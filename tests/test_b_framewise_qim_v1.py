"""CPU artificial tensors only. No scientific cache/model/media is opened."""
import ast
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from main.tube_state import b_framewise_qim_v1 as p
from main.tube_state.b_framewise_qim_v1_solver import solve, Limits, project_budget
from runtime.wan.b_framewise_qim_v1_operator import HeadOperator, feature_delta, replay_head
from experiments.wan_state_clock import b_framewise_qim_v1_experiment as runner

pytestmark = pytest.mark.unit
torch.set_num_threads(2)


def test_bch_algebra_bit_order_and_seven_errors():
    # Independent Horner GF multiply/reduce, not the encoder's division path.
    def mult(a,b):
        poly = 0
        for i in range(6):
            if b & (1 << i): poly ^= a << i
        for k in range(10,5,-1):
            if poly & (1 << k): poly ^= 67 << (k-6)
        return poly
    for j in (0,1,31,181,65536,92671):
        bits = p.encode(j)
        assert sum(b << (23-i) for i,b in enumerate(bits[:24])) == j
        alpha = 1
        for root in range(1,15):
            alpha = mult(alpha,2); value = 0
            for b in bits: value = mult(value,alpha)^b
            assert value == 0
        damaged = bits.copy()
        for i in (0,4,13,26,39,51,62): damaged[i] ^= 1
        assert p.decode(damaged)['j'] == j
    assert p.generator().bit_length() == 40
    with pytest.raises(ValueError): p.encode(92672)


def test_dct_qim_scale_ties_and_partial_replica_merge():
    atom = p.basis(); assert float(atom.square().sum()) == pytest.approx(1.)
    values = torch.tensor([-16.,0.,16.,32.],dtype=torch.float64)
    assert p.nearest(values,0).tolist() == [-32.,0.,0.,32.]
    d0,d1,s = p.distances(torch.tensor([0.,8.,16.,24.]))
    assert s.tolist() == [-1.,0.,1.,0.]
    rgb = torch.zeros((1,320,512,3),dtype=torch.float64)
    rgb[0,200:208,16:24] = 16/255*atom[:,:,None]
    q = p.read_rgb(rgb)['q']; assert float(q[0,0,0,0]) == pytest.approx(16.)
    # P^T is independently checked by an inner product, across every tile/block.
    torch.manual_seed(10)
    r = torch.randn((1,10,3,58,74),dtype=torch.float64); t = torch.randn((1,10,63),dtype=torch.float64)
    assert float((p.project(r)*t).sum()) == pytest.approx(float((r*p.project_transpose(t)).sum()),abs=1e-12)
    soft = torch.zeros((2,5,2,63),dtype=torch.float64); mask = torch.zeros_like(soft,dtype=torch.bool)
    word = torch.tensor(p.encode(181))*2-1
    soft[0,1,0,:31] = word[:31]; soft[0,1,1,31:] = word[31:]
    mask[0,1,0,:31] = True; mask[0,1,1,31:] = True
    result = p.merge_replicas(soft,mask)
    assert result['candidates'][1]['j'] == 181
    assert result['candidates'][6]['status'] == 'ERASURE'  # no cross-frame mixing


def test_startup_boundary_and_global_clock():
    js,_ = p.schedule(6); mask = p.support(js >= 0)
    assert (mask.flatten(1).sum(1)//3).tolist() == [8510,17020,25530,34040,42550,42550]
    js2,_ = p.schedule(1,181)
    assert sorted(js2[0,:5].tolist()) == [177,178,179,180,181]
    assert not bool(mask[:,5:,:,57].any())


def test_operator_full_conv_and_transpose_no_temporal_truncation():
    torch.manual_seed(9)
    weight = torch.randn((3,2,3,3,3),dtype=torch.float64)/10
    active = p.schedule(5)[0] >= 0
    op = HeadOperator(weight,active)
    a = torch.randn(op.input_shape,dtype=torch.float64)
    features = feature_delta(a,active,dtype=torch.float64,device='cpu')
    full = F.conv3d(F.pad(features,(1,1,1,1,2,0)),weight)/2
    actual = p.pack(full[0].permute(1,2,3,0)); predicted = op.forward(a)
    assert torch.allclose(actual,predicted,atol=2e-15,rtol=1e-12)
    r = torch.randn(op.output_shape,dtype=torch.float64)
    assert float((predicted*r).sum()) == pytest.approx(float((a*op.adjoint(r)).sum()),rel=1e-12,abs=1e-10)
    # A singleton perturbation still contributes in later native chunks.
    a[1:] = 0
    assert float(op.forward(a)[1:3].abs().max()) > 0
    assert float(op.forward(a)[3:].abs().max()) == 0


class FakeCausal(torch.nn.Conv3d):
    def __init__(self):
        super().__init__(2,3,3,padding=0); self._padding=(1,1,1,1,2,0)
    def forward(self,x,cache_x=None):
        padding = list(self._padding)
        if cache_x is not None: x=torch.cat((cache_x,x),dim=2);padding[4]-=cache_x.shape[2]
        return super().forward(F.pad(x,padding))


def test_native_modified_tail_replay_matches_joint_not_stale_clone():
    torch.manual_seed(8); conv=FakeCausal().eval().requires_grad_(False)
    active=p.schedule(5)[0]>=0
    a=torch.randn((5,10,2,7,9),dtype=torch.float64)/10
    h=torch.randn((1,2,5,320,512))
    expected=conv(h+feature_delta(a,active,dtype=torch.float32,device='cpu'))[0].permute(1,2,3,0)/2+.5
    result=torch.cat([x for _,x in replay_head(conv,[h[:,:,:1],h[:,:,1:]],a,active)])
    assert torch.allclose(result,expected,atol=5e-7,rtol=1e-6)
    # Demonstrate that using the old tail really is observable in this fixture.
    changed=h[:,:,1:]+feature_delta(a[1:],active[1:],dtype=torch.float32,device='cpu')
    stale=conv(changed,h[:,:,:1])[0].permute(1,2,3,0)/2+.5
    assert float((stale-result[1:]).abs().max()) > 1e-4


class Matrix:
    def __init__(self,m):
        self.m=torch.tensor(m,dtype=torch.float64);self.input_shape=(1,self.m.shape[1]);self.output_shape=(1,self.m.shape[0])
        self.bound=float(torch.linalg.matrix_norm(self.m,2))
    def forward(self,a):return a@self.m.T
    def adjoint(self,r):return r@self.m


def test_solver_analytic_lexicographic_nullspace_and_active_box():
    # r=(a0+a1,a2), r0=1. Primary optimum r=(1,0), secondary a=(.5,.5,0).
    op=Matrix([[1,1,0],[0,0,1]])
    d=torch.tensor([[1.]],dtype=torch.float64)
    answer=solve(op,d,torch.ones_like(d,dtype=torch.bool),torch.ones(op.output_shape,dtype=torch.bool),
        measure=lambda r:r[:,:1],transpose=lambda z:F.pad(z,(0,1)),measurement_bound=1,rms=2,linf=2)
    assert answer['status']=='SOLVED_APPROXIMATE_KKT'
    assert torch.allclose(answer['coefficients'],torch.tensor([[.5,.5,0.]],dtype=torch.float64),atol=2e-6)
    # Box-only projection is wrong here; the sphere is also active.
    v=torch.tensor([[4.,1.]],dtype=torch.float64)
    projected=project_budget(v,torch.ones_like(v,dtype=torch.bool),rms=1/math.sqrt(2),linf=.8)
    assert projected.tolist()[0]==pytest.approx([.8,.6],abs=1e-12)


def test_solver_unresolved_not_false_infeasible_and_resource_limits():
    op=Matrix([[1]])
    kwargs=dict(measure=lambda x:x,transpose=lambda x:x,measurement_bound=1,rms=.1,linf=.1)
    d=torch.ones((1,1),dtype=torch.float64);mask=torch.ones_like(d,dtype=torch.bool)
    result=solve(op,d,mask,mask,limits=Limits(iterations=20),**kwargs)
    assert result['status']=='NUMERICALLY_UNRESOLVED'
    assert solve(op,d,mask,mask,limits=Limits(memory_bytes=1),**kwargs)['status']=='RESOURCE_UNRESOLVED'
    assert solve(Matrix([[0]]),d,mask,mask,**kwargs)['status']=='CERTIFIED_INFEASIBLE_ZERO_RESPONSE'


def config():
    return runner.load(Path(__file__).resolve().parents[1]/'experiments/wan_state_clock/configs/b_framewise_qim_v1.json')


def test_missing_input_and_interruption_recovery_preserve_denominators(tmp_path):
    def forbidden(*args):raise AssertionError('backend must not start')
    result=runner.run(config(),tmp_path/'missing',tmp_path/'out',forbidden)
    assert result['stop']['kind']=='INPUT_MISSING'
    assert result['denominators']['time_read_slots']==912240
    assert all(v['started']==0 for v in result['counts'].values())
    states=runner.load(tmp_path/'out/state.json');states['BASE/preclip']['status']='STARTED';runner.dump(tmp_path/'out/state.json',states)
    recovered=runner.run(config(),tmp_path/'missing',tmp_path/'out',forbidden)
    assert recovered['rows']['BASE/preclip']['status']=='UNRESOLVED_INTERRUPTED'
    assert recovered['physical_calls_this_invocation']==0 and len(recovered['rows'])==24


@pytest.mark.parametrize('valid_truth',[True,False])
def test_runner_fake_path_raw_before_truth_and_actual_identity_cache(tmp_path,monkeypatch,valid_truth):
    from main.tube_state import b_local_identity_dct_v1 as identity
    source=tmp_path/'source';terminal=source/config()['terminal_relative'];terminal.parent.mkdir(parents=True);terminal.write_text('fake terminal never loaded')
    # Exact envelope produced by b_terminal_three_group_v1.run; truth stays post-only.
    runner.dump(source/'protocol.json',dict(config=dict(writer=dict(bits=[0]*32 if valid_truth else [0])),
        denominators={},scientific_pass=None))
    events=[];original=runner.load
    def checked_load(path):
        if Path(path)==source/'protocol.json':
            assert len(list((tmp_path/'out/raw').rglob('*.npz')))==8
            assert len(list((tmp_path/'out/identity').rglob('*.json')))==16
            events.append('truth')
        return original(path)
    monkeypatch.setattr(runner,'load',checked_load)
    class Fake:
        def __init__(self,cfg,out,event):self.event=event
        def capture(self,path):self.event('D181',False);self.event('D181',True);return 'base'
        def read_layer(self,path,layer):
            q=np.zeros((181,5,2,63));return dict(q=q,d0=q,d1=q+16,soft=q-1,mask=np.ones(q.shape,dtype=bool))
        def media(self,path,arm):
            for k in ('codec_encode','codec_probe','codec_decode'):self.event(k,False);self.event(k,True)
            return 'media'
        def identity(self,media,start,phase,view):
            self.event('E41',False)
            # Actual B1 API with an artificial normalized latent, no encoder.
            cache=identity.decode(identity.project(torch.zeros(identity.LOCAL_SHAPE)),'watermark',view_id=view,key_id='key00')
            self.event('E41',True);return cache
        def solve(self,base,progress):return dict(status='SOLVED_APPROXIMATE_KKT')
        def apply(self,base):return 'qim',dict(status='NATIVE_RESIDUAL_WITHIN_TOLERANCE')
        def close(self):pass
    result=runner.run(config(),source,tmp_path/'out',Fake)
    assert events==['truth']
    assert result['counts']['E41']['returned']==16
    post=original(tmp_path/'out/evaluation.json')
    assert all(x['status']=='RETURNED' for x in original(tmp_path/'out/state.json').values())
    if valid_truth:
        assert result['status']=='COMPLETED_FIXED_RECORDS' and result['evaluation_status']=='POST_RECORDED'
        assert len(post['time'])==8 and len(post['identity'])==2560
        assert post['identity_comparison']['common_complete_observed_slots']==320
        assert post['identity_comparison']['base_common_ber']==0
    else:
        assert result['status']=='STOPPED_FIXED_ATTEMPT' and result['stop']['kind']=='POST_UNRESOLVED'
        assert post['status']=='POST_UNRESOLVED' and result['evaluation_status']=='POST_UNRESOLVED'
        def forbidden(*args):raise AssertionError('spent failed attempt must not reenter')
        recovery=runner.run(config(),source,tmp_path/'out',forbidden)
        assert recovery['physical_calls_this_invocation']==0
        assert recovery['saved']['result']['evaluation_status']=='POST_UNRESOLVED'


@pytest.mark.parametrize('state',[
    ['not a row mapping'], {'BASE/preclip':{'status':[]}}, {'BASE/preclip':{'status':{}}}])
def test_corrupt_recovery_keeps_unknown_rows_without_reentry(tmp_path,state):
    runner.dump(tmp_path/'attempt.json',dict(status='STARTED'))
    runner.dump(tmp_path/'state.json',state)
    result=runner.recover(tmp_path)
    assert len(result['rows'])==24
    assert all(x['status']=='UNRESOLVED_RECEIPT' for x in result['rows'].values())
    assert result['physical_calls_this_invocation']==0 and result['denominators']['identity_local_bit_slots']==2560


def test_layering_and_cli_import_inert():
    root=Path(__file__).resolve().parents[1]
    for folder in ('main/tube_state','runtime/wan'):
        for file in (root/folder).glob('b_framewise_qim_v1*.py'):
            tree=ast.parse(file.read_text())
            imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
            forbidden=('runtime','experiments','governance') if folder.startswith('main') else ('experiments','governance')
            assert all(not x.startswith(forbidden) for x in imports)
    import experiments.wan_state_clock.b_framewise_qim_v1_run
