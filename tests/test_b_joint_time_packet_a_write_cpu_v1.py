"""Small artificial stage-P checks; no real q0, model, full solver or A-CPU run."""
import json
import struct
from pathlib import Path
import numpy as np
import pytest
import torch
from main.tube_state import b_joint_time_packet_a_writer_v1 as writer
from main.tube_state import b_joint_time_packet_a_v1 as packet
from main.tube_state import b_framewise_qim_v1 as old
from runtime.wan import b_joint_time_packet_a_write_cpu_v1 as runtime
from experiments.wan_state_clock import b_joint_time_packet_a_write_cpu_v1_experiment as experiment
from experiments.wan_state_clock import b_joint_time_packet_a_write_cpu_v1_records as rec

pytestmark = pytest.mark.unit


def test_schedule_age_replica_order_and_unwritten_startup():
    fields = writer.schedule(6)
    for n in range(6):
        for replica in range(2):
            for lane in range(5):
                tile = 5*replica+lane
                legal = [j for j in range(max(0,n-4),n+1) if j%5==lane]
                if not legal:
                    assert not fields['active'][n,tile]
                    assert fields['messages'][n,tile] == -1
                    assert (fields['bits'][n,tile] == -1).all()
                else:
                    j = legal[0]; msg = 8*j+n-j
                    assert fields['messages'][n,tile] == msg
                    assert packet.decode_raw24(fields['bits'][n,tile].int().tolist())['msg'] == msg
    assert torch.equal(fields['bits'][:,:5],fields['bits'][:,5:])
    assert fields['messages'][4,2] == 18  # j2, age2; old code carried only j2.


def test_q0_replica_major_and_original_distance_units():
    q = np.arange(181*5*2*63, dtype=np.float64).reshape(181,5,2,63)/10
    d0,d1,soft = old.distances(torch.from_numpy(q))
    raw = dict(q=q,d0=d0.numpy(),d1=d1.numpy(),soft=soft.numpy(),mask=np.ones(q.shape,dtype=bool))
    converted = writer.validate_raw_q0(raw)
    for lane,replica in ((0,1),(3,0),(4,1)):
        assert converted[90,5*replica+lane,17] == q[90,lane,replica,17]
    wrong = dict(raw, soft=raw['soft']*2)
    with pytest.raises(ValueError,match='units'):writer.validate_raw_q0(wrong)
    wrong = dict(raw,mask=raw['mask'].copy());wrong['mask'][0,4,1,2]=False
    with pytest.raises(ValueError,match='complete'):writer.validate_raw_q0(wrong)


def test_targets_lower_ties_and_no_startup_filler():
    fields = writer.schedule(5)
    q0 = 16*fields['bits']+16  # exact midpoint, including negative/unwritten sentinel.
    target = writer.targets(q0, fields)
    writer.validate_targets(q0,target,fields)
    assert torch.equal(target[fields['active']],16*fields['bits'][fields['active']])
    assert torch.equal(target[~fields['active']],q0[~fields['active']])
    damaged = target.clone();damaged[0,1,0]+=1
    with pytest.raises(ValueError,match='unwritten'):writer.validate_targets(q0,damaged,fields)


def tiny_head():
    return dict(weight=torch.ones((3,1,3,3,3),dtype=torch.float32),bias=torch.zeros(3,dtype=torch.float32),
        causal_contract=dict(padding=[1,1,1,1,2,0],stride=[1,1,1],dilation=[1,1,1],groups=1))


def test_head_exact_byte_ranges_no_full_checkpoint_required(tmp_path):
    weight=np.arange(81,dtype='<f4').reshape(3,1,3,3,3);bias=np.array([1,2,3],dtype='<f4')
    header=json.dumps({'decoder.conv_out.bias':dict(dtype='F32',shape=[3],data_offsets=[0,12]),
        'decoder.conv_out.weight':dict(dtype='F32',shape=list(weight.shape),data_offsets=[12,336])}).encode()
    hp=tmp_path/'header';hp.write_bytes(struct.pack('<Q',len(header))+header)
    data=tmp_path/'head';data.write_bytes(bias.tobytes()+weight.tobytes())
    tensors,receipt=runtime.read_head_tensors(hp,data,8+len(header))
    assert torch.equal(tensors['weight'],torch.from_numpy(weight))
    assert torch.equal(tensors['bias'],torch.from_numpy(bias))
    assert receipt['requested_tensors']['weight']['bytes']==324
    data.write_bytes(data.read_bytes()[:-1])
    with pytest.raises(ValueError,match='incomplete'):runtime.read_head_tensors(hp,data,8+len(header))
    hp.write_bytes(hp.read_bytes()[:-1])
    with pytest.raises(ValueError,match='header'):runtime.read_head_tensors(hp,data,8+len(header))


def test_adapter_uses_original_solver_limits_and_preserves_cap(monkeypatch,tmp_path):
    from main.tube_state.b_framewise_qim_v1_solver import solve as original
    assert runtime.solve is original
    backend=runtime.Backend(tiny_head(),tmp_path,lambda *args:None,rec.dump)
    backend.q0=torch.zeros((181,10,63),dtype=torch.float64)
    backend.target()
    def fake_solve(op,d,target_mask,budget_mask,*,limits,progress):
        assert (limits.iterations,limits.check_every,limits.seconds,limits.memory_bytes)==(4000,20,7200,16*1024**3)
        assert d.dtype==torch.float64 and d.device.type=='cpu'
        assert torch.equal(d,(backend.targets-backend.q0)/255)
        assert torch.equal(target_mask,backend.fields['active'][:,:,None].expand(-1,-1,63))
        return dict(status='NUMERICALLY_UNRESOLVED',reason='iteration cap',iterations=4000,
                    coefficients=torch.zeros(op.input_shape,dtype=torch.float64),metrics={'target':0.125})
    monkeypatch.setattr(runtime,'solve',fake_solve)
    answer=backend.solve(lambda row:None)
    assert answer['status']=='NUMERICALLY_UNRESOLVED' and answer['native_handoff_eligible']
    assert (tmp_path/'coefficients.pt').is_file() and not (tmp_path/'prediction.pt').exists()
    # A caught call/resource failure is preserved, never made handoff eligible.
    monkeypatch.setattr(runtime,'solve',lambda *a,**k:dict(status='NUMERICALLY_UNRESOLVED',reason='RuntimeError: test failure',
        coefficients=torch.zeros(backend.operator.input_shape,dtype=torch.float64)))
    answer=backend.solve(lambda row:None)
    assert answer['finite_candidate_saved'] and not answer['native_handoff_eligible']


def config(tmp_path):
    head=tmp_path/'head';head.write_text('fake')
    paths={s:tmp_path/(s+'.npz') for s in rec.SOURCES}
    for path in paths.values():path.write_text('fake')
    return dict(head=str(head),q0={s:dict(file=str(paths[s]),source_id=rec.SOURCE_IDS[s],role='BASE',
        layer='preclip',units='255*RGB_luma_DCT',frames=181) for s in rec.SOURCES})


class FakeBackend:
    fail_n01=False
    def __init__(self,head,output,event,dump):self.source=Path(output).name;self.event=event
    def read_q0(self,spec):
        if self.fail_n01 and self.source=='n01':raise FileNotFoundError('artificial missing q0')
    def target(self):pass
    def solve(self,progress):
        for kind in ('K','KT'):self.event(kind,False);self.event(kind,True)
        return dict(native_handoff_eligible=True)
    def predict(self):self.event('K',False);self.event('K',True)
    def handoff(self,source):return dict(status='ARTIFICIAL_READY',source=source)


def test_one_attempt_counts_partial_source_and_readonly_reentry(tmp_path,monkeypatch):
    cfg=config(tmp_path);output=tmp_path/'run'
    monkeypatch.setattr(FakeBackend,'fail_n01',True)
    # Even a physically absent source input cannot block the other source.
    Path(cfg['q0']['n01']['file']).unlink()
    result=experiment.run(cfg,output,backend_factory=FakeBackend,head_loader=lambda path:{})
    assert result['handoff_status']=='CPU_HANDOFF_PARTIAL'
    rows=rec.load(output/'operations.json')
    assert rows['rows']['n01/q0_load']['status']=='FAILED'
    assert rows['rows']['n01/solver']['started']==0
    assert rows['rows']['e02/solver']['returned']==1
    assert rows['operator_counts']['e02/solver/K']['returned']==1
    assert rows['operator_counts']['e02/prediction/K']['returned']==1
    assert rows['operator_counts']['e02/solver/KT']['returned']==1
    assert rec.load(output/'cpu_handoff.json')['source_eligibility']==dict(n01=False,e02=True)
    frozen=(output/'operations.json').read_bytes()
    def forbidden(*args):raise AssertionError('consumed reentry opened scientific input')
    again=experiment.run({},output,backend_factory=forbidden,head_loader=forbidden)
    assert again['physical_calls_this_invocation']==0
    assert (output/'operations.json').read_bytes()==frozen


def test_interruption_and_head_prepare_failure_are_consumed(tmp_path):
    cfg=config(tmp_path);output=tmp_path/'run'
    class Interrupted(FakeBackend):
        def solve(self,progress):raise KeyboardInterrupt('artificial stop')
    result=experiment.run(cfg,output,backend_factory=Interrupted,head_loader=lambda path:{})
    rows=rec.load(output/'operations.json')['rows']
    assert rows['n01/solver']['status']=='INTERRUPTED'
    assert rows['e02/q0_load']['status']=='NOT_STARTED'
    head=tmp_path/'head_attempt'
    def fail(*args):raise RuntimeError('artificial CPU load failure')
    result=experiment.prepare_head(tmp_path/'model',head,prepare=fail)
    assert result['status']=='HEAD_PREPARATION_FAILED'
    assert experiment.prepare_head('missing',head,prepare=fail)['physical_calls_this_invocation']==0


def test_catalog_and_K_KT_independent_caps(tmp_path):
    catalog=rec.operation_catalog()
    assert len(catalog['operator_ids']['n01']['solver_K'])==4200
    assert catalog['bounds']['K']==8402 and catalog['bounds']['KT']==8400
    assert catalog['active_target_slots_per_source']==112770
    journal=rec.Journal(tmp_path);journal.current='n01/solver'
    journal.counts['n01/solver/KT']=dict(planned=4200,started=4200,returned=4200)
    with pytest.raises(rec.BudgetExceeded):journal.event('n01','KT',False)
    assert journal.counts['n01/solver/KT']['started']==4200
    journal.current='n01/prediction';journal.event('n01','K',False);journal.event('n01','K',True)
    assert journal.counts['n01/prediction/K']['returned']==1
    with pytest.raises(rec.BudgetExceeded):journal.event('n01','KT',False)


def test_cli_consumed_stage_before_missing_or_unreadable_config(tmp_path):
    import subprocess
    import sys
    output=tmp_path/'consumed';output.mkdir()
    rec.dump(output/'attempt.json',dict(status='CONSUMED_ON_START'))
    before=(output/'attempt.json').read_bytes()
    unreadable=tmp_path/'config-directory';unreadable.mkdir()
    damaged=tmp_path/'damaged.json';damaged.write_text('{broken')
    for config_path in (tmp_path/'missing.json',unreadable,damaged):
        completed=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_joint_time_packet_a_write_cpu_v1_run',
            'solve','--config',str(config_path),'--output',str(output)],cwd=Path(__file__).resolve().parents[1],
            check=True,capture_output=True,text=True)
        result=json.loads(completed.stdout)
        assert result['status']=='READ_ONLY_CONSUMED_STAGE'
        assert result['physical_calls_this_invocation']==result['scientific_reads_this_invocation']==0
    assert (output/'attempt.json').read_bytes()==before
    assert sorted(p.name for p in output.iterdir())==['attempt.json']

