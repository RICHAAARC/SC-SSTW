import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from diffusers import UniPCMultistepScheduler
from main.tube_state import projection_margin as carrier, state_clock, flow_control
from runtime.wan.flow_step import controlled_step
from runtime.wan.flow_generation import continue_steps

torch.set_num_threads(1)
pytestmark = pytest.mark.unit


def scheduler():
    s = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3.)
    s.set_timesteps(50)
    s.set_begin_index(0)
    return s


@pytest.fixture(scope='module')
def book():
    return state_clock.codebook(b'WanProjection-first-validation-key-v1')


def test_analytic_loss_sign_basis_and_gradient(book):
    z = np.random.default_rng(4).normal(size=carrier.SHAPE).astype(np.float32)
    u, rec = flow_control.request(z, book, 0)
    d, c = book['directions'].astype(float), book['codes'][0]
    p = np.asarray(rec['projection'])
    grad = -2/carrier.SUPPORT_COUNT * (np.maximum(0, 1-c*p)*c)[:,None]*d
    u_blocks = carrier.blocks(u).reshape(1760,1024)
    np.testing.assert_allclose(u_blocks, -grad*1760/(6*(d*d).sum(axis=1)[:,None]), rtol=1e-5, atol=1e-7)
    direction = np.random.default_rng(6).normal(size=grad.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-3
    blocks = carrier.blocks(z).reshape(1760,1024).astype(float)
    def loss(x):
        return np.mean(np.maximum(0,1-c*np.einsum('ij,ij->i',x,d))**2)
    fd = (loss(blocks+eps*direction)-loss(blocks-eps*direction))/(2*eps)
    assert fd == pytest.approx(float((grad*direction).sum()), rel=1e-4, abs=1e-8)
    assert flow_control.projection_record(z+u,book,0)['loss'] < rec['loss']
    assert np.count_nonzero(u[:,:,0]) == np.count_nonzero(u[:,:,45]) == 0
    sigma = .292
    np.testing.assert_allclose(z-sigma*(-u/sigma),z+u,atol=5e-7)


def test_unipc_off_equivalence_affine_budgets_and_tail(book):
    s = scheduler()
    z = torch.randn(carrier.SHAPE, generator=torch.Generator().manual_seed(2))
    for i in range(44):
        z = s.step((z*.05).bfloat16().float(),s.timesteps[i],z,return_dict=False)[0]
    assert float(s.sigmas[44]) == pytest.approx(.292,abs=.002)
    original = copy.deepcopy(s)
    off = z.clone()
    totals = {'u':0.,'D':0.}
    counts = []
    for i in range(44,50):
        v = (z*.05).bfloat16().float()
        if i < 47:
            sigma = float(s.sigmas[s.step_index])
            u,_ = flow_control.request((z-sigma*v).numpy(),book,0)
            before = copy.deepcopy(s)
            z,record,arrays = controlled_step(s,z,v,s.timesteps[i],torch.from_numpy(u),.015,totals,lambda *x:counts.append(x))
            assert record['u']['support_rms'] <= .005+2e-6
            assert record['D']['support_rms'] <= .005+2e-6
            assert record['affine_error']['support_rms'] < record['numerical_tolerance']
            assert before.step_index == i and s.step_index == i+1
        else:
            z = s.step(v,s.timesteps[i],z,return_dict=False)[0]
        off = original.step((off*.05).bfloat16().float(),original.timesteps[i],off,return_dict=False)[0]
    assert totals['u'] <= .015+6e-6 and totals['D'] <= .015+6e-6
    assert counts.count(('shadow_step',True)) == 6
    assert s.step_index == 50 and s.lower_order_nums > 0
    assert not torch.equal(off,z)
    # A zero control follows bitwise identical UniPC updates with unchanged history.
    a,b = scheduler(),scheduler()
    x=y=torch.ones((1,16,46,1,1))
    for i in range(50):
        v=(x*.05).bfloat16().float()
        if i in (44,45,46):
            x,_,_ = controlled_step(a,x,v,a.timesteps[i],torch.zeros_like(x),0.,{'u':0.,'D':0.},lambda *args:None)
        else:
            x=a.step(v,a.timesteps[i],x,return_dict=False)[0]
        y=b.step((y*.05).bfloat16().float(),b.timesteps[i],y,return_dict=False)[0]
        assert torch.equal(x,y)


def fake_setup(monkeypatch, module, fail_flow=False):
    class VAE:
        def to(self,*args): return self
        def parameters(self): return iter([torch.zeros(1)])
        def clear_cache(self): pass
    class Transformer:
        def __call__(self,hidden_states,**kwargs): return ((hidden_states*.05).bfloat16(),)
    def prepare(config):
        return SimpleNamespace(vae=VAE(),transformer=Transformer(),scheduler=scheduler()),torch.ones(carrier.SHAPE),None,None,torch.float32
    monkeypatch.setattr(module,'prepare_generation',prepare)
    monkeypatch.setattr(module,'decode_normalized_latent',lambda vae,z: torch.full((181,2,2,3),.5+float(z.mean())*.01))
    videos = {}
    def encode(rgb,path,*args):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'fake')
        videos[str(path)] = rgb.clone()
    monkeypatch.setattr(module,'encode_rgb',encode)
    monkeypatch.setattr(module,'read_mp4',lambda path:videos[str(path)])
    monkeypatch.setattr(module,'reencode_rgb24_readback',lambda vae,rgb:torch.zeros((1,16,1+(len(rgb)-1)//4,40,64)))
    received = []
    def read(obs,book):
        received.append({g:tuple(v.shape) for g,v in obs.items()})
        return {'rankings':{k:{'message_unique':True,'best':{'message':0}} for k in ('global_matched','global_state','local_matched','local_without_update','local_state')}}
    monkeypatch.setattr(module.state_clock,'read',read)
    monkeypatch.setattr(module.state_clock,'report',lambda detection,truth,delta:{'truth':truth})
    if fail_flow:
        real = module.controlled_step
        calls = [0]
        def fail_once(*args,**kwargs):
            calls[0]+=1
            if calls[0]==1: raise RuntimeError('injected first-flow failure')
            return real(*args,**kwargs)
        monkeypatch.setattr(module,'controlled_step',fail_once)
    return received


@pytest.mark.parametrize('failure',[False,True])
def test_fixed_runner_failure_retention_and_counts(monkeypatch,tmp_path,failure):
    from experiments.wan_state_clock import flow_run as module
    received = fake_setup(monkeypatch,module,failure)
    config=json.loads(Path('experiments/wan_state_clock/configs/generate_replication.json').read_text())
    result=module.run(config,tmp_path/'run')
    assert tuple(result['videos'])==module.ARMS
    assert len(received)==5
    assert result['videos']['FLOW_B']['status']=='COMPLETE'
    assert all(len(v['observations'])==4 for v in result['videos'].values())
    if not failure:
        assert not result['failures']
        assert result['precision']['cfg_output_dtype']=='torch.bfloat16'
        assert result['precision']['scheduler_input_dtype']=='torch.float32'
        expected=torch.ones(carrier.SHAPE)
        s=scheduler()
        for i in range(50):
            expected=s.step((expected*.05).bfloat16().float(),s.timesteps[i],expected,return_dict=False)[0]
        assert torch.equal(torch.load(tmp_path/'run/OFF_terminal.pt',weights_only=True),expected)
        for k,n in result['fixed_calls'].items():
            assert result['actual_calls'][k+'_completed']==n
            assert result['actual_calls'][k+'_attempted']==n
        assert received[0][0][2]==46 and received[0][1][2]==45
        assert result['first_round_criteria_met'] is False
    else:
        assert any('injected' in row['error'] for row in result['failures'])
        assert result['first_round_criteria_met'] is None
        assert result['videos']['FLOW_A']['status']=='PARTIAL_OR_FAILED'
    assert json.loads((tmp_path/'run/result.json').read_text())['actual_calls']==result['actual_calls']


def test_notebook_static_bundle():
    import ast,base64,io,zipfile
    root=Path('.')
    notebook=json.loads((root/'notebooks/flow_tube_state_colab.ipynb').read_text())
    assert ''.join(notebook['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in notebook['cells']:
        if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
    source=''.join(next(c for c in notebook['cells'] if c['id']=='source')['source'])
    tree=ast.parse(source)
    payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PAYLOAD' for t in n.targets))
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        for name in archive.namelist():
            if name.endswith('.py'): assert archive.read(name)==(root/name).read_bytes()
    assert 'experiments.wan_state_clock.flow_run' in ''.join(notebook['cells'][-1]['source'])
