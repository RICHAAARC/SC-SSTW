import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from diffusers import UniPCMultistepScheduler
from main.tube_state import state_clock,projection_margin as carrier
from main.tube_state import velocity_coefficients as method
from runtime.wan.velocity_direction import tail,detached_scheduler,response_coefficients

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def scheduler():
    s=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3.)
    s.set_timesteps(50);s.set_begin_index(0)
    return s


class Model:
    def __call__(self,hidden_states,timestep,encoder_hidden_states,**kwargs):
        return (torch.sin(hidden_states*.1)+.001*encoder_hidden_states,)


@pytest.fixture(scope='module')
def setup():
    book=state_clock.codebook(b'WanProjection-first-validation-key-v1')
    directions=torch.tensor(book['directions'])
    codes=torch.tensor(book['codes'])
    s=scheduler()
    z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(6))*.1
    for i in range(44):z=s.step(torch.sin(z*.1),s.timesteps[i],z,return_dict=False)[0]
    return book,directions,codes,s,z


def test_torch_blocks_loss_and_scatter_match_numpy(setup):
    book,d,c,s,z=setup
    np.testing.assert_array_equal(method.blocks(z).numpy(),carrier.blocks(z.numpy()).reshape(1760,1024))
    a=torch.randn(1760,generator=torch.Generator().manual_seed(8),requires_grad=True)
    scattered=method.scatter(a,d)
    expect=np.zeros(carrier.SHAPE,dtype=np.float32)
    carrier.put_blocks(expect,(a.detach()[:,None]*d).numpy())
    np.testing.assert_array_equal(scattered.detach().numpy(),expect)
    assert not scattered[:,:,0].count_nonzero() and not scattered[:,:,45].count_nonzero()
    p=np.einsum('ij,ij->i',carrier.blocks(z.numpy()).reshape(1760,1024).astype(float),book['directions'].astype(float))
    assert float(method.terminal_loss(z,d,c[0]))==pytest.approx(np.maximum(0,1-book['codes'][0]*p).dot(np.maximum(0,1-book['codes'][0]*p))/1760)
    loss=method.terminal_loss(z+scattered,d,c[0]);g=torch.autograd.grad(loss,a)[0]
    assert torch.isfinite(g).all() and g.count_nonzero()


def test_scalar_response_matches_random_live_history_and_clone(setup):
    _,d,c,s,z=setup
    calls=[]
    rows=response_coefficients(s,lambda *x:calls.append(x))
    assert calls.count(('response_probe_step',True))==6
    assert s.step_index==44
    live=detached_scheduler(s)
    x=z.clone().requires_grad_()
    for row in rows:
        i=row['index'];v=torch.sin(x*.1)
        delta=torch.randn_like(x)*.01
        base,ctrl=detached_scheduler(live),detached_scheduler(live)
        without=base.step(v.detach(),live.timesteps[i],x.detach(),return_dict=False)[0]
        with_control=ctrl.step((v+delta).detach(),live.timesteps[i],x.detach(),return_dict=False)[0]
        torch.testing.assert_close(with_control-without,row['h']*delta,atol=1e-7,rtol=2e-3)
        x=live.step(v+delta,live.timesteps[i],x,return_dict=False)[0]
        assert x.grad_fn is not None
        clone=detached_scheduler(live)
        assert clone.last_sample.grad_fn is None
        assert any(t.grad_fn is not None for t in live.model_outputs if t is not None)
        assert clone.step_index==live.step_index and clone.this_order==live.this_order
    assert s.step_index==44


def run_tail(setup,a,checkpoint=False,record=None,input_dtype=torch.float32):
    _,d,c,s,z=setup
    counts={}
    def count(k,done):counts[k+('_completed' if done else '_attempted')]=counts.get(k+('_completed' if done else '_attempted'),0)+1
    out=tail(SimpleNamespace(transformer=Model()),s,z,torch.tensor(1.),torch.tensor(-1.),input_dtype,5.,a,d,.02,count,
        record or (lambda *a:None),use_checkpoint=checkpoint,responses=response_coefficients(s,lambda *a:None))
    return out,counts


def test_true_tail_ad_checkpoint_finite_difference_and_history(setup):
    _,d,c,s,z=setup
    a=torch.zeros((3,1760),requires_grad=True)
    out,counts=run_tail(setup,a,True)
    g=torch.autograd.grad(method.terminal_loss(out,d,c[0]),a)[0]
    assert bool((g.abs().sum(dim=1)>0).all())
    assert counts['transformer_completed']==12 and counts['scheduler_step_completed']==6 and counts['shadow_step_completed']==6
    assert counts['transformer_replay_completed']==10
    assert s.step_index==44
    b=torch.zeros_like(a,requires_grad=True)
    native,_=run_tail(setup,b,False)
    native_g=torch.autograd.grad(method.terminal_loss(native,d,c[0]),b)[0]
    torch.testing.assert_close(out,native,rtol=0,atol=0)
    torch.testing.assert_close(g,native_g,rtol=1e-6,atol=1e-9)
    responses=response_coefficients(s,lambda *a:None)
    q,info=method.direction_and_amplitude(g,d,[r['sigma'] for r in responses],[r['h'] for r in responses],.02)
    assert info['rho']==.1
    assert info['epsilon']==pytest.approx(.1*.999*info['epsilon_cap'])
    eps=info['epsilon']
    records=[]
    with torch.no_grad():
        plus,_=run_tail(setup,eps*q,record=lambda row,arrays:records.append(row))
        minus,_=run_tail(setup,-eps*q)
        fd=float((method.terminal_loss(plus,d,c[0])-method.terminal_loss(minus,d,c[0]))/(2*eps))
    assert fd==pytest.approx(info['AD_directional_derivative'],rel=.02,abs=1e-7)
    assert all(r['step_within_budget'] and r['sum_within_budget'] and r['response_valid'] for r in records)
    assert all(r['live_shadow_error']['support_rms']==0 for r in records)
    # Deliberate huge coefficients are retained and marked, not silently clipped.
    large=[]
    with torch.no_grad():run_tail(setup,1000*q,record=lambda row,arrays:large.append((row,arrays)))
    assert any(not row['step_within_budget'] for row,_ in large)
    torch.testing.assert_close(large[0][1]['requested_delta_velocity'],method.scatter(1000*q[0],d))


@pytest.mark.parametrize('failure',[False,'direction','plus_forward','zero_a_forward','zero_a_backward'])
def test_six_rows_counts_failure_and_symmetric_coefficients(monkeypatch,tmp_path,setup,failure):
    from experiments.wan_state_clock import velocity_direction_run as runner
    def prepare(config,*,load_vae):
        assert load_vae is False
        return SimpleNamespace(transformer=Model(),scheduler=scheduler(),vae=None),torch.zeros(carrier.SHAPE),torch.tensor(1.),torch.tensor(-1.),torch.float32
    monkeypatch.setattr(runner,'prepare_generation',prepare)
    if failure=='zero_a_backward':
        original_grad=torch.autograd.grad
        attempts=[0]
        def failed_backward(*args,**kwargs):
            attempts[0]+=1
            if attempts[0]==1:raise RuntimeError('injected backward operator OOM')
            return original_grad(*args,**kwargs)
        monkeypatch.setattr(torch.autograd,'grad',failed_backward)
    if failure=='direction':
        original=runner.method.direction_and_amplitude
        calls=[0]
        def broken(*args):
            calls[0]+=1
            if calls[0]==1:raise ValueError('injected zero/nonfinite direction')
            return original(*args)
        monkeypatch.setattr(runner.method,'direction_and_amplitude',broken)
    if failure in ('plus_forward','zero_a_forward'):
        original=runner.tail
        calls=[0]
        def failed_tail(*args,**kwargs):
            calls[0]+=1
            if calls[0]==(3 if failure=='plus_forward' else 1):raise RuntimeError('injected tail OOM')
            return original(*args,**kwargs)
        monkeypatch.setattr(runner,'tail',failed_tail)
    config=json.loads(Path('experiments/wan_state_clock/configs/generate_replication.json').read_text())
    result=runner.run(config,tmp_path/'run')
    assert tuple(result['conditions'])==runner.CONDITIONS
    assert result['artifact_paths']['result_directory']==str((tmp_path/'run').resolve())
    assert result['artifact_paths']['effective_config']==str((tmp_path/'run/config.json').resolve())
    assert all(all(k in row for k in ('elapsed_seconds','resources','endpoint','loss')) for row in result['conditions'].values())
    if failure=='direction':
        assert result['conditions']['PLUS_A']['status']=='MISSING_DIRECTION'
        assert result['conditions']['MINUS_A']['status']=='MISSING_DIRECTION'
        assert result['conditions']['MINUS_B']['status']=='COMPLETE'
        assert result['failures']
    elif failure=='plus_forward':
        assert result['conditions']['PLUS_A']['status']=='FAILED_FORWARD'
        assert result['conditions']['MINUS_A']['status']=='COMPLETE'
        assert result['conditions']['MINUS_B']['status']=='COMPLETE'
        assert any('injected tail OOM' in row['error'] for row in result['failures'])
    elif failure=='zero_a_backward':
        assert result['conditions']['ZERO_A']['status']=='FAILED_AFTER_FORWARD'
        assert result['conditions']['ZERO_B']['status']=='COMPLETE'
        assert result['conditions']['PLUS_A']['status']=='MISSING_DIRECTION'
        assert result['conditions']['MINUS_B']['status']=='COMPLETE'
        stages=[r['stage'] for r in result['conditions']['ZERO_A']['resource_snapshots']]
        assert stages==['forward_complete','before_backward','failure_with_traceback_live',
                        'condition_finally_after_graph_release','after_condition_return']
        assert 'failed_backward' in result['failures'][0]['traceback']
        assert 'injected backward operator OOM' in result['failures'][0]['traceback']
    elif failure=='zero_a_forward':
        assert result['conditions']['ZERO_A']['status']=='FAILED_FORWARD'
        assert result['conditions']['ZERO_B']['status']=='COMPLETE'
        assert result['conditions']['PLUS_B']['status']=='MISSING_DIRECTION'
        assert result['conditions']['PLUS_B']['resources'] is None
        assert result['zero_repeat_floor'] is None and result['R'] is None
    else:
        assert not result['failures']
        for k,n in result['fixed_calls'].items():assert result['actual_calls'][k+'_completed']==n
        assert result['actual_calls']['transformer_replay_completed']==20
        assert result['zero_repeat_floor']['terminal_difference']['global_rms']==0
        assert result['zero_repeat_floor']['loss_difference_by_message']==[0,0]
        assert not result['over_budget_conditions'] and not result['response_check_failed_conditions']
        for suffix in ('A','B'):
            plus=torch.load(tmp_path/f'run/PLUS_{suffix}_coefficients.pt',weights_only=True)
            minus=torch.load(tmp_path/f'run/MINUS_{suffix}_coefficients.pt',weights_only=True)
            assert torch.equal(plus,-minus)
            assert result['direction_comparisons'][suffix]['scientific_pass'] is None
    for row in result['conditions'].values():
        if row['status']=='COMPLETE':
            stages=[r['stage'] for r in row['resource_snapshots']]
            assert stages[0]=='forward_complete' and stages[-1]=='after_condition_return'
    assert all(isinstance(row['traceback'],str) for row in result['failures'])
    persisted=json.loads((tmp_path/'run/result.json').read_text())
    assert persisted['actual_calls']==result['actual_calls']


def test_bf16_checkpoint_and_zero_forward_repeat(setup):
    _,d,c,s,z=setup
    a=torch.zeros((3,1760),requires_grad=True)
    rows=[]
    out,counts=run_tail(setup,a,True,record=lambda row,arrays:rows.append(row),input_dtype=torch.bfloat16)
    g=torch.autograd.grad(method.terminal_loss(out,d,c[0]),a)[0]
    b=torch.zeros_like(a,requires_grad=True)
    native,_=run_tail(setup,b,False,input_dtype=torch.bfloat16)
    native_g=torch.autograd.grad(method.terminal_loss(native,d,c[0]),b)[0]
    torch.testing.assert_close(out,native,rtol=0,atol=0)
    torch.testing.assert_close(g,native_g,rtol=0,atol=0)
    assert torch.isfinite(g).all() and bool((g.abs().sum(dim=1)>0).all())
    assert all(r['cfg_dtype']=='torch.bfloat16' and r['scheduler_dtype']=='torch.float32' for r in rows)
    assert s.step_index==44 and counts['scheduler_step_completed']==6
    # No finite-difference agreement assertion: BF16 AD is only a surrogate.


def test_fp32_effective_control_rounding_and_swallowing(setup):
    rows=[]
    with torch.no_grad():
        run_tail(setup,torch.full((3,1760),1e-12),record=lambda row,arrays:rows.append((row,arrays)))
    assert rows[0][0]['swallowed_fraction']>.9
    for row,arrays in rows:
        assert row['requested_U']['support_rms']>0
        torch.testing.assert_close(arrays['U'],-row['sigma']*arrays['effective_delta_velocity'],rtol=0,atol=0)
        assert row['addition_rounding_error']['support_rms']>0


def test_zero_nonfinite_direction_explicit(setup):
    _,d,_,s,_=setup
    for g in (torch.zeros((3,1760)),torch.full((3,1760),float('nan'))):
        with pytest.raises(ValueError):method.direction_and_amplitude(g,d,[.3,.25,.2],[-.05]*3,.02)


def test_prepare_generation_optional_vae_preserves_default(monkeypatch):
    import diffusers
    from runtime.wan import generation
    calls=[]
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1,dtype=torch.bfloat16))
            self.config=SimpleNamespace(in_channels=16)
    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
            self.config=SimpleNamespace(scale_factor_temporal=4,scale_factor_spatial=8)
    def from_pretrained(model,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(config=SimpleNamespace(),text_encoder=torch.nn.Identity(),transformer=Transformer(),
            vae=VAE(),vae_scale_factor_temporal=4,vae_scale_factor_spatial=8,scheduler=scheduler(),
            encode_prompt=lambda **kwargs:(torch.ones(1),torch.zeros(1)),
            prepare_latents=lambda *args:torch.zeros(carrier.SHAPE))
    loaded=[]
    def load(config):loaded.append(True);return VAE()
    monkeypatch.setattr(diffusers.WanPipeline,'from_pretrained',from_pretrained)
    monkeypatch.setattr(generation,'load_frozen_vae',load)
    real_device=torch.device
    monkeypatch.setattr(torch,'device',lambda value:real_device('cpu' if value=='cuda' else value))
    config=json.loads(Path('experiments/wan_state_clock/configs/generate_replication.json').read_text())
    default=generation.prepare_generation(config)
    assert default[0].vae is not None and loaded==[True] and 'vae' not in calls[0]
    without=generation.prepare_generation(config,load_vae=False)
    assert without[0].vae is None and loaded==[True] and calls[1]['vae'] is None
    assert torch.equal(default[1],without[1])


def test_notebook_has_pinned_source_and_fixed_runner():
    root=Path('.')
    nb=json.loads((root/'notebooks/velocity_direction_colab.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    source=''.join(nb['cells'][2]['source'])
    assert "SOURCE_COMMIT = '5beff27c5aed479f7f4209ca42b77f61e6d9997e'" in source
    assert "SOURCE_URL = 'https://github.com/RICHAAARC/SC-SSTW.git'" in source
    assert "'fetch', '--depth', '1', 'origin', SOURCE_COMMIT" in source
    assert "'checkout', '--detach', SOURCE_COMMIT" in source
    assert 'if ACTUAL_SOURCE_COMMIT != SOURCE_COMMIT:' in source
    assert 'PAYLOAD' not in json.dumps(nb) and 'base64' not in json.dumps(nb)
    launch=''.join(nb['cells'][3]['source'])
    assert "BASE_CONFIG = SOURCE / 'experiments/wan_state_clock/configs/generate_replication.json'" in launch
    assert "'archive', '--format=zip', '--output', str(SOURCE_ARCHIVE), SOURCE_COMMIT" in launch
    original=json.loads((root/'experiments/wan_state_clock/configs/generate_replication.json').read_text())
    config=copy.deepcopy(original)
    env={'config':config,'SOURCE_URL':'https://github.com/RICHAAARC/SC-SSTW.git','ACTUAL_SOURCE_COMMIT':'5beff27c5aed479f7f4209ca42b77f61e6d9997e'}
    assignments=[n for n in ast.parse(launch).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Subscript) and ast.unparse(n.targets[0]).startswith('config[') and ast.unparse(n.targets[0])!="config['artifact_paths']"]
    exec(compile(ast.Module(body=assignments,type_ignores=[]),'config-overrides','exec'),env)
    assert config['source_commit']==env['ACTUAL_SOURCE_COMMIT']
    assert config['source_url']==env['SOURCE_URL']
    assert config['generation']['role']=='shared_0_43_then_two_zero_and_four_signed_real_terminal_tails'
    assert config['output_drive_parent']=='/content/drive/MyDrive/Video-WM/VelocityDirection'
    for key in ('source_snapshot','source_commit','source_url'):config.pop(key)
    config['generation']['role']=original['generation']['role']
    config['output_drive_parent']=original['output_drive_parent']
    assert config==original
    assert 'experiments.wan_state_clock.velocity_direction_run' in launch
    assert "OUTPUT = Path('/content/drive/MyDrive/Video-WM/VelocityDirection') / RUN_ID" in launch
    assert "LOG = OUTPUT.parent / f'{RUN_ID}.launcher.log'" in launch
    assert "SOURCE_ARCHIVE = OUTPUT.parent / f'{RUN_ID}.source.zip'" in launch
    assert "config['artifact_paths'] = {'launcher_log': str(LOG), 'source_archive': str(SOURCE_ARCHIVE), 'source_directory': str(SOURCE)}" in launch
