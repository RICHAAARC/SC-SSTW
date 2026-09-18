"""CPU native solver + tiny fake Transformer; optional exact official 0.40 source."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from main.tube_state import grow_frequency as method
from runtime.wan import grow_control_transfer as rt
from experiments.wan_state_clock import grow_control_transfer_run as runner

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


@pytest.fixture
def native(monkeypatch):
    import diffusers
    source=os.environ.get('DIFFUSERS_UNIPC_SOURCE')
    if source:
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()=='5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872'
        name='diffusers.schedulers._grow_audit_unipc040'
        spec=importlib.util.spec_from_file_location(name,source);module=importlib.util.module_from_spec(spec)
        import sys
        monkeypatch.setitem(sys.modules,name,module);spec.loader.exec_module(module)
        monkeypatch.setattr(diffusers,'UniPCMultistepScheduler',module.UniPCMultistepScheduler)
    def make():
        s=diffusers.UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3,solver_order=2)
        s.set_timesteps(50);return s
    return make


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1));self.config=SimpleNamespace(_commit_hash='fake');self.grad_enabled=[]
    def forward(self,hidden_states,timestep,encoder_hidden_states,**kwargs):
        self.grad_enabled.append(torch.is_grad_enabled())
        return (hidden_states*self.weight+encoder_hidden_states*.001,)


def test_independent_native_response_and_snapshot(native):
    s=native();z=torch.randn((1,2,3,4),generator=torch.Generator().manual_seed(1))
    for i in range(10):z=s.step(z*.1,s.timesteps[i],z,return_dict=False)[0]
    before=rt.fingerprint((z,s.__dict__));calls=[];r=rt.scalar_response(s,lambda k,d:calls.append((k,d)))
    assert before==rt.fingerprint((z,s.__dict__)) and calls.count(('response_probe_step',True))==2
    u=torch.randn(z.shape)*.01;v=z*.1;off,mark=copy.deepcopy(s),copy.deepcopy(s)
    a=off.step(v,s.timesteps[10],z,return_dict=False)[0]
    b=mark.step(v-u/r['sigma'],s.timesteps[10],z,return_dict=False)[0]
    assert rt.check_delta(b-a,r['K_clean']*u,a,b)['pass']
    assert rt.check_delta(mark.last_sample-off.last_sample,r['K_corrected_sample']*u,mark.last_sample,off.last_sample)['pass']
    assert not rt.check_delta(b-a,(r['K_clean']+1)*u,a,b)['pass']


@pytest.mark.parametrize('eta',[.1,0.])
def test_fixed_fake_full_run_and_zero_response_continues(tmp_path,monkeypatch,native,eta):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12));monkeypatch.setattr(method,'ETA',eta)
    model=Model();pipe=SimpleNamespace(transformer=model,scheduler=native())
    z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(4))*.2
    monkeypatch.setattr(runner,'prepare_generation',lambda *a,**kw:(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    result=runner.run(tmp_path/'out')
    assert result['numerical_gate_pass'] and result['status']=='EXECUTION_COMPLETE'
    assert result['actual_calls']['one_step']['transformer_completed']==22
    assert result['actual_calls']['one_step']['scheduler_step_completed']==13
    assert result['actual_calls']['one_step']['response_probe_step_completed']==2
    assert result['actual_calls']['one_step']['local_gradient_completed']==2
    assert result['actual_calls']['tail']['transformer_completed']==234
    assert result['actual_calls']['tail']['scheduler_step_completed']==117
    assert not any(model.grad_enabled) and model.weight.grad is None
    assert all(len(a['terminal']['per_time'])==46 for a in result['arms'].values())
    assert result['scientific_pass'] is None
    check=result['arms']['A']['numerical_checks']['native_transmission']
    assert 'tolerance_over_expected_maxabs' in check
    first=json.loads((tmp_path/'out/observations/A_11.json').read_text())
    assert first['clean_delta_decomposition']['pass']
    assert 'delta_next_model_input' in first and len(first['predicted_clean_delta']['selected_coefficients'])==46
    contrasts=[json.loads((tmp_path/'out/observations'/f'AB_{i:02d}.json').read_text()) for i in range(10,51)]
    assert all(r['status']=='MEASURED' for r in contrasts)
    assert contrasts[1]['clean_delta_decomposition']['pass']
    assert contrasts[-1]['cast_is_actual_model_input'] is False
    pre=torch.load(tmp_path/'out/snapshots/pre10.pt',weights_only=False)
    post=torch.load(tmp_path/'out/snapshots/A_post10.pt',weights_only=False)
    assert pre['scheduler_state']['_step_index']==10 and post['scheduler_state']['_step_index']==11
    assert (tmp_path/'out/conditioning.pt').exists()
    if eta==0:assert check['expected_delta_maxabs']==0 and check['pass']


def test_invalid_independent_response_skips_with_fixed_rows(tmp_path,monkeypatch,native):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12))
    pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(runner,'prepare_generation',lambda *a,**kw:(pipe,torch.zeros(method.SHAPE),torch.tensor(1.),torch.tensor(-1.),torch.float32))
    original=rt.scalar_response
    def corrupt(*args,**kwargs):
        r=original(*args,**kwargs);r['K_clean']+=1;return r
    monkeypatch.setattr(rt,'scalar_response',corrupt)
    result=runner.run(tmp_path/'out')
    assert not result['numerical_gate_pass'] and len(result['arms'])==3
    assert json.loads((tmp_path/'out/observations/AB_11.json').read_text())['status']=='SKIPPED_NUMERICAL_INVALIDITY'
    assert result['actual_calls']['tail']['transformer_attempted']==0
    assert all(a['tail_status']=='SKIPPED_NUMERICAL_INVALIDITY' and len(a['terminal']['per_time'])==46 for a in result['arms'].values())


def test_actual_bfloat16_input_cast_observed_without_model(monkeypatch):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12))
    z=torch.ones(method.SHAPE);delta=torch.full_like(z,1e-4)
    row=rt.transmission(z+delta,z,delta,method.codebook(b'cast'),torch.bfloat16)
    assert row['delta_z']['rms']>0
    assert row['delta_next_model_input']['rms']==0
    assert row['delta_next_model_input']['nonzero_fraction']==0


def test_setup_failure_keeps_all_ab_indices_and_arm_rows(tmp_path,monkeypatch):
    def unavailable(*args,**kwargs):raise RuntimeError('fake unavailable loader')
    monkeypatch.setattr(runner,'prepare_generation',unavailable)
    result=runner.run(tmp_path/'out')
    assert result['status']=='WITH_RETAINED_FAILURES' and len(result['arms'])==3
    rows=[json.loads((tmp_path/'out/observations'/f'AB_{i:02d}.json').read_text()) for i in range(10,51)]
    assert rows[0]['status']=='MISSING_AB_STATE'
    assert all(v['status']=='SKIPPED_NUMERICAL_INVALIDITY' for v in rows[1:])
    assert result['actual_calls']['tail']['transformer_attempted']==0
    assert all(len(a['terminal']['per_time'])==46 for a in result['arms'].values())
