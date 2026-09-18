"""Fixed late window: fake model, native scheduler, no model downloads."""
import copy
from types import SimpleNamespace
import pytest
import torch
from test_grow_control_transfer import native,Model
from main.tube_state import grow_frequency as method
from runtime.wan import grow_late_control as rt
from experiments.wan_state_clock import grow_late_control_run as runner

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def test_late_native_count_endpoint_and_old_unchanged(monkeypatch,native):
    runner.validate(runner.load(runner.MANIFEST))
    assert method.CONTROL_INDICES==tuple(range(10,30))
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12))
    z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(4))*.2
    model=Model();s=native();pipe=SimpleNamespace(transformer=model,scheduler=s)
    calls=[];rows=[];artifacts={}
    output=rt.generate(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,method.codebook(b'test'),0,
        lambda k,d:calls.append((k,d)),rows.append,lambda k,v:artifacts.update({k:v.clone()}),control_indices=rt.CONTROL_INDICES)
    assert calls.count(('transformer',True))==100 and calls.count(('scheduler_step',True))==50
    assert calls.count(('response_probe_step',True))==20 and calls.count(('local_gradient',True))==20
    assert [r['index'] for r in rows if r['controlled']]==list(range(30,50))
    assert rows[-1]['sigma']>0 and rows[-1]['next_sigma']==0 and rows[-1]['terminal_lower_order']==1
    assert rows[-1]['terminal_vs_last_controlled_clean_rms']<1e-6
    torch.testing.assert_close(output,artifacts['last49_predicted_clean_after'])
    assert all('actual_after' in r and r['control_induced_delta_rms']>=0 for r in rows if r['controlled'])
    assert not any(model.grad_enabled) and model.weight.grad is None


def test_invalid_control_sigma_and_manifest_fail(monkeypatch,native):
    bad=runner.load(runner.MANIFEST);bad['frequency']['controlled_indices']=list(range(10,30))
    with pytest.raises(ValueError):runner.validate(bad)
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12));s=native();s.sigmas[49]=0
    with pytest.raises(ValueError,match='no skipping'):
        rt.generate(SimpleNamespace(transformer=Model(),scheduler=s),torch.zeros(method.SHAPE),torch.tensor(1.),torch.tensor(-1.),
            torch.float32,5.,method.codebook(b'test'),0,lambda *x:None,lambda x:None,lambda *x:None,control_indices=rt.CONTROL_INDICES)


def test_gate_requires_all_eight_and_no_erasures():
    cases={c:runner.empty('EXECUTION_COMPLETE')|{'exit_code':0} for c in runner.CASES}
    for case in cases.values():
        for arm in ('A','B'):
            case['videos'][arm].update(status='COMPLETE',terminal={'status':'COMPLETE','aggregate':{'bit_erasures':0},
                'payload_comparisons_reporting_only':{'aggregate':[{'exact_payload_match':arm=='A'},{'exact_payload_match':arm=='B'}]}})
    assert runner.media_gate(cases)['media_eligible']
    cases[runner.CASES[0]]['exit_code']=1
    assert runner.media_gate(cases)['exact_no_erasure']==6
    cases[runner.CASES[0]]['exit_code']=0
    cases[runner.CASES[-1]]['videos']['B']['terminal']['aggregate']['bit_erasures']=1
    assert runner.media_gate(cases)['exact_no_erasure']==7
    assert not runner.media_gate({})['media_eligible']


def test_late_matches_original_runtime_except_indices(monkeypatch,native):
    from runtime.wan.grow_frequency import generate as original
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12))
    z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(14))*.2
    book=method.codebook(b'oracle');args=(z,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,book,0,lambda *x:None,lambda x:None)
    actual=rt.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*args,lambda *x:None,control_indices=rt.CONTROL_INDICES)
    monkeypatch.setattr(method,'CONTROL_INDICES',rt.CONTROL_INDICES)
    expected=original(SimpleNamespace(transformer=Model(),scheduler=native()),*args)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_failure_roster_retained(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    result=runner.run_all(tmp_path/'all')
    assert result['video_denominator']==12 and len(result['cases'])==4
    assert all(len(c['videos'])==3 for c in result['cases'].values())
    assert not result['media_gate']['media_eligible']


def test_fake_case_terminal_only_counts(tmp_path,monkeypatch,native):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12))
    pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(4))*.2
    monkeypatch.setattr(runner,'prepare_generation',lambda *a,**kw:(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    result=runner.run_case(runner.CASES[0],tmp_path/'case')
    assert result['status']=='EXECUTION_COMPLETE'
    assert result['actual_calls']['transformer_completed']==300
    assert result['actual_calls']['scheduler_step_completed']==150
    assert result['actual_calls']['local_gradient_completed']==40
    assert result['actual_calls']['response_probe_step_completed']==40
    assert len(result['videos'])==3 and (tmp_path/'case/latents/OFF_terminal.pt').exists()
    assert result['historical_comparison']['status']=='NOT_ESTABLISHED'
    assert result['tensor_fingerprints']['initial'] and result['schedule_sha256']
