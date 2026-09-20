"""Synthetic native scheduler and fixed seven-arm mechanism tests; no real video experiment."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest,torch
from main.tube_state import projection_margin as carrier,terminal_guidance as method,flow_control
from runtime.wan import tube_multistep as rt,tube_terminal_guidance as prior
from experiments.wan_state_clock import flow_tube_multistep_run as run
import test_flow_tube_state_guidance as fixture
from test_flow_tube_state_guidance import native,book,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def test_full_early_request_native_response_and_history(native,book):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());initial=torch.zeros(carrier.SHAPE);count=lambda *a:None
    z,s,_=rt.shared_prefix(pipe,initial,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,count)
    v=rt.velocity(pipe,z,s,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,44,count)
    original=copy.deepcopy(s);finger=rt.fingerprint(vars(original));off=copy.deepcopy(s).step(v,s.timesteps[44],z.clone(),return_dict=False)[0]
    clean=z-float(s.sigmas[44])*v;u,_,_=method.correction(clean.numpy(),book,0);old_u,_=flow_control.request(clean.numpy(),book,0)
    np.testing.assert_allclose(u,3*old_u,rtol=2e-5,atol=1e-7)
    actual,row=rt.advance(s,z,v,book,0,44,count)
    expected_D=rt.measures(actual-off)
    assert row['actual_D']==expected_D and row['u']==rt.measures(torch.from_numpy(u))
    assert not torch.allclose(actual-off,torch.from_numpy(u),rtol=1e-4,atol=1e-7)
    assert rt.fingerprint(vars(original))==finger and s.step_index==45
    assert row['requested_minimum_margin']>1-1e-5 and not actual.requires_grad


def test_seven_arm_shared_forks_baseline_equivalence_and_media(monkeypatch,native,tmp_path,book):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case';g=fixture.generate_fixture(monkeypatch,native,path)
    assert len(g['videos'])==7 and all(g['actual_calls'][k+'_completed']==run.PLAN[k] for k in ('transformer','scheduler_step','shadow_step'))
    for suffix in ('A','B'):
        a=g['videos']['EARLY_ONLY_'+suffix];b=g['videos']['EARLY_LAST_'+suffix]
        assert a['pre49_input_fingerprint']==b['pre49_input_fingerprint'] and a['pre49_history_fingerprint']==b['pre49_history_fingerprint']
        assert a['cumulative_control']['controlled_steps']==3 and b['cumulative_control']['controlled_steps']==4
        assert not a['final_step']['controlled'] and b['final_step']['controlled']
        assert b['cumulative_control']['actual_D']['support']['sum_rms']==pytest.approx(a['cumulative_control']['actual_D']['support']['sum_rms']+b['final_step']['actual_D']['support_rms'])
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());initial=torch.zeros(carrier.SHAPE)
    z,v,s,_=prior.prepare_last(pipe,initial,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,lambda *a:None)
    expected_off,_=prior.final_step(s,z,v,lambda *a:None)
    torch.testing.assert_close(torch.load(path/'OFF_terminal.pt',weights_only=True),expected_off,rtol=0,atol=0)
    for m,suffix in enumerate(('A','B')):
        u,_,_=method.correction((z-float(s.sigmas[49])*v).numpy(),book,m)
        expected,_=prior.final_step(s,z,v,lambda *a:None,u=torch.from_numpy(u))
        torch.testing.assert_close(torch.load(path/('LAST_'+suffix+'_terminal.pt'),weights_only=True),expected,rtol=0,atol=0)
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path);m=run.media_case(run.CASES[0],path)
    assert m['status']=='EXECUTION_COMPLETE',m['failures']
    assert lengths==[181,177,177,177]*7 and len(blind)==7
    assert all(m['actual_calls'][k+'_completed']==v for k,v in run.PLAN.items())


def test_partial_media_and_fixed_failure_denominators(monkeypatch,native,tmp_path):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case';fixture.generate_fixture(monkeypatch,native,path)
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path,fail='partial_off');r=run.media_case(run.CASES[0],path)
    assert r['status']=='WITH_RETAINED_FAILURES' and 'OFF' not in reads
    assert all(v['status']=='COMPLETE' for k,v in r['videos'].items() if k!='OFF')
    assert all(v['saved_quality_vs_off']['status']=='MISSING_OFF_REFERENCE' for k,v in r['videos'].items() if k!='OFF')
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    root=run.run_all(tmp_path/'full');assert root['video_denominator']==28 and root['receiver_encode_denominator']==112
    assert sum(len(v['observations']) for c in root['cases'].values() for v in c['videos'].values())==112
    assert len(root['mechanism_summary'])==28 and all(v['missing_or_failed']==8 for mode in root['recovery_summary'].values() for v in mode.values())
