"""CPU/fake implementation checks only; no real model or scientific execution."""
from types import SimpleNamespace
import math
import pytest,torch
from main.tube_state import projection_margin as carrier,response_selection as decision
from runtime.wan import tube_response_selection as runtime
from experiments.wan_state_clock import flow_tube_response_selection_run as run
import test_flow_tube_state_guidance as fixture
from test_flow_tube_state_guidance import native,book,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

def test_decision_fixed_eligibility_tie_skip_missing_and_units():
    assert decision.select(1.,.8,1.2,.1)['sign']==1
    assert decision.select(1.,1.2,.8,.1)['sign']==-1
    assert decision.select(1.,1.,1.2,.1)['status']=='SKIP_NONIMPROVING'
    assert decision.select(1.,.8,.8-1e-7,.1)['status']=='SELECT_PLUS_TIE'
    # Only minus eligible: never allow tie preference to choose ineligible plus.
    assert decision.select(1.,1.-.75e-6,1.-1.25e-6,.1)['sign']==-1
    row=decision.select(1.,.8,1.2,.1)
    assert row['central_difference']==pytest.approx(-2.)
    assert row['nonlinear_symmetric_difference']==pytest.approx(0.)
    assert decision.select(1.,1.,1.,0.,True)['status']=='ZERO_DIRECTION'
    assert decision.select(1.,1.,1.,0.,True)['central_difference'] is None
    allplus={c:{'response_decisions':{k:decision.select(1.,.8,1.2,.1) for k in ('A','B')}} for c in run.CASES}
    assert run.selection_summary(allplus)['all_plus_degenerates_to_LOCAL'] is True
    for args in ((1.,None,1.,.1),(1.,float('nan'),1.,.1),(1.,.8,.9,0.)):
        with pytest.raises(ValueError):decision.select(*args)

def test_native_same_history_sigma_and_fresh_signed_replays(native,book):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());counts=[];count=lambda *v:counts.append(v)
    prompt=torch.tensor(1.);negative=torch.tensor(-1.)
    nodes,snapshots,_=runtime.reference(pipe,torch.zeros(carrier.SHAPE),prompt,negative,torch.float32,5.,count)
    snapshot=snapshots[46];z=nodes[46]['z'];v=nodes[46]['v'];before=runtime.fingerprint(vars(snapshot))
    raw,_,_=run.method.correction(nodes[46]['clean'].numpy(),book,0)
    q,epsilon,d=runtime.prepare_direction(snapshot,z,v,nodes[47]['z'],torch.from_numpy(raw),.01,count)
    assert d['unit_direction']['support_rms']==pytest.approx(1.,rel=1e-6)
    plus,pr,_,pa=runtime.replay(pipe,snapshot,z,v,nodes[47]['z'],q,epsilon,1,.01,nodes,prompt,negative,torch.float32,5.,book,count)
    minus,mr,_,ma=runtime.replay(pipe,snapshot,z,v,nodes[47]['z'],q,epsilon,-1,.01,nodes,prompt,negative,torch.float32,5.,book,count)
    formal,fr,_,_=runtime.replay(pipe,snapshot,z,v,nodes[47]['z'],q,epsilon,1,.01,nodes,prompt,negative,torch.float32,5.,book,count)
    torch.testing.assert_close(plus,formal,rtol=0,atol=0)
    assert formal.data_ptr()!=plus.data_ptr()
    assert pr['actual_D']['support_rms']==pytest.approx(.01,rel=1e-5)
    torch.testing.assert_close(pa['actual_D'],-ma['actual_D'],rtol=1e-4,atol=1e-7)
    assert float((pa['actual_D']*q).sum())>0
    assert pr['history_fingerprint']==mr['history_fingerprint']==fr['history_fingerprint']==before==runtime.fingerprint(vars(snapshot))
    assert not any(pipe.transformer.flags) and not plus.requires_grad and not minus.requires_grad

def test_complete_generation_independent_formal_budget_and_media(monkeypatch,native,tmp_path):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case'
    calls=[];original=runtime.replay
    def replay(*args,**kw):calls.append(args[7]);return original(*args,**kw)
    monkeypatch.setattr(runtime,'replay',replay)
    g=fixture.generate_fixture(monkeypatch,native,path)
    assert len(calls)==8 and len(g['videos'])==5
    for key in ('transformer','scheduler_step','unit_response_probe_step','terminal_backward'):assert g['actual_calls'][key+'_completed']==run.PLAN[key]
    for suffix in ('A','B'):
        for label in ('plus','minus'):
            pathkey='probe_'+suffix+'_'+label
            assert g['calls_by_path'][pathkey]['transformer_completed']==6
            assert g['calls_by_path'][pathkey]['scheduler_step_completed']==4
            assert math.isfinite(g['elapsed_seconds_by_path'][pathkey])
        for group in ('LOCAL','RESPONSE'):
            item=g['videos'][group+'_'+suffix]
            assert item['formal_replay_equivalence']['maxabs']==0
            assert item['control']['terminal_reused_from_probe'] is False
            assert g['calls_by_path']['formal_'+group+'_'+suffix]['scheduler_step_completed']==4
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path);r=run.media_case(run.CASES[0],path)
    assert r['status']=='EXECUTION_COMPLETE',r['failures']
    assert lengths==[181,177,177,177]*5 and len(blind)==5
    assert all(r['actual_calls'][k+'_completed']==n for k,n in run.PLAN.items())

def test_zero_direction_keeps_probes_formal_media_and_counts(monkeypatch,native,tmp_path):
    monkeypatch.setattr(fixture,'run',run)
    def zero(clean,*a):return clean*0,clean.copy(),{}
    monkeypatch.setattr(run.method,'correction',zero)
    path=tmp_path/'zero';g=fixture.generate_fixture(monkeypatch,native,path)
    for suffix in ('A','B'):
        assert g['response_decisions'][suffix]['status']=='ZERO_DIRECTION'
        for group in ('LOCAL','RESPONSE'):
            item=g['videos'][group+'_'+suffix]
            assert item['control']['epsilon']==0 and item['control']['expected_signed_arm_D_support_rms']==0
            assert item['control']['actual_D']['support_rms']==0
    assert g['actual_calls']['transformer_completed']==148 and g['actual_calls']['scheduler_step_completed']==84
    fixture.media_fixture(monkeypatch,path);r=run.media_case(run.CASES[0],path)
    assert r['status']=='EXECUTION_COMPLETE' and r['actual_calls']['vae_encode_completed']==20

def test_missing_probe_preserves_local_and_fixed_failures(monkeypatch,native,tmp_path):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(run,'prepare_generation',lambda *a,**kw:(pipe,torch.zeros(carrier.SHAPE),torch.tensor(1.),torch.tensor(-1.),torch.float32))
    original=runtime.replay
    def fail_minus(*args,**kw):
        if args[7]==-1:raise RuntimeError('synthetic missing negative probe')
        return original(*args,**kw)
    monkeypatch.setattr(runtime,'replay',fail_minus)
    g=run.generate_case(run.CASES[0],tmp_path/'missing')
    assert g['status']=='WITH_RETAINED_FAILURES'
    assert all(g['videos']['LOCAL_'+s]['status']=='TERMINAL_PERSISTED' for s in ('A','B'))
    assert all(g['videos']['RESPONSE_'+s]['status']=='FAILED' for s in ('A','B'))
    assert all(g['response_decisions'][s]['status']=='INVALID' for s in ('A','B'))
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    r=run.run_all(tmp_path/'root')
    assert r['video_denominator']==10 and r['receiver_encode_denominator']==40
    assert sum(len(v['observations']) for c in r['cases'].values() for v in c['videos'].values())==40
    assert r['selection_summary']['counts']['invalid']==4
    assert r['paired_summary']['fixed_pair_denominator']==4 and r['paired_summary']['terminal_complete']==0
