"""CPU synthetic checks only: objective identity, detached feedback, native sign and complete budget."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest,torch
from main.tube_state import projection_margin as carrier,flow_control,terminal_feedback as objective
from runtime.wan import tube_terminal_feedback as runtime
from experiments.wan_state_clock import flow_tube_terminal_feedback_run as run
import test_flow_tube_state_guidance as fixture
from test_flow_tube_state_guidance import native,book,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

def test_objective_matches_existing_basis_and_analytic_gradient(book):
    z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(19))*.1
    leaf=z.clone().requires_grad_(True)
    for message in (0,1):
        gradient,row=objective.terminal_gradient(leaf,book,message)
        original=flow_control.projection_record(z.numpy(),book,message)
        assert row['loss']==pytest.approx(.5*original['loss'],rel=1e-12)
        blocks=carrier.blocks(z.numpy()).reshape(1760,1024).astype(float);d=book['directions'].astype(float);c=book['codes'][message]
        expected=-(np.maximum(0,1-c*(blocks*d).sum(1))*c)[:,None]*d/1760
        np.testing.assert_allclose(carrier.blocks(gradient.numpy()).reshape(1760,1024),expected,rtol=2e-6,atol=1e-10)
        assert not gradient.requires_grad and leaf.grad is None
        assert torch.count_nonzero(gradient[:,:,0])==torch.count_nonzero(gradient[:,:,45])==0

def test_native_preview_identity_tail_actual_loss_decreases(native,book):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());count=lambda *a:None;prompt=torch.tensor(1.);negative=torch.tensor(-1.)
    nodes,snapshots,_=runtime.reference(pipe,torch.zeros(carrier.SHAPE),prompt,negative,torch.float32,5.,count)
    before=runtime.fingerprint(vars(snapshots[46]));z=nodes[46]['z'];v=nodes[46]['v']
    preview,record=runtime.preview(pipe,snapshots[46],z,v,prompt,negative,torch.float32,5.,count)
    torch.testing.assert_close(preview,nodes[50]['z'],rtol=0,atol=0)
    gradient,_=objective.terminal_gradient(preview,book,0)
    after,s,row,_=runtime.matched_direction(snapshots[46],z,v,nodes[47]['z'],-gradient,gradient,.01,count)
    assert row['unit_direction']['support_rms']==pytest.approx(1.,rel=1e-6)
    assert row['actual_D']['support_rms']==pytest.approx(.01,rel=1e-5)
    assert row['predicted_terminal_loss_decrease_identity_tail']>0
    # Identity continuation is the stated approximation: its real objective must descend.
    baseline_next=nodes[47]['z'];assert objective.value(after,book,0)<objective.value(baseline_next,book,0)
    # Also verify the actual native linear-model free tail rather than only normalization.
    terminal,_=runtime.continue_single(pipe,after,s,nodes,prompt,negative,torch.float32,5.,46,book,count)
    assert objective.value(terminal,book,0)<objective.value(preview,book,0)
    assert runtime.fingerprint(vars(snapshots[46]))==before and not any(pipe.transformer.flags)
    with pytest.raises(ValueError,match='direction'):runtime.matched_direction(snapshots[46],z,v,nodes[47]['z'],torch.zeros_like(z),gradient,.01,count)

def test_five_fixed_arms_preview_cost_and_media(monkeypatch,native,tmp_path):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case';g=fixture.generate_fixture(monkeypatch,native,path)
    assert len(g['videos'])==5
    for key in ('transformer','scheduler_step','unit_response_probe_step','terminal_backward'):assert g['actual_calls'][key+'_completed']==run.PLAN[key]
    assert g['calls_by_path']['shared_terminal_preview']['transformer_completed']==6
    assert g['calls_by_path']['shared_terminal_preview']['scheduler_step_completed']==4
    assert g['equivalence']['preview_vs_OFF']['maxabs']==0
    for suffix in ('A','B'):
        local=g['videos']['LOCAL_'+suffix]['control'];feedback=g['videos']['TERMINAL_FEEDBACK_'+suffix]['control']
        assert local['history_fingerprint']==feedback['history_fingerprint']==g['preview']['history_fingerprint']
        assert local['input_fingerprint']==feedback['input_fingerprint']==g['preview']['input_fingerprint']
        assert local['target_D_support_rms']==feedback['target_D_support_rms']
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path);r=run.media_case(run.CASES[0],path)
    assert r['status']=='EXECUTION_COMPLETE',r['failures']
    assert len(blind)==5 and lengths==[181,177,177,177]*5
    assert all(r['actual_calls'][k+'_completed']==n for k,n in run.PLAN.items())

def test_failures_preserve_media_and_pair_denominators(monkeypatch,native,tmp_path):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case';fixture.generate_fixture(monkeypatch,native,path)
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path,fail='partial_off');r=run.media_case(run.CASES[0],path)
    assert 'OFF' not in reads and r['status']=='WITH_RETAINED_FAILURES'
    assert all(v['status']=='COMPLETE' for a,v in r['videos'].items() if a!='OFF')
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    result=run.run_all(tmp_path/'all')
    assert result['video_denominator']==10 and result['receiver_encode_denominator']==40
    assert result['paired_summary']['fixed_pair_denominator']==4 and len(result['paired_summary']['rows'])==4
    assert all(r['MP4_local_state_margin_gain_vs_LOCAL'] is None for r in result['paired_summary']['rows'])
