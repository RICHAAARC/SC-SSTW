from __future__ import annotations
import json
import pytest
import torch
from main.tube_state import local_joint_same_point_dual_v1 as method
from runtime.wan import local_joint_same_point_dual_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import vae as adapter
from tests.test_local_joint_readout_m0_v1 import TinyWan,PROTOCOL,SHAPE,SUPPORT

pytestmark=pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def synthetic_read(gap=.005,tie=False):
    m=torch.tensor([-.2,.01,.03,0.]*8,dtype=torch.double,requires_grad=True).reshape(4,8)
    c=torch.full((22,),-.1,dtype=torch.double)
    c[0]=gap;c[5]=0
    if tie:c[15]=0
    c.requires_grad_()
    actual=c[0]-c[1:].max()
    return dict(margins=m,correlations=c,objectives=actual.reshape(1),active=dict(other_offsets=[5,15] if tie else [5]))


def test_exact_scalar_derivatives_and_different_tie_semantics():
    read=synthetic_read()
    mean=method.scalar("NEXT_MEAN",read)
    gm,gc=torch.autograd.grad(mean,(read["margins"],read["correlations"]),retain_graph=True)
    expected=torch.zeros(32,dtype=torch.double);expected[list(method.NONPOSITIVE_ROSTER)]=1/16
    torch.testing.assert_close(gm.flatten(),expected,rtol=0,atol=0)
    assert gc[0]==1 and gc[5]==-1 and gc[16]==0
    loss=method.scalar("NEXT_ALLBIT",read)
    pm,sc=torch.autograd.grad(loss,(read["margins"],read["correlations"]))
    torch.testing.assert_close(pm.flatten()[:4],torch.tensor([-.025,-.0125,0,-.025],dtype=torch.double))
    assert sc[0]==pytest.approx(-.1) and sc[5]==pytest.approx(.1)
    for name in method.ARMS:
        with pytest.raises(method.UndefinedActiveTie):method.scalar(name,synthetic_read(tie=True))
    inactive=synthetic_read(gap=.01,tie=True)
    loss=method.scalar("NEXT_ALLBIT",inactive)
    pm,sc=torch.autograd.grad(loss,(inactive["margins"],inactive["correlations"]))
    assert torch.count_nonzero(pm)>0 and torch.count_nonzero(sc)==0
    with pytest.raises(method.UndefinedActiveTie):method.scalar("NEXT_MEAN",inactive)


def test_step_sign_zero_and_original_ball_without_old_half_projection():
    base=torch.zeros(SHAPE);start=base.clone()
    g=runtime.masked(torch.ones_like(base),SUPPORT)
    plus,mp=method.construct("NEXT_MEAN",base,start,g,SUPPORT)
    minus,ma=method.construct("NEXT_ALLBIT",base,start,g,SUPPORT)
    torch.testing.assert_close(plus,-minus)
    assert mp["actual_increment_l2"]==pytest.approx(.1,abs=1e-7)
    assert ma["predicted_scalar_change"]<0 and mp["predicted_scalar_change"]>0
    value,row=method.construct("NEXT_MEAN",base,start,g*0,SUPPORT)
    assert value is None and row["status"]=="ZERO_GRADIENT"
    at_edge=.99*g/g.norm()
    with pytest.raises(ValueError,match="radius"):
        method.construct("NEXT_MEAN",base,at_edge,g,SUPPORT)


def setup(tmp_path):
    source=tmp_path/"source";source.mkdir()
    base=torch.randn(SHAPE,generator=torch.Generator().manual_seed(4))*.2
    d=runtime.masked(torch.randn(SHAPE,generator=torch.Generator().manual_seed(5)),SUPPORT)
    start=base+.1*d/d.norm()
    torch.save(dict(terminal=start),source/"candidate_unit.pt")
    for name,z in (("BASE",base),("SUM_STEP",start)):
        folder=source/name;folder.mkdir()
        torch.save(z,folder/"terminal_latent.pt")
        rgb=adapter.decode_normalized_latent(TinyWan(),z)
        torch.save(rgb,folder/"float_rgb.pt")
        read=method.tensor_readout(rgb,key=runtime.KEY,message=bytes.fromhex("8001a55a"),protocol=PROTOCOL)
        report=method.read_report("NEXT_MEAN",read)
        runtime.write_json(folder/"readout.json",report)
    history=source/"history.json";runtime.write_json(history,report)
    config=dict(schema="local-joint-same-point-dual-v1",model={},device="cpu",
        carrier=dict(key=runtime.KEY,message_hex="8001a55a",rho=.5,cap=1.),step=.1,arms=list(method.ARMS),
        inputs=dict(sum_run=str(source),historical_allbit_metrics=str(history)))
    return config,base,start


def invoke(config,output,model):
    return runtime.run(config,output,loader=lambda *_a,**_k:model,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)


def test_complete_dual_budget_same_point_quality_and_report_only_resume(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config,base,start=setup(tmp_path);out=tmp_path/"out";model=TinyWan()
    result=invoke(config,out,model)
    assert result["status"]=="COMPLETE",result["failures"]
    assert model.calls==4
    assert result["counts"]["vae_load_completed"]==1
    assert result["counts"]["decoder_vjp_completed"]==2
    assert result["counts"]["decode_completed"]==4
    assert result["ordinary_chunks"]==dict(attempted=92,completed=92)
    assert result["physical_replay_chunks"]["forward"]==dict(attempted=92,completed=92)
    assert result["physical_replay_chunks"]["recompute"]["attempted"]==92
    assert result["observation_accounting"]["four_endpoints"]["observed_metrics"]==220
    assert result["observation_accounting"]["repeated_S_gradient_forwards"]["observed_metrics"]==110
    for name in method.ARMS:
        unit=runtime.tensor(out/"units"/(name+".pt"))
        assert torch.equal(unit["input_terminal"],start)
        candidate=runtime.tensor(out/name/"terminal_latent.pt")
        assert float((candidate-base).double().norm())<=1
        assert float((candidate-start).double().norm())==pytest.approx(.1,abs=1e-6)
        for ref in ("BASE","SUM_STEP"):
            q=resume.read_json(out/"quality"/(name+"_vs_"+ref)/"quality.json")
            assert len(q["framewise"])==181 and len(q["temporal_framewise"])==180 and q["peak_events"]
    report=resume.read_json(out/"comparison.json")
    assert all(len(r["payload_rows"])==32 and len(r["state_rows"])==22 for r in report["views"].values())
    assert not runtime.model_work_remaining(out)
    (out/"comparison.json").unlink()
    (out/"quality/NEXT_MEAN_vs_BASE/frames/frame_001.png").unlink()
    scoring_calls=[]
    def forbidden_scoring(*_a,**_k):
        scoring_calls.append(True)
        raise AssertionError("completed scoring must not execute again")
    monkeypatch.setattr(runtime.observer,"evaluate_saved",forbidden_scoring)
    model=TinyWan();again=invoke(config,out,model)
    assert again["status"]=="COMPLETE" and model.calls==0
    assert scoring_calls==[]
    assert not any(again["current_attempt_counts"].values())
    assert (out/"quality/NEXT_MEAN_vs_BASE/frames/frame_001.png").exists()
    metrics_path=out/"NEXT_ALLBIT/metrics.json";metrics_bytes=metrics_path.read_bytes();metrics_path.unlink()
    model=TinyWan();missing=invoke(config,out,model)
    assert missing["status"]=="ENGINEERING_FAILURE" and model.calls==0 and scoring_calls==[]
    assert any("committed scoring missing/invalid" in r["reason"] for r in missing["failures"])
    metrics_path.write_bytes(metrics_bytes)
    # Losing a completed gradient must not silently issue a replacement VJP.
    (out/"units/NEXT_MEAN.pt").unlink()
    assert not runtime.model_work_remaining(out)
    model=TinyWan();broken=invoke(config,out,model)
    assert broken["status"]=="ENGINEERING_FAILURE" and model.calls==0
    assert any("committed gradient missing" in r["reason"] for r in broken["failures"])


def test_missing_reference_rgb_retains_four_pairs_and_no_hidden_decode(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config,_,_=setup(tmp_path)
    (tmp_path/"source/BASE/float_rgb.pt").unlink()
    model=TinyWan();result=invoke(config,tmp_path/"out",model)
    assert result["status"]=="COMPLETE_WITH_MISSING_EVIDENCE" and model.calls==4
    assert len(result["quality"])==4
    assert all(result["quality"][n+"_vs_BASE"]["status"]=="MISSING" for n in method.ARMS)
    assert all(result["quality"][n+"_vs_SUM_STEP"]["status"]=="SCORED" for n in method.ARMS)


def test_interrupted_second_gradient_replays_only_uncommitted_unit(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config,_,_=setup(tmp_path);out=tmp_path/"out";atomic=resume.atomic_tensor
    def interrupt(path,value):
        if str(path).endswith("units/NEXT_ALLBIT.pt"):raise KeyboardInterrupt("expiry before commit")
        return atomic(path,value)
    monkeypatch.setattr(resume,"atomic_tensor",interrupt)
    with pytest.raises(KeyboardInterrupt):invoke(config,out,TinyWan())
    state=runtime.read_state(out)
    assert state["status"]=="INTERRUPTED" and state["counts"]["decoder_vjp_completed"]==2
    monkeypatch.setattr(resume,"atomic_tensor",atomic)
    model=TinyWan();result=invoke(config,out,model)
    assert result["status"]=="COMPLETE" and model.calls==3
    assert result["counts"]["decoder_vjp_completed"]==3
    assert result["current_attempt_counts"]["decoder_vjp_completed"]==1
    assert result["physical_replay_chunks"]["forward"]["attempted"]==138
    assert result["current_attempt_replay_chunks"]["forward"]["attempted"]==46
    assert result["failure_history"]


def test_failed_arm_does_not_remove_other_arm_or_fixed_rows(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config,_,_=setup(tmp_path);out=tmp_path/"out"
    result=invoke(config,out,TinyWan(fail_decode=1))
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["views"]["NEXT_ALLBIT"]["status"]=="SCORED"
    assert result["counts"]["decode_attempted"]==3
    report=resume.read_json(out/"comparison.json")
    assert len(report["views"]["NEXT_MEAN"]["payload_rows"])==32
    assert len(report["views"]["NEXT_MEAN"]["state_rows"])==22


def test_shared_load_failure_is_not_automatically_retried(tmp_path):
    config,_,_=setup(tmp_path)
    def fail(*_a,**_k):raise RuntimeError("unavailable")
    result=runtime.run(config,tmp_path/"out",loader=fail,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["vae_load_attempted"]==1 and result["counts"]["decode_attempted"]==0


def test_zero_gradients_commit_observations_and_resume_without_endpoint_decode(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config,_,_=setup(tmp_path);out=tmp_path/"out"
    monkeypatch.setattr(method,"scalar",lambda name,read:read["margins"].sum()*0)
    model=TinyWan();result=invoke(config,out,model)
    assert result["status"]=="COMPLETE_WITH_UNDEFINED_ARMS" and model.calls==2
    assert all(result["units"][n]["status"]=="ZERO_GRADIENT" for n in method.ARMS)
    assert all(result["views"][n]["status"]=="MISSING" for n in method.ARMS)
    assert result["observation_accounting"]["repeated_S_gradient_forwards"]["observed_metrics"]==110
    assert not runtime.model_work_remaining(out)
    model=TinyWan();again=invoke(config,out,model)
    assert again["status"]=="COMPLETE_WITH_UNDEFINED_ARMS" and model.calls==0
