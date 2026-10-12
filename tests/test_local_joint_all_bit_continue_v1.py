from __future__ import annotations
import json
from pathlib import Path
import pytest
import torch
from main.tube_state import local_joint_all_bit_continue_v1 as method
from runtime.wan import local_joint_all_bit_continue_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from runtime.wan import vae as adapter
from test_local_joint_all_bit_v1 import inputs as old_inputs, threads
from test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark=pytest.mark.quick


def inputs(tmp_path):
    config=old_inputs(tmp_path)
    base=torch.load(config["inputs"]["terminal_latent"],weights_only=True)
    planned=torch.load(config["inputs"]["two_mean_delta"],weights_only=True)
    start=(base.double()+planned.double()).float()
    path=Path(config["inputs"]["terminal_latent"]).parent/"source_step_2_unit.pt"
    # Only the actual terminal is valid; old input and planned delta must be ignored.
    torch.save(dict(terminal=start,delta=torch.full_like(start,float("nan")),input_terminal=-base),path)
    config.update(schema="local-joint-all-bit-continue-v1",objective=dict(method.PARAMETERS))
    config["inputs"]={"terminal_latent":config["inputs"]["terminal_latent"],"start_step_unit":str(path)}
    return config,base,start


def execute(config,tmp_path,loader=None):
    return runtime.run(config,tmp_path/"out",loader=loader or (lambda *_a,**_k:TinyWan()),
                       protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)


def test_projection_rounding_zero_and_actual_input_arithmetic():
    base=torch.tensor([1.25,-.375],dtype=torch.float32)
    saved,projected,report=method.project_terminal(base,torch.tensor([.6,.8],dtype=torch.float64))
    assert report["nearest_fp32_actual_l2"]==pytest.approx(1.000000023841858)
    assert report["inward_repair_needed"] and report["inward_fp32_components"]==2
    assert float((saved.double()-base.double()).norm())<=1
    assert bool(((saved.double()-base.double()).abs()<=projected.abs()).all())
    interior,_,report=method.project_terminal(base,torch.tensor([.3,.4],dtype=torch.float64))
    assert not report["inward_repair_needed"]
    assert torch.equal(interior,(base.double()+torch.tensor([.3,.4],dtype=torch.float64)).float())
    # First .10 step from r~.8436 is inside; second outward step projects at radius1.
    base=torch.zeros(2,dtype=torch.float32)
    current=torch.tensor([.84360858,0.],dtype=torch.float32)
    first,r1=method.descent_terminal(base,current,torch.tensor([-2.,0.]))
    second,r2=method.descent_terminal(base,first,torch.tensor([-2.,0.]))
    assert r1["cap_scale"]==1 and r2["cap_scale"]<1
    assert float((second-base).double().norm())<=1
    assert r1["step_size"]==.1 and "half_step" not in r1
    none,r=method.descent_terminal(base,current,torch.zeros(2))
    assert none is None and r["status"]=="ZERO_GRADIENT"
    zero,r=method.descent_terminal(base,torch.tensor([.1,0.]),torch.tensor([1.,0.]))
    # The actual FP32 input is used; its tiny difference from exact .1 is retained.
    assert zero[0]==(torch.tensor(.1,dtype=torch.float32).double()-.1).float()
    assert r["status"]=="DEFINED"
    for invalid in (torch.tensor([float("nan"),0.]),torch.tensor([float("inf"),0.])):
        with pytest.raises(ValueError): method.descent_terminal(base,current,invalid)


def test_full_continuation_actual_start_two_fresh_gradients_and_reports(tmp_path):
    config,base,start=inputs(tmp_path)
    result=execute(config,tmp_path)
    assert result["status"]=="COMPLETE",result["failures"]
    assert result["counts"]["decode_completed"]==6 and result["counts"]["decoder_vjp_completed"]==2
    assert result["counts"]["vae_load_completed"]==1
    assert result["ordinary_chunks"]==dict(attempted=184,completed=184)
    assert result["totals"]==dict(expected_windows=352,observed_windows=352,expected_chips=5632,observed_chips=5632,expected_metrics=220,observed_metrics=220)
    assert tuple(result["views"])==method.VIEWS
    out=tmp_path/"out"
    assert torch.equal(torch.load(out/"START_SAVED/terminal_latent.pt",weights_only=True),start)
    current=start
    for k,target in enumerate(("CONT_MID","CONT_FINAL"),1):
        checkpoint=torch.load(out/"gradients"/f"step_{k}_checkpoint.pt",weights_only=True)
        assert torch.equal(checkpoint["input_terminal"],current)
        direct=TinyWan();z=current.clone().requires_grad_();mean,std=adapter._scale_tensors(direct,z)
        rgb=(direct.decode(z*std+mean)[0][0].permute(1,2,3,0)/2+.5).clamp(0,1)
        read=method.tensor_readout(rgb,key=runtime.engine.KEY,message=bytes.fromhex("8001a55a"),protocol=PROTOCOL)
        gradient=runtime.engine.masked(torch.autograd.grad(method.scalar_loss(read),z)[0],SUPPORT)
        torch.testing.assert_close(checkpoint["gradient"],gradient,rtol=0,atol=0)
        # This fixture stays strictly inside the ball; independent unprojected formula.
        candidate=current.double()-base.double()-.1*gradient.double()/gradient.double().norm()
        assert float(candidate.norm())<1
        expected=(base.double()+candidate).float()
        unit=torch.load(out/f"step_{k}_unit.pt",weights_only=True)
        assert torch.equal(unit["terminal"],expected)
        assert unit["delta"].dtype==torch.float64
        assert torch.equal(unit["delta"],expected.double()-base.double())
        assert float(unit["delta"].norm())<=1
        assert unit["report"]["predicted_loss_change"]==float((gradient.double()*(expected.double()-current.double())).sum())
        assert "half_step" not in unit["report"]
        assert result["gradient_readouts"][f"step_{k}"]["loss"]["parameters"]==method.PARAMETERS
        current=expected
    comparison=json.loads((out/"comparison.json").read_text())
    assert comparison["primary_comparison"]["reference"]=="START_SAVED"
    assert comparison["steps"]["step_1"]["reference"]=="START_SAVED"
    assert len(comparison["bits_32"])==32
    tracked=json.loads((out/"tracked_bits.json").read_text())
    assert len(tracked["views"]["CONT_FINAL"]["six_original_negative_bits"])==6
    assert len(tracked["views"]["CONT_FINAL"]["original_26_positive_bits"])==26
    for name in method.VIEWS:
        quality=json.loads((out/name/"quality.json").read_text())
        assert len(quality["framewise"])==181 and len(quality["temporal_framewise"])==180
        images=json.loads((out/name/"frames.json").read_text())
        assert images["frame_indices"]==result["quality_frames"]
        assert all((out/name/item["path"]).is_file() for item in images["frames"]+images["crops"])
        assert result["views"][name]["loss"]["parameters"]==method.PARAMETERS


@pytest.mark.parametrize("point,remaining",[("g1",(1,3,1)),("g2",(1,1,0)),("mid_rgb",(1,2,1)),("final_rgb",(0,0,0))])
def test_continuation_resume_completed_gradients_and_rgb(tmp_path,monkeypatch,point,remaining):
    config,base,start=inputs(tmp_path)
    original=resume.atomic_tensor
    stop_name={"g1":"step_1_checkpoint.pt","g2":"step_2_checkpoint.pt","mid_rgb":"CONT_MID/float_rgb.pt","final_rgb":"CONT_FINAL/float_rgb.pt"}[point]
    def stop(path,value):
        original(path,value)
        if str(path).endswith(stop_name): raise KeyboardInterrupt("expiry after complete unit")
    with monkeypatch.context() as patch:
        patch.setattr(resume,"atomic_tensor",stop)
        first=execute(config,tmp_path)
    assert first["status"]=="INTERRUPTED"
    if point=="g1":
        (tmp_path/"out/result.json").write_text('{"torn":')
    # Resume with all original inputs gone: durable copied inputs must suffice.
    for path in config["inputs"].values(): Path(path).unlink()
    if point=="final_rgb":
        config["device"]="cuda"
        def forbid(*_a,**_k): raise AssertionError("CPU tail must not load/use CUDA")
        loader=forbid
        for name in ("is_available","reset_peak_memory_stats","empty_cache","max_memory_allocated","max_memory_reserved"):
            monkeypatch.setattr(torch.cuda,name,forbid)
        assert not runtime.model_work_remaining(tmp_path/"out")
    else:
        loader=lambda *_a,**_k:TinyWan()
    result=execute(config,tmp_path,loader)
    assert result["status"]=="COMPLETE",result["failures"]
    c=result["current_attempt_counts"]
    assert (c["vae_load_completed"],c["decode_completed"],c["decoder_vjp_completed"])==remaining
    assert result["counts"]["decode_completed"]==6 and result["counts"]["decoder_vjp_completed"]==2
    assert result["totals"]["observed_metrics"]==220
    def forbidden(*_a,**_k): raise AssertionError("completed task is report only")
    monkeypatch.setattr(torch,"load",forbidden)
    again=execute(config,tmp_path,forbidden)
    assert again["status"]=="COMPLETE" and all(v==0 for v in again["current_attempt_counts"].values())


def test_base_reader_failure_retains_quality_and_all_new_steps(tmp_path,monkeypatch):
    config,_,_=inputs(tmp_path)
    original=runtime.engine.observer.read_rgb
    calls=0
    def read(*a,**k):
        nonlocal calls
        calls+=1
        if calls==1: raise OSError("BASE reader failure")
        return original(*a,**k)
    monkeypatch.setattr(runtime.engine.observer,"read_rgb",read)
    result=execute(config,tmp_path)
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decode_completed"]==6 and result["counts"]["decoder_vjp_completed"]==2
    assert result["totals"]["expected_metrics"]==220 and result["totals"]["observed_metrics"]==165
    assert (tmp_path/"out/CONT_FINAL/quality.json").is_file()


@pytest.mark.parametrize("terminal",["zero","tie"])
def test_continuation_terminal_outcomes_do_not_retry(tmp_path,monkeypatch,terminal):
    config,_,_=inputs(tmp_path)
    with monkeypatch.context() as patch:
        if terminal=="zero":
            original=torch.autograd.grad
            patch.setattr(torch.autograd,"grad",lambda *a,**k:(torch.zeros_like(original(*a,**k)[0]),))
        else:
            original=method.loss_summary
            def tied(*a,**k):
                row=original(*a,**k);row["active_state_tie"]=True;return row
            patch.setattr(method,"loss_summary",tied)
        first=execute(config,tmp_path)
    assert first["status"]=="COMPLETE"
    assert first["views"]["CONT_MID"]["status"]==first["views"]["CONT_FINAL"]["status"]=="MISSING"
    assert first["counts"]["decoder_vjp_completed"]==(1 if terminal=="zero" else 0)
    assert not runtime.model_work_remaining(tmp_path/"out")
    def forbidden(*a,**k): raise AssertionError("terminal outcome must not retry")
    monkeypatch.setattr(torch,"load",forbidden)
    result=execute(config,tmp_path,forbidden)
    assert result["status"]=="COMPLETE" and all(x==0 for x in result["current_attempt_counts"].values())


def test_missing_start_keeps_base_and_fixed_missing_slots(tmp_path):
    config,_,_=inputs(tmp_path)
    Path(config["inputs"]["start_step_unit"]).unlink()
    result=execute(config,tmp_path)
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decode_completed"]==1 and result["counts"]["decoder_vjp_attempted"]==0
    assert result["totals"]["expected_metrics"]==220 and result["totals"]["observed_metrics"]==55
    assert result["views"]["START_SAVED"]["status"]=="MISSING"
