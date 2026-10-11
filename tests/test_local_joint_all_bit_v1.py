from __future__ import annotations

import json
import math

import pytest
import torch

from main.tube_state import local_joint_all_bit_v1 as method
from runtime.wan import local_joint_all_bit_v1 as runtime
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan import vae as adapter
from test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT, inputs as m0_inputs

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(tmp_path):
    config=m0_inputs(tmp_path)
    config["inputs"]["two_mean_delta"]=config["inputs"].pop("bridge_delta")
    config["objective"]=dict(method.PARAMETERS)
    # Saved TWO_MEAN is a cumulative delta, not a unit direction.
    path=config["inputs"]["two_mean_delta"]
    torch.save(.82*torch.load(path,weights_only=True),path)
    return config


def execute(config,tmp_path,model):
    return runtime.run(config,tmp_path/"out",loader=lambda *_a,**_k:model,
                       protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)


def read_fixture(margins,gap,offsets=(2,)):
    return dict(margins=margins,objectives=gap.expand(5),active=dict(other_offsets=list(offsets)),tied=True)


def test_huber_values_boundaries_and_all32_coefficients():
    x=torch.tensor([-.4,0.,.005,.01,.02,.3],dtype=torch.double,requires_grad=True)
    values=method.huber_deficit(x,.02)
    torch.testing.assert_close(values,torch.tensor([.41,.01,.005625,.0025,0.,0.],dtype=torch.double))
    derivative=torch.autograd.grad(values.sum(),x)[0]
    torch.testing.assert_close(derivative,torch.tensor([-1.,-1.,-.75,-.5,0.,0.],dtype=torch.double))
    margins=torch.tensor([-.3,0.,.006972974,.02]+[.03]*28,dtype=torch.double,requires_grad=True)
    gap=torch.tensor(-.1,dtype=torch.double,requires_grad=True)
    read=read_fixture(margins,gap)
    dm,dg=torch.autograd.grad(method.scalar_loss(read),(margins,gap))
    report=method.read_summary(read)
    torch.testing.assert_close(dm,torch.tensor(report["payload_coefficients"],dtype=torch.double))
    assert dm[0] == dm[1] == -.025
    assert dm[2]/dm[0] == pytest.approx(.6513513)
    assert torch.count_nonzero(dm)==3 and dg == -.2
    # Equal margins at any payload-min tie are legal, all 32 receive their own coefficient.
    m=torch.full((4,8),-.1,dtype=torch.double,requires_grad=True)
    g=torch.tensor(.01,dtype=torch.double,requires_grad=True)
    dm,dg=torch.autograd.grad(method.scalar_loss(read_fixture(m,g,(1,2))),(m,g))
    torch.testing.assert_close(dm,torch.full_like(m,-.025))
    assert dg == 0
    with pytest.raises(method.UndefinedActiveTie):
        method.scalar_loss(read_fixture(m,g-.0001,(1,2)))
    # State term resumes below gamma; its C1 boundary derivative is exactly zero.
    assert method.read_summary(read_fixture(m,g-.005))["state_coefficient"] == pytest.approx(-.1)


def test_original_reader_scalar_directional_difference():
    rgb=(.2+.6*torch.rand(PROTOCOL.video_shape,generator=torch.Generator().manual_seed(9))).double().requires_grad_()
    read=method.tensor_readout(rgb,key=runtime.KEY,message=bytes.fromhex("8001a55a"),protocol=PROTOCOL)
    gradient=torch.autograd.grad(method.scalar_loss(read),rgb)[0]
    direction=torch.randn(rgb.shape,generator=torch.Generator().manual_seed(8),dtype=torch.double)
    direction/=direction.norm()
    def value(x):
        return method.scalar_loss(method.tensor_readout(x,key=runtime.KEY,message=bytes.fromhex("8001a55a"),protocol=PROTOCOL))
    h=1e-4
    central=float((value(rgb.detach()+h*direction)-value(rgb.detach()-h*direction))/(2*h))
    assert central == pytest.approx(float((gradient*direction).sum()),abs=1e-9)


def test_descent_budget_cancellation_zero_and_projection():
    first,report=method.descent_delta(torch.zeros(2),torch.tensor([-2.,0.]))
    torch.testing.assert_close(first,torch.tensor([.5,0.]))
    cancel,report=method.descent_delta(first,torch.tensor([2.,0.]))
    assert report["status"]=="DEFINED" and report["cumulative_zero"] and report["cap_scale"]==1
    assert cancel.norm()==0 and report["predicted_loss_change"]==-1.
    second,report=method.descent_delta(first,torch.tensor([0.,3.]))
    assert float(second.norm())==pytest.approx(math.sqrt(.5)) and report["cap_scale"]==1
    projected,report=method.descent_delta(torch.tensor([.500001,0.]),torch.tensor([-1.,0.]))
    assert report["cap_scale"]<1 and float(projected.norm())==pytest.approx(1.)
    none,report=method.descent_delta(first,torch.zeros(2))
    assert none is None and report["status"]=="ZERO_GRADIENT"
    with pytest.raises(FloatingPointError):
        method.descent_delta(first,torch.tensor([float("nan"),0.]))


def test_full_runner_two_fresh_vjps_saved_comparator_and_resources(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config=inputs(tmp_path);model=TinyWan();result=execute(config,tmp_path,model)
    assert result["status"]=="COMPLETE",result["failures"]
    assert result["counts"]==dict(vae_load_attempted=1,vae_load_completed=1,decode_attempted=6,
        decode_completed=6,decoder_vjp_attempted=2,decoder_vjp_completed=2,encode=0,dit=0,native=0,scheduler=0,codec=0)
    assert result["ordinary_chunks"]==dict(attempted=184,completed=184)
    assert result["totals"]==dict(expected_windows=352,expected_chips=5632,expected_metrics=220,
                                   observed_windows=352,observed_chips=5632,observed_metrics=220)
    assert set(result["replay"])=={"step_1","step_2"}
    for replay in result["replay"].values():
        for phase in ("forward","recompute"):
            assert replay["counts"]["vae_chunk"][phase]==dict(attempted=46,completed=46)
        assert replay["boundary_storage"]["closed"]
    assert not list(tmp_path.glob("wan-vae-boundary-*"))
    out=tmp_path/"out"
    old=torch.load(config["inputs"]["two_mean_delta"],weights_only=True)
    torch.testing.assert_close(torch.load(out/"TWO_MEAN_SAVED/delta.pt",weights_only=True),old,rtol=0,atol=0)
    z0=torch.load(config["inputs"]["terminal_latent"],weights_only=True)
    previous=torch.zeros_like(z0)
    for k,name in enumerate(("ALLBIT_HALF","ALLBIT_TWO"),1):
        g=torch.load(out/"gradients"/f"step_{k}.pt",weights_only=True)
        # Direct causal CPU decoder gives an independent reference for each fresh VJP.
        direct=TinyWan();z=(z0+previous).requires_grad_();mean,std=adapter._scale_tensors(direct,z)
        decoded=direct.decode(z*std+mean)[0]
        rgb=(decoded[0].permute(1,2,3,0)/2+.5).clamp(0,1)
        read=method.tensor_readout(rgb,key=runtime.KEY,message=bytes.fromhex("8001a55a"),protocol=PROTOCOL)
        expected_g=runtime.masked(torch.autograd.grad(method.scalar_loss(read),z)[0],SUPPORT)
        torch.testing.assert_close(g,expected_g,rtol=0,atol=0)
        expected,report=method.descent_delta(previous,g)
        actual=torch.load(out/name/"delta.pt",weights_only=True)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        terminal=torch.load(out/name/"terminal_latent.pt",weights_only=True)
        assert result["views"][name]["displacement"]["actual_terminal_minus_base_l2"]==float((terminal.double()-z0.double()).norm())
        assert actual.double().norm()<=1+1e-7
        read_report=result["gradient_readouts"][f"step_{k}"]
        assert read_report["original_metric_count"]==55 and len(read_report["values"]["margins"])==32
        assert read_report["max_abs_difference_55"]<1e-12
        previous=actual
    assert not torch.equal(torch.load(out/"ALLBIT_HALF/delta.pt",weights_only=True),old*.5)
    comparison=json.loads((out/"comparison.json").read_text())
    assert len(comparison["bits_32"])==32
    assert comparison["primary_comparison"]["reference"]=="TWO_MEAN_SAVED"
    assert comparison["primary_comparison"]["target"]=="ALLBIT_TWO"
    for name in method.VIEWS:
        q=json.loads((out/name/"quality.json").read_text())
        assert len(q["framewise"])==181 and len(q["temporal_framewise"])==180
        images=json.loads((out/name/"frames.json").read_text())
        assert images["frame_indices"]==result["quality_frames"]
        for item in images["frames"]+images["crops"]:
            assert (out/name/item["path"]).is_file()
        assert all("max_abs" in row for row in q["framewise"])


@pytest.mark.parametrize("which",[1,2])
def test_zero_gradient_retains_fixed_slots(tmp_path,monkeypatch,which):
    original=torch.autograd.grad;calls=0
    def inject(*a,**kw):
        nonlocal calls
        calls+=1;value=original(*a,**kw)[0]
        return (torch.zeros_like(value) if calls==which else value,)
    monkeypatch.setattr(torch.autograd,"grad",inject)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"]=="COMPLETE" and result["direction_status"]=="ZERO_GRADIENT"
    assert result["counts"]["decoder_vjp_completed"]==which
    assert result["counts"]["decode_completed"]==2*which+1
    assert result["totals"]["observed_metrics"]==55*(which+1)
    assert result["totals"]["expected_metrics"]==220
    assert result["views"]["ALLBIT_TWO"]["status"]=="MISSING"


def test_saved_comparator_missing_does_not_gate_new_two_steps(tmp_path):
    config=inputs(tmp_path);config["inputs"]["two_mean_delta"]=str(tmp_path/"absent"/"delta.pt")
    result=execute(config,tmp_path,TinyWan())
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decode_completed"]==5 and result["counts"]["decoder_vjp_completed"]==2
    assert result["views"]["TWO_MEAN_SAVED"]["status"]=="MISSING"
    assert result["views"]["ALLBIT_TWO"]["status"]=="SCORED"


@pytest.mark.parametrize("decode,completed,metrics",[(1,1,55),(3,2,110),(5,4,165)])
def test_actual_failure_keeps_independent_control_and_missing_slots(tmp_path,decode,completed,metrics):
    result=execute(inputs(tmp_path),tmp_path,TinyWan(fail_decode=decode))
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decode_completed"]==completed
    assert result["totals"]["observed_metrics"]==metrics and result["totals"]["expected_metrics"]==220
    assert result["views"]["TWO_MEAN_SAVED"]["status"]=="SCORED"


@pytest.mark.parametrize("place",["ordinary","gradient"])
def test_active_state_tie_no_arbitrary_vjp(tmp_path,monkeypatch,place):
    actual=method.loss_summary
    def tied(*a,**kw):
        row=actual(*a,**kw);row["active_state_tie"]=True;return row
    if place=="ordinary":
        monkeypatch.setattr(method,"loss_summary",tied)
    else:
        actual_read=method.tensor_readout
        def read(*a,**kw):
            result=actual_read(*a,**kw)
            result["active"]["other_offsets"]=[1,2]
            result["objectives"]=torch.cat((result["objectives"][:1]*0-.1,result["objectives"][1:]))
            return result
        monkeypatch.setattr(method,"tensor_readout",read)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"]=="COMPLETE"
    assert result["direction_status"]=="UNDEFINED_ACTIVE_STATE_MAX_TIE"
    assert result["counts"]["decoder_vjp_attempted"]==0
    assert result["counts"]["decode_completed"]==(2 if place=="ordinary" else 3)
    assert result["totals"]["observed_metrics"]==110


def test_quality_failure_does_not_gate_method_and_interruption_is_report_only(tmp_path,monkeypatch):
    def fail(*a,**kw): raise OSError("quality IO")
    monkeypatch.setattr(runtime,"quality",fail)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decoder_vjp_completed"]==2 and result["counts"]["decode_completed"]==6
    assert result["totals"]["observed_metrics"]==220
    # Simulate a reaped child with stale totals, then prove takeover only reads reports.
    result["status"]="RUNNING"
    result["model_calls"]["step_2_vjp"]["status"]="RUNNING"
    runtime.write_json(tmp_path/"out/result.json",result)
    monkeypatch.setattr(torch,"load",lambda *a,**kw:(_ for _ in ()).throw(AssertionError("no tensor reload")))
    recovered=runtime.finalize_interrupted(tmp_path/"out","reaped")
    assert recovered["status"]=="INTERRUPTED" and recovered["totals"]["observed_metrics"]==220
    assert recovered["model_calls"]["step_2_vjp"]["status"]=="INTERRUPTED_COMPLETION_UNKNOWN"


def test_actual_peak_neighborhoods_zero_ties_coordinates_and_unclipped_values(tmp_path):
    base=torch.zeros(PROTOCOL.video_shape)
    rgb=base.clone();rgb[92]=.4;rgb[116,3,4,2]=.5
    q=visual.quality(rgb,base,PROTOCOL)
    events={row["kind"]:row for row in q["peak_events"]}
    assert events["frame_rmse"]["from_frame"]==92
    assert events["frame_max_abs"]["from_frame"]==116
    assert events["temporal_rmse"]["from_frame"]==91
    assert events["temporal_max_abs"]["from_frame"]==115
    assert q["peak_pixel"]==dict(frame=116,y=3,x=4,channel=2,max_abs=.5)
    assert q["framewise"][116]["max_abs"]==.5 and q["display"]["clipped_pixels"]==257
    zero=visual.quality(base,base,PROTOCOL)
    assert zero["peak_events"]==[]
    frames,crops=visual.selection({"ALLBIT_TWO":q,"BASE":zero},181)
    assert set(range(90,94))|set(range(114,118)) <= set(frames)
    assert set(method.FRAMES)<=set(frames) and len(frames)<30
    assert len(crops)==6
    for name,value,report in (("BASE",base,zero),("ALLBIT_TWO",rgb,q)):
        runtime.save_tensor(tmp_path/name/"float_rgb.pt",value)
        runtime.write_json(tmp_path/name/"quality.json",report)
    result=visual.export_visuals(tmp_path,base)
    assert result["views"]["BASE"]["frame_indices"]==result["views"]["ALLBIT_TWO"]["frame_indices"]
    assert result["views"]["TWO_MEAN_SAVED"]["status"]=="MISSING"
    for name in ("BASE","ALLBIT_TWO"):
        for item in result["views"][name]["frames"]+result["views"][name]["crops"]:
            assert (tmp_path/name/item["path"]).is_file()
    from PIL import Image
    with Image.open(tmp_path/"ALLBIT_TWO/frames/frame_116.png") as image:
        assert image.size==(440,64)


def test_valid_opposing_vjps_decode_cumulative_zero_without_rollback(tmp_path,monkeypatch):
    actual=torch.autograd.grad;calls=0
    def opposite(*a,**kw):
        nonlocal calls
        calls+=1
        value=actual(*a,**kw)[0]
        direction=torch.zeros_like(value)
        direction[0,0,1,0,0]=-1. if calls==1 else 1.
        return (direction,)
    monkeypatch.setattr(torch.autograd,"grad",opposite)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"]=="COMPLETE"
    assert result["steps"]["step_2"]["status"]=="DEFINED" and result["steps"]["step_2"]["cumulative_zero"]
    assert result["counts"]["decode_completed"]==6 and result["counts"]["decoder_vjp_completed"]==2
    assert result["views"]["ALLBIT_TWO"]["status"]=="SCORED"
    assert torch.load(tmp_path/"out/ALLBIT_TWO/delta.pt",weights_only=True).norm()==0


def test_nonfinite_second_vjp_keeps_half_and_does_not_fake_final(tmp_path,monkeypatch):
    actual=torch.autograd.grad;calls=0
    def nonfinite(*a,**kw):
        nonlocal calls
        calls+=1;value=actual(*a,**kw)[0]
        return (torch.full_like(value,float("nan")) if calls==2 else value,)
    monkeypatch.setattr(torch.autograd,"grad",nonfinite)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decoder_vjp_completed"]==2 and result["counts"]["decode_completed"]==5
    assert result["views"]["ALLBIT_HALF"]["status"]=="SCORED"
    assert result["views"]["ALLBIT_TWO"]["status"]=="MISSING"
    assert not (tmp_path/"out/ALLBIT_TWO/delta.pt").exists()


@pytest.mark.parametrize("target,step,decode,metrics",[("ALLBIT_HALF","step_1",4,110),("ALLBIT_TWO","step_2",6,165)])
@pytest.mark.parametrize("failure_kind",["decode","readout"])
def test_target_failure_preserves_formed_direction_and_specific_view_reason(tmp_path,monkeypatch,target,step,decode,metrics,failure_kind):
    if failure_kind=="readout":
        original=runtime.observer.read_rgb;reads=0
        def injected(*a,**kw):
            nonlocal reads
            reads+=1
            if reads==(3 if target=="ALLBIT_HALF" else 4):
                raise OSError("injected target readout failure")
            return original(*a,**kw)
        monkeypatch.setattr(runtime.observer,"read_rgb",injected)
    result=execute(inputs(tmp_path),tmp_path,TinyWan(fail_decode=decode if failure_kind=="decode" else None))
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decode_attempted"]==decode
    assert result["counts"]["decode_completed"]==decode-(failure_kind=="decode")
    assert result["totals"]["observed_metrics"]==metrics and result["totals"]["expected_metrics"]==220
    saved=json.loads((tmp_path/"out"/(step+".json")).read_text())
    assert saved==result["steps"][step] and saved["status"]=="DEFINED"
    assert result["views"][target]["direction_status"]=="DEFINED"
    assert result["views"][target]["status"]=="MISSING"
    assert "injected" in result["views"][target]["reason"]
    assert any(row["stage"]==target and "injected" in row["reason"] for row in result["failures"])
    if target=="ALLBIT_HALF":
        assert result["steps"]["step_2"]["status"]=="MISSING_DEPENDENCY"
        assert result["model_calls"]["step_2_gradient_decode"]["status"]=="MISSING_DEPENDENCY"
        assert result["counts"]["decoder_vjp_completed"]==1
        assert "ALLBIT_HALF" in result["views"]["ALLBIT_TWO"]["reason"]
