from __future__ import annotations

import json
import math

import pytest
import torch

from main.tube_state import local_joint_mean_refresh_v1 as method
from runtime.wan import local_joint_mean_refresh_v1 as runtime
from runtime.wan import local_joint_readout_m0_v1 as m0
from test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT, inputs as m0_inputs

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def inputs(tmp_path):
    config = m0_inputs(tmp_path)
    saved = config["inputs"].pop("bridge_delta")
    config["inputs"].update(mean_delta=saved, common_delta=saved)
    return config


def execute(config, tmp_path, model):
    return runtime.run(config, tmp_path/"out", loader=lambda *_a, **_k:model,
        protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)


def test_one_average_vjp_matches_original_magnitude_gradient_mean():
    # Differently sized component gradients detect accidental per-objective normalization.
    x = torch.tensor([.7, -.2], dtype=torch.float64, requires_grad=True)
    objectives = torch.stack((x[0], 20*x[1], (x*x).sum(), 3*x[0]-x[1], x[0]*x[1]))
    individual = [torch.autograd.grad(y, x, retain_graph=True)[0] for y in objectives]
    scalar = method.mean_objective(dict(objectives=objectives, tied=False))
    average = torch.autograd.grad(scalar, x)[0]
    torch.testing.assert_close(average, torch.stack(individual).mean(0), rtol=0, atol=1e-15)
    normalized = torch.stack([g/g.norm() for g in individual]).mean(0)
    assert not torch.allclose(average/average.norm(), normalized/normalized.norm())


def test_original_base_budget_no_refill_and_actual_increment_prediction():
    mid = method.midpoint(torch.tensor([1.,0.]))
    for g, expected in [(torch.tensor([0.,3.]), math.sqrt(.5)),
                         (torch.tensor([-2.,0.]), 0.),
                         (torch.tensor([2.,0.]), 1.)]:
        delta, report = method.refresh_delta(mid,g)
        assert float(delta.double().norm()) == pytest.approx(expected)
        assert report["cumulative_post_cap_l2"] == float(delta.double().norm())
        assert report["predicted_average_change"] == pytest.approx(float((g.double()*(delta-mid).double()).sum()))
        assert report["cap_scale"] == 1. and report["no_radius_refill"]
    # Numerical projection branch is relative to the original origin, not the midpoint.
    delta, report = method.refresh_delta(torch.tensor([.500001,0.]), torch.tensor([1.,0.]))
    assert report["cap_scale"] < 1
    assert float(delta.double().norm()) == pytest.approx(1., abs=1e-7)
    assert report["actual_second_increment_l2"] < .5


def test_zero_tie_nonfinite_and_large_finite_norm():
    zero, report = method.refresh_delta(torch.tensor([.5,0.]),torch.zeros(2))
    assert zero is None and report["status"] == "ZERO_MEAN"
    with pytest.raises(method.UndefinedActiveTie):
        method.mean_objective(dict(tied=True, objectives=torch.ones(5)))
    with pytest.raises(FloatingPointError):
        method.refresh_delta(torch.zeros(2),torch.tensor([float("nan"),0.]))
    with pytest.raises(FloatingPointError):
        method.mean_objective(dict(tied=False, objectives=torch.tensor([1.,2.,3.,4.,float("inf")])))
    delta, _ = method.refresh_delta(torch.zeros(2,dtype=torch.float64),torch.tensor([1e100,1e100],dtype=torch.float64))
    assert float(delta.norm()) == pytest.approx(.5)


def test_full_fixed_runner_and_complete_temporal_quality(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    config=inputs(tmp_path); result=execute(config,tmp_path,TinyWan())
    assert result["status"] == "COMPLETE",result["failures"]
    assert result["counts"] == dict(vae_load_attempted=1,vae_load_completed=1,
        decode_attempted=6,decode_completed=6,decoder_vjp_attempted=1,decoder_vjp_completed=1,
        encode=0,dit=0,native=0,codec=0)
    assert result["ordinary_chunks"] == dict(attempted=230,completed=230)
    replay=result["replay"]["midpoint"]
    assert replay["counts"]["vae_chunk"]["forward"] == dict(attempted=46,completed=46)
    assert replay["counts"]["vae_chunk"]["recompute"] == dict(attempted=46,completed=46)
    assert replay["boundary_storage"]["closed"]
    assert result["totals"]["observed_metrics"] == result["totals"]["expected_metrics"] == 275
    assert result["totals"]["observed_chips"] == 7040
    assert not list(tmp_path.glob("wan-vae-boundary-*"))
    out=tmp_path/"out"
    saved=torch.load(config["inputs"]["mean_delta"],weights_only=True)
    mid=torch.load(out/"MID_MEAN/delta.pt",weights_only=True)
    two=torch.load(out/"TWO_MEAN/delta.pt",weights_only=True)
    torch.testing.assert_close(mid,.5*saved,rtol=0,atol=0)
    assert two.double().norm() <= 1+1e-7
    gradient=torch.load(out/"gradients/midpoint_mean.pt",weights_only=True)
    assert float((gradient-runtime.masked(gradient,SUPPORT)).abs().sum()) == 0
    expected,report=method.refresh_delta(mid,gradient)
    torch.testing.assert_close(two,expected,rtol=0,atol=0)
    comparison=json.loads((out/"comparison.json").read_text())
    assert comparison["primary_comparison"]["reference"] == "ONE_MEAN"
    assert comparison["primary_comparison"]["target"] == "TWO_MEAN"
    assert len(comparison["metrics"]["TWO_MEAN"]["margins"]) == 32
    assert len(comparison["metrics"]["TWO_MEAN"]["correlations"]) == 22
    q=json.loads((out/"TWO_MEAN/quality.json").read_text())
    assert len(q["temporal_framewise"]) == 180
    assert q["temporal_framewise"][0]["from_frame"] == 0
    assert q["temporal_framewise"][-1]["to_frame"] == 180
    rgb=torch.load(out/"TWO_MEAN/float_rgb.pt",weights_only=True)
    base=torch.load(out/"BASE/float_rgb.pt",weights_only=True)
    residual=rgb.double()-base.double()
    for item in q["temporal_framewise"]:
        change=residual[item["to_frame"]]-residual[item["from_frame"]]
        assert item["rmse"] == pytest.approx(float(change.square().mean().sqrt()),abs=1e-15)
        assert item["max_abs"] == float(change.abs().max())
    from PIL import Image
    with Image.open(out/"TWO_MEAN/frames.png") as im:
        assert im.width == 8*PROTOCOL.video_shape[2]
    assert result["quality_frames"] == [1,44,88,112,116,120,132,176]
    # The shared helper's default M0 structure/numbers are unchanged.
    old=m0.quality(rgb,base,PROTOCOL)
    assert "temporal_framewise" not in old and "temporal_peak" not in old
    assert old == {k:v for k,v in q.items() if k not in ("temporal_framewise","temporal_peak")}


@pytest.mark.parametrize("tie_at",[1,2])
def test_tie_retains_independent_comparators_and_no_vjp(tmp_path,monkeypatch,tie_at):
    actual=method.tensor_readout; calls=0
    def tied(*args,**kwargs):
        nonlocal calls
        calls+=1
        read=actual(*args,**kwargs)
        if calls==tie_at: read["tied"]=True
        return read
    monkeypatch.setattr(method,"tensor_readout",tied)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"] == "COMPLETE"
    assert result["direction_status"] == "UNDEFINED_ACTIVE_MIN_MAX_TIE"
    assert result["counts"]["decode_completed"] == 3+tie_at
    assert result["counts"]["decoder_vjp_attempted"] == 0
    assert result["totals"]["observed_metrics"] == 220
    assert result["totals"]["expected_metrics"] == 275
    assert all(result["views"][x]["status"] == "SCORED" for x in method.VIEWS[:4])
    assert result["views"]["TWO_MEAN"]["status"] == "MISSING"


@pytest.mark.parametrize("bad_gradient",["zero","nonfinite"])
def test_degenerate_gradient_no_fallback_or_repeated_vjp(tmp_path,monkeypatch,bad_gradient):
    actual=torch.autograd.grad
    calls=0
    def injected(*args,**kwargs):
        nonlocal calls
        calls+=1
        value=actual(*args,**kwargs)[0]
        return (torch.zeros_like(value) if bad_gradient=="zero" else torch.full_like(value,float("nan")),)
    monkeypatch.setattr(torch.autograd,"grad",injected)
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert calls == result["counts"]["decoder_vjp_completed"] == 1
    assert result["counts"]["decode_completed"] == 5
    assert result["totals"]["observed_metrics"] == 220
    assert result["views"]["TWO_MEAN"]["direction_status"] == "UNDEFINED"
    assert result["direction_status"] == ("ZERO_MEAN" if bad_gradient=="zero" else "UNDEFINED_MIDPOINT_OR_GRADIENT_FAILURE")


@pytest.mark.parametrize("missing",["mean_delta","common_delta"])
def test_missing_input_keeps_other_controls(tmp_path,missing):
    config=inputs(tmp_path)
    config["inputs"][missing]=str(tmp_path/"absent"/"delta.pt")
    result=execute(config,tmp_path,TinyWan())
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["views"]["BASE"]["status"] == "SCORED"
    if missing=="mean_delta":
        assert result["views"]["ONE_COMMON"]["status"] == "SCORED"
        assert result["counts"]["decode_completed"] == 2
        assert result["counts"]["decoder_vjp_attempted"] == 0
    else:
        assert result["views"]["TWO_MEAN"]["status"] == "SCORED"
        assert result["counts"]["decode_completed"] == 5
        assert result["counts"]["decoder_vjp_completed"] == 1


def test_gradient_failure_preserves_four_scored_views(tmp_path):
    result=execute(inputs(tmp_path),tmp_path,TinyWan(fail_decode=5))
    assert result["counts"]["decode_attempted"] == 5 and result["counts"]["decode_completed"] == 4
    assert result["counts"]["decoder_vjp_attempted"] == 0
    assert result["totals"]["observed_metrics"] == 220
    assert result["status"] == "ENGINEERING_FAILURE"


def test_quality_io_failure_does_not_gate_fixed_calls(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,"quality",lambda *_a,**_k:(_ for _ in ()).throw(OSError("quality IO")))
    result=execute(inputs(tmp_path),tmp_path,TinyWan())
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["counts"]["decode_completed"] == 6
    assert result["counts"]["decoder_vjp_completed"] == 1
    assert result["totals"]["observed_metrics"] == 275
    assert all(x["artifacts"]["quality"]=="MISSING" for x in result["views"].values())


def test_interruption_recovers_persisted_observations_only(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,"quality",lambda *_a,**_k:(_ for _ in ()).throw(KeyboardInterrupt("after metrics")))
    model=TinyWan(); result=execute(inputs(tmp_path),tmp_path,model)
    assert result["status"]=="INTERRUPTED" and model.calls==1
    assert result["totals"]["observed_metrics"] == 55
    result["status"]="RUNNING"
    result["views"]["BASE"].update(status="MISSING",observed_metrics=0,observed_windows=0,observed_chips=0)
    runtime.write_json(tmp_path/"out/result.json",result)
    recovered=runtime.finalize_interrupted(tmp_path/"out","reaped child")
    assert recovered["totals"]["observed_metrics"] == 55 and model.calls==1
    comparison=json.loads((tmp_path/"out/comparison.json").read_text())
    assert comparison["primary_comparison"]["objective_change"] == [None]*5
    assert len(comparison["metrics"]) == 5
