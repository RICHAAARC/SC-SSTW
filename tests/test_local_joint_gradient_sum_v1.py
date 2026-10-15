from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from main.tube_state import local_joint_gradient_sum_v1 as method
from main.tube_state import local_joint_terminal_bridge_v1 as observer
from runtime.wan import local_joint_gradient_sum_v1 as runtime
from runtime.wan.local_joint_readout_m0_v1 import metric_values
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_exact_unequal_raw_gradient_sum_has_no_separate_normalization():
    base = torch.zeros(SHAPE)
    g_a = torch.zeros(SHAPE)
    g_g = torch.zeros(SHAPE)
    start, end = SUPPORT.time_range
    y0, y1, x0, x1 = SUPPORT.rois[0]
    g_a[:, :, start:end, y0:y1, x0:x1] = 3
    g_g[:, :, start:end, y0:y1, x0:x1] = -1
    candidate, report = method.construct_candidate(
        base, {"aggregate_payload": g_a, "state_G16": g_g}, SUPPORT)
    raw_sum = g_a.double() + g_g.double()
    expected = base.double() + method.STEP * raw_sum / raw_sum.norm()
    torch.testing.assert_close(candidate.double(), expected, rtol=0, atol=2e-8)
    assert report["scalar_weights"] == {"aggregate_payload": 1.0, "state_G16": 1.0}
    assert report["per_gradient_normalization"] is False
    assert report["summed_direction_normalized"] is True
    assert report["stored_BASE_distance"] == pytest.approx(.1, abs=2e-8)
    assert report["endpoint_projection"] is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_summary_rejects_nonfinite_readout_values(bad):
    values = dict(margins=[.1] * 32, correlations=[.2] * 22)
    values["margins"][7] = bad
    with pytest.raises(ValueError, match="finite complete readout"):
        method.summarize(values)
    values = dict(margins=[.1] * 32, correlations=[.2] * 22)
    values["correlations"][16] = bad
    with pytest.raises(ValueError, match="finite complete readout"):
        method.summarize(values)


def setup_source(tmp_path):
    source = tmp_path / "alignment"
    (source / "inputs").mkdir(parents=True)
    (source / "units").mkdir()
    generator = torch.Generator().manual_seed(37)
    base = torch.randn(SHAPE, generator=generator) * .1
    source_model = TinyWan()
    base_rgb = runtime.adapter.decode_normalized_latent(source_model, base)
    rows = observer.read_rgb(base_rgb, key=runtime.KEY, rho=.5, protocol=PROTOCOL)
    evaluation = observer.evaluate_saved(rows, key=runtime.KEY, message=bytes.fromhex("8001a55a"),
                                         rho=.5, protocol=PROTOCOL)
    values = metric_values(evaluation)
    q = runtime.prior.raw_q(rows)
    readout = dict(values=dict(margins=values["margins"],
                               correlations=values["correlations"],
                               state_gap=values["state_gap"]))
    torch.save(dict(base=base), source / "inputs" / "fixed_inputs.pt")
    gradients = {
        "aggregate_payload": runtime.masked(torch.randn(SHAPE, generator=generator), SUPPORT),
        "state_G16": runtime.masked(torch.randn(SHAPE, generator=generator) * .2, SUPPORT),
    }
    for name, gradient in gradients.items():
        torch.save(dict(name=name, input_terminal=base, gradient=gradient,
                        gradient_normalized=False, q=q, readout=readout),
                   source / "units" / (name + ".pt"))
    base_rgb_path = tmp_path / "base_float_rgb.pt"
    torch.save(base_rgb, base_rgb_path)
    config = dict(schema="local-joint-gradient-sum-v1", device="cpu", model={},
        carrier=dict(key=runtime.KEY, message_hex="8001a55a", rho=.5, cap=1.),
        gradient_weights=dict(aggregate_payload=1, state_G16=1), step=.1,
        inputs=dict(alignment_run=str(source), base_float_rgb=str(base_rgb_path)))
    return config, base


def invoke(config, output, model):
    return runtime.run(config, output, loader=lambda *_args, **_kwargs: model,
                       protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)


def test_one_decode_zero_vjp_complete_and_report_only_resume(tmp_path):
    config, base = setup_source(tmp_path)
    output = tmp_path / "out"
    model = TinyWan()
    result = invoke(config, output, model)
    assert result["status"] == "COMPLETE", result["failures"]
    assert model.calls == 1
    assert result["counts"]["vae_load_completed"] == 1
    assert result["counts"]["decode_completed"] == 1
    assert result["counts"]["decoder_vjp_completed"] == 0
    assert result["ordinary_chunks"] == dict(attempted=46, completed=46)
    assert result["totals"]["observed_metrics"] == 110
    assert result["expected_cost"]["source_points"] == 1
    assert result["expected_cost"]["new_source_endpoints"] == 1
    assert result["expected_cost"]["independent_source_videos"] == 0
    assert result["expected_cost"]["independent_samples"] == 0
    assert result["totals"]["independent_samples"] == 0
    unit = torch.load(output / "candidate_unit.pt", weights_only=True)
    assert float((unit["terminal"].double() - base.double()).norm()) == pytest.approx(.1, abs=2e-7)
    comparison = json.loads((output / "comparison.json").read_text())
    assert len(comparison["bits_32"]) == 32
    assert len(comparison["states"]["SUM_STEP"]["correlations"]) == 22
    assert comparison["descriptive_distances"]["no_threshold_gate"]
    assert (output / "SUM_STEP" / "quality.json").is_file()
    assert (output / "SUM_STEP" / "frames.json").is_file()
    assert (output / "candidate_commits" / "gradient_sum.json").is_file()
    assert (output / "rgb_commits" / "SUM_STEP.json").is_file()
    assert (output / "scoring_commits" / "SUM_STEP.json").is_file()
    assert runtime.resume.Journal(output, views=method.VIEWS, steps=runtime.STEPS).interval == 60

    config["inputs"]["alignment_run"] = str(tmp_path / "missing")
    config["inputs"]["base_float_rgb"] = str(tmp_path / "missing_rgb.pt")
    assert not runtime.model_work_remaining(output)
    again = runtime.run(config, output, protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert again["status"] == "COMPLETE"
    assert not any(again["current_attempt_counts"].values())
    assert again["counts"] == result["counts"]


def test_fault_after_candidate_commit_resumes_without_reconstruction(tmp_path, monkeypatch):
    config, _ = setup_source(tmp_path)
    output = tmp_path / "out"
    original = runtime.prior.copy_saved_file

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("fault after candidate persistence")

    monkeypatch.setattr(runtime.prior, "copy_saved_file", fail_copy)
    first = invoke(config, output, TinyWan())
    assert first["status"] == "ENGINEERING_FAILURE"
    assert (output / "candidate_unit.pt").is_file()
    assert first["counts"]["vae_load_attempted"] == 0
    saved = torch.load(output / "candidate_unit.pt", weights_only=True)["terminal"].clone()

    monkeypatch.setattr(runtime.prior, "copy_saved_file", original)
    complete = invoke(config, output, TinyWan())
    assert complete["status"] == "COMPLETE", complete["failures"]
    assert torch.equal(torch.load(output / "candidate_unit.pt", weights_only=True)["terminal"], saved)
    assert complete["failure_history"]


def test_formula_mismatched_saved_candidate_fails_before_decode(tmp_path, monkeypatch):
    config, _ = setup_source(tmp_path)
    output = tmp_path / "out"
    original = runtime.prior.copy_saved_file

    def fail_copy(*_args, **_kwargs):
        raise RuntimeError("stop after candidate persistence")

    monkeypatch.setattr(runtime.prior, "copy_saved_file", fail_copy)
    first = invoke(config, output, TinyWan())
    assert first["status"] == "ENGINEERING_FAILURE"
    unit = torch.load(output / "candidate_unit.pt", weights_only=True)
    start, _ = SUPPORT.time_range
    y0, _, x0, _ = SUPPORT.rois[0]
    unit["terminal"] = unit["terminal"].clone()
    unit["terminal"][0, 0, start, y0, x0] += 1e-3
    torch.save(unit, output / "candidate_unit.pt")
    monkeypatch.setattr(runtime.prior, "copy_saved_file", original)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("formula mismatch must fail before model loading")

    result = runtime.run(config, output, loader=forbidden,
                         protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert result["status"] == "ENGINEERING_FAILURE"
    assert not any(result["current_attempt_counts"].values())
    assert result["counts"]["vae_load_attempted"] == 0
    assert "does not match BASE + 0.10" in result["failures"][-1]["reason"]


def test_missing_peak_structures_are_recomputed_from_saved_rgb(tmp_path):
    config, _ = setup_source(tmp_path)
    output = tmp_path / "out"
    result = invoke(config, output, TinyWan())
    assert result["status"] == "COMPLETE"
    base_quality_path = output / "BASE" / "quality.json"
    sum_quality_path = output / "SUM_STEP" / "quality.json"
    base_quality = json.loads(base_quality_path.read_text())
    sum_quality = json.loads(sum_quality_path.read_text())
    del base_quality["peak_events"]
    del sum_quality["peak_pixel"]
    base_quality_path.write_text(json.dumps(base_quality))
    sum_quality_path.write_text(json.dumps(sum_quality))
    assert not runtime.reports_complete(output, PROTOCOL.video_shape[0])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("quality repair must use saved RGB without model work")

    repaired = runtime.run(config, output, loader=forbidden,
                           protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert repaired["status"] == "COMPLETE", repaired["failures"]
    assert not any(repaired["current_attempt_counts"].values())
    base_quality = json.loads(base_quality_path.read_text())
    sum_quality = json.loads(sum_quality_path.read_text())
    assert base_quality["peak_events"] == []
    assert isinstance(sum_quality["peak_pixel"], dict)
    assert runtime.quality_complete(base_quality, PROTOCOL.video_shape[0])
    assert runtime.quality_complete(sum_quality, PROTOCOL.video_shape[0])
    assert runtime.reports_complete(output, PROTOCOL.video_shape[0])


def test_empty_visual_index_cache_is_rebuilt_from_saved_rgb(tmp_path):
    config, _ = setup_source(tmp_path)
    output = tmp_path / "out"
    result = invoke(config, output, TinyWan())
    assert result["status"] == "COMPLETE"
    for name in method.VIEWS:
        path = output / name / "frames.json"
        index = json.loads(path.read_text())
        assert index["frames"]
        index["frames"] = []
        index["crops"] = []
        path.write_text(json.dumps(index))
    assert not runtime.reports_complete(output, PROTOCOL.video_shape[0])

    def forbidden(*_args, **_kwargs):
        raise AssertionError("visual cache repair must use saved RGB without model work")

    repaired = runtime.run(config, output, loader=forbidden,
                           protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert repaired["status"] == "COMPLETE", repaired["failures"]
    assert not any(repaired["current_attempt_counts"].values())
    reports = {name: json.loads((output / name / "quality.json").read_text()) for name in method.VIEWS}
    expected_frames, expected_crops = runtime.visual.selection(
        reports, PROTOCOL.video_shape[0], views=method.VIEWS)
    for name in method.VIEWS:
        index = json.loads((output / name / "frames.json").read_text())
        assert runtime.visual_index_complete(index, expected_frames, expected_crops)
        paths = [item["path"] for item in index["frames"] + index["crops"]]
        assert len(paths) == len(set(paths))
        assert len(index["frames"]) == len(expected_frames)
        assert len(index["crops"]) == len(expected_crops)
    assert runtime.reports_complete(output, PROTOCOL.video_shape[0])


def test_missing_saved_inputs_fail_with_zero_model_calls(tmp_path):
    config = dict(schema="local-joint-gradient-sum-v1", device="cpu", model={},
        carrier=dict(key=runtime.KEY, message_hex="8001a55a", rho=.5, cap=1.),
        gradient_weights=dict(aggregate_payload=1, state_G16=1), step=.1,
        inputs=dict(alignment_run=str(tmp_path / "missing"),
                    base_float_rgb=str(tmp_path / "missing_rgb.pt")))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("missing saved inputs must not trigger replacement model work")

    result = runtime.run(config, tmp_path / "out", loader=forbidden,
                         protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert result["status"] == "ENGINEERING_FAILURE"
    assert not any(result["counts"].values())
    assert "no replacement gradient, BASE decode or VJP" in result["failures"][-1]["reason"]


def test_fresh_process_executes_fixed_fake_path(tmp_path):
    config, _ = setup_source(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "fresh"
    program = """
import json, sys
from runtime.wan import local_joint_gradient_sum_v1 as runtime
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT
config=json.load(open(sys.argv[1]))
result=runtime.run(config,sys.argv[2],loader=lambda *_a,**_k:TinyWan(),
                   protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
print(json.dumps({'status':result['status'],'counts':result['counts']}))
raise SystemExit(0 if result['status']=='COMPLETE' else 1)
"""
    completed = subprocess.run([sys.executable, "-c", program, str(config_path), str(output)],
                               cwd=runtime.Path(__file__).resolve().parents[1],
                               capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.splitlines()[-1])
    assert payload["status"] == "COMPLETE"
    assert payload["counts"]["decode_completed"] == 1


def test_rgb_commit_survives_scoring_fault_without_second_decode(tmp_path, monkeypatch):
    config, _ = setup_source(tmp_path)
    output = tmp_path / "out"
    original = runtime.observer.read_rgb
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("fault after RGB persistence")
        return original(*args, **kwargs)

    monkeypatch.setattr(runtime.observer, "read_rgb", fail_once)
    first_model = TinyWan()
    first = invoke(config, output, first_model)
    assert first["status"] == "ENGINEERING_FAILURE"
    assert first_model.calls == 1
    assert (output / "SUM_STEP" / "float_rgb.pt").is_file()
    assert (output / "rgb_commits" / "SUM_STEP.json").is_file()

    second_model = TinyWan()
    complete = invoke(config, output, second_model)
    assert complete["status"] == "COMPLETE", complete["failures"]
    assert second_model.calls == 0
    assert complete["counts"]["decode_completed"] == 1
    assert complete["failure_history"]
