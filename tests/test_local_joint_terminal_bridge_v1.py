from __future__ import annotations

import copy
import json
from pathlib import Path
import types

import pytest
import torch

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc
from main.tube_state import local_joint_terminal_bridge_v1 as math_diag
from runtime.wan import local_joint_terminal_bridge_v1 as runtime
from runtime.wan import vae as vae_adapter
from runtime.wan.local_joint_state_payload_provider_v1 import LatentSupport

pytestmark = pytest.mark.quick
PROTOCOL = carrier.CarrierProtocol((181, 16, 16, 3), 1, 8, 22,
    ((0, 8, 0, 8), (0, 8, 8, 16), (8, 16, 0, 8), (8, 16, 8, 16)))
SHAPE = (1, 3, 181, 16, 16)
SUPPORT = LatentSupport((1, 177), ((0, 4, 0, 4), (0, 4, 12, 16), (12, 16, 0, 4), (12, 16, 12, 16)))


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class FakeVAE(torch.nn.Module):
    def __init__(self, *, fail_decode=0, bad_encode=False, output=None):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = types.SimpleNamespace(latents_mean=[.1, .2, .3], latents_std=[.5, .8, 1.2])
        self.events = []
        self.fail_decode, self.bad_encode, self.output = fail_decode, bad_encode, output
        self.cache_clears = 0

    def clear_cache(self):
        self.cache_clears += 1

    def boundary(self, kind):
        assert not torch.is_grad_enabled()
        self.events.append(kind)
        if self.output:
            on_disk = json.loads((self.output/"result.json").read_text())
            assert on_disk["counts"][kind+"_attempted"] == self.events.count(kind)
            assert sum(row["status"] == "RUNNING" for row in on_disk["model_calls"].values()) == 1

    def encode(self, video):
        self.boundary("encode")
        value = video.clone()
        if self.bad_encode:
            value[0, 0, 0, 0, 0] = float("nan")
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=lambda: value))

    def decode(self, value, return_dict=False):
        self.boundary("decode")
        if self.events.count("decode") == self.fail_decode:
            raise RuntimeError("sentinel decode failure")
        return (value.clone(),)


def inputs(tmp_path, *, zero=False):
    folder = tmp_path/"saved_joint"
    folder.mkdir()
    generator = torch.Generator().manual_seed(42)
    rgb = torch.zeros(PROTOCOL.video_shape) if zero else torch.rand(PROTOCOL.video_shape, generator=generator)
    vae = FakeVAE()
    with torch.inference_mode():
        z = vae_adapter.reencode_rgb24_readback(vae, rgb) + .01
    torch.save(z, folder/"terminal.pt")
    torch.save(rgb, folder/"float.pt")
    config = dict(source_run=runtime.RUN_ID, source_arm="JOINT", model=runtime.MODEL,
        carrier=dict(key=runtime.KEY, message_hex="8001a55a", rho=.5, cap=1.), device="cpu",
        inputs=dict(terminal_latent=str(folder/"terminal.pt"), float_rgb=str(folder/"float.pt")))
    return config, rgb


def execute(config, tmp_path, model):
    return runtime.run(config, tmp_path/"output", loader=lambda *_a, **_k: model,
                       protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)


def test_observer_preserves_original_carrier_and_clipping():
    rgb = torch.rand(PROTOCOL.video_shape, generator=torch.Generator().manual_seed(8))
    reference, receipt = carrier.apply_carrier_rgb(rgb, key=runtime.KEY, message=bytes.fromhex("8001a55a"), rho=.5, protocol=PROTOCOL)
    pre = rgb.clone()
    seen = []
    def observer(frame, roi, pixels):
        y0, y1, x0, x1 = PROTOCOL.rois[roi]
        pre[frame, y0:y1, x0:x1] = pixels
        pixels.fill_(999)  # a diagnostic callback cannot mutate the writer
        seen.append((frame, roi))
    actual, receipt2 = carrier.apply_carrier_rgb(rgb, key=runtime.KEY, message=bytes.fromhex("8001a55a"), rho=.5,
                                                protocol=PROTOCOL, diagnostic_observer=observer)
    assert torch.equal(reference, actual)
    assert receipt == receipt2
    assert len(seen) == 22*8*4
    assert torch.equal(pre.clamp(0, 1), actual)
    assert receipt["clipped_low_values"] + receipt["clipped_high_values"] > 0


def test_known_grid_adapter_matches_original_full_catalog():
    rgb = torch.rand(PROTOCOL.video_shape, generator=torch.Generator().manual_seed(2))
    raw = math_diag.read_rgb(rgb, key=runtime.KEY, rho=.5, protocol=PROTOCOL)
    actual = math_diag.evaluate_saved(raw, key=runtime.KEY, message=bytes.fromhex("8001a55a"), rho=.5, protocol=PROTOCOL)
    full = json.loads(json.dumps([r.to_dict() for r in carrier.observe_catalog(rgb, runtime.KEY, protocol=PROTOCOL)]))
    expected = posthoc.evaluate_observations(full, key=runtime.KEY, message=bytes.fromhex("8001a55a"), protocol=PROTOCOL)
    assert actual["state"] == expected["state"]
    assert actual["payload"] == expected["payload"]
    assert len(actual["target_errors"]) == 1408
    assert len(raw) == 88


def test_full_fixed_pipeline_saves_all_layers_and_exact_call_order(tmp_path):
    config, _ = inputs(tmp_path)
    model = FakeVAE(output=tmp_path/"output")
    result = execute(config, tmp_path, model)
    assert result["status"] == "COMPLETE", result["failures"]
    assert model.events == ["encode"]*2 + ["decode"]*4
    assert result["counts"]["encode_completed"] == 2
    assert result["counts"]["decode_completed"] == 4
    assert result["actual_model_execution"] == "INJECTED_TEST_DOUBLE"
    assert result["stages"]["base"]["views"]["preclip"]["reason"] == "saved_base_preclip_unavailable"
    for name in math_diag.STAGES:
        for view in math_diag.VIEWS:
            if (name, view) == ("base", "preclip"):
                continue
            raw = json.loads((tmp_path/"output"/name/(view+"_raw.json")).read_text())
            assert len(raw["rows"]) == 88
            assert sum(len(r["state_chips"])+len(r["payload_chips"]) for r in raw["rows"]) == 1408
            metric = json.loads((tmp_path/"output"/name/(view+"_metrics.json")).read_text())
            assert len(metric["state"]["correlations"]) == 22
            assert len(metric["payload"]["metrics"]) == 32
    paired = json.loads((tmp_path/"output/paired_comparisons.json").read_text())
    assert len(paired["comparisons"]) == 14
    assert all(len(p["chips"]) == 1408 for p in paired["comparisons"].values())
    assert all(p["missing_values"] == 0 for p in paired["comparisons"].values())
    assert (tmp_path/"output/tensors/reconstruction_residual.pt").exists()


def test_negative_or_missing_statistics_do_not_gate_remaining_calls(tmp_path):
    config, _ = inputs(tmp_path, zero=True)
    model = FakeVAE()
    result = execute(config, tmp_path, model)
    assert result["status"] == "COMPLETE"
    assert model.events == ["encode"]*2 + ["decode"]*4
    assert result["stages"]["candidate"]["views"]["postclip"]["descriptive_condition"]["met"] is None
    assert result["scientific_pass"] is False


@pytest.mark.parametrize("failure", ["float_missing", "nonfinite_input", "bad_encode", "decode_failure"])
def test_failure_counts_and_retained_dependencies(tmp_path, failure):
    config, rgb = inputs(tmp_path)
    if failure == "float_missing":
        Path(config["inputs"]["float_rgb"]).unlink()
    if failure == "nonfinite_input":
        rgb[0, 0, 0, 0] = float("nan")
        torch.save(rgb, config["inputs"]["float_rgb"])
    model = FakeVAE(fail_decode=1 if failure == "decode_failure" else 0, bad_encode=failure == "bad_encode")
    result = execute(config, tmp_path, model)
    assert result["status"] == "ENGINEERING_FAILURE"
    assert len(result["stages"]) == 6
    if failure in ("float_missing", "nonfinite_input"):
        assert model.events == []
        assert result["counts"]["vae_load_attempted"] == 0
        assert result["counts"]["decode_attempted"] == 0  # no 1D fallback
    elif failure == "bad_encode":
        assert model.events == ["encode"]
        assert result["counts"]["encode_completed"] == 1  # model returned; adapter rejected NaN
        assert result["model_calls"]["encode_candidate"]["status"] == "MISSING_DEPENDENCY"
    else:
        assert model.events == ["encode"]*2 + ["decode"]*4
        assert result["counts"]["decode_attempted"] == 4
        assert result["counts"]["decode_completed"] == 3
        assert result["stages"]["posterior_reconstruction"]["status"] == "FAILED"
        assert result["stages"]["capped_reinjection"]["status"] == "COMPLETE"


def test_external_takeover_preserves_completed_read_and_unknown_call(tmp_path):
    result = runtime.initial_result({})
    result["counts"]["encode_attempted"] = 2
    result["counts"]["encode_completed"] = 1
    result["model_calls"]["encode_base"]["status"] = "RETURNED"
    result["model_calls"]["encode_candidate"]["status"] = "RUNNING"
    result["stages"]["base"].update(status="COMPLETE")
    result["stages"]["base"]["views"]["postclip"] = dict(status="OBSERVED", state_gap=-.1)
    runtime.write_json(tmp_path/"result.json", result)
    actual = runtime.finalize_interrupted(tmp_path, "killed test child")
    assert actual["status"] == "INTERRUPTED"
    assert actual["counts"]["encode_completed"] == 1
    assert actual["model_calls"]["encode_candidate"]["status"] == "INTERRUPTED_COMPLETION_UNKNOWN"
    assert actual["stages"]["base"]["views"]["postclip"]["state_gap"] == -.1
    assert runtime.finalize_interrupted(tmp_path, "second attempt") == actual


def test_decode_observer_receives_unclamped_same_decode_and_cannot_change_output():
    model = FakeVAE()
    z = torch.randn(SHAPE, generator=torch.Generator().manual_seed(6))
    observed = []
    def capture(x):
        observed.append(x.clone())
        x.fill_(999)
    actual = vae_adapter.decode_normalized_latent(model, z, diagnostic_observer=capture)
    mean, std = vae_adapter._scale_tensors(model, z)
    expected = ((z*std+mean)[0].permute(1, 2, 3, 0)/2+.5).clamp(0, 1)
    assert model.events == ["decode"]
    assert torch.equal(actual, expected)
    assert torch.equal(observed[0].clamp(0, 1), expected)
    assert (observed[0] < 0).any() or (observed[0] > 1).any()
