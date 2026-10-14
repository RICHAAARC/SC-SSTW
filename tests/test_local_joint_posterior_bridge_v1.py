from __future__ import annotations

import json

import pytest
import torch

from main.tube_state import local_joint_posterior_bridge_v1 as method
from runtime.wan import local_joint_posterior_bridge_v1 as runtime
from runtime.wan import local_joint_all_bit_quality_v1 as visual
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT


pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_inputs(seed=51):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(SHAPE, generator=generator) * .1
    raw = torch.randn(SHAPE, generator=generator) * .15
    residual = torch.randn(SHAPE, generator=generator) * .08
    return base, raw, residual


def test_construct_is_one_fixed_masked_residual_compensated_endpoint():
    base, raw, residual = make_inputs()
    terminal, report = method.construct_candidate(base, raw, residual, SUPPORT)
    direction = method.masked(raw.double() - residual.double(), SUPPORT)
    expected = (base.double() + method.NUMERICAL_RADIUS * direction / direction.norm()).float()
    torch.testing.assert_close(terminal, expected, rtol=0, atol=0)
    actual = terminal.double() - base.double()
    assert report["status"] == "FIXED_POSTERIOR_BRIDGE_CANDIDATE"
    assert report["lambda_residual"] == 1
    assert report["numerical_radius"] == .99998
    assert report["scan"] is report["endpoint_projection"] is False
    assert report["independent_algebraic_source_validation"] is False
    assert float(actual.norm()) <= 1
    assert torch.count_nonzero(actual - method.masked(actual, SUPPORT)) == 0
    assert len(report["roi_norms"]) == 4 and len(report["channel_norms"]) == SHAPE[1]

    with pytest.raises(ValueError, match="nonzero finite norm"):
        method.construct_candidate(base, torch.zeros_like(raw), torch.zeros_like(residual), SUPPORT)
    with pytest.raises(ValueError, match="shape"):
        method.construct_candidate(base, raw[:, :, :-1], residual, SUPPORT)


def save_readout(source, name, rgb):
    rows = runtime.observer.read_rgb(rgb, key=runtime.KEY, rho=.5, protocol=PROTOCOL)
    evaluation = runtime.observer.evaluate_saved(
        rows, key=runtime.KEY, message=bytes.fromhex("8001a55a"), rho=.5, protocol=PROTOCOL)
    directory = source / name
    directory.mkdir(parents=True)
    (directory / "postclip_raw.json").write_text(json.dumps(dict(status="OBSERVED", rows=rows)))
    (directory / "postclip_metrics.json").write_text(json.dumps(evaluation))


def setup_source(tmp_path):
    source = tmp_path / "bridge"
    (source / "tensors").mkdir(parents=True)
    base, raw, residual = make_inputs()
    torch.save(raw, source / "tensors/raw.pt")
    torch.save(residual, source / "tensors/reconstruction_residual.pt")
    base_terminal = tmp_path / "base_terminal.pt"
    base_rgb_path = tmp_path / "base_float_rgb.pt"
    torch.save(base, base_terminal)
    setup_model = TinyWan()
    base_rgb = runtime.adapter.decode_normalized_latent(setup_model, base)
    torch.save(base_rgb, base_rgb_path)
    old_direction = method.masked(raw.double(), SUPPORT)
    capped_terminal = (base.double() + .7 * old_direction / old_direction.norm()).float()
    capped_rgb = runtime.adapter.decode_normalized_latent(TinyWan(), capped_terminal)
    save_readout(source, "base", base_rgb)
    save_readout(source, "capped_reinjection", capped_rgb)
    old_quality = visual.quality(capped_rgb, base_rgb, PROTOCOL)
    (source / "result.json").write_text(json.dumps(dict(stages=dict(capped_reinjection=dict(
        views=dict(postclip=dict(rgb_vs_base=dict(
            rmse=old_quality["rmse"], max_abs=old_quality["max_abs"]))))))))
    config = dict(
        schema="local-joint-posterior-bridge-v1",
        device="cpu",
        model={},
        carrier=dict(key=runtime.KEY, message_hex="8001a55a", rho=.5, cap=1.,
                     lambda_residual=1., numerical_radius=.99998),
        inputs=dict(bridge_run=str(source), base_terminal=str(base_terminal),
                    base_float_rgb=str(base_rgb_path)),
    )
    return config, base, raw, residual


def test_runtime_one_decode_complete_result_and_zero_model_resume(tmp_path, monkeypatch):
    config, base, raw, residual = setup_source(tmp_path)
    output = tmp_path / "out"
    model = TinyWan()
    result = runtime.run(config, output, loader=lambda *_args, **_kwargs: model,
                         protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert result["status"] == "COMPLETE", result["failures"]
    assert model.calls == 1
    assert result["counts"]["vae_load_completed"] == 1
    assert result["counts"]["decode_completed"] == 1
    assert result["counts"]["decoder_vjp_completed"] == 0
    assert result["counts"]["encode"] == 0
    assert result["ordinary_chunks"] == dict(attempted=46, completed=46)
    assert result["totals"]["observed_metrics"] == 165
    assert (output / "candidate_unit.pt").is_file()
    assert (output / "rgb_commits/POSTERIOR_BRIDGE.json").is_file()
    assert (output / "scoring_commits/POSTERIOR_BRIDGE.json").is_file()
    assert not (output / "CAPPED/float_rgb.pt").exists()

    unit = torch.load(output / "candidate_unit.pt", map_location="cpu", weights_only=True)
    expected, _ = method.construct_candidate(base, raw, residual, SUPPORT)
    torch.testing.assert_close(unit["terminal"], expected, rtol=0, atol=0)
    comparison = resume.read_json(output / "comparison.json")
    assert len(comparison["bits_32"]) == 32
    assert all(len(row["roi_margins"][name]) == 4
               for row in comparison["bits_32"] for name in method.VIEWS)
    assert all(len(comparison["states"][name]["correlations"]) == 22 for name in method.VIEWS)
    assert comparison["descriptive_target"]["delivery_gate"] is False
    for name in method.RGB_VIEWS:
        quality = resume.read_json(output / name / "quality_vs_BASE.json")
        assert len(quality["framewise"]) == PROTOCOL.video_shape[0]
        assert len(quality["temporal_framewise"]) == PROTOCOL.video_shape[0] - 1
        assert "peak_events" in quality and "peak_pixel" in quality
        assert quality["rgb_input_semantics"] == "decoder_output_clamped_float_[0,1]"
        assert quality["numeric_residual_metrics_use_full_range"] is True
        assert quality["display_residual_range"] == [-.25, .25]
        assert quality["display"]["rgb_input_semantics"] == quality["rgb_input_semantics"]
        index = resume.read_json(output / name / "frames_vs_BASE.json")
        assert index["rgb_input_semantics"] == quality["rgb_input_semantics"]
        assert index["numeric_residual_metrics_use_full_range"] is True
        assert index["display_residual_range"] == [-.25, .25]
    capped_quality = resume.read_json(output / "CAPPED/quality_vs_BASE.json")
    assert capped_quality["status"] == "SAVED_SUMMARY_ONLY"
    assert capped_quality["framewise"] is None
    assert runtime.reports_complete(output, PROTOCOL.video_shape[0])

    # A report-only resume upgrades older complete artifacts with explicit RGB
    # semantics and must not load the model or repeat the saved decode.
    semantics = ("rgb_input_semantics", "numeric_residual_metrics_use_full_range",
                 "display_residual_range")
    for name in method.RGB_VIEWS:
        for filename in ("quality_vs_BASE.json", "frames_vs_BASE.json"):
            path = output / name / filename
            row = resume.read_json(path)
            for field in semantics:
                row.pop(field)
            path.write_text(json.dumps(row))
    assert not runtime.reports_complete(output, PROTOCOL.video_shape[0])

    config["inputs"] = {name: str(tmp_path / "missing" / name) for name in config["inputs"]}
    monkeypatch.setattr(runtime.generation, "load_frozen_vae",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("model called")))
    assert not runtime.model_work_remaining(output)
    again = runtime.run(config, output, protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert again["status"] == "COMPLETE"
    assert not any(again["current_attempt_counts"].values())
    assert again["last_resume_action"] == "report_only; zero model calls"
    assert runtime.reports_complete(output, PROTOCOL.video_shape[0])
