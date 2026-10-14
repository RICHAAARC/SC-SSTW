from __future__ import annotations

import json

import pytest
import torch

from main.tube_state import local_joint_base_alignment_v1 as method
from runtime.wan import local_joint_base_alignment_v1 as runtime
from runtime.wan import generation
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_fixed_scalars_ignore_fresh_roster_and_active_offset():
    margins = torch.arange(32, dtype=torch.double, requires_grad=True).reshape(4, 8)
    correlations = torch.arange(22, dtype=torch.double, requires_grad=True)/100
    read = dict(margins=margins, correlations=correlations)
    expected = margins.reshape(-1)[list(method.NONPOSITIVE_ROSTER)].mean()
    assert method.scalar("aggregate_payload", read) == expected
    assert method.scalar("state_G16", read) == correlations[0]-correlations[16]
    payload_gradient = torch.autograd.grad(method.scalar("aggregate_payload", read), margins,
                                           retain_graph=True)[0].reshape(-1)
    expected_gradient = torch.zeros(32, dtype=torch.double)
    expected_gradient[list(method.NONPOSITIVE_ROSTER)] = 1/16
    torch.testing.assert_close(payload_gradient, expected_gradient, rtol=0, atol=0)
    state_gradient = torch.autograd.grad(method.scalar("state_G16", read), correlations)[0]
    expected_state = torch.zeros(22, dtype=torch.double)
    expected_state[0], expected_state[16] = 1, -1
    torch.testing.assert_close(state_gradient, expected_state, rtol=0, atol=0)
    report = method.readout_report("state_G16", read)
    assert report["fresh_active_wrong_offsets"] == [21]
    assert not report["fresh_active_includes_fixed_16"]
    assert not report["fresh_roster_matches_historical"]
    assert report["fixed_definition_unchanged_by_fresh_readout"]


def test_projection_preserves_raw_slopes_and_accepts_zero_gradient():
    gradients = dict(aggregate_payload=torch.tensor([3., 4.]), state_G16=torch.zeros(2))
    directions = dict(old_bridge=torch.tensor([2., 0.]),
                      posterior_bridge=torch.tensor([0., -7.]),
                      base_to_cont_final=torch.tensor([1., 1.]))
    report = method.analyze(gradients, directions)
    row = report["projections"]["aggregate_payload"]
    assert row["old_bridge"]["raw_unit_direction_slope"] == pytest.approx(3.)
    assert row["old_bridge"]["masked_gradient_l2"] == pytest.approx(5.)
    assert row["old_bridge"]["cosine"] == pytest.approx(.6)
    zero = report["projections"]["state_G16"]
    assert all(item["raw_unit_direction_slope"] == 0 for item in zero.values())
    assert all(item["cosine"] is None for item in zero.values())
    assert report["unit_direction_cosines"]["old_bridge"]["posterior_bridge"] == 0
    assert report["gradient_geometry"]["gram"]["aggregate_payload"]["aggregate_payload"] == 25
    assert report["gradient_geometry"]["pairwise_cosines"]["aggregate_payload"]["state_G16"] is None
    assert report["gradient_normalization"] is False and report["no_sign_or_success_gate"]

    directions["posterior_bridge"] = torch.zeros(2)
    report = method.analyze(gradients, directions)
    undefined = report["projections"]["aggregate_payload"]["posterior_bridge"]
    assert undefined["raw_unit_direction_slope"] is None
    assert undefined["unit_direction_status"] == "UNDEFINED_ZERO_DIRECTION"


def setup(tmp_path):
    source = tmp_path/"source"
    bridge = source/"bridge"
    bridge.joinpath("tensors").mkdir(parents=True)
    base = torch.randn(SHAPE, generator=torch.Generator().manual_seed(4))*.2
    displacement = runtime.masked(torch.randn(SHAPE, generator=torch.Generator().manual_seed(5)), SUPPORT)
    displacement = displacement/float(displacement.double().norm())*.8
    final = base+displacement
    raw = torch.randn(SHAPE, generator=torch.Generator().manual_seed(6))
    residual = torch.randn(SHAPE, generator=torch.Generator().manual_seed(7))*.2
    six = source/"six.pt"
    torch.save(dict(base=base, final=final, reference={}, source_locators={},
                    input_field="fixture", source_arm="CONT_FINAL"), six)
    torch.save(raw, bridge/"tensors/raw.pt")
    torch.save(residual, bridge/"tensors/reconstruction_residual.pt")
    config = dict(schema="local-joint-base-alignment-v1", device="cpu", model={},
        carrier=dict(key=runtime.KEY, message_hex="8001a55a", rho=.5, cap=1.),
        nonpositive_roster=list(method.NONPOSITIVE_ROSTER), fixed_wrong_offset=16,
        inputs=dict(six_bit_inputs=str(six), bridge_run=str(bridge)))
    return config, base


def invoke(config, output, model):
    return runtime.run(config, output, loader=lambda *_args, **_kwargs: model,
                       protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)


def test_two_serial_vjps_save_full_evidence_and_complete_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT", str(tmp_path))
    config, base = setup(tmp_path)
    output = tmp_path/"out"
    model = TinyWan()
    result = invoke(config, output, model)
    assert result["status"] == "COMPLETE", result["failures"]
    assert model.calls == 2
    assert result["counts"]["vae_load_completed"] == 1
    assert result["counts"]["decode_completed"] == 2
    assert result["counts"]["decoder_vjp_completed"] == 2
    assert result["physical_replay_chunks"] == dict(forward=92, recompute=92)
    assert result["totals"]["observed_windows"] == 176
    assert result["totals"]["observed_chips"] == 2816
    assert result["totals"]["observed_metrics"] == 110
    for name in method.UNITS:
        unit = torch.load(output/"units"/(name+".pt"), weights_only=True)
        assert torch.equal(unit["input_terminal"], base)
        assert tuple(unit["q"].shape) == (22, 4, 16)
        assert len(unit["readout"]["values"]["margins"]) == 32
        assert len(unit["readout"]["values"]["correlations"]) == 22
        assert torch.equal(unit["gradient"], runtime.masked(unit["raw_gradient"], SUPPORT))
        assert (output/"unit_commits"/(name+".json")).is_file()
    diagnostic = json.loads((output/"diagnostic.json").read_text())
    assert set(diagnostic["projections"]) == set(method.UNITS)
    assert all(set(row) == set(method.DIRECTIONS) for row in diagnostic["projections"].values())
    assert diagnostic["direction_sources"]["raw"] == "bridge_run/tensors/raw.pt"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("complete report-only resume must not load a model")
    monkeypatch.setattr(generation, "load_frozen_vae", forbidden)
    assert not runtime.model_work_remaining(output)
    again = runtime.run(config, output, protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert again["status"] == "COMPLETE"
    assert not any(again["current_attempt_counts"].values())
    assert again["counts"] == result["counts"]


def test_interrupted_second_unit_retires_running_and_counts_rework(tmp_path, monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT", str(tmp_path))
    config, _ = setup(tmp_path)
    output = tmp_path/"out"
    model = TinyWan()
    atomic = resume.atomic_tensor

    def interrupt(path, value):
        if str(path).endswith("units/state_G16.pt"):
            raise KeyboardInterrupt("expiry before second unit commit")
        return atomic(path, value)

    monkeypatch.setattr(resume, "atomic_tensor", interrupt)
    with pytest.raises(KeyboardInterrupt, match="expiry"):
        invoke(config, output, model)
    interrupted = runtime.read_state(output)
    assert interrupted["status"] == "INTERRUPTED"
    assert interrupted["failure_history"]
    assert interrupted["counts"]["decode_completed"] == 2
    assert interrupted["counts"]["decoder_vjp_completed"] == 2
    assert (output/"units/aggregate_payload.pt").is_file()
    assert not (output/"units/state_G16.pt").exists()

    monkeypatch.setattr(resume, "atomic_tensor", atomic)
    complete = invoke(config, output, model)
    assert complete["status"] == "COMPLETE", complete["failures"]
    assert complete["current_attempt_counts"] == dict(
        vae_load_attempted=1, vae_load_completed=1,
        decode_attempted=1, decode_completed=1,
        decoder_vjp_attempted=1, decoder_vjp_completed=1)
    assert complete["counts"]["decode_completed"] == 3
    assert complete["counts"]["decoder_vjp_completed"] == 3
    assert complete["physical_replay_chunks"] == dict(forward=138, recompute=138)
    assert complete["failure_history"]


def test_committed_missing_unit_is_data_failure_without_replacement_vjp(tmp_path, monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT", str(tmp_path))
    config, _ = setup(tmp_path)
    output = tmp_path/"out"
    complete = invoke(config, output, TinyWan())
    (output/"units/state_G16.pt").unlink()
    assert not runtime.model_work_remaining(output)
    model = TinyWan()
    result = invoke(config, output, model)
    assert result["status"] == "ENGINEERING_FAILURE"
    assert model.calls == 0
    assert result["counts"] == complete["counts"]
    assert "missing committed gradient unit" in result["failures"][-1]["reason"]


def test_first_unit_decode_failure_does_not_drop_second_fixed_unit(tmp_path, monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT", str(tmp_path))
    config, _ = setup(tmp_path)
    result = invoke(config, tmp_path/"out", TinyWan(fail_decode=1))
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["units"]["aggregate_payload"]["status"] == "FAILED"
    assert result["units"]["state_G16"]["status"] == "COMPLETE"
    assert result["counts"]["decode_attempted"] == 2
    assert result["counts"]["decode_completed"] == 1
    assert result["counts"]["decoder_vjp_completed"] == 1
    assert result["totals"]["observed_metrics"] == 55
    assert len(result["failure_history"]) == 1


def test_shared_model_load_failure_blocks_second_unit_without_fake_retry(tmp_path):
    config, _ = setup(tmp_path)

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("injected VAE load failure")

    result = runtime.run(config, tmp_path/"out", loader=unavailable,
                         protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert result["status"] == "ENGINEERING_FAILURE"
    assert result["units"]["aggregate_payload"]["status"] == "FAILED"
    assert result["units"]["state_G16"]["status"] == "BLOCKED_SHARED_MODEL_LOAD"
    assert result["counts"]["vae_load_attempted"] == 1
    assert result["counts"]["vae_load_completed"] == 0
    assert result["counts"]["decode_attempted"] == 0

    recovered = invoke(config, tmp_path/"out", TinyWan())
    assert recovered["status"] == "COMPLETE", recovered["failures"]
    assert recovered["current_attempt_counts"] == dict(
        vae_load_attempted=1, vae_load_completed=1,
        decode_attempted=2, decode_completed=2,
        decoder_vjp_attempted=2, decoder_vjp_completed=2)
    assert recovered["counts"]["vae_load_attempted"] == 2
    assert recovered["counts"]["vae_load_completed"] == 1
    assert recovered["failure_history"]
