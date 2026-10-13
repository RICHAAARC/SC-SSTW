from __future__ import annotations

import json

import pytest
import torch

from main.tube_state import local_joint_ball_probe_v1 as method
from runtime.wan import local_joint_ball_probe_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from tests.test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark = pytest.mark.quick


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def start_values():
    margins = [.04 + .003 * i for i in range(32)]
    for bit, value in zip(method.BITS, (-.17, -.10, -.23, -.07, -.06, -.12)):
        margins[bit] = value
    margins[29] = .021
    return dict(margins=margins, correlations=[.1] + [.02] * 21, state_gap=.08)


def test_one_shot_full_ball_solver_fp32_is_inside_and_guarded():
    base = torch.zeros(1, 1, 1, 1)
    start = torch.tensor([[[[.8]]]])
    # Six raw target margins share one direction; guard senses all permit it.
    gradients = [torch.ones_like(base) * scale for scale in (1, 1.1, 1.2, 1.3, 1.4, 1.5)]
    gradients += [torch.ones_like(base), -torch.ones_like(base), torch.ones_like(base), -torch.ones_like(base)]
    terminal, report = method.solve_candidate(gradients, base, start, start_values())
    assert terminal is not None, report
    assert report["status"] == "FIXED_BALL_CANDIDATE"
    assert report["solver_radius_limit"] == 1 - method.FP32_RADIUS_MARGIN
    assert report["fp32"]["BASE_radius"] <= 1
    assert report["fp32"]["predictions"]["max_normalized_guard_violation"] <= method.RESIDUAL_TOL
    assert report["endpoint_projection"] is False


def setup_source(tmp_path):
    source = tmp_path / "protected"
    (source / "inputs").mkdir(parents=True)
    (source / "units").mkdir()
    generator = torch.Generator().manual_seed(23)
    base = torch.randn(SHAPE, generator=generator) * .1
    delta = runtime.masked(torch.randn(SHAPE, generator=generator), SUPPORT).double()
    delta *= .9 / delta.norm()
    start = (base.double() + delta).float()
    values = start_values()
    old = [runtime.masked(torch.randn(SHAPE, generator=generator), SUPPORT) for _ in method.BITS]
    protections = [runtime.masked(torch.randn(SHAPE, generator=generator), SUPPORT) for _ in method.PROTECTIONS]
    bundle = dict(base=base, start=start, old_gradients=old, reference=values,
                  barrier=method.barrier_parameters(values))
    torch.save(bundle, source / "inputs/fixed_inputs.pt")
    for name, gradient in zip(method.PROTECTIONS, protections):
        torch.save(dict(name=name, input_terminal=start, gradient=gradient), source / "units" / (name + ".pt"))
    torch.save(dict(terminal=start.clone()), source / "candidate_unit.pt")
    model = TinyWan()
    for name, terminal in dict(BASE=base, START_FINAL=start, PROTECTED=start).items():
        directory = source / name
        directory.mkdir()
        rgb = runtime.adapter.decode_normalized_latent(model, terminal)
        rows = runtime.observer.read_rgb(rgb, key=runtime.KEY, rho=.5, protocol=PROTOCOL)
        evaluation = runtime.observer.evaluate_saved(rows, key=runtime.KEY, message=bytes.fromhex("8001a55a"), rho=.5, protocol=PROTOCOL)
        torch.save(terminal, directory / "terminal_latent.pt")
        torch.save(rgb, directory / "float_rgb.pt")
        torch.save(runtime.raw_q(rows), directory / "q.pt")
        (directory / "raw.json").write_text(json.dumps(rows))
        (directory / "metrics.json").write_text(json.dumps(evaluation))
    config = dict(schema="local-joint-ball-probe-v1", device="cpu", model={},
                  carrier=dict(key=runtime.KEY, message_hex="8001a55a", rho=.5, cap=1.),
                  inputs=dict(protected_run=str(source)))
    return config, base, start


def test_runtime_one_decode_reuses_three_views_and_complete_resume(tmp_path, monkeypatch):
    config, base, start = setup_source(tmp_path)
    output = tmp_path / "out"

    def fixed(*_args):
        prediction = method.linear_predictions(
            torch.load(output / "inputs/fixed_inputs.pt", weights_only=True)["gradients"],
            start_values(), start, start,
        ) if (output / "inputs/fixed_inputs.pt").exists() else dict(
            changes={}, target_margins={}, minimum_target_margin=-.23,
            normalized_signed_guard_changes={name: 0. for name in method.PROTECTIONS},
            max_normalized_guard_violation=0.,
        )
        return start.clone(), dict(status="FIXED_BALL_CANDIDATE", fixture=True,
            fp32=dict(BASE_radius=float((start.double()-base.double()).norm()),
                      distance_from_START=0., predictions=prediction))

    monkeypatch.setattr(method, "solve_candidate", fixed)
    model = TinyWan()
    result = runtime.run(config, output, loader=lambda *_a, **_k: model,
                         protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert result["status"] == "COMPLETE", result["failures"]
    assert model.calls == 1
    assert result["counts"]["vae_load_completed"] == 1
    assert result["counts"]["decode_completed"] == 1
    assert result["counts"]["decoder_vjp_completed"] == 0
    assert result["ordinary_chunks"] == dict(attempted=46, completed=46)
    assert result["totals"]["observed_metrics"] == 220
    assert (output / "candidate_unit.pt").is_file()
    assert (output / "rgb_commits/BALL_PROBE.json").is_file()
    assert (output / "scoring_commits/BALL_PROBE.json").is_file()
    assert len(resume.read_json(output / "comparison.json")["bits_32"]) == 32

    config["inputs"]["protected_run"] = str(tmp_path / "missing")
    monkeypatch.setattr(runtime.generation, "load_frozen_vae", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("model called")))
    assert not runtime.model_work_remaining(output)
    again = runtime.run(config, output, protocol=PROTOCOL, latent_shape=SHAPE, support=SUPPORT)
    assert again["status"] == "COMPLETE"
    assert not any(again["current_attempt_counts"].values())
