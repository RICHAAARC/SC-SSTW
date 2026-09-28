"""Small CPU native-UniPC checks for the fixed no-gradient controller."""
from __future__ import annotations

import copy
from collections import Counter

import numpy as np
import pytest

from main.tube_state import rgb_dct_structured_feedback as method

pytestmark = pytest.mark.unit


def test_fixed_basis_grid_and_positive_guard():
    raw = np.ones((1, 16, 46, 40, 64), dtype=np.float32)
    basis, rows = method.split_basis(raw)
    assert [row["time"] for row in rows] == [[1, 15], [15, 30], [30, 45]]
    for direction in basis:
        assert np.isclose(np.sqrt(np.mean(direction[:, :, 1:45].astype(np.float64) ** 2)), 1)
        assert not np.any(direction[:, :, (0, 45)])
    q = np.array([0.001] + [-0.002] * 29)
    J = np.zeros((30, 3))
    J[1:, 0] = 2
    selection = method.select_coefficients(q, J, method.R_STAR,
                                            (True, False, False))
    assert selection["coefficients"][0] > 0
    assert selection["coefficients"][1:] == [0, 0]
    assert selection["predictions_tried"] == 125
    lowered = q.copy()
    lowered[0] = -1e-6
    verdict = method.judge_candidate(q, lowered, .001, .01, 0)
    assert verdict["reason"] == "POSITIVE_GROUP_LOSS"
    with pytest.raises(FloatingPointError):
        method.judge_candidate(q, q, float("nan"), .01, 0)


def test_real_unipc_tail_history_and_accepted_first_step():
    import torch
    from diffusers import UniPCMultistepScheduler
    from runtime.wan import trajectory
    from runtime.wan.rgb_dct_structured_feedback_backend import WanStructuredFeedbackBackend

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", thresholding=False,
        predict_x0=True, lower_order_final=True, final_sigmas_type="zero",
        use_flow_sigmas=True, flow_shift=3.0)
    scheduler.set_timesteps(50, device="cpu")
    class FakeTransformer:
        def __call__(self, hidden_states, timestep, encoder_hidden_states,
                     attention_kwargs, return_dict):
            return (0.13 * hidden_states + 0.002 * timestep.float().reshape(1, 1, 1, 1, 1),)
    class FakePipe:
        transformer = FakeTransformer()
    counts = Counter()
    def count(kind, completed):
        counts[(kind, completed)] += 1
    backend = WanStructuredFeedbackBackend(
        {"generation": {"guidance_scale": 5.0}}, count)
    backend.pipe = FakePipe()
    backend.prompt = torch.zeros(1, 1, 1)
    backend.negative = torch.zeros(1, 1, 1)
    backend.dtype = torch.float32
    backend.phase = "TRANSFORMER"
    z = torch.zeros((1, 1, 46, 1, 1))
    for index in range(46):
        v = backend.velocity(index, z, scheduler)
        z = trajectory.native_step(scheduler, z, v, index,
                                   lambda _kind, done: count("scheduler_prefix", done))
    v46 = backend.velocity(46, z, scheduler)
    frozen = copy.deepcopy(scheduler)
    history_before = trajectory.fingerprint(vars(frozen))
    base = backend.rollout(46, z, frozen, v46)
    delta = torch.zeros_like(z)
    delta[:, :, 1:15] = -0.01
    probe = backend.rollout(46, z, frozen, v46, delta,
                            baseline_next=base["next_z"])
    accepted = backend.rollout(46, z, frozen, v46, 2 * delta,
                               baseline_next=base["next_z"])
    assert trajectory.fingerprint(vars(frozen)) == history_before
    assert probe["immediate"]["support_rms"] > 0
    assert accepted["immediate"]["support_rms"] > probe["immediate"]["support_rms"]
    assert counts[("scheduler_feedback", False)] == 12
    assert counts[("scheduler_feedback", True)] == 12
    next_z = accepted["next_z"]
    next_history = accepted["next_scheduler"]
    assert trajectory.fingerprint(vars(next_history)) == accepted["next_history_fingerprint"]
    v47 = backend.velocity(47, next_z, next_history)
    continued = backend.rollout(47, next_z, next_history, v47)
    assert continued["terminal_fingerprint"] == accepted["terminal_fingerprint"]
    assert base["terminal_fingerprint"] != accepted["terminal_fingerprint"]


def _tiny_feedback_backend():
    import torch
    from diffusers import UniPCMultistepScheduler
    from runtime.wan import trajectory
    from runtime.wan.rgb_dct_structured_feedback_backend import WanStructuredFeedbackBackend

    scheduler = UniPCMultistepScheduler(
        prediction_type="flow_prediction", thresholding=False,
        predict_x0=True, lower_order_final=True, final_sigmas_type="zero",
        use_flow_sigmas=True, flow_shift=3.0)
    scheduler.set_timesteps(50, device="cpu")
    class FakeTransformer:
        def __call__(self, hidden_states, timestep, encoder_hidden_states,
                     attention_kwargs, return_dict):
            return (0.13 * hidden_states +
                    0.002 * timestep.float().reshape(1, 1, 1, 1, 1),)
    class FakePipe:
        transformer = FakeTransformer()
    backend = WanStructuredFeedbackBackend(
        {"generation": {"guidance_scale": 5.0}}, lambda *_: None)
    backend.pipe = FakePipe()
    backend.prompt = torch.zeros(1, 1, 1)
    backend.negative = torch.zeros(1, 1, 1)
    backend.dtype = torch.float32
    backend.phase = "TRANSFORMER"
    z = torch.zeros((1, 1, 46, 1, 1))
    for index in range(50):
        v = backend.velocity(index, z, scheduler)
        if index in (46, 49):
            backend.nodes[index] = {"z": z.clone(), "v": v.clone()}
            backend.snapshots[index] = copy.deepcopy(scheduler)
        z = trajectory.native_step(scheduler, z, v, index, lambda *_: None)
    backend.off_terminal = z.clone()
    backend.to_vae_phase = lambda name: _set_fake_phase(backend, "VAE", name)
    backend.to_transformer_phase = lambda name: _set_fake_phase(backend, "TRANSFORMER", name)
    return backend


def _set_fake_phase(backend, phase, name):
    backend.phase = phase
    return {"name": name, "fake_cpu_phase": phase}


def test_feedback_arm_accept_and_zero_update_keep_native_history(tmp_path):
    import torch
    from experiments.wan_state_clock import rgb_dct_structured_terminal_feedback_run as run

    config = run.load_config()
    store = run.Store(tmp_path / "result.json",
                      run.initial_result(config, tmp_path, "0" * 40))
    store.active_case = run.CASE_IDS[0]
    store.save()
    backend = _tiny_feedback_backend()
    backend.count = store.count
    directions = []
    for lo, hi in method.BINS:
        direction = np.zeros((1, 1, 46, 1, 1), dtype=np.float32)
        direction[:, :, lo:hi] = np.sqrt(44 / (hi - lo))
        directions.append(direction)
    def score_improvement(terminal, _key):
        improvement = float((terminal - backend.off_terminal).abs().mean()) * 100
        value = -0.01 + improvement
        return {"q": [value] * 30, "C": 30 if value > 0 else 0,
                "score": value}
    backend.score_terminal = score_improvement
    run._feedback_arm(store, run.CASE_IDS[0], "MULTI46_47_48_49",
                      backend, directions, b"fixed-key")
    rows = store.case(run.CASE_IDS[0])["controls"]["MULTI46_47_48_49"]
    assert any(row["outcome"] == "ACCEPT" for row in rows)
    assert store.case(run.CASE_IDS[0])["budgets"]["MULTI46_47_48_49"]["spent"] > 0
    assert backend.terminals["MULTI46_47_48_49"].shape == torch.Size((1, 1, 46, 1, 1))
    assert all(row["outcome"] != "PENDING" for row in rows)
    assert any(row.get("continuation_diagnostic") for row in rows[1:])

    backend2 = _tiny_feedback_backend()
    backend2.count = store.count
    backend2.score_terminal = lambda *_: {"q": [-0.01] * 30, "C": 0,
                                          "score": -0.01}
    run._feedback_arm(store, run.CASE_IDS[0], "MULTI46_47_48_FREE49",
                      backend2, directions, b"fixed-key")
    rejection = store.case(run.CASE_IDS[0])["controls"]["MULTI46_47_48_FREE49"]
    assert [row["outcome"] for row in rejection[:3]] == ["ZERO_OPTIMUM"] * 3
    assert rejection[-1]["outcome"] == "FREE49"
    assert store.case(run.CASE_IDS[0])["budgets"]["MULTI46_47_48_FREE49"]["spent"] == 0
    assert "MULTI46_47_48_FREE49" in backend2.terminals
    assert len(store.data["cases"]) == 2
    assert sum(len(case["slots"]) for case in store.data["cases"].values()) == 10


def test_fake_ten_slot_supervision_and_timeout_retention(tmp_path):
    from experiments.wan_state_clock import rgb_dct_structured_terminal_feedback_run as run

    config = run.load_config()
    def complete_worker(output, _config, case_id):
        store = run.Store.open(output / "result.json")
        case = store.case(case_id)
        case["environment_receipt"] = {"status": "VALID"}
        case["resources"] = {"receipt_valid": True}
        case["release_receipt"] = {"status": "COMPLETED"}
        for arm, slot in case["slots"].items():
            slot.update(status="SCORED", frames_used=181,
                        positive_groups=0 if arm == "OFF" else 30,
                        decision="H0" if arm == "OFF" else "H1")
        store.save()
        return {"status": "COMPLETED", "error": None}
    output = tmp_path / "complete"
    output.mkdir()
    run._supervise(output, config, "0" * 40, worker_fn=complete_worker)
    result = run.Store.open(output / "result.json").data
    assert result["status"] == "FIXED_STRUCTURED_FEEDBACK_DEVELOPMENT_COMPLETE"
    assert result["scored_media_slots"] == 10
    assert result["scored_frames"] == 1810

    def timeout_worker(output, _config, case_id):
        if case_id == run.CASE_IDS[0]:
            return {"status": "TIMEOUT", "error": "WORKER_TIMEOUT",
                    "elapsed_seconds": 28800}
        return complete_worker(output, _config, case_id)
    output = tmp_path / "timeout"
    output.mkdir()
    run._supervise(output, config, "0" * 40, worker_fn=timeout_worker)
    result = run.Store.open(output / "result.json").data
    assert result["status"] == "INCOMPLETE"
    assert result["scored_media_slots"] == 5
    assert result["invalid_media_slots"] == 5
    assert len(result["cases"][run.CASE_IDS[0]]["slots"]) == 5
    assert all(slot["status"] == "NOT_RUN_RESOURCE_FAILURE"
               for slot in result["cases"][run.CASE_IDS[0]]["slots"].values())


def test_nonzero_prediction_three_true_rejects_still_returns_terminal(tmp_path,
                                                                    monkeypatch):
    from experiments.wan_state_clock import rgb_dct_structured_terminal_feedback_run as run

    config = run.load_config()
    store = run.Store(tmp_path / "result.json",
                      run.initial_result(config, tmp_path, "0" * 40))
    store.active_case = run.CASE_IDS[0]
    store.save()
    backend = _tiny_feedback_backend()
    backend.count = store.count
    def score(terminal, _key):
        improvement = float((terminal - backend.off_terminal).abs().mean()) * 100
        value = -0.01 + improvement
        return {"q": [value] * 30, "C": 30 if value > 0 else 0,
                "score": value}
    backend.score_terminal = score
    monkeypatch.setattr(method, "judge_candidate", lambda *_: dict(
        accepted=False, reason="NO_TRUE_HINGE_IMPROVEMENT",
        gained_groups=[], lost_groups=[], baseline_hinge=0.1,
        candidate_hinge=0.1, true_hinge_improvement=0.0))
    directions = []
    for lo, hi in method.BINS:
        direction = np.zeros((1, 1, 46, 1, 1), dtype=np.float32)
        direction[:, :, lo:hi] = np.sqrt(44 / (hi - lo))
        directions.append(direction)
    run._feedback_arm(store, run.CASE_IDS[0], "ONLY49", backend,
                      directions, b"fixed-key")
    point = store.case(run.CASE_IDS[0])["controls"]["ONLY49"][0]
    assert point["outcome"] == "NO_ACCEPTABLE_CANDIDATE"
    assert len(point["candidates"]) == 3
    assert all(row["tail_computed"] and row["decoded"] for row in point["candidates"])
    assert all(row["decision"] == "NO_TRUE_HINGE_IMPROVEMENT"
               for row in point["candidates"])
    assert all(len(row["predicted_q"]) == 30 and len(row["q"]) == 30
               for row in point["candidates"])
    assert store.case(run.CASE_IDS[0])["budgets"]["ONLY49"]["spent"] == 0
    assert "ONLY49" in backend.terminals
    assert store.case(run.CASE_IDS[0])["calls"]["scheduler_feedback"]["completed"] == 7


def test_run_case_five_arm_wiring_and_isolated_arm_failure(tmp_path, monkeypatch):
    from experiments.wan_state_clock import rgb_dct_structured_terminal_feedback_run as run

    config = run.load_config()
    def setup_backend(store):
        backend = _tiny_feedback_backend()
        backend.count = store.count
        backend.prepare_off = lambda _path: {"fake_cpu_generation": True}
        directions = []
        for lo, hi in method.BINS:
            direction = np.zeros((1, 1, 46, 1, 1), dtype=np.float32)
            direction[:, :, lo:hi] = np.sqrt(44 / (hi - lo))
            directions.append(direction)
        backend.build_basis = lambda _key: (
            np.broadcast_to(np.float32(0.5), run.RGB_SHAPE),
            directions, directions[0],
            {"clipping": {"fake": True},
             "single_geometry": {"status": "READY"},
             "bins": [{"status": "READY"}] * 3})
        backend.control_single49 = lambda _direction, _target: (
            "READY", backend.terminals.setdefault(
                "SINGLE49", backend.off_terminal.clone()),
            {"same_history": True})
        backend.score_terminal = lambda *_: {"q": [-0.01] * 30,
                                               "C": 0, "score": -0.01}
        backend.decode = lambda _terminal: np.broadcast_to(
            np.float32(0.5), run.RGB_SHAPE)
        backend.resources = lambda: {"receipt_valid": True,
                                     "cuda": {"peak_allocated": 0,
                                              "peak_reserved": 0}}
        backend.release = lambda: _set_fake_phase(backend, "RELEASED", "RELEASE")
        return backend
    def memory_score(_rgb, _key):
        return {"status": "SCORED", "reason": None, "score": 0.0,
                "group_scores": [-0.01] * 30, "positive_groups": 0,
                "frames_used": 181, "loss": .0001, "decision": None}
    def encode(_rgb, path, _fps, _crf):
        path.write_bytes(b"fake-mp4")
    def score_mp4(path, _key):
        arm = path.parent.name
        c = 0 if arm == "OFF" else 30
        q = [-0.01] * 30 if c == 0 else [0.01] * 30
        return ({"status": "SCORED", "reason": None, "score": .01,
                 "group_scores": q, "positive_groups": c,
                 "frames_used": 181,
                 "spec_sha256": run.receiver.SPEC_SHA256,
                 "key_id": config["receiver_key_id"]},
                np.broadcast_to(np.float32(0.5), run.RGB_SHAPE))
    monkeypatch.setattr(run, "_quantized_rgb", lambda rgb: rgb)
    monkeypatch.setattr(run, "_score_memory_layer", memory_score)
    monkeypatch.setattr(run, "_paired_quality", lambda *_: {"fake": True})

    for inject_only49_error in (False, True):
        output = tmp_path / ("failed_arm" if inject_only49_error else "complete")
        output.mkdir()
        store = run.Store(output / "result.json",
                          run.initial_result(config, output, "0" * 40))
        case_id = run.CASE_IDS[0]
        store.case(case_id)["environment_receipt"] = {"status": "VALID"}
        store.save()
        backend = setup_backend(store)
        if inject_only49_error:
            original_state = backend.state
            def fail_only49(index):
                if index == 49:
                    raise RuntimeError("fake ONLY49 control failure")
                return original_state(index)
            backend.state = fail_only49
        run.run_case(store, case_id, config, backend,
                     encode_fn=encode, score_fn=score_mp4)
        case = store.case(case_id)
        assert case["slots"]["OFF"]["status"] == "SCORED"
        assert case["slots"]["SINGLE49"]["status"] == "SCORED"
        assert case["slots"]["MULTI46_47_48_49"]["status"] == "SCORED"
        assert case["slots"]["MULTI46_47_48_FREE49"]["status"] == "SCORED"
        assert case["budgets"]["MULTI46_47_48_49"]["status"] == "COMPLETE"
        assert case["release_receipt"]["status"] == "COMPLETED"
        assert backend.phase == "RELEASED"
        assert case["calls"]["mp4_save"]["completed"] == (4 if inject_only49_error else 5)
        assert case["calls"]["mp4_read"]["completed"] == (4 if inject_only49_error else 5)
        assert case["calls"]["score"]["completed"] == (4 if inject_only49_error else 5)
        assert case["slots"]["ONLY49"]["status"] == (
            "ENGINEERING_INVALID" if inject_only49_error else "SCORED")
