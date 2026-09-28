"""CPU convex certificates and same native-loop failure/continuation contracts."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from main.tube_state import rgb_dct_structured_feedback as fixed
from main.tube_state import rgb_dct_continuous_solver as solver

pytestmark = pytest.mark.unit


def test_small_optimum_lost_by_original_lattice():
    q = np.full(30, -0.000001)
    J = np.zeros((30, 3)); J[:, 0] = 1
    old = fixed.select_coefficients(q, J, fixed.R_STAR)
    new = solver.select_coefficients(q, J, fixed.R_STAR)
    assert old["coefficients"] == [0., 0., 0.]
    assert new["status"] == "CERTIFIED"
    assert np.isclose(new["coefficients"][0], 0.000101 / 1.0001, rtol=1e-10)
    assert new["predicted_objective"] < old["predicted_objective"]
    assert new["global_objective_gap_upper_bound"] <= new["tolerance"]


@pytest.mark.parametrize("radius,available", [(0., (True, True, True)),
    (fixed.R_STAR, (False, False, False)), (fixed.R_STAR, (True, False, True))])
def test_zero_and_unavailable_directions(radius, available):
    q = np.full(30, -0.001)
    J = np.zeros((30, 3)); J[:, 1] = 1
    result = solver.select_coefficients(q, J, radius, available)
    assert result["status"] == "ZERO_OPTIMUM"
    assert result["coefficients"] == [0., 0., 0.]


def test_positive_q_flat_hinge_zero_optimum():
    result = solver.select_coefficients(np.full(30, .1), np.ones((30, 3)), fixed.R_STAR)
    assert result["status"] == "ZERO_OPTIMUM"
    assert result["global_objective_gap_upper_bound"] == 0


def test_budget_boundary_and_unavailable_coordinate():
    q = np.full(30, -1.)
    J = np.ones((30, 3)); J[:, 1] = 100
    result = solver.select_coefficients(q, J, 0.001, (True, False, True))
    a = np.array(result["coefficients"])
    assert a[1] == 0 and a.sum() <= .001
    assert a.sum() >= np.nextafter(.001, 0) - 1e-18
    assert result["global_objective_gap_upper_bound"] <= result["tolerance"]


def test_numerical_failure_and_no_uncertified_return():
    q, J = np.full(30, -.001), np.ones((30, 3))
    with pytest.raises(solver.SolverFailure) as caught:
        solver.select_coefficients(q, J, fixed.R_STAR, max_iterations=0)
    assert caught.value.receipt["reason"] == "MAX_ITERATIONS_WITHOUT_CERTIFICATE"
    assert caught.value.receipt["coefficients"] is None
    assert caught.value.receipt["status"] == "FAILED"
    for matrix in (J * np.nan, J * 1e308):
        with pytest.raises(solver.SolverFailure):
            solver.select_coefficients(q, matrix, fixed.R_STAR)
    with pytest.raises(solver.SolverFailure):
        solver.select_coefficients(q, J, fixed.R_STAR + 1e-10)


def test_certified_tiny_nonzero_is_never_called_zero():
    q = np.full(30, fixed.MARGIN - 1e-12)
    result = solver.select_coefficients(q, np.ones((30, 3)), fixed.R_STAR)
    assert result["status"] == "CERTIFIED"
    assert sum(result["coefficients"]) > 0


def _directions():
    directions = []
    for lo, hi in fixed.BINS:
        direction = np.zeros((1, 1, 46, 1, 1), dtype=np.float32)
        direction[:, :, lo:hi] = np.sqrt(44 / (hi - lo))
        directions.append(direction)
    return directions


def test_real_loop_solver_failure_stops_sequential_chain(tmp_path):
    from experiments.wan_state_clock import rgb_dct_structured_terminal_feedback_run as run
    from test_rgb_dct_structured_feedback import _tiny_feedback_backend
    backend = _tiny_feedback_backend()
    backend.score_terminal = lambda *_: dict(q=[-.01] * 30, C=0, score=-.01)
    store = run.Store(tmp_path / "result.json", run.initial_result(run.load_config(), tmp_path, "0" * 40))
    store.active_case = run.CASE_IDS[0]
    backend.count = store.count
    calls = []
    def failure(*_):
        calls.append("solver")
        raise solver.SolverFailure(dict(status="FAILED", reason="FORCED_NONCONVERGENCE", coefficients=None))
    with pytest.raises(solver.SolverFailure):
        run._feedback_arm(store, run.CASE_IDS[0], "MULTI46_47_48_49", backend,
                          _directions(), b"key", coefficient_solver=failure)
    rows = store.case(run.CASE_IDS[0])["controls"]["MULTI46_47_48_49"]
    assert len(rows) == 1 and rows[0]["outcome"] == "SOLVER_FAILURE"
    assert store.case(run.CASE_IDS[0])["budgets"]["MULTI46_47_48_49"]["status"] == "FAILED"
    assert "MULTI46_47_48_49" not in backend.terminals
    assert calls == ["solver"]


def test_candidate_shared_sequence_refreshes_current_q_and_history(tmp_path):
    from experiments.wan_state_clock import rgb_dct_structured_terminal_feedback_run as run
    from experiments.wan_state_clock import video_trajectory_blind_validation_run as candidate
    from test_rgb_dct_structured_feedback import _tiny_feedback_backend
    backend = _tiny_feedback_backend()
    def score(terminal, _key):
        value = -.01 + float((terminal - backend.off_terminal).abs().mean()) * 100
        return dict(q=[value]*30, C=30 if value > 0 else 0, score=value)
    backend.score_terminal = score
    store = run.Store(tmp_path / "result.json", run.initial_result(run.load_config(), tmp_path, "0"*40))
    store.active_case = run.CASE_IDS[0]
    backend.count = store.count
    candidate.feedback_arm(store, run.CASE_IDS[0], "MULTI46_47_48_49", backend, _directions(), b"key")
    rows = store.case(run.CASE_IDS[0])["controls"]["MULTI46_47_48_49"]
    assert rows[0]["outcome"] == "ACCEPT"
    assert rows[1]["baseline_history"] == rows[0]["committed_next_history_fingerprint"]
    assert rows[1]["baseline"]["q"] != rows[0]["baseline"]["q"]
    assert len(rows[1]["probes"]) == 3
    assert rows[1]["prediction"]["solver"] == solver.SPEC
    assert rows[1]["continuation_diagnostic"]["terminal_fingerprint_match"]


def test_fixed_six_store_failure_retains_every_point(tmp_path):
    from experiments.wan_state_clock import video_trajectory_blind_validation_run as run
    candidate, historical = run.load_config()
    data = run.initial_result(tmp_path, tmp_path / "saved", candidate, historical, {}, None)
    store = run.SixPointStore(tmp_path / "result.json", data)
    run._mark_pending(store, run.shared.CASE_IDS[0], "MISSING_SAVED_STATE")
    assert data["point_counts"] == dict(expected=6, accepted=0, evaluated=0, pending=3, failed=3)
    assert len(json.loads(store.path.read_text())["cases"]) == 2


def test_independent_point_uses_same_real_loop_and_does_not_advance(tmp_path):
    from experiments.wan_state_clock import video_trajectory_blind_validation_run as candidate
    from test_rgb_dct_structured_feedback import _tiny_feedback_backend
    from runtime.wan import trajectory
    backend = _tiny_feedback_backend()
    backend.score_terminal = lambda *_: dict(q=[-.01]*30, C=0, score=-.01)
    config, historical = candidate.load_config()
    store = candidate.SixPointStore(tmp_path / "result.json", candidate.initial_result(
        tmp_path, tmp_path / "saved", config, historical, {}, None))
    case_id = candidate.shared.CASE_IDS[0]
    store.active_case = case_id
    backend.count = store.count
    z, history, v = backend.state(46)
    baseline = backend.rollout(46, z, history, v)
    reference = dict(current_state_fingerprint=trajectory.fingerprint((z, vars(history), v)),
        baseline_history=baseline["source_history_fingerprint"],
        baseline_terminal_fingerprint=baseline["terminal_fingerprint"],
        baseline=dict(q=[-.01]*30), jacobian=np.zeros((30,3)).tolist())
    original_velocity = backend.velocity
    calls = []
    def velocity(*args):
        calls.append(args[0]); return original_velocity(*args)
    backend.velocity = velocity
    candidate.feedback_arm(store, case_id, "FIXED_T46", backend, _directions(), b"key",
                           independent_point=46, reference_point=reference)
    point = store.case(case_id)["controls"]["FIXED_T46"][0]
    assert point["outcome"] == "ZERO_OPTIMUM"
    assert point["saved_reference_identity"] == dict(state_match=True, history_match=True, terminal_match=True)
    assert calls == [47,48,49]*4  # baseline + probes, no next-point velocity
    assert store.data["point_counts"]["evaluated"] == 1
    assert trajectory.fingerprint(vars(backend.snapshots[46])) == reference["baseline_history"]


def test_post_verdict_history_failure_invalidates_point_not_evidence(tmp_path):
    from experiments.wan_state_clock import video_trajectory_blind_validation_run as candidate
    from test_rgb_dct_structured_feedback import _tiny_feedback_backend
    from runtime.wan import trajectory
    backend = _tiny_feedback_backend()
    def score(terminal, _key):
        value = -.01 + float((terminal - backend.off_terminal).abs().mean()) * 100
        return dict(q=[value]*30, C=30 if value > 0 else 0, score=value)
    backend.score_terminal = score
    config, historical = candidate.load_config()
    store = candidate.SixPointStore(tmp_path / "result.json", candidate.initial_result(
        tmp_path, tmp_path / "saved", config, historical, {}, None))
    case_id = candidate.shared.CASE_IDS[0]
    store.active_case = case_id
    z, history, v = backend.state(46)
    baseline = backend.rollout(46, z, history, v)
    reference = dict(index=46,
        current_state_fingerprint=trajectory.fingerprint((z, vars(history), v)),
        baseline_history=baseline["source_history_fingerprint"],
        baseline_terminal_fingerprint=baseline["terminal_fingerprint"],
        baseline=score(baseline["terminal"], b"key"), jacobian=np.zeros((30,3)).tolist())
    saved_case = dict(controls={"MULTI46_47_48_49": [reference]})
    original_rollout = backend.rollout
    def corrupt_candidate_history(*args, **kwargs):
        result = original_rollout(*args, **kwargs)
        if len(args) > 4 and args[4] is not None:
            result["next_history_fingerprint"] = "injected-history-mismatch"
        return result
    backend.rollout = corrupt_candidate_history
    backend.count = store.count
    with pytest.raises(RuntimeError, match="committed first-step history identity"):
        candidate.run_independent_points(store, case_id, backend, _directions(), b"key", saved_case)
    case = store.case(case_id)
    point = case["controls"]["FIXED_T46"][0]
    assert point["outcome"] == "ENGINEERING_FAILURE"
    assert point["prior_outcome"] == "ACCEPT"
    assert any(trial["decision"] == "ACCEPT" for trial in point["candidates"])
    assert case["budgets"]["FIXED_T46"]["status"] == "FAILED"
    assert case["budgets"]["FIXED_T46"]["spent"] > 0
    assert store.data["point_counts"] == dict(expected=6, accepted=0, evaluated=0, pending=5, failed=1)
    assert len(case["failures"]) == 1
    assert "FIXED_T46" not in backend.terminals
    persisted = json.loads(store.path.read_text())
    assert persisted["point_counts"] == store.data["point_counts"]
    assert all(case["controls"][f"FIXED_T{i}"][0]["outcome"] == "PENDING" for i in (47,48))


@pytest.mark.parametrize("failed_index,worker_reason", [
    (46, "WORKER_EXIT_-9"), (47, "WORKER_TIMEOUT")])
def test_worker_death_after_verdict_recovers_only_incomplete_points(
        tmp_path, failed_index, worker_reason):
    from experiments.wan_state_clock import video_trajectory_blind_validation_run as candidate
    from test_rgb_dct_structured_feedback import _tiny_feedback_backend
    from runtime.wan import trajectory
    backend = _tiny_feedback_backend()
    def score(terminal, _key):
        value = -.01 + float((terminal - backend.off_terminal).abs().mean()) * 100
        return dict(q=[value]*30, C=30 if value > 0 else 0, score=value)
    backend.score_terminal = score
    config, historical = candidate.load_config()
    store = candidate.SixPointStore(tmp_path / "result.json", candidate.initial_result(
        tmp_path, tmp_path / "saved", config, historical, {}, None))
    case_id = candidate.shared.CASE_IDS[0]
    store.active_case = case_id
    references = []
    for index in (46, 47):
        z, history, v = backend.state(index)
        baseline = backend.rollout(index, z, history, v)
        references.append(dict(index=index,
            current_state_fingerprint=trajectory.fingerprint((z, vars(history), v)),
            baseline_history=baseline["source_history_fingerprint"],
            baseline_terminal_fingerprint=baseline["terminal_fingerprint"],
            baseline=score(baseline["terminal"], b"key"), jacobian=np.zeros((30,3)).tolist()))
        if index == 46:
            next_z, next_history = baseline["next_z"], baseline["next_scheduler"]
            backend.nodes[47] = dict(z=next_z, v=backend.velocity(47, next_z, next_history))
            backend.snapshots[47] = next_history
    saved_case = dict(controls={"MULTI46_47_48_49": references})
    backend.count = store.count
    original_save = store.save
    class HardWorkerDeath(BaseException):
        pass
    def save_then_die():
        original_save()
        case = store.case(case_id)
        arm = f"FIXED_T{failed_index}"
        if (case["controls"][arm][0]["outcome"] == "ACCEPT"
                and case["budgets"].get(arm, {}).get("status") == "RUNNING"):
            raise HardWorkerDeath(worker_reason)
    store.save = save_then_die
    with pytest.raises(HardWorkerDeath):
        candidate.run_independent_points(store, case_id, backend, _directions(), b"key", saved_case)
    # Parent has only the persisted checkpoint; the child's Exception handler
    # never ran. This is the exact crash/timeout window after verdict save.
    recovered = candidate.SixPointStore.open(store.path)
    provisional = recovered.case(case_id)["controls"][f"FIXED_T{failed_index}"][0]
    assert provisional["outcome"] == "ACCEPT"
    assert recovered.case(case_id)["budgets"][f"FIXED_T{failed_index}"]["status"] == "RUNNING"
    assert recovered.case(case_id)["failures"] == []
    completed_before = copy.deepcopy(recovered.case(case_id)["controls"]["FIXED_T46"]) if failed_index == 47 else None
    candidate.recover_worker_failure(recovered, case_id, worker_reason)
    case = recovered.case(case_id)
    row = case["controls"][f"FIXED_T{failed_index}"][0]
    assert row["outcome"] == "ENGINEERING_FAILURE" and row["prior_outcome"] == "ACCEPT"
    assert any(trial["decision"] == "ACCEPT" for trial in row["candidates"])
    budget = case["budgets"][f"FIXED_T{failed_index}"]
    assert budget["status"] == "FAILED" and budget["spent"] > 0
    expected_complete = failed_index - 46
    assert recovered.data["point_counts"] == dict(expected=6, accepted=expected_complete,
        evaluated=expected_complete, failed=3-expected_complete, pending=3)
    if completed_before is not None:
        assert case["controls"]["FIXED_T46"] == completed_before
        assert case["budgets"]["FIXED_T46"]["status"] == "COMPLETE"
    assert case["controls"]["FIXED_T48"][0]["outcome"] == "NOT_RUN_FAILURE"
    assert case["status"] == "FAILED" and case["worker_error"] == worker_reason
    assert json.loads(store.path.read_text())["point_counts"] == recovered.data["point_counts"]
