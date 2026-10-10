from itertools import combinations_with_replacement
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from experiments.paper_results_v1 import receiver_controls as m
from experiments.paper_results_v1 import receiver_controls_cli as cli


def exhaustive(a, cap):
    candidates = []
    for path in combinations_with_replacement(range(a.shape[1]), len(a)):
        if m.path_statistics(list(path))["max_dwell"] <= cap:
            total = float(sum(a[t, s] for t, s in enumerate(path)))
            candidates.append((total, list(path)))
    return sorted(candidates, key=lambda x: x[0], reverse=True)


def test_duration_top_two_matches_independent_complete_enumeration():
    q = np.array([[5, -4, 1, 0], [2, 7, -3, 1], [-2, 3, 9, -1],
                  [-5, 1, 5, 8], [-4, 0, 2, 11]], dtype=float)
    expected = exhaustive(q, 4)
    actual = m._duration_top_two(q)
    assert actual["best_score"] == expected[0][0]
    assert actual["runner_up_score"] == expected[1][0]
    assert actual["best_path"] == expected[0][1]
    assert actual["runner_up_path"] != actual["best_path"]
    assert m.path_sum(q, actual["runner_up_path"]) == expected[1][0]


def test_duration_ties_retain_distinct_complete_witnesses():
    q = np.zeros((5, 4))
    expected = exhaustive(q, 4)
    actual = m._duration_top_two(q)
    assert len(expected) > 2 and actual["status"] == "UNRESOLVED"
    assert actual["path"] is None and actual["score_gap"] == 0
    assert actual["best_path"] != actual["runner_up_path"]
    for path in (actual["best_path"], actual["runner_up_path"]):
        assert (0.0, path) in expected


def test_four_repeats_are_legal_fifth_is_model_limitation_not_bad_input():
    four = np.array([[5., 0.]]*4 + [[0., 5.]]*4)
    exact = m._duration_top_two(four)
    assert exact["status"] == "ESTIMATED"
    assert exact["path"] == [0]*4 + [1]*4
    five = np.array([[5., 0.]]*5 + [[0., 5.]]*3)
    limited = m._duration_top_two(five)
    assert limited["status"] == "ESTIMATED"
    assert limited["path"] != [0]*5+[1]*3
    assert m.path_statistics(limited["path"])["max_dwell"] == 4
    assert not cli.duration_truth_feasible([[dict(source_index=s, weight=1.)] for s in [0]*5+[1]*3])


@pytest.mark.parametrize("n", [1, 2, 3])
def test_short_free_start_and_centered_ambiguity(n):
    q = np.full((n, 181), -2.)
    q[:, 87] = 3.
    assert m.select_score_path(q, "D4")["path"] == [87]*n
    background, centered = m.center_received_scores(q)
    assert np.all(centered == 0) and background[87] == 3
    for grammar in ("U", "D4"):
        answer = m.select_score_path(centered, grammar)
        assert answer["status"] == "UNRESOLVED" and answer["path"] is None
        assert answer["diagnostic_best_path"] != answer["diagnostic_runner_up_path"]


def test_free_start_arbitrary_deletion_and_finite_duration_capacity():
    q = np.full((4, 181), -100.)
    path = [37, 37, 101, 175]
    q[np.arange(4), path] = 5
    assert m.select_score_path(q, "D4")["path"] == path
    assert m.select_score_path(np.zeros((725, 181)), "D4")["reason"] == "NO_COMPLETE_DURATION_PATH"
    one = m._duration_top_two(np.array([[1.], [2.], [3.], [4.]]))
    assert one["status"] == "ESTIMATED" and one["runner_up_path"] is None and one["score_gap"] is None


def test_centering_fixed_median_even_length_and_constant_column_invariance():
    q = np.tile(np.array([0., 2., 8., 10.])[:, None], (1, 181)) + np.arange(181)
    bias, centered = m.center_received_scores(q)
    bias2, centered2 = m.center_received_scores(q + np.arange(181)*4)
    np.testing.assert_array_equal(bias, 5+np.arange(181))
    np.testing.assert_array_equal(centered, centered2)
    np.testing.assert_array_equal(bias2-bias, np.arange(181)*4)
    assert list(inspect.signature(m.select_score_path).parameters) == ["value", "grammar"]
    assert list(inspect.signature(m.center_received_scores).parameters) == ["value"]


def test_mixed_support_membership_and_weighted_distance_are_different():
    truth = [[dict(source_index=2, weight=.25), dict(source_index=3, weight=.75)]]
    report = cli.evaluate_path([2], truth)
    assert report["in_support_fraction"] == 1 and report["support_error_sum"] == 0
    assert report["weighted_error_sum"] == .75


def test_invalid_numerical_inputs_are_real_failures():
    with pytest.raises(ValueError):
        m.select_score_path(np.full((1, 181), np.nan), "D4")
    with pytest.raises(ValueError):
        m.select_score_path(np.zeros((0, 181)), "D4")
    with pytest.raises(ValueError):
        m.center_received_scores(np.zeros((3, 180)))


def write_fixture(root, pilot, attack, q):
    path = cli.score_path(root, pilot, attack)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, signed_projection=q, rho=np.ones_like(q))
    return path


def test_runner_persists_all_slots_and_selectors_before_truth(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    # Two available cases, one malformed; all other fixed cases stay MISSING.
    good = np.full((3, 181), -2.)
    good[np.arange(3), [37, 38, 90]] = 5.
    write_fixture(inputs, "p1", "full", good)
    write_fixture(inputs, "p2", "full", good)
    write_fixture(inputs, "p1", "crop37_126", np.zeros((2, 180)))
    output = tmp_path / "run"
    truth_calls = []
    original_select = m.select_score_path
    def selected(q, grammar):
        # The failed p1 posthoc stage must not prevent this later p2 condition.
        return original_select(q, grammar)
    monkeypatch.setattr(m, "select_score_path", selected)
    def truth(attack):
        truth_calls.append(attack)
        stored = json.loads((output / "run_state.json").read_text())
        pilot = "p1" if len(truth_calls) == 1 else "p2"
        rr = [r for r in stored["rows"] if (r["pilot"], r["attack"]) == (pilot, attack)]
        assert len(rr) == 4 and all("selector" in r for r in rr)
        assert all((output / "records" / pilot / attack / (r["arm"]+".json")).is_file() for r in rr)
        if pilot == "p1":
            raise ValueError("posthoc fixture failure")
        return [[dict(source_index=s, weight=1.)] for s in [37, 38, 90]]
    monkeypatch.setattr(cli, "truth_for_report", truth)
    result = cli.run(inputs, output)
    assert len(result["rows"]) == 120
    assert result["status"] == "COMPLETE_WITH_FAILURES"
    assert sum(r["status"] == "MISSING" for r in result["rows"]) == 27*4
    assert sum(r["status"] == "FAILED" for r in result["rows"]) == 4
    p1 = [r for r in result["rows"] if (r["pilot"], r["attack"]) == ("p1", "full")]
    p2 = [r for r in result["rows"] if (r["pilot"], r["attack"]) == ("p2", "full")]
    assert all(r["status"] == "ESTIMATED" and r["evaluation"]["status"] == "FAILED" for r in p1)
    assert all(r["evaluation"]["status"] == "EVALUATED" for r in p2)
    summary = cli.reports(output, result)
    assert all(r["planned_conditions"] == 30 and r["recorded_rows"] == 30 for r in summary["arms"])
    assert result["model_calls"] == result["media_calls"] == 0


def test_interruption_keeps_prior_selector_and_remaining_denominator(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    q = np.full((1, 181), -1.); q[0, 52] = 1.
    write_fixture(inputs, "p1", "full", q)
    original = m.select_score_path
    calls = []
    def interrupt(value, grammar):
        calls.append(grammar)
        if len(calls) == 2:
            raise KeyboardInterrupt("fixed interruption")
        return original(value, grammar)
    monkeypatch.setattr(m, "select_score_path", interrupt)
    output = tmp_path / "run"
    state = cli.run(inputs, output)
    assert state["status"] == "INTERRUPTED"
    assert len(state["rows"]) == 120 and len(calls) == 2
    assert state["rows"][0]["selector"]["path"] == [52]
    assert state["rows"][1]["status"] == "FAILED"
    assert all(r["status"] == "NOT_EXECUTED" for r in state["rows"][2:])
    before = json.loads((output / "run_state.json").read_text())
    cli.main(["--report-only", "--output", str(output)])
    assert json.loads((output / "run_state.json").read_text()) == before


def test_direct_cli_without_git_or_model_import(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    files = ("main/__init__.py", "main/tube_state/__init__.py",
             "main/tube_state/video_trajectory_temporal_edit_receiver_v1.py",
             "experiments/__init__.py", "experiments/paper_results_v1/__init__.py",
             "experiments/paper_results_v1/receiver_controls.py",
             "experiments/paper_results_v1/receiver_controls_cli.py")
    for name in files:
        target = source/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if (repo/name).is_file():
            shutil.copyfile(repo/name, target)
        else:
            target.write_text("")
    assert not (source / ".git").exists()
    output = tmp_path/"missing_input_run"
    result = subprocess.run([sys.executable, "-B", str(source/"experiments/paper_results_v1/receiver_controls_cli.py"),
                             "--diagnostics-root", str(tmp_path/"absent"), "--output", str(output)],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 1, result.stderr
    state = json.loads((output/"run_state.json").read_text())
    assert len(state["rows"]) == 120 and all(r["status"] == "MISSING" for r in state["rows"])
    assert state["model_calls"] == state["media_calls"] == 0
