"""CPU-only fixed 30-case / four-arm saved-score comparison and posthoc report.

Run with python -m experiments.paper_results_v1.receiver_controls_cli. The
selector module never receives filenames, case labels, recipes, or truth.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from experiments.paper_results_v1 import receiver_controls as method

ATTACKS = ("full", "crop37_126", "speed075", "speed125", "mean3", "mean5",
           "delete_g0", "delete_g1", "delete_g2", "repeat_g0", "repeat_g1", "repeat_g2",
           "interp_g0", "interp_g1", "interp_g2")
PILOTS = ("p1", "p2")
OLD_AUDIT = "a-line-temporal-attack-20261010T031617441784Z-2536172d-audit"
NEW_AUDIT = "a-line-m05-sync-root-cause-20261010"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def score_path(diagnostics, pilot, attack):
    root = Path(diagnostics)
    if attack == "full":
        return root / OLD_AUDIT / "records" / ("pilot_0" + pilot[-1]) / "clock" / "PAYLOAD_FRAMEWISE_M05" / attack / "K0.npz"
    return root / NEW_AUDIT / "evidence" / pilot / "PAYLOAD_FRAMEWISE_M05" / attack / "K0.npz"


def load_scores(path):
    with np.load(path, allow_pickle=False) as data:
        numerator = np.asarray(data["signed_projection"], dtype=np.float64)
        rho = np.asarray(data["rho"], dtype=np.float64)
    if numerator.shape != rho.shape:
        raise ValueError("signed_projection/rho shapes differ")
    if not np.isfinite(rho).all() or np.any(rho <= 0):
        raise ValueError("rho must be finite positive")
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        return method.received_scores(numerator / rho)


def truth_for_report(attack):
    # This function is invoked only after the four selector records are durable.
    from experiments.paper_results_v1.attack_matrix import fixed_attack_specs, expand_weighted_truth
    spec = next(s for s in fixed_attack_specs() if s["attack_id"] == attack)
    return expand_weighted_truth(spec)


def duration_truth_feasible(truth):
    """Posthoc boolean reachability only; never used by a selector."""
    reachable = np.zeros((181, 4), dtype=bool)
    for item in truth[0]:
        reachable[item["source_index"], 0] = True
    for support in truth[1:]:
        current = np.zeros_like(reachable)
        earlier = np.cumsum(reachable.any(axis=1))
        for item in support:
            s = item["source_index"]
            current[s, 0] = s > 0 and earlier[s-1] > 0
            current[s, 1:] = reachable[s, :-1]
        reachable = current
    return bool(reachable.any())


def evaluate_path(path, truth):
    if len(path) != len(truth):
        raise ValueError("path and posthoc truth lengths differ")
    frames = []
    for t, (selected, support) in enumerate(zip(path, truth)):
        allowed = [x for x in support if x["weight"] > 0]
        if not allowed:
            raise ValueError("truth support is empty")
        minimum = min(abs(selected-x["source_index"]) for x in allowed)
        weighted = math.fsum(x["weight"]*abs(selected-x["source_index"]) for x in allowed)
        frames.append(dict(received_index=t, selected_source=selected,
                           positive_weight_support=allowed, minimum_support_error=minimum,
                           weighted_error=weighted))
    errors = [r["minimum_support_error"] for r in frames]
    weighted = [r["weighted_error"] for r in frames]
    inside = sum(e == 0 for e in errors)
    return dict(status="EVALUATED", frames=len(path), start_error=errors[0],
                in_support_frames=inside, in_support_fraction=inside/len(path),
                all_frames_in_support=inside == len(path),
                support_error_sum=sum(errors), support_error_mean=sum(errors)/len(path),
                support_error_max=max(errors), weighted_error_sum=math.fsum(weighted),
                weighted_error_mean=math.fsum(weighted)/len(path), weighted_error_max=max(weighted),
                frame_rows=frames)


def initial_state(diagnostics):
    return dict(schema="m05-receiver-controls-v1", status="RUNNING", started_at=time.time(),
                evidence_level="same_batch_development_not_confirmation", fixed_max_dwell=4,
                planned_conditions=30, planned_rows=120, model_calls=0, media_calls=0,
                background="per_input_column_median", arms=list(method.ARMS), failures=[],
                rows=[dict(pilot=p, attack=a, arm=arm, status="NOT_EXECUTED", reason="not_started",
                           evaluation=dict(status="NOT_EVALUATED", reason="not_started"))
                      for p in PILOTS for a in ATTACKS for arm in method.ARMS],
                sources=[dict(pilot=p, attack=a, path=str(score_path(diagnostics, p, a)))
                         for p in PILOTS for a in ATTACKS])


def reports(output, state):
    output = Path(output)
    compact = []
    for row in state["rows"]:
        selector = row.get("selector", {})
        stats = selector.get("path_statistics") or {}
        ev = row.get("evaluation", {})
        c = {k: row.get(k) for k in ("pilot", "attack", "arm", "status", "reason", "received_frames")}
        c.update(evaluation_status=ev.get("status"), evaluation_reason=ev.get("reason"),
                 source_start=stats.get("source_start"), source_end=stats.get("source_end"),
                 max_dwell=stats.get("max_dwell"), distinct_sources=stats.get("distinct_sources"),
                 objective=selector.get("best_score"), runner_up_objective=selector.get("runner_up_score"),
                 gap=selector.get("score_gap"), exact_tie=selector.get("exact_tie"),
                 raw_q_path_sum=row.get("raw_q_path_sum"), centered_q_path_sum=row.get("centered_q_path_sum"),
                 start_error=ev.get("start_error"), in_support_frames=ev.get("in_support_frames"),
                 in_support_fraction=ev.get("in_support_fraction"),
                 all_frames_in_support=ev.get("all_frames_in_support"),
                 support_error_sum=ev.get("support_error_sum"), support_error_mean=ev.get("support_error_mean"),
                 support_error_max=ev.get("support_error_max"), weighted_error_sum=ev.get("weighted_error_sum"),
                 weighted_error_mean=ev.get("weighted_error_mean"), weighted_error_max=ev.get("weighted_error_max"),
                 truth_d4_feasible=row.get("truth_d4_feasible"))
        compact.append(c)
    target = output / "case_arm_rows.csv"
    temporary = target.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(compact[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(compact)
    temporary.replace(target)
    arms = []
    for arm in method.ARMS:
        rows = [r for r in compact if r["arm"] == arm]
        valid = [r for r in rows if r["evaluation_status"] == "EVALUATED"]
        count = len(valid)
        estimated = [r for r in rows if r["status"] == "ESTIMATED"]
        n = sum(r["received_frames"] for r in valid)
        arms.append(dict(arm=arm, planned_conditions=30, recorded_rows=len(rows),
            status_counts=dict(Counter(r["status"] for r in rows)),
            evaluation_status_counts=dict(Counter(r["evaluation_status"] for r in rows)),
            evaluated_conditions=count, evaluated_received_frames=n,
            received_frames_with_loaded_input=sum(r["received_frames"] or 0 for r in rows),
            support_error_mean_equal_case=None if not count else math.fsum(r["support_error_mean"] for r in valid)/count,
            weighted_error_mean_equal_case=None if not count else math.fsum(r["weighted_error_mean"] for r in valid)/count,
            in_support_fraction_equal_case=None if not count else math.fsum(r["in_support_fraction"] for r in valid)/count,
            all_frames_in_support_conditions=sum(bool(r["all_frames_in_support"]) for r in valid),
            zero_start_error_conditions=sum(r["start_error"] == 0 for r in valid),
            in_support_frames=sum(r["in_support_frames"] for r in valid),
            in_support_fraction_evaluated_frames=None if not n else sum(r["in_support_frames"] for r in valid)/n,
            support_error_mean_evaluated_frames=None if not n else sum(r["support_error_sum"] for r in valid)/n,
            weighted_error_mean_evaluated_frames=None if not n else math.fsum(r["weighted_error_sum"] for r in valid)/n,
            maximum_dwell_among_estimated=None if not estimated else max(r["max_dwell"] for r in estimated)))
    summary = dict(status=state["status"], planned_rows=120, recorded_rows=len(compact),
                   model_calls=0, media_calls=0, evidence_level=state["evidence_level"], arms=arms,
                   failure_count=len(state["failures"]),
                   note="All 30 cases per arm retained; error means condition on evaluated paths and show coverage. Different-score objective totals are not improvement measures.")
    write_json(output / "summary.json", summary)
    return summary


def run(diagnostics, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    state = initial_state(diagnostics)
    write_json(output / "run_state.json", state)
    reports(output, state)
    try:
        for pilot in PILOTS:
            for attack in ATTACKS:
                rows = [r for r in state["rows"] if (r["pilot"], r["attack"]) == (pilot, attack)]
                source = score_path(diagnostics, pilot, attack)
                directory = output / "records" / pilot / attack
                directory.mkdir(parents=True, exist_ok=True)
                try:
                    q = load_scores(source)
                except Exception as exc:
                    status = "MISSING" if isinstance(exc, OSError) else "FAILED"
                    for row in rows:
                        row.update(status=status, reason=f"{type(exc).__name__}: {exc}")
                        row["evaluation"] = dict(status="NOT_EVALUATED", reason=status)
                    state["failures"].append(dict(pilot=pilot, attack=attack, stage="input", reason=repr(exc)))
                    write_json(output / "run_state.json", state)
                    continue
                for row in rows:
                    row["received_frames"] = len(q)
                centered = background = None
                center_error = None
                try:
                    background, centered = method.center_received_scores(q)
                    np.savez_compressed(directory / "scores.npz", q=q, background=background, centered=centered)
                except Exception as exc:
                    center_error = f"{type(exc).__name__}: {exc}"
                    state["failures"].append(dict(pilot=pilot, attack=attack, stage="center_or_save", reason=center_error))
                    if centered is not None:
                        # Save failure is evidence failure, not a changed selector.
                        for row in rows:
                            row["score_artifact_error"] = center_error
                for row in rows:
                    row.update(status="RUNNING", reason=None)
                    write_json(output / "run_state.json", state)
                    try:
                        local = centered if row["arm"].startswith("CENTERED") else q
                        if local is None:
                            raise ValueError(center_error)
                        selector = method.select_score_path(local, row["arm"].split("_")[-1])
                        row.update(status=selector["status"], reason=selector["reason"], selector=selector)
                        row["raw_q_path_sum"] = method.path_sum(q, selector["path"])
                        row["centered_q_path_sum"] = None if centered is None else method.path_sum(centered, selector["path"])
                    except Exception as exc:
                        row.update(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                        state["failures"].append(dict(pilot=pilot, attack=attack, arm=row["arm"], stage="selector", reason=repr(exc)))
                    # Durable selector output precedes any truth construction.
                    write_json(directory / (row["arm"] + ".json"), row)
                    write_json(output / "run_state.json", state)
                try:
                    truth = truth_for_report(attack)
                    if len(truth) != len(q):
                        raise ValueError("received frame length differs from fixed posthoc recipe")
                    feasible = duration_truth_feasible(truth)
                    for row in rows:
                        row["truth_d4_feasible"] = feasible
                        selector = row.get("selector", {})
                        if row["status"] == "ESTIMATED":
                            row["evaluation"] = evaluate_path(selector["path"], truth)
                        else:
                            row["evaluation"] = dict(status="NOT_EVALUATED", reason=row["status"])
                except Exception as exc:
                    for row in rows:
                        row["evaluation"] = dict(status="FAILED", reason=f"{type(exc).__name__}: {exc}")
                    state["failures"].append(dict(pilot=pilot, attack=attack, stage="posthoc", reason=repr(exc)))
                # Legacy history is optional reporting, not a selector input or gate.
                try:
                    old = json.loads(source.with_suffix(".json").read_text())["estimate"]
                    raw = rows[0].get("selector", {})
                    rows[0]["legacy_comparison"] = dict(
                        best_path_equal=raw.get("diagnostic_best_path") == old.get("best_path"),
                        best_score_difference=None if raw.get("best_score") is None else raw["best_score"]-old["best_score"])
                except Exception as exc:
                    rows[0]["legacy_comparison"] = dict(status="UNAVAILABLE", reason=repr(exc))
                for row in rows:
                    write_json(directory / (row["arm"] + ".json"), row)
                write_json(output / "run_state.json", state)
                reports(output, state)
                print(f"{pilot}/{attack}: " + ", ".join(r["arm"] + "=" + r["status"] for r in rows), flush=True)
        state["status"] = "COMPLETE" if not state["failures"] else "COMPLETE_WITH_FAILURES"
    except BaseException as exc:
        state["status"] = "INTERRUPTED" if not isinstance(exc, Exception) else "FAILED"
        state["failures"].append(dict(stage="run", reason=f"{type(exc).__name__}: {exc}"))
        for row in state["rows"]:
            if row["status"] == "RUNNING":
                row.update(status="FAILED", reason="interrupted_or_failed: " + repr(exc))
                row["evaluation"] = dict(status="NOT_EVALUATED", reason=row["reason"])
    finally:
        state["finished_at"] = time.time()
        write_json(output / "run_state.json", state)
        reports(output, state)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-only", action="store_true", help="refresh CSV/summary from an existing partial or complete state without selection")
    args = parser.parse_args(argv)
    if args.report_only:
        state = json.loads((args.output / "run_state.json").read_text())
        summary = reports(args.output, state)
        print(json.dumps(summary))
        return 0
    if args.diagnostics_root is None:
        parser.error("--diagnostics-root is required for the fixed comparison")
    result = run(args.diagnostics_root, args.output)
    print(json.dumps(dict(status=result["status"], output=str(args.output), rows=len(result["rows"]), failures=len(result["failures"]))))
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
