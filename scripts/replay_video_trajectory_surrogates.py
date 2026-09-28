"""Replay exactly six saved convex proxies on CPU; no model, media or GPU call."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from main.tube_state import rgb_dct_continuous_solver as solver
from main.tube_state import rgb_dct_structured_feedback as fixed
from experiments.wan_state_clock import video_trajectory_blind_validation_run as candidate


def replay(result_path, reference_path):
    config, _ = candidate.load_config()
    if candidate.shared._sha(result_path) != config["saved_result_sha256"]:
        raise ValueError("fixed saved result changed")
    result = json.loads(result_path.read_text())
    reference = json.loads(reference_path.read_text())
    report = dict(status="CPU_SURROGATE_ONLY", real_tail_executed=False,
        fixed_denominator=6, result_sha256=candidate.shared._sha(result_path),
        prior_offline_review_sha256=candidate.shared._sha(reference_path),
        source_receipt=candidate.source_receipt(), points=[])
    for case_id in candidate.shared.CASE_IDS:
        for index in candidate.POINTS:
            p = next(p for p in result["cases"][case_id]["controls"]["MULTI46_47_48_49"] if p["index"] == index)
            row = dict(source=case_id, index=index, status="PENDING")
            report["points"].append(row)
            try:
                q, J = np.array(p["baseline"]["q"]), np.array(p["jacobian"])
                rebuilt = np.column_stack([(np.array(probe["q"])-q)/probe["finite_difference_denominator"] for probe in p["probes"]])
                assert np.array_equal(J, rebuilt)
                legacy = fixed.select_coefficients(q, J, p["prediction"]["radius"])
                assert legacy["coefficients"] == [0.,0.,0.]
                answer = solver.select_coefficients(q, J, p["prediction"]["radius"])
                prior = next(r for r in reference["points"] if r["source"] == case_id and r["index"] == index)["continuous"]
                a = np.array(answer["coefficients"])
                assert answer["predicted_objective"] <= prior["objective"] + 2 * answer["tolerance"]
                assert answer["global_objective_gap_upper_bound"] <= answer["tolerance"]
                scales = []
                for scale in fixed.SCALES:
                    predicted = q + J @ (a*scale)
                    lost = np.flatnonzero((q > 0) & (predicted <= 0)).tolist()
                    gain = fixed.weighted_hinge(q,q)-fixed.weighted_hinge(predicted,q)
                    scales.append(dict(scale=scale, predicted_C=int((predicted>0).sum()),
                        predicted_lost_groups=lost, predicted_hinge_improvement=gain,
                        proxy_positive_and_hinge_guard=not lost and gain >= fixed.MIN_IMPROVEMENT,
                        no_true_native_measurement=True))
                row.update(status="CERTIFIED", legacy=legacy, continuous=answer,
                    prior_objective_absolute_delta=abs(answer["predicted_objective"]-prior["objective"]),
                    reconstructed_J_exact=True, surrogate_scales=scales)
            except Exception as exc:
                row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}",
                           solver_receipt=getattr(exc,"receipt",None))
    report["counts"] = dict(expected=6, certified=sum(p["status"]=="CERTIFIED" for p in report["points"]),
        failed=sum(p["status"]=="FAILED" for p in report["points"]))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    report=replay(args.result,args.reference)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    candidate.shared._atomic_json(args.output,report)
    print(json.dumps(report["counts"]))
    if report["counts"]["failed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
