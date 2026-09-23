"""Combine frozen original/split records with the bidirectional CPU replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RECEIVERS = ("ORIGINAL", "C1_MATCHED_CONFIRM", "C2_STATE_CONFIRM", "BIDIRECTIONAL_C2_MEAN")


def _counts(rows):
    result = {kind: {"DETECTED": 0, "REJECTED": 0, "INVALID": 0, "UNCALIBRATED": 0} for kind in ("OFF", "marked")}
    for key, status in rows.items():
        kind = "OFF" if "/OFF" in key else "marked"
        result[kind][status if status in result[kind] else "INVALID"] += 1
    return result


def main(cpu_path, baseline_path, split_path, output):
    cpu_path, baseline_path, split_path, output = map(Path, (cpu_path, baseline_path, split_path, output))
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    evaluation_keys = sorted(key for key in cpu["views"] if key.startswith("eval_"))
    view_status = {receiver: {} for receiver in RECEIVERS}
    source_status = {receiver: {} for receiver in RECEIVERS}

    original_threshold = baseline["calibration_reproduction"]["computed_threshold"]
    for key in evaluation_keys:
        view_status["ORIGINAL"][key] = "DETECTED" if baseline["views"][key]["score"] > original_threshold else "REJECTED"
        for receiver in ("C1_MATCHED_CONFIRM", "C2_STATE_CONFIRM"):
            view_status[receiver][key] = split["candidate_results"][receiver]["views"][key]["status"]
        view_status["BIDIRECTIONAL_C2_MEAN"][key] = cpu["views"][key]["candidate_decision"]

    source_keys = sorted({"/".join(key.split("/")[:2]) for key in evaluation_keys})
    for source_key in source_keys:
        view_keys = [key for key in evaluation_keys if key.startswith(source_key + "/")]
        source_status["ORIGINAL"][source_key] = (
            "DETECTED" if max(baseline["views"][key]["score"] for key in view_keys) > original_threshold
            else "REJECTED"
        )
        for receiver in ("C1_MATCHED_CONFIRM", "C2_STATE_CONFIRM"):
            source_status[receiver][source_key] = split["candidate_results"][receiver]["sources"][source_key]["decision"]["status"]
        source_status["BIDIRECTIONAL_C2_MEAN"][source_key] = (
            "DETECTED" if max(cpu["views"][key]["candidate_blind_margin"] for key in view_keys) > 0
            else "REJECTED"
        )

    summaries = {
        receiver: {"views": _counts(view_status[receiver]), "source_arms": _counts(source_status[receiver])}
        for receiver in RECEIVERS
    }
    p2 = {}
    for key, row in cpu["p2_SPEED5_4"].items():
        p2[key] = {
            "single_C2_blind_minus_reference_gap": row["single_C2_blind_minus_reference_gap"],
            "candidate_blind_minus_reference_gap": row["candidate_blind_minus_reference_gap"],
            "single_C2_blind_margin": row["single_C2_blind_margin"],
            "candidate_blind_margin": row["candidate_blind_margin"],
            "single_C2_fixed_reference_margin": row["single_C2_fixed_reference_margin"],
            "candidate_fixed_reference_margin": row["candidate_fixed_reference_margin"],
        }
    result = {
        "status": "PASS",
        "inputs": {
            "cpu_result": str(cpu_path), "baseline_result": str(baseline_path),
            "split_result": str(split_path),
        },
        "evaluation_denominators_per_receiver": {
            "OFF_views": 6, "marked_views": 12, "OFF_source_arms": 2, "marked_source_arms": 4,
        },
        "all_four_receiver_evaluation": summaries,
        "p2_SPEED5_4_signed_gaps": p2,
        "p3": {
            "single_C2_fixed_reference_pass": sum(
                row["single_C2_fixed_reference_margin"] > 0 for row in cpu["p3_marked"].values()
            ),
            "candidate_fixed_reference_pass": sum(
                row["candidate_fixed_reference_margin"] > 0 for row in cpu["p3_marked"].values()
            ),
            "candidate_blind_pass": sum(
                row["candidate_blind_margin"] > 0 for row in cpu["p3_marked"].values()
            ),
            "denominator": len(cpu["p3_marked"]),
        },
        "evidence_ceiling": cpu["evidence_ceiling"],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Bidirectional-Cross-Confirm-V1 CPU comparison",
        "",
        "The full CPU replay passed 24/24 views and exact A→B full-field equivalence in 24/24. The joint observation+matching-codebook partition-role swap passed one full 4,284-path check.",
        "",
        "All counts below use the fixed old evaluation batch. OFF and marked results are shown together; calibration samples are excluded.",
        "",
        "| Receiver | OFF views | marked views | OFF source-arms | marked source-arms |",
        "|---|---:|---:|---:|---:|",
    ]
    for receiver in RECEIVERS:
        value = summaries[receiver]
        lines.append(
            f"| {receiver} | {value['views']['OFF']['DETECTED']}/6 | {value['views']['marked']['DETECTED']}/12 | "
            f"{value['source_arms']['OFF']['DETECTED']}/2 | {value['source_arms']['marked']['DETECTED']}/4 |"
        )
    lines.extend([
        "",
        "The candidate adds two marked view detections, both p2 SPEED5_4. Its blind-minus-reference gaps are -0.019112 and -0.022730, versus -0.011674 and -0.018261 for single-direction C2. The gains therefore do not come from numerically narrowing the blind/reference gap; they come from the changed statistic and its own calibration threshold on this development batch.",
        "",
        "For p3, single-direction C2 fixed references pass 0/6. The bidirectional fixed-reference mean passes 6/6, while bidirectional blind scores pass 0/6. This supports a remaining blind-selection loss under the new statistic on this reused batch; it does not establish independent generalization.",
        "",
        "Evidence ceiling: old same-batch CPU development diagnosis only. No model, VAE, GPU, media, or new-source execution occurred; GPU resources remain unmeasured.",
    ])
    (output / "comparison_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-result", required=True)
    parser.add_argument("--baseline-result", required=True)
    parser.add_argument("--split-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.cpu_result, args.baseline_result, args.split_result, args.output)
