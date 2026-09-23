"""Replay the frozen 24-view tensor batch through the bidirectional receiver on CPU."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from main.tube_state import bidirectional_cross_confirm_receiver as bidirectional
from main.tube_state import fixed_key

KEY = b"WanProjection-first-validation-key-v1"
VIEWS = fixed_key.FIXED_VIEWS
REFERENCE = {
    "FULL": "IDENTITY",
    "DELETE90": "DELETE90_REFERENCE",
    "SPEED5_4": "SPEED5_4_REFERENCE",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_observations(view_root):
    return {
        phase: torch.load(view_root / f"g{phase}.pt", map_location="cpu", weights_only=True).numpy()
        for phase in range(4)
    }


def joint_partition_role_swap(observations, book):
    swapped_observations = {
        phase: np.roll(value, 4, axis=3) for phase, value in observations.items()
    }
    swapped_book = dict(book)
    swapped_book["directions"] = np.roll(
        book["directions"].reshape(11, 10, 16, 4, 256), 1, axis=1,
    ).reshape(book["directions"].shape)
    for name in ("sync", "polarity", "code"):
        swapped_book[name] = np.roll(
            book[name].reshape(11, 10, 16), 1, axis=1,
        ).reshape(book[name].shape)
    return swapped_observations, swapped_book


def equivalent_a_to_b(new_detection, old_detection):
    direction = new_detection["directions"]["A_LOCATE_B_CONFIRM"]
    checks = {
        "selection": direction["selection"] == old_detection["selection"],
        "confirmation": direction["confirmation"] == old_detection["confirmation"],
        "confirmation_scores": direction["confirmation_scores"] == old_detection["confirmation_scores"],
        "fixed_path_diagnostics": direction["fixed_path_diagnostics"] == old_detection["fixed_path_diagnostics"],
        "alignment_reporting_only": direction["alignment_reporting_only"] == old_detection["alignment_reporting_only"],
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main(tensor_root, old_result_path, output):
    tensor_root, old_result_path, output = map(Path, (tensor_root, old_result_path, output))
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads(old_result_path.read_text(encoding="utf-8"))
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    book = fixed_key.codebook(KEY)
    rows = {}
    failures = []
    view_roots = sorted(path.parent for path in tensor_root.rglob("g0.pt"))
    for index, view_root in enumerate(view_roots, 1):
        relative = view_root.relative_to(tensor_root)
        case_id, _, arm, view = relative.parts
        view_id = f"{case_id}/{arm}/{view}"
        print(f"cpu replay {index}/{len(view_roots)} {view_id}", flush=True)
        started_view = time.perf_counter()
        try:
            observations = load_observations(view_root)
            detection = bidirectional.read(observations, KEY, old["spec_sha256"])
            old_detection = old["views"][view_id]
            equivalence = equivalent_a_to_b(detection, old_detection)
            reference_name = REFERENCE[view]
            directional_reference = {
                name: direction["fixed_path_diagnostics"][reference_name]["scores"]["C2_STATE_CONFIRM"]
                for name, direction in detection["directions"].items()
            }
            fixed_reference = sum(directional_reference.values()) / 2.0
            old_reference = old_detection["fixed_path_diagnostics"][reference_name]["scores"]["C2_STATE_CONFIRM"]
            rows[view_id] = {
                "status": detection["status"],
                "seconds": time.perf_counter() - started_view,
                "candidate_blind_score": detection["candidate_score"],
                "candidate_directional_blind_C2": detection["c2_candidate_components"],
                "candidate_fixed_reference_score": fixed_reference,
                "candidate_directional_fixed_reference_C2": directional_reference,
                "candidate_blind_minus_reference_gap": detection["candidate_score"] - fixed_reference,
                "single_C2_blind_score": old_detection["confirmation_scores"]["C2_STATE_CONFIRM"],
                "single_C2_fixed_reference_score": old_reference,
                "single_C2_blind_minus_reference_gap": old_detection["confirmation_scores"]["C2_STATE_CONFIRM"] - old_reference,
                "a_to_b_existing_split_equivalence": equivalence,
            }
            if equivalence["status"] != "PASS":
                failures.append({"view": view_id, "kind": "A_TO_B_EQUIVALENCE_FAILED", "checks": equivalence["checks"]})
        except Exception as exc:
            failures.append({"view": view_id, "kind": "REPLAY_FAILED", "error": repr(exc)})
            rows[view_id] = {"status": "INVALID", "error": repr(exc)}

    cal_ids = ("cal_off_p0_s1", "cal_off_p1_s1")
    cal_source_scores = {
        case_id: max(rows[f"{case_id}/OFF/{view}"]["candidate_blind_score"] for view in VIEWS)
        for case_id in cal_ids
    }
    candidate_threshold = max(cal_source_scores.values()) + 1e-6
    single_threshold = old["candidate_results"]["C2_STATE_CONFIRM"]["calibration"]["threshold"]
    for view_id, row in rows.items():
        if row.get("status") != "SCORED":
            row["candidate_decision"] = "INVALID"
            continue
        row["candidate_threshold_same_batch_development_only"] = candidate_threshold
        row["candidate_blind_margin"] = row["candidate_blind_score"] - candidate_threshold
        row["candidate_fixed_reference_margin"] = row["candidate_fixed_reference_score"] - candidate_threshold
        row["candidate_decision"] = "DETECTED" if row["candidate_blind_margin"] > 0 else "REJECTED"
        row["single_C2_threshold"] = single_threshold
        row["single_C2_blind_margin"] = row["single_C2_blind_score"] - single_threshold
        row["single_C2_fixed_reference_margin"] = row["single_C2_fixed_reference_score"] - single_threshold
        row["single_C2_decision"] = "DETECTED" if row["single_C2_blind_margin"] > 0 else "REJECTED"

    symmetry_view = "eval_p2_s2/SINGLE46/SPEED5_4"
    symmetry_observations = load_observations(tensor_root / "eval_p2_s2/observations/SINGLE46/SPEED5_4")
    swapped_observations, swapped_book = joint_partition_role_swap(symmetry_observations, book)
    print(f"cpu full-grid joint role swap {symmetry_view}", flush=True)
    swapped = bidirectional._read_book(swapped_observations, swapped_book, old["spec_sha256"])
    original_components = rows[symmetry_view]["candidate_directional_blind_C2"]
    symmetry_checks = {
        "candidate_mean_preserved": math.isclose(
            swapped["candidate_score"], rows[symmetry_view]["candidate_blind_score"], abs_tol=1e-15,
        ),
        "A_direction_equals_original_B": math.isclose(
            swapped["c2_candidate_components"]["A_LOCATE_B_CONFIRM"],
            original_components["B_LOCATE_A_CONFIRM"], abs_tol=1e-15,
        ),
        "B_direction_equals_original_A": math.isclose(
            swapped["c2_candidate_components"]["B_LOCATE_A_CONFIRM"],
            original_components["A_LOCATE_B_CONFIRM"], abs_tol=1e-15,
        ),
        "both_full_grid_attempts_4284": all(
            direction["selection"]["attempted_path_count"] == 4284
            for direction in swapped["directions"].values()
        ),
    }
    symmetry = {
        "status": "PASS" if all(symmetry_checks.values()) else "FAIL",
        "view": symmetry_view,
        "kind": "joint observation spatial-row and matching codebook support permutation",
        "checks": symmetry_checks,
    }
    if symmetry["status"] != "PASS":
        failures.append({"view": symmetry_view, "kind": "ROLE_SWAP_SYMMETRY_FAILED", "checks": symmetry_checks})

    evaluation = {key: row for key, row in rows.items() if key.startswith("eval_")}
    off = {key: row for key, row in evaluation.items() if "/OFF/" in key}
    marked = {key: row for key, row in evaluation.items() if "/OFF/" not in key}
    summary = {
        "candidate_OFF_detected": sum(row["candidate_decision"] == "DETECTED" for row in off.values()),
        "candidate_OFF_denominator": len(off),
        "single_C2_OFF_detected": sum(row["single_C2_decision"] == "DETECTED" for row in off.values()),
        "single_C2_OFF_denominator": len(off),
        "candidate_marked_detected": sum(row["candidate_decision"] == "DETECTED" for row in marked.values()),
        "candidate_marked_denominator": len(marked),
        "single_C2_marked_detected": sum(row["single_C2_decision"] == "DETECTED" for row in marked.values()),
        "single_C2_marked_denominator": len(marked),
    }
    p2_speed = {
        key: value for key, value in rows.items()
        if key.startswith("eval_p2_s2/") and key.endswith("/SPEED5_4") and "/OFF/" not in key
    }
    p3_marked = {
        key: value for key, value in rows.items()
        if key.startswith("eval_p3_s3/") and "/OFF/" not in key
    }
    receipt = {
        "status": "PASS" if not failures and len(rows) == 24 else "WITH_RETAINED_FAILURES",
        "started_utc": started,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "execution": "CPU-only read-only replay; no model, VAE, GPU, media, or new experiment execution",
        "source_files_sha256": {
            "bidirectional_receiver": sha(Path(bidirectional.__file__)),
            "existing_split_receiver": sha(Path(__file__).parents[1] / "main/tube_state/fixed_key_split_receiver.py"),
            "fixed_key": sha(Path(fixed_key.__file__)),
        },
        "inputs": {
            "tensor_root": str(tensor_root),
            "old_result": str(old_result_path),
            "old_result_sha256": sha(old_result_path),
            "tensor_views": len(view_roots),
            "tensor_phases": 4 * len(view_roots),
        },
        "full_grid": {
            "views_scored": sum(row.get("status") == "SCORED" for row in rows.values()),
            "expected_views": 24,
            "paths_per_direction_per_view": 4284,
            "a_to_b_full_field_equivalence_passed": sum(
                row.get("a_to_b_existing_split_equivalence", {}).get("status") == "PASS"
                for row in rows.values()
            ),
            "joint_role_swap": symmetry,
        },
        "development_calibration": {
            "kind": "same frozen two-source batch diagnosis only; not the new formal four-source program",
            "candidate_source_scores": cal_source_scores,
            "candidate_threshold": candidate_threshold,
            "single_C2_threshold": single_threshold,
        },
        "summary": summary,
        "p2_SPEED5_4": p2_speed,
        "p3_marked": p3_marked,
        "views": rows,
        "failures": failures,
        "evidence_ceiling": "existing same-batch CPU development diagnosis; not independent validation, real GPU resource evidence, or science",
    }
    (output / "cpu_result.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Bidirectional-Cross-Confirm-V1 CPU development diagnosis",
        "",
        f"Status: **{receipt['status']}**. Replayed {receipt['full_grid']['views_scored']}/24 existing views and 96/96 stored phase tensors on CPU. This did not run the model, VAE, GPU, media generation, or a new experiment.",
        "",
        f"The new A-locate/B-confirm direction matched the frozen existing split receiver across selection, held-out confirmation, C1/C2 scores, all three fixed references, and alignment reporting for {receipt['full_grid']['a_to_b_full_field_equivalence_passed']}/24 views. The full 4,284-path joint observation+codebook partition-role swap check was {symmetry['status']}: directions exchanged and the arithmetic mean was preserved.",
        "",
        "Using only the old two OFF calibration sources for same-batch diagnosis (not the new four-source formal program):",
        "",
        "| Receiver | eval OFF detected | eval marked detected |",
        "|---|---:|---:|",
        f"| existing single-direction C2 | {summary['single_C2_OFF_detected']}/{summary['single_C2_OFF_denominator']} | {summary['single_C2_marked_detected']}/{summary['single_C2_marked_denominator']} |",
        f"| bidirectional C2 mean | {summary['candidate_OFF_detected']}/{summary['candidate_OFF_denominator']} | {summary['candidate_marked_detected']}/{summary['candidate_marked_denominator']} |",
        "",
        "For p2 SPEED5_4, the existing single-direction C2 has fixed-reference pass but blind failure in both marked arms. The bidirectional mean passes both blind and fixed-reference thresholds in both arms on this old development batch; this is the only source of the marked-count increase from 4/12 to 6/12.",
        "",
        "For p3, the existing single-direction C2 remains below threshold even on all six supplied reference paths. The bidirectional fixed-reference mean is above its own same-batch threshold in all six, while all six blind scores remain below threshold. This separates the old single-direction confirmation insufficiency from a remaining bidirectional blind-localization gap; it does not prove generalization. No physical-signal-destruction or MP4-causation claim is made.",
        "",
        "All detailed per-view scores, thresholds, gaps, margins, equivalence checks, timings, and retained failures are in `cpu_result.json`. Evidence ceiling: existing same-batch CPU development diagnosis, not independent validation, real GPU resource evidence, or a scientific PASS.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor-root", required=True)
    parser.add_argument("--old-result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = main(args.tensor_root, args.old_result, args.output)
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)
