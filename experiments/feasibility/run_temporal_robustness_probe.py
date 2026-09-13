"""Run CPU-only SC-SSTW temporal robustness diagnostics.

This probe is synthetic-only feasibility triage. It checks whether AISB public
alignment plus owner/wrong-key state synchronization survives synthetic clock
edits. It does not use video, GPU, saved-video observations, fixed-FPR, or paper
claims.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "feasibility"))

from main.sc_sstw.aisb import (
    best_non_overlapping_sequence,
    make_double_redundant_templates,
    scan_burst_candidates,
    template_observation_pairs,
)
from main.sc_sstw.calibration import calibrate_from_pilot_pairs, equalize_observations
from main.sc_sstw.sync import dynamic_time_sync
from experiments.feasibility.synthetic.channel import generate_observations, make_random_channel
from run_aisb_probe import (
    _candidate_key,
    _double_redundant_mixed_states,
    _double_redundant_template_by_id,
)


def _burst_plan() -> list[tuple[int, str]]:
    return [
        (9, "double_redundant_alpha"),
        (35, "double_redundant_beta"),
        (64, "double_redundant_gamma"),
        (92, "double_redundant_alpha"),
        (118, "double_redundant_beta"),
    ]


def _missing_pair(case_index: int, ordinal: int) -> tuple[int, int]:
    first = (case_index + 2 * ordinal) % 12
    second = (first + 5 + case_index // 5) % 12
    if second == first:
        second = (second + 1) % 12
    return tuple(sorted((first, second)))


def _edit_clock_distorted[T](
    values: list[T],
    burst_plan: list[tuple[int, str]],
    *,
    case_index: int,
    double_missing: bool,
) -> tuple[list[T], list[tuple[int, str, tuple[int, int] | None]]]:
    template_length = 12
    missing_by_start = {
        start: (_missing_pair(case_index, ordinal) if double_missing else None)
        for ordinal, (start, _) in enumerate(burst_plan)
    }
    protected: set[int] = set()
    for start, _ in burst_plan:
        missing = set(missing_by_start[start] or ())
        for offset in range(template_length):
            if offset not in missing:
                protected.add(start + offset)

    crop_start = case_index % 3
    crop_end = len(values) - (case_index % 4)
    edited: list[T] = []
    source_to_first: dict[int, int] = {}
    for source_index, value in enumerate(values[crop_start:crop_end], start=crop_start):
        missing_here = any(
            source_index == start + missing
            for start, missing_pair in missing_by_start.items()
            for missing in (missing_pair or ())
        )
        if missing_here:
            continue
        if source_index not in protected and (source_index * 5 + case_index) % 19 in {0, 1, 2}:
            continue
        source_to_first.setdefault(source_index, len(edited))
        edited.append(value)
        if source_index not in protected and (source_index + 3 * case_index) % 17 in {0, 1}:
            edited.append(value)
        if source_index not in protected and source_index > len(values) // 2 and (source_index + case_index) % 29 == 0:
            edited.append(value)

    truth: list[tuple[int, str, tuple[int, int] | None]] = []
    for start, template_id in burst_plan:
        missing = set(missing_by_start[start] or ())
        present_offsets = [offset for offset in range(template_length) if offset not in missing]
        if all(start + offset in source_to_first for offset in present_offsets):
            missing_value = missing_by_start[start]
            truth.append((source_to_first[start + present_offsets[0]], template_id, missing_value))
    return edited, truth


def _case(case_index: int, *, mode: str, residual_threshold: float = 0.015) -> dict[str, object]:
    if mode not in {"clock_distortion", "combined"}:
        raise ValueError("mode must be clock_distortion or combined")
    double_missing = mode == "combined"
    owner_key = f"owner_temporal_{mode}_{case_index}"
    true_message = f"message_{case_index % 16}"
    wrong_keys = [f"wrong_temporal_{mode}_{case_index}_{index}" for index in range(16)]
    burst_plan = _burst_plan()
    length = 150
    states = _double_redundant_mixed_states(owner_key, length, burst_plan)
    observations = generate_observations(
        states,
        make_random_channel(41000 + 1000 * int(double_missing) + case_index, relation_count=16, noise_std=0.016),
        seed=42000 + 1000 * int(double_missing) + case_index,
    )
    edited, truth = _edit_clock_distorted(observations, burst_plan, case_index=case_index, double_missing=double_missing)
    templates = make_double_redundant_templates()
    candidates = scan_burst_candidates(
        edited,
        templates,
        allow_double_deletion=double_missing,
        top_k_per_start=10 if double_missing else 1,
    )
    accepted = best_non_overlapping_sequence(candidates, burst_length=templates[0].length, residual_threshold=residual_threshold, maximize_count=double_missing)
    accepted_set = {_candidate_key(candidate) for candidate in accepted}
    truth_set = set(truth)
    candidate_by_key = {_candidate_key(candidate): candidate for candidate in candidates}
    truth_covered_by_candidates = all(
        (candidate := candidate_by_key.get(key)) is not None and candidate.residual <= residual_threshold
        for key in truth_set
    )
    pilot_pairs = []
    template_by_id = _double_redundant_template_by_id()
    for candidate in accepted:
        pilot_pairs.extend(template_observation_pairs(candidate, edited, template_by_id[candidate.template_id]))
    calibration = calibrate_from_pilot_pairs(pilot_pairs)
    equalized = equalize_observations(edited, calibration)
    owner_score = dynamic_time_sync(equalized, _double_redundant_mixed_states(owner_key, length, burst_plan)).score
    wrong_scores = [
        dynamic_time_sync(equalized, _double_redundant_mixed_states(wrong_key, length, burst_plan)).score
        for wrong_key in wrong_keys
    ]
    best_wrong_score = max(wrong_scores)
    margin = owner_score - best_wrong_score
    false_positive = len(accepted_set - truth_set)
    false_negative = len(truth_set - accepted_set)
    return {
        "case_index": case_index,
        "mode": mode,
        "truth_count": len(truth_set),
        "accepted_count": len(accepted_set),
        "alignment_exact": accepted_set == truth_set,
        "truth_covered_by_candidates": truth_covered_by_candidates,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "owner_score": owner_score,
        "best_wrong_score": best_wrong_score,
        "score_margin": margin,
        "calibration_pilot_mse": calibration.pilot_reconstruction_mse,
        "pass": (
            (accepted_set == truth_set or truth_covered_by_candidates)
            and margin > 0.02
            and calibration.pilot_reconstruction_mse < 0.02
        ),
    }


def _summarize(cases: list[dict[str, object]]) -> dict[str, object]:
    margins = [float(case["score_margin"]) for case in cases]
    return {
        "case_count": len(cases),
        "pass_count": sum(1 for case in cases if case["pass"]),
        "alignment_exact_count": sum(1 for case in cases if case["alignment_exact"]),
        "truth_covered_count": sum(1 for case in cases if case["truth_covered_by_candidates"]),
        "false_positive": sum(int(case["false_positive"]) for case in cases),
        "false_negative": sum(int(case["false_negative"]) for case in cases),
        "score_margin_min": min(margins) if margins else None,
        "score_margin_mean": sum(margins) / max(1, len(margins)),
        "cases": cases,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU-only temporal robustness diagnostics.")
    parser.add_argument("--mode", choices=["clock_distortion", "combined"], required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--case-count", type=int, default=64)
    parser.add_argument("--case-jsonl", type=Path, default=None)
    parser.add_argument("--residual-threshold", type=float, default=0.015)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases = [_case(index, mode=args.mode, residual_threshold=args.residual_threshold) for index in range(args.start_index, args.start_index + args.case_count)]
    if args.case_jsonl is not None:
        with args.case_jsonl.open("w", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps(case, sort_keys=True) + "\n")
    report = {
        "status": "temporal_robustness_synthetic_only_no_video_no_gpu_no_claim",
        "mode": args.mode,
        "residual_threshold": args.residual_threshold,
        "summary": _summarize(cases),
        "synthetic_construction_pass": all(case["pass"] for case in cases),
        "paper_claim": False,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
