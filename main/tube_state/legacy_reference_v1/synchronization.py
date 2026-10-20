
from __future__ import annotations

import math
from statistics import median

from .digest import compute_object_digest


OFFSET_SEARCH_MIN = -16
OFFSET_SEARCH_MAX = 16
DEFAULT_SYNC_SCALE = 1.0


def search_best_offset(
    temporal_scores: dict[int, float],
    sync_codes: dict[int, int],
    offset_search_min: int = OFFSET_SEARCH_MIN,
    offset_search_max: int = OFFSET_SEARCH_MAX,
    ground_truth_offset: int | None = None,
    center_prior_enabled: bool = True,
) -> dict[str, float | int | None]:
    if not isinstance(temporal_scores, dict) or not temporal_scores:
        raise ValueError("temporal_scores must be a non-empty dictionary")
    if not isinstance(sync_codes, dict) or not sync_codes:
        raise ValueError("sync_codes must be a non-empty dictionary")
    if not isinstance(offset_search_min, int) or not isinstance(offset_search_max, int):
        raise TypeError("offset search bounds must be integers")
    if offset_search_min > offset_search_max:
        raise ValueError("offset_search_min must not exceed offset_search_max")

    offset_scores: dict[int, float] = {}
    for offset_candidate in range(offset_search_min, offset_search_max + 1):
        offset_scores[offset_candidate] = round(
            sum(
                float(temporal_score)
                * float(sync_codes.get(temporal_index - offset_candidate, 0))
                for temporal_index, temporal_score in temporal_scores.items()
            ),
            6,
        )

    return build_offset_search_result(
        offset_scores,
        ground_truth_offset,
        center_prior_enabled=center_prior_enabled,
    )


def build_offset_search_result(
    offset_scores: dict[int, float],
    ground_truth_offset: int | None = None,
    center_prior_enabled: bool = True,
) -> dict[str, float | int | None]:
    if not isinstance(offset_scores, dict) or not offset_scores:
        raise ValueError("offset_scores must be a non-empty dictionary")

    candidates = [
        {
            "offset": int(offset_candidate),
            "scale": DEFAULT_SYNC_SCALE,
            "score": float(candidate_score),
        }
        for offset_candidate, candidate_score in offset_scores.items()
    ]
    return _build_search_result(
        candidates,
        ground_truth_offset=ground_truth_offset,
        ground_truth_scale=None,
        alignment_mode="offset",
        center_prior_enabled=center_prior_enabled,
    )


def build_offset_scale_search_result(
    candidate_scores: dict[tuple[int, float], float],
    ground_truth_offset: int | None = None,
    ground_truth_scale: float | None = None,
    center_prior_enabled: bool = True,
) -> dict[str, float | int | str | None | bool]:
    if not isinstance(candidate_scores, dict) or not candidate_scores:
        raise ValueError("candidate_scores must be a non-empty dictionary")

    candidates: list[dict[str, float | int]] = []
    for candidate_key, candidate_score in candidate_scores.items():
        if (
            not isinstance(candidate_key, tuple)
            or len(candidate_key) != 2
            or not isinstance(candidate_key[0], int)
            or not isinstance(candidate_key[1], (int, float))
        ):
            raise TypeError("candidate_scores keys must be (offset, scale) tuples")
        candidates.append(
            {
                "offset": int(candidate_key[0]),
                "scale": round(float(candidate_key[1]), 6),
                "score": float(candidate_score),
            }
        )
    return _build_search_result(
        candidates,
        ground_truth_offset=ground_truth_offset,
        ground_truth_scale=ground_truth_scale,
        alignment_mode="offset_scale",
        center_prior_enabled=center_prior_enabled,
    )


def _build_search_result(
    candidates: list[dict[str, float | int]],
    ground_truth_offset: int | None,
    ground_truth_scale: float | None,
    alignment_mode: str,
    center_prior_enabled: bool,
) -> dict[str, float | int | str | None | bool]:
    if not candidates:
        raise ValueError("candidates must be non-empty")

    ranked_candidates = sorted(
        candidates,
        key=lambda item: (
            -float(item["score"]),
            abs(int(item["offset"])) if center_prior_enabled else int(item["offset"]),
            (
                abs(float(item["scale"]) - DEFAULT_SYNC_SCALE)
                if center_prior_enabled
                else float(item["scale"])
            ),
            int(item["offset"]),
            float(item["scale"]),
        ),
    )
    best_candidate = ranked_candidates[0]
    best_offset = int(best_candidate["offset"])
    best_scale = round(float(best_candidate["scale"]), 6)
    best_raw_score = float(best_candidate["score"])
    all_scores = [float(candidate["score"]) for candidate in candidates]
    second_or_median = _resolve_second_or_median_score(ranked_candidates)
    peak_margin = round(best_raw_score - second_or_median, 6)
    positive_margin = round(max(0.0, peak_margin), 6)
    peak_rank = None
    alignment_error = None
    if ground_truth_offset is not None:
        alignment_error = abs(best_offset - ground_truth_offset)
        peak_rank = next(
            (
                rank
                for rank, candidate in enumerate(ranked_candidates, start=1)
                if _candidate_matches_ground_truth(
                    candidate,
                    ground_truth_offset,
                    ground_truth_scale,
                )
            ),
            None,
        )
    scale_error = None
    if ground_truth_scale is not None:
        scale_error = round(abs(best_scale - float(ground_truth_scale)), 6)

    sorted_candidates = sorted(
        [
            {
                "offset": int(candidate["offset"]),
                "scale": round(float(candidate["scale"]), 6),
                "score": round(float(candidate["score"]), 6),
            }
            for candidate in candidates
        ],
        key=lambda item: (int(item["offset"]), float(item["scale"])),
    )

    return {
        "sync_search_enabled": True,
        "sync_estimated_offset": best_offset,
        "sync_ground_truth_offset": ground_truth_offset,
        "sync_alignment_error": alignment_error,
        "sync_peak_rank": peak_rank,
        "sync_score": positive_margin,
        "sync_search_space_size": len(candidates),
        "sync_search_space_digest": compute_object_digest(sorted_candidates),
        "sync_score_median": round(median(all_scores), 6),
        "sync_estimated_scale": best_scale,
        "sync_ground_truth_scale": (
            None if ground_truth_scale is None else round(float(ground_truth_scale), 6)
        ),
        "sync_scale_error": scale_error,
        "sync_alignment_mode": alignment_mode,
        "S_sync_peak_best": round(best_raw_score, 6),
        "S_sync_peak_second_or_median": round(second_or_median, 6),
        "S_sync_peak_margin": peak_margin,
        "S_sync_positive_margin": positive_margin,
    }


def _resolve_second_or_median_score(
    ranked_candidates: list[dict[str, float | int]],
) -> float:
    if len(ranked_candidates) == 1:
        return float(ranked_candidates[0]["score"])
    return float(ranked_candidates[1]["score"])


def _candidate_matches_ground_truth(
    candidate: dict[str, float | int],
    ground_truth_offset: int,
    ground_truth_scale: float | None,
) -> bool:
    if int(candidate["offset"]) != int(ground_truth_offset):
        return False
    if ground_truth_scale is None:
        return True
    return abs(float(candidate["scale"]) - float(ground_truth_scale)) <= 1e-6


def _normalize_sync_score(all_scores: list[float], best_score: float) -> float:
    if len(all_scores) == 1:
        return round(float(best_score), 6)
    mean_score = sum(all_scores) / len(all_scores)
    variance = sum((score - mean_score) ** 2 for score in all_scores) / len(all_scores)
    if variance == 0.0:
        return round(float(best_score), 6)
    return round((float(best_score) - mean_score) / math.sqrt(variance), 6)
