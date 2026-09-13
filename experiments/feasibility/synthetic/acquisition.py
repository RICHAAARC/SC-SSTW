"""Observation-only public-pilot re-acquisition prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math

from main.sc_sstw.calibration import CalibrationResult, calibrate_from_pilot_pairs, equalize_observations
from main.sc_sstw.linalg import Vector, matvec
from .state import PilotPattern


@dataclass(frozen=True)
class PilotAcquisitionResult:
    """Best public-pilot sequence found without oracle source indices."""

    calibration: CalibrationResult
    observed_indices: tuple[int, ...]
    pilot_direction_indices: tuple[int, ...]
    acquisition_score: float


def _calibration_objective(
    observations: list[Vector],
    observed_indices: tuple[int, ...],
    pilot_direction_indices: tuple[int, ...],
    pilot_directions: tuple[tuple[float, float], ...],
) -> tuple[float, CalibrationResult | None]:
    if len(observed_indices) < 4:
        return math.inf, None
    pairs = [
        (pilot_directions[direction_index], observations[observed_index])
        for observed_index, direction_index in zip(
            observed_indices,
            pilot_direction_indices,
            strict=True,
        )
    ]
    try:
        calibration = calibrate_from_pilot_pairs(pairs)
    except ValueError:
        return math.inf, None
    sample_stride = max(1, len(observations) // 16)
    sampled_observations = observations[::sample_stride][:16]
    equalized = equalize_observations(sampled_observations, calibration)
    unit_circle_mse = sum(
        ((state[0] * state[0] + state[1] * state[1]) ** 0.5 - 1.0) ** 2
        for state in equalized
    ) / max(1, len(equalized))
    # Penalize degenerate channels and state-space distortion; reward longer
    # consistent pilot chains.
    count_reward = 0.0020 * len(observed_indices)
    condition_penalty = max(0.0, calibration.condition_number - 8.0) * 0.02
    return (
        calibration.pilot_reconstruction_mse
        + 0.04 * unit_circle_mse
        + condition_penalty
        - count_reward,
        calibration,
    )


def acquire_pilots_by_periodic_beam(
    observations: list[Vector],
    *,
    pilot_pattern: PilotPattern,
    max_observed_gap: int | None = None,
    beam_width: int = 160,
    minimum_pilot_count: int = 8,
) -> PilotAcquisitionResult:
    """Find a plausible periodic public-pilot chain from observations alone.

    This is deliberately a small feasibility heuristic. It assumes crop/deletion
    edits but no arbitrary permutation. It does not use the owner key.
    """

    if len(observations) < 8:
        raise ValueError("at least eight observations are required")
    if pilot_pattern.period <= 0:
        raise ValueError("pilot period must be positive")
    pilot_directions = pilot_pattern.directions
    if len(pilot_directions) < 4:
        raise ValueError("at least four public pilot directions are required")
    max_gap = max_observed_gap or pilot_pattern.period * 2
    beams: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
    first_search = min(len(observations), pilot_pattern.period * 2)
    for observed_index in range(first_search):
        for direction_index in range(len(pilot_directions)):
            beams.append((0.0, (observed_index,), (direction_index,)))

    best: tuple[float, CalibrationResult, tuple[int, ...], tuple[int, ...]] | None = None
    for _ in range(max(1, len(observations) // max(1, pilot_pattern.period))):
        expanded: list[tuple[float, tuple[int, ...], tuple[int, ...]]] = []
        for _, observed_indices, direction_indices in beams:
            last_observed = observed_indices[-1]
            last_direction = direction_indices[-1]
            for gap in range(1, max_gap + 1):
                next_observed = last_observed + gap
                if next_observed >= len(observations):
                    continue
                # Usually the next pilot direction is next in the public code.
                # Allow one missing pilot as deletion/crop tolerance.
                for direction_step in (1, 2):
                    if pilot_pattern.cyclic:
                        next_direction = (last_direction + direction_step) % len(pilot_directions)
                    else:
                        next_direction = last_direction + direction_step
                        if next_direction >= len(pilot_directions):
                            continue
                    candidate_observed = observed_indices + (next_observed,)
                    candidate_directions = direction_indices + (next_direction,)
                    objective, calibration = _calibration_objective(
                        observations,
                        candidate_observed,
                        candidate_directions,
                        pilot_directions,
                    )
                    ranked_objective = objective
                    if calibration is not None and len(candidate_observed) >= minimum_pilot_count:
                        if best is None or ranked_objective < best[0]:
                            best = (
                                ranked_objective,
                                calibration,
                                candidate_observed,
                                candidate_directions,
                            )
                    # Before four pilots, keep path alive with neutral objective.
                    expanded.append((ranked_objective, candidate_observed, candidate_directions))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[0])
        beams = expanded[:beam_width]

    if best is None:
        raise ValueError("pilot acquisition failed")
    objective, calibration, observed_indices, direction_indices = best
    return PilotAcquisitionResult(
        calibration=calibration,
        observed_indices=observed_indices,
        pilot_direction_indices=direction_indices,
        acquisition_score=objective,
    )
