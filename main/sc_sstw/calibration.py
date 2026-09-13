"""Pilot-based channel calibration and state equalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .linalg import Matrix, Vector, condition_number_2d_columns, invert_2x2, least_squares, matmul, matvec, sub, transpose


class PublicPilotSchedule(Protocol):
    """Minimal public schedule required for carrier-independent calibration."""

    def pilot_at(self, index: int) -> tuple[float, float] | None: ...


@dataclass(frozen=True)
class CalibrationResult:
    """Estimated video-specific affine relation channel."""

    matrix: Matrix
    bias: Vector
    condition_number: float
    pilot_reconstruction_mse: float


def calibrate_channel(
    observations: list[Vector],
    *,
    pilot_pattern: PublicPilotSchedule,
    source_indices: list[int] | None = None,
) -> CalibrationResult:
    """Estimate A,b from public pilots only.

    The same result is intended to be reused for owner and wrong keys.
    """

    if source_indices is not None and len(source_indices) != len(observations):
        raise ValueError("source_indices length must match observations")
    indexed_sources = source_indices or list(range(len(observations)))
    indices = [
        observed_index
        for observed_index, source_index in enumerate(indexed_sources)
        if pilot_pattern.pilot_at(source_index) is not None
    ]
    if len(indices) < 4:
        raise ValueError("at least four pilot observations are required")
    pilot_pairs: list[tuple[tuple[float, float], Vector]] = []
    for index in indices:
        pilot = pilot_pattern.pilot_at(indexed_sources[index])
        if pilot is None:
            raise RuntimeError("pilot index mismatch")
        pilot_pairs.append((pilot, observations[index]))
    return calibrate_from_pilot_pairs(pilot_pairs)


def calibrate_from_pilot_pairs(
    pilot_pairs: list[tuple[tuple[float, float], Vector]],
) -> CalibrationResult:
    """Estimate A,b from explicit public-pilot labels and observations."""

    if len(pilot_pairs) < 4:
        raise ValueError("at least four pilot pairs are required")
    features: Matrix = []
    targets: Matrix = []
    for pilot, observation in pilot_pairs:
        features.append([pilot[0], pilot[1], 1.0])
        targets.append(observation)
    beta = least_squares(features, targets)
    beta_t = transpose(beta)
    matrix = [[row[0], row[1]] for row in beta_t]
    bias = [row[2] for row in beta_t]
    error = 0.0
    count = 0
    for feature, target in zip(features, targets, strict=True):
        predicted = matvec(beta_t, feature)
        error += sum((a - b) ** 2 for a, b in zip(predicted, target, strict=True))
        count += len(target)
    return CalibrationResult(
        matrix=matrix,
        bias=bias,
        condition_number=condition_number_2d_columns(matrix),
        pilot_reconstruction_mse=error / max(1, count),
    )


def equalize_observations(
    observations: list[Vector],
    calibration: CalibrationResult,
    *,
    ridge: float = 1e-4,
) -> list[tuple[float, float]]:
    """Map q observations back into the shared 2D state space."""

    a_t = transpose(calibration.matrix)
    normal = matmul(a_t, calibration.matrix)
    inverse = invert_2x2(normal, ridge=ridge)
    projector = matmul(inverse, a_t)
    equalized: list[tuple[float, float]] = []
    for observation in observations:
        centered = sub(observation, calibration.bias)
        state = matvec(projector, centered)
        equalized.append((state[0], state[1]))
    return equalized
