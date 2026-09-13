"""Candidate-key scoring after shared channel calibration."""

from __future__ import annotations

from main.sc_sstw.calibration import CalibrationResult, calibrate_channel, equalize_observations
from .state import PilotPattern, generate_state_sequence
from main.sc_sstw.sync import SyncResult, dynamic_time_sync


def score_key(
    observations: list[list[float]],
    candidate_key: str,
    *,
    pilot_pattern: PilotPattern,
    candidate_length: int | None = None,
    source_indices: list[int] | None = None,
) -> tuple[SyncResult, float, float]:
    """Score a key using key-independent calibration.

    Returns synchronization result, condition number, and pilot reconstruction
    MSE. Calibration does not depend on `candidate_key`.
    """

    calibration = calibrate_channel(
        observations,
        pilot_pattern=pilot_pattern,
        source_indices=source_indices,
    )
    equalized = equalize_observations(observations, calibration)
    candidate = generate_state_sequence(
        candidate_key,
        candidate_length or len(observations),
        pilot_pattern=pilot_pattern,
    )
    sync = dynamic_time_sync(equalized, candidate)
    return sync, calibration.condition_number, calibration.pilot_reconstruction_mse


def score_key_with_calibration(
    observations: list[list[float]],
    candidate_key: str,
    *,
    calibration: CalibrationResult,
    pilot_pattern: PilotPattern,
    candidate_length: int | None = None,
) -> SyncResult:
    """Score a candidate key using a precomputed key-independent calibration."""

    equalized = equalize_observations(observations, calibration)
    candidate = generate_state_sequence(
        candidate_key,
        candidate_length or len(observations),
        pilot_pattern=pilot_pattern,
    )
    return dynamic_time_sync(equalized, candidate)
