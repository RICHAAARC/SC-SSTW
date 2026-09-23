"""Blind bidirectional locator/confirmation receiver with one frozen C2 mean score."""
from __future__ import annotations

import math

from . import fixed_key
from . import fixed_key_split_receiver as split

PROTOCOL_ID = "SC-SSTW-Bidirectional-Cross-Confirm-V1"
CANDIDATE_ID = "BIDIRECTIONAL_C2_MEAN"
CALIBRATION_SOURCE_IDS = (
    "new_cal_c0_s1",
    "new_cal_c1_s2",
    "new_cal_c2_s3",
    "new_cal_c3_s4",
)


def _select(locator):
    """Run the unchanged 4,284-path blind grid on the supplied partition."""
    best, attempted, scored = None, 0, 0
    for path in split.state_clock.clock_paths():
        attempted += 1
        row = locator.score_path(path)
        if row["status"] == "SCORED" and math.isfinite(row["score"]):
            scored += 1
            if best is None or fixed_key._rank(row) < fixed_key._rank(best):
                best = row
    return dict(best=best, attempted_path_count=attempted, scored_path_count=scored)


def _direction(observations, book, locator_indices, locator_name,
               confirmation_indices, confirmation_name):
    locator = split.PartitionEvidence(observations, book, locator_indices, locator_name)
    selection = _select(locator)
    confirmation = split.PartitionEvidence(
        observations, book, confirmation_indices, confirmation_name,
    )
    if selection["best"] is None:
        return dict(
            status="INVALID", reason="no usable locator path", selection=selection,
            locator_partition=locator_name, confirmation_partition=confirmation_name,
        )
    selected_path = dict(selection["best"]["path"])
    heldout = confirmation.score_path(selected_path)
    confirmation_values = split.confirmation_scores(heldout)
    diagnostics = {}
    for name, path in fixed_key.REFERENCE_PATHS.items():
        locator_row = locator.score_path(path)
        confirmation_row = confirmation.score_path(path)
        diagnostics[name] = dict(
            locator=locator_row,
            confirmation=confirmation_row,
            scores=split.confirmation_scores(confirmation_row),
        )
    assert selection["attempted_path_count"] == 4284
    assert confirmation.path_evaluations == 4
    alignment_input = dict(
        best=selection["best"],
        fixed_path_diagnostics={name: value["locator"] for name, value in diagnostics.items()},
    )
    status = (
        "SCORED"
        if heldout.get("status") == "SCORED"
        and all(
            confirmation_values[name] is not None
            and math.isfinite(confirmation_values[name])
            for name in split.CANDIDATES
        )
        else "INVALID"
    )
    return dict(
        status=status,
        locator_partition=locator_name,
        confirmation_partition=confirmation_name,
        selection=selection,
        confirmation=heldout,
        confirmation_scores=confirmation_values,
        fixed_path_diagnostics=diagnostics,
        confirmation_path_evaluations=confirmation.path_evaluations,
        alignment_reporting_only={
            view: fixed_key.alignment_report(alignment_input, view)
            for view in fixed_key.FIXED_VIEWS
        },
    )


def _read_book(observations, book, spec_sha):
    """Core read with an already-bound codebook; exposed for invariance tests."""
    if (
        set(observations) != set(range(4))
        or any(
            value.ndim != 5
            or value.shape[:2] != (1, 16)
            or value.shape[3:] != (40, 64)
            or not split.np.isfinite(value).all()
            for value in observations.values()
        )
    ):
        return dict(
            status="INVALID", reason="four finite valid phase tensors required",
            candidate_score=None, candidate_id=CANDIDATE_ID,
            receiver_protocol_id=PROTOCOL_ID, spec_sha256=spec_sha,
        )
    directions = {
        "A_LOCATE_B_CONFIRM": _direction(observations, book, split.A, "A", split.B, "B"),
        "B_LOCATE_A_CONFIRM": _direction(observations, book, split.B, "B", split.A, "A"),
    }
    c2 = {
        name: row.get("confirmation_scores", {}).get("C2_STATE_CONFIRM")
        for name, row in directions.items()
    }
    valid = all(
        directions[name].get("status") == "SCORED"
        and score is not None and math.isfinite(score)
        for name, score in c2.items()
    )
    score = float(sum(c2.values()) / 2.0) if valid else None
    return dict(
        status="SCORED" if valid else "INVALID",
        reason=None if valid else "both directional C2 confirmations must be valid",
        spec_sha256=spec_sha,
        key_id=book["key_id"],
        receiver_protocol_id=PROTOCOL_ID,
        candidate_id=CANDIDATE_ID,
        candidate_score=score,
        aggregation="arithmetic_mean_of_two_directional_C2_scores",
        directions=directions,
        c1_diagnostic_only={
            name: row.get("confirmation_scores", {}).get("C1_MATCHED_CONFIRM")
            for name, row in directions.items()
        },
        c2_candidate_components=c2,
        alignment_reporting_only={
            view: {
                name: row.get("alignment_reporting_only", {}).get(
                    view, {"status": "UNAVAILABLE"},
                )
                for name, row in directions.items()
            }
            for view in fixed_key.FIXED_VIEWS
        },
    )


def read(observations, key, spec_sha):
    """Return one score: mean(A-locate/B-C2, B-locate/A-C2)."""
    return _read_book(observations, fixed_key.codebook(key), spec_sha)


def source_statistic(views, spec_sha, key_id):
    expected = dict(
        spec_sha256=spec_sha,
        key_id=key_id,
        receiver_protocol_id=PROTOCOL_ID,
        candidate_id=CANDIDATE_ID,
    )
    if (
        set(views) != set(fixed_key.FIXED_VIEWS)
        or any(
            value.get("status") != "SCORED"
            or any(value.get(key) != expected_value for key, expected_value in expected.items())
            for value in views.values()
        )
    ):
        return dict(status="INVALID", statistic=None, winning_view=None, **expected)
    scores = {name: views[name].get("candidate_score") for name in fixed_key.FIXED_VIEWS}
    if any(score is None or not math.isfinite(score) for score in scores.values()):
        return dict(status="INVALID", statistic=None, winning_view=None, **expected)
    winner = max(fixed_key.FIXED_VIEWS, key=lambda view: (scores[view], view))
    return dict(
        status="SCORED", statistic=scores[winner], winning_view=winner, **expected,
    )


def calibrate(calibration_sources, spec_sha, key_id, guard=1e-6):
    binding = dict(
        candidate_id=CANDIDATE_ID,
        spec_sha256=spec_sha,
        key_id=key_id,
        receiver_protocol_id=PROTOCOL_ID,
    )
    ok = (
        set(calibration_sources) == set(CALIBRATION_SOURCE_IDS)
        and all(
            row.get("status") == "SCORED"
            and math.isfinite(row.get("statistic", math.nan))
            and all(row.get(key) == value for key, value in binding.items())
            for row in calibration_sources.values()
        )
    )
    if not math.isfinite(guard) or guard <= 0:
        raise ValueError("positive finite calibration guard required")
    return dict(
        status="FROZEN" if ok else "UNCALIBRATED",
        threshold=(
            max(row["statistic"] for row in calibration_sources.values()) + guard
            if ok else None
        ),
        guard=guard,
        sources=calibration_sources,
        source_count=4,
        empirical_rank_resolution="1/5",
        claim="functional OFF calibration; not a low-FPR estimate",
        **binding,
    )


def decide(score, status, calibration, spec_sha, key_id):
    if status != "SCORED" or score is None or not math.isfinite(score):
        return dict(status="INVALID", detected=None, statistic=score)
    binding = dict(
        candidate_id=CANDIDATE_ID,
        spec_sha256=spec_sha,
        key_id=key_id,
        receiver_protocol_id=PROTOCOL_ID,
    )
    if calibration.get("status") != "FROZEN" or any(
        calibration.get(key) != value for key, value in binding.items()
    ):
        return dict(status="UNCALIBRATED", detected=None, statistic=score)
    threshold = calibration["threshold"]
    return dict(
        status="DETECTED" if score > threshold else "REJECTED",
        detected=score > threshold,
        statistic=score,
        threshold=threshold,
    )
