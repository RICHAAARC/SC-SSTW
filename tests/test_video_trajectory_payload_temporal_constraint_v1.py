"""Minimal CPU construction checks, not real writer/receiver evidence."""
from dataclasses import replace

import numpy as np
import pytest

from main.tube_state import video_trajectory_payload_framewise_sync_v1 as original
from main.tube_state import video_trajectory_payload_temporal_constraint_v1 as candidate

pytestmark = pytest.mark.unit
PUBLIC = replace(original.PUBLIC, source_height=32, source_width=32)


def rows_for(before):
    return [{
        "source_tubelet": b, "source_start": 4 * b, "source_stop": min(4 * b + 4, 181),
        "patch_y": 0, "patch_x": 0, "sync_sign": original.sync_sign("watermark", b, PUBLIC),
        "signed_projection_before": q, "raw_target_delta_l2": max(0.0, 0.5 - q),
        "target_margin": 0.5,
    } for b, q in enumerate(before)]


def test_full_time_constraints_optimum_and_cross_boundary_coupling():
    # Only tubelet zero is active; every other projection remains zero.
    rows = rows_for([0.0] + [0.75] * 45)
    baseline, solved, receipt = candidate.construct_temporal_constraint_delta(rows, "watermark", PUBLIC)
    assert receipt["constraint_count"] == 46
    assert receipt["active_original_m05_constraints"] == 1
    assert np.count_nonzero(baseline[4:]) == 0
    assert np.linalg.norm(solved[4]) > 1e-6  # coupling crosses t=3 -> t=4
    assert receipt["projection_residual_max_abs"] < 1e-12
    assert receipt["stationarity_residual_max_abs"] < 1e-12
    assert receipt["delta_t"]["l2_squared"] > receipt["delta0"]["l2_squared"]
    assert receipt["delta_t"]["dt_squared"] < receipt["delta0"]["dt_squared"]
    assert receipt["delta_t"]["lambda1_objective"] < receipt["delta0"]["lambda1_objective"]
    assert receipt["delta_t"]["tubelet_boundary_edges"] == 45
    assert receipt["delta_t"]["tubelet_internal_edges"] == 135
    # Independent KKT: H*solved has no component orthogonal to the A rows.
    gradient = solved.copy()
    edges = np.diff(solved, axis=0)
    gradient[:-1] -= edges
    gradient[1:] += edges
    orthogonal = gradient.copy()
    for b in range(46):
        block = slice(4 * b, min(4 * b + 4, 181))
        direction = original.sync_sign("watermark", b, PUBLIC) * original.patch_direction(
            "watermark", b, 0, 0, PUBLIC).astype(np.float64)
        before = float(np.sum(baseline[block] * direction))
        after = float(np.sum(solved[block] * direction))
        assert after == pytest.approx(before, abs=1e-12)
        orthogonal[block] -= np.sum(gradient[block] * direction) * direction
    assert np.max(np.abs(orthogonal)) < 1e-12


def test_zero_increment_keeps_all_inactive_constraints():
    baseline, solved, receipt = candidate.construct_temporal_constraint_delta(
        rows_for([0.75] * 46), "watermark", PUBLIC)
    assert not np.any(baseline)
    assert not np.any(solved)
    assert receipt["constraint_count"] == 46
    assert receipt["delta_t"]["lambda1_objective"] == 0
    assert receipt["delta_t"]["lag1_uncentered_cosine"] is None


def test_missing_constraint_is_not_silently_dropped():
    with pytest.raises(ValueError, match="all fixed M05 projection rows"):
        candidate.construct_temporal_constraint_delta(rows_for([0.0] * 46)[:-1], "watermark", PUBLIC)
