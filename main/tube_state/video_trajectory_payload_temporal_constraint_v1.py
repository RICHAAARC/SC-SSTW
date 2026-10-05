"""Isolated ideal-float64 M05 temporal constraint candidate; no runtime path.

Minimize ||delta||^2 + ||D_t delta||^2 under all original signed tubelet
projection equalities. D_t spans all 180 adjacent pairs of the 181-frame source.
This constructs perturbations only, without source-addition rounding or media.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from . import video_trajectory_payload_framewise_sync_v1 as framewise


def perturbation_structure(delta: np.ndarray) -> dict[str, Any]:
    """Describe a full perturbation, not RGB motion or perceived flicker."""
    value = np.asarray(delta, dtype=np.float64)
    if value.ndim != 4 or value.shape[0] != 181 or not np.isfinite(value).all():
        raise ValueError("expected a finite 181-frame [T,C,H,W] perturbation")
    energy = float(np.sum(value * value))
    edge_energy = np.sum(np.diff(value, axis=0) ** 2, axis=(1, 2, 3))
    boundary = np.arange(1, 181) % 4 == 0
    roughness = float(np.sum(edge_energy))
    prev_energy = float(np.sum(value[:-1] ** 2))
    next_energy = float(np.sum(value[1:] ** 2))
    lag1_inner = float(np.sum(value[:-1] * value[1:]))
    return {
        "l2_squared": energy,
        "l2": float(np.sqrt(energy)),
        "dt_squared": roughness,
        "dt_boundary_squared": float(np.sum(edge_energy[boundary])),
        "dt_internal_squared": float(np.sum(edge_energy[~boundary])),
        "lambda1_objective": energy + roughness,
        "lag1_inner_product": lag1_inner,
        "lag1_uncentered_cosine": (
            lag1_inner / float(np.sqrt(prev_energy * next_energy))
            if prev_energy and next_energy else None
        ),
        "normalized_roughness_dt2_over_l2_squared": roughness / energy if energy else None,
        "adjacent_edges": 180,
        "tubelet_boundary_edges": int(np.sum(boundary)),
        "tubelet_internal_edges": int(np.sum(~boundary)),
    }


def construct_temporal_constraint_delta(
    m05_rows: Sequence[dict[str, Any]],
    key: str,
    public: framewise.PublicProtocol = framewise.PUBLIC,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return ideal M05 delta, the fixed-lambda=1 solution, and its receipt.

    All 46 rows per spatial patch, including inactive zero increments, are
    required. Public seeds, signs and directions are reused without alteration.
    No source latent, payload or receiver input is used.
    """
    framewise.public_receipt(public)
    if public.source_frames != 181 or (public.patch_height, public.patch_width) != (4, 4):
        raise ValueError("this candidate fixes the original 181-frame, 4x4-patch support")
    patches = framewise.spatial_patch_coordinates(public.latent_height, public.latent_width, public)
    expected = {(b, y, x) for b in range(46) for y, x in patches}
    indexed: dict[tuple[int, int, int], dict[str, Any]] = {}
    for row in m05_rows:
        coordinate = (row["source_tubelet"], row["patch_y"], row["patch_x"])
        if coordinate not in expected or coordinate in indexed:
            raise ValueError("unexpected or duplicate M05 projection row")
        b, _, _ = coordinate
        if (row["source_start"], row["source_stop"]) != (4 * b, min(4 * b + 4, 181)):
            raise ValueError("M05 row support differs from the public tubelet")
        if row["sync_sign"] != framewise.sync_sign(key, b, public):
            raise ValueError("M05 row sign differs from the unchanged public codebook")
        before, raw = float(row["signed_projection_before"]), float(row["raw_target_delta_l2"])
        if (row["target_margin"] != 0.5 or not np.isfinite([before, raw]).all()
                or raw != max(0.0, 0.5 - before)):
            raise ValueError("row is not an unchanged M05 ideal increment")
        indexed[coordinate] = row
    if set(indexed) != expected:
        raise ValueError("all fixed M05 projection rows, including inactive rows, are required")

    # Natural endpoints: H diagonal (2,3,...,3,2), off-diagonal -1.
    # Time is never split at tubelet boundaries; only spatial patches separate.
    difference = np.diff(np.eye(181, dtype=np.float64), axis=0)
    hessian = np.eye(181, dtype=np.float64) + difference.T @ difference
    h_inverse = np.linalg.solve(hessian, np.eye(181, dtype=np.float64))
    shape = (181, 4, public.latent_height, public.latent_width)
    delta0, delta_t = np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=np.float64)
    projection_rows: list[dict[str, Any]] = []
    max_stationarity = 0.0
    for y, x in patches:
        signed_directions = [
            framewise.sync_sign(key, b, public)
            * framewise.patch_direction(key, b, y, x, public).astype(np.float64).reshape(-1, 64)
            for b in range(46)
        ]
        # W = H^-1 A^T, using each A row's at-most-four-frame support.
        w = np.empty((181, 64, 46), dtype=np.float64)
        baseline = np.zeros((181, 64), dtype=np.float64)
        b_vector = np.empty(46, dtype=np.float64)
        for b, direction in enumerate(signed_directions):
            start, stop = 4 * b, min(4 * b + 4, 181)
            raw = float(indexed[(b, y, x)]["raw_target_delta_l2"])
            baseline[start:stop] = raw * direction
            b_vector[b] = float(np.sum(baseline[start:stop] * direction))
            w[:, :, b] = h_inverse[:, start:stop] @ direction
        schur = np.empty((46, 46), dtype=np.float64)
        for b, direction in enumerate(signed_directions):
            start, stop = 4 * b, min(4 * b + 4, 181)
            schur[b] = np.einsum("tc,tck->k", direction, w[start:stop])
        multipliers = np.linalg.solve(schur, b_vector)
        solved = np.einsum("tck,k->tc", w, multipliers)
        adjoint = np.zeros_like(solved)
        cast_only = solved.astype(np.float32).astype(np.float64)
        for b, direction in enumerate(signed_directions):
            start, stop = 4 * b, min(4 * b + 4, 181)
            adjoint[start:stop] = multipliers[b] * direction
            projection = float(np.sum(solved[start:stop] * direction))
            cast_projection = float(np.sum(cast_only[start:stop] * direction))
            projection_rows.append({
                "source_tubelet": b, "patch_y": y, "patch_x": x,
                "active_original_m05": bool(indexed[(b, y, x)]["raw_target_delta_l2"] > 0),
                "delta0_signed_projection": float(b_vector[b]),
                "delta_t_signed_projection": projection,
                "projection_residual": projection - float(b_vector[b]),
                "delta_t_cast_only_float32_projection_residual": cast_projection - float(b_vector[b]),
            })
        max_stationarity = max(max_stationarity, float(np.max(np.abs(hessian @ solved - adjoint))))
        delta0[:, :, y:y + 4, x:x + 4] = baseline.reshape(181, 4, 4, 4)
        delta_t[:, :, y:y + 4, x:x + 4] = solved.reshape(181, 4, 4, 4)
    residuals = np.array([row["projection_residual"] for row in projection_rows])
    cast_residuals = np.array([
        row["delta_t_cast_only_float32_projection_residual"] for row in projection_rows
    ])
    return delta0, delta_t, {
        "construction": "full181-equality-constrained-temporal-lambda1",
        "public_method_version": public.method_version,
        "key_id": framewise.key_identifier(key),
        "lambda": 1.0,
        "target_margin": 0.5,
        "dtype": "float64 ideal perturbations",
        "source_shape": list(shape),
        "spatial_patches": len(patches),
        "constraints_per_patch": 46,
        "constraint_count": len(projection_rows),
        "active_original_m05_constraints": sum(row["active_original_m05"] for row in projection_rows),
        "projection_residual_max_abs": float(np.max(np.abs(residuals))),
        "projection_residual_l2": float(np.linalg.norm(residuals)),
        "stationarity_residual_max_abs": max_stationarity,
        "cast_only_float32_projection_residual_max_abs": float(np.max(np.abs(cast_residuals))),
        "delta0": perturbation_structure(delta0),
        "delta_t": perturbation_structure(delta_t),
        "projection_rows": projection_rows,
        "evidence_ceiling": (
            "Ideal latent perturbation construction only, fixed lambda=1 and all full-support constraints. "
            "No source latent: (z+delta)-z float32 rounding is unmeasured. Full projections do not guarantee "
            "partial-crop or wrong-offset projections, VAE/MP4 scores, payload, RGB quality, perception, "
            "or generalization. L2 budget is not held fixed."
        ),
    }
