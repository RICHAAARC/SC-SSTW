"""Deterministic FP64 convex solver for the unchanged three-direction surrogate.

A Frank-Wolfe gap bounds objective suboptimality over the entire nonnegative
simplex. This is a numerical convex-surrogate certificate, not a real-tail claim.
"""
from __future__ import annotations

import math
import numpy as np
from main.tube_state import rgb_dct_structured_feedback as fixed

SPEC = dict(name="projected_gradient_fw_gap_v1", dtype="float64",
            max_iterations=200000, absolute_gap_tolerance=1e-18,
            relative_gap_tolerance=1e-11, roundoff_factor=64)


class SolverFailure(RuntimeError):
    def __init__(self, receipt):
        self.receipt = receipt
        super().__init__(receipt["reason"])


def _project(x, radius):
    positive = np.maximum(x, 0.0)
    if positive.sum() <= radius:
        return positive
    ordered = np.sort(positive)[::-1]
    excess = np.cumsum(ordered) - radius
    active = np.flatnonzero(ordered - excess / np.arange(1, len(x) + 1) > 0)
    if not len(active):
        return np.zeros_like(x)
    k = int(active[-1])
    projected = np.maximum(positive - excess[k] / (k + 1), 0.0)
    # Inward rounding: the returned point is feasible without budget tolerance.
    total = float(projected.sum())
    if total > radius:
        projected *= np.nextafter(radius / total, 0.0)
    return projected


def select_coefficients(q, jacobian, remaining,
                        available=(True, True, True), *,
                        max_iterations=SPEC["max_iterations"],
                        absolute_tolerance=SPEC["absolute_gap_tolerance"],
                        relative_tolerance=SPEC["relative_gap_tolerance"]):
    receipt = dict(solver=dict(SPEC), status="FAILED", reason=None,
                   coefficients=None, predicted_q=None, iterations=0)
    def fail(reason):
        receipt["reason"] = reason
        raise SolverFailure(receipt.copy())
    q = np.asarray(q, dtype=np.float64)
    J = np.asarray(jacobian, dtype=np.float64)
    mask = np.asarray(available)
    if (q.shape != (30,) or J.shape != (30, 3) or mask.shape != (3,)
            or mask.dtype != np.bool_):
        fail("INVALID_SHAPE_OR_AVAILABILITY")
    if not np.isfinite(q).all() or not np.isfinite(J).all():
        fail("NONFINITE_INPUT")
    if not math.isfinite(remaining) or not 0 <= remaining <= fixed.R_STAR:
        fail("INVALID_REMAINING_BUDGET")
    if (not isinstance(max_iterations, int) or max_iterations < 0
            or not math.isfinite(absolute_tolerance) or absolute_tolerance < 0
            or not math.isfinite(relative_tolerance) or relative_tolerance < 0):
        fail("INVALID_NUMERICAL_SETTINGS")
    radius = min(fixed.POINT_CAP, remaining)
    weights = np.where(q > 0, 4.0, 1.0)
    # Unavailable directions cannot influence projection, smoothness or the LMO.
    J = J.copy()
    J[:, ~mask] = 0.0
    a = np.zeros(3)
    receipt.update(radius=radius, available=mask.tolist(),
                   max_iterations=max_iterations)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            hessian_bound = 2 / 30 * J.T @ (weights[:, None] * J)
            hessian_bound += 2 * fixed.RIDGE * np.eye(3)
            lipschitz = float(np.linalg.eigvalsh(hessian_bound)[-1])
            if not math.isfinite(lipschitz) or lipschitz <= 0:
                fail("INVALID_LIPSCHITZ_BOUND")
            def evaluate(coeff):
                prediction = q + J @ coeff
                residual = np.maximum(fixed.MARGIN - prediction, 0.0)
                objective = float(np.mean(weights * residual ** 2)
                                  + fixed.RIDGE * (coeff @ coeff))
                gradient = -2 / 30 * J.T @ (weights * residual) + 2 * fixed.RIDGE * coeff
                if not math.isfinite(objective) or not np.isfinite(gradient).all():
                    fail("NONFINITE_ITERATE")
                return objective, gradient, prediction
            f0, g0, _ = evaluate(a)
            tolerance = max(absolute_tolerance, relative_tolerance * f0)
            receipt.update(zero_objective=f0, tolerance=tolerance,
                           derivative_at_zero=g0.tolist(), lipschitz_bound=lipschitz)
            zero_kkt = radius == 0 or not mask.any() or np.all(g0[mask] >= 0)
            for iteration in range(max_iterations + 1):
                f, gradient, predicted = evaluate(a)
                linear_minimum = radius * min(0.0, float(gradient[mask].min())) if mask.any() else 0.0
                raw_gap = float(gradient @ a - linear_minimum)
                # A conservative FP64 rounding cushion; not interval arithmetic.
                scale = (float(np.abs(gradient) @ a) + abs(linear_minimum)
                         + f + f0)
                rounding = SPEC["roundoff_factor"] * np.finfo(np.float64).eps * scale
                gap = max(0.0, raw_gap) + rounding
                receipt.update(iterations=iteration, predicted_objective=f,
                    gradient=gradient.tolist(), raw_frank_wolfe_gap=raw_gap,
                    roundoff_cushion=rounding, global_objective_gap_upper_bound=gap)
                if raw_gap < -rounding:
                    fail("NEGATIVE_GAP_NUMERICAL_ERROR")
                if (zero_kkt or np.any(a > 0)) and gap <= tolerance:
                    if (np.any(a < 0) or np.any(a[~mask] != 0)
                            or float(a.sum()) > radius or f > f0 + rounding):
                        fail("FINAL_FEASIBILITY_OR_DESCENT_ERROR")
                    receipt.update(status="ZERO_OPTIMUM" if not np.any(a) else "CERTIFIED",
                        reason=None, coefficients=a.tolist(), predicted_q=predicted.tolist())
                    return receipt
                if iteration == max_iterations:
                    fail("MAX_ITERATIONS_WITHOUT_CERTIFICATE")
                updated = _project(a - gradient / lipschitz, radius)
                updated[~mask] = 0.0
                if np.array_equal(updated, a):
                    fail("STAGNATION_WITHOUT_CERTIFICATE")
                a = updated
    except SolverFailure:
        raise
    except (FloatingPointError, np.linalg.LinAlgError, OverflowError) as exc:
        fail(f"NUMERICAL_FAILURE:{type(exc).__name__}")
