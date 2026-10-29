"""Pure direct constraints for the unchanged saved QIM candidate; no optimization/IO."""
import torch
from . import b_framewise_qim_v1 as p
from .b_framewise_qim_v1_solver import budget_metrics


def validate_targets(q0, targets, bits, active):
    if (q0.shape != targets.shape or targets.shape != bits.shape or active.shape != targets.shape[:2]
            or q0.dtype != torch.float64 or targets.dtype != torch.float64
            or not bool(torch.isfinite(q0).all() and torch.isfinite(targets).all())):
        raise ValueError('finite FP64 fixed target geometry required')
    if not torch.equal(targets, p.nearest(q0, bits)):
        raise ValueError('saved targets differ from original class/nearest/tie semantics; not rewritten')


def direct_constraints(q0, projected_prediction, targets, prediction, mask, active):
    if not bool(torch.isfinite(prediction).all() and torch.isfinite(projected_prediction).all()):
        raise FloatingPointError('nonfinite actual Aa')
    target = float(torch.where(active[:, :, None], q0+255*projected_prediction-targets, 0).abs().max())
    budget = budget_metrics(prediction, mask)
    accepted = target <= 1e-3 and budget['excess'] <= 1e-7
    return dict(status='DIRECT_CONSTRAINTS_WITHIN_TOLERANCE' if accepted else 'DIRECT_CONSTRAINTS_FAILED',
                target_max=target, budget=budget, optimality='UNPROVEN', old_solver_status_unchanged=True)


def native_chunk_metrics(value, base, prediction, targets, active, mask):
    if value.shape != base.shape or value.dtype != torch.float32 or not bool(torch.isfinite(value).all() and torch.isfinite(base).all()):
        raise ValueError('complete finite FP32 native chunk and saved BASE required')
    delta = value.double()-base.double(); packed = p.pack(delta)
    budget = budget_metrics(packed, mask)
    full_mask = torch.zeros_like(delta, dtype=torch.bool)
    for tile, (y, x) in enumerate(p.ROIS):
        height = min(58, 321-y)
        full_mask[:, y-1:y-1+height, x-1:x+73] = mask[:, tile, :, :height].permute(0, 2, 3, 1)
    outside = torch.where(full_mask, 0, delta).abs().flatten(1).amax(1)
    budget['outside_full_canvas'] = float(outside.max())
    budget['excess'] = max(budget['excess'], budget['outside_full_canvas'])
    errors = torch.where(active[:, :, None], 255*p.project(p.pack(value.double()))-targets, 0).abs().flatten(1).amax(1)
    prediction_errors = (packed-prediction).abs().flatten(1).amax(1)
    frames = [dict(target_max=float(errors[i]), delta_linf=float(delta[i].abs().max()),
                   prediction_error_max=float(prediction_errors[i]), outside_full_canvas=float(outside[i])) for i in range(len(value))]
    return dict(budget=budget, target_max=float(errors.max()),
                prediction_error_max=float(prediction_errors.max()), frames=frames)
