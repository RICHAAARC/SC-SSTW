"""UniPC control with two disposable shadow steps and one real history update."""
from __future__ import annotations
import copy
import torch


def measures(x):
    return {'support_rms': float(x[:, :, 1:45].double().square().mean().sqrt()),
            'global_rms': float(x.double().square().mean().sqrt())}


def controlled_step(scheduler, sample, velocity, timestep, u, radius, totals, count):
    if velocity.dtype != torch.float32:
        raise ValueError('new flow path requires the completed CFG output in FP32')
    if scheduler.config.prediction_type != 'flow_prediction' or scheduler.config.thresholding or not scheduler.predict_x0:
        raise ValueError('control requires flow_prediction UniPC predict_x0 without thresholding')
    sigma = float(scheduler.sigmas[scheduler.step_index])
    if not sigma > 0:
        raise ValueError('active sigma must be positive')
    raw_delta = -u / sigma
    # Same sample, history, timestep, and existing CFG velocity. No model calls.
    count('shadow_step', False)
    base = copy.deepcopy(scheduler).step(velocity, timestep, sample, return_dict=False)[0]
    count('shadow_step', True)
    count('shadow_step', False)
    raw = copy.deepcopy(scheduler).step(velocity + raw_delta, timestep, sample, return_dict=False)[0]
    count('shadow_step', True)
    raw_d = raw - base
    ur, dr = measures(u)['support_rms'], measures(raw_d)['support_rms']
    limits = (min(radius/3., max(0., radius-totals['u'])),
              min(radius/3., max(0., radius-totals['D'])))
    scale = min(1., limits[0]/ur if ur else 1., limits[1]/dr if dr else 1.)
    # Reserve float32 rounding headroom; budgets remain upper bounds, never targets.
    if scale < 1.:
        scale *= 1. - 1e-5
    final_velocity = velocity + scale*raw_delta
    count('scheduler_step', False)
    updated = scheduler.step(final_velocity, timestep, sample, return_dict=False)[0]
    count('scheduler_step', True)
    actual_d = updated - base
    used_u, used_d = measures(scale*u), measures(actual_d)
    error = measures(actual_d-scale*raw_d)
    if not torch.isfinite(updated).all() or not torch.isfinite(final_velocity).all():
        raise FloatingPointError('nonfinite controlled UniPC step')
    # Numerical tolerance is recorded, not a new scientific threshold.
    tolerance = 2e-6 * max(1., measures(sample)['support_rms'])
    if error['support_rms'] > tolerance:
        raise RuntimeError('UniPC fixed-history affine scaling check failed')
    if used_u['support_rms'] > limits[0]+tolerance or used_d['support_rms'] > limits[1]+tolerance:
        raise RuntimeError('actual per-step/cumulative budget exceeded numerical tolerance')
    totals['u'] += used_u['support_rms']
    totals['D'] += used_d['support_rms']
    record = {'sigma': sigma, 'scale': scale, 'radius': radius,
              'request_u': measures(u), 'raw_delta_velocity': measures(raw_delta),
              'delta_velocity': measures(final_velocity-velocity), 'raw_D': measures(raw_d),
              'u': used_u, 'D': used_d, 'cumulative_sum_rms': dict(totals),
              'affine_error': error, 'numerical_tolerance': tolerance}
    arrays = {'requested_u': u.cpu(), 'raw_delta_velocity': raw_delta.cpu(),
              'applied_u': (scale*u).cpu(), 'applied_delta_velocity': (final_velocity-velocity).cpu(),
              'actual_D': actual_d.cpu()}
    return updated, record, arrays
