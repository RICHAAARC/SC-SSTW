"""DCT2 identity at the existing 50-step conditional-clean/native seam.

This component does not load a model, write terminal state or execute media.
The common runner owns preparation, state construction, IO and failure slots.
"""
from __future__ import annotations

import math
import torch

from main.tube_state import b_local_identity_dct_v1 as payload
from runtime.wan import b_segmented_payload_v1 as native_payload
from runtime.wan.conditional_joint import trajectory


def guided_velocity(z, conditional, unconditional, sigma, target, index, count=None):
    if type(index) is not int or not 0 <= index < 50:
        raise ValueError("fixed native step 0..49 required")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0:
        raise ValueError("finite positive native sigma required")
    z, c, u = z.float(), conditional.float(), unconditional.float()
    for value in (z, c, u):
        payload._validate_tensor(value, payload.LATENT_SHAPE)
    enabled = index >= 25
    detail = dict(enabled=enabled, arm="DCT2_MULTI", state_control_cfg="float32")
    if enabled:
        if count:
            count("payload_gradient", False)
        delta, receipt = payload.local_delta(z - float(sigma) * c, target)
        if count:
            count("payload_gradient", True)
        c = c - delta / float(sigma)
        detail.update(payload=receipt, pilot=None,
                      payload_delta_l2=float(delta.double().square().sum().sqrt()),
                      cfg_clean_delta_rms=5 * receipt["conditional_clean_delta_rms"])
    velocity = u + 5.0 * (c - u)
    if not bool(torch.isfinite(velocity).all()):
        raise FloatingPointError("nonfinite DCT2 CFG velocity")
    return velocity.detach(), detail


def run_trajectory(pipe, initial, scheduler, prompt, negative, dtype, key, bits, count, record):
    """Same call signature as the old trajectory; identity arithmetic is DCT2."""
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:
        raise ValueError("fresh native scheduler history required")
    z = initial.detach().float().clone()
    target = payload.build_target(z, key, bits)
    for index in range(50):
        # Shared helper only executes conditional/unconditional DiT forwards.
        c, u = native_payload.predict_branches(pipe, z, scheduler, prompt, negative, dtype, index, count)
        sigma = float(scheduler.sigmas[index])
        velocity, detail = guided_velocity(z, c, u, sigma, target, index, count)
        z = trajectory.native_step(scheduler, z, velocity, index, count, kind="native_step")
        if record:
            record(dict(index=index, sigma=sigma, cursor_after=scheduler.step_index, **detail))
    if scheduler.step_index != 50:
        raise RuntimeError("native DCT2 trajectory incomplete")
    return z.detach().cpu(), dict(arm="B_LOCAL_IDENTITY_DCT2", steps=50,
                                  active_indices=list(range(25, 50)),
                                  segment_source_times=[[1, 11], [12, 22], [23, 33], [34, 44]],
                                  net_bits=32, target_count=payload.TARGET_COUNT,
                                  beta=payload.BETA, eta=payload.ETA, guard_times=[0, 45],
                                  writer_fft_calls=0, terminal_payload_rewrite=False,
                                  state_writer_called=False, denoiser_backward=False, tail_backward=False)
