"""Fixed-budget temperature-one payload control on live 44/46 histories."""
from __future__ import annotations

import math

import torch

from main.tube_state import payload_codec
from . import trajectory

CONTROL_STEPS = (44, 46)
R_STAR = 0.042943312697648145


def clean_direction(clean, book: dict, payload: int, count=None):
    """Autograd only on a detached CPU clean leaf; no model graph is retained."""
    with torch.enable_grad():
        leaf = clean.detach().cpu().double().clone().requires_grad_(True)
        value, parts = payload_codec.payload_sync_loss(leaf, book, payload)
        if count is not None:
            count("clean_leaf_backward", False)
        gradient, = torch.autograd.grad(value, leaf)
        if count is not None:
            count("clean_leaf_backward", True)
    gradient = gradient.detach()
    norm = trajectory.measures(gradient)["support_rms"]
    if not math.isfinite(norm) or norm <= 0 or not torch.isfinite(gradient).all():
        raise ValueError("zero/nonfinite clean payload gradient; no fallback")
    raw = -gradient.float()
    return raw, {
        "objective": "mean-rival-tanh-plus-explicit-pilot",
        "temperature": 1,
        "pilot_loss_weight": payload_codec.PILOT_LOSS_WEIGHT,
        "loss": float(value.detach()),
        "data_loss": float(parts["data_loss"].detach()),
        "pilot_loss": float(parts["pilot_loss"].detach()),
        "gradient_support_rms": norm,
        "meaning": "CPU detached current clean leaf; no terminal transport, model/VAE graph, or receiver derivative",
    }


@torch.no_grad()
def prepare_direction(snapshot, z, v, zero_next, raw, target, index, count, *, allowed_steps=CONTROL_STEPS):
    before = trajectory.fingerprint(vars(snapshot))
    if index not in allowed_steps or snapshot.step_index != index:
        raise ValueError("invalid control cursor")
    sigma = float(snapshot.sigmas[index])
    if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(target) or target <= 0:
        raise ValueError("positive finite sigma and response target required")
    raw = raw.detach().to(device=z.device, dtype=torch.float32)
    norm = trajectory.measures(raw)["support_rms"]
    if not math.isfinite(norm) or norm <= 0 or not torch.isfinite(raw).all():
        raise ValueError("zero/nonfinite direction; no fallback")
    unit = raw / norm
    probe, _ = trajectory.zero_step(snapshot, z, v - unit / sigma, index, count, "unit_response_probe_step")
    unit_response = trajectory.measures(probe - zero_next)
    response = unit_response["support_rms"]
    if not math.isfinite(response) or response <= 0:
        raise ValueError("zero/nonfinite same-history unit response")
    epsilon = target / response
    if trajectory.fingerprint(vars(snapshot)) != before:
        raise RuntimeError("unit probe polluted source scheduler history")
    return unit, epsilon, {
        "index": index,
        "sigma": sigma,
        "target_D_support_rms": target,
        "raw_direction": trajectory.measures(raw),
        "unit_direction": trajectory.measures(unit),
        "unit_D": unit_response,
        "epsilon": epsilon,
        "input_fingerprint": trajectory.fingerprint(z),
        "history_fingerprint": before,
    }


@torch.no_grad()
def controlled_step(snapshot, z, v, zero_next, unit, epsilon, target, index, count):
    before = trajectory.fingerprint(vars(snapshot))
    sigma = float(snapshot.sigmas[index])
    applied_u = epsilon * unit
    controlled_velocity = v - applied_u / sigma
    if not torch.isfinite(controlled_velocity).all():
        raise FloatingPointError("nonfinite controlled velocity")
    after, history = trajectory.zero_step(snapshot, z, controlled_velocity, index, count)
    actual = trajectory.measures(after - zero_next)
    relative_error = abs(actual["support_rms"] - target) / target
    if not all(math.isfinite(value) for value in actual.values()) or not math.isfinite(relative_error):
        raise FloatingPointError("nonfinite actual native response")
    if relative_error > 2e-5:
        raise RuntimeError("fixed native-response budget mismatch")
    if trajectory.fingerprint(vars(snapshot)) != before:
        raise RuntimeError("formal control polluted source scheduler history")
    return after, history, {
        "index": index,
        "sigma": sigma,
        "epsilon": epsilon,
        "target_D_support_rms": target,
        "actual_D": actual,
        "matching_relative_error": relative_error,
        "u": trajectory.measures(applied_u),
        "delta_velocity": trajectory.measures(controlled_velocity - v),
        "input_fingerprint": trajectory.fingerprint(z),
        "history_fingerprint": before,
        "zero_next_fingerprint": trajectory.fingerprint(zero_next),
        "controlled_next_fingerprint": trajectory.fingerprint(after),
    }, {"applied_u": applied_u.cpu(), "actual_D": (after - zero_next).cpu()}


@torch.no_grad()
def finish(pipe, z, scheduler, prompt, negative, dtype, guidance, start, count):
    for index in range(start, 50):
        v = trajectory.velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count)
        z = trajectory.native_step(scheduler, z, v, index, count)
    return z.detach()


def cumulative(rows: list[dict]) -> dict:
    values = [row["actual_D"] for row in rows]
    return {
        "controlled_steps": len(rows),
        "actual_D": {
            space: {
                "sum_rms": sum(value[space + "_rms"] for value in values),
                "sum_rms_squared": sum(value[space + "_rms"] ** 2 for value in values),
                "peak_rms": max((value[space + "_rms"] for value in values), default=0.0),
            }
            for space in ("support", "global")
        },
        "meaning": "sum of per-step same-history native responses; not terminal displacement or equal energy",
    }
