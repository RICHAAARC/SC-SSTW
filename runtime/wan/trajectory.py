"""Shared native Wan trajectory primitives for the adopted 44/46 controller.

This module contains only the successful live-history mechanics: fresh native
prefix generation, CFG velocity, scheduler-history copies, response measures,
and fingerprints.  It has no experiment, receiver, or historical runner import.
"""
from __future__ import annotations

import copy
import hashlib
import json

import torch


def measures(value) -> dict:
    tensor = value.detach().double()
    return {
        "support_rms": float(tensor[:, :, 1:45].square().mean().sqrt()),
        "global_rms": float(tensor.square().mean().sqrt()),
        "peak_abs": float(tensor.abs().max()),
    }


def _fingerprint_value(value):
    if torch.is_tensor(value):
        # NumPy cannot expose bfloat16 values. Reinterpret contiguous tensor
        # storage as bytes; reshape handles scalar tensors as well. For NumPy
        # supported dtypes these are the same bytes as .numpy().tobytes().
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {str(key): _fingerprint_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_fingerprint_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def fingerprint(value) -> str:
    encoded = json.dumps(_fingerprint_value(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@torch.no_grad()
def velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count):
    timestep = scheduler.timesteps[index].expand(z.shape[0])
    values = []
    for embedding in (prompt, negative):
        count("transformer", False)
        values.append(pipe.transformer(
            hidden_states=z.to(dtype), timestep=timestep, encoder_hidden_states=embedding,
            attention_kwargs=None, return_dict=False,
        )[0])
        count("transformer", True)
    result = (values[1] + guidance * (values[0] - values[1])).float()
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite CFG velocity")
    return result


@torch.no_grad()
def native_step(scheduler, z, v, index, count, kind="scheduler_step"):
    cursor = scheduler.step_index
    if not (cursor == index or (index == 0 and cursor is None)):
        raise ValueError("native scheduler cursor mismatch")
    count(kind, False)
    result = scheduler.step(v, scheduler.timesteps[index], z.clone(), return_dict=False)[0]
    count(kind, True)
    if scheduler.step_index != index + 1 or not torch.isfinite(result).all():
        raise RuntimeError("nonfinite native state or scheduler cursor divergence")
    return result.detach()


def validate_scheduler(scheduler) -> None:
    config = scheduler.config
    if config.prediction_type != "flow_prediction" or config.thresholding or not scheduler.predict_x0:
        raise ValueError("native nonthresholded flow-prediction scheduler required")
    if len(scheduler.timesteps) != 50 or not config.lower_order_final or float(scheduler.sigmas[-1]) != 0.0:
        raise ValueError("fixed native 50-step zero-sigma schedule required")


@torch.no_grad()
def reference(pipe, initial, prompt, negative, dtype, guidance, count):
    """Create the same-run OFF path plus independent complete histories at 44/46."""
    scheduler = pipe.scheduler
    validate_scheduler(scheduler)
    z = initial.detach()
    nodes = {}
    snapshots = {}
    for index in range(50):
        v = velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count)
        if index in range(44, 50):
            nodes[index] = {"z": z.cpu().clone(), "v": v.cpu().clone(),
                            "clean": (z - float(scheduler.sigmas[index]) * v).cpu().clone()}
        if index in (44, 46):
            snapshots[index] = copy.deepcopy(scheduler)
        z = native_step(scheduler, z, v, index, count)
    nodes[50] = {"z": z.cpu().clone()}
    return nodes, snapshots


@torch.no_grad()
def prefix44(pipe, initial, prompt, negative, dtype, guidance, count):
    """Run the shared fresh-noise prefix through step 43 and fork at state 44."""
    scheduler = pipe.scheduler
    validate_scheduler(scheduler)
    z = initial.detach()
    for index in range(44):
        v = velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count)
        z = native_step(scheduler, z, v, index, count)
    return z.detach(), copy.deepcopy(scheduler)


@torch.no_grad()
def zero_step(snapshot, z, v, index, count, kind="scheduler_step"):
    before = fingerprint(vars(snapshot))
    copied = copy.deepcopy(snapshot)
    result = native_step(copied, z, v, index, count, kind)
    if fingerprint(vars(snapshot)) != before:
        raise RuntimeError("zero step polluted source scheduler history")
    return result, copied
