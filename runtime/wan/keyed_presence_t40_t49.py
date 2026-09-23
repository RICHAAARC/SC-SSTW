"""Fresh-noise fixed-marker tanh controls on independent live T40..T49 histories."""
from __future__ import annotations

import copy
import gc
import math

import torch

from main.tube_state import fixed_key
from . import fixed_key_control, payload_control, trajectory
from .generation import prepare_generation

ARMS = {
    "OFF": (),
    "T44_46": (44, 46),
    "T40_48": tuple(range(40, 49)),
    "T40_49": tuple(range(40, 50)),
    "T49_ONLY": (49,),
}
R_STAR = fixed_key_control.R_STAR


@torch.no_grad()
def prefix40(pipe, initial, prompt, negative, dtype, guidance, count):
    scheduler = pipe.scheduler
    trajectory.validate_scheduler(scheduler)
    z = initial.detach()
    for index in range(40):
        v = trajectory.velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count)
        z = trajectory.native_step(scheduler, z, v, index, count)
    if scheduler.step_index != 40:
        raise RuntimeError("T40 prefix scheduler cursor mismatch")
    return z.detach(), copy.deepcopy(scheduler)


@torch.no_grad()
def branch(pipe, z40, snapshot40, prompt, negative, dtype, guidance, controls, book, count):
    """One fork; every shadow/probe/formal step uses this fork's full history."""
    z = z40.clone().to(next(pipe.transformer.parameters()).device)
    scheduler = copy.deepcopy(snapshot40)
    rows, gradients, probes, tensors = [], [], [], {}
    target = R_STAR / len(controls) if controls else None
    for index in range(40, 50):
        if scheduler.step_index != index:
            raise RuntimeError("branch native scheduler cursor mismatch")
        v = trajectory.velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count)
        if index not in controls:
            z = trajectory.native_step(scheduler, z, v, index, count)
            continue
        history_before = trajectory.fingerprint(vars(scheduler))
        zero, _ = trajectory.zero_step(scheduler, z, v, index, count, "zero_shadow_step")
        sigma = float(scheduler.sigmas[index])
        if sigma <= 0:
            raise ValueError("controlled step requires positive native sigma")
        raw, gradient = fixed_key_control.clean_direction(z - sigma * v, book, count)
        gradient["index"] = index
        q, epsilon, probe = payload_control.prepare_direction(
            scheduler, z, v, zero, raw, target, index, count,
            allowed_steps=range(40, 50),
        )
        z, scheduler, row, arrays = payload_control.controlled_step(
            scheduler, z, v, zero, q, epsilon, target, index, count,
        )
        if row["history_fingerprint"] != history_before or probe["history_fingerprint"] != history_before:
            raise RuntimeError("same-history control mismatch")
        rows.append(row)
        gradients.append(gradient)
        probes.append(probe)
        tensors[str(index)] = arrays
    if not bool(torch.isfinite(z).all()):
        raise FloatingPointError("nonfinite terminal state")
    return z.detach().cpu(), rows, gradients, probes, tensors


def generate_terminals(config, key: bytes, arms, count):
    """Generate all requested forks from one fresh T0 noise, retaining per-arm failures."""
    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(arm not in ARMS for arm in arms):
        raise ValueError("invalid fixed arm roster")
    count("generation", False)
    pipe, initial, prompt, negative, dtype = prepare_generation(config, load_vae=False)
    count("generation", True)
    try:
        z40, snapshot40 = prefix40(
            pipe, initial, prompt, negative, dtype, config["generation"]["guidance_scale"], count,
        )
        state40 = trajectory.fingerprint(z40)
        history40 = trajectory.fingerprint(vars(snapshot40))
        book = fixed_key.codebook(key)
        out = {}
        for arm in arms:
            try:
                terminal, controls, gradients, probes, tensors = branch(
                    pipe, z40, snapshot40, prompt, negative, dtype,
                    config["generation"]["guidance_scale"], ARMS[arm], book, count,
                )
                if trajectory.fingerprint(z40) != state40 or trajectory.fingerprint(vars(snapshot40)) != history40:
                    raise RuntimeError("branch polluted T40 source state/history")
                out[arm] = dict(status="GENERATED", terminal=terminal,
                    control_steps=controls, clean_gradients=gradients, unit_probes=probes,
                    control_tensors=tensors,
                    cumulative_native_response=payload_control.cumulative(controls),
                    terminal_fingerprint=trajectory.fingerprint(terminal),
                    nominal_terminal=fixed_key.nominal_record(terminal, book, include_state=True))
            except Exception as exc:
                out[arm] = dict(status="FAILED_GENERATION", error=repr(exc))
        off = out.get("OFF")
        for arm, row in out.items():
            if row["status"] != "GENERATED" or off is None or off["status"] != "GENERATED":
                row["terminal_minus_off"] = {"status": "UNAVAILABLE", "measures": None,
                    "reason": "marked or same-source OFF terminal unavailable"}
                continue
            change = trajectory.measures(row["terminal"] - off["terminal"])
            row["terminal_minus_off"] = {"status": "REPORTED" if all(math.isfinite(x) for x in change.values()) else "FAILED_NONFINITE",
                "measures": change if all(math.isfinite(x) for x in change.values()) else None,
                "meaning": "reporting-only net terminal displacement from same-source OFF; not native cumulative budget or blind detection"}
        return dict(arms=out, initial_noise_fingerprint=trajectory.fingerprint(initial),
            state40_fingerprint=state40, history40_fingerprint=history40)
    finally:
        pipe.transformer = None
        pipe = initial = prompt = negative = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
