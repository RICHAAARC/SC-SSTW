"""Public fixed-key SC-SSTW generation and independent receive entrypoints.

The fixed experiment imports these functions.  This module never imports an
experiment runner, and the receive entrypoint rebuilds its public codebook from
only a key plus an explicit receiver protocol.
"""
from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

from main.tube_state import fixed_key
from runtime.wan import fixed_key_control, trajectory
from runtime.wan.integrated_core import encode_four_phases, eligible_detection
from runtime.wan.generation import load_frozen_vae, prepare_generation
from runtime.wan.io import encode_rgb, read_mp4
from runtime.wan.vae import _clear_cache, decode_normalized_latent, reencode_rgb24_readback

PROTOCOL_PATH = Path(__file__).with_name("fixed_key_protocol.json")
CONTROL_ARMS = {"OFF": (), "SINGLE46": (46,), "MULTI44_46": (44, 46)}
PAIRED_ARM_SPECS = {
    "OFF": ((), None),
    "LEGACY_SINGLE46": ((46,), fixed_key_control.DEFAULT_OBJECTIVE),
    "MSE_SINGLE46": ((46,), fixed_key_control.WINDOW_STATE_MSE_OBJECTIVE),
    "LEGACY_MULTI44_46": ((44, 46), fixed_key_control.DEFAULT_OBJECTIVE),
    "MSE_MULTI44_46": ((44, 46), fixed_key_control.WINDOW_STATE_MSE_OBJECTIVE),
}


def load_protocol(path: str | Path | None = None) -> dict:
    return json.loads((PROTOCOL_PATH if path is None else Path(path)).read_text(encoding="utf-8"))


def validate_protocol(protocol: dict) -> None:
    """Require the complete canonical method/model/media definition.

    Prompt and seed are deliberately absent from this document and are supplied
    to :func:`generation_config`.  Every persisted protocol field is fixed;
    sharing only the receiver ID is not calibration-compatible.
    """
    canonical = load_protocol()
    if not isinstance(protocol, dict) or protocol != canonical:
        raise ValueError("protocol must exactly match the canonical fixed-key protocol")


def generation_config(protocol: dict, prompt: str, seed: int) -> dict:
    validate_protocol(protocol)
    result = {"model": copy.deepcopy(protocol["model"]), "generation": copy.deepcopy(protocol["generation"])}
    result["generation"].update(prompt=str(prompt), seed=int(seed))
    return result


def _release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _count(count: Callable[[str, bool], None] | None, kind: str, completed: bool) -> None:
    if count is not None:
        count(kind, completed)


def _marker_branch(pipe, z44, snapshot44, prompt, negative, dtype, guidance, controls,
                    book, count, objective=fixed_key_control.DEFAULT_OBJECTIVE):
    z = z44.to(next(pipe.transformer.parameters()).device)
    scheduler = copy.deepcopy(snapshot44)
    control_rows, gradient_rows, probe_rows, arrays = [], [], [], {}
    per_step = fixed_key_control.R_STAR / len(controls) if controls else None
    for index in range(44, 50):
        velocity = trajectory.velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count)
        if index in controls:
            zero_next, _ = trajectory.zero_step(scheduler, z, velocity, index, count, "zero_shadow_step")
            sigma = float(scheduler.sigmas[index])
            clean = z - sigma * velocity
            # Preserve the legacy positional call for existing integrations that
            # monkeypatch the historical three-argument default objective.
            if objective == fixed_key_control.DEFAULT_OBJECTIVE:
                raw, gradient = fixed_key_control.clean_direction(clean, book, count)
            else:
                raw, gradient = fixed_key_control.clean_direction(clean, book, count, objective=objective)
            gradient["index"] = index
            unit, epsilon, probe = fixed_key_control.prepare_direction(
                scheduler, z, velocity, zero_next, raw, per_step, index, count,
            )
            z, scheduler, row, tensors = fixed_key_control.controlled_step(
                scheduler, z, velocity, zero_next, unit, epsilon, per_step, index, count,
            )
            control_rows.append(row); gradient_rows.append(gradient); probe_rows.append(probe)
            arrays[str(index)] = tensors
        else:
            z = trajectory.native_step(scheduler, z, velocity, index, count)
    return z.detach().cpu(), control_rows, gradient_rows, probe_rows, arrays


def generate_key_terminals(config: dict, key: bytes,
                               arms: Iterable[str] = ("MULTI44_46",),
                               count: Callable[[str, bool], None] | None = None,
                               *, objective: str = fixed_key_control.DEFAULT_OBJECTIVE) -> dict:
    """Run fresh noise through the shared 44/46 writer and return CPU terminals with nominal marker evidence."""
    count = count or (lambda kind, completed: None)
    arms = tuple(arms)
    if not arms or any(arm not in CONTROL_ARMS for arm in arms):
        raise ValueError("arms must be selected from OFF, SINGLE46, MULTI44_46")
    objective_record = fixed_key_control.objective_identity(objective)
    _count(count, "generation", False)
    pipe, initial, prompt, negative, dtype = prepare_generation(config, load_vae=False)
    _count(count, "generation", True)
    try:
        z44, snapshot44 = trajectory.prefix44(
            pipe, initial, prompt, negative, dtype, config["generation"]["guidance_scale"], count,
        )
        book = fixed_key.codebook(key)
        rows = {}
        for arm in arms:
            try:
                terminal, controls, gradients, probes, arrays = _marker_branch(
                    pipe, z44, snapshot44, prompt, negative, dtype,
                    config["generation"]["guidance_scale"], CONTROL_ARMS[arm], book, count, objective,
                )
                rows[arm] = {"status": "GENERATED", "terminal": terminal, "control_steps": controls,
                             "clean_gradients": gradients, "unit_probes": probes, "control_tensors": arrays,
                             "cumulative_native_response": fixed_key_control.cumulative(controls),
                             "terminal_fingerprint": trajectory.fingerprint(terminal),
                             "nominal_terminal": fixed_key.nominal_record(terminal, book, include_state=True)}
            except Exception as exc:
                rows[arm] = {"status": "FAILED_GENERATION", "error": repr(exc)}
        return {"arms": rows, "book": book, "writer_objective": objective_record,
                "initial_noise_fingerprint": trajectory.fingerprint(initial),
                "state44_fingerprint": trajectory.fingerprint(z44)}
    finally:
        pipe.transformer = None
        pipe = initial = prompt = negative = None
        _release()


def generate_paired_key_terminals(config: dict, key: bytes,
                                  count: Callable[[str, bool], None] | None = None) -> dict:
    """Generate the fixed OFF/legacy/MSE five-fork comparison from one state 44."""
    count = count or (lambda kind, completed: None)
    pipe = None
    _count(count, "generation", False)
    pipe, initial, prompt, negative, dtype = prepare_generation(config, load_vae=False)
    _count(count, "generation", True)
    try:
        z44, snapshot44 = trajectory.prefix44(
            pipe, initial, prompt, negative, dtype, config["generation"]["guidance_scale"], count,
        )
        snapshot_fingerprint = trajectory.fingerprint(vars(snapshot44))
        book = fixed_key.codebook(key)
        rows = {}
        for arm, (controls, objective) in PAIRED_ARM_SPECS.items():
            try:
                branch_objective = objective or fixed_key_control.DEFAULT_OBJECTIVE
                terminal, control_rows, gradients, probes, arrays = _marker_branch(
                    pipe, z44.clone(), snapshot44, prompt, negative, dtype,
                    config["generation"]["guidance_scale"], controls, book, count, branch_objective,
                )
                if trajectory.fingerprint(vars(snapshot44)) != snapshot_fingerprint:
                    raise RuntimeError("paired branch polluted source scheduler snapshot")
                rows[arm] = {
                    "status": "GENERATED",
                    "terminal": terminal,
                    "writer_objective": None if objective is None else fixed_key_control.objective_identity(objective),
                    "control_steps": control_rows,
                    "clean_gradients": gradients,
                    "unit_probes": probes,
                    "control_tensors": arrays,
                    "cumulative_native_response": fixed_key_control.cumulative(control_rows),
                    "terminal_fingerprint": trajectory.fingerprint(terminal),
                    "nominal_terminal": fixed_key.nominal_record(terminal, book, include_state=True),
                }
            except Exception as exc:
                rows[arm] = {"status": "FAILED_GENERATION", "error": repr(exc)}
        return {
            "arms": rows,
            "book": book,
            "initial_noise_fingerprint": trajectory.fingerprint(initial),
            "state44_fingerprint": trajectory.fingerprint(z44),
            "scheduler44_fingerprint": snapshot_fingerprint,
        }
    finally:
        if pipe is not None:
            pipe.transformer = None
        pipe = initial = prompt = negative = None
        _release()


def generate_video(prompt: str, seed: int, key: bytes,
                   output_path: str | Path, protocol: dict | None = None,
                   arm: str = "MULTI44_46", count: Callable[[str, bool], None] | None = None,
                   *, objective: str = fixed_key_control.DEFAULT_OBJECTIVE) -> dict:
    """Public writer: prompt/seed/key to one persisted MP4."""
    protocol = copy.deepcopy(protocol or load_protocol())
    config = generation_config(protocol, prompt, seed)
    generated = generate_key_terminals(config, key, (arm,), count, objective=objective)
    if generated["arms"][arm]["status"] != "GENERATED":
        raise RuntimeError(generated["arms"][arm]["error"])
    terminal = generated["arms"][arm].pop("terminal")
    vae = None
    try:
        vae = load_frozen_vae(config)
        _count(count, "vae_decode", False)
        rgb = decode_normalized_latent(vae, terminal.to(next(vae.parameters()).device)).detach().cpu()
        _count(count, "vae_decode", True)
        _count(count, "mp4_save", False)
        encode_rgb(rgb, Path(output_path), config["generation"]["fps"], protocol["video"]["crf"])
        _count(count, "mp4_save", True)
    finally:
        terminal = vae = None
        _release()
    writer = {key: value for key, value in generated["arms"][arm].items() if key != "control_tensors"}
    writer["objective"] = generated["writer_objective"]
    return {"status": "VIDEO_PERSISTED", "path": str(Path(output_path)),
            "arm": arm, "receiver_protocol_id": protocol["receiver_protocol_id"],
            "key_id": fixed_key.key_identifier(key), "writer": writer}


def receive_pixels(pixels, key: bytes, protocol: dict, calibration: dict | None, *, vae=None,
                   count: Callable[[str, bool], None] | None = None,
                   on_phase: Callable[[int, Any, dict], None] | None = None) -> dict:
    """Independent receiver core; writer terminal, truth, and trajectory are absent."""
    validate_protocol(protocol)
    own_vae = vae is None
    if own_vae:
        vae = load_frozen_vae(protocol)
    try:
        observations, phases = encode_four_phases(pixels, vae, count, on_phase)
        detection = fixed_key.read(observations, fixed_key.codebook(key))
        complete = all(phases.get(str(phase), {}).get("status") == "COMPLETE" for phase in range(4))
        status = "SCORED" if complete and detection.get("status") == "SCORED" else "PARTIAL_OR_FAILED"
        formal = eligible_detection(status, phases, detection)
        decision = fixed_key.decide(formal, calibration)
        return {"status": status, "observations": phases, "detection": detection,
                "formal_detection": formal, "decision": decision}
    finally:
        if own_vae:
            vae = None; _release()


def receive_mp4(mp4_path: str | Path, key: bytes, protocol: dict, calibration: dict | None,
                *, vae=None, count: Callable[[str, bool], None] | None = None,
                on_phase: Callable[[int, Any, dict], None] | None = None) -> dict:
    """Public receiver: one MP4 plus key, protocol, and calibration only."""
    _count(count, "mp4_read", False)
    pixels = read_mp4(Path(mp4_path))
    _count(count, "mp4_read", True)
    try:
        return receive_pixels(pixels, key, protocol, calibration, vae=vae, count=count, on_phase=on_phase)
    finally:
        pixels = None; _release()
