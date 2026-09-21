"""Public four-bit SC-SSTW generation and independent receive entrypoints.

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

from main.tube_state import payload_codec
from runtime.wan import payload_control, trajectory
from runtime.wan.generation import load_frozen_vae, prepare_generation
from runtime.wan.io import encode_rgb, read_mp4
from runtime.wan.vae import _clear_cache, decode_normalized_latent, reencode_rgb24_readback

PROTOCOL_PATH = Path(__file__).with_name("integrated_payload_protocol.json")
CONTROL_ARMS = {"OFF": (), "SINGLE46": (46,), "MULTI44_46": (44, 46)}


def load_protocol(path: str | Path | None = None) -> dict:
    return json.loads((PROTOCOL_PATH if path is None else Path(path)).read_text(encoding="utf-8"))


def validate_protocol(protocol: dict) -> None:
    if protocol.get("receiver_protocol_id") != payload_codec.RECEIVER_PROTOCOL_ID:
        raise ValueError("unsupported receiver protocol")
    generation = protocol.get("generation", {})
    if (generation.get("frames"), generation.get("steps")) != (181, 50):
        raise ValueError("the public writer requires the fixed 181-frame/50-step Wan protocol")
    if protocol.get("control", {}).get("R_star") != payload_control.R_STAR:
        raise ValueError("fixed control budget mismatch")
    if protocol.get("payload", {}).get("pilot_loss_weight") != payload_codec.PILOT_LOSS_WEIGHT:
        raise ValueError("fixed pilot coefficient mismatch")


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


def _payload_branch(pipe, z44, snapshot44, prompt, negative, dtype, guidance, controls,
                    book, payload, count):
    z = z44.to(next(pipe.transformer.parameters()).device)
    scheduler = copy.deepcopy(snapshot44)
    control_rows, gradient_rows, probe_rows, arrays = [], [], [], {}
    per_step = payload_control.R_STAR / len(controls) if controls else None
    for index in range(44, 50):
        velocity = trajectory.velocity(pipe, z, scheduler, prompt, negative, dtype, guidance, index, count)
        if index in controls:
            zero_next, _ = trajectory.zero_step(scheduler, z, velocity, index, count, "zero_shadow_step")
            sigma = float(scheduler.sigmas[index])
            raw, gradient = payload_control.clean_direction(z - sigma * velocity, book, payload, count)
            gradient["index"] = index
            unit, epsilon, probe = payload_control.prepare_direction(
                scheduler, z, velocity, zero_next, raw, per_step, index, count,
            )
            z, scheduler, row, tensors = payload_control.controlled_step(
                scheduler, z, velocity, zero_next, unit, epsilon, per_step, index, count,
            )
            control_rows.append(row); gradient_rows.append(gradient); probe_rows.append(probe)
            arrays[str(index)] = tensors
        else:
            z = trajectory.native_step(scheduler, z, velocity, index, count)
    return z.detach().cpu(), control_rows, gradient_rows, probe_rows, arrays


def generate_payload_terminals(config: dict, payload: int | str, key: bytes,
                               arms: Iterable[str] = ("MULTI44_46",),
                               count: Callable[[str, bool], None] | None = None) -> dict:
    """Run fresh noise through the shared 44/46 writer and return CPU terminals."""
    payload = payload_codec.parse_payload(payload)
    count = count or (lambda kind, completed: None)
    arms = tuple(arms)
    if not arms or any(arm not in CONTROL_ARMS for arm in arms):
        raise ValueError("arms must be selected from OFF, SINGLE46, MULTI44_46")
    _count(count, "generation", False)
    pipe, initial, prompt, negative, dtype = prepare_generation(config, load_vae=False)
    _count(count, "generation", True)
    try:
        z44, snapshot44 = trajectory.prefix44(
            pipe, initial, prompt, negative, dtype, config["generation"]["guidance_scale"], count,
        )
        book = payload_codec.codebook(key)
        rows = {}
        for arm in arms:
            try:
                terminal, controls, gradients, probes, arrays = _payload_branch(
                    pipe, z44, snapshot44, prompt, negative, dtype,
                    config["generation"]["guidance_scale"], CONTROL_ARMS[arm], book, payload, count,
                )
                rows[arm] = {"status": "GENERATED", "terminal": terminal, "control_steps": controls,
                             "clean_gradients": gradients, "unit_probes": probes, "control_tensors": arrays,
                             "cumulative_native_response": payload_control.cumulative(controls),
                             "terminal_fingerprint": trajectory.fingerprint(terminal)}
            except Exception as exc:
                rows[arm] = {"status": "FAILED_GENERATION", "error": repr(exc)}
        return {"payload": payload, "arms": rows, "book": book,
                "initial_noise_fingerprint": trajectory.fingerprint(initial),
                "state44_fingerprint": trajectory.fingerprint(z44)}
    finally:
        pipe.transformer = None
        pipe = initial = prompt = negative = None
        _release()


def generate_video(prompt: str, seed: int, payload: int | str, key: bytes,
                   output_path: str | Path, protocol: dict | None = None,
                   arm: str = "MULTI44_46", count: Callable[[str, bool], None] | None = None) -> dict:
    """Public writer: prompt/seed/nibble/key to one persisted MP4."""
    protocol = copy.deepcopy(protocol or load_protocol())
    config = generation_config(protocol, prompt, seed)
    generated = generate_payload_terminals(config, payload, key, (arm,), count)
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
    return {"status": "VIDEO_PERSISTED", "path": str(Path(output_path)), "payload": generated["payload"],
            "arm": arm, "receiver_protocol_id": protocol["receiver_protocol_id"],
            "key_id": payload_codec.key_identifier(key), "writer": writer}


def eligible_detection(view_status: str, phase_rows: dict, raw: dict) -> dict:
    """One formal adapter shared by standalone and fixed-run receivers."""
    incomplete = [str(phase) for phase in range(4) if phase_rows.get(str(phase), {}).get("status") != "COMPLETE"]
    if view_status != "SCORED" or incomplete or raw.get("status") != "SCORED":
        return {"status": "INVALID", "reason": "VIEW_NOT_PROTOCOL_ELIGIBLE", "view_status": view_status,
                "incomplete_phases": incomplete, "raw_detection_status": raw.get("status")}
    return raw


def encode_four_phases(pixels, vae, count: Callable[[str, bool], None] | None = None,
                       on_phase: Callable[[int, Any, dict], None] | None = None) -> tuple[dict, dict]:
    observations, phase_rows = {}, {}
    for phase in range(4):
        encoded = None
        try:
            shifted = pixels[phase:]
            groups = (len(shifted) - 1) // 4
            used = 1 + 4 * groups
            _count(count, "vae_encode", False)
            encoded = reencode_rgb24_readback(vae, shifted[:used]).detach().cpu()
            _count(count, "vae_encode", True)
            observations[phase] = encoded.numpy()
            row = {"status": "COMPLETE", "frames_used": used, "tail_discarded": len(shifted) - used}
            phase_rows[str(phase)] = row
            if on_phase is not None:
                on_phase(phase, encoded, row)
        except Exception as exc:
            phase_rows[str(phase)] = {"status": "FAILED", "error": repr(exc)}
        finally:
            encoded = None
            _clear_cache(vae); _release()
    return observations, phase_rows


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
        detection = payload_codec.read(observations, payload_codec.codebook(key))
        complete = all(phases.get(str(phase), {}).get("status") == "COMPLETE" for phase in range(4))
        status = "SCORED" if complete and detection.get("status") == "SCORED" else "PARTIAL_OR_FAILED"
        formal = eligible_detection(status, phases, detection)
        decision = payload_codec.decide(formal, calibration)
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
