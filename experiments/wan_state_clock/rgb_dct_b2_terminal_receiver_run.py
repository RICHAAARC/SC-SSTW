"""Fixed two-source OFF/SINGLE49/TERMINAL49_RECEIVER RGB-DCT experiment."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import os
import re
import resource
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from main.tube_state import rgb_dct_group_consistency as receiver
from main.tube_state.rgb_dct_t49_carrier import (
    AMPLITUDE, RGB_SHAPE, apply_carrier, lift_direction,
)
from runtime.wan.rgb_dct_multistep_adapter import score_mp4_once


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "configs/rgb_dct_b2_terminal_receiver_v1.json"
PROTOCOL_PATH = ROOT / "docs/rgb_dct_b2_terminal_receiver_protocol.md"
MODULE = "experiments.wan_state_clock.rgb_dct_b2_terminal_receiver_run"
OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/RGB-DCT-B2-Terminal-Receiver-V1")
CASE_IDS = ("eval_copperkettle_s2501", "eval_paperplane_s2502")
ARMS = ("OFF", "SINGLE49", "TERMINAL49_RECEIVER")
PROMPTS = (
    "locked camera, a copper kettle resting on a plain wooden table, steady soft indoor light, no people, no cuts",
    "locked camera, a white paper airplane gliding slowly across a plain blue background, steady soft light, no people, no cuts",
)
SEEDS = (2026092501, 2026092502)
R_STAR = 0.042943312697648145
PLAN = dict(
    generation=2, transformer_prefix=200, scheduler_prefix=100,
    vae_gradient_decode=2, vae_decode=4, vae_encode=4, vae_vjp=2,
    unit_response_probe_step=4, scheduler_step=4,
    mp4_save=6, mp4_read=6, score=6,
)
DERIVED = dict(transformer=200, scheduler_and_probe=108,
               vae_decode=6, backward_vjp=2)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if _sha(PROTOCOL_PATH) != config["protocol_sha256"]:
        raise ValueError("B2 terminal-receiver protocol byte identity mismatch")
    if config["frozen_task_card_sha256"] != "93ecd1c876b701cd2e6f0b1d8430661be9764ebee1a404b93b0e73d73b74ee69":
        raise ValueError("frozen B2 task card identity changed")
    if (config["receiver_spec_sha256"] != receiver.SPEC_SHA256
            or config["baseline_receiver_spec_sha256"] != receiver.baseline.SPEC_SHA256):
        raise ValueError("frozen RGB-DCT receiver changed")
    if receiver.score_rgb(None, config["key_utf8"].encode())["key_id"] != config["receiver_key_id"]:
        raise ValueError("fixed receiver key changed")
    expected_cases = [dict(id=case_id, seed=seed, prompt=prompt)
                      for case_id, seed, prompt in zip(CASE_IDS, SEEDS, PROMPTS, strict=True)]
    if config["cases"] != expected_cases:
        raise ValueError("fixed two-source roster changed")
    if config["model"] != dict(
        id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        revision="0fad780a534b6463e45facd96134c9f345acfa5b",
        transformer_dtype="bfloat16", vae_dtype="float32",
    ):
        raise ValueError("fixed Wan model changed")
    if config["generation"] != dict(
        height=320, width=512, frames=181, fps=8, steps=50,
        guidance_scale=5.0, max_sequence_length=512,
        negative_prompt="text, watermark, logo, camera motion, cuts, multiple objects, flicker",
    ):
        raise ValueError("fixed generation changed")
    if config["media"] != dict(
        fps=8, crf=18, codec="libx264", pixel_format="yuv420p",
        receiver_readback="ffmpeg_rgb24",
    ):
        raise ValueError("fixed media contract changed")
    if config["observation_layers"] != [
        "float_rgb", "rgb8_quantized", "mp4_rgb24"
    ]:
        raise ValueError("fixed receiver evidence layers changed")
    if config["proxy"] != dict(
        loss="mean(relu(-q_g)^2)", frames=181, blocks=160, groups=30,
        numpy_rtol=0.00002, numpy_atol=0.000002,
    ):
        raise ValueError("fixed proxy changed")
    if config["control"] != dict(
        gradient_index=49, reference_index=49, latent_support_time=[1, 45],
        target_D_support_rms=R_STAR, single_update=True,
        no_scan_no_retry_no_fallback=True,
    ) or AMPLITUDE != 0.1:
        raise ValueError("fixed writer control changed")
    if config["decision_rule"] != dict(
        statistic="C=sum(q_g>0)", threshold=24,
        rule="C>=24:H1;C<24:H0", mp4_only=True,
    ):
        raise ValueError("fixed MP4 decision changed")
    if config["fixed_denominator"] != dict(
        sources=2, arms_per_source=3, mp4_score_slots=6, frames=1086,
    ):
        raise ValueError("fixed denominator changed")
    if config["call_plan_max"] != PLAN or config["derived_call_totals"] != DERIVED:
        raise ValueError("fixed call plan changed")
    if config["resources"] != dict(
        case_timeout_seconds=14400, worker_processes_per_case=1,
        case_execution="serial fresh child processes",
        termination_grace_seconds=10,
        vae_transformer_separate_phases=True,
        replay_limits_per_source={"transformer_block": 256, "vae_chunk": 64},
    ):
        raise ValueError("fixed resource plan changed")
    return config


def environment_receipt(config: dict) -> dict:
    """Validate required APIs while recording, not constraining, CUDA suffix."""
    import diffusers
    import torch
    from torch.utils.checkpoint import checkpoint

    public = str(torch.__version__).split("+", 1)[0]
    suffix = (str(torch.__version__).split("+", 1)[1]
              if "+" in str(torch.__version__) else None)
    expected = config["runtime"]
    if diffusers.__version__ != expected["diffusers_version"]:
        raise RuntimeError(f"diffusers version mismatch: {diffusers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for fixed Wan execution")
    if "use_reentrant" not in inspect.signature(checkpoint).parameters:
        raise RuntimeError("non-reentrant torch checkpoint API unavailable")
    hooks = getattr(torch.autograd.graph, "saved_tensors_hooks", None)
    if not callable(hooks):
        raise RuntimeError("saved_tensors_hooks API unavailable")
    return dict(
        torch_version=str(torch.__version__), torch_public_version=public,
        torch_version_policy=expected["torch_version_policy"],
        torch_build_suffix=suffix,
        cuda_build_suffix_policy=expected["cuda_build_suffix_policy"],
        torch_cuda_version=torch.version.cuda,
        cuda_available=True, cuda_device_name=torch.cuda.get_device_name(0),
        diffusers_version=diffusers.__version__,
        non_reentrant_checkpoint=True, saved_tensors_hooks=True,
    )


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _slot(case_dir: Path, arm: str) -> dict:
    return dict(
        status="PENDING", reason=None, score=None, group_scores=None,
        positive_groups=None, c_margin=None, decision=None,
        quality_vs_off=None, attempted=False, expected_frames=181, frames_used=0,
        path=str(case_dir / "received_videos" / arm / "FULL.mp4"),
        sha256=None, bytes=None, receiver_status=None,
        layers={
            name: dict(status="PENDING", reason=None, score=None,
                       group_scores=None, positive_groups=None,
                       frames_used=0, decision=None)
            for name in ("float_rgb", "rgb8_quantized", "mp4_rgb24")
        },
    )


def initial_result(config: dict, output: Path, source_sha: str) -> dict:
    cases = {}
    for source in config["cases"]:
        case_id = source["id"]
        cases[case_id] = dict(
            status="PENDING", stage="PENDING", prompt=source["prompt"], seed=source["seed"],
            slots={arm: _slot(output / case_id, arm) for arm in ARMS},
            generation=None, proxy=None, gradient=None, gradient_effect=None,
            lifts={}, controls={}, phase_receipts=[], recompute_ledger=None,
            calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
            environment_receipt={"status": "PENDING"},
            worker_receipt={"status": "PENDING"},
            release_receipt={"status": "PENDING"},
            resources={}, elapsed_seconds=None, failures=[],
        )
    return dict(
        status="RUNNING", experiment_id=config["name"], source_sha=source_sha,
        config_path=str(CONFIG_PATH), config_sha256=_sha(CONFIG_PATH),
        protocol_path=str(PROTOCOL_PATH), protocol_sha256=config["protocol_sha256"],
        frozen_task_card_sha256=config["frozen_task_card_sha256"],
        receiver_spec_sha256=config["receiver_spec_sha256"],
        baseline_receiver_spec_sha256=config["baseline_receiver_spec_sha256"],
        key_id=config["receiver_key_id"], key_utf8=config["key_utf8"],
        model=config["model"], runtime_environment={
            case_id: {"status": "PENDING"} for case_id in CASE_IDS
        },
        generation_config=config["generation"], media_config=config["media"],
        proxy_config=config["proxy"], control_config=config["control"],
        decision_rule=config["decision_rule"], fixed_denominator=config["fixed_denominator"],
        call_plan_max=config["call_plan_max"], derived_call_totals=DERIVED,
        call_completion_definition={
            "mp4_read": "completed only for one valid full 181-frame RGB24 readback",
            "score": "completed only for a finite frozen-receiver MP4 C score",
            "vae_vjp": "one full 181-frame FP32 VAE loss-to-terminal cotangent",
        },
        resources_config=config["resources"],
        calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
        attempted_media_slots=0, scored_media_slots=0, invalid_media_slots=0,
        pending_media_slots=6, scored_frames=0, reporting=None,
        cases=cases, output_dir=str(output), result_path=str(output / "result.json"),
        evidence_ceiling=config["evidence_ceiling"],
    )


class Store:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data
        self.active_case: str | None = None

    @classmethod
    def open(cls, path: Path) -> "Store":
        return cls(path, json.loads(path.read_text(encoding="utf-8")))

    def case(self, case_id: str) -> dict:
        return self.data["cases"][case_id]

    def save(self) -> None:
        slots = [slot for case in self.data["cases"].values()
                 for slot in case["slots"].values()]
        if len(slots) != 6:
            raise RuntimeError("fixed six-slot denominator changed")
        statuses = [slot["status"] for slot in slots]
        self.data["scored_media_slots"] = statuses.count("SCORED")
        self.data["pending_media_slots"] = statuses.count("PENDING")
        self.data["invalid_media_slots"] = (
            6 - self.data["scored_media_slots"] - self.data["pending_media_slots"]
        )
        self.data["attempted_media_slots"] = self.data["calls"]["mp4_save"]["attempted"]
        self.data["scored_frames"] = sum(
            slot["frames_used"] for slot in slots if slot["status"] == "SCORED"
        )
        _atomic_json(self.path, self.data)

    def count(self, kind: str, completed: bool) -> None:
        if kind not in PLAN:
            raise ValueError(f"unplanned call kind: {kind}")
        field = "completed" if completed else "attempted"
        total = self.data["calls"][kind]
        total[field] += 1
        if total["attempted"] > PLAN[kind] or total["completed"] > total["attempted"]:
            raise RuntimeError(f"fixed {kind} call cap/order exceeded")
        if self.active_case is not None:
            row = self.case(self.active_case)["calls"][kind]
            row[field] += 1
            if row["attempted"] > PLAN[kind] // 2 or row["completed"] > row["attempted"]:
                raise RuntimeError(f"per-source {kind} call cap/order exceeded")
        self.save()

    def replay(self, summary: dict) -> None:
        if self.active_case is None:
            raise RuntimeError("replay event outside active source")
        self.case(self.active_case)["recompute_ledger"] = copy.deepcopy(summary)
        self.save()


def _encode_rgb_np(rgb: np.ndarray, path: Path, fps: int, crf: int) -> None:
    import torch
    from runtime.wan.io import encode_rgb
    encode_rgb(torch.from_numpy(np.ascontiguousarray(rgb)), path, fps, crf)


def _validate_rgb(rgb: np.ndarray) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.shape != RGB_SHAPE or rgb.dtype != np.float32:
        raise ValueError("fixed FP32 [181,320,512,3] RGB required")
    for frame in rgb:
        if not np.isfinite(frame).all() or np.any(frame < 0) or np.any(frame > 1):
            raise ValueError("finite RGB in [0,1] required")


def _quantized_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return the exact RGB8 raster used by runtime.wan.io before FFmpeg."""
    import torch
    from runtime.wan.vae import quantize_rgb8_no_codec

    pixels = quantize_rgb8_no_codec(
        torch.from_numpy(np.ascontiguousarray(rgb))
    )
    return pixels.to(dtype=torch.float32).div(255).numpy()


def _layer_record(scored: dict) -> dict:
    q = scored.get("group_scores")
    return dict(
        status=scored.get("status"), reason=scored.get("reason"),
        score=(float(scored["score"])
               if isinstance(scored.get("score"), (int, float)) else None),
        group_scores=([float(value) for value in q]
                      if isinstance(q, list) and len(q) == 30 else None),
        positive_groups=scored.get("positive_groups"),
        frames_used=scored.get("frames_used", 0), decision=None,
    )


def _score_memory_layer(rgb: np.ndarray, key: bytes) -> dict:
    try:
        record = _layer_record(receiver.score_rgb(rgb, key))
    except Exception as exc:
        return dict(
            status="INVALID", reason=f"{type(exc).__name__}: {exc}", score=None,
            group_scores=None, positive_groups=None, frames_used=0, decision=None,
        )
    return record


def _require_scored_layer(record: dict) -> None:
    if (record["status"] != "SCORED" or record["group_scores"] is None
            or not isinstance(record["positive_groups"], int)
            or sum(value > 0 for value in record["group_scores"])
            != record["positive_groups"]):
        raise RuntimeError(
            f"in-memory receiver layer invalid: {record['reason'] or 'MALFORMED'}"
        )


def _paired_quality(off: np.ndarray, marked: np.ndarray) -> dict:
    if off.shape != RGB_SHAPE or marked.shape != RGB_SHAPE:
        raise ValueError("paired full MP4 RGB required")
    square_sum = count = 0.0
    for first, second in zip(off, marked, strict=True):
        difference = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
        if not np.isfinite(difference).all():
            raise ValueError("nonfinite paired MP4 difference")
        square_sum += float(np.square(difference).sum())
        count += difference.size
    mse = square_sum / count
    return dict(
        rgb_rmse=math.sqrt(mse),
        rgb_psnr_db=(-10 * math.log10(mse) if mse > 0 else None),
        rgb_psnr_infinite=(mse == 0), compared_frames=181,
        diagnostic_only=True, no_visual_pass_threshold=True,
    )


def _save_and_score(store: Store, case_id: str, arm: str, rgb: np.ndarray,
                    key: bytes, config: dict, *, encode_fn=_encode_rgb_np,
                    score_fn=score_mp4_once, off_received=None):
    slot = store.case(case_id)["slots"][arm]
    path = Path(slot["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    slot.update(status="PREPARING", attempted=True)
    store.save()
    try:
        _validate_rgb(rgb)
        slot["layers"]["float_rgb"] = _score_memory_layer(rgb, key)
        store.save()
        _require_scored_layer(slot["layers"]["float_rgb"])
        quantized = _quantized_rgb(rgb)
        slot["layers"]["rgb8_quantized"] = _score_memory_layer(quantized, key)
        store.save()
        _require_scored_layer(slot["layers"]["rgb8_quantized"])
        store.count("mp4_save", False)
        encode_fn(rgb, path, config["media"]["fps"], config["media"]["crf"])
        store.count("mp4_save", True)
        slot.update(sha256=_sha(path), bytes=path.stat().st_size)
        if slot["bytes"] <= 0:
            raise ValueError("empty MP4")
        store.save()
        store.count("mp4_read", False)
        store.count("score", False)
        scored, received = score_fn(path, key)
        slot["layers"]["mp4_rgb24"] = _layer_record(scored)
        slot["receiver_status"] = scored.get("status")
        slot["frames_used"] = scored.get("frames_used", 0)
        valid = (
            scored.get("status") == "SCORED"
            and isinstance(scored.get("score"), (int, float))
            and not isinstance(scored.get("score"), bool)
            and math.isfinite(scored["score"])
            and isinstance(scored.get("positive_groups"), int)
            and not isinstance(scored.get("positive_groups"), bool)
            and 0 <= scored["positive_groups"] <= 30
            and isinstance(scored.get("group_scores"), list)
            and len(scored["group_scores"]) == 30
            and all(isinstance(q, (int, float)) and math.isfinite(q)
                    for q in scored["group_scores"])
            and sum(q > 0 for q in scored["group_scores"]) == scored["positive_groups"]
            and slot["frames_used"] == 181
            and scored.get("spec_sha256") == receiver.SPEC_SHA256
            and scored.get("key_id") == config["receiver_key_id"]
        )
        if not valid:
            slot.update(status="INVALID", reason=scored.get("reason") or "RECEIVER_OUTPUT_INVALID")
        else:
            store.count("mp4_read", True)
            store.count("score", True)
            slot.update(
                status="SCORED", reason=None, score=float(scored["score"]),
                group_scores=[float(q) for q in scored["group_scores"]],
                positive_groups=scored["positive_groups"],
                c_margin=scored["positive_groups"] - 24,
            )
            store.save()
            slot["decision"] = receiver.decide(slot["positive_groups"])
            slot["layers"]["mp4_rgb24"]["decision"] = slot["decision"]
            if arm != "OFF" and off_received is not None:
                slot["quality_vs_off"] = _paired_quality(off_received, received)
    except Exception as exc:
        slot.update(status="ENGINEERING_INVALID", reason=f"{type(exc).__name__}: {exc}")
        received = None
    store.save()
    return received if slot["status"] == "SCORED" else None


def _fail_arm(store: Store, case_id: str, arm: str, stage: str, exc: Exception) -> None:
    case = store.case(case_id)
    case["failures"].append(dict(
        arm=arm, stage=stage, error=f"{type(exc).__name__}: {exc}",
        traceback=traceback.format_exc(),
    ))
    slot = case["slots"][arm]
    if slot["status"] in ("PENDING", "PREPARING"):
        slot.update(status="ENGINEERING_INVALID", reason=f"{type(exc).__name__}: {exc}")
    store.save()


def _no_media_result(store: Store, case_id: str, arm: str, state: str) -> None:
    """A finite zero without the fixed MP4 is engineering-invalid evidence."""
    store.case(case_id)["slots"][arm].update(
        status="ENGINEERING_INVALID", reason=f"FINITE_{state}_NO_MP4", attempted=True,
    )
    store.save()


def _dependency_invalid(store: Store, case_id: str, arm: str, reason: str) -> None:
    slot = store.case(case_id)["slots"][arm]
    if slot["status"] in ("PENDING", "PREPARING"):
        slot.update(status="NOT_RUN_ENGINEERING_INVALID", reason=reason,
                    attempted=False, decision=None)
    store.save()


def _mark_pending(store: Store, case_id: str, status: str, reason: str) -> None:
    for slot in store.case(case_id)["slots"].values():
        if slot["status"] in ("PENDING", "PREPARING"):
            slot.update(status=status, reason=reason)
    store.save()


def _finalize_case(store: Store, case_id: str) -> None:
    case = store.case(case_id)
    states = [slot["status"] for slot in case["slots"].values()]
    receipts_valid = (
        case["environment_receipt"].get("status") == "VALID"
        and case["release_receipt"].get("status") == "COMPLETED"
        and case["resources"].get("receipt_valid") is True
    )
    if (all(state == "SCORED" for state in states)
            and not case["failures"] and receipts_valid):
        case["status"] = "SCORED"
    elif case["failures"] or any("INVALID" in state or "FAILURE" in state for state in states):
        case["status"] = "ENGINEERING_INVALID"
    else:
        case["status"] = "INCOMPLETE"
    store.save()


def _single49_lift(store: Store, case_id: str, off_rgb: np.ndarray, key: bytes,
                   backend):
    case = store.case(case_id)
    case["stage"] = "SINGLE49_POSITIVE_VAE_LIFT"
    case["slots"]["SINGLE49"]["attempted"] = True
    store.save()
    plus, clipping = apply_carrier(off_rgb, key, +1)
    encoded_plus = backend.encode(plus)
    encoded_base = backend.encode(off_rgb)
    masked, geometry = lift_direction(encoded_plus - encoded_base)
    case["lifts"]["SINGLE49"] = dict(
        index=49, clipping=clipping, geometry=geometry,
    )
    store.save()
    if geometry["status"] == "ZERO_LIFT_DIRECTION":
        _no_media_result(store, case_id, "SINGLE49", "ZERO_LIFT_DIRECTION")
        return None
    return masked


def _gradient_effect(off_proxy: dict, marked_rgb: np.ndarray, key: bytes) -> dict:
    """Record finite RGB loss and fixed group sign flips without selection."""
    from main.tube_state.rgb_dct_terminal_gradient import numpy_proxy

    marked = numpy_proxy(marked_rgb, key)
    before = np.asarray(off_proxy["q"], dtype=np.float64)
    after = marked["q"]
    if before.shape != (30,) or after.shape != (30,):
        raise ValueError("full 30-group gradient diagnostic required")
    return dict(
        off_loss=float(off_proxy["loss"]), marked_loss=marked["loss"],
        actual_float_rgb_loss_delta=marked["loss"] - float(off_proxy["loss"]),
        off_q=[float(value) for value in before],
        marked_q=[float(value) for value in after],
        gain_groups=[int(i) for i in np.flatnonzero((before <= 0) & (after > 0))],
        loss_groups=[int(i) for i in np.flatnonzero((before > 0) & (after <= 0))],
        off_c=int(np.count_nonzero(before > 0)),
        marked_c=int(np.count_nonzero(after > 0)),
    )


def run_case(store: Store, case_id: str, config: dict, backend,
             *, encode_fn=_encode_rgb_np, score_fn=score_mp4_once) -> None:
    """Run all three fixed same-source arms with independent failure retention."""
    case = store.case(case_id)
    store.active_case = case_id
    key = config["key_utf8"].encode()
    case_dir = Path(store.data["output_dir"]) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    off_received = off_rgb = single_direction = None
    try:
        case.update(status="RUNNING", stage="SHARED_OFF_TRAJECTORY")
        store.save()
        case["generation"] = backend.prepare_off(case_dir / "state")
        case["stage"] = "FP32_VAE_TERMINAL_GRADIENT"
        store.save()
        case["phase_receipts"].append(backend.to_vae_phase("GRADIENT_VAE"))
        store.save()
        try:
            off_rgb, case["proxy"] = backend.terminal_cotangent(key)
            case["gradient"] = backend.terminal_gradient_receipt
            store.save()
            case["stage"] = "OFF_MP4_SAVE_READ_SCORE"
            store.save()
            off_received = _save_and_score(
                store, case_id, "OFF", off_rgb, key, config,
                encode_fn=encode_fn, score_fn=score_fn,
            )
        except Exception as exc:
            _fail_arm(store, case_id, "OFF", case["stage"], exc)
            _fail_arm(store, case_id, "TERMINAL49_RECEIVER", case["stage"], exc)
            _dependency_invalid(
                store, case_id, "SINGLE49",
                f"DEPENDENCY_SHARED_OFF_GRADIENT_FAILED:{type(exc).__name__}:{exc}",
            )
        if off_rgb is not None:
            try:
                single_direction = _single49_lift(
                    store, case_id, off_rgb, key, backend
                )
            except Exception as exc:
                _fail_arm(store, case_id, "SINGLE49", case["stage"], exc)

        if not any(case["slots"][arm]["status"] == "PENDING"
                   for arm in ("SINGLE49", "TERMINAL49_RECEIVER")):
            return

        case["stage"] = "TRANSFORMER_RESTORE"
        store.save()
        case["phase_receipts"].append(backend.to_transformer_phase("T49_CONTROL_RESUME"))
        store.save()

        if case["slots"]["TERMINAL49_RECEIVER"]["status"] == "PENDING":
            try:
                case["stage"] = "TERMINAL49_RECEIVER_NATIVE_RESPONSE"
                store.save()
                state, _, control = backend.control_terminal49_receiver(R_STAR)
                case["controls"]["TERMINAL49_RECEIVER"] = control
                store.save()
                if state != "READY":
                    _no_media_result(store, case_id, "TERMINAL49_RECEIVER", state)
            except Exception as exc:
                _fail_arm(store, case_id, "TERMINAL49_RECEIVER", case["stage"], exc)

        if single_direction is not None and case["slots"]["SINGLE49"]["status"] == "PENDING":
            try:
                case["stage"] = "SINGLE49_NATIVE_CONTROL"
                store.save()
                state, _, control = backend.control_single49(single_direction, R_STAR)
                case["controls"]["SINGLE49"] = control
                store.save()
                if state != "READY":
                    _no_media_result(store, case_id, "SINGLE49", state)
            except Exception as exc:
                _fail_arm(store, case_id, "SINGLE49", case["stage"], exc)

        case["stage"] = "FINAL_FP32_VAE_MEDIA"
        store.save()
        case["phase_receipts"].append(backend.to_vae_phase("FINAL_MEDIA_VAE"))
        store.save()
        for arm in ("SINGLE49", "TERMINAL49_RECEIVER"):
            if case["slots"][arm]["status"] != "PENDING" or arm not in backend.terminals:
                continue
            try:
                case["stage"] = f"{arm}_DECODE_SAVE_READ_SCORE"
                store.save()
                rgb = backend.decode(backend.terminals[arm])
                if arm == "TERMINAL49_RECEIVER":
                    case["gradient_effect"] = _gradient_effect(case["proxy"], rgb, key)
                    store.save()
                _save_and_score(
                    store, case_id, arm, rgb, key, config,
                    encode_fn=encode_fn, score_fn=score_fn,
                    off_received=off_received,
                )
            except Exception as exc:
                _fail_arm(store, case_id, arm, case["stage"], exc)
    except Exception as exc:
        case["failures"].append(dict(
            stage=case["stage"], error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        ))
        if case["stage"] in {
            "SHARED_OFF_TRAJECTORY", "FP32_VAE_TERMINAL_GRADIENT",
        }:
            _dependency_invalid(
                store, case_id, "SINGLE49",
                f"DEPENDENCY_SHARED_OFF_FAILED:{type(exc).__name__}:{exc}",
            )
        _mark_pending(store, case_id, "NOT_RUN_ENGINEERING_INVALID",
                      f"{type(exc).__name__}: {exc}")
    finally:
        case["elapsed_seconds"] = time.perf_counter() - started
        try:
            case["resources"] = backend.resources()
            case["resources"].update(
                cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                wall_elapsed_seconds=case["elapsed_seconds"],
            )
        except Exception as exc:
            case["resources"] = {"status": "UNAVAILABLE", "reason": repr(exc)}
            case["failures"].append(dict(stage="RESOURCE_RECEIPT", error=repr(exc)))
        try:
            backend.release()
            case["release_receipt"] = {"status": "COMPLETED"}
        except Exception as exc:
            case["release_receipt"] = {"status": "FAILED", "reason": repr(exc)}
            case["failures"].append(dict(stage="RELEASE", error=repr(exc)))
        _finalize_case(store, case_id)
        store.active_case = None
        gc.collect()


def finalize_experiment(store: Store) -> None:
    expected = {case_id: {arm: ("H0" if arm == "OFF" else "H1") for arm in ARMS}
                for case_id in CASE_IDS}
    slots = [(case_id, arm, store.case(case_id)["slots"][arm])
             for case_id in CASE_IDS for arm in ARMS]
    decisions = {case_id: {arm: store.case(case_id)["slots"][arm]["decision"]
                           for arm in ARMS} for case_id in CASE_IDS}
    complete = (
        all(slot["status"] == "SCORED" for _, _, slot in slots)
        and all(store.case(case_id)["status"] == "SCORED" for case_id in CASE_IDS)
        and all(not store.case(case_id)["failures"] for case_id in CASE_IDS)
        and all(store.case(case_id)["environment_receipt"].get("status") == "VALID"
                for case_id in CASE_IDS)
        and all(store.case(case_id)["worker_receipt"].get("status") == "COMPLETED"
                for case_id in CASE_IDS)
        and all(store.case(case_id)["release_receipt"].get("status") == "COMPLETED"
                for case_id in CASE_IDS)
        and all(store.case(case_id)["resources"].get("receipt_valid") is True
                for case_id in CASE_IDS)
    )
    mismatches = [
        dict(case_id=case_id, arm=arm, expected=expected[case_id][arm],
             observed=slot["decision"], positive_groups=slot["positive_groups"])
        for case_id, arm, slot in slots
        if slot["status"] == "SCORED" and slot["decision"] != expected[case_id][arm]
    ]
    comparisons = {}
    for case_id in CASE_IDS:
        by_arm = store.case(case_id)["slots"]
        off_c = by_arm["OFF"]["positive_groups"]
        comparisons[case_id] = {
            arm: dict(
                c=by_arm[arm]["positive_groups"],
                c_minus_off=(by_arm[arm]["positive_groups"] - off_c
                             if by_arm[arm]["positive_groups"] is not None
                             and off_c is not None else None),
                quality_vs_off=by_arm[arm]["quality_vs_off"],
            ) for arm in ARMS[1:]
        }
    store.data["reporting"] = dict(
        joined_after_decision_persistence=True, expected=expected,
        decisions=decisions, observed_misclassifications=mismatches,
        same_source_comparison=comparisons,
    )
    store.data["status"] = (
        "FIXED_B2_TERMINAL_RECEIVER_COMPLETE" if complete and not mismatches
        else "FIXED_B2_TERMINAL_RECEIVER_NEGATIVE" if complete else "INCOMPLETE"
    )
    store.save()


def _source_sha() -> str:
    value = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("source Git SHA unavailable")
    return value


def _validate_output(output: Path) -> None:
    if output.parent != OUTPUT_PARENT or not re.fullmatch(r"\d{8}T\d{12}Z", output.name):
        raise ValueError("fresh timestamp output under fixed Drive parent required")
    if output.exists():
        if not (output / "setup_receipt.json").is_file() or (output / "result.json").exists():
            raise FileExistsError("output is not a fresh notebook setup directory")
    else:
        if not OUTPUT_PARENT.is_dir():
            raise FileNotFoundError("fixed Drive output parent absent")
        output.mkdir(exist_ok=False)


def _proc_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def _persist_worker_monitor(output: Path, case_id: str, receipt: dict) -> None:
    _atomic_json(output / f"{case_id}_worker_monitor.json", receipt)


def _run_worker(output: Path, config: dict, case_id: str) -> dict:
    env = os.environ.copy()
    env["RGB_DCT_B2_TERMINAL_RECEIVER_INTERNAL_CASE"] = case_id
    local_root = Path("/content") if Path("/content").is_dir() else Path("/tmp")
    spool_owner = tempfile.TemporaryDirectory(
        prefix=f"wan-vae-worker-{case_id}-", dir=local_root
    )
    env["RGB_DCT_BOUNDARY_SPOOL_ROOT"] = spool_owner.name
    command = [sys.executable, "-u", "-m", MODULE, "--output", str(output)]
    worker_log = output / f"{case_id}_worker.log"
    started = time.monotonic()
    receipt = dict(
        status="STARTING", error=None, exit_code=None, elapsed_seconds=0.0,
        parent_observed_peak_rss_kib=None, boundary_spool_cleanup="PENDING",
        timeout_seconds=(
            config["resources"]["case_timeout_seconds"]
        ), termination=None, log_path=str(worker_log),
    )
    _persist_worker_monitor(output, case_id, receipt)
    child = None
    with worker_log.open("w", encoding="utf-8") as stream:
        try:
            child = subprocess.Popen(
                command, cwd=ROOT, env=env, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
            receipt["status"] = "RUNNING"
            deadline = started + config["resources"]["case_timeout_seconds"]
            while child.poll() is None and time.monotonic() < deadline:
                rss = _proc_rss_kib(child.pid)
                if rss is not None:
                    receipt["parent_observed_peak_rss_kib"] = max(
                        receipt["parent_observed_peak_rss_kib"] or 0, rss
                    )
                receipt["elapsed_seconds"] = time.monotonic() - started
                _persist_worker_monitor(output, case_id, receipt)
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            if child.poll() is None:
                receipt.update(status="TERMINATING", error="WORKER_TIMEOUT",
                               termination="SIGTERM_SENT")
                _persist_worker_monitor(output, case_id, receipt)
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                grace_deadline = time.monotonic() + config["resources"]["termination_grace_seconds"]
                while child.poll() is None and time.monotonic() < grace_deadline:
                    rss = _proc_rss_kib(child.pid)
                    if rss is not None:
                        receipt["parent_observed_peak_rss_kib"] = max(
                            receipt["parent_observed_peak_rss_kib"] or 0, rss
                        )
                    receipt["elapsed_seconds"] = time.monotonic() - started
                    _persist_worker_monitor(output, case_id, receipt)
                    time.sleep(min(0.5, max(0.0, grace_deadline - time.monotonic())))
                if child.poll() is None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    receipt["termination"] = "SIGTERM_THEN_SIGKILL"
                else:
                    receipt["termination"] = "SIGTERM_GRACEFUL"
                child.wait()
                receipt["status"] = "TIMEOUT"
            else:
                receipt["exit_code"] = child.returncode
                if child.returncode == 0:
                    receipt["status"] = "COMPLETED"
                else:
                    receipt.update(status="FAILED",
                                   error=f"WORKER_EXIT_{child.returncode}")
        except Exception as exc:
            receipt.update(
                status="START_FAILED",
                error=f"WORKER_START_FAILED:{type(exc).__name__}:{exc}",
            )
    if child is not None and child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        receipt.update(status="FAILED", error="WORKER_MONITOR_ABORTED")
    try:
        spool_owner.cleanup()
        receipt["boundary_spool_cleanup"] = "COMPLETED"
    except Exception as exc:
        receipt.update(status="FAILED", boundary_spool_cleanup="FAILED",
                       error=f"BOUNDARY_SPOOL_CLEANUP:{type(exc).__name__}:{exc}")
    receipt["elapsed_seconds"] = time.monotonic() - started
    _persist_worker_monitor(output, case_id, receipt)
    return receipt


def _supervise(output: Path, config: dict, source_sha: str,
               *, worker_fn: Callable = _run_worker) -> None:
    result_path = output / "result.json"
    store = Store(result_path, initial_result(config, output, source_sha))
    store.save()
    for case_id in CASE_IDS:
        receipt = worker_fn(output, config, case_id)
        if receipt is None:  # CPU/fake worker compatibility.
            receipt = dict(status="COMPLETED", error=None, exit_code=0,
                           elapsed_seconds=0.0, parent_observed_peak_rss_kib=None,
                           termination=None)
        store = Store.open(result_path)
        case = store.case(case_id)
        case["worker_receipt"] = receipt
        if receipt["status"] != "COMPLETED":
            failure = receipt.get("error") or receipt["status"]
            case = store.case(case_id)
            case["failures"].append(dict(
                stage="WORKER_SUPERVISION", error=failure,
                 log_path=str(output / f"{case_id}_worker.log"),
            ))
            if not case.get("resources") or case["resources"].get("receipt_valid") is not True:
                receipt["cuda_peak"] = {
                    "status": "UNAVAILABLE",
                    "reason": f"{receipt['status']}:{failure}",
                }
                case["resources"] = dict(
                    receipt_valid=False,
                    parent_observed_peak_rss_kib=receipt.get(
                        "parent_observed_peak_rss_kib"
                    ),
                    cuda={"status": "UNAVAILABLE",
                          "reason": f"{receipt['status']}:{failure}"},
                    wall_elapsed_seconds=receipt.get("elapsed_seconds"),
                )
            else:
                receipt["cuda_peak"] = {
                    "status": "AVAILABLE",
                    "peak_allocated": case["resources"]["cuda"].get("peak_allocated"),
                    "peak_reserved": case["resources"]["cuda"].get("peak_reserved"),
                }
            _mark_pending(
                store, case_id,
                "NOT_RUN_RESOURCE_FAILURE" if receipt["status"] == "TIMEOUT"
                else "NOT_RUN_WORKER_FAILURE",
                failure,
            )
        _finalize_case(store, case_id)
        store.case(case_id)["worker_log_path"] = str(output / f"{case_id}_worker.log")
        store.case(case_id)["worker_monitor_path"] = str(
            output / f"{case_id}_worker_monitor.json"
        )
        store.save()
    finalize_experiment(store)


def _record_child_environment(store: Store, case_id: str, config: dict) -> bool:
    """Persist the fresh-child environment result before model construction."""
    case = store.case(case_id)
    case["stage"] = "ENVIRONMENT_CHECK"
    store.save()
    try:
        receipt = environment_receipt(config)
        receipt["status"] = "VALID"
        case["environment_receipt"] = receipt
        store.data["runtime_environment"][case_id] = receipt
        store.save()
        return True
    except Exception as exc:
        failure = f"{type(exc).__name__}: {exc}"
        receipt = dict(status="FAILED", stage="ENVIRONMENT_CHECK",
                       reason=failure, traceback=traceback.format_exc())
        case["environment_receipt"] = receipt
        store.data["runtime_environment"][case_id] = receipt
        case["failures"].append(dict(stage="ENVIRONMENT_CHECK", error=failure,
                                      traceback=traceback.format_exc()))
        case["resources"] = dict(
            receipt_valid=False,
            cuda={"status": "UNAVAILABLE",
                  "reason": f"ENVIRONMENT_CHECK:{failure}"},
        )
        case["release_receipt"] = {
            "status": "NOT_RUN", "reason": "ENVIRONMENT_CHECK_FAILED"
        }
        _mark_pending(store, case_id, "NOT_RUN_ENGINEERING_INVALID",
                      f"ENVIRONMENT_CHECK:{failure}")
        _finalize_case(store, case_id)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    output = args.output
    internal = os.environ.get("RGB_DCT_B2_TERMINAL_RECEIVER_INTERNAL_CASE")
    if internal:
        if internal not in CASE_IDS or not (output / "result.json").is_file():
            raise ValueError("invalid fixed worker invocation")
        source = next(row for row in config["cases"] if row["id"] == internal)
        runtime_config = copy.deepcopy(config)
        runtime_config["generation"].update(prompt=source["prompt"], seed=source["seed"])
        store = Store.open(output / "result.json")
        if store.data["config_sha256"] != _sha(CONFIG_PATH) or store.data["source_sha"] != _source_sha():
            raise ValueError("worker source/config binding changed")
        if store.case(internal)["status"] != "PENDING":
            raise ValueError("fixed source already attempted; no retry")
        if not _record_child_environment(store, internal, config):
            return
        from runtime.wan.rgb_dct_b2_terminal_receiver_backend import WanB2TerminalReceiverBackend
        backend = WanB2TerminalReceiverBackend(
            runtime_config, store.count, replay_callback=store.replay,
        )
        run_case(store, internal, config, backend)
        return
    _validate_output(output)
    _supervise(output, config, _source_sha())


if __name__ == "__main__":
    main()
