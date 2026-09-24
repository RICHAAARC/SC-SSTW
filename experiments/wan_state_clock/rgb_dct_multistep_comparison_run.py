"""Fixed two-source, four-arm RGB-DCT T44/T46/T49 comparison."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from main.tube_state import rgb_dct_group_consistency as receiver
from main.tube_state.rgb_dct_t49_carrier import AMPLITUDE, RGB_SHAPE, apply_carrier, lift_direction
from runtime.wan.rgb_dct_multistep_adapter import score_mp4_once

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "configs/rgb_dct_multistep_comparison_v1.json"
PROTOCOL_PATH = ROOT / "docs/rgb_dct_multistep_comparison_protocol_20260924.md"
MODULE = "experiments.wan_state_clock.rgb_dct_multistep_comparison_run"
OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/RGB-DCT-Multistep-Comparison-V1")
CASE_IDS = ("eval_kite_s2421", "eval_pottery_s2422")
ARMS = ("OFF", "SINGLE49", "SINGLE46", "MULTI44_46")
PROMPTS = (
    "locked camera, a single red kite drifting steadily across a clear blue sky, stable daylight, no people, no cuts",
    "locked camera, a blue ceramic vase rotating slowly on a plain display turntable, stable indoor light, no people, no cuts",
)
SEEDS = (2026092421, 2026092422)
R_STAR = 0.042943312697648145
PLAN = dict(generation=2, transformer=232, transformer_validation=12,
            scheduler_step=122, shadow_step=8, unit_response_probe_step=8,
            vae_decode=14, vae_encode=16, backward=0,
            mp4_save=8, mp4_read=8, score=8)
TOTAL_NATIVE_STEPS = 138
TOTAL_TRANSFORMER = 244


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if _sha(PROTOCOL_PATH) != config["protocol_sha256"]:
        raise ValueError("adopted protocol byte identity mismatch")
    if config["receiver_spec_sha256"] != receiver.SPEC_SHA256:
        raise ValueError("RGB-DCT receiver specification changed")
    if config["baseline_receiver_spec_sha256"] != receiver.baseline.SPEC_SHA256:
        raise ValueError("original RGB-DCT feature specification changed")
    if (config["receiver_baseline_source_sha"] != "c967db03767768871ba0cd3b49208aa308f51b48"
            or config["model"] != dict(
                id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                revision="0fad780a534b6463e45facd96134c9f345acfa5b",
                transformer_dtype="bfloat16", vae_dtype="float32")):
        raise ValueError("frozen receiver baseline/model changed")
    if receiver.score_rgb(None, config["key_utf8"].encode())["key_id"] != config["receiver_key_id"]:
        raise ValueError("fixed receiver key changed")
    expected = [dict(id=case_id, seed=seed, prompt=prompt)
                for case_id, seed, prompt in zip(CASE_IDS, SEEDS, PROMPTS, strict=True)]
    if config["cases"] != expected:
        raise ValueError("fixed two-source prompt/seed roster changed")
    if config["generation"] != dict(
            height=320, width=512, frames=181, fps=8, steps=50,
            guidance_scale=5.0, max_sequence_length=512,
            negative_prompt="text, watermark, logo, camera motion, cuts, multiple objects, flicker"):
        raise ValueError("fixed Wan generation changed")
    if config["media"] != dict(fps=8, crf=18, codec="libx264",
                               pixel_format="yuv420p", receiver_readback="ffmpeg_rgb24"):
        raise ValueError("fixed media contract changed")
    if (config["carrier"] != dict(
            amplitude=0.1, rgb_channels="same additive carrier in R, G, B",
            positive_only_native_lift=True, latent_support_time=[1, 45])
            or AMPLITUDE != 0.1
            or config["control"] != dict(
                single_indices=[49, 46], multi_indices=[44, 46],
                target_D_support_rms=R_STAR,
                multi_each_target_D_support_rms=R_STAR / 2,
                sigma50=0.0, no_gradient_no_scan_no_fallback=True)):
        raise ValueError("fixed writer mechanism changed")
    if config["decision_rule"] != dict(statistic="C=sum(q_g>0)", threshold=24,
                                      rule="C>=24:H1;C<24:H0", reference_calibration=False):
        raise ValueError("fixed C>=24 decision rule changed")
    if config["fixed_denominator"] != dict(sources=2, mp4_score_slots=8, frames=1448):
        raise ValueError("fixed denominator changed")
    if config["call_plan_max"] != PLAN | {"native_scheduler_total": TOTAL_NATIVE_STEPS,
                                          "transformer_total": TOTAL_TRANSFORMER}:
        raise ValueError("fixed call plan changed")
    if config["resources"] != dict(
            case_timeout_seconds=10800, worker_processes_per_case=1,
            case_execution="serial fresh child processes",
            vae_transformer_separate_phases=True,
            same_input_cfg_validations_per_case=3):
        raise ValueError("fixed resource plan changed")
    return config


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _slot(case_dir: Path, arm: str) -> dict:
    return dict(status="PENDING", reason=None, score=None, group_scores=None,
                positive_groups=None, c_margin=None, decision=None, quality_vs_off=None,
                attempted=False,
                expected_frames=181, frames_used=0,
                path=str(case_dir / "received_videos" / arm / "FULL.mp4"),
                sha256=None, bytes=None, receiver_status=None, clipping=None)


def initial_result(config: dict, output: Path, source_sha: str) -> dict:
    cases = {}
    for case in config["cases"]:
        case_id = case["id"]
        cases[case_id] = dict(
            status="PENDING", stage="PENDING",
            prompt=case["prompt"], seed=case["seed"],
            slots={arm: _slot(output / case_id, arm) for arm in ARMS},
            generation=None, phase_receipts=[], lifts={}, controls={},
            arm_metrics={}, cfg_restoration_checks=[],
            calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
            resources={}, elapsed_seconds=None, failures=[])
    return dict(
        status="RUNNING", experiment_id=config["name"],
        source_sha=source_sha, config_path=str(CONFIG_PATH),
        config_sha256=_sha(CONFIG_PATH), protocol_path=str(PROTOCOL_PATH),
        protocol_sha256=config["protocol_sha256"],
        receiver_baseline_source_sha=config["receiver_baseline_source_sha"],
        receiver_spec_sha256=config["receiver_spec_sha256"],
        baseline_receiver_spec_sha256=config["baseline_receiver_spec_sha256"],
        key_id=config["receiver_key_id"], key_utf8=config["key_utf8"],
        model=config["model"], generation_config=config["generation"],
        media_config=config["media"], carrier_config=config["carrier"],
        control_config=config["control"], decision_rule=config["decision_rule"],
        fixed_denominator=config["fixed_denominator"],
        call_plan_max=config["call_plan_max"],
        call_completion_definition={
            "mp4_read": "completed only for a valid full RGB24 readback with SCORED receiver output",
            "score": "completed only for a valid full-media C score",
        },
        resources_config=config["resources"],
        calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
        attempted_media_slots=0, scored_media_slots=0, invalid_media_slots=0,
        pending_media_slots=8, scored_frames=0,
        reporting=None,
        cases=cases, output_dir=str(output), result_path=str(output / "result.json"),
        evidence_ceiling=config["evidence_ceiling"])


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
        statuses = [slot["status"] for slot in slots]
        if len(slots) != 8:
            raise RuntimeError("fixed eight-slot denominator changed")
        self.data["scored_media_slots"] = statuses.count("SCORED")
        self.data["pending_media_slots"] = statuses.count("PENDING")
        self.data["attempted_media_slots"] = self.data["calls"]["mp4_save"]["attempted"]
        self.data["invalid_media_slots"] = 8 - self.data["scored_media_slots"] - self.data["pending_media_slots"]
        self.data["scored_frames"] = sum(slot["frames_used"] for slot in slots
                                          if slot["status"] == "SCORED")
        self.data["native_scheduler_calls"] = {
            name: self.data["calls"]["scheduler_step"][name]
            + self.data["calls"]["shadow_step"][name]
            + self.data["calls"]["unit_response_probe_step"][name]
            for name in ("attempted", "completed")}
        if self.data["native_scheduler_calls"]["attempted"] > TOTAL_NATIVE_STEPS:
            raise RuntimeError("native scheduler cap exceeded")
        self.data["transformer_calls"] = {
            name: self.data["calls"]["transformer"][name]
            + self.data["calls"]["transformer_validation"][name]
            for name in ("attempted", "completed")}
        if self.data["transformer_calls"]["attempted"] > TOTAL_TRANSFORMER:
            raise RuntimeError("total Transformer call cap exceeded")
        _atomic_json(self.path, self.data)

    def count(self, kind: str, completed: bool) -> None:
        if kind not in PLAN:
            raise ValueError("unplanned call kind")
        field = "completed" if completed else "attempted"
        row = self.data["calls"][kind]
        row[field] += 1
        if row["attempted"] > PLAN[kind] or row["completed"] > row["attempted"]:
            raise RuntimeError(f"fixed {kind} call cap/order exceeded")
        if self.active_case is not None:
            self.case(self.active_case)["calls"][kind][field] += 1
        self.save()


def _encode_rgb_np(rgb: np.ndarray, path: Path, fps: int, crf: int) -> None:
    import torch
    from runtime.wan.io import encode_rgb
    encode_rgb(torch.from_numpy(np.ascontiguousarray(rgb)), path, fps, crf)


def _validate_rgb_for_mp4(rgb: np.ndarray) -> None:
    if (not isinstance(rgb, np.ndarray) or rgb.shape != RGB_SHAPE
            or rgb.dtype not in (np.dtype("float32"), np.dtype("float64"))):
        raise ValueError("fixed float32/float64 [181,320,512,3] RGB required")
    for frame in rgb:
        if not np.isfinite(frame).all() or np.any(frame < 0) or np.any(frame > 1):
            raise ValueError("finite RGB in [0,1] required before MP4 quantization")


def _save_and_score(
    store: Store, case_id: str, arm: str, rgb: np.ndarray, key: bytes,
    config: dict, *, encode_fn: Callable = _encode_rgb_np,
    score_fn: Callable = score_mp4_once, clipping: dict | None = None,
    off_received: np.ndarray | None = None,
) -> np.ndarray | None:
    slot = store.case(case_id)["slots"][arm]
    path = Path(slot["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    slot.update(status="PREPARING", attempted=True, clipping=clipping)
    store.save()
    try:
        _validate_rgb_for_mp4(rgb)
        store.count("mp4_save", False)
        encode_fn(rgb, path, config["media"]["fps"], config["media"]["crf"])
        store.count("mp4_save", True)
        slot.update(sha256=_sha(path), bytes=path.stat().st_size)
        if slot["bytes"] <= 0:
            raise ValueError("empty MP4")
        store.save()  # Actual media identity precedes receiver score and decision.
        store.count("mp4_read", False)
        store.count("score", False)
        scored, received = score_fn(path, key)  # Blind C sees only MP4 path and key.
        slot["receiver_status"] = scored.get("status")
        slot["frames_used"] = scored.get("frames_used", 0)
        if (scored.get("status") != "SCORED"
                or not isinstance(scored.get("score"), (int, float))
                or isinstance(scored.get("score"), bool)
                or not math.isfinite(scored["score"])
                or not isinstance(scored.get("positive_groups"), int)
                or isinstance(scored.get("positive_groups"), bool)
                or not 0 <= scored["positive_groups"] <= 30
                or not isinstance(scored.get("group_scores"), list)
                or len(scored["group_scores"]) != 30
                or not all(isinstance(q, (int, float)) and math.isfinite(q)
                           for q in scored["group_scores"])
                or sum(q > 0 for q in scored["group_scores"]) != scored["positive_groups"]
                or slot["frames_used"] != 181
                or scored.get("spec_sha256") != receiver.SPEC_SHA256
                or scored.get("key_id") != config["receiver_key_id"]):
            slot.update(status="INVALID",
                        reason=scored.get("reason") or "RECEIVER_OUTPUT_INVALID")
        else:
            store.count("mp4_read", True)
            store.count("score", True)
            slot.update(status="SCORED", score=float(scored["score"]),
                        group_scores=[float(q) for q in scored["group_scores"]],
                        positive_groups=scored["positive_groups"],
                        c_margin=scored["positive_groups"] - 24, reason=None)
            store.save()  # Media identity and receiver C precede the blind decision.
            slot["decision"] = receiver.decide(slot["positive_groups"])
            if arm != "OFF" and off_received is not None:
                try:
                    slot["quality_vs_off"] = _paired_mp4_quality(off_received, received)
                except Exception as quality_exc:
                    slot["quality_error"] = f"{type(quality_exc).__name__}: {quality_exc}"
    except Exception as exc:
        slot.update(status="ENGINEERING_INVALID",
                    reason=f"{type(exc).__name__}: {exc}", score=None)
    store.save()
    return received if slot["status"] == "SCORED" else None


def _paired_mp4_quality(off: np.ndarray, marked: np.ndarray) -> dict:
    """Diagnostics on the two already decoded full MP4s, with no extra read."""
    if not isinstance(off, np.ndarray) or not isinstance(marked, np.ndarray) or off.shape != RGB_SHAPE or marked.shape != RGB_SHAPE:
        raise ValueError("paired full MP4 RGB required")
    abs_sum = square_sum = temporal_square_sum = count = temporal_count = 0.0
    previous_off = previous_marked = None
    for off_frame, marked_frame in zip(off, marked, strict=True):
        first = np.asarray(off_frame, dtype=np.float64)
        second = np.asarray(marked_frame, dtype=np.float64)
        difference = second - first
        if not np.isfinite(difference).all():
            raise ValueError("nonfinite paired MP4 difference")
        abs_sum += float(np.abs(difference).sum())
        square_sum += float(np.square(difference).sum())
        count += difference.size
        if previous_off is not None:
            temporal_difference = (second - previous_marked) - (first - previous_off)
            temporal_square_sum += float(np.square(temporal_difference).sum())
            temporal_count += temporal_difference.size
        previous_off, previous_marked = first, second
    mse = square_sum / count
    return dict(rgb_mae=abs_sum / count, rgb_rmse=math.sqrt(mse),
                rgb_psnr_db=(-10 * math.log10(mse) if mse > 0 else None),
                rgb_psnr_infinite=(mse == 0),
                temporal_difference_rmse=math.sqrt(temporal_square_sum / temporal_count),
                compared_frames=181)


def _fail_arm(store: Store, case_id: str, arm: str, stage: str,
              exc: Exception, status: str = "ENGINEERING_INVALID") -> None:
    case = store.case(case_id)
    case["failures"].append(dict(arm=arm, stage=stage,
                                 error=f"{type(exc).__name__}: {exc}",
                                 traceback=traceback.format_exc()))
    slot = case["slots"][arm]
    if slot["status"] in ("PENDING", "PREPARING"):
        slot.update(status=status, reason=f"{type(exc).__name__}: {exc}")
    store.save()


def _method_negative(store: Store, case_id: str, arm: str, state: str) -> None:
    store.case(case_id)["slots"][arm].update(status=state,
                                             reason=f"FINITE_{state}", attempted=True)
    store.save()


def _mark_pending(store: Store, case_id: str, status: str, reason: str) -> None:
    for slot in store.case(case_id)["slots"].values():
        if slot["status"] in ("PENDING", "PREPARING"):
            slot.update(status=status, reason=reason, score=None)
    store.save()


def _finalize_case(store: Store, case_id: str) -> None:
    case = store.case(case_id)
    states = [row["status"] for row in case["slots"].values()]
    if all(state == "SCORED" for state in states) and not case["failures"]:
        case["status"] = "SCORED"
    elif case["failures"] or any("INVALID" in state or "FAILURE" in state for state in states):
        case["status"] = "ENGINEERING_INVALID"
    elif "ZERO_LIFT_DIRECTION" in states or "ZERO_NATIVE_RESPONSE" in states:
        case["status"] = "METHOD_NEGATIVE"
    else:
        case["status"] = "INCOMPLETE"
    store.save()


def _make_lift(store: Store, case_id: str, arm: str, base_rgb: np.ndarray,
               key: bytes, backend, step_index: int) -> np.ndarray | None:
    case = store.case(case_id)
    case["stage"] = f"{arm}_POSITIVE_FLOAT_RGB_LIFT"
    case["slots"][arm]["attempted"] = True
    store.save()
    plus, clipping = apply_carrier(base_rgb, key, +1)
    encoded_plus = backend.encode(plus)
    encoded_base = backend.encode(base_rgb)
    masked, lift = lift_direction(encoded_plus - encoded_base)
    case["lifts"].setdefault(arm, []).append(dict(index=step_index,
        status=lift["status"], clipping=clipping, geometry=lift))
    store.save()
    del plus, encoded_plus, encoded_base
    if lift["status"] == "ZERO_LIFT_DIRECTION":
        _method_negative(store, case_id, arm, "ZERO_LIFT_DIRECTION")
        return None
    if lift["status"] != "READY":
        raise ValueError("unexpected positive lift state")
    return masked


def _arm_metrics(backend, terminal, controls: list[dict]) -> dict:
    support = [row["actual_D"]["support_rms"] for row in controls]
    return dict(step_indices=[row["index"] for row in controls],
                sum_support_rms=math.fsum(support),
                sum_support_rms_squared=math.fsum(value * value for value in support),
                step_peak_abs=[row["actual_D"]["peak_abs"] for row in controls],
                net_terminal_from_off=backend.net_from_off(terminal))


def _score_terminals(store: Store, case_id: str, config: dict, backend,
                     key: bytes, off_received: np.ndarray | None,
                     arms: tuple[str, ...], encode_fn: Callable,
                     score_fn: Callable) -> None:
    case = store.case(case_id)
    for arm in arms:
        if arm not in backend.terminals or case["slots"][arm]["status"] != "PENDING":
            continue
        try:
            case["stage"] = f"{arm}_FINAL_DECODE_SAVE_SCORE"
            store.save()
            rgb = backend.decode(backend.terminals[arm])
            _save_and_score(store, case_id, arm, rgb, key, config,
                            encode_fn=encode_fn, score_fn=score_fn,
                            off_received=off_received)
            del rgb
        except Exception as exc:
            _fail_arm(store, case_id, arm, case["stage"], exc)


def run_case(
    store: Store, case_id: str, config: dict, backend,
    *, encode_fn: Callable = _encode_rgb_np, score_fn: Callable = score_mp4_once,
) -> None:
    """Run all four same-noise arms; an arm failure cannot replace later arms."""
    case = store.case(case_id)
    store.active_case = case_id
    key = config["key_utf8"].encode("utf-8")
    case_dir = Path(store.data["output_dir"]) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    directions: dict[str, np.ndarray] = {}
    off_received = None
    try:
        case.update(status="RUNNING", stage="SHARED_OFF_TRAJECTORY")
        store.save()
        case["generation"] = backend.prepare_off(case_dir / "state")
        case["stage"] = "FIRST_SEPARATE_VAE_PHASE"
        store.save()
        case["phase_receipts"].append(backend.to_vae_phase("FIRST_LIFT_VAE"))
        store.save()
        off_rgb = None
        try:
            case["stage"] = "OFF_DECODE_SAVE_SCORE"
            off_rgb = backend.decode(backend.off_terminal)
            off_received = _save_and_score(store, case_id, "OFF", off_rgb, key, config,
                                           encode_fn=encode_fn, score_fn=score_fn)
        except Exception as exc:
            _fail_arm(store, case_id, "OFF", case["stage"], exc)
        for arm, base_index in (("SINGLE49", 49), ("SINGLE46", 46), ("MULTI44_46", 44)):
            try:
                if base_index == 49:
                    if off_rgb is None:
                        raise RuntimeError("normal T49 X0 decode unavailable")
                    base_rgb = off_rgb
                else:
                    case["stage"] = f"{arm}_EARLY_CLEAN_PROXY_DECODE"
                    store.save()
                    base_rgb = backend.decode(backend.clean_proxy(base_index))
                direction = _make_lift(store, case_id, arm, base_rgb, key, backend,
                                       base_index)
                if direction is not None:
                    directions[arm] = direction
                if base_index != 49:
                    del base_rgb
            except Exception as exc:
                _fail_arm(store, case_id, arm, case["stage"], exc)
        del off_rgb
        case["stage"] = "FIRST_TRANSFORMER_RESTORE"
        store.save()
        case["phase_receipts"].append(backend.to_transformer_phase("FIRST_RESUME"))
        for index in (44, 46):
            case["cfg_restoration_checks"].append(backend.verify_cfg(index))
            store.save()

        if "SINGLE49" in directions:
            terminal = control_history = None
            try:
                case["stage"] = "SINGLE49_CONTROL"
                store.save()
                state, terminal, control_history, control = backend.control_from_off(
                    49, directions.pop("SINGLE49"), R_STAR)
                case["controls"]["SINGLE49"] = [control]
                store.save()
                if state == "READY":
                    backend.terminals["SINGLE49"] = terminal.detach().cpu()
                    case["arm_metrics"]["SINGLE49"] = _arm_metrics(
                        backend, backend.terminals["SINGLE49"], [control])
                else:
                    _method_negative(store, case_id, "SINGLE49", state)
            except Exception as exc:
                _fail_arm(store, case_id, "SINGLE49", case["stage"], exc)
            finally:
                terminal = control_history = None

        if "SINGLE46" in directions:
            z47 = scheduler = terminal = None
            try:
                case["stage"] = "SINGLE46_CONTROL_AND_FREE_CONTINUATION"
                store.save()
                state, z47, scheduler, control = backend.control_from_off(
                    46, directions.pop("SINGLE46"), R_STAR)
                case["controls"]["SINGLE46"] = [control]
                store.save()
                if state == "READY":
                    terminal = backend.continue_normal(z47, scheduler, 47, 50)
                    backend.terminals["SINGLE46"] = terminal
                    case["arm_metrics"]["SINGLE46"] = _arm_metrics(backend, terminal, [control])
                else:
                    _method_negative(store, case_id, "SINGLE46", state)
            except Exception as exc:
                _fail_arm(store, case_id, "SINGLE46", case["stage"], exc)
            finally:
                z47 = scheduler = terminal = None

        if "MULTI44_46" in directions:
            try:
                case["stage"] = "MULTI44_CONTROL_AND_LIVE_T46"
                store.save()
                state, control44 = backend.multi44_to_46(
                    directions.pop("MULTI44_46"), R_STAR / 2)
                case["controls"]["MULTI44_46"] = [control44]
                store.save()
                if state != "READY":
                    _method_negative(store, case_id, "MULTI44_46", state)
                else:
                    case["stage"] = "MULTI46_SEPARATE_VAE_PHASE"
                    store.save()
                    case["phase_receipts"].append(backend.to_vae_phase("MULTI46_LIFT_VAE"))
                    _score_terminals(store, case_id, config, backend, key,
                                     off_received, ("SINGLE49", "SINGLE46"),
                                     encode_fn, score_fn)
                    base46 = backend.decode(backend.multi_clean_proxy())
                    direction46 = _make_lift(store, case_id, "MULTI44_46", base46,
                                              key, backend, 46)
                    del base46
                    case["stage"] = "MULTI46_TRANSFORMER_RESTORE"
                    store.save()
                    case["phase_receipts"].append(backend.to_transformer_phase("MULTI46_RESUME"))
                    case["cfg_restoration_checks"].append(backend.verify_cfg(46, multi=True))
                    store.save()
                    if direction46 is not None:
                        z47 = scheduler = terminal = None
                        case["stage"] = "MULTI46_CONTROL_AND_FREE_CONTINUATION"
                        store.save()
                        state46, z47, scheduler, control46 = backend.control_multi46(
                            direction46, R_STAR / 2)
                        case["controls"]["MULTI44_46"].append(control46)
                        store.save()
                        if state46 == "READY":
                            terminal = backend.continue_normal(z47, scheduler, 47, 50)
                            backend.terminals["MULTI44_46"] = terminal
                            case["arm_metrics"]["MULTI44_46"] = _arm_metrics(
                                backend, terminal, case["controls"]["MULTI44_46"])
                        else:
                            _method_negative(store, case_id, "MULTI44_46", state46)
                        z47 = scheduler = terminal = None
            except Exception as exc:
                _fail_arm(store, case_id, "MULTI44_46", case["stage"], exc)
                z47 = scheduler = terminal = None

        case["stage"] = "FINAL_SEPARATE_VAE_PHASE"
        store.save()
        if backend.phase == "TRANSFORMER":
            case["phase_receipts"].append(backend.to_vae_phase("FINAL_MEDIA_VAE"))
        elif backend.phase != "VAE":
            raise RuntimeError("no viable VAE phase for remaining terminal media")
        _score_terminals(store, case_id, config, backend, key, off_received,
                         ARMS[1:], encode_fn, score_fn)
    except Exception as exc:
        case["failures"].append(dict(stage=case["stage"],
                                     error=f"{type(exc).__name__}: {exc}",
                                     traceback=traceback.format_exc()))
        _mark_pending(store, case_id, "NOT_RUN_ENGINEERING_INVALID",
                      f"{type(exc).__name__}: {exc}")
    finally:
        case["elapsed_seconds"] = time.perf_counter() - start
        try:
            case["resources"] = backend.resources()
            case["resources"]["cpu_peak_rss_kib"] = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss
            case["resources"]["wall_elapsed_seconds"] = case["elapsed_seconds"]
        except Exception as exc:
            case["resources"] = {"status": "UNAVAILABLE", "reason": repr(exc)}
        try:
            backend.release()
        except Exception as exc:
            case["failures"].append(dict(stage="RELEASE", error=repr(exc)))
        _finalize_case(store, case_id)
        store.active_case = None
        gc.collect()


def finalize_experiment(store: Store) -> None:
    expected = {case_id: {arm: ("H0" if arm == "OFF" else "H1")
                          for arm in store.case(case_id)["slots"]}
                for case_id in CASE_IDS}
    decisions = {case_id: {arm: slot["decision"]
                           for arm, slot in store.case(case_id)["slots"].items()}
                 for case_id in CASE_IDS}
    slots = [(case_id, arm, slot) for case_id in CASE_IDS
             for arm, slot in store.case(case_id)["slots"].items()]
    complete = all(slot["status"] == "SCORED" and slot["decision"] in ("H0", "H1")
                   for _, _, slot in slots) and all(
                       store.case(case_id)["status"] == "SCORED" for case_id in CASE_IDS)
    errors = [dict(case_id=case_id, arm=arm, expected=expected[case_id][arm],
                   observed=slot["decision"], positive_groups=slot["positive_groups"])
              for case_id, arm, slot in slots if slot["status"] == "SCORED"
              and slot["decision"] != expected[case_id][arm]]
    diagnostics = {}
    for case_id in CASE_IDS:
        slots_by_arm = store.case(case_id)["slots"]
        off_c = slots_by_arm["OFF"]["positive_groups"]
        diagnostics[case_id] = {
            arm: dict(c=slots_by_arm[arm]["positive_groups"],
                      c_minus_off=(slots_by_arm[arm]["positive_groups"] - off_c
                                   if slots_by_arm[arm]["positive_groups"] is not None
                                   and off_c is not None else None),
                      quality_vs_off=slots_by_arm[arm]["quality_vs_off"])
            for arm in ARMS[1:]}
    store.data["reporting"] = dict(
        joined_after_decision_persistence=True,
        expected=expected, decisions=decisions, observed_misclassifications=errors,
        same_source_comparison=diagnostics)
    store.data["status"] = (
        "FIXED_MULTISTEP_COMPARISON_COMPLETE" if complete and not errors
        else "FIXED_MULTISTEP_COMPARISON_NEGATIVE" if complete else "INCOMPLETE")
    store.save()


def _source_sha() -> str:
    value = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("source Git SHA unavailable")
    return value


def _validate_output(output: Path) -> None:
    if output.parent != OUTPUT_PARENT or not re.fullmatch(r"\d{8}T\d{12}Z", output.name):
        raise ValueError("new timestamp output under fixed Drive parent required")
    if output.exists():
        if not (output / "setup_receipt.json").is_file() or (output / "result.json").exists():
            raise FileExistsError("output directory is not fresh notebook setup")
    else:
        if not OUTPUT_PARENT.is_dir():
            raise FileNotFoundError("Drive output parent absent")
        output.mkdir(exist_ok=False)


def _run_worker(output: Path, config: dict, case_id: str) -> str | None:
    env = os.environ.copy()
    env["RGB_DCT_MULTISTEP_INTERNAL_CASE"] = case_id
    command = [sys.executable, "-u", "-m", MODULE, "--output", str(output)]
    worker_log = output / f"{case_id}_worker.log"
    print("case start:", case_id, "log:", worker_log, flush=True)
    with worker_log.open("w", encoding="utf-8") as stream:
        try:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=config["resources"]["case_timeout_seconds"])
                failure = None if code == 0 else f"WORKER_EXIT_{code}"
                if failure is not None:
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                failure = "WORKER_TIMEOUT"
        except Exception as exc:
            failure = f"WORKER_START_FAILED:{type(exc).__name__}:{exc}"
    return failure


def _supervise(output: Path, config: dict, source_sha: str,
               *, worker_fn: Callable = _run_worker) -> None:
    result_path = output / "result.json"
    store = Store(result_path, initial_result(config, output, source_sha))
    store.save()  # Two sources/eight slots/1448 planned frames before model work.
    for case_id in CASE_IDS:
        failure = worker_fn(output, config, case_id)
        store = Store.open(result_path)
        if failure is not None:
            case = store.case(case_id)
            case["failures"].append(dict(stage="WORKER", error=failure,
                                         log_path=str(output / f"{case_id}_worker.log")))
            _mark_pending(store, case_id,
                          "NOT_RUN_RESOURCE_FAILURE" if failure == "WORKER_TIMEOUT"
                          else "NOT_RUN_WORKER_FAILURE", failure)
            _finalize_case(store, case_id)
        store.case(case_id)["worker_log_path"] = str(output / f"{case_id}_worker.log")
        store.save()
        print("case complete:", case_id, store.case(case_id)["status"], flush=True)
    finalize_experiment(store)
    for case_id in CASE_IDS:
        for arm, row in store.case(case_id)["slots"].items():
            print(case_id, arm, row["status"], row["score"], row["decision"], flush=True)
    print("result:", result_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    output = args.output
    internal_case = os.environ.get("RGB_DCT_MULTISTEP_INTERNAL_CASE")
    if internal_case:
        if internal_case not in CASE_IDS or not (output / "result.json").is_file():
            raise ValueError("invalid fixed worker invocation")
        case_config = next(row for row in config["cases"] if row["id"] == internal_case)
        runtime_config = copy.deepcopy(config)
        runtime_config["generation"].update(prompt=case_config["prompt"],
                                            seed=case_config["seed"])
        store = Store.open(output / "result.json")
        if store.data["config_sha256"] != _sha(CONFIG_PATH) or store.data["source_sha"] != _source_sha():
            raise ValueError("worker source/config binding changed")
        if store.case(internal_case)["status"] != "PENDING":
            raise ValueError("fixed source already attempted; no worker retry")
        from runtime.wan.rgb_dct_multistep_backend import WanMultiBackend
        run_case(store, internal_case, config,
                 WanMultiBackend(runtime_config, store.count))
        return
    _validate_output(output)
    _supervise(output, config, _source_sha())


if __name__ == "__main__":
    main()
