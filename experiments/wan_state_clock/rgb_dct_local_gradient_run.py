"""Fixed two-source OFF/SINGLE49/LOCAL44_46_48 RGB-DCT trial."""
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
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable

import numpy as np

from main.tube_state import rgb_dct_group_consistency as receiver
from main.tube_state.rgb_dct_t49_carrier import RGB_SHAPE, apply_carrier, lift_direction
from runtime.wan.rgb_dct_local_gradient_adapter import read_original_mp4, score_original_rgb

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "configs/rgb_dct_local_gradient_v1.json"
PROTOCOL_PATH = ROOT / "docs/rgb_dct_local_gradient_v1_protocol_20260924.md"
MODULE = "experiments.wan_state_clock.rgb_dct_local_gradient_run"
OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/RGB-DCT-Local-Gradient-V1")
CASE_IDS = ("eval_clock_s2431", "eval_umbrella_s2432")
ARMS = ("OFF", "SINGLE49", "LOCAL44_46_48")
PROMPTS = (
    "locked camera, a round wall clock with black hands on a plain white wall, steady indoor light, no people, no cuts",
    "locked camera, a yellow umbrella turning slowly against a plain gray background, steady soft light, no people, no cuts",
)
SEEDS = (2026092431, 2026092432)
R_STAR = 0.042943312697648145
PLAN_PER_CASE = dict(generation=1, transformer=112, transformer_validation=6,
                     scheduler_step=57, shadow_step=4,
                     unit_response_probe_step=4, vae_decode=6, vae_encode=2,
                     backward=3, vae_chunk_forward=138,
                     vae_chunk_recompute=138,
                     mp4_save=3, mp4_read=3, score=3)
PLAN = {name: value * 2 for name, value in PLAN_PER_CASE.items()}
TOTAL_NATIVE_STEPS = 130
TOTAL_TRANSFORMER = 236


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if _sha(PROTOCOL_PATH) != config["protocol_sha256"]:
        raise ValueError("adopted protocol byte identity mismatch")
    if (config["receiver_spec_sha256"] != receiver.SPEC_SHA256
            or config["baseline_receiver_spec_sha256"] != receiver.baseline.SPEC_SHA256):
        raise ValueError("RGB-DCT receiver specification changed")
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
        raise ValueError("fixed two-source roster changed")
    if config["generation"] != dict(
            height=320, width=512, frames=181, fps=8, steps=50,
            guidance_scale=5.0, max_sequence_length=512,
            negative_prompt="text, watermark, logo, camera motion, cuts, multiple objects, flicker"):
        raise ValueError("fixed Wan generation changed")
    if config["media"] != dict(fps=8, crf=18, codec="libx264",
                               pixel_format="yuv420p", receiver_readback="ffmpeg_rgb24"):
        raise ValueError("fixed media contract changed")
    expected_control = dict(
        local_indices=[44, 46, 48], single_reference_index=49,
        target_D_support_rms=R_STAR,
        local_each_target_D_support_rms=R_STAR / 3,
        latent_support_time=[1, 45], loss="mean(relu(-q_g)^2)",
        proxy_q_atol=1e-9, proxy_q_rtol=1e-8,
        no_scan_retry_tuning_or_fallback=True)
    if config["control"] != expected_control:
        raise ValueError("fixed local gradient mechanism changed")
    if config["checkpoint"] != dict(
            kind="native causal-frame nonreentrant checkpoint with CPU boundary packing",
            latent_frames_per_decode=46, gradient_decodes_per_source=3,
            vae_chunk_forward_max_per_source=138,
            vae_chunk_recompute_max_per_source=138, spatial_tiling=False):
        raise ValueError("fixed VAE checkpoint plan changed")
    if config["runtime"] != dict(torch_base_version="2.11.0",
                                 diffusers_version="0.40.0",
                                 cuda_required=True,
                                 cuda_build_suffix_gate=False):
        raise ValueError("fixed runtime requirement changed")
    if config["decision_rule"] != dict(statistic="C=sum(q_g>0)", threshold=24,
                                      rule="C>=24:H1;C<24:H0",
                                      reference_calibration=False):
        raise ValueError("fixed C>=24 decision rule changed")
    if config["fixed_denominator"] != dict(sources=2, mp4_score_slots=6, frames=1086):
        raise ValueError("fixed denominator changed")
    if config["call_plan_max"] != PLAN | {
            "native_scheduler_total": TOTAL_NATIVE_STEPS,
            "transformer_total": TOTAL_TRANSFORMER}:
        raise ValueError("fixed total call plan changed")
    if config.get("per_source_call_plan_max") != PLAN_PER_CASE | {
            "native_scheduler_total": TOTAL_NATIVE_STEPS // 2,
            "transformer_total": TOTAL_TRANSFORMER // 2}:
        raise ValueError("fixed per-source call plan changed")
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
                positive_groups=None, c_margin=None, decision=None,
                quality_vs_off=None, attempted=False, expected_frames=181,
                frames_used=0,
                path=str(case_dir / "received_videos" / arm / "FULL.mp4"),
                sha256=None, bytes=None, receiver_status=None,
                receiver_output=None, clipping=None, raw_failures=[])


def initial_result(config: dict, output: Path, source_sha: str) -> dict:
    cases = {}
    for case in config["cases"]:
        case_id = case["id"]
        cases[case_id] = dict(
            status="PENDING", stage="PENDING", prompt=case["prompt"], seed=case["seed"],
            slots={arm: _slot(output / case_id, arm) for arm in ARMS},
            environment=None, generation=None, phase_receipts=[], controls={},
            proxy_gradients=[], live_nodes=[], arm_metrics={},
            cfg_restoration_checks=[],
            calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
            resources={}, elapsed_seconds=None, failures=[])
    return dict(
        status="RUNNING", experiment_id=config["name"], source_sha=source_sha,
        config_path=str(CONFIG_PATH), config_sha256=_sha(CONFIG_PATH),
        protocol_path=str(PROTOCOL_PATH), protocol_sha256=config["protocol_sha256"],
        receiver_baseline_source_sha=config["receiver_baseline_source_sha"],
        receiver_spec_sha256=config["receiver_spec_sha256"],
        baseline_receiver_spec_sha256=config["baseline_receiver_spec_sha256"],
        key_id=config["receiver_key_id"], key_utf8=config["key_utf8"],
        model=config["model"], runtime_config=config["runtime"],
        generation_config=config["generation"], media_config=config["media"],
        control_config=config["control"], decision_rule=config["decision_rule"],
        checkpoint_config=config["checkpoint"],
        fixed_denominator=config["fixed_denominator"],
        per_source_call_plan_max=config["per_source_call_plan_max"],
        call_plan_max=config["call_plan_max"], resources_config=config["resources"],
        call_completion_definition={
            "backward": "completed only after a finite autograd gradient is returned",
            "mp4_save": "completed only after a nonempty MP4 path and SHA-256 are retained",
            "mp4_read": "completed only after a valid full finite RGB24 readback",
            "score": "completed only for a valid full-media C score"},
        calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
        attempted_media_slots=0, scored_media_slots=0, invalid_media_slots=0,
        pending_media_slots=6, scored_frames=0, reporting=None, cases=cases,
        output_dir=str(output), result_path=str(output / "result.json"),
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
        if len(slots) != 6:
            raise RuntimeError("fixed six-slot denominator changed")
        statuses = [slot["status"] for slot in slots]
        self.data["attempted_media_slots"] = sum(bool(slot["attempted"]) for slot in slots)
        self.data["scored_media_slots"] = statuses.count("SCORED")
        self.data["pending_media_slots"] = statuses.count("PENDING")
        self.data["invalid_media_slots"] = 6 - self.data["scored_media_slots"] - self.data["pending_media_slots"]
        self.data["scored_frames"] = sum(slot["frames_used"] for slot in slots
                                          if slot["status"] == "SCORED")
        self.data["native_scheduler_calls"] = {
            name: self.data["calls"]["scheduler_step"][name]
            + self.data["calls"]["shadow_step"][name]
            + self.data["calls"]["unit_response_probe_step"][name]
            for name in ("attempted", "completed")}
        self.data["transformer_calls"] = {
            name: self.data["calls"]["transformer"][name]
            + self.data["calls"]["transformer_validation"][name]
            for name in ("attempted", "completed")}
        if self.data["native_scheduler_calls"]["attempted"] > TOTAL_NATIVE_STEPS:
            raise RuntimeError("native scheduler call cap exceeded")
        if self.data["transformer_calls"]["attempted"] > TOTAL_TRANSFORMER:
            raise RuntimeError("Transformer call cap exceeded")
        _atomic_json(self.path, self.data)

    def count(self, kind: str, completed: bool) -> None:
        if kind not in PLAN:
            raise ValueError("unplanned call kind")
        field = "completed" if completed else "attempted"
        row = self.data["calls"][kind]
        row[field] += 1
        if row["attempted"] > PLAN[kind] or row["completed"] > row["attempted"]:
            raise RuntimeError(f"fixed {kind} total cap/order exceeded")
        if self.active_case is not None:
            case_row = self.case(self.active_case)["calls"][kind]
            case_row[field] += 1
            if (case_row["attempted"] > PLAN_PER_CASE[kind]
                    or case_row["completed"] > case_row["attempted"]):
                raise RuntimeError(f"fixed per-source {kind} cap/order exceeded")
        self.save()


def _runtime_environment(config: dict) -> dict:
    import diffusers
    import torch
    from diffusers import AutoencoderKLWan, WanPipeline

    receipt = dict(python=sys.version, executable=sys.executable,
                   torch=str(torch.__version__), torch_cuda_runtime=torch.version.cuda,
                   diffusers=str(diffusers.__version__), cuda_available=torch.cuda.is_available(),
                   cuda_device=(torch.cuda.get_device_name(0)
                                if torch.cuda.is_available() else None),
                   required_api=dict(autograd_grad=callable(torch.autograd.grad),
                                     wan_pipeline=WanPipeline is not None,
                                     wan_vae=AutoencoderKLWan is not None))
    expected = config["runtime"]
    if receipt["torch"].split("+", 1)[0] != expected["torch_base_version"]:
        raise RuntimeError("fixed torch base version required; CUDA build suffix is unrestricted")
    if receipt["diffusers"] != expected["diffusers_version"]:
        raise RuntimeError("fixed diffusers version required")
    if not receipt["cuda_available"] or not all(receipt["required_api"].values()):
        raise RuntimeError("CUDA and required Wan/autograd APIs are unavailable")
    return receipt


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


def _paired_mp4_quality(off: np.ndarray, marked: np.ndarray) -> dict:
    if (not isinstance(off, np.ndarray) or not isinstance(marked, np.ndarray)
            or off.shape != RGB_SHAPE or marked.shape != RGB_SHAPE):
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


def _json_error_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_json_error_value(item) for item in value]
    return repr(value)


def _exception_record(stage: str, exc: Exception) -> dict:
    record = dict(stage=stage, exception_class=type(exc).__name__,
                  message=str(exc), traceback=traceback.format_exc())
    for source, target in (("cmd", "command"), ("returncode", "return_code"),
                           ("stdout", "stdout"), ("stderr", "stderr"),
                           ("output", "output")):
        if hasattr(exc, source):
            record[target] = _json_error_value(getattr(exc, source))
    return record


def _save_and_score(store: Store, case_id: str, arm: str, rgb: np.ndarray,
                    key: bytes, config: dict, *, encode_fn: Callable,
                    read_fn: Callable, receiver_fn: Callable,
                    clipping: dict | None = None,
                    off_received: np.ndarray | None = None) -> np.ndarray | None:
    slot = store.case(case_id)["slots"][arm]
    path = Path(slot["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    slot.update(status="PREPARING", attempted=True, clipping=clipping)
    store.save()
    failure_stage = "RGB_VALIDATION"
    received = None
    try:
        _validate_rgb_for_mp4(rgb)
        failure_stage = "MP4_ENCODE"
        store.count("mp4_save", False)
        encode_fn(rgb, path, config["media"]["fps"], config["media"]["crf"])
        failure_stage = "MP4_IDENTITY"
        slot.update(sha256=_sha(path), bytes=path.stat().st_size)
        if slot["bytes"] <= 0:
            raise ValueError("empty MP4")
        store.count("mp4_save", True)
        store.save()
        failure_stage = "MP4_RGB24_READ"
        store.count("mp4_read", False)
        received = read_fn(path)
        _validate_rgb_for_mp4(received)
        store.count("mp4_read", True)
        failure_stage = "BLIND_RECEIVER_SCORE"
        store.count("score", False)
        scored = receiver_fn(received, key)
        slot["receiver_output"] = (scored if isinstance(scored, dict)
                                   else {"raw_repr": repr(scored)})
        if not isinstance(scored, dict):
            raise TypeError("blind receiver output must be a dict")
        slot["receiver_status"] = scored.get("status")
        slot["frames_used"] = scored.get("frames_used", 0)
        valid = (scored.get("status") == "SCORED"
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
                 and scored.get("key_id") == config["receiver_key_id"])
        if not valid:
            slot.update(status="INVALID",
                        reason=scored.get("reason") or "RECEIVER_OUTPUT_INVALID")
        else:
            store.count("score", True)
            slot.update(status="SCORED", score=float(scored["score"]),
                        group_scores=[float(q) for q in scored["group_scores"]],
                        positive_groups=scored["positive_groups"],
                        c_margin=scored["positive_groups"] - 24, reason=None)
            store.save()
            slot["decision"] = receiver.decide(slot["positive_groups"])
            if arm != "OFF" and off_received is not None:
                try:
                    slot["quality_vs_off"] = _paired_mp4_quality(off_received, received)
                except Exception as quality_exc:
                    slot["quality_error"] = f"{type(quality_exc).__name__}: {quality_exc}"
                    failure = _exception_record(
                        f"{arm}_PAIRED_MP4_QUALITY", quality_exc) | {"arm": arm}
                    slot["quality_failure"] = failure
                    store.case(case_id)["failures"].append(failure)
    except Exception as exc:
        slot.update(status="ENGINEERING_INVALID", reason=f"{type(exc).__name__}: {exc}",
                    score=None)
        slot["raw_failures"].append(_exception_record(failure_stage, exc))
    store.save()
    return received if slot["status"] == "SCORED" else None


def _fail_arm(store: Store, case_id: str, arm: str, stage: str,
              exc: Exception, status: str = "ENGINEERING_INVALID") -> None:
    case = store.case(case_id)
    failure = _exception_record(stage, exc) | {"arm": arm}
    case["failures"].append(failure)
    slot = case["slots"][arm]
    slot["raw_failures"].append(failure)
    slot["attempted"] = True
    if slot["status"] in ("PENDING", "PREPARING"):
        slot.update(status=status, reason=f"{type(exc).__name__}: {exc}")
    store.save()


def _method_negative(store: Store, case_id: str, arm: str, state: str) -> None:
    store.case(case_id)["slots"][arm].update(status=state,
                                             reason=f"FINITE_{state}", attempted=True)
    store.save()


def _engineering_invalid_state(store: Store, case_id: str, arm: str,
                               state: str, receipt: dict) -> None:
    slot = store.case(case_id)["slots"][arm]
    slot.update(status="ENGINEERING_INVALID", reason=state, attempted=True)
    slot["raw_failures"].append(dict(
        stage=f"{arm}_{state}", exception_class=state,
        message="finite exact-zero local gradient is unusable for native control",
        traceback=None, receipt=receipt))
    store.save()


def _mark_pending(store: Store, case_id: str, status: str, reason: str) -> None:
    for slot in store.case(case_id)["slots"].values():
        if slot["status"] in ("PENDING", "PREPARING"):
            slot.update(status=status, reason=reason, score=None, attempted=True)
    store.save()


def _finalize_case(store: Store, case_id: str) -> None:
    case = store.case(case_id)
    states = [row["status"] for row in case["slots"].values()]
    if all(state == "SCORED" for state in states) and not case["failures"]:
        case["status"] = "SCORED"
    elif case["failures"] or any("INVALID" in state or "FAILURE" in state for state in states):
        case["status"] = "ENGINEERING_INVALID"
    elif any(state in ("ZERO_LIFT_DIRECTION", "ZERO_NATIVE_RESPONSE") for state in states):
        case["status"] = "METHOD_NEGATIVE"
    else:
        case["status"] = "INCOMPLETE"
    store.save()


def _single49_direction(store: Store, case_id: str, off_rgb: np.ndarray,
                        key: bytes, backend) -> np.ndarray | None:
    case = store.case(case_id)
    case["stage"] = "SINGLE49_POSITIVE_FLOAT_RGB_LIFT"
    case["slots"]["SINGLE49"]["attempted"] = True
    store.save()
    plus, clipping = apply_carrier(off_rgb, key, +1)
    encoded_plus = backend.encode(plus)
    encoded_base = backend.encode(off_rgb)
    masked, lift = lift_direction(encoded_plus - encoded_base)
    case["arm_metrics"]["SINGLE49_LIFT"] = dict(status=lift["status"],
                                                 clipping=clipping, geometry=lift)
    store.save()
    if lift["status"] == "ZERO_LIFT_DIRECTION":
        _method_negative(store, case_id, "SINGLE49", "ZERO_LIFT_DIRECTION")
        return None
    if lift["status"] != "READY":
        raise ValueError("unexpected SINGLE49 lift state")
    return masked


def _ensure_final_vae(store: Store, case_id: str, backend) -> None:
    if backend.phase == "TRANSFORMER":
        store.case(case_id)["phase_receipts"].append(
            backend.to_vae_phase("FINAL_MEDIA_VAE"))
        store.save()
    elif backend.phase != "VAE":
        raise RuntimeError("no viable VAE phase for terminal media")


def run_case(store: Store, case_id: str, config: dict, backend,
             *, encode_fn: Callable = _encode_rgb_np,
             read_fn: Callable = read_original_mp4,
             receiver_fn: Callable = score_original_rgb,
             environment_fn: Callable = _runtime_environment) -> None:
    case = store.case(case_id)
    store.active_case = case_id
    key = config["key_utf8"].encode("utf-8")
    case_dir = Path(store.data["output_dir"]) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    off_received = None
    try:
        case.update(status="RUNNING", stage="RUNTIME_ENVIRONMENT")
        store.save()
        case["environment"] = environment_fn(config)
        case["stage"] = "SHARED_OFF_TRAJECTORY"
        store.save()
        case["generation"] = backend.prepare_off(case_dir / "state")
        case["phase_receipts"].append(backend.to_vae_phase("OFF_AND_SINGLE49_VAE"))
        store.save()
        off_rgb = None
        single_direction = None
        try:
            case["stage"] = "OFF_DECODE_SAVE_SCORE"
            off_rgb = backend.decode(backend.off_terminal)
            off_received = _save_and_score(store, case_id, "OFF", off_rgb, key, config,
                                           encode_fn=encode_fn, read_fn=read_fn,
                                           receiver_fn=receiver_fn)
        except Exception as exc:
            _fail_arm(store, case_id, "OFF", case["stage"], exc)
        try:
            if off_rgb is None:
                raise RuntimeError("normal T49 X0 decode unavailable")
            single_direction = _single49_direction(store, case_id, off_rgb, key, backend)
        except Exception as exc:
            _fail_arm(store, case_id, "SINGLE49", case["stage"], exc)
        off_rgb = None
        case["phase_receipts"].append(backend.to_transformer_phase("FIRST_TRANSFORMER_RESUME"))
        store.save()

        if single_direction is not None:
            try:
                case["stage"] = "SINGLE49_NATIVE_CONTROL"
                state, terminal, control = backend.control_single49(single_direction, R_STAR)
                case["controls"]["SINGLE49"] = [control]
                store.save()
                if state == "READY":
                    backend.terminals["SINGLE49"] = terminal.detach().cpu()
                    case["arm_metrics"]["SINGLE49"] = dict(
                        step_indices=[49], step_support_rms=[control["actual_D"]["support_rms"]],
                        net_terminal_from_off=backend.net_from_off(terminal))
                else:
                    _method_negative(store, case_id, "SINGLE49", state)
            except Exception as exc:
                _fail_arm(store, case_id, "SINGLE49", case["stage"], exc)

        try:
            local_slot = case["slots"]["LOCAL44_46_48"]
            local_slot["attempted"] = True
            case["stage"] = "LOCAL_LIVE_T44_CFG"
            case["live_nodes"].append(backend.start_local44())
            store.save()
            local_controls = []
            for position, index in enumerate((44, 46, 48)):
                case["stage"] = f"LOCAL_T{index}_GRADIENT_VAE"
                case["phase_receipts"].append(backend.to_vae_phase(f"LOCAL_T{index}_VAE"))
                store.save()
                state, direction, gradient = backend.local_gradient(key)
                case["proxy_gradients"].append(gradient)
                store.save()
                if state != "READY":
                    if state == "ZERO_LOCAL_GRADIENT":
                        _engineering_invalid_state(
                            store, case_id, "LOCAL44_46_48", state, gradient)
                    else:
                        _method_negative(store, case_id, "LOCAL44_46_48", state)
                    break
                case["stage"] = f"LOCAL_T{index}_TRANSFORMER_RESTORE"
                case["phase_receipts"].append(
                    backend.to_transformer_phase(f"LOCAL_T{index}_RESUME"))
                receipt = backend.verify_local_cfg()
                case["cfg_restoration_checks"].append(receipt)
                store.save()
                case["stage"] = f"LOCAL_T{index}_NATIVE_CONTROL"
                state, control = backend.control_local(direction, R_STAR / 3)
                local_controls.append(control)
                case["controls"]["LOCAL44_46_48"] = local_controls
                store.save()
                if state != "READY":
                    _method_negative(store, case_id, "LOCAL44_46_48", state)
                    break
                if position < 2:
                    target = (46, 48)[position]
                    case["stage"] = f"LOCAL_FREE_TO_T{target}_AND_LIVE_CFG"
                    case["live_nodes"].append(backend.advance_local_to(target))
                    store.save()
                else:
                    terminal = backend.finish_local()
                    supports = [row["actual_D"]["support_rms"] for row in local_controls]
                    case["arm_metrics"]["LOCAL44_46_48"] = dict(
                        step_indices=[44, 46, 48], step_support_rms=supports,
                        sum_support_rms=math.fsum(supports),
                        sum_support_rms_squared=math.fsum(value * value for value in supports),
                        net_terminal_from_off=backend.net_from_off(terminal))
                    store.save()
        except Exception as exc:
            _fail_arm(store, case_id, "LOCAL44_46_48", case["stage"], exc)

        case["stage"] = "FINAL_SEPARATE_VAE_PHASE"
        _ensure_final_vae(store, case_id, backend)
        for arm in ARMS[1:]:
            if arm not in backend.terminals or case["slots"][arm]["status"] != "PENDING":
                continue
            try:
                case["stage"] = f"{arm}_FINAL_DECODE_SAVE_SCORE"
                store.save()
                rgb = backend.decode(backend.terminals[arm])
                _save_and_score(store, case_id, arm, rgb, key, config,
                                encode_fn=encode_fn, read_fn=read_fn,
                                receiver_fn=receiver_fn,
                                off_received=off_received)
                del rgb
            except Exception as exc:
                _fail_arm(store, case_id, arm, case["stage"], exc)
    except Exception as exc:
        case["failures"].append(_exception_record(case["stage"], exc))
        _mark_pending(store, case_id, "NOT_RUN_ENGINEERING_INVALID",
                      f"{type(exc).__name__}: {exc}")
    finally:
        case["elapsed_seconds"] = time.perf_counter() - started
        try:
            case["resources"] = backend.resources()
            case["resources"]["cpu_peak_rss_kib"] = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss
            case["resources"]["wall_elapsed_seconds"] = case["elapsed_seconds"]
        except Exception as exc:
            case["resources"] = {"status": "UNAVAILABLE", "reason": repr(exc)}
            case["failures"].append(_exception_record("RESOURCES", exc))
        try:
            backend.release()
        except Exception as exc:
            case["failures"].append(_exception_record("RELEASE", exc))
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
    complete = (all(slot["status"] == "SCORED" and slot["decision"] in ("H0", "H1")
                    for _, _, slot in slots)
                and all(store.case(case_id)["status"] == "SCORED"
                        and not store.case(case_id)["failures"]
                        for case_id in CASE_IDS))
    errors = [dict(case_id=case_id, arm=arm, expected=expected[case_id][arm],
                   observed=slot["decision"], positive_groups=slot["positive_groups"])
              for case_id, arm, slot in slots if slot["status"] == "SCORED"
              and slot["decision"] != expected[case_id][arm]]
    diagnostics = {}
    for case_id in CASE_IDS:
        rows = store.case(case_id)["slots"]
        off_c = rows["OFF"]["positive_groups"]
        diagnostics[case_id] = {
            arm: dict(c=rows[arm]["positive_groups"],
                      c_minus_off=(rows[arm]["positive_groups"] - off_c
                                   if rows[arm]["positive_groups"] is not None
                                   and off_c is not None else None),
                      quality_vs_off=rows[arm]["quality_vs_off"])
            for arm in ARMS[1:]}
    store.data["reporting"] = dict(joined_after_decision_persistence=True,
                                    expected=expected, decisions=decisions,
                                    observed_misclassifications=errors,
                                    same_source_comparison=diagnostics)
    store.data["status"] = (
        "FIXED_LOCAL_GRADIENT_COMPLETE" if complete and not errors
        else "FIXED_LOCAL_GRADIENT_NEGATIVE" if complete else "INCOMPLETE")
    store.save()


def _source_sha() -> str:
    value = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                    text=True).strip()
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


def _run_worker(output: Path, config: dict, case_id: str) -> dict:
    """Monitor child RSS and retain a parent-owned local spool on hard exit."""
    env = os.environ.copy()
    env["RGB_DCT_LOCAL_GRADIENT_INTERNAL_CASE"] = case_id
    local_root = Path("/content") if Path("/content").is_dir() else Path("/tmp")
    command = [sys.executable, "-u", "-m", MODULE, "--output", str(output)]
    worker_log = output / f"{case_id}_worker.log"
    monitor_path = output / f"{case_id}_worker_monitor.json"
    started = time.monotonic()
    receipt = dict(status="STARTING", error=None, exit_code=None,
                   elapsed_seconds=0.0, parent_observed_peak_rss_kib=None,
                   parent_observed_peak_spool_bytes=0,
                   boundary_spool_cleanup="PENDING", termination=None,
                   timeout_seconds=config["resources"]["case_timeout_seconds"],
                   log_path=str(worker_log))
    spool_owner = None
    child = None
    try:
        spool_owner = tempfile.TemporaryDirectory(
            prefix=f"wan-vae-worker-{case_id}-", dir=local_root)
        env["RGB_DCT_BOUNDARY_SPOOL_ROOT"] = spool_owner.name
        _atomic_json(monitor_path, receipt)
        print("case start:", case_id, "log:", worker_log, flush=True)
        with worker_log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            receipt["status"] = "RUNNING"
            deadline = started + config["resources"]["case_timeout_seconds"]
            while child.poll() is None and time.monotonic() < deadline:
                rss = _proc_rss_kib(child.pid)
                if rss is not None:
                    receipt["parent_observed_peak_rss_kib"] = max(
                        receipt["parent_observed_peak_rss_kib"] or 0, rss)
                receipt["parent_observed_peak_spool_bytes"] = max(
                    receipt["parent_observed_peak_spool_bytes"],
                    _spool_bytes(Path(spool_owner.name)))
                receipt["elapsed_seconds"] = time.monotonic() - started
                _atomic_json(monitor_path, receipt)
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
            if child.poll() is None:
                receipt.update(status="TERMINATING", error="WORKER_TIMEOUT",
                               termination="SIGTERM_SENT")
                _atomic_json(monitor_path, receipt)
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                grace_deadline = time.monotonic() + 10
                while child.poll() is None and time.monotonic() < grace_deadline:
                    rss = _proc_rss_kib(child.pid)
                    if rss is not None:
                        receipt["parent_observed_peak_rss_kib"] = max(
                            receipt["parent_observed_peak_rss_kib"] or 0, rss)
                    receipt["elapsed_seconds"] = time.monotonic() - started
                    _atomic_json(monitor_path, receipt)
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
                    receipt.update(status="FAILED", error=f"WORKER_EXIT_{child.returncode}")
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    except BaseException as exc:
        monitor_error = f"WORKER_MONITOR_FAILED:{type(exc).__name__}:{exc}"
        receipt.update(status="FAILED", error=receipt["error"] or monitor_error,
                       monitor_error=monitor_error)
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            receipt["termination"] = "MONITOR_ABORT_SIGKILL"
        if spool_owner is None:
            receipt["boundary_spool_cleanup"] = "NOT_CREATED"
        else:
            try:
                spool_owner.cleanup()
                receipt["boundary_spool_cleanup"] = "COMPLETED"
            except Exception as exc:
                cleanup_error = f"BOUNDARY_SPOOL_CLEANUP:{type(exc).__name__}:{exc}"
                receipt.update(status="FAILED", boundary_spool_cleanup="FAILED",
                               error=receipt["error"] or cleanup_error,
                               cleanup_error=cleanup_error)
        receipt["elapsed_seconds"] = time.monotonic() - started
        try:
            _atomic_json(monitor_path, receipt)
        except Exception as exc:
            write_error = f"WORKER_MONITOR_FINAL_WRITE:{type(exc).__name__}:{exc}"
            receipt.update(status="FAILED", error=receipt["error"] or write_error,
                           final_monitor_write_error=write_error)
    return receipt


def _proc_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def _spool_bytes(root: Path) -> int:
    total = 0
    for directory, _, files in os.walk(root):
        for name in files:
            try:
                total += (Path(directory) / name).stat().st_size
            except FileNotFoundError:
                pass
    return total


def _supervise(output: Path, config: dict, source_sha: str,
               *, worker_fn: Callable = _run_worker) -> None:
    result_path = output / "result.json"
    store = Store(result_path, initial_result(config, output, source_sha))
    store.save()
    for case_id in CASE_IDS:
        try:
            receipt = worker_fn(output, config, case_id)
        except Exception as exc:
            receipt = dict(status="FAILED",
                           error=f"WORKER_SUPERVISION_FAILED:{type(exc).__name__}:{exc}",
                           exception_class=type(exc).__name__,
                           traceback=traceback.format_exc(),
                           parent_observed_peak_rss_kib=None,
                           parent_observed_peak_spool_bytes=None)
        if isinstance(receipt, dict):
            failure = (receipt.get("error") or receipt.get("status")) if (
                receipt.get("status") != "COMPLETED") else None
        else:
            failure = receipt
            receipt = dict(status="COMPLETED" if failure is None else "FAILED",
                           error=failure, parent_observed_peak_rss_kib=None)
        store = Store.open(result_path)
        store.case(case_id)["worker_receipt"] = receipt
        store.case(case_id).setdefault("resources", {})["parent_observed_peak_rss_kib"] = (
            receipt.get("parent_observed_peak_rss_kib"))
        store.case(case_id)["resources"]["parent_observed_peak_spool_bytes"] = (
            receipt.get("parent_observed_peak_spool_bytes"))
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
    internal_case = os.environ.get("RGB_DCT_LOCAL_GRADIENT_INTERNAL_CASE")
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
        from runtime.wan.rgb_dct_local_gradient_backend import WanLocalGradientBackend
        run_case(store, internal_case, config,
                 WanLocalGradientBackend(runtime_config, store.count))
        return
    _validate_output(output)
    _supervise(output, config, _source_sha())


if __name__ == "__main__":
    main()
