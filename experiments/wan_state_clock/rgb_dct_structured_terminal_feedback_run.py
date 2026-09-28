"""Fixed two-source, five-arm RGB-DCT structured terminal feedback run."""
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
from main.tube_state.rgb_dct_t49_carrier import (
    AMPLITUDE, RGB_SHAPE,
)
from main.tube_state import rgb_dct_structured_feedback as method
from runtime.wan.rgb_dct_multistep_adapter import score_mp4_once
from runtime.wan.rgb_dct_terminal_gradient_backend import _move_scheduler


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "configs/rgb_dct_structured_terminal_feedback_v1.json"
PROTOCOL_PATH = ROOT / "docs/rgb_dct_structured_terminal_feedback_v1_protocol.md"
MODULE = "experiments.wan_state_clock.rgb_dct_structured_terminal_feedback_run"
OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/RGB-DCT-Structured-Terminal-Feedback-V1")
CASE_IDS = ("eval_copperkettle_s2501", "eval_paperplane_s2502")
ARMS = ("OFF", "SINGLE49", "ONLY49", "MULTI46_47_48_49",
        "MULTI46_47_48_FREE49")
PROMPTS = (
    "locked camera, a copper kettle resting on a plain wooden table, steady soft indoor light, no people, no cuts",
    "locked camera, a white paper airplane gliding slowly across a plain blue background, steady soft light, no people, no cuts",
)
SEEDS = (2026092501, 2026092502)
R_STAR = 0.042943312697648145
PLAN = dict(
    generation=2, transformer_prefix=200, scheduler_prefix=100,
    transformer_feedback=400, scheduler_feedback=282,
    vae_decode=122, vae_encode=4,
    unit_response_probe_step=2, scheduler_step=2,
    memory_receiver_score=132,
    mp4_save=10, mp4_read=10, score=10,
)
DERIVED = dict(transformer_max=600, scheduler_max=386,
               vae_decode_max=122, vae_encode_max=4,
               memory_receiver_score_max=132, backward_vjp=0)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if _sha(PROTOCOL_PATH) != config["protocol_sha256"]:
        raise ValueError("structured feedback protocol byte identity mismatch")
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
    if config["control"] != dict(
        points=[46, 47, 48, 49], bins=[[1, 15], [15, 30], [30, 45]],
        latent_support_time=[1, 45], target_D_support_rms=R_STAR,
        total_immediate_rms_cap=R_STAR, point_cap="remaining_total",
        probe_velocity_radius=method.PROBE_RADIUS, margin=method.MARGIN,
        ridge=method.RIDGE, min_true_hinge_improvement=method.MIN_IMPROVEMENT,
        immediate_peak_cap=method.PEAK_CAP,
        lattice_fractions=[0.0, 0.25, 0.5, 0.75, 1.0],
        fallback_scales=list(method.SCALES),
        positive_group_weight=4.0,
    ) or AMPLITUDE != 0.1:
        raise ValueError("fixed writer control changed")
    if config["decision_rule"] != dict(
        statistic="C=sum(q_g>0)", threshold=24,
        rule="C>=24:H1;C<24:H0", mp4_only=True,
    ):
        raise ValueError("fixed MP4 decision changed")
    if config["fixed_denominator"] != dict(
        sources=2, arms_per_source=5, mp4_score_slots=10, frames=1810,
    ):
        raise ValueError("fixed denominator changed")
    if config["call_plan_max"] != PLAN or config["derived_call_totals"] != DERIVED:
        raise ValueError("fixed call plan changed")
    if config["resources"] != dict(
        case_timeout_seconds=28800, worker_processes_per_case=1,
        case_execution="serial fresh child processes",
        termination_grace_seconds=10,
        vae_transformer_separate_phases=True,
        no_gradient_graph_spool=True,
    ):
        raise ValueError("fixed resource plan changed")
    return config


def environment_receipt(config: dict) -> dict:
    """Validate required APIs while recording, not constraining, CUDA suffix."""
    import diffusers
    import torch

    public = str(torch.__version__).split("+", 1)[0]
    suffix = (str(torch.__version__).split("+", 1)[1]
              if "+" in str(torch.__version__) else None)
    expected = config["runtime"]
    if diffusers.__version__ != expected["diffusers_version"]:
        raise RuntimeError(f"diffusers version mismatch: {diffusers.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for fixed Wan execution")
    return dict(
        torch_version=str(torch.__version__), torch_public_version=public,
        torch_version_policy=expected["torch_version_policy"],
        torch_build_suffix=suffix,
        cuda_build_suffix_policy=expected["cuda_build_suffix_policy"],
        torch_cuda_version=torch.version.cuda,
        cuda_available=True, cuda_device_name=torch.cuda.get_device_name(0),
        diffusers_version=diffusers.__version__,
        no_grad_tail=True,
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
                       group_scores=None, positive_groups=None, loss=None,
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
            generation=None, basis=None, budgets={},
            lifts={}, controls={}, phase_receipts=[],
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
        receiver_spec_sha256=config["receiver_spec_sha256"],
        baseline_receiver_spec_sha256=config["baseline_receiver_spec_sha256"],
        key_id=config["receiver_key_id"], key_utf8=config["key_utf8"],
        model=config["model"], runtime_environment={
            case_id: {"status": "PENDING"} for case_id in CASE_IDS
        },
        generation_config=config["generation"], media_config=config["media"],
        control_config=config["control"],
        decision_rule=config["decision_rule"], fixed_denominator=config["fixed_denominator"],
        call_plan_max=config["call_plan_max"], derived_call_totals=DERIVED,
        call_completion_definition={
            "mp4_read": "completed only for one valid full 181-frame RGB24 readback",
            "score": "completed only for a finite frozen-receiver MP4 C score",
            "vae_decode": "completed only for a finite full 181-frame FP32 VAE decode",
        },
        resources_config=config["resources"],
        calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
        attempted_media_slots=0, scored_media_slots=0, invalid_media_slots=0,
        pending_media_slots=10, scored_frames=0, reporting=None,
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
        if len(slots) != 10:
            raise RuntimeError("fixed ten-slot denominator changed")
        statuses = [slot["status"] for slot in slots]
        self.data["scored_media_slots"] = statuses.count("SCORED")
        self.data["pending_media_slots"] = statuses.count("PENDING")
        self.data["invalid_media_slots"] = (
            10 - self.data["scored_media_slots"] - self.data["pending_media_slots"]
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
    valid_q = (
        isinstance(q, list) and len(q) == 30
        and all(isinstance(value, (int, float)) and math.isfinite(value)
                for value in q)
    )
    loss = (float(np.square(np.maximum(-np.asarray(q, dtype=np.float64), 0.0)).mean())
            if valid_q else None)
    return dict(
        status=scored.get("status"), reason=scored.get("reason"),
        score=(float(scored["score"])
               if isinstance(scored.get("score"), (int, float)) else None),
        group_scores=([float(value) for value in q] if valid_q else None),
        positive_groups=scored.get("positive_groups"),
        frames_used=scored.get("frames_used", 0), loss=loss, decision=None,
    )


def _score_memory_layer(rgb: np.ndarray, key: bytes) -> dict:
    try:
        record = _layer_record(receiver.score_rgb(rgb, key))
    except Exception as exc:
        return dict(
            status="INVALID", reason=f"{type(exc).__name__}: {exc}", score=None,
            group_scores=None, positive_groups=None, loss=None,
            frames_used=0, decision=None,
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
        store.count("memory_receiver_score", False)
        slot["layers"]["float_rgb"] = _score_memory_layer(rgb, key)
        store.save()
        _require_scored_layer(slot["layers"]["float_rgb"])
        store.count("memory_receiver_score", True)
        quantized = _quantized_rgb(rgb)
        store.count("memory_receiver_score", False)
        slot["layers"]["rgb8_quantized"] = _score_memory_layer(quantized, key)
        store.save()
        _require_scored_layer(slot["layers"]["rgb8_quantized"])
        store.count("memory_receiver_score", True)
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


def _feedback_arm(store: Store, case_id: str, arm: str, backend,
                  directions: list[np.ndarray], key: bytes) -> None:
    """Continue one independent main trajectory from accepted first-step history."""
    import torch
    from runtime.wan import trajectory

    case = store.case(case_id)
    points = ((49,) if arm == "ONLY49" else
              (46, 47, 48) if arm == "MULTI46_47_48_FREE49" else
              (46, 47, 48, 49))
    z, scheduler, v = backend.state(points[0])
    spent = squared = max_peak = cumulative_peak = 0.0
    cumulative_tensor = torch.zeros_like(z)
    records = []
    case["controls"][arm] = records
    case["budgets"][arm] = dict(status="RUNNING", spent=0.0)
    store.save()
    final = None
    expected_terminal_fp = expected_q = expected_terminal = None
    for index in points:
        case["stage"] = f"{arm}_T{index}_NATIVE_BASELINE_PROBES"
        store.save()
        source_fp = trajectory.fingerprint((z, vars(scheduler), v))
        point = dict(index=index, current_state_fingerprint=source_fp,
                     baseline_history=None, baseline_terminal_fingerprint=None,
                     baseline_rollout_status="ATTEMPTED", probes=[], candidates=[],
                     prediction=None, outcome="PENDING", spent_before=spent)
        records.append(point)
        store.save()
        baseline = backend.rollout(index, z, scheduler, v)
        point.update(baseline_rollout_status="COMPLETED",
                     baseline_history=baseline["source_history_fingerprint"],
                     baseline_terminal_fingerprint=baseline["terminal_fingerprint"])
        store.save()
        if expected_terminal_fp is not None:
            point["continuation_diagnostic"] = dict(
                terminal_fingerprint_match=(baseline["terminal_fingerprint"]
                                            == expected_terminal_fp),
                terminal_delta=trajectory.measures(
                    baseline["terminal"] - expected_terminal),
                diagnostic_only=True)
            store.save()
        sigma = float(scheduler.sigmas[index])
        if not math.isfinite(sigma) or sigma <= 0:
            raise FloatingPointError("nonfinite/nonpositive frozen sigma")
        probes = []
        for j, direction in enumerate(directions):
            point["probes"].append(dict(j=j, rollout_status="ATTEMPTED"))
            store.save()
            delta = torch.as_tensor(-method.PROBE_RADIUS * direction / sigma,
                                    dtype=torch.float32)
            probe = backend.rollout(index, z, scheduler, v, delta,
                                    baseline_next=baseline["next_z"])
            probes.append(probe)
            point["probes"][j].update(
                rollout_status="COMPLETED",
                terminal_fingerprint=probe["terminal_fingerprint"],
                source_history_fingerprint=probe["source_history_fingerprint"],
                final_history_fingerprint=probe["final_history_fingerprint"],
                immediate=probe["immediate"])
            store.save()

        case["stage"] = f"{arm}_T{index}_VAE_BASELINE_PROBES"
        store.save()
        case["phase_receipts"].append(backend.to_vae_phase(f"{arm}_T{index}_PROBE_VAE"))
        store.save()
        baseline_score = backend.score_terminal(baseline["terminal"], key)
        base_q = np.asarray(baseline_score["q"], dtype=np.float64)
        if expected_q is not None:
            q_delta = base_q - expected_q
            point["continuation_diagnostic"].update(
                q_delta=q_delta.tolist(), max_abs_q_delta=float(np.max(np.abs(q_delta))),
                gained_groups=[int(i) for i in np.flatnonzero(
                    (expected_q <= 0) & (base_q > 0))],
                lost_groups=[int(i) for i in np.flatnonzero(
                    (expected_q > 0) & (base_q <= 0))])
            store.save()
        point["baseline"] = dict(**baseline_score,
            weighted_hinge=method.weighted_hinge(base_q, base_q))
        store.save()
        jacobian = np.zeros((30, 3), dtype=np.float64)
        available = []
        for j, probe in enumerate(probes):
            scored = backend.score_terminal(probe["terminal"], key)
            response = probe["immediate"]
            probe_rms = response["support_rms"]
            if (not all(math.isfinite(response[field]) for field in
                        ("support_rms", "global_rms", "peak_abs"))
                    or any(response[field] < 0 for field in
                           ("support_rms", "global_rms", "peak_abs"))):
                raise FloatingPointError("invalid native probe response")
            if probe_rms:
                jacobian[:, j] = (np.asarray(scored["q"]) - base_q) / probe_rms
            available.append(bool(probe_rms))
            probe_q = np.asarray(scored["q"], dtype=np.float64)
            point["probes"][j].update(dict(q=scored["q"], C=scored["C"],
                weighted_hinge=method.weighted_hinge(probe_q, base_q),
                gained_groups=[int(i) for i in np.flatnonzero((base_q <= 0) & (probe_q > 0))],
                lost_groups=[int(i) for i in np.flatnonzero((base_q > 0) & (probe_q <= 0))],
                immediate=response, finite_difference_denominator=probe_rms,
                status="READY" if probe_rms else "ZERO_NATIVE_RESPONSE",
                terminal_fingerprint=probe["terminal_fingerprint"],
                source_history_fingerprint=probe["source_history_fingerprint"],
                final_history_fingerprint=probe["final_history_fingerprint"]))
            store.save()
        case["phase_receipts"].append(backend.to_transformer_phase(
            f"{arm}_T{index}_PROBE_TRANSFORMER_RESUME"))
        if trajectory.fingerprint((z, vars(scheduler), v)) != source_fp:
            raise RuntimeError("main state/history changed across probe VAE phase")
        store.save()

        choice = method.select_coefficients(base_q, jacobian,
                                             method.R_STAR - spent, tuple(available))
        point["jacobian"] = jacobian.tolist()
        point["prediction"] = choice
        predicted_q = np.asarray(choice["predicted_q"], dtype=np.float64)
        choice["predicted_hinge"] = method.weighted_hinge(predicted_q, base_q)
        choice["gained_groups"] = [int(i) for i in np.flatnonzero(
            (base_q <= 0) & (predicted_q > 0))]
        choice["lost_groups"] = [int(i) for i in np.flatnonzero(
            (base_q > 0) & (predicted_q <= 0))]
        point["probe_positive_groups"] = [int(i) for i in np.flatnonzero(base_q > 0)]
        store.save()
        coeff = np.asarray(choice["coefficients"], dtype=np.float64)
        candidates = []
        if np.any(coeff > 0):
            for scale in method.SCALES:
                scale_predicted_q = base_q + jacobian @ (coeff * scale)
                row = dict(scale=scale, coefficients=(coeff * scale).tolist(),
                           tail_rollout_status="ATTEMPTED", immediate=None,
                           tail_computed=False, decoded=False,
                           predicted_q=scale_predicted_q.tolist(),
                           predicted_weighted_hinge=method.weighted_hinge(
                               scale_predicted_q, base_q),
                           predicted_regularized_objective=(
                               method.weighted_hinge(scale_predicted_q, base_q)
                               + method.RIDGE * float((coeff * scale) @ (coeff * scale))),
                           predicted_gained_groups=[int(i) for i in np.flatnonzero(
                               (base_q <= 0) & (scale_predicted_q > 0))],
                           predicted_lost_groups=[int(i) for i in np.flatnonzero(
                               (base_q > 0) & (scale_predicted_q <= 0))],
                           q=None, decision="PENDING")
                point["candidates"].append(row)
                store.save()
                combined = np.zeros_like(directions[0])
                for j in range(3):
                    if coeff[j] > 0:
                        combined += (-method.PROBE_RADIUS / sigma
                                     * float(scale * coeff[j] / point["probes"][j]["finite_difference_denominator"])
                                     * directions[j])
                candidate = backend.rollout(
                    index, z, scheduler, v,
                    torch.as_tensor(combined, dtype=torch.float32),
                    baseline_next=baseline["next_z"])
                response = candidate["immediate"]
                if (not all(math.isfinite(response[field]) for field in
                            ("support_rms", "global_rms", "peak_abs"))
                        or any(response[field] < 0 for field in
                               ("support_rms", "global_rms", "peak_abs"))):
                    raise FloatingPointError("invalid true-candidate native response")
                row.update(tail_rollout_status="COMPLETED", tail_computed=True,
                    immediate=response,
                    source_history_fingerprint=candidate["source_history_fingerprint"],
                    next_history_fingerprint=candidate["next_history_fingerprint"],
                    terminal_fingerprint=candidate["terminal_fingerprint"])
                if (response["support_rms"] > method.R_STAR - spent
                        or response["peak_abs"] > method.PEAK_CAP):
                    row["decision"] = ("NATIVE_RMS_BUDGET"
                        if response["support_rms"] > method.R_STAR - spent
                        else "NATIVE_PEAK_CAP")
                candidates.append(candidate)
                store.save()
        if candidates and any(row["decision"] == "PENDING"
                              for row in point["candidates"]):
            case["stage"] = f"{arm}_T{index}_VAE_TRUE_CANDIDATES"
            store.save()
            case["phase_receipts"].append(backend.to_vae_phase(
                f"{arm}_T{index}_CANDIDATE_VAE"))
            store.save()
            first_accepted = False
            for row, candidate in zip(point["candidates"], candidates, strict=True):
                if row["decision"] != "PENDING":
                    continue
                if first_accepted:
                    row["decision"] = "TAIL_COMPUTED_NOT_SCORED_AFTER_ACCEPT"
                    store.save()
                    continue
                scored = backend.score_terminal(candidate["terminal"], key)
                row["decoded"] = True
                row["q"] = scored["q"]
                row["C"] = scored["C"]
                verdict = method.judge_candidate(
                    base_q, np.asarray(scored["q"]),
                    row["immediate"]["support_rms"],
                    row["immediate"]["peak_abs"], spent)
                row.update(verdict)
                row["decision"] = verdict["reason"]
                first_accepted = verdict["accepted"]
                store.save()
            case["phase_receipts"].append(backend.to_transformer_phase(
                f"{arm}_T{index}_CANDIDATE_TRANSFORMER_RESUME"))
            if trajectory.fingerprint((z, vars(scheduler), v)) != source_fp:
                raise RuntimeError("main state/history changed across candidate VAE phase")
            store.save()

        accepted_index = next((j for j, row in enumerate(point["candidates"])
                               if row["decision"] == "ACCEPT"), None)
        if accepted_index is None:
            chosen = baseline
            expected_q = base_q
            point["outcome"] = ("ZERO_OPTIMUM" if not candidates else
                                "NO_ACCEPTABLE_CANDIDATE")
        else:
            chosen = candidates[accepted_index]
            expected_q = np.asarray(point["candidates"][accepted_index]["q"],
                                    dtype=np.float64)
            response = chosen["immediate"]
            spent += response["support_rms"]
            squared += response["support_rms"] ** 2
            max_peak = max(max_peak, response["peak_abs"])
            cumulative_tensor += chosen["immediate_tensor"]
            cumulative_peak = max(cumulative_peak,
                                  float(cumulative_tensor.abs().max()))
            point["outcome"] = "ACCEPT"
            point["accepted_scale"] = point["candidates"][accepted_index]["scale"]
        point["spent_after"] = spent
        expected_terminal_fp = chosen["terminal_fingerprint"]
        expected_terminal = chosen["terminal"]
        point["committed_next_history_fingerprint"] = chosen["next_history_fingerprint"]
        case["budgets"][arm] = dict(status="RUNNING", spent=spent,
            squared_immediate_rms_sum=squared, max_single_step_peak=max_peak,
            max_running_response_sum_peak=cumulative_peak)
        store.save()
        z, scheduler = chosen["next_z"], chosen["next_scheduler"]
        if (trajectory.fingerprint(vars(scheduler))
                != point["committed_next_history_fingerprint"]):
            raise RuntimeError("committed first-step history identity changed")
        final = chosen["terminal"]
        if index < 49:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            v = backend.velocity(index + 1, z.to(device),
                                 _move_scheduler(copy.deepcopy(scheduler), device))
            v = v.detach().cpu()
    if arm == "MULTI46_47_48_FREE49":
        case["stage"] = f"{arm}_FREE_T49"
        store.save()
        free = backend.rollout(49, z, scheduler, v)
        final = free["terminal"]
        case["controls"][arm].append(dict(index=49, outcome="FREE49",
            source_history_fingerprint=free["source_history_fingerprint"],
            final_history_fingerprint=free["final_history_fingerprint"]))
        store.save()
    backend.terminals[arm] = final
    terminal_delta = trajectory.measures(final - backend.off_terminal)
    case["budgets"][arm] = dict(status="COMPLETE", spent=spent,
        squared_immediate_rms_sum=squared, max_single_step_peak=max_peak,
        max_running_response_sum_peak=cumulative_peak,
        final_terminal_delta=terminal_delta,
        final_terminal_peak_abs=terminal_delta["peak_abs"])
    store.save()


def run_case(store: Store, case_id: str, config: dict, backend,
             *, encode_fn=_encode_rgb_np, score_fn=score_mp4_once) -> None:
    """Run five same-source arms, persisting every stage and rejection."""
    case = store.case(case_id)
    store.active_case = case_id
    key = config["key_utf8"].encode()
    case_dir = Path(store.data["output_dir"]) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    off_received = None
    try:
        case.update(status="RUNNING", stage="SHARED_OFF_TRAJECTORY")
        store.save()
        case["generation"] = backend.prepare_off(case_dir / "state")
        case["stage"] = "INITIAL_FP32_VAE_BASIS"
        store.save()
        case["phase_receipts"].append(backend.to_vae_phase("INITIAL_BASIS_VAE"))
        store.save()
        off_rgb, directions, single_direction, case["basis"] = backend.build_basis(key)
        case["lifts"]["SINGLE49"] = dict(
            index=49, clipping=case["basis"]["clipping"],
            geometry=case["basis"]["single_geometry"])
        store.save()
        case["stage"] = "OFF_MP4_SAVE_READ_SCORE"
        store.save()
        off_received = _save_and_score(
            store, case_id, "OFF", off_rgb, key, config,
            encode_fn=encode_fn, score_fn=score_fn)
        del off_rgb
        case["stage"] = "TRANSFORMER_RESTORE"
        store.save()
        case["phase_receipts"].append(
            backend.to_transformer_phase("INITIAL_CONTROL_RESUME"))
        store.save()
        try:
            case["stage"] = "SINGLE49_NATIVE_CONTROL"
            store.save()
            case["slots"]["SINGLE49"]["attempted"] = True
            if case["basis"]["single_geometry"]["status"] == "ZERO_LIFT_DIRECTION":
                _no_media_result(store, case_id, "SINGLE49", "ZERO_LIFT_DIRECTION")
            else:
                state, _, control = backend.control_single49(single_direction, R_STAR)
                case["controls"]["SINGLE49"] = control
                store.save()
                if state != "READY":
                    _no_media_result(store, case_id, "SINGLE49", state)
        except Exception as exc:
            _fail_arm(store, case_id, "SINGLE49", case["stage"], exc)
        del single_direction
        for arm in ("ONLY49", "MULTI46_47_48_49", "MULTI46_47_48_FREE49"):
            try:
                case["slots"][arm]["attempted"] = True
                case["stage"] = f"{arm}_CONTROL"
                store.save()
                _feedback_arm(store, case_id, arm, backend, directions, key)
            except Exception as exc:
                _fail_arm(store, case_id, arm, case["stage"], exc)
                if backend.phase == "VAE":
                    case["phase_receipts"].append(
                        backend.to_transformer_phase(f"{arm}_ERROR_RESUME"))
                    store.save()
        del directions
        case["stage"] = "FINAL_FP32_VAE_MEDIA"
        store.save()
        case["phase_receipts"].append(backend.to_vae_phase("FINAL_MEDIA_VAE"))
        store.save()
        for arm in ARMS[1:]:
            if case["slots"][arm]["status"] != "PENDING" or arm not in backend.terminals:
                continue
            try:
                case["stage"] = f"{arm}_DECODE_SAVE_READ_SCORE"
                store.save()
                rgb = backend.decode(backend.terminals[arm])
                _save_and_score(store, case_id, arm, rgb, key, config,
                                encode_fn=encode_fn, score_fn=score_fn,
                                off_received=off_received)
                del rgb
            except Exception as exc:
                _fail_arm(store, case_id, arm, case["stage"], exc)
    except Exception as exc:
        case["failures"].append(dict(
            stage=case["stage"], error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc()))
        _mark_pending(store, case_id, "NOT_RUN_ENGINEERING_INVALID",
                      f"{type(exc).__name__}: {exc}")
    finally:
        case["elapsed_seconds"] = time.perf_counter() - started
        try:
            case["resources"] = backend.resources()
            case["resources"].update(
                cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                wall_elapsed_seconds=case["elapsed_seconds"])
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
    gate_arms = ("OFF", "SINGLE49", "MULTI46_47_48_49")
    expected = {case_id: {arm: ("H0" if arm == "OFF" else "H1")
                          for arm in gate_arms}
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
        if (arm in gate_arms and slot["status"] == "SCORED"
            and slot["decision"] != expected[case_id][arm])
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
        comparison_not_a_pass_gate="ONLY49",
        ablation_not_a_pass_gate="MULTI46_47_48_FREE49",
        main_arm="MULTI46_47_48_49",
        control_arms=["OFF", "SINGLE49"],
        ten_slot_complete=complete,
        main_arm_decisions={case_id: decisions[case_id]["MULTI46_47_48_49"]
                            for case_id in CASE_IDS},
        only49_decisions={case_id: decisions[case_id]["ONLY49"]
                          for case_id in CASE_IDS},
        free49_decisions={case_id: decisions[case_id]["MULTI46_47_48_FREE49"]
                          for case_id in CASE_IDS},
        decisions=decisions, observed_misclassifications=mismatches,
        same_source_comparison=comparisons,
    )
    store.data["status"] = (
        "FIXED_STRUCTURED_FEEDBACK_DEVELOPMENT_COMPLETE" if complete and not mismatches
        else "FIXED_STRUCTURED_FEEDBACK_DEVELOPMENT_NEGATIVE" if complete else "INCOMPLETE"
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
    env["RGB_DCT_STRUCTURED_FEEDBACK_INTERNAL_CASE"] = case_id
    command = [sys.executable, "-u", "-m", MODULE, "--output", str(output)]
    worker_log = output / f"{case_id}_worker.log"
    started = time.monotonic()
    receipt = dict(
        status="STARTING", error=None, exit_code=None, elapsed_seconds=0.0,
        parent_observed_peak_rss_kib=None,
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
    internal = os.environ.get("RGB_DCT_STRUCTURED_FEEDBACK_INTERNAL_CASE")
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
        from runtime.wan.rgb_dct_structured_feedback_backend import WanStructuredFeedbackBackend
        backend = WanStructuredFeedbackBackend(runtime_config, store.count)
        run_case(store, internal, config, backend)
        return
    _validate_output(output)
    _supervise(output, config, _source_sha())


if __name__ == "__main__":
    main()
