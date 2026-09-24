"""Fixed 2 REF_OFF + 2 EVAL RGB-DCT group-consistency functional run."""
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
from runtime.wan.rgb_dct_group_consistency_adapter import score_mp4

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "configs/rgb_dct_group_consistency_v1.json"
PROTOCOL_PATH = ROOT / "docs/rgb_dct_group_consistency_h0_h1_protocol_20260924.md"
MODULE = "experiments.wan_state_clock.rgb_dct_group_consistency_run"
OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/RGB-DCT-Group-Consistency-H0-H1-V1")
REF_IDS = ("ref_metronome_s2411", "ref_pond_s2412")
EVAL_IDS = ("eval_bicycle_s2413", "eval_windmill_s2414")
CASE_IDS = REF_IDS + EVAL_IDS
PROMPTS = (
    "locked camera, a wooden metronome swinging steadily on a plain desk, stable indoor light, no people, no cuts",
    "locked camera, gentle ripples spreading across a small pond beside still stones, stable daylight, no people, no cuts",
    "locked camera, a blue bicycle rolling slowly along an empty paved path, stable daylight, no people, no cuts",
    "locked camera, a small white windmill turning slowly in a grassy field, stable daylight, no people, no cuts",
)
SEEDS = (2026092411, 2026092412, 2026092413, 2026092414)
PLAN = dict(generation=4, transformer=400, scheduler_step=202,
            unit_response_probe_step=2, vae_decode=6, vae_encode=4,
            backward=0, mp4_save=6, mp4_read=6, score=6)
TOTAL_NATIVE_STEPS = 204


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
    expected = [dict(id=case_id, seed=seed, prompt=prompt,
                     role="REF_OFF" if index < 2 else "EVAL")
                for index, (case_id, seed, prompt) in
                enumerate(zip(CASE_IDS, SEEDS, PROMPTS, strict=True))]
    if config["cases"] != expected:
        raise ValueError("fixed REF_OFF/EVAL prompt/seed roster changed")
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
                index=49, target_D_support_rms=0.042943312697648145,
                sigma50=0.0, no_gradient_no_scan_no_fallback=True)):
        raise ValueError("fixed writer mechanism changed")
    if config["decision_rule"] != dict(statistic="C=sum(q_g>0)", threshold=24,
                                      rule="C>=24:H1;C<24:H0", reference_calibration=False):
        raise ValueError("fixed C>=24 decision rule changed")
    if config["fixed_denominator"] != dict(sources=4, mp4_score_slots=6, frames=1086):
        raise ValueError("fixed denominator changed")
    if config["call_plan_max"] != PLAN | {"native_scheduler_total": TOTAL_NATIVE_STEPS}:
        raise ValueError("fixed call plan changed")
    if config["resources"] != dict(
            case_timeout_seconds=10800, worker_processes_per_case=1,
            case_execution="serial fresh child processes",
            vae_after_transformer_release=True):
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
                positive_groups=None, decision=None,
                expected_frames=181, frames_used=0,
                path=str(case_dir / "received_videos" / arm / "FULL.mp4"),
                sha256=None, bytes=None, receiver_status=None, clipping=None)


def initial_result(config: dict, output: Path, source_sha: str) -> dict:
    cases = {}
    for case in config["cases"]:
        case_id = case["id"]
        arms = ("OFF",) if case["role"] == "REF_OFF" else ("OFF", "NATIVE_H1")
        cases[case_id] = dict(
            status="PENDING", stage="PENDING", role=case["role"],
            prompt=case["prompt"], seed=case["seed"],
            slots={arm: _slot(output / case_id, arm) for arm in arms},
            generation=None, vae=None, lift=None, native_control=None,
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
        pending_media_slots=6, scored_frames=0,
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
        if len(slots) != 6:
            raise RuntimeError("fixed six-slot denominator changed")
        self.data["scored_media_slots"] = statuses.count("SCORED")
        self.data["pending_media_slots"] = statuses.count("PENDING")
        self.data["attempted_media_slots"] = self.data["calls"]["mp4_save"]["attempted"]
        self.data["invalid_media_slots"] = 6 - self.data["scored_media_slots"] - self.data["pending_media_slots"]
        self.data["scored_frames"] = sum(slot["frames_used"] for slot in slots
                                          if slot["status"] == "SCORED")
        self.data["native_scheduler_calls"] = {
            name: self.data["calls"]["scheduler_step"][name]
            + self.data["calls"]["unit_response_probe_step"][name]
            for name in ("attempted", "completed")}
        if self.data["native_scheduler_calls"]["attempted"] > TOTAL_NATIVE_STEPS:
            raise RuntimeError("native scheduler cap exceeded")
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
    score_fn: Callable = score_mp4, clipping: dict | None = None,
) -> None:
    slot = store.case(case_id)["slots"][arm]
    path = Path(slot["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    slot.update(status="PREPARING", clipping=clipping)
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
        scored = score_fn(path, key)  # Blind receiver sees only MP4 path and key.
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
                        positive_groups=scored["positive_groups"], reason=None)
            store.save()  # Media identity and receiver C precede the blind decision.
            slot["decision"] = receiver.decide(slot["positive_groups"])
    except Exception as exc:
        slot.update(status="ENGINEERING_INVALID",
                    reason=f"{type(exc).__name__}: {exc}", score=None)
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
    elif "ZERO_LIFT_DIRECTION" in states or "ZERO_NATIVE_RESPONSE" in states:
        case["status"] = "METHOD_NEGATIVE"
    else:
        case["status"] = "ENGINEERING_INVALID"
    store.save()


def run_case(
    store: Store, case_id: str, config: dict, backend,
    *, encode_fn: Callable = _encode_rgb_np, score_fn: Callable = score_mp4,
) -> None:
    """Run one fresh source; injection points support CPU fake verification."""
    case = store.case(case_id)
    store.active_case = case_id
    key = config["key_utf8"].encode("utf-8")
    case_dir = Path(store.data["output_dir"]) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        case.update(status="RUNNING", stage="GENERATE_PREFIX")
        store.save()
        prefix = backend.generate_prefix(case_dir / "state")
        case["generation"] = prefix.metadata
        case["stage"] = "VAE_LOAD_DECODE_OFF"
        store.save()
        case["vae"] = backend.load_vae()
        x0 = backend.decode(prefix.z0)
        _save_and_score(store, case_id, "OFF", x0, key, config,
                        encode_fn=encode_fn, score_fn=score_fn)
        if case["role"] == "REF_OFF":
            return
        case["stage"] = "INTERNAL_FLOAT_RGB_POSITIVE_LIFT"
        store.save()
        plus, clipping = apply_carrier(x0, key, +1)
        case["internal_positive_clipping"] = clipping
        encoded_plus = backend.encode(plus)  # Floating P+, never a PIXEL+ MP4.
        encoded_x0 = backend.encode(x0)
        masked, lift = lift_direction(encoded_plus - encoded_x0)
        case["lift"] = lift
        store.save()
        del encoded_plus, encoded_x0, plus
        native = case["slots"]["NATIVE_H1"]
        if lift["status"] == "ZERO_LIFT_DIRECTION":
            native.update(status="ZERO_LIFT_DIRECTION",
                          reason="FINITE_ZERO_MASKED_LIFT")
            store.save()
            return
        case["stage"] = "NATIVE_T49_CONTROL"
        store.save()
        state, terminal, control = backend.controlled_terminal(
            prefix, masked, config["control"]["target_D_support_rms"])
        case["native_control"] = control
        store.save()
        if state == "ZERO_NATIVE_RESPONSE":
            native.update(status="ZERO_NATIVE_RESPONSE",
                          reason="FINITE_ZERO_UNIT_RESPONSE")
            store.save()
            return
        if state != "READY":
            raise ValueError("unexpected native control state")
        case["stage"] = "DECODE_SAVE_NATIVE"
        store.save()
        rgb_native = backend.decode(terminal)
        _save_and_score(store, case_id, "NATIVE_H1", rgb_native, key, config,
                        encode_fn=encode_fn, score_fn=score_fn)
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
    for case_id in EVAL_IDS:
        pair = store.case(case_id)["slots"]
        diagnostics[case_id] = (
            pair["NATIVE_H1"]["score"] - pair["OFF"]["score"]
            if all(pair[arm]["status"] == "SCORED" for arm in ("OFF", "NATIVE_H1"))
            else None)
    store.data["reporting"] = dict(
        joined_after_decision_persistence=True,
        expected=expected, decisions=decisions, observed_misclassifications=errors,
        same_source_native_minus_off_diagnostic=diagnostics)
    store.data["status"] = (
        "FUNCTIONAL_BLIND_PRESENCE_OBSERVED" if complete and not errors
        else "FIXED_FUNCTIONAL_NEGATIVE" if complete else "INCOMPLETE")
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
    env["RGB_DCT_GROUP_INTERNAL_CASE"] = case_id
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
    store.save()  # Four sources/six slots/1086 planned frames before model work.
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
    internal_case = os.environ.get("RGB_DCT_GROUP_INTERNAL_CASE")
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
        from runtime.wan.rgb_dct_t49_backend import WanT49Backend
        run_case(store, internal_case, config,
                 WanT49Backend(runtime_config, store.count))
        return
    _validate_output(output)
    _supervise(output, config, _source_sha())


if __name__ == "__main__":
    main()
