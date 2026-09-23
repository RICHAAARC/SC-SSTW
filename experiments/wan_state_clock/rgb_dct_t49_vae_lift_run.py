"""Fixed two-source, eight-MP4 RGB/DCT T49 VAE-lift development trial.

The user runs the pinned Colab notebook. This runner has no calibration,
threshold, key search, backward pass, arm selection, or retry path.
"""
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

from main.tube_state import rgb_dct_presence as receiver
from main.tube_state.rgb_dct_t49_carrier import AMPLITUDE, apply_carrier, lift_direction
from runtime.wan.rgb_dct_presence_adapter import score_mp4


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).parent / "configs/rgb_dct_t49_vae_lift_v1.json"
PROTOCOL_PATH = ROOT / "docs/rgb_dct_t49_vae_lift_protocol_20260923.md"
MODULE = "experiments.wan_state_clock.rgb_dct_t49_vae_lift_run"
OUTPUT_PARENT = Path("/content/drive/MyDrive/Video-WM/RGB-DCT-T49-VAE-Lift-V1")
CASE_IDS = ("truck_new_s11", "dog_weak_new_s12")
ARMS = ("OFF", "PIXEL_PLUS", "PIXEL_MINUS", "NATIVE_PLUS")
PLAN = dict(generation=2, transformer=200, scheduler_step=102,
            unit_response_probe_step=2, vae_decode=4, vae_encode=4,
            backward=0, mp4_save=8, mp4_read=8, score=8)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if _sha(PROTOCOL_PATH) != config["protocol_sha256"]:
        raise ValueError("published frozen protocol byte identity mismatch")
    if config["receiver_spec_sha256"] != receiver.SPEC_SHA256:
        raise ValueError("RGB-DCT receiver specification changed")
    if (config["receiver_baseline_source_sha"] != "c967db03767768871ba0cd3b49208aa308f51b48"
            or config["model"] != dict(
                id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                revision="0fad780a534b6463e45facd96134c9f345acfa5b",
                transformer_dtype="bfloat16", vae_dtype="float32",
            )):
        raise ValueError("frozen receiver baseline/model identity changed")
    key = config["key_utf8"].encode("utf-8")
    if receiver.score_rgb(None, key)["key_id"] != config["receiver_key_id"]:
        raise ValueError("fixed receiver key identity changed")
    if tuple(case["id"] for case in config["cases"]) != CASE_IDS or tuple(config["media"]["arms"]) != ARMS:
        raise ValueError("fixed two-source/eight-arm roster changed")
    if config["cases"] != [
        dict(id="truck_new_s11", seed=2026092311,
             prompt="locked camera, a single turquoise cargo truck moving slowly along a straight empty road, stable daylight, no people, no cuts",
             role="DEVELOPMENT_SOURCE_NEW_PAIR"),
        dict(id="dog_weak_new_s12", seed=2026092312,
             prompt="locked camera, a single black dog walking slowly across an empty park lawn, stable cloudy light, no people, no cuts",
             role="DEVELOPMENT_SOURCE_NEW_PAIR"),
    ]:
        raise ValueError("fixed prompt/seed pairs changed")
    if config["media"] != dict(
        arms=list(ARMS), fps=8, crf=18, codec="libx264",
        pixel_format="yuv420p", receiver_readback="ffmpeg_rgb24",
    ):
        raise ValueError("fixed media codec/roster changed")
    if config["call_plan_max"] != PLAN | {"native_scheduler_total": 104}:
        raise ValueError("fixed call plan changed")
    if (config["carrier"] != dict(
            amplitude=0.1,
            rgb_channels="same additive carrier in R, G, B",
            dct_delta_without_clip="plus/minus 0.2*c_t*s_b",
            positive_only_native_lift=True, latent_support_time=[1, 45],
        ) or AMPLITUDE != 0.1 or config["control"] != dict(
            index=49, target_D_support_rms=0.042943312697648145,
            sigma50=0.0, native_response_ratio_floor=0.1,
            direct_gate="S(PIXEL_PLUS)>S(OFF)>S(PIXEL_MINUS)",
            no_gradient_no_scan_no_fallback=True,
        )):
        raise ValueError("fixed carrier/native criterion changed")
    if config["fixed_denominator"] != dict(sources=2, mp4_score_slots=8, frames=1448):
        raise ValueError("fixed source/media/frame denominator changed")
    if config["resources"] != dict(
        case_timeout_seconds=10800, worker_processes_per_case=1,
        case_execution="serial fresh child processes",
        vae_after_transformer_release=True,
    ):
        raise ValueError("fixed resource execution contract changed")
    if config["generation"] != dict(
        height=320, width=512, frames=181, fps=8, steps=50,
        guidance_scale=5.0, max_sequence_length=512,
        negative_prompt="text, watermark, logo, camera motion, cuts, multiple objects, flicker",
    ):
        raise ValueError("fixed Wan generation geometry/schedule changed")
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
    return dict(
        status="PENDING", reason=None, score=None, pre_encode_score=None,
        pre_encode_status=None, receiver_status=None,
        score_change_after_codec=None, expected_frames=181, frames_used=0,
        path=str(case_dir / "received_videos" / arm / "FULL.mp4"),
        sha256=None, bytes=None, clipping=None,
    )


def initial_result(config: dict, output: Path, source_sha: str) -> dict:
    cases = {}
    for case in config["cases"]:
        case_dir = output / case["id"]
        cases[case["id"]] = dict(
            status="PENDING", prompt=case["prompt"], seed=case["seed"],
            role=case["role"], prompt_category_provenance="CATEGORY_SEEN_IN_OLDER_DEVELOPMENT",
            slots={arm: _slot(case_dir, arm) for arm in ARMS},
            channel_gate=None, lift=None, native_control=None,
            comparison=None, generation=None, vae=None,
            calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
            resources={}, stage="PENDING", elapsed_seconds=None,
            failures=[],
        )
    return dict(
        status="RUNNING", experiment_id=config["name"],
        source_sha=source_sha, config_path=str(CONFIG_PATH),
        config_sha256=_sha(CONFIG_PATH), protocol_path=str(PROTOCOL_PATH),
        protocol_sha256=config["protocol_sha256"],
        receiver_baseline_source_sha=config["receiver_baseline_source_sha"],
        receiver_spec_sha256=config["receiver_spec_sha256"],
        key_id=config["receiver_key_id"], key_utf8=config["key_utf8"],
        model=config["model"], generation_config=config["generation"],
        media_config=config["media"], carrier_config=config["carrier"],
        control_config=config["control"], resources_config=config["resources"],
        fixed_denominator=config["fixed_denominator"],
        call_plan_max=config["call_plan_max"],
        calls={kind: {"attempted": 0, "completed": 0} for kind in PLAN},
        attempted_media_slots=0, scored_media_slots=0, invalid_media_slots=0,
        pending_media_slots=8, scored_frames=0, cases=cases, output_dir=str(output),
        result_path=str(output / "result.json"),
        evidence_ceiling=config["evidence_ceiling"],
        codec_score_change_meaning="actual MP4 receiver score minus floating pre-encode receiver score; not pixel RMSE, video quality, or physical retention",
        decisions_are_development_only=True,
    )


class Store:
    def __init__(self, path: Path, data: dict):
        self.path = path
        self.data = data
        self.active_case: str | None = None

    @classmethod
    def open(cls, path: Path) -> "Store":
        return cls(path, json.loads(path.read_text(encoding="utf-8")))

    def save(self) -> None:
        statuses = [slot["status"] for case in self.data["cases"].values()
                    for slot in case["slots"].values()]
        self.data["scored_media_slots"] = statuses.count("SCORED")
        self.data["pending_media_slots"] = statuses.count("PENDING")
        self.data["attempted_media_slots"] = self.data["calls"]["mp4_save"]["attempted"]
        self.data["invalid_media_slots"] = 8 - self.data["scored_media_slots"] - self.data["pending_media_slots"]
        self.data["scored_frames"] = sum(
            slot["frames_used"] for case in self.data["cases"].values()
            for slot in case["slots"].values() if slot["status"] == "SCORED"
        )
        self.data["native_scheduler_calls"] = {
            name: self.data["calls"]["scheduler_step"][name]
            + self.data["calls"]["unit_response_probe_step"][name]
            for name in ("attempted", "completed")
        }
        if self.data["native_scheduler_calls"]["attempted"] > 104:
            raise RuntimeError("fixed native scheduler call cap exceeded")
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
            self.data["cases"][self.active_case]["calls"][kind][field] += 1
        self.save()

    def case(self, case_id: str) -> dict:
        return self.data["cases"][case_id]


def channel_gate(slots: dict) -> dict:
    direct = {arm: slots[arm] for arm in ARMS[:3]}
    if any(row["status"] != "SCORED" for row in direct.values()):
        return dict(status="INVALID", reason="DIRECT_MP4_SCORE_MISSING", direct_delta=None)
    off, plus, minus = (direct[arm]["score"] for arm in ARMS[:3])
    positive = plus > off > minus
    return dict(status="POSITIVE" if positive else "NEGATIVE",
                reason=None if positive else "FIXED_DIRECT_DIRECTION_FAILED",
                direct_delta=plus - off, negative_delta=minus - off,
                rule="S(PIXEL_PLUS)>S(OFF)>S(PIXEL_MINUS)")


def case_comparison(slots: dict, ratio_floor: float) -> dict:
    if any(slots[arm]["status"] != "SCORED" for arm in ARMS):
        return dict(status="UNAVAILABLE", direct_delta=None, native_delta=None,
                    native_to_direct_ratio=None)
    off = slots["OFF"]["score"]
    direct = slots["PIXEL_PLUS"]["score"] - off
    negative = slots["PIXEL_MINUS"]["score"] - off
    native = slots["NATIVE_PLUS"]["score"] - off
    controlled = direct > 0 and negative < 0 and native >= ratio_floor * direct
    return dict(status="DEVELOPMENT_CONTROLLED" if controlled else "FIXED_CONSTRUCTION_NEGATIVE",
                direct_delta=direct, negative_delta=negative, native_delta=native,
                native_to_direct_ratio=native / direct if direct > 0 else None,
                native_ratio_floor=ratio_floor,
                meaning="normalized receiver-score increment ratio only; not a physical retention or detection threshold")


def _encode_rgb_np(rgb: np.ndarray, path: Path, fps: int, crf: int) -> None:
    import torch
    from runtime.wan.io import encode_rgb

    encode_rgb(torch.from_numpy(np.ascontiguousarray(rgb)), path, fps, crf)


def _save_and_score(
    store: Store, case_id: str, arm: str, rgb: np.ndarray, key: bytes,
    config: dict, *, encode_fn: Callable = _encode_rgb_np,
    score_fn: Callable = score_mp4,
    clipping: dict | None = None,
) -> None:
    slot = store.case(case_id)["slots"][arm]
    path = Path(slot["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    slot.update(status="PREPARING", clipping=clipping)
    store.save()
    try:
        pre = receiver.score_rgb(rgb, key)
        slot["pre_encode_status"] = pre["status"]
        if pre["status"] != "SCORED":
            raise ValueError("nonfinite/invalid pre-encode RGB score: " + str(pre["reason"]))
        slot["pre_encode_score"] = pre["score"]
        store.save()
        store.count("mp4_save", False)
        encode_fn(rgb, path, config["media"]["fps"], config["media"]["crf"])
        store.count("mp4_save", True)
        slot.update(sha256=_sha(path), bytes=path.stat().st_size)
        store.save()
        store.count("mp4_read", False)
        store.count("score", False)
        scored = score_fn(path, key)  # The receiver sees only a path and key.
        slot["receiver_status"] = scored.get("status")
        slot["frames_used"] = scored.get("frames_used", 0)
        if (scored.get("status") != "SCORED" or not isinstance(scored.get("score"), (int, float))
                or not math.isfinite(scored["score"]) or slot["frames_used"] != 181
                or scored.get("spec_sha256") != receiver.SPEC_SHA256
                or scored.get("key_id") != config["receiver_key_id"]):
            slot.update(status="DECODE_INVALID", reason=scored.get("reason") or "RECEIVER_OUTPUT_INVALID")
        else:
            store.count("mp4_read", True)
            store.count("score", True)
            slot.update(status="SCORED", score=float(scored["score"]), reason=None,
                        score_change_after_codec=float(scored["score"] - pre["score"]))
    except Exception as exc:
        slot.update(status="ENGINEERING_INVALID", reason=f"{type(exc).__name__}: {exc}", score=None)
    store.save()


def _mark_pending(store: Store, case_id: str, status: str, reason: str) -> None:
    for slot in store.case(case_id)["slots"].values():
        if slot["status"] in ("PENDING", "PREPARING"):
            slot.update(status=status, reason=reason, score=None)
    store.save()


def _finalize_case(store: Store, case_id: str, config: dict) -> None:
    case = store.case(case_id)
    case["comparison"] = case_comparison(case["slots"], config["control"]["native_response_ratio_floor"])
    if case["comparison"]["status"] == "DEVELOPMENT_CONTROLLED":
        case["status"] = "DEVELOPMENT_CONTROLLED"
    elif any(case["slots"]["NATIVE_PLUS"]["status"] == negative for negative in (
        "NOT_RUN_CHANNEL_NEGATIVE", "ZERO_LIFT_DIRECTION", "ZERO_NATIVE_RESPONSE",
    )) or case["comparison"]["status"] == "FIXED_CONSTRUCTION_NEGATIVE":
        case["status"] = "FIXED_CONSTRUCTION_NEGATIVE"
    else:
        case["status"] = "ENGINEERING_INVALID"
    store.save()


def run_case(
    store: Store, case_id: str, config: dict, backend,
    *, encode_fn: Callable = _encode_rgb_np, score_fn: Callable = score_mp4,
) -> None:
    """Run one fixed case; injection points are for CPU fake tests only."""
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
        case["stage"] = "DIRECT_RGB_CARRIER"
        store.save()
        plus, plus_clipping = apply_carrier(x0, key, +1)
        _save_and_score(store, case_id, "PIXEL_PLUS", plus, key, config,
                        encode_fn=encode_fn, score_fn=score_fn,
                        clipping=plus_clipping)
        minus, minus_clipping = apply_carrier(x0, key, -1)
        _save_and_score(store, case_id, "PIXEL_MINUS", minus, key, config,
                        encode_fn=encode_fn, score_fn=score_fn,
                        clipping=minus_clipping)
        del minus
        gate = channel_gate(case["slots"])
        case["channel_gate"] = gate
        store.save()  # All three physical MP4 scores precede this truth join.
        native = case["slots"]["NATIVE_PLUS"]
        if gate["status"] != "POSITIVE":
            native.update(status="NOT_RUN_CHANNEL_NEGATIVE" if gate["status"] == "NEGATIVE"
                          else "NOT_RUN_CHANNEL_INVALID", reason=gate["reason"])
            store.save()
            return
        case["stage"] = "POSITIVE_VAE_LIFT"
        store.save()
        encoded_plus = backend.encode(plus)  # Float P+, not RGB8 or MP4 readback.
        encoded_x0 = backend.encode(x0)
        raw = encoded_plus - encoded_x0  # Never E(P+) - z0.
        masked, lift = lift_direction(raw)
        case["lift"] = lift
        store.save()
        del encoded_plus, encoded_x0, plus
        if lift["status"] == "ZERO_LIFT_DIRECTION":
            native.update(status="ZERO_LIFT_DIRECTION", reason="FINITE_ZERO_MASKED_LIFT")
            store.save()
            return
        case["stage"] = "NATIVE_T49_CONTROL"
        store.save()
        state, terminal, control = backend.controlled_terminal(
            prefix, masked, config["control"]["target_D_support_rms"]
        )
        case["native_control"] = control
        store.save()
        if state == "ZERO_NATIVE_RESPONSE":
            native.update(status="ZERO_NATIVE_RESPONSE", reason="FINITE_ZERO_UNIT_RESPONSE")
            store.save()
            return
        if state != "READY":
            raise ValueError("unexpected native control state")
        case["stage"] = "DECODE_SAVE_NATIVE"
        store.save()
        rgb_native = backend.decode(terminal)
        _save_and_score(store, case_id, "NATIVE_PLUS", rgb_native, key, config,
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
            case["resources"]["cpu_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            case["resources"]["wall_elapsed_seconds"] = case["elapsed_seconds"]
        except Exception as exc:
            case["resources"] = {"status": "UNAVAILABLE", "reason": repr(exc)}
        try:
            backend.release()
        except Exception as exc:
            case["failures"].append(dict(stage="RELEASE", error=repr(exc)))
        _finalize_case(store, case_id, config)
        store.active_case = None
        gc.collect()


def finalize_experiment(store: Store) -> None:
    statuses = [store.case(case_id)["status"] for case_id in CASE_IDS]
    if statuses == ["DEVELOPMENT_CONTROLLED"] * 2 and store.data["scored_media_slots"] == 8:
        store.data["status"] = "DEVELOPMENT_CONTROLLABILITY_OBSERVED"
    elif "ENGINEERING_INVALID" in statuses:
        store.data["status"] = "INCOMPLETE_ENGINEERING_INVALID"
    else:
        store.data["status"] = "FIXED_CONSTRUCTION_NEGATIVE"
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


def _supervise(output: Path, config: dict, source_sha: str) -> None:
    result_path = output / "result.json"
    store = Store(result_path, initial_result(config, output, source_sha))
    store.save()  # All 8 PENDING slots before any model/media work.
    for case_id in CASE_IDS:
        env = os.environ.copy()
        env["RGB_DCT_T49_INTERNAL_CASE"] = case_id
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
        store = Store.open(result_path)
        if failure is not None:
            case = store.case(case_id)
            case["failures"].append(dict(stage="WORKER", error=failure,
                                         log_path=str(worker_log)))
            _mark_pending(store, case_id, "NOT_RUN_RESOURCE_FAILURE" if failure == "WORKER_TIMEOUT"
                          else "NOT_RUN_WORKER_FAILURE", failure)
            _finalize_case(store, case_id, config)
        store.data["cases"][case_id]["worker_log_path"] = str(worker_log)
        store.save()
        print("case complete:", case_id, store.case(case_id)["status"], flush=True)
    finalize_experiment(store)
    for case_id in CASE_IDS:
        for arm in ARMS:
            row = store.case(case_id)["slots"][arm]
            print(case_id, arm, row["status"], row["score"], flush=True)
    print("result:", result_path, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config()
    output = args.output
    internal_case = os.environ.get("RGB_DCT_T49_INTERNAL_CASE")
    if internal_case:
        if internal_case not in CASE_IDS or not (output / "result.json").is_file():
            raise ValueError("invalid fixed worker invocation")
        case_config = next(row for row in config["cases"] if row["id"] == internal_case)
        runtime_config = copy.deepcopy(config)
        runtime_config["generation"].update(prompt=case_config["prompt"],
                                            seed=case_config["seed"])
        store = Store.open(output / "result.json")
        from runtime.wan.rgb_dct_t49_backend import WanT49Backend

        run_case(store, internal_case, config,
                 WanT49Backend(runtime_config, store.count))
        return
    _validate_output(output)
    _supervise(output, config, _source_sha())


if __name__ == "__main__":
    main()
