"""Post-MP4 quality plus fixed wrong-key and unwatermarked controls.

This follow-up consumes the completed residual-joint run and its checkpoint
ancestor.  It never reconstructs a writer or republishes media.  K1 clock
selection is recomputed from the saved receiver-visible framewise latent.
OFF_NATIVE starts from the checkpoint's saved MP4 readback.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import time

from experiments.paper_results_v1.resume_state import StateJournal, atomic_bytes, read_state, unit_attempt
from experiments.paper_results_v1.residual_joint_v1 import (
    CASES, CODEC, DELETE_POSITIONS, DELETE_SHAPE, GROUPS, RGB_SHAPE,
    _atomic_npy, _atomic_rgb8, _atomic_torch, _default_payload_read,
    _default_wan_encode, _load_npy, _load_npy_from_value, _load_rgb_path,
    _load_torch, _posthoc_path_metrics, _select_clock, delete_spec,
)


SCHEMA = "paper-results-v1-residual-controls-v1"
ARMS = ("WM_K0_REUSED", "WM_K1", "OFF_K0")
NEW_ARMS = ("WM_K1", "OFF_K0")
QUALITY_REFERENCES = ("CANDIDATE_PRE", "NATIVE_PRE", "NATIVE_POST")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def reason(exc):
    return f"{type(exc).__name__}: {exc}"


def _slot_id(arm, group, case):
    return f"{arm.lower()}/{group.lower()}/{case}"


def _slot_ids():
    return [_slot_id(arm, group, case) for arm in ARMS for group in GROUPS for case in CASES]


def _quality_ids():
    return [f"{group.lower()}/{case}/{reference.lower()}" for group in GROUPS for case in CASES
            for reference in QUALITY_REFERENCES]


def _shape(group):
    return RGB_SHAPE if group == "CLEAN_MP4" else DELETE_SHAPE


def _key_label(arm):
    return "K1" if arm == "WM_K1" else "K0"


def _new_rows():
    rows = []
    for arm in ARMS:
        for group in GROUPS:
            for case in CASES:
                reused = arm == "WM_K0_REUSED"
                rows.append({
                    "slot_id": _slot_id(arm, group, case), "arm": arm, "group": group,
                    "case_id": case, "key_label": _key_label(arm), "control_role": (
                        "POSITIVE_REUSED" if reused else "WRONG_KEY" if arm == "WM_K1" else "UNWATERMARKED"
                    ),
                    "status": "PENDING", "reason": None, "planned_bits": 32, "missing_bits": 32,
                    "received_frames": 181 if group == "CLEAN_MP4" else 163,
                    "media": {"status": "PENDING", "reason": None},
                    "clock": {"status": "PENDING", "reason": None},
                    "payload": {"status": "PENDING", "reason": None},
                })
    return rows


def _new_quality_rows():
    return [{
        "quality_id": f"{group.lower()}/{case}/{reference.lower()}",
        "group": group, "case_id": case, "reference": reference,
        "status": "PENDING", "reason": None, "quality_gate": None,
    } for group in GROUPS for case in CASES for reference in QUALITY_REFERENCES]


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.path = self.output / "residual_controls_state.json"
        self.data = read_state(self.path)
        if self.data.get("schema") != SCHEMA:
            raise ValueError("residual-controls schema changed")
        if [row.get("slot_id") for row in self.data.get("rows", [])] != _slot_ids():
            raise ValueError("fixed 24-row denominator changed")
        if [row.get("quality_id") for row in self.data.get("quality_rows", [])] != _quality_ids():
            raise ValueError("fixed quality plan changed")
        self.journal = StateJournal(self.path)
        self.journal.seed_cache(self.data)
        retired = []
        for phase, status in list(self.data.get("phases", {}).items()):
            if status == "RUNNING":
                self.data["phases"][phase] = "INTERRUPTED"; retired.append("phase/" + phase)
        for row in self.data["rows"]:
            for name in ("media", "clock", "payload"):
                if row[name].get("status") == "RUNNING":
                    row[name].update(status="INTERRUPTED", reason="previous process ended before unit completion")
                    retired.append(row["slot_id"] + "/" + name)
        for row in self.data["quality_rows"]:
            if row.get("status") == "RUNNING":
                row.update(status="INTERRUPTED", reason="previous process ended before unit completion")
                retired.append("quality/" + row["quality_id"])
        for collection in ("edit_calls", "quality_calls", "framewise_loads", "framewise_encode_clips",
                           "framewise_physical_batches", "clock_calls", "wan_loads", "wan_encode_clips",
                           "wan_reuses", "wan_cache_repairs", "payload_read_calls"):
            for index, record in enumerate(self.data.get(collection, [])):
                if record.get("status") == "RUNNING":
                    record.update(status="INTERRUPTED", reason="previous process ended before completion",
                                  finished_at=time.time())
                    retired.append(f"{collection}/{index}")
        for unit_id, attempts in self.data.get("unit_attempts", {}).items():
            for index, attempt in enumerate(attempts):
                if attempt.get("status") == "RUNNING":
                    attempt.update(status="INTERRUPTED", previous_status="RUNNING",
                                   reason="previous process ended before completion", finished_at=time.time())
                    retired.append(f"unit_attempts/{unit_id}/{index}")
        if retired:
            self.data.setdefault("resume_recoveries", []).append({"at": time.time(), "retired_running": retired})
            self.save()

    def save(self):
        self.journal.save(self.data)


def _validate_config(config):
    if config.get("codec") != CODEC:
        raise ValueError("codec must remain libx264/8fps/CRF18/yuv420p")
    keys = config.get("keys", {})
    if not all(isinstance(keys.get(name), str) and keys[name] for name in ("K0", "K1")):
        raise ValueError("nonempty existing K0 and K1 are required")
    if keys["K0"] == keys["K1"]:
        raise ValueError("K1 must differ from K0")
    bits = config.get("payload_bits")
    if not isinstance(bits, list) or len(bits) != 32 or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
        raise ValueError("payload_bits must contain 32 binary integers")
    for name in ("wan", "framewise"):
        model = config.get("models", {}).get(name)
        if not isinstance(model, dict) or not isinstance(model.get("local_snapshot_path"), str):
            raise ValueError(f"models.{name}.local_snapshot_path is required")
    if config["models"]["framewise"].get("batch_frames") != 8:
        raise ValueError("framewise batch_frames must remain 8")
    return keys, list(bits)


def _joint_row(state, group, case):
    return next(row for row in state["rows"] if row["group"] == group and row["case_id"] == case)


def _checkpoint_artifact(state, case, arm, phase):
    wanted = f"{case}/{arm}/{phase}"
    matches = [row for row in state.get("artifacts", []) if row.get("artifact_id") == wanted]
    if len(matches) != 1:
        raise ValueError(f"checkpoint artifact cardinality changed: {wanted}")
    return matches[0]


def initialize(source_joint_run, checkpoint_run, output, config):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    state_path = output / "residual_controls_state.json"
    cfg = read_json(config); keys, payload_bits = _validate_config(cfg)
    source_joint_run = Path(source_joint_run).resolve()
    checkpoint_run = Path(checkpoint_run).resolve()
    source_joint_state_path = source_joint_run / "run_state/residual_joint_state.json"
    checkpoint_state_path = checkpoint_run / "run_state/run_state.json"
    if state_path.is_file():
        store = Store(output)
        if store.data.get("keys") != keys:
            raise ValueError("keys differ from initialized control run")
        return store

    rows = _new_rows(); errors = []
    quality_rows = _new_quality_rows()
    source_joint = {"rows": [], "sources": {}}
    checkpoint = {"artifacts": []}
    source_problem = checkpoint_problem = None
    try:
        source_joint = read_json(source_joint_state_path)
        if source_joint.get("schema") != "paper-results-v1-residual-joint-v1":
            raise ValueError("source is not residual-joint-v1")
        if source_joint.get("key") != keys["K0"]:
            raise ValueError("K0 differs from completed residual-joint run")
    except Exception as exc:
        source_problem = reason(exc)
        errors.append({"at": time.time(), "phase": "initialize", "source": str(source_joint_state_path),
                       "reason": source_problem})
        source_joint = {"rows": [], "sources": {}}
    try:
        checkpoint = read_json(checkpoint_state_path)
        if not isinstance(checkpoint.get("artifacts"), list):
            raise ValueError("checkpoint artifacts must be a list")
    except Exception as exc:
        checkpoint_problem = reason(exc)
        errors.append({"at": time.time(), "phase": "initialize", "source": str(checkpoint_state_path),
                       "reason": checkpoint_problem})
        checkpoint = {"artifacts": []}

    def fail(row, component, why):
        row[component].update(status="MISSING", reason=why)
        row.update(status="FAILED", reason=why, missing_bits=32)
        errors.append({"at": time.time(), "phase": "initialize", "slot_id": row["slot_id"], "reason": why})

    for group in GROUPS:
        for case in CASES:
            reused = next(row for row in rows if row["arm"] == "WM_K0_REUSED" and row["group"] == group and row["case_id"] == case)
            wrong = next(row for row in rows if row["arm"] == "WM_K1" and row["group"] == group and row["case_id"] == case)
            matches = [row for row in source_joint.get("rows", [])
                       if row.get("group") == group and row.get("case_id") == case]
            if len(matches) != 1:
                why = source_problem or f"source residual-joint row cardinality {len(matches)} for {group}/{case}"
                fail(reused, "payload", why); fail(wrong, "media", why)
                for quality in quality_rows:
                    if quality["group"] == group and quality["case_id"] == case:
                        quality.update(status="FAILED", reason=why)
                continue
            source = matches[0]
            if source.get("payload", {}).get("status") == "READ":
                reused.update(status="READ", reason=None, missing_bits=0)
                reused["media"] = {**source.get("media", {}), "reuse": "SOURCE_RESIDUAL_JOINT"}
                reused["clock"] = {**source.get("clock", {}), "reuse": "SOURCE_RESIDUAL_JOINT"}
                reused["payload"] = {**source.get("payload", {}), "reuse": "SOURCE_RESIDUAL_JOINT"}
            else:
                fail(reused, "payload", "source K0 row is not a completed READ")
            if source.get("media", {}).get("status") == "AVAILABLE":
                wrong["media"] = {**source["media"], "reuse": "SAME_RECEIVED_RGB_AS_WM_K0"}
                wrong.update(status="MEDIA_READY")
                wrong["clock"]["source_framewise_latent"] = source.get("clock", {}).get("latent_path")
                wrong["clock"]["source_k0_selector_record"] = source.get("clock", {}).get("selector_record")
            else:
                fail(wrong, "media", "source watermarked received RGB unavailable")

    for case in CASES:
        try:
            artifact = _checkpoint_artifact(checkpoint, case, "OFF_NATIVE", "POST")
        except Exception as exc:
            why = checkpoint_problem or reason(exc)
            for group in GROUPS:
                fail(next(row for row in rows if row["arm"] == "OFF_K0" and row["group"] == group and row["case_id"] == case),
                     "media", why)
        else:
            for group in GROUPS:
                off = next(row for row in rows if row["arm"] == "OFF_K0" and row["group"] == group and row["case_id"] == case)
                off["media"]["source_artifact"] = artifact
        for reference, phase in (("NATIVE_PRE", "PRE"), ("NATIVE_POST", "POST")):
            try:
                _checkpoint_artifact(checkpoint, case, "PAYLOAD_NATIVE", phase)
            except Exception as exc:
                why = checkpoint_problem or reason(exc)
                for quality in quality_rows:
                    if quality["case_id"] == case and quality["reference"] == reference:
                        quality.update(status="FAILED", reason=why)

    state = {
        "schema": SCHEMA, "status": "INITIALIZED", "source_joint_run": str(source_joint_run),
        "source_joint_state": str(source_joint_state_path), "checkpoint_run": str(checkpoint_run),
        "checkpoint_state": str(checkpoint_state_path), "effective_config": str(Path(config).resolve()),
        "keys": keys, "payload_bits": payload_bits, "rows": rows, "quality_rows": quality_rows,
        "fixed_denominator": {"cases": list(CASES), "groups": list(GROUPS), "arms": list(ARMS),
                              "rows": 24, "bits": 768, "rows_per_arm": 8, "bits_per_row": 32},
        "method": {
            "wm_k0": "reuse completed same-run K0 rows without model calls",
            "wm_k1": "saved received RGB and framewise latent -> fresh K1 CENTERED_U -> operation-specific Wan input -> K1 payload read",
            "off_k0": "saved OFF_NATIVE MP4 readback -> same clean/delete conditions -> K0 CENTERED_U -> Wan payload read",
            "truth_boundary": "clock and Wan phases complete with no pending model unit before known probe bits are compared",
            "off_interpretation": "absence/false-match control only; no inserted payload truth and no presence decision",
            "thresholds": None, "off_k1": "NOT_IN_FIXED_SCOPE",
            "off_framewise_cache_audit": (
                "checkpoint/recovery records contain completed OFF_NATIVE POST/delete_g0 encode calls "
                "and key-score sidecars, but expose no raw key-independent [T,4,40,64] POST latent; "
                "PRE, other-attack, and Wan temporal-16-channel tensors are not substitutes"
            ),
        },
        "planned_budget": {
            "generation_or_dit": 0, "writer": 0, "reconstruction": 0, "codec_calls": 0,
            "cpu_off_delete_edits": 4, "quality_pairs": 24,
            "framewise_loads_uninterrupted_upper_bound": 1, "new_framewise_encode_clips": 8,
            "clean_framewise_physical_batches": 92, "delete_framewise_physical_batches": 84,
            "new_framewise_physical_batches": 176, "framewise_batch_frames": 8,
            "new_centered_u_selections": 16, "wan_loads_uninterrupted_upper_bound": 1,
            "new_wan_encode_clips_upper_bound": 16,
            "framewise_cache_reuse_note": (
                "zero reusable OFF POST caches were located in the targeted checkpoint/recovery audit; "
                "the runner reuses only its own durable output on resume"
            ),
            "wan_encode_budget_note": (
                "with all blind operations supported, eight OFF plus zero-to-eight K1 fresh encodes; "
                "unsupported clock/visible-span or failures can make actual calls fewer than eight while rows remain fixed"
            ),
            "operation_equal_k1_wan_reuse_upper_bound": 8, "new_payload_reads": 16,
        },
        "phases": {"quality": "PENDING", "media": "PENDING", "clock": "PENDING", "wan": "PENDING"},
        "unit_attempts": {}, "errors": errors, "edit_calls": [], "quality_calls": [],
        "framewise_loads": [], "framewise_encode_clips": [], "framewise_physical_batches": [],
        "clock_calls": [], "wan_loads": [], "wan_encode_clips": [], "wan_reuses": [],
        "wan_cache_repairs": [], "payload_read_calls": [],
    }
    atomic_json(state_path, state)
    store = Store(output); report(store)
    return store


def _row(store, arm, group, case):
    return next(row for row in store.data["rows"] if row["arm"] == arm and row["group"] == group and row["case_id"] == case)


def _source_joint(store):
    return read_json(store.data["source_joint_state"])


def _checkpoint(store):
    return read_json(store.data["checkpoint_state"])


def _mapped_indices(group):
    if group == "CLEAN_MP4":
        return list(range(181))
    deleted = set(DELETE_POSITIONS)
    return [index for index in range(181) if index not in deleted]


def _quality_reference_path(store, case, reference):
    if reference == "CANDIDATE_PRE":
        return Path(_source_joint(store)["sources"][case]["path"])
    phase = "PRE" if reference == "NATIVE_PRE" else "POST"
    return Path(_checkpoint_artifact(_checkpoint(store), case, "PAYLOAD_NATIVE", phase)["path"])


def _quality_metrics(candidate_path, reference_path, group):
    import numpy as np
    shape = _shape(group); mapping = _mapped_indices(group)
    expected_candidate = math.prod(shape); expected_reference = math.prod(RGB_SHAPE)
    if not Path(candidate_path).is_file() or Path(candidate_path).stat().st_size != expected_candidate:
        raise FileNotFoundError(f"candidate received RGB unreadable: {candidate_path}")
    if not Path(reference_path).is_file() or Path(reference_path).stat().st_size != expected_reference:
        raise FileNotFoundError(f"quality reference RGB unreadable: {reference_path}")
    candidate = np.memmap(candidate_path, np.uint8, "r", shape=tuple(shape))
    reference = np.memmap(reference_path, np.uint8, "r", shape=tuple(RGB_SHAPE))
    frames, temporal = [], []
    sq_sum = abs_sum = changed = count = 0; maximum = 0; prior = None
    tsq_sum = tabs_sum = tchanged = tcount = 0; tmaximum = 0
    for received_index, source_index in enumerate(mapping):
        delta = candidate[received_index].astype(np.int16) - reference[source_index].astype(np.int16)
        absolute = np.abs(delta); square = delta.astype(np.float64) ** 2
        frame_mse = float(square.mean()); frame_mae = float(absolute.mean())
        flat_peak = int(absolute.argmax()); y, x, channel = (int(value) for value in np.unravel_index(flat_peak, absolute.shape))
        frames.append({"received_frame": received_index, "source_frame": source_index,
                       "mse_rgb8": frame_mse, "rmse_rgb8": math.sqrt(frame_mse),
                       "mae_rgb8": frame_mae, "max_abs_rgb8": int(absolute.max()),
                       "peak_pixel": {"y": y, "x": x, "channel": channel,
                                      "absolute_error_rgb8": int(absolute.reshape(-1)[flat_peak])},
                       "changed_channel_fraction": float(np.mean(absolute > 0)),
                       "psnr_db": None if frame_mse == 0 else 10.0 * math.log10(255.0 ** 2 / frame_mse)})
        sq_sum += float(square.sum()); abs_sum += int(absolute.sum()); changed += int((absolute > 0).sum())
        count += int(delta.size); maximum = max(maximum, int(absolute.max()))
        if prior is not None:
            flicker = delta - prior; flicker_abs = np.abs(flicker); flicker_sq = flicker.astype(np.float64) ** 2
            mse = float(flicker_sq.mean())
            temporal_flat = int(flicker_abs.argmax())
            ty, tx, tchannel = (int(value) for value in np.unravel_index(temporal_flat, flicker_abs.shape))
            temporal.append({"from_received_frame": received_index - 1, "to_received_frame": received_index,
                             "from_source_frame": mapping[received_index - 1], "to_source_frame": source_index,
                             "mse_residual_delta_rgb8": mse, "rmse_residual_delta_rgb8": math.sqrt(mse),
                             "mae_residual_delta_rgb8": float(flicker_abs.mean()),
                             "max_abs_residual_delta_rgb8": int(flicker_abs.max()),
                             "peak_pixel": {"y": ty, "x": tx, "channel": tchannel,
                                            "absolute_error_rgb8": int(flicker_abs.reshape(-1)[temporal_flat])}})
            tsq_sum += float(flicker_sq.sum()); tabs_sum += int(flicker_abs.sum())
            tchanged += int((flicker_abs > 0).sum()); tcount += int(flicker.size)
            tmaximum = max(tmaximum, int(flicker_abs.max()))
        prior = delta
    peak_rows = {
        "frame_rmse": max(frames, key=lambda row: row["rmse_rgb8"]),
        "frame_max_pixel_abs": max(frames, key=lambda row: row["max_abs_rgb8"]),
        "transition_rmse": max(temporal, key=lambda row: row["rmse_residual_delta_rgb8"]),
        "transition_max_pixel_delta_abs": max(temporal, key=lambda row: row["max_abs_residual_delta_rgb8"]),
    }
    peaks = {}
    for kind, peak in peak_rows.items():
        temporal_peak = kind.startswith("transition_")
        anchor = peak["to_received_frame"] if temporal_peak else peak["received_frame"]
        if temporal_peak:
            start = max(0, peak["to_received_frame"] - 2); stop = min(len(temporal), peak["to_received_frame"] + 1)
            neighborhood = temporal[start:stop]
        else:
            start = max(0, anchor - 1); stop = min(len(frames), anchor + 2)
            neighborhood = frames[start:stop]
        peaks[kind] = {"metric": dict(peak), "anchor_received_frame": anchor,
                       "peak_pixel": dict(peak["peak_pixel"]), "neighborhood": neighborhood}
    return {
        "full_clip": {"frames": len(frames), "channels": count, "mse_rgb8": sq_sum / count,
                      "rmse_rgb8": math.sqrt(sq_sum / count), "mae_rgb8": abs_sum / count,
                      "max_abs_rgb8": maximum, "changed_channel_fraction": changed / count,
                      "psnr_db": None if sq_sum == 0 else 10.0 * math.log10(255.0 ** 2 / (sq_sum / count))},
        "per_frame": frames,
        "temporal": {"aggregate": {"transitions": len(temporal), "mse_residual_delta_rgb8": tsq_sum / tcount,
                                    "rmse_residual_delta_rgb8": math.sqrt(tsq_sum / tcount),
                                    "mae_residual_delta_rgb8": tabs_sum / tcount,
                                    "max_abs_residual_delta_rgb8": tmaximum,
                                    "changed_channel_fraction": tchanged / tcount}, "rows": temporal},
        "peaks": peaks,
        "mapping": "identity" if group == "CLEAN_MP4" else "fixed delete_g0 applied to reference",
    }


def _save_quality_peak_visuals(candidate_path, reference_path, group, peaks, output):
    """Save actual RGB8 candidate/reference/fixed-range difference triptychs."""
    import numpy as np
    from PIL import Image, ImageDraw
    candidate = np.memmap(candidate_path, np.uint8, "r", shape=tuple(_shape(group)))
    reference = np.memmap(reference_path, np.uint8, "r", shape=tuple(RGB_SHAPE))
    mapping = _mapped_indices(group); output = Path(output); output.mkdir(parents=True, exist_ok=True)

    def save_triptych(arrays, labels, path):
        images = [Image.fromarray(np.asarray(array, dtype=np.uint8), "RGB") for array in arrays]
        width = sum(image.width for image in images); height = max(image.height for image in images)
        canvas = Image.new("RGB", (width, height + 22), "white"); x_offset = 0
        draw = ImageDraw.Draw(canvas)
        for image, label in zip(images, labels):
            canvas.paste(image, (x_offset, 22)); draw.text((x_offset + 3, 4), label, fill="black")
            x_offset += image.width
        temporary = path.with_name(path.name + ".tmp")
        canvas.save(temporary, format="PNG"); os.replace(temporary, path)

    artifacts = {}
    for kind, peak in peaks.items():
        received_index = peak["anchor_received_frame"]; source_index = mapping[received_index]
        candidate_frame = np.asarray(candidate[received_index])
        reference_frame = np.asarray(reference[source_index])
        difference = np.abs(candidate_frame.astype(np.int16) - reference_frame.astype(np.int16)).astype(np.uint8)
        full_path = output / f"{kind}-frame-triptych.png"
        save_triptych((candidate_frame, reference_frame, difference),
                      ("candidate", "reference", "abs diff [0,255]"), full_path)
        pixel = peak["peak_pixel"]; y, x = pixel["y"], pixel["x"]
        height, width = candidate_frame.shape[:2]
        y0, y1 = max(0, y - 16), min(height, y + 17)
        x0, x1 = max(0, x - 16), min(width, x + 17)
        crop_path = output / f"{kind}-crop-triptych.png"
        save_triptych((candidate_frame[y0:y1, x0:x1], reference_frame[y0:y1, x0:x1], difference[y0:y1, x0:x1]),
                      ("candidate crop", "reference crop", "abs diff [0,255]"), crop_path)
        artifacts[kind] = {"frame_triptych": str(full_path), "crop_triptych": str(crop_path),
                           "display_difference": "absolute RGB8 difference, fixed 0..255; no amplification or auto-normalization",
                           "crop": {"y0": y0, "y1": y1, "x0": x0, "x1": x1}}
    return artifacts


def run_quality(store):
    store.data["phases"]["quality"] = "RUNNING"; store.save()
    with store.journal.periodic():
        for row in store.data["quality_rows"]:
            if row["status"] in ("EVALUATED", "FAILED"):
                continue
            call = {"quality_id": row["quality_id"], "status": "RUNNING", "started_at": time.time()}
            store.data["quality_calls"].append(call); row.update(status="RUNNING", reason=None); store.save()
            try:
                source = _joint_row(_source_joint(store), row["group"], row["case_id"])
                candidate = source["media"]["received_path"]
                reference = _quality_reference_path(store, row["case_id"], row["reference"])
                with unit_attempt(store, "quality/" + row["quality_id"], operation="post_mp4_cpu_quality"):
                    metrics = _quality_metrics(candidate, reference, row["group"])
                    record = store.output / "quality" / row["group"].lower() / row["case_id"] / (row["reference"].lower() + ".json")
                    visuals = _save_quality_peak_visuals(candidate, reference, row["group"], metrics["peaks"],
                                                         record.parent / (row["reference"].lower() + "-peaks"))
                    for kind, artifact in visuals.items():
                        metrics["peaks"][kind]["visuals"] = artifact
                    atomic_json(record, {"quality_id": row["quality_id"], "candidate_semantics": "ACTUAL_WM_RECEIVED_RGB",
                                         "reference_semantics": row["reference"], "quality_gate": None,
                                         "difference_display_range": "fixed 0..255 absolute RGB8; no auto-normalization",
                                         **metrics})
                row.update(status="EVALUATED", reason=None, record=str(record), full_clip=metrics["full_clip"],
                           peaks=metrics["peaks"], temporal_aggregate=metrics["temporal"]["aggregate"])
                call.update(status="COMPLETE", finished_at=time.time())
            except BaseException as exc:
                terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                row.update(status=terminal, reason=reason(exc)); call.update(status=terminal, reason=reason(exc), finished_at=time.time())
                if not isinstance(exc, Exception):
                    raise
            finally:
                store.save()
    store.data["phases"]["quality"] = "COMPLETE"; store.save(); report(store)
    return store


def run_media(store, *, load_rgb=None, apply_edit=None):
    from experiments.paper_results_v1.attack_matrix import apply_edit_rgb8
    load_rgb = load_rgb or _load_rgb_path; apply_edit = apply_edit or apply_edit_rgb8
    store.data["phases"]["media"] = "RUNNING"; store.save()
    with store.journal.periodic():
        for case in CASES:
            clean = _row(store, "OFF_K0", "CLEAN_MP4", case)
            artifact = clean["media"].get("source_artifact", {})
            if clean["media"]["status"] not in ("AVAILABLE", "FAILED", "MISSING"):
                try:
                    path = Path(artifact.get("path", ""))
                    if artifact.get("status") != "AVAILABLE" or not path.is_file() or path.stat().st_size != math.prod(RGB_SHAPE):
                        raise FileNotFoundError(f"saved OFF_NATIVE POST unreadable: {path}")
                    clean["media"].update(status="AVAILABLE", received_path=str(path), source_mp4=artifact.get("mp4", {}).get("path"),
                                          reuse="CHECKPOINT_OFF_NATIVE_POST_READBACK", reason=None)
                    clean.update(status="MEDIA_READY", reason=None)
                except Exception as exc:
                    clean["media"].update(status="FAILED", reason=reason(exc)); clean.update(status="FAILED", reason=reason(exc))
                store.save()
            attacked = _row(store, "OFF_K0", "DELETE_G0", case)
            if attacked["media"]["status"] in ("AVAILABLE", "FAILED", "MISSING"):
                continue
            if clean["media"]["status"] != "AVAILABLE":
                attacked["media"].update(status="FAILED", reason="clean OFF_NATIVE POST unavailable")
                attacked.update(status="FAILED", reason=attacked["media"]["reason"]); store.save(); continue
            call = {"slot_id": attacked["slot_id"], "status": "RUNNING", "started_at": time.time()}
            store.data["edit_calls"].append(call); attacked["media"].update(status="RUNNING", reason=None); store.save()
            try:
                with unit_attempt(store, "media/" + attacked["slot_id"], operation="fixed_delete_g0"):
                    rgb = load_rgb(clean["media"]["received_path"], RGB_SHAPE)
                    edited, receipt = apply_edit(rgb, delete_spec())
                    saved = _atomic_rgb8(store.output / "media/off_k0" / case / "delete_g0_received.rgb8", edited, DELETE_SHAPE)
                attacked["media"].update(status="AVAILABLE", reason=None, received_path=saved["path"],
                                          source_mp4=clean["media"].get("source_mp4"), edit_receipt=receipt)
                attacked.update(status="MEDIA_READY", reason=None); call.update(status="COMPLETE", finished_at=time.time())
            except BaseException as exc:
                terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                attacked["media"].update(status=terminal, reason=reason(exc)); attacked.update(status=terminal, reason=reason(exc))
                call.update(status=terminal, reason=reason(exc), finished_at=time.time())
                if not isinstance(exc, Exception): raise
            finally:
                store.save()
    store.data["phases"]["media"] = "COMPLETE"; store.save(); report(store)
    return store


def _control_framewise_path(store, row):
    return store.output / "latents/framewise" / row["arm"].lower() / row["group"].lower() / (row["case_id"] + ".npy")


def _default_framewise_loader(store):
    from experiments.paper_results_v1.real_backends import load_local_framewise_backend
    return load_local_framewise_backend(read_json(store.data["effective_config"])["models"]["framewise"])


def run_clock(store, *, load_backend=None, load_rgb=None, select_clock=None):
    load_backend = load_backend or (lambda: _default_framewise_loader(store))
    load_rgb = load_rgb or _load_rgb_path; select_clock = select_clock or _select_clock
    candidates = [row for row in store.data["rows"] if row["arm"] in NEW_ARMS and row["media"]["status"] == "AVAILABLE"
                  and row["clock"]["status"] not in ("CLOCK_READY", "UNRESOLVED", "UNSUPPORTED", "FAILED")]
    store.data["phases"]["clock"] = "RUNNING"; store.save(); backend = None
    with store.journal.periodic():
        try:
            for row in candidates:
                encode_call = selector_call = None
                try:
                    frames = row["received_frames"]
                    if row["arm"] == "WM_K1":
                        latent_path = Path(row["clock"].get("source_framewise_latent", ""))
                        latent = _load_npy(latent_path, frames)
                        latent_reuse = "SOURCE_WM_K0_RECEIVER_VISIBLE_FRAMEWISE_LATENT"
                    else:
                        latent_path = _control_framewise_path(store, row); latent_reuse = None
                        if latent_path.is_file():
                            latent = _load_npy(latent_path, frames)
                        else:
                            if backend is None:
                                load_record = {"status": "RUNNING", "started_at": time.time()}
                                store.data["framewise_loads"].append(load_record); store.save()
                                try: backend = load_backend()
                                except BaseException as exc:
                                    load_record.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                                       reason=reason(exc), finished_at=time.time()); store.save(); raise
                                load_record.update(status="COMPLETE", finished_at=time.time()); store.save()
                            rgb = load_rgb(row["media"]["received_path"], _shape(row["group"]))
                            encode_call = {"slot_id": row["slot_id"], "frames": frames, "status": "RUNNING", "started_at": time.time()}
                            store.data["framewise_encode_clips"].append(encode_call); store.save()
                            def batch_observer(event, start, stop, error):
                                if event == "STARTED":
                                    item = {"slot_id": row["slot_id"], "start": start, "stop": stop, "status": "RUNNING", "started_at": time.time()}
                                    store.data["framewise_physical_batches"].append(item)
                                else:
                                    item = store.data["framewise_physical_batches"][-1]
                                    item.update(status="COMPLETE" if event == "RETURNED" else ("FAILED" if isinstance(error, Exception) else "INTERRUPTED"),
                                                finished_at=time.time())
                                    if error is not None: item["reason"] = reason(error)
                                store.save()
                            try:
                                with unit_attempt(store, "clock-encode/" + row["slot_id"], operation="framewise_clock_encode"):
                                    latent = _load_npy_from_value(backend.encode(rgb, batch_observer=batch_observer), frames)
                                    _atomic_npy(latent_path, latent)
                            except BaseException as exc:
                                encode_call.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                                   reason=reason(exc), finished_at=time.time())
                                store.save(); raise
                            encode_call.update(status="COMPLETE", finished_at=time.time()); store.save()
                    key = store.data["keys"][row["key_label"]]
                    selector_call = {"slot_id": row["slot_id"], "key_label": row["key_label"], "status": "RUNNING", "started_at": time.time()}
                    store.data["clock_calls"].append(selector_call); store.save()
                    evidence, selection = select_clock(latent, key)
                    folder = store.output / "clock" / row["arm"].lower() / row["group"].lower() / row["case_id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    import numpy as np
                    scores = folder / "scores.npz"
                    with scores.with_suffix(".npz.tmp").open("wb") as stream:
                        np.savez_compressed(stream, signed_projection=evidence["signed_projection"], rho=evidence["rho"])
                        stream.flush(); os.fsync(stream.fileno())
                    os.replace(scores.with_suffix(".npz.tmp"), scores)
                    record = folder / "blind_selection.json"
                    atomic_json(record, {"schema": SCHEMA, "slot_id": row["slot_id"], "key_label": row["key_label"],
                                         "receiver": "CENTERED_U", "truth_inputs": False, "message_inputs": False,
                                         "attack_inputs": False, "estimate": selection["estimate"],
                                         "operation": selection["operation"], "score_sidecar": str(scores)})
                    operation = selection["operation"]
                    status = "CLOCK_READY" if operation is not None and operation.get("status") == "SUPPORTED" else (
                        "UNSUPPORTED" if operation is not None else "UNRESOLVED")
                    row["clock"].update(status=status, reason=selection.get("reason") if status != "CLOCK_READY" else None,
                                        selector_record=str(record), latent_path=str(latent_path), latent_reuse=latent_reuse,
                                        selector_status=selection["estimate"]["status"], score_gap=selection["estimate"].get("score_gap"),
                                        output_frames=operation.get("output_frames") if operation else None,
                                        wan_support=operation.get("wan_support") if operation else None)
                    row.update(status=status, reason=row["clock"].get("reason")); selector_call.update(status="COMPLETE", finished_at=time.time())
                except BaseException as exc:
                    terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                    row["clock"].update(status=terminal, reason=reason(exc)); row.update(status=terminal, reason=reason(exc))
                    for call in (encode_call, selector_call):
                        if call is not None and call.get("status") == "RUNNING":
                            call.update(status=terminal, reason=reason(exc), finished_at=time.time())
                    for batch in store.data["framewise_physical_batches"]:
                        if batch.get("slot_id") == row["slot_id"] and batch.get("status") == "RUNNING":
                            batch.update(status=terminal, reason=reason(exc), finished_at=time.time())
                    if not isinstance(exc, Exception): raise
                finally:
                    store.save()
        finally:
            if backend is not None: backend.close()
    store.data["phases"]["clock"] = "COMPLETE"; store.save(); report(store)
    return store


def _source_k0_wan_latent(store, row, operation):
    if row["arm"] != "WM_K1":
        return None
    selector = Path(row["clock"].get("source_k0_selector_record", ""))
    if not selector.is_file():
        return None
    source_operation = read_json(selector).get("operation")
    if not isinstance(source_operation, dict) or source_operation != operation:
        return None
    path = Path(store.data["source_joint_run"]) / "run_state/latents/wan" / row["group"].lower() / (row["case_id"] + ".pt")
    if not path.is_file():
        store.data["wan_cache_repairs"].append({
            "slot_id": row["slot_id"], "status": "MISSING_REENCODE", "path": str(path),
            "reason": "equal-operation source K0 Wan latent is missing; perform fresh encode",
            "finished_at": time.time(),
        })
        store.save()
        return None
    try:
        _load_torch(path, operation["output_frames"])
    except Exception as exc:
        store.data["wan_cache_repairs"].append({
            "slot_id": row["slot_id"], "status": "INVALID_REENCODE", "path": str(path),
            "reason": reason(exc), "finished_at": time.time(),
        })
        store.save()
        return None
    return path


def _control_wan_path(store, row):
    return store.output / "latents/wan" / row["arm"].lower() / row["group"].lower() / (row["case_id"] + ".pt")


def _default_wan_loader(store):
    from runtime.wan.generation import load_frozen_vae
    config = read_json(store.data["effective_config"])
    return load_frozen_vae({"model": {"id": config["models"]["wan"]["local_snapshot_path"]}},
                           device=config["models"]["wan"].get("device", "cuda"))


def _save_payload(store, row, result, attempt):
    import numpy as np
    root = store.output / "payload" / row["arm"].lower() / row["group"].lower() / row["case_id"]
    root.mkdir(parents=True, exist_ok=True)
    value = dict(result); signed = value.pop("signed_votes"); zeros = value.pop("zero_mask")
    sidecar = root / f"votes-attempt-{attempt}.npz"
    with sidecar.with_suffix(".npz.tmp").open("wb") as stream:
        np.savez_compressed(stream, signed_votes=signed, zero_mask=zeros); stream.flush(); os.fsync(stream.fileno())
    os.replace(sidecar.with_suffix(".npz.tmp"), sidecar)
    record = root / f"read-attempt-{attempt}.json"
    atomic_json(record, {**value, "key_label": row["key_label"],
                         "vote_sidecar": {"path": str(sidecar), "arrays": ["signed_votes", "zero_mask"]}})
    row["payload"].update(status="READ", reason=None, read_record=str(record), decoded_bits=value["decoded_bits"],
                          votes=value["votes"], R=value["R"], output_frames=value["output_frames"],
                          original_reader_match=value["original_reader_match"])
    row.update(status="READ", reason=None, missing_bits=0)


def run_wan(store, *, load_vae=None, encode=None, payload_read=None, load_rgb=None):
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import operate_map
    load_vae = load_vae or (lambda: _default_wan_loader(store)); encode = encode or _default_wan_encode
    payload_read = payload_read or _default_payload_read; load_rgb = load_rgb or _load_rgb_path
    candidates = [row for row in store.data["rows"] if row["arm"] in NEW_ARMS and row["clock"]["status"] == "CLOCK_READY"
                  and row["payload"]["status"] not in ("READ", "FAILED")]
    store.data["phases"]["wan"] = "RUNNING"; store.save(); vae = None
    with store.journal.periodic():
        try:
            for row in candidates:
                encode_call = read_call = None
                try:
                    operation = read_json(row["clock"]["selector_record"]).get("operation")
                    if not isinstance(operation, dict) or operation.get("status") != "SUPPORTED":
                        raise ValueError("durable blind selection lacks supported operation")
                    output_frames = operation["output_frames"]
                    latent_path = _control_wan_path(store, row); reuse_path = _source_k0_wan_latent(store, row, operation)
                    latent = None
                    if latent_path.is_file():
                        try:
                            latent = _load_torch(latent_path, output_frames)
                        except Exception as exc:
                            store.data["wan_cache_repairs"].append({
                                "slot_id": row["slot_id"], "status": "INVALID_REENCODE", "path": str(latent_path),
                                "reason": reason(exc), "finished_at": time.time(),
                            })
                            store.save()
                    if latent is None and reuse_path is not None:
                        latent = _load_torch(reuse_path, output_frames)
                        reuse = {"slot_id": row["slot_id"], "status": "COMPLETE", "source_latent": str(reuse_path),
                                 "semantics": "same received RGB and equal full blind operation maps", "finished_at": time.time()}
                        store.data["wan_reuses"].append(reuse); row["payload"]["wan_latent_reuse"] = reuse; store.save()
                    if latent is None:
                        if vae is None:
                            load_record = {"status": "RUNNING", "started_at": time.time()}
                            store.data["wan_loads"].append(load_record); store.save()
                            try: vae = load_vae()
                            except BaseException as exc:
                                load_record.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                                   reason=reason(exc), finished_at=time.time()); store.save(); raise
                            load_record.update(status="COMPLETE", finished_at=time.time()); store.save()
                        received = load_rgb(row["media"]["received_path"], _shape(row["group"]))
                        corrected = operate_map(received, operation["received_index_map"])
                        encode_call = {"slot_id": row["slot_id"], "input_frames": row["received_frames"],
                                       "output_frames": output_frames, "status": "RUNNING", "started_at": time.time()}
                        store.data["wan_encode_clips"].append(encode_call); store.save()
                        def observer(event, error):
                            if event == "RETURNED": encode_call.update(status="COMPLETE", finished_at=time.time())
                            elif event == "FAILED":
                                encode_call.update(status="FAILED" if isinstance(error, Exception) else "INTERRUPTED",
                                                   reason=reason(error), finished_at=time.time())
                            store.save()
                        try:
                            with unit_attempt(store, "wan-encode/" + row["slot_id"], operation="wan_encode_blind_corrected_rgb"):
                                latent = encode(vae, corrected, observer).detach().cpu().float().contiguous()
                                times = (output_frames - 1) // 4 + 1
                                if tuple(latent.shape) != (1, 16, times, 40, 64) or not bool(latent.isfinite().all()):
                                    raise ValueError("Wan encoder returned shape/nonfinite mismatch")
                                _atomic_torch(latent_path, latent)
                        except BaseException as exc:
                            encode_call.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                                               reason=reason(exc), finished_at=time.time())
                            store.save(); raise
                        if encode_call["status"] == "RUNNING": encode_call.update(status="COMPLETE", finished_at=time.time())
                        store.save()
                    read_call = {"slot_id": row["slot_id"], "key_label": row["key_label"], "status": "RUNNING", "started_at": time.time()}
                    store.data["payload_read_calls"].append(read_call)
                    attempt = sum(item.get("slot_id") == row["slot_id"] for item in store.data["payload_read_calls"]); store.save()
                    with unit_attempt(store, "payload-read/" + row["slot_id"], operation="wan_payload_read"):
                        result = payload_read(latent, store.data["keys"][row["key_label"]], output_frames,
                                              operation["source_coordinate_map"])
                        bits = result.get("decoded_bits")
                        if result.get("status") != "READ" or not isinstance(bits, list) or len(bits) != 32 or any(type(bit) is not int or bit not in (0, 1) for bit in bits):
                            raise ValueError("payload reader did not return 32 binary integer bits")
                        _save_payload(store, row, result, attempt)
                    read_call.update(status="COMPLETE", finished_at=time.time(), read_record=row["payload"]["read_record"])
                except BaseException as exc:
                    terminal = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
                    row["payload"].update(status=terminal, reason=reason(exc)); row.update(status=terminal, reason=reason(exc), missing_bits=32)
                    for call in (encode_call, read_call):
                        if call is not None and call.get("status") == "RUNNING":
                            call.update(status=terminal, reason=reason(exc), finished_at=time.time())
                    if not isinstance(exc, Exception): raise
                finally:
                    store.save()
        finally:
            vae = None
            import gc; gc.collect()
            try:
                import torch
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            except ImportError: pass
    store.data["phases"]["wan"] = "COMPLETE"; store.save(); report(store)
    return store


def _count(records):
    return dict(sorted(Counter(item.get("status", "UNKNOWN") for item in records).items()))


def _atomic_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def pending_model_units(store):
    result = []
    for row in store.data["rows"]:
        if row["arm"] not in NEW_ARMS:
            continue
        if row["media"]["status"] == "AVAILABLE" and row["clock"]["status"] in ("PENDING", "RUNNING", "INTERRUPTED"):
            if row["arm"] == "OFF_K0" and not _control_framewise_path(store, row).is_file():
                result.append("framewise/" + row["slot_id"])
            result.append("clock-selection/" + row["slot_id"])
        if row["clock"]["status"] == "CLOCK_READY" and row["payload"]["status"] in ("PENDING", "RUNNING", "INTERRUPTED"):
            result.append("wan-or-reuse-and-read/" + row["slot_id"])
    return result


def report(store):
    truth = store.data["payload_bits"]
    phases = store.data.get("phases", {})
    truth_joined = (phases.get("clock") == "COMPLETE" and phases.get("wan") == "COMPLETE"
                    and not pending_model_units(store))
    rows = []
    for source in store.data["rows"]:
        decoded = source["payload"].get("decoded_bits")
        observed = source["payload"].get("status") == "READ" and isinstance(decoded, list) and len(decoded) == 32
        compared = observed and truth_joined
        errors = sum(a != b for a, b in zip(decoded, truth)) if compared else None
        path_metrics = _posthoc_path_metrics(source) if truth_joined else {
            "status": "NOT_JOINED",
            "reason": "post-hoc truth requires complete clock and Wan phases with no pending model unit",
        }
        clock_exact = (
            path_metrics.get("status") == "EVALUATED_POSTHOC"
            and path_metrics.get("frames") == source.get("received_frames")
            and path_metrics.get("minimum_support_absolute_error_sum") == 0
            and path_metrics.get("weighted_absolute_error_sum") == 0
        ) if truth_joined else None
        row = {
            "slot_id": source["slot_id"], "arm": source["arm"], "group": source["group"], "case_id": source["case_id"],
            "key_label": source["key_label"], "control_role": source["control_role"],
            "status": "EVALUATED" if compared else "OBSERVED_NOT_JOINED" if observed else source["status"],
            "reason": source.get("reason"), "received_frames": source.get("received_frames"),
            "selector_status": source["clock"].get("selector_status"), "score_gap": source["clock"].get("score_gap"),
            "output_frames": source["payload"].get("output_frames", source["clock"].get("output_frames")),
            "wan_support": source["clock"].get("wan_support"), "R": source["payload"].get("R"),
            "original_reader_match": source["payload"].get("original_reader_match"),
            "planned_bits": 32, "missing_bits": 0 if observed else 32, "decoded_bits": decoded,
            "truth_join": "POSTHOC_AFTER_CLOCK_WAN_COMPLETE_NO_PENDING_MODEL_UNITS" if truth_joined else "NOT_JOINED",
            "bit_errors_against_known_probe": errors, "clock_path_error": path_metrics,
            "clock_truth_exact": clock_exact,
            "clock_truth_exact_interpretation": "descriptive post-hoc path equality only; not presence, FPR, or a gate",
            "interpretation": ("positive recovery" if source["arm"] == "WM_K0_REUSED" else
                               "wrong-key control; equality is not presence" if source["arm"] == "WM_K1" else
                               "unwatermarked absence control; equality is not an inserted payload success or presence decision"),
            "clock_record": source["clock"].get("selector_record"), "payload_record": source["payload"].get("read_record"),
            "wan_latent_reuse": source["payload"].get("wan_latent_reuse"),
        }
        if source["arm"] == "WM_K0_REUSED":
            row["exact_payload_recovery"] = errors == 0 if compared else None
        else:
            row["false_target_exact_match"] = errors == 0 if compared else None
        rows.append(row)
    arms = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm]; evaluated = [row for row in selected if row["status"] == "EVALUATED"]
        value = {"fixed_rows": 8, "fixed_bits": 256, "status_counts": _count(selected),
                 "evaluated_rows": len(evaluated), "missing_rows": 8 - len(evaluated),
                 "bit_errors_against_known_probe": (sum(row["bit_errors_against_known_probe"] for row in evaluated)
                                                     if truth_joined else None)}
        if arm == "WM_K0_REUSED":
            value.update(exact_payload_recoveries=(sum(row.get("exact_payload_recovery") is True for row in selected)
                                                   if truth_joined else None),
                         fixed_denominator_exact_payload_recovery_rate=(
                             sum(row.get("exact_payload_recovery") is True for row in selected) / 8 if truth_joined else None))
        else:
            value.update(false_target_exact_matches=(sum(row.get("false_target_exact_match") is True for row in selected)
                                                     if truth_joined else None),
                         fixed_denominator_false_target_match_rate=(
                             sum(row.get("false_target_exact_match") is True for row in selected) / 8 if truth_joined else None))
        arms[arm] = value
    groups = {}
    for group in GROUPS:
        grouped = {}
        for arm in ARMS:
            selected = [row for row in rows if row["group"] == group and row["arm"] == arm]
            evaluated = [row for row in selected if row["status"] == "EVALUATED"]
            value = {"fixed_rows": 4, "evaluated_rows": len(evaluated), "missing_rows": 4 - len(evaluated)}
            if arm == "WM_K0_REUSED":
                value.update(exact_payload_recoveries=(sum(row.get("exact_payload_recovery") is True for row in selected)
                                                       if truth_joined else None),
                             fixed_denominator_exact_payload_recovery_rate=(
                                 sum(row.get("exact_payload_recovery") is True for row in selected) / 4 if truth_joined else None))
            else:
                value.update(false_target_exact_matches=(sum(row.get("false_target_exact_match") is True for row in selected)
                                                         if truth_joined else None),
                             fixed_denominator_false_target_match_rate=(
                                 sum(row.get("false_target_exact_match") is True for row in selected) / 4 if truth_joined else None))
            grouped[arm] = value
        groups[group] = grouped
    all_evaluated = all(row["status"] == "EVALUATED" for row in rows)
    quality_complete = all(row["status"] in ("EVALUATED", "FAILED") for row in store.data["quality_rows"])
    terminal = store.data.get("phases", {}).get("wan") == "COMPLETE" and quality_complete and not pending_model_units(store)
    summary = {
        "schema": SCHEMA, "status": ("COMPLETE" if all_evaluated and quality_complete else
                                      "COMPLETE_WITH_RETAINED_FAILURES" if terminal else "IN_PROGRESS"),
        "fixed_denominator": store.data["fixed_denominator"], "truth_joined": truth_joined,
        "arms": arms, "groups": groups,
        "quality": {"fixed_pairs": 24, "status_counts": _count(store.data["quality_rows"]), "quality_gate": None},
        "controls_are_presence_threshold_free": True, "off_has_no_inserted_payload_truth": True,
        "claim_limit": "same four saved candidates and saved OFF siblings under clean MP4/delete_g0; controls and CPU quality only; no FPR, threshold, generalization, or scientific PASS",
        "actual_calls": {"codec": 0, "generation_or_dit": 0, "writer": 0, "quality": _count(store.data["quality_calls"]),
                         "cpu_off_delete_edits": _count(store.data["edit_calls"]), "framewise_loads": _count(store.data["framewise_loads"]),
                         "framewise_encode_clips": _count(store.data["framewise_encode_clips"]),
                         "framewise_physical_batches": _count(store.data["framewise_physical_batches"]),
                         "clock_selections": _count(store.data["clock_calls"]), "wan_loads": _count(store.data["wan_loads"]),
                         "wan_encode_clips": _count(store.data["wan_encode_clips"]), "wan_reuses": _count(store.data["wan_reuses"]),
                         "wan_cache_repairs": _count(store.data["wan_cache_repairs"]),
                         "payload_reads": _count(store.data["payload_read_calls"])},
        "pending_model_units": pending_model_units(store),
    }
    store.data["status"] = summary["status"]; store.save()
    atomic_json(store.output / "summary.json", summary)
    atomic_json(store.output / "evaluation_report.json", {"summary": summary, "rows": rows,
                                                            "quality_rows": store.data["quality_rows"],
                                                            "appendix": {"post_mp4_quality_pairs": store.data["quality_rows"]}})
    compact = [{k: v for k, v in row.items() if k not in ("decoded_bits", "wan_latent_reuse")} for row in rows]
    _atomic_csv(store.output / "control_rows.csv", compact)
    return summary


def record_external_failure(store, message):
    store.data["errors"].append({"at": time.time(), "phase": "external", "reason": str(message)})
    for phase, status in list(store.data["phases"].items()):
        if status == "RUNNING": store.data["phases"][phase] = "INTERRUPTED"
    for row in store.data["rows"]:
        for name in ("media", "clock", "payload"):
            if row[name].get("status") == "RUNNING": row[name].update(status="INTERRUPTED", reason=str(message))
    store.save(); return report(store)
