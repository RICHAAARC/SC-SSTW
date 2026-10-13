"""Fixed same-map layer readout for four saved confirmation sources.

CPU reuse and reporting are separate from the only GPU phase.  The GPU phase
reads saved RGB8 PRE artifacts, keeps one Wan VAE alive, and performs at most
the twelve fixed encodes.  Missing artifacts remain rows in the denominator;
there is no media decode, generation, or codec fallback.
"""
from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

from experiments.paper_results_v1.resume_state import StateJournal, atomic_bytes, read_state, unit_attempt

SCHEMA = "paper-results-v1-layer-readout-v1"
SOURCE_RUN_ID = "20261012T032440550166Z-c3026cfb"
CASES = ("confirm_01", "confirm_02", "confirm_07", "confirm_03")
LAYERS = ("TERMINAL", "NATIVE_PRE", "RECON_PRE", "M05_PRE", "M05_POST")
ENCODE_LAYERS = ("NATIVE_PRE", "RECON_PRE", "M05_PRE")
IDENTITY_MAP = list(range(181))
ARTIFACT_IDS = {
    "TERMINAL": "{case}/PAYLOAD_NATIVE/TERMINAL",
    "NATIVE_PRE": "{case}/PAYLOAD_NATIVE/PRE",
    "RECON_PRE": "{case}/PAYLOAD_FRAMEWISE_RECON/PRE",
    "M05_PRE": "{case}/PAYLOAD_FRAMEWISE_M05/PRE",
    "M05_POST": "{case}/PAYLOAD_FRAMEWISE_M05/POST",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def reason(exc):
    return f"{type(exc).__name__}: {exc}"


def _fixed_slots():
    return [f"{case}/{layer}" for case in CASES for layer in LAYERS]


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.path = self.output / "layer_readout_state.json"
        self.data = read_state(self.path)
        if self.data.get("schema") != SCHEMA:
            raise ValueError("layer readout schema changed")
        slots = [row.get("slot_id") for row in self.data.get("rows", [])]
        if slots != _fixed_slots():
            raise ValueError("fixed four-source/five-layer denominator changed")
        self.journal = StateJournal(self.path)
        self.journal.seed_cache(self.data)

    def save(self):
        self.journal.save(self.data)

    def resolve(self, value, document=None):
        """Resolve ordinary saved locators after moving the complete source run."""
        raw = str(value)
        path = Path(raw)
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        if document is not None:
            base = Path(document).parent
            candidates.append(base / (path.name if path.is_absolute() else path))
        source = Path(self.data["source_run"])
        marker = "/" + source.name + "/"
        if marker in raw.replace("\\", "/"):
            suffix = raw.replace("\\", "/").split(marker, 1)[1]
            candidates.append(source / suffix)
        for folder in ("run_state", "confirmation_receiver", "attack_run"):
            token = f"/{folder}/"
            if token in raw.replace("\\", "/"):
                candidates.append(source / folder / raw.replace("\\", "/").split(token, 1)[1])
        if not path.is_absolute():
            candidates.append(source / path)
        seen = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"saved artifact unavailable: {value}")


def _source_state_path(source_run):
    for relative in ("run_state/run_state.json", "run_state.json"):
        candidate = Path(source_run) / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("source run_state.json is unavailable")


def initialize(source_run, output, config):
    output = Path(output).resolve()
    state_path = output / "layer_readout_state.json"
    if state_path.is_file():
        return Store(output)
    source_run = Path(source_run).resolve()
    source_state = _source_state_path(source_run)
    cfg = read_json(config)
    source_config_path = source_run / "effective_confirmation_config.json"
    source_cfg = read_json(source_config_path)
    key = cfg.get("keys", {}).get("K0")
    payload = cfg.get("payload_bits")
    if not isinstance(key, str) or not key or key != source_cfg.get("keys", {}).get("K0"):
        raise ValueError("K0 must equal the saved source run K0")
    if payload != source_cfg.get("payload_bits") or len(payload or []) != 32 or any(type(x) is not int or x not in (0, 1) for x in payload):
        raise ValueError("payload must equal the saved 32-bit binary payload")
    wan = cfg.get("models", {}).get("wan")
    if not isinstance(wan, dict) or not isinstance(wan.get("local_snapshot_path"), str):
        raise ValueError("Wan VAE model locator is required")
    rows = []
    for case in CASES:
        for layer in LAYERS:
            rows.append({
                "slot_id": f"{case}/{layer}", "case_id": case, "layer": layer,
                "artifact_id": ARTIFACT_IDS[layer].format(case=case),
                "key_label": "K0", "receiver": "CENTERED_U", "map": "IDENTITY_0_180",
                "status": "PENDING_ENCODE" if layer in ENCODE_LAYERS else "PENDING_REUSE",
                "reason": None, "planned_bits": 32, "missing_bits": 32,
            })
    state = {
        "schema": SCHEMA, "status": "INITIALIZED", "source_run": str(source_run),
        "source_state": str(source_state), "source_config": str(source_config_path.resolve()),
        "effective_config": str(Path(config).resolve()), "key": key, "wan": wan,
        "fixed_denominator": {
            "cases": list(CASES), "layers": list(LAYERS), "rows": 20,
            "planned_bits": 640, "reused_cpu_rows": 8, "new_wan_encode_rows": 12,
            "identity_map": "0..180", "key_label": "K0", "R": 44,
        },
        "planned_budget": {
            "wan_vae_loads_successful_run": 1, "wan_encode_forwards": 12,
            "wan_decode": 0, "vjp": 0, "generation_or_dit": 0,
            "framewise_model_loads": 0, "codec_or_media_calls": 0,
        },
        "rows": rows, "unit_attempts": {}, "vae_loads": [], "wan_encode_calls": [], "errors": [], "phases": {},
        "actual_other_model_calls": 0, "actual_decode_calls": 0,
        "actual_generation_calls": 0, "actual_codec_or_media_calls": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(state_path, state)
    store = Store(output)
    report(store)
    return store


def _artifact(store, row):
    source = read_json(store.data["source_state"])
    matches = [item for item in source.get("artifacts", []) if item.get("artifact_id") == row["artifact_id"]]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one source artifact {row['artifact_id']}; found {len(matches)}")
    item = matches[0]
    if item.get("status") != "AVAILABLE" or not isinstance(item.get("path"), str):
        raise FileNotFoundError(f"source artifact is not AVAILABLE: {row['artifact_id']}")
    return item


def _load_rgb8(store, row):
    import numpy as np
    import torch
    item = _artifact(store, row)
    if item.get("shape") != [181, 320, 512, 3] or item.get("dtype") != "uint8":
        raise ValueError(f"saved PRE RGB8 geometry changed: {row['artifact_id']}")
    path = store.resolve(item["path"])
    expected = 181 * 320 * 512 * 3
    if path.stat().st_size != expected:
        raise ValueError(f"saved PRE RGB8 byte count changed: {path.stat().st_size} != {expected}")
    pixels = np.fromfile(path, dtype=np.uint8)
    if pixels.size != expected:
        raise ValueError("saved PRE RGB8 is incomplete")
    return torch.from_numpy(pixels.reshape(181, 320, 512, 3))


def _probe_rgb8(store, row):
    item = _artifact(store, row)
    if item.get("shape") != [181, 320, 512, 3] or item.get("dtype") != "uint8":
        raise ValueError(f"saved PRE RGB8 geometry changed: {row['artifact_id']}")
    path = store.resolve(item["path"])
    expected = 181 * 320 * 512 * 3
    if path.stat().st_size != expected:
        raise ValueError(f"saved PRE RGB8 byte count changed: {path.stat().st_size} != {expected}")
    return path


def _load_terminal(store, row):
    import torch
    item = _artifact(store, row)
    path = store.resolve(item["path"])
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("normalized", value.get("latent"))
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 16, 46, 40, 64):
        raise ValueError("saved terminal must be [1,16,46,40,64]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("saved terminal contains nonfinite values")
    return value.float()


def _validate_result(result):
    import numpy as np
    if result.get("status") != "READ" or result.get("R") != 44 or result.get("output_frames") != 181:
        raise ValueError("readout must be the fixed R44 identity measurement")
    if result.get("truth_inputs") is not False or result.get("original_reader_match") is not True:
        raise ValueError("readout truth boundary or original-reader comparison failed")
    bits = result.get("decoded_bits")
    if not isinstance(bits, list) or len(bits) != 32 or any(x not in (0, 1) for x in bits):
        raise ValueError("readout must contain all 32 decoded bits")
    if not isinstance(result.get("votes"), list) or len(result["votes"]) != 32:
        raise ValueError("readout must contain all 32 vote summaries")
    signs = np.asarray(result.get("signed_votes"))
    zeros = np.asarray(result.get("zero_mask"))
    if signs.shape != (4, 44, 240) or zeros.shape != signs.shape:
        raise ValueError("raw vote sidecar must be [channel,time,coordinate]=[4,44,240]")
    if not np.isin(signs, (-1, 0, 1)).all() or not np.array_equal(signs == 0, zeros):
        raise ValueError("raw sign votes and zero mask are inconsistent")
    return signs.astype(np.int8, copy=False), zeros.astype(bool, copy=False)


def _atomic_npz(path, **arrays):
    import numpy as np
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _save_read(store, row, result, evidence_origin):
    signs, zeros = _validate_result(result)
    folder = store.output / "reads" / row["case_id"] / row["layer"]
    sidecar = folder / "votes.npz"
    _atomic_npz(sidecar, signed_votes=signs, zero_mask=zeros)
    document = {k: v for k, v in result.items() if k not in ("signed_votes", "zero_mask", "time_bit_rows", "key_id")}
    document.update({
        "schema": SCHEMA, "slot_id": row["slot_id"], "case_id": row["case_id"],
        "layer": row["layer"], "key_label": "K0", "receiver": "CENTERED_U",
        "source_coordinate_map": IDENTITY_MAP, "received_index_map": IDENTITY_MAP,
        "vote_sidecar": {"path": str(sidecar), "coordinate_order": "[payload_channel,receiver_latent,frequency_coordinate]"},
        "evidence_origin": evidence_origin,
    })
    detail = folder / "read.json"
    atomic_json(detail, document)
    row.update(
        status="READ", reason=None, missing_bits=0, decoded_bits=list(result["decoded_bits"]),
        votes=result["votes"], R=44, output_frames=181, detail_record=str(detail),
        vote_sidecar=str(sidecar), original_reader_match=True, evidence_origin=evidence_origin,
    )
    store.save()


def _source_receiver_rows(store):
    cached = getattr(store, "_source_receiver_rows_cache", None)
    if cached is not None:
        return cached
    source = Path(store.data["source_run"])
    candidates = (
        source / "confirmation_receiver/evaluation_report.json",
        source / "confirmation_receiver/confirmation_state.json",
        source / "evaluation_report.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        document = read_json(path)
        rows = document.get("receiver_rows", document.get("rows"))
        if isinstance(rows, list):
            store._source_receiver_rows_cache = (rows, path)
            return store._source_receiver_rows_cache
    raise FileNotFoundError("confirmation receiver rows are unavailable")


def _post_result(store, row):
    import numpy as np
    rows, source_document = _source_receiver_rows(store)
    matches = [item for item in rows if (
        item.get("case_id"), item.get("arm"), item.get("attack_id"),
        item.get("key_label"), item.get("receiver")
    ) == (row["case_id"], "PAYLOAD_FRAMEWISE_M05", "full", "K0", "CENTERED_U")]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one fixed M05 POST receiver row; found {len(matches)}")
    source_row = matches[0]
    operation = source_row.get("operation", {})
    if source_row.get("status") != "EVALUATED" or source_row.get("truth_inputs") is not False:
        raise ValueError("saved M05 POST row is not a blind evaluated read")
    if source_row.get("R") != 44 or operation.get("wan_support") != 44 or operation.get("output_frames") != 181:
        raise ValueError("saved M05 POST row is not R44 over 181 frames")
    if operation.get("source_coordinate_map") != IDENTITY_MAP or operation.get("received_index_map") != IDENTITY_MAP:
        raise ValueError("saved M05 POST row is not the identity map")
    detail_value = source_row.get("detail_record")
    if isinstance(detail_value, dict):
        detail_value = detail_value.get("path")
    if not isinstance(detail_value, str):
        raise FileNotFoundError("saved M05 POST logical detail path is missing")
    detail_path = store.resolve(detail_value, source_document)
    detail = read_json(detail_path)
    if detail.get("slot_id") != source_row.get("slot_id") or detail.get("source_coordinate_map") != IDENTITY_MAP:
        raise ValueError("saved logical M05 POST annotation is not the fixed identity read")
    physical_value = source_row.get("physical_read_record")
    if isinstance(physical_value, dict):
        physical_value = physical_value.get("path")
    if not isinstance(physical_value, str):
        physical_value = detail.get("physical_read_record")
    if not isinstance(physical_value, str):
        raise FileNotFoundError("saved M05 POST physical read path is missing")
    physical_path = store.resolve(physical_value, detail_path)
    physical = read_json(physical_path)
    if (
        physical.get("R") != 44 or physical.get("output_frames") != 181
        or physical.get("truth_inputs") is not False
        or physical.get("original_reader_match") is not True
    ):
        raise ValueError("saved M05 POST physical read has incompatible R44/blind semantics")
    if physical.get("decoded_bits") != source_row.get("decoded_bits") or physical.get("votes") != source_row.get("votes"):
        raise ValueError("saved logical and physical M05 POST reads disagree")
    sidecar_value = physical.get("vote_sidecar", {}).get("path")
    if not isinstance(sidecar_value, str):
        raise FileNotFoundError("saved M05 POST raw vote sidecar path is missing")
    sidecar = store.resolve(sidecar_value, physical_path)
    with np.load(sidecar, allow_pickle=False) as arrays:
        signs = arrays["signed_votes"].copy()
        zeros = arrays["zero_mask"].copy()
    result = {
        "status": "READ", "R": source_row["R"], "output_frames": operation["output_frames"],
        "decoded_bits": source_row["decoded_bits"], "votes": source_row["votes"],
        "bit_rows": physical.get("bit_rows", source_row.get("bit_rows", [])),
        "signed_votes": signs, "zero_mask": zeros,
        "original_reader_match": source_row.get("original_reader_match"), "truth_inputs": False,
    }
    return result, {
        "kind": "REUSED_SAVED_M05_POST_IDENTITY_READ", "source_slot": source_row.get("slot_id"),
        "source_detail": str(detail_path), "source_physical_read": str(physical_path),
        "source_vote_sidecar": str(sidecar),
    }


def _mark_failure(store, row, exc, phase):
    row.update(status="MISSING" if isinstance(exc, FileNotFoundError) else "FAILED", reason=reason(exc))
    row.setdefault("resume_history", []).append({"at": time.time(), "phase": phase, "reason": reason(exc)})
    store.data.setdefault("errors", []).append({"at": time.time(), "slot_id": row["slot_id"], "phase": phase, "reason": reason(exc)})
    store.save()


def _restore_durable_read(store, row):
    if not row.get("detail_record") or not row.get("vote_sidecar"):
        return False
    try:
        detail = read_json(row["detail_record"])
        import numpy as np
        with np.load(row["vote_sidecar"], allow_pickle=False) as arrays:
            result = dict(detail, signed_votes=arrays["signed_votes"].copy(), zero_mask=arrays["zero_mask"].copy())
        _validate_result(result)
    except Exception:
        return False
    row.update(status="READ", reason=None, missing_bits=0)
    store.save()
    return True


def reuse(store, *, load_terminal=None, payload_read=None, post_result=None):
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import read_payload_general
    load_terminal = load_terminal or (lambda row: _load_terminal(store, row))
    payload_read = payload_read or read_payload_general
    post_result = post_result or (lambda row: _post_result(store, row))
    store.data["phases"]["reuse"] = "RUNNING"; store.save()
    try:
        with store.journal.periodic():
            for row in store.data["rows"]:
                if row["layer"] not in ("TERMINAL", "M05_POST") or row["status"] != "PENDING_REUSE":
                    continue
                if _restore_durable_read(store, row):
                    continue
                try:
                    if row["layer"] == "TERMINAL":
                        latent = load_terminal(row)
                        result = payload_read(latent, store.data["key"], 181, IDENTITY_MAP)
                        origin = {"kind": "REUSED_SAVED_PAYLOAD_NATIVE_TERMINAL", "artifact_id": row["artifact_id"]}
                    else:
                        result, origin = post_result(row)
                    _save_read(store, row, result, origin)
                except Exception as exc:
                    _mark_failure(store, row, exc, "reuse")
        store.data["phases"]["reuse"] = "COMPLETE"
    except BaseException as exc:
        store.data["phases"]["reuse"] = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
        store.data["errors"].append({"at": time.time(), "phase": "reuse", "reason": reason(exc)})
        raise
    finally:
        store.save(); report(store)
    return store


def _atomic_torch(path, tensor):
    import torch
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(tensor, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_latent(path):
    import torch
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != (1, 16, 46, 40, 64):
        raise ValueError("durable PRE latent must be [1,16,46,40,64]")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("durable PRE latent contains nonfinite values")
    return value.float()


def _latent_path(store, row):
    return store.output / "latents" / row["case_id"] / f"{row['layer']}.pt"


def pending_model_rows(store):
    """Return fixed slots which still require a new physical VAE encode.

    FAILED and MISSING are retained scientific rows, not implicit retry work.
    ENCODED rows are also excluded: their durable latent must be read or their
    missing durable output reported, never silently replaced by another encode.
    """
    pending = []
    for row in store.data["rows"]:
        if row["layer"] not in ENCODE_LAYERS:
            continue
        if row["status"] not in ("PENDING_ENCODE", "RUNNING", "INTERRUPTED"):
            continue
        if _latent_path(store, row).is_file() or (row.get("latent_path") and Path(row["latent_path"]).is_file()):
            continue
        pending.append(row["slot_id"])
    return pending


def read(store, *, load_vae=None, encode=None, payload_read=None, load_rgb=None, probe_rgb=None):
    from runtime.wan.generation import load_frozen_vae
    from runtime.wan.vae import reencode_rgb24_readback
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import read_payload_general
    load_vae = load_vae or (lambda: load_frozen_vae(
        {"model": {"id": store.data["wan"]["local_snapshot_path"]}},
        device=store.data["wan"].get("device", "cuda"),
    ))
    encode = encode or reencode_rgb24_readback
    payload_read = payload_read or read_payload_general
    load_rgb = load_rgb or (lambda row: _load_rgb8(store, row))
    probe_rgb = probe_rgb or (lambda row: _probe_rgb8(store, row))
    store.data["phases"]["read"] = "RUNNING"; store.save()
    vae = None
    try:
        with store.journal.periodic():
            todo = [row for row in store.data["rows"] if row["layer"] in ENCODE_LAYERS
                    and row["status"] in ("PENDING_ENCODE", "RUNNING", "INTERRUPTED", "ENCODED")]
            # A durable latent completes the expensive unit without another encode.
            needs_model = []
            for row in todo:
                latent_path = Path(row.get("latent_path") or _latent_path(store, row))
                if latent_path.is_file():
                    try:
                        row["latent_path"] = str(latent_path); store.save()
                        result = payload_read(_load_latent(latent_path), store.data["key"], 181, IDENTITY_MAP)
                        _save_read(store, row, result, {"kind": "NEW_SAVED_PRE_ENCODE", "artifact_id": row["artifact_id"], "latent_path": str(latent_path)})
                    except Exception as exc:
                        _mark_failure(store, row, exc, "durable-latent-read")
                else:
                    if row["status"] == "ENCODED":
                        _mark_failure(store, row, FileNotFoundError(f"durable encoded latent unavailable: {latent_path}"),
                                      "durable-latent-read")
                    else:
                        needs_model.append(row)
            # Validate each saved RGB8 before loading the VAE. Missing rows remain in the denominator.
            prepared = []
            for row in needs_model:
                try:
                    probe_rgb(row)
                    prepared.append(row)
                except Exception as exc:
                    _mark_failure(store, row, exc, "load-saved-pre-rgb8")
            if prepared:
                load_record = {"attempt": len(store.data["vae_loads"]) + 1, "status": "RUNNING", "started_at": time.time()}
                store.data["vae_loads"].append(load_record); store.save()
                try:
                    vae = load_vae()
                except BaseException as exc:
                    load_record.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED", finished_at=time.time(), reason=reason(exc))
                    store.save(); raise
                load_record.update(status="COMPLETE", finished_at=time.time()); store.save()
            for row in prepared:
                latent = rgb8 = None
                try:
                    rgb8 = load_rgb(row)
                    row.update(status="RUNNING", reason=None); store.save()
                    with unit_attempt(store, row["slot_id"], operation="wan_vae_encode_saved_pre_rgb8"):
                        call_record = None
                        def observe_encode(event, error):
                            nonlocal call_record
                            if event == "STARTED":
                                call_record = {"slot_id": row["slot_id"], "status": "RUNNING", "started_at": time.time()}
                                store.data["wan_encode_calls"].append(call_record)
                            elif event == "RETURNED" and call_record is not None:
                                call_record.update(status="COMPLETE", finished_at=time.time())
                            elif event == "FAILED" and call_record is not None:
                                call_record.update(status="FAILED" if isinstance(error, Exception) else "INTERRUPTED",
                                                   finished_at=time.time(), reason=reason(error))
                            else:
                                raise ValueError("invalid Wan encode observer transition")
                            store.save()
                        normalized = encode(vae, rgb8.float().div(255.0), encode_observer=observe_encode)
                        latent = normalized.detach().cpu().float().contiguous()
                        if tuple(latent.shape) != (1, 16, 46, 40, 64):
                            raise ValueError("PRE encode must produce [1,16,46,40,64]")
                        import torch
                        if not bool(torch.isfinite(latent).all()):
                            raise ValueError("PRE encode produced nonfinite values")
                        latent_path = _latent_path(store, row)
                        _atomic_torch(latent_path, latent)
                        row.update(status="ENCODED", latent_path=str(latent_path), encode_finished_at=time.time())
                        store.save()
                    result = payload_read(latent, store.data["key"], 181, IDENTITY_MAP)
                    _save_read(store, row, result, {"kind": "NEW_SAVED_PRE_ENCODE", "artifact_id": row["artifact_id"], "latent_path": row["latent_path"]})
                except Exception as exc:
                    _mark_failure(store, row, exc, "encode-or-read")
                finally:
                    normalized = latent = rgb8 = None
        store.data["phases"]["read"] = "COMPLETE"
    except BaseException as exc:
        store.data["phases"]["read"] = "FAILED" if isinstance(exc, Exception) else "INTERRUPTED"
        store.data["errors"].append({"at": time.time(), "phase": "read", "reason": reason(exc)})
        raise
    finally:
        vae = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            store.save(); report(store)
    return store


def _vote_metrics(sidecar, truth):
    import numpy as np
    with np.load(sidecar, allow_pickle=False) as arrays:
        signs = arrays["signed_votes"]
        zeros = arrays["zero_mask"]
        if signs.shape != (4, 44, 240) or not np.array_equal(signs == 0, zeros):
            raise ValueError("durable vote sidecar is inconsistent")
        result = []
        for bit in range(32):
            channel, slot = divmod(bit, 8)
            values = signs[channel, :, slot::8].reshape(-1)
            ones = int((values > 0).sum())
            nonpositive = int(values.size - ones)
            binary = (values > 0).astype(np.int8).tolist()
            decoded = int(Counter(binary).most_common(1)[0][0])
            aligned_margin = ((ones - nonpositive) if truth[bit] else (nonpositive - ones)) / int(values.size)
            result.append({
                "bit_index": bit, "payload_channel": channel, "slot_in_channel": slot,
                "truth": truth[bit], "decoded": decoded, "correct": decoded == truth[bit],
                "ones": ones, "nonpositive": nonpositive, "count": int(values.size),
                "coefficient_zeros": int((values == 0).sum()), "aligned_vote_margin": aligned_margin,
            })
    return result


def report(store):
    # This is the only truth/message join. Reuse and encode/read never inspect payload_bits.
    try:
        truth = read_json(store.data["source_config"])["payload_bits"]
        if len(truth) != 32 or any(x not in (0, 1) for x in truth):
            raise ValueError("source truth is not 32 binary bits")
    except Exception as exc:
        truth = None
        store.data["report_reason"] = reason(exc)
    compact = []
    bit_rows = []
    for row in store.data["rows"]:
        if row["status"] in ("READ", "EVALUATED") and truth is not None:
            try:
                metrics = _vote_metrics(row["vote_sidecar"], truth)
                decoded = [item["decoded"] for item in metrics]
                if decoded != row["decoded_bits"]:
                    raise ValueError(f"saved raw votes disagree with decoded bits: {row['slot_id']}")
                row.update(status="EVALUATED", missing_bits=0, bit_errors=sum(not item["correct"] for item in metrics),
                           exact_recovery=all(item["correct"] for item in metrics),
                           minimum_aligned_vote_margin=min(item["aligned_vote_margin"] for item in metrics))
                for item in metrics:
                    bit_rows.append({"slot_id": row["slot_id"], "case_id": row["case_id"], "layer": row["layer"], **item})
            except Exception as exc:
                previous = {key: row.get(key) for key in (
                    "status", "missing_bits", "bit_errors", "exact_recovery", "minimum_aligned_vote_margin",
                    "decoded_bits", "votes", "detail_record", "vote_sidecar",
                )}
                event = {"at": time.time(), "phase": "report-read", "reason": reason(exc), "previous": previous}
                row["previous_evaluation"] = previous
                row.setdefault("evaluation_history", []).append(event)
                row.update(status="MISSING" if isinstance(exc, (FileNotFoundError, OSError)) else "FAILED",
                           reason=reason(exc), missing_bits=32)
                for key in ("bit_errors", "exact_recovery", "minimum_aligned_vote_margin", "decoded_bits", "votes"):
                    row.pop(key, None)
                store.data.setdefault("errors", []).append({"at": event["at"], "slot_id": row["slot_id"],
                                                             "phase": "report-read", "reason": event["reason"]})
        compact.append({k: row.get(k) for k in (
            "slot_id", "case_id", "layer", "status", "reason", "artifact_id", "key_label", "receiver",
            "map", "R", "missing_bits", "bit_errors", "exact_recovery", "minimum_aligned_vote_margin",
            "detail_record", "vote_sidecar", "latent_path",
        )})
    status_counts = dict(Counter(row["status"] for row in store.data["rows"]))
    encode_attempts = [attempt for attempts in store.data.get("unit_attempts", {}).values() for attempt in attempts]
    physical_encodes = store.data.get("wan_encode_calls", [])
    by_case = {}
    earliest_wrong = []
    for case in CASES:
        case_rows = [row for row in store.data["rows"] if row["case_id"] == case]
        by_case[case] = [{k: row.get(k) for k in ("layer", "status", "bit_errors", "exact_recovery")}
                         for row in case_rows]
        for bit_index in range(32) if truth is not None else ():
            prior_complete = True
            for row in case_rows:
                if row["status"] != "EVALUATED":
                    prior_complete = False
                    continue
                if row["decoded_bits"][bit_index] != truth[bit_index]:
                    earliest_wrong.append({"case_id": case, "bit_index": bit_index, "layer": row["layer"],
                                           "prior_layers_complete": prior_complete})
                    break
    summary = {
        "schema": SCHEMA, "status": "COMPLETE" if status_counts == {"EVALUATED": 20} else "PARTIAL_WITH_RETAINED_STATUSES",
        "fixed_denominator": store.data["fixed_denominator"], "status_counts": status_counts,
        "evaluated_rows": status_counts.get("EVALUATED", 0),
        "evaluated_bits": status_counts.get("EVALUATED", 0) * 32,
        "bit_errors": sum(row.get("bit_errors", 0) for row in store.data["rows"] if row["status"] == "EVALUATED"),
        "exact_rows": sum(bool(row.get("exact_recovery")) for row in store.data["rows"] if row["status"] == "EVALUATED"),
        "slot_attempts": {"attempted": len(encode_attempts), "statuses": dict(Counter(x["status"] for x in encode_attempts))},
        "actual_wan_encode_calls": {"attempted": len(physical_encodes), "statuses": dict(Counter(x["status"] for x in physical_encodes))},
        "actual_vae_loads": {"attempted": len(store.data["vae_loads"]), "statuses": dict(Counter(x["status"] for x in store.data["vae_loads"]))},
        "actual_wan_decode": 0, "actual_vjp": 0, "actual_generation_or_dit": 0,
        "actual_framewise_model_loads": 0, "actual_codec_or_media_calls": 0,
        "truth_join": "REPORT_ONLY_AFTER_FIXED_IDENTITY_READS",
        "pending_model_rows": pending_model_rows(store),
        "by_case": by_case, "earliest_observed_wrong_bits": earliest_wrong,
    }
    store.data["status"] = summary["status"]
    store.save()
    atomic_json(store.output / "summary.json", summary)
    atomic_json(store.output / "evaluation_report.json", {"summary": summary, "rows": store.data["rows"], "bit_rows": bit_rows})
    fields = list(compact[0])
    with (store.output / "layer_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(compact)
    bit_fields = list(bit_rows[0]) if bit_rows else ["slot_id", "case_id", "layer", "bit_index"]
    with (store.output / "layer_bit_rows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=bit_fields, lineterminator="\n"); writer.writeheader(); writer.writerows(bit_rows)
    return summary


def record_external_failure(store, message):
    store.data.setdefault("errors", []).append({"at": time.time(), "phase": "external", "reason": str(message)})
    if store.data.get("phases", {}).get("read") == "RUNNING":
        store.data["phases"]["read"] = "INTERRUPTED"
    for row in store.data["rows"]:
        if row.get("status") == "RUNNING":
            row.update(status="INTERRUPTED", reason=str(message))
            row.setdefault("resume_history", []).append({"at": time.time(), "phase": "external", "reason": str(message)})
    store.save(); return report(store)
