"""Recoverable unit commits and bounded periodic snapshots; no identity gates.

A checkpoint contains the last committed state, not an in-flight CUDA kernel.
The timer copies the on-disk commit, never concurrently traverses mutable rows.
"""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import threading
import time
import warnings
from urllib.parse import quote

CHECKPOINT_SECONDS = 60
CHECKPOINT_UNITS = 32
SCIENTIFIC_TERMINAL = {"READ", "EVALUATED", "UNSUPPORTED", "UNRESOLVED", "NOT_EXECUTED_BY_NOTEBOOK"}


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _readable(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("state must be a JSON object")
    return value


def state_candidates(path):
    path = Path(path)
    others = [path.with_name(path.name + ".previous")]
    folder = path.parent / "checkpoints" / path.stem
    others += list(folder.glob("*.json")) if folder.is_dir() else []
    return [path, *sorted((p for p in others if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)]


def state_exists(path):
    return any(p.is_file() for p in state_candidates(path))


def read_state(path):
    problems = []
    for candidate in state_candidates(path):
        try:
            value = _readable(candidate)
            if not isinstance(value.get("phases"), dict):
                raise ValueError("run state phases must be an object")
        except (OSError, ValueError) as exc:
            problems.append({"path": str(candidate), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if problems:
            event = {"at": time.time(), "loaded_from": str(candidate), "problems": problems}
            value.setdefault("state_recovery_events", []).append(event)
            warnings.warn("Recovered readable run state: " + json.dumps(event), RuntimeWarning)
        # Unit commits newer than a recovered whole-state snapshot are replayed
        # independently. No contiguous sequence or identity comparison is used.
        units = Path(path).parent / "committed_units" / Path(path).stem
        snapshot_time = value.get("state_saved_at_ns", candidate.stat().st_mtime_ns)
        for collection, identity in UNIT_COLLECTIONS.items():
            rows = value.get(collection)
            if not isinstance(rows, list):
                continue
            index = {row[identity]: row for row in rows if identity in row}
            for receipt in (units / collection).glob("*.json"):
                if receipt.stat().st_mtime_ns < snapshot_time:
                    continue
                try:
                    committed = _readable(receipt)
                    if committed.get("saved_at_ns", receipt.stat().st_mtime_ns) < snapshot_time:
                        continue
                    row = committed.get("row", committed)
                    if not isinstance(row, dict):
                        raise ValueError("unit row must be an object")
                    key = row[identity]
                    if not isinstance(key, str) or not isinstance(row.get("status"), str):
                        raise ValueError("unit requires a string identifier and status")
                    if key not in index and collection != "physical_inputs":
                        raise ValueError("unknown unit ID outside the fixed plan: " + key)
                    required = set(index.get(key, {})) & {
                        "case_id", "cohort", "arm", "method", "attack_id", "key_label",
                        "receiver", "mode", "stage", "phase", "planned_bits", "missing_bits", "kind",
                        "candidate", "reference", "left", "right", "view_id", "protocol", "map_id",
                        "artifact_id", "received_index_map", "output_frames", "logical_slots", "key_reads",
                    }
                    if collection == "physical_inputs":
                        required |= {"artifact_id", "received_index_map", "output_frames", "logical_slots", "key_reads"}
                    missing = required - row.keys()
                    if missing:
                        raise ValueError("unit is missing core fields: " + ", ".join(sorted(missing)))
                    if collection == "artifacts" and row["status"] == "AVAILABLE" and not isinstance(row.get("path"), str):
                        raise ValueError("available artifact is missing its data path")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    warnings.warn(f"Unreadable unit commit {receipt}: {exc}", RuntimeWarning)
                    continue
                if key in index:
                    index[key].clear(); index[key].update(row)
                else:
                    rows.append(row)
                    index[key] = row
        for receipt in (units / "attempts").glob("*.json"):
            if receipt.stat().st_mtime_ns < snapshot_time:
                continue
            try:
                committed = _readable(receipt)
                if committed.get("saved_at_ns", receipt.stat().st_mtime_ns) < snapshot_time:
                    continue
                record = committed.get("row", committed)
                if not isinstance(record, dict):
                    raise ValueError("attempt record must be an object")
                if not isinstance(record.get("unit_id"), str) or not isinstance(record.get("attempts"), list):
                    raise ValueError("attempt commit requires unit_id and attempts list")
                if any(not isinstance(a, dict) or not isinstance(a.get("status"), str) for a in record["attempts"]):
                    raise ValueError("invalid attempt status record")
                value.setdefault("unit_attempts", {})[record["unit_id"]] = record["attempts"]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                warnings.warn(f"Unreadable attempt commit {receipt}: {exc}", RuntimeWarning)
        return value
    raise OSError("No readable run state/checkpoint: " + json.dumps(problems))


class StateJournal:
    def __init__(self, path, *, clock=time.monotonic, interval=CHECKPOINT_SECONDS):
        self.path = Path(path)
        self.clock, self.interval = clock, interval
        self.anchor, self.commits = clock(), 0
        self.lock = threading.RLock()
        self.row_cache = {}

    def seed_cache(self, value):
        # A new phase process must not rewrite thousands of unchanged Drive files.
        for collection, identity in UNIT_COLLECTIONS.items():
            for row in value.get(collection, []):
                if identity in row:
                    key = (collection, str(row[identity]))
                    if value.get("resume_format") == "unit-checkpoint-v1":
                        self.row_cache[key] = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
        if value.get("resume_format") == "unit-checkpoint-v1":
            for name, attempts in value.get("unit_attempts", {}).items():
                self.row_cache[("attempts", name)] = json.dumps({"unit_id": name, "attempts": attempts}, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()

    def save(self, value):
        with self.lock:
            save_time = time.time_ns()
            value["resume_format"] = "unit-checkpoint-v1"
            for collection, identity in UNIT_COLLECTIONS.items():
                for row in value.get(collection, []):
                    if identity not in row or (collection != "physical_inputs" and row.get("status") in ("PLANNED", "NOT_EXECUTED_BY_NOTEBOOK") and not row.get("resume_history")):
                        continue
                    key = (collection, str(row[identity]))
                    serialized = json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
                    if self.row_cache.get(key) != serialized:
                        unit_path = self.path.parent / "committed_units" / self.path.stem / collection / (quote(key[1], safe="") + ".json")
                        atomic_bytes(unit_path, json.dumps({"saved_at_ns": save_time, "row": row}, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
                        self.row_cache[key] = serialized
            for name, attempts in value.get("unit_attempts", {}).items():
                serialized = json.dumps({"unit_id": name, "attempts": attempts}, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
                if self.row_cache.get(("attempts", name)) != serialized:
                    atomic_bytes(self.path.parent / "committed_units" / self.path.stem / "attempts" / (quote(name, safe="") + ".json"), json.dumps({"saved_at_ns": save_time, "row": {"unit_id": name, "attempts": attempts}}, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
                    self.row_cache[("attempts", name)] = serialized
            value["state_saved_at_ns"] = save_time
            payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
            # Rotate only a readable prior state; a damaged primary cannot
            # overwrite the previous good recovery point.
            if self.path.is_file():
                try:
                    prior = self.path.read_bytes()
                    if not isinstance(json.loads(prior), dict):
                        raise ValueError("not a state object")
                except (OSError, ValueError):
                    pass
                else:
                    atomic_bytes(self.path.with_name(self.path.name + ".previous"), prior)
            atomic_bytes(self.path, payload)
            self.commits += 1
            if self.commits >= CHECKPOINT_UNITS or self.clock() - self.anchor >= self.interval:
                self.checkpoint("unit-or-time-interval")

    def checkpoint(self, reason):
        with self.lock:
            if not self.path.is_file():
                return
            value = _readable(self.path)
            folder = self.path.parent / "checkpoints" / self.path.stem
            latest, previous = folder / "latest.json", folder / "previous.json"
            if latest.is_file():
                try:
                    _readable(latest)
                except (OSError, ValueError):
                    pass
                else:
                    atomic_bytes(previous, latest.read_bytes())
            atomic_bytes(latest, self.path.read_bytes())
            counts = {key: dict(Counter(row.get("status", "UNKNOWN") for row in value[key]))
                      for key in ("rows", "artifacts", "receiver_rows", "baseline_rows", "quality_rows")
                      if isinstance(value.get(key), list)}
            progress = {"updated_at": time.time(), "reason": reason,
                        "state": str(self.path), "recoverable_checkpoint": str(latest),
                        "resume_granularity": "last durable unit; no unsaved CUDA/denoise-step state",
                        "status_counts": counts, "phases": value.get("phases", {})}
            atomic_bytes(self.path.parent / "progress.json", json.dumps(progress, ensure_ascii=False).encode())
            self.anchor, self.commits = self.clock(), 0

    @contextmanager
    def periodic(self):
        stop = threading.Event()
        def tick():
            while not stop.wait(self.interval):
                try:
                    self.checkpoint("60-second-last-committed-state")
                except Exception as exc:
                    warnings.warn(f"Checkpoint write failed: {type(exc).__name__}: {exc}", RuntimeWarning)
        worker = threading.Thread(target=tick, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join()
            self.checkpoint("phase-exit")


def remember(row, reason):
    snapshot = {key: copy.deepcopy(value) for key, value in row.items() if key != "resume_history"}
    row.setdefault("resume_history", []).append({"at": time.time(), "reason": reason, "prior": snapshot})


def retry_row(row, pending="PLANNED"):
    if row.get("status") in SCIENTIFIC_TERMINAL or row.get("status") in ("AVAILABLE", "EXTRACTED", "CLOCK_READY"):
        return False
    if row.get("status") != pending:
        remember(row, "resume missing technical unit")
        row.update(status=pending, reason=None)
    return True


@contextmanager
def unit_attempt(store, unit_id, **fields):
    attempts = store.data.setdefault("unit_attempts", {}).setdefault(unit_id, [])
    for old in attempts:
        if old.get("status") == "RUNNING":
            old.update(status="INTERRUPTED", previous_status="RUNNING", reason="process ended before a durable return")
    attempt = {"attempt": len(attempts) + 1, "status": "RUNNING", "started_at": time.time(), **fields}
    attempts.append(attempt)
    store.save()
    try:
        yield attempt
    except BaseException as exc:
        attempt.update(status="FAILED" if isinstance(exc, Exception) else "INTERRUPTED",
                       finished_at=time.time(), reason=f"{type(exc).__name__}: {exc}")
        raise
    else:
        attempt.update(status="COMPLETE", finished_at=time.time())
    finally:
        store.save()


def durable_artifact(row):
    """Status protects committed media from regeneration, even if local cache vanished."""
    return row.get("status") == "AVAILABLE"


# Dynamic groups must commit before logical rows which reference them.
UNIT_COLLECTIONS = {"physical_inputs": "physical_id", "artifacts": "artifact_id", "rows": "slot_id",
                    "receiver_rows": "slot_id", "baseline_rows": "slot_id",
                    "quality_rows": "quality_id"}
