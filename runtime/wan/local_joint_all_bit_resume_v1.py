"""Durable complete-unit recovery for the adopted two-step All-Bit task.

The timer snapshots last committed JSON; it never reads a live autograd graph.
Tensor units live beside the snapshot on Drive. Temporary activation spools do not.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time
import warnings

INTERVAL_SECONDS = 60


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode()


def read_json(path):
    def invalid(value):
        raise ValueError("nonfinite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def atomic_tensor(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _valid_state(path):
    value = read_json(path)
    if not isinstance(value, dict) or not all(isinstance(value.get(k), dict) for k in ("views", "steps", "counts", "model_calls")):
        raise ValueError("incomplete result state")
    if not {"BASE","TWO_MEAN_SAVED","ALLBIT_HALF","ALLBIT_TWO"} <= value["views"].keys() or not {"step_1","step_2"} <= value["steps"].keys():
        raise ValueError("result is missing fixed method slots")
    return value


def read_state(output):
    output = Path(output)
    candidates = [output / "result.json", output / "result.json.previous",
                  output / "checkpoints/latest.json", output / "checkpoints/previous.json"]
    problems = []
    value = None
    for path in candidates:
        if not path.exists():
            continue
        try:
            value = _valid_state(path)
            break
        except (OSError, ValueError, TypeError) as exc:
            problems.append(dict(path=str(path), reason=repr(exc)))
    if value is None:
        if problems:
            raise OSError("No readable All-Bit state: " + json.dumps(problems))
        return None
    if problems:
        value.setdefault("recovery_events", []).append(dict(loaded_from=str(path), problems=problems))
    # A complete call receipt survives a torn whole-state snapshot. No sequence or
    # source identity is required; partial .tmp writes are never treated as commits.
    calls = {row["id"]: row for row in value.get("call_history", [])}
    for path in (output / "calls").glob("*.json"):
        try:
            row = read_json(path)
            if not isinstance(row.get("id"), str) or row.get("kind") not in ("vae_load", "decode", "decoder_vjp"):
                raise ValueError("invalid model-call receipt")
            calls[row["id"]] = row
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            value.setdefault("recovery_events", []).append(dict(path=str(path), reason=repr(exc)))
    value["call_history"] = sorted(calls.values(), key=lambda r: r["id"])
    recount(value)
    return value


def recount(value):
    if not value.get("call_history"):
        return
    calls = value["call_history"]
    for kind in ("vae_load", "decode", "decoder_vjp"):
        value["counts"][kind + "_attempted"] = sum(r["kind"] == kind for r in calls)
        value["counts"][kind + "_completed"] = sum(r["kind"] == kind and r["status"] == "RETURNED" for r in calls)
    value["ordinary_chunks"] = {phase: sum(r.get("ordinary_chunks", {}).get(phase, 0) for r in calls)
                                for phase in ("attempted", "completed")}
    for row in calls:
        value["model_calls"][row["unit"]] = {k: v for k, v in row.items() if k != "id"}
        if "replay" in row:
            value["replay"][row["unit"].removesuffix("_gradient_decode")] = row["replay"]


class Journal:
    def __init__(self, output, interval=INTERVAL_SECONDS):
        self.output = Path(output)
        self.interval = interval
        self.lock = threading.RLock()
        self.call_cache = {}

    def save(self, value):
        with self.lock:
            for row in value.get("call_history", []):
                payload = json_bytes(row)
                if self.call_cache.get(row["id"]) != payload:
                    atomic_bytes(self.output / "calls" / (row["id"] + ".json"), payload)
                    self.call_cache[row["id"]] = payload
            value["state_saved_at_unix"] = time.time()
            value["resume_granularity"] = "last complete durable unit; unfinished decode/VJP is recomputed"
            path = self.output / "result.json"
            if path.exists():
                try:
                    _valid_state(path)
                except (OSError, ValueError, TypeError):
                    pass
                else:
                    atomic_bytes(self.output / "result.json.previous", path.read_bytes())
            atomic_bytes(path, json_bytes(value))

    def checkpoint(self, reason):
        with self.lock:
            path = self.output / "result.json"
            if not path.exists():
                return
            value = _valid_state(path)
            latest = self.output / "checkpoints/latest.json"
            if latest.exists():
                try:
                    _valid_state(latest)
                except (OSError, ValueError, TypeError):
                    pass
                else:
                    atomic_bytes(self.output / "checkpoints/previous.json", latest.read_bytes())
            atomic_bytes(latest, path.read_bytes())
            progress = dict(updated_at_unix=time.time(), reason=reason,
                recoverable_state="checkpoints/latest.json", status=value["status"],
                views={name: row["status"] for name, row in value["views"].items()},
                steps={name: row["status"] for name, row in value["steps"].items()},
                active_calls=[r["unit"] for r in value.get("call_history", []) if r["status"] == "RUNNING"],
                actual_counts=value["counts"], resume_granularity=value.get("resume_granularity"))
            atomic_bytes(self.output / "progress.json", json_bytes(progress))

    @contextmanager
    def periodic(self):
        stop = threading.Event()
        def tick():
            while not stop.wait(self.interval):
                try:
                    self.checkpoint("60-second last committed state")
                except Exception as exc:
                    warnings.warn("Checkpoint write failed: " + repr(exc), RuntimeWarning)
        worker = threading.Thread(target=tick, daemon=True)
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join()
            self.checkpoint("attempt exit")
