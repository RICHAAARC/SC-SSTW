"""Ordinary one-attempt journal. No imports of models, code identity or governance."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    def invalid(value):
        raise ValueError("invalid JSON number: " + value)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=invalid)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def reserve_local(output):
    """Any prior artifact consumes this directory; a missing final JSON is not retry permission."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        return False
    output.mkdir(parents=True, exist_ok=True)
    with (output / "attempt.json").open("x", encoding="utf-8") as stream:
        json.dump(dict(status="RESERVED", consumed=True, reserved_at=now()), stream)
        stream.flush()
        os.fsync(stream.fileno())
    return True


class Journal:
    def __init__(self, output, planned):
        self.output = Path(output)
        self.rows = {row["id"]: dict(row) for row in planned}
        self.events = self.output / "events.jsonl"
        if self.events.exists():
            raise RuntimeError("An existing journal is read-only; no implicit resume/retry")
        write_json(self.output / "plan.json", list(self.rows.values()))

    def event(self, value):
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(time=now(), **value), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def call(self, key, function):
        row = self.rows[key]
        if row["status"] != "NOT_STARTED":
            raise RuntimeError("Call slot already consumed: " + key)
        row["status"] = "STARTED"
        self.event(dict(id=key, kind=row["kind"], status="STARTED"))
        started = time.monotonic()
        try:
            value = function()
        except BaseException as error:
            row.update(status="FAILED", error=f"{type(error).__name__}: {error}", seconds=time.monotonic()-started)
            self.event(row)
            raise
        row.update(status="RETURNED", seconds=time.monotonic()-started)
        self.event(row)
        return value

    def snapshot(self):
        write_json(self.output / "ledger.json", list(self.rows.values()))
        counts = dict(Counter(row["status"] for row in self.rows.values()))
        by_kind = {}
        for row in self.rows.values():
            by_kind.setdefault(row["kind"], Counter())[row["status"]] += 1
        return dict(status_counts=counts, physical_calls={k: dict(v) for k, v in by_kind.items()})


def recover(output):
    """Read-only crash accounting. Never starts a call or overwrites saved evidence."""
    output = Path(output)
    rows = {r["id"]: r for r in read_json(output / "plan.json")} if (output / "plan.json").is_file() else {}
    errors = []
    if (output / "events.jsonl").is_file():
        for i, line in enumerate((output / "events.jsonl").read_text().splitlines(), 1):
            try:
                event = json.loads(line)
                rows[event["id"]].update(event)
            except (ValueError, KeyError) as error:
                errors.append(dict(line=i, error=str(error)))
    for row in rows.values():
        if row["status"] == "STARTED":
            row["status"] = "INTERRUPTED_CONSUMED"
    result = dict(mode="READ_ONLY", consumed=output.exists() and any(output.iterdir()),
                  operations=list(rows.values()), read_errors=errors)
    for name in ("attempt.json", "result.json", "evaluation.json", "preparation_failure.json"):
        if (output / name).is_file():
            try:
                result[name] = read_json(output / name)
            except (OSError, ValueError) as error:
                errors.append(dict(file=name, error=str(error)))
    return result
