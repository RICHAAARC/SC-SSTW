"""Standalone fixed saved-terminal bridge CLI; no notebook or Git required."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import sys

from runtime.wan.local_joint_terminal_bridge_v1 import run


def source_identity(config_path):
    """Describe actual imported repository source/config without an admission check."""
    root = Path(__file__).resolve().parents[2]
    paths = {Path(__file__).resolve(), config_path.resolve()}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(root) and path.suffix == ".py":
                paths.add(path)
    files, errors = {}, {}
    for path in sorted(paths):
        label = str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        try:
            files[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            errors[label] = repr(exc)
    return dict(kind="actual_imported_source_and_config", git_commit=None, blocking=False,
        files=files, errors=errors,
        content_sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, _frame):
        raise SystemExit(f"external signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        identity = source_identity(args.config)
    except Exception as exc:
        identity = dict(status="UNAVAILABLE", reason=repr(exc), blocking=False)
    result = run(json.loads(args.config.read_text(encoding="utf-8")), args.output, source_identity=identity)
    print(json.dumps(dict(status=result["status"], counts=result["counts"], output=str(args.output)), indent=2), flush=True)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
