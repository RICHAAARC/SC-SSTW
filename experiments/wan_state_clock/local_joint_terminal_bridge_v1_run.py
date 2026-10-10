"""Standalone fixed saved-terminal bridge CLI; no notebook or Git required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

from runtime.wan.local_joint_terminal_bridge_v1 import run


def source_identity(config_path):
    return dict(workspace=str(Path(__file__).resolve().parents[2]), config_path=str(config_path))


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
