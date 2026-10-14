"""CLI for post-MP4 quality, wrong-key, and OFF controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.paper_results_v1 import residual_controls_v1 as run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-joint-run")
    parser.add_argument("--checkpoint-run")
    parser.add_argument("--config")
    parser.add_argument("--phase", choices=("init", "quality", "media", "clock", "wan", "report", "record-failure"), required=True)
    parser.add_argument("--reason", default="external child failure")
    args = parser.parse_args(argv)
    if args.phase == "init":
        if not args.source_joint_run or not args.checkpoint_run or not args.config:
            parser.error("init requires --source-joint-run, --checkpoint-run, and --config")
        store = run.initialize(args.source_joint_run, args.checkpoint_run, args.output, args.config)
    else:
        store = run.Store(args.output)
        if args.phase == "record-failure": run.record_external_failure(store, args.reason)
        elif args.phase == "report": run.report(store)
        else: getattr(run, "run_" + args.phase)(store)
    print(json.dumps({"phase": args.phase, "summary": run.report(run.Store(args.output))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
