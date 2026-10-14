"""CLI for the fixed residual clean-MP4/delete_g0 joint receiver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.paper_results_v1 import residual_joint_v1 as run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--config")
    parser.add_argument("--phase", choices=("init", "quality", "media", "clock", "wan", "report", "record-failure"), required=True)
    parser.add_argument("--reason", default="external child failure")
    args = parser.parse_args(argv)
    if args.phase == "init":
        if not args.source_run or not args.config:
            parser.error("init requires --source-run and --config")
        store = run.initialize(args.source_run, args.output, args.config)
    else:
        store = run.Store(args.output)
        if args.phase == "record-failure":
            run.record_external_failure(store, args.reason)
        elif args.phase == "report":
            run.report(store)
        else:
            getattr(run, "run_" + args.phase)(store)
    summary = run.report(run.Store(args.output))
    print(json.dumps({"phase": args.phase, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
