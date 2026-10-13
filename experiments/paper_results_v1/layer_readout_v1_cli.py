"""CLI for the fixed Paper Results V1 four-source layer readout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.paper_results_v1 import layer_readout_v1 as run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--config")
    parser.add_argument("--phase", choices=("init", "reuse", "read", "report", "record-failure"), required=True)
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
        else:
            getattr(run, args.phase)(store)
    summary = run.report(run.Store(args.output))
    failures = [{k: row.get(k) for k in ("slot_id", "status", "reason")}
                for row in run.Store(args.output).data["rows"] if row["status"] in ("MISSING", "FAILED", "INTERRUPTED")]
    print(json.dumps({"phase": args.phase, "summary": summary, "retained_failures": failures}, indent=2))
    return 2 if args.phase in ("reuse", "read") and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
