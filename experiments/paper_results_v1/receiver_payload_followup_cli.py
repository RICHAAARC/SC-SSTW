"""Standalone fixed receiver payload follow-up; prepare/report are CPU-only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.paper_results_v1 import receiver_payload_followup as followup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-run")
    parser.add_argument("--config")
    parser.add_argument("--temp-root")
    parser.add_argument("--phase", choices=("init", "prepare", "read", "report", "record-failure"), required=True)
    parser.add_argument("--reason", default="external child failure")
    args = parser.parse_args(argv)
    if args.phase == "init":
        if not args.source_run or not args.config:
            parser.error("init requires --source-run and --config")
        followup.initialize(args.source_run, args.output, args.config, args.temp_root)
    else:
        store = followup.Store(args.output)
        if args.phase == "record-failure":
            followup.record_external_failure(store, args.reason)
        else:
            getattr(followup, args.phase)(store)
    store = followup.Store(args.output)
    summary = followup.report(store)
    failures = [{k: row.get(k) for k in ("slot_id", "status", "reason")}
                for row in store.data["rows"] if row["status"] in ("MISSING", "FAILED")]
    print(json.dumps({"phase": args.phase, "summary": summary, "retained_failures": failures}, indent=2))
    return 2 if args.phase in ("prepare", "read") and failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
