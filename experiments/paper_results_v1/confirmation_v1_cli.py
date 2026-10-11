"""CLI for the fixed eight-source confirmation receiver."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_results_v1 import confirmation_v1 as method


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attack-output", type=Path)
    parser.add_argument("--temp-root", type=Path)
    parser.add_argument("--case-id", choices=method.CASES)
    parser.add_argument("--phase", required=True, choices=("plan", "init", "prepare", "read", "report", "record-failure"))
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    config = method.validate_config(method._read(args.config))
    if args.phase == "plan":
        value = method.build_plan(config); method._atomic_json(args.output, value)
    elif args.phase == "init":
        if args.attack_output is None:
            parser.error("--attack-output is required for init")
        value = method.initialize(args.attack_output, args.output, config, temp_root=args.temp_root).data
    else:
        store = method.ConfirmationStore(args.output, config, temp_root=args.temp_root)
        if args.phase in ("prepare", "read", "record-failure") and args.case_id is None:
            parser.error("--case-id is required for this phase")
        if args.phase == "prepare": value = method.prepare_case(store, args.case_id).data
        elif args.phase == "read": value = method.read_case(store, args.case_id).data
        elif args.phase == "record-failure":
            method.record_external_failure(store, args.case_id, args.reason or "external failure")
            value = store.data
        else: value = method.report(store)
    print(json.dumps({"phase": args.phase, "status": value.get("status", "COMPLETE")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
