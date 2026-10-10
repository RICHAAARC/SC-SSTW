"""CLI for the independent fixed temporal-attack evaluation."""
from __future__ import annotations

import argparse
import json
import sys

from experiments.paper_results_v1.attack_eval import ATTEMPT_CASES, PHASES, record_external_failure, run_phase


def parser():
    value = argparse.ArgumentParser()
    value.add_argument("--config", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--phase", required=True, choices=("plan", "init", "record-failure", *PHASES))
    value.add_argument("--case-id", choices=ATTEMPT_CASES)
    value.add_argument("--failed-phase", choices=tuple(phase for phase in PHASES if phase != "evaluate"))
    value.add_argument("--reason")
    value.add_argument("--temp-root")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.phase not in ("plan", "init", "evaluate") and args.case_id is None:
        parser().error("--case-id is required for an execution phase")
    if args.phase == "record-failure":
        if not args.failed_phase or not args.reason:
            parser().error("record-failure requires --failed-phase and --reason")
        result = record_external_failure(
            args.config, args.output, args.failed_phase, args.case_id, args.reason,
            temp_root=args.temp_root,
        )
        print(json.dumps({"phase": args.phase, "case_id": args.case_id, "status": result["status"]}))
        return 0
    result = run_phase(
        args.config, args.output, args.phase, case_id=args.case_id, temp_root=args.temp_root,
    )
    print(json.dumps({"phase": args.phase, "case_id": args.case_id, "status": "COMPLETE", "result_type": type(result).__name__}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
