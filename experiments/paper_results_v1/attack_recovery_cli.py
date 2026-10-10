from __future__ import annotations

import argparse

from experiments.paper_results_v1 import attack_eval
from experiments.paper_results_v1 import attack_recovery


def main(argv=None):
    parser = argparse.ArgumentParser(description="Resume fixed two-pilot temporal attack evaluation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temp-root")
    parser.add_argument("--phase", required=True)
    parser.add_argument("--case-id")
    parser.add_argument("--reason")
    args = parser.parse_args(argv)
    if args.phase == "init":
        attack_recovery.initialize(args.config, args.source_run, args.output, temp_root=args.temp_root)
        return 0
    config = attack_eval.validate_config(attack_eval.read_json(args.config))
    store = attack_eval.AttackRunStore(args.output, config, temp_root=args.temp_root)
    if args.phase == "record-failure":
        attack_recovery.record_external_failure(store, args.reason.split(":", 1)[0], args.case_id, args.reason)
    elif args.phase == "evaluate":
        attack_recovery.evaluate(store, config)
    else:
        if args.case_id not in attack_eval.ATTEMPT_CASES:
            raise ValueError("recovery only attempts pilot_01/pilot_02")
        attack_recovery.run_case_phase(store, config, args.phase, args.case_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
