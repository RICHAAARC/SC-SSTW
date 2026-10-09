"""Independent staged CLI for the concrete Paper Results V1 real workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_results_v1.report import read_json
from experiments.paper_results_v1.real_eval import (
    PHASES,
    RunStore,
    build_plan,
    run_phase,
    static_preflight,
    validate_real_config,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute one explicit Paper Results V1 phase. Each model family runs "
            "in a separate process invocation. No weights are downloaded."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("preflight", "plan", "init", *PHASES))
    parser.add_argument("--case-id")
    args = parser.parse_args(argv)
    config = validate_real_config(read_json(args.config.resolve()))

    if args.phase == "preflight":
        report = static_preflight(config)
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "preflight.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "report": str(path.resolve())}, sort_keys=True))
        return 0 if report["status"] == "READY" else 2

    if args.phase == "plan":
        if args.case_id:
            parser.error("plan does not accept --case-id")
        plan = build_plan(config)
        report = {
            "schema_version": config["schema_version"],
            "study_id": config["study_id"],
            "adoption_status": config["adoption"]["status"],
            "fixed_denominator": {
                "cases": len(config["cases"]),
                "artifacts": len(plan["artifacts"]),
                "receiver_slots": len(plan["receiver_slots"]),
                "receiver_bits": 32 * len(plan["receiver_slots"]),
                "baseline_slots": len(plan["baseline_slots"]),
                "comparison_slots": len(plan["comparison_slots"]),
                "quality_rows": len(plan["quality_rows"]),
                "cost_rows": len(plan["costs"]),
            },
            **plan,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / "plan.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": "PLANNED", "report": str(path.resolve()), "fixed_denominator": report["fixed_denominator"]}, sort_keys=True))
        return 0

    if args.phase == "init":
        if args.case_id:
            parser.error("init does not accept --case-id")
        store = RunStore(args.output, config, create=True)
        print(json.dumps({
            "status": store.data["status"],
            "state": str(store.path),
            "fixed_denominator": {
                "cases": len(config["cases"]),
                "artifacts": len(store.data["artifacts"]),
                "receiver_slots": len(store.data["receiver_slots"]),
                "baseline_slots": len(store.data["baseline_slots"]),
                "comparison_slots": len(store.data["comparison_slots"]),
                "quality_rows": len(store.data["quality_rows"]),
                "cost_rows": len(store.data["costs"]),
            },
        }, sort_keys=True))
        return 0

    if not config["cases"]:
        parser.error("execution config has an empty case roster")
    if args.case_id and args.case_id not in {case["case_id"] for case in config["cases"]}:
        parser.error(f"unknown --case-id {args.case_id!r}")
    if args.phase == "evaluate":
        if args.case_id:
            parser.error("evaluate aggregates all cases and does not accept --case-id")
    elif not args.case_id:
        parser.error(f"{args.phase} requires --case-id")
    store = RunStore(args.output, config)
    try:
        result = run_phase(store, config, args.phase, case_id=args.case_id)
    except BaseException as exc:
        print(json.dumps({
            "status": "FAILED_RETAINED",
            "phase": args.phase,
            "case_id": args.case_id,
            "error": f"{type(exc).__name__}: {exc}",
            "state": str(store.path),
        }, sort_keys=True))
        return 1
    print(json.dumps({
        "status": "COMPLETE",
        "phase": args.phase,
        "case_id": args.case_id,
        "state": str(store.path),
        "report_status": result.get("status") if isinstance(result, dict) else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
