"""CLI for a predeclared Paper Results V1 manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_results_v1.report import build_report, load_inputs, read_json, validate_manifest, write_report


def _result(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("--result requires RESULT_ID=PATH")
    result_id, path = value.split("=", 1)
    if not result_id or not path:
        raise argparse.ArgumentTypeError("--result requires nonempty RESULT_ID and PATH")
    return result_id, Path(path)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Left-join saved Video-WM outputs onto an explicit fixed-denominator manifest."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--result", action="append", default=[], type=_result, metavar="RESULT_ID=PATH")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = validate_manifest(read_json(args.manifest))
    report = build_report(manifest, load_inputs(args.result))
    write_report(report, args.output_dir)
    print(json.dumps({
        "report_status": report["report_status"],
        "slots": report["manifest_denominator"]["slots"],
        "slot_state_counts": report["manifest_denominator"]["slot_state_counts"],
        "output": str(args.output_dir.resolve()),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
