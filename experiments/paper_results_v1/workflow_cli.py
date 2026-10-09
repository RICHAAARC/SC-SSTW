"""CLI for an explicit Paper Results V1 artifact workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_results_v1.report import build_report, load_inputs, read_json, validate_manifest, write_report
from experiments.paper_results_v1.workflow import (
    run_workflow,
    validate_workflow_manifest,
    write_workflow_report,
)


def _resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base / path


def _link_main_report(declaration, *, base_dir, output_dir):
    status = declaration.get("status")
    report_id = declaration.get("report_id", "UNDECLARED")
    if status != "DECLARED":
        return {"report_id": report_id, "status": status or "PENDING_NOT_DECLARED"}
    try:
        manifest_path = _resolve(base_dir, declaration["manifest_path"])
        main_manifest = validate_manifest(read_json(manifest_path))
        entries = []
        for item in declaration.get("results", []):
            entries.append((item["result_id"], _resolve(base_dir, item["path"])))
        main_report = build_report(main_manifest, load_inputs(entries))
        main_output = output_dir / "main_report"
        write_report(main_report, main_output)
        return {
            "report_id": report_id,
            "status": "LINKED",
            "report_status": main_report["report_status"],
            "schema_version": main_report["schema_version"],
            "manifest_denominator": main_report["manifest_denominator"],
            "output_dir": "main_report",
        }
    except Exception as exc:
        return {"report_id": report_id, "status": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Execute explicitly injected outer workflow callbacks over a fixed artifact plan."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--fixture-backends",
        action="store_true",
        help="Use deterministic CPU fixture callbacks; valid only for SYNTHETIC_FIXTURE_ONLY manifests.",
    )
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = validate_workflow_manifest(read_json(manifest_path))
    if args.fixture_backends and manifest["evidence_role"] != "SYNTHETIC_FIXTURE_ONLY":
        parser.error("--fixture-backends requires evidence_role SYNTHETIC_FIXTURE_ONLY")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist")
    args.output_dir.mkdir(parents=True)
    operations = {}
    native_adapters = {}
    if args.fixture_backends:
        from experiments.paper_results_v1.fixture_backends import (
            build_fixture_native_adapters,
            build_fixture_operations,
        )
        operations = build_fixture_operations()
        native_adapters = build_fixture_native_adapters()
    link = _link_main_report(
        manifest["main_report"], base_dir=manifest_path.parent, output_dir=args.output_dir,
    )
    workflow_report = run_workflow(
        manifest,
        operations=operations,
        native_adapters=native_adapters,
        base_dir=manifest_path.parent,
        main_report_link=link,
    )
    write_workflow_report(workflow_report, args.output_dir)
    print(json.dumps({
        "output": str(args.output_dir.resolve()),
        "report_status": workflow_report["report_status"],
        "denominator": workflow_report["manifest_denominator"],
        "main_report_status": link["status"],
    }, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
