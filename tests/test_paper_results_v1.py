"""CPU/static checks for manifest-first paper reporting; no model or media run."""
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.paper_results_v1 import report


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "experiments/paper_results_v1"


def fixture_manifest():
    return report.read_json(FIXTURE / "synthetic_fixture.manifest.json")


def fixture_inputs():
    return report.load_inputs([("fixture", FIXTURE / "synthetic_fixture.result.json")])


def test_fixed_denominator_retains_failure_missing_excluded_unsupported_and_pair_gap():
    value = report.build_report(fixture_manifest(), fixture_inputs())
    assert value["report_status"] == "COMPLETE_WITH_RETAINED_ISSUES"
    assert value["manifest_denominator"]["slots"] == 6
    assert value["manifest_denominator"]["slot_state_counts"] == {
        "EXCLUDED": 1, "FAILED": 1, "MISSING": 1, "OBSERVED": 2, "UNSUPPORTED": 1,
    }
    assert value["paired_sync_effect"][0]["interpretation"] == "NO_BER_GAIN_OBSERVED"
    assert value["paired_sync_effect"][1]["state"] == "NOT_EVALUABLE"
    assert value["paired_sync_effect"][1]["reason"] == "RAW or SYNC member is not evaluable"
    assert value["measurements"][0]["state"] == "OBSERVED"
    assert value["measurements"][1]["state"] == "MISSING"
    excluded = next(row for row in value["conditional_recovery"] if row["source_id"] == "synthetic_source_3")
    unsupported = next(row for row in value["conditional_recovery"] if row["source_id"] == "synthetic_source_4")
    assert excluded["fixed_denominator_slots"] == 1 and excluded["eligible_recovery_slots"] == 0
    assert unsupported["fixed_denominator_slots"] == 1 and unsupported["eligible_recovery_slots"] == 0


def test_current_conditional_schema_adapter_keeps_blind_and_truth_outputs_separate():
    value = report.build_report(fixture_manifest(), fixture_inputs())
    raw_blind = next(row for row in value["blind_rows"] if row["slot_id"] == "fixture/raw")
    raw_truth = next(row for row in value["truth_rows"] if row["slot_id"] == "fixture/raw")
    assert "truth" not in raw_blind and "bit_errors" not in json.dumps(raw_blind["blind"])
    assert raw_blind["blind"]["physical_read"] == "synthetic-raw"
    assert "blind" not in raw_truth and raw_truth["truth"]["bit_errors"] == 0
    assert raw_truth["truth"]["bit_denominator"] == 32
    assert value["manifest_denominator"]["observed_unique_physical_key_read_identities"] == 1
    raw_group = next(row for row in value["conditional_recovery"] if row["source_id"] == "synthetic_source" and row["receiver_mode"] == "RAW")
    assert raw_group["fixed_bit_denominator"] == raw_group["eligible_bit_denominator"] == 32


def test_duplicate_manifest_and_duplicate_result_inputs_are_conflicts_not_dropped():
    manifest = fixture_manifest()
    manifest["slots"][1]["slot_id"] = manifest["slots"][0]["slot_id"]
    value = report.build_report(manifest, fixture_inputs())
    duplicates = [row for row in value["truth_rows"] if row["slot_id"] == "fixture/raw"]
    assert len(duplicates) == 2 and all(row["state"] == "CONFLICT" for row in duplicates)
    assert value["manifest_denominator"]["slots"] == 6

    inputs = report.load_inputs([
        ("fixture", FIXTURE / "synthetic_fixture.result.json"),
        ("fixture", FIXTURE / "synthetic_fixture.result.json"),
    ])
    value = report.build_report(fixture_manifest(), inputs)
    assert value["manifest_denominator"]["slot_state_counts"]["CONFLICT"] == 4
    assert len(value["input_records"]) == 2


def test_unplanned_saved_rows_are_retained_and_do_not_expand_denominator(tmp_path):
    saved = report.read_json(FIXTURE / "synthetic_fixture.result.json")
    saved["payload_reads"]["unplanned/K0/RAW"] = {
        "status": "READ", "mode": "RAW", "physical_read": "extra",
    }
    saved["posthoc"]["unplanned/K0/RAW"] = {
        "status": "EVALUATED_TRUTH", "key_role": "CORRECT_KEY", "bit_errors": 0,
    }
    saved["payload_reads"]["unplanned/K0/PAYLOAD_ONLY"] = {
        "status": "FAILED", "mode": "RAW", "error": "synthetic unplanned failure",
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    value = report.build_report(fixture_manifest(), report.load_inputs([("fixture", path)]))
    assert value["manifest_denominator"]["slots"] == 6
    assert value["unplanned_observations"] == [
        {
            "result_id": "fixture", "record_index": 0, "locator": "unplanned/K0/PAYLOAD_ONLY",
            "state": "UNPLANNED", "payload_exists": True, "payload_status": "FAILED",
            "posthoc_exists": False, "posthoc_status": None,
        },
        {
            "result_id": "fixture", "record_index": 0, "locator": "unplanned/K0/RAW",
            "state": "UNPLANNED", "payload_exists": True, "payload_status": "READ",
            "posthoc_exists": True, "posthoc_status": "EVALUATED_TRUTH",
        },
    ]


def test_32bit_contract_rejects_manifest_and_saved_payload_mismatches(tmp_path):
    for value in (True, 31, 64):
        manifest = fixture_manifest()
        manifest["comparability"]["message_length_bits"] = value
        with pytest.raises(report.ManifestError, match="message_length_bits"):
            report.validate_manifest(manifest)
        manifest = fixture_manifest()
        manifest["slots"][0]["planned_bits"] = value
        with pytest.raises(report.ManifestError, match="planned_bits"):
            report.validate_manifest(manifest)

    saved = report.read_json(FIXTURE / "synthetic_fixture.result.json")
    saved["payload_reads"]["global_00/K0/BASELINE"]["planned_final_bits"] = 64
    path = tmp_path / "wrong-bits.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    value = report.build_report(fixture_manifest(), report.load_inputs([("fixture", path)]))
    raw = next(row for row in value["truth_rows"] if row["slot_id"] == "fixture/raw")
    assert raw["state"] == "CONFLICT" and raw["truth"] == {}
    assert "planned_final_bits" in raw["reason"]


@pytest.mark.parametrize("mismatch", ["view", "input_id", "result_id"])
def test_pair_requires_same_declared_and_saved_condition_identity(tmp_path, mismatch):
    manifest = fixture_manifest()
    saved = report.read_json(FIXTURE / "synthetic_fixture.result.json")
    saved["posthoc"]["global_00/K0/BASELINE"]["bit_errors"] = 8
    if mismatch == "view":
        manifest["slots"][1]["view"] = "global_alt"
        saved["payload_reads"]["global_00/K0/EST_ALIGN"]["view_id"] = "global_alt"
    elif mismatch == "input_id":
        saved["payload_reads"]["global_00/K0/EST_ALIGN"]["input_id"] = "input_alt"
    else:
        manifest["result_bindings"]["fixture_alt"] = copy.deepcopy(manifest["result_bindings"]["fixture"])
        manifest["slots"][1]["result_id"] = "fixture_alt"
    path = tmp_path / "pair.json"
    path.write_text(json.dumps(saved), encoding="utf-8")
    inputs = [("fixture", path)]
    if mismatch == "result_id":
        inputs.append(("fixture_alt", path))
    value = report.build_report(manifest, report.load_inputs(inputs))
    pair = value["paired_sync_effect"][0]
    assert pair["state"] == "NOT_EVALUABLE" and "interpretation" not in pair
    assert "mismatch" in pair["reason"]


def test_malformed_and_nonfinite_inputs_are_retained_and_report_writes(tmp_path):
    malformed_cases = {
        "payload-container": lambda saved: saved.update(payload_reads=None),
        "posthoc-container": lambda saved: saved.update(posthoc=None),
        "payload-row": lambda saved: saved["payload_reads"].__setitem__("global_00/K0/BASELINE", None),
        "posthoc-row": lambda saved: saved["posthoc"].__setitem__("global_00/K0/BASELINE", None),
    }
    for name, mutate in malformed_cases.items():
        saved = report.read_json(FIXTURE / "synthetic_fixture.result.json")
        mutate(saved)
        malformed = tmp_path / f"{name}.json"
        malformed.write_text(json.dumps(saved), encoding="utf-8")
        value = report.build_report(fixture_manifest(), report.load_inputs([("fixture", malformed)]))
        raw = next(row for row in value["truth_rows"] if row["slot_id"] == "fixture/raw")
        assert raw["state"] == "CONFLICT" and "object" in raw["reason"]
        report.write_report(value, tmp_path / f"{name}-output")

    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    value = report.build_report(fixture_manifest(), report.load_inputs([("fixture", non_object)]))
    assert all(row["state"] == "FAILED" for row in value["truth_rows"][:3])
    report.write_report(value, tmp_path / "non-object-output")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"status":"SYNTHETIC_FIXTURE_ONLY","quality":{"x":NaN}}', encoding="utf-8")
    value = report.build_report(fixture_manifest(), report.load_inputs([("fixture", nonfinite)]))
    assert value["input_records"][0]["status"] == "FAILED"
    report.write_report(value, tmp_path / "nonfinite-output")


def test_historical_manifest_is_explicit_44_slots_and_separates_roles_and_oracle():
    manifest = report.read_json(FIXTURE / "historical_conditional_joint.manifest.json")
    report.validate_manifest(manifest)
    assert len(manifest["slots"]) == 44
    assert len({row["slot_id"] for row in manifest["slots"]}) == 44
    assert {row["key_label"] for row in manifest["slots"]} == {"K0", "K1"}
    assert {row["key_role"] for row in manifest["slots"]} == {"CORRECT_KEY", "WRONG_KEY"}
    assert sum(row["receiver_mode"] == "ORACLE" for row in manifest["slots"]) == 4
    assert len(manifest["pairs"]) == 9
    assert sum(pair["included"] for pair in manifest["pairs"]) == 8
    assert manifest["method"]["evidence_ceiling"].startswith("Historical same-source")


def test_cli_outputs_json_csv_markdown_from_no_git_source_copy(tmp_path):
    source = tmp_path / "source-copy"
    (source / "experiments").mkdir(parents=True)
    shutil.copy(ROOT / "experiments/__init__.py", source / "experiments/__init__.py")
    shutil.copytree(FIXTURE, source / "experiments/paper_results_v1")
    output = tmp_path / "out"
    interpreter = [sys.executable]
    if not os.access(sys.executable, os.X_OK):
        interpreter = ["/lib64/ld-linux-x86-64.so.2", sys.executable]
    completed = subprocess.run(
        interpreter + [
            "-m", "experiments.paper_results_v1.cli",
            "--manifest", "experiments/paper_results_v1/synthetic_fixture.manifest.json",
            "--result", "fixture=experiments/paper_results_v1/synthetic_fixture.result.json",
            "--output-dir", str(output),
        ],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not (source / ".git").exists()
    assert json.loads(completed.stdout)["slots"] == 6
    assert {path.name for path in output.iterdir()} == {
        "report.json", "report.md", "blind_rows.csv", "truth_rows.csv",
        "paired_sync_effect.csv", "measurements.csv",
    }
    with (output / "truth_rows.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 6 and {row["state"] for row in rows} >= {"FAILED", "MISSING", "OBSERVED"}
