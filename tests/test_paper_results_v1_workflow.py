"""CPU/static checks for native adapters and explicit artifact workflow."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.paper_results_v1 import report
from experiments.paper_results_v1.fixture_backends import (
    build_fixture_native_adapters,
    build_fixture_operations,
)
from experiments.paper_results_v1.native_adapters import (
    NativeAdapterError,
    RivaGANPathBackend,
    VideoSealLoadedBackend,
    VideoSealNativeAdapter,
)
from experiments.paper_results_v1.workflow import (
    WorkflowManifestError,
    _quality_record,
    run_workflow,
    validate_workflow_manifest,
)


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/paper_results_v1"


def workflow_manifest():
    return report.read_json(PACKAGE / "workflow_fixture.manifest.json")


def test_explicit_workflow_runs_callbacks_and_retains_fixed_denominators():
    value = run_workflow(
        workflow_manifest(),
        operations=build_fixture_operations(),
        native_adapters=build_fixture_native_adapters(),
        base_dir=PACKAGE,
        main_report_link={"status": "LINKED", "report_id": "unit"},
    )
    assert value["report_status"] == "COMPLETE_WITH_RETAINED_ISSUES"
    assert value["manifest_denominator"] == {
        "cases": 2,
        "source_artifacts": 2,
        "plan_steps": 16,
        "native_jobs": 4,
        "quality_pairs": 6,
        "cost_rows": 28,
        "source_state_counts": {"AVAILABLE": 1, "MISSING": 1},
        "plan_state_counts": {"BLOCKED_DEPENDENCY": 8, "SUCCEEDED": 8},
        "native_state_counts": {"BLOCKED_DEPENDENCY": 2, "SUCCEEDED": 2},
        "quality_state_counts": {"BLOCKED_DEPENDENCY": 3, "OBSERVED": 3},
    }
    assert len(value["artifacts"]) == 26
    assert {row["state"] for row in value["quality_records"]} == {"OBSERVED", "BLOCKED_DEPENDENCY"}
    observed = [row for row in value["quality_records"] if row["state"] == "OBSERVED"]
    assert len(observed) == 3
    assert all(row["metric_semantics"] == "PAIRWISE_ABSOLUTE_NOT_ADDITIVE_DECOMPOSITION" for row in observed)
    assert all(row["pairing_verification_scope"].endswith("NOT_PHYSICAL_MEDIA_VERIFICATION") for row in observed)

    videoseal = next(row for row in value["native_records"] if row["job_id"] == "fixture_ok/videoseal")
    vs_extract = videoseal["native"]["extract"]
    assert vs_extract["truth_input_supplied"] is False
    assert vs_extract["output_shapes"]["preds"] == [2, 5, 1, 1]
    assert vs_extract["output_element_counts"]["preds"] == 10
    assert len(vs_extract["raw_native_output"]["preds"]) == 2
    assert videoseal["native"]["main_32bit_mapping"] == "PENDING_NOT_APPLIED"

    rivagan = next(row for row in value["native_records"] if row["job_id"] == "fixture_ok/rivagan")
    riva_extract = rivagan["native"]["extract"]
    assert riva_extract["truth_input_supplied"] is False
    assert riva_extract["frame_shapes"] == [[32], [32]]
    assert len(riva_extract["frame_soft_outputs"][0]) == 32
    assert len(riva_extract["frame_native_zero_threshold_bits"][1]) == 32
    assert riva_extract["soft_value_count"] == 64
    assert rivagan["native"]["sequence_32bit_recovery"] == "PENDING_REDUCER_NOT_APPLIED"
    assert "bit_accuracy" not in json.dumps(rivagan["native"])


def test_missing_callbacks_and_failed_callback_never_create_success_rows():
    manifest = workflow_manifest()
    no_backends = run_workflow(manifest, base_dir=PACKAGE)
    assert no_backends["manifest_denominator"]["plan_steps"] == 16
    assert no_backends["manifest_denominator"]["plan_state_counts"].get("SUCCEEDED", 0) == 0
    assert len(no_backends["artifacts"]) == 26
    assert all(row["status"] != "SUCCEEDED" for row in no_backends["native_records"])

    no_codec_operations = build_fixture_operations()
    del no_codec_operations["fixture_codec_roundtrip"]
    no_codec = run_workflow(
        manifest,
        operations=no_codec_operations,
        native_adapters=build_fixture_native_adapters(),
        base_dir=PACKAGE,
    )
    native_without_codec = next(
        row for row in no_codec["native_records"] if row["job_id"] == "fixture_ok/videoseal"
    )
    assert native_without_codec["status"] == "SKIPPED_BACKEND_UNAVAILABLE"
    assert native_without_codec["native"]["status"] == "PARTIAL_WITH_RETAINED_FAILURE"
    assert native_without_codec["native"]["embed"]["raw_native_output"]["msgs"] == [[0, 1, 1, 0]]
    pre = next(row for row in no_codec["artifacts"] if row["artifact_id"] == "fixture_ok/videoseal/pre")
    post = next(row for row in no_codec["artifacts"] if row["artifact_id"] == "fixture_ok/videoseal/post")
    assert pre["status"] == "AVAILABLE" and post["status"] == "SKIPPED"

    operations = build_fixture_operations()

    def fail_payload(value, context):
        del value, context
        raise RuntimeError("fixture transform failure")

    operations["fixture_payload_native"] = fail_payload
    failed = run_workflow(
        manifest,
        operations=operations,
        native_adapters=build_fixture_native_adapters(),
        base_dir=PACKAGE,
    )
    p0 = next(row for row in failed["plan"] if row["step_id"] == "fixture_ok/PAYLOAD_NATIVE/TRANSFORM")
    p0_codec = next(row for row in failed["plan"] if row["step_id"] == "fixture_ok/PAYLOAD_NATIVE/CODEC")
    p1 = next(row for row in failed["plan"] if row["step_id"] == "fixture_ok/PAYLOAD_FRAMEWISE_RECON/TRANSFORM")
    assert p0["state"] == "FAILED"
    assert p0_codec["state"] == p1["state"] == "BLOCKED_DEPENDENCY"
    assert len(failed["plan"]) == 16 and len(failed["quality_records"]) == 6

    malformed_source = workflow_manifest()
    malformed_source["cases"][0]["source"]["input"]["value"] = [[0.0], [0.0, 1.0]]
    retained = run_workflow(malformed_source, base_dir=PACKAGE)
    source_row = next(row for row in retained["artifacts"] if row["artifact_id"] == "fixture_ok/source")
    assert source_row["status"] == "FAILED" and "ragged" in source_row["reason"]
    assert len(retained["plan"]) == 16


def test_manifest_dependency_and_native_message_contracts_are_explicit():
    manifest = workflow_manifest()
    manifest["cases"][0]["stages"]["PAYLOAD_FRAMEWISE_RECON"]["input_artifact_id"] = "wrong"
    with pytest.raises(WorkflowManifestError, match="input_artifact_id"):
        validate_workflow_manifest(manifest)

    manifest = workflow_manifest()
    manifest["native_jobs"][1]["native_message_bits"] = [0, 1]
    with pytest.raises(WorkflowManifestError, match="exactly 32"):
        validate_workflow_manifest(manifest)

    manifest = workflow_manifest()
    manifest["native_jobs"][0]["input_artifact_id"] = "fixture_ok/p0/pre"
    with pytest.raises(WorkflowManifestError, match="OFF_NATIVE PRE"):
        validate_workflow_manifest(manifest)

    template = report.read_json(PACKAGE / "workflow.template.json")
    validate_workflow_manifest(template)
    assert template["cases"] == [] and template["native_jobs"] == []
    assert run_workflow(template)["report_status"] == "PENDING_EMPTY_ROSTER"


def test_videoseal_loaded_backend_uses_native_calls_without_truth_or_capacity_mapping():
    calls = []

    class Model:
        def embed(self, media, *, msgs, is_video, lowres_attenuation):
            calls.append(("embed", copy.deepcopy(msgs), is_video, lowres_attenuation))
            return {"imgs_w": media, "msgs": msgs}

        def detect(self, media, *, is_video):
            calls.append(("detect", media, is_video))
            return {"preds": [[[[0.1]], [[-0.2]], [[0.3]]]]}

    adapter = VideoSealNativeAdapter(
        VideoSealLoadedBackend(Model(), lambda bits: [bits]),
        native_message_length=2,
        backend_metadata={
            "source_version": "unit", "model_version": "unit", "weight_identity": "unit",
            "detect_output_layout": "T,1+K,H,W",
        },
    )
    _media, record = adapter.run([[0.2]], [0, 1])
    assert calls[0] == ("embed", [[0, 1]], True, False)
    assert calls[1][0] == "detect" and len(calls[1]) == 3
    assert record["extract"]["raw_native_output"]["preds"] == [[[[0.1]], [[-0.2]], [[0.3]]]]
    assert record["extract"]["truth_input_supplied"] is False
    assert record["main_32bit_mapping"] == "PENDING_NOT_APPLIED"
    with pytest.raises(NativeAdapterError, match="exactly 2"):
        adapter.run([[0.2]], [0])
    with pytest.raises(NativeAdapterError, match="backend_metadata missing"):
        VideoSealNativeAdapter(
            VideoSealLoadedBackend(Model(), lambda bits: [bits]),
            native_message_length=2,
            backend_metadata={"model_version": "unit"},
        )


def test_rivagan_path_backend_discloses_hidden_native_codec():
    calls = []

    class Model:
        def encode(self, source, bits, target):
            calls.append(("encode", source, bits, target))

        def decode(self, source):
            calls.append(("decode", source))
            return [[-0.1, 0.1] * 16]

    backend = RivaGANPathBackend(Model())
    output = backend.embed("source.mp4", [0, 1] * 16, output_uri="native.mp4")
    decoded = list(backend.extract(output))
    assert calls[0][0] == "encode" and calls[1] == ("decode", "native.mp4")
    assert len(decoded[0]) == 32
    assert backend.transport_metadata["native_path_codec"] == "opencv_mp4v"
    assert backend.transport_metadata["native_path_fps"] == 20
    assert backend.transport_metadata["codec_comparability"] == "NATIVE_PATH_CODEC_NOT_SHARED_PLANNED_CODEC"


def test_quality_identical_psnr_is_null_and_declared_identity_mismatch_conflicts():
    metadata = {
        "case_id": "c", "source_id": "s", "source_content_id": "content", "noise_id": "n",
        "noise_seed": 1, "codec_id": "codec", "codec_parameters": {"x": 1},
    }
    artifacts = {
        "a": {"artifact_id": "a", "status": "AVAILABLE", "_value": [0.0, 0.0], **metadata},
        "b": {"artifact_id": "b", "status": "AVAILABLE", "_value": [0.0, 0.0], **metadata},
    }
    pair = {
        "quality_id": "q", "case_id": "c", "comparison": "IDENTICAL_FIXTURE",
        "reference_artifact_id": "a", "candidate_artifact_id": "b", "data_range": 1.0,
    }
    row = _quality_record(pair, artifacts)
    assert row["state"] == "OBSERVED" and row["mse"] == 0.0
    assert row["psnr_db"] is None and row["psnr_status"] == "IDENTICAL_INFINITE"
    artifacts["b"]["codec_id"] = "other"
    row = _quality_record(pair, artifacts)
    assert row["state"] == "CONFLICT" and "codec_id" in row["reason"]


def test_workflow_cli_runs_from_no_git_source_copy(tmp_path):
    source = tmp_path / "source-copy"
    (source / "experiments").mkdir(parents=True)
    shutil.copy(ROOT / "experiments/__init__.py", source / "experiments/__init__.py")
    shutil.copytree(PACKAGE, source / "experiments/paper_results_v1")
    output = tmp_path / "workflow-output"
    interpreter = [sys.executable]
    if not os.access(sys.executable, os.X_OK):
        interpreter = ["/lib64/ld-linux-x86-64.so.2", sys.executable]
    completed = subprocess.run(
        interpreter + [
            "-m", "experiments.paper_results_v1.workflow_cli",
            "--manifest", "experiments/paper_results_v1/workflow_fixture.manifest.json",
            "--fixture-backends", "--output-dir", str(output),
        ],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    )
    assert not (source / ".git").exists()
    summary = json.loads(completed.stdout)
    assert summary["denominator"]["plan_steps"] == 16
    assert summary["main_report_status"] == "LINKED"
    assert {path.name for path in output.iterdir()} == {
        "workflow_report.json", "workflow_report.md", "plan.csv", "artifacts.csv",
        "quality.csv", "cost.csv", "native_records.json", "main_report",
    }
    value = json.loads((output / "workflow_report.json").read_text(encoding="utf-8"))
    assert value["manifest_denominator"]["native_state_counts"] == {
        "BLOCKED_DEPENDENCY": 2, "SUCCEEDED": 2,
    }
