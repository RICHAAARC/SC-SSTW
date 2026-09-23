"""Fake-media CPU checks; no real MP4, Drive, Colab, or model execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path

import pytest

from main.tube_state.fixed_key import key_identifier as historical_key_identifier
from scripts import build_rgb_dct_off_diagnostic_notebook as builder
from scripts import rgb_dct_off_diagnostic as diagnostic


pytestmark = pytest.mark.unit
SOURCE_SHA = "a" * 40


def test_fixed_manifest_byte_identity_and_old_key_provenance():
    manifest = diagnostic._load_manifest()
    assert hashlib.sha256(diagnostic.MANIFEST.read_bytes()).hexdigest() == diagnostic.MANIFEST_SHA256
    assert [r["id"] for r in manifest["rows"]] == list(diagnostic.CASE_IDS)
    assert [r["role"] for r in manifest["rows"]] == ["calibration_off"] * 4 + ["evaluation"] * 4
    assert diagnostic.KEY == b"WanProjection-first-validation-key-v1"
    assert diagnostic.LEGACY_KEY_ID == historical_key_identifier(diagnostic.KEY) == "2deb9bf5842d9b2c"
    assert diagnostic.NEW_KEY_ID == "785b91ae6b23bfc9"


def test_every_slot_persists_before_read_and_failures_keep_the_fixed_denominator(tmp_path, monkeypatch):
    manifest = diagnostic._load_manifest()
    run_root, output = tmp_path / "historical", tmp_path / "new-output"
    original_bytes = {}
    for row in manifest["rows"]:
        if row["id"] == diagnostic.CASE_IDS[1]:
            continue
        path = run_root / row["id"] / row["path"]
        path.parent.mkdir(parents=True)
        original_bytes[row["id"]] = row["id"].encode()
        path.write_bytes(original_bytes[row["id"]])

    snapshots = []
    write = diagnostic._write_result

    def observed_write(path, result):
        snapshots.append(copy.deepcopy(result))
        write(path, result)

    monkeypatch.setattr(diagnostic, "_write_result", observed_write)
    hash_calls, score_calls = [], []

    def fake_hash(path):
        case_id = path.parents[2].name
        if not hash_calls:
            persisted = json.loads((output / "result.json").read_text())
            assert [row["status"] for row in persisted["rows"]] == ["PENDING"] * 8
        hash_calls.append(case_id)
        if case_id == diagnostic.CASE_IDS[2]:
            return "0" * 64
        return next(row["sha256"] for row in manifest["rows"] if row["id"] == case_id)

    def fake_score(path, key):
        case_id = path.parents[2].name
        score_calls.append(case_id)
        assert key == diagnostic.KEY
        if case_id == diagnostic.CASE_IDS[3]:
            return dict(status="INVALID", score=None, reason="FULL_RGB_SHAPE_REQUIRED", frames_used=0)
        if case_id == diagnostic.CASE_IDS[4]:
            raise RuntimeError("fake decode failure")
        return dict(status="SCORED", score=0.125, frames_used=181,
                    spec_sha256=diagnostic.SPEC_SHA256, key_id=diagnostic.NEW_KEY_ID)

    result = diagnostic.run_diagnostic(
        run_root, output, manifest, SOURCE_SHA, score_fn=fake_score, hash_fn=fake_hash
    )
    assert len(snapshots) == 10  # Eight pending, then eight updates, then final status.
    assert [row["status"] for row in snapshots[0]["rows"]] == ["PENDING"] * 8
    assert snapshots[0]["attempted_count"] == 0
    assert [r["status"] for r in result["rows"]] == [
        "SCORED", "MISSING", "HASH_MISMATCH", "DECODE_INVALID",
        "EXCEPTION", "SCORED", "SCORED", "SCORED",
    ]
    assert score_calls == [diagnostic.CASE_IDS[i] for i in (0, 3, 4, 5, 6, 7)]
    assert diagnostic.CASE_IDS[2] in hash_calls and diagnostic.CASE_IDS[1] not in hash_calls
    assert result["fixed_denominator"] == result["attempted_count"] == 8
    assert (result["scored_count"], result["invalid_count"], result["pending_count"]) == (4, 4, 0)
    assert result["rows"][2]["actual_sha256"] == "0" * 64
    assert all(row["score"] is None for row in result["rows"] if row["status"] != "SCORED")
    assert result["source_sha"] == SOURCE_SHA and result["spec_sha256"] == diagnostic.SPEC_SHA256
    assert json.loads((output / "result.json").read_text()) == result
    for case_id, raw in original_bytes.items():
        assert (run_root / case_id / "received_videos/OFF/FULL.mp4").read_bytes() == raw


def test_all_missing_and_existing_result_retained(tmp_path):
    manifest = diagnostic._load_manifest()
    output = tmp_path / "new-output"
    result = diagnostic.run_diagnostic(tmp_path / "missing", output, manifest, SOURCE_SHA)
    assert result["status"] == "COMPLETE_WITH_INVALID"
    assert result["attempted_count"] == result["invalid_count"] == 8
    assert all(row["status"] == "MISSING" and row["score"] is None for row in result["rows"])
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(tmp_path / "missing", output, manifest, SOURCE_SHA)
    assert json.loads((output / "result.json").read_text()) == result


def test_cli_output_location_is_fixed():
    with pytest.raises(ValueError):
        diagnostic._checked_output(Path("/tmp/other/20260923T120000000000Z"))
    with pytest.raises(ValueError):
        diagnostic._checked_output(diagnostic.OUTPUT_PARENT / "reused")


def test_builder_temporary_notebook_has_fixed_cpu_run_all_flow(tmp_path):
    with pytest.raises(ValueError):
        builder.build("not-a-sha", tmp_path / "bad.ipynb")
    path = builder.build(SOURCE_SHA, tmp_path / "temporary-only.ipynb")
    notebook = json.loads(path.read_text())
    codes = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert codes[0] == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert notebook["metadata"]["source_commit"] == SOURCE_SHA
    assert all(cell.get("outputs", []) == [] for cell in notebook["cells"])
    assert all(cell.get("execution_count") is None for cell in notebook["cells"] if cell["cell_type"] == "code")
    for code in codes:
        ast.parse(code)
    text = "\n".join(codes)
    assert str(diagnostic.RUN_ROOT) in text
    assert str(diagnostic.OUTPUT_PARENT) in text
    assert "checkout', '--detach', SOURCE_SHA" in text
    assert "scripts.rgb_dct_off_diagnostic" in text
    assert "--run-root" in text and "--output" in text
    assert "--manifest" not in text and "--key" not in text
    assert "pip install" not in text and "cuda" not in text.lower()
