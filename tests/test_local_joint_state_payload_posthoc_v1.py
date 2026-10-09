"""CPU-only checks for the adopted known-grid descriptive posthoc."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import pytest

from main.tube_state import local_joint_state_payload_carrier_v1 as carrier
from main.tube_state import local_joint_state_payload_posthoc_v1 as method
from experiments.wan_state_clock import local_joint_state_payload_posthoc_v1_run as posthoc


pytestmark = pytest.mark.unit
PROTOCOL = carrier.CarrierProtocol(
    video_shape=(181, 8, 32, 3), segment_start=1, segment_frames=8, segment_count=22,
    rois=((0, 8, 0, 8), (0, 8, 8, 16), (0, 8, 16, 24), (0, 8, 24, 32)),
)


def _chip(component, component_index, pair_index, q, plus, minus, *, zero=0):
    return dict(component=component, component_index=component_index, pair_index=pair_index,
                plus_coord=list(plus), minus_coord=list(minus), status="SCORED",
                energy_plus=(1 + q) / 2, energy_minus=(1 - q) / 2, q=q,
                expected_frames=8, available_frames=8, expected_samples=8,
                available_samples=8, supported_samples=8 - zero,
                zero_energy_samples=zero, failed_samples=0, error=None)


def rows(key: str, message: bytes, amplitude: float):
    states = carrier.state_matrix(key)
    fragments = carrier.message_fragments(message)
    output = []
    for spec in carrier.phase_window_catalog(PROTOCOL):
        slot = spec.slot if 0 <= spec.slot < 22 else 0
        pairs = carrier.coefficient_pairs(key, spec.roi_index)
        state = [_chip("state", 8 * spec.roi_index + bit, bit,
                       amplitude * states[slot][8 * spec.roi_index + bit], *pairs[bit]) for bit in range(8)]
        payload = [_chip("payload", bit, 8 + bit,
                         amplitude * (2 * fragments[slot % 4][bit] - 1), *pairs[8 + bit]) for bit in range(8)]
        output.append(dict(spec=dict(observation_id=spec.observation_id, phase=spec.phase, slot=spec.slot,
                                     roi_index=spec.roi_index, requested_frames=list(spec.requested_frames),
                                     received_frames=list(spec.received_frames), availability=spec.availability),
                           state_status="SCORED", payload_status="SCORED",
                           state_chips=state, payload_chips=payload, truth_used=False))
    return output


def test_fixed_correlations_payload_counts_strict_ties_and_zero_subsupport():
    key, message = "adopted-key", bytes.fromhex("8001a55a")
    joint_rows = rows(key, message, .5)
    # Partial zero-energy tile support is valid while pooled support remains.
    target = next(row for row in joint_rows if row["spec"]["observation_id"] == "phase1:slot0:roi0")
    target["state_chips"][0].update(supported_samples=3, zero_energy_samples=5)
    value = method.evaluate_observations(joint_rows, key=key, message=message, protocol=PROTOCOL)
    assert len(value["state"]["correlations"]) == 22
    assert value["state"]["c0"] == pytest.approx(.5) and value["state"]["c0_minus_max_other"] > 0
    assert value["state"]["evidence"][0]["evidence"]["valid_for_metric"] is True
    assert len(value["payload"]["metrics"]) == 32
    assert [item["expected_evidence_count"] for item in value["payload"]["metrics"][::8]] == [24, 24, 20, 20]
    assert all(item["signed_mean"] == pytest.approx(.5) for item in value["payload"]["metrics"])
    assert method.descriptive_condition(value)["met"] is True
    with pytest.raises(ValueError, match="chip index mapping mismatch"):
        method.evaluate_observations(joint_rows, key=key + "-wrong", message=message, protocol=PROTOCOL)

    off = method.evaluate_observations(rows(key, message, 0.0), key=key, message=message, protocol=PROTOCOL)
    assert off["state"]["global_max_ties"] == list(range(22))
    assert off["state"]["c0_minus_max_other"] == 0.0
    assert method.descriptive_condition(off)["met"] is False


@pytest.mark.parametrize("failure", ("zero", "nan", "partial", "missing"))
def test_missing_nonfinite_zero_and_partial_do_not_drop_or_reweight(failure):
    key, message = "adopted-key", bytes.fromhex("8001a55a")
    value_rows = rows(key, message, .5)
    row = next(item for item in value_rows if item["spec"]["observation_id"] == "phase1:slot0:roi0")
    chip = row["state_chips"][0]
    if failure == "zero":
        chip.update(q=None, energy_plus=0.0, energy_minus=0.0, supported_samples=0, zero_energy_samples=8, status="MISSING")
        row["state_status"] = "PARTIAL"
    elif failure == "nan":
        chip["q"] = float("nan")
    elif failure == "missing":
        chip.update(q=None, energy_plus=None, energy_minus=None, status="MISSING",
                    available_frames=0, available_samples=0, supported_samples=0)
        row["state_status"] = "MISSING"
    else:
        chip.update(status="PARTIAL", failed_samples=1, available_samples=7)
        row["state_status"] = "PARTIAL"
    value = method.evaluate_observations(value_rows, key=key, message=message, protocol=PROTOCOL)
    assert value["state"]["status"] == "MISSING"
    assert len(value["state"]["correlations"]) == 22
    assert all(item["value"] is None and item["missing_items"] == 1 for item in value["state"]["correlations"])
    assert len(value["state"]["evidence"]) == 704
    condition = method.descriptive_condition(value)
    assert condition["classification"] == (
        "CONSTRUCTION_SUPPORT_GAP" if failure == "zero" else "ENGINEERING_FAILURE"
    )
    complete = method.evaluate_observations(rows(key, message, .5), key=key, message=message, protocol=PROTOCOL)
    comparison = method.compare_arms(complete, value)
    assert comparison["expected_values"] == 55 and comparison["missing_values"] == 23

    payload_rows = rows(key, message, .5)
    row = next(item for item in payload_rows if item["spec"]["observation_id"] == "phase1:slot0:roi0")
    row["payload_chips"][0].update(q=None, supported_samples=0, zero_energy_samples=8, status="MISSING")
    row["payload_status"] = "PARTIAL"
    payload = method.evaluate_observations(payload_rows, key=key, message=message, protocol=PROTOCOL)
    missing = [item for item in payload["payload"]["metrics"] if item["status"] == "MISSING"]
    assert [(item["fragment"], item["bit"], item["missing_items"]) for item in missing] == [(0, 0, 1)]
    assert missing[0]["expected_evidence_count"] == 24 and len(missing[0]["evidence"]) == 24
    assert method.compare_arms(complete, payload)["missing_values"] == 1


def test_condition_and_attribution_keep_finite_negative_and_missing_control_distinct():
    key, message = "adopted-key", bytes.fromhex("8001a55a")
    negative = method.descriptive_condition(
        method.evaluate_observations(rows(key, message, 0.0), key=key, message=message, protocol=PROTOCOL))
    positive = method.descriptive_condition(
        method.evaluate_observations(rows(key, message, .5), key=key, message=message, protocol=PROTOCOL))
    unavailable = dict(status="MISSING", met=None, classification="ENGINEERING_FAILURE")
    assert negative["classification"] == "VALID_FINITE_NEGATIVE"
    assert posthoc._attribution(negative, positive) == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    assert posthoc._attribution(unavailable, positive) == "ATTRIBUTION_UNRESOLVED_OFF_UNAVAILABLE"


def _write_rows(path: Path, value):
    encoded = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _fixture_run(root: Path, config: dict):
    observations = {}
    for arm, amplitude in (("OFF", 0.0), ("JOINT", .5)):
        observations[arm] = {}
        for layer in method.LAYERS:
            for label, key in (("CORRECT", config["carrier"]["key"]), ("WRONG", config["carrier"]["wrong_key"])):
                path = root / f"{arm}-{layer}-{label}.json"
                sha = _write_rows(path, rows(key, bytes.fromhex(config["carrier"]["message_hex"]), amplitude))
                observations[arm][f"{layer}/{label}"] = dict(
                    status="SAVED", path=str(path), sha256=sha, rows=768,
                    fixed_directory_rows=768, key_label=label, layer=layer, key=key, truth_used=False)
    result = dict(
        status="COMPLETE", stage="COMPLETE", config=config, actual_model_calls=False,
        source_identity={"kind": "dependency_injected_cpu_fixture"},
        execution=dict(kind="dependency_injected_cpu_fixture", attempted=True, completed=True,
                       scientific_interpretation=False, blind_recovery=False, fpr_evidence=False),
        arms={arm: dict(status="COMPLETE", initial_fingerprint="shared-initial",
                        terminal_fingerprint=f"{arm.lower()}-terminal", observations=value)
              for arm, value in observations.items()},
    )
    path = root / "result.json"; path.write_text(json.dumps(result))
    manifest = dict(schema="local-joint-raw-observation-manifest-v1", truth_loaded=False,
                    arms={arm: {
                        name: {key: item for key, item in receipt.items() if key != "key"}
                        for name, receipt in value.items()
                    } for arm, value in observations.items()})
    (root / "raw_observation_manifest.json").write_text(json.dumps(manifest))
    return path


def test_fixed_config_is_the_adopted_single_source_two_arm_roster():
    config = json.loads(posthoc.FIXED_CONFIG.read_text())
    posthoc.runtime.validate_config(config)
    assert config["source"] == {
        "source_id": "yellow_sailboat_dev_s2026100701", "development_only": True}
    assert config["arms"] == ["OFF", "JOINT"]
    assert config["carrier"] == {
        "key": "local-joint-state-payload-v1-first-mechanism",
        "wrong_key": "local-joint-state-payload-v1-first-mechanism-wrong",
        "message_hex": "8001a55a", "rho": .5, "cap": 1.0,
    }
    assert config["generation"]["seed"] == 2026100701
    assert config["media"] == {"fps": 8, "codec": "libx264", "crf": 18, "pixel_format": "yuv420p"}


def test_saved_posthoc_cli_no_git_seals_before_truth_and_compares_arms(tmp_path):
    release = tmp_path / "release"
    for name in posthoc.SOURCE_CLOSURE:
        target = release / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(posthoc.ROOT / name, target)
    config = json.loads((release / "experiments/wan_state_clock/configs/local_joint_state_payload_v1.json").read_text())
    run_result = _fixture_run(tmp_path, config)
    output = release / "posthoc-output"
    child = subprocess.run([
        "/home/richar/projects/CEG-WM/alive/CEG-WM/.venv/bin/python", "-B", "-m",
        "experiments.wan_state_clock.local_joint_state_payload_posthoc_v1_run",
        "--run-result", str(run_result), "--config",
        str(release / "experiments/wan_state_clock/configs/local_joint_state_payload_v1.json"),
        "--output", str(output),
    ], cwd=release, env={**__import__("os").environ, "PYTHONPATH": str(release), "CUDA_VISIBLE_DEVICES": ""},
       capture_output=True, text=True)
    assert child.returncode == 0, child.stderr
    seal = json.loads((output / "raw_observation_seal.json").read_text())
    result = json.loads((output / "posthoc_result.json").read_text())
    assert seal["truth_loaded"] is False and len(seal["entries"]) == 12
    assert seal["manifest_sha256"] == hashlib.sha256(
        (tmp_path / "raw_observation_manifest.json").read_bytes()).hexdigest()
    assert all(item["status"] == "SEALED" and item["rows"] == 768 for item in seal["entries"].values())
    assert result["truth"]["loaded_after_raw_seal"] is True
    assert result["mp4_conditions"]["JOINT"]["met"] is True
    assert result["mp4_conditions"]["OFF"]["met"] is False
    assert result["mp4_attribution"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    assert result["outcome_classification"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    assert result["run_result"]["status"] == "COMPLETE"
    assert result["run_result"]["execution"]["completed"] is True
    assert result["run_result"]["arms"]["OFF"]["initial_fingerprint"] == "shared-initial"
    assert set(result["correct_key_conditions"]["JOINT"]) == set(method.LAYERS)
    assert all(result["correct_key_conditions"]["JOINT"][layer]["met"] is True for layer in method.LAYERS)
    assert all(result["wrong_key_conditions"]["JOINT"][layer]["auxiliary_only"] is True
               for layer in method.LAYERS)
    assert result["evaluations"]["JOINT"]["mp4"]["WRONG"]["auxiliary_wrong_key"] is True
    assert result["scientific_pass"] is False


def test_metadata_differences_and_optional_hash_errors_do_not_block_posthoc(tmp_path, monkeypatch):
    config = json.loads(posthoc.FIXED_CONFIG.read_text())
    run_result = _fixture_run(tmp_path, config)
    result = json.loads(run_result.read_text())
    result["arms"]["OFF"]["observations"]["float_rgb/CORRECT"]["sha256"] = "0" * 64
    result["config"]["model"]["revision"] = "different-source-record"
    run_result.write_text(json.dumps(result))
    # The source snapshot is optional; the already imported implementation can
    # evaluate readable observations even without its source/config sidecars.
    monkeypatch.setattr(posthoc, "ROOT", tmp_path / "missing-source")
    monkeypatch.setattr(posthoc, "_sha", lambda path: (_ for _ in ()).throw(PermissionError("optional hash read")))
    output = tmp_path / "binding-output"
    persisted = posthoc.run(run_result, posthoc.FIXED_CONFIG, output)
    seal = json.loads((output / "raw_observation_seal.json").read_text())
    assert seal["truth_loaded"] is False and len(seal["entries"]) == 12
    assert persisted["truth"]["loaded_after_raw_seal"] is True
    assert persisted["metadata_comparison"]["manifest_matches_run_receipt"] is False
    assert persisted["metadata_comparison"]["config_matches_run_receipt"] is False
    assert persisted["source_identity"]["file_errors"]
    assert len(persisted["identity_errors"]) == 3
    assert persisted["outcome_classification"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"


@pytest.mark.parametrize("declared_sha", ("stale", None))
def test_raw_receipt_hash_is_advisory(tmp_path, declared_sha):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps([{}] * 768))
    receipt = dict(status="SAVED", path=str(path), layer="mp4", key_label="CORRECT", sha256=declared_sha)
    loaded, sealed = posthoc._load_raw_receipt(receipt, expected_layer="mp4", expected_label="CORRECT")
    assert len(loaded) == 768 and sealed["status"] == "SEALED"
    assert sealed["sha256_matches"] is (None if declared_sha is None else False)
    path.write_text(json.dumps([{}] * 767))
    loaded, failure = posthoc._load_raw_receipt(receipt, expected_layer="mp4", expected_label="CORRECT")
    assert loaded is None and "768 rows" in failure["reason"]


def test_raw_receipt_layer_label_remains_a_method_boundary(tmp_path):
    path = tmp_path / "raw.json"; path.write_text(json.dumps([{}] * 768))
    receipt = dict(status="SAVED", path=str(path), layer="float_rgb", key_label="CORRECT")
    loaded, failure = posthoc._load_raw_receipt(receipt, expected_layer="mp4", expected_label="CORRECT")
    assert loaded is None and "layer/key label mismatch" in failure["reason"]


def test_posthoc_truth_mismatch_is_rejected_after_raw_seal(tmp_path):
    config = json.loads(posthoc.FIXED_CONFIG.read_text())
    run_result = _fixture_run(tmp_path, config)
    result = json.loads(run_result.read_text())
    result["config"]["carrier"]["message_hex"] = "00000000"
    run_result.write_text(json.dumps(result))
    output = tmp_path / "wrong-truth-output"
    with pytest.raises(ValueError, match="key/message truth"):
        posthoc.run(run_result, posthoc.FIXED_CONFIG, output)
    seal = json.loads((output / "raw_observation_seal.json").read_text())
    assert seal["truth_loaded"] is False and len(seal["entries"]) == 12


def test_incomplete_run_cannot_be_overridden_by_positive_local_mp4_evidence(tmp_path):
    config = json.loads(posthoc.FIXED_CONFIG.read_text())
    run_result = _fixture_run(tmp_path, config)
    result = json.loads(run_result.read_text())
    result["status"] = "FAILED"
    result["stage"] = "FAILED"
    result["execution"]["completed"] = False
    result["arms"]["JOINT"]["status"] = "FAILED"
    run_result.write_text(json.dumps(result))
    output = tmp_path / "incomplete-output"
    persisted = posthoc.run(run_result, posthoc.FIXED_CONFIG, output)
    assert persisted["mp4_attribution"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    assert persisted["status"] == "INCOMPLETE" and persisted["outcome_classification"] == "INCOMPLETE"
    assert persisted["run_result"]["execution"]["completed"] is False
    assert persisted["run_result"]["arms"]["JOINT"]["status"] == "FAILED"


def test_wrong_key_failure_is_auxiliary_when_correct_chain_is_complete(tmp_path):
    config = json.loads(posthoc.FIXED_CONFIG.read_text())
    run_result = _fixture_run(tmp_path, config)
    result = json.loads(run_result.read_text())
    manifest_path = tmp_path / "raw_observation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for owner in (result["arms"]["JOINT"]["observations"], manifest["arms"]["JOINT"]):
        owner["mp4/WRONG"] = dict(status="FAILED", layer="mp4", key_label="WRONG",
                                   truth_used=False, reason="fixture wrong-key failure")
    result["status"] = "COMPLETE_WITH_OBSERVATION_FAILURE"
    result["arms"]["JOINT"]["status"] = "COMPLETE_WITH_OBSERVATION_FAILURE"
    run_result.write_text(json.dumps(result))
    manifest_path.write_text(json.dumps(manifest))
    persisted = posthoc.run(run_result, posthoc.FIXED_CONFIG, tmp_path / "wrong-aux-output")
    assert persisted["wrong_key_conditions"]["JOINT"]["mp4"]["classification"] == "ENGINEERING_FAILURE"
    assert persisted["mp4_attribution"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    assert persisted["outcome_classification"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"


def test_unavailable_complete_raw_file_preserves_fixed_metrics_and_other_evidence(tmp_path):
    config = json.loads(posthoc.FIXED_CONFIG.read_text())
    run_result = _fixture_run(tmp_path, config)
    result = json.loads(run_result.read_text())
    manifest_path = tmp_path / "raw_observation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    missing_path = tmp_path / "does-not-exist.json"
    for owner in (result["arms"]["OFF"]["observations"], manifest["arms"]["OFF"]):
        owner["mp4/CORRECT"]["path"] = str(missing_path)
    run_result.write_text(json.dumps(result))
    manifest_path.write_text(json.dumps(manifest))

    persisted = posthoc.run(run_result, posthoc.FIXED_CONFIG, tmp_path / "missing-raw-output")
    unavailable = persisted["evaluations"]["OFF"]["mp4"]["CORRECT"]
    evaluation = unavailable["evaluation"]
    assert unavailable["status"] == "ENGINEERING_FAILURE"
    assert evaluation["status"] == "ENGINEERING_FAILURE" and evaluation["observed_evidence_items"] == 0
    assert evaluation["scope"]["state_denominator"] == 704
    assert len(evaluation["state"]["correlations"]) == 22
    assert "FileNotFoundError" in unavailable["reason"]
    assert all(row["value"] is None and row["reason"] == unavailable["reason"]
               for row in evaluation["state"]["correlations"])
    assert evaluation["state"]["c0_minus_max_other"] is None
    assert len(evaluation["payload"]["metrics"]) == 32
    assert [row["expected_evidence_count"] for row in evaluation["payload"]["metrics"][::8]] == [24, 24, 20, 20]
    assert all(row["signed_mean"] is None and row["evidence"] == []
               for row in evaluation["payload"]["metrics"])
    comparison = persisted["correct_key_arm_differences"]["mp4"]
    assert comparison["expected_values"] == 55 and comparison["missing_values"] == 55
    assert len(comparison["state_correlations"]) == 22 and comparison["state_gap"] is None
    assert len(comparison["payload_signed_means"]) == 32
    assert persisted["evaluations"]["JOINT"]["mp4"]["CORRECT"]["status"] == "EVALUATED"
    assert persisted["evaluations"]["JOINT"]["mp4"]["WRONG"]["status"] == "EVALUATED"
    assert persisted["mp4_attribution"] == "ATTRIBUTION_UNRESOLVED_OFF_UNAVAILABLE"
    assert persisted["outcome_classification"] == "ENGINEERING_FAILURE"


@pytest.mark.parametrize(("field", "nonfinite"), (
    ("q", float("nan")), ("q", float("inf")),
    ("energy_plus", float("nan")), ("energy_minus", float("inf")),
), ids=("q-nan", "q-infinity", "energy-plus-nan", "energy-minus-infinity"))
def test_persisted_nonfinite_raw_cli_writes_strict_engineering_result(tmp_path, field, nonfinite):
    config = json.loads(posthoc.FIXED_CONFIG.read_text())
    run_result = _fixture_run(tmp_path, config)
    result = json.loads(run_result.read_text())
    manifest_path = tmp_path / "raw_observation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = result["arms"]["OFF"]["observations"]["float_rgb/CORRECT"]
    raw_path = Path(receipt["path"])
    raw_rows = json.loads(raw_path.read_text())
    target = next(row for row in raw_rows if row["spec"]["observation_id"] == "phase1:slot0:roi0")
    target["state_chips"][0][field] = nonfinite
    encoded = json.dumps(raw_rows, separators=(",", ":"), allow_nan=True).encode()
    raw_path.write_bytes(encoded)
    sha = hashlib.sha256(encoded).hexdigest()
    receipt["sha256"] = sha
    manifest["arms"]["OFF"]["float_rgb/CORRECT"]["sha256"] = sha
    run_result.write_text(json.dumps(result))
    manifest_path.write_text(json.dumps(manifest))

    output = tmp_path / "nonfinite-output"
    assert posthoc.main(["--run-result", str(run_result), "--config", str(posthoc.FIXED_CONFIG),
                         "--output", str(output)]) == 0
    persisted_text = (output / "posthoc_result.json").read_text()
    assert "NaN" not in persisted_text and "Infinity" not in persisted_text
    persisted = json.loads(persisted_text)
    evidence = persisted["evaluations"]["OFF"]["float_rgb"]["CORRECT"]["evaluation"][
        "state"]["invalid_items"][0]["evidence"]
    assert evidence[field] is None
    assert evidence["missing_reason"] == (
        "q_missing_or_nonfinite" if field == "q" else "energy_missing_or_nonfinite")
    assert persisted["correct_key_conditions"]["OFF"]["float_rgb"][
        "classification"] == "ENGINEERING_FAILURE"
    assert persisted["mp4_attribution"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    assert persisted["outcome_classification"] == "ENGINEERING_FAILURE"
