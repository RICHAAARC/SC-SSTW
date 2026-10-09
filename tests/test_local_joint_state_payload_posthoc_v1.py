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

    payload_rows = rows(key, message, .5)
    row = next(item for item in payload_rows if item["spec"]["observation_id"] == "phase1:slot0:roi0")
    row["payload_chips"][0].update(q=None, supported_samples=0, zero_energy_samples=8, status="MISSING")
    row["payload_status"] = "PARTIAL"
    payload = method.evaluate_observations(payload_rows, key=key, message=message, protocol=PROTOCOL)
    missing = [item for item in payload["payload"]["metrics"] if item["status"] == "MISSING"]
    assert [(item["fragment"], item["bit"], item["missing_items"]) for item in missing] == [(0, 0, 1)]
    assert missing[0]["expected_evidence_count"] == 24 and len(missing[0]["evidence"]) == 24


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
    result = dict(config=config, arms={arm: dict(observations=value) for arm, value in observations.items()})
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
    assert result["attribution"] == "DESCRIPTIVE_PROGRESS_WITH_OFF_CONTRAST"
    assert set(result["correct_key_conditions"]["JOINT"]) == set(method.LAYERS)
    assert all(result["correct_key_conditions"]["JOINT"][layer]["met"] is True for layer in method.LAYERS)
    assert all(result["wrong_key_conditions"]["JOINT"][layer]["auxiliary_only"] is True
               for layer in method.LAYERS)
    assert result["evaluations"]["JOINT"]["mp4"]["WRONG"]["auxiliary_wrong_key"] is True
    assert result["scientific_pass"] is False
