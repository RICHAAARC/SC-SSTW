"""Persistence at model/reader boundaries, with no real model execution."""
import json
import os
import signal
import subprocess
import sys
import types

import pytest

from runtime.wan import local_joint_terminal_bridge_v1 as runtime
from main.tube_state import local_joint_terminal_bridge_v1 as diagnostic
from main.tube_state import local_joint_state_payload_posthoc_v1 as posthoc

pytestmark = pytest.mark.quick


def test_before_adapter_failure_returned_model_call_is_already_saved(tmp_path):
    result = runtime.initial_result({})
    def save():
        runtime.write_json(tmp_path/"result.json", result)
    tracked = runtime.TrackedVAE(types.SimpleNamespace(encode=lambda _: "returned-value"), result, save,
                                 execution_kind="INJECTED_TEST_DOUBLE")
    tracked.current = "encode_base"
    assert tracked.encode(None) == "returned-value"
    on_disk = json.loads((tmp_path/"result.json").read_text())
    assert on_disk["counts"]["encode_completed"] == 1
    assert on_disk["model_calls"]["encode_base"]["status"] == "RETURNED"
    with pytest.raises(RuntimeError, match="repeated"):
        tracked.encode(None)
    assert result["counts"]["encode_attempted"] == 1


def test_primary_model_error_survives_secondary_persistence_error():
    result = runtime.initial_result({})
    primary = RuntimeError("primary model failure")
    def encode(_):
        raise primary
    calls = []
    def save():
        calls.append(None)
        if len(calls) == 2:
            raise OSError("secondary disk failure")
    tracked = runtime.TrackedVAE(types.SimpleNamespace(encode=encode), result, save,
                                 execution_kind="INJECTED_TEST_DOUBLE")
    tracked.current = "encode_base"
    with pytest.raises(RuntimeError) as caught:
        tracked.encode(None)
    assert caught.value is primary
    assert result["counts"]["encode_attempted"] == 1
    assert result["counts"]["encode_completed"] == 0
    assert "secondary disk failure" in str(getattr(primary, "__notes__", []))


def _raw_fixture(q):
    # Synthetic saved records, not model/reader measurements. They exercise
    # serialization and comparison only, with all fixed IDs and chip slots.
    rows = []
    for spec in diagnostic.known_catalog():
        chips = [dict(pair_index=i, status="SCORED", q=q, energy_plus=1+q,
                      energy_minus=1-q, energy_total=2.) for i in range(16)]
        rows.append(dict(spec=dict(observation_id=spec.observation_id),
                         state_chips=chips[:8], payload_chips=chips[8:]))
    return dict(status="OBSERVED", truth_used=False, observed_windows=88, rows=rows)


def _metric_fixture(value):
    metric = posthoc.missing_evaluation("fixture")
    for field in ("directory_placeholder", "status", "engineering_reason"):
        metric.pop(field, None)
    metric["state"].update(status="SCORED", c0_minus_max_other=value)
    for row in metric["state"]["correlations"]:
        row.update(status="SCORED", value=value, reason=None)
    metric["payload"]["status"] = "SCORED"
    for row in metric["payload"]["metrics"]:
        row.update(status="SCORED", signed_mean=value, reason=None)
    return metric


@pytest.mark.skipif(os.name != "posix", reason="the Colab child SIGKILL path is POSIX")
@pytest.mark.parametrize("cut", ["placeholders", "raw_saved", "metrics_saved"])
def test_sigkill_takeover_completes_fixed_reports_from_persisted_evidence(tmp_path, monkeypatch, cut):
    result = runtime.initial_result({})
    result["counts"].update(encode_attempted=2, encode_completed=1)
    result["model_calls"]["encode_base"]["status"] = "RETURNED"
    result["model_calls"]["encode_candidate"]["status"] = "RUNNING"
    files = {}
    for name in diagnostic.STAGES:
        for view in diagnostic.VIEWS:
            missing = "saved_base_preclip_unavailable" if (name, view) == ("base", "preclip") else "not_run"
            files[f"{name}/{view}_raw.json"] = dict(status="MISSING", reason=missing,
                expected_windows=88, expected_chips=1408, observed_windows=0, rows=[])
            files[f"{name}/{view}_metrics.json"] = posthoc.missing_evaluation(missing)
    saved_paths = []
    if cut != "placeholders":
        files["base/postclip_raw.json"] = _raw_fixture(.1)
        files["base/postclip_metrics.json"] = _metric_fixture(.1)
        files["candidate/postclip_raw.json"] = _raw_fixture(.4)
        saved_paths = ["base/postclip_raw.json", "base/postclip_metrics.json", "candidate/postclip_raw.json"]
        result["stages"]["base"]["status"] = "COMPLETE"
        result["stages"]["base"]["views"]["postclip"].update(status="OBSERVED", state_gap=.1)
        # The child has saved raw, but its result.json metadata can lag a write.
        result["stages"]["candidate"]["status"] = "RUNNING"
        result["stages"]["candidate"]["views"]["postclip"].update(status="RUNNING", reason=None)
    if cut == "metrics_saved":
        files["candidate/postclip_metrics.json"] = _metric_fixture(.4)
        saved_paths.append("candidate/postclip_metrics.json")
    files["result.json"] = result
    child_code = """
import json, os, pathlib, signal, sys
root = pathlib.Path(sys.argv[1])
for relative, value in json.load(sys.stdin).items():
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as stream:
        json.dump(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
os.kill(os.getpid(), signal.SIGKILL)
"""
    child = subprocess.run([sys.executable, "-c", child_code, str(tmp_path)],
                           input=json.dumps(files), text=True, capture_output=True)
    assert child.returncode == -signal.SIGKILL, child.stderr
    before = {name: (tmp_path/name).read_bytes() for name in saved_paths}
    assert not (tmp_path/"paired_comparisons.json").exists()
    def forbidden(*_a, **_k):
        raise AssertionError("report takeover must not read pixels, recompute metrics or execute a model")
    monkeypatch.setattr(runtime.generation, "load_frozen_vae", forbidden)
    monkeypatch.setattr(runtime.vae_adapter, "decode_normalized_latent", forbidden)
    monkeypatch.setattr(runtime.vae_adapter, "reencode_rgb24_readback", forbidden)
    monkeypatch.setattr(diagnostic, "read_rgb", forbidden)
    monkeypatch.setattr(diagnostic, "evaluate_saved", forbidden)
    reason = "notebook reaped SIGKILL fixture child"
    recovered = runtime.finalize_interrupted(tmp_path, reason)
    assert recovered["status"] == "INTERRUPTED"
    assert recovered["counts"] == result["counts"]
    assert recovered["model_calls"]["encode_base"]["status"] == "RETURNED"
    assert recovered["model_calls"]["encode_candidate"]["status"] == "INTERRUPTED_COMPLETION_UNKNOWN"
    assert all((tmp_path/name).read_bytes() == value for name, value in before.items())
    comparisons = json.loads((tmp_path/"paired_comparisons.json").read_text())["comparisons"]
    assert len(comparisons) == 14
    for pair in comparisons.values():
        assert len(pair["chips"]) == 1408
        assert len(pair["state_correlations"]) + len(pair["payload_signed_means"]) + 1 == 55
        for row in pair["state_correlations"] + pair["payload_signed_means"]:
            assert row["status"] == ("MISSING" if row["value"] is None else "SCORED")
        for chip in pair["chips"]:
            assert chip["signed_q_delta_status"] == ("MISSING" if chip["signed_q_delta"] is None else "SCORED")
            assert all(chip["delta_status"][key] == ("MISSING" if value is None else "SCORED")
                       for key, value in chip["deltas"].items())
    pair = comparisons["base/postclip -> candidate/postclip"]
    if cut == "placeholders":
        assert all(chip["deltas"]["q"] is None for chip in pair["chips"])
    else:
        assert all(chip["deltas"]["q"] == pytest.approx(.3) for chip in pair["chips"])
        assert recovered["stages"]["base"]["views"]["postclip"]["state_gap"] == .1
    if cut == "metrics_saved":
        assert pair["state_gap"] == pytest.approx(.3)
        assert pair["missing_values"] == 0
        assert recovered["stages"]["candidate"]["views"]["postclip"]["status"] == "OBSERVED"
        assert recovered["stages"]["candidate"]["status"] == "INTERRUPTED"  # no stage-completion inference
    else:
        assert pair["state_gap_status"] == "MISSING"
        assert pair["missing_values"] == 55
    for name in diagnostic.STAGES:
        for view_name in diagnostic.VIEWS:
            view = recovered["stages"][name]["views"][view_name]
            raw = json.loads((tmp_path/name/f"{view_name}_raw.json").read_text())
            metric = json.loads((tmp_path/name/f"{view_name}_metrics.json").read_text())
            if raw["status"] == "MISSING":
                assert raw["reason"] == view["reason"] != "not_run"
            if metric.get("directory_placeholder"):
                assert metric["engineering_reason"] == view["reason"] != "not_run"
                assert all(c["reason"] == view["reason"] for c in metric["state"]["correlations"])
    assert recovered["stages"]["base"]["views"]["preclip"]["reason"] == "saved_base_preclip_unavailable"
    snapshots = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    assert runtime.finalize_interrupted(tmp_path, "a repeated callback") == recovered
    assert all(path.read_bytes() == data for path, data in snapshots.items())
