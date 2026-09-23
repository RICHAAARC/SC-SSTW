import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.wan_state_clock import content_background_existence_run as runner
from main.tube_state import content_background_existence as candidate
from main.tube_state import fixed_key
from runtime.wan import payload_control
from scripts.build_content_background_existence_notebook import build


ROOT = Path(__file__).parents[1]
pytestmark = pytest.mark.unit


def observations():
    return {phase: np.zeros((1, 16, 2, 40, 64), dtype=np.float32) for phase in range(4)}


def test_wrong_keys_are_exact_unique_raw_digests():
    expected = tuple(
        hashlib.sha256(f"SC-SSTW-Content-Background-Existence-V1/wrong/{index:02d}".encode()).digest()
        for index in range(16)
    )
    assert candidate.WRONG_KEYS == expected
    assert len(set(candidate.WRONG_KEYS)) == 16


def test_read_runs_correct_once_and_each_wrong_over_full_family(monkeypatch):
    correct = b"correct key"
    calls = []
    values = {correct: 10.0, **{key: float(index) for index, key in enumerate(candidate.WRONG_KEYS)}}
    monkeypatch.setattr(candidate.fixed_key, "codebook", lambda key: {"key": key})

    def fake_read(obs, book):
        calls.append(book["key"])
        score = values[book["key"]]
        return {
            "status": "SCORED", "attempted_path_count": 4284, "scored_path_count": 4000,
            "existence_statistic": score, "best": {"path": {"g": 0}, "score": score},
        }

    monkeypatch.setattr(candidate.fixed_key, "read", fake_read)
    result = candidate.read(observations(), correct, "a" * 64)
    wrong = np.arange(16, dtype=np.float64)
    expected = (10.0 - wrong.mean()) / (wrong.std(ddof=0) + 1e-6)
    assert result["status"] == "SCORED"
    assert calls == [correct, *candidate.WRONG_KEYS]
    assert result["correct_detection"] is result["searches"][0]["detection"]
    assert result["wrong_scores"] == wrong.tolist()
    assert result["wrong_population_std"] == wrong.std(ddof=0)
    assert result["std_ddof"] == 0
    assert result["statistic"] == expected
    assert result["actual_search_count"] == 17
    assert result["actual_path_attempts"] == 17 * 4284
    assert all(row["attempted_path_count"] == 4284 for row in result["searches"])


def test_invalid_phase_preserves_all_fixed_key_slots(monkeypatch):
    calls = []
    monkeypatch.setattr(candidate.fixed_key, "read", lambda *args: calls.append(args))
    result = candidate.read({0: observations()[0]}, b"correct", "b" * 64)
    assert result["status"] == "INVALID"
    assert len(result["searches"]) == 17
    assert all(row["status"] == "NOT_RUN" for row in result["searches"])
    assert calls == []
    assert result["actual_search_count"] == 0
    assert result["actual_path_attempts"] == 0


def test_wrong_failure_still_preserves_successful_correct_original(monkeypatch):
    correct = b"correct"
    monkeypatch.setattr(candidate.fixed_key, "codebook", lambda key: {"key": key})

    def fake_read(obs, book):
        if book["key"] == candidate.WRONG_KEYS[3]:
            raise RuntimeError("fixed failure")
        return {
            "status": "SCORED", "attempted_path_count": 4284, "scored_path_count": 4284,
            "existence_statistic": 1.0, "best": {"path": {}, "score": 1.0},
        }

    monkeypatch.setattr(candidate.fixed_key, "read", fake_read)
    result = candidate.read(observations(), correct, "c" * 64)
    assert result["status"] == "INVALID"
    assert result["correct_detection"]["status"] == "SCORED"
    assert result["correct_score"] == 1.0
    assert result["actual_search_count"] == 17
    assert result["actual_path_attempts"] == 16 * 4284
    assert result["searches"][4]["status"] == "INVALID"


def test_population_std_at_or_below_epsilon_is_invalid(monkeypatch):
    monkeypatch.setattr(candidate.fixed_key, "codebook", lambda key: {"key": key})
    monkeypatch.setattr(candidate.fixed_key, "read", lambda obs, book: {
        "status": "SCORED", "attempted_path_count": 4284, "scored_path_count": 4284,
        "existence_statistic": 1.0, "best": {"path": {}, "score": 1.0},
    })
    result = candidate.read(observations(), b"correct", "d" * 64)
    assert result["status"] == "INVALID"
    assert result["wrong_population_std"] == 0.0
    assert result["correct_detection"]["status"] == "SCORED"


def test_four_source_calibration_is_exact_and_bound():
    spec = "e" * 64
    correct_id = fixed_key.key_identifier(b"correct")
    ids = tuple(f"cal-{index}" for index in range(4))
    sources = {
        source_id: {
            "status": "SCORED", "statistic": float(index),
            "receiver_protocol_id": candidate.PROTOCOL_ID,
            "candidate_id": candidate.RECEIVER_ID, "spec_sha256": spec,
            "correct_key_id": correct_id,
        }
        for index, source_id in enumerate(ids)
    }
    calibration = candidate.calibrate(sources, ids, spec, correct_id)
    assert calibration["status"] == "FROZEN"
    assert calibration["threshold"] == 3.0 + 1e-6
    assert calibration["source_count"] == 4
    assert candidate.calibrate(dict(list(sources.items())[:3]), ids, spec, correct_id)["status"] == "UNCALIBRATED"


def test_runner_uses_candidate_correct_detection_as_original(monkeypatch):
    marker = {"status": "SCORED", "existence_statistic": 0.5}
    candidate_row = {"status": "SCORED", "correct_detection": marker}
    calls = []

    def candidate_read(obs, key, spec):
        calls.append((obs, key, spec))
        return candidate_row

    monkeypatch.setattr(runner.candidate, "read", candidate_read)
    monkeypatch.setattr(runner.split_receiver, "read", lambda *args: {"status": "SCORED"})
    phases = {str(phase): {"status": "COMPLETE"} for phase in range(4)}
    rows = runner._receiver_rows(observations(), phases, b"correct", "f" * 64)
    assert len(calls) == 1
    assert rows["ORIGINAL"]["detection"] is marker
    assert rows[candidate.RECEIVER_ID]["detection"] is candidate_row


def test_source_records_reject_failed_phase_even_with_scored_detection():
    originals = runner._configure_shared()
    item = runner.shared.empty_arm("OFF")
    runner._restore_shared(originals)
    spec = "9" * 64
    key_id = fixed_key.key_identifier(b"correct")
    for view in runner.VIEWS:
        item["views"][view]["observations"] = {
            str(phase): {"status": "COMPLETE"} for phase in runner.PHASES
        }
        split = {
            "status": "SCORED", "spec_sha256": spec, "key_id": key_id,
            "receiver_protocol_id": runner.split_receiver.PROTOCOL_ID,
            "confirmation_scores": {name: 0.2 for name in runner.split_receiver.CANDIDATES},
        }
        normalized = {
            "status": "SCORED", "statistic": 2.0,
            "receiver_protocol_id": candidate.PROTOCOL_ID,
            "candidate_id": candidate.RECEIVER_ID, "spec_sha256": spec,
            "correct_key_id": key_id,
        }
        for receiver in runner.split_receiver.CANDIDATES:
            item["views"][view]["receivers"][receiver] = {
                "status": "SCORED", "detection": split, "decision": {},
            }
        item["views"][view]["receivers"][candidate.RECEIVER_ID] = {
            "status": "SCORED", "detection": normalized, "decision": {},
        }
    item["views"]["FULL"]["observations"]["2"]["status"] = "FAILED"
    assert runner._source_record(item, "C1_MATCHED_CONFIRM", spec, key_id)["status"] == "INVALID"
    assert runner._source_record(item, candidate.RECEIVER_ID, spec, key_id)["status"] == "INVALID"
    assert Path(runner.shared.__file__) in runner._source_files()
    assert Path(payload_control.__file__) in runner._source_files()


def test_manifest_matches_fixed_new_source_roster_and_call_plan():
    config = json.loads((runner.MANIFEST).read_text())
    runner.validate_manifest(config)
    assert len(config["cases"]) == 8
    assert [case["seed"] for case in config["cases"]] == list(range(2026092301, 2026092309))
    assert config["fixed_denominator"]["physical_source_arms"] == 16
    assert config["expected_complete_calls"] == runner.TOTAL_PLAN
    assert config["candidate_search_accounting"] == {
        "keys_per_view": 17, "full_blind_searches": 816, "path_attempts": 3495744,
    }


def test_builder_emits_fixed_unexecuted_notebook(tmp_path):
    source = "1" * 40
    output = build(source, tmp_path / "candidate.ipynb")
    notebook = json.loads(output.read_text())
    assert notebook["cells"][0]["source"] == [
        "from google.colab import drive\n", "drive.mount('/content/drive')"
    ]
    text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "torch==2.11.0" in text and "diffusers==0.40.0" in text
    assert "transformers').split('.')[0]) >= 5" not in text
    assert "'diffusers==0.40.0', 'transformers', 'accelerate'" in text
    assert "environment_setup.json" in text
    assert "content_background_existence_run" in text
    assert "check=False" in text and "runner produced no retained result.json" in text
    assert "get_device_name" in text and "A100" not in text
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell.get("execution_count") is None and not cell.get("outputs")
    assert notebook["metadata"]["source_commit"] == source


def _fake_original(score, key_id):
    return {
        "status": "SCORED", "existence_statistic": score,
        "receiver_protocol_id": fixed_key.PROTOCOL_ID, "key_id": key_id,
        "best": {"path": {}}, "fixed_path_diagnostics": {},
    }


def _fake_split(score, spec, key_id):
    return {
        "status": "SCORED", "receiver_protocol_id": runner.split_receiver.PROTOCOL_ID,
        "spec_sha256": spec, "key_id": key_id,
        "confirmation_scores": {name: score for name in runner.split_receiver.CANDIDATES},
        "alignment_reporting_only": {}, "fixed_path_diagnostics": {},
    }


def _fake_candidate(score, spec, key_id, original):
    return {
        "status": "SCORED", "receiver_protocol_id": candidate.PROTOCOL_ID,
        "candidate_id": candidate.RECEIVER_ID, "spec_sha256": spec,
        "correct_key_id": key_id, "statistic": score,
        "correct_detection": original, "wrong_scores": [0.0] * 16,
        "wrong_mean": 0.0, "wrong_population_std": 1.0,
    }


def _case_calls(role):
    values = {
        "generation": 1,
        "transformer": 100 if role == "calibration_off" else 124,
        "scheduler_step": 50 if role == "calibration_off" else 62,
        "zero_shadow_step": 0 if role == "calibration_off" else 3,
        "unit_response_probe_step": 0 if role == "calibration_off" else 3,
        "clean_leaf_backward": 0 if role == "calibration_off" else 3,
        "vae_decode": 1 if role == "calibration_off" else 3,
        "mp4_save": 3 if role == "calibration_off" else 9,
        "mp4_read": 3 if role == "calibration_off" else 9,
        "vae_encode": 12 if role == "calibration_off" else 36,
    }
    return {kind + "_" + suffix: value for kind, value in values.items() for suffix in ("attempted", "completed")}


def test_stubbed_run_freezes_four_calibrations_before_evaluation_and_counts_exact(monkeypatch, tmp_path):
    config = runner.load(runner.MANIFEST)
    case_map = {case["id"]: case for case in config["cases"]}
    spec = runner.sha(runner.SPEC)
    key_id = fixed_key.key_identifier(config["key_utf8"].encode())
    order = []

    def fake_child(command, log_path):
        case_id = command[command.index("--case-id") + 1]
        stage = command[command.index("--stage") + 1]
        case_root = Path(command[command.index("--output") + 1])
        case_root.mkdir(parents=True, exist_ok=True)
        case = case_map[case_id]
        order.append((case_id, stage))
        record = runner.shared.empty_case(case)
        record.update(status="GENERATION_COMPLETE" if stage == "generate" else "EXECUTION_COMPLETE", failures=[], actual_calls=_case_calls(case["role"]))
        if stage == "generate":
            runner.shared.dump(case_root / "generation.json", record)
            return 0
        if case["role"] == "evaluation":
            calibrations = json.loads((case_root.parent / "calibrations.json").read_text())
            assert all(row["status"] == "FROZEN" for row in calibrations.values())
        for arm, item in record["videos"].items():
            item["status"] = "MEDIA_COMPLETE"
            for view_index, view in enumerate(runner.VIEWS):
                score = (0.1 + 0.01 * view_index) if case["role"] == "calibration_off" else (0.2 + 0.01 * view_index)
                original = _fake_original(score, key_id)
                split = _fake_split(score, spec, key_id)
                normalized = _fake_candidate(score + 1.0, spec, key_id, original)
                view_row = item["views"][view]
                view_row["status"] = "SCORED"
                view_row["observations"] = {str(phase): {"status": "COMPLETE"} for phase in runner.PHASES}
                view_row["receivers"] = {
                    "ORIGINAL": {"status": "SCORED", "detection": original, "decision": {}},
                    "C1_MATCHED_CONFIRM": {"status": "SCORED", "detection": split, "decision": {}},
                    "C2_STATE_CONFIRM": {"status": "SCORED", "detection": split, "decision": {}},
                    candidate.RECEIVER_ID: {"status": "SCORED", "detection": normalized, "decision": {}},
                }
        runner.shared.dump(case_root / "result.json", record)
        return 0

    monkeypatch.setattr(runner.shared, "_run_child", fake_child)
    result = runner.run_all(runner.MANIFEST, tmp_path / "run")
    assert result["status"] == "EXECUTION_COMPLETE"
    assert all(row["status"] == "FROZEN" and row["source_count"] == 4 for row in result["calibrations"].values())
    assert all(row["status"] == "EXACT" for row in result["call_accounting"].values())
    assert result["fixed_denominator"]["physical_source_arms"] == 16
    assert result["summary"]["per_receiver_and_arm"][candidate.RECEIVER_ID]["OFF"]["views"]["denominator"] == 12
    assert result["summary"]["same_source_marked_off_reporting_only"][candidate.RECEIVER_ID]["denominator"] == 24
    assert result["summary"]["decision_overlap_candidate_vs_baselines"]["ORIGINAL"]["evaluation_marked_views"]["denominator"] == 24
    assert [case for case, stage in order if stage == "generate"] == [case["id"] for case in config["cases"]]
