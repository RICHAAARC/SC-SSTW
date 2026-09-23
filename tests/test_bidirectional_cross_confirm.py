import ast
import importlib.metadata
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from experiments.wan_state_clock import bidirectional_cross_confirm_run as runner
from experiments.wan_state_clock import window_state_mse_paired_run as shared
from main.tube_state import bidirectional_cross_confirm_receiver as bidirectional
from main.tube_state import fixed_key
from main.tube_state import fixed_key_split_receiver as split

KEY = b"WanProjection-first-validation-key-v1"
SOURCE = "a" * 40
pytestmark = pytest.mark.unit


def _observations(seed=7):
    rng = np.random.default_rng(seed)
    return {
        phase: rng.normal(0, 0.01, size=(1, 16, 46, 40, 64)).astype(np.float32)
        for phase in range(4)
    }


def _one_path_select(locator):
    row = locator.score_path(fixed_key.REFERENCE_PATHS["IDENTITY"])
    return {"best": row, "attempted_path_count": 4284, "scored_path_count": 1}


def _joint_partition_role_swap(observations, book):
    swapped_observations = {
        phase: np.roll(value, 4, axis=3) for phase, value in observations.items()
    }
    swapped_book = dict(book)
    swapped_book["directions"] = np.roll(
        book["directions"].reshape(11, 10, 16, 4, 256), 1, axis=1,
    ).reshape(book["directions"].shape)
    for name in ("sync", "polarity", "code"):
        swapped_book[name] = np.roll(
            book[name].reshape(11, 10, 16), 1, axis=1,
        ).reshape(book[name].shape)
    return swapped_observations, swapped_book


def test_a_locator_b_confirmation_matches_existing_split_fields(monkeypatch):
    monkeypatch.setattr(bidirectional, "_select", _one_path_select)
    observations = _observations()
    book = fixed_key.codebook(KEY)
    direction = bidirectional._direction(observations, book, split.A, "A", split.B, "B")

    locator = split.PartitionEvidence(observations, book, split.A, "A")
    old_selection = _one_path_select(locator)
    confirmation = split.PartitionEvidence(observations, book, split.B, "B")
    old_heldout = confirmation.score_path(old_selection["best"]["path"])
    assert direction["selection"] == old_selection
    assert direction["confirmation"] == old_heldout
    assert direction["confirmation_scores"] == split.confirmation_scores(old_heldout)
    assert direction["confirmation_path_evaluations"] == 4


def test_joint_observation_and_codebook_partition_swap_exchanges_directions_and_preserves_mean(monkeypatch):
    monkeypatch.setattr(bidirectional, "_select", _one_path_select)
    observations = _observations(11)
    book = fixed_key.codebook(KEY)
    original = bidirectional._read_book(observations, book, "spec")
    swapped_observations, swapped_book = _joint_partition_role_swap(observations, book)
    swapped = bidirectional._read_book(swapped_observations, swapped_book, "spec")
    assert swapped["candidate_score"] == pytest.approx(original["candidate_score"], abs=1e-15)
    assert swapped["c2_candidate_components"]["A_LOCATE_B_CONFIRM"] == pytest.approx(
        original["c2_candidate_components"]["B_LOCATE_A_CONFIRM"], abs=1e-15,
    )
    assert swapped["c2_candidate_components"]["B_LOCATE_A_CONFIRM"] == pytest.approx(
        original["c2_candidate_components"]["A_LOCATE_B_CONFIRM"], abs=1e-15,
    )


def test_candidate_invalid_if_either_direction_invalid(monkeypatch):
    observations = {
        phase: np.broadcast_to(np.zeros(1, dtype=np.float32), (1, 16, 46, 40, 64))
        for phase in range(4)
    }

    def fake_direction(observations, book, locator_indices, locator_name, *args):
        if locator_name == "A":
            return {"status": "INVALID", "confirmation_scores": {"C1_MATCHED_CONFIRM": None, "C2_STATE_CONFIRM": None}}
        return {"status": "SCORED", "confirmation_scores": {"C1_MATCHED_CONFIRM": 1.0, "C2_STATE_CONFIRM": 2.0}}

    monkeypatch.setattr(bidirectional, "_direction", fake_direction)
    result = bidirectional.read(observations, KEY, "spec")
    assert result["status"] == "INVALID"
    assert result["candidate_score"] is None


def _bound_source(receiver, case_id, score):
    if receiver == "ORIGINAL":
        return {
            "status": "SCORED", "statistic": score,
            "receiver_protocol_id": fixed_key.PROTOCOL_ID,
            "key_id": fixed_key.key_identifier(KEY),
        }
    if receiver == bidirectional.CANDIDATE_ID:
        return {
            "status": "SCORED", "statistic": score,
            "candidate_id": receiver, "spec_sha256": "spec",
            "receiver_protocol_id": bidirectional.PROTOCOL_ID,
            "key_id": fixed_key.key_identifier(KEY),
        }
    return {
        "status": "SCORED", "statistic": score,
        "candidate_id": receiver, "spec_sha256": "spec",
        "receiver_protocol_id": split.PROTOCOL_ID,
        "key_id": fixed_key.key_identifier(KEY),
    }


@pytest.mark.parametrize("receiver", runner.RECEIVERS)
def test_each_receiver_requires_own_complete_four_source_calibration(receiver):
    sources = {
        case_id: _bound_source(receiver, case_id, index / 10)
        for index, case_id in enumerate(bidirectional.CALIBRATION_SOURCE_IDS)
    }
    calibration = runner._freeze_four_sources(
        sources, receiver, "spec", fixed_key.key_identifier(KEY), 1e-6,
    )
    assert calibration["status"] == "FROZEN"
    assert calibration["source_count"] == 4
    assert calibration["threshold"] == pytest.approx(0.300001)
    sources.pop(bidirectional.CALIBRATION_SOURCE_IDS[-1])
    assert runner._freeze_four_sources(
        sources, receiver, "spec", fixed_key.key_identifier(KEY), 1e-6,
    )["status"] == "UNCALIBRATED"


def test_shared_runner_overrides_restore_and_legacy_plan_is_present_from_first_save(monkeypatch, tmp_path):
    before = {name: getattr(shared, name) for name in runner._SHARED_OVERRIDES}
    captured = {}

    def fake_generate(case_id, config_path, output):
        case = next(row for row in runner.load(config_path)["cases"] if row["id"] == case_id)
        captured["plan"] = shared.generation_call_plan(case)
        generated = {
            "arms": {"OFF": {}, "SINGLE46": {}},
            "writer_objective": {"id": "legacy"},
        }
        captured["off_objective"] = shared.branch_writer_objective(generated["arms"]["OFF"], case, generated)
        captured["marked_objective"] = shared.branch_writer_objective(generated["arms"]["SINGLE46"], case, generated)
        return {"status": "GENERATION_COMPLETE"}

    monkeypatch.setattr(shared, "generate_case", fake_generate)
    case_id = "new_eval_e0_s5"
    result = runner.generate_case(case_id, runner.MANIFEST, tmp_path)
    assert result["status"] == "GENERATION_COMPLETE"
    assert captured["plan"] == {
        "generation": 1, "transformer": 124, "scheduler_step": 62,
        "zero_shadow_step": 3, "unit_response_probe_step": 3,
        "clean_leaf_backward": 3,
    }
    assert captured["off_objective"] is None
    assert captured["marked_objective"] == {"id": "legacy"}
    assert all(getattr(shared, name) is value for name, value in before.items())


def test_manifest_and_notebook_builder_static_contract(monkeypatch, tmp_path):
    runner.validate_manifest(runner.load(runner.MANIFEST))
    from scripts import build_bidirectional_cross_confirm_notebook as builder

    path = builder.build(SOURCE, tmp_path / "candidate.ipynb")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert "".join(notebook["cells"][0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
    assert notebook["metadata"]["source_commit"] == SOURCE
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
    install = "".join(notebook["cells"][3]["source"])
    assert "torch==2.11.0" in install and "diffusers==0.40.0" in install
    assert "transformers" in install and "transformers<5" not in install
    run = "".join(notebook["cells"][6]["source"])
    assert "bidirectional_cross_confirm_run" in run
    assert "if not (OUTPUT / 'result.json').exists()" in run
    results = "".join(notebook["cells"][7]["source"])
    assert "comparison_records" in results and "BIDIRECTIONAL_C2_MEAN" in results
    assert "evaluation-only:" in results and "OFF_views=12" in results


def _fake_original(score):
    return {
        "status": "SCORED", "existence_statistic": score,
        "receiver_protocol_id": fixed_key.PROTOCOL_ID,
        "key_id": fixed_key.key_identifier(KEY),
        "best": None, "fixed_path_diagnostics": {},
    }


def _fake_split(score, spec_sha):
    diagnostics = {
        name: {"scores": {candidate: score - 0.01 for candidate in split.CANDIDATES}}
        for name in fixed_key.REFERENCE_PATHS
    }
    return {
        "status": "SCORED", "spec_sha256": spec_sha,
        "key_id": fixed_key.key_identifier(KEY),
        "receiver_protocol_id": split.PROTOCOL_ID,
        "confirmation_scores": {candidate: score for candidate in split.CANDIDATES},
        "fixed_path_diagnostics": diagnostics,
        "alignment_reporting_only": {view: {"status": "REPORTED"} for view in runner.VIEWS},
    }


def _fake_bidirectional(score, spec_sha):
    directions = {
        direction: {
            "fixed_path_diagnostics": {
                name: {"scores": {"C2_STATE_CONFIRM": score - 0.02}}
                for name in fixed_key.REFERENCE_PATHS
            }
        }
        for direction in ("A_LOCATE_B_CONFIRM", "B_LOCATE_A_CONFIRM")
    }
    return {
        "status": "SCORED", "candidate_score": score,
        "candidate_id": bidirectional.CANDIDATE_ID,
        "spec_sha256": spec_sha, "key_id": fixed_key.key_identifier(KEY),
        "receiver_protocol_id": bidirectional.PROTOCOL_ID,
        "c2_candidate_components": {
            "A_LOCATE_B_CONFIRM": score, "B_LOCATE_A_CONFIRM": score,
        },
        "directions": directions,
        "alignment_reporting_only": {view: {"status": "REPORTED"} for view in runner.VIEWS},
    }


def _counts(role):
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
    return {
        kind + "_" + status: value
        for kind, value in values.items() for status in ("attempted", "completed")
    }


def test_run_all_retains_missing_calibration_phase_and_all_slots(monkeypatch, tmp_path):
    config = runner.load(runner.MANIFEST)
    case_map = {case["id"]: case for case in config["cases"]}
    spec_sha = runner.sha(runner.SPEC)

    def child(command, log_path):
        case_id = command[command.index("--case-id") + 1]
        stage = command[command.index("--stage") + 1]
        case_root = Path(command[command.index("--output") + 1])
        case_root.mkdir(parents=True, exist_ok=True)
        case = case_map[case_id]
        record = runner.empty_case(case)
        record["actual_calls"] = _counts(case["role"])
        if stage == "generate":
            record["status"] = "GENERATION_COMPLETE"
            runner.dump(case_root / "generation.json", record)
            return 0
        record["status"] = "EXECUTION_COMPLETE"
        for arm_index, (arm, item) in enumerate(record["videos"].items()):
            item["status"] = "MEDIA_COMPLETE"
            for view_index, view in enumerate(runner.VIEWS):
                score = 0.1 + arm_index * 0.2 + view_index * 0.01
                view_row = item["views"][view]
                view_row["status"] = "SCORED"
                view_row["observations"] = {
                    str(phase): {"status": "COMPLETE"} for phase in runner.PHASES
                }
                if case_id == bidirectional.CALIBRATION_SOURCE_IDS[0] and view == "FULL":
                    view_row["observations"]["0"]["status"] = "FAILED"
                original = _fake_original(score)
                split_detection = _fake_split(score, spec_sha)
                candidate = _fake_bidirectional(score, spec_sha)
                view_row["receivers"] = {
                    "ORIGINAL": {"status": "SCORED", "detection": original, "decision": {"status": "NOT_RUN"}},
                    "C1_MATCHED_CONFIRM": {"status": "SCORED", "detection": split_detection, "decision": {"status": "NOT_RUN"}},
                    "C2_STATE_CONFIRM": {"status": "SCORED", "detection": split_detection, "decision": {"status": "NOT_RUN"}},
                    bidirectional.CANDIDATE_ID: {"status": "SCORED", "detection": candidate, "decision": {"status": "NOT_RUN"}},
                }
        runner.dump(case_root / "result.json", record)
        return 0

    monkeypatch.setattr(runner, "_run_child", child)
    result = runner.run_all(runner.MANIFEST, tmp_path / "retained")
    assert result["status"] == "WITH_RETAINED_FAILURES"
    assert all(value["status"] == "UNCALIBRATED" for value in result["calibrations"].values())
    assert sum(len(case["videos"]) for case in result["cases"].values()) == 16
    assert sum(
        len(arm["views"]) for case in result["cases"].values() for arm in case["videos"].values()
    ) == 48
    assert len(result["comparison_records"]) == 36
    assert all(
        row[bidirectional.CANDIDATE_ID]["fixed_reference_margin_vs_own_threshold"] is None
        for row in result["comparison_records"]
    )
    assert all(value["status"] == "EXACT" for value in result["call_accounting"].values())
