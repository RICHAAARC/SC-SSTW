"""CPU/fake/static checks for the paired runner; no model, GPU, or real media."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from main.tube_state import fixed_key
from main.tube_state import fixed_key_split_receiver as split_receiver
from main.tube_state import projection_margin as carrier
from runtime.wan import fixed_key_control, fixed_key_core, generation, integrated_core, io, payload_control, trajectory, vae
from experiments.wan_state_clock import window_state_mse_paired_run as paired

pytestmark = pytest.mark.unit
torch.set_num_threads(1)
KEY = b"WanProjection-first-validation-key-v1"
SOURCE_TOKEN = "1" * 40


class FakeScheduler:
    def __init__(self):
        self.config = SimpleNamespace(
            prediction_type="flow_prediction", thresholding=False, lower_order_final=True,
        )
        self.predict_x0 = True
        self.timesteps = torch.arange(50, 0, -1)
        self.sigmas = torch.linspace(1, 0, 51)
        self.step_index = None
        self.history = []

    def step(self, velocity, timestep, state, return_dict=False):
        if self.step_index is None:
            self.step_index = 0
        self.history.append(float(state.mean()))
        self.step_index += 1
        return (state - 0.01 * velocity,)


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, hidden_states, **kwargs):
        return (hidden_states * 0.1 + self.weight * 0.01,)


def fake_pipe():
    return SimpleNamespace(scheduler=FakeScheduler(), transformer=FakeTransformer())


def test_manifest_denominators_source_receipt_and_frozen_receiver():
    config = paired.load(paired.MANIFEST)
    paired.validate_manifest(config)
    cases = {case["id"]: paired.empty_case(case) for case in config["cases"]}
    assert sum(len(case["videos"]) for case in cases.values()) == 12
    assert sum(len(arm["views"]) for case in cases.values() for arm in case["videos"].values()) == 36
    assert sum(
        len(view["observations"])
        for case in cases.values() for arm in case["videos"].values() for view in arm["views"].values()
    ) == 144
    assert sum(
        len(view["receivers"])
        for case in cases.values() for arm in case["videos"].values() for view in arm["views"].values()
    ) == 108
    assert sum(
        len(arm["receiver_source_decisions"])
        for case in cases.values() for arm in case["videos"].values()
    ) == 36
    source_files = paired._source_files()
    for module in (payload_control, generation, integrated_core, io, vae):
        assert Path(module.__file__) in source_files
    assert "prepare_direction.__code__" not in Path(paired.__file__).read_text()
    assert hashlib.sha256(Path(split_receiver.__file__).read_bytes()).hexdigest() == (
        "c69cb4ac718d9f30744a562fa9f6d549de4926fa71686e52d5df408d519ec641"
    )


def test_paired_core_one_prefix_five_cloned_forks_exact_calls(monkeypatch):
    monkeypatch.setattr(
        fixed_key_core, "prepare_generation",
        lambda *args, **kwargs: (
            fake_pipe(), torch.zeros(carrier.SHAPE), torch.tensor(1.0), torch.tensor(-1.0), torch.float32,
        ),
    )
    passed_states = []
    snapshot_records = []
    original_branch = fixed_key_core._marker_branch

    def branch(pipe, state44, snapshot44, *args, **kwargs):
        passed_states.append((state44, state44.detach().clone()))
        before = trajectory.fingerprint(vars(snapshot44))
        result = original_branch(pipe, state44, snapshot44, *args, **kwargs)
        snapshot_records.append((before, trajectory.fingerprint(vars(snapshot44))))
        return result

    monkeypatch.setattr(fixed_key_core, "_marker_branch", branch)
    events = []
    config = fixed_key_core.generation_config(fixed_key_core.load_protocol(), "fixed prompt", 17)
    result = fixed_key_core.generate_paired_key_terminals(
        config, KEY, lambda kind, complete: events.append((kind, complete)),
    )
    assert tuple(result["arms"]) == paired.EVALUATION_ARMS
    assert len({state.data_ptr() for state, _ in passed_states}) == 5
    for _, state in passed_states[1:]:
        torch.testing.assert_close(state, passed_states[0][1], rtol=0, atol=0)
    assert all(before == after for before, after in snapshot_records)
    assert events.count(("transformer", True)) == 148
    assert events.count(("scheduler_step", True)) == 74
    for kind in ("zero_shadow_step", "unit_response_probe_step", "clean_leaf_backward"):
        assert events.count((kind, True)) == 6
    assert result["arms"]["OFF"]["writer_objective"] is None
    for prefix, objective in (
        ("LEGACY", fixed_key_control.DEFAULT_OBJECTIVE),
        ("MSE", fixed_key_control.WINDOW_STATE_MSE_OBJECTIVE),
    ):
        for schedule in ("SINGLE46", "MULTI44_46"):
            row = result["arms"][prefix + "_" + schedule]
            assert row["writer_objective"]["id"] == objective
            assert all(gradient["objective"] == objective for gradient in row["clean_gradients"])
            assert row["cumulative_native_response"]["actual_D"]["support"]["sum_rms"] == pytest.approx(
                fixed_key_control.R_STAR, rel=2e-5,
            )


def _original_detection(score):
    return {
        "status": "SCORED", "existence_statistic": score,
        "receiver_protocol_id": fixed_key.PROTOCOL_ID,
        "key_id": fixed_key.key_identifier(KEY), "best": None,
        "fixed_path_diagnostics": {
            name: {"score": score - 0.01} for name in fixed_key.REFERENCE_PATHS
        },
    }


def _split_detection(score, spec_sha):
    return {
        "status": "SCORED", "spec_sha256": spec_sha,
        "key_id": fixed_key.key_identifier(KEY),
        "receiver_protocol_id": split_receiver.PROTOCOL_ID,
        "confirmation_scores": {
            "C1_MATCHED_CONFIRM": score,
            "C2_STATE_CONFIRM": score - 0.02,
        },
        "fixed_path_diagnostics": {
            name: {"scores": {
                "C1_MATCHED_CONFIRM": score - 0.01,
                "C2_STATE_CONFIRM": score - 0.03,
            }}
            for name in fixed_key.REFERENCE_PATHS
        },
        "alignment_reporting_only": {
            view: {"status": "REPORTED", "windows": []} for view in paired.VIEWS
        },
    }


def test_receiver_exceptions_are_isolated_and_failed_phase_cannot_calibrate(monkeypatch):
    phases = {str(phase): {"status": "COMPLETE"} for phase in paired.PHASES}
    monkeypatch.setattr(fixed_key, "read", lambda *args: (_ for _ in ()).throw(RuntimeError("original")))
    monkeypatch.setattr(split_receiver, "read", lambda *args: _split_detection(0.2, "spec"))
    rows = paired._receiver_rows({}, phases, KEY, "spec")
    assert rows["ORIGINAL"]["status"] == "PARTIAL_OR_FAILED"
    assert rows["C1_MATCHED_CONFIRM"]["status"] == "SCORED"
    monkeypatch.setattr(fixed_key, "read", lambda *args: _original_detection(0.2))
    monkeypatch.setattr(split_receiver, "read", lambda *args: (_ for _ in ()).throw(RuntimeError("split")))
    rows = paired._receiver_rows({}, phases, KEY, "spec")
    assert rows["ORIGINAL"]["status"] == "SCORED"
    assert rows["C1_MATCHED_CONFIRM"]["status"] == "PARTIAL_OR_FAILED"

    item = paired.empty_arm("OFF")
    for view in paired.VIEWS:
        item["views"][view]["status"] = "SCORED"
        item["views"][view]["observations"] = {
            str(phase): {"status": "COMPLETE"} for phase in paired.PHASES
        }
        item["views"][view]["receivers"]["C1_MATCHED_CONFIRM"] = {
            "status": "SCORED", "detection": _split_detection(0.2, "spec"),
            "decision": {"status": "NOT_RUN", "detected": None},
        }
    item["views"]["FULL"]["observations"]["2"]["status"] = "FAILED"
    source = paired._source_record(
        item, "C1_MATCHED_CONFIRM", "spec", fixed_key.key_identifier(KEY),
    )
    assert source["status"] == "INVALID"


def test_eval_media_reads_and_encodes_each_unique_view_once(monkeypatch, tmp_path):
    config = paired.load(paired.MANIFEST)
    case = next(row for row in config["cases"] if row["role"] == "evaluation")
    case_root = tmp_path / case["id"]
    case_root.mkdir()
    record = paired.empty_case(case)
    record.update(status="GENERATION_COMPLETE", failures=[], actual_calls={})
    paired.dump(case_root / "generation.json", record)
    for arm in paired.case_arms(case):
        torch.save(torch.zeros(1), case_root / f"{arm}_terminal.pt")

    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    reads = []
    four_phase_calls = []
    receiver_observation_ids = {}
    spec_sha = paired.sha(paired.SPEC)

    monkeypatch.setattr(paired, "load_frozen_vae", lambda protocol: FakeVAE())
    monkeypatch.setattr(
        paired, "decode_normalized_latent",
        lambda vae, terminal: torch.zeros(181, 2, 2, 3),
    )

    def save_mp4(rgb, path, fps, crf):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"stub-mp4")

    def read_mp4(path):
        reads.append(path)
        return torch.zeros(181, 2, 2, 3)

    def encode_phases(pixels, vae, count, on_phase):
        observations = {phase: object() for phase in paired.PHASES}
        four_phase_calls.append(observations)
        phases = {}
        for phase in paired.PHASES:
            count("vae_encode", False)
            count("vae_encode", True)
            row = {"status": "COMPLETE", "frames_used": 177, "tail_discarded": 0}
            phases[str(phase)] = row
            on_phase(phase, torch.zeros(1), row)
        return observations, phases

    def original_read(observations, book):
        receiver_observation_ids.setdefault(id(observations), []).append("ORIGINAL")
        return _original_detection(0.2)

    def split_read(observations, key, received_spec_sha):
        assert received_spec_sha == spec_sha
        receiver_observation_ids.setdefault(id(observations), []).append("SPLIT")
        return _split_detection(0.2, spec_sha)

    monkeypatch.setattr(paired, "encode_rgb", save_mp4)
    monkeypatch.setattr(paired, "read_mp4", read_mp4)
    monkeypatch.setattr(paired, "encode_four_phases", encode_phases)
    monkeypatch.setattr(fixed_key, "read", original_read)
    monkeypatch.setattr(split_receiver, "read", split_read)
    result = paired.media_case(case["id"], paired.MANIFEST, case_root)
    assert result["status"] == "EXECUTION_COMPLETE"
    assert len(reads) == 15 and len(four_phase_calls) == 15
    assert all(receiver_observation_ids[id(observations)] == ["ORIGINAL", "SPLIT"] for observations in four_phase_calls)
    expected = {"vae_decode": 5, "mp4_save": 15, "mp4_read": 15, "vae_encode": 60}
    for kind, value in expected.items():
        assert result["actual_calls"][kind + "_attempted"] == value
        assert result["actual_calls"][kind + "_completed"] == value


def _full_call_counts(role):
    values = {
        "generation": 1,
        "transformer": 100 if role == "calibration_off" else 148,
        "scheduler_step": 50 if role == "calibration_off" else 74,
        "zero_shadow_step": 0 if role == "calibration_off" else 6,
        "unit_response_probe_step": 0 if role == "calibration_off" else 6,
        "clean_leaf_backward": 0 if role == "calibration_off" else 6,
        "vae_decode": 1 if role == "calibration_off" else 5,
        "mp4_save": 3 if role == "calibration_off" else 15,
        "mp4_read": 3 if role == "calibration_off" else 15,
        "vae_encode": 12 if role == "calibration_off" else 60,
    }
    return {
        kind + "_" + status: value
        for kind, value in values.items() for status in ("attempted", "completed")
    }


def test_runner_stub_exact_counts_calibration_order_and_pair_records(monkeypatch, tmp_path):
    config = paired.load(paired.MANIFEST)
    case_map = {case["id"]: case for case in config["cases"]}
    spec_sha = paired.sha(paired.SPEC)
    order = []

    def child(command, log_path):
        case_id = command[command.index("--case-id") + 1]
        stage = command[command.index("--stage") + 1]
        case_root = Path(command[command.index("--output") + 1])
        case_root.mkdir(parents=True, exist_ok=True)
        case = case_map[case_id]
        order.append((case_id, stage))
        record = paired.empty_case(case)
        record["actual_calls"] = _full_call_counts(case["role"])
        record["fresh_generation"] = {
            "initial_noise_fingerprint": case_id + "-noise",
            "state44_fingerprint": case_id + "-state44",
            "scheduler44_fingerprint": case_id + "-scheduler44",
        }
        if stage == "generate":
            record["status"] = "GENERATION_COMPLETE"
            paired.dump(case_root / "generation.json", record)
            return 0
        if case["role"] == "evaluation":
            assert paired.load(case_root.parent / "calibrations.json")["ORIGINAL"]["status"] == "FROZEN"
        record["status"] = "EXECUTION_COMPLETE"
        for arm, item in record["videos"].items():
            item["status"] = "MEDIA_COMPLETE"
            base = 0.1 if arm == "OFF" else (0.3 if arm.startswith("LEGACY") else 0.4)
            for offset, view in enumerate(paired.VIEWS):
                view_row = item["views"][view]
                view_row["status"] = "SCORED"
                view_row["observations"] = {
                    str(phase): {"status": "COMPLETE"} for phase in paired.PHASES
                }
                score = base + 0.01 * offset
                split = _split_detection(score, spec_sha)
                view_row["receivers"] = {
                    "ORIGINAL": {"status": "SCORED", "detection": _original_detection(score), "decision": {"status": "NOT_RUN", "detected": None}},
                    "C1_MATCHED_CONFIRM": {"status": "SCORED", "detection": split, "decision": {"status": "NOT_RUN", "detected": None}},
                    "C2_STATE_CONFIRM": {"status": "SCORED", "detection": split, "decision": {"status": "NOT_RUN", "detected": None}},
                }
        paired.dump(case_root / "result.json", record)
        return 0

    monkeypatch.setattr(paired, "_run_child", child)
    result = paired.run_all(paired.MANIFEST, tmp_path / "ok")
    assert result["status"] == "EXECUTION_COMPLETE"
    assert all(row["status"] == "EXACT" for row in result["call_accounting"].values())
    assert all(calibration["status"] == "FROZEN" for calibration in result["calibrations"].values())
    assert len(result["paired_comparisons"]) == 12
    assert all(len(row["views"]) == 3 for row in result["paired_comparisons"])
    assert all(row["source_score_difference_mse_minus_legacy"] == pytest.approx(0.1) for row in result["paired_comparisons"])
    assert all(row["shared_off_reference"]["meaning"].startswith("one physical OFF") for row in result["paired_comparisons"])
    assert order[:4] == [
        ("cal_off_p0_s1", "generate"), ("cal_off_p0_s1", "media"),
        ("cal_off_p1_s1", "generate"), ("cal_off_p1_s1", "media"),
    ]


def test_runner_retains_slots_on_truncated_child_json(monkeypatch, tmp_path):
    def child(command, log_path):
        stage = command[command.index("--stage") + 1]
        case_root = Path(command[command.index("--output") + 1])
        case_root.mkdir(parents=True, exist_ok=True)
        target = case_root / ("generation.json" if stage == "generate" else "result.json")
        target.write_text("{truncated", encoding="utf-8")
        return 9

    monkeypatch.setattr(paired, "_run_child", child)
    result = paired.run_all(paired.MANIFEST, tmp_path / "truncated")
    assert result["status"] == "WITH_RETAINED_FAILURES"
    assert sum(failure.get("kind") == "CHILD_RESULT_INVALID" for failure in result["failures"]) == 8
    assert sum(len(case["videos"]) for case in result["cases"].values()) == 12
    assert sum(
        len(view["receivers"])
        for case in result["cases"].values()
        for arm in case["videos"].values()
        for view in arm["views"].values()
    ) == 108
    assert all(case.get("media_exit_code") == 9 for case in result["cases"].values())


def test_notebook_builder_stdlib_schema_ast_and_nonzero_result_handoff(monkeypatch, tmp_path):
    from scripts import build_window_state_mse_paired_notebook as builder

    notebook_path = builder.build(SOURCE_TOKEN, tmp_path / "paired.ipynb")
    notebook = json.loads(notebook_path.read_text())
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] == 5
    assert notebook["metadata"]["source_commit"] == SOURCE_TOKEN
    assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
    assert "".join(notebook["cells"][0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
    for cell in notebook["cells"]:
        assert set(("cell_type", "metadata", "source", "id")) <= set(cell)
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []
    install = "".join(notebook["cells"][3]["source"])
    assert "torch==2.11.0" in install and "diffusers==0.40.0" in install and "environment_setup.json" in install
    commands = []

    class Child:
        def __init__(self, command, **kwargs):
            commands.append(command)
            self.stdout = iter(["stub output\n"])

        def wait(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Child)
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "stub")
    install_output = tmp_path / "install"
    install_output.mkdir()
    exec(install, {"OUTPUT": install_output})
    pip = [command for command in commands if "install" in command and "pip" in command]
    assert pip[0][4:] == [
        "torch==2.11.0", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128",
    ]
    assert pip[1][4:] == [
        "diffusers==0.40.0", "transformers", "accelerate", "ftfy", "sentencepiece",
        "safetensors", "huggingface_hub", "numpy", "Pillow",
    ]
    ast.parse(commands[-1][-1])
    assert "environment_setup.json" in commands[-1][-1]
    run_code = "".join(notebook["cells"][6]["source"])
    run_output = tmp_path / "retained"
    run_output.mkdir()
    (run_output / "result.json").write_text("{}", encoding="utf-8")

    class Completed:
        returncode = 7

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Completed())
    exec(run_code, {"OUTPUT": run_output, "REPO": tmp_path})
    receipt = json.loads((run_output / "execution_receipt.json").read_text())
    assert receipt["returncode"] == 7

    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        exec(run_code, {"OUTPUT": missing, "REPO": tmp_path})

    partial = {
        "status": "WITH_RETAINED_FAILURES", "cases": {}, "failures": [{"stage": "stub"}],
    }
    (run_output / "result.json").write_text(json.dumps(partial), encoding="utf-8")
    exec("".join(notebook["cells"][7]["source"]), {"OUTPUT": run_output})
