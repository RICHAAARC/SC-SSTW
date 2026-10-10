"""Static/CPU checks for the concrete staged real-evaluation entry."""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from experiments.paper_results_v1 import real_backends, real_eval, report


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/paper_results_v1"


def proposal():
    return report.read_json(PACKAGE / "real_eval.proposal.json")


def adopted():
    return report.read_json(PACKAGE / "real_eval.adopted.json")


def one_case_config():
    value = proposal()
    value["study_id"] = "synthetic_real_runner_state_fixture"
    value["cases"] = [copy.deepcopy(value["cases"][0])]
    return value


def interpreter():
    value = [sys.executable]
    if not os.access(sys.executable, os.X_OK):
        value = ["/lib64/ld-linux-x86-64.so.2", sys.executable]
    return value


def test_proposal_plan_has_separate_pilot_and_confirmation_fixed_denominators():
    config = real_eval.validate_real_config(proposal())
    plan = real_eval.build_plan(config)
    assert len(config["cases"]) == 10
    assert len(plan["artifacts"]) == 690
    assert len(plan["costs"]) == 110
    assert len(plan["receiver_slots"]) == 1600
    assert len(plan["baseline_slots"]) == 180
    assert len(plan["comparison_slots"]) == 0
    assert len(plan["quality_rows"]) == 70
    assert plan["receiver_resource_plan"] == {
        "sync_framewise_encodes": 360,
        "raw_unique_map_arm_encodes": 320,
        "aligned_or_path_encode_upper_bound": 880,
        "wan_physical_encode_upper_bound": 1200,
        "wan_reads": 1600,
    }
    confirmation = [row for row in plan["receiver_slots"] if row["case_id"].startswith("confirm_")]
    pilot = [row for row in plan["receiver_slots"] if row["case_id"].startswith("pilot_")]
    assert len(confirmation) == 1280 and len(pilot) == 320
    full = [row for row in config["cases"][0]["observations"] if row["observation_id"].endswith("global_00")]
    assert len(full) == 4
    assert all(row["analysis_role"] == "FULL_GEOMETRY_CONTROL_EXCLUDED_FROM_SYNC_GAIN" for row in full)
    assert config["payload_hex"] == "A6D39C5E"
    assert config["payload_bits"] != [int(bit) for byte in b"OKOK" for bit in f"{byte:08b}"]


def test_adopted_plan_freezes_native_tie_rules_and_paper_comparison_denominators():
    config = real_eval.validate_real_config(adopted())
    plan = real_eval.build_plan(config)
    audit = report.read_json(PACKAGE / "source_identity_audit.json")
    assert config["adoption"]["status"] == "ADOPTED_METHOD_DEFINITION_LOCAL_ONLY"
    assert config["real_execution_authorized"] is False
    assert config["keys"] == {"K0": "watermark", "K1": "watermark-wrong"}
    assert config["roster_adoption"].startswith("FIXED_TWO_PILOT_PLUS_EIGHT_CONFIRMATION_USER_ADOPTED")
    assert {case["source_status"] for case in config["cases"]} == {
        "USER_ADOPTED_LIMITED_LOCAL_NO_MATCH_UNPROVEN"
    }
    assert audit["candidate_case_ids"] == [case["case_id"] for case in config["cases"]]
    assert audit["result"] == "NO_MATCH_IN_SEARCHED_LOCAL_HISTORY"
    assert audit["unseen_status"] == "UNPROVEN_LIMITED_LOCAL_SEARCH"
    assert config["evaluation_rules"]["videoseal_32"]["rule"] == real_eval.VIDEOSEAL_NATIVE_TIE_RULE
    assert config["evaluation_rules"]["rivagan_sequence"]["rule"] == real_eval.RIVAGAN_NATIVE_TIE_RULE
    assert len(plan["comparison_slots"]) == 180

    for method in real_eval.BASELINES:
        confirmation = [
            row for row in plan["comparison_slots"]
            if row["cohort"] == "CONFIRMATION_CANDIDATE" and row["baseline_method"] == method
        ]
        nonfull = [row for row in confirmation if row["analysis_role"] == "PRIMARY_SYNC"]
        full = [row for row in confirmation if row["analysis_role"] != "PRIMARY_SYNC"]
        assert len(nonfull) == 64 and len(full) == 8
        assert all(row["main_arm"] == "PAYLOAD_FRAMEWISE_M05" for row in confirmation)
        assert all(row["main_key_label"] == "K0" for row in confirmation)
        assert all(row["main_mode"] == "GLOBAL" for row in nonfull if row["protocol"] == "GLOBAL")
        assert all(row["main_mode"] == "PATH" for row in nonfull if row["protocol"] == "SINGLE_JUMP")
        assert all(row["main_mode"] == "RAW" for row in full)
        pilot = [
            row for row in plan["comparison_slots"]
            if row["cohort"] == "PILOT_EXCLUDED_FROM_CONFIRMATION" and row["baseline_method"] == method
        ]
        assert len(pilot) == 18


def test_run_store_failure_retains_all_planned_rows_and_per_case_phase_state(tmp_path):
    config = one_case_config()
    store = real_eval.RunStore(tmp_path / "run", config, create=True)
    denominator = (
        len(store.data["artifacts"]), len(store.data["receiver_slots"]), len(store.data["costs"]),
    )
    store.phase_start("generate", "pilot_01")
    store.phase_failure("generate", RuntimeError("synthetic retained failure"), "pilot_01")
    store.cost(
        "pilot_01", "generate", status="FAILED", seconds=0.0,
        reason="RuntimeError: synthetic retained failure",
    )
    reopened = real_eval.RunStore(tmp_path / "run", config)
    assert denominator == (
        len(reopened.data["artifacts"]), len(reopened.data["receiver_slots"]), len(reopened.data["costs"]),
    )
    assert reopened.data["phases"]["generate"]["cases"]["pilot_01"]["status"] == "FAILED"
    terminals = [
        row for row in reopened.data["artifacts"]
        if row["artifact_id"] in (
            "pilot_01/OFF_NATIVE/TERMINAL", "pilot_01/PAYLOAD_NATIVE/TERMINAL",
        )
    ]
    assert len(terminals) == 2 and all(row["status"] == "FAILED" for row in terminals)
    assert sum(row["status"] == "PLANNED" for row in reopened.data["receiver_slots"]) == 160


@pytest.mark.parametrize(
    ("phase", "slot_key", "slot_filter", "expected_slots", "artifact_filter", "expected_artifacts"),
    (
        (
            "baseline-extract-videoseal", "baseline_slots",
            lambda row: row["case_id"] == "pilot_01" and row["method"] == "videoseal",
            9,
            lambda row: row["artifact_id"].startswith("pilot_01/videoseal/NATIVE_SOFT/"),
            9,
        ),
        (
            "receiver-sync", "receiver_slots",
            lambda row: row["case_id"] == "pilot_01",
            160,
            lambda row: row["artifact_id"].startswith("pilot_01/observation/"),
            36,
        ),
    ),
)
def test_timed_failure_retains_original_error_and_every_fixed_phase_row(
    tmp_path, phase, slot_key, slot_filter, expected_slots, artifact_filter, expected_artifacts,
):
    config = one_case_config()
    store = real_eval.RunStore(tmp_path / phase, config, create=True)
    denominator = {
        "artifacts": len(store.data["artifacts"]),
        "baseline_slots": len(store.data["baseline_slots"]),
        "receiver_slots": len(store.data["receiver_slots"]),
        "costs": len(store.data["costs"]),
    }

    def fail_inside_phase():
        raise RuntimeError("original phase root cause")

    with pytest.raises(RuntimeError, match="original phase root cause"):
        real_eval._run_timed(store, phase, "pilot_01", fail_inside_phase)

    phase_case = store.data["phases"][phase]["cases"]["pilot_01"]
    assert phase_case["status"] == "FAILED"
    assert phase_case["failures"] == ["RuntimeError: original phase root cause"]
    cost = next(row for row in store.data["costs"] if row["cost_id"] == f"pilot_01/{phase}")
    assert cost["status"] == "FAILED"
    assert cost["reason"] == "RuntimeError: original phase root cause"
    slots = [row for row in store.data[slot_key] if slot_filter(row)]
    assert len(slots) == expected_slots
    assert all(row["status"] == "FAILED" and row["reason"] == cost["reason"] for row in slots)
    artifacts = [row for row in store.data["artifacts"] if artifact_filter(row)]
    assert len(artifacts) == expected_artifacts
    assert all(row["status"] == "FAILED" and row["reason"] == cost["reason"] for row in artifacts)
    assert denominator == {
        "artifacts": len(store.data["artifacts"]),
        "baseline_slots": len(store.data["baseline_slots"]),
        "receiver_slots": len(store.data["receiver_slots"]),
        "costs": len(store.data["costs"]),
    }


def test_identity_and_revision_metadata_do_not_gate_or_rewrite_the_plan(tmp_path):
    config = one_case_config()
    config["models"]["wan"]["revision"] = "different-local-wan-revision"
    config["models"]["framewise"]["revision"] = "different-local-framewise-revision"
    for method in ("videoseal", "rivagan"):
        config["models"][method].pop("source_commit", None)
        config["models"][method].pop("checkpoint_sha256", None)
    config["models"]["videoseal"].pop("card_sha256", None)
    validated = real_eval.validate_real_config(config)
    assert len(real_eval.build_plan(validated)["receiver_slots"]) == 160

    output = tmp_path / "identity-observation"
    real_eval.RunStore(output, config, create=True)
    changed = copy.deepcopy(config)
    changed["models"]["wan"]["revision"] = "another-recorded-revision"
    reopened = real_eval.RunStore(output, changed)
    assert reopened.data["config_path_semantics"] == "LOADED_VALUES_USED_WITHOUT_DIGEST_ADMISSION_GATE"
    assert "config_identity_observation" not in reopened.data
    assert len(reopened.data["receiver_slots"]) == 160


def test_file_digest_metadata_is_not_required_for_real_file_use(monkeypatch, tmp_path):
    path = tmp_path / "usable.bin"
    path.write_bytes(b"usable bytes")
    resolved, actual = real_backends.require_local_file(
        path, expected_sha256="declared-different", label="fixture",
    )
    assert resolved == path.resolve()
    assert actual is None


def test_rgb8_reader_uses_shape_and_byte_count_without_digest_admission(monkeypatch, tmp_path):
    class Array:
        def reshape(self, shape):
            self.shape = tuple(shape)
            return self

        def copy(self):
            return self

    fake_numpy = types.SimpleNamespace(
        uint8="uint8",
        prod=lambda shape: __import__("math").prod(shape),
        frombuffer=lambda raw, dtype: Array(),
    )
    fake_torch = types.SimpleNamespace(from_numpy=lambda value: value)
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    from runtime.wan.rgb8_source import read_rgb8_source

    path = tmp_path / "editable.rgb8"
    path.write_bytes(b"\x00\x01\x02\x03\x04\x05")
    result = read_rgb8_source(
        path, expected_sha256="declared-different", shape=(1, 1, 2, 3),
    )
    assert result.shape == (1, 1, 2, 3)

    monkeypatch.setattr(
        real_backends, "file_sha256",
        lambda _path: (_ for _ in ()).throw(OSError("digest observer unavailable")),
    )
    resolved, actual = real_backends.require_local_file(
        path, expected_sha256="declared", label="fixture",
    )
    assert resolved == path.resolve()
    assert actual is None


def test_preflight_uses_existing_baseline_files_without_digest_admission(tmp_path):
    config = one_case_config()
    for method, package in (("videoseal", "videoseal/__init__.py"), ("rivagan", "rivagan/rivagan.py")):
        root = tmp_path / method
        target = root / package
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# fixture\n", encoding="utf-8")
        checkpoint = root / "checkpoint.bin"
        checkpoint.write_bytes(b"fixture checkpoint")
        config["models"][method]["source_root"] = str(root)
        config["models"][method]["checkpoint_path"] = str(checkpoint)
        config["models"][method]["checkpoint_sha256"] = "declared-different"
    card = tmp_path / "videoseal/card.yaml"
    card.write_text("args: {nbits: 256}\n", encoding="utf-8")
    config["models"]["videoseal"]["card_path"] = str(card)
    config["models"]["videoseal"]["card_sha256"] = "declared-different"

    checks = {row["name"]: row for row in real_eval.static_preflight(config)["checks"]}
    for name in ("videoseal:card_path", "videoseal:checkpoint", "rivagan:checkpoint"):
        assert checks[name]["status"] == "READY"
        assert "sha256_status" not in checks[name]


def test_real_cli_preflight_plan_and_init_work_from_no_git_copy(tmp_path):
    source = tmp_path / "source-copy"
    (source / "experiments").mkdir(parents=True)
    shutil.copy(ROOT / "experiments/__init__.py", source / "experiments/__init__.py")
    shutil.copytree(PACKAGE, source / "experiments/paper_results_v1")
    assert not (source / ".git").exists()
    config = source / "experiments/paper_results_v1/real_eval.proposal.json"

    plan_output = tmp_path / "plan"
    completed = subprocess.run(
        interpreter() + [
            "-m", "experiments.paper_results_v1.real_cli", "--config", str(config),
            "--output", str(plan_output), "--phase", "plan",
        ],
        cwd=source, text=True, capture_output=True, check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["fixed_denominator"] == {
        "artifacts": 690, "baseline_slots": 180, "cases": 10, "cost_rows": 110,
        "comparison_slots": 0, "quality_rows": 70, "receiver_bits": 51200,
        "receiver_slots": 1600,
    }

    preflight_output = tmp_path / "preflight"
    completed = subprocess.run(
        interpreter() + [
            "-m", "experiments.paper_results_v1.real_cli", "--config", str(config),
            "--output", str(preflight_output), "--phase", "preflight",
        ],
        cwd=source, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 2
    preflight = json.loads((preflight_output / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["status"] == "BLOCKED"
    assert preflight["loads_models"] is preflight["executes_media"] is False
    assert any(row["name"] == "videoseal:checkpoint" and row["status"] == "BLOCKED" for row in preflight["checks"])

    init_output = tmp_path / "state"
    subprocess.run(
        interpreter() + [
            "-m", "experiments.paper_results_v1.real_cli", "--config", str(config),
            "--output", str(init_output), "--phase", "init",
        ],
        cwd=source, text=True, capture_output=True, check=True,
    )
    state = json.loads((init_output / "run_state.json").read_text(encoding="utf-8"))
    assert len(state["artifacts"]) == 690 and len(state["receiver_slots"]) == 1600
    assert len(state["baseline_slots"]) == 180 and len(state["quality_rows"]) == 70
    assert len(state["comparison_slots"]) == 0
    assert all(row["status"] == "PLANNED" for row in state["receiver_slots"])


def test_off_trajectory_keeps_shared_predict_cfg_and_scheduler_without_payload(monkeypatch):
    class BoolReceipt:
        def all(self):
            return True

    fake_torch = types.SimpleNamespace(isfinite=lambda value: BoolReceipt())

    class Tensor:
        def __init__(self, value):
            self.value = float(value)

        def detach(self):
            return self

        def float(self):
            return self

        def clone(self):
            return Tensor(self.value)

        def cpu(self):
            return self

        def __add__(self, other):
            return Tensor(self.value + other.value)

        def __sub__(self, other):
            return Tensor(self.value - other.value)

        def __mul__(self, scalar):
            return Tensor(self.value * scalar)

        __rmul__ = __mul__

    class Scheduler:
        def __init__(self):
            self.step_index = None
            self.sigmas = [1.0] * 50

    velocities = []

    def native_step(scheduler, z, velocity, index, count, kind):
        assert kind == "native_step"
        count(kind, False)
        assert scheduler.step_index in (None, index)
        scheduler.step_index = index + 1
        velocities.append(velocity.value)
        count(kind, True)
        return z

    fake_runtime = types.SimpleNamespace(
        trajectory=types.SimpleNamespace(validate_scheduler=lambda scheduler: None, native_step=native_step),
        predict_branches=lambda pipe, z, scheduler, prompt, negative, dtype, index, count: (
            Tensor(2.0), Tensor(1.0),
        ),
    )
    import runtime.wan
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "runtime.wan.video_trajectory_conditional_joint_v1", fake_runtime)
    monkeypatch.setattr(runtime.wan, "video_trajectory_conditional_joint_v1", fake_runtime, raising=False)
    counts = []
    rows = []
    terminal, receipt = real_eval.run_off_trajectory(
        object(), Tensor(0.0), Scheduler(), object(), object(), "float",
        lambda name, done: counts.append((name, done)), rows.append,
    )
    assert terminal.value == 0.0
    assert velocities == [6.0] * 50
    assert len(rows) == 50 and all(row["payload_enabled"] is False for row in rows)
    assert counts == [item for _ in range(50) for item in (("native_step", False), ("native_step", True))]
    assert receipt["cfg"] == "unconditional + 5*(conditional-unconditional)"


def test_videoseal_loader_uses_explicit_local_card_and_checkpoint(monkeypatch, tmp_path):
    source = tmp_path / "videoseal-source"
    (source / "videoseal").mkdir(parents=True)
    (source / "videoseal/__init__.py").write_text("", encoding="utf-8")
    card = source / "videoseal_1.0.yaml"
    checkpoint = source / "local.pth"
    card.write_text("fixture-card", encoding="utf-8")
    checkpoint.write_bytes(b"fixture-local-checkpoint")
    calls = []

    class Model:
        def __init__(self):
            self.embedder = types.SimpleNamespace(msg_processor=types.SimpleNamespace(nbits=256))

        def eval(self):
            calls.append("eval")
            return self

        def to(self, device):
            calls.append(("to", device))
            return self

    fake_torch = types.SimpleNamespace(
        device=lambda value: value, float32="float32", uint8="uint8",
    )
    card_config = types.SimpleNamespace(args=types.SimpleNamespace(nbits=256))
    modules = {
        "torch": fake_torch,
        "omegaconf": types.SimpleNamespace(OmegaConf=types.SimpleNamespace(load=lambda path: card_config)),
        "videoseal.utils.cfg": types.SimpleNamespace(
            setup_model=lambda config, path: calls.append(("setup_model", config, path)) or Model(),
        ),
    }
    original_import = real_backends.importlib.import_module
    monkeypatch.setattr(
        real_backends.importlib, "import_module",
        lambda name: modules[name] if name in modules else original_import(name),
    )
    config = {
        "source_root": str(source), "source_commit": "unit-pin",
        "card_path": str(card), "card_sha256": real_backends.file_sha256(card),
        "checkpoint_path": str(checkpoint), "checkpoint_sha256": real_backends.file_sha256(checkpoint),
        "model_card_name": "videoseal_1.0_256bit", "device": "cpu",
        "native_message_length": 256, "lowres_attenuation": True,
        "detect_output_layout": "T,1+K", "inline_element_limit": 100,
    }
    adapter = real_backends.load_videoseal_adapter(config, native_output_store=lambda *args: None)
    assert calls[0][0] == "setup_model"
    assert calls[0][2] == str(checkpoint.resolve())
    assert adapter.native_message_length == 256
    assert adapter.backend_metadata["card_args_nbits"] == 256
    assert adapter.backend_metadata["model_msg_processor_nbits"] == 256
    assert adapter.inline_element_limit == 1


def test_baseline_extract_preserves_nine_logical_rows_with_eight_native_calls(monkeypatch, tmp_path):
    config = one_case_config()
    store = real_eval.RunStore(tmp_path / "run", config, create=True)

    class Media:
        def __init__(self, frames=181):
            self.frames = frames

        def __getitem__(self, mapping):
            return Media(len(mapping))

        def numpy(self):
            return self

    class Adapter:
        def __init__(self):
            self.calls = []

        def extract(self, media):
            self.calls.append(media.frames)
            return {"frame_soft_outputs": [[1.0] * 32 for _ in range(media.frames)]}

    adapter = Adapter()
    monkeypatch.setattr(real_eval.ArtifactFiles, "load_rgb8", staticmethod(lambda receipt: Media()))
    monkeypatch.setattr(real_eval, "_baseline_adapter", lambda config, method, files: adapter)
    monkeypatch.setattr(real_eval, "close_adapter", lambda value: None)
    real_eval.phase_baseline_extract(store, config, "pilot_01", "rivagan")
    slots = [row for row in store.data["baseline_slots"] if row["case_id"] == "pilot_01" and row["method"] == "rivagan"]
    assert len(slots) == 9 and all(row["status"] == "AVAILABLE" for row in slots)
    assert len(adapter.calls) == 8
    assert len({row["artifact_id"] for row in slots}) == 9


def test_receiver_sync_expands_declared_map_and_persists_every_observation(monkeypatch, tmp_path):
    config = one_case_config()
    store = real_eval.RunStore(tmp_path / "run", config, create=True)
    for arm in real_eval.MAIN_ARMS:
        store.artifact(real_eval._artifact_id("pilot_01", arm, "POST"), status="AVAILABLE", sha256=f"sha-{arm}")

    class Media:
        def __init__(self, frames=181):
            self.frames = frames

        def __len__(self):
            return self.frames

    class Model:
        def __init__(self):
            self.encodes = 0

        def encode(self, received):
            self.encodes += 1
            return received

        def score(self, latent, key, protocol):
            return {"key": key, "protocol": protocol}

        def close(self):
            pass

    model = Model()
    fake_runtime = types.SimpleNamespace(
        Inputs=types.SimpleNamespace(construct=lambda full, mapping: Media(len(mapping))),
    )
    fake_method = types.SimpleNamespace(
        estimates=lambda raw, protocol, frames, key: {"frames": frames},
        blind_operation=lambda logical, estimates: {"received_index_map": list(range(logical["frames"]))},
    )
    import runtime.wan
    import main.tube_state
    monkeypatch.setitem(sys.modules, "runtime.wan.video_trajectory_conditional_joint_v1", fake_runtime)
    monkeypatch.setattr(runtime.wan, "video_trajectory_conditional_joint_v1", fake_runtime, raising=False)
    monkeypatch.setitem(sys.modules, "main.tube_state.video_trajectory_conditional_joint_v1", fake_method)
    monkeypatch.setattr(main.tube_state, "video_trajectory_conditional_joint_v1", fake_method, raising=False)
    monkeypatch.setattr(real_eval, "load_local_framewise_backend", lambda config: model)
    monkeypatch.setattr(real_eval.ArtifactFiles, "load_rgb8", staticmethod(lambda receipt: Media()))
    real_eval.phase_receiver_sync(store, config, "pilot_01")
    receipt = store.data["records"]["pilot_01"]["blind_plan"]
    saved = json.loads(Path(receipt["path"]).read_text(encoding="utf-8"))
    assert len(saved["observations"]) == 36 and len(saved["slots"]) == 160
    assert saved["observations"]["off_native/global_00"]["frames"] == 181
    assert saved["observations"]["off_native/global_01"]["frames"] == 177
    assert model.encodes == 36


@pytest.mark.parametrize(
    ("frames", "expected_status", "expected_bit", "expected_ties"),
    [
        ([[0.0] * 32, [0.0] * 32], "UNEVALUABLE_ZERO_TIE", 1, 32),
        ([[1.0] * 32, [1.0] * 32], "EVALUATED", 1, 0),
        ([[1.0] * 32], "FAILED", None, None),
        ([[float("nan")] + [1.0] * 31] * 2, "FAILED", None, None),
    ],
)
def test_rivagan_sequence_rule_keeps_zero_and_failure_semantics(frames, expected_status, expected_bit, expected_ties):
    result = real_eval._rivagan_sequence_result(
        {"frame_soft_outputs": frames}, [1] * 32, expected_frames=2,
    )
    assert result["status"] == expected_status
    if expected_status in ("EVALUATED", "UNEVALUABLE_ZERO_TIE"):
        assert result["native_decoded_bits"] == [expected_bit] * 32
        assert result["tie_count"] == expected_ties
        if expected_status == "UNEVALUABLE_ZERO_TIE":
            assert result["decoded_bits"] is None
            assert result["exact_recovery"] is False
    else:
        assert "reason" in result


def test_effective32_zero_tie_preserves_each_native_zero_rule_but_is_not_exact():
    videoseal = real_eval._effective32_decision(
        [0.0] * 32, [0] * 32, rule="vs", zero_decodes_one=False, strict_zero_tie=True,
    )
    rivagan = real_eval._effective32_decision(
        [0.0] * 32, [1] * 32, rule="riva", zero_decodes_one=True, strict_zero_tie=True,
    )
    assert videoseal["native_decoded_bits"] == [0] * 32
    assert rivagan["native_decoded_bits"] == [1] * 32
    assert videoseal["status"] == rivagan["status"] == "UNEVALUABLE_ZERO_TIE"
    assert videoseal["exact_recovery"] is rivagan["exact_recovery"] is False


def test_adopted_native_tie_rules_keep_native_bits_while_strict_rules_remain_unevaluable():
    vs_native = real_eval._effective32_decision(
        [0.0] * 32, [0] * 32, rule=real_eval.VIDEOSEAL_NATIVE_TIE_RULE,
        zero_decodes_one=False, strict_zero_tie=False,
    )
    riva_native = real_eval._rivagan_sequence_result(
        {"frame_soft_outputs": [[0.0] * 32, [0.0] * 32]},
        [1] * 32,
        expected_frames=2,
        rule=real_eval.RIVAGAN_NATIVE_TIE_RULE,
    )
    assert vs_native["status"] == riva_native["status"] == "EVALUATED"
    assert vs_native["decoded_bits"] == [0] * 32
    assert riva_native["decoded_bits"] == [1] * 32
    assert vs_native["exact_recovery"] is riva_native["exact_recovery"] is True
    assert vs_native["tie_count"] == riva_native["tie_count"] == 32
    assert vs_native["tie_policy"] == riva_native["tie_policy"] == "NATIVE_BIT_RETAINED"

    strict = real_eval._rivagan_sequence_result(
        {"frame_soft_outputs": [[0.0] * 32, [0.0] * 32]},
        [1] * 32,
        expected_frames=2,
        rule=real_eval.RIVAGAN_STRICT_RULE,
    )
    assert strict["status"] == "UNEVALUABLE_ZERO_TIE"
    assert strict["native_decoded_bits"] == [1] * 32
    assert strict["decoded_bits"] is None


def test_fixed_comparison_left_join_keeps_missing_rows_and_source_denominator():
    config = real_eval.validate_real_config(adopted())
    slots = [
        row for row in real_eval.build_plan(config)["comparison_slots"]
        if row["case_id"] == "confirm_01" and row["baseline_method"] == "videoseal"
    ]
    global_01 = next(row for row in slots if row["view_id"] == "global_01")
    global_02 = next(row for row in slots if row["view_id"] == "global_02")
    receiver_rows = [
        {
            "slot_id": global_01["main_slot_id"], "status": "EVALUATED_TRUTH",
            "exact_recovery": True, "bit_errors": 0, "main_vote_tie_count": 2,
            "tie_policy": real_eval.MAIN_VOTE_TIE_POLICY,
            "tie_evidence_status": "AVAILABLE_ORIGINAL_READOUT_VOTES",
            "tie_evidence_reason": None,
        },
        {"slot_id": global_02["main_slot_id"], "status": "EVALUATED_TRUTH", "exact_recovery": False, "bit_errors": 2},
    ]
    baseline_rows = [
        {
            "slot_id": global_01["baseline_slot_id"], "status": "EVALUATED",
            "exact_recovery": False, "bit_errors": 1, "tie_count": 1,
            "tie_policy": "NATIVE_BIT_RETAINED",
        },
    ]
    rows = real_eval._evaluate_comparison_rows(slots, receiver_rows, baseline_rows)
    assert len(rows) == 9
    complete = next(row for row in rows if row["view_id"] == "global_01")
    assert complete["status"] == "EVALUATED_PAIR"
    assert complete["main_vote_tie_count"] == 2
    assert complete["main_tie_semantics"] == "FINAL_BIT_COUNTER_VOTE_EQUALITY"
    assert complete["baseline_tie_count"] == 1
    assert complete["baseline_tie_semantics"] == real_eval.BASELINE_TIE_SEMANTICS
    incomplete = next(row for row in rows if row["view_id"] == "global_02")
    assert incomplete["status"] == "UNEVALUABLE_PAIR"
    assert incomplete["baseline_status"] == "MISSING_ROW"
    assert incomplete["exact_success_difference_main_minus_baseline"] is None

    summary = real_eval._comparison_source_summaries(rows)[0]
    assert summary["fixed_nonfull_view_denominator"] == 8
    assert summary["evaluable_nonfull_pairs"] == 1
    assert summary["unavailable_nonfull_pairs"] == 7
    assert summary["main_unavailable_nonfull"] == 6
    assert summary["baseline_unavailable_nonfull"] == 7
    assert summary["main_exact_successes_fixed_nonfull"] == 1
    assert summary["main_observed_errors_nonfull"] == 1
    assert summary["baseline_observed_errors_nonfull"] == 1
    assert summary["observed_exact_success_difference_sum"] == 1
    assert summary["exact_success_difference_compatible_range"] == [-6, 7]
    assert summary["full_control_planned"] == 1
    cohort = real_eval._comparison_cohort_summaries([summary])[
        "CONFIRMATION_CANDIDATE|videoseal"
    ]
    assert cohort["exact_success_difference_compatible_range"] == [-6, 7]


@pytest.mark.parametrize(
    ("main_exact", "baseline_exact", "expected"),
    [
        (True, None, (0, 1)),
        (False, None, (-1, 0)),
        (None, True, (-1, 0)),
        (None, False, (0, 1)),
        (None, None, (-1, 1)),
        (True, False, (1, 1)),
    ],
)
def test_exact_success_difference_bounds_use_every_known_side(main_exact, baseline_exact, expected):
    assert real_eval._exact_success_difference_bounds(main_exact, baseline_exact) == expected


@pytest.mark.parametrize("frames", [180, 182])
def test_videoseal_reducer_rejects_missing_or_extra_frames(monkeypatch, tmp_path, frames):
    sidecar = tmp_path / "preds.npz"
    sidecar.write_bytes(b"synthetic-sidecar")

    class Preds:
        ndim = 2
        shape = (frames, 257)

    class Values:
        files = ["preds"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, key):
            assert key == "preds"
            return Preds()

    fake_numpy = types.SimpleNamespace(
        load=lambda path, allow_pickle: Values(),
        isfinite=lambda value: types.SimpleNamespace(all=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setattr(real_eval, "file_sha256", lambda path: "fixture-sha")
    result = real_eval._videoseal_32_result(
        {
            "storage": "LOSSLESS_SIDECAR",
            "lossless_native_output": {"uri": str(sidecar), "sha256": "fixture-sha"},
        },
        [0] * 32,
        native_length=256,
        expected_frames=181,
        rule=real_eval.VIDEOSEAL_STRICT_RULE,
    )
    assert result["status"] == "FAILED"
    assert "shape" in result["reason"]


def test_expensive_phase_is_single_attempt_but_evaluate_can_regenerate(tmp_path):
    config = one_case_config()
    store = real_eval.RunStore(tmp_path / "run", config, create=True)
    store.phase_start("generate", "pilot_01")
    store.phase_failure("generate", RuntimeError("synthetic first attempt"), "pilot_01")
    with pytest.raises(real_eval.RealEvalConfigError, match="already attempted"):
        store.phase_start("generate", "pilot_01")
    assert store.data["phases"]["generate"]["cases"]["pilot_01"]["failures"] == [
        "RuntimeError: synthetic first attempt"
    ]

    store.phase_start("evaluate")
    store.phase_failure("evaluate", RuntimeError("synthetic report failure"))
    store.phase_start("evaluate")
    assert store.data["phases"]["evaluate"]["status"] == "RUNNING"
    assert store.data["phases"]["evaluate"]["failures"] == [
        "RuntimeError: synthetic report failure"
    ]


def test_multicase_phase_status_is_derived_after_each_case_transition(tmp_path):
    config = proposal()
    config["study_id"] = "synthetic_multicase_phase_state_fixture"
    config["cases"] = copy.deepcopy(config["cases"][:2])
    first, second = [case["case_id"] for case in config["cases"]]

    failed_then_complete = real_eval.RunStore(tmp_path / "failed-then-complete", config, create=True)
    assert failed_then_complete.data["phases"]["generate"]["status"] == "PLANNED"
    failed_then_complete.phase_start("generate", first)
    failed_then_complete.phase_failure("generate", RuntimeError("first failed"), first)
    assert failed_then_complete.data["phases"]["generate"]["status"] == "FAILED"
    failed_then_complete.phase_start("generate", second)
    assert failed_then_complete.data["phases"]["generate"]["status"] == "RUNNING"
    failed_then_complete.phase_finish("generate", second)
    assert failed_then_complete.data["phases"]["generate"]["status"] == "FAILED"
    assert failed_then_complete.data["phases"]["generate"]["cases"][first]["failures"] == [
        "RuntimeError: first failed"
    ]

    complete_then_failed = real_eval.RunStore(tmp_path / "complete-then-failed", config, create=True)
    complete_then_failed.phase_start("generate", first)
    complete_then_failed.phase_finish("generate", first)
    assert complete_then_failed.data["phases"]["generate"]["status"] == "PARTIAL"
    assert complete_then_failed.data["phases"]["generate"]["cases"][second]["status"] == "PLANNED"
    complete_then_failed.phase_start("generate", second)
    assert complete_then_failed.data["phases"]["generate"]["status"] == "RUNNING"
    complete_then_failed.phase_failure("generate", RuntimeError("second failed"), second)
    assert complete_then_failed.data["phases"]["generate"]["status"] == "FAILED"


def test_evaluate_isolates_bad_receiver_baseline_and_sidecar_receipts(tmp_path):
    config = one_case_config()
    config["evaluation_rules"]["videoseal_32"]["status"] = "ADOPTED_FOR_EXECUTION"
    config["evaluation_rules"]["rivagan_sequence"]["status"] = "ADOPTED_FOR_EXECUTION"
    store = real_eval.RunStore(tmp_path / "run", config, create=True)
    case_records = store.data["records"].setdefault("pilot_01", {})
    first_receiver = store.data["receiver_slots"][0]["slot_id"]
    second_receiver = store.data["receiver_slots"][1]["slot_id"]
    third_receiver = store.data["receiver_slots"][2]["slot_id"]
    fourth_receiver = store.data["receiver_slots"][3]["slot_id"]
    votes = [
        {"ones": 2, "zeros": 2, "count": 4},
        *[{"ones": 3, "zeros": 1, "count": 4} for _ in range(31)],
    ]
    case_records["blind_reads"] = real_eval._json_dump(
        tmp_path / "bad_receiver.json",
        {
            "slots": {
                first_receiver: {
                    "status": "READ", "decoded_bits": config["payload_bits"],
                    "detail": {"original_readout": {"votes": votes}},
                },
                second_receiver: {"status": "READ", "decoded_bits": config["payload_bits"]},
                third_receiver: {
                    "status": "READ", "decoded_bits": config["payload_bits"],
                    "detail": {"original_readout": {"votes": votes[:-1]}},
                },
                fourth_receiver: [],
            },
            "physical_encodes": [],
        },
    )

    vs_slot = next(row for row in store.data["baseline_slots"] if row["method"] == "videoseal")
    vs_slot["record"] = real_eval._json_dump(
        tmp_path / "bad_sidecar_record.json",
        {
            "storage": "LOSSLESS_SIDECAR",
            "lossless_native_output": {
                "uri": str(tmp_path / "missing.npz"), "sha256": "missing",
            },
        },
    )
    riva_slot = next(row for row in store.data["baseline_slots"] if row["method"] == "rivagan")
    riva_slot["record"] = real_eval._json_dump(tmp_path / "bad_baseline.json", [])
    store.phase_start("generate", "pilot_01")
    store.phase_failure("generate", RuntimeError("retained before report"), "pilot_01")
    store.save()

    report_value = real_eval.phase_evaluate(store, config)
    assert report_value["status"] == "COMPLETE_WITH_RETAINED_ISSUES"
    assert len(report_value["receiver_rows"]) == 160
    assert report_value["receiver_rows"][0]["status"] == "EVALUATED_TRUTH"
    assert report_value["receiver_rows"][0]["main_vote_tie_count"] == 1
    assert report_value["receiver_rows"][0]["tie_evidence_status"] == "AVAILABLE_ORIGINAL_READOUT_VOTES"
    assert report_value["receiver_rows"][1]["status"] == "EVALUATED_TRUTH"
    assert report_value["receiver_rows"][1]["main_vote_tie_count"] is None
    assert report_value["receiver_rows"][1]["tie_evidence_status"] == "UNAVAILABLE_MISSING"
    assert report_value["receiver_rows"][2]["status"] == "EVALUATED_TRUTH"
    assert report_value["receiver_rows"][2]["exact_recovery"] is True
    assert report_value["receiver_rows"][2]["main_vote_tie_count"] is None
    assert report_value["receiver_rows"][2]["tie_evidence_status"] == "UNAVAILABLE_INVALID"
    assert report_value["receiver_rows"][3]["status"] == "FAILED"
    assert sum(row["status"] == "FAILED" for row in report_value["receiver_rows"]) == 157
    assert len(report_value["baseline_rows"]) == 18
    assert all(row["status"] == "FAILED" for row in report_value["baseline_rows"])
    assert report_value["receiver_resource_counts"]["actual_unique_physical_encodes_by_case"]["pilot_01"]["status"] == "FAILED"
    assert report_value["phase_records"]["generate"]["cases"]["pilot_01"]["failures"] == [
        "RuntimeError: retained before report"
    ]
    assert (store.output / "evaluation_report.json").is_file()
    assert (store.output / "receiver_rows.csv").is_file()
    assert (store.output / "baseline_rows.csv").is_file()


def test_legacy_strict_run_state_without_comparison_manifest_reopens_without_reinterpretation(tmp_path):
    config = one_case_config()
    output = tmp_path / "legacy-strict-run"
    store = real_eval.RunStore(output, config, create=True)
    assert store.data["comparison_slots"] == []
    store.data.pop("comparison_slots")
    store.save()

    reopened = real_eval.RunStore(output, config)
    report_value = real_eval.phase_evaluate(reopened, config)
    assert report_value["fixed_denominator"]["planned_comparison_rows"] == 0
    assert report_value["comparison_rows"] == []
    assert report_value["comparison_source_summaries"] == []
    assert report_value["comparison_cohort_summaries"] == {}
    assert {row["status"] for row in report_value["baseline_rows"]} == {"PENDING_RULE_NOT_ADOPTED"}
