"""Static and CPU-only checks for the self-contained two-pilot Colab handoff."""
from __future__ import annotations

import ast
import base64
import copy
import hashlib
import io
import json
import os
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import pytest

from experiments.paper_results_v1.colab_orchestration import (
    build_scope_summary,
    execute_fixed_sequence,
    prepare_baseline_environment,
)
from experiments.paper_results_v1 import real_eval, report as report_io


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/paper_results_v1_two_pilot_colab.ipynb"
pytestmark = pytest.mark.unit


def _notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _assignment(source, name):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"assignment {name} not found")


def _interpreter():
    if os.access(sys.executable, os.X_OK):
        return [sys.executable]
    return ["/lib64/ld-linux-x86-64.so.2", sys.executable]


def test_notebook_is_clean_fixed_run_all_with_exact_mount_cell():
    notebook = _notebook()
    assert notebook["nbformat"] == 4
    assert "".join(notebook["cells"][0]["source"]) == (
        "from google.colab import drive\ndrive.mount('/content/drive')\n"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))
    joined = "\n".join(_code(notebook))
    assert "force_remount" not in joined
    assert "RUN = False" not in joined and "MODE =" not in joined
    setup = _code(notebook)[1]
    assert _assignment(setup, "PILOT_CASES") == ("pilot_01", "pilot_02")
    assert _assignment(setup, "CONFIRMATION_CASES") == tuple(f"confirm_{index:02d}" for index in range(1, 9))
    assert _assignment(setup, "FIXED_FULL_DENOMINATOR") == {
        "cases": 10, "artifacts": 690, "receiver_slots": 1600,
        "baseline_slots": 180, "comparison_slots": 180,
        "quality_rows": 70, "cost_rows": 110,
    }


def test_notebook_embeds_verified_no_git_source_closure_and_full_fixed_plan(tmp_path):
    notebook = _notebook()
    setup = _code(notebook)[1]
    raw = base64.b64decode(_assignment(setup, "PORTABLE_B64"))
    assert hashlib.sha256(raw).hexdigest() == _assignment(setup, "PORTABLE_ZIP_SHA256")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        names = archive.namelist()
        assert not any(".git" in Path(name).parts for name in names)
        assert {
            "experiments/__init__.py",
            "experiments/paper_results_v1/real_cli.py",
            "experiments/paper_results_v1/real_eval.py",
            "experiments/paper_results_v1/real_eval.adopted.json",
            "main/tube_state/video_trajectory_conditional_joint_v1.py",
            "runtime/wan/video_trajectory_conditional_joint_v1.py",
            "portable_manifest.json",
        }.issubset(names)
        for name in names:
            if name.endswith(".py"):
                ast.parse(archive.read(name).decode("utf-8"), filename=name)
        archive.extractall(tmp_path)
    manifest = json.loads((tmp_path / "portable_manifest.json").read_text(encoding="utf-8"))
    for name, receipt in manifest["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == receipt["sha256"]
    assert not (tmp_path / ".git").exists()
    config = tmp_path / "experiments/paper_results_v1/real_eval.adopted.json"
    output = tmp_path / "plan"
    completed = subprocess.run(
        _interpreter() + [
            "-m", "experiments.paper_results_v1.real_cli", "--config", str(config),
            "--output", str(output), "--phase", "plan",
        ],
        cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(tmp_path)},
        text=True, capture_output=True, check=True,
    )
    summary = json.loads(completed.stdout)
    assert summary["fixed_denominator"] == {
        "cases": 10, "artifacts": 690, "receiver_slots": 1600,
        "receiver_bits": 51200, "baseline_slots": 180,
        "comparison_slots": 180, "quality_rows": 70, "cost_rows": 110,
    }
    config_data = json.loads(config.read_text(encoding="utf-8"))
    assert [case["case_id"] for case in config_data["cases"]] == [
        "pilot_01", "pilot_02", *[f"confirm_{index:02d}" for index in range(1, 9)],
    ]


def test_notebook_freezes_identity_downloads_stage_order_and_failure_retention():
    notebook = _notebook()
    code = _code(notebook)
    joined = "\n".join(code)
    assert "870ca7fb33578b90f14c602016b6c2788096226e" in joined
    assert "efffa72a4ca46d4d5051f6970c96424c2cdab441" in joined
    assert "4d9928350509f808b6b57d48d2f958aef811d332" in joined
    assert "https://dl.fbaipublicfiles.com/videoseal/y_256b_img.pth" in joined
    assert "0fad780a534b6463e45facd96134c9f345acfa5b" in joined
    assert "31f26fdeee1355a5c34592e401dd41e45d25a493" in joined
    assert "torch==1.0.1.post2" not in joined
    assert "pip\", \"install\", \"-r" not in joined
    assert '"--no-deps"' not in joined
    assert '"--constraint", str(BASELINE_CORE_CONSTRAINTS)' in joined
    assert "antlr4-python3-runtime==4.9.*" in joined
    assert "PyYAML>=5.1.0" in joined
    assert "from videoseal.utils.cfg import setup_model" in joined
    assert "_install_rivagan_pickle_classes" in joined
    assert "IMPORT_READY_NO_MODEL_LOADED" in joined
    assert "IMPORT_READY_NO_CHECKPOINT_LOADED" in joined
    assert "MAIN_PIP_CHECK_DIAGNOSTIC\", check=False" in joined

    execute = next(source for source in code if "phase_interpreters" in source)
    assert "execute_fixed_sequence(" in execute
    assert "persist_execution" in execute
    assert "check=False" in execute
    assert "FINAL_FIXED_DENOMINATOR_EVALUATE" in execute
    phases = _assignment(code[1], "PHASE_ORDER")
    assert phases == (
        "generate", "decode", "framewise", "baseline-embed-videoseal",
        "baseline-embed-rivagan", "codec", "quality", "baseline-extract-videoseal",
        "baseline-extract-rivagan", "receiver-sync", "receiver-read",
    )
    assert "confirmation_status\": \"NOT_EXECUTED_BY_NOTEBOOK" in joined
    assert "historical_evidence_is_not_this_run\": True" in joined


def test_launch_exception_is_saved_immediately_then_later_attempt_and_evaluate_run():
    calls = []
    snapshots = []

    def invoke(case_id, phase):
        calls.append((case_id, phase))
        if phase == "generate":
            raise OSError("synthetic launch failure")
        return 0

    progress = execute_fixed_sequence(
        pilot_ids=("pilot_01",), phases=("generate", "decode"),
        preflight=lambda: 0, invoke=invoke,
        evaluate=lambda: calls.append((None, "evaluate")) or 0,
        persist=lambda value: snapshots.append(copy.deepcopy(value)),
    )
    assert calls == [("pilot_01", "generate"), ("pilot_01", "decode"), (None, "evaluate")]
    assert progress["phase_attempts"] == [
        {
            "case_id": "pilot_01", "phase": "generate", "returncode": None,
            "status": "FAILED_RETAINED", "reason": "OSError: synthetic launch failure",
        },
        {"case_id": "pilot_01", "phase": "decode", "returncode": 0, "status": "COMPLETE"},
    ]
    assert any(len(row["phase_attempts"]) == 1 for row in snapshots)
    assert snapshots[-1]["evaluate"] == {"status": "COMPLETE", "returncode": 0}


def test_keyboard_interrupt_stops_expensive_attempts_but_persists_evaluate_cleanup():
    calls = []
    snapshots = []

    def invoke(case_id, phase):
        calls.append((case_id, phase))
        raise KeyboardInterrupt("synthetic user stop")

    with pytest.raises(KeyboardInterrupt, match="synthetic user stop"):
        execute_fixed_sequence(
            pilot_ids=("pilot_01", "pilot_02"), phases=("generate", "decode"),
            preflight=lambda: 0, invoke=invoke,
            evaluate=lambda: calls.append((None, "evaluate")) or 0,
            persist=lambda value: snapshots.append(copy.deepcopy(value)),
        )
    assert calls == [("pilot_01", "generate"), (None, "evaluate")]
    assert snapshots[-1]["interrupted"] is True
    assert snapshots[-1]["phase_attempts"][0]["status"] == "INTERRUPTED"
    assert snapshots[-1]["evaluate"]["status"] == "COMPLETE"


def test_handoff_status_uses_final_report_and_separates_confirmation_projection():
    state = {
        "receiver_rows": [],
        "receiver_slots": [
            {"case_id": "pilot_01", "status": "PLANNED"},
            {"case_id": "confirm_01", "status": "PLANNED"},
        ],
        "baseline_rows": [],
        "baseline_slots": [
            {"case_id": "pilot_01", "status": "PLANNED"},
            {"case_id": "confirm_01", "status": "PLANNED"},
        ],
        "comparison_rows": [],
        "comparison_slots": [
            {"case_id": "pilot_01", "status": "PLANNED"},
            {"case_id": "confirm_01", "status": "PLANNED"},
        ],
        "quality_rows": [
            {"case_id": "pilot_01", "status": "PLANNED"},
            {"case_id": "confirm_01", "status": "PLANNED"},
        ],
    }
    report = {
        "receiver_rows": [
            {"case_id": "pilot_01", "status": "EVALUATED_TRUTH"},
            {"case_id": "confirm_01", "status": "FAILED"},
        ],
        "baseline_rows": [
            {"case_id": "pilot_01", "status": "EVALUATED"},
            {"case_id": "confirm_01", "status": "FAILED"},
        ],
        "comparison_rows": [
            {"case_id": "pilot_01", "status": "EVALUATED_PAIR"},
            {"case_id": "confirm_01", "status": "UNEVALUABLE_PAIR"},
        ],
        "quality_rows": [
            {"case_id": "pilot_01", "status": "EVALUATED"},
            {"case_id": "confirm_01", "status": "PLANNED"},
        ],
        "comparison_source_summaries": [
            {"case_id": "pilot_01", "fixed_nonfull_view_denominator": 8},
            {"case_id": "confirm_01", "fixed_nonfull_view_denominator": 8},
        ],
        "comparison_cohort_summaries": {
            "PILOT_EXCLUDED_FROM_CONFIRMATION": {"sources": 1},
            "CONFIRMATION_CANDIDATE": {"sources": 1},
        },
    }
    summary = build_scope_summary(
        state, report, pilot_ids=("pilot_01",), confirmation_ids=("confirm_01",),
    )
    assert summary["pilot_status_source"] == "evaluation_report_final_rows"
    assert summary["pilot_status_counts"]["receiver"] == {"EVALUATED_TRUTH": 1}
    assert summary["pilot_status_counts"]["comparison"] == {"EVALUATED_PAIR": 1}
    assert summary["pilot_comparison_source_summaries"] == [
        {"case_id": "pilot_01", "fixed_nonfull_view_denominator": 8}
    ]
    assert summary["confirmation"]["immutable_plan_counts"] == {
        "receiver": 1, "baseline": 1, "comparison": 1, "quality": 1,
    }
    assert summary["confirmation"]["report_projection_status_counts"]["comparison"] == {
        "UNEVALUABLE_PAIR": 1
    }
    fallback = build_scope_summary(
        state, None, pilot_ids=("pilot_01",), confirmation_ids=("confirm_01",),
    )
    assert fallback["pilot_status_source"] == "run_state_fallback_not_evaluated"
    assert fallback["pilot_status_counts"]["receiver"] == {"PLANNED": 1}
    assert fallback["confirmation"]["report_projection_status_counts"] is None


def test_actual_adopted_empty_evidence_report_keeps_both_pilot_method_cohorts(tmp_path):
    config = report_io.read_json(
        ROOT / "experiments/paper_results_v1/real_eval.adopted.json"
    )
    store = real_eval.RunStore(tmp_path / "empty-evidence-run", config, create=True)
    evaluation = real_eval.phase_evaluate(store, config)
    summary = build_scope_summary(
        store.data, evaluation,
        pilot_ids=("pilot_01", "pilot_02"),
        confirmation_ids=tuple(f"confirm_{index:02d}" for index in range(1, 9)),
    )
    assert len(summary["pilot_comparison_source_summaries"]) == 4
    assert {
        row["source_summary_id"] for row in summary["pilot_comparison_source_summaries"]
    } == {
        "pilot_01/videoseal", "pilot_01/rivagan",
        "pilot_02/videoseal", "pilot_02/rivagan",
    }
    assert set(summary["pilot_comparison_cohort_summaries"]) == {
        "PILOT_EXCLUDED_FROM_CONFIRMATION|videoseal",
        "PILOT_EXCLUDED_FROM_CONFIRMATION|rivagan",
    }
    assert all(
        row["cohort"] == "PILOT_EXCLUDED_FROM_CONFIRMATION"
        and row["fixed_source_denominator"] == 2
        for row in summary["pilot_comparison_cohort_summaries"].values()
    )
    assert summary["pilot_status_counts"]["comparison"] == {
        "UNEVALUABLE_PAIR": 36
    }
    assert summary["confirmation"]["immutable_plan_counts"] == {
        "receiver": 1280, "baseline": 144, "comparison": 144, "quality": 56,
    }


def test_baseline_environment_receipts_separate_resolution_probe_and_model_status():
    calls = []

    success = prepare_baseline_environment(
        install=lambda: calls.append("install"),
        probe=lambda: calls.append("probe"),
    )
    assert calls == ["install", "probe"]
    assert success == {
        "dependency_install_status": "COMPLETE",
        "entry_import_probe_status": "IMPORT_READY_NO_MODEL_OR_WEIGHT_LOADED",
        "model_compatibility_status": "NOT_VALIDATED_REQUIRES_REAL_EMBED_EXTRACT",
    }

    calls.clear()
    install_failure = prepare_baseline_environment(
        install=lambda: (_ for _ in ()).throw(OSError("resolver unavailable")),
        probe=lambda: calls.append("probe must not run"),
    )
    assert calls == []
    assert install_failure["dependency_install_status"] == "FAILED"
    assert install_failure["entry_import_probe_status"] == "BLOCKED_DEPENDENCY_INSTALL_FAILED"
    assert install_failure["dependency_install_reason"] == "OSError: resolver unavailable"

    probe_failure = prepare_baseline_environment(
        install=lambda: None,
        probe=lambda: (_ for _ in ()).throw(ImportError("entry unavailable")),
    )
    assert probe_failure["dependency_install_status"] == "COMPLETE"
    assert probe_failure["entry_import_probe_status"] == "FAILED"
    assert probe_failure["entry_import_probe_reason"] == "ImportError: entry unavailable"
    assert probe_failure["model_compatibility_status"] == "NOT_VALIDATED_REQUIRES_REAL_EMBED_EXTRACT"


def test_generated_code_cells_run_in_order_from_fresh_isolated_kernel_with_boundary_stubs(tmp_path):
    isolated_python = Path("/usr/bin/python3")
    assert isolated_python.is_file()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    script = isolated / "run_notebook_boundary_stub.py"
    identity_injection = textwrap.dedent('''\
        def verified_checkout(label, url, commit, destination):
            destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
            if label.startswith("VIDEOSEAL"):
                (destination / "videoseal/cards").mkdir(parents=True, exist_ok=True)
                (destination / "videoseal/__init__.py").write_text("")
                (destination / "videoseal/cards/videoseal_1.0.yaml").write_text("args:\\n  nbits: 256\\n")
            else:
                (destination / "rivagan").mkdir(parents=True, exist_ok=True)
                (destination / "rivagan/__init__.py").write_text("")
                (destination / "rivagan/rivagan.py").write_text("class RivaGAN: pass\\n")
            return destination
        def cached_download(label, url, destination):
            destination = Path(destination); destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((label + "-stub").encode())
            return destination, sha256_file(destination)
    ''')
    script.write_text(textwrap.dedent(f'''\
        import json, os, subprocess, sys, types
        from pathlib import Path

        notebook_path = Path({str(NOTEBOOK)!r})
        sandbox = Path({str(isolated)!r})
        content_root = sandbox / "content"
        drive_root = sandbox / "drive" / "MyDrive" / "Video-WM"
        content_root.mkdir(parents=True, exist_ok=True)
        drive_root.mkdir(parents=True, exist_ok=True)

        google = types.ModuleType("google")
        colab = types.ModuleType("google.colab")
        drive = types.ModuleType("google.colab.drive")
        drive.mount = lambda path: None
        colab.drive = drive
        colab.userdata = types.SimpleNamespace(get=lambda key: None)
        google.colab = colab
        sys.modules.update({{
            "google": google, "google.colab": colab, "google.colab.drive": drive,
        }})
        huggingface_hub = types.ModuleType("huggingface_hub")
        snapshot_calls = []
        def snapshot_download(*, repo_id, revision, local_dir, allow_patterns):
            snapshot_calls.append((repo_id, revision, tuple(allow_patterns)))
            root = Path(local_dir); root.mkdir(parents=True, exist_ok=True)
            if repo_id.startswith("Wan-AI/"):
                for name in (
                    "model_index.json", "scheduler/scheduler_config.json",
                    "tokenizer/tokenizer_config.json", "text_encoder/config.json",
                    "transformer/config.json", "vae/config.json",
                    "transformer/stub.safetensors",
                ):
                    path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("{{}}")
            else:
                (root / "config.json").write_text("{{}}")
                (root / "diffusion_pytorch_model.safetensors").write_bytes(b"stub")
            return str(root)
        huggingface_hub.snapshot_download = snapshot_download
        sys.modules["huggingface_hub"] = huggingface_hub

        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        def adapted(source):
            return source.replace(
                "/content/drive/MyDrive/Video-WM", str(drive_root)
            ).replace("/content", str(content_root))

        namespace = {{"__name__": "__main__"}}
        os.chdir(sandbox)
        exec(compile(adapted(cells[0]), "cell-0", "exec"), namespace)
        exec(compile(adapted(cells[1]), "cell-1", "exec"), namespace)
        portable_root = namespace["PORTABLE_ROOT"]
        assert sys.path[0] == str(portable_root)
        assert Path(sys.path[0]).is_dir()

        identity = adapted(cells[2])
        injection = {identity_injection!r}
        identity = identity.replace("baseline_setup = {{}}", injection + "\\nbaseline_setup = {{}}", 1)
        exec(compile(identity, "cell-2", "exec"), namespace)
        assert (namespace["RUN_OUTPUT"] / "run_state.json").is_file()

        original_logged = namespace["logged"]
        original_check_output = subprocess.check_output
        environment_commands = []
        def environment_logged(command, stage, **kwargs):
            environment_commands.append((stage, list(command)))
            return 0
        def check_output_stub(command, *args, **kwargs):
            if "pip" in command and "freeze" in command:
                return "" if kwargs.get("text") else b""
            return original_check_output(command, *args, **kwargs)
        namespace["logged"] = environment_logged
        subprocess.check_output = check_output_stub
        try:
            exec(compile(adapted(cells[3]), "cell-3", "exec"), namespace)
        finally:
            subprocess.check_output = original_check_output
            namespace["logged"] = original_logged
        assert len(snapshot_calls) == 2
        assert all(
            row["dependency_install_status"] == "COMPLETE"
            and row["entry_import_probe_status"] == "IMPORT_READY_NO_MODEL_OR_WEIGHT_LOADED"
            for row in namespace["baseline_setup"].values()
        )
        assert any("--constraint" in command for _stage, command in environment_commands)
        assert not any("--no-deps" in command for _stage, command in environment_commands)

        phase_commands = []
        def phase_logged(command, stage, **kwargs):
            phase = command[command.index("--phase") + 1] if "--phase" in command else None
            if phase in ("preflight", "evaluate"):
                return original_logged(command, stage, **kwargs)
            phase_commands.append((stage, list(command)))
            return 1
        namespace["logged"] = phase_logged
        exec(compile(adapted(cells[4]), "cell-4", "exec"), namespace)
        namespace["logged"] = original_logged
        exec(compile(adapted(cells[5]), "cell-5", "exec"), namespace)

        progress = json.loads((namespace["OUTPUT_ROOT"] / "pilot_phase_attempts.json").read_text())
        assert len(progress["phase_attempts"]) == 22
        assert {{row["case_id"] for row in progress["phase_attempts"]}} == {{"pilot_01", "pilot_02"}}
        assert progress["evaluate"]["status"] == "COMPLETE"
        assert len(phase_commands) == 22
        handoff = json.loads((namespace["OUTPUT_ROOT"] / "handoff_summary.json").read_text())
        assert handoff["pilot_status_source"] == "evaluation_report_final_rows"
        assert handoff["execution_scope"]["attempted"] == ["pilot_01", "pilot_02"]
        assert handoff["confirmation"]["execution_scope_status"] == "NOT_EXECUTED_BY_NOTEBOOK"
        assert (namespace["RUN_OUTPUT"] / "evaluation_report.json").is_file()
        assert "torch" not in sys.modules
        print(json.dumps({{"status": "BOUNDARY_STUB_COMPLETE", "phases": len(phase_commands)}}))
    '''), encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    completed = subprocess.run(
        [str(isolated_python), "-I", str(script)], cwd=isolated,
        env=environment, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert '"status": "BOUNDARY_STUB_COMPLETE"' in completed.stdout
