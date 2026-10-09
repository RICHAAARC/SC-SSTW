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
    ensure_baseline_interpreter,
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


def test_portable_identity_differences_are_recorded_but_bad_archives_still_fail(tmp_path):
    setup = _code(_notebook())[1]
    original = base64.b64decode(_assignment(setup, "PORTABLE_B64"))

    def variant(*, drop_manifest=False, change_file=False, unsafe=False):
        stream = io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(stream, "w") as target:
            for info in source.infolist():
                if drop_manifest and info.filename == "portable_manifest.json":
                    continue
                payload = source.read(info.filename)
                if change_file and info.filename == "experiments/__init__.py":
                    payload += b"# recorded difference\n"
                target.writestr(info, payload)
            if unsafe:
                target.writestr("../escape.txt", b"unsafe")
        return stream.getvalue()

    def extraction_namespace(root, raw):
        source = setup.split("\ntry:\n    SOURCE_MANIFEST = extract_portable_source()", 1)[0]
        content = root / "content"
        drive = root / "drive" / "MyDrive" / "Video-WM"
        source = source.replace("/content/drive/MyDrive/Video-WM", str(drive)).replace("/content", str(content))
        namespace = {"__name__": "__main__"}
        exec(compile(source, "portable-setup-prefix", "exec"), namespace)
        namespace["PORTABLE_B64"] = base64.b64encode(raw).decode()
        namespace["PORTABLE_ZIP_SHA256"] = "declared-different"
        return namespace

    changed = extraction_namespace(tmp_path / "changed", variant(change_file=True))
    changed["extract_portable_source"]()
    changed_receipt = json.loads((changed["OUTPUT_ROOT"] / "portable_source_receipt.json").read_text())
    assert changed_receipt["zip_sha256_status"] == "RECORDED_DIFFERENCE"
    assert changed_receipt["file_observations"]["experiments/__init__.py"]["status"] == "RECORDED_DIFFERENCE"
    assert changed_receipt["identity_differences_are_blocking"] is False

    absent = extraction_namespace(tmp_path / "absent", variant(drop_manifest=True))
    assert absent["extract_portable_source"]() == {"files": {}}
    absent_receipt = json.loads((absent["OUTPUT_ROOT"] / "portable_source_receipt.json").read_text())
    assert absent_receipt["manifest_status"] == "ABSENT_OPTIONAL"

    corrupt = extraction_namespace(tmp_path / "corrupt", b"not a zip archive")
    with pytest.raises(zipfile.BadZipFile):
        corrupt["extract_portable_source"]()

    unsafe = extraction_namespace(tmp_path / "unsafe", variant(unsafe=True))
    with pytest.raises(RuntimeError, match="unsafe portable archive member"):
        unsafe["extract_portable_source"]()


def test_cached_dirty_checkout_is_preserved_and_recorded(tmp_path):
    identity_source = _code(_notebook())[2]
    tree = ast.parse(identity_source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "prepared_checkout"
    )
    calls = []
    namespace = {
        "Path": Path,
        "subprocess": subprocess,
        "logged": lambda *args, **kwargs: calls.append((args, kwargs)),
        "source_checkout_receipts": {},
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "prepared-checkout", "exec"), namespace)

    checkout = tmp_path / "cached-source"
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    (checkout / "source.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "source.py"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-m", "fixture"],
        check=True, capture_output=True,
    )
    (checkout / "source.py").write_text("value = 2\n", encoding="utf-8")
    namespace["prepared_checkout"]("SOURCE", "unused://remote", "different-commit", checkout)

    receipt = namespace["source_checkout_receipts"]["SOURCE"]
    assert calls == []
    assert receipt["created_now"] is False
    assert receipt["status"] == "RECORDED_DIFFERENCE"
    assert receipt["dirty_paths"] == [" M source.py"]


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
    assert '"--constraint", str(BASELINE_REPAIR_CONSTRAINTS)' in joined
    assert 'PYTHON, "-m", "pip", "--python", str(vpython), "install"' in joined
    assert 'str(vpython), "-m", "pip", "install"' not in joined
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
    assert calls == ["probe"]
    assert success == {
        "dependency_install_status": "NOT_NEEDED_IMPORT_READY",
        "entry_import_probe_status": "IMPORT_READY_NO_MODEL_OR_WEIGHT_LOADED",
        "model_compatibility_status": "NOT_VALIDATED_REQUIRES_REAL_EMBED_EXTRACT",
    }

    calls.clear()
    install_failure = prepare_baseline_environment(
        install=lambda: (_ for _ in ()).throw(OSError("resolver unavailable")),
        probe=lambda: (_ for _ in ()).throw(ImportError("initial missing entry")),
    )
    assert calls == []
    assert install_failure["dependency_install_status"] == "FAILED"
    assert install_failure["entry_import_probe_status"] == "BLOCKED_DEPENDENCY_INSTALL_FAILED"
    assert install_failure["dependency_install_reason"] == "OSError: resolver unavailable"
    assert install_failure["initial_entry_import_probe_reason"] == "ImportError: initial missing entry"

    probe_calls = []
    probe_failure = prepare_baseline_environment(
        install=lambda: None,
        probe=lambda: probe_calls.append("probe") or (_ for _ in ()).throw(ImportError("entry unavailable")),
    )
    assert probe_calls == ["probe", "probe"]
    assert probe_failure["dependency_install_status"] == "COMPLETE_AFTER_IMPORT_FAILURE"
    assert probe_failure["entry_import_probe_status"] == "FAILED"
    assert probe_failure["entry_import_probe_reason"] == "ImportError: entry unavailable"
    assert probe_failure["model_compatibility_status"] == "NOT_VALIDATED_REQUIRES_REAL_EMBED_EXTRACT"


@pytest.mark.parametrize("initial_probe_result", ("nonzero", "oserror"))
def test_baseline_interpreter_rebuilds_partial_venv_without_ensurepip_and_validates_main_stack(
    tmp_path, initial_probe_result,
):
    venv = tmp_path / "baseline-venv"
    python = venv / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("partial interpreter", encoding="utf-8")
    stale = venv / "stale-marker"
    stale.write_text("left by failed ensurepip", encoding="utf-8")
    commands = []

    def run(command, stage, **kwargs):
        commands.append((stage, list(command), dict(kwargs)))
        if stage.endswith("_VENV_CREATE_WITHOUT_PIP"):
            created = Path(command[-1]) / "bin/python"
            created.parent.mkdir(parents=True, exist_ok=True)
            created.write_text("complete interpreter", encoding="utf-8")
            return 0
        if "-c" in command:
            compile(command[command.index("-c") + 1], stage, "exec")
            assert command[-1] == str(venv)
            if stage.endswith("_VENV_REUSE_PROBE"):
                if initial_probe_result == "oserror":
                    raise OSError("partial interpreter cannot start")
                return 1
            return 0
        raise AssertionError(command)

    prepared, receipt = ensure_baseline_interpreter(
        method="videoseal", venv=venv, main_python="/usr/bin/python3", run=run,
    )
    assert prepared == str(python)
    assert receipt["status"] == "REBUILT_AND_VALIDATED"
    assert receipt["initial_probe_returncode"] == (1 if initial_probe_result == "nonzero" else None)
    assert receipt["initial_probe_error"] == (
        None if initial_probe_result == "nonzero"
        else "OSError: partial interpreter cannot start"
    )
    assert receipt["ensurepip_used"] is False
    assert not stale.exists()
    create = next(command for stage, command, _ in commands if stage.endswith("_VENV_CREATE_WITHOUT_PIP"))
    assert create == [
        "/usr/bin/python3", "-m", "venv", "--without-pip",
        "--system-site-packages", str(venv),
    ]

    reuse_commands = []
    reused, reuse_receipt = ensure_baseline_interpreter(
        method="videoseal", venv=venv, main_python="/usr/bin/python3",
        run=lambda command, stage, **kwargs: reuse_commands.append((stage, command)) or 0,
    )
    assert reused == str(python)
    assert reuse_receipt["status"] == "REUSED_VALIDATED"
    assert len(reuse_commands) == 1 and "-c" in reuse_commands[0][1]
    assert reuse_commands[0][1][-1] == str(venv)


def test_generated_code_cells_run_in_order_from_fresh_isolated_kernel_with_boundary_stubs(tmp_path):
    isolated_python = Path("/usr/bin/python3")
    assert isolated_python.is_file()
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    script = isolated / "run_notebook_boundary_stub.py"
    identity_injection = textwrap.dedent('''\
        def prepared_checkout(label, url, commit, destination):
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
        changed_config = json.loads(namespace["EFFECTIVE_CONFIG"].read_text())
        changed_config["identity_observation_fixture"] = "different after init"
        namespace["atomic_json"](namespace["EFFECTIVE_CONFIG"], changed_config)

        original_logged = namespace["logged"]
        original_check_output = subprocess.check_output
        environment_commands = []
        entry_probe_attempts = {{}}
        def environment_logged(command, stage, **kwargs):
            environment_commands.append((stage, list(command)))
            if stage.endswith("_VENV_CREATE_WITHOUT_PIP"):
                vpython = Path(command[-1]) / "bin/python"
                vpython.parent.mkdir(parents=True, exist_ok=True)
                vpython.write_text("stub interpreter")
            if "-c" in command:
                compile(command[command.index("-c") + 1], stage, "exec")
            if stage.endswith("_ENTRY_IMPORT_PROBE"):
                entry_probe_attempts[stage] = entry_probe_attempts.get(stage, 0) + 1
                if entry_probe_attempts[stage] == 1:
                    raise subprocess.CalledProcessError(1, command)
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
            row["dependency_install_status"] == "COMPLETE_AFTER_IMPORT_FAILURE"
            and row["entry_import_probe_status"] == "IMPORT_READY_NO_MODEL_OR_WEIGHT_LOADED"
            for row in namespace["baseline_setup"].values()
        )
        assert not any("--no-deps" in command for _stage, command in environment_commands)
        create_commands = [
            command for stage, command in environment_commands
            if stage.endswith("_VENV_CREATE_WITHOUT_PIP")
        ]
        assert len(create_commands) == 2
        assert all("--without-pip" in command and "--system-site-packages" in command for command in create_commands)
        install_commands = [
            command for stage, command in environment_commands
            if stage.endswith("_ISOLATED_DEPENDENCIES")
        ]
        assert len(install_commands) == 2
        assert all(
            command[:4] == [namespace["PYTHON"], "-m", "pip", "--python"]
            and "install" in command and "--constraint" in command
            for command in install_commands
        )
        assert not any(
            command[0] in namespace["baseline_pythons"].values()
            and command[1:4] == ["-m", "pip", "install"]
            for command in install_commands
        )

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
        stages = json.loads(namespace["STAGE_RECEIPTS"].read_text())
        identity_stage = next(row for row in stages if row["stage"] == "EFFECTIVE_CONFIG_IDENTITY_OBSERVED")
        assert identity_stage["status"] == "RECORDED_DIFFERENCE"
        state = json.loads((namespace["RUN_OUTPUT"] / "run_state.json").read_text())
        assert state["config_identity_observation"]["status"] == "RECORDED_DIFFERENCE"
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
