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
import zipfile
from pathlib import Path

import pytest

from experiments.paper_results_v1.colab_orchestration import (
    build_scope_summary,
    execute_fixed_sequence,
)


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
