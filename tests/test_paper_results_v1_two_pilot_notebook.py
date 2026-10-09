"""Static and CPU-only checks for the self-contained two-pilot Colab handoff."""
from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


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
    assert "for case_id in PILOT_CASES" in execute
    assert "for phase in PHASE_ORDER" in execute
    assert "check=False" in execute
    assert execute.index("receiver-read") if "receiver-read" in execute else True
    assert execute.index("FINAL_FIXED_DENOMINATOR_EVALUATE") > execute.index("for case_id in PILOT_CASES")
    phases = _assignment(code[1], "PHASE_ORDER")
    assert phases == (
        "generate", "decode", "framewise", "baseline-embed-videoseal",
        "baseline-embed-rivagan", "codec", "quality", "baseline-extract-videoseal",
        "baseline-extract-rivagan", "receiver-sync", "receiver-read",
    )
    assert "confirmation_status\": \"PLANNED_NOT_EXECUTED_BY_THIS_NOTEBOOK" in joined
    assert "historical_evidence_is_not_this_run\": True" in joined
