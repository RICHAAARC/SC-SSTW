"""Static checks for the fixed source-bound user-run notebook."""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_integrated_payload_notebook as builder

ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "integrated_payload_v1_colab.ipynb"
pytestmark = pytest.mark.unit


def read_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_notebook_is_fixed_run_all_and_bound_to_source_commit():
    notebook = read_notebook()
    assert notebook["metadata"]["source_commit"] == builder.SOURCE_SHA
    assert notebook["metadata"]["notebook_binding_kind"] == "immutable_source_commit_pending_publication"
    assert notebook["cells"][0]["cell_type"] == "code"
    assert "".join(notebook["cells"][0]["source"]).strip() == "from google.colab import drive\ndrive.mount('/content/drive')"
    combined = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    assert builder.SOURCE_SHA in combined
    assert "experiments.wan_state_clock.integrated_payload_run" in combined
    assert "integrated_payload_v1.json" in combined
    assert "/content/drive/MyDrive/Video-WM/SC-SSTW-Core-Integration" in combined
    assert "force_remount" not in combined and "@param" not in combined and "rglob" not in combined and "glob(" not in combined
    assert "MODE" not in combined and "--case-id" not in combined and "--stage" not in combined
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_notebook_builder_is_byte_reproducible():
    before = NOTEBOOK.read_bytes()
    completed = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_integrated_payload_notebook.py")], cwd=ROOT, check=False)
    assert completed.returncode == 0
    assert NOTEBOOK.read_bytes() == before
