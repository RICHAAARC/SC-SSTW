from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import build_local_joint_base_alignment_v1_notebook as builder
from runtime.wan import local_joint_base_alignment_v1 as runtime

pytestmark = pytest.mark.quick


def code(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def test_actual_notebook_nbformat_ast_config_and_portable_zip_are_current():
    notebook = json.loads((builder.ROOT/"notebooks"/(builder.NAME+"_colab.ipynb")).read_text())
    assert notebook == builder.build_notebook()
    assert notebook["nbformat"] == 4 and notebook["nbformat_minor"] >= 5
    assert code(notebook, 0).splitlines() == ["from google.colab import drive", "drive.mount('/content/drive')"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    assignment = next(node for node in ast.parse(code(notebook, 2)).body
                      if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "FIXED_CONFIG"
                              for target in node.targets))
    config = ast.literal_eval(assignment.value)
    assert config == json.loads((builder.ROOT/builder.CONFIG_PATH).read_text())
    assert config["nonpositive_roster"] == list(runtime.method.NONPOSITIVE_ROSTER)
    assert config["fixed_wrong_offset"] == 16
    assert config["inputs"]["six_bit_inputs"].endswith(
        "resumable-six-bit-final-v1/run/inputs/fixed_inputs.pt")
    assert config["inputs"]["bridge_run"].endswith("20261010T031603868487Z/run")
    package = builder.portable_archive()
    assert package == (builder.ROOT/"notebooks"/(builder.NAME+"_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        for name in archive.namelist():
            assert archive.read(name) == (builder.ROOT/name).read_bytes()
    source = "\n".join(code(notebook, index) for index in range(len(notebook["cells"])))
    for forbidden in ("B64", "base64", "hashlib", "git rev-parse", "force_remount", "strict", "override"):
        assert forbidden not in source


def test_no_git_portable_cli_help_and_missing_input_zero_calls(tmp_path):
    source = tmp_path/"source"
    with zipfile.ZipFile(io.BytesIO(builder.portable_archive())) as archive:
        archive.extractall(source)
    assert not (source/".git").exists()
    for path in source.rglob("*.py"):
        ast.parse(path.read_text())
    command = [sys.executable, "-m", "experiments.wan_state_clock.local_joint_base_alignment_v1_run"]
    help_result = subprocess.run(command+["--help"], cwd=source, capture_output=True, text=True)
    assert help_result.returncode == 0
    config = json.loads((source/builder.CONFIG_PATH).read_text())
    config["inputs"] = dict(six_bit_inputs=str(tmp_path/"missing/six.pt"),
                            bridge_run=str(tmp_path/"missing/bridge"))
    config_path = tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path/"output"
    run = subprocess.run(command+["--config", str(config_path), "--output", str(output)],
                         cwd=source, capture_output=True, text=True)
    assert run.returncode == 1, run.stderr
    result = runtime.read_state(output)
    assert result["status"] == "ENGINEERING_FAILURE"
    assert not any(result["counts"].values())
    assert result["totals"]["expected_metrics"] == 110
