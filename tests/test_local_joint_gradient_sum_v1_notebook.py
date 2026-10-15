from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import zipfile

import pytest

from scripts import build_local_joint_gradient_sum_v1_notebook as builder

pytestmark = pytest.mark.quick


def code(notebook, index):
    return "".join(notebook["cells"][index]["source"])


def test_actual_notebook_is_direct_fixed_task_and_zip_is_current():
    import nbformat

    notebook = json.loads((builder.ROOT / "notebooks" / (builder.NAME + "_colab.ipynb")).read_text())
    assert notebook == builder.build_notebook()
    nbformat.validate(notebook)
    assert code(notebook, 0).splitlines() == ["from google.colab import drive", "drive.mount('/content/drive')"]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    node = next(node for node in ast.parse(code(notebook, 2)).body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "FIXED_CONFIG" for target in node.targets))
    config = ast.literal_eval(node.value)
    assert config == json.loads((builder.ROOT / builder.CONFIG_PATH).read_text())
    assert config["gradient_weights"] == {"aggregate_payload": 1, "state_G16": 1}
    assert config["step"] == .1
    assert config["inputs"]["alignment_run"].endswith("resumable-base-alignment-v1/run")
    assert config["inputs"]["base_float_rgb"].endswith("run/joint/float_rgb.pt")
    source = "\n".join(code(notebook, index) for index in range(len(notebook["cells"])))
    for forbidden in ("B64", "base64", "hashlib", "git rev-parse", "force_remount",
                      "strict", "override"):
        assert forbidden not in source
    assert "1 ordinary decode" in source
    assert "result['status'] != 'COMPLETE'" in code(notebook, 5)
    package = builder.portable_archive()
    assert package == (builder.ROOT / "notebooks" / (builder.NAME + "_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        for name in archive.namelist():
            assert archive.read(name) == (builder.ROOT / name).read_bytes()


def test_no_git_portable_cli_help_is_real(tmp_path):
    source = tmp_path / "source"
    with zipfile.ZipFile(io.BytesIO(builder.portable_archive())) as archive:
        archive.extractall(source)
    assert not (source / ".git").exists()
    command = [sys.executable, "-m", "experiments.wan_state_clock.local_joint_gradient_sum_v1_run", "--help"]
    result = subprocess.run(command, cwd=source, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--output" in result.stdout
