"""Fixed P0 delivery checks; local archive/AST/CLI only, no network/model calls."""
import ast
import json
import subprocess
import sys
import zipfile

import pytest
from scripts import build_local_joint_p0_input_vjp_v1_notebook as builder

pytestmark = pytest.mark.unit


def test_notebook_builder_mount_and_fixed_budget_match():
    path = builder.ROOT / "notebooks" / (builder.NAME + "_colab.ipynb")
    notebook = json.loads(path.read_text())
    assert notebook == builder.build_notebook()
    assert "".join(notebook["cells"][0]["source"]) == "from google.colab import drive\ndrive.mount('/content/drive')"
    codes = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for code in codes:
        ast.parse(code)
    source = "\n".join(codes)
    for prohibited in ("force_remount", "files.upload", "b64decode", "sha256", "manifest", "expected_sha", "strict=", "override="):
        assert prohibited not in source
    assert "target.is_relative_to(WORKSPACE.resolve())" in source
    assert "finalize_interrupted" in source and "start_new_session=True" in source
    assert "local_joint_p0_input_vjp_v1_run" in source


def test_editable_zip_standalone_cli_without_git(tmp_path):
    archive_path = builder.ROOT / "notebooks" / (builder.NAME + "_portable_source.zip")
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        for name in builder.PORTABLE_FILES:
            assert archive.read(name) == (builder.ROOT / name).read_bytes()
        archive.extractall(tmp_path)
    # Exercise an edited ordinary source copy: there is no source-digest admission.
    config = tmp_path / builder.CONFIG_PATH
    value = json.loads(config.read_text())
    value["schema"] = "edited optional source label"
    config.write_text(json.dumps(value))
    result = subprocess.run([sys.executable, "-m", "experiments.wan_state_clock.local_joint_p0_input_vjp_v1_run", "--help"],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout and "--output" in result.stdout
