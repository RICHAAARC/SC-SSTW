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
    assert "NEEDS_MODEL = needs_model_work(RUN_OUTPUT)" in source
    assert "read_state(RUN_OUTPUT) is None" not in source
    assert "Current attempt:" in source and "Earlier zero-call preflight snapshots:" in source
    assert "Failures (includes earlier preflight/report history):" in source


@pytest.mark.parametrize("physical_started", [False, True])
def test_dependency_cell_uses_first_model_work_decision(tmp_path, physical_started):
    from runtime.wan import local_joint_p0_input_vjp_v1 as runtime
    result = runtime.initial_result({})
    result.update(status="ENGINEERING_FAILURE", failures=[dict(stage="saved_source", reason="old dtype check")])
    if physical_started:
        result["calls"] = [dict(id="prep", kind="transformer_prepare", branch=None, status="FAILED")]
    runtime.recount(result)
    runtime.write_json(tmp_path / "result.json", result)
    commands = []
    def logged(command, stage):
        commands.append((command, stage))
        return 0
    def fail(stage, exc):
        raise AssertionError((stage, exc))
    scope = dict(RUN_OUTPUT=tmp_path, FIXED_CONFIG={"device": "cuda"}, PYTHON=sys.executable,
                 logged=logged, fail=fail, ATTEMPT=tmp_path, sys=sys, write_json=lambda *_: None)
    code = "".join(builder.build_notebook()["cells"][4]["source"])
    exec(compile(code, "notebook_dependency_cell", "exec"), scope)
    assert scope["NEEDS_MODEL"] is (not physical_started)
    assert ("diffusers" in commands[0][0][-1]) is (not physical_started)
    assert any(stage == "CUDA_AVAILABILITY" for _, stage in commands) is (not physical_started)


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
