from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile
import urllib.request

import pytest

from scripts import build_local_joint_terminal_bridge_v1_notebook as builder
from runtime.wan import local_joint_terminal_bridge_v1 as runtime

pytestmark = pytest.mark.quick


def notebook():
    return json.loads((builder.ROOT/"notebooks"/(builder.NAME+"_colab.ipynb")).read_text())


def code(nb, i):
    return "".join(nb["cells"][i]["source"])


def setup(tmp_path, monkeypatch):
    nb = notebook()
    mounts = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path: mounts.append(path))
    colab.userdata = types.SimpleNamespace(get=lambda _name: None)
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = {"__name__": "__notebook_stub__"}
    exec(compile(code(nb, 0), "mount", "exec"), namespace)
    source = code(nb, 2).replace(
        "/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Terminal-Bridge", str(tmp_path/"drive"))
    source = source.replace("/content/Video-WM-Local-Joint-Terminal-Bridge-", str(tmp_path/"source-"))
    exec(compile(source, "setup", "exec"), namespace)
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_a, **_k: io.BytesIO(builder.portable_archive()))
    # Download is stubbed; extraction is real, local and model-free.
    exec(compile(code(nb, 3), "source", "exec"), namespace)
    assert mounts == ["/content/drive"]
    return nb, namespace


def test_new_notebook_and_archive_are_current_and_empty():
    nb = notebook()
    assert nb == builder.build_notebook()
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    archive = builder.portable_archive()
    assert archive == (builder.ROOT/"notebooks"/(builder.NAME+"_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert set(z.namelist()) == set(builder.PORTABLE_FILES)
        assert all(z.read(name) == (builder.ROOT/name).read_bytes() for name in builder.PORTABLE_FILES)
    assert nb["metadata"]["accelerator"] == "GPU"
    source = "\n".join("".join(cell["source"]) for cell in nb["cells"])
    assert "B64" not in source and "base64" not in source and "hashlib" not in source
    assert "urlopen(SOURCE_URL)" in source
    environment = code(nb, 4)
    assert "apt-get" not in environment
    assert "WanPipeline" not in environment
    assert "assert torch.cuda.is_available()" in environment
    assert "version_differences" in environment  # recorded, no equality assertion


@pytest.mark.parametrize("outcome", ["success", "child_killed", "environment_failure"])
def test_run_all_sequential_stub_and_failure_takeover(tmp_path, monkeypatch, outcome):
    nb, ns = setup(tmp_path, monkeypatch)
    stages = []
    def logged(command, stage, **kwargs):
        stages.append(stage)
        if "-c" in command:
            ast.parse(command[command.index("-c")+1])
        if stage == "DEPENDENCY_PROBE":
            if outcome == "environment_failure":
                raise RuntimeError("sentinel dependency failure")
            ns["write_json"](ns["OUTPUT"]/"dependency_probe.json", dict(versions={}, cuda_available=True, device="STUB"))
            return 0
        if stage == "DEPENDENCY_REPORT":
            return 1
        assert stage == "FIXED_TERMINAL_BRIDGE"
        assert "experiments.wan_state_clock.local_joint_terminal_bridge_v1_run" in command
        result = runtime.initial_result(ns["FIXED_CONFIG"])
        result["counts"].update(encode_attempted=2, encode_completed=2, decode_attempted=1 if outcome == "child_killed" else 4,
                                decode_completed=0 if outcome == "child_killed" else 4)
        result["stages"]["base"].update(status="COMPLETE")
        result["stages"]["base"]["views"]["postclip"] = dict(status="OBSERVED", state_gap=-.2)
        if outcome == "child_killed":
            result["model_calls"]["posterior_reconstruction"]["status"] = "RUNNING"
        else:
            result["status"] = "COMPLETE"
        runtime.write_json(ns["RUN_OUTPUT"]/"result.json", result)
        return -9 if outcome == "child_killed" else 0
    ns["logged"] = logged
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: "test-only==0\n")
    if outcome == "environment_failure":
        with pytest.raises(RuntimeError, match="sentinel dependency failure"):
            exec(compile(code(nb, 4), "environment", "exec"), ns)
        assert stages == ["DEPENDENCY_PROBE"]
        assert (ns["OUTPUT"]/"notebook_failure.json").exists()
        assert not ns["RUN_OUTPUT"].exists()
        return
    exec(compile(code(nb, 4), "environment", "exec"), ns)
    if outcome == "child_killed":
        with pytest.raises(RuntimeError, match="incomplete"):
            exec(compile(code(nb, 5), "run", "exec"), ns)
        result = json.loads((ns["RUN_OUTPUT"]/"result.json").read_text())
        assert result["status"] == "INTERRUPTED"
        assert result["counts"]["decode_completed"] == 0
        assert result["model_calls"]["posterior_reconstruction"]["status"] == "INTERRUPTED_COMPLETION_UNKNOWN"
        assert result["stages"]["base"]["views"]["postclip"]["state_gap"] == -.2
        comparisons = json.loads((ns["RUN_OUTPUT"]/"paired_comparisons.json").read_text())["comparisons"]
        assert len(comparisons) == 14
        assert all(len(pair["chips"]) == 1408 and pair["missing_values"] == 55
                   for pair in comparisons.values())
        assert all(chip["signed_q_delta_status"] == "MISSING"
                   for pair in comparisons.values() for chip in pair["chips"])
        missing = json.loads((ns["RUN_OUTPUT"]/"candidate/postclip_metrics.json").read_text())
        assert "child exited with return code -9" in missing["engineering_reason"]
    else:
        exec(compile(code(nb, 5), "run", "exec"), ns)
        exec(compile(code(nb, 6), "summary", "exec"), ns)
    assert stages == ["DEPENDENCY_PROBE", "DEPENDENCY_REPORT", "FIXED_TERMINAL_BRIDGE"]
    assert (ns["OUTPUT"]/"execution_receipt.json").exists()
    assert not (ns["WORKSPACE"]/".git").exists()


def test_no_git_archive_cli_runs_missing_input_path_without_models(tmp_path):
    package = builder.portable_archive()
    source = tmp_path/"source"
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        archive.extractall(source)
    config = json.loads((source/builder.CONFIG_PATH).read_text())
    config["inputs"] = dict(terminal_latent=str(tmp_path/"absent_inputs/terminal.pt"),
                            float_rgb=str(tmp_path/"absent_inputs/float.pt"))
    config_path = source/builder.CONFIG_PATH
    config_path.write_text(json.dumps(config))
    completed = subprocess.run([sys.executable, "-m", "experiments.wan_state_clock.local_joint_terminal_bridge_v1_run",
        "--config", str(config_path), "--output", str(tmp_path/"output")], cwd=source, text=True, capture_output=True)
    assert completed.returncode == 1, completed.stderr
    result = json.loads((tmp_path/"output/result.json").read_text())
    assert result["status"] == "ENGINEERING_FAILURE"
    assert all(value == 0 for value in result["counts"].values())
    assert result["source_identity"] == dict(workspace=str(source), config_path=str(config_path))
    assert (tmp_path/"output/paired_comparisons.json").exists()
