from __future__ import annotations

import ast
import io
import json
import subprocess
import sys
import types
import urllib.request
import zipfile

import pytest

from scripts import build_local_joint_readout_m0_v1_notebook as builder
from runtime.wan import local_joint_readout_m0_v1 as runtime

pytestmark = pytest.mark.quick


def code(nb, i):
    return "".join(nb["cells"][i]["source"])


def test_notebook_and_ordinary_archive_are_current():
    notebook=json.loads((builder.ROOT/"notebooks"/(builder.NAME+"_colab.ipynb")).read_text())
    assert notebook == builder.build_notebook()
    assert code(notebook,0).splitlines() == ["from google.colab import drive","drive.mount('/content/drive')"]
    assert notebook["metadata"]["accelerator"] == "GPU"
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    package=builder.portable_archive()
    assert package == (builder.ROOT/"notebooks"/(builder.NAME+"_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        assert all(archive.read(name) == (builder.ROOT/name).read_bytes() for name in builder.PORTABLE_FILES)
    source="\n".join("".join(x["source"]) for x in notebook["cells"])
    for forbidden in ("B64", "base64", "hashlib", "git rev-parse", "apt-get", "WanPipeline"):
        assert forbidden not in source
    assert "urlopen(SOURCE_URL)" in source and "SOURCE_REF =" in source


@pytest.mark.parametrize("outcome",["complete","killed","environment_failure"])
def test_fresh_namespace_ordered_run_all_with_stubs(tmp_path,monkeypatch,outcome):
    nb=builder.build_notebook()
    mounts=[]
    google=types.ModuleType("google"); colab=types.ModuleType("google.colab")
    colab.drive=types.SimpleNamespace(mount=mounts.append)
    colab.userdata=types.SimpleNamespace(get=lambda _name: None)
    google.colab=colab
    monkeypatch.setitem(sys.modules,"google",google)
    monkeypatch.setitem(sys.modules,"google.colab",colab)
    ns={"__name__":"__fresh_notebook_stub__"}
    exec(compile(code(nb,0),"mount","exec"),ns)
    setup=code(nb,2).replace("/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Readout-M0",str(tmp_path/"drive"))
    setup=setup.replace("/content/Video-WM-Local-Joint-Readout-M0-",str(tmp_path/"source-"))
    exec(compile(setup,"setup","exec"),ns)
    monkeypatch.setattr(urllib.request,"urlopen",lambda *_a,**_k:io.BytesIO(builder.portable_archive()))
    # Only download is stubbed; actual local extraction uses the notebook cell.
    old_path=list(sys.path)
    try:
        exec(compile(code(nb,3),"source","exec"),ns)
    finally:
        sys.path[:]=old_path
    assert mounts == ["/content/drive"] and not (ns["WORKSPACE"]/".git").exists()
    stages=[]
    def logged(command,stage,**kwargs):
        stages.append(stage)
        if "-c" in command: ast.parse(command[command.index("-c")+1])
        if stage == "DEPENDENCY_PROBE":
            if outcome == "environment_failure": raise RuntimeError("dependency sentinel")
            ns["write_json"](ns["OUTPUT"]/"dependency_probe.json",dict(versions={},cuda_available=True,device="CPU_TEST_STUB"))
            return 0
        if stage == "DEPENDENCY_REPORT": return 1
        assert stage == "FIXED_READOUT_M0"
        assert "experiments.wan_state_clock.local_joint_readout_m0_v1_run" in command
        assert command[command.index("--config")+1] == str(ns["OUTPUT"]/"fixed_config.json")
        result=runtime.initial_result(ns["FIXED_CONFIG"])
        if outcome == "killed":
            result["counts"]["decode_attempted"]=1
            result["model_calls"]["BASE"]["status"]="RUNNING"
        else:
            result["status"]="COMPLETE"
        runtime.write_json(ns["RUN_OUTPUT"]/"result.json",result)
        return -9 if outcome == "killed" else 0
    ns["logged"]=logged
    monkeypatch.setattr(subprocess,"check_output",lambda *_a,**_k:"fixture-only==0\n")
    if outcome == "environment_failure":
        with pytest.raises(RuntimeError,match="dependency sentinel"):
            exec(compile(code(nb,4),"environment","exec"),ns)
        assert stages == ["DEPENDENCY_PROBE"]
        assert (ns["OUTPUT"]/"notebook_failure.json").exists()
        return
    exec(compile(code(nb,4),"environment","exec"),ns)
    if outcome == "killed":
        with pytest.raises(RuntimeError,match="engineering interruption"):
            exec(compile(code(nb,5),"run","exec"),ns)
        result=json.loads((ns["RUN_OUTPUT"]/"result.json").read_text())
        assert result["counts"]["decode_completed"] == 0
        assert result["model_calls"]["BASE"]["status"] == "INTERRUPTED_COMPLETION_UNKNOWN"
        assert result["totals"]["expected_metrics"] == 330 and result["totals"]["observed_metrics"] == 0
        assert result["status"] == "INTERRUPTED"
        assert len(json.loads((ns["RUN_OUTPUT"]/"comparison.json").read_text())["metrics"]) == 6
    else:
        exec(compile(code(nb,5),"run","exec"),ns)
        exec(compile(code(nb,6),"summary","exec"),ns)
    assert stages == ["DEPENDENCY_PROBE","DEPENDENCY_REPORT","FIXED_READOUT_M0"]
    assert (ns["OUTPUT"]/"execution_receipt.json").exists()


def test_standalone_no_git_zip_cli_missing_input_never_loads_model(tmp_path):
    source=tmp_path/"source"
    with zipfile.ZipFile(io.BytesIO(builder.portable_archive())) as archive:
        archive.extractall(source)
    config=json.loads((source/builder.CONFIG_PATH).read_text())
    config["inputs"]={"terminal_latent":str(tmp_path/"missing/terminal.pt"),
                      "bridge_delta":str(tmp_path/"missing/capped.pt")}
    (source/builder.CONFIG_PATH).write_text(json.dumps(config))
    completed=subprocess.run([sys.executable,"-m","experiments.wan_state_clock.local_joint_readout_m0_v1_run",
        "--config",str(source/builder.CONFIG_PATH),"--output",str(tmp_path/"out")],
        cwd=source,text=True,capture_output=True)
    assert completed.returncode == 1,completed.stderr
    result=json.loads((tmp_path/"out/result.json").read_text())
    assert all(x == 0 for x in result["counts"].values())
    assert result["totals"]["expected_metrics"] == 330 and result["totals"]["observed_metrics"] == 0
    assert result["status"] == "ENGINEERING_FAILURE"
