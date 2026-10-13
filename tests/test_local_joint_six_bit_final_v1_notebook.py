from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import subprocess
import sys
import types
import urllib.request
import zipfile

import pytest

from scripts import build_local_joint_six_bit_final_v1_notebook as builder
from runtime.wan import local_joint_six_bit_final_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume

pytestmark = pytest.mark.quick


def code(nb,i):
    return "".join(nb["cells"][i]["source"])


def test_actual_notebook_config_and_zip_are_current():
    nb = json.loads((builder.ROOT/"notebooks"/(builder.NAME+"_colab.ipynb")).read_text())
    assert nb == builder.build_notebook()
    assert code(nb,0).splitlines() == ["from google.colab import drive","drive.mount('/content/drive')"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    node = next(n for n in ast.parse(code(nb,2)).body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == "FIXED_CONFIG" for t in n.targets))
    config = ast.literal_eval(node.value)
    assert config == json.loads((builder.ROOT/builder.CONFIG_PATH).read_text())
    assert config["source_run"] == "resumable-task-v1-2x010"
    assert config["inputs"]["final_step_unit"].endswith("resumable-task-v1-2x010/run/step_2_unit.pt")
    package = builder.portable_archive()
    assert package == (builder.ROOT/"notebooks"/(builder.NAME+"_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        for name in archive.namelist(): assert archive.read(name) == (builder.ROOT/name).read_bytes()
    source = "\n".join(code(nb,i) for i in range(len(nb["cells"])))
    for forbidden in ("B64","base64","hashlib","git rev-parse","force_remount","WanPipeline"):
        assert forbidden not in source


@pytest.mark.parametrize("outcome",["complete","killed"])
def test_ordered_fresh_and_repeated_notebook_with_stubs(tmp_path,monkeypatch,outcome):
    nb = builder.build_notebook(); mounts=[]
    google = types.ModuleType("google"); colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mounts.append)
    colab.userdata = types.SimpleNamespace(get=lambda _:None); google.colab = colab
    monkeypatch.setitem(sys.modules,"google",google); monkeypatch.setitem(sys.modules,"google.colab",colab)
    ns = {"__name__":"__fresh_notebook__"}
    exec(compile(code(nb,0),"mount","exec"),ns)
    setup = code(nb,2).replace("/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Six-Bit-Final",str(tmp_path/"drive"))
    setup = setup.replace("/content/Video-WM-Local-Joint-Six-Bit-Final-",str(tmp_path/"source-"))
    exec(compile(setup,"setup","exec"),ns)
    task,attempt = ns["OUTPUT"],ns["ATTEMPT_OUTPUT"]
    monkeypatch.setattr(urllib.request,"urlopen",lambda *_a,**_k:io.BytesIO(builder.portable_archive()))
    old_path = list(sys.path)
    old_modules = {k:v for k,v in sys.modules.items() if k.split(".",1)[0] in ("main","runtime","experiments")}
    try:
        exec(compile(code(nb,3),"source","exec"),ns)
        import importlib
        loaded = importlib.import_module("runtime.wan.local_joint_six_bit_final_v1")
        assert Path(loaded.__file__).is_relative_to(ns["WORKSPACE"])
    finally:
        sys.path[:] = old_path
        for k in list(sys.modules):
            if k.split(".",1)[0] in ("main","runtime","experiments"): sys.modules.pop(k,None)
        sys.modules.update(old_modules)
    assert mounts == ["/content/drive"] and not (ns["WORKSPACE"]/".git").exists()
    stages=[]
    def logged(command,stage,**kwargs):
        stages.append(stage)
        if stage == "DEPENDENCY_PROBE":
            ast.parse(command[-1])
            if ns.get("second"):
                assert "cuda" not in command[-1] and "diffusers" not in command[-1]
            ns["write_json"](task/"dependency_probe.json",dict(model_work_remaining=not ns.get("second",False)))
            return 0
        if stage == "DEPENDENCY_REPORT": return 0
        assert stage == "FIXED_SIX_BIT_FINAL"
        assert "experiments.wan_state_clock.local_joint_six_bit_final_v1_run" in command
        result = runtime.initial_result(ns["FIXED_CONFIG"])
        result["attempt_id"] = "stub"
        if outcome == "killed":
            result["call_history"].append(dict(id="stub_0",kind="decode",unit="bit_04_gradient_decode",status="RUNNING",attempt_id="stub"))
        else: result["status"] = "COMPLETE"
        resume.Journal(ns["RUN_OUTPUT"],views=runtime.method.VIEWS,steps=()).save(result)
        return -9 if outcome == "killed" else 0
    ns["logged"] = logged
    exec(compile(code(nb,4),"environment","exec"),ns)
    if outcome == "killed":
        with pytest.raises(RuntimeError,match="engineering interruption"):
            exec(compile(code(nb,5),"run","exec"),ns)
        result = runtime.read_state(ns["RUN_OUTPUT"])
        assert result["status"] == "INTERRUPTED" and result["counts"]["decode_completed"] == 0
        assert result["totals"]["expected_metrics"] == 330
        return
    exec(compile(code(nb,5),"run","exec"),ns)
    exec(compile(code(nb,6),"summary","exec"),ns)
    ns["FIXED_CONFIG"]["inputs"]["terminal_latent"] = "edited locator"
    ns["write_json"](task/"fixed_config.json",ns["FIXED_CONFIG"])
    exec(compile(setup,"setup_again","exec"),ns)
    assert ns["OUTPUT"] == task and ns["ATTEMPT_OUTPUT"] != attempt
    assert ns["FIXED_CONFIG"]["inputs"]["terminal_latent"] == "edited locator"
    ns["second"] = True; ns["logged"] = logged
    monkeypatch.setattr(runtime,"model_work_remaining",lambda _:False)
    exec(compile(code(nb,4),"environment_cpu_tail","exec"),ns)
    exec(compile(code(nb,5),"rerun","exec"),ns)
    exec(compile(code(nb,6),"summary_again","exec"),ns)


def test_no_git_archive_cli_missing_input_zero_calls(tmp_path):
    source = tmp_path/"source"
    with zipfile.ZipFile(io.BytesIO(builder.portable_archive())) as archive: archive.extractall(source)
    for path in source.rglob("*.py"): ast.parse(path.read_text())
    config = json.loads((source/builder.CONFIG_PATH).read_text())
    config["inputs"] = {k:str(tmp_path/"missing"/k) for k in config["inputs"]}
    (source/builder.CONFIG_PATH).write_text(json.dumps(config))
    command = [sys.executable,"-m","experiments.wan_state_clock.local_joint_six_bit_final_v1_run"]
    help_result = subprocess.run(command+["--help"],cwd=source,capture_output=True,text=True)
    assert help_result.returncode == 0
    run = subprocess.run(command+["--config",str(source/builder.CONFIG_PATH),"--output",str(tmp_path/"out")],
                         cwd=source,capture_output=True,text=True)
    assert run.returncode == 1,run.stderr
    result = runtime.read_state(tmp_path/"out")
    assert result["status"] == "ENGINEERING_FAILURE" and not any(result["counts"].values())
    assert result["totals"]["expected_metrics"] == 330 and len(result["views"]) == 6
