from __future__ import annotations

import ast
import io
import json
from pathlib import Path
import sys
import types
import urllib.request
import zipfile

import pytest

from scripts import build_local_joint_protected_common_v1_notebook as builder
from runtime.wan import local_joint_protected_common_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume

pytestmark = pytest.mark.quick


def code(nb,i): return "".join(nb["cells"][i]["source"])


def test_notebook_config_source_archive_contract():
    import nbformat
    nb=json.loads((builder.ROOT/"notebooks"/(builder.NAME+"_colab.ipynb")).read_text())
    assert nb == builder.build_notebook()
    nbformat.validate(nb)
    assert code(nb,0).splitlines() == ["from google.colab import drive","drive.mount('/content/drive')"]
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"])); assert cell["outputs"] == [] and cell["execution_count"] is None
    node=next(n for n in ast.parse(code(nb,2)).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == "FIXED_CONFIG" for t in n.targets))
    config=ast.literal_eval(node.value)
    assert config == json.loads((builder.ROOT/builder.CONFIG_PATH).read_text())
    assert config["step_size"] == .1 and config["carrier"]["cap"] == 1
    assert config["inputs"]["tangent_run"].endswith("resumable-tangent-common-v1/run")
    package=builder.portable_archive()
    assert package == (builder.ROOT/"notebooks"/(builder.NAME+"_portable_source.zip")).read_bytes()
    with zipfile.ZipFile(io.BytesIO(package)) as archive:
        assert set(archive.namelist()) == set(builder.PORTABLE_FILES)
        for name in archive.namelist(): assert archive.read(name) == (builder.ROOT/name).read_bytes()
    source="\n".join(code(nb,i) for i in range(len(nb["cells"])))
    for forbidden in ("B64","base64","hashlib","git rev-parse","force_remount","WanPipeline"):
        assert forbidden not in source
    assert "PAUSED_FOR_SESSION_RESTART" in code(nb,5)


@pytest.mark.parametrize("outcome",["pause","no_candidate","complete","killed"])
def test_ordered_notebook_pause_is_normal_and_complete_repeat_no_gpu(tmp_path,monkeypatch,outcome):
    nb=builder.build_notebook(); mounts=[]
    google=types.ModuleType("google"); colab=types.ModuleType("google.colab")
    colab.drive=types.SimpleNamespace(mount=mounts.append)
    colab.userdata=types.SimpleNamespace(get=lambda _:None); google.colab=colab
    monkeypatch.setitem(sys.modules,"google",google); monkeypatch.setitem(sys.modules,"google.colab",colab)
    ns={"__name__":"__fresh_notebook__"}
    exec(compile(code(nb,0),"mount","exec"),ns)
    setup=code(nb,2).replace("/content/drive/MyDrive/Video-WM/Local-Joint-State-Payload-V1-Protected-Common",str(tmp_path/"drive"))
    setup=setup.replace("/content/Video-WM-Local-Joint-Protected-Common-",str(tmp_path/"source-"))
    exec(compile(setup,"setup","exec"),ns)
    monkeypatch.setattr(urllib.request,"urlopen",lambda *_a,**_k:io.BytesIO(builder.portable_archive()))
    old_path=list(sys.path)
    old_modules={k:v for k,v in sys.modules.items() if k.split(".",1)[0] in ("main","runtime","experiments")}
    try:
        exec(compile(code(nb,3),"source","exec"),ns)
        import importlib
        loaded=importlib.import_module("runtime.wan.local_joint_protected_common_v1")
        assert Path(loaded.__file__).is_relative_to(ns["WORKSPACE"])
    finally:
        sys.path[:]=old_path
        for k in list(sys.modules):
            if k.split(".",1)[0] in ("main","runtime","experiments"): sys.modules.pop(k,None)
        sys.modules.update(old_modules)
    assert mounts == ["/content/drive"] and not (ns["WORKSPACE"]/".git").exists()
    stages=[]
    def logged(command,stage,**kwargs):
        stages.append(stage)
        if stage == "DEPENDENCY_PROBE":
            ast.parse(command[-1])
            if ns.get("second"): assert "cuda" not in command[-1] and "diffusers" not in command[-1]
            ns["write_json"](ns["OUTPUT"]/"dependency_probe.json",dict(model_work_remaining=not ns.get("second",False)))
            return 0
        if stage == "DEPENDENCY_REPORT": return 1
        assert stage == "FIXED_PROTECTED_COMMON"
        assert "experiments.wan_state_clock.local_joint_protected_common_v1_run" in command
        result=runtime.initial_result(ns["FIXED_CONFIG"]); result["attempt_id"]="stub"
        result["status"]={"pause":"PAUSED_FOR_SESSION_RESTART","no_candidate":"COMPLETE_NO_CANDIDATE","complete":"COMPLETE","killed":"RUNNING"}[outcome]
        if outcome == "killed":
            result["call_history"].append(dict(id="stub_0",kind="decode",unit="bit29_gradient_decode",status="RUNNING",attempt_id="stub"))
        else:
            ns["write_json"](ns["RUN_OUTPUT"]/"comparison.json",dict(bits_32=[dict(margins={},protected_gain=None,retained_tangent_gain_ratio=None)]*32,
                protections={},quality_vs_BASE={}))
        resume.Journal(ns["RUN_OUTPUT"],views=runtime.method.VIEWS,steps=runtime.STEPS).save(result)
        return -9 if outcome == "killed" else 0
    ns["logged"]=logged
    exec(compile(code(nb,4),"environment","exec"),ns)
    if outcome == "killed":
        with pytest.raises(RuntimeError,match="engineering interruption"):
            exec(compile(code(nb,5),"run","exec"),ns)
        result=runtime.read_state(ns["RUN_OUTPUT"])
        assert result["status"] == "INTERRUPTED"
        assert result["call_history"][0]["status"] == "INTERRUPTED_COMPLETION_UNKNOWN"
    else:
        exec(compile(code(nb,5),"run","exec"),ns)
        exec(compile(code(nb,6),"summary","exec"),ns)
        assert not list((ns["RUN_OUTPUT"]/"failures").glob("*"))
        if outcome != "pause":
            original=ns["ATTEMPT_OUTPUT"]; task=ns["OUTPUT"]
            exec(compile(setup,"setup_again","exec"),ns); ns["logged"]=logged; ns["second"]=True
            exec(compile(code(nb,4),"environment_again","exec"),ns)
            exec(compile(code(nb,5),"run_again","exec"),ns)
            assert ns["OUTPUT"] == task and ns["ATTEMPT_OUTPUT"] != original
