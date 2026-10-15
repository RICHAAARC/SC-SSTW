from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile
import pytest
import torch
from diffusers import UniPCMultistepScheduler
from runtime.wan import local_joint_paired_transfer_v1 as runtime
from runtime.wan import local_joint_paired_transfer_native_v1 as native
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from main.tube_state import local_joint_paired_transfer_v1 as method
from tests.test_local_joint_readout_m0_v1 import TinyWan,PROTOCOL,SHAPE,SUPPORT

pytestmark=pytest.mark.quick
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)
def config():
    c=json.loads((ROOT/"experiments/wan_state_clock/configs/local_joint_paired_transfer_v1.json").read_text())
    c.update(device="cpu",transformer_dtype="float32");return c
class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1),requires_grad=False);self.calls=0
    def forward(self,hidden_states,timestep,encoder_hidden_states,**_):
        self.calls+=1
        return (.03*torch.tanh(hidden_states*1.3)+.001*encoder_hidden_states.mean()+.00001*timestep.reshape(-1,1,1,1,1),)
def prepare(*_a,**_k):
    scheduler=UniPCMultistepScheduler(prediction_type="flow_prediction",use_flow_sigmas=True,flow_shift=3)
    scheduler.set_timesteps(50);scheduler.set_begin_index(0)
    return types.SimpleNamespace(transformer=Transformer(),scheduler=scheduler),torch.randn(SHAPE,generator=torch.Generator().manual_seed(8))*.2,torch.ones(1),torch.zeros(1),torch.float32
def invoke(c,out,models=None):
    if models is None:models=[]
    def loader(*_a,**_k):
        m=TinyWan();models.append(m);return m
    return runtime.run(c,out,preparer=prepare,loader=loader,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
def assert_complete(r):
    assert r["status"]=="COMPLETE",r["failures"]
    assert all(v["status"]=="SCORED" for v in r["views"].values())
def test_full_paired_paths_budget_quality_and_report_only(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    out=tmp_path/"out";models=[];c=config();r=invoke(c,out,models);assert_complete(r)
    assert len(models)==2 and [m.calls for m in models]==[2,1]
    assert r["counts"]["generation_prepare_completed"]==1
    assert r["counts"]["dit_completed"]==102 and r["counts"]["native_step_completed"]==52
    assert r["counts"]["decoder_vjp_completed"]==1 and r["counts"]["decode_completed"]==3
    assert r["ordinary_chunks"]==dict(attempted=92,completed=92)
    assert r["physical_replay_chunks"]["forward"]==dict(attempted=46,completed=46)
    assert r["physical_replay_chunks"]["recompute"]["attempted"]==46
    assert r["observation_accounting"]["endpoints"]["observed_metrics"]==165
    unit=runtime.tensor(out/"units/gradient.pt");zero=runtime.tensor(out/"ZERO/terminal_latent.pt")
    assert torch.equal(unit["input_terminal"],zero) and not (out/"ZERO/float_rgb.pt").exists()
    tr=r["transport"];assert tr["direct_terminal_effect"]["l2"]==pytest.approx(.1,abs=1e-6)
    assert tr["conditional_clean_operand"]["l2"]==pytest.approx(.02,abs=1e-7)
    assert tr["nominal_CFG_clean"]["l2"]==pytest.approx(.1,abs=1e-7)
    assert abs(tr["final_native_effect"]["l2"]-.1)>.001
    for name in ("TERMINAL","NATIVE"):
        q=resume.read_json(out/"quality"/(name+"_vs_ZERO")/"quality.json")
        assert len(q["framewise"])==181 and len(q["temporal_framewise"])==180 and q["peak_events"]
    report=resume.read_json(out/"comparison.json")
    assert all(len(x["payload_rows"])==32 and len(x["state_rows"])==22 for x in report["views"].values())
    assert not report["scientific_pass"] and not runtime.model_work_remaining(out)
    (out/"comparison.json").unlink()
    image=next((out/"quality/TERMINAL_vs_ZERO/frames").glob("*.png"));image.unlink()
    monkeypatch.setattr(runtime.observer,"evaluate_saved",lambda *_a,**_k:pytest.fail("rescore"))
    monkeypatch.setattr(runtime.previous.visual,"quality",lambda *_a,**_k:pytest.fail("quality rescore"))
    models=[];r=invoke(c,out,models);assert_complete(r)
    assert models==[] and not any(r["current_attempt_counts"].values()) and image.exists()
    assert r["quality"]["TERMINAL_vs_ZERO"]["rendered_images"]==[image.relative_to(out/"quality/TERMINAL_vs_ZERO").as_posix()]
    assert r["quality"]["NATIVE_vs_ZERO"]["rendered_images"]==[]
    changed=copy.deepcopy(c);changed["generation"]["prompt"]="new requested prompt";changed["model"]["revision"]="new-note"
    r=invoke(changed,out,models);assert_complete(r)
    assert r["config"]["generation"]["prompt"]==c["generation"]["prompt"]
    assert r["requested_config"]["generation"]["prompt"]=="new requested prompt"
    quality_path=out/"quality/TERMINAL_vs_ZERO/quality.json";quality_bytes=quality_path.read_bytes();quality_path.unlink()
    r=invoke(c,out,models)
    assert r["status"]=="ENGINEERING_FAILURE" and not models and not any(r["current_attempt_counts"].values())
    assert any("committed quality missing" in x["reason"] for x in r["failures"])
    quality_path.write_bytes(quality_bytes)
    (out/"units/gradient.pt").unlink()
    assert not runtime.model_work_remaining(out)
    r=invoke(c,out,models)
    assert r["status"]=="ENGINEERING_FAILURE" and not models
    assert any("committed gradient missing" in x["reason"] for x in r["failures"])

@pytest.mark.parametrize("phase,next_index",[("baseline",13),("tail",49),("tail",50)])
def test_native_interruption_restarts_only_uncommitted_step(tmp_path,monkeypatch,phase,next_index):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    out=tmp_path/"out";atomic=resume.atomic_tensor
    def stop(path,value):
        atomic(path,value)
        if str(path).endswith("/native/"+phase+".pt") and value["next_index"]==next_index:
            raise KeyboardInterrupt("expiry after atomic native unit")
    monkeypatch.setattr(resume,"atomic_tensor",stop)
    with pytest.raises(KeyboardInterrupt):invoke(config(),out)
    before=runtime.read_state(out)
    assert before["status"]=="INTERRUPTED"
    monkeypatch.setattr(resume,"atomic_tensor",atomic)
    r=invoke(config(),out);assert_complete(r)
    assert r["counts"]["dit_completed"]==102 and r["counts"]["native_step_completed"]==52
    assert r["counts"]["generation_prepare_completed"]==(1 if next_index==50 else 2)
    assert r["current_attempt_counts"]["dit_completed"]==(76 if phase=="baseline" else 0 if next_index==50 else 2)
    assert len(r["failure_history"])>=1
    if phase=="tail":assert r["current_attempt_counts"]["decoder_vjp_attempted"]==0

def test_vjp_failure_retains_completed_ZERO_forward(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    def fail(*_a,**_k):raise RuntimeError("injected VJP failure")
    monkeypatch.setattr(torch.autograd,"grad",fail)
    out=tmp_path/"out";r=invoke(config(),out)
    assert r["status"]=="ENGINEERING_FAILURE"
    assert r["views"]["ZERO"]["status"]=="SCORED" and r["units"]["gradient"]["status"]=="FAILED"
    assert r["counts"]["decoder_vjp_attempted"]==1 and r["counts"]["decoder_vjp_completed"]==0
    assert r["counts"]["decoder_vjp_failed"]==r["current_attempt_counts"]["decoder_vjp_failed"]==1
    assert r["counts"]["decoder_vjp_interrupted_completion_unknown"]==0
    assert len(list((out/"failures").glob("*_ZERO_forward.pt")))==1
    artifact=runtime.tensor(next((out/"failures").glob("*_ZERO_forward.pt")))
    assert artifact["status"]=="FORWARD_ONLY_VJP_FAILED" and not artifact["gradient_complete"]
    assert r["counts"]["dit_completed"]==100 and r["counts"]["decode_completed"]==1
    changed=config();changed["generation"]["prompt"]="other"
    with pytest.raises(ValueError,match="original source"):runtime.validate_config(changed)

def test_failure_and_unknown_counts_remain_distinct_across_attempts():
    r=runtime.initial_result(config());r["attempt_id"]="new"
    statuses=("FAILED","INTERRUPTED_COMPLETION_UNKNOWN","RETURNED","RUNNING")
    r["call_history"]=[dict(id=str(i),attempt_id="old" if i==0 else "new",unit=str(i),
        kind="transformer_conditional",status=s) for i,s in enumerate(statuses)]
    runtime.refresh_counts(r)
    assert r["counts"]["dit_attempted"]==4 and r["counts"]["dit_completed"]==1
    assert r["counts"]["dit_failed"]==r["counts"]["dit_interrupted_completion_unknown"]==r["counts"]["dit_running"]==1
    assert r["current_attempt_counts"]["dit_attempted"]==3 and r["current_attempt_counts"]["dit_failed"]==0
    assert r["current_attempt_counts"]["dit_interrupted_completion_unknown"]==1

def test_interrupted_native_DiT_unknown_count_and_tail_resume(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    original=Transformer.forward
    def stop(self,*a,**k):
        if self.calls==100:raise KeyboardInterrupt("expiry inside native final conditional DiT")
        return original(self,*a,**k)
    monkeypatch.setattr(Transformer,"forward",stop)
    out=tmp_path/"out"
    with pytest.raises(KeyboardInterrupt):invoke(config(),out)
    r=runtime.read_state(out)
    assert r["counts"]["dit_attempted"]==101 and r["counts"]["dit_completed"]==100
    assert r["counts"]["dit_failed"]==0 and r["counts"]["dit_interrupted_completion_unknown"]==1
    assert r["current_attempt_counts"]["transformer_conditional_interrupted_completion_unknown"]==1
    monkeypatch.setattr(Transformer,"forward",original)
    r=invoke(config(),out);assert_complete(r)
    assert r["counts"]["dit_attempted"]==103 and r["counts"]["dit_completed"]==102
    assert r["counts"]["dit_interrupted_completion_unknown"]==1 and r["counts"]["dit_failed"]==0
    assert r["current_attempt_counts"]["dit_attempted"]==r["current_attempt_counts"]["dit_completed"]==2
    assert r["current_attempt_counts"]["dit_interrupted_completion_unknown"]==0

def test_fixed_rows_keep_terminal_on_native_failure(tmp_path,monkeypatch):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    original=runtime.predict_branches
    def fail(pipe,z,s,p,n,d,i,count):
        if i==49 and pipe.transformer.calls>=100:raise RuntimeError("injected final DiT failure")
        return original(pipe,z,s,p,n,d,i,count)
    monkeypatch.setattr(runtime,"predict_branches",fail)
    out=tmp_path/"out";r=invoke(config(),out)
    assert r["status"]=="ENGINEERING_FAILURE"
    assert r["views"]["ZERO"]["status"]==r["views"]["TERMINAL"]["status"]=="SCORED"
    report=resume.read_json(out/"comparison.json")
    assert report["bounded_outcome"]=="UNRESOLVED_MISSING_OR_UNDEFINED"
    assert len(report["views"]["NATIVE"]["payload_rows"])==32
    assert {x["status"] for x in report["views"]["NATIVE"]["state_rows"]}=={"FAILED"}

@pytest.mark.parametrize("undefined",["ZERO_GRADIENT","UNDEFINED_ACTIVE_TIE"])
def test_undefined_preserves_ZERO_and_skips_dependent_models(tmp_path,monkeypatch,undefined):
    monkeypatch.setenv("M0_BOUNDARY_SPOOL_ROOT",str(tmp_path))
    if undefined=="ZERO_GRADIENT":monkeypatch.setattr(method,"scalar",lambda read:read["margins"].sum()*0)
    else:
        def tie(_):raise method.UndefinedActiveTie("injected active tie")
        monkeypatch.setattr(method,"scalar",tie)
    out=tmp_path/"out";models=[];r=invoke(config(),out,models)
    assert r["status"]=="COMPLETE_WITH_UNDEFINED_ARMS" and r["views"]["ZERO"]["status"]=="SCORED"
    assert r["counts"]["dit_completed"]==100 and r["counts"]["decode_completed"]==1
    assert len(models)==1 and r["units"]["gradient"]["status"]==undefined
    assert not runtime.model_work_remaining(out)

def test_notebook_zip_cli_and_adopted_semantics(tmp_path):
    from scripts import build_local_joint_paired_transfer_v1_notebook as builder
    import nbformat
    nb=builder.build_notebook();nbformat.validate(nbformat.from_dict(nb))
    assert json.loads((ROOT/"notebooks/local_joint_paired_transfer_v1_colab.ipynb").read_text())==nb
    for cell in nb["cells"]:
        if cell["cell_type"]=="code":ast.parse("".join(cell["source"]));assert cell["outputs"]==[]
    assert "".join(nb["cells"][0]["source"]).strip()=="from google.colab import drive\ndrive.mount('/content/drive')"
    ns={}
    tree=ast.parse("".join(nb["cells"][2]["source"]))
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="FIXED_CONFIG" for x in n.targets))
    assert ast.literal_eval(assignment.value)==json.loads((ROOT/builder.CONFIG_PATH).read_text())
    assert ast.literal_eval(assignment.value)["selection_status"]=="ADOPTED_BY_USER_20261015"
    assert runtime.initial_result(config())["selection_status"]=="ADOPTED_BY_USER_20261015"
    assert "pending user confirmation" not in method.CEILING and "adopted by the user" in method.CEILING
    assert "User-adopted recipe" in "".join(nb["cells"][1]["source"])
    archive=ROOT/"notebooks/local_joint_paired_transfer_v1_portable_source.zip"
    with zipfile.ZipFile(archive) as z:
        assert all(z.read(n)==(ROOT/n).read_bytes() for n in dict.fromkeys(builder.PORTABLE_FILES))
        z.extractall(tmp_path/"editable")
    cp=subprocess.run([sys.executable,"-m","experiments.wan_state_clock.local_joint_paired_transfer_v1_run","--help"],cwd=tmp_path/"editable",capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
    assert not (tmp_path/"editable/.git").exists()

@pytest.mark.parametrize("probe_failure",[None,"once","persistent"])
def test_environment_notes_optional_but_real_probe_errors_preserved(tmp_path,monkeypatch,probe_failure):
    import builtins
    from scripts import build_local_joint_paired_transfer_v1_notebook as builder
    code="".join(builder.build_notebook()["cells"][4]["source"])
    stages=[];failures=[];imports=[];original_import=builtins.__import__;original_write=Path.write_text
    # The optional write fails and leaves an old wrong-shaped JSON note; neither
    # that read problem nor the later receipt write may become a model gate.
    original_write(tmp_path/"dependency_probe.json","[]")
    def imported(name,*a,**k):
        if name=="diffusers":
            imports.append(name)
            if probe_failure=="persistent" or (probe_failure=="once" and len(imports)==1):
                raise ImportError("injected real probe import failure")
        return original_import(name,*a,**k)
    def optional_write(path,*a,**k):
        if path.name=="dependency_probe.json":raise OSError("injected optional note write failure")
        return original_write(path,*a,**k)
    monkeypatch.setattr(builtins,"__import__",imported);monkeypatch.setattr(Path,"write_text",optional_write)
    def logged(command,stage,check=True,**_):
        stages.append(stage);rc=0
        if "-c" in command:
            try:exec(command[-1],{})
            except ImportError:rc=1
        if check and rc:raise subprocess.CalledProcessError(rc,command)
        return rc
    def optional_receipt(path,value):raise OSError("injected optional receipt write failure")
    ns=dict(RUN_OUTPUT=tmp_path/"run",OUTPUT=tmp_path,ATTEMPT_OUTPUT=tmp_path,FIXED_CONFIG={"device":"cpu"},
        PYTHON=sys.executable,json=json,logged=logged,write_json=optional_receipt,record_failure=lambda *args:failures.append(args))
    if probe_failure=="persistent":
        with pytest.raises(subprocess.CalledProcessError):exec(code,ns)
        assert failures and "DEPENDENCY_REPAIR" in stages and "DEPENDENCY_REPROBE" in stages
    else:
        exec(code,ns)
        assert not failures and ("DEPENDENCY_REPAIR" in stages)==(probe_failure=="once")
        assert ns["environment_receipt"]["environment_note_error"]
