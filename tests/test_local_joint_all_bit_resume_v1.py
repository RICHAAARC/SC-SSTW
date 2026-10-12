from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest
import torch

from runtime.wan import local_joint_all_bit_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from test_local_joint_all_bit_v1 import inputs, execute, threads
from test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT

pytestmark = pytest.mark.quick


def test_periodic_snapshot_is_recoverable_and_torn_primary_uses_previous(tmp_path):
    store=resume.Journal(tmp_path,interval=.02)
    value=runtime.initial_result({})
    store.save(value)
    fired=threading.Event()
    original=store.checkpoint
    def checkpoint(reason):
        original(reason)
        if reason.startswith("60-second"):
            fired.set()
    store.checkpoint=checkpoint
    with store.periodic():
        assert fired.wait(2)
    snapshot=json.loads((tmp_path/"checkpoints/latest.json").read_text())
    assert snapshot["views"]==value["views"] and snapshot["steps"]==value["steps"]
    assert (tmp_path/"progress.json").is_file()
    value["direction_status"]="DURABLE_SECOND_STATE"
    store.save(value)
    (tmp_path/"result.json").write_text('{"half_written":')
    (tmp_path/"result.json.tmp").write_text("half a new write")
    recovered=resume.read_state(tmp_path)
    assert recovered["direction_status"]=="NOT_RUN"
    assert recovered["recovery_events"][0]["loaded_from"].endswith("result.json.previous")
    # A broken primary must never replace the previous readable fallback.
    store.save(recovered)
    assert json.loads((tmp_path/"result.json.previous").read_text())["direction_status"]=="NOT_RUN"


@pytest.mark.parametrize("step,wrong_input",[("step_1",False),("step_2",False),("step_2",True)])
def test_complete_gradient_survives_interrupt_before_delta(tmp_path,monkeypatch,step,wrong_input):
    config=inputs(tmp_path)
    original=resume.atomic_tensor
    def stop(path,value):
        original(path,value)
        if Path(path).name==step+"_checkpoint.pt":
            raise KeyboardInterrupt("expiry immediately after complete gradient")
    with monkeypatch.context() as patch:
        patch.setattr(resume,"atomic_tensor",stop)
        first=execute(config,tmp_path,TinyWan())
    assert first["status"]=="INTERRUPTED"
    bundle=torch.load(tmp_path/"out/gradients"/(step+"_checkpoint.pt"),weights_only=True)
    assert torch.equal(bundle["input_terminal"],torch.load(tmp_path/"out"/("BASE" if step=="step_1" else "ALLBIT_HALF")/"terminal_latent.pt",weights_only=True))
    if wrong_input:
        bundle["input_terminal"]=bundle["input_terminal"]+.1
        resume.atomic_tensor(tmp_path/"out/gradients"/(step+"_checkpoint.pt"),bundle)
    result=execute(config,tmp_path,TinyWan())
    assert result["status"]=="COMPLETE",result["failures"]
    assert result["counts"]["decode_attempted"]==6+wrong_input
    assert result["counts"]["decoder_vjp_attempted"]==2+wrong_input
    assert result["counts"]["vae_load_attempted"]==2
    assert len([r for r in result["call_history"] if r["unit"]==step+"_vjp"])==1+wrong_input
    if wrong_input:
        assert any("different step input" in r["reason"] for r in result["recovery_events"])
    assert result["attempt_history"][-1]["status"]=="INTERRUPTED"
    assert result["attempt_history"][-1]["failures"]
    assert result["totals"]["observed_metrics"]==220


def test_unsaved_vjp_restarts_whole_unit_and_retains_physical_cost(tmp_path,monkeypatch):
    config=inputs(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(torch.autograd,"grad",lambda *_a,**_k:(_ for _ in ()).throw(KeyboardInterrupt("incomplete VJP")))
        first=execute(config,tmp_path,TinyWan())
    assert first["status"]=="INTERRUPTED"
    (tmp_path/"out/gradients/step_1_checkpoint.pt.tmp").write_bytes(b"half tensor")
    result=execute(config,tmp_path,TinyWan())
    assert result["status"]=="COMPLETE"
    assert result["counts"]["decode_attempted"]==7
    assert result["counts"]["decoder_vjp_attempted"]==3
    assert result["counts"]["decoder_vjp_completed"]==2
    attempts=[r for r in result["call_history"] if r["unit"]=="step_1_vjp"]
    assert [r["status"] for r in attempts]==["INTERRUPTED_COMPLETION_UNKNOWN","RETURNED"]
    assert len([r for r in result["call_history"] if r["unit"]=="step_1_gradient_decode" and "replay" in r])==2


def test_saved_rgb_then_cpu_scoring_and_complete_report_never_touch_cuda(tmp_path,monkeypatch):
    config=inputs(tmp_path)
    original=runtime.save_tensor
    def stop(path,tensor):
        original(path,tensor)
        if Path(path).parts[-2:]==("ALLBIT_TWO","float_rgb.pt"):
            raise KeyboardInterrupt("saved RGB before raw and score")
    with monkeypatch.context() as patch:
        patch.setattr(runtime,"save_tensor",stop)
        first=execute(config,tmp_path,TinyWan())
    assert first["status"]=="INTERRUPTED" and first["counts"]["decode_completed"]==6
    assert not runtime.model_work_remaining(tmp_path/"out")
    def forbidden(*_a,**_k):
        raise AssertionError("completed model work must not run")
    for name in ("is_available","reset_peak_memory_stats","max_memory_allocated","max_memory_reserved","empty_cache"):
        monkeypatch.setattr(torch.cuda,name,forbidden)
    config["device"]="cuda"  # A GPU-authored task must finish its CPU-only tail on CPU.
    result=runtime.run(config,tmp_path/"out",loader=forbidden,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert result["status"]=="COMPLETE",result["failures"]
    assert result["counts"]==first["counts"] and result["totals"]["observed_metrics"]==220
    assert runtime.reports_complete(tmp_path/"out")
    # A kill after the last artifact but before final status is report-only too.
    result["status"]="RUNNING"
    resume.Journal(tmp_path/"out").save(result)
    with monkeypatch.context() as patch:
        patch.setattr(torch,"load",forbidden)
        patch.setattr(runtime,"quality",forbidden)
        patch.setattr(runtime.observer,"read_rgb",forbidden)
        again=runtime.run(config,tmp_path/"out",loader=forbidden,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert again["status"]=="COMPLETE" and again["counts"]==first["counts"]
    assert again["last_resume_action"].startswith("report_only")
    directory=tmp_path/"out/ALLBIT_TWO"
    raw,score=(directory/"raw.json").read_bytes(),(directory/"metrics.json").read_bytes()
    (directory/"float_rgb.pt").write_bytes(b"unreadable saved RGB")
    (directory/"quality.json").unlink()
    assert not runtime.model_work_remaining(tmp_path/"out")
    partial=runtime.run(config,tmp_path/"out",loader=forbidden,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert partial["status"]=="ENGINEERING_FAILURE"
    assert partial["counts"]==first["counts"] and partial["totals"]["observed_metrics"]==220
    assert any("without re-decode" in r["reason"] for r in partial["failures"])
    assert (directory/"raw.json").read_bytes()==raw and (directory/"metrics.json").read_bytes()==score


def test_cross_process_hard_exit_recovers_without_ephemeral_inputs(tmp_path):
    ephemeral=tmp_path/"content"
    ephemeral.mkdir()
    config=inputs(ephemeral)
    drive=tmp_path/"drive"
    config_path=tmp_path/"config.json"
    config_path.write_text(json.dumps(config))
    root=Path(__file__).resolve().parents[1]
    script=tmp_path/"worker.py"
    script.write_text("""
import json, os, sys
from pathlib import Path
import torch
torch.set_num_threads(1)
from runtime.wan import local_joint_all_bit_v1 as runtime
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from test_local_joint_readout_m0_v1 import TinyWan, PROTOCOL, SHAPE, SUPPORT
config=json.loads(Path(sys.argv[1]).read_text())
if sys.argv[3]=='interrupt':
    original=resume.atomic_tensor
    def stop(path,value):
        original(path,value)
        if Path(path).name=='step_1_checkpoint.pt': os._exit(75)
    resume.atomic_tensor=stop
result=runtime.run(config,sys.argv[2],loader=lambda *_a,**_k:TinyWan(),protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
assert result['status']=='COMPLETE',result['failures']
""")
    env={**os.environ,"PYTHONPATH":os.pathsep.join((str(root),str(root/"tests"))),"M0_BOUNDARY_SPOOL_ROOT":str(ephemeral)}
    command=[sys.executable,str(script),str(config_path),str(drive)]
    stopped=subprocess.run(command+["interrupt"],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert stopped.returncode==75,stopped.stderr
    assert (drive/"gradients/step_1_checkpoint.pt").is_file()
    # Temporary inputs and activation spools disappear with the old /content VM.
    shutil.rmtree(ephemeral)
    ephemeral.mkdir()
    finished=subprocess.run(command+["resume"],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert finished.returncode==0,finished.stderr
    result=json.loads((drive/"result.json").read_text())
    assert result["counts"]["decode_attempted"]==6 and result["counts"]["decoder_vjp_attempted"]==2
    assert result["totals"]["observed_metrics"]==220
    assert result["attempt_history"][-1]["status"]=="RUNNING"


@pytest.mark.parametrize("terminal",["zero","tie"])
def test_method_terminal_resume_is_report_only_with_missing_dependent_views(tmp_path,monkeypatch,terminal):
    config=inputs(tmp_path)
    with monkeypatch.context() as patch:
        if terminal=="zero":
            original=torch.autograd.grad
            patch.setattr(torch.autograd,"grad",lambda *a,**k:(torch.zeros_like(original(*a,**k)[0]),))
        else:
            original=runtime.loss_from_values
            def tied(values):
                report=original(values)
                if report is not None: report["active_state_tie"]=True
                return report
            patch.setattr(runtime,"loss_from_values",tied)
        first=execute(config,tmp_path,TinyWan())
    assert first["status"]=="COMPLETE"
    assert first["views"]["ALLBIT_TWO"]["status"]=="MISSING"
    assert not runtime.model_work_remaining(tmp_path/"out")
    def forbidden(*_a,**_k): raise AssertionError("terminal method must not retry")
    monkeypatch.setattr(torch,"load",forbidden)
    monkeypatch.setattr(torch.cuda,"is_available",forbidden)
    config["device"]="cuda"
    again=runtime.run(config,tmp_path/"out",loader=forbidden,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)
    assert again["status"]=="COMPLETE" and again["counts"]==first["counts"]
    assert again["last_resume_action"].startswith("report_only")


def test_base_rgb_is_quality_reference_even_if_base_reader_fails(tmp_path,monkeypatch):
    config=inputs(tmp_path)
    original=runtime.observer.read_rgb
    calls=0
    def read(*a,**k):
        nonlocal calls
        calls+=1
        if calls==1: raise OSError("BASE reader unavailable after RGB save")
        return original(*a,**k)
    monkeypatch.setattr(runtime.observer,"read_rgb",read)
    result=execute(config,tmp_path,TinyWan())
    assert result["status"]=="ENGINEERING_FAILURE"
    assert result["counts"]["decode_completed"]==2 and result["counts"]["decoder_vjp_attempted"]==0
    assert result["views"]["BASE"]["status"]=="MISSING"
    assert result["views"]["TWO_MEAN_SAVED"]["status"]=="SCORED"
    assert (tmp_path/"out/TWO_MEAN_SAVED/quality.json").is_file()
