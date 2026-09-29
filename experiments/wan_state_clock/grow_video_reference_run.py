"""Fixed OFF/MULTI/LAST -> MP4 -> blind32 bits, two fresh child phases."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from main.tube_state import grow_video_reference as method
from runtime.wan import grow_video_reference as backend

ROOT=Path(__file__).resolve().parents[2]
CONFIG=Path(__file__).parent/"configs/grow_video_reference_v1.json"
MODULE="experiments.wan_state_clock.grow_video_reference_run"
KEY_IDS=("CORRECT","WRONG")
TARGET_IDS=("REGISTERED","WRONG_MESSAGE")
DENOMINATOR=dict(sources=1,arms=3,mp4=3,layers_per_arm=4,keys=2,raw_reads=24,evaluations=48)


def dump(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    with temp.open("w") as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write("\n")
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def environment():
    packages={}
    for name in ("torch","torchvision","diffusers","transformers","accelerate","numpy","imageio","imageio-ffmpeg"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return dict(python=sys.version,executable=sys.executable,packages=packages)


def load_config():
    cfg=json.loads(CONFIG.read_text())
    c=cfg["control"];g=cfg["generation"]
    if (cfg["fixed_denominator"]!=DENOMINATOR or cfg["arms"]!=list(method.ARMS)
        or c!={"channels":[0,1,2,3],"payload_bits":32,"alpha":0.5,"band_min":0.2,"band_max":0.5,
            "all_latent_times":46,"reference_mask_count":1600,"reference_eta":200.0,
            "actual_mask_count":44160,"derived_eta":5520.0,"MULTI":list(range(25,50)),
            "LAST":[49],"cfg_compensation":False,"total_budget_matching":False}
        or (g["height"],g["width"],g["frames"],g["steps"],g["guidance_scale"])!=(320,512,181,50,5.0)):
        raise ValueError("fixed construction/denominator mismatch")
    if len(method.message_bits(cfg["message"]))!=32 or len(method.message_bits(cfg["wrong_message"]))!=32:
        raise ValueError("32-bit truth evaluation targets required")
    method.layout_receipt(cfg["key"],cfg["wrong_key"])
    return cfg


def initial_result(output):
    rows={f"{a}/{l}/{k}":dict(status="PENDING",arm=a,layer=l,key_id=k,decoded_bits=None,truth_used=False)
          for a in method.ARMS for l in method.LAYERS for k in KEY_IDS}
    sources=("main/tube_state/grow_video_reference.py","runtime/wan/grow_video_reference.py",
        "runtime/wan/generation.py","runtime/wan/trajectory.py","runtime/wan/vae.py","runtime/wan/io.py",
        "experiments/wan_state_clock/grow_video_reference_run.py",
        "experiments/wan_state_clock/configs/grow_video_reference_v1.json",
        "experiments/wan_state_clock/requirements-grow-video-reference.txt")
    return dict(status="RUNNING",stage="INITIALIZE",fixed_denominator=DENOMINATOR,
        source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_files={p:backend.file_sha256(ROOT/p) for p in sources},
        config_sha256=backend.file_sha256(CONFIG),environment=environment(),output=str(output),
        generation={a:dict(status="PENDING",steps=[],terminal_path=str(output/a/"terminal.pt")) for a in method.ARMS},
        videos={a:dict(status="PENDING",path=str(output/a/(a+".mp4"))) for a in method.ARMS},
        reads=rows,evaluations={f"{key}/{target}":dict(status="PENDING",arm=row["arm"],layer=row["layer"],
            key_id=row["key_id"],target_id=target,bit_errors=None,ber=None,exact_bits=None)
            for key,row in rows.items() for target in TARGET_IDS},
        calls={},workers={},failures=[],quality={a:dict(status="PENDING",diagnostic_only=True) for a in method.ARMS},
        model_loaded={"generation":False,"media":False},
        actual_model_calls=False,evidence_ceiling="One-source fixed dependent comparisons; MP4 primary; no scientific PASS inferred from execution")


class Store:
    def __init__(self,output,*,create=False):
        self.output=Path(output);self.path=self.output/"result.json"
        if create:
            self.output.mkdir(parents=True,exist_ok=False);self.data=initial_result(self.output)
        else:self.data=json.loads(self.path.read_text())
        self.save()
    def save(self):
        d=self.data
        if len(d["generation"])!=3 or len(d["videos"])!=3 or len(d["reads"])!=24 or len(d["evaluations"])!=48:
            raise ValueError("fixed denominator changed")
        d["counts"]=dict(generated=sum(x["status"]=="COMPLETE" for x in d["generation"].values()),
            saved_mp4=sum(x["status"]=="SAVED" for x in d["videos"].values()),
            reads=sum(x["status"]=="READ" for x in d["reads"].values()),
            evaluated=sum(x["status"]=="EVALUATED" for x in d["evaluations"].values()))
        dump(self.path,d)
    def count(self,kind,completed):
        ledger=self.data["calls"].setdefault(self.data["stage"],{})
        row=ledger.setdefault(kind,dict(attempted=0,completed=0))
        row["completed" if completed else "attempted"]+=1
        if kind.startswith(("transformer_","vae_")) and not completed:self.data["actual_model_calls"]=True
        self.save()
    def failure(self,stage,exc):
        self.data["failures"].append(dict(stage=stage,error=f"{type(exc).__name__}: {exc}"))
        self.save()
    def blind(self):
        dump(self.output/"blind_readouts.json",dict(reads=self.data["reads"],public_payload_bits=32,
            truth_inputs=False,evaluated=False,primary_layer="mp4",fixed_read_rows=24))
    def set_reads(self,arm,layer,rows=None,error=None):
        for key in KEY_IDS:
            row=self.data["reads"][f"{arm}/{layer}/{key}"]
            if rows is None:row.update(status="MISSING_READOUT",error=error)
            else:row.update(rows[key])
        self.blind();self.save()


def generation_worker(store,cfg):
    import torch
    from runtime.wan.generation import prepare_generation
    from runtime.wan import trajectory
    device,dtype=backend.execution_device_dtype()
    store.data["stage"]="GENERATION_LOAD";store.save()
    pipe,initial,prompt,negative,input_dtype=prepare_generation(cfg,load_vae=False,device=device,model_dtype=dtype)
    store.data["model_loaded"]["generation"]=True
    pristine=copy.deepcopy(pipe.scheduler)
    initial_fp=trajectory.fingerprint(initial)
    history_fp=trajectory.fingerprint(vars(pristine))
    store.data["generation_setup"]=dict(device=device,transformer_dtype=str(input_dtype),
        state_dtype="torch.float32",control_dtype="torch.float32",cfg_dtype="torch.float32",
        initial_noise_sha256=initial_fp,pristine_history_sha256=history_fp,
        model=cfg["model"],case_id=cfg["case_id"],seed=cfg["generation"]["seed"],
        layout=method.layout_receipt(cfg["key"],cfg["wrong_key"]))
    store.save()
    for arm in method.ARMS:
        store.data["stage"]="GENERATE_"+arm;row=store.data["generation"][arm]
        row.update(status="RUNNING",initial_noise_sha256=initial_fp,initial_history_sha256=history_fp);store.save()
        def record(value):row["steps"].append(value);store.save()
        terminal,meta=backend.run_trajectory(pipe,initial,copy.deepcopy(pristine),prompt,negative,
            input_dtype,arm,cfg["key"],method.message_bits(cfg["message"]),store.count,record)
        if trajectory.fingerprint(initial)!=initial_fp or trajectory.fingerprint(vars(pristine))!=history_fp:
            raise RuntimeError("arm modified shared initial noise/pristine history")
        path=Path(row["terminal_path"]);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(terminal,path)
        row.update(status="COMPLETE",terminal_file_sha256=backend.file_sha256(path),metadata=meta)
        store.save();del terminal
    # Worker exit releases the transformer, embeddings and all scheduler history.


def media_worker(store,cfg):
    import torch
    from runtime.wan.generation import load_frozen_vae
    from runtime.wan import vae as vae_adapter,io
    if not any(x["status"]=="COMPLETE" for x in store.data["generation"].values()):
        raise RuntimeError("no completed generation terminal for media")
    store.data["stage"]="MEDIA_VAE_LOAD";store.save()
    device="cuda" if torch.cuda.is_available() else "cpu"
    vae=load_frozen_vae(cfg,device=device)
    store.data["model_loaded"]["media"]=True
    store.data["media_setup"]=dict(device=device,vae_dtype=str(next(vae.parameters()).dtype),
        scaling="Wan:decode(normalized*std+mean);encode(mode-mean)/std",model=cfg["model"])
    store.save()
    keys=dict(CORRECT=cfg["key"],WRONG=cfg["wrong_key"])
    for arm in method.ARMS:
        generated=store.data["generation"][arm];video=store.data["videos"][arm]
        if generated["status"]!="COMPLETE":
            video.update(status="MISSING_TERMINAL")
            for layer in method.LAYERS:store.set_reads(arm,layer,error="generation incomplete")
            continue
        try:
            if backend.file_sha256(generated["terminal_path"])!=generated["terminal_file_sha256"]:
                raise RuntimeError("saved terminal identity mismatch")
            terminal=torch.load(generated["terminal_path"],map_location="cpu",weights_only=True)
            store.data["stage"]="READ_"+arm+"_terminal"
            try:store.set_reads(arm,"terminal",backend.read_diagnostic_latent(terminal,keys,count=store.count))
            except Exception as exc:store.failure(store.data["stage"],exc);store.set_reads(arm,"terminal",error=str(exc))
            store.data["stage"]="DECODE_"+arm
            rgb=backend.counted(store.count,"vae_decode",lambda:vae_adapter.decode_normalized_latent(
                vae,terminal.to(device=device,dtype=torch.float32))).detach().cpu()
            del terminal
            if tuple(rgb.shape)!=method.PUBLIC.video_shape:raise ValueError("decoded video geometry mismatch")
            store.data["stage"]="SAVE_"+arm;video["status"]="SAVING";store.save()
            try:
                backend.counted(store.count,"mp4_save",lambda:io.encode_rgb(rgb,Path(video["path"]),cfg["media"]["fps"],cfg["media"]["crf"]))
                video.update(status="SAVED",sha256=backend.file_sha256(video["path"]),bytes=Path(video["path"]).stat().st_size)
            except Exception as exc:
                video.update(status="FAILED",error=str(exc));store.failure(store.data["stage"],exc)
            store.save()
            for layer in ("float_rgb","rgb8"):
                store.data["stage"]="READ_"+arm+"_"+layer
                observed=None
                try:
                    observed=rgb if layer=="float_rgb" else vae_adapter.quantize_rgb8_no_codec(rgb).float()/255
                    rows=backend.reencode_diagnostic_rgb(observed,keys,method.PUBLIC,vae,count=store.count)
                    store.set_reads(arm,layer,rows)
                except Exception as exc:store.failure(store.data["stage"],exc);store.set_reads(arm,layer,error=str(exc))
                finally:del observed
            del rgb
            store.data["stage"]="READ_"+arm+"_mp4"
            if video["status"]=="SAVED":
                try:
                    rows=backend.read_mp4_payload(video["path"],keys,method.PUBLIC,vae,count=store.count)
                    if any(row["mp4_sha256"]!=video["sha256"] for row in rows.values()):raise RuntimeError("MP4 identity mismatch")
                    store.set_reads(arm,"mp4",rows)
                except Exception as exc:store.failure(store.data["stage"],exc);store.set_reads(arm,"mp4",error=str(exc))
            else:store.set_reads(arm,"mp4",error="MP4 save incomplete")
        except Exception as exc:
            store.failure(store.data["stage"],exc)
            if video["status"]!="SAVED":video.update(status="FAILED",error=str(exc))
            for layer in method.LAYERS:
                if any(store.data["reads"][f"{arm}/{layer}/{k}"]["status"]=="PENDING" for k in KEY_IDS):
                    store.set_reads(arm,layer,error=str(exc))
        store.save()
    store.blind()


def recover_unfinished(store,phase,error):
    if phase=="generation":
        for row in store.data["generation"].values():
            if row["status"] in ("PENDING","RUNNING"):row.update(status="NOT_COMPLETED",error=error)
    else:
        for row in store.data["videos"].values():
            if row["status"] in ("PENDING","SAVING"):row.update(status="NOT_COMPLETED",error=error)
        for row in store.data["reads"].values():
            if row["status"]=="PENDING":row.update(status="MISSING_READOUT",error=error)
        store.blind()
    store.save()


def evaluate_saved_reads(store,cfg):
    path=store.output/"blind_readouts.json"
    if not path.is_file():store.blind()
    before=backend.file_sha256(path);raw=json.loads(path.read_text())
    truth=dict(REGISTERED=method.message_bits(cfg["message"]),WRONG_MESSAGE=method.message_bits(cfg["wrong_message"]))
    for identity,read in raw["reads"].items():
        for target,bits in truth.items():
            row=store.data["evaluations"][identity+"/"+target]
            row["status"]="MISSING_READOUT"
            if read["status"]=="READ":
                actual=read["decoded_bits"]
                if len(actual)!=32 or any(type(x) is not int or x not in (0,1) for x in actual):raise ValueError("invalid raw payload")
                errors=sum(a!=b for a,b in zip(actual,bits,strict=True))
                row.update(status="EVALUATED",bit_errors=errors,ber=errors/32,bit_accuracy=1-errors/32,
                    exact_bits=errors==0,decoded_string_diagnostic=bytes(int(''.join(map(str,actual[i:i+8])),2) for i in range(0,32,8)).decode('utf-8',errors='ignore'))
    if backend.file_sha256(path)!=before:raise RuntimeError("evaluation changed blind evidence")
    store.data["blind_readouts_sha256"]=before;store.save()


def quality_diagnostics(store):
    """Three extra codec reads; framewise FP64 accumulation, no VAE/truth."""
    from runtime.wan.io import read_mp4
    store.data["stage"]="QUALITY_MP4_DIAGNOSTICS"
    reference=None
    for arm in method.ARMS:
        row=store.data["quality"][arm];video=store.data["videos"][arm]
        observed=None
        try:
            if video["status"]!="SAVED":
                row.update(status="MISSING_MP4");continue
            if backend.file_sha256(video["path"])!=video["sha256"]:
                raise RuntimeError("quality MP4 identity mismatch")
            observed=backend.counted(store.count,"quality_mp4_read",lambda:read_mp4(Path(video["path"])))
            if tuple(observed.shape)!=method.PUBLIC.video_shape:raise ValueError("quality MP4 geometry mismatch")
            if arm=="OFF":
                reference=observed;squared=0.0
            elif reference is None:
                row.update(status="MISSING_OFF_REFERENCE");continue
            else:
                squared=0.0
                for index in range(observed.shape[0]):
                    delta=observed[index].double()-reference[index].double()
                    squared+=float(delta.square().sum())
            mse=squared/observed.numel()
            row.update(status="MEASURED",rgb_rmse=math.sqrt(mse),
                psnr_db=-10*math.log10(mse) if mse else None,identical_pixels=mse==0,
                source="same-format MP4 RGB24 readback relative to OFF",threshold=None,
                mp4_sha256=video["sha256"])
        except Exception as exc:
            row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}")
        finally:
            observed=None;store.save()


def _stop_worker(child):
    """Reap a started worker before loading its last disk state or proceeding."""
    if child.poll() is None:
        try:child.terminate()
        except ProcessLookupError:pass
    try:return child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        child.kill()
        return child.wait(timeout=10)


def run_worker_phase(output,phase):
    command=[sys.executable,"-u","-m",MODULE,"--output",str(output),"--worker",phase]
    start=time.perf_counter();child=None;code=None;error=None;cleanup_error=None;halt=None
    try:
        with (output/(phase+".log")).open("w") as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end="",flush=True);log.write(line);log.flush()
            code=child.wait()
    except BaseException as exc:
        error=f"{type(exc).__name__}: {exc}"
        if not isinstance(exc,Exception):halt="INTERRUPTED"
        if child is not None:
            try:code=_stop_worker(child)
            except BaseException as cleanup:
                cleanup_error=f"{type(cleanup).__name__}: {cleanup}"
                halt=halt or "CLEANUP_FAILED"
    finally:
        if child is not None and child.stdout is not None:
            try:child.stdout.close()
            except Exception:pass
    # Never save the parent's stale pre-launch Store over a worker checkpoint.
    # A child may persist its final completed data during terminate/wait itself.
    store=Store(output)
    status=("START_FAILED" if child is None else "MONITOR_FAILED") if error else ("COMPLETE" if code==0 else "NONZERO_EXIT")
    row=dict(command=command,status=status,returncode=code,elapsed_seconds=time.perf_counter()-start,
        child_started=child is not None,last_child_stage=store.data["stage"])
    if error:row["error"]=error
    if cleanup_error:row["cleanup_error"]=cleanup_error
    store.data["workers"][phase]=row
    if status!="COMPLETE":
        reason=error or f"worker exited{code}"
        store.data["failures"].append(dict(stage="WORKER_"+phase.upper(),error=reason,cleanup_error=cleanup_error))
        recover_unfinished(store,phase,reason)
    store.save()
    return store,halt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--worker",choices=("generation","media"))
    args=parser.parse_args();cfg=load_config()
    if args.worker:
        store=Store(args.output)
        try:(generation_worker if args.worker=="generation" else media_worker)(store,cfg)
        except Exception as exc:
            store.failure(store.data["stage"],exc);recover_unfinished(store,args.worker,str(exc));raise
        return
    store=Store(args.output,create=True)
    halt=None
    for phase in ("generation","media"):
        store,halt=run_worker_phase(args.output,phase)
        if halt:
            if phase=="generation":
                store.data["workers"]["media"]=dict(status="NOT_STARTED_"+halt,returncode=None,child_started=False)
            recover_unfinished(store,"media",halt)
            break
    evaluate_saved_reads(store,cfg)
    done=(store.data["counts"]==dict(generated=3,saved_mp4=3,reads=24,evaluated=48)
          and all(row["status"]=="COMPLETE" for row in store.data["workers"].values()))
    store.data.update(status=halt or ("EXECUTION_COMPLETE" if done else "INCOMPLETE"),stage="READ_EVALUATION_FINISHED")
    store.save()
    if halt:
        for row in store.data["quality"].values():row.update(status="NOT_RUN_"+halt)
    else:quality_diagnostics(store)
    store.data["stage"]="FINISHED"
    store.save();print(json.dumps(dict(status=store.data["status"],counts=store.data["counts"])))
    if halt=="INTERRUPTED":raise SystemExit(130)
    if not done:raise SystemExit(1)


if __name__=="__main__":main()
