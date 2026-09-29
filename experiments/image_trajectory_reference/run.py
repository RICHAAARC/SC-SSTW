"""Fixed official GROW writer -> PNG -> independent bits -> separate evaluation."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .official_bridge import (ROOT, UPSTREAM_SHA, PublicReaderConfig, layout_receipt,
                              load_official, read_saved_png, sha256, verify_source)

CONFIG_PATH = Path(__file__).with_name("config.json")
ARMS = ("OFF", "GROW")
KEYS = ("CORRECT", "WRONG")
TARGETS = ("REGISTERED", "WRONG_MESSAGE")
DENOMINATOR = dict(sources=1, png_images=2, blind_readouts=4, evaluations=8)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w",encoding="utf-8") as stream:
        json.dump(data,stream,indent=2,allow_nan=False)
        stream.write("\n");stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def load_config(upstream_root):
    cfg = json.loads(CONFIG_PATH.read_text())
    GrowConfig, _, codec = load_official(upstream_root)
    defaults = GrowConfig()
    if (cfg["upstream_commit"] != UPSTREAM_SHA or cfg["fixed_denominator"] != DENOMINATOR
            or cfg["public_payload_bits"] != 32):
        raise ValueError("fixed upstream/denominator/payload contract mismatch")
    if any(getattr(defaults,k) != v for k,v in cfg["official_config"].items()):
        raise ValueError("official default parameter changed")
    if any(len(codec.message_to_bits(value)) != 32 for value in
           (cfg["official_config"]["message"],cfg["wrong_message"])):
        raise ValueError("fixed 32-bit evaluation targets required")
    if cfg["wrong_message"] == cfg["official_config"]["message"]:
        raise ValueError("wrong message equals registered message")
    return cfg


def initial_result(output):
    output = Path(output)
    return dict(status="RUNNING", stage="INITIALIZE", fixed_denominator=DENOMINATOR,
        config_sha256=sha256(CONFIG_PATH),
        source_sha=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        source_files={str(p.relative_to(ROOT)):sha256(p) for p in (
            Path(__file__),Path(__file__).with_name("official_bridge.py"),CONFIG_PATH)},
        real_model_executed=False, model_loaded=False, model_calls_executed=False,
        calls={}, upstream=None, assets=None, environment=None,
        images={arm:dict(status="PENDING",path=str(output/f"{arm}.png"),sha256=None,
                         bytes=None,seed=42,error=None) for arm in ARMS},
        reads={f"{arm}/{key}":dict(status="PENDING",image_id=arm,key_id=key,
            decoded_bits=None,truth_used=False,error=None) for arm in ARMS for key in KEYS},
        evaluations={f"{arm}/{key}/{target}":dict(status="PENDING",image_id=arm,
            key_id=key,target_id=target,bit_errors=None,ber=None,bit_accuracy=None,
            exact_bits=None) for arm in ARMS for key in KEYS for target in TARGETS},
        failures=[], quality=None,output_dir=str(output), elapsed_seconds=None,
        evidence_ceiling="One fixed official-code saved-PNG comparison; no paper-reproduction, population-FPR, robustness or video claim")


def save_result(path,result):
    if (len(result["images"])!=2 or len(result["reads"])!=4 or len(result["evaluations"])!=8):
        raise RuntimeError("fixed 2/4/8 denominator changed")
    result["counts"] = dict(saved_images=sum(r["status"]=="SAVED" for r in result["images"].values()),
        readouts=sum(r["status"]=="READ" for r in result["reads"].values()),
        evaluated=sum(r["status"]=="EVALUATED" for r in result["evaluations"].values()))
    atomic_json(path,result)


def preserve_unfinished(result,reason):
    for group in ("images","reads","evaluations"):
        for row in result[group].values():
            if row["status"] in ("PENDING","GENERATING","READING"):
                row.update(status="NOT_COMPLETED",error=reason)


def environment_receipt():
    packages=("torch","torchvision","diffusers","transformers","accelerate","numpy","Pillow","tqdm","PyYAML","huggingface_hub","safetensors")
    versions={}
    for name in packages:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return dict(python=sys.version,packages=versions,selection="active compatible runtime; recorded, not an exact-version gate")


def load_real_pipeline(config, record_assets):
    """Load the selected SD2.1-base mirror; metadata APIs are not a prerequisite."""
    import torch
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    asset=config["model_asset"]
    model_id,revision=asset["repo_id"],asset["revision"]
    device="cuda" if torch.cuda.is_available() else "cpu"
    receipt=dict(model_id=model_id,original_model_id=asset["original_repo_id"],
        source_kind=asset["kind"],status="LOADING",stage="SCHEDULER_LOAD",
        revision=revision,device=device,metadata_api_required=False,
        asset_source_changed=True)
    if device=="cpu":
        receipt["runtime_note"]="CUDA unavailable: running on CPU; GPU is recommended for speed."
    record_assets(receipt)
    try:
        scheduler=DDIMScheduler.from_pretrained(model_id,subfolder="scheduler",revision=revision)
        receipt.update(stage="PIPELINE_LOAD");record_assets(receipt)
        pipe=StableDiffusionPipeline.from_pretrained(model_id,scheduler=scheduler,
            torch_dtype=torch.float32,revision=revision,use_safetensors=True).to(device)
    except Exception as exc:
        receipt.update(status="ASSET_ACCESS_REQUIRED" if asset_access_error(exc) else "LOAD_FAILED",
                       error=f"{type(exc).__name__}: {exc}",
                       http_status=getattr(getattr(exc,"response",None),"status_code",None))
        record_assets(receipt)
        raise
    receipt.update(status="LOADED",stage="LOADED",scheduler=type(pipe.scheduler).__name__,
        scheduler_config=dict(pipe.scheduler.config),dtype="float32",
        device_name=torch.cuda.get_device_name(0) if device=="cuda" else "CPU",
        all_components_revision=revision)
    record_assets(receipt)
    return pipe


def asset_access_error(exc):
    """Classify access/missing assets through wrapped Hub errors, not OOM."""
    seen=set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        status=getattr(getattr(exc,"response",None),"status_code",None)
        if status in (401,403,404) or isinstance(exc,FileNotFoundError):
            return True
        exc=exc.__cause__ or exc.__context__
    return False


def observe_calls(pipe,result,persist):
    """Hooks observe actual upstream calls without extra model evaluations."""
    handles=[]
    for name,module in (("unet",pipe.unet),("vae_encoder",pipe.vae.encoder),("vae_decoder",pipe.vae.decoder)):
        def before(_module,_inputs,kind=name):
            ledger=result["calls"].setdefault(result["stage"],{})
            ledger.setdefault(kind,dict(attempted=0,completed=0))["attempted"]+=1
            result["model_calls_executed"]=True
            result["real_model_executed"]=result["model_loaded"]
            persist()
        def after(_module,_inputs,_output,kind=name):
            result["calls"][result["stage"]][kind]["completed"]+=1
            persist()
        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    return handles

def evaluate_readouts(blind_path, config, codec):
    """Truth enters only after persisted readout bytes are reopened."""
    blind=json.loads(Path(blind_path).read_text())
    targets=dict(REGISTERED=config["official_config"]["message"],WRONG_MESSAGE=config["wrong_message"])
    rows={}
    for identity,read in blind["reads"].items():
        for target_id,message in targets.items():
            row=dict(status="MISSING_READOUT",image_id=read["image_id"],key_id=read["key_id"],
                target_id=target_id,bit_errors=None,ber=None,bit_accuracy=None,exact_bits=None)
            if read["status"]=="READ":
                actual=read["decoded_bits"];truth=codec.message_to_bits(message)
                if len(actual)!=32 or len(truth)!=32:
                    raise ValueError("fixed bit denominator mismatch")
                errors=sum(a!=b for a,b in zip(actual,truth,strict=True))
                row.update(status="EVALUATED",bit_errors=errors,ber=errors/32,
                           bit_accuracy=1-errors/32,exact_bits=errors==0)
            rows[f"{identity}/{target_id}"]=row
    if len(rows)!=8: raise ValueError("fixed evaluation denominator mismatch")
    return rows


def run_fixed(output,upstream_root,*,pipeline_factory=load_real_pipeline,device=None):
    """Only the dependency factory/device can be replaced by CPU test fixtures."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    result=initial_result(output);result_path=output/"result.json"
    save_result(result_path,result)
    started=time.perf_counter();pipe=None;handles=[]
    try:
        result["upstream"]=verify_source(upstream_root)
        config=load_config(upstream_root)
        result["environment"]=environment_receipt()
        result["layout"]=layout_receipt(config["official_config"]["secret_key"],config["wrong_key"])
        GrowConfig,Watermarker,codec=load_official(upstream_root)
        result["stage"]="MODEL_ASSETS"
        save_result(result_path,result)
        def asset_record(value):
            result["assets"]=dict(value);save_result(result_path,result)
        pipe=pipeline_factory(config,asset_record)
        result["model_loaded"]=(pipeline_factory is load_real_pipeline)
        result["execution_kind"]="REAL_MODEL" if result["model_loaded"] else "CPU_FAKE_COMPONENTS_OFFICIAL_FUNCTIONS"
        handles=observe_calls(pipe,result,lambda:save_result(result_path,result))
        device=device or str(getattr(pipe,"device",(result.get("assets") or {}).get("device","cpu")))
        result["runtime_device"]=device
        writer_config=dict(config["official_config"]);writer_config["device"]=device
        writer=Watermarker(GrowConfig(**writer_config),pipe=pipe)
        for arm in ARMS:
            row=result["images"][arm]
            result["stage"]=f"GENERATE_{arm}";row["status"]="GENERATING";save_result(result_path,result)
            try:
                function=writer.generate_normal if arm=="OFF" else writer.generate_with_watermark
                image=function(config["case"]["prompt"],seed=42)
                image.save(row["path"],format="PNG")
                del image
                row.update(status="SAVED",sha256=sha256(row["path"]),bytes=Path(row["path"]).stat().st_size,
                           official_function=function.__name__)
            except Exception as exc:
                row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}")
                result["failures"].append(dict(stage=result["stage"],error=row["error"]))
            save_result(result_path,result)
        del writer
        blind_path=output/"blind_readouts.json"
        key_values=dict(CORRECT=config["official_config"]["secret_key"],WRONG=config["wrong_key"])
        for identity,row in result["reads"].items():
            arm,key_id=row["image_id"],row["key_id"]
            result["stage"]=f"BLIND_READ_{arm}_{key_id}"
            if result["images"][arm]["status"]!="SAVED":
                row.update(status="MISSING_IMAGE",error="generation did not save a PNG")
            else:
                row["status"]="READING";save_result(result_path,result)
                try:
                    public=PublicReaderConfig(device=device,dtype="float32",secret_key=key_values[key_id])
                    decoded=read_saved_png(result["images"][arm]["path"],public,pipe,upstream_root)
                    if decoded["png_sha256"]!=result["images"][arm]["sha256"]:
                        raise RuntimeError("saved PNG identity changed")
                    row.update(decoded)
                except Exception as exc:
                    row.update(status="FAILED",error=f"{type(exc).__name__}: {exc}")
                    result["failures"].append(dict(stage=result["stage"],error=row["error"]))
            atomic_json(blind_path,dict(reads=result["reads"],public_payload_bits=32,
                evaluation_performed=False,no_truth_inputs=True))
            save_result(result_path,result)
        # All four fixed read rows now exist on disk, even after read failures.
        result["stage"]="SEPARATE_TRUTH_EVALUATION"
        result["blind_readouts_sha256_before_evaluation"]=sha256(blind_path)
        save_result(result_path,result)
        result["evaluations"]=evaluate_readouts(blind_path,config,codec)
        if sha256(blind_path)!=result["blind_readouts_sha256_before_evaluation"]:
            raise RuntimeError("evaluation changed blind evidence")
        if all(row["status"]=="SAVED" for row in result["images"].values()):
            from PIL import Image
            import numpy as np
            with Image.open(output/"OFF.png") as a,Image.open(output/"GROW.png") as b:
                delta=np.asarray(a.convert("RGB"),dtype=np.float64)/255-np.asarray(b.convert("RGB"),dtype=np.float64)/255
            mse=float(np.mean(delta*delta))
            result["quality"]=dict(paired_png_rmse=math.sqrt(mse),
                psnr_db=-10*math.log10(mse) if mse else None,
                identical_png_pixels=mse==0,diagnostic_only=True,no_quality_pass_threshold=True)
        result["status"]="FIXED_REFERENCE_COMPLETE" if all(row["status"]=="READ" for row in result["reads"].values()) else "INCOMPLETE"
        result["stage"]="FINISHED"
    except Exception as exc:
        reason=f"{type(exc).__name__}: {exc}"
        asset=result.get("assets") or {}
        result["status"]="ASSET_ACCESS_REQUIRED" if asset.get("status")=="ASSET_ACCESS_REQUIRED" else "ENGINEERING_FAILURE"
        result["failures"].append(dict(stage=result["stage"],error=reason,traceback=traceback.format_exc()))
        preserve_unfinished(result,reason)
    finally:
        for handle in handles: handle.remove()
        if pipe is not None:
            del pipe
        result["elapsed_seconds"]=time.perf_counter()-started
        save_result(result_path,result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run_fixed(args.output,args.upstream_root)
    print(json.dumps(dict(status=result["status"],counts=result["counts"])))
    if result["status"]!="FIXED_REFERENCE_COMPLETE": raise SystemExit(1)


if __name__=="__main__": main()
