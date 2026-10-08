"""Isolated source workers for trajectory attribution V1; no receiver decisions."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
from main.tube_state import grow_video_reference as payload
from runtime.wan import video_trajectory_attribution_v1 as runtime

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    temp=path.with_suffix(path.suffix+".tmp");temp.write_bytes(raw);os.replace(temp,path)
    return dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

class Store:
    def __init__(self,output,cfg,source_id):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.path=self.output/"source_preparation.json";self.cfg=cfg;self.source_id=source_id
        if self.path.exists():self.data=json.loads(self.path.read_text())
        else:
            source=cfg["sources"][source_id]
            self.data=dict(status="RUNNING",stage="INITIALIZED",source_id=source_id,source_role=source["role"],
                prompt=source["prompt"],seed=source["seed"],trajectories={a:dict(status="PENDING") for a in ("OFF","A","B")},
                rasters={},received={},calls={},events={},failures=[])
            self.save()
    def save(self):dump(self.path,self.data)
    def count(self,name,done):
        row=self.data["calls"].setdefault(name,dict(attempted=0,completed=0))
        row["completed" if done else "attempted"]+=1;self.save()
    def call(self,name,fn):
        self.count(name,False);value=fn();self.count(name,True);return value
    def event(self,condition,stage,row):
        self.data["events"].setdefault(condition,{})[stage]=row;self.save()
    def failure(self,exc):
        self.data["failures"].append(dict(stage=self.data["stage"],error=f"{type(exc).__name__}: {exc}"))
        self.data["status"]="FAILED";self.save()

def generation_config(cfg,source_id):
    source=cfg["sources"][source_id]
    return dict(model=cfg["model"],generation=dict(cfg["generation_common"],prompt=source["prompt"],seed=source["seed"]))

def generation_phase(store):
    import torch
    cfg=generation_config(store.cfg,store.source_id)
    device,dtype=runtime.execution_device_dtype();pipe=initial=prompt=negative=None
    try:
        store.data["stage"]="GENERATION_LOAD";store.save()
        pipe,initial,prompt,negative,input_dtype=store.call("generation_load",lambda:runtime.generation.prepare_generation(cfg,load_vae=False,device=device,model_dtype=dtype))
        pristine=copy.deepcopy(pipe.scheduler)
        initial_sha=runtime.trajectory.fingerprint(initial);history_sha=runtime.trajectory.fingerprint(vars(pristine))
        for arm in ("OFF","A","B"):
            row=store.data["trajectories"][arm];row.update(status="RUNNING",steps=[]);store.save()
            message=store.cfg["trajectories"][arm]["message"]
            bits=[0]*32 if message is None else payload.message_bits(message)
            def record(value,r=row):r["steps"].append(value);store.save()
            terminal,receipt=runtime.run_native_arm(pipe,initial,pristine,prompt,negative,input_dtype,arm,
                store.cfg["keys"]["K0"],bits,store.count,record)
            if runtime.trajectory.fingerprint(initial)!=initial_sha or runtime.trajectory.fingerprint(vars(pristine))!=history_sha:
                raise RuntimeError("shared initial/pristine history changed")
            path=store.output/"terminals"/(arm+".pt");path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(terminal,path)
            row.update(status="COMPLETE",path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),receipt=receipt)
            store.save();terminal=None
        store.data["generation_identity"]=dict(initial_sha256=initial_sha,pristine_history_sha256=history_sha)
        store.data["stage"]="GENERATED";store.save()
    finally:
        pipe=initial=prompt=negative=None;runtime.release()

def native_decode_phase(store):
    import torch
    cfg=generation_config(store.cfg,store.source_id);device,_=runtime.execution_device_dtype();model=None
    try:
        store.data["stage"]="NATIVE_DECODE_LOAD";store.save()
        model=store.call("wan_vae_load",lambda:runtime.generation.load_frozen_vae(cfg,device=device))
        for arm,row in store.data["trajectories"].items():
            if row["status"]!="COMPLETE":raise ValueError("trajectory unavailable "+arm)
            path=Path(row["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row["sha256"]:raise ValueError("terminal identity "+arm)
            terminal=torch.load(path,map_location="cpu",weights_only=True)
            rgb=store.call("wan_decode",lambda:runtime.vae.decode_normalized_latent(model,terminal.to(next(model.parameters()).device)))
            raster=store.call("raster_save",lambda:runtime.media.save_raster(runtime.vae.quantize_rgb8_no_codec(rgb),store.output/arm/"P0.rgb8"))
            store.data["rasters"][arm+"_P0"]={**raster,"shape":[181,320,512,3]};store.save()
            terminal=rgb=None
        store.data["stage"]="NATIVE_DECODED";store.save()
    finally:
        model=None;runtime.release()

def framewise_media_phase(store,backend_type=runtime.FramewiseBackend,input_type=runtime.Inputs,media=runtime.media):
    model=None
    try:
        store.data["stage"]="FRAMEWISE_LOAD";store.save();model=store.call("framewise_vae_load",lambda:backend_type(store.cfg))
        for arm in ("OFF","A","B"):
            source=input_type.read(store.data["rasters"][arm+"_P0"])
            z=store.call("framewise_writer_encode",lambda:model.encode(source))
            outputs=[]
            if arm=="OFF":outputs=[("OFF",z)]
            elif arm=="A":
                outputs.append(("A_P1",z))
                written,receipt=store.call("writer_sync",lambda:model.write(z.copy(),store.cfg["keys"]["K0"]))
                store.data["rasters"]["A_M05_WRITER"]=dump(store.output/"A"/"writer_receipt.json",receipt)
                outputs.append(("A_M05",written))
            else:
                written,receipt=store.call("writer_sync",lambda:model.write(z.copy(),store.cfg["keys"]["K0"]))
                store.data["rasters"]["B_M05_WRITER"]=dump(store.output/"B"/"writer_receipt.json",receipt)
                outputs.append(("B_M05",written))
            for condition,value in outputs:
                q8=store.call("framewise_writer_decode",lambda v=value:model.decode(v))
                raster=store.call("raster_save",lambda q=q8,c=condition:media.save_raster(q,store.output/c/"source.rgb8"))
                store.data["rasters"][condition]={**raster,"shape":[181,320,512,3]};store.save()
            source=z=None
        if model is not None:model.close();model=None
        for condition in ("OFF","A_P1","A_M05","B_M05"):
            raster=store.data["rasters"][condition]
            media.mp4_roundtrip(raster["path"],raster["sha256"],store.output/condition/"source.mp4",
                store.output/condition/"received.rgb8",count=store.count,
                event=lambda stage,row,c=condition:store.event(c,stage,row))
            received=store.data["events"][condition]["rgb24"]
            if received["status"]!="SAVED":raise RuntimeError("received RGB missing "+condition)
            store.data["received"][condition]={**received,"shape":[181,320,512,3]};store.save()
        store.data.update(status="COMPLETE",stage="FINISHED");store.save()
    finally:
        if model is not None:model.close()

def run_phase(output,phase,cfg,source_id):
    store=Store(output,cfg,source_id)
    try:
        if phase=="generation":generation_phase(store)
        elif phase=="native_decode":native_decode_phase(store)
        elif phase=="framewise_media":framewise_media_phase(store)
        else:raise ValueError("fixed worker phase")
        return store.data
    except BaseException as exc:
        store.failure(exc);raise
