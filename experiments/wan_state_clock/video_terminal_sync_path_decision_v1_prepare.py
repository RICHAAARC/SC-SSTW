"""Separate source and matched P1/M05 worker processes; no receiver selection."""
from pathlib import Path
import argparse,gc,json,hashlib,sys
import numpy as np
from experiments.wan_state_clock import video_trajectory_payload_gt_v1_run as prior
from main.tube_state import video_trajectory_payload_gt_v1 as writer
from runtime.wan import framewise_autoencoder_kl as framewise,rgb8_source,vae
from runtime.wan import video_local_fourier_rm_same_raster as media
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_terminal_sync_path_decision_v1_preparation.json"
class Store:
    profile="G"
    def __init__(self,output,cfg,phase):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True);self.path=self.output/"source_preparation.json";self.cfg=cfg
        self.identity=prior.sha256_file(CONFIG)
        if phase=="source":
            if self.path.exists():raise FileExistsError("single source attempt only")
            self.data=dict(status="RUNNING",stage="INITIALIZED",config_sha256=self.identity,source_calls={k:dict(attempted=0,completed=0) for k in cfg["source_preparation_planned_calls"]},calls={k:dict(attempted=0,completed=0) for k in cfg["m05_planned_calls"]},source_preparation=dict(status="PENDING",arm="PAYLOAD_MULTI",steps=[dict(index=i,status="PENDING") for i in range(50)],rgb_path=str(self.output/"prepared_source/source.rgb8"),protocol=cfg["generation"]),writer=dict(status="PENDING"),rasters={k:dict(status="PENDING") for k in ("P1","M05")},transport={k:dict(status="PENDING",events={}) for k in ("P1","M05")},received_sources={},failures=[],actual_generation_calls=False)
        else:
            self.data=json.loads(self.path.read_text())
            if self.data["status"]!="SOURCE_READY" or self.data["config_sha256"]!=self.identity:raise ValueError("source identity/stage")
        self.save()
    def save(self):prior.dump_json(self.path,self.data)
    def count(self,k,done):self.data["calls"][k]["completed" if done else "attempted"]+=1;self.save()
    def call(self,k,fn):
        self.count(k,False);x=fn();self.count(k,True);return x
    def event(self,condition,stage,row):
        self.data["transport"][condition]["events"][stage]=row
        if stage=="rgb24" and row["status"]=="SAVED":self.data["transport"][condition]["status"]="COMPLETE"
        self.save()
    def fail(self,exc):
        reason=f"{type(exc).__name__}: {exc}";self.data["status"]="FAILED";self.data["failures"].append(dict(stage=self.data["stage"],error=reason))
        def finish(x):
            if isinstance(x,dict):
                if x.get("status") in ("RUNNING","PENDING"):x.update(status="NOT_COMPLETED",error=reason)
                for v in x.values():finish(v)
            elif isinstance(x,list):
                for v in x:finish(v)
        finish(self.data);self.save()
    def integrity(self):
        for table,plan in (("source_calls","source_preparation_planned_calls"),("calls","m05_planned_calls")):
            self.data[table+"_integrity"]={k:dict(expected=n,actual=self.data[table][k],match=self.data[table][k]==dict(attempted=n,completed=n)) for k,n in self.cfg[plan].items()}
        self.save()
class Backend:
    def __init__(self,cfg):self.cfg=cfg;self.model=None
    def read(self,s):return rgb8_source.read_rgb8_source(s["path"],expected_sha256=s["sha256"],shape=tuple(s["shape"]))
    def load(self):
        import torch
        self.model=framewise.load_frozen_framewise_vae(device="cuda" if torch.cuda.is_available() else "cpu")
    def encode(self,rgb):return framewise.encode_rgb_frames(self.model,rgb.float().div(255),batch_frames=8).numpy()
    def write(self,z):return writer.write_condition(z,self.cfg["key"],"M05_FRAMEWISE_SYNC",writer.public_protocol("G"))
    def decode(self,z):
        import torch
        return vae.quantize_rgb8_no_codec(framewise.decode_rgb_frames(self.model,torch.from_numpy(z.copy()),batch_frames=8))
    def save(self,rgb,p):return media.save_raster(rgb,p)
    def transport(self,raster,root,count,event):
        media.mp4_roundtrip(raster["path"],raster["sha256"],root/"source.mp4",root/"received.rgb8",count=count,event=event)
    def close(self):
        import torch
        self.model=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
def matched_worker(store,cfg,backend_type=Backend):
    backend=backend_type(cfg);z=written=None
    try:
        source=store.call("source_read",lambda:backend.read(store.data["source_protocol"]))
        store.call("framewise_vae_load",backend.load)
        z=store.call("framewise_writer_encode",lambda:backend.encode(source));before=hashlib.sha256(z.tobytes()).hexdigest()
        written,receipt=store.call("writer_sync",lambda:backend.write(z.copy()))
        if hashlib.sha256(z.tobytes()).hexdigest()!=before or receipt["target_margin"]!=0.5:raise ValueError("shared original/target.5 identity")
        p=store.output/"writer_receipt.M05.json.gz";prior.dump_gzip_json(p,receipt)
        store.data["writer"]=dict(status="SAVED",path=str(p),sha256=prior.sha256_file(p),row_count=len(receipt["rows"]),target_margin=0.5,original_latent_sha256=before);store.save()
        for condition,value in (("P1",z),("M05",written)):
            store.data["stage"]=condition+"_DECODE";store.save()
            rgb=store.call("framewise_writer_decode",lambda v=value:backend.decode(v))
            store.data["rasters"][condition]=store.call("raster_save",lambda:backend.save(rgb,store.output/condition/"source.rgb8"));store.save()
        source=z=written=rgb=None
    finally:backend.close()
    for condition in ("P1","M05"):
        store.data["stage"]=condition+"_MP4";store.save()
        backend.transport(store.data["rasters"][condition],store.output/condition,store.count,lambda stage,row,c=condition:store.event(c,stage,row))
        row=store.data["transport"][condition]["events"]["rgb24"]
        store.data["received_sources"][condition]={k:row[k] for k in ("path","sha256","shape","bytes")};store.save()
def run_phase(output,phase,*,cfg=None,source_fn=prior.source_worker,matched_fn=matched_worker):
    cfg=json.loads(CONFIG.read_text()) if cfg is None else cfg;store=Store(output,cfg,phase)
    try:
        if phase=="source":source_fn(store,cfg);store.data["status"]="SOURCE_READY"
        elif phase=="matched":matched_fn(store,cfg);store.data["status"]="COMPLETE"
        else:raise ValueError("source/matched only")
        store.integrity()
        fields=["source_calls_integrity"]+(["calls_integrity"] if phase=="matched" else [])
        if not all(x["match"] for field in fields for x in store.data[field].values()):raise RuntimeError("preparation budget mismatch")
        store.save();return store.data
    except BaseException as exc:store.fail(exc);store.integrity();raise
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path);p.add_argument("--phase",choices=("source","matched"),required=True);a=p.parse_args();run_phase(a.output,a.phase)
