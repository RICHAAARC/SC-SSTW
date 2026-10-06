"""Fixed source then M05 workers. Neither worker performs any receiver selection."""
from pathlib import Path
import argparse,gc,json,sys,importlib.metadata
import numpy as np
from experiments.wan_state_clock import video_trajectory_payload_gt_v1_run as generation
from main.tube_state import video_trajectory_payload_gt_v1 as writer
from runtime.wan import framewise_autoencoder_kl as framewise
from runtime.wan import rgb8_source,vae
from runtime.wan import video_local_fourier_rm_same_raster as media

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_receiver_independent_source_v1_preparation.json"
def load_config():return json.loads(CONFIG.read_text())
def sha(p):return generation.sha256_file(p)
def dump(p,x):generation.dump_json(p,x)

class Store:
    profile="G"
    def __init__(self,output,cfg,phase):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True);self.path=self.output/"source_preparation.json";self.cfg=cfg
        if phase=="source":
            if self.path.exists():raise FileExistsError("one fixed source attempt; existing preparation receipt")
            self.data=dict(status="RUNNING",stage="INITIALIZED",config_sha256=sha(CONFIG),
                source_calls={k:dict(attempted=0,completed=0) for k in cfg["source_preparation_planned_calls"]},
                calls={k:dict(attempted=0,completed=0) for k in cfg["m05_planned_calls"]},
                source_preparation=dict(status="PENDING",arm="PAYLOAD_MULTI",steps=[dict(index=i,status="PENDING") for i in range(50)],
                    rgb_path=str(self.output/"prepared_source/source.rgb8"),protocol=cfg["generation"]),
                environment={},
                writer=dict(status="PENDING",condition="M05_FRAMEWISE_SYNC"),raster=dict(status="PENDING"),
                transport=dict(status="PENDING",events={}),failures=[],actual_generation_calls=False)
            self.save()
        else:
            self.data=json.loads(self.path.read_text())
            if self.data["status"]!="SOURCE_READY":raise RuntimeError("source worker must have finished before M05")
            if self.data["config_sha256"]!=sha(CONFIG):raise RuntimeError("preparation protocol changed")
    def save(self):dump(self.path,self.data)
    def count(self,name,completed):
        self.data["calls"][name]["completed" if completed else "attempted"]+=1;self.save()
    def call(self,name,fn):
        self.count(name,False);v=fn();self.count(name,True);return v
    def event(self,stage,row):
        self.data["transport"]["events"][stage]=row
        if stage=="rgb24" and row["status"]=="SAVED":self.data["transport"]["status"]="COMPLETE"
        elif row["status"]=="FAILED":self.data["transport"]["status"]="FAILED"
        self.save()
    def fail(self,exc):
        reason=f"{type(exc).__name__}: {exc}"
        self.data["failures"].append(dict(stage=self.data["stage"],error=reason));self.data["status"]="FAILED"
        for row in [self.data["source_preparation"],self.data["writer"],self.data["raster"],self.data["transport"],*self.data["transport"]["events"].values()]:
            if row["status"] in ("PENDING","RUNNING"):row.update(status="NOT_COMPLETED",error=reason)
        for row in self.data["source_preparation"]["steps"]:
            if row["status"]=="PENDING":row.update(status="NOT_COMPLETED",error=reason)
        self.save()
    def integrity(self):
        for table,plan,out in [("source_calls","source_preparation_planned_calls","generation_call_integrity"),("calls","m05_planned_calls","m05_call_integrity")]:
            self.data[out]={k:dict(expected=v,actual=self.data[table][k],match=self.data[table][k]==dict(attempted=v,completed=v)) for k,v in self.cfg[plan].items()}
        self.save()

def m05_worker(store,cfg):
    """Exactly one original encoded copy, M05 projection, decode and full MP4 chain."""
    import torch
    source=encoded=written=decoded=None;frame_vae=None
    try:
        store.data["stage"]="M05_SOURCE_READ";store.save()
        spec=store.data["source_protocol"]
        source=store.call("source_read",lambda:rgb8_source.read_rgb8_source(spec["path"],expected_sha256=spec["sha256"],shape=tuple(spec["shape"])))
        store.data["stage"]="M05_ENCODE_WRITE_DECODE";store.save()
        frame_vae=store.call("framewise_vae_load",lambda:framewise.load_frozen_framewise_vae(device="cuda" if torch.cuda.is_available() else "cpu"))
        encoded=store.call("framewise_writer_encode",lambda:framewise.encode_rgb_frames(frame_vae,source.float().div(255.0),batch_frames=cfg["framewise_vae"]["batch_frames"]))
        original=encoded.numpy().copy();before=generation.hashlib.sha256(original.tobytes()).hexdigest()
        written,receipt=store.call("writer_sync",lambda:writer.write_condition(original.copy(),cfg["key"],"M05_FRAMEWISE_SYNC",writer.public_protocol("G")))
        if generation.hashlib.sha256(original.tobytes()).hexdigest()!=before:raise ValueError("encoded writer source mutated")
        path=store.output/"writer_receipt.M05_FRAMEWISE_SYNC.json.gz";generation.dump_gzip_json(path,receipt)
        store.data["writer"]=dict(status="SAVED",condition="M05_FRAMEWISE_SYNC",path=str(path),sha256=sha(path),
            row_count=len(receipt["rows"]),target_margin=receipt["target_margin"],input_scaled_latent_sha256=before,independent_copy=True)
        store.save()
        decoded=store.call("framewise_writer_decode",lambda:framewise.decode_rgb_frames(frame_vae,torch.from_numpy(written.copy()),batch_frames=cfg["framewise_vae"]["batch_frames"]))
        raster=store.call("raster_save",lambda:media.save_raster(vae.quantize_rgb8_no_codec(decoded),store.output/"M05_FRAMEWISE_SYNC/source.rgb8"))
        store.data["raster"]=raster;store.save()
        source=encoded=written=decoded=original=None
        frame_vae=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        store.data["stage"]="M05_MP4";store.save()
        received=media.mp4_roundtrip(raster["path"],raster["sha256"],store.output/"M05_FRAMEWISE_SYNC/source.mp4",
            store.output/"M05_FRAMEWISE_SYNC/received.rgb8",count=store.count,event=store.event)
        row=store.data["transport"]["events"]["rgb24"]
        store.data["received_source"]=dict(path=row["path"],sha256=row["sha256"],shape=row["shape"],
            bytes=row["bytes"],frames=int(received.shape[0]),dtype="uint8")
        store.save()
    finally:
        source=encoded=written=decoded=frame_vae=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

def run_phase(output,phase,*,cfg=None,source_fn=generation.source_worker,m05_fn=m05_worker):
    cfg=load_config() if cfg is None else cfg;store=Store(output,cfg,phase)
    try:
        if phase=="source":
            source_fn(store,cfg)
            store.data["status"]="SOURCE_READY"
        elif phase=="m05":
            m05_fn(store,cfg);store.data["status"]="COMPLETE"
        else:raise ValueError("fixed source/m05 phase")
        store.integrity()
        needed=["generation_call_integrity"]+(["m05_call_integrity"] if phase=="m05" else [])
        if not all(x["match"] for k in needed for x in store.data[k].values()):raise RuntimeError("preparation call budget incomplete")
        store.save()
        return store.data
    except BaseException as exc:
        store.fail(exc);store.integrity();raise

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--phase",choices=("source","m05"),required=True)
    a=p.parse_args();run_phase(a.output,a.phase)
if __name__=="__main__":main()
