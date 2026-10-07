"""Two isolated preparation workers, never a receiver or historical runner."""
from pathlib import Path
import gc,gzip,hashlib,json,os
from runtime.wan import video_trajectory_conditional_joint_v1 as backend

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);b=(json.dumps(value,indent=2,allow_nan=False)+"\n").encode()
    if p.suffix==".gz":b=gzip.compress(b,mtime=0)
    t=p.with_suffix(p.suffix+".tmp");t.write_bytes(b);os.replace(t,p)
    return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
class Store:
    def __init__(self,output,cfg,phase):
        self.output=Path(output);self.path=self.output/"source_preparation.json";self.cfg=cfg
        ident=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        if phase=="source":
            if self.path.exists():raise FileExistsError("single preparation attempt")
            self.data=dict(status="RUNNING",stage="SOURCE",config_content_sha256=ident,source_calls={k:dict(attempted=0,completed=0) for k in cfg["source_preparation_planned_calls"]},calls={k:dict(attempted=0,completed=0) for k in cfg["m05_planned_calls"]},steps=[dict(index=i,status="PENDING") for i in range(50)],writer=dict(status="PENDING"),rasters={k:dict(status="PENDING") for k in ("P1","M05")},transport={k:dict(status="PENDING",events={}) for k in ("P1","M05")},received_sources={},failures=[])
        else:
            self.data=json.loads(self.path.read_text())
            if self.data["status"]!="SOURCE_READY" or self.data["config_content_sha256"]!=ident:raise ValueError("source identity/stage")
        self.save()
    def save(self):dump(self.path,self.data)
    def count(self,table,name,done):self.data[table][name]["completed" if done else "attempted"]+=1;self.save()
    def call(self,table,name,fn):
        self.count(table,name,False);value=fn();self.count(table,name,True);return value
    def event(self,condition,stage,row):
        self.data["transport"][condition]["events"][stage]=row
        if stage=="rgb24" and row["status"]=="SAVED":self.data["transport"][condition]["status"]="COMPLETE"
        self.save()
    def failure(self,exc):
        reason=f"{type(exc).__name__}: {exc}";self.data["failures"].append(dict(stage=self.data["stage"],error=reason));self.data["status"]="FAILED"
        def settle(x):
            if isinstance(x,dict):
                if x.get("status") in ("PENDING","RUNNING","ENCODING"):x.update(status="NOT_COMPLETED",error=reason)
                for v in x.values():settle(v)
            elif isinstance(x,list):
                for v in x:settle(v)
        settle(self.data);self.save()
    def integrity(self,phase):
        pairs=[("source_calls","source_preparation_planned_calls")]+([("calls","m05_planned_calls")] if phase=="matched" else [])
        self.data["call_integrity"]={table:{k:dict(expected=n,actual=self.data[table][k],match=self.data[table][k]==dict(attempted=n,completed=n)) for k,n in self.cfg[plan].items()} for table,plan in pairs};self.save()
        return all(x["match"] for table in self.data["call_integrity"].values() for x in table.values())

def source_worker(store,cfg):
    import torch
    device,dtype=backend.execution_device_dtype();pipe=initial=prompt=negative=terminal=native=rgb=None
    call=lambda name,fn:store.call("source_calls",name,fn)
    try:
        pipe,initial,prompt,negative,input_dtype=call("generation_load",lambda:backend.generation.prepare_generation(cfg,load_vae=False,device=device,model_dtype=dtype))
        before=backend.trajectory.fingerprint(initial)
        def record(row):store.data["steps"][row["index"]]=dict(row,status="COMPLETE");store.save()
        terminal,receipt=call("trajectory",lambda:backend.run_trajectory(pipe,initial,pipe.scheduler,prompt,negative,input_dtype,cfg["key"],backend.layout.message_bits(cfg["message"]),lambda name,done:store.count("source_calls",name,done),record))
        if backend.trajectory.fingerprint(initial)!=before:raise ValueError("initial source state changed")
        store.data.update(initial_sha256=before,trajectory_receipt=receipt);store.save()
        pipe=initial=prompt=negative=None;backend.release()
        native=call("wan_vae_load",lambda:backend.generation.load_frozen_vae(cfg,device=device))
        rgb=call("wan_decode",lambda:backend.vae.decode_normalized_latent(native,terminal.to(next(native.parameters()).device)))
        raster=call("raster_save",lambda:backend.media.save_raster(backend.vae.quantize_rgb8_no_codec(rgb),store.output/"P0/source.rgb8"))
        store.data["source_protocol"]={**raster,"shape":[181,320,512,3]};store.save()
    finally:
        pipe=initial=prompt=negative=terminal=native=rgb=None;backend.release()

def matched_worker(store,cfg,backend_type=backend.FramewiseBackend,input_type=backend.Inputs,media=backend.media):
    model=None;source=z=written=rgb=None
    call=lambda name,fn:store.call("calls",name,fn)
    try:
        source=call("source_read",lambda:input_type.read(store.data["source_protocol"]))
        model=call("framewise_vae_load",lambda:backend_type(cfg))
        z=call("framewise_writer_encode",lambda:model.encode(source));before=hashlib.sha256(z.tobytes()).hexdigest()
        written,receipt=call("writer_sync",lambda:model.write(z.copy(),cfg["key"]))
        if hashlib.sha256(z.tobytes()).hexdigest()!=before or receipt["target_margin"]!=.5:raise ValueError("shared original/target .5")
        ref=dump(store.output/"writer_receipt.M05.json.gz",receipt)
        store.data["writer"]=dict(status="SAVED",**ref,target_margin=.5,row_count=len(receipt["rows"]),original_latent_sha256=before);store.save()
        for condition,value in (("P1",z),("M05",written)):
            store.data["stage"]=condition+"_DECODE";store.save()
            rgb=call("framewise_writer_decode",lambda:model.decode(value))
            store.data["rasters"][condition]=call("raster_save",lambda:media.save_raster(rgb,store.output/condition/"source.rgb8"));store.save()
    finally:
        source=z=written=rgb=None
        if model is not None:model.close()
    for condition in ("P1","M05"):
        store.data["stage"]=condition+"_MP4";store.save();r=store.data["rasters"][condition]
        media.mp4_roundtrip(r["path"],r["sha256"],store.output/condition/"source.mp4",store.output/condition/"received.rgb8",count=lambda name,done:store.count("calls",name,done),event=lambda stage,row,c=condition:store.event(c,stage,row))
        store.data["received_sources"][condition]=store.data["transport"][condition]["events"]["rgb24"].copy();store.save()

def run_phase(output,phase,cfg,source_fn=source_worker,matched_fn=matched_worker):
    store=Store(output,cfg,phase)
    try:
        if phase=="source":source_fn(store,cfg);store.data["status"]="SOURCE_READY"
        elif phase=="matched":matched_fn(store,cfg);store.data["status"]="COMPLETE"
        else:raise ValueError("source/matched")
        if not store.integrity(phase):raise RuntimeError("preparation call budget mismatch")
        store.save();return store.data
    except BaseException as exc:store.failure(exc);store.integrity(phase);raise
