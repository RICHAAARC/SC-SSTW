"""Explicit-input fixed M05 writer/MP4/FULL receiver engineering entry.

It retains target .5 and original fixed media math, without claiming the old
four-condition experiment was rerun by this reduced entry.
"""
from pathlib import Path
import argparse,copy,json,sys
from main.tube_state import grow_video_reference as payload
from main.tube_state import video_trajectory_payload_framewise_sync_v1 as method
from runtime.wan.video_trajectory_payload_framewise_sync_v1 import M05Backend
from runtime.wan.video_trajectory_receiver_estimated_align_v1 import WanBackend
from experiments.wan_state_clock import receiver_records as records
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/"experiments/wan_state_clock/configs/video_trajectory_payload_framewise_sync_m05_v1.json"

def load_config(path=CONFIG):return records.public_config(path)
class Store(records.Store):
    def __init__(self,output,cfg):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.cfg=cfg
        self.data=dict(status="RUNNING",stage="INITIALIZED",**records.source_metadata(ROOT,cfg,CONFIG),
            fixed_denominator=dict(source=1,condition=1,sync_reads=2,payload_reads=2,final_bits=64),
            calls={k:dict(attempted=0,completed=0) for k in cfg["planned_calls"]},failures=[],
            writer=dict(status="PENDING"),media={},sync_reads={k:dict(status="PENDING") for k,_ in records.keys(cfg)},
            payload_reads={k:dict(status="PENDING") for k,_ in records.keys(cfg)},posthoc={k:dict(status="PENDING",planned_bits=32) for k,_ in records.keys(cfg)})
        self.save()
    def count(self,name,completed):
        self.data["calls"][name]["completed" if completed else "attempted"]+=1;self.save()
    def event(self,name,row):self.data["media"][name]=row;self.save()

def run(output,*,cfg=None,writer_type=M05Backend,wan_type=WanBackend):
    cfg=copy.deepcopy(load_config() if cfg is None else cfg);store=Store(output,cfg);writer=wan=None;received=None;abort=None
    try:
        source=store.call("source_read",lambda:writer_type.read_source(cfg["source"]))
        store.data["source_input"]=dict(cfg["source"],verified=True);store.save()
        writer=store.call("framewise_load",lambda:writer_type(cfg["framewise_model"]))
        z=store.call("writer_encode",lambda:writer.encode(source));del source
        changed,receipt=store.call("writer_apply",lambda:writer.write(z,cfg["key"]));del z
        saved=records.dump(store.output/"writer_receipt.json.gz",receipt);store.data["writer"]=dict(status="WRITTEN",target=0.5,**saved);store.save()
        rgb=store.call("writer_decode",lambda:writer.decode(changed));del changed
        received=writer.transport(rgb,store.output,store.count,store.event);del rgb
        fresh=store.call("framewise_receiver_encode",lambda:writer.encode(received))
        for label,key in records.keys(cfg):
            try:
                row=store.call("sync_read",lambda k=key:writer.score(fresh,k))
                saved=records.dump(store.output/"sync"/(label+".json.gz"),dict(readout=row,truth_inputs=False))
                store.data["sync_reads"][label]=dict(status="SAVED",**saved)
            except Exception as exc:store.failure(label+"/SYNC",exc)
        del fresh
    except BaseException as exc:
        # The codec records RUNNING before subprocess work. Preserve completed
        # receipts and close only unfinished events, including KeyboardInterrupt.
        for row in store.data["media"].values():
            if row.get("status") in ("PENDING","RUNNING"):
                row.update(status="INTERRUPTED" if not isinstance(exc,Exception) else "FAILED",error=f"{type(exc).__name__}: {exc}")
        store.failure("M05_PREPARATION",exc)
        if not isinstance(exc,Exception):abort=exc
    finally:
        if writer is not None:
            try:writer.close()
            except BaseException as exc:
                store.failure("FRAMEWISE_RELEASE",exc)
                if abort is None and not isinstance(exc,Exception):abort=exc
    try:
        if received is not None and abort is None:
            wan=store.call("wan_load",lambda:wan_type(cfg["model"]))
            latent=store.call("wan_encode",lambda:wan.encode(received))
            for label,key in records.keys(cfg):
                try:
                    detail=store.call("payload_read",lambda k=key:wan.read(latent,k,181))
                    saved=records.dump(store.output/"payload"/(label+".json.gz"),detail)
                    store.data["payload_reads"][label]=dict(status="READ",**saved,decoded_bits=detail["original_readout"]["decoded_bits"],votes=detail["original_readout"]["votes"])
                except Exception as exc:store.failure(label+"/PAYLOAD",exc)
            del latent
    except BaseException as exc:
        store.failure("PAYLOAD",exc)
        if not isinstance(exc,Exception):abort=exc
    finally:
        received=None
        if wan is not None:
            try:wan.close()
            except BaseException as exc:
                store.failure("WAN_RELEASE",exc)
                if abort is None and not isinstance(exc,Exception):abort=exc
    for group in ("sync_reads","payload_reads"):
        for key,row in store.data[group].items():
            if row["status"]=="PENDING":row.update(status="FAILED",error="NOT_COMPLETED")
    if store.data["writer"]["status"]=="PENDING":store.data["writer"].update(status="FAILED",error="NOT_COMPLETED")
    seal=records.dump(store.output/"blind_readouts.json",dict(sync_reads=store.data["sync_reads"],payload_reads=store.data["payload_reads"],truth_inputs=False))
    store.data["blind_seal"]=seal
    try:
        path=Path(cfg["posthoc_config"])
        if records.digest(path)!=cfg["posthoc_config_sha256"]:raise ValueError("posthoc identity mismatch")
        expected=payload.message_bits(records.read(path)["message"])
        if len(expected)!=32:raise ValueError("posthoc message must encode exactly32 bits")
        # Validate the entire fixed read set before writing any evaluation row.
        for row in store.data["payload_reads"].values():
            bits=row.get("decoded_bits")
            if row["status"]=="READ" and (not isinstance(bits,list) or len(bits)!=32 or any(b not in (0,1) for b in bits)):
                raise ValueError("decoded payload must contain exactly32 binary bits")
        for key,row in store.data["payload_reads"].items():
            bits=row.get("decoded_bits");store.data["posthoc"][key]=dict(status="EVALUATED_POSTSEAL" if bits is not None else "FAILED",key_role="REGISTERED" if key=="K0" else "WRONG_KEY",bit_errors=sum(a!=b for a,b in zip(bits,expected)) if bits is not None else None,bit_rows=[dict(bit_index=i,expected=b,decoded=bits[i] if bits is not None else None) for i,b in enumerate(expected)])
    except BaseException as exc:
        store.failure("POSTHOC",exc)
        if abort is None and not isinstance(exc,Exception):abort=exc
        for row in store.data["posthoc"].values():
            if row["status"]=="PENDING":row.update(status="FAILED",error=str(exc),bit_rows=[dict(bit_index=i,status="FAILED") for i in range(32)])
    if records.digest(seal["path"])!=seal["sha256"]:raise RuntimeError("blind seal modified")
    store.data["call_integrity"]={k:dict(expected=n,actual=store.data["calls"][k],match=store.data["calls"][k]==dict(attempted=n,completed=n)) for k,n in cfg["planned_calls"].items()}
    store.data.update(status="COMPLETE" if not store.data["failures"] and all(x["match"] for x in store.data["call_integrity"].values()) else "INCOMPLETE",stage="FINISHED",scientific_pass=False,evidence_ceiling="Explicit-input engineering entry; original-version fixed M05 evidence only. No new joint real run, threshold/FPR/generalization.")
    store.save()
    if abort is not None:raise abort
    return store.data

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    r=run(a.output,cfg=load_config(a.config));print(json.dumps(dict(status=r["status"],result=str(a.output/"result.json"))));return 0 if r["status"]=="COMPLETE" else 1
if __name__=="__main__":sys.exit(main())
