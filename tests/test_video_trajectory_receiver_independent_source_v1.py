"""Fixed1B source/M05 integration only: CPU fakes, no real models or codec."""
import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import video_trajectory_receiver_independent_source_v1_run as runner
from experiments.wan_state_clock import video_trajectory_receiver_independent_source_v1_prepare as prepmod
from experiments.wan_state_clock import video_trajectory_payload_gt_v1_run as generation
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as method
from main.tube_state import grow_video_reference as layout
from runtime.wan import video_trajectory_receiver_estimated_align_v1 as runtime
from scripts import build_video_trajectory_receiver_independent_source_notebook as builder
pytestmark=pytest.mark.unit
PROMPT="locked camera, a small blue toy car slowly rolling left to right across a wooden tabletop, steady soft daylight, no people, no cuts"

def fake_source(store,cfg):
    assert cfg["generation"]["prompt"]==PROMPT and cfg["generation"]["seed"]==2026100601
    raw=np.broadcast_to(np.arange(181,dtype=np.uint8)[:,None,None,None],(181,2,2,3)).copy()
    path=Path(store.data["source_preparation"]["rgb_path"]);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw.tobytes())
    spec=dict(path=str(path),sha256=runner.digest(path),bytes=raw.nbytes,shape=list(raw.shape),frames=181,dtype="uint8")
    store.data["source_calls"]={k:dict(attempted=n,completed=n) for k,n in cfg["source_preparation_planned_calls"].items()}
    store.data["source_preparation"].update(status="SAVED",steps=[dict(index=i,status="COMPLETE") for i in range(50)],rgb_receipt=spec)
    store.data["source_protocol"]=spec;store.data["actual_generation_calls"]=True;store.save()

def fake_m05(store,cfg):
    assert store.data["status"]=="SOURCE_READY" and cfg["condition"]=="M05_FRAMEWISE_SYNC"
    src=store.data["source_protocol"];out=store.output/"M05_FRAMEWISE_SYNC/received.rgb8";out.parent.mkdir()
    out.write_bytes(Path(src["path"]).read_bytes())
    store.data["received_source"]={**src,"path":str(out)}
    store.data["calls"]={k:dict(attempted=n,completed=n) for k,n in cfg["m05_planned_calls"].items()}
    store.data["writer"].update(status="SAVED",target_margin=0.5)
    store.data["transport"].update(status="COMPLETE")
    store.save()

class FW:
    closed=False;mode="max"
    def __init__(self,cfg):FW.closed=False
    def encode(self,rgb):return (len(rgb),int(rgb[0,0,0,0])),dict(fake=True)
    def score(self,z,key):
        n,start=z;offset=start if key=="watermark" else start+1 if start%4==2 else start-1
        if FW.mode=="p0":offset=0
        d=method.missing_sync(n,key,"fake")
        for c in d["candidate_rows"]:c.update(status="SCORED",score=float(c["source_offset"]==offset))
        for x in d["local_rows"]:x.update(status="SCORED",q=0.0)
        d["counts"]["failed_candidates"]=0;d.update(status="COMPLETE")
        d["summary"].update(best_score=1.0,top_offsets=[offset],canonical_offset=offset,unique=True)
        return d
    def close(self):FW.closed=True

class Wan:
    read_source=staticmethod(runtime.WanBackend.read_source)
    pixel_receipt=staticmethod(runtime.WanBackend.pixel_receipt)
    operate_clip=staticmethod(runtime.WanBackend.operate_clip)
    def __init__(self,cfg):assert FW.closed
    def encode(self,rgb):
        class L:shape=(1,16,(len(rgb)-1)//4+1,40,64)
        return L()
    def read(self,z,key,n):
        R=method.support(n);s=np.ones((4,R,240),np.int8);s[:,R//2:]=-1
        d=method.detailed_votes(s,layout.coordinates(key),np.zeros_like(s),n)
        d.update(original_reader_match=True,original_readout=dict(status="READ",R=R,truth_used=False,
            decoded_bits=[x["decoded"] for x in d["bit_rows"]],votes=[{k:x[k] for k in ("ones","zeros","count")} for x in d["bit_rows"]]))
        return d
    def close(self):pass

def fixtures(tmp_path,monkeypatch,fail=None):
    cfg=copy.deepcopy(runner.load_config());prep=prepmod.load_config()
    for row in cfg["inputs"].values():row.update(shape=[row["frames"],2,2,3],bytes=row["frames"]*12)
    pf=tmp_path/"prepare.json";pf.write_text(json.dumps(prep));cfg["preparation_config_sha256"]=runner.digest(pf)
    monkeypatch.setattr(runner,"PREPARATION",pf);monkeypatch.setattr(prepmod,"CONFIG",pf)
    out=tmp_path/"run";events=[]
    def worker(store,phase):
        assert len(store.data["payload_reads"])==16 and not (out/"blind_sync_readouts.json").exists()
        events.append(phase)
        def failed_source(s,c):
            s.data["stage"]="SOURCE_FAKE"
            s.data["source_calls"]["generation_load"]["attempted"]=1
            raise KeyboardInterrupt("source interrupt") if fail=="source_interrupt" else RuntimeError("source failed")
        def failed_m05(s,c):
            s.data["stage"]="M05_MP4"
            s.data["calls"]["mp4_save"]["attempted"]=1
            raise KeyboardInterrupt("codec interrupt") if fail=="codec_interrupt" else RuntimeError("codec failed")
        prepmod.run_phase(store.output/"source_preparation",phase,cfg=prep,
            source_fn=failed_source if fail and fail.startswith("source") else fake_source,
            m05_fn=failed_m05 if fail and fail.startswith("codec") else fake_m05)
        store.data.setdefault("preparation_workers",{})[phase]=dict(status="COMPLETE",fake=True);store.save()
    def source_fn(store,p):return runner.prepare_new_source(store,p,worker=worker)
    FW.mode="max"
    return cfg,prep,out,events,source_fn

def test_fixed_protocol_and_source_worker_reuse():
    cfg=runner.load_config();prep=prepmod.load_config()
    assert prep["generation"]["prompt"]==PROMPT and prep["generation"]["seed"]==2026100601
    assert prepmod.run_phase.__kwdefaults__["source_fn"] is generation.source_worker
    old=json.loads((runner.ROOT/"experiments/wan_state_clock/configs/video_trajectory_payload_g_v1.json").read_text())
    assert prep["control"]==old["control"] and prep["key"]==old["key"] and prep["message"]==old["message"]
    assert {k:v for k,v in prep["generation"].items() if k not in ("prompt","seed")}=={k:v for k,v in old["generation"].items() if k not in ("prompt","seed")}
    assert prep["condition"]=="M05_FRAMEWISE_SYNC" and prep["projection_target"]==0.5
    assert prep["source"]["path"] is prep["source"]["sha256"] is None
    assert cfg["fixed_denominator"]==runner.accepted.load_config()["fixed_denominator"]
    assert cfg["physical_upper_bounds"]==dict(wan_receiver_encode=12,payload_read=16)
    assert runner.construct_received is runner.accepted.construct_received and runner.seal_sync_and_plan is runner.accepted.seal_sync_and_plan

@pytest.mark.parametrize("phase_mode,physical_encodes,physical_reads", [("max",12,16),("p0",4,8)])
def test_two_worker_order_identity_seals_key_independence_and_aliases(tmp_path,monkeypatch,phase_mode,physical_encodes,physical_reads):
    cfg,prep,out,events,source_fn=fixtures(tmp_path,monkeypatch);FW.mode=phase_mode;original=Path.read_bytes
    def guard(p):
        if p==runner.POSTHOC:assert (out/"blind_payload_readouts.json").is_file()
        return original(p)
    monkeypatch.setattr(Path,"read_bytes",guard)
    class GuardFW(FW):
        def __init__(self,c):
            assert events==["source","m05"] and (out/"source_preparation/source_preparation.json").is_file()
            super().__init__(c)
    class GuardWan(Wan):
        def __init__(self,c):
            assert (out/"blind_sync_readouts.json").is_file() and (out/"physical_plan.json").is_file()
            super().__init__(c)
    r=runner.run(out,cfg=cfg,framewise_type=GuardFW,wan_type=GuardWan,source_fn=source_fn)
    assert events==["source","m05"] and r["status"]=="COMPLETE"
    assert r["generated_source"]["completed_steps"]==50
    assert r["generated_source"]["source_calls"]=={k:dict(attempted=n,completed=n) for k,n in prep["source_preparation_planned_calls"].items()}
    assert r["generated_source"]["m05_calls"]=={k:dict(attempted=n,completed=n) for k,n in prep["m05_planned_calls"].items()}
    assert all(x["match"] for x in r["call_integrity"].values())
    assert r["counts"]["logical"]==dict(reads=16,votes=506880,time_bit_rows=16896,final_bits=512)
    assert r["calls"]["wan_receiver_encode"]["completed"]==physical_encodes and r["calls"]["payload_read"]["completed"]==physical_reads
    assert sum(x["candidate_scores"] for x in r["sync_reads"].values())==392
    assert sum(x["candidate_tubelet_rows"] for x in r["sync_reads"].values())==9456
    record=runner.read(out/"preparation_receipt.json")
    for oid,item in prep["crops"].items():
        raw=np.broadcast_to(np.arange(item["start"],item["start"]+item["length"],dtype=np.uint8)[:,None,None,None],(item["length"],2,2,3)).copy()
        assert r["observations"][oid]["sha256"]==hashlib.sha256(raw.tobytes()).hexdigest()
        assert record["crops"][oid]["source_frame_map"]==list(range(item["start"],item["start"]+item["length"]))
        if phase_mode=="max":assert r["estimates"][oid+"/K0"]["phase"]!=r["estimates"][oid+"/K1"]["phase"]
        else:
            assert not r["sync_posthoc"][oid+"/K0"]["phase_correct"]
            assert r["payload_reads"][oid+"/K0/EST_ALIGN"]["alias_of"]==oid+"/K0/BASELINE"
    for f in ["blind_sync_readouts.json","physical_plan.json","blind_payload_readouts.json"]:
        text=(out/f).read_text()
        assert all(x not in text for x in ['"source_start"','"message"','"writer"','"source_frame_map"'])
    assert "historical_comparisons" not in r

@pytest.mark.parametrize("fail",["source_failure","codec_failure","source_interrupt","codec_interrupt"])
def test_source_or_codec_failure_keeps_all_slots_no_model_or_old_source(tmp_path,monkeypatch,fail):
    cfg,prep,out,events,source_fn=fixtures(tmp_path,monkeypatch,fail)
    class Never(FW):
        def __init__(self,c):raise AssertionError("receiver model loaded after failed preparation")
    if fail.endswith("interrupt"):
        with pytest.raises(KeyboardInterrupt):runner.run(out,cfg=cfg,framewise_type=Never,wan_type=Wan,source_fn=source_fn)
        r=runner.read(out/"result.json")
    else:r=runner.run(out,cfg=cfg,framewise_type=Never,wan_type=Wan,source_fn=source_fn)
    assert r["status"]=="INCOMPLETE" and r["stage"]=="FINISHED"
    assert events==(["source"] if fail.startswith("source") else ["source","m05"])
    assert r["calls"]["framewise_vae_load"]["attempted"]==r["calls"]["wan_vae_load"]["attempted"]==0
    assert len(r["sync_reads"])==8 and len(r["payload_reads"])==16
    assert all(x["status"]=="FAILED" for x in r["payload_reads"].values())
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==512
    assert sum(len(runner.read(x["detail_path"])["time_bit_rows"]) for x in r["payload_reads"].values())==16896
    for f in ["preparation_receipt.json","blind_sync_readouts.json","physical_plan.json","blind_payload_readouts.json"]:assert (out/f).is_file()
    saved=runner.read(out/"source_preparation/source_preparation.json")
    assert saved["status"]=="FAILED"
    assert all(x["status"]!="PENDING" for x in saved["source_preparation"]["steps"])

def test_m05_only_real_orchestration_with_cpu_stubs(tmp_path,monkeypatch):
    cfg=prepmod.load_config();store=prepmod.Store(tmp_path/"prep",cfg,"source");fake_source(store,cfg)
    store.data["status"]="SOURCE_READY";store.save();events=[]
    monkeypatch.setattr(prepmod.framewise,"load_frozen_framewise_vae",lambda **kw:object())
    monkeypatch.setattr(prepmod.framewise,"encode_rgb_frames",lambda *a,**kw:torch.zeros((181,4,2,2)))
    def write(z,key,condition,public):
        events.append(condition);assert not z.any() and condition=="M05_FRAMEWISE_SYNC"
        z+=1;return z,dict(rows=[dict(fake=True)],target_margin=0.5)
    monkeypatch.setattr(prepmod.writer,"write_condition",write)
    monkeypatch.setattr(prepmod.framewise,"decode_rgb_frames",lambda *a,**kw:torch.zeros((181,2,2,3)))
    def save(q,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(q.numpy().tobytes())
        return dict(status="SAVED",path=str(path),sha256=runner.digest(path),bytes=path.stat().st_size,shape=list(q.shape),dtype="uint8")
    monkeypatch.setattr(prepmod.media,"save_raster",save)
    def transport(path,h,mp4,rgb_path,*,count,event):
        assert runner.digest(path)==h
        for name,stage in [("mp4_save","mp4"),("mp4_probe","probe"),("mp4_readback","rgb24")]:
            count(name,False);count(name,True)
            row=dict(status="COMPLETE" if stage=="probe" else "SAVED")
            if stage=="rgb24":row.update(save(torch.zeros((181,2,2,3),dtype=torch.uint8),rgb_path))
            event(stage,row)
        return torch.zeros((181,2,2,3),dtype=torch.uint8)
    monkeypatch.setattr(prepmod.media,"mp4_roundtrip",transport)
    prepmod.m05_worker(store,cfg)
    assert events==["M05_FRAMEWISE_SYNC"]
    assert store.data["calls"]=={k:dict(attempted=n,completed=n) for k,n in cfg["m05_planned_calls"].items()}
    assert store.data["writer"]["independent_copy"] and store.data["received_source"]["frames"]==181

def test_worker_catchable_interrupt_terminates_and_records(tmp_path,monkeypatch):
    cfg=runner.load_config();store=runner.Store(tmp_path/"run",cfg);calls=[]
    class BrokenStream:
        def __iter__(self):raise KeyboardInterrupt("fake process interrupt")
        def close(self):calls.append("close")
    class Process:
        stdout=BrokenStream()
        def __init__(self,*a,**kw):calls.append("spawn");assert "start_new_session" not in kw
        def poll(self):return None
        def terminate(self):calls.append("terminate")
        def wait(self,timeout=None):calls.append("wait");return -15
    monkeypatch.setattr(runner.subprocess,"Popen",Process)
    with pytest.raises(KeyboardInterrupt):runner.launch_preparation_worker(store,"source")
    assert calls==["spawn","terminate","wait","close"]
    assert store.data["preparation_workers"]["source"]["status"]=="FAILED" and len(store.data["payload_reads"])==16

def test_fixed_notebook_and_source_binding_static(tmp_path):
    p=builder.build(output=tmp_path/"draft.ipynb");nb=json.loads(p.read_text());codes=[c for c in nb["cells"] if c["cell_type"]=="code"]
    assert len(codes)==5 and "".join(codes[0]["source"])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in codes:assert c["outputs"]==[] and c["execution_count"] is None;ast.parse("".join(c["source"]))
    setup="".join(codes[1]["source"]);assert setup.index("if SOURCE_SHA is None:")<setup.index("OUTPUT.mkdir")
    with pytest.raises(RuntimeError,match="UNPUBLISHED_DRAFT"):exec(setup,{})
    env="".join(codes[2]["source"])
    for token in ["AutoencoderKL","AutoencoderKLWan","WanPipeline","sentencepiece==0.2.2","ftfy==6.3.1","ffmpeg","ffprobe"]:assert token in env
    assert nb["metadata"]["candidate_binding"]["source_sha"] is None
    cfg=runner.load_config()
    assert runner.digest(runner.PREPARATION)==cfg["preparation_config_sha256"]
    assert runner.digest(runner.POSTHOC)==cfg["posthoc_config_sha256"]
    assert all((runner.ROOT/p).is_file() for p in cfg["source_files"])
    assert str(runner.POSTHOC.relative_to(runner.ROOT)) not in cfg["source_files"]

def test_worker_terminated_before_child_failure_handler_settles_receipt(tmp_path):
    cfg=runner.load_config();store=runner.Store(tmp_path/"run",cfg);prep=prepmod.load_config()
    def terminated(store,phase):
        prepmod.Store(store.output/"source_preparation",prep,phase)
        raise RuntimeError("worker exited before its failure handler")
    with pytest.raises(RuntimeError):
        runner.prepare_new_source(store,prep,worker=terminated)
    saved=runner.read(store.output/"source_preparation/source_preparation.json")
    assert saved["status"]=="INCOMPLETE"
    assert all(x["status"]=="NOT_COMPLETED" for x in saved["source_preparation"]["steps"])
    assert saved["writer"]["status"]==saved["transport"]["status"]=="NOT_COMPLETED"
