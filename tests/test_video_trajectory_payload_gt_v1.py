"""Targeted G/T CPU/fake/static checks; no real generation, VAE or codec."""
from __future__ import annotations
import ast,copy,hashlib,json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from experiments.wan_state_clock import video_trajectory_payload_gt_v1_run as runner
from main.tube_state import video_trajectory_payload_gt_v1 as method
from main.tube_state import video_trajectory_payload_framewise_sync_v1 as original
from runtime.wan import video_trajectory_payload_gt_v1 as backend
from scripts import build_video_trajectory_payload_gt_notebooks as builder

pytestmark=pytest.mark.unit

class FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        self.config=SimpleNamespace(scaling_factor=0.18215,force_upcast=True)
    def encode(self,value):
        pooled=F.avg_pool2d(value.float(),8)
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.cat((pooled,pooled[:,:1]),dim=1)))
    def decode(self,value):
        return SimpleNamespace(sample=F.interpolate(value[:,:3].float(),scale_factor=8,mode="nearest"))

def fake_runtime(monkeypatch,tmp_path,profile):
    cfg=copy.deepcopy(runner.load_config(profile))
    public=replace(method.public_protocol(profile),source_height=32,source_width=32)
    source=(np.arange(181*32*32*3,dtype=np.uint32)%256).astype(np.uint8).reshape(181,32,32,3)
    path=tmp_path/"source.rgb8";path.write_bytes(source.tobytes())
    cfg["source"].update(path=str(path),sha256=hashlib.sha256(source.tobytes()).hexdigest(),shape=list(source.shape),bytes=source.nbytes)
    cfg["framewise_vae"]["batch_frames"]=64
    monkeypatch.setattr(runner,"load_config",lambda p:cfg)
    monkeypatch.setattr(method,"public_protocol",lambda p:public)
    monkeypatch.setattr(runner.framewise,"load_frozen_framewise_vae",lambda **kw:FakeVAE())
    from runtime.wan import generation,video_local_fourier_rm_old8_two_state_v1 as old8
    monkeypatch.setattr(generation,"load_frozen_vae",lambda *a,**kw:FakeVAE())
    monkeypatch.setattr(runner.wan_adapter,"reencode_rgb24_readback",
        lambda vae,rgb:torch.zeros((1,16,(rgb.shape[0]-1)//4+1,1,1)))
    payload_calls=[]
    def fake_payload(normalized,key,R):
        payload_calls.append((key,R))
        return dict(status="READ",decoded_bits=[1]*32,
            votes=[dict(ones=R*15,zeros=R*15,count=R*30) for _ in range(32)],
            truth_used=False,R=R)
    monkeypatch.setattr(backend.payload_method,"payload_read",fake_payload)
    def save(q8,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        raw=q8.numpy().tobytes();path.write_bytes(raw)
        return dict(status="SAVED",path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),shape=list(q8.shape),dtype="uint8")
    monkeypatch.setattr(runner.media,"save_raster",save)
    def transport(raster_path,raster_sha,mp4_path,rgb_path,*,count,event):
        raw=Path(raster_path).read_bytes();assert hashlib.sha256(raw).hexdigest()==raster_sha
        for name in ("mp4_save","mp4_probe","mp4_readback"):
            count(name,False)
            if name=="mp4_save":
                Path(mp4_path).write_bytes(b"fake codec")
                event("mp4",dict(status="SAVED",path=str(mp4_path),sha256=hashlib.sha256(b"fake codec").hexdigest()))
            elif name=="mp4_probe":event("probe",dict(status="COMPLETE"))
            else:
                Path(rgb_path).write_bytes(raw)
                event("rgb24",dict(status="SAVED",path=str(rgb_path),sha256=raster_sha))
            count(name,True)
        return torch.from_numpy(np.frombuffer(raw,np.uint8).reshape(source.shape).copy())
    monkeypatch.setattr(runner.media,"mp4_roundtrip",transport)
    monkeypatch.setattr(old8,"execution_device_dtype",lambda:("cpu",torch.float32))
    monkeypatch.setattr(generation,"prepare_generation",lambda *a,**kw:
        (SimpleNamespace(scheduler=SimpleNamespace()),torch.zeros((1,)),None,None,torch.float32))
    def trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,record,*,diagnostic):
        assert arm=="PAYLOAD_MULTI" and diagnostic is None
        for i in range(50):
            for name in ("transformer_conditional","transformer_unconditional","native_step")+(
                ("local_control","payload_gradient") if i>=25 else ()):
                count(name,False);count(name,True)
            record(dict(index=i,enabled=i>=25))
        return torch.zeros((1,)),dict(fake=True,arm=arm)
    monkeypatch.setattr(old8,"run_trajectory",trajectory)
    monkeypatch.setattr(runner.wan_adapter,"decode_normalized_latent",lambda *a:torch.from_numpy(source.copy()).float()/255)
    return cfg,public,payload_calls

@pytest.mark.parametrize("profile",["G","T"])
def test_fake_complete_rosters_blind_seal_and_source_budget(tmp_path,monkeypatch,profile):
    cfg,public,payload_calls=fake_runtime(monkeypatch,tmp_path,profile)
    store=runner.Store(tmp_path/profile,profile=profile,create=True)
    if profile=="G":
        runner.source_worker(store,cfg)
        reloaded=runner.Store(store.output,profile=profile)
        assert reloaded.data["source_preparation"]["status"]=="SAVED"
        assert Path(reloaded.data["source_protocol"]["path"]).is_file()
        assert not reloaded.data["calls"]  # identity persisted before comparison
        store.data["workers"]["source"]={"status":"COMPLETE"}
    runner.media_worker(store,cfg)
    store.data["workers"]["media"]={"status":"COMPLETE"}
    store.save()
    snapshot=store.blind_snapshot();before=runner.sha256_file(snapshot)
    runner.evaluate(store,cfg)
    assert runner.sha256_file(snapshot)==before
    assert runner.finish(store,cfg)
    fixed=cfg["fixed_denominator"]
    for name in ("observations","sync_readouts","sync_candidate_scores","sync_candidate_tubelet_rows","payload_reads","quality"):
        assert store.data["counts"][name]==fixed[name]
    assert len(payload_calls)==fixed["payload_reads"]
    assert sum(R==22 for _,R in payload_calls)==(8 if profile=="G" else 0)
    for sid,row in store.data["payload_reads"].items():
        assert all(bit["decoded"]==1 and bit["margin"]==0 for bit in row["bit_rows"])
        assert all(bit["count"]==(660 if row["R"]==22 else 1320) for bit in row["bit_rows"])
    for sid,row in store.data["sync_posthoc"].items():
        assert row["true_offset"]==cfg["views"][row["view"]]["truth_start_posthoc_only"]
        assert (row["true_score_gap"] is None)==(row["view"]=="FULL181")
        assert "error" not in row
    for qid,row in store.data["quality"].items():
        if qid.endswith("_vs_P1_FRAMEWISE_RECON"):
            residual=row["condition_minus_p1_residual"]
            expected={"PRECODEC":(180,45,135),"FULL181_MP4":(180,45,135),
                "CROP177_MP4":(176,44,132),"SHORT89_MP4":(88,22,66)}[qid.split("/")[0]]
            assert (residual["edges"],residual["boundary_edges"],residual["internal_edges"])==expected
    if profile=="T":
        writer=store.data["writers"]["T05_TEMPORAL_CONSTRAINT"]
        receipt=runner.read_gzip_json(writer["path"])
        assert len(receipt["rows"])==46 and len(receipt["crop177_partial_rows"])==2
        assert writer["input_scaled_latent_sha256"]==store.data["writers"]["M05_FRAMEWISE_SYNC"]["input_scaled_latent_sha256"]

def test_g_source_failure_retains_full_plan(tmp_path,monkeypatch):
    cfg=runner.load_config("G")
    from runtime.wan import generation
    monkeypatch.setattr(generation,"prepare_generation",lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("fake source failed")))
    store=runner.Store(tmp_path/"failed",profile="G",create=True)
    with pytest.raises(RuntimeError,match="fake source failed"):runner.source_worker(store,cfg)
    runner.settle(store,"fake source failed")
    runner.evaluate(store,cfg)
    assert len(store.data["sync_reads"])==24
    assert sum(row["planned_candidate_scores"] for row in store.data["sync_reads"].values())==792
    assert sum(row["planned_local_rows"] for row in store.data["sync_reads"].values())==19280
    plans=json.loads(Path(store.data["public_candidate_rosters"]["path"]).read_text())
    assert len(plans["89"])==93
    assert all(row["status"]=="NOT_COMPLETED" for row in store.data["source_preparation"]["steps"])
    assert all(row["status"]=="MISSING_READ" for row in store.data["sync_posthoc"].values())
    assert not store.data["calls"]

def test_original_payload_short_support_and_ties_are_not_recomputed(monkeypatch):
    normalized=torch.zeros((1,16,23,40,64),dtype=torch.float32)
    row=backend.read_payload(normalized,"watermark",89)
    original_row=runner.payload_method.payload_read(normalized,"watermark",22)
    assert row["decoded_bits"]==original_row["decoded_bits"]
    assert all(bit["count"]==660 for bit in row["bit_rows"])
    def tie(*a):
        return dict(status="READ",decoded_bits=[1]*32,
            votes=[dict(ones=330,zeros=330,count=660) for _ in range(32)],truth_used=False,R=22)
    monkeypatch.setattr(backend.payload_method,"payload_read",tie)
    assert all(bit["decoded"]==1 and bit["normalized_margin"]==0 for bit in backend.read_payload(normalized,"watermark",89)["bit_rows"])

def test_t_actual_delta_full_inactive_and_partial_receipt():
    public=replace(original.PUBLIC,source_height=32,source_width=32)
    source=np.random.default_rng(7).normal(0,2,(181,4,4,4)).astype(np.float32)
    saved=source.copy()
    written,receipt=method.apply_temporal_writer(source,"watermark",public)
    actual=written.astype(np.float64)-source.astype(np.float64)
    assert np.array_equal(source,saved)
    assert receipt["applied_float32_delta_l2"]==pytest.approx(np.linalg.norm(actual.ravel()),abs=1e-12)
    assert abs(receipt["applied_float32_delta_l2"]-receipt["constructed_delta_l2"])>1e-9
    assert len(receipt["rows"])==46 and any(row["constraint_increment"]==0 for row in receipt["rows"])
    assert receipt["max_constructed_full_residual"]<1e-12
    assert receipt["max_applied_full_residual"]>0
    assert [row["rho"] for row in receipt["crop177_partial_rows"]]==[0.75,0.5]
    assert any(abs(row["constructed_minus_m05"])>1e-6 for row in receipt["crop177_partial_rows"])

@pytest.mark.parametrize("profile",["G","T"])
def test_notebook_static_unbound_guard_and_roundtrip(tmp_path,profile):
    path=builder.build(profile,output=tmp_path/(profile+".ipynb"))
    value=json.loads(path.read_text())
    codes=["".join(cell["source"]) for cell in value["cells"] if cell["cell_type"]=="code"]
    assert len(codes)==5
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in value["cells"]:
        if cell["cell_type"]=="code":
            ast.parse("".join(cell["source"]));assert cell["outputs"]==[] and cell["execution_count"] is None
    scope={}
    with pytest.raises(RuntimeError,match="UNPUBLISHED_DRAFT"):exec(codes[1],scope)
    assert "OUTPUT" not in scope
    assert ("WanPipeline" in codes[2])==(profile=="G")
    sha="a"*40
    bound=builder.build(profile,sha,tmp_path/(profile+"-bound.ipynb"))
    bound_value=json.loads(bound.read_text())
    assert bound_value["metadata"]["candidate_binding"]["source_sha"]==sha
    assert "".join(bound_value["cells"][3]["source"])==codes[2]
    assert "".join(bound_value["cells"][4]["source"])==codes[3]
    assert "".join(bound_value["cells"][5]["source"])==codes[4]

def test_worker_interrupt_preserves_saved_first_writer_and_full_plan(tmp_path,monkeypatch):
    store=runner.Store(tmp_path/"interrupt",profile="T",create=True)
    first="M05_FRAMEWISE_SYNC";store.data["writers"][first]={"status":"SAVED","path":"retained-receipt"}
    store.save()
    class Stream:
        closed=False
        def __iter__(self):raise KeyboardInterrupt("fake interrupt")
        def close(self):self.closed=True
    class Child:
        stdout=Stream();terminated=False
        def poll(self):return None
        def terminate(self):self.terminated=True
        def wait(self,timeout=None):return -15
    child=Child()
    monkeypatch.setattr(runner.subprocess,"Popen",lambda *a,**kw:child)
    store,interrupted=runner.run_worker_phase(store)
    assert interrupted and child.terminated and child.stdout.closed
    assert store.data["writers"][first]["status"]=="SAVED"
    assert store.data["writers"]["T05_TEMPORAL_CONSTRAINT"]["status"]=="NOT_COMPLETED"
    assert len(store.data["sync_reads"])==16 and len(store.data["payload_reads"])==16
    assert all(row["error"]=="KeyboardInterrupt: fake interrupt" for row in store.data["sync_reads"].values())
