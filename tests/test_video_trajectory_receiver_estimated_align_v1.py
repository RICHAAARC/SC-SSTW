"""Focused CPU/fake/static checks; no model, Drive, GPU, media or experiment run."""
import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from main.tube_state import grow_video_reference as layout
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as method
from runtime.wan import video_trajectory_receiver_estimated_align_v1 as backend
from experiments.wan_state_clock import video_trajectory_receiver_estimated_align_v1_run as runner
from scripts import build_video_trajectory_receiver_estimated_align_notebook as builder
pytestmark=pytest.mark.unit

def fake_sync(n,key,offset=None,tied=False):
    d=method.missing_sync(n,key,"fake fixture")
    if offset is None:offset=0 if n==181 else (1 if key=="watermark" else (3 if n==177 else 59))
    for r in d["candidate_rows"]:r.update(status="SCORED",score=1.0 if r["source_offset"]==offset else 0.0)
    for r in d["local_rows"]:r.update(status="SCORED",q=0.0)
    if tied and n!=181:d["candidate_rows"][0]["score"]=1.0
    top=[r["source_offset"] for r in d["candidate_rows"] if r["score"]==1.0]
    d.update(status="COMPLETE")
    d["counts"]["failed_candidates"]=0
    d["summary"].update(top_offsets=top,canonical_offset=top[0] if len(top)==1 else None,
        unique=len(top)==1,best_score=1.0)
    return d

def rawdetail(n,key="watermark",first=1):
    R=method.support(n);signs=np.full((4,R,240),-first,np.int8);signs[:,:R//2]=first
    d=method.detailed_votes(signs,layout.coordinates(key),np.zeros_like(signs),n)
    d.update(original_reader_match=True,original_readout=dict(status="READ",R=R,truth_used=False,
        decoded_bits=[x["decoded"] for x in d["bit_rows"]],
        votes=[{k:x[k] for k in ("ones","zeros","count")} for x in d["bit_rows"]]))
    return d

def test_phase_maps_keep_length_and_do_not_mutate():
    for n in (181,177,89):
        rgb=torch.arange(n,dtype=torch.uint8).reshape(n,1,1,1).expand(n,1,1,3).clone();before=rgb.clone()
        for p in range(4):
            out=backend.WanBackend.operate_clip(rgb,p)
            assert out[:,0,0,0].tolist()==[0]*p+list(range(n-p))
            assert len(out)==n and torch.equal(rgb,before)
            out[0]=255
            assert torch.equal(rgb,before)
        with pytest.raises(ValueError):method.phase_map(n,4)

def test_complete_unique_estimate_and_all_unresolved_cases():
    for n in (181,177,89):
        for key in ("watermark","watermark-wrong"):
            d=fake_sync(n,key);e=method.estimate(d,n,key)
            assert e["status"]=="ESTIMATED" and e["phase"]==e["offset"]%4 and not e["sync_accepted"]
    d=fake_sync(177,"watermark",tied=True)
    assert method.estimate(d,177,"watermark")["reason"]=="TIED_TOP"
    near=fake_sync(177,"watermark");near["candidate_rows"][0]["score"]=1-0.5e-12
    near["summary"].update(top_offsets=[0,1],canonical_offset=None,unique=False)
    assert method.estimate(near,177,"watermark")["phase"] is None
    invalid=[]
    x=fake_sync(177,"watermark");x["candidate_rows"].pop();invalid.append(x)
    x=fake_sync(177,"watermark");x["candidate_rows"][0]["score"]=float("nan");invalid.append(x)
    x=fake_sync(177,"watermark");x["local_rows"][0]["status"]="FAILED";invalid.append(x)
    x=fake_sync(177,"watermark");x["counts"]["failed_candidates"]=1;invalid.append(x)
    x=fake_sync(177,"watermark");x["summary"]["canonical_offset"]=0;invalid.append(x)
    x=fake_sync(177,"watermark-wrong");invalid.append(x)
    for d in invalid:
        e=method.estimate(d,177,"watermark");assert e["status"]=="UNRESOLVED" and e["offset"] is e["phase"] is None

def test_phase_plan_aliases_key_independence_and_fixed_denominators():
    cfg=runner.load_config();est={}
    for oid,s in cfg["inputs"].items():
        for k,label in runner.keys(cfg):est[oid+"/"+k]=method.estimate(fake_sync(s["frames"],label),s["frames"],label)
    plan=method.physical_plan(cfg["inputs"],est)
    assert plan["planned_calls"]==dict(wan_receiver_encode=7,payload_read=10)
    assert len(plan["logical_slots"])==12
    assert sum(x["planned_votes"] for x in plan["logical_slots"].values())==422400
    assert sum(x["planned_time_bit_rows"] for x in plan["logical_slots"].values())==14080
    assert plan["planned_physical_votes"]==337920 and plan["planned_physical_time_bit_rows"]==11264
    first=next(iter(cfg["inputs"]))
    for k in ("K0","K1"):
        assert plan["logical_slots"][first+"/"+k+"/EST_ALIGN"]["alias_of"]==first+"/"+k+"/BASELINE"
    for e in est.values():e.update(phase=0,offset=0)
    p0=method.physical_plan(cfg["inputs"],est)
    assert p0["planned_calls"]==dict(wan_receiver_encode=3,payload_read=6)
    for oid in list(cfg["inputs"])[1:]:
        for k in ("K0","K1"):est[oid+"/"+k].update(phase=1,offset=1)
    same=method.physical_plan(cfg["inputs"],est)
    assert same["planned_calls"]==dict(wan_receiver_encode=5,payload_read=10)
    assert len(same["encodes"])==5 and len(same["reads"])==10

@pytest.mark.parametrize("n",[181,177,89])
def test_real_cpu_fft_R44_R22_matches_original(n):
    old=torch.get_num_threads();torch.set_num_threads(1)
    try:
        torch.manual_seed(17);z=torch.randn(1,16,(n-1)//4+1,40,64)
        for key in ("watermark","watermark-wrong"):
            d=backend.read_detailed(z,key,n)
            assert d["original_reader_match"] and d["R"]==method.support(n)
            signs=np.asarray(d["signed_votes"])
            for b,r in enumerate(d["bit_rows"]):
                seq=signs[b//8,:,b%8::8].reshape(-1)
                assert len(seq)==30*method.support(n) and int((seq==1).sum())==r["ones"]
        zero=backend.read_detailed(torch.zeros_like(z),"watermark",n)
        assert all(x["decoded"]==0 and x["ones"]==0 for x in zero["bit_rows"])
        assert np.asarray(zero["zero_coefficient_mask"]).all()
    finally:torch.set_num_threads(old)

def test_strict_zero_and_original_counter_tie_order():
    for n in (177,89):
        for first in (-1,1):
            d=rawdetail(n,first=first)
            assert all(x["tie"] and x["decoded"]==(first+1)//2 and x["count"]==30*method.support(n) for x in d["bit_rows"])
        R=method.support(n)
        with pytest.raises(ValueError):
            method.detailed_votes(np.ones((4,R,240)),layout.coordinates("watermark"),np.ones((4,R,240)),n)

def fixture_config(tmp_path,monkeypatch):
    cfg=copy.deepcopy(runner.load_config());reporting=json.loads(runner.POSTHOC.read_text())
    old={"payload_reads":{}};oldseal={"sync_reads":{}}
    for i,(oid,spec) in enumerate(cfg["inputs"].items()):
        n=spec["frames"]
        x=((np.arange(n*12,dtype=np.uint32)+i)%256).astype(np.uint8).reshape(n,2,2,3)
        p=tmp_path/(oid+".rgb8");p.write_bytes(x.tobytes())
        spec.update(path=str(p),shape=list(x.shape),bytes=x.nbytes,sha256=runner.digest(p))
        for k,label in runner.keys(cfg):
            sid=oid+"/"+k;f=tmp_path/"oldsync"/oid/(k+".json.gz")
            receipt=runner.dump(f,dict(readout=fake_sync(n,label)))
            oldseal["sync_reads"][sid]=dict(path=str(f),sha256=receipt["sha256"])
            old["payload_reads"][sid]=rawdetail(n,label)["original_readout"]
    oldp=tmp_path/"old.json";seal=tmp_path/"oldblind.json"
    runner.dump(oldp,old);runner.dump(seal,oldseal)
    reporting["history"]=dict(result_path=str(oldp),result_sha256=runner.digest(oldp),
        blind_path=str(seal),blind_sha256=runner.digest(seal))
    post=tmp_path/"posthoc.json";runner.dump(post,reporting)
    cfg["posthoc_config"]=str(post);cfg["posthoc_config_sha256"]=runner.digest(post);monkeypatch.setattr(runner,"POSTHOC",post)
    return cfg

class FakeFW:
    closed=False
    def __init__(self,cfg):type(self).closed=False
    def encode(self,rgb):return len(rgb),dict(fake=True)
    def score(self,n,key):return fake_sync(n,key)
    def close(self):type(self).closed=True

class FakeWan:
    closed=False
    read_source=staticmethod(backend.WanBackend.read_source)
    operate_clip=staticmethod(backend.WanBackend.operate_clip)
    pixel_receipt=staticmethod(backend.WanBackend.pixel_receipt)
    def __init__(self,model):
        assert FakeFW.closed
        type(self).closed=False
    def encode(self,rgb):
        class Latent:
            shape=(1,16,(len(rgb)-1)//4+1,40,64)
        return Latent()
    def read(self,z,key,n):return rawdetail(n,key)
    def close(self):type(self).closed=True

def test_fixed_success_seals_before_plan_payload_truth_history(tmp_path,monkeypatch):
    cfg=fixture_config(tmp_path,monkeypatch);out=tmp_path/"run"
    original=Path.read_bytes
    protected={runner.POSTHOC,tmp_path/"old.json",tmp_path/"oldblind.json"}
    def guarded(path):
        if path in protected or "oldsync" in path.parts:
            assert (out/"blind_payload_readouts.json").is_file()
        return original(path)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    class GuardWan(FakeWan):
        def __init__(self,model):
            assert (out/"blind_sync_readouts.json").is_file() and (out/"physical_plan.json").is_file()
            assert not (out/"blind_payload_readouts.json").exists()
            super().__init__(model)
    r=runner.run(out,cfg=cfg,framewise_type=FakeFW,wan_type=GuardWan)
    assert r["status"]=="COMPLETE" and r["stage"]=="FINISHED" and FakeFW.closed and GuardWan.closed
    assert r["counts"]==dict(logical=dict(reads=12,votes=422400,time_bit_rows=14080,final_bits=384),
        physical=dict(reads=10,votes=337920,time_bit_rows=11264,final_bits=320),sync_reads=6)
    assert all(x["match"] for x in r["call_integrity"].values())
    assert all(x["baseline_aggregate_equal"] and x["baseline_decoded_equal"] and x["sync_candidate_rows_equal"] for x in r["historical_comparisons"].values())
    assert sum(x["time_bit_rows"] for x in r["payload_posthoc"].values())==14080
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==384
    assert {x["key_role"] for x in r["payload_posthoc"].values()}=={"REGISTERED","WRONG_KEY"}
    for name in ("blind_sync_readouts.json","physical_plan.json","blind_payload_readouts.json"):
        sealed=runner.read(out/name);assert sealed["truth_inputs"] is False
        assert not any(x in json.dumps(sealed) for x in ('"expected"','"source_frame_map_posthoc"','"message"'))
    assert cfg["posthoc_config_sha256"]==r["posthoc_config_sha256"]

def test_unresolved_estimates_preserve_readable_baselines(tmp_path,monkeypatch):
    cfg=fixture_config(tmp_path,monkeypatch)
    class Tied(FakeFW):
        def score(self,n,key):return fake_sync(n,key,tied=True)
        def close(self):FakeFW.closed=True
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=Tied,wan_type=FakeWan)
    assert r["status"]=="INCOMPLETE" and r["counts"]["logical"]["reads"]==8
    assert r["physical_plan"]["planned_calls"]==dict(wan_receiver_encode=3,payload_read=6)
    assert all(x["status"]=="READ" for k,x in r["payload_reads"].items() if k.endswith("/BASELINE"))
    failed=[x for x in r["payload_reads"].values() if x["status"]=="FAILED"]
    assert len(failed)==4 and all(runner.read(x["detail_path"])["signed_votes"] is None for x in failed)

def test_missing_sources_keep_all_slots_no_model(tmp_path,monkeypatch):
    cfg=fixture_config(tmp_path,monkeypatch)
    for spec in cfg["inputs"].values():spec["path"]+=".missing"
    class Never(FakeFW):
        def __init__(self,cfg):raise AssertionError("no source -> no model")
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=Never,wan_type=FakeWan)
    assert r["status"]=="INCOMPLETE" and len(r["payload_reads"])==12 and len(r["sync_reads"])==6
    assert r["calls"]["framewise_vae_load"]["attempted"]==r["calls"]["wan_vae_load"]["attempted"]==0
    assert r["counts"]["logical"]["votes"]==0
    assert sum(len(runner.read(x["detail_path"])["time_bit_rows"]) for x in r["payload_reads"].values())==14080
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==384

@pytest.mark.parametrize("stage",["framewise","wan"])
def test_interrupt_closes_models_and_seals_fixed_failures(tmp_path,monkeypatch,stage):
    cfg=fixture_config(tmp_path,monkeypatch);out=tmp_path/"run"
    class InterruptFW(FakeFW):
        def encode(self,rgb):raise KeyboardInterrupt("fake FW interrupt")
        def close(self):FakeFW.closed=True
    class InterruptWan(FakeWan):
        def encode(self,rgb):raise KeyboardInterrupt("fake Wan interrupt")
    with pytest.raises(KeyboardInterrupt):
        runner.run(out,cfg=cfg,framewise_type=InterruptFW if stage=="framewise" else FakeFW,
            wan_type=InterruptWan if stage=="wan" else FakeWan)
    r=runner.read(out/"result.json")
    assert r["status"]=="INCOMPLETE" and r["stage"]=="FINISHED" and FakeFW.closed
    if stage=="wan":assert InterruptWan.closed
    assert len(r["payload_reads"])==12 and all(x["status"]=="FAILED" for x in r["payload_reads"].values())
    assert all(x["status"]!="ENCODING" for x in r.get("alignment_receipts",{}).values())
    assert all((out/f).is_file() for f in ("blind_sync_readouts.json","physical_plan.json","blind_payload_readouts.json"))

def test_early_posthoc_failure_closes_all_reporting_slots(tmp_path,monkeypatch):
    cfg=fixture_config(tmp_path,monkeypatch)
    runner.POSTHOC.unlink()
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=FakeFW,wan_type=FakeWan)
    assert r["status"]=="INCOMPLETE" and r["counts"]["logical"]["reads"]==12
    for kind in ("payload_posthoc","sync_posthoc","same_run_differences","historical_comparisons"):
        assert all(x["status"]=="FAILED" and x["error"] for x in r[kind].values())
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==384

def test_notebook_mount_guard_bindings_and_fixed_environment(tmp_path):
    import nbformat
    n=nbformat.read(builder.build(output=tmp_path/"draft.ipynb"),as_version=4);nbformat.validate(n)
    codes=[c for c in n.cells if c.cell_type=="code"];assert len(codes)==5
    assert codes[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in codes:assert not c.outputs and c.execution_count is None;ast.parse(c.source)
    assert codes[1].source.index("if SOURCE_SHA is None:")<codes[1].source.index("OUTPUT.mkdir")
    with pytest.raises(RuntimeError,match="UNPUBLISHED_DRAFT"):exec(codes[1].source,{})
    assert "AutoencoderKL,AutoencoderKLWan" in codes[2].source and "--config" in codes[3].source
    assert "CONFIG_PATH = None" in codes[1].source and "REQUIRES_MEDIA = False" in codes[1].source
