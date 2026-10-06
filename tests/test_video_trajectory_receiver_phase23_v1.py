"""New 1A boundary checks only; unchanged Stage2 suite is not rerun."""
import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from main.tube_state import grow_video_reference as layout
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as method
from runtime.wan import video_trajectory_receiver_estimated_align_v1 as accepted
from experiments.wan_state_clock import video_trajectory_receiver_phase23_v1_run as runner
from experiments.wan_state_clock import video_trajectory_receiver_estimated_align_v1_run as shared
from scripts import build_video_trajectory_receiver_phase23_notebook as builder
pytestmark=pytest.mark.unit

def sync_fixture(n,key,offset,tied=False):
    d=method.missing_sync(n,key,"fake")
    for c in d["candidate_rows"]:c.update(status="SCORED",score=1.0 if c["source_offset"]==offset else 0.0)
    for x in d["local_rows"]:x.update(status="SCORED",q=0.0)
    if tied:
        for c in d["candidate_rows"]:c["score"]=0.0
    best=max(c["score"] for c in d["candidate_rows"])
    top=[c["source_offset"] for c in d["candidate_rows"] if c["score"]==best]
    d.update(status="COMPLETE");d["counts"]["failed_candidates"]=0
    d["summary"].update(best_score=best,top_offsets=top,canonical_offset=top[0] if len(top)==1 else None,unique=len(top)==1)
    return d

def votes_fixture(n,key):
    R=method.support(n);signs=np.ones((4,R,240),np.int8);signs[:,R//2:]=-1
    d=method.detailed_votes(signs,layout.coordinates(key),np.zeros_like(signs),n)
    d.update(original_reader_match=True,original_readout=dict(status="READ",R=R,truth_used=False,
        decoded_bits=[x["decoded"] for x in d["bit_rows"]],
        votes=[{k:x[k] for k in ("ones","zeros","count")} for x in d["bit_rows"]]))
    return d

class FakeFW:
    closed=False;inputs=[];mode="max"
    def __init__(self,cfg):FakeFW.closed=False;FakeFW.inputs=[]
    def encode(self,rgb):
        FakeFW.inputs.append(rgb.clone())
        return (len(rgb),int(rgb[0,0,0,0])),dict(fake=True)
    def score(self,observed,key):
        n,first=observed
        offset=first if key=="watermark" else (first+1 if first%4==2 else first-1)
        if FakeFW.mode=="wrong_p0" and key=="watermark":offset=0
        d=sync_fixture(n,key,offset,tied=FakeFW.mode=="tie")
        if FakeFW.mode=="missing" and key=="watermark":d["candidate_rows"].pop()
        return d
    def close(self):FakeFW.closed=True

class FakeWan:
    closed=False;source_reads=0;seen=[]
    @staticmethod
    def read_source(spec):
        FakeWan.source_reads+=1
        return accepted.WanBackend.read_source(spec)
    operate_clip=staticmethod(accepted.WanBackend.operate_clip)
    pixel_receipt=staticmethod(accepted.WanBackend.pixel_receipt)
    def __init__(self,cfg):assert FakeFW.closed;FakeWan.closed=False;FakeWan.seen=[]
    def encode(self,rgb):
        FakeWan.seen.append(rgb.clone())
        class Latent:shape=(1,16,(len(rgb)-1)//4+1,40,64)
        return Latent()
    def read(self,latent,key,n):return votes_fixture(n,key)
    def close(self):FakeWan.closed=True

def fixture(tmp_path,monkeypatch):
    cfg=copy.deepcopy(runner.load_config())
    prep=json.loads(runner.PREPARATION.read_text());post=json.loads(runner.POSTHOC.read_text())
    full=np.broadcast_to(np.arange(181,dtype=np.uint8)[:,None,None,None],(181,2,2,3)).copy()
    p=tmp_path/"full.rgb8";p.write_bytes(full.tobytes())
    prep["source"].update(path=str(p),shape=list(full.shape),bytes=full.nbytes,sha256=runner.digest(p))
    for oid,s in cfg["inputs"].items():s.update(shape=[s["frames"],2,2,3],bytes=s["frames"]*12)
    pf=tmp_path/"preparation.json";hf=tmp_path/"posthoc.json"
    runner.dump(pf,prep);runner.dump(hf,post)
    cfg["preparation_config_sha256"]=runner.digest(pf);cfg["posthoc_config_sha256"]=runner.digest(hf)
    monkeypatch.setattr(runner,"PREPARATION",pf);monkeypatch.setattr(runner,"POSTHOC",hf)
    FakeFW.mode="max";FakeWan.source_reads=0
    return cfg,prep,torch.from_numpy(full)

def test_fixture_maps_and_direct_reuse():
    x=torch.arange(181,dtype=torch.uint8).reshape(181,1,1,1).expand(181,1,1,3).clone();before=x.clone()
    for n,start in ((177,2),(177,3),(89,38),(89,39)):
        y=runner.construct_received(x,start,n)
        assert y[:,0,0,0].tolist()==list(range(start,start+n)) and torch.equal(x,before)
        y[0]=255;assert torch.equal(x,before)
    assert runner.method is method and runner.WanBackend is accepted.WanBackend and runner.FramewiseBackend is accepted.FramewiseBackend
    assert runner.run_payload is shared.run_payload and runner.seal_payload is shared.seal_payload
    with pytest.raises(ValueError):runner.construct_received(x,5,177)

def test_single_read_four_crops_blind_choice_seals_and_max_budget(tmp_path,monkeypatch):
    cfg,prep,full=fixture(tmp_path,monkeypatch);out=tmp_path/"run";original=Path.read_bytes
    def guarded(p):
        if p==runner.POSTHOC:assert (out/"blind_payload_readouts.json").is_file()
        return original(p)
    monkeypatch.setattr(Path,"read_bytes",guarded)
    class GuardWan(FakeWan):
        def __init__(self,c):
            assert (out/"blind_sync_readouts.json").is_file() and (out/"physical_plan.json").is_file()
            assert not (out/"blind_payload_readouts.json").exists()
            super().__init__(c)
    result=runner.run(out,cfg=cfg,framewise_type=FakeFW,wan_type=GuardWan)
    assert result["status"]=="COMPLETE" and FakeWan.source_reads==1
    assert all(x["match"] for x in result["call_integrity"].values())
    assert result["calls"]["source_slice"]==dict(attempted=4,completed=4)
    assert result["physical_plan"]["planned_calls"]==dict(wan_receiver_encode=12,payload_read=16)
    assert result["counts"]==dict(logical=dict(reads=16,votes=506880,time_bit_rows=16896,final_bits=512),
        physical=dict(reads=16,votes=506880,time_bit_rows=16896,final_bits=512),sync_reads=8)
    assert sum(x["candidate_scores"] for x in result["sync_reads"].values())==392
    assert sum(x["candidate_tubelet_rows"] for x in result["sync_reads"].values())==9456
    for i,(oid,x) in enumerate(prep["crops"].items()):
        clip=full[x["start"]:x["start"]+x["length"]]
        assert torch.equal(FakeFW.inputs[i],clip)
        assert result["observations"][oid]["sha256"]==hashlib.sha256(clip.numpy().tobytes()).hexdigest()
        assert result["estimates"][oid+"/K0"]["phase"]==x["start"]%4
        assert result["estimates"][oid+"/K0"]["phase"]!=result["estimates"][oid+"/K1"]["phase"]
        assert result["sync_posthoc"][oid+"/K0"]["offset_correct"]
        assert not result["sync_posthoc"][oid+"/K1"]["offset_correct"]
    for gid,item in result["physical_plan"]["encodes"].items():
        assert item["received_index_map"]==method.phase_map(item["frames"],item["phase"])
    assert all(x["sha256"] is None for x in cfg["inputs"].values()),"caller config mutated"
    for file in ("blind_sync_readouts.json","physical_plan.json","blind_payload_readouts.json"):
        s=(out/file).read_text()
        assert all(x not in s for x in ('"source_start"','"source_frame_map"','"expected"','"message"','"view"'))
    assert "historical_comparisons" not in result and "history" not in json.dumps(result["calls"])
    assert sum(len(x["bit_rows"]) for x in result["payload_posthoc"].values())==512
    assert sum(x["time_bit_rows"] for x in result["payload_posthoc"].values())==16896

def test_wrong_estimate_p0_alias_is_not_corrected_by_truth(tmp_path,monkeypatch):
    cfg,_,_=fixture(tmp_path,monkeypatch);FakeFW.mode="wrong_p0"
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=FakeFW,wan_type=FakeWan)
    assert r["status"]=="COMPLETE"  # execution only; truth is not a success selector
    assert r["physical_plan"]["planned_calls"]==dict(wan_receiver_encode=8,payload_read=12)
    for oid in cfg["inputs"]:
        sid=oid+"/K0"
        assert r["estimates"][sid]["offset"]==r["estimates"][sid]["phase"]==0
        assert not r["sync_posthoc"][sid]["offset_correct"] and not r["sync_posthoc"][sid]["phase_correct"]
        assert r["payload_reads"][sid+"/EST_ALIGN"]["alias_of"]==sid+"/BASELINE"

@pytest.mark.parametrize("mode",["tie","missing"])
def test_unresolved_keeps_all_baselines_and_fixed_aligned_failures(tmp_path,monkeypatch,mode):
    cfg,_,_=fixture(tmp_path,monkeypatch);FakeFW.mode=mode
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=FakeFW,wan_type=FakeWan)
    assert r["status"]=="INCOMPLETE" and len(r["payload_reads"])==16
    assert all(x["status"]=="READ" for k,x in r["payload_reads"].items() if k.endswith("/BASELINE"))
    failures=[x for x in r["payload_reads"].values() if x["status"]=="FAILED"]
    assert len(failures)==(8 if mode=="tie" else 4)
    assert all(runner.read(x["detail_path"])["signed_votes"] is None for x in failures)

def test_missing_full_keeps_sixteen_slots_without_loading_models(tmp_path,monkeypatch):
    cfg,prep,_=fixture(tmp_path,monkeypatch);Path(prep["source"]["path"]).unlink()
    class Never(FakeFW):
        def __init__(self,c):raise AssertionError("missing source must not load VAE")
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=Never,wan_type=FakeWan)
    assert FakeWan.source_reads==1 and r["calls"]["source_slice"]["attempted"]==0
    assert len(r["payload_reads"])==16 and all(x["status"]=="FAILED" for x in r["payload_reads"].values())
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==512
    assert sum(len(runner.read(x["detail_path"])["time_bit_rows"]) for x in r["payload_reads"].values())==16896

def test_one_slice_failure_does_not_hide_other_observations(tmp_path,monkeypatch):
    cfg,_,_=fixture(tmp_path,monkeypatch);original=runner.construct_received
    def fail_one(full,start,length):
        if start==3:raise ValueError("fake failed slice")
        return original(full,start,length)
    monkeypatch.setattr(runner,"construct_received",fail_one)
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=FakeFW,wan_type=FakeWan)
    assert r["status"]=="INCOMPLETE" and r["calls"]["source_slice"]==dict(attempted=4,completed=3)
    assert r["counts"]["logical"]["reads"]==12 and len(r["payload_reads"])==16
    assert sum(x["status"]=="FAILED" for x in r["observations"].values())==1

@pytest.mark.parametrize("stage",["slice","framewise","wan"])
def test_catchable_interrupt_seals_sixteen_slots_and_releases(tmp_path,monkeypatch,stage):
    cfg,_,_=fixture(tmp_path,monkeypatch);out=tmp_path/"run"
    original=runner.construct_received
    if stage=="slice":
        def interrupted(full,start,length):
            if start==3:raise KeyboardInterrupt("fake slice interrupt")
            return original(full,start,length)
        monkeypatch.setattr(runner,"construct_received",interrupted)
    class InterruptFW(FakeFW):
        def encode(self,rgb):raise KeyboardInterrupt("fake FW interrupt")
    class InterruptWan(FakeWan):
        def encode(self,rgb):raise KeyboardInterrupt("fake Wan interrupt")
    with pytest.raises(KeyboardInterrupt):
        runner.run(out,cfg=cfg,framewise_type=InterruptFW if stage=="framewise" else FakeFW,
            wan_type=InterruptWan if stage=="wan" else FakeWan)
    r=runner.read(out/"result.json")
    assert r["status"]=="INCOMPLETE" and r["stage"]=="FINISHED"
    assert len(r["payload_reads"])==16 and all(x["status"]=="FAILED" for x in r["payload_reads"].values())
    if stage in ("framewise","wan"):assert FakeFW.closed
    if stage=="wan":assert FakeWan.closed
    for file in ("preparation_receipt.json","blind_sync_readouts.json","physical_plan.json","blind_payload_readouts.json"):
        assert (out/file).is_file()
    assert all(x["status"]!="ENCODING" for x in r.get("alignment_receipts",{}).values())

def test_notebook_and_configuration_static_boundaries(tmp_path):
    p=builder.build(output=tmp_path/"draft.ipynb");doc=json.loads(p.read_text())
    codes=[x for x in doc["cells"] if x["cell_type"]=="code"];assert len(codes)==5
    assert "".join(codes[0]["source"])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in codes:
        assert c["outputs"]==[] and c["execution_count"] is None;ast.parse("".join(c["source"]))
    setup="".join(codes[1]["source"])
    assert setup.index("if SOURCE_SHA is None:")<setup.index("OUTPUT.mkdir")
    with pytest.raises(RuntimeError,match="UNPUBLISHED_DRAFT"):exec(setup,{})
    assert doc["metadata"]["candidate_binding"]["source_sha"] is None
    env="".join(codes[2]["source"]);assert "AutoencoderKL, AutoencoderKLWan" in env
    assert all(t not in env for t in ("ffmpeg","ffprobe","WanPipeline","apt-get","ftfy"))
    assert "historical_comparisons" not in "".join(codes[4]["source"])
    cfg=runner.load_config();prep=json.loads(runner.PREPARATION.read_text());post=json.loads(runner.POSTHOC.read_text())
    assert cfg["preparation_config_sha256"]==runner.digest(runner.PREPARATION) and cfg["posthoc_config_sha256"]==runner.digest(runner.POSTHOC)
    assert [(x["length"],x["start"]) for x in prep["crops"].values()]==[(177,2),(177,3),(89,38),(89,39)]
    for oid,x in prep["crops"].items():assert post["truth"][oid]["source_start"]==x["start"]
    assert all(k not in json.dumps(cfg["inputs"]) for k in ('"source_start"','"start"','"phase"','"message"'))
    assert runner.POSTHOC.relative_to(runner.ROOT).as_posix() not in cfg["source_files"]
    assert runner.CONFIG.relative_to(runner.ROOT).as_posix() in cfg["source_files"]
    assert "experiments/wan_state_clock/video_trajectory_receiver_phase23_v1_run.py" in cfg["source_files"]
    assert "main/tube_state/video_trajectory_receiver_estimated_align_v1.py" in cfg["source_files"]
