"""Targeted CPU/fake/static evidence only; no real VAE, Drive, GPU or codec."""
import ast,copy,gzip,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from main.tube_state import grow_video_reference as layout
from main.tube_state import video_trajectory_receiver_origin_v1 as method
from runtime.wan import video_trajectory_receiver_origin_v1 as backend
from experiments.wan_state_clock import video_trajectory_receiver_origin_v1_run as runner
from scripts import build_video_trajectory_receiver_origin_notebook as builder
pytestmark=pytest.mark.unit

def rawdetail(first=1):
    signs=np.full((4,44,240),-first,dtype=np.int8);signs[:,:22]=first
    d=method.detailed_votes(signs,layout.coordinates("watermark"),np.zeros_like(signs))
    d.update(original_reader_match=True,original_readout=dict(status="READ",R=44,truth_used=False,
        decoded_bits=[x["decoded"] for x in d["bit_rows"]],
        votes=[{k:x[k] for k in ("ones","zeros","count")} for x in d["bit_rows"]]))
    return d

def test_exact_counter_order_and_zero_votes():
    one=rawdetail(1);zero=rawdetail(-1)
    assert all(x["tie"] and x["decoded"]==1 and x["count"]==1320 for x in one["bit_rows"])
    assert all(x["decoded"]==0 for x in zero["bit_rows"])
    assert len(one["time_bit_rows"])==1408 and one["detailed_vote_count"]==42240
    d=method.detailed_votes(-np.ones((4,44,240),np.int8),layout.coordinates("watermark"),np.ones((4,44,240),np.int8))
    assert all(x["zeros"]==1320 and x["decoded"]==0 for x in d["bit_rows"])
    assert not any(k in d for k in ("condition","source_start","truth","expected"))
    assert [x["receiver_latent_index"] for x in d["time_bit_rows"][:44]]==list(range(1,45))

@pytest.mark.parametrize("key",["watermark","watermark-wrong"])
def test_real_cpu_fft_matches_unchanged_reader(key):
    torch.manual_seed(717)
    z=torch.randn(1,16,45,40,64)
    d=backend.read_detailed(z,key)
    assert d["original_reader_match"]
    assert len(d["signed_votes"])==4
    assert all(x["count"]==1320 for x in d["bit_rows"])
    # Reconstruct every original ordered bit sequence from the detailed tensor.
    signs=np.asarray(d["signed_votes"])
    for b,row in enumerate(d["bit_rows"]):
        seq=signs[b//8,:,b%8::8].reshape(-1)
        assert (seq==1).sum()==row["ones"]
        assert len(seq)==1320

def fixture_config(tmp_path):
    cfg=copy.deepcopy(runner.load_config())
    for i,(c,spec) in enumerate(cfg["inputs"].items()):
        x=((np.arange(181*2*2*3,dtype=np.uint32)+i)%256).astype(np.uint8).reshape(181,2,2,3)
        p=tmp_path/(c+".rgb8");p.write_bytes(x.tobytes())
        spec.update(path=str(p),shape=list(x.shape),bytes=x.nbytes,sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                    start1_sha256=hashlib.sha256(x[1:178].tobytes()).hexdigest())
    old={"payload_reads":{}}
    for spec in cfg["inputs"].values():
        for k in ("K0","K1"):old["payload_reads"][spec["historical_observation_id"]+"/"+k]=rawdetail()["original_readout"]
    prior=tmp_path/"old.json";prior.write_text(json.dumps(old))
    cfg["historical_result"].update(path=str(prior),sha256=runner.digest(prior))
    return cfg

class FakeBackend:
    calls=[];closed=False
    read_source=staticmethod(backend.ReceiverBackend.read_source)
    slice_source=staticmethod(backend.ReceiverBackend.slice_source)
    pixel_receipt=staticmethod(backend.ReceiverBackend.pixel_receipt)
    def __init__(self,model):self.calls.clear();type(self).closed=False
    def encode(self,rgb):
        self.calls.append(("encode",tuple(rgb.shape)))
        return torch.zeros(1,16,45,40,64)
    def read(self,z,key):
        self.calls.append(("read",key))
        return rawdetail()
    def close(self):type(self).closed=True

def test_fixed_success_roster_postseal_history_and_truth(tmp_path,monkeypatch):
    cfg=fixture_config(tmp_path);out=tmp_path/"run"
    original_digest=runner.digest
    def guarded(p):
        if str(p)==cfg["historical_result"]["path"]:
            assert (out/"blind_receiver_readouts.json").is_file()
        return original_digest(p)
    monkeypatch.setattr(runner,"digest",guarded)
    r=runner.run(out,cfg=cfg,backend_type=FakeBackend)
    assert r["status"]=="COMPLETE" and FakeBackend.closed
    assert r["counts"]==dict(observations=4,payload_reads=8,detailed_votes=337920,time_bit_rows=11264,final_bit_rows=256)
    assert all(v["match"] for v in r["call_integrity"].values())
    assert sum(x[0]=="encode" for x in FakeBackend.calls)==4
    assert sum(x[0]=="read" for x in FakeBackend.calls)==8
    blind=json.loads((out/"blind_receiver_readouts.json").read_text())
    assert blind["truth_inputs"] is False
    assert all("condition" not in x and "source_start" not in x for x in blind["observations"].values())
    assert all(x["votes_equal"] and x["decoded_equal"] for x in r["historical_comparisons"].values())
    for row in r["payload_posthoc"].values():
        detailed=runner.read_detail(row["time_bit_path"])
        assert len(detailed["time_bit_rows"])==1408 and len(row["bit_rows"])==32
        assert all(x["signed_normalized_margin"]==0 for x in row["bit_rows"])
    assert {x["source_start_posthoc_only"] for x in r["payload_posthoc"].values()}=={0,1}

def test_missing_sources_keep_fixed_failures_without_model_load(tmp_path):
    cfg=fixture_config(tmp_path)
    for s in cfg["inputs"].values():s["path"]+=".missing"
    class NeverLoad(FakeBackend):
        def __init__(self,model):raise AssertionError("no source -> no model")
    r=runner.run(tmp_path/"run",cfg=cfg,backend_type=NeverLoad)
    assert r["status"]=="INCOMPLETE" and len(r["observations"])==4 and len(r["payload_reads"])==8
    assert r["calls"]["wan_vae_load"]["attempted"]==0
    assert r["counts"]["detailed_votes"]==0
    for row in r["payload_reads"].values():
        d=runner.read_detail(row["detail_path"])
        assert d["status"]=="FAILED" and len(d["time_bit_rows"])==1408 and d["signed_votes"] is None
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==256

def test_interrupt_persists_seal_failures_and_releases_model(tmp_path):
    cfg=fixture_config(tmp_path);out=tmp_path/"run"
    class Interrupted(FakeBackend):
        def encode(self,rgb):raise KeyboardInterrupt("fake bounded interrupt")
    with pytest.raises(KeyboardInterrupt):runner.run(out,cfg=cfg,backend_type=Interrupted)
    r=json.loads((out/"result.json").read_text())
    assert r["status"]=="INCOMPLETE" and r["stage"]=="FINISHED" and Interrupted.closed
    assert all(x["status"]=="FAILED" for x in r["observations"].values())
    assert all(x["status"]=="FAILED" for x in r["payload_reads"].values())
    assert (out/"blind_receiver_readouts.json").is_file()

def test_static_notebook_guard_binding_and_no_media_setup(tmp_path):
    p=builder.build(output=tmp_path/"draft.ipynb")
    n=json.loads(p.read_text());codes=[c for c in n["cells"] if c["cell_type"]=="code"]
    assert len(codes)==5
    assert "".join(codes[0]["source"])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in codes:assert c["outputs"]==[] and c["execution_count"] is None;ast.parse("".join(c["source"]))
    setup="".join(codes[1]["source"])
    assert setup.index("if SOURCE_SHA is None:")<setup.index("OUTPUT.mkdir")
    with pytest.raises(RuntimeError,match="UNPUBLISHED_DRAFT"):exec(setup,{})
    env="".join(codes[2]["source"])
    assert all(x not in env for x in ("ffmpeg","ffprobe","apt-get","WanPipeline","AutoencoderKL,","ftfy"))
    assert "AutoencoderKLWan" in env and "pip','check" in env
    pub=json.loads(builder.build("1"*40,output=tmp_path/"bound.ipynb").read_text())
    for index in (0,3,4,5):assert n["cells"][index]==pub["cells"][index]
    assert "".join(n["cells"][2]["source"]).replace("SOURCE_SHA = None","SOURCE_SHA = '"+'1'*40+"'")=="".join(pub["cells"][2]["source"])


def test_early_posthoc_failure_closes_history_slots(tmp_path,monkeypatch):
    cfg=fixture_config(tmp_path)
    def broken(store):raise RuntimeError("fake posthoc failure after seal")
    monkeypatch.setattr(runner,"join_posthoc",broken)
    r=runner.run(tmp_path/"run",cfg=cfg,backend_type=FakeBackend)
    assert r["status"]=="INCOMPLETE"
    assert all(x["status"]=="NOT_EVALUATED" and "posthoc failed" in x["error"] for x in r["historical_comparisons"].values())
    assert all(x["status"]=="FAILED" and len(x["bit_rows"])==32 for x in r["payload_posthoc"].values())
