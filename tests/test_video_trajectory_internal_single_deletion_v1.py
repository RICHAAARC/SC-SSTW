"""Isolated single-deletion semantics; fake resources only, no experiment execution."""
import ast,copy,hashlib,json
from pathlib import Path
import numpy as np
import pytest
import torch
from main.tube_state import grow_video_reference as layout
from main.tube_state import video_trajectory_internal_single_deletion_v1 as method
from runtime.wan import video_trajectory_internal_single_deletion_v1 as runtime
from runtime.wan.video_trajectory_receiver_estimated_align_v1 import read_detailed
from experiments.wan_state_clock import video_trajectory_internal_single_deletion_v1_run as runner
from scripts import build_video_trajectory_internal_single_deletion_notebook as builder
pytestmark=pytest.mark.unit


def h(f="H0",b=2,k=None):return dict(family=f,b=b,k=k)

def fake_sync(path,key,tied=False):
    mapping=method.path_map(path)
    cache=[dict(r=r,d=d,status="SCORED",numerator=0.0 if tied else 40.0*int(mapping[r]==r+d),rho=160.0 if r+d==180 else 40.0)
        for r in range(177) for d in range(5)]
    return method.reduce_cache(cache,key)

def detail(key):
    signs=np.ones((4,44,240),np.int8);signs[:,22:]=-1
    d=method.prior.detailed_votes(signs,layout.coordinates(key),np.zeros_like(signs),177)
    d.update(original_reader_match=True,original_readout=dict(status="READ",R=44,truth_used=False,
        decoded_bits=[x["decoded"] for x in d["bit_rows"]],votes=[{k:x[k] for k in ("ones","zeros","count")} for x in d["bit_rows"]]))
    return d

class FW:
    closed=False;mode="normal"
    def __init__(self,cfg):FW.closed=False
    def encode(self,x):return (int(x[88,0,0,0])==91),dict(fake=True)
    def score(self,is_d,key):
        path=h("H1",2,88) if is_d else h()
        if key!="watermark":path=h("H1",1,80)
        if FW.mode=="wrong":path=h(b=0)
        return fake_sync(path,key,FW.mode=="tie")
    def close(self):FW.closed=True

class Wan:
    closed=False;reads=0;encodes=0;source_reads=0
    read_source=staticmethod(runtime.WanBackend.read_source)
    pixel_receipt=staticmethod(runtime.WanBackend.pixel_receipt)
    operate_map=staticmethod(runtime.WanBackend.operate_map)
    def __init__(self,cfg):assert FW.closed;Wan.closed=False;Wan.reads=Wan.encodes=0
    def encode(self,x):
        Wan.encodes+=1
        return torch.zeros(1) # fake shape; production adapter is separately tested with real CPU latents
    cache_latent=staticmethod(lambda x:x.clone())
    restore_latent=staticmethod(lambda x:x)
    def read(self,z,key,n):Wan.reads+=1;return detail(key)
    def close(self):Wan.closed=True

def fixture(tmp_path,monkeypatch):
    cfg=copy.deepcopy(runner.load_config());prep=runner.read(runner.PREPARATION)
    full=np.broadcast_to(np.arange(181,dtype=np.uint8)[:,None,None,None],(181,2,2,3)).copy()
    p=tmp_path/"full.rgb8";p.write_bytes(full.tobytes());prep["source"].update(path=str(p),shape=list(full.shape),bytes=full.nbytes,sha256=runner.digest(p))
    for row in cfg["inputs"].values():row.update(shape=[177,2,2,3],bytes=177*12)
    f=tmp_path/"prep.json";runner.dump(f,prep);cfg["preparation_config_sha256"]=runner.digest(f);monkeypatch.setattr(runner,"PREPARATION",f)
    FW.mode="normal";FW.closed=Wan.closed=False
    return cfg,prep,torch.from_numpy(full)

def test_709_paths_corrections_and_alias_edges():
    paths=method.hypotheses();assert len(paths)==709 and len({tuple(method.path_map(x)) for x in paths})==709
    for x in paths:
        m=method.path_map(x);assert len(m)==177 and 0<=min(m)<max(m)<=180
        assert sum(b-a==2 for a,b in zip(m,m[1:]))==int(x["family"]=="H1")
        op=method.correction(x);assert len(op["received_index_map"])==177
        assert all(i in range(177) for i in op["received_index_map"])
    c=method.path_map(h());d=method.path_map(h("H1",2,88))
    a=[c[i] for i in method.correction(h())["received_index_map"]]
    b=[d[i] for i in method.correction(h("H1",2,88))["received_index_map"]]
    assert [i for i in range(177) if a[i]!=b[i]]==[90] and (a[90],b[90])==(90,89)
    assert method.correction(h("H1",2,88))["synthetic_output_indices"]==[0,1,90]
    assert method.correction(h("H1",3,176))["received_index_map"]==method.correction(h(b=3))["received_index_map"]
    assert method.correction(h(b=4))["received_index_map"]==method.identity()["received_index_map"]


def test_h0_original_score_equivalence_and_source180_rho():
    z=np.random.default_rng(12).normal(size=(177,4,40,64)).astype(np.float32)
    a=method.score(z,"watermark");b=method.sync.score_received_latent(z,"watermark",method.PUBLIC)
    for x,y in zip(a["candidate_rows"][:5],b["candidate_rows"]):
        assert x["b"]==y["source_offset"]
        assert x["score"]==pytest.approx(y["score"],abs=method.PUBLIC.tie_atol,rel=0)
        assert x["rho"]==pytest.approx(y["denominator"],abs=1e-10)
    assert a["candidate_rows"][0]["rho"]==pytest.approx(7080)
    assert a["candidate_rows"][4]["rho"]==pytest.approx(7200)
    assert len(a["local_top"])==45 and len(a["frame_cache"])==885


def test_unique_tie_incomplete_nonfinite_and_local_grid():
    x=fake_sync(h("H1",2,88),"watermark")
    assert method.estimates(x,"watermark")["joint"]["path"]==h("H1",2,88)
    assert x["local_top"][0]["canonical_displacement"]==2 and x["local_top"][-1]["canonical_displacement"]==3
    tied=fake_sync(h(),"watermark",True)
    assert all(e["status"]=="UNRESOLVED" for e in method.estimates(tied,"watermark").values())
    for mode in ("missing","nonfinite"):
        b=copy.deepcopy(x)
        if mode=="missing":b["candidate_rows"].pop()
        else:b["frame_cache"][0]["numerator"]=float("nan")
        assert method.estimates(b,"watermark")["joint"]["status"]=="UNRESOLVED"
    assert method.estimates(x,"wrong-key")["joint"]["status"]=="UNRESOLVED"


def test_canonical_vote_compatibility_and_first_encounter_tie():
    torch.set_num_threads(2)
    z=torch.randn((1,16,45,40,64),generator=torch.Generator().manual_seed(31));z[0,0]=0
    d=read_detailed(z,"watermark",177)
    assert d["original_reader_match"] and d["detailed_vote_count"]==42240
    assert np.asarray(d["signed_votes"])[0].min()==np.asarray(d["signed_votes"])[0].max()==-1
    assert np.asarray(d["zero_coefficient_mask"])[0].all()
    t=detail("watermark")
    assert all(x["decoded"]==1 and x["ones"]==x["zeros"]==660 for x in t["bit_rows"])


def test_success_seal_order_no_blind_overwrite_and_oracle_cache(tmp_path,monkeypatch):
    cfg,prep,full=fixture(tmp_path,monkeypatch);out=tmp_path/"run";original=Path.read_bytes;blinded={}
    # Guard the real checked() semantic reads, including actual production config paths.
    def guard(p):
        if p==runner.ORACLE:
            assert (out/"blind_payload.json").is_file()
            blinded["bytes"]=(out/"blind_payload.json").read_bytes()
        if p==runner.POSTHOC:assert (out/"oracle_payload.json").is_file()
        return original(p)
    monkeypatch.setattr(Path,"read_bytes",guard)
    r=runner.run(out,cfg=cfg,framewise_type=FW,wan_type=Wan)
    assert r["status"]=="COMPLETE" and all(x["match"] for x in r["call_integrity"].values())
    assert r["counts"]["logical"]==dict(reads=16,votes=675840,time_bit_rows=22528,final_bits=512)
    assert r["counts"]["blind"]["reads"]==12 and r["counts"]["oracle"]["reads"]==4
    assert (out/"blind_payload.json").read_bytes()==blinded["bytes"]
    assert FW.closed and Wan.closed and r["calls"]["source_read"]["completed"]==1
    for oid,indices in prep["source_frame_maps"].items():
        clip=runner.construct(full,indices);assert clip[:,0,0,0].tolist()==indices
        assert r["observations"][oid]["sha256"]==hashlib.sha256(clip.numpy().tobytes()).hexdigest()
        assert r["path_posthoc"][oid+"/K0"]["joint"]["map_correct"]==177
        for key in ("K0","K1"):
            row=r["payload_reads"][oid+"/"+key+"/TRUTH_PATH"]
            assert row["status"]=="READ"
            assert oid+"/"+key+"/TRUTH_PATH" in r["physical_reads"][row["physical_read"]]["logical_slots"]
    # D oracle K1 reuses the successful blind K0 map latent; C K1 already read its H0 map.
    assert r["stage_calls"]["oracle"]["wan_receiver_encode"]["attempted"]==0
    assert r["stage_calls"]["oracle"]["payload_read"]["completed"]==1
    for filename in ("blind_sync_readouts.json","blind_plan.json","blind_payload.json"):
        s=(out/filename).read_text();assert all(v not in s for v in ('"message"','"source_frame_map"','"true_b"','"view"'))
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==512
    assert sum(x["time_bit_rows"] for x in r["payload_posthoc"].values())==22528


@pytest.mark.parametrize("failure",["encode","read"])
def test_oracle_never_retries_cached_blind_failures(tmp_path,monkeypatch,failure):
    cfg,_,_=fixture(tmp_path,monkeypatch)
    class Broken(Wan):
        def encode(self,x):
            if failure=="encode":raise ValueError("fake encode failure")
            return super().encode(x)
        def read(self,z,key,n):raise ValueError("fake read failure")
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=FW,wan_type=Broken)
    assert len(r["payload_reads"])==16 and all(x["status"]=="FAILED" for x in r["payload_reads"].values())
    assert r["stage_calls"]["oracle"]["wan_receiver_encode"]["attempted"]==0
    # Only the never-before-read D oracle K1/map pair may be attempted after successful encodes.
    assert r["stage_calls"]["oracle"]["payload_read"]["attempted"]==(1 if failure=="read" else 0)
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==512


@pytest.mark.parametrize("mode",["wrong","tie"])
def test_no_truth_fallback_and_retained_unresolved_slots(tmp_path,monkeypatch,mode):
    cfg,_,_=fixture(tmp_path,monkeypatch);FW.mode=mode
    r=runner.run(tmp_path/"run",cfg=cfg,framewise_type=FW,wan_type=Wan)
    assert len(r["payload_reads"])==16
    assert all(x["status"]=="READ" for lid,x in r["payload_reads"].items() if lid.endswith("/RAW") or lid.endswith("/TRUTH_PATH"))
    blind=[x for lid,x in r["payload_reads"].items() if lid.endswith("/GLOBAL_ALIGN") or lid.endswith("/PATH_ALIGN")]
    if mode=="tie":assert all(x["status"]=="FAILED" for x in blind)
    else:
        assert all(x["alias_of"].endswith("/RAW") for x in blind)
        assert all(not x["joint"]["b_correct"] for x in r["path_posthoc"].values())


@pytest.mark.parametrize("failure",["missing","framewise_interrupt","wan_interrupt"])
def test_failure_denominators_and_release(tmp_path,monkeypatch,failure):
    cfg,prep,_=fixture(tmp_path,monkeypatch);out=tmp_path/"run"
    if failure=="missing":Path(prep["source"]["path"]).unlink()
    class InterruptFW(FW):
        def encode(self,x):raise KeyboardInterrupt("fake FW")
    class InterruptWan(Wan):
        def encode(self,x):raise KeyboardInterrupt("fake Wan")
    args=dict(cfg=cfg,framewise_type=InterruptFW if failure=="framewise_interrupt" else FW,wan_type=InterruptWan if failure=="wan_interrupt" else Wan)
    if failure=="missing":runner.run(out,**args)
    else:
        with pytest.raises(KeyboardInterrupt):runner.run(out,**args)
    r=runner.read(out/"result.json")
    assert r["status"]=="INCOMPLETE" and r["stage"]=="FINISHED"
    assert len(r["payload_reads"])==16 and all(x["status"]=="FAILED" for x in r["payload_reads"].values())
    assert sum(len(x["bit_rows"]) for x in r["payload_posthoc"].values())==512
    assert sum(len(runner.read(x["detail_path"])["time_bit_rows"]) for x in r["payload_reads"].values())==22528
    assert all(x["status"]!="ENCODING" for x in r["alignment_receipts"].values())
    if failure!="missing":assert FW.closed
    if failure=="wan_interrupt":assert Wan.closed


def test_notebook_and_source_boundaries(tmp_path):
    doc=json.loads(builder.build(output=tmp_path/"draft.ipynb").read_text());codes=[x for x in doc["cells"] if x["cell_type"]=="code"]
    assert len(codes)==5 and ''.join(codes[0]["source"])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in codes:assert c["execution_count"] is None and not c["outputs"];ast.parse(''.join(c["source"]))
    setup=''.join(codes[1]["source"]);assert setup.index('if SOURCE_SHA is None:')<setup.index('OUTPUT.mkdir')
    with pytest.raises(RuntimeError,match="UNPUBLISHED_DRAFT"):exec(setup,{})
    cfg=runner.load_config()
    for path,field in ((runner.PREPARATION,"preparation"),(runner.ORACLE,"oracle"),(runner.POSTHOC,"posthoc")):
        assert runner.digest(path)==cfg[field+"_config_sha256"]
    assert all(p.relative_to(runner.ROOT).as_posix() not in cfg["source_files"] for p in (runner.ORACLE,runner.POSTHOC))
    assert all(k not in json.dumps(cfg["inputs"]) for k in ('"b"','"k"','"message"','"source_frame_map"'))
    assert runtime.WanBackend.read is read_detailed
    assert doc["metadata"]["candidate_binding"]["source_sha"] is None

def test_full_map_cache_maximum_and_phase0_alias():
    inputs=runner.load_config()["inputs"];est={}
    for oid in inputs:
        for key,b,k in (("K0",1,70),("K1",3,90)):
            est[oid+"/"+key]={name:dict(status="ESTIMATED",path=path,reason=None) for name,path in
                (("H0",h(b=b)),("joint",h("H1",b,k)))}
    p=method.plan_blind(inputs,est)
    assert (len(p["encodes"]),len(p["reads"]),len(p["logical_slots"]))==(10,12,12)
    for oid,truth in runner.read(runner.ORACLE)["paths"].items():
        for key in ("K0","K1"):method.add_slot(p,oid,key,"TRUTH_PATH",method.correction(truth),inputs[oid],oracle=True)
    assert (len(p["encodes"]),len(p["reads"]),len(p["logical_slots"]))==(12,16,16)
    for oid in inputs:
        for key in ("K0","K1"):
            est[oid+"/"+key]={name:dict(status="ESTIMATED",path=h(b=0),reason=None) for name in ("H0","joint")}
    p=method.plan_blind(inputs,est)
    assert (len(p["encodes"]),len(p["reads"]))==(2,4)
    assert sum(x.get("alias_of") is not None for x in p["logical_slots"].values())==8
