"""Card51 fixed ECC cases and synthetic CPU dispatch only; no research inputs."""
import ast
import inspect
from itertools import combinations
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
import torch

from main.tube_state import local_payload_secded_39_32_v1 as ecc
from main.tube_state import local_joint_payload_protection_v1 as m
from runtime.wan import local_joint_payload_protection_v1 as runtime
from scripts.build_local_joint_payload_protection_v1_notebook import build, ROOT, STEM, SOURCE_FILES

pytestmark=pytest.mark.unit
torch.set_num_threads(1)
CASES=[("clean",())]+[("single",(i,)) for i in range(39)]+[("double",pair) for pair in combinations(range(39),2)]+[
    ("miscorrection",(1,2,4)),("undetected",(0,1,2,3)),("invalid",(0,1,38)),("erasure",(0,))]


@pytest.mark.parametrize("kind,positions",CASES,ids=[f"{kind}-{'-'.join(map(str,p))}" for kind,p in CASES])
def test_fixed_785_ecc_cases(kind,positions):
    assert len(CASES)==785
    word=[0]*39
    for i in positions:word[i]=None if kind == "erasure" else 1
    out=ecc.decode(word)
    if kind == "clean":
        assert out["status"] == "CODE_CONSISTENT" and out["message"] == [0]*32
    elif kind == "single":
        assert out["status"] == "SINGLE_ERROR_RULE_APPLIED" and out["returned_codeword"] == [0]*39
        assert out["flipped_coordinate"] == positions[0] and out["syndrome_reductions"] == 2
    elif kind == "double":
        assert out["status"] == "DETECTED_UNCORRECTABLE" and out["message"] == [None]*32
        assert not out["rule_applied"] and out["syndrome_reductions"] == 1
    elif kind == "miscorrection":
        assert out["flipped_coordinate"] == 7 and out["message"][ecc.DATA_COORDINATES.index(7)] == 1
    elif kind == "undetected":
        assert out["status"] == "CODE_CONSISTENT" and out["message"] != [0]*32
    elif kind == "invalid":
        assert out["status"] == "INVALID_SHORTENED_SYNDROME" and out["message"] == [None]*32
    else:
        assert out["status"] == "UNRESOLVED" and out["syndrome_reductions"] == 0 and out["message"] == [None]*32


def test_encoder_basis_systematic_and_blind_interfaces():
    for b in range(32):
        bits=[int(i==b) for i in range(32)];word=ecc.encode(bits)
        assert ecc.syndrome(word) == (0,0)
        assert [word[i] for i in ecc.DATA_COORDINATES] == bits
    assert list(inspect.signature(ecc.decode).parameters) == ["received"]
    assert list(inspect.signature(m.select_pilot_path).parameters) == ["pilot_scores"]
    assert list(inspect.signature(m.despread).parameters) == ["raw","selection"]
    with pytest.raises(ValueError):ecc.decode([0]*38)
    with pytest.raises(ValueError):ecc.decode([0]*38+[float("nan")])


@pytest.fixture(scope="module")
def synthetic_carrier():
    public=m.public_codebook("watermark")
    zero=torch.zeros(m.LATENT_SHAPE)
    bits=[int(i%3 == 0) for i in range(32)]
    raw=m.observe(zero,"watermark",public)
    candidate,record=m.write_projection(zero,raw,ecc.encode(bits),public)
    return public,zero,bits,raw,candidate,record


def test_prefix61_geometry_and_one_materialization_from_P(synthetic_carrier):
    public,zero,bits,raw,candidate,record=synthetic_carrier
    for key in m.KEYS.values():
        new=public if key=="watermark" else m.public_codebook(key)
        old=m.base.public_codebook(key)
        assert torch.equal(new["directions"][:,:54],old["directions"])
        assert new["permutations"]==old["permutations"] and new["masks"]==old["masks"]
        assert [x[:54] for x in new["selected_basis"]]==old["selected_basis"]
        for roi in new["directions"]:
            flat=roi.reshape(61,-1)
            assert torch.allclose(flat@flat.T,torch.eye(61,dtype=torch.float64),atol=1e-12)
    old=m.base.public_codebook("watermark")
    j0,old_record=m.base.write_projection(zero,m.base.read_local(zero,"watermark",old),bits,old)
    assert record["fp32_materializations"]==1 and record["original54_continuous_distance_squared_display"]==pytest.approx(old_record["continuous_distance_squared_display"])
    assert record["added_parity_continuous_distance_squared_display"]==154
    delta=m.base.delta_statistics(zero,candidate)
    assert delta["outside_support"]["linf"]==delta["legacy_channels_0_3"]["linf"]==0
    expected=zero.double()
    for k,row in enumerate(record["rows"]):
        coeff=torch.tensor(row["state_score_correction"]+row["data_score_correction"]+row["parity_score_correction"],dtype=torch.float64)
        for r,(y0,y1,x0,x1) in enumerate(m.ROIS):
            expected[0,4:16,1+2*k:3+2*k,y0:y1,x0:x1]+=(coeff@public["directions"][r].reshape(61,-1)).reshape(12,2,8,8)
    assert torch.equal(candidate,expected.float()) and not torch.equal(candidate,j0)
    assert list(inspect.signature(m.write_projection).parameters)[0]=="source_p"
    assert inspect.getsource(m.write_projection).count("out.float()") == 1


def test_unknown_initial_exact_ties_and_payload_mapping(synthetic_carrier):
    public,zero,bits,raw,candidate,record=synthetic_carrier
    shifted=candidate.clone()
    for k in m.STATES:
        src=(k+7)%22
        shifted[:,:,1+2*k:3+2*k]=candidate[:,:,1+2*src:3+2*src]
    observed=m.observe(shifted,"watermark",public)
    selection=m.select_pilot_path([r[:22] for r in observed["pooled_scores"]])
    assert selection["selected_initial_state"] == 7 and selection["path_scores"]==22 and selection["pilot_terms_reduced"]==484
    decoded=m.despread(observed,selection)
    assert all(p["data_bits"]==bits and p["codeword_bits"]==ecc.encode(bits) for p in decoded["positions"])
    assert [r["decoded"] for r in m.aggregate([p["data_soft"] for p in decoded["positions"]])]==bits
    tied=m.select_pilot_path([[0.0]*22 for _ in range(22)])
    assert tied["optimal_initial_states"]==list(range(22)) and tied["selected_path"] is None
    assert all(p["codeword_bits"]==[None]*39 for p in m.despread(raw,tied)["positions"])
    near=[[0.0]*22 for _ in range(22)]
    near[0][3]=1.0;near[0][4]=__import__('math').nextafter(1.0,2.0)
    assert m.select_pilot_path(near)["selected_initial_state"]==4


def test_actual_backend_rejects_saved_non_fp32(tmp_path):
    backend=runtime.WanBackend("not-loaded")
    for dtype in (torch.float16,torch.float64,torch.float32):
        path=tmp_path/(str(dtype)+".pt")
        value=torch.tensor(0,dtype=dtype).expand(m.LATENT_SHAPE)
        torch.save(value,path)
        if dtype == torch.float32:
            loaded=backend.load_terminal(path)
            assert loaded.dtype==dtype and torch.equal(loaded,value)
        else:
            with pytest.raises(ValueError,match="already be FP32"):backend.load_terminal(path)


class StubBackend:
    device="cpu";blocks={}
    def __init__(self,fail=None):self.calls=[];self.fail=fail
    def note(self,name):
        self.calls.append(name)
        if self.fail==name:
            self.fail=None
            raise RuntimeError("synthetic failure "+name)
    def load_terminal(self,path):
        self.note("source/"+str(path))
        # Frozen fixture identities; not actual research terminals.
        return torch.zeros(m.LATENT_SHAPE) if str(path)=="P" else torch.full(m.LATENT_SHAPE,0.01)
    def load(self):self.note("load");return dict(synthetic=True,device="cpu")
    def decode(self,value):
        self.note("decode");self.feature=value
        rgb=torch.tensor(.5).expand(m.VIDEO_SHAPE)
        return dict(preclip=rgb,float_rgb=rgb)
    def quantize(self,rgb):self.note("quantize");return torch.tensor(128,dtype=torch.uint8).expand(m.VIDEO_SHAPE)
    def save_mp4(self,pixels,path):self.note("save");self.pixels=pixels;return dict(synthetic=True)
    def probe(self,path):self.note("probe");return dict(width=512,height=320)
    def read_mp4(self,path,probe):self.note("read");return self.pixels
    def encode(self,rgb):self.note("encode");return self.feature.clone()


@pytest.fixture
def cheap_dispatch(monkeypatch):
    monkeypatch.setattr(torch,"save",lambda value,path:Path(path).write_text("synthetic fixture only"))
    monkeypatch.setattr(runtime,"distortion",lambda *args,**kwargs:dict(status="SCORED",frame_denominator=181,adjacent_denominator=180,synthetic=True))
    monkeypatch.setattr(runtime,"read_latent_bits",lambda *args:dict(status="READ",decoded_bits=[0]*32,
        votes=[dict(ones=0,zeros=2,count=2,tie=False,first_vote=0) for _ in range(32)],truth_used=False))


def test_fixed_three_arm_dispatch_and_persistence(tmp_path,monkeypatch,cheap_dispatch):
    original_decode=ecc.decode
    original_evaluate=runtime.evaluate_all
    def checked_decode(received):
        ledger=json.loads((tmp_path/"cpu_ledger.json").read_text())
        active=[n for n,r in ledger.items() if r["status"]=="STARTED" and "/ecc/" in n]
        assert len(active)==1
        slot=active[0].rsplit("/ecc/",1)[0]
        assert (tmp_path/"blind"/slot/"selection.json").is_file()
        assert (tmp_path/"blind"/slot/"raw_symbols.json").is_file()
        assert not (tmp_path/"evaluation.json").exists()
        return original_decode(received)
    def checked_evaluate(raw,truth):
        assert (tmp_path/"blind_completion.json").exists()
        assert len(list((tmp_path/"blind").rglob("local.json")))==12
        assert len(list((tmp_path/"blind").rglob("legacy.json")))==12
        return original_evaluate(raw,truth)
    monkeypatch.setattr(ecc,"decode",checked_decode)
    monkeypatch.setattr(runtime,"evaluate_all",checked_evaluate)
    backend=StubBackend()
    result=runtime.run("P","J0",tmp_path,backend=backend)
    assert result["status"]=="EXECUTION_COMPLETE",result.get("fatal_error")
    assert len(runtime.units())==sum(r["returned"] for r in result["counts"].values())==69
    assert len(runtime.cpu_units())==sum(r["returned"] for r in result["cpu_counts"].values())==304
    assert result["counts"]["ordinary_decode"]["returned"]==result["counts"]["posterior_encode"]["returned"]==3
    assert result["cpu_arithmetic"]["path_scores"]==264 and result["cpu_arithmetic"]["pilot_terms"]==5808
    assert result["cpu_arithmetic"]["syndrome_reductions"]<=528
    evaluation=result["evaluation"]
    assert len(evaluation)==12 and sum(r["local"]["raw_symbols"]["denominator"] for r in evaluation.values())==10296
    assert sum(r["local"]["ecc_data"]["denominator"] for r in evaluation.values())==8448
    assert sum(r["legacy"]["denominator"] for r in evaluation.values())==384
    local=evaluation["J_ECC/mp4/CORRECT"]["local"]
    assert local["state"]["correct"]==22 and local["raw_data"]["correct"]==local["ecc_data"]["correct"]==704
    count=len(backend.calls)
    assert runtime.run("NO","NO",tmp_path,backend=backend)["status"]=="EXISTING_ATTEMPT_NOT_REEXECUTED"
    assert len(backend.calls)==count


@pytest.mark.parametrize("failure",["source/J0","encode"])
def test_failed_calls_keep_denominators_and_independent_arms(tmp_path,cheap_dispatch,failure):
    backend=StubBackend(failure)
    result=runtime.run("P","J0",tmp_path,backend=backend)
    assert result["status"]=="INCOMPLETE"
    assert len(result["calls"])==69 and len(result["cpu_calls"])==304
    assert sum(r["local"]["state"]["denominator"] for r in result["evaluation"].values())==264
    assert sum(r["local"]["ecc_data"]["denominator"] for r in result["evaluation"].values())==8448
    assert result["evaluation"]["J_ECC/mp4/CORRECT"]["local"]["ecc_data"]["correct"]==704
    if failure=="source/J0":
        assert result["evaluation"]["J0/mp4/CORRECT"]["local"]["ecc_data"]["unavailable"]==704
        assert result["counts"]["ordinary_decode"]["returned"]==2
    else:
        assert result["counts"]["posterior_encode"]["started"]==3
        assert result["counts"]["posterior_encode"]["failed"]==1
        assert result["evaluation"]["P/mp4/CORRECT"]["local"]["raw_symbols"]["unavailable"]==858


def test_ecc_failure_retains_raw_and_missing_post(tmp_path,monkeypatch,cheap_dispatch):
    original=ecc.decode
    counter=0
    def fail_once(received):
        nonlocal counter
        counter+=1
        if counter==1:raise RuntimeError("synthetic ECC fault")
        return original(received)
    monkeypatch.setattr(ecc,"decode",fail_once)
    result=runtime.run("P","J0",tmp_path,backend=StubBackend())
    assert result["status"]=="INCOMPLETE" and result["cpu_counts"]["ecc_block"]["failed"]==1
    local=result["evaluation"]["P/terminal/CORRECT"]["local"]
    assert local["raw_data"]["unavailable"]==0 and local["ecc_data"]["unavailable"]==32
    assert local["ecc_data"]["erasures"]==672


def test_quality_peaks_all_metrics_ties_and_one_measurement(monkeypatch):
    def stats(rms,linf):return dict(rms=rms,linf=linf)
    original=dict(status="SCORED",frame_denominator=3,adjacent_denominator=2,
        frames=[dict(frame=2,whole=stats(4,7),outside_nominal_support=stats(3,6)),
                dict(frame=0,whole=stats(2,8),outside_nominal_support=stats(3,5)),
                dict(frame=1,whole=stats(4,8),outside_nominal_support=stats(1,6))],
        adjacent=[dict(frame_before=1,frame_after=2,whole=stats(5,9),outside_union_nominal_support=stats(2,7)),
                  dict(frame_before=0,frame_after=1,whole=stats(5,11),outside_union_nominal_support=stats(6,7))])
    calls=[]
    def saved_distortion(left,right,*,meaning):
        calls.append((left,right,meaning));return original
    monkeypatch.setattr(runtime,"distortion",saved_distortion)
    output=runtime.measured_quality("synthetic-left","synthetic-right",meaning="synthetic saved rows")
    assert calls==[("synthetic-left","synthetic-right","synthetic saved rows")]
    assert "peaks" not in original and output["frames"]==original["frames"] and output["adjacent"]==original["adjacent"]
    expected={
        ("frames","whole","rms"):(4,[{"frame":1},{"frame":2}]),
        ("frames","whole","linf"):(8,[{"frame":0},{"frame":1}]),
        ("frames","outside_nominal_support","rms"):(3,[{"frame":0},{"frame":2}]),
        ("frames","outside_nominal_support","linf"):(6,[{"frame":1},{"frame":2}]),
        ("adjacent","whole","rms"):(5,[{"frame_before":0,"frame_after":1},{"frame_before":1,"frame_after":2}]),
        ("adjacent","whole","linf"):(11,[{"frame_before":0,"frame_after":1}]),
        ("adjacent","outside_union_nominal_support","rms"):(6,[{"frame_before":0,"frame_after":1}]),
        ("adjacent","outside_union_nominal_support","linf"):(7,[{"frame_before":0,"frame_after":1},{"frame_before":1,"frame_after":2}])}
    for (rows,region,metric),(value,locations) in expected.items():
        peak=output["peaks"][rows][region][metric]
        assert peak["status"]=="SCORED" and peak["value"]==value and peak["locations"]==locations
    assert "exact ==" in output["peaks"]["tie_policy"]
    for rows,regions in (("frames",("whole","outside_nominal_support")),("adjacent",("whole","outside_union_nominal_support"))):
        for row in original[rows]:
            for region in regions:row[region]=stats(0.0,0.0)
    zeros=runtime.with_quality_peaks(original)
    assert zeros["peaks"]["frames"]["whole"]["rms"]["value"]==0.0
    assert len(zeros["peaks"]["frames"]["whole"]["rms"]["locations"])==3
    original["frames"][0]["whole"]["rms"]=None
    missing=runtime.with_quality_peaks(original)["peaks"]["frames"]["whole"]
    assert missing["rms"]["status"]=="UNAVAILABLE" and missing["rms"]["value"] is None and missing["rms"]["locations"]==[]
    assert missing["linf"]["status"]=="SCORED"


@pytest.mark.parametrize("status",["NOT_ENTERED","FAILED","INTERRUPTED"])
def test_quality_peaks_unavailable_never_fabricate_positions(status):
    quality=runtime.with_quality_peaks(runtime.shared.unavailable_quality(status))
    assert quality["status"]==status
    for group in ("frames","adjacent"):
        for region in quality["peaks"][group].values():
            for peak in region.values():
                assert peak["status"]=="UNAVAILABLE" and peak["source_status"]==status
                assert peak["value"] is None and peak["locations"]==[]
    assert len(runtime.units())==69 and len(runtime.cpu_units())==304


def test_notebook_bundle_editable_cli_and_read_only(tmp_path,monkeypatch):
    import nbformat
    notebook_path,archive_path=build()
    notebook=json.loads(notebook_path.read_text())
    nbformat.validate(nbformat.from_dict(notebook))
    codes=["".join(c["source"]) for c in notebook["cells"] if c["cell_type"]=="code"]
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for code in codes:ast.parse(code)
    assert "raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1" in codes[1]
    assert "is_relative_to" in codes[1] and "S_ISLNK" in codes[1]
    assert 'print("Quality peaks",name,row.get("peaks"' in codes[3]
    with zipfile.ZipFile(archive_path) as bundle:
        assert sorted(bundle.namelist())==sorted(SOURCE_FILES)
        for name in SOURCE_FILES:
            assert bundle.read(name)==(ROOT/name).read_bytes()
            if name.endswith('.py'):ast.parse(bundle.read(name))
        bundle.extractall(tmp_path/"editable-source")
    (tmp_path/".git").mkdir()
    cli=subprocess.run([sys.executable,"-m","experiments.wan_state_clock."+STEM+"_run","--help"],cwd=tmp_path/"editable-source",capture_output=True,text=True)
    assert cli.returncode==0,cli.stderr
    for residue in ("writer_encoding.json.tmp","cpu_ledger.json","blind/unfinished.json.tmp"):
        output=tmp_path/residue.replace('/','_')
        marker=output/residue;marker.parent.mkdir(parents=True);marker.write_text("synthetic residue")
        backend=StubBackend()
        assert runtime.run("NO","NO",output,backend=backend)["status"]=="EXISTING_ATTEMPT_WITHOUT_CHECKPOINT"
        assert not backend.calls
    source_cell=codes[1].replace("/content/drive/MyDrive/Video-WM/B-Local-Payload-Protection-V1/fixed-clean-triple-v1",str(output))
    import urllib.request
    monkeypatch.setattr(urllib.request,"urlretrieve",lambda *args:pytest.fail("existing records must skip download"))
    scope={}
    exec(compile(source_cell,"existing-source-cell","exec"),scope)
    assert scope["EXISTING_ARTIFACTS"] and "SOURCE_ROOT" not in scope
    monkeypatch.setattr(subprocess,"Popen",lambda *args,**kwargs:pytest.fail("existing records must skip runner"))
    exec(compile(codes[2],"existing-setup-cell","exec"),scope)
