"""Bounded record-only regressions: no models, FFT, codecs or scientific replay."""
import ast
import copy
import importlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.wan_state_clock import receiver_records as records
from scripts import notebook_common as notebook
from runtime.wan import provenance
ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def no_scientific_execution(monkeypatch):
    import numpy as np
    import torch
    def forbidden(*a,**kw):raise AssertionError("real FFT/model/codec execution is excluded")
    for owner in (np.fft,torch.fft):
        for name in ("fft","fftn","fft2","ifft","ifftn","ifft2"):
            monkeypatch.setattr(owner,name,forbidden)


def config(name):return json.loads((ROOT/"experiments/wan_state_clock/configs"/(name+".json")).read_text())

def test_source_record_needs_neither_git_manifest_nor_file_inventory(tmp_path,monkeypatch):
    monkeypatch.setattr(provenance.subprocess,"check_output",lambda *a,**k:pytest.fail("no Git inherited from parent"))
    root=tmp_path/"source";root.mkdir();(tmp_path/".git").mkdir()
    assert provenance.source_identity(root)["git_commit"] is None
    assert provenance.source_identity(root)["files"]=={}
    assert records.digest(tmp_path/"absent") is None
    from experiments.wan_state_clock import grow_video_reference_run as grow
    monkeypatch.setattr(grow,"ROOT",root)
    store=grow.Store(tmp_path/"run",create=True,config=tmp_path/"no-location-note")
    assert store.data["config_sha256"] is None
    assert [len(store.data[k]) for k in ("generation","videos","reads","evaluations")]==[3,3,24,48]

@pytest.mark.parametrize("name",["video_trajectory_payload_framewise_sync_m05_v1","video_trajectory_receiver_estimated_align_v1","video_trajectory_internal_single_deletion_v1"])
def test_public_config_ignores_locator_versions_preserves_method(tmp_path,name):
    cfg=config(name);cfg.pop("environment_pins",None);cfg["model"].pop("revision",None)
    cfg.get("framewise_model",{}).pop("revision",None)
    cfg.get("framewise_model",{})["role"]="optional note"
    for field in list(cfg):
        if field.endswith("sha256"):cfg.pop(field)
    for spec in cfg.get("inputs",{}).values():spec.pop("sha256",None)
    p=tmp_path/"input.json";p.write_text(json.dumps(cfg));assert records.public_config(p)["name"]==name
    cfg["fixed_denominator"]={};p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="fixed public protocol"):records.public_config(p)

@pytest.mark.parametrize("module",["runtime.wan.fixed_rgb_media","runtime.wan.conditional_joint.fixed_rgb_media"])
def test_rgb_bytes_uses_actual_size_not_expected_digest(tmp_path,monkeypatch,module):
    media=importlib.import_module(module);monkeypatch.setattr(media,"RGB_BYTES",6)
    p=tmp_path/"rgb";p.write_bytes(bytes(range(6)))
    assert media._bytes(p)==bytes(range(6));assert media._bytes(p,"obsolete")==bytes(range(6))
    p.write_bytes(b"bad")
    with pytest.raises(ValueError):media._bytes(p,"obsolete")


def test_rgb_reader_missing_digest_and_real_size_failure(tmp_path):
    from runtime.wan.rgb8_source import read_rgb8_source
    p=tmp_path/"rgb";p.write_bytes(bytes(range(6)))
    assert tuple(read_rgb8_source(p,shape=(1,1,2,3)).shape)==(1,1,2,3)
    assert read_rgb8_source(p,expected_sha256="different",shape=(1,1,2,3)).tolist()==[[[[0,1,2],[3,4,5]]]]
    with pytest.raises(ValueError,match="byte count"):read_rgb8_source(p,shape=(2,1,2,3))


def test_snapshot_checks_nested_decisions_not_location_notes(tmp_path):
    rows={"A":dict(status="ESTIMATED",joint=dict(path=dict(family="H1",b=2,k=90),phase=2),votes={"0":5,"1":8},sha256="old")}
    p=tmp_path/"blind.json";records.dump(p,dict(truth_inputs=False,estimates=rows))
    changed=copy.deepcopy(rows);changed["A"]["sha256"]="different";changed["A"]["revision"]="optional"
    records.check_snapshot(p,{"estimates":changed})
    changed["A"]["joint"]["path"]["k"]=91
    with pytest.raises(ValueError,match="decision differs"):records.check_snapshot(p,{"estimates":changed})
    with pytest.raises(ValueError,match="slot set"):records.check_snapshot(p,{"estimates":{}})
    with pytest.raises(ValueError,match="blind/oracle"):records.check_snapshot(p,truth_inputs=True)


def test_detail_checks_actual_decoded_and_support(tmp_path):
    p=tmp_path/"detail.json"
    bits=[dict(bit_index=b,payload_channel=b//8,decoded=0,ones=0,zeros=660,count=660) for b in range(32)]
    times=[dict(bit_index=b,payload_channel=b//8,receiver_latent_index=t,decoded=0,ones=0,zeros=30,count=30) for b in range(32) for t in range(1,23)]
    detail=dict(status="READ",R=22,received_frames=89,bit_rows=bits,time_bit_rows=times,original_readout=dict(decoded_bits=[0]*32,votes=[dict(ones=0,zeros=660,count=660) for _ in range(32)]))
    records.dump(p,detail);row=dict(status="READ",R=22,received_frames=89,decoded_bits=[0]*32,votes=detail["original_readout"]["votes"],detail_sha256="obsolete")
    records.checked_detail(p,row);row["R"]=44
    with pytest.raises(ValueError,match="support"):records.checked_detail(p,row)
    row["R"]=22;row["decoded_bits"][0]=1
    with pytest.raises(ValueError,match="decision"):records.checked_detail(p,row)


def test_estimated_posthoc_requires_saved_blind_decisions_before_truth(tmp_path,monkeypatch):
    from experiments.wan_state_clock import video_trajectory_receiver_estimated_align_v1_run as run
    truth=tmp_path/"truth.json";truth.write_text('{"message":"OKOK"}')
    store=SimpleNamespace(output=tmp_path,cfg=dict(posthoc_config=str(truth)),data=dict(payload_reads={},physical_reads={}))
    with pytest.raises(FileNotFoundError):run.load_posthoc(store)
    records.dump(tmp_path/"blind_payload_readouts.json",dict(truth_inputs=False,logical_reads={},physical_reads={}))
    assert run.load_posthoc(store)==dict(message="OKOK")


def test_conditional_store_does_not_open_oracle_or_truth(tmp_path,monkeypatch):
    from experiments.wan_state_clock import video_trajectory_conditional_joint_v1_run as run
    cfg=config("video_trajectory_conditional_joint_v1");cfg.pop("environment_pins");cfg["_config_path"]="missing optional config note"
    for name in ("oracle","posthoc","preparation"):cfg[name+"_config"]=str(tmp_path/(name+".json"));cfg.pop(name+"_config_sha256")
    monkeypatch.setattr(run,"read",lambda path:pytest.fail("Store must not read companions"))
    store=run.Store(tmp_path/"run",cfg)
    assert len(store.data["payload_reads"])==44 and store.data["config_sha256"] is None


def test_preparation_resume_metadata_missing_and_real_semantics(tmp_path):
    from experiments.wan_state_clock import video_trajectory_conditional_joint_v1_prepare as prep
    cfg=config("video_trajectory_conditional_joint_v1_preparation")
    output=tmp_path/"prepare";store=prep.Store(output,cfg,"source")
    store.data.update(status="SOURCE_READY",source_protocol=dict(shape=[181,320,512,3]));store.save()
    changed=copy.deepcopy(cfg);changed.pop("environment_pins");changed.pop("method_version",None);changed["model"].pop("revision");changed["framewise_vae"].pop("revision")
    assert prep.Store(output,changed,"matched").data["resume_semantics"]=="CHECKED"
    data=json.loads(store.path.read_text());data.pop("preparation_inputs");data.pop("config_content_sha256",None);store.path.write_text(json.dumps(data))
    assert prep.Store(output,changed,"matched").data["resume_semantics"]=="OLD_RECORD_UNAVAILABLE"
    changed["generation"]["seed"]+=1
    with pytest.raises(ValueError,match="preparation semantics: generation"):prep.Store(output,changed,"matched")


BUILDERS=("build_grow_video_reference_notebook","build_video_trajectory_payload_framewise_sync_m05_notebook","build_video_trajectory_receiver_estimated_align_notebook","build_video_trajectory_internal_single_deletion_notebook","build_video_trajectory_conditional_joint_notebook")
@pytest.mark.parametrize("name",BUILDERS)
def test_notebook_generated_copy_schema_and_optional_source(tmp_path,name):
    import nbformat
    builder=importlib.import_module("scripts."+name);p=builder.build(output=tmp_path/"candidate.ipynb")
    assert p.read_bytes()==builder.OUTPUT.read_bytes()
    nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
    assert nb.cells[0].source.strip()=="from google.colab import drive\ndrive.mount('/content/drive')"
    for cell in nb.cells:
        if cell.cell_type=="code":ast.parse(cell.source);assert not cell.outputs and cell.execution_count is None
    builder.build(source_sha="arbitrary location note",output=p)


def test_notebook_setup_and_environment_use_existing_source_without_git(tmp_path,monkeypatch):
    source=tmp_path/"source";entry=source/"pkg/run.py";entry.parent.mkdir(parents=True);entry.write_text("")
    cfg=tmp_path/"input.json";cfg.write_text(json.dumps(dict(name="fixed",fixed_denominator={"rows":4})))
    scope=dict(CONFIG_PATH=str(cfg),CONFIG_NAME="fixed",FIXED={"rows":4},SOURCE_SHA=None,SOURCE_DIR=str(source),SOURCE_URL=None,ENTRYPOINT="pkg.run",OUTPUT_ROOT=str(tmp_path/"out"),REQUIRES_MEDIA=False)
    exec(notebook.SETUP,scope);calls=[]
    scope["logged"]=lambda cmd,stage,**kw:calls.append((cmd,stage)) or 0
    def unavailable(*a,**kw):raise OSError("optional source/environment records unavailable")
    monkeypatch.setattr(subprocess,"check_output",unavailable)
    exec(notebook.ENVIRONMENT,scope)
    assert [stage for _,stage in calls]==["DEPENDENCY_PROBE"]
    assert not any("git" in cmd or "install" in cmd for cmd,_ in calls)
    cfg.write_text('{"name":"fixed","fixed_denominator":{}}')
    with pytest.raises(ValueError,match="denominator"):scope["validate_config"]()


def test_grow_worker_close_error_is_secondary_and_latest_state_survives(tmp_path,monkeypatch):
    from experiments.wan_state_clock import grow_video_reference_run as run
    output=tmp_path/"run";run.Store(output,create=True);events=[]
    class Stream:
        def __iter__(self):raise KeyboardInterrupt("primary interruption")
        def close(self):events.append("close");raise OSError("secondary close")
    class Child:
        stdout=Stream()
        def poll(self):return None
        def terminate(self):events.append("terminate")
        def wait(self,timeout=None):
            events.append("wait");store=run.Store(output);store.data["stage"]="LAST_CHILD_CHECKPOINT";store.save();return -15
    monkeypatch.setattr(run.subprocess,"Popen",lambda *a,**k:Child())
    store,halt=run.run_worker_phase(output,"generation",config=tmp_path/"selected.json")
    assert halt=="INTERRUPTED" and events==["terminate","wait","close"]
    row=store.data["workers"]["generation"]
    assert "primary interruption" in row["error"] and "secondary close" in row["cleanup_error"]
    assert row["last_child_stage"]=="LAST_CHILD_CHECKPOINT"
    assert str(tmp_path/"selected.json") in row["command"]
    assert len(store.data["generation"])==3


def test_conditional_interruption_keeps_all_fixed_slots_without_truth_reads(tmp_path,monkeypatch):
    from experiments.wan_state_clock import video_trajectory_conditional_joint_v1_run as run
    cfg=run.load_config()
    cfg["oracle_config"]=str(tmp_path/"unread-oracle.json");cfg["posthoc_config"]=str(tmp_path/"unread-truth.json")
    def stop(store):raise KeyboardInterrupt("injected preparation interruption")
    def never(*a,**kw):pytest.fail("interrupted run cannot call models or quality")
    output=tmp_path/"interrupted"
    with pytest.raises(KeyboardInterrupt):run.run(output,cfg=cfg,source_fn=stop,framewise_type=never,wan_type=never,quality_fn=never)
    result=run.read(output/"result.json")
    assert len(result["payload_reads"])==len(result["posthoc"])==44
    assert result["counts"]["logical_final_bits"]==1408
    assert result["counts"]["logical_time_bit_rows"]==53504
    assert result["stage"]=="FINISHED"
    assert all(row["status"] not in ("PENDING","RUNNING") for row in result["payload_reads"].values())
    assert (output/"blind_payload_seal.json").is_file()

@pytest.mark.parametrize("module",["video_trajectory_internal_single_deletion_v1_run","video_trajectory_conditional_joint_v1_run"])
def test_payload_boundary_keeps_saved_decisions_before_oracle(tmp_path,module):
    run=importlib.import_module("experiments.wan_state_clock."+module)
    row=dict(status="READ",mode="RAW",oracle=False,decoded_bits=[0]*32,physical_read="physical",R=44)
    path=tmp_path/"blind.json";records.dump(path,dict(logical_reads={"slot":row},truth_inputs=False))
    store=SimpleNamespace(data=dict(payload_reads={"slot":row},blind_payload_seal=dict(path=str(path))))
    run.check_payload(store,False)
    row["decoded_bits"][0]=1
    with pytest.raises(ValueError,match="decision differs"):run.check_payload(store,False)
