"""Five review reproductions; synthetic tables/fake subprocesses only."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.wan_state_clock import receiver_records as records
from scripts import notebook_common as notebook
ROOT=Path(__file__).resolve().parents[1]


def detail_and_row(R=22):
    bits=[dict(bit_index=b,payload_channel=b//8,decoded=0,ones=0,zeros=30*R,count=30*R) for b in range(32)]
    times=[dict(bit_index=b,payload_channel=b//8,receiver_latent_index=t,decoded=0,ones=0,zeros=30,count=30) for b in range(32) for t in range(1,R+1)]
    original=dict(decoded_bits=[0]*32,votes=[dict(ones=0,zeros=30*R,count=30*R) for _ in range(32)])
    detail=dict(status="READ",R=R,received_frames=89 if R==22 else 177,bit_rows=bits,time_bit_rows=times,original_readout=original)
    row=dict(status="READ",R=R,received_frames=detail["received_frames"],**copy.deepcopy(original))
    return detail,row


def test_conditional_actual_model_id_checked_versions_optional(tmp_path):
    from experiments.wan_state_clock import video_trajectory_conditional_joint_v1_run as run
    cfg=records.read(run.CONFIG);cfg.pop("environment_pins",None);cfg["model"].pop("revision",None);cfg["model"]["role"]="optional changed note"
    p=tmp_path/"selected.json";p.write_text(json.dumps(cfg));assert run.load_config(p)["model"]["id"]==cfg["model"]["id"]
    cfg["model"]["id"]="different-receiver";p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError,match="receiver model id"):run.load_config(p)

@pytest.mark.parametrize("R",[22,44])
def test_saved_detail_support_valid_without_hashes_or_truth(tmp_path,R):
    detail,row=detail_and_row(R);detail["revision"]="note";row["sha256"]="not compared"
    p=tmp_path/"detail.json";records.dump(p,detail)
    assert records.checked_detail(p,row)["status"]=="READ"

@pytest.mark.parametrize("change",["decoded","failed_promoted","short_bits","duplicate_bit","short_time","duplicate_time","time_index","unknown_bit_status","saved_votes","time_totals"])
def test_saved_detail_known_contradictions_are_rejected(tmp_path,change):
    detail,row=detail_and_row()
    if change=="decoded":
        # Internally valid detail counts, but the completed saved decision is zero.
        for item in detail["bit_rows"]+detail["time_bit_rows"]:item.update(decoded=1,ones=item["count"],zeros=0)
    elif change=="failed_promoted":row["status"]="FAILED"
    elif change=="short_bits":detail["bit_rows"]=detail["bit_rows"][:1]
    elif change=="duplicate_bit":detail["bit_rows"][1]=copy.deepcopy(detail["bit_rows"][0])
    elif change=="short_time":detail["time_bit_rows"].pop()
    elif change=="duplicate_time":detail["time_bit_rows"][1]=copy.deepcopy(detail["time_bit_rows"][0])
    elif change=="time_index":detail["time_bit_rows"][0]["receiver_latent_index"]=0
    elif change=="unknown_bit_status":detail["bit_rows"][0]["status"]="UNKNOWN"
    elif change=="saved_votes":row["votes"][0].update(ones=1,zeros=659)
    elif change=="time_totals":detail["time_bit_rows"][0].update(ones=1,zeros=29)
    p=tmp_path/"detail.json";records.dump(p,detail)
    with pytest.raises(ValueError):records.checked_detail(p,row)


def test_unknown_saved_row_and_failed_detail_do_not_create_success(tmp_path):
    detail,row=detail_and_row();row["status"]="UNRESOLVED";detail["status"]="FAILED";detail.pop("bit_rows");detail.pop("original_readout")
    for item in detail["time_bit_rows"]:
        for field in ("decoded","ones","zeros","count"):item.pop(field)
        item["status"]="FAILED"
    p=tmp_path/"failed.json";records.dump(p,detail)
    assert records.checked_detail(p,row)["status"]=="FAILED"


def test_internal_atomic_detail_write_interrupt_closes_fixed_rows(tmp_path,monkeypatch):
    from experiments.wan_state_clock import video_trajectory_internal_single_deletion_v1_run as run
    cfg=run.load_config();store=run.Store(tmp_path/"run",cfg)
    oid=next(iter(cfg["inputs"]));lid=oid+"/K0/RAW";rid="one/read";gid="one"
    plan=dict(encodes={gid:dict(observation_id=oid,received_index_map=list(range(177)),frames=177)},reads={rid:dict(encode_id=gid,key_label="K0",frames=177,logical_slots=[lid])},logical_slots={lid:dict(status="PLANNED",physical_read=rid,oracle=False)},truth_inputs=False)
    store.data["blind_plan_seal"]=records.dump(store.output/"blind_plan.json",plan)
    store.data["blind_sync_seal"]=dict(path="not consumed here",sha256=None)
    run.install_plan(store,plan)
    detail,_=detail_and_row(44)
    class Backend:
        @staticmethod
        def operate_map(source,mapping):return source
        @staticmethod
        def pixel_receipt(value):return {}
        @staticmethod
        def encode(value):return SimpleNamespace(shape=(1,))
        @staticmethod
        def cache_latent(value):return value
        @staticmethod
        def restore_latent(value):return value
        @staticmethod
        def read(*args):return detail
    actual_dump=run.dump
    def interrupt_after_atomic_write(path,value):
        receipt=actual_dump(path,value)
        if str(path).endswith("one/read.votes.json.gz"):raise KeyboardInterrupt("after atomic detail write")
        return receipt
    monkeypatch.setattr(run,"dump",interrupt_after_atomic_write)
    with pytest.raises(KeyboardInterrupt):run.run_plan(store,{oid:object()},Backend(),{},plan,"blind")
    monkeypatch.setattr(run,"dump",actual_dump)
    original=Path(store.data["physical_reads"][rid]["detail_path"]).read_bytes()
    calls=copy.deepcopy(store.data["calls"])
    run.seal_payload(store,run.BLIND,"blind_payload",False)
    assert store.data["physical_reads"][rid]["status"]=="FAILED"
    assert store.data["payload_reads"][lid]["status"]=="FAILED"
    assert store.data["physical_reads"][rid]["error"]=="INTERRUPTED_BEFORE_READ_COMMIT"
    assert Path(store.data["physical_reads"][rid]["detail_path"]).read_bytes()==original
    assert store.data["calls"]==calls and calls["payload_read"]==dict(attempted=1,completed=1)
    assert len(store.data["payload_reads"])==16
    assert all(row["status"] not in ("PENDING","RUNNING") for row in run.read(store.data["blind_payload_seal"]["path"])["logical_reads"].values())
    with pytest.raises(ValueError,match="completion"):records.checked_detail(store.data["payload_reads"][lid]["detail_path"],store.data["payload_reads"][lid])

@pytest.mark.parametrize("invalid",[None,"name","denominator"])
def test_notebook_explicit_clone_precedes_config_read_without_network(tmp_path,monkeypatch,invalid):
    source=tmp_path/"source";cfg=source/"config.json";calls=[]
    scope=dict(CONFIG_PATH=str(cfg),CONFIG_NAME="fixed",FIXED={"rows":4},SOURCE_SHA=None,SOURCE_DIR=str(source),SOURCE_URL="https://example.invalid/explicit.git",ENTRYPOINT="pkg.run",OUTPUT_ROOT=str(tmp_path/"out"),REQUIRES_MEDIA=False)
    exec(notebook.SETUP,scope)
    assert not source.exists() and "INPUT_CONFIG" not in scope
    def fake_logged(command,stage,**kw):
        calls.append(stage)
        if stage=="SOURCE_CLONE":
            entry=source/"pkg/run.py";entry.parent.mkdir(parents=True);entry.write_text("")
            cfg.write_text(json.dumps(dict(name="wrong" if invalid=="name" else "fixed",fixed_denominator={} if invalid=="denominator" else {"rows":4})))
        return 0
    scope["logged"]=fake_logged
    monkeypatch.setattr(scope["subprocess"],"check_output",lambda *a,**k:(_ for _ in ()).throw(OSError("optional note unavailable")))
    if invalid:
        with pytest.raises(ValueError):exec(notebook.ENVIRONMENT,scope)
        assert calls==["SOURCE_CLONE"]
    else:
        exec(notebook.ENVIRONMENT,scope)
        assert calls==["SOURCE_CLONE","DEPENDENCY_PROBE"] and scope["INPUT_CONFIG"]["name"]=="fixed"
        cfg.write_text('{"name":"wrong","fixed_denominator":{"rows":4}}')
        with pytest.raises(ValueError):scope["validate_config"]()


def test_grow_relative_config_cli_and_direct_worker_use_caller_absolute_path(tmp_path,monkeypatch):
    from experiments.wan_state_clock import grow_video_reference_run as run
    caller=tmp_path/"caller";caller.mkdir();selected=caller/"selected.json";selected.write_bytes(run.CONFIG.read_bytes())
    monkeypatch.chdir(caller);commands=[]
    class Stream:
        def __iter__(self):return iter(())
        def close(self):pass
    class Child:
        stdout=Stream()
        def wait(self,timeout=None):return 1
    def fake_spawn(command,**kwargs):
        actual=Path(command[command.index("--config")+1]);assert actual==selected.resolve();assert actual.read_bytes()==selected.read_bytes()
        assert kwargs["cwd"]==run.ROOT;commands.append(command);return Child()
    monkeypatch.setattr(run.subprocess,"Popen",fake_spawn)
    output=tmp_path/"cli";monkeypatch.setattr(run.sys,"argv",["fake","--config","selected.json","--output",str(output)])
    with pytest.raises(SystemExit) as exc:run.main()
    assert exc.value.code==1 and [x[-1] for x in commands]==["generation","media"]
    assert records.read(output/"result.json")["config_path"]==str(selected.resolve())
    direct=tmp_path/"direct";run.Store(direct,create=True)
    run.run_worker_phase(direct,"generation",config="selected.json")
    assert len(commands)==3
