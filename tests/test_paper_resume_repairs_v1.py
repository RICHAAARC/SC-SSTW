"""Targeted R2 artificial tests for independent review findings."""
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import pytest
from runtime.wan import paper_checkpoint_v1 as checkpoint,paper_suite_records_v1 as rec


def test_backward_wall_clock_does_not_rewind_rng(tmp_path,monkeypatch):
    rows={key:dict(status='NOT_STARTED',started=0,returned=0) for key in ('a','b','c')}
    rec.write(tmp_path/'state.json',rows);random.seed(57);np.random.seed(57)
    wall=[1000.]
    def clock():wall[0]-=.1;return wall[0]
    monkeypatch.setattr(checkpoint.time,'time',clock)
    journal=checkpoint.DurableJournal(tmp_path,tmp_path/'state.json',tmp_path/'cache')
    draw=lambda:(random.random(),np.random.random())
    a=journal.call('a',draw);b=journal.call('b',draw);expected=draw()
    random.seed(99);np.random.seed(99)
    resumed=checkpoint.DurableJournal(tmp_path,tmp_path/'state.json',tmp_path/'cache')
    assert resumed.call('a',lambda:pytest.fail('reran a'))==a
    assert resumed.call('b',lambda:pytest.fail('reran b'))==b
    assert resumed.call('c',draw)==expected
    assert rec.read(tmp_path/'checkpoints/last_rng.json')['commit_index']==3


def test_time_backend_restores_internal_counts_and_rejects_missing(tmp_path,monkeypatch):
    from runtime.wan import b_paper_cohort_backend_v1 as backend
    internal={'input0':dict(bch_started=99,bch_returned=99,role_packets_started=7,role_packets_returned=7)}
    rec.write(tmp_path/'internal_calls.json',internal)
    rec.write(tmp_path/'state.json',{'time/input0/hard_receive':dict(status='RETURNED')})
    resources=SimpleNamespace(check=lambda:None)
    monkeypatch.setattr(backend,'GenerationBackend',lambda *a:SimpleNamespace(media_backend=lambda:SimpleNamespace(resources=resources)))
    actual=backend.TimeBackend({},tmp_path,tmp_path/'cache',None,rec.write)
    assert actual.internal==internal
    (tmp_path/'internal_calls.json').unlink()
    with pytest.raises(ValueError,match='cannot reset'):backend.TimeBackend({},tmp_path,tmp_path/'cache',None,rec.write)
    # Actual return committed before its ledger row: it also must not reset inner consumption.
    rec.write(tmp_path/'state.json',{'time/input0/hard_receive':dict(status='STARTED')})
    rec.write(tmp_path/'checkpoints/returns/time/input0/hard_receive.json',dict(saved=True))
    with pytest.raises(ValueError,match='cannot reset'):backend.TimeBackend({},tmp_path,tmp_path/'cache',None,rec.write)
    rec.write(tmp_path/'internal_calls.json',internal)
    rec.write(tmp_path/'state.json',{'time/input0/relative_compare':dict(status='RETURNED')})
    with pytest.raises(ValueError,match='cannot reset'):backend.TimeBackend({},tmp_path,tmp_path/'cache',None,rec.write)


def test_parent_decode_value_is_reused_and_missing_cache_uses_recovery(tmp_path,monkeypatch):
    from runtime.wan.native_baseline_storage_v4 import Budget,UnitStore
    from runtime.wan.native_baseline_compact_reuse_v1 import Reuse,CompactOperations
    import runtime.wan.native_baseline_compact_reuse_v1 as reuse_module
    import runtime.wan.native_baseline_io_v4 as io
    from main.baselines import public_compact12_v1 as p
    parent=tmp_path/'parent';cache=tmp_path/'cache';cache.mkdir();value=cache/'rgb.npy';np.save(value,np.zeros((2,1),np.uint8))
    source=UnitStore(parent,'media/S/OFF',Budget(parent,'parent',limits=p.physical_budget()))
    source.perform('G1/probe/c.mp4','video_probe',lambda:{})
    source.perform('G1/decode','video_decode',lambda:dict(path=str(value)))
    out=tmp_path/'shard';budget=Budget(out,'new',limits=p.physical_budget(),saved=dict(base={'video_decode':1,'video_probe':1}))
    calls=[]
    ops=CompactOperations(out,tmp_path/'newcache',budget,SimpleNamespace(probe=lambda *a,**kw:pytest.fail('duplicate probe'),decode=lambda *a:calls.append(1) or np.ones((2,1),np.uint8)),
        Reuse(out,parent,cache,same_experiment=True))
    monkeypatch.setattr(reuse_module,'load_rgb',lambda path,n:np.load(path))
    monkeypatch.setattr(io,'validate_rgb',lambda *a:None)
    store=ops.store('media/S/OFF');np.testing.assert_array_equal(ops.decoded(store,'G1/decode','c.mp4',2,tmp_path/'newcache/rgb.npy'),np.zeros((2,1)))
    assert calls==[] and budget.base['video_decode']==1 and budget.extra['video_decode']==0
    value.unlink()
    assert ops.decoded(store,'G1/decode','c.mp4',2,tmp_path/'newcache/rgb.npy').all()
    assert calls==[1] and budget.base['video_decode']==1 and budget.extra['video_decode']==1


def test_parent_missing_query_never_falls_back(tmp_path,monkeypatch):
    from runtime.wan.native_baseline_compact_shards_v1 import run_shard
    parent=tmp_path/'parent';parent.mkdir();(parent/'physical_calls.jsonl').write_text('{}\n')
    old=tmp_path/'old';rec.write(old/'queries.json',dict(other=True))
    with pytest.raises(ValueError,match='query registry'):
        run_shard(0,parent=parent,cache_root=tmp_path/'cache',old=old,old_cache=tmp_path/'oldcache',vendor_factory=lambda:pytest.fail('model preparation'))
    assert not (parent/'shards/part0/queries.json').exists()


def test_bad_parent_budget_is_unknown_not_zero(tmp_path):
    from runtime.wan.paper_execution_v1 import PublicCost
    (tmp_path/'physical_calls.jsonl').write_text('broken\n')
    value=PublicCost(tmp_path,[]).snapshot()
    assert value['status']=='UNKNOWN_CONSUMPTION' and value['base'] is None and not value['counts_complete']


def test_resource_history_retained(tmp_path):
    checkpoint.resource_record(tmp_path,'generation_resources',dict(peak=7))
    checkpoint.resource_record(tmp_path,'generation_resources',dict(peak=5))
    files=list((tmp_path/'resource_history/generation_resources').glob('*.json'))
    assert sorted(rec.read(p)['peak'] for p in files)==[5,7]
    assert rec.read(tmp_path/'generation_resources.json')['peak']==5


def test_only_finalizer_marks_public_complete(tmp_path,monkeypatch):
    from runtime.wan import paper_execution_v1 as execution,paper_suite_tables_v1 as tables,paper_query_statistics_v1 as queries
    from main.tube_state import paper_suite_v1 as p
    monkeypatch.setattr(p,'REPORT_ROOT',str(tmp_path/'report'))
    monkeypatch.setattr(queries,'run',lambda *a:dict(status='COMPLETED'))
    monkeypatch.setattr(tables,'build',lambda *a:dict(components=[dict(status='COMPLETED')]*24))
    monkeypatch.setattr(execution,'collect_public',lambda *a:dict(status='INCOMPLETE_RETAINED'))
    value=execution.finalize(output=tmp_path/'exec',cohort_root=tmp_path/'c',wan_root=tmp_path/'w',query_root=tmp_path/'q',public_root=tmp_path/'public')
    assert value['status']=='PARTIAL_REPORT_SAVED'


def test_finalizer_collects_parent_and_shard_cost_logs(tmp_path,monkeypatch):
    from runtime.wan import paper_execution_v1 as execution,native_baseline_compact_tables_v1 as tables
    from runtime.wan.native_baseline_storage_v4 import append_json
    parent=tmp_path/'parent';part=parent/'shards/part1'
    append_json(parent/'model_resources.jsonl',dict(backend='wam',peak=9))
    append_json(part/'model_resources.jsonl',dict(backend='pixelseal',peak=7))
    append_json(part/'quality_resources.jsonl',dict(forwards=4))
    append_json(part/'model_loads.jsonl',dict(status='STARTED'))
    def build(destination,sources,roster,queries,budget,old,source_roots):
        assert len(sources)==532 and len(source_roots)==532
        rec.write(Path(destination)/'tables/D_resources.json',budget.snapshot())
        return dict(status='INCOMPLETE_RETAINED')
    monkeypatch.setattr(tables,'build_tables',build)
    dest=tmp_path/'final';execution.collect_public(dest,parent,tmp_path/'old')
    value=rec.read(dest/'tables/D_resources.json')
    assert {row['peak'] for row in value['model_resources']}=={9,7}
    assert value['quality_resources'][0]['forwards']==4 and len(value['model_loads'])==1


def test_cohort_notebook_does_not_skip_last_paused_checkpoint(tmp_path):
    from scripts.build_b_paper_cohort_v1_notebook import build
    import json
    root=tmp_path/'results';source=tmp_path/'source'
    for i in range(1,5):
        out=root/f'paper-joint-t{i:02d}'/'attempt01';rec.write(out/'attempt.json',dict(status='COMPLETED'))
    rec.write(root/'paper-joint-t04/attempt01/checkpoint_control.json',dict(cache='fake'))
    rec.write(root/'paper-joint-t04/attempt01/result.json',dict(status='PAUSED_BOUNDARY'))
    bootstrap=source/'runtime/wan/b_paper_cohort_colab_v1.py';bootstrap.parent.mkdir(parents=True)
    bootstrap.write_text("def run_colab(**kwargs):return dict(status='FAKE_CONTINUED',arguments=kwargs)\n")
    nb=json.loads(build(tmp_path).read_text());code=''.join(nb['cells'][4]['source'])
    code=code.replace("Path('/content/drive/MyDrive/Video-WM_Result/main_method/paper_cohort_v1')",repr(str(root)))
    # Replace only the fixture's location, leaving actual continuation selection unchanged.
    code=code.replace('OUTPUT_ROOT = '+repr(str(root)),'OUTPUT_ROOT = Path('+repr(str(root))+')')
    ns=dict(BATCH='TEST-A',SOURCE_DIRECTORY=str(source),SOURCE_REF='main',MAX_SOURCES=1,MAX_HOURS=4,PREVIOUS_RUNTIME_STOPPED=False)
    exec(compile(code,'synthetic_notebook_cell','exec'),ns)
    assert ns['RUN']['status']=='FAKE_CONTINUED'
