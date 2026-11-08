"""Artificial CPU-only continuation and ownership checks; no real data/model/codec."""
from collections import Counter
from pathlib import Path
import importlib.util
import json
import numpy as np
import pytest
from main.tube_state import paper_execution_v1 as schedule,paper_suite_v1 as p
from main.baselines import public_compact12_v1 as public
from runtime.wan import paper_suite_records_v1 as rec
ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tests'/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def roster():return public.selected_sources(rec.read(ROOT/'main/baselines/protocols/public_native_fpr001_v4.sources.json')['sources'])


def test_static_ownership_and_total_allowances():
    sources=roster();limits=[schedule.shard_budget(sources,i) for i in range(4)]
    assert set(schedule.main_sources(0)).isdisjoint(schedule.main_sources(1))
    assert set(schedule.main_sources(0)+schedule.main_sources(1))==set(p.SOURCES)
    assert sum(len(v['source_ids']) for v in limits)==532
    assert len(set(s for v in limits for s in v['source_ids']))==532
    for field in ('base','recovery'):
        assert dict(sum((Counter(v[field]) for v in limits),Counter()))=={k:v for k,v in public.physical_budget()[field].items() if v}
    assert sum(v['runtime_limit'] for v in limits)==65
    events=[dict(status='STARTED',unit='source/'+sources[0]['id'],operation='download',category='download',count=1,recovery=False),
        dict(status='STARTED',unit='loaders',operation='TEST-00/wam',category='model_load',count=1,recovery=True)]
    spent=schedule.inherited_spend(sources,events)
    parts=[schedule.shard_budget(sources,i,spent,5) for i in range(4)]
    assert sum(v['runtime_limit'] for v in parts)==60
    assert sum(v['recovery']['model_load'] for v in parts)==public.physical_budget()['recovery']['model_load']
    assert sum(v['inherited']['base'].get('download',0) for v in parts)==1
    with pytest.raises(ValueError):schedule.inherited_spend(sources,[dict(status='UNREADABLE_EVENT')])


def test_shard_cursors_keep_shared_pack_work_together(tmp_path):
    from runtime.wan.native_baseline_compact_shards_v1 import ShardCursor
    tasks=[]
    for shard in range(4):
        cursor=ShardCursor(tmp_path/str(shard),roster(),shard)
        own={x['id'] for x in schedule.owned_packs(roster(),shard)}
        assert all(t['pack'] in own for t in cursor.tasks if t['kind']!='post')
        tasks.extend(t['id'] for t in cursor.tasks if t['kind']!='post')
        for pack in own:
            group=[t for t in cursor.tasks if t['pack']==pack]
            assert {t['backend'] for t in group if t['kind']=='native'}==set(p.BACKENDS)
        cursor.state.update(next_task=5,active_task=None,pending={'0':[1]})
        assert cursor.queue()[0][0]==5
    assert len(tasks)==len(set(tasks))


def test_main_worker_source_boundary_and_no_cross_shard(tmp_path):
    from runtime.wan.paper_execution_v1 import main_worker
    cohort=tmp_path/'c';wan=tmp_path/'w';calls=[]
    def c(sid,**kw):calls.append(('c',sid));rec.write(cohort/sid/'attempt01/result.json',dict(status='COMPLETED'));return dict(status='COMPLETED')
    def w(sid,*args,**kw):calls.append(('w',sid));rec.write(wan/sid/'attempt01/result.json',dict(status='COMPLETED'));return dict(status='COMPLETED')
    for i in range(2):
        result=main_worker(0,1,100,output=tmp_path/'o',cohort_root=cohort,wan_root=wan,cohort_call=c,wan_call=w)
        assert result['status']=='PAUSED_BOUNDARY'
    assert calls==[(kind,sid) for sid in schedule.main_sources(0)[:2] for kind in ('c','w')]


def test_cohort_stage_and_identity_operation_resume(tmp_path,monkeypatch):
    from runtime.wan import b_paper_cohort_resume_v1 as resume,b_paper_cohort_worker_v1 as worker
    from runtime.wan import paper_checkpoint_v1 as checkpoint
    from main.tube_state import b_paper_cohort_v1 as cp
    fake=module('test_b_paper_cohort_v1');sid=cp.ROSTER[0]['id'];output=tmp_path/'c'/sid/'attempt01'
    fake.FakeIdentity.output=output;stage_calls=[];identity_calls=0;original=checkpoint.Boundary
    def stage(stage,out,cache,seconds,limit,session):
        nonlocal identity_calls
        stage_calls.append(stage)
        if stage=='identity':identity_calls+=1
        class Limited(original):
            def __init__(self,*a,**kw):super().__init__(*a,**kw);self.max_items=35 if stage=='identity' and identity_calls==1 else None
        original_journal=checkpoint.DurableJournal
        class LocalBoundary(original_journal):
            def call(self,key,function):
                value=super().call(key,function)
                if stage=='identity' and identity_calls==1 and key=='input10/B_NEW/q0/local':self.boundary.max_items=self.boundary.completed
                return value
        with monkeypatch.context() as m:
            m.setattr(checkpoint,'DurableJournal',LocalBoundary)
            return worker.worker(stage,out,cache,generation_factory=fake.FakeGeneration,time_factory=fake.FakeTime,identity_factory=fake.FakeIdentity)
    kw=dict(output_root=tmp_path/'c',cache_root=tmp_path/'cache',owner_root=tmp_path/'owner',worker_runner=stage)
    first=resume.run_source(sid,max_stages=1,**kw);assert first['status']=='PAUSED_BOUNDARY'
    assert (tmp_path/'cache'/sid/'OFF.fixture').is_file()
    second=resume.run_source(sid,**kw);assert second['status']=='PAUSED_BOUNDARY'
    saved_local=checkpoint.load_value(rec.read(output/'checkpoints/returns/input10/B_NEW/q0/local.json')['value'])
    assert saved_local['memo']
    count=rec.read(output/'state.json');done={k for k,v in count.items() if v['status']=='RETURNED'}
    third=resume.run_source(sid,**kw);assert third['status']=='COMPLETED',third
    assert stage_calls==['prepare','solve','replay','receive','identity','identity']
    rows=rec.read(output/'state.json');assert all(v['started']==v['returned']==1 for v in rows.values())
    assert done and all(rows[k]['started']==1 for k in done)
    assert not (tmp_path/'cache'/sid).exists() and (output/'raw_terminal_slots.json').is_file()
    inner=rec.read(output/'stages/prepare/internal_calls.json')
    assert inner['transformer_conditional']['started']==100
    assert resume.run_source(sid,**kw)['status']=='READ_ONLY'


def test_cohort_unknown_call_or_stage_does_not_restart(tmp_path):
    from runtime.wan import b_paper_cohort_resume_v1 as resume,b_paper_cohort_records_v1 as cr
    from main.tube_state import b_paper_cohort_v1 as cp
    sid=cp.ROSTER[0]['id'];out=tmp_path/'c'/sid/'attempt01';cr.reserve(out,cp.source_config(sid))
    rec.write(out/'checkpoint_control.json',dict(cache=str(tmp_path/'cache'/sid),stages={}))
    rows=rec.read(out/'state.json');rows['generate_OFF'].update(status='STARTED',started=1);rec.write(out/'state.json',rows)
    result=resume.run_source(sid,output_root=tmp_path/'c',cache_root=tmp_path/'cache',owner_root=tmp_path/'o',worker_runner=lambda *a:pytest.fail('uncertain consumed call repeated'))
    assert result['status']=='STOPPED_FIXED_ATTEMPT' and 'UNRECOVERABLE' in result['failure']
    assert rec.read(out/'state.json')['generate_OFF']['started']==1


def test_wan_values_queries_and_counts_survive_pause(tmp_path,monkeypatch):
    from runtime.wan import paper_wan_comparison_v1 as wan,paper_checkpoint_v1 as checkpoint
    fake=module('test_paper_suite_v1');sid=p.SOURCES[0];cohort=tmp_path/'c';output=tmp_path/'w';models=[];media=fake.FakeMedia()
    rec.write(cohort/sid/'attempt01/result.json',dict(status='COMPLETED'))
    monkeypatch.setattr(wan,'received_shape',lambda n:(n,1,1,3));monkeypatch.setattr(wan,'quality',lambda a,b:dict(frames=len(a),rgb_mse=0))
    def factory(name):obj=fake.FakeAdapter(name);models.append(obj);return obj
    original=checkpoint.Boundary
    class Limited(original):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.max_items=40
    with monkeypatch.context() as m:
        m.setattr(checkpoint,'Boundary',Limited)
        first=wan.run_source(sid,cohort,output,adapter_factory=factory,media=media)
    assert first['status']=='PAUSED_BOUNDARY'
    queries=(output/sid/'attempt01/queries.json').read_bytes()
    second=wan.run_source(sid,cohort,output,adapter_factory=factory,media=media)
    assert second['status']=='COMPLETED',second
    assert (output/sid/'attempt01/queries.json').read_bytes()==queries
    assert media.calls==dict(encode=20,probe=26,decode=26)
    assert sum(m.counts['writer'] for m in models)==49 and sum(m.counts['reader'] for m in models)==350
    assert sum(m.counts['load'] for m in models)==4
    rows=rec.read(output/sid/'attempt01/operations.json');assert Counter(v['kind'] for v in rows.values() if v['status']=='RETURNED')==p.WAN_BUDGET
    from runtime.wan.native_baseline_storage_v4 import events
    restore=list(events(output/sid/'attempt01/restore_overhead.jsonl'))
    assert [r['status'] for r in restore]==['STARTED','RETURNED']


def test_atomic_writes_and_active_owner(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from runtime.wan.native_baseline_storage_v4 import runtime_session
    from runtime.wan.paper_execution_v1 import active_worker_check
    target=tmp_path/'shared.json'
    with ThreadPoolExecutor(8) as pool:list(pool.map(lambda i:rec.write(target,dict(i=i,data='x'*100)),range(100)))
    assert type(rec.read(target)['i']) is int and not list(tmp_path.glob('*.pending*'))
    with runtime_session(tmp_path/'workers/main0'):
        with pytest.raises(RuntimeError):active_worker_check(True,tmp_path,tmp_path/'public')


def test_all_thin_builders_and_cpu_finalizer(tmp_path):
    from scripts.build_paper_execution_v1_notebooks import ENTRIES,build
    import nbformat,ast
    for stem in ENTRIES:
        path=build(stem,tmp_path);assert path.read_bytes()==(ROOT/'notebooks'/path.name).read_bytes()
        notebook=nbformat.read(path,as_version=4);nbformat.validate(notebook)
        assert len(notebook.cells)==7 and notebook.cells[0].source.strip()=="from google.colab import drive\ndrive.mount('/content/drive')"
        for cell in notebook.cells:
            if cell.cell_type=='code':ast.parse(cell.source);assert not cell.outputs
        if stem=='paper_finalize_v1':assert 'accelerator' not in notebook.metadata
