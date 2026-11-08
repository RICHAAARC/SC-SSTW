"""Synthetic engineering coverage only; no native model, media or real saved-data replay."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from main.baselines import public_compact12_v1 as p
from main.baselines.public_fpr001_v4 import batches,writer_batches,read_batch,views as old_views
from runtime.wan.native_baseline_storage_v1 import read_json,write_json
from runtime.wan.native_baseline_storage_v4 import Budget,UnitStore,Unavailable
from runtime.wan.native_baseline_compact_reuse_v1 import Reuse,CompactOperations,ReadOnlyStore
from runtime.wan.native_baseline_io_v4 import saved_raw
from runtime.wan import native_baselines_compact_v1 as r
ROOT=Path(__file__).resolve().parents[1]


def test_frozen_matrix_budget_queries(tmp_path):
    roster=read_json(ROOT/'main/baselines/protocols/public_native_fpr001_v4.sources.json')['sources']
    sources=p.selected_sources(roster);v=p.views();limits=p.physical_budget()
    assert len(sources)==532 and sources[0]['global_ordinal']==65 and sources[31]['global_ordinal']==96
    assert len([s for s in sources if s['split']=='TEST'])==500
    assert len(v)==12 and len(old_views())==40 and sum(x['frames'] for x in v)==1838
    assert sum(x['frames']-1 for x in v)==1826
    assert sum(len(batches(x['frames'],8)) for x in v)==236
    assert sum(len(batches(x['frames'],32)) for x in v)==64
    assert limits['base']['detect_wam']==(32+500*2)*236
    assert limits['base']['detect_pixelseal']==(32+500*3)*64
    assert limits['base']['detect_videoseal']==(32+500*2)*64
    assert limits['base']['video_encode']==(32+500*5)*12
    assert limits['base']['video_decode']==(32*3+500*7)*11+(32+500*5)
    assert limits['base']['lpips']==500*4*3*12
    assert limits['base']['model_load']==3*(2+32)
    assert sum(x['upper'] for x in p.nominal_bounds().values())<.001
    old=tmp_path/'old';old.mkdir();queries={}
    from main.baselines.public_fpr001_v4 import allocate_queries
    queries=allocate_queries(roster,lambda n:b'\x12'*n);write_json(old/'queries.json',queries)
    out=tmp_path/'out';out.mkdir()
    selected,full,saved,plans=r.inputs(out,old)
    assert saved==queries and len(selected)==532 and full==roster
    assert read_json(out/'query_origin.json')['draws_this_package']==0
    assert all(t['source'] not in {s['id'] for s in roster[:64]} for t in r.CompactCursor(out,selected).tasks)
    assert not (old/'units').exists()


def fixture(tmp_path,complete=True):
    old=tmp_path/'old';out=tmp_path/'out';out.mkdir(parents=True);oldcache=tmp_path/'oldcache'
    budget=Budget(old,'old');unit='native/S/wam/OFF';store=UnitStore(old,unit,budget)
    view=p.views()[0];epoch=dict(id='epoch01',input=str(oldcache/'missing.npy'),whole_view_recovery=False)
    write_json(store.folder/'views'/view['id']/'epochs.json',[epoch])
    for start,n in batches(view['frames'],8)[:None if complete else 1]:
        vals=dict(bits=np.zeros((n,32),np.uint8),valid=np.ones(n,bool),selected_count=np.ones(n),
                  center_count=np.ones(n),sums=np.zeros((n,32)),means=np.zeros((n,32)))
        store.perform(view['id']+'/epoch01/batch'+str(start),'detect_wam',lambda v=vals:v,arrays=True)
    b=Budget(out,'new',limits=p.physical_budget());reuse=Reuse(out,old,oldcache)
    ops=CompactOperations(out,tmp_path/'newcache',b,None,reuse)
    return old,out,ops,view


@pytest.mark.parametrize('complete',[True,False])
def test_saved_epoch_reuse_zero_model_and_old_immutable(tmp_path,monkeypatch,complete):
    old,out,ops,view=fixture(tmp_path,complete);before={str(f.relative_to(old)):f.read_bytes() for f in old.rglob('*') if f.is_file()}
    store=ops.store('native/S/wam/OFF');imported=ops.reuse.view(store,view)
    raw,epoch=saved_raw(store,view['id'],181,'wam');assert raw['observed'].sum()==(181 if complete else 8)
    cursor=SimpleNamespace(state={},view=lambda *a:None)
    run=object.__new__(r.CompactRunner);run.ops=ops;run.cursor=cursor;run.plans={'S':{}};run.queries={}
    run.progress=lambda **k:None
    run.get_adapter=lambda *a:pytest.fail('old complete/partial without saved input must not load model')
    monkeypatch.setattr(r,'write_post',lambda *a:None)
    result=run.read_arm_compact(dict(id='S'),'wam','OFF','TEST-00',[0])
    assert result==([] if complete else [0]) and dict(ops.budget.base)=={}
    assert before=={str(f.relative_to(old)):f.read_bytes() for f in old.rglob('*') if f.is_file()}
    assert not (old/'units/new').exists()
    ReadOnlyStore(old,'absent');assert not (old/'units/absent').exists()


def test_imported_failures_and_missing_writer_not_retried(tmp_path):
    old,out,ops,view=fixture(tmp_path)
    folder=old/'units/writer/S/wam';folder.mkdir(parents=True)
    from runtime.wan.native_baseline_storage_v4 import append_json
    append_json(folder/'calls.jsonl',dict(operation='batch0',status='STARTED',physical_key='batch0'))
    append_json(folder/'calls.jsonl',dict(operation='batch0',status='FAILED_PERMANENT'))
    store=ops.store('writer/S/wam',cache_arrays=True)
    with pytest.raises(Unavailable):store.perform('batch0','embed_wam',lambda:pytest.fail('old failure retried'))
    with pytest.raises(Unavailable):ops.writer(dict(id='S'),'wam',None,[0]*32,None)
    assert dict(ops.budget.base)=={}


def test_fresh_reader_only_twelve_views(tmp_path,monkeypatch):
    old=tmp_path/'old';out=tmp_path/'out';out.mkdir()
    ops=CompactOperations(out,tmp_path/'cache',Budget(out,'test',limits=p.physical_budget()),None,Reuse(out,old,tmp_path/'oldcache'))
    run=object.__new__(r.CompactRunner);run.ops=ops;run.cursor=SimpleNamespace(state={},view=lambda *a:None)
    run.plans={'S':{}};run.queries={};run.progress=lambda **k:None;run.get_adapter=lambda *a:object()
    called=[]
    def extract(ops,source,b,arm,view,adapter,pixels):
        called.append(view['id']);return dict(observed=np.ones(view['frames'],bool)),dict(id='synthetic')
    monkeypatch.setattr(r,'extract_view',extract);monkeypatch.setattr(r,'write_post',lambda *a:None)
    monkeypatch.setattr(r,'read_saved_view',lambda store,v,b:(dict(observed=np.full(v['frames'],v['id'] in called,bool)),None,[]))
    assert run.read_arm_compact(dict(id='S'),'wam','OFF','CAL-00',0)==[]
    assert called==list(p.VIEW_IDS)


def test_report_missing_collision_and_dev_reference(tmp_path,monkeypatch):
    from runtime.wan import native_baseline_compact_tables_v1 as t
    monkeypatch.setattr(t,'action_sensitivity',lambda *a:dict(synthetic=True))
    sources=[dict(id='C',split='CAL',action='x'),dict(id='T',split='TEST',action='y')]
    dev=dict(id='D',split='DEV',action='z');out=tmp_path/'out';out.mkdir();old=tmp_path/'old'
    q={m+'/'+s['id']:dict(wrong_query_collision=s['split']=='TEST') for m in p.METHODS for s in sources}
    result=t.build_tables(out,sources,sources+[dev],q,Budget(out,'check',limits=p.physical_budget()),old)
    assert result['logical_views']==144 and result['execution_incomplete_views']==144
    assert result['status']=='INCOMPLETE_RETAINED' and not old.exists()
    table=read_json(out/'tables/B_ownership.json')
    assert all(x['channel'] not in ('FAMILY40','ALL40_ROBUST') for x in table)
    collision=[x for x in table if x['verification']=='wrong_query']
    assert all(x['planned']==1 and x['not_applicable']==1 and x['eligible']==0 and x['positive']==0 for x in collision)
    assert len(read_json(out/'tables/DEV_saved_reference.json'))==96


def test_notebooks_and_distinct_default_entry(tmp_path):
    import importlib.util
    from runtime.wan.paper_suite_colab_v1 import FULL_STAGES,FPR_STAGES
    assert 'public_compact' in FULL_STAGES and 'public' not in FULL_STAGES
    assert 'public_compact' not in FPR_STAGES and 'wan' not in FPR_STAGES
    for stem in ('paper_suite_v1','native_baselines_public_compact12_v1'):
        spec=importlib.util.spec_from_file_location(stem,ROOT/'scripts'/('build_'+stem+'_notebook.py'))
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        path=mod.build(tmp_path);assert path.read_bytes()==(ROOT/'notebooks'/path.name).read_bytes()
        notebook=json.loads(path.read_text());assert ''.join(notebook['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    old=json.loads((ROOT/'notebooks/native_baselines_public_fpr001_v4.ipynb').read_text())
    assert 'native_baseline_colab_v4' in json.dumps(old)


def runner_for(ops,monkeypatch):
    run=object.__new__(r.CompactRunner);run.ops=ops;run.cursor=SimpleNamespace(state={},view=lambda *a:None)
    run.plans={'S':{}};run.queries={};run.progress=lambda **k:None
    run.get_adapter=lambda *a:pytest.fail('no fresh model for unreadable consumed evidence')
    monkeypatch.setattr(r,'write_post',lambda *a:None)
    return run


def test_unreadable_raw_and_receipts_remain_unknown(tmp_path,monkeypatch):
    old,out,ops,view=fixture(tmp_path,True)
    # A missing complete batch container is an actual read error; no reconstruction.
    for file in (old/'units/native/S/wam/OFF').glob('arrays-*.bin'):file.unlink()
    run=runner_for(ops,monkeypatch)
    assert run.read_arm_compact(dict(id='S'),'wam','OFF','TEST-00',[0])==[0]
    assert read_json(out/'units/native/S/wam/OFF/views/G1/T00/read_errors.json')['batch_errors']
    # Another old unit has an unreadable consumed receipt and no usable epoch.
    folder=old/'units/native/S/wam/wam';folder.mkdir(parents=True)
    (folder/'calls.jsonl').write_text('{bad old receipt')
    assert run.read_arm_compact(dict(id='S'),'wam','wam','TEST-00',[0,1])==[0,1]
    assert dict(ops.budget.base)=={}
    assert ops.store('native/S/wam/wam').record_errors


def test_partial_saved_input_only_never_started_batches(tmp_path,monkeypatch):
    old,out,ops,view=fixture(tmp_path,False)
    folder=old/'units/native/S/wam/OFF';ep=read_json(folder/'views/G1/T00/epochs.json')
    Path(ep[-1]['input']).parent.mkdir(parents=True);Path(ep[-1]['input']).touch()
    from runtime.wan.native_baseline_storage_v4 import append_json
    for status in ('STARTED','INFRA_FAILED'):
        append_json(folder/'calls.jsonl',dict(operation='G1/T00/epoch01/batch8',physical_key='G1/T00/batch8',status=status))
    run=runner_for(ops,monkeypatch);calls=[]
    def detect(rgb):
        n=len(rgb);calls.append(n)
        return dict(bits=np.zeros((n,32),np.uint8),valid=np.ones(n,bool),selected_count=np.ones(n),
                    center_count=np.ones(n),sums=np.zeros((n,32)),means=np.zeros((n,32)))
    run.get_adapter=lambda *a:SimpleNamespace(detect=detect)
    monkeypatch.setattr(r,'load_rgb',lambda *a:np.zeros((181,1),np.uint8))
    assert run.read_arm_compact(dict(id='S'),'wam','OFF','TEST-00',[0])==[0]
    assert len(calls)==21 and sum(calls)==165 and ops.budget.base['detect_wam']==21
    raw,epoch=saved_raw(ops.store('native/S/wam/OFF'),'G1/T00',181,'wam')
    assert raw['observed'].sum()==173 and not raw['observed'][8:16].any()
    # Never-started tail now has own durable results; reentry performs zero additional readers.
    run.get_adapter=lambda *a:pytest.fail('reentry redid old or newly completed batches')
    assert run.read_arm_compact(dict(id='S'),'wam','OFF','TEST-00',[0])==[0]
    assert ops.budget.base['detect_wam']==21


def test_bad_dev_quality_does_not_block_and_legacy_segment(tmp_path,monkeypatch):
    from runtime.wan import native_baseline_compact_tables_v1 as t
    dev=dict(id='D',split='DEV',action='z');old=tmp_path/'old'
    bad=old/'units/quality/D/wam/quality/precodec.json';bad.parent.mkdir(parents=True);bad.write_text('[]')
    rows=t.quality_rows(old,[dev]);assert len(rows)==12 and rows[0]['read_error'] and rows[0]['observed_frames']==0
    write_json(bad,dict(frames=[dict(mse='bad')],lpips_observed_frames=181))
    assert t.quality_rows(old,[dev])[0]['lpips_observed_frames']==0
    old,out,ops,view=fixture(tmp_path/'legacy',True)
    log=old/'units/native/S/wam/OFF/calls.jsonl'
    records=[json.loads(l) for l in log.read_text().splitlines()]
    for row in records:row.pop('array_file',None);row.pop('array_record',None)
    log.write_text(''.join(json.dumps(v)+'\n' for v in records))
    store=ops.store('native/S/wam/OFF');ops.reuse.view(store,view)
    assert saved_raw(store,'G1/T00',181,'wam')[0]['observed'].all()
