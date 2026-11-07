"""Minimal artificial CPU checks for the user-authorized direct checkpoint semantics."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest
from runtime.wan import native_baselines_public_v4 as runner
from runtime.wan.native_baseline_checkpoint_v4 import ComputationCursor, tasks_for
from runtime.wan.native_baseline_storage_v4 import Budget, UnitStore, ArrayLog, append_json, RecoveryExhausted
from runtime.wan.native_baseline_storage_v1 import read_json,write_json
from main.baselines.public_fpr001_v4 import views
pytestmark=pytest.mark.unit


def sources():return [dict(id='ucf-dev-'+x,split='DEV') for x in ('001','056','057')]


def legacy(output):
    output.mkdir(parents=True,exist_ok=True)
    (output/'runner.log').write_text(str(dict(pack='DEV-bank',source='ucf-dev-056',backend='wam',at=100))+'\n'+
        str(dict(stage='RESTORE_VIEW_RECEIPTS',source='ucf-dev-014',backend='wam',arm='OFF',at=200))+'\n')
    append_json(output/'physical_calls.jsonl',dict(status='STARTED',unit='native/ucf-dev-056/wam/wam',
        operation='G2/T08/epoch01/batch72',category='detect_wam',recovery=False,count=1,at=101))


def test_legacy_uses_real_compute_not_restore_scan(tmp_path):
    legacy(tmp_path)
    with patch.object(Path,'glob',side_effect=AssertionError('no directory scan')),patch.object(Path,'rglob',side_effect=AssertionError('no history scan')):
        cursor=ComputationCursor(tmp_path,sources())
    task=cursor.tasks[cursor.state['active_task']]
    assert (task['source'],task['backend'],task['arm'])==('ucf-dev-056','wam','wam')
    assert views()[cursor.state['view_index']]['id']=='G2/T08'


def test_scheduler_skips_history_without_completion_checks(tmp_path,monkeypatch):
    legacy(tmp_path);b=Budget(tmp_path,'new');cursor=ComputationCursor(tmp_path,sources(),b)
    selected=cursor.tasks[cursor.state['active_task']];actions=[];loads=[]
    class Adapter:
        counts={}
        def close(self):pass
    loaders=SimpleNamespace(watermark=lambda pack,backend:(loads.append((pack,backend)),Adapter())[1],close=lambda:None)
    ops=SimpleNamespace(output=tmp_path,cache=tmp_path/'cache',budget=b,_stores={},failure=lambda *a:None)
    run=runner.Runner(ops,sources(),{}, {},loaders,None,cursor)
    run.ensure_media=lambda s,a,model:actions.append(('media',s['id'],a))
    run.read_arm=lambda s,b,a,model,selection:(actions.append(('native',s['id'],a,selection)),[])[1]
    run.quality=lambda s,a,model:actions.append(('quality',s['id'],a))
    monkeypatch.setattr(runner,'quality_complete',lambda *a:True)
    monkeypatch.setattr(runner,'scheduling_receipt',lambda *a:(_ for _ in ()).throw(AssertionError('no completed-history verification')),raising=False)
    monkeypatch.setattr(run,'source_complete',lambda *a:(_ for _ in ()).throw(AssertionError('no source sweep')),raising=False)
    monkeypatch.setattr(runner,'build_tables',lambda *a:{'status':'artificial complete'})
    assert run.run()['status']=='artificial complete'
    assert actions[0][:3]==('native','ucf-dev-056','wam')
    assert not any(a[1]=='ucf-dev-001' and a[2] in ('wam','OFF') for a in actions[:next(i for i,a in enumerate(actions) if a[0]=='media' and a[2]=='pixelseal')])
    assert cursor.state['status']=='DONE' and loads==[('DEV-bank','wam'),('DEV-bank','pixelseal'),('DEV-bank','videoseal')]


def test_committed_batch_is_not_opened_until_used(tmp_path):
    b=Budget(tmp_path,'one');store=UnitStore(tmp_path,'u',b);calls=[]
    store.perform('batch','detect_wam',lambda:(calls.append(1),{'x':np.arange(2)})[1],arrays=True)
    restored=UnitStore(tmp_path,'u',Budget(tmp_path,'two'))
    with patch.object(ArrayLog,'index',side_effect=AssertionError('no global index')),patch.object(ArrayLog,'read_record',wraps=ArrayLog.read_record) as read:
        assert restored.reusable('batch') and read.call_count==0
        assert restored.perform('batch','detect_wam',lambda:(_ for _ in ()).throw(AssertionError('no rerun')),arrays=True)['x'].tolist()==[0,1]
        assert read.call_count==1
    store.segment.unlink();again=UnitStore(tmp_path,'u',Budget(tmp_path,'three'))
    assert again.reusable('batch')
    with pytest.raises(OSError):again.perform('batch','detect_wam',lambda:(_ for _ in ()).throw(AssertionError('no replacement call')),arrays=True)
    assert calls==[1]


def test_budget_restores_only_new_tail_and_keeps_limit(tmp_path):
    import runtime.wan.native_baseline_storage_v4 as storage
    limits={'base':{'detect_wam':2},'recovery':{'detect_wam':1}}
    b=Budget(tmp_path,'one',limits);b.charge('detect_wam','u','first');saved=b.resume_state()
    b.charge('detect_wam','u','second');original=storage.events;offsets=[]
    def read(path,offset=0):offsets.append(offset);yield from original(path,offset)
    with patch.object(storage,'events',read):restored=Budget(tmp_path,'two',limits,saved=saved)
    assert offsets==[saved['ledger_offset']] and restored.base['detect_wam']==2
    with pytest.raises(RecoveryExhausted):restored.charge('detect_wam','u','third')


def test_cursor_preserves_only_remaining_views(tmp_path):
    legacy(tmp_path);b=Budget(tmp_path,'one');cursor=ComputationCursor(tmp_path,sources(),b)
    index=cursor.state['active_task'];cursor.begin(index,[2,5]);cursor.view(5,[5],[2])
    write_json(tmp_path/'progress.json',dict(stage='RESTORE_VIEW_RECEIPTS',source='ucf-dev-001'))
    again=ComputationCursor(tmp_path,sources(),b)
    assert again.queue()[0]==(index,[5]) and again.state['failed_views']==[2]
    again.finish(index,[2]);reloaded=ComputationCursor(tmp_path,sources(),b)
    assert reloaded.queue()[0]==(index,[2])


def test_thin_entry_uses_cursor_without_bootstrap_history_scans(tmp_path):
    from scripts.build_native_baselines_public_fpr001_v4_notebook import build
    n=read_json(build(tmp_path));actual=read_json(runner.ROOT/'notebooks/native_baselines_public_fpr001_v4.ipynb')
    assert n==actual and ''.join(n['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    tree=ast.parse((runner.ROOT/'runtime/wan/native_baseline_colab_v4.py').read_text())
    run=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='run_colab')
    calls={x.func.id for x in ast.walk(run) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)}
    assert not calls.intersection({'native_complete','tables_ready','clean_stopped_sources'})


def test_query_validation_only_touches_used_query():
    records=runner.SavedQueries({'wam/current':dict(query=[0]*32,wrong_query=None),
                                'wam/history':dict(query='unused malformed history')})
    assert records['wam/current']['wrong_query'] is None
    with pytest.raises(ValueError):records['wam/history']
