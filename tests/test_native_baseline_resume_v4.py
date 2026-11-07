"""Narrow artificial CPU checks for resuming v4, no model/codec/media execution."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest

pytestmark=pytest.mark.unit
from runtime.wan import native_baselines_public_v4 as runner
from runtime.wan import native_baseline_colab_v4 as colab
from runtime.wan.native_baseline_storage_v4 import ArrayLog, UnitStore
from runtime.wan.native_baseline_storage_v1 import write_json, read_json
from main.baselines.public_fpr001_v4 import views


def test_scheduling_never_opens_arrays_and_legacy_is_cached(tmp_path):
    source={'id':'one'};folder=tmp_path/'units/native/one/wam/OFF'
    for v in views():write_json(folder/'views'/v['id']/'status.json',dict(complete=True,observed=v['frames'],planned=v['frames']))
    with patch.object(runner,'UnitStore',side_effect=AssertionError('no arrays in scheduling')):
        assert runner.scheduling_receipt(tmp_path,source,'wam','OFF')
    for f in (folder/'views').rglob('status.json'):f.unlink()
    assert runner.scheduling_receipt(tmp_path,source,'wam','OFF')
    assert not runner.native_complete(tmp_path,source,'wam','OFF')  # hint is not scientific evidence


def test_arrays_lazy_header_seek_and_one_payload(tmp_path):
    folder=tmp_path/'units/u';folder.mkdir(parents=True)
    log=ArrayLog(folder/'arrays-old.bin')
    log.append('a',{'x':np.arange(2)})
    log.append('b',{'x':np.arange(3)})
    with patch.object(ArrayLog,'read_record',wraps=ArrayLog.read_record) as load:
        store=UnitStore(tmp_path,'u',SimpleNamespace(runtime_id='new'))
        assert load.call_count==0
        assert len(store.arrays)==2 and load.call_count==0
        assert store.arrays['b']['arrays']['x'].tolist()==[0,1,2]
        assert load.call_count==1


def test_lazy_plans_do_not_read_unneeded_source(tmp_path):
    sources=[dict(id='a',split='DEV'),dict(id='b',split='DEV')]
    write_json(tmp_path/'editor/a/plan.json',{'saved':'a'})
    (tmp_path/'editor/b').mkdir();(tmp_path/'editor/b/plan.json').write_text('broken but unneeded')
    plans=runner.SourcePlans(tmp_path,sources)
    assert len(plans)==0 and plans['a']=={'saved':'a'} and len(plans)==1


def test_progress_old_runtime_is_never_current(tmp_path):
    write_json(tmp_path/'runtime_owner.json',dict(id='new',pid=42,started_at=200))
    write_json(tmp_path/'progress.json',dict(runtime_id='old',at=100,source='056'))
    row,last=colab.current_progress(tmp_path,42,200)
    assert row['status']=='WAITING_FOR_CURRENT_PROGRESS' and row['historical_progress']['source']=='056'
    write_json(tmp_path/'progress.json',dict(runtime_id='new',at=201,stage='RESTORE_PHYSICAL_BUDGET',processed=7))
    row,last=colab.current_progress(tmp_path,42,200,last)
    assert row['status']=='CURRENT_PROGRESS' and row['processed']==7
    row,last=colab.current_progress(tmp_path,42,200,last)
    assert row['status']=='CURRENT_STAGE_UNCHANGED'


def test_old_source_cleanup_keeps_active_and_results(tmp_path):
    work=tmp_path/'work';work.mkdir();(work/'project').mkdir();(work/'current-source-idle').mkdir()
    keep=work/'current-source-busy';keep.mkdir()
    (work/'unrelated').mkdir();(work/'current-source-alias').symlink_to(work/'unrelated',target_is_directory=True)
    results=tmp_path/'Video-WM_Result';results.mkdir();(results/'keep').write_text('evidence')
    (work/'current-source-link').symlink_to(results,target_is_directory=True)
    proc=tmp_path/'proc';(proc/'123').mkdir(parents=True)
    (proc/'123/cwd').symlink_to(keep,target_is_directory=True)
    (proc/'123/cmdline').write_bytes(b'python');(proc/'123/environ').write_bytes(b'')
    report=colab.clean_stopped_sources(work,proc)
    assert len(report['removed'])==2 and keep.is_dir() and (work/'unrelated').is_dir()
    assert (results/'keep').read_text()=='evidence' and (work/'current-source-link').exists()


def test_completed_views_do_not_extract_or_post_again(tmp_path):
    source={'id':'one','split':'DEV'}
    for v in views():
        write_json(tmp_path/'units/native/one/wam/OFF/views'/v['id']/'status.json',dict(complete=True,observed=v['frames'],planned=v['frames']))
        write_json(tmp_path/'post/one/wam/OFF'/(v['id']+'.json'),{'saved':True})
    ops=SimpleNamespace(output=tmp_path,cache=tmp_path/'cache',budget=SimpleNamespace(runtime_id='new'))
    run=runner.Runner(ops,[source],{}, {}, None,None)
    with patch.object(runner,'extract_view',side_effect=AssertionError('must skip completed view')):
        run.read_arm(source,'wam','OFF',None)
    assert read_json(tmp_path/'units/native/one/wam/OFF/scheduling.json')['complete']


def test_final_post_does_not_trust_completion_hint(tmp_path):
    from runtime.wan import native_baseline_evaluation_v4 as evaluation
    folder=tmp_path/'units/native/a/wam/OFF';folder.mkdir(parents=True)
    write_json(folder/'scheduling.json',{'complete':True})
    view=dict(id='G1/T00',frames=2,operation='T00',generation='G1')
    write_json(folder/'views/G1/T00/status.json',dict(complete=True,observed=2,planned=2))
    with patch.object(evaluation,'views',return_value=[view]),patch.object(evaluation,'write_post') as post:
        evaluation.refresh_post(tmp_path,[dict(id='a',split='CAL')],{}, {'a':{}},SimpleNamespace(runtime_id='check'))
    assert not post.call_args.args[5]['observed'].any()
    assert not read_json(folder/'scheduling.json')['complete']
    assert read_json(folder/'views/G1/T00/status.json')['observed']==0


def test_bootstrap_refreshes_source_without_raw_prescan(tmp_path,monkeypatch):
    output=tmp_path/'result';work=tmp_path/'work';cache=tmp_path/'cache'
    monkeypatch.setattr(colab,'OUTPUT',output);monkeypatch.setattr(colab,'WORK',work);monkeypatch.setattr(colab,'CACHE',cache)
    write_json(output/'result.json',dict(status='ALL_NATIVE_OBSERVED',quality_planned_pairs=181,quality_complete_stages=1))
    downloaded=[];launched=[]
    def download(repo,ref,destination):
        downloaded.append(destination);destination.mkdir(parents=True);return destination
    monkeypatch.setattr(colab,'source_download',download)
    monkeypatch.setattr(colab,'clean_stopped_sources',lambda: {})
    monkeypatch.setattr(colab,'tables_ready',lambda out:False)
    monkeypatch.setattr(colab,'show_results',lambda:None)
    monkeypatch.setattr(colab,'stop_group',lambda worker:None)
    monkeypatch.setattr(colab.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    def popen(args,**kwargs):
        launched.append(kwargs['cwd']);return SimpleNamespace(pid=42,returncode=0,poll=lambda:0)
    monkeypatch.setattr(colab.subprocess,'Popen',popen)
    with patch.object(runner,'native_complete',side_effect=AssertionError('no bootstrap raw scan')):
        assert colab.run_colab()['status']=='RUN_FINISHED_WITH_TABLES'
        assert colab.run_colab()['status']=='RUN_FINISHED_WITH_TABLES'
    assert len(launched)==2 and launched[0]!=launched[1]
    assert all(not path.exists() for path in downloaded)
    assert (output/'result.json').is_file()


def test_nonessential_source_cleanup_failure_does_not_block(tmp_path):
    work=tmp_path/'work';(work/'project').mkdir(parents=True)
    proc=tmp_path/'proc';proc.mkdir()
    with patch.object(colab.shutil,'rmtree',side_effect=OSError('artificial locked temporary source')):
        row=colab.clean_stopped_sources(work,proc)
    assert row['removed']==[] and len(row['cleanup_errors'])==1
    assert (work/'project').is_dir()
