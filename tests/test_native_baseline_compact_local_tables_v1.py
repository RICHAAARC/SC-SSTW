"""Synthetic I/O and publication checks; no research records or models."""
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import zipfile

import pytest

from main.baselines import public_compact12_v1 as p
from runtime.wan import native_baseline_compact_recovery_v1 as repair
from runtime.wan import native_baseline_compact_local_tables_v1 as local
from runtime.wan import native_baseline_compact_tables_v1 as tables
from runtime.wan.native_baseline_evaluation_v4 import missing_post, post_path
from runtime.wan.native_baseline_storage_v1 import read_json, write_json

pytestmark=pytest.mark.unit


def fixture(tmp_path, monkeypatch):
    ctx=object.__new__(repair.Context)
    ctx.parent=tmp_path/'original';ctx.output=tmp_path/'repair';ctx.old=tmp_path/'old'
    ctx.sources=[dict(id='C',split='CAL',action='a'),dict(id='S',split='TEST',action='a')]
    ctx.roster=ctx.sources+[dict(id='D',split='DEV',action='a')]
    ctx.by_id={s['id']:s for s in ctx.roster};ctx.query_cache={}
    monkeypatch.setattr(local.schedule,'owned_packs',lambda sources,shard:[dict(sources=sources)] if shard==0 else [])
    monkeypatch.setattr(repair,'REBUILD_ARMS',set())
    monkeypatch.setattr(tables,'action_sensitivity',lambda *args:dict(synthetic=True))
    queries={m+'/'+s['id']:dict(wrong_query_collision=False) for s in ctx.sources for m in p.METHODS}
    write_json(ctx.parent/'shards/part0/queries.json',queries)
    for source in ctx.roster:
        root=ctx.old if source['split']=='DEV' else ctx.parent/'shards/part0'
        for method in p.METHODS:
            for arm in ('OFF',) if source['split']=='CAL' else ('OFF',method):
                for view in p.views():
                    row=missing_post(source,method,arm,view)
                    row['ownership']=dict(decision=arm!='OFF',complete=True)
                    if arm!='OFF':row['wrong_query']=dict(decision=False,complete=True)
                    row['payload']['status']='SYNTHETIC_OBSERVED'
                    write_json(post_path(root,source['id'],method,arm,view['id']),row)
            if source['split']!='CAL':
                q=dict(frames=[dict(mse=.001,ssim=.9,psnr_db=30)]*181,
                       edges=[dict(residual_difference_rms=.02)]*180)
                for stage in ('precodec','G1','G2-identity'):
                    write_json(root/'units/quality'/source['id']/method/'quality'/(stage+'.json'),q)
    # A new unresolved result must win over the old complete observation.
    view=p.views()[0]
    write_json(post_path(ctx.part(0),'S','wam','wam',view['id']),missing_post(ctx.sources[1],'wam','wam',view))
    return ctx


def snapshots(*roots):
    return {str(f):f.read_bytes() for root in roots for f in root.rglob('*') if f.is_file()}


def test_local_zip_matches_original_tables_and_preserves_source_records(tmp_path,monkeypatch):
    ctx=fixture(tmp_path,monkeypatch)
    expected=repair.collect(ctx)
    expected_tables={f.name:json.loads(f.read_text()) for f in (ctx.output/'global/tables').glob('*.json')}
    before=snapshots(ctx.parent,ctx.old,ctx.output)
    result=local.collect_local(ctx,tmp_path/'local')
    assert {k:v for k,v in result.items() if k!='delivery'}==expected
    assert result['execution_incomplete_views']==1
    assert result['logical_views']==3*len(p.METHODS)*len(p.VIEW_IDS)
    archive=Path(result['delivery']['archive'])
    with zipfile.ZipFile(archive) as package:
        for name,rows in expected_tables.items():
            assert json.loads(package.read('global/tables/'+name))==rows
        assert any(n.startswith('inputs/') for n in package.namelist())
    assert all(Path(path).read_bytes()==data for path,data in before.items())
    assert read_json(ctx.output/'aggregation_latest.json')['archive']==str(archive)


def test_unreadable_selected_overlay_does_not_fall_back_or_publish(tmp_path,monkeypatch):
    ctx=fixture(tmp_path,monkeypatch)
    selected=post_path(ctx.part(0),'S','wam','wam',p.VIEW_IDS[0])
    original=local.shutil.copyfile
    def fail(src,dst):
        if src==selected:raise OSError('synthetic unreadable new record')
        return original(src,dst)
    monkeypatch.setattr(local.shutil,'copyfile',fail)
    with pytest.raises(OSError,match='unreadable'):
        local.collect_local(ctx,tmp_path/'local')
    assert not (ctx.output/'aggregation_latest.json').exists()
    progress=list((tmp_path/'local').glob('*/progress.json'))
    assert read_json(progress[0])['stage']=='FAILED'


def test_interrupted_upload_retains_partial_without_completion_receipt(tmp_path,monkeypatch):
    work=tmp_path/'work';(work/'inputs').mkdir(parents=True);(work/'global').mkdir()
    write_json(work/'global/result.json',dict(status='SYNTHETIC'))
    write_json(work/'stage_report.json',dict(status='STAGED'))
    original=local.shutil.copyfileobj
    def interrupted(source,dest,*args,**kwargs):
        if str(getattr(dest,'name','')).endswith('.partial'):
            dest.write(source.read(10));raise OSError('synthetic interrupted upload')
        return original(source,dest,*args,**kwargs)
    monkeypatch.setattr(local.shutil,'copyfileobj',interrupted)
    with pytest.raises(OSError,match='interrupted upload'):
        local.publish_archive(work,tmp_path/'formal','one',lambda *a,**k:None)
    folder=tmp_path/'formal/aggregation_runs/one'
    assert (folder/'global_results.zip.partial').is_file()
    assert not (folder/'global_results.zip').exists()
    assert not (folder/'receipt.json').exists()
    assert (work/'global_results.zip').is_file()


def test_tables_only_never_calls_recovery(tmp_path,monkeypatch):
    from runtime.wan import native_baseline_compact_recovery_run_v1 as runner
    ctx=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(runner,'Context',lambda *a:ctx)
    monkeypatch.setattr(runner,'check_previous_worker',lambda *a:None)
    monkeypatch.setattr(runner,'runtime_session',lambda *a,**k:nullcontext())
    def forbidden(*a,**k):raise AssertionError('raw/model/codec recovery must not run')
    monkeypatch.setattr(runner,'cpu_recover',forbidden)
    result=runner.run(tables_only=True,local_work=tmp_path/'local',vendor_factory=forbidden)
    assert result['new_model_calls']==result['new_codec_calls']==0
    assert result['status']=='GLOBAL_TABLES_SAVED'


def test_cpu_notebook_and_builder_are_synchronized(tmp_path):
    import ast
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('builder',root/'scripts/build_native_baseline_compact_recovery_v1_notebook.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    generated=builder.build_tables_only(tmp_path)
    assert generated.read_bytes()==(root/'notebooks'/generated.name).read_bytes()
    nb=json.loads(generated.read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert 'accelerator' not in nb['metadata']
    assert "CONFIG['tables_only'] = True" in ''.join(nb['cells'][2]['source'])
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))


def test_local_workspace_cannot_be_formal_output(tmp_path,monkeypatch):
    ctx=fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='outside formal'):
        local.collect_local(ctx,ctx.output/'cache')
