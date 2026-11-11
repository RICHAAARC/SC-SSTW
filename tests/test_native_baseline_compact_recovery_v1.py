"""Artificial recovery checks; never models, media codecs or saved research arrays."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from runtime.wan.native_baseline_storage_v1 import read_json, write_json
from runtime.wan.native_baseline_storage_v4 import ArrayLog, append_json, Budget
from runtime.wan import native_baseline_compact_recovery_records_v1 as records
from runtime.wan import native_baseline_compact_recovery_v1 as repair
from runtime.wan.native_baseline_compact_recovery_gpu_v1 import allow_group, saved_budget
from main.baselines import public_compact12_v1 as p
from main.baselines.public_fpr001_v4 import batches, read_batch

ROOT=Path(__file__).resolve().parents[1]


def make_store(tmp_path,b='pixelseal',frames=181):
    root=tmp_path/'original';unit='native/S/'+b+'/OFF';folder=root/'units'/unit
    view=dict(id='G1/T00',frames=frames,operation='T00',generation='G1')
    log=ArrayLog(folder/'arrays-old.bin')
    for start,n in batches(frames,read_batch(b)):
        key=f'G1/T00/epoch01/batch{start}'
        values=dict(bits=np.zeros((n,256 if b=='pixelseal' else 32),np.uint8),valid=np.ones(n,bool))
        if b=='pixelseal':values['logits']=np.zeros((n,257),np.float32)
        location=log.append(key,values,runtime_id='old')
        append_json(folder/'calls.jsonl',dict(operation=key,status='COMMITTED',runtime_id='old',array_file=log.path.name,
            array_record={k:v for k,v in location.items() if k!='path'}))
    return root,unit,view


def test_fixed_repair_matrix():
    targets=records.targets()
    assert len(targets)==87 and len(records.groups(targets))==70 and len(records.quality_targets())==11
    assert len(records.execution_targets())==96 and len(records.groups())==79 and len(records.execution_quality_targets())==14
    assert len({tuple(sorted(r.items())) for r in targets})==87
    assert {r['source'] for r in targets+records.quality_targets()}=={'ucf-0937','ucf-0641','ucf-0642','ucf-0643','ucf-0517','ucf-0532'}


def test_missing_registry_recovers_exact_saved_batches_without_old_write(tmp_path):
    root,unit,view=make_store(tmp_path)
    before={str(f.relative_to(root)):f.read_bytes() for f in root.rglob('*') if f.is_file()}
    store=records.SavedStore(root,unit);epoch,basis=records.select_epoch(store,view)
    raw,details=records.read_raw(store,view,'pixelseal',epoch)
    assert basis['basis']=='UNIQUE_RECEIPT_EPOCH' and epoch['id']=='epoch01'
    assert raw['observed'].all() and len(details)==6 and all(r['observed'] for r in details)
    assert before=={str(f.relative_to(root)):f.read_bytes() for f in root.rglob('*') if f.is_file()}


def test_missing_registry_never_falls_back_from_new_started_epoch(tmp_path):
    root,unit,view=make_store(tmp_path)
    append_json(root/'units'/unit/'calls.jsonl',dict(operation='G1/T00/epoch02/batch0',status='STARTED'))
    store=records.SavedStore(root,unit);epoch,basis=records.select_epoch(store,view)
    assert epoch is None and basis['epoch_ids']==['epoch01','epoch02']
    write_json(store.folder/'views/G1/T00/epochs.json',[dict(id='epoch01'),dict(id='epoch02',input='/missing')])
    epoch,basis=records.select_epoch(store,view);raw,details=records.read_raw(store,view,'pixelseal',epoch)
    assert epoch['id']=='epoch02' and not raw['observed'].any() and details[0]['receipt_status']=='STARTED'


def test_imported_receipts_then_own_and_absolute_array_relocation(tmp_path):
    root,unit,view=make_store(tmp_path)
    folder=root/'units'/unit;rows=[json.loads(s) for s in (folder/'calls.jsonl').read_text().splitlines()]
    for row in rows:row['array_file']='/content/drive/MyDrive/old/'+str((folder/'arrays-old.bin').relative_to(root))
    write_json(folder/'old_import.json',dict(events=rows))
    (folder/'calls.jsonl').write_text('')
    locate=records.Locator(remaps=[('/content/drive/MyDrive/old',root)])
    store=records.SavedStore(root,unit,locate);epoch,_=records.select_epoch(store,view)
    assert records.read_raw(store,view,'pixelseal',epoch)[0]['observed'].all()
    append_json(folder/'calls.jsonl',dict(operation=rows[0]['operation'],status='INVALID_RETURN'))
    store=records.SavedStore(root,unit,locate)
    raw,details=records.read_raw(store,view,'pixelseal',epoch)
    assert not raw['observed'][:32].any() and details[0]['receipt_status']=='INVALID_RETURN'


def test_committed_is_not_readable_and_no_automatic_gpu_replacement(tmp_path):
    root,unit,view=make_store(tmp_path)
    (root/'units'/unit/'arrays-old.bin').write_bytes(b'broken')
    store=records.SavedStore(root,unit);epoch,_=records.select_epoch(store,view)
    raw,details=records.read_raw(store,view,'pixelseal',epoch)
    assert not raw['observed'].any() and all(r.get('error') for r in details)
    assert not allow_group(dict(status='RAW_INCOMPLETE',missing_batches=details))
    assert not allow_group(dict(status='INPUT_LOST_PARTIAL_EPOCH',missing_batches=[]))


def context(tmp_path,root):
    ctx=object.__new__(repair.Context);ctx.output=tmp_path/'repair';ctx.parent=root;ctx.old=tmp_path/'oldv4'
    ctx.cache=tmp_path/'cache';ctx.old_cache=tmp_path/'oldcache';ctx.locate=records.Locator()
    ctx.by_id={'S':dict(id='S',split='TEST',action='synthetic')};ctx.view_map={v['id']:v for v in p.views()}
    ctx.roots=lambda shard,overlay=True:([ctx.output] if overlay else [])+[root]
    ctx.part=lambda shard:ctx.output
    ctx.store=lambda shard,unit,overlay=True:records.SavedStore(root,unit)
    ctx.query=lambda shard:{'pixelseal/S':{'query':[0]*256},'frameindex/S':{'query':[0]*248}}
    ctx.plan=lambda shard,sid:{'maps':{'T00':list(range(181))}}
    return ctx


def test_shared_off_raw_has_two_independent_post_calls_and_preserves_old_complete(tmp_path,monkeypatch):
    root,unit,view=make_store(tmp_path);ctx=context(tmp_path,root);called=[]
    from runtime.wan import native_baseline_evaluation_v4 as evaluation
    def post(output,source,method,arm,view,raw,epoch,queries,plan,container):
        called.append((method,arm,int(raw['observed'].sum()),epoch['id']))
        return {'ownership':{'complete':True}}
    monkeypatch.setattr(evaluation,'write_post',post)
    rows=[dict(shard=2,source='S',method=m,arm='OFF',view='G1/T00') for m in ('pixelseal','frameindex')]
    result=repair.recover_group(ctx,(2,'S','pixelseal','OFF','G1/T00'),rows)
    assert result['cpu_recovered']==2 and called==[('pixelseal','OFF',181,'epoch01'),('frameindex','OFF',181,'epoch01')]
    write_json(root/'post/S/frameindex/OFF/G1/T00.json',dict(ownership=dict(complete=True)))
    called.clear();result=repair.recover_group(ctx,(2,'S','pixelseal','OFF','G1/T00'),rows)
    assert result['already_complete']==1 and result['cpu_recovered']==1 and called[0][0]=='pixelseal'


def test_quality_only_requested_stage_and_old_partial_preserved(tmp_path,monkeypatch):
    root=tmp_path/'original';ctx=context(tmp_path,root);called=[]
    old=root/'units/quality/S/frameindex/quality/G1.json';write_json(old,dict(frames=[],edges=[],old=True))
    monkeypatch.setattr(repair,'saved_rgb',lambda c,s,sid,arm,stage:called.append((arm,stage)) or np.zeros(1))
    from runtime.wan import native_baseline_quality_v4 as quality
    monkeypatch.setattr(quality,'cpu_quality',lambda a,b:dict(frames=[{}]*181,edges=[{}]*180))
    result=repair.recover_quality(ctx,dict(shard=2,source='S',method='frameindex',stage='G1'))
    assert result['status']=='RECOVERED_SAVED_RGB' and called==[('frameindex','G1'),('OFF','G1')]
    assert read_json(old)==dict(frames=[],edges=[],old=True)
    assert (ctx.output/'units/quality/S/frameindex/quality/G1.json').is_file()


def test_original_counts_constrain_but_are_not_new_ledger_cost(tmp_path):
    parent=tmp_path/'original';output=tmp_path/'repair'
    write_json(parent/'shards/part2/physical_counts.json',dict(base={'detect_pixelseal':10},recovery={'detect_pixelseal':2},ledger_offset=999))
    ctx=SimpleNamespace(parent=parent)
    saved=saved_budget(ctx,2,output);assert saved['ledger_offset']==0
    b=Budget(output,'now',limits=p.physical_budget(),saved=saved)
    b.charge('detect_pixelseal','native/S/pixelseal/OFF','G1/T00/epoch01/batch0',recovery=True)
    assert b.base['detect_pixelseal']==10 and b.extra['detect_pixelseal']==3
    b.snapshot();again=Budget(output,'next',limits=p.physical_budget(),saved=saved_budget(ctx,2,output))
    assert again.extra['detect_pixelseal']==3
    newonly=Budget(output,'read',limits=p.physical_budget())
    assert not newonly.base and newonly.extra['detect_pixelseal']==1


def test_notebook_builder_matches_and_cpu_entry_has_no_eager_gpu_import(tmp_path):
    path=ROOT/'scripts/build_native_baseline_compact_recovery_v1_notebook.py'
    spec=importlib.util.spec_from_file_location('builder',path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    produced=builder.build(tmp_path);saved=ROOT/'notebooks'/produced.name
    assert produced.read_bytes()==saved.read_bytes()
    n=json.loads(saved.read_text());assert ''.join(n['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert all(not c.get('outputs') for c in n['cells'])
    assert 'PREVIOUS_RUNTIME_STOPPED = True' in produced.read_text()
    import ast
    for c in n['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))


def test_quality_codec_uses_saved_media_then_quality_only_reconstruction(tmp_path,monkeypatch):
    parent=tmp_path/'original';ctx=repair.Context(parent,tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    write_json(parent/'shards/part2/physical_counts.json',dict(base={},recovery={}))
    original=parent/'saved.mp4';original.parent.mkdir(exist_ok=True);original.write_bytes(b'original')
    ctx.store=lambda *a,**k:SimpleNamespace(row=lambda key:dict(status='COMMITTED',value={'path':str(original)}) if key=='G1/encode' else dict(status='NOT_STARTED'))
    from runtime.wan import native_baseline_media_v4 as media
    from runtime.wan import native_baseline_io_v4 as io
    calls=[]
    class FakeMedia:
        def probe(self,*a,**k):calls.append('probe');return {}
        def decode(self,*a,**k):calls.append('decode');return np.zeros((2,1,1,3),np.uint8)
        def encode(self,pixels,path,*a,**k):calls.append('encode');Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(b'new');return dict(path=str(path))
    monkeypatch.setattr(media,'MediaV4',FakeMedia)
    monkeypatch.setattr(io,'validate_rgb',lambda *a:None)
    monkeypatch.setattr(io,'load_rgb',lambda path,*a:np.load(path,allow_pickle=False))
    monkeypatch.setattr(repair,'saved_rgb',lambda *a:np.zeros((2,1,1,3),np.uint8))
    ctx.decode_saved_media(2,'S','frameindex','G1');ctx.decode_saved_media(2,'S','frameindex','G1')
    assert calls==['probe','decode']
    ctx.decode_saved_media(2,'S','frameindex','G2-identity')
    assert calls==['probe','decode','encode','probe','decode']
    assert original.read_bytes()==b'original' and not (ctx.part(2)/'units/native').exists()
    value=read_json(ctx.part(2)/'units/quality_media/S/frameindex/G2-identity/material.json')
    assert value['used_only_for_quality'] and not value['used_by_native_reader']
    assert dict(ctx.cpu_budgets[2].extra)==dict(video_probe=2,video_decode=2,video_encode=1)


def test_quality_missing_original_writer_never_loads_or_reembeds(tmp_path,monkeypatch):
    parent=tmp_path/'original';ctx=repair.Context(parent,tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    write_json(parent/'shards/part2/physical_counts.json',dict(base={},recovery={}))
    def missing(*a):raise FileNotFoundError('original writer unavailable')
    monkeypatch.setattr(repair,'saved_rgb',missing)
    with pytest.raises(FileNotFoundError):ctx.quality_media(2,'S','frameindex','G1')
    assert not ctx.cpu_budgets[2].base and not ctx.cpu_budgets[2].extra
    assert not (ctx.part(2)/'model_loads.jsonl').exists()


def test_selected_receipt_root_cannot_inherit_unrelated_old_epoch(tmp_path,monkeypatch):
    root,unit,view=make_store(tmp_path);ctx=context(tmp_path,root)
    path=root/'units'/unit/'calls.jsonl'
    path.write_text(path.read_text().replace('epoch01','epoch02'))
    # The batch array headers deliberately retain epoch01, so this checks
    # association, not successful readback of mismatched artificial records.
    other=tmp_path/'other';write_json(other/'units'/unit/'views/G1/T00/epochs.json',[dict(id='epoch01')])
    ctx.roots=lambda shard,overlay=True:[ctx.output,root,other]
    from runtime.wan import native_baseline_evaluation_v4 as evaluation
    monkeypatch.setattr(evaluation,'write_post',lambda *a,**k:{'ownership':{'complete':False}})
    row=dict(shard=2,source='S',method='pixelseal',arm='OFF',view='G1/T00')
    result=repair.recover_group(ctx,(2,'S','pixelseal','OFF','G1/T00'),[row])
    assert result['epoch']['id']=='epoch02' and result['association']['basis']=='UNIQUE_RECEIPT_EPOCH'


def test_post_uses_same_first_existing_root_as_global_tables(tmp_path):
    root=tmp_path/'original';ctx=context(tmp_path,root);other=tmp_path/'oldv4'
    relative='post/S/pixelseal/OFF/G1/T00.json'
    write_json(root/relative,{'ownership':{'complete':False}})
    write_json(other/relative,{'ownership':{'complete':True}})
    ctx.roots=lambda shard,overlay=True:[ctx.output,root,other]
    row=dict(shard=2,source='S',method='pixelseal',arm='OFF',view='G1/T00')
    value,path=repair.post_value(ctx,row)
    assert value is None and path==root/relative


def test_quality_failed_call_is_local_but_disk_failure_stops(tmp_path,monkeypatch):
    from runtime.wan.native_baseline_storage_v4 import Unavailable,SpaceUnavailable
    ctx=context(tmp_path,tmp_path/'original')
    row=dict(shard=2,source='S',method='frameindex',stage='G1')
    def failed(*a):raise Unavailable('saved failed decode')
    monkeypatch.setattr(repair,'saved_rgb',failed)
    assert repair.recover_quality(ctx,row)['status']=='PAIRED_MATERIAL_UNAVAILABLE'
    def disk(*a):raise SpaceUnavailable('real disk failure')
    monkeypatch.setattr(repair,'saved_rgb',disk)
    with pytest.raises(SpaceUnavailable):repair.recover_quality(ctx,row)


def test_complete_arm_rebuild_keeps_old_receipts_out_of_new_writer(tmp_path):
    from runtime.wan.native_baseline_compact_recovery_gpu_v1 import RepairReuse,RepairOperations,replacement_unit
    parent=tmp_path/'parent';ctx=repair.Context(parent,tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    source='ucf-0642';unit='writer/'+source+'/frameindex';part=parent/'shards/part2'
    write_json(part/'queries.json',{'frameindex/'+source:{'query':[0]*248},'pixelseal/'+source:{'query':[0]*256}})
    old=part/'units'/unit
    append_json(old/'calls.jsonl',dict(operation='batch0',status='COMMITTED',array_file='/missing-old-array'))
    b=Budget(ctx.part(2),'new',limits=p.physical_budget());reuse=RepairReuse(ctx,2)
    ops=RepairOperations(ctx.part(2),ctx.cache/'new',b,None,reuse)
    store=ops.store(unit,cache_arrays=True)
    assert store.row('batch0')['status']=='NOT_STARTED'
    imported=read_json(store.folder/'old_import.json')
    assert imported['events']==[] and len(imported['retained_original_events'])==1
    store.perform('batch0','embed_pixelseal',lambda:{'rgb':np.zeros((1,1,1,3),np.uint8)},arrays=True)
    assert not b.base and b.extra['embed_pixelseal']==1
    assert store.row('batch0')['recovery'] is True
    assert replacement_unit('writer/ucf-0643/pixelseal')
    assert len((old/'calls.jsonl').read_text().splitlines())==1


def test_saved_complete_work_never_constructs_a_model(tmp_path,monkeypatch):
    from runtime.wan import native_baseline_compact_recovery_gpu_v1 as gpu
    ctx=repair.Context(tmp_path/'parent',tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    for shard in (1,2,3):write_json(ctx.parent/'shards'/f'part{shard}'/'physical_counts.json',dict(base={},recovery={}))
    monkeypatch.setattr(gpu,'recover_group',lambda *a:dict(status='ALREADY_COMPLETE'))
    monkeypatch.setattr(gpu,'quality_targets',lambda:[])
    def forbidden():raise AssertionError('complete slots must never load a model')
    result=gpu.run_physical(ctx,dict(views=[]),forbidden)
    assert len(result)==79 and all(r['status']=='SKIPPED_OR_UNRESOLVED' for r in result)
    assert not any(ctx.output.rglob('model_loads.jsonl'))


def entry_module():
    spec=importlib.util.spec_from_file_location('recovery_entry',ROOT/'scripts/native_baseline_compact_recovery_v1_entry.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_cpu_package_preserves_existing_files_and_counts_costs_once(tmp_path):
    import zipfile
    output=tmp_path/'repair';output.mkdir();part=output/'shards/part2'
    row=dict(status='STARTED',runtime_id='local',unit='cpu_quality_media/S/OFF/G1',operation='decode',category='video_decode',at=1.,count=1,recovery=True)
    write_json(part/'physical_counts.json',dict(base={},recovery={},ledger_offset=0))
    post=part/'post/S/frameindex/frameindex/G1/T00.json';write_json(post,{'existing':True})
    with zipfile.ZipFile(output/'local_cpu_recovery.zip','w') as archive:
        archive.writestr('shards/part2/physical_calls.jsonl',json.dumps(row)+'\n')
        archive.writestr('shards/part2/physical_counts.json',json.dumps({'base':{'ignored':99}}))
        archive.writestr('shards/part2/post/S/frameindex/frameindex/G1/T00.json','{"old_cpu":true}')
        archive.writestr('shards/part2/units/quality/S/frameindex/quality/G1.json','{"quality":true}')
        archive.writestr('shards/part2/runtime.json','{"status":"RUNNING"}')
    entry=entry_module();first=entry.import_cpu_package(output);again=entry.import_cpu_package(output)
    assert len(first['new_cost_events'])==1 and not again['new_cost_events']
    assert read_json(post)=={'existing':True} and not (part/'runtime.json').exists()
    budget=Budget(part,'colab',limits=p.physical_budget(),saved=read_json(part/'physical_counts.json'))
    assert budget.extra['video_decode']==1 and not budget.base


def test_cpu_package_rejects_unsafe_members_before_copying(tmp_path):
    import zipfile
    with zipfile.ZipFile(tmp_path/'local_cpu_recovery.zip','w') as archive:
        archive.writestr('cpu_recovery.json','{}')
        archive.writestr('../escaped.json','{}')
    with pytest.raises(ValueError):entry_module().import_cpu_package(tmp_path)
    assert not (tmp_path/'cpu_recovery.json').exists()


def test_missing_original_partial_positive_arm_stops_before_model(tmp_path,monkeypatch):
    from runtime.wan import native_baseline_compact_recovery_gpu_v1 as gpu
    ctx=repair.Context(tmp_path/'parent',tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    for shard in (1,2,3):write_json(ctx.parent/'shards'/f'part{shard}'/'physical_counts.json',dict(base={},recovery={}))
    ctx.query=lambda shard:{'frameindex/ucf-0641':dict(query=[0]*248)}
    ctx.plan=lambda *a:{'maps':{}}
    row=dict(shard=2,source='ucf-0641',method='frameindex',arm='frameindex',view='G1/T00')
    monkeypatch.setattr(gpu,'groups',lambda:{(2,'ucf-0641','pixelseal','frameindex','G1/T00'):[row]})
    monkeypatch.setattr(gpu,'recover_group',lambda *a:dict(status='RAW_INCOMPLETE',missing_batches=[]))
    monkeypatch.setattr(gpu,'quality_targets',lambda:[])
    def forbidden():raise AssertionError('missing partial writer must not cause a model load')
    result=gpu.run_physical(ctx,dict(views=[]),forbidden)
    assert len(result)==1 and 'NEEDS_WRITER_INPUT_DECISION' in result[0]['error']
    assert not any(ctx.output.rglob('model_loads.jsonl'))


def test_collector_keeps_original_and_new_resource_details(tmp_path,monkeypatch):
    from runtime.wan import native_baseline_compact_tables_v1 as tables
    from runtime.wan import paper_execution_v1 as paper
    ctx=repair.Context(tmp_path/'parent',tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    ctx.sources=[];ctx.query=lambda shard:{}
    monkeypatch.setattr(paper.PublicCost,'snapshot',lambda self:dict(base={'detect_pixelseal':5},recovery={},counts_complete=True))
    called=[]
    def build(output,sources,roster,queries,cost,old,roots):
        called.append(roots);write_json(output/'tables/D_resources.json',cost.snapshot());return {'complete':True}
    monkeypatch.setattr(tables,'build_tables',build)
    append_json(ctx.parent/'shards/part2/model_resources.jsonl',dict(backend='old',actual_forward_hooks={'detect':5}))
    append_json(ctx.part(2)/'model_resources.jsonl',dict(backend='new',actual_forward_hooks={'detect':1}))
    b=Budget(ctx.part(2),'new',limits=p.physical_budget());b.charge('detect_pixelseal','native/S/pixelseal/OFF','one',recovery=True)
    result=repair.collect(ctx);value=read_json(ctx.output/'global/tables/D_resources.json')
    assert result=={'complete':True} and len(called)==1
    assert [r['backend'] for r in value['model_resources']]==['old','new']
    assert value['base']['detect_pixelseal']==5 and value['recovery']['detect_pixelseal']==1


def test_unreadable_receipt_cannot_imply_a_unique_old_epoch(tmp_path):
    root,unit,view=make_store(tmp_path)
    with (root/'units'/unit/'calls.jsonl').open('a') as stream:stream.write('{torn new epoch receipt\n')
    store=records.SavedStore(root,unit);epoch,basis=records.select_epoch(store,view)
    assert epoch is None and basis['basis']=='UNREADABLE_RECEIPTS'
    write_json(store.folder/'views/G1/T00/epochs.json',[dict(id='epoch01',input=None)])
    epoch,basis=records.select_epoch(store,view)
    assert epoch['id']=='epoch01' and basis['basis']=='SAVED_LAST_EPOCH'


def test_replacement_arm_never_uses_old_complete_post_raw_or_quality(tmp_path,monkeypatch):
    from runtime.wan.native_baseline_evaluation_v4 import missing_post
    from runtime.wan.native_baseline_compact_tables_v1 import rows_for
    ctx=repair.Context(tmp_path/'parent',tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    sid='ucf-0643';method='pixelseal';source=ctx.by_id[sid];original=ctx.parent/'shards/part2'
    for view in p.views():
        value=missing_post(source,method,method,view);value['ownership'].update(complete=True,decision=True)
        write_json(original/f'post/{sid}/{method}/{method}/{view["id"]}.json',value)
    unit=f'native/{sid}/{method}/{method}'
    append_json(original/'units'/unit/'calls.jsonl',dict(operation='G1/T00/epoch01/batch0',status='COMMITTED'))
    for stage in ('precodec','G1','G2-identity'):
        write_json(original/f'units/quality/{sid}/{method}/quality/{stage}.json',dict(frames=[{}]*181,edges=[{}]*180))
    repair.prepare_replacement_slots(ctx)
    row=dict(shard=2,source=sid,method=method,arm=method,view='G1/T00')
    assert repair.post_value(ctx,row)[0] is None and not ctx.store(2,unit).rows
    values=rows_for(ctx.output,source,method,method,{sid:ctx.roots(2)})
    assert len(values)==12 and not any(v['ownership']['complete'] for v in values)
    def missing(*a):raise FileNotFoundError('new writer not returned')
    monkeypatch.setattr(repair,'saved_rgb',missing)
    assert repair.recover_quality(ctx,dict(shard=2,source=sid,method=method,stage='G1'))['status']=='PAIRED_MATERIAL_UNAVAILABLE'
    # One new successful view must not make the remaining 11 fall back to old success.
    view=p.views()[0];value=missing_post(source,method,method,view);value['ownership'].update(complete=True,decision=False)
    write_json(ctx.part(2)/f'post/{sid}/{method}/{method}/{view["id"]}.json',value)
    repair.prepare_replacement_slots(ctx)
    values=rows_for(ctx.output,source,method,method,{sid:ctx.roots(2)})
    assert sum(v['ownership']['complete'] for v in values)==1


def test_cpu_import_checks_owners_before_touching_ledger(tmp_path):
    import os,socket,zipfile
    output=tmp_path/'repair';output.mkdir()
    with zipfile.ZipFile(output/'local_cpu_recovery.zip','w') as archive:archive.writestr('cpu_recovery.json','{}')
    write_json(output/'shards/part2/runtime_owner.json',dict(status='ACTIVE',host=socket.gethostname(),pid=os.getpid()))
    before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError):entry_module().import_cpu_package(output,tmp_path/'parent')
    assert before=={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_cpu_import_never_leaves_partial_final_file(tmp_path,monkeypatch):
    import zipfile
    with zipfile.ZipFile(tmp_path/'local_cpu_recovery.zip','w') as archive:archive.writestr('cpu_recovery.json','{"valid":true}')
    entry=entry_module();real=entry.os.replace
    def interrupted(source,target):
        if Path(target).name=='cpu_recovery.json':raise OSError('simulated publication interruption')
        return real(source,target)
    monkeypatch.setattr(entry.os,'replace',interrupted)
    with pytest.raises(OSError):entry.import_cpu_package(tmp_path,tmp_path/'parent')
    assert not (tmp_path/'cpu_recovery.json').exists()
    monkeypatch.setattr(entry.os,'replace',real)
    entry.import_cpu_package(tmp_path,tmp_path/'parent')
    assert read_json(tmp_path/'cpu_recovery.json')=={'valid':True}


@pytest.mark.parametrize('status,error,allowed',[
    ('FAILED_PERMANENT','two extra infrastructure calls exhausted',True),
    ('FAILED_PERMANENT','actual model configuration invalid',False),
    ('INVALID_RETURN','actual nonfinite writer output',False)])
def test_rebuild_distinguishes_retired_infra_gate_from_method_failure(tmp_path,status,error,allowed):
    from runtime.wan.native_baseline_compact_recovery_gpu_v1 import RepairReuse,RepairOperations
    from runtime.wan.native_baseline_storage_v4 import Unavailable
    ctx=repair.Context(tmp_path/'parent',tmp_path/'repair',tmp_path/'cache',tmp_path/'old',tmp_path/'oldcache')
    part=ctx.parent/'shards/part2';unit='writer/ucf-0643/pixelseal'
    write_json(part/'queries.json',{'pixelseal/ucf-0643':{'query':[0]*256}})
    append_json(part/'units'/unit/'calls.jsonl',dict(operation='batch0',status=status,error=error))
    budget=Budget(ctx.part(2),'new',limits=p.physical_budget())
    ops=RepairOperations(ctx.part(2),ctx.cache/'new',budget,None,RepairReuse(ctx,2))
    if not allowed:
        with pytest.raises(Unavailable):ops.store(unit,cache_arrays=True)
        assert not budget.base and not budget.extra
    else:
        store=ops.store(unit,cache_arrays=True)
        old=read_json(store.folder/'old_import.json')['retained_original_events']
        assert old[0]['error']==error and store.row('batch0')['status']=='NOT_STARTED'
        store.perform('batch0','embed_pixelseal',lambda:dict(rgb=np.zeros((1,1,1,3),np.uint8)),arrays=True)
        assert budget.extra['embed_pixelseal']==1
