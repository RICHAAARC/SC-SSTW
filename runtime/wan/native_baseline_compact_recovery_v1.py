"""Repair only the 87 saved compact12 view gaps and 11 quality-stage gaps."""
from collections import Counter
from pathlib import Path
import time

from main.baselines import public_compact12_v1 as p
from main.baselines.public_fpr001_v4 import backend, writer_batches
from main.tube_state import paper_execution_v1 as schedule
from .native_baseline_storage_v1 import read_json, write_json
from .native_baseline_storage_v4 import append_json, events, Unavailable, SpaceUnavailable
from .native_baseline_compact_recovery_records_v1 import (
    execution_targets, execution_quality_targets as quality_targets, groups, Locator, SavedStore, select_epoch,
    read_raw, complete_post, first_file, replacement_unit, REBUILD_ARMS)

OUTPUT=Path(p.OUTPUT)/'recovery/repair01'


class Context:
    def __init__(self,parent=p.OUTPUT,output=OUTPUT,cache=p.CACHE,old=p.OLD_OUTPUT,old_cache=p.OLD_CACHE):
        self.parent=Path(parent);self.output=Path(output);self.cache=Path(cache)
        self.old=Path(old);self.old_cache=Path(old_cache)
        self.locate=Locator(remaps=((p.OUTPUT,self.parent),(p.CACHE,self.cache),(p.OLD_OUTPUT,self.old),(p.OLD_CACHE,self.old_cache)))
        self.query_cache={};self.plan_cache={};self.view_map={v['id']:v for v in p.views()}
        self.cpu_budgets={};self.cpu_decode_enabled=True;self.runtime_id='cpu-repair-'+str(time.time_ns())
        candidates=[self.parent,*[self.parent/'shards'/f'part{i}' for i in range(4)],self.old]
        path=first_file(candidates,'roster.json') or first_file(candidates,'sources.json')
        if path is None:
            path=Path(__file__).resolve().parents[2]/'main/baselines/protocols/public_native_fpr001_v4.sources.json'
        roster=read_json(path);self.roster=roster['sources'] if isinstance(roster,dict) else roster
        self.sources=p.selected_sources(self.roster);self.by_id={s['id']:s for s in self.roster}
        self.pack={s['id']:pack['id'] for pack in schedule.public_packs(self.sources) for s in pack['sources']}

    def part(self,shard):return self.output/'shards'/f'part{shard}'
    def roots(self,shard,overlay=True):
        old=[self.parent/'shards'/f'part{shard}',self.parent,self.old]
        return [self.part(shard),*old] if overlay else old
    def cache_for(self,root,shard):
        if root==self.part(shard):return self.cache/'recovery/repair01'/f'part{shard}'
        if root==self.parent/'shards'/f'part{shard}':return self.cache/'shards'/f'part{shard}'
        return self.old_cache if root==self.old else self.cache
    def query(self,shard):
        from .native_baselines_public_v4 import SavedQueries
        if shard not in self.query_cache:
            path=first_file(self.roots(shard,False),'queries.json')
            if path is None:raise FileNotFoundError('saved query registry unavailable for part'+str(shard))
            self.query_cache[shard]=SavedQueries(read_json(path))
        return self.query_cache[shard]
    def plan(self,shard,sid):
        key=(shard,sid)
        if key not in self.plan_cache:
            path=first_file(self.roots(shard,False),f'editor/{sid}/plan.json')
            if path is None:raise FileNotFoundError('saved editor plan unavailable: '+sid)
            value=read_json(path)
            if not isinstance(value,dict) or not isinstance(value.get('maps'),dict):raise ValueError('invalid saved editor plan: '+sid)
            self.plan_cache[key]=value
        return self.plan_cache[key]
    def store(self,shard,unit,overlay=True):
        roots=[self.part(shard)] if replacement_unit(unit) else self.roots(shard,overlay)
        # An empty directory cannot mask saved receipts in a fallback package.
        root=next((r for r in roots if any((r/'units'/unit/f).is_file() for f in ('calls.jsonl','old_import.json'))),roots[0])
        directory=self.cache_for(root,shard)/'arrays'/unit if unit.startswith('writer/') else None
        return SavedStore(root,unit,self.locate,directory)

    def cpu_budget(self,shard):
        from .native_baseline_storage_v4 import Budget
        from .native_baseline_compact_recovery_records_v1 import saved_budget
        if shard not in self.cpu_budgets:
            inherited=self.parent/'shards'/f'part{shard}'/'inherited_budget.json'
            limits=read_json(inherited).get('limits',p.physical_budget()) if inherited.is_file() else p.physical_budget()
            self.cpu_budgets[shard]=Budget(self.part(shard),self.runtime_id,limits=limits,saved=saved_budget(self,shard,self.part(shard)))
        return self.cpu_budgets[shard]

    def quality_media(self,shard,sid,arm,stage):
        """Reconstruct only missing quality material from original saved writer pixels."""
        from .native_baseline_media_v4 import MediaV4
        from .native_baseline_io_v4 import file_available
        from .native_baseline_storage_v4 import UnitStore
        original=self.store(shard,f'media/{sid}/{arm}');key='G1/encode' if stage=='G1' else 'G2/T00/encode'
        row=original.row(key);value=row.get('value')
        path=self.locate(value['path']) if isinstance(value,dict) and value.get('path') else None
        if row.get('status')=='COMMITTED' and path is not None and path.is_file():return path,'ORIGINAL_SAVED_MEDIA'
        budget=self.cpu_budget(shard);store=UnitStore(self.part(shard),f'quality_media/{sid}/{arm}/{stage}',budget)
        prior=store.row('encode').get('value')
        if file_available(prior):return Path(prior['path']),'QUALITY_ONLY_RECONSTRUCTION'
        # No model call is reachable: precodec reads only the saved original writer arrays.
        pixels=saved_rgb(self,shard,sid,arm,'precodec' if stage=='G1' else 'G1')
        def encode():
            path=store.folder/f"clip-{store.attempts['encode']:02}.mp4"
            return MediaV4().encode(pixels,path,25,codec='h264',crf=18)
        try:
            value=store.perform('encode','video_encode',encode,recovery=True,output_available=file_available)
            write_json(store.folder/'material.json',dict(source='original saved writer/canonical pixels' if stage=='G1' else 'original G1 or quality-only G1 reconstruction',
                stage=stage,used_only_for_quality=True,used_by_native_reader=False,path=value['path']))
            return Path(value['path']),'QUALITY_ONLY_RECONSTRUCTION'
        finally:budget.snapshot()

    def decode_saved_media(self,shard,sid,arm,stage):
        from .native_baseline_media_v4 import MediaV4
        from .native_baseline_io_v4 import save_array, load_rgb, validate_rgb, file_available
        from .native_baseline_storage_v4 import UnitStore
        if not self.cpu_decode_enabled:raise FileNotFoundError('saved RGB unavailable; codec not enabled')
        path,basis=self.quality_media(shard,sid,arm,stage)
        budget=self.cpu_budget(shard);media=MediaV4()
        receipt=UnitStore(self.part(shard),f'cpu_quality_media/{sid}/{arm}/{stage}',budget)
        target=self.cache_for(self.part(shard),shard)/'quality-decoded'/sid/arm/(stage+'.npy')
        try:
            receipt.perform('probe','video_probe',lambda:media.probe(path,[181,320,512,3],25),recovery=True)
            def decode():
                rgb=media.decode(path,[181,320,512,3]);validate_rgb(rgb)
                return save_array(target,rgb)
            value=receipt.perform('decode','video_decode',decode,recovery=True,output_available=file_available)
            write_json(receipt.folder/'material.json',dict(source=basis,media_path=str(path),used_only_for_quality=True,used_by_native_reader=False))
            return load_rgb(value['path'])
        finally:budget.snapshot()


def post_value(ctx,row):
    relative=f"post/{row['source']}/{row['method']}/{row['arm']}/{row['view']}.json"
    roots=[ctx.part(row['shard'])] if (row['source'],row['arm']) in REBUILD_ARMS else ctx.roots(row['shard'])
    for root in roots:
        path=root/relative
        if not path.is_file():continue
        try:
            value=read_json(path)
            if complete_post(value):return value,path
        except (OSError,ValueError,TypeError):pass
        # The table reader takes the first existing row, including an incomplete
        # one. Do not silently select an older, more favourable complete row.
        return None,path
    return None,None


def recover_group(ctx,key,rows):
    from .native_baseline_evaluation_v4 import write_post
    shard,sid,b,arm,vid=key;view=ctx.view_map[vid]
    pending=[r for r in rows if post_value(ctx,r)[0] is None]
    report=dict(shard=shard,source=sid,backend=b,arm=arm,view=vid,
        logical_slots=len(rows),already_complete=len(rows)-len(pending),cpu_recovered=0)
    if not pending:return dict(report,status='ALREADY_COMPLETE',missing_batches=[])
    store=ctx.store(shard,f'native/{sid}/{b}/{arm}')
    # Registries belong to the selected receipt lineage. A registry in another
    # root may describe an older epoch and must never override these receipts.
    epoch,association=select_epoch(store,view)
    raw,details=read_raw(store,view,b,epoch)
    report.update(epoch=epoch,association=association,observed_frames=int(raw['observed'].sum()),
        planned_frames=view['frames'],batches=details,receipt_errors=store.record_errors,
        missing_batches=[r for r in details if not r['observed']],raw_root=str(store.folder))
    try:
        if epoch is not None:
            queries=ctx.query(shard);plan=ctx.plan(shard,sid)
            for row in pending:
                # Native observations are fixed before each independent logical evaluation.
                value=write_post(ctx.part(shard),ctx.by_id[sid],row['method'],arm,view,
                    raw,epoch,queries,plan,'repair-cpu')
                report['cpu_recovered']+=complete_post(value)
    except (OSError,ValueError,TypeError,KeyError) as error:report['post_error']=f'{type(error).__name__}: {error}'
    if report['cpu_recovered']==len(pending):report['status']='RECOVERED_SAVED_RAW'
    elif epoch is None:report['status']='EPOCH_ASSOCIATION_UNRESOLVED'
    elif report.get('post_error'):report['status']='POST_INPUT_UNAVAILABLE'
    elif raw['observed'].any() and not (epoch.get('input') and Path(epoch['input']).is_file()):report['status']='INPUT_LOST_PARTIAL_EPOCH'
    else:report['status']='RAW_INCOMPLETE'
    return report


def saved_rgb(ctx,shard,sid,arm,stage):
    """Original matched arrays only: no decoder, writer, model or codec here."""
    from .native_baseline_io_v4 import load_rgb, validate_rgb
    import numpy as np
    roots=[ctx.part(shard)] if (sid,arm) in REBUILD_ARMS else ctx.roots(shard)
    if stage=='precodec' and arm=='OFF':
        path=first_file(roots,f'canonical/{sid}.npy')
        if path:return load_rgb(path)
        raise FileNotFoundError('saved canonical RGB unavailable: '+sid)
    if stage=='precodec':
        store=ctx.store(shard,f'writer/{sid}/{arm}');pieces=[]
        if store.record_errors:raise ValueError('unreadable writer receipts')
        for start,n in writer_batches(arm):
            key='batch'+str(start)
            if store.row(key).get('status') not in ('COMMITTED','RETURNED_NOT_COMMITTED'):raise FileNotFoundError('saved writer batch unavailable: '+key)
            value=store.arrays[key]['arrays']['rgb'];validate_rgb(value,n);pieces.append(value)
        return np.concatenate(pieces)
    store=ctx.store(shard,f'media/{sid}/{arm}')
    keys=['G1/decode'] if stage=='G1' else ['G2/T00/decode/'+backend(arm)] if arm!='OFF' else ['G2/T00/decode/'+b for b in ('pixelseal','wam','videoseal')]
    for key in keys:
        value=store.row(key).get('value')
        if isinstance(value,dict) and value.get('path') and ctx.locate(value['path']).is_file():return load_rgb(ctx.locate(value['path']))
    for root in roots:
        cache=ctx.cache_for(root,shard)
        paths=[cache/'g1'/sid/(arm+'.npy')] if stage=='G1' else [cache/'derived'/sid/arm/'G2/T00'/(b+'.npy') for b in ('pixelseal','wam','videoseal')]
        for path in paths:
            if path.is_file():return load_rgb(path)
    return ctx.decode_saved_media(shard,sid,arm,stage)


def recover_quality(ctx,row):
    from .native_baselines_public_v4 import quality_stage_complete
    from .native_baseline_quality_v4 import cpu_quality
    shard,sid,method,stage=(row[k] for k in ('shard','source','method','stage'))
    relative=f'units/quality/{sid}/{method}/quality/{stage}.json'
    roots=[ctx.part(shard)] if (sid,method) in REBUILD_ARMS else ctx.roots(shard)
    for root in roots:
        path=root/relative
        if path.is_file():
            try:
                if quality_stage_complete(read_json(path)):return dict(row,status='ALREADY_COMPLETE',saved_path=str(path))
            except (OSError,ValueError,TypeError):pass
            break
    try:
        # Positive material is the known sparse dependency; check it before decoding OFF.
        right=saved_rgb(ctx,shard,sid,method,stage);left=saved_rgb(ctx,shard,sid,'OFF',stage)
        result=cpu_quality(left,right)
        result.update(lpips_observed_frames=0,lpips_mean=None,lpips_policy='not requested',recovery=dict(source='saved original matched arrays/media; missing quality media may be reconstructed only from original saved writer/canonical pixels',new_model_calls=0,codec_costs='separate repair shard ledger',materials='units/quality_media and units/cpu_quality_media; never inserted into native reader epochs'))
        path=ctx.part(shard)/relative;write_json(path,result)
        return dict(row,status='RECOVERED_SAVED_RGB',saved_path=str(path))
    except SpaceUnavailable:
        raise
    except (Unavailable,OSError,ValueError,TypeError,KeyError) as error:
        return dict(row,status='PAIRED_MATERIAL_UNAVAILABLE',error=f'{type(error).__name__}: {error}')


def prepare_replacement_slots(ctx):
    """Keep the authorized replacement whole arm, even if new slots fail."""
    from .native_baseline_evaluation_v4 import missing_post
    for sid,method in sorted(REBUILD_ARMS):
        shard=2;marker=dict(source=sid,method=method,status='AUTHORIZED_WHOLE_ARM_REPLACEMENT',
            views=list(p.VIEW_IDS),quality=['precodec','G1','G2-identity'],old_results='retained in original shard',
            selection='new arm only, including missing/failed new slots; never fall back to old successes')
        for view in p.views():
            path=ctx.part(shard)/f'post/{sid}/{method}/{method}/{view["id"]}.json'
            if not path.is_file():write_json(path,dict(missing_post(ctx.by_id[sid],method,method,view),replacement=marker))
        for stage in marker['quality']:
            path=ctx.part(shard)/f'units/quality/{sid}/{method}/quality/{stage}.json'
            if not path.is_file():write_json(path,dict(frames=[],edges=[],replacement=marker))
        write_json(ctx.part(shard)/f'replacements/{sid}/{method}.json',marker)


def cpu_recover(ctx):
    tick=time.monotonic()
    prepare_replacement_slots(ctx)
    views=[recover_group(ctx,key,rows) for key,rows in groups().items()]
    quality=[recover_quality(ctx,row) for row in quality_targets()]
    result=dict(status='CPU_RECOVERY_RECORDED',fixed_logical_views=len(execution_targets()),fixed_quality_stages=len(quality_targets()),
        original_gap_logical_views=87,original_gap_quality_stages=11,replacement_additional_views=9,replacement_additional_quality_stages=3,
        raw_view_groups=len(groups()),views=views,quality=quality,
        logical_already_complete=sum(r['already_complete'] for r in views),
        logical_recovered_from_saved_raw=sum(r['cpu_recovered'] for r in views),
        logical_remaining=sum(r['logical_slots']-r['already_complete']-r['cpu_recovered'] for r in views),
        quality_statuses=dict(Counter(r['status'] for r in quality)),
        new_model_calls=0,cpu_codec_costs=[dict(shard=s,base=dict(b.base),recovery=dict(b.extra),scope='cumulative inherited plus repair; additional calls remain in repair ledger') for s,b in ctx.cpu_budgets.items()],cpu_seconds=time.monotonic()-tick,
        missing_scope='unavailable at the supplied paths; a partial local mirror does not establish cloud absence')
    ctx.output.mkdir(parents=True,exist_ok=True)
    previous=ctx.output/'cpu_recovery.json'
    if previous.is_file():append_json(ctx.output/'cpu_recovery_history.jsonl',read_json(previous))
    write_json(previous,result);return result


def collect(ctx):
    """Merge source-level fixed slots, never averaged shard rates or intervals."""
    from .native_baseline_compact_tables_v1 import build_tables
    from .paper_execution_v1 import PublicCost
    from .native_baseline_storage_v4 import Budget
    prepare_replacement_slots(ctx)
    queries={};source_roots={}
    for shard in range(4):
        try:q=ctx.query(shard)
        except (OSError,ValueError,TypeError):q={}
        for pack in schedule.owned_packs(ctx.sources,shard):
            for source in pack['sources']:
                sid=source['id'];source_roots[sid]=ctx.roots(shard)
                for method in p.METHODS:
                    key=method+'/'+sid;queries[key]=q.get(key,dict(wrong_query_collision=None))
    class Cost:
        def snapshot(self):
            value=PublicCost(ctx.parent,ctx.sources).snapshot();extra=[]
            for shard in (1,2,3):
                # This reads only the repair ledger, without the inherited budget snapshot.
                budget=Budget(ctx.part(shard),'read-only',limits=p.physical_budget())
                extra.append(dict(shard=shard,base=dict(budget.base),recovery=dict(budget.extra)))
            value['repair_additional_costs']=extra
            if value.get('counts_complete'):
                for field in ('base','recovery'):
                    total=Counter(value[field])
                    for row in extra:total.update(row[field])
                    value[field]=dict(total)
            value['old_packages_unchanged']=True;return value
    result=build_tables(ctx.output/'global',ctx.sources,ctx.roster,queries,Cost(),ctx.old,source_roots)
    resource_path=ctx.output/'global/tables/D_resources.json';resources=read_json(resource_path)
    roots=[ctx.parent,*[ctx.parent/'shards'/f'part{i}' for i in range(4)],ctx.output,*[ctx.part(i) for i in range(4)]]
    for field,filename in [('model_resources','model_resources.jsonl'),('quality_resources','quality_resources.jsonl'),
                           ('model_loads','model_loads.jsonl'),('import_receipts','imports.jsonl')]:
        resources[field]=[dict(row,record_root=str(root)) for root in roots for row in events(root/filename)]
    resources['old_v4_cost_scope']='separate prior protocol; references are not new compact calls'
    resources['local_cpu_imports']=list(events(ctx.output/'cpu_package_imports.jsonl'))
    write_json(resource_path,resources);return result
