"""Targeted physical continuation after the saved-only compact12 repair pass."""
from pathlib import Path
import time

from main.baselines import public_compact12_v1 as p
from main.baselines.public_fpr001_v4 import backend
from .native_baseline_storage_v1 import read_json, write_json
from .native_baseline_storage_v4 import Budget, Unavailable, SpaceUnavailable, append_json, events, runtime_session, retired_recovery_gate
from .native_baseline_compact_reuse_v1 import Reuse, CompactOperations
from .native_baseline_compact_recovery_records_v1 import (groups, execution_quality_targets as quality_targets,
    SavedStore, saved_budget, replacement_unit, REBUILD_ARMS)
from .native_baseline_compact_recovery_v1 import recover_group, recover_quality, saved_rgb, post_value


class RepairOperations(CompactOperations):
    def store(self,unit,**kwargs):
        store=super().store(unit,**kwargs)
        if not getattr(store,'repair_costs_wrapped',False):
            original=store.perform
            def perform(key,*args,**options):
                # Repair work is additional infrastructure cost, never a reset
                # of the already consumed original scientific allowance.
                options['recovery']=True
                return original(key,*args,**options)
            store.perform=perform;store.repair_costs_wrapped=True
        return store


class RepairReuse(Reuse):
    def __init__(self,ctx,shard):
        root=ctx.parent/'shards'/f'part{shard}'
        super().__init__(ctx.part(shard),root,ctx.cache/'shards'/f'part{shard}',same_experiment=True)
        self.ctx=ctx;self.shard=shard
        self.roots=[(r,ctx.cache_for(r,shard)) for r in ctx.roots(shard,False)]

    def old_unit(self,unit):
        root=next((r for r,c in self.roots if any((r/'units'/unit/f).is_file() for f in ('calls.jsonl','old_import.json'))),self.old)
        parts=unit.split('/');method=parts[3] if parts[0]=='native' and len(parts)>3 else parts[2] if len(parts)>2 else None
        if method not in (None,'OFF') and (root/'units'/unit).is_dir():
            path=root/'queries.json'
            if not path.is_file():raise Unavailable('saved writer query record unavailable: '+unit)
            key=method+'/'+parts[1]
            if read_json(path)[key]['query']!=self.ctx.query(self.shard)[key]['query']:
                raise Unavailable('saved writer message differs from original repair query: '+unit)
        return root,self.ctx.cache_for(root,self.shard)

    def import_unit(self,unit,newfolder):
        marker=newfolder/'old_import.json'
        if marker.is_file():return read_json(marker)
        root,cache=self.old_unit(unit)
        store=SavedStore(root,unit,self.ctx.locate,cache/'arrays'/unit if unit.startswith('writer/') else None)
        converted=[]
        for old in store.history:
            if not isinstance(old,dict) or 'operation' not in old:continue
            row=dict(old);origin=row.get('imported_from',str(store.folder))
            same=row.get('same_experiment',root!=self.ctx.old)
            row.update(imported_from=origin,same_experiment=same)
            if row.get('array_file'):
                path=self.ctx.locate(row['array_file'])
                if not path.is_absolute():path=(cache/'arrays'/unit if unit.startswith('writer/') else store.folder)/path
                row['array_file']=str(path)
            elif row.get('status') in ('COMMITTED','RETURNED_NOT_COMMITTED') and unit.split('/')[0] in ('native','writer','quality'):
                row['array_file']=str((cache/'arrays'/unit if unit.startswith('writer/') else store.folder)/f"arrays-{row.get('runtime_id')}.bin")
            if isinstance(row.get('value'),dict) and row['value'].get('path'):
                row['value']=dict(row['value'],path=str(self.ctx.locate(row['value']['path'])))
            converted.append(row)
        replacement=replacement_unit(unit)
        if replacement and any(r.get('status') in ('INVALID_RETURN','FAILED_PERMANENT') and not retired_recovery_gate(r) for r in store.rows.values()):
            raise Unavailable('method/format failure retained; whole-arm input recovery does not authorize a method retry: '+unit)
        result=dict(old_root=str(root),unit=unit,events=[] if replacement else converted,read_errors=store.record_errors,
            original_receipts_preserved=True,new_physical_calls=0)
        if replacement:
            result.update(retained_original_events=converted,execution_basis='AUTHORIZED_WHOLE_ARM_REBUILD',
                prevents='old writer pixels or native batches mixing with the replacement arm')
        write_json(marker,result)
        append_json(self.output/'imports.jsonl',dict(unit=unit,source=str(store.folder),imported_events=len(converted),new_physical_calls=0))
        return result

    def view(self,store,view):
        if replacement_unit(store.unit):
            return dict(imported=False,epoch=None,same_experiment=False,source='complete missing arm replacement')
        return super().view(store,view)


def allow_group(report):
    if report['status'] in ('ALREADY_COMPLETE','RECOVERED_SAVED_RAW'):return False
    if report.get('receipt_errors') or report.get('post_error'):return False
    if any(r.get('receipt_status')=='COMMITTED' for r in report.get('missing_batches',[])):return False
    if report['status']=='INPUT_LOST_PARTIAL_EPOCH':return False
    if report['status']=='EPOCH_ASSOCIATION_UNRESOLVED':
        return report['association'].get('basis')=='NO_UNIQUE_RECEIPT_EPOCH' and not report['association'].get('epoch_ids')
    return True


def physical_plan(cpu):
    rows=[]
    for report in cpu['views']:
        if report['status'] in ('ALREADY_COMPLETE','RECOVERED_SAVED_RAW'):continue
        rows.append(dict(shard=report['shard'],source=report['source'],backend=report['backend'],arm=report['arm'],view=report['view'],
            status='PENDING_SAVED_MEDIA_CHECK' if allow_group(report) else 'RETAIN_UNRESOLVED',
            original_status=report['status'],missing_batch_starts=[b['start'] for b in report.get('missing_batches',[])],
            complete_arm_rebuild=(report['source'],report['arm']) in REBUILD_ARMS))
    return dict(status='TARGETED_PHYSICAL_PLAN',rows=rows,
        complete_arm_work=[dict(source=s,arm=a,views=list(p.VIEW_IDS),
            basis='FULL_REPLACEMENT_KEEP_OLD_RESULTS' if s=='ucf-0643' else 'ALL_ORIGINAL_POSITIVE_VIEWS_MISSING') for s,a in sorted(REBUILD_ARMS)],
        semantics='missing batch starts are candidates, not actual calls; media availability and saved receipts decide physical work; all new costs in repair ledger')


def run_physical(ctx,cpu,vendor_factory,previous_runtime_stopped=True):
    from .native_baseline_colab_v4 import check_previous_worker
    from .native_baseline_media_v4 import MediaV4
    from .native_baseline_models_v4 import NativeAdapterV4
    from .native_baselines_public_v4 import Loaders
    from .native_baseline_io_v4 import extract_view, file_available
    from .native_baseline_quality_v4 import cpu_quality
    from .native_baseline_evaluation_v4 import write_post
    from .native_baseline_compact_recovery_records_v1 import read_raw
    output=[];vendor=None;selected=groups()
    write_json(ctx.output/'physical_plan.json',physical_plan(cpu))
    for shard in (1,2,3):
        folder=ctx.part(shard)
        check_previous_worker(ctx.parent/'shards'/f'part{shard}',previous_runtime_stopped)
        with runtime_session(folder,previous_runtime_stopped=previous_runtime_stopped) as owner:
            original=ctx.parent/'shards'/f'part{shard}'/'inherited_budget.json'
            limits=read_json(original).get('limits',p.physical_budget()) if original.is_file() else p.physical_budget()
            budget=Budget(folder,owner['id'],limits=limits,saved=saved_budget(ctx,shard,folder))
            ctx.cpu_budgets[shard]=budget
            reuse=RepairReuse(ctx,shard);ops=RepairOperations(folder,ctx.cache_for(folder,shard),budget,MediaV4(),reuse)
            loaders=None;adapter=None;loaded=None
            def close_model():
                nonlocal adapter,loaded
                if adapter is not None:
                    append_json(folder/'model_resources.jsonl',dict(runtime_id=owner['id'],backend=loaded,
                        actual_forward_hooks=dict(getattr(adapter,'counts',{})),**(adapter.resources() if hasattr(adapter,'resources') else {})))
                    adapter.close();adapter=None;loaded=None
            def model(sid,b):
                nonlocal vendor,loaders,adapter,loaded
                if loaded!=b:
                    close_model()
                    if vendor is None:vendor=vendor_factory()
                    if loaders is None:
                        loaders=Loaders(ops,vendor,NativeAdapterV4,None)
                        loaders.rows=list(events(ctx.parent/'shards'/f'part{shard}'/'model_loads.jsonl'))+loaders.rows
                    adapter=loaders.watermark(ctx.pack[sid],b);loaded=b
                return adapter
            def g1(sid,arm):
                source=ctx.by_id[sid];store=ops.store(f'media/{sid}/{arm}')
                row=store.row('G1/encode')
                if row.get('status')=='COMMITTED':
                    if not file_available(row.get('value')):raise Unavailable('original committed G1 media unavailable; no re-embedding substitution')
                    return ops.g1_saved(source,arm)
                # A new original stream is allowed only where no old observations survive.
                b=backend(arm) if arm!='OFF' else None
                if b and (sid,arm) not in REBUILD_ARMS:
                    raise Unavailable('NEEDS_WRITER_INPUT_DECISION: original G1 missing; no partial-arm re-embedding')
                canonical=ops.source(source)
                written=canonical if arm=='OFF' else ops.writer(source,arm,canonical,ctx.query(shard)[arm+'/'+sid]['query'],model(sid,b))
                return ops.g1(source,arm,lambda:written)
            try:
                for key,rows in selected.items():
                    part,sid,b,arm,vid=key
                    if part!=shard:continue
                    report=recover_group(ctx,key,rows)
                    if not allow_group(report):
                        output.append(dict(key=list(key),status='SKIPPED_OR_UNRESOLVED',reason=report['status']));continue
                    # Validate saved query and editor plan before any physical work.
                    try:
                        queries=ctx.query(shard);plan=ctx.plan(shard,sid)
                        for row in rows:queries[row['method']+'/'+sid]
                        view=ctx.view_map[vid];store=ops.store(f'native/{sid}/{b}/{arm}')
                        if store.record_errors:raise Unavailable('unreadable original unit receipts')
                        epoch=None if (sid,arm) in REBUILD_ARMS else report.get('epoch')
                        if epoch:
                            epoch=dict(epoch)
                            if not epoch.get('input'):
                                epoch['input']=str(ops.cache/'received'/sid/b/arm/vid/(epoch['id']+'.npy'))
                            path=store.folder/'views'/vid/'epochs.json'
                            if not path.is_file():write_json(path,[epoch])
                        def pixels():
                            received=g1(sid,arm)
                            return ops.received(ctx.by_id[sid],arm,view,plan,received,
                                lambda donor,a:g1(donor,a),decode_tag=b)
                        # Check media availability before loading a model. Missing
                        # original G1 on a partial positive arm is not re-embedded.
                        media_store=ops.store(f'media/{sid}/{arm}')
                        saved_input=bool(epoch and epoch.get('input') and Path(epoch['input']).is_file())
                        if not saved_input and arm!='OFF' and (sid,arm) not in REBUILD_ARMS and not file_available(media_store.row('G1/encode').get('value')):
                            raise Unavailable('NEEDS_WRITER_INPUT_DECISION: original G1 missing for partial arm')
                        raw,chosen=extract_view(ops,ctx.by_id[sid],b,arm,view,model(sid,b),pixels)
                        # Fixed raw epoch saved first; each method gets its own unchanged evaluation.
                        for row in rows:
                            if post_value(ctx,row)[0] is None:
                                write_post(folder,ctx.by_id[sid],row['method'],arm,view,raw,chosen,queries,plan,owner['id'])
                        output.append(dict(key=list(key),status='PHYSICAL_RETURNED',observed=int(raw['observed'].sum()),planned=view['frames'],epoch=chosen))
                    except SpaceUnavailable:raise
                    except Exception as error:
                        ops.failure('repair/'+sid+'/'+arm+'/'+vid,error)
                        output.append(dict(key=list(key),status='FAILED_RETAINED',error=f'{type(error).__name__}: {error}'))
                for row in quality_targets():
                    if row['shard']!=shard:continue
                    prior=recover_quality(ctx,row)
                    if prior['status'] in ('ALREADY_COMPLETE','RECOVERED_SAVED_RGB'):continue
                    output.append(dict(**row,status='QUALITY_UNRESOLVED',error=prior.get('error')))
            finally:
                close_model()
                if loaders is not None:loaders.close()
                budget.snapshot()
    write_json(ctx.output/'physical_recovery.json',dict(status='TARGETED_CONTINUATION_RETURNED',rows=output))
    return output
