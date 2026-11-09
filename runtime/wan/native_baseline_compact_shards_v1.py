"""Disjoint complete public packs with apportioned original global allowances."""
from pathlib import Path
import signal
from main.baselines import public_compact12_v1 as p
from main.tube_state import paper_execution_v1 as schedule
from . import native_baselines_compact_v1 as base
from .native_baselines_public_v4 import Loaders
from .native_baseline_storage_v1 import read_json,write_json
from .native_baseline_storage_v4 import Budget,runtime_session,events
from .native_baseline_colab_v4 import check_previous_worker
from .native_baseline_compact_reuse_v1 import Reuse,CompactOperations
from .paper_checkpoint_v1 import Boundary,BoundaryPause


class ShardCursor(base.CompactCursor):
    def __init__(self,output,sources,shard):
        # Parent cursor indices are never interpreted in this independent task list.
        super().__init__(output,sources)
        packs={p['id'] for p in schedule.owned_packs(sources,shard)}
        self.tasks=[t for t in self.tasks if t['pack'] in packs or t['kind']=='post']
        if not 0<=self.state['next_task']<=len(self.tasks):raise ValueError('invalid shard computation cursor')
        if self.state.get('shard',shard)!=shard:raise ValueError('cursor belongs to another task shard')
        self.state['shard']=shard
    def queue(self):
        ordered=super().queue();active=self.state.get('active_task');next_task=self.state['next_task']
        primary=[row for row in ordered if row[0]>=next_task or row[0]==active]
        retry=[row for row in ordered if row[0]<next_task and row[0]!=active]
        return [row for row in primary if self.tasks[row[0]]['kind']!='post']+retry+[row for row in primary if self.tasks[row[0]]['kind']=='post']


def inherited(output,parent,sources,shard):
    path=Path(output)/'inherited_budget.json'
    if path.is_file():return read_json(path)
    rows=list(events(Path(parent)/'physical_calls.jsonl'))
    spend=schedule.inherited_spend(sources,rows)
    runtimes=len(list(events(Path(parent)/'runtimes.jsonl')))
    limits=schedule.shard_budget(sources,shard,spend,runtimes)
    pack_ids=set(limits['pack_ids'])
    loads=[r for r in events(Path(parent)/'model_loads.jsonl') if r.get('key','').split('/')[0] in pack_ids or r.get('category')=='lpips_load' and shard==0]
    value=dict(limits=limits,parent_model_loads=loads,parent=str(parent))
    write_json(path,value);return value


def run_shard(shard,vendor=None,max_packs=None,seconds=None,previous_runtime_stopped=False,
              parent=p.OUTPUT,cache_root=p.CACHE,old=p.OLD_OUTPUT,old_cache=p.OLD_CACHE,vendor_factory=None):
    from .native_baseline_media_v4 import MediaV4
    from .native_baseline_models_v4 import NativeAdapterV4
    from .native_baseline_quality_v4 import LPIPS,measure
    from .paper_execution_v1 import OUTPUT as EXECUTION_OUTPUT
    check_previous_worker(EXECUTION_OUTPUT/'owners/finalizer',previous_runtime_stopped)
    parent=Path(parent);output=parent/'shards'/('part'+str(shard));cache=Path(cache_root)/'shards'/('part'+str(shard))
    roots=[Path(v).resolve() for v in (parent,cache_root,old,old_cache)]
    for i,a in enumerate(roots):
        for b in roots[i+1:]:
            if a.is_relative_to(b) or b.is_relative_to(a):raise ValueError('separate result and cache roots required')
    if 'Video-WM_Result' in cache.resolve().parts:raise ValueError('cache cannot be inside protected results')
    checkpoint=output/'compute_checkpoint.json'
    if checkpoint.is_file() and read_json(checkpoint).get('status')=='DONE':
        state=read_json(checkpoint)
        if not base.unresolved_gate_slots(state) or not any(base.retired_recovery_gate(r) for r in events(output/'failures.jsonl')):
            return dict(status='READ_ONLY_COMPLETED',shard=shard,output=str(output))
    # Same experiment's unsharded runner must be stopped before redistributing its unfinished work.
    check_previous_worker(parent,previous_runtime_stopped)
    if (parent/'physical_calls.jsonl').exists() and not (parent/'queries.json').is_file():
        raise ValueError('consumed compact parent query registry unavailable; cannot replace it with another registry')
    old_inputs=parent if (parent/'queries.json').is_file() else Path(old)
    roster_path=next((path for path in (output/'roster.json',parent/'roster.json',Path(old)/'sources.json') if path.is_file()),None)
    if roster_path is None:
        config=read_json(base.DEFAULT_CONFIG);roster_path=base.DEFAULT_CONFIG.parent/config['source_list']
    roster=read_json(roster_path)
    if isinstance(roster,dict):roster=roster['sources']
    sources=p.selected_sources(roster)
    frozen_path=output/'inherited_budget.json'
    if frozen_path.is_file():preview=read_json(frozen_path)['limits']
    else:preview=schedule.shard_budget(sources,shard,schedule.inherited_spend(sources,list(events(parent/'physical_calls.jsonl'))),len(list(events(parent/'runtimes.jsonl'))))
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def stop(signum,frame):raise KeyboardInterrupt('runtime stopped')
    for sig in previous:signal.signal(sig,stop)
    try:
        with runtime_session(output,previous_runtime_stopped=previous_runtime_stopped,runtime_limit=preview['runtime_limit']) as runtime:
            own=schedule.owned_packs(sources,shard);selected=[s for pack in own for s in pack['sources']]
            sources,roster,queries,plans=base.inputs(output,old_inputs,query_sources=selected)
            frozen=inherited(output,parent,sources,shard);limits=frozen['limits']
            cursor=ShardCursor(output,sources,shard)
            base.restore_recovery_pending(output,cursor)
            if not cursor.queue():return dict(status='READ_ONLY_COMPLETED',shard=shard,output=str(output))
            if vendor_factory is not None:vendor=vendor_factory()
            saved=cursor.state.get('budget') or dict(limits['inherited'],ledger_offset=0)
            budget=Budget(output,runtime['id'],limits=limits,saved=saved);cursor.budget=budget;cursor.save()
            reuse=Reuse(output,parent,cache_root,fallback=old,fallback_cache=old_cache,same_experiment=True)
            ops=CompactOperations(output,cache,budget,MediaV4(),reuse)
            loaders=Loaders(ops,vendor,NativeAdapterV4,LPIPS)
            loaders.rows=frozen['parent_model_loads']+loaders.rows;loaders.initial_lpips_recovery=shard!=0
            runner=base.CompactRunner(ops,selected,queries,plans,loaders,measure,cursor,roster=roster if shard==0 else [])
            runner.boundary=Boundary(max_packs,seconds);runner.bounded_schedule=True
            try:
                result=runner.run();write_json(output/'completion.json',dict(status='TASK_QUEUE_RETURNED',result=result))
                return dict(status='SHARD_RETURNED',shard=shard,output=str(output),result=result)
            except BoundaryPause as error:
                value=dict(status='PAUSED_BOUNDARY',shard=shard,output=str(output),reason=str(error),checkpoint=str(cursor.path))
                write_json(output/'pause.json',value);return value
            finally:loaders.close();budget.snapshot();cursor.save()
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
