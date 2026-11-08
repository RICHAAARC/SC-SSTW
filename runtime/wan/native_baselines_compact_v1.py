"""Compact CAL32/TEST500 scheduler; original v4 and DEV remain read-only."""
import gc
import os
from pathlib import Path
import signal
import time
from main.baselines import public_compact12_v1 as p
from main.baselines.public_fpr001_v4 import allocate_queries,batches,read_batch,backend
from .native_baselines_public_v4 import SavedQueries,SourcePlans,Loaders,Runner,quality_complete
from .native_baseline_storage_v1 import read_json,write_json
from .native_baseline_storage_v4 import Budget,Unavailable,runtime_session,append_json
from .native_baseline_checkpoint_v4 import ComputationCursor,tasks_for
from .native_baseline_io_v4 import extract_view,saved_raw,empty_raw,load_rgb,validate_native,remove_cache,SpaceUnavailable
from .native_baseline_evaluation_v4 import write_post
from .native_baseline_compact_reuse_v1 import Reuse,CompactOperations,read_saved_view

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'main/baselines/protocols/public_compact12_v1.json'


def inputs(output,old,config=DEFAULT_CONFIG):
    config_path=Path(config);config=p.validate_config(read_json(config_path));output=Path(output)
    roster_path=output/'roster.json'
    if roster_path.is_file():roster=read_json(roster_path)
    else:
        old_roster=Path(old)/'sources.json'
        roster=read_json(old_roster) if old_roster.is_file() else read_json(config_path.parent/config['source_list'])
        if isinstance(roster,dict):roster=roster['sources']
        p.selected_sources(roster);write_json(roster_path,roster)
    sources=p.selected_sources(roster)
    if (output/'protocol.json').is_file():p.validate_config(read_json(output/'protocol.json'))
    else:write_json(output/'protocol.json',config)
    write_json(output/'sources.json',sources)
    query_path=output/'queries.json'
    if query_path.is_file():queries=read_json(query_path)
    elif (Path(old)/'queries.json').is_file():
        queries=read_json(Path(old)/'queries.json');write_json(query_path,queries)
        write_json(output/'query_origin.json',dict(source=str(Path(old)/'queries.json'),draws_this_package=0))
    else:
        if (output/'physical_calls.jsonl').exists() or (Path(old)/'physical_calls.jsonl').exists():
            raise Unavailable('actual saved query registry missing after physical work; cannot replace it')
        queries=allocate_queries(roster,os.urandom);write_json(query_path,queries)
        write_json(output/'query_origin.json',dict(source='independent draws before this package work',new_registry=True))
    return sources,roster,SavedQueries(queries),SourcePlans(output,roster)


class CompactCursor(ComputationCursor):
    def __init__(self,output,sources,budget=None):
        self.output=Path(output);self.path=self.output/'compute_checkpoint.json';self.tasks=tasks_for(sources);self.budget=budget
        if self.path.is_file():self.state=read_json(self.path)
        elif (self.output/'physical_calls.jsonl').exists():raise Unavailable('compact computation checkpoint missing after physical work')
        else:self.state=dict(next_task=0,active_task=None,view_index=0,pending={},status='ACTIVE',origin='new compact package')
        index=self.state['next_task']
        if not isinstance(index,int) or not 0<=index<=len(self.tasks):raise ValueError('invalid compact task cursor')
        if self.state.get('protocol',p.EXPERIMENT)!=p.EXPERIMENT:raise ValueError('cursor contains another method task list')
        self.state['protocol']=p.EXPERIMENT


class CompactRunner(Runner):
    def __init__(self,*args,roster,**kwargs):
        super().__init__(*args,**kwargs);self.roster=roster;self.adapter=None;self.loaded=None

    def close_adapter(self):
        if self.adapter is not None:
            value=self.adapter.resources() if hasattr(self.adapter,'resources') else {}
            append_json(self.ops.output/'model_resources.jsonl',dict(pack=self.loaded[0],backend=self.loaded[1],
                runtime_id=self.ops.budget.runtime_id,actual_forward_hooks=dict(getattr(self.adapter,'counts',{})),**value))
            self.adapter.close();self.adapter=None;self.loaded=None;gc.collect()

    def get_adapter(self,pack,b):
        if self.loaded!=(pack,b):
            self.close_adapter();self.progress(stage='LOAD_CURRENT_MODEL',pack=pack,backend=b)
            self.adapter=self.loaders.watermark(pack,b);self.loaded=(pack,b)
        return self.adapter

    def read_arm_compact(self,source,b,arm,pack,selection):
        available=p.views();remaining=list(selection) if isinstance(selection,list) else list(range(selection,len(available)))
        failed=list(self.cursor.state.get('failed_views',[]));g1=None
        methods=('pixelseal','frameindex') if b=='pixelseal' and arm=='OFF' else (b if arm=='OFF' else arm,)
        store=self.ops.store('native/'+source['id']+'/'+b+'/'+arm)
        for index in list(remaining):
            view=available[index];self.cursor.view(index,remaining,failed)
            self.progress(stage='COMPUTE_VIEW',source=source['id'],backend=b,arm=arm,view=view['id'],processed=index+1,total=12)
            try:
                imported=self.ops.reuse.view(store,view)
                raw,epoch,read_errors=read_saved_view(store,view,b)
                if not raw['observed'].all() and not store.record_errors:
                    def pixels():
                        nonlocal g1
                        if g1 is None:g1=self.ops.g1_saved(source,arm)
                        return self.ops.received(source,arm,view,self.plans[source['id']],g1,self.donor,decode_tag=b)
                    if imported['imported']:
                        # Only never-started batches may extend the one saved old epoch, on its exact saved input.
                        prefix=view['id']+'/'+epoch['id'];todo=[(s,n) for s,n in batches(view['frames'],read_batch(b))
                            if store.row(prefix+'/batch'+str(s))['status']=='NOT_STARTED']
                        if todo and Path(epoch['input']).is_file():
                            received=load_rgb(epoch['input'],view['frames']);adapter=self.get_adapter(pack,b)
                            for start,n in todo:
                                self.guard('imported/'+prefix+'/batch'+str(start),lambda s=start,k=n:store.perform(
                                    prefix+'/batch'+str(s),'detect_'+b,lambda:adapter.detect(received[s:s+k]),arrays=True,
                                    validate=lambda value:validate_native(value,k,b),physical_key=view['id']+'/batch'+str(s)))
                            del received
                        raw,epoch,read_errors=read_saved_view(store,view,b)
                    elif any(r.get('imported_from') and k.startswith(view['id']+'/') for k,r in store.rows.items()):
                        self.ops.failure('native/'+source['id']+'/'+view['id'],Unavailable('old reader receipts exist but epoch unavailable'))
                    else:
                        prefix=view['id']+'/'+epoch['id'] if epoch else None
                        needs_reader=not prefix or any(store.row(prefix+'/batch'+str(start))['status']!='COMMITTED' for start,n in batches(view['frames'],read_batch(b)))
                        adapter=self.get_adapter(pack,b) if needs_reader else None
                        raw,epoch=extract_view(self.ops,source,b,arm,view,adapter,pixels)
            except SpaceUnavailable:raise
            except Exception as error:self.ops.failure('read/'+source['id']+'/'+view['id'],error)
            raw,epoch,read_errors=read_saved_view(store,view,b)
            if read_errors or store.record_errors:
                write_json(store.folder/'views'/view['id']/'read_errors.json',dict(batch_errors=read_errors,receipt_errors=store.record_errors))
            if not raw['observed'].all() and index not in failed:failed.append(index)
            if raw['observed'].all() and index in failed:failed.remove(index)
            for method in methods:write_post(self.ops.output,source,method,arm,view,raw,epoch,self.queries,self.plans[source['id']],self.ops.budget.runtime_id)
            write_json(store.folder/'views'/view['id']/'status.json',dict(complete=bool(raw['observed'].all()),observed=int(raw['observed'].sum()),planned=view['frames']))
            if view['generation']=='G2' and view['operation']!='T00':
                scratch=self.ops.cache/'derived'/source['id']/arm/view['id']/(b+'.npy')
                if scratch.is_file():remove_cache(scratch,self.ops.cache)
            remaining.remove(index);self.cursor.view(remaining[0] if remaining else 12,remaining,failed)
        return failed

    def quality_compact(self,source,method,pack):
        self.ops.reuse.quality(source['id'],method)
        if quality_complete(self.ops.output,source,method):return
        # A partial old quality unit stays a partial reference. No old completed LPIPS/writer rerun.
        store=self.ops.store('quality/'+source['id']+'/'+method)
        if store.record_errors or any(r.get('imported_from') for r in store.rows.values()):return
        writer=self.ops.store('writer/'+source['id']+'/'+method,cache_arrays=True)
        old_writer=any(r.get('imported_from') for r in writer.rows.values())
        adapter=None if old_writer else self.get_adapter(pack,backend(method))
        self.quality(source,method,adapter)

    def run(self):
        from .native_baseline_compact_tables_v1 import build_tables
        result=None
        try:
            for index,selection in self.cursor.queue():
                task=self.cursor.tasks[index];self.cursor.begin(index,selection);self.progress(stage='COMPUTE_TASK',task_index=index,**task)
                source=self.by_id.get(task['source']);kind=task['kind'];failed=[]
                if kind=='media':
                    media_store=self.ops.store('media/'+source['id']+'/'+task['arm']);row=media_store.row('G1/encode')
                    if media_store.record_errors:failed=[0]
                    elif row.get('imported_from') and row['status']!='COMMITTED':failed=[0]
                    elif row['status']!='COMMITTED':
                        adapter=self.get_adapter(task['pack'],task['backend']) if task['backend'] else None
                        self.guard(task['id'],lambda:self.ensure_media(source,task['arm'],adapter))
                        if self.ops.store('media/'+source['id']+'/'+task['arm']).row('G1/encode')['status']!='COMMITTED':failed=[0]
                elif kind=='native':failed=self.read_arm_compact(source,task['backend'],task['arm'],task['pack'],selection)
                elif kind=='quality':
                    self.guard(task['id'],lambda:self.quality_compact(source,task['arm'],task['pack']))
                    if not quality_complete(self.ops.output,source,task['arm']):failed=[0]
                    else:
                        path=self.ops.cache/'arrays/writer'/source['id']/task['arm']
                        if path.exists():remove_cache(path,self.ops.cache)
                elif kind=='cleanup':
                    self.close_adapter();self.ops._stores.clear()
                    unfinished={self.cursor.tasks[int(k)]['source'] for k in self.cursor.state.get('pending',{})}
                    for sid in task['source_ids']:
                        if sid in unfinished:continue
                        for path in (self.ops.cache/'derived'/sid,self.ops.cache/'g1'/sid,self.ops.cache/'sources'/sid):
                            if path.exists():remove_cache(path,self.ops.cache)
                elif kind=='post':
                    self.close_adapter();self.loaders.close()
                    result=build_tables(self.ops.output,self.sources,self.roster,self.queries,self.ops.budget,self.ops.reuse.old)
                self.cursor.finish(index,failed)
            return result if result is not None else read_json(self.ops.output/'result.json')
        finally:self.close_adapter()


def run(output=p.OUTPUT,cache=p.CACHE,old=p.OLD_OUTPUT,old_cache=p.OLD_CACHE,vendor=None,
        config=DEFAULT_CONFIG,previous_runtime_stopped=False):
    from .native_baseline_media_v4 import MediaV4
    from .native_baseline_models_v4 import NativeAdapterV4
    from .native_baseline_quality_v4 import LPIPS,measure
    output=Path(output);cache=Path(cache);old=Path(old);old_cache=Path(old_cache)
    paths=[x.resolve() for x in (output,cache,old,old_cache)]
    for i in range(len(paths)):
        for j in range(i+1,len(paths)):
            if paths[i].is_relative_to(paths[j]) or paths[j].is_relative_to(paths[i]):raise ValueError('new and old result/cache trees must be separate')
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def stop(signum,frame):raise KeyboardInterrupt('stop signal '+str(signum))
    for sig in previous:signal.signal(sig,stop)
    try:
        with runtime_session(output,previous_runtime_stopped=previous_runtime_stopped) as runtime:
            sources,roster,queries,plans=inputs(output,old,config)
            cursor=CompactCursor(output,sources)
            budget=Budget(output,runtime['id'],limits=p.physical_budget(),saved=cursor.state.get('budget'));cursor.budget=budget;cursor.save()
            ops=CompactOperations(output,cache,budget,MediaV4(),Reuse(output,old,old_cache))
            loaders=Loaders(ops,vendor,NativeAdapterV4,LPIPS)
            runner=CompactRunner(ops,sources,queries,plans,loaders,measure,cursor,roster=roster)
            try:return runner.run()
            except BaseException as error:
                write_json(output/'interruption.json',dict(status='INTERRUPTED_RETAINED',error=f'{type(error).__name__}: {error}',at=time.time()))
                raise
            finally:loaders.close();budget.snapshot()
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
