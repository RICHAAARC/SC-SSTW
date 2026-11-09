"""One resumable scheduler for the fixed public v4 protocol. Passive on import."""
from collections import Counter
import gc
import os
from pathlib import Path
import signal
import time
import errno

from main.baselines.public_fpr001_v4 import (BACKENDS, METHODS, OUTPUT, CACHE, views,
    validate_config, validate_sources, allocate_queries, source_plan, physical_budget)
from .native_baseline_storage_v1 import read_json, write_json
from .native_baseline_storage_v4 import (Budget, UnitStore, Unavailable, RecoveryExhausted, runtime_session,
    events, append_json, infrastructure, retired_recovery_gate)
from .native_baseline_io_v4 import (MediaOperations, SpaceUnavailable, check_space, extract_view,
    remove_cache, load_rgb, saved_raw)
from .native_baseline_evaluation_v4 import write_post, build_tables
from .native_baseline_checkpoint_v4 import ComputationCursor

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'main/baselines/protocols/public_native_fpr001_v4.json'


def load_inputs(config=DEFAULT_CONFIG):
    config=Path(config);value=validate_config(read_json(config))
    source_path=config.parent/value['source_list'];source_data=read_json(source_path)
    sources=source_data['sources'] if isinstance(source_data,dict) else source_data
    return value,validate_sources(sources)


def initialize(output,config,sources):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Existing semantic inputs are used, not compared through hashes or source identities.
    if (output/'sources.json').is_file():sources=validate_sources(read_json(output/'sources.json'))
    else:write_json(output/'sources.json',sources)
    if (output/'protocol.json').is_file():validate_config(read_json(output/'protocol.json'))
    else:write_json(output/'protocol.json',config)
    path=output/'queries.json'
    if path.is_file():queries=read_json(path)
    else:
        # A lost query record after physical work cannot be silently replaced with a new query.
        if (output/'physical_calls.jsonl').is_file():raise Unavailable('saved query file is actually missing after physical work; restore it before continuing')
        queries=allocate_queries(sources,os.urandom);write_json(path,queries)
    return sources,SavedQueries(queries),SourcePlans(output,sources)


class SavedQueries(dict):
    """Check the query actually being used, without validating completed historical work."""
    def __getitem__(self,key):
        value=super().__getitem__(key);method,sid=key.split('/',1);n=248 if method=='frameindex' else METHODS[method]
        for field in ('query','wrong_query'):
            if field not in value:continue
            q=value[field]
            if field=='wrong_query' and q is None:continue
            if len(q)!=n or any(bit not in (0,1) for bit in q):raise ValueError('invalid current query: '+key+'/'+field)
        if method=='frameindex' and value['query']!=super().__getitem__('pixelseal/'+sid)['query'][:248]:
            raise ValueError('current FrameIndex query must use Pixel query prefix')
        return value


class SourcePlans(dict):
    """Load the frozen editor plan only when its source is actually needed."""
    def __init__(self,output,sources):
        super().__init__();self.output=Path(output);self.sources={s['id']:s for s in sources}
        self.donors=[s for s in sources if s['split']=='DEV']

    def __missing__(self,sid):
        path=self.output/'editor'/sid/'plan.json'
        if path.is_file():value=read_json(path)
        else:value=source_plan(self.sources[sid],self.donors);write_json(path,value)
        self[sid]=value;return value



class Loaders:
    """Record every actual load; successful reloads need no recovery quota."""
    def __init__(self,ops,vendor,adapter_factory,quality_factory):
        self.ops=ops;self.vendor=vendor;self.adapter_factory=adapter_factory;self.quality_factory=quality_factory
        self.path=ops.output/'model_loads.jsonl';self.rows=list(events(self.path));self.extra_watermark=0;self.lpips=None

    def perform(self,key,category,model):
        previous=[r for r in self.rows if r.get('key')==key and r.get('status')=='STARTED']
        last=[r for r in self.rows if r.get('key')==key]
        if last and last[-1]['status']=='FAILED_PERMANENT' and not retired_recovery_gate(last[-1]):raise Unavailable('model load method/config failure retained: '+key)
        if last and last[-1].get('runtime_id')==self.ops.budget.runtime_id and last[-1]['status']!='RETURNED' and not retired_recovery_gate(last[-1]):
            raise Unavailable('no same-runtime failed model-load retry: '+key)
        recovery=bool(previous) or category=='lpips_load' and (getattr(self,'initial_lpips_recovery',False) or bool([r for r in self.rows if r.get('category')=='lpips_load']))
        self.ops.budget.charge(category,'loaders',key,recovery=recovery)
        if recovery and category=='model_load':self.extra_watermark+=1
        record=dict(key=key,category=category,recovery=recovery,runtime_id=self.ops.budget.runtime_id,status='STARTED',at=time.time())
        append_json(self.path,record);self.rows.append(record);start=time.monotonic()
        try:
            value=model.load()
        except BaseException as error:
            record=dict(record,status='INFRA_FAILED' if infrastructure(error) or isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED_PERMANENT',error=str(error),seconds=time.monotonic()-start)
            append_json(self.path,record);self.rows.append(record);model.close()
            if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
            raise Unavailable(str(error)) from error
        record=dict(record,status='RETURNED',seconds=time.monotonic()-start,details=value)
        append_json(self.path,record);self.rows.append(record);return model

    def watermark(self,pack,backend):
        return self.perform(pack+'/'+backend,'model_load',self.adapter_factory(backend,self.vendor))

    def quality(self):
        if self.lpips is None:self.lpips=self.perform('lpips','lpips_load',self.quality_factory())
        return self.lpips

    def close(self):
        if self.lpips:
            append_json(self.ops.output/'quality_resources.jsonl',dict(runtime_id=self.ops.budget.runtime_id,
                actual_alex_forwards=getattr(self.lpips,'forwards',None)))
            self.lpips.close();self.lpips=None


def native_complete(output,source,backend,arm,completion_cache=None):
    """Explicit historical diagnostic only; never called by the default resume scheduler."""
    key=(source['id'],backend,arm)
    if completion_cache is not None and key in completion_cache:return completion_cache[key]
    folder=Path(output)/'units/native'/source['id']/backend/arm/'views'
    complete=False
    try:
        if all((folder/v['id']/'status.json').is_file() and read_json(folder/v['id']/'status.json').get('complete') for v in views()):
            from types import SimpleNamespace
            store=UnitStore(output,'native/'+source['id']+'/'+backend+'/'+arm,SimpleNamespace(runtime_id='read-only'))
            complete=all(saved_raw(store,v['id'],v['frames'],backend)[0]['observed'].all() for v in views())
    except (OSError,ValueError,KeyError,TypeError):complete=False
    complete=bool(complete)
    if completion_cache is not None:completion_cache[key]=complete
    return complete


def quality_complete(output,source,method):
    folder=Path(output)/'units/quality'/source['id']/method/'quality'
    for stage in ('precodec','G1','G2-identity'):
        path=folder/(stage+'.json')
        if not path.is_file():return False
        value=read_json(path)
        if not quality_stage_complete(value):return False
    return True


def quality_stage_complete(value):
    # LPIPS is no longer a requested metric. Its missing rows stay missing.
    import math
    frames=value.get('frames',[]);edges=value.get('edges',[])
    if not isinstance(frames,list) or not isinstance(edges,list) or len(frames)!=181 or len(edges)!=180:return False
    def finite(x):return type(x) in (int,float) and math.isfinite(x)
    return all(isinstance(r,dict) and finite(r.get('mse')) and finite(r.get('ssim'))
        and (r.get('psnr_db') is None or finite(r['psnr_db'])) for r in frames) and all(
        isinstance(r,dict) and finite(r.get('residual_difference_rms')) for r in edges)


class Runner:
    def __init__(self,ops,sources,queries,plans,loaders,measure,cursor=None):
        self.ops=ops;self.sources=sources;self.queries=queries;self.plans=plans;self.loaders=loaders;self.measure=measure
        self.by_id={s['id']:s for s in sources}
        self._completion={};self.started=time.monotonic();self.cursor=cursor or ComputationCursor(ops.output,sources,ops.budget)

    def progress(self,**value):
        value.update(at=time.time(),runtime_id=self.ops.budget.runtime_id,elapsed_seconds=time.monotonic()-self.started)
        write_json(self.ops.output/'progress.json',value);print(value,flush=True)

    def guard(self,where,call):
        try:return call()
        except SpaceUnavailable:raise
        except Exception as error:
            if isinstance(error,OSError) and error.errno==errno.ENOSPC:raise SpaceUnavailable('actual storage write failed: disk full') from error
            self.ops.failure(where,error);return None

    def donor(self,sid,arm):return self.ops.g1_saved(self.by_id[sid],arm)

    def prepare_off(self,source):
        canonical=self.ops.source(source)
        return self.ops.g1(source,'OFF',lambda:canonical)

    def prepare_wm(self,source,method,adapter):
        # A completed native/quality stream needs neither writer replay nor its cleaned cache.
        media=self.ops.store('media/'+source['id']+'/'+method)
        if media.row('G1/encode')['status']=='COMMITTED' and quality_complete(self.ops.output,source,method):
            return self.ops.g1_saved(source,method)
        canonical=self.ops.source(source)
        written=self.ops.writer(source,method,canonical,self.queries[method+'/'+source['id']]['query'],adapter)
        return self.ops.g1(source,method,lambda:written)

    def ensure_media(self,source,arm,adapter=None):
        store=self.ops.store('media/'+source['id']+'/'+arm)
        row=store.row('G1/encode')
        if row['status']=='COMMITTED':return
        self.progress(stage='PREPARE_MISSING_MEDIA',source=source['id'],arm=arm)
        return self.prepare_off(source) if arm=='OFF' else self.prepare_wm(source,arm,adapter)

    def read_arm(self,source,backend,arm,adapter,start_view=0):
        methods=('pixelseal','frameindex') if backend=='pixelseal' and arm=='OFF' else (backend if arm=='OFF' else arm,)
        g1=None
        def pixels(view):
            nonlocal g1
            if g1 is None:g1=self.ops.g1_saved(source,arm)
            return self.ops.received(source,arm,view,self.plans[source['id']],g1,self.donor,decode_tag=backend)
        available=views()
        remaining=list(start_view) if isinstance(start_view,list) else list(range(start_view,len(available)))
        failed=list(self.cursor.state.get('failed_views',[]))
        while remaining:
            index=remaining[0];view=available[index]
            self.cursor.view(index,remaining,failed)
            self.progress(stage='COMPUTE_VIEW',source=source['id'],backend=backend,arm=arm,view=view['id'],processed=index+1,total=40)
            raw,epoch=extract_view(self.ops,source,backend,arm,view,adapter,lambda v=view:pixels(v))
            if not raw['observed'].all() and index not in failed:failed.append(index)
            elif raw['observed'].all() and index in failed:failed.remove(index)
            # All native arrays above are durable before query verification or truth accounting.
            for method in methods:
                write_post(self.ops.output,source,method,arm,view,raw,epoch,self.queries,self.plans[source['id']],self.ops.budget.runtime_id)
            # The exact epoch input remains on Drive if any batch is missing.
            # Decoder scratch can be dropped after being copied to that epoch. G2-ID stays for quality.
            if view['generation']=='G2' and view['operation']!='T00':
                scratch=self.ops.cache/'derived'/source['id']/arm/view['id']/(backend+'.npy')
                if scratch.is_file():remove_cache(scratch,self.ops.cache)
            remaining.pop(0)
            self.cursor.view(remaining[0] if remaining else 40,remaining,failed)
        del g1
        return failed

    def quality(self,source,method,adapter):
        if quality_complete(self.ops.output,source,method):return
        from main.baselines.public_fpr001_v4 import backend
        self.progress(stage='QUALITY_MISSING',source=source['id'],method=method)
        canonical=self.ops.source(source);store=self.ops.store('quality/'+source['id']+'/'+method)
        model=None  # User accepted absent LPIPS; do not load or rerun it.
        for stage in ('precodec','G1','G2-identity'):
            path=store.folder/'quality'/(stage+'.json')
            if path.is_file() and quality_stage_complete(read_json(path)):continue
            def run_stage():
                if stage=='precodec':
                    left=canonical
                    right=self.ops.writer(source,method,canonical,self.queries[method+'/'+source['id']]['query'],adapter)
                else:
                    left=self.ops.g1_saved(source,'OFF');right=self.ops.g1_saved(source,method)
                    if stage=='G2-identity':
                        view=next(v for v in views() if v['id']=='G2/T00')
                        left=self.ops.received(source,'OFF',view,self.plans[source['id']],left,self.donor,decode_tag=backend(method))
                        right=self.ops.received(source,method,view,self.plans[source['id']],right,self.donor,decode_tag=backend(method))
                return self.measure(store,stage,left,right,model)
            self.guard('quality/'+source['id']+'/'+method+'/'+stage,run_stage)

    def run(self):
        cursor=self.cursor;adapter=None;loaded=None;result=None
        def close_adapter():
            nonlocal adapter,loaded
            if adapter is not None:
                resources=adapter.resources() if hasattr(adapter,'resources') else {}
                append_json(self.ops.output/'model_resources.jsonl',dict(pack=loaded[0],backend=loaded[1],runtime_id=self.ops.budget.runtime_id,
                    actual_forward_hooks=dict(getattr(adapter,'counts',{})),**resources))
                adapter.close();adapter=None;loaded=None;self.ops._stores.clear();gc.collect()
        try:
            for index,selection in cursor.queue():
                task=cursor.tasks[index];cursor.begin(index,selection)
                self.progress(stage='COMPUTE_TASK',task_index=index,**task)
                kind=task['kind'];source=self.by_id.get(task['source'])
                wanted=(task['pack'],task['backend']) if task['backend'] else None
                if wanted!=loaded:
                    close_adapter()
                    if wanted:
                        self.progress(stage='LOAD_CURRENT_MODEL',pack=wanted[0],backend=wanted[1],source=task['source'])
                        adapter=self.loaders.watermark(*wanted);loaded=wanted
                failed=[]
                try:
                    if kind=='media':self.ensure_media(source,task['arm'],adapter)
                    elif kind=='native':failed=self.read_arm(source,task['backend'],task['arm'],adapter,selection)
                    elif kind=='quality':
                        self.quality(source,task['arm'],adapter)
                        if not quality_complete(self.ops.output,source,task['arm']):failed=[0]
                        elif not any(cursor.tasks[int(k)]['source']==source['id'] and cursor.tasks[int(k)]['arm']==task['arm'] for k in cursor.state.get('pending',{})):
                            for path in (self.ops.cache/'arrays/writer'/source['id']/task['arm'],self.ops.cache/'arrays/jpeg'/source['id']/task['arm'],self.ops.cache/'derived'/source['id']/task['arm']):
                                if path.exists():remove_cache(path,self.ops.cache)
                    elif kind=='cleanup':
                        unfinished={cursor.tasks[int(k)]['source'] for k in cursor.state.get('pending',{})}
                        for sid in task['source_ids']:
                            if sid in unfinished:continue
                            for path in (self.ops.cache/'derived'/sid,self.ops.cache/'g1'/sid,self.ops.cache/'sources'/sid):
                                if path.exists():remove_cache(path,self.ops.cache)
                    elif kind=='post':
                        self.loaders.close()
                        result=build_tables(self.ops.output,self.sources,self.queries,self.plans,self.ops.budget)
                except SpaceUnavailable:raise
                except Exception as error:
                    self.ops.failure(task['id'],error)
                    self.progress(stage='CURRENT_TASK_READ_OR_COMPUTE_FAILED',task=task,error=f'{type(error).__name__}: {error}')
                    # Keep the current computation and its precise view cursor for the next runtime.
                    raise
                cursor.finish(index,failed)
            return result if result is not None else read_json(self.ops.output/'result.json')
        finally:close_adapter()


def resource_projection(output,budget):
    totals=Counter();counts=Counter()
    for path in (Path(output)/'units').rglob('calls.jsonl'):
        for row in events(path):
            if row.get('status')=='RETURNED_NOT_COMMITTED' and row.get('category'):
                totals[row['category']]+=row.get('seconds',0);counts[row['category']]+=row.get('count',1)
    return dict(basis='actual returned DEV calls; not an additional pilot',classes={k:dict(
        observed_calls=counts[k],seconds=totals[k],mean_seconds=totals[k]/counts[k],
        remaining_base_calls=max(0,budget.limits['base'][k]-budget.base[k]),
        projected_remaining_seconds=max(0,budget.limits['base'][k]-budget.base[k])*totals[k]/counts[k]) for k in counts})


def run(output=OUTPUT,cache=CACHE,vendor=None,config=DEFAULT_CONFIG,previous_runtime_stopped=False):
    from .native_baseline_media_v4 import MediaV4
    from .native_baseline_models_v4 import NativeAdapterV4
    from .native_baseline_quality_v4 import LPIPS,measure
    config,sources=load_inputs(config);output=Path(output);cache=Path(cache)
    if cache.resolve().is_relative_to(output.resolve()) or output.resolve().is_relative_to(cache.resolve()):raise ValueError('cache and protected results must be separate trees')
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
    def stop(signum,frame):raise KeyboardInterrupt('stop signal '+str(signum))
    for sig in previous:signal.signal(sig,stop)
    try:
        with runtime_session(output,previous_runtime_stopped=previous_runtime_stopped) as runtime:
            started=time.monotonic()
            def restoring(stage,**extra):
                row=dict(stage=stage,runtime_id=runtime['id'],at=time.time(),elapsed_seconds=time.monotonic()-started,**extra)
                write_json(output/'progress.json',row);print(row,flush=True)
            restoring('READ_SAVED_QUERY_RECORD',processed=0,total=len(sources))
            sources,queries,plans=initialize(output,config,sources)
            cursor=ComputationCursor(output,sources)
            restoring('RESTORE_COMPUTE_CHECKPOINT',position=cursor.state)
            budget=Budget(output,runtime['id'],progress=restoring,saved=cursor.state.get('budget'))
            cursor.budget=budget;cursor.save()
            ops=MediaOperations(output,cache,budget,MediaV4())
            loaders=Loaders(ops,vendor,NativeAdapterV4,LPIPS);runner=Runner(ops,sources,queries,plans,loaders,measure,cursor)
            try:return runner.run()
            except BaseException as error:
                write_json(output/'interruption.json',dict(status='PAUSED_SPACE' if isinstance(error,SpaceUnavailable) else 'INTERRUPTED_RETAINED',
                    error=f'{type(error).__name__}: {error}',runtime_id=runtime['id'],at=time.time(),resume='same Drive checkpoint; infrastructure costs recorded without recovery quota'))
                raise
            finally:loaders.close();budget.snapshot()
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
