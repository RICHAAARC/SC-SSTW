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
    events, append_json, infrastructure)
from .native_baseline_io_v4 import (MediaOperations, SpaceUnavailable, check_space, extract_view,
    remove_cache, load_rgb, saved_raw)
from .native_baseline_evaluation_v4 import write_post, build_tables

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'experiments/wan_state_clock/configs/public_native_fpr001_v4.json'


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
    for s in sources:
        for method,bits in METHODS.items():
            q=queries[method+'/'+s['id']];n=248 if method=='frameindex' else bits
            if len(q['query'])!=n or any(x not in (0,1) for x in q['query']):raise ValueError('unreadable/invalid saved query')
            if s['split']!='CAL' and (len(q['wrong_query'])!=n or any(x not in (0,1) for x in q['wrong_query'])):raise ValueError('invalid saved wrong-query')
        if queries['frameindex/'+s['id']]['query']!=queries['pixelseal/'+s['id']]['query'][:248]:raise ValueError('FrameIndex owner must use the fixed Pixel query prefix')
    donors=[s for s in sources if s['split']=='DEV'];plans={}
    for s in sources:
        path=output/'editor'/s['id']/'plan.json'
        if path.is_file():plans[s['id']]=read_json(path)
        else:plans[s['id']]=source_plan(s,donors);write_json(path,plans[s['id']])
    return sources,queries,plans


class Loaders:
    """Actual model loads, including bounded new-runtime reloads; no automatic retry loops."""
    def __init__(self,ops,vendor,adapter_factory,quality_factory):
        self.ops=ops;self.vendor=vendor;self.adapter_factory=adapter_factory;self.quality_factory=quality_factory
        self.path=ops.output/'model_loads.jsonl';self.rows=list(events(self.path));self.extra_watermark=0;self.lpips=None

    def perform(self,key,category,model):
        previous=[r for r in self.rows if r.get('key')==key and r.get('status')=='STARTED']
        last=[r for r in self.rows if r.get('key')==key]
        if last and last[-1]['status']=='FAILED_PERMANENT':raise Unavailable('model load method/config failure retained: '+key)
        if any(r.get('runtime_id')==self.ops.budget.runtime_id for r in previous):raise Unavailable('no same-runtime model-load retry: '+key)
        recovery=bool(previous) or category=='lpips_load' and bool([r for r in self.rows if r.get('category')=='lpips_load'])
        if recovery and category=='model_load' and self.extra_watermark>=3:raise RecoveryExhausted('three additional watermark loads used in this runtime')
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
        if len(value.get('frames',[]))!=181 or value.get('lpips_observed_frames')!=181:return False
    return True


class Runner:
    def __init__(self,ops,sources,queries,plans,loaders,measure):
        self.ops=ops;self.sources=sources;self.queries=queries;self.plans=plans;self.loaders=loaders;self.measure=measure
        self.by_id={s['id']:s for s in sources}
        self._completion={}

    def progress(self,**value):
        value.update(at=time.time(),runtime_id=self.ops.budget.runtime_id)
        write_json(self.ops.output/'progress.json',value);print(value,flush=True)

    def guard(self,where,call):
        try:return call()
        except SpaceUnavailable:raise
        except Exception as error:
            if isinstance(error,OSError) and error.errno==errno.ENOSPC:raise SpaceUnavailable('actual storage write failed: disk full') from error
            self.ops.failure(where,error);return None

    def donor(self,sid,arm):return self.ops.g1_saved(self.by_id[sid],arm)

    def complete(self,source,backend,arm):return native_complete(self.ops.output,source,backend,arm,self._completion)

    def source_complete(self,source):
        return all(self.complete(source,b,'OFF') for b in BACKENDS) and (
            source['split']=='CAL' or all(self.complete(source,'pixelseal' if m=='frameindex' else m,m)
                and quality_complete(self.ops.output,source,m) for m in METHODS))

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

    def read_arm(self,source,backend,arm,adapter):
        methods=('pixelseal','frameindex') if backend=='pixelseal' and arm=='OFF' else (backend if arm=='OFF' else arm,)
        g1=None
        def pixels(view):
            nonlocal g1
            if g1 is None:g1=self.ops.g1_saved(source,arm)
            return self.ops.received(source,arm,view,self.plans[source['id']],g1,self.donor,decode_tag=backend)
        complete=True
        for view in views():
            raw,epoch=extract_view(self.ops,source,backend,arm,view,adapter,lambda v=view:pixels(v))
            complete=complete and bool(raw['observed'].all())
            # All native arrays above are durable before query verification or truth accounting.
            for method in methods:
                write_post(self.ops.output,source,method,arm,view,raw,epoch,self.queries,self.plans[source['id']],self.ops.budget.runtime_id)
            # The exact epoch input remains on Drive if any batch is missing.
            # Decoder scratch can be dropped after being copied to that epoch. G2-ID stays for quality.
            if view['generation']=='G2' and view['operation']!='T00':
                scratch=self.ops.cache/'derived'/source['id']/arm/view['id']/(backend+'.npy')
                if scratch.is_file():remove_cache(scratch,self.ops.cache)
        del g1
        self._completion[(source['id'],backend,arm)]=complete

    def quality(self,source,method,adapter):
        if quality_complete(self.ops.output,source,method):return
        from main.baselines.public_fpr001_v4 import backend
        canonical=self.ops.source(source);store=self.ops.store('quality/'+source['id']+'/'+method)
        model=self.guard('load/LPIPS',self.loaders.quality)
        for stage in ('precodec','G1','G2-identity'):
            path=store.folder/'quality'/(stage+'.json')
            if path.is_file() and read_json(path).get('lpips_observed_frames')==181:continue
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

    def backend_pack(self,pack,sources,backend,*,dev=False):
        wm_methods=('pixelseal','frameindex') if backend=='pixelseal' else (backend,)
        def needed(s):
            return not self.complete(s,backend,'OFF') or s['split']!='CAL' and any(
                not self.complete(s,backend,m) or not quality_complete(self.ops.output,s,m) for m in wm_methods)
        if not any(needed(s) for s in sources):return
        adapter=self.loaders.watermark(pack,backend)
        try:
            # All DEV same-method WM clips exist before any T12 donor lookup.
            if dev:
                for begin in range(0,len(sources),16):
                    for source in sources[begin:begin+16]:
                        for method in wm_methods:self.guard('prepare/'+source['id']+'/'+method,lambda s=source,m=method:self.prepare_wm(s,m,adapter))
            for source in sources:
                if not needed(source):continue
                self.progress(pack=pack,source=source['id'],backend=backend)
                self.guard('native/'+source['id']+'/'+backend+'/OFF',lambda:self.read_arm(source,backend,'OFF',adapter))
                if source['split']!='CAL':
                    for method in wm_methods:
                        if self.complete(source,backend,method) and quality_complete(self.ops.output,source,method):continue
                        if not dev:self.guard('prepare/'+source['id']+'/'+method,lambda m=method:self.prepare_wm(source,m,adapter))
                        self.guard('native/'+source['id']+'/'+method,lambda m=method:self.read_arm(source,backend,m,adapter))
                        self.guard('quality/'+source['id']+'/'+method,lambda m=method:self.quality(source,m,adapter))
                        if self.complete(source,backend,method) and quality_complete(self.ops.output,source,method):
                            for path in (self.ops.cache/'arrays/writer'/source['id']/method,self.ops.cache/'arrays/jpeg'/source['id']/method,self.ops.cache/'derived'/source['id']/method):
                                if path.exists():remove_cache(path,self.ops.cache)
            resources=adapter.resources() if hasattr(adapter,'resources') else {}
            append_json(self.ops.output/'model_resources.jsonl',dict(pack=pack,backend=backend,runtime_id=self.ops.budget.runtime_id,
                actual_forward_hooks=dict(getattr(adapter,'counts',{})),**resources))
        finally:adapter.close();self.ops._stores.clear();gc.collect()

    def run(self):
        if all(self.source_complete(s) for s in self.sources):
            self.progress(stage='CPU_POST_FROM_ACTUAL_CHECKPOINTS')
            self.loaders.close()
            return build_tables(self.ops.output,self.sources,self.queries,self.plans,self.ops.budget)
        dev=[s for s in self.sources if s['split']=='DEV']
        for source in dev:self.guard('prepare/'+source['id']+'/OFF',lambda s=source:self.prepare_off(s))
        for backend in BACKENDS:self.guard('DEV/'+backend,lambda b=backend:self.backend_pack('DEV-bank',dev,b,dev=True))
        write_json(self.ops.output/'dev_resource_projection.json',resource_projection(self.ops.output,self.ops.budget))
        for split in ('CAL','TEST'):
            selected=[s for s in self.sources if s['split']==split]
            for start in range(0,len(selected),16):
                pack=f'{split}-{start//16:02}';sources=selected[start:start+16]
                if all(self.source_complete(s) for s in sources):continue
                check_space(self.ops.output);check_space(self.ops.cache)
                for source in sources:
                    if not self.source_complete(source):self.guard('prepare/'+source['id']+'/OFF',lambda s=source:self.prepare_off(s))
                for backend in BACKENDS:self.guard(pack+'/'+backend,lambda b=backend:self.backend_pack(pack,sources,b))
                for source in sources:
                    done=self.source_complete(source)
                    if done:
                        for path in (self.ops.cache/'derived'/source['id'],self.ops.cache/'g1'/source['id'],self.ops.cache/'sources'/source['id']):
                            if path.exists():remove_cache(path,self.ops.cache)
                self.ops.budget.snapshot()
        self.progress(stage='CPU_POST_AND_TABLES')
        self.loaders.close()
        return build_tables(self.ops.output,self.sources,self.queries,self.plans,self.ops.budget)


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
            sources,queries,plans=initialize(output,config,sources)
            budget=Budget(output,runtime['id']);ops=MediaOperations(output,cache,budget,MediaV4())
            loaders=Loaders(ops,vendor,NativeAdapterV4,LPIPS);runner=Runner(ops,sources,queries,plans,loaders,measure)
            try:return runner.run()
            except BaseException as error:
                write_json(output/'interruption.json',dict(status='PAUSED_SPACE' if isinstance(error,SpaceUnavailable) else 'INTERRUPTED_RETAINED',
                    error=f'{type(error).__name__}: {error}',runtime_id=runtime['id'],at=time.time(),resume='same Drive package within fixed infrastructure budgets'))
                raise
            finally:loaders.close();budget.snapshot()
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
