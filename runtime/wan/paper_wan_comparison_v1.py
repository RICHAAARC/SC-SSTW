"""New fixed Wan panel: reuse C4 media and official native adapters, no C4 replay."""
from pathlib import Path
import gc
import os
import time
from main.tube_state import paper_suite_v1 as p
from . import paper_suite_records_v1 as rec


def save_npz(path,arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:np.savez_compressed(stream,**arrays)


def quality(a,b):
    import numpy as np
    per_frame=[float(np.mean((x.astype(np.float64)-y.astype(np.float64))**2)/255**2) for x,y in zip(a,b)]
    mse=float(np.mean(per_frame))
    return dict(frames=len(a),rgb_mse=mse,psnr_db=float(-10*np.log10(mse)) if mse else None,
                perfect_match=mse==0,per_frame_rgb_mse=per_frame,reference='saved OFF same second-generation view')


def run_source(sid,cohort_root=p.COHORT_ROOT,output_root=p.WAN_ROOT,vendor=None,*,adapter_factory=None,media=None):
    import numpy as np
    from main.baselines.public_fpr001_v4 import allocate_queries,frame_messages
    from main.baselines.public_metrics_v4 import evaluate_view
    from .native_baseline_io_v4 import empty_raw,validate_native
    if sid not in p.SOURCES:raise ValueError('fixed twelve-source roster required')
    output=Path(output_root)/sid/'attempt01';source=Path(cohort_root)/sid/'attempt01'
    if not (source/'result.json').is_file():
        return dict(status='DEPENDENCY_PENDING',source=sid,output=str(output),reason='C4 terminal result unavailable; no new attempt consumed')
    if not rec.reserve(output,dict(source=sid,budget=p.WAN_BUDGET,scope=p.COMPARISON_SCOPE)):
        return dict(status='READ_ONLY',output=str(output),saved=rec.optional(output/'result.json'))
    journal=rec.Journal(output,p.wan_plan());started=time.monotonic();failure=None;interrupted=False
    queries=allocate_queries([dict(id=sid,split='TEST')],os.urandom);rec.write(output/'queries.json',queries)
    maps=p.construction_maps();rec.write(output/'editor.json',maps)
    if adapter_factory is None:
        from .native_baseline_models_v4 import NativeAdapterV4
        adapter_factory=lambda backend:NativeAdapterV4(backend,vendor)
    if media is None:
        from .paper_wan_media_v1 import CohortMedia
        media=CohortMedia()
    raws={};qualities=[];resources=[];posts=[];off={};adapter=None
    def load_media(path,n,key):
        journal.call(key+'/codec_probe',lambda:media.probe(path,(n,320,512,3),fps=8))
        rgb=journal.call(key+'/codec_decode',lambda:media.decode(path,(n,320,512,3)))
        if rgb.dtype!=np.uint8 or tuple(rgb.shape)!=(n,320,512,3):raise ValueError('actual received RGB geometry invalid')
        return rgb
    def detect(pixels,physical,arm,view):
        n=len(pixels);raw=empty_raw(n,physical)
        for start,k in p.batches(n,8 if physical=='wam' else 32):
            key=f'{physical}/{arm}/{view}/read{start}'
            def call():
                value=adapter.detect(pixels[start:start+k]);validate_native(value,k,physical)
                save_npz(output/'raw_batches'/physical/arm/view/f'{start}.npz',value);return value
            value=journal.call(key,call)
            raw['observed'][start:start+k]=True
            for name in ('bits','valid','logits'):
                if name in raw:raw[name][start:start+k]=value[name]
        save_npz(output/'raw'/physical/arm/(view+'.npz'),raw)
        return raw
    try:
        canonical=load_media(source/'media/OFF/first.mp4',181,'OFF/canonical')
        for i,(view,n) in enumerate(zip(p.VIEWS,p.LENGTHS)):
            off[view]=load_media(source/f'media/input{i}/received.mp4',n,'OFF/'+view)
        for physical in p.BACKENDS:
            adapter=adapter_factory(physical)
            receipt=journal.call(physical+'/load',adapter.load);rec.write(output/'models'/(physical+'.json'),receipt)
            try:
                for view in p.VIEWS:raws[(physical,'OFF',view)]=detect(off[view],physical,'OFF',view)
                logicals=('pixelseal','frameindex') if physical=='pixelseal' else (physical,)
                for method in logicals:
                    marked=np.empty_like(canonical);q=queries[method+'/'+sid]['query']
                    for start,k in p.writer_batches(method):
                        msg=frame_messages(q,start,k) if method=='frameindex' else q
                        def embed():
                            value=adapter.embed_batch(canonical[start:start+k],msg,method=method,start=start)
                            if value.dtype!=np.uint8 or value.shape!=canonical[start:start+k].shape:
                                raise ValueError('writer returned invalid RGB geometry')
                            return value
                        marked[start:start+k]=journal.call(method+'/write'+str(start),embed)
                    for view,n in zip(p.VIEWS,p.LENGTHS):
                        path=output/'media'/method/(view+'.mp4');edited=marked[maps[view]]
                        journal.call(method+'/'+view+'/codec_encode',lambda:media.encode(edited,path,fps=8))
                        rgb=load_media(path,n,method+'/'+view)
                        # Same physical Pixel model is reused; logical FI reader has its own fixed operations.
                        raw=empty_raw(n,physical)
                        for start,k in p.batches(n,8 if physical=='wam' else 32):
                            def call():
                                value=adapter.detect(rgb[start:start+k]);validate_native(value,k,physical)
                                save_npz(output/'raw_batches'/method/'WM'/view/f'{start}.npz',value);return value
                            value=journal.call(f'{method}/WM/{view}/read{start}',call)
                            raw['observed'][start:start+k]=True
                            for name in ('bits','valid','logits'):
                                if name in raw:raw[name][start:start+k]=value[name]
                        save_npz(output/'raw'/method/'WM'/(view+'.npz'),raw);raws[(method,'WM',view)]=raw
                        qrow=journal.call(method+'/'+view+'/quality',lambda:quality(rgb,off[view]))
                        qualities.append(dict(source=sid,method=method,view=view,**qrow));rec.write(output/'quality.json',qualities)
                    del marked
            finally:
                resources.append(dict(backend=physical,counts=dict(adapter.counts),**adapter.resources()))
                adapter.close();adapter=None;rec.write(output/'resources.json',resources);gc.collect()
        # Every physical raw is saved before any query verification or truth scoring.
        rec.write(output/'blind_terminal.json',dict(status='ALL_NATIVE_RAW_SAVED',truth_used_for_detection=False))
        for method in p.METHODS:
            for arm in ('OFF','WM'):
                for view in p.VIEWS:
                    raw=raws[(p.backend(method) if arm=='OFF' else method,arm,view)]
                    mapping=[dict(kind='target',source_id=sid,index=i) for i in maps[view]]
                    def post():
                        value,arrays=evaluate_view(method,arm,raw,queries[method+'/'+sid],mapping,queries)
                        folder=output/'post'/method/arm
                        save_npz(folder/(view+'.npz'),arrays)
                        rec.write(folder/(view+'.json'),value);return value
                    value=journal.call(f'{method}/{arm}/{view}/post',post)
                    posts.append(dict(source=sid,method=method,arm=arm,view=view,**value))
    except BaseException as error:
        failure=f'{type(error).__name__}: {error}';interrupted=isinstance(error,(KeyboardInterrupt,InterruptedError))
    finally:
        if adapter is not None:
            try:adapter.close()
            except Exception as error:failure=failure or str(error)
    result=dict(status='COMPLETED' if not failure else 'STOPPED_FIXED_ATTEMPT',source=sid,
        failure=failure,interrupted=interrupted,planned=p.WAN_BUDGET,posts_returned=len(posts),
        wall_seconds=time.monotonic()-started,scope=p.COMPARISON_SCOPE)
    rec.write(output/'result.json',result);return result


def run_all(cohort_root=p.COHORT_ROOT,output_root=p.WAN_ROOT,vendor=None):
    rows=[]
    for sid in p.SOURCES:
        result=run_source(sid,cohort_root,output_root,vendor);rows.append(result)
        print(sid,result['status'],flush=True)
        if result.get('interrupted'):break
    return dict(status='INTERRUPTED' if rows[-1].get('interrupted') else 'RETURNED',sources=rows)
