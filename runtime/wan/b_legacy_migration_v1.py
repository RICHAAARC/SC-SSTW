"""Card77 user-only E12 operator comparison on existing C73 pixel tensors."""
from pathlib import Path
import json
import time
import resource
import torch
from main.tube_state import b_legacy_migration_v1 as method
from runtime.wan import c73_identity_v1 as c73
from runtime.wan.b_legacy_migration_evaluation_v1 import evaluate

dump=c73.dump
DEFAULT_MODEL_ID=c73.DEFAULT_MODEL_ID


def fixed_denominator():
    return dict(arms=3,layers=2,histories=2,targets=2,keys=2,logical_vectors=48,
        identity_slots=2208,positive_slots=8,negative_slots=2200,new_vectors=36,
        new_identity_slots=1656,reused_vectors=12,reused_identity_slots=552,
        new_positive_slots=6,new_negative_slots=1650,VAE_load=1,E=12,
        full_E=6,local_E=6,nominal_encoder_chunks=294,D=0,codec=0,quant=0,FFT=0,
        new_candidates=0,source_generation=0,pixel_tensor_loads=6,saved_local_JSON_reads=12,
        new_ROI_dots=9216,new_pooled_coordinates=2304,net_user_bits=32,ledger_units=72)


def protocol():
    return dict(scope="existing Wan C73 operator control, not equivalent old-method migration",
        fixed_denominator=fixed_denominator(),A_ID=.5,rho_strict_less=1/16,
        full="181 received frames, one cache reset E, normalized posterior mode, float32",
        local="same five-context/four-target nine-frame reset as C73",
        targets=[[61,16],[109,28]],writer_changed=False,default_receiver_changed=False,
        old_framewise_control="archived score replay and isolated CPU core only; no new old-VAE run",
        MP4_local="saved same-C73 observations reused across execution runs, not a new E",
        attribution_limit="readout runtime differences cannot be attributed exclusively to history",
        new_payload_read=False,resource_hard_caps=None,scientific_pass=None)


def units():
    values={"model":"vae_load","evaluation":"post_numeric_evaluation",
            "reference/prior_result":"saved_context_read","reference/legacy":"archived_summary_read"}
    values.update({f"book/{k}":"identity_book" for k in method.KEY_IDS})
    for arm in method.ARMS:
        for layer in method.LAYERS:
            values[f"load/{arm}/{layer}"]="saved_pixel_tensor_load"
            values[f"full_E/{arm}/{layer}"]="posterior_encode"
            for key in method.KEY_IDS:values[f"full_read/{arm}/{layer}/{key}"]="full_identity_observation"
        for target in range(2):
            values[f"prepare/{arm}/{target}"]="local_preparation"
            values[f"local_E/{arm}/{target}"]="posterior_encode"
            for key in method.KEY_IDS:values[f"local_read/{arm}/{target}/{key}"]="local_identity_observation"
    for role in method.plan():
        if role["layer"]=="mp4" and role["history"]=="local":values['reuse/'+role['row']]="saved_observation_read"
    for pair in ("P0_to_S_base","P0_to_S_ID","S_base_to_S_ID"):
        for layer in method.LAYERS:values[f"reference/quality/{pair}/{layer}"]="saved_quality_read"
    return values


class Journal(c73.Journal):
    def __init__(self,output,backend):
        self.output,self.backend=output,backend
        self.data={k:dict(kind=v,status="NOT_ENTERED") for k,v in units().items()}
        self.save()


class WanBackend(c73.WanBackend):
    def load_pixels(self,path):
        value=torch.load(path,map_location="cpu",weights_only=True)
        return c73.shared.validate_rgb(value,pixels=True)

    def encode_full(self,pixels):
        from runtime.wan.vae import reencode_rgb24_readback
        c73.shared.validate_rgb(pixels,pixels=True)
        z=reencode_rgb24_readback(self.vae,pixels.float()/255).detach().cpu().float()
        method.identity.legacy.validate_latent(z)
        return z


def execution_artifacts(output):
    output=Path(output)
    names=("attempt.json","result.json","checkpoint.json","ledger.json","protocol.json","numeric_completion.json","evaluation.json")
    found=[n+s for n in names for s in ("", ".tmp") if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ("blind","features","references","reused"):
        p=output/folder
        if p.is_dir():found.extend(str(f.relative_to(output)) for f in p.rglob("*") if f.is_file() or f.is_symlink())
    return sorted(found)


def run(c73_root,output,legacy_summary,model_id=DEFAULT_MODEL_ID,*,backend=None):
    output=Path(output);source=Path(c73_root)
    output.mkdir(parents=True,exist_ok=True)
    artifacts=execution_artifacts(output)
    if artifacts:
        value=c73.existing_attempt(output,artifacts);value['fixed_denominator']=fixed_denominator();return value
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(started_unix=time.time(),source=str(source),scope="Card77 E12 only; failure/interruption consumes attempt"),stream)
            c73.flush_file(stream)
    except FileExistsError:
        value=c73.existing_attempt(output,['attempt.json']);value['fixed_denominator']=fixed_denominator();return value
    backend=backend or WanBackend(model_id)
    journal=Journal(output,backend)
    roles=method.plan()
    observations={r['row']:method.unavailable('NOT_ENTERED',2 if r['history']=='local' else method.TARGETS[r['target']][1]) for r in roles}
    result=dict(status='RUNNING',protocol=protocol(),fixed_denominator=fixed_denominator(),scientific_pass=None)
    books={};pixels={};row_operations={};started=time.perf_counter()
    def checkpoint():
        result['counts']=journal.counts();dump(output/'checkpoint.json',result)
    def tensor(path,value):
        p=output/path;p.parent.mkdir(parents=True,exist_ok=True);torch.save(value,p);return value
    def save_observation(role,value,operation):
        observations[role['row']]=value;row_operations[role['row']]=operation
        dump(output/'blind'/(role['row']+'.json'),value)
    def matching(**fields):
        return [r for r in roles if all(r[k]==v for k,v in fields.items())]
    def record_call(slot,operation,path):
        def call():
            value=operation();dump(output/path,value);return value
        return journal.call(slot,call)
    dump(output/'protocol.json',protocol())
    for name,value in observations.items():dump(output/'blind'/(name+'.json'),value)
    checkpoint()
    try:
        for key,secret in zip(method.KEY_IDS,method.identity.KEYS):
            b=journal.call('book/'+key,lambda secret=secret:method.identity.codebook(secret))
            if b is not None:books[key]=b
        for arm in method.ARMS:
            for layer in method.LAYERS:
                filename='rgb8.pt' if layer=='rgb8' else 'readback.pt'
                value=journal.call(f'load/{arm}/{layer}',lambda arm=arm,filename=filename:backend.load_pixels(source/'media'/arm/filename))
                if value is not None:pixels[arm,layer]=value
        # Reuse saved blind observations verbatim, no new projection/classify/E.
        for role in matching(layer='mp4',history='local'):
            slot='reuse/'+role['row'];row_operations[role['row']]=slot
            index=method.ARMS.index(role['arm'])*6+(0 if role['target']==0 else 3)
            origin=source/'blind/local'/f'{index:02d}'/(role['key']+'.json')
            def reuse(origin=origin):
                raw=json.loads(origin.read_text());vectors=raw.get('vectors',[])
                if len(vectors)!=1 or [x.get('identity') for x in vectors[0].get('identities',[])]!=list(range(46)):
                    raise ValueError('saved C73 local record requires one vector and all46 identities')
                return dict(origin=str(origin),new_physical_observation=False,original=raw)
            value=record_call(slot,reuse,Path('reused')/(role['row']+'.json'))
            if value is not None:save_observation(role,value['original']['vectors'][0],slot)
        loaded=journal.call('model',backend.load) if pixels else None
        if loaded is not None:
            result['model']=loaded
            for arm in method.ARMS:
                # Each arm/layer failure remains independent of other arms.
                for layer in method.LAYERS:
                    if (arm,layer) not in pixels:continue
                    slot=f'full_E/{arm}/{layer}'
                    z=journal.call(slot,lambda arm=arm,layer=layer:tensor(Path('features/full')/arm/(layer+'.pt'),backend.encode_full(pixels[arm,layer])))
                    if z is None:continue
                    for key in method.KEY_IDS:
                        read=f'full_read/{arm}/{layer}/{key}'
                        selected=matching(arm=arm,layer=layer,history='full',key=key)
                        for role in selected:row_operations[role['row']]=read
                        if key not in books:continue
                        value=record_call(read,lambda z=z,key=key:method.observe_full(z,books[key]),Path('blind/raw_full')/arm/layer/(key+'.json'))
                        if value is not None:
                            for role,v in zip(selected,value):save_observation(role,v,read)
                if (arm,'rgb8') not in pixels:continue
                for target,(start,time_index) in enumerate(method.TARGETS):
                    def prepare(arm=arm,start=start,target=target):
                        clip,receipt=method.identity.prepare_local(pixels[arm,'rgb8'],start)
                        dump(output/'blind/preparation'/arm/(str(target)+'.json'),receipt)
                        return clip
                    clip=journal.call(f'prepare/{arm}/{target}',prepare)
                    if clip is None:continue
                    z=journal.call(f'local_E/{arm}/{target}',lambda clip=clip,arm=arm,target=target:tensor(Path('features/local')/arm/(str(target)+'.pt'),backend.encode_local(clip)))
                    if z is None:continue
                    for key in method.KEY_IDS:
                        slot=f'local_read/{arm}/{target}/{key}'
                        role=matching(arm=arm,layer='rgb8',history='local',target=target,key=key)[0]
                        row_operations[role['row']]=slot
                        if key not in books:continue
                        value=record_call(slot,lambda z=z,key=key:method.observe_local(z,books[key]),Path('blind/raw_local')/arm/str(target)/(key+'.json'))
                        if value is not None:save_observation(role,value,slot)
                    checkpoint()
    except BaseException as exc:
        result['fatal_error']=f'{type(exc).__name__}: {exc}'
    for role in roles:
        op=row_operations.get(role['row'])
        dependencies=[]
        if role['layer']=='mp4' and role['history']=='local':
            dependencies=['reuse/'+role['row']]
        elif role['history']=='full':
            dependencies=[f"full_read/{role['arm']}/{role['layer']}/{role['key']}",
                          f"full_E/{role['arm']}/{role['layer']}",f"load/{role['arm']}/{role['layer']}",'model','book/'+role['key']]
        else:
            dependencies=[f"local_read/{role['arm']}/{role['target']}/{role['key']}",
                          f"local_E/{role['arm']}/{role['target']}",f"prepare/{role['arm']}/{role['target']}",
                          f"load/{role['arm']}/rgb8",'model','book/'+role['key']]
        failed=next((d for d in dependencies if journal.data[d]['status'] in ('FAILED','INTERRUPTED','STARTED')),None)
        if failed is not None:
            status=journal.data[failed]['status']
            observations[role['row']]=method.unavailable(status if failed==op else 'UPSTREAM_'+status,2 if role['history']=='local' else method.TARGETS[role['target']][1])
            observations[role['row']]['unavailable_from']=dict(operation=failed,status=status)
        elif observations[role['row']]['status']=='NOT_ENTERED' and result.get('fatal_error'):
            observations[role['row']]['attempt_interruption']=result['fatal_error']
        dump(output/'blind'/(role['row']+'.json'),observations[role['row']])
    dump(output/'numeric_completion.json',dict(completed_unix=time.time(),vectors=48,new_vectors=36,reused_vectors=12,
        identity_slots=2208,all_fixed_rows_persisted=True,semantic_roles_used=False))
    def post():
        saved={r['row']:json.loads((output/'blind'/(r['row']+'.json')).read_text()) for r in roles}
        value=evaluate(saved);dump(output/'evaluation.json',value);return value
    evaluated=journal.call('evaluation',post)
    result['evaluation_summary']=None if evaluated is None else {k:v for k,v in evaluated.items() if k not in ['rows','vectors']}
    # Historical writer traces, prior labels and P/S comparisons stay post-only.
    old=record_call('reference/prior_result',lambda:json.loads((source/'result.json').read_text()),Path('references/prior_C73_result.json'))
    if old is not None:
        result['readout_context']=dict(prior_model=old.get('model'),current_model=result.get('model'),
            comparison_is_cross_run=True,metadata_difference_is_not_admission_gate=True)
    history=record_call('reference/legacy',lambda:json.loads(Path(legacy_summary).read_text()),Path('references/legacy_archived_summary.json'))
    if history is not None:result['legacy_reference_summary']=dict(saved_events=history.get('records'),physical_calls=0,scope='archived framewise actual score replay, not new physical reproduction')
    for pair in ('P0_to_S_base','P0_to_S_ID','S_base_to_S_ID'):
        for layer in method.LAYERS:
            slot=f'reference/quality/{pair}/{layer}'
            record_call(slot,lambda pair=pair,layer=layer:json.loads((source/'quality'/pair/(layer+'.json')).read_text()),Path('references/quality')/pair/(layer+'.json'))
    result.update(calls=journal.data,counts=journal.counts(),seconds=time.perf_counter()-started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optional_file_fsync_unsupported_errnos=sorted(c73.SYNC_UNAVAILABLE_ERRNOS))
    if backend.device=='cuda':result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    result['status']='EXECUTION_COMPLETE' if all(r['status']=='RETURNED' for r in journal.data.values()) and not result.get('fatal_error') else 'INCOMPLETE'
    dump(output/'result.json',result);checkpoint();return result
