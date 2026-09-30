"""Matched five-layer saved-terminal replay; user-run VAE/codec only, no writer."""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from main.tube_state import video_overlap_zero_mean_state as state
from main.tube_state import video_overlap_zero_mean_control as payload
from runtime.wan import zero_mean_c1_layered_transport as adapter
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_c1_layered_transport_v1.json'
MODULE='experiments.wan_state_clock.zero_mean_c1_layered_transport_run'
ARMS=('OFF','PAYLOAD_MULTI','OVERLAP_MULTI')
LAYERS=('TERMINAL_PREFIX44','FLOAT_CLAMPED','RGB8_NO_CODEC','MATCHED_SOURCE_MP4','MATCHED_FULL_RESAVED')
KEYS=('CORRECT','WRONG')
GOOD=('COMPLETE','NO_ENERGY')

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def dump_gzip(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:
        f.write(gzip.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode(),mtime=0));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def load_gzip(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))

def environment():
    versions={}
    for name in ('torch','torchvision','diffusers','transformers','accelerate','numpy','imageio','imageio-ffmpeg'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return dict(python=sys.version,executable=sys.executable,packages=versions)

def load_config():
    cfg=json.loads(CONFIG.read_text())
    if cfg['arms']!=list(ARMS) or cfg['layers']!=list(LAYERS) or cfg['geometry']['R']!=44:raise ValueError('fixed roster mismatch')
    for path,sha in cfg['fixed_input']['inherited_method_files'].items():
        if adapter.file_sha256(ROOT/path)!=sha:raise ValueError('inherited source identity mismatch: '+path)
    return cfg

def initial_result(output,cfg,input_root):
    normalized={};reads={};paths={};messages={};rasters={};media={};clamp={}
    shared={};catalog=state.catalog(44);cp=output/'catalog_R44.json.gz';dump_gzip(cp,dict(rows=catalog))
    for kid,key in cfg['keys'].items():
        f=state.family_receipt(key,44,np.ones((44,4),bool));fp=output/f'family_R44_{kid}.json.gz';dump_gzip(fp,f)
        if len(catalog)!=3915 or len(f['valid_catalog_indices'])!=174:raise ValueError('original family mismatch')
        shared[kid]=dict(catalog_path=str(cp),catalog_sha256=adapter.file_sha256(cp),family_path=str(fp),
            family_sha256=adapter.file_sha256(fp),catalog=3915,valid=174,excluded=3741,class_count=len(f['classes']))
    for arm in ARMS:
        clamp[arm]=dict(status='PENDING')
        for kind in ('FLOAT_CLAMPED','RGB8_NO_CODEC'):
            rasters[f'{arm}/{kind}']=dict(status='PENDING',path=str(output/arm/(kind+'.pt')))
        for kind in LAYERS[3:]:
            media[f'{arm}/{kind}']=dict(status='PENDING',read_status='PENDING',path=str(output/arm/(kind+'.mp4')))
        for layer in LAYERS:
            sid=f'{arm}/{layer}';normalized[sid]=dict(status='PENDING',path=str(output/arm/(layer+'.normalized.pt')),
                R=44,full_latent_steps=46,extra_regular_index=45,fixed_phase=0)
            for kid in KEYS:
                rid=sid+'/'+kid
                reads[rid]=dict(status='PENDING',payload_status='PENDING',path=str(output/arm/(layer+'.'+kid+'.raw.json.gz')),
                    shared=shared[kid],scored=0,accepted_payload=False,state_path_accepted=False)
                paths[rid]=dict(status='PENDING',accepted_payload=False,state_path_accepted=False)
                for target in cfg['messages']:messages[rid+'/'+target]=dict(status='PENDING',bit_errors=None,accepted_payload=False)
    source_paths=list(cfg['fixed_input']['inherited_method_files'])+[
        'runtime/wan/zero_mean_c1_layered_transport.py',str(CONFIG.relative_to(ROOT)),
        str(Path(__file__).relative_to(ROOT)),'experiments/wan_state_clock/requirements-grow-video-reference.txt']
    return dict(status='RUNNING',stage='INITIALIZE',normalized=normalized,reads=reads,path_posthoc=paths,message_evaluations=messages,
        rasters=rasters,media=media,decode_diagnostics=clamp,shared=shared,input_root=str(input_root),input_identity=dict(status='PENDING'),
        inputs={r['arm']:dict(r,status='PENDING') for r in cfg['fixed_input']['required_terminal']},
        historical={str(i):dict(r,status='PENDING') for i,r in enumerate(cfg['fixed_input']['optional_historical'])},
        historical_comparisons={},calls={k:dict(attempted=0,completed=0) for k in cfg['planned_calls']},
        model_loaded=False,actual_model_calls=False,workers={},failures=[],source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files={p:adapter.file_sha256(ROOT/p) for p in source_paths},config_sha256=adapter.file_sha256(CONFIG),
        environment=environment(),model=cfg['model'],method_version=state.PUBLIC.method_version,
        fixed_denominator=cfg['fixed_denominator'],planned_calls=cfg['planned_calls'],basis_layout={k:state.basis_layout(v) for k,v in cfg['keys'].items()},
        evidence_ceiling=cfg['evidence_ceiling'],score_status='UNCALIBRATED_DIAGNOSTIC',science_status='NO_AUTOMATIC_SCIENTIFIC_PASS')

class Store:
    def __init__(self,output,*,create=False,cfg=None,input_root=None):
        self.output=Path(output);self.path=self.output/'result.json'
        if create:
            if self.output.resolve().is_relative_to(Path(input_root).resolve()):raise ValueError('output must be outside the read-only input run')
            self.output.mkdir(parents=True,exist_ok=False);self.data=initial_result(self.output,cfg,input_root)
        else:self.data=json.loads(self.path.read_text())
        self.save()
    def save(self):
        d=self.data
        sizes=dict(normalized=15,reads=30,path_posthoc=30,message_evaluations=60,rasters=6,media=6,decode_diagnostics=3,inputs=3,historical=15)
        if any(len(d[k])!=v for k,v in sizes.items()):raise ValueError('fixed table size changed')
        d['counts']=dict(normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),
            path_reads=sum(r['status'] in GOOD for r in d['reads'].values()),
            payload_reads=sum(r['payload_status']=='READ' for r in d['reads'].values()),
            valid_costs=sum(r['scored'] for r in d['reads'].values()),
            catalog_slots=sum(r['shared']['catalog'] for r in d['reads'].values()),
            structural_exclusions=sum(r['shared']['excluded'] for r in d['reads'].values()),
            path_posthoc=sum(r['status']=='EVALUATED' for r in d['path_posthoc'].values()),
            message_evaluations=sum(r['status']=='EVALUATED' for r in d['message_evaluations'].values()),
            rasters=sum(r['status']=='SAVED' for r in d['rasters'].values()),
            mp4_saved=sum(r['status']=='SAVED' for r in d['media'].values()),
            mp4_read=sum(r['read_status']=='READ' for r in d['media'].values()),
            decode_diagnostics=sum(r['status']=='MEASURED' for r in d['decode_diagnostics'].values()))
        dump(self.path,d)
    def blind(self):
        dump(self.output/'blind_readouts.json',{k:self.data[k] for k in ('normalized','reads','shared','fixed_denominator','method_version','basis_layout')})
    def checkpoint(self):
        self.save();self.blind() # result is canonical; blind can always be rebuilt.
    def failure(self,stage,exc):
        self.data['failures'].append(dict(stage=stage,error=f'{type(exc).__name__}: {exc}'));self.save()
    def call(self,kind,fn):
        row=self.data['calls'][kind];row['attempted']+=1
        if kind.startswith('vae_'):self.data['actual_model_calls']=True
        self.save();value=fn();row['completed']+=1;self.save();return value

def validate_latent(z):
    import torch
    if tuple(z.shape)!=(1,16,46,40,64) or z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):raise ValueError('full normalized tensor geometry/dtype/finite mismatch')
    return z

def save_tensor(path,value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    return dict(path=str(path),sha256=adapter.file_sha256(path),shape=list(value.shape),dtype=str(value.dtype),status='SAVED')

def reopen_tensor(row):
    import torch
    if adapter.file_sha256(row['path'])!=row['sha256']:raise ValueError('saved tensor identity mismatch')
    return torch.load(row['path'],map_location='cpu',weights_only=True)

def preflight(store,cfg):
    root=Path(store.data['input_root']);identity=store.data['input_identity']
    try:
        rp=root/'result.json'
        if adapter.file_sha256(rp)!=cfg['fixed_input']['result_sha256']:raise ValueError('fixed source result hash mismatch')
        previous=json.loads(rp.read_text())
        if previous['source_sha']!=cfg['fixed_input']['source_sha']:raise ValueError('fixed source SHA mismatch')
        identity.update(status='VERIFIED',path=str(rp),sha256=adapter.file_sha256(rp),source_sha=previous['source_sha'],
            historical_environment=previous.get('environment'),historical_model=previous.get('generation_setup',{}).get('model'))
    except Exception as exc:
        identity.update(status='FAILED',error=str(exc));store.failure('INPUT_IDENTITY',exc)
    for row in store.data['inputs'].values():
        try:
            if identity['status']!='VERIFIED':raise ValueError('fixed source result not verified')
            p=root/row['relative_path']
            if adapter.file_sha256(p)!=row['sha256']:raise ValueError('terminal hash mismatch')
            row.update(status='VERIFIED',path=str(p))
        except Exception as exc:row.update(status='FAILED',error=str(exc))
    # Optional evidence is checked independently and never changes primary availability.
    for row in store.data['historical'].values():
        p=root/row['relative_path']
        try:
            if adapter.file_sha256(p)!=row['sha256']:raise ValueError('historical identity mismatch')
            row.update(status='VERIFIED',path=str(p))
        except Exception as exc:row.update(status='MISSING_OR_INVALID',path=str(p),error=str(exc))
    store.checkpoint()

def read_normalized(z,key,call):
    """Truth-free adapter; no layer, arm, filename, writer evidence or target argument."""
    received=z[0,:,1:45].numpy().transpose(1,0,2,3)
    inf=call('new_path_infer',lambda:state.infer(received,key,np.ones((44,4),bool)))
    try:bits=call('new_payload_read',lambda:payload.payload_read(z,key,44))
    except Exception as exc:bits=dict(status='FAILED',error=str(exc),decoded_bits=None)
    # Complete saved tensor contains row45; an extra projection is diagnostic, not scored.
    try:extra=dict(status='PROJECTED',values=state.extract(z[0,:,1:].numpy().transpose(1,0,2,3),key,np.ones((45,4),bool))[-1].tolist())
    except Exception as exc:extra=dict(status='FAILED',error=str(exc))
    return dict(inference=inf,payload=bits,extra_regular45_projection=extra,fixed_phase=0,truth_used=False,writer_inputs=False)

def score_layer(store,arm,layer,cfg):
    sid=f'{arm}/{layer}';normal=store.data['normalized'][sid]
    try:z=validate_latent(reopen_tensor(normal))
    except Exception as exc:
        for kid in KEYS:store.data['reads'][sid+'/'+kid].update(status='NOT_COMPLETED',payload_status='NOT_COMPLETED',error=str(exc))
        store.checkpoint();return
    for kid,key in cfg['keys'].items():
        row=store.data['reads'][sid+'/'+kid]
        try:
            raw=read_normalized(z,key,store.call);inf=raw['inference']
            if inf['summary']['status'] in GOOD:
                if inf['counts']['scored']!=174 or len(inf['path_costs'])!=174 or not np.isfinite(inf['path_costs']).all():raise ValueError('truncated/nonfinite path costs')
                if len(inf['class_costs'])!=row['shared']['class_count'] or not np.isfinite(inf['class_costs']).all():raise ValueError('truncated/nonfinite class costs')
            raw['normalized_sha256']=normal['sha256'];dump_gzip(Path(row['path']),raw)
            row.update(status=inf['summary']['status'],payload_status=raw['payload']['status'],sha256=adapter.file_sha256(row['path']),
                scored=inf['counts']['scored'],summary=inf['summary'],payload=raw['payload'])
        except Exception as exc:row.update(status='FAILED',payload_status='FAILED',error=f'{type(exc).__name__}: {exc}')
        store.checkpoint()

def terminal_layers(store,cfg):
    import torch
    for arm in ARMS:
        source=store.data['inputs'][arm];row=store.data['normalized'][arm+'/'+LAYERS[0]]
        try:
            if source['status']!='VERIFIED':raise ValueError('verified terminal unavailable')
            validate_latent(reopen_tensor(source))
            row.update(status='SAVED',path=source['path'],sha256=source['sha256'],reused=True,shape=[1,16,46,40,64],dtype='torch.float32')
            store.checkpoint();score_layer(store,arm,LAYERS[0],cfg)
        except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure('TERMINAL_'+arm,exc)
        store.checkpoint()

def encoded_layer(store,arm,layer,rgb,frozen,cfg):
    row=store.data['normalized'][arm+'/'+layer]
    try:
        z=store.call('vae_encode',lambda:adapter.encode_normalized(frozen,rgb))
        row.update(save_tensor(row['path'],z));del z
        store.checkpoint();score_layer(store,arm,layer,cfg)
    except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure('ENCODE_'+arm+'/'+layer,exc)
    store.checkpoint()

def media_edge(store,arm,layer,rgb_or_q8,*,q8=False):
    row=store.data['media'][arm+'/'+layer]
    try:
        row['status']='SAVING';store.save()
        if q8:store.call('new_mp4_save',lambda:adapter.encode_raster(rgb_or_q8,row['path']))
        else:store.call('new_mp4_save',lambda:adapter.io.encode_rgb(rgb_or_q8,Path(row['path']),8,18))
        row.update(status='SAVED',sha256=adapter.file_sha256(row['path']));store.save()
        observed=store.call('new_mp4_read',lambda:adapter.read_full_mp4(row['path']))
        row.update(read_status='READ',readback_shape=list(observed.shape));store.save();return observed
    except Exception as exc:
        if row['status']=='SAVED':row['read_status']='FAILED'
        else:row.update(status='FAILED',read_status='NOT_COMPLETED')
        row['error']=str(exc);store.failure('MEDIA_'+arm+'/'+layer,exc);return None

def replay_worker(store,cfg):
    import torch
    from runtime.wan.generation import load_frozen_vae
    store.data['stage']='VAE_LOAD';store.save()
    frozen=load_frozen_vae(cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    store.data.update(model_loaded=True,vae_runtime=dict(device=str(next(frozen.parameters()).device),dtype=str(next(frozen.parameters()).dtype),
        latents_mean=list(frozen.config.latents_mean),latents_std=list(frozen.config.latents_std),posterior='mode',
        full_frame_encode=True,frames=181,normalization='(raw-mean)/std'))
    store.save()
    for arm in ARMS:
        store.data['stage']='REPLAY_'+arm;store.save();src=store.data['inputs'][arm]
        try:
            if src['status']!='VERIFIED':raise ValueError('verified terminal unavailable')
            z=validate_latent(reopen_tensor(src))
            rgb,stats=store.call('vae_decode',lambda:adapter.decode_with_clamp_receipt(frozen,z));del z
            store.data['decode_diagnostics'][arm]=stats
            rr=store.data['rasters'][arm+'/FLOAT_CLAMPED'];rr.update(save_tensor(rr['path'],rgb));del rgb;store.save()
            rgb=reopen_tensor(rr);encoded_layer(store,arm,'FLOAT_CLAMPED',rgb,frozen,cfg)
            q8=adapter.shared_vae.quantize_rgb8_no_codec(rgb);del rgb
            qr=store.data['rasters'][arm+'/RGB8_NO_CODEC'];qr.update(save_tensor(qr['path'],q8));del q8;store.save()
            q8=reopen_tensor(qr);encoded_layer(store,arm,'RGB8_NO_CODEC',q8.float()/255.,frozen,cfg)
            received=media_edge(store,arm,'MATCHED_SOURCE_MP4',q8,q8=True);del q8
            if received is not None:
                encoded_layer(store,arm,'MATCHED_SOURCE_MP4',received,frozen,cfg)
                full=media_edge(store,arm,'MATCHED_FULL_RESAVED',received);del received
                if full is not None:encoded_layer(store,arm,'MATCHED_FULL_RESAVED',full,frozen,cfg);del full
        except Exception as exc:
            if store.data['decode_diagnostics'][arm]['status']=='PENDING':store.data['decode_diagnostics'][arm]=dict(status='FAILED',error=str(exc))
            store.failure('REPLAY_'+arm,exc)
        store.checkpoint()

def recover_unfinished(store,reason):
    for group in ('normalized','reads','rasters','media','decode_diagnostics'):
        for row in store.data[group].values():
            if row['status'] in ('PENDING','SAVING','RUNNING'):row.update(status='NOT_COMPLETED',error=reason)
            if group=='reads' and row['payload_status']=='PENDING':row['payload_status']='NOT_COMPLETED'
            if group=='media' and row['read_status']=='PENDING':row['read_status']='NOT_COMPLETED'
    store.checkpoint()

def candidate_terms(obs,template):
    return dict(cost=float(np.mean((obs-template)**2)),observed_energy=float(np.sum(obs*obs)),
        template_energy=float(np.sum(template*template)),dot=float(np.sum(obs*template)),
        squared_error_by_time_block_age=np.sum((obs-template)**2,axis=-1).tolist())

def historical_opponents(store):
    """Hash-bound old FULL phase0 canonical; never a global phase winner."""
    result={}
    for row in store.data['historical'].values():
        if row['kind']!='FULL_phase0_raw':continue
        value=dict(status='MISSING_REFERENCE',role='original FULL phase0 fixed opponent only')
        try:
            if row['status']!='VERIFIED' or adapter.file_sha256(row['path'])!=row['sha256']:raise ValueError('historical raw unavailable or changed')
            raw=load_gzip(row['path'])['inference']
            if raw['received_length']!=44:raise ValueError('historical support mismatch')
            i=raw['summary']['canonical_catalog_index']
            if i is None or i not in raw['valid_catalog_indices']:raise ValueError('historical canonical unavailable')
            value.update(status='AVAILABLE',raw_sha256=row['sha256'],catalog_index=i,path=state.catalog(44)[i],fixed_phase=0)
        except Exception as exc:value['error']=str(exc)
        result[row['arm']+'/'+row['key_id']]=value
    return result

def evaluate_saved(store,cfg):
    """Only this stage receives registered path/message; reads only already committed raw."""
    store.data['stage']='POSTHOC';store.checkpoint();blind=store.output/'blind_readouts.json';before=adapter.file_sha256(blind)
    opponents=historical_opponents(store)
    catalog=state.catalog(44);true_index=next(i for i,r in enumerate(catalog) if r['taus']==list(range(1,45)) and r['event_type']=='ZERO_EDIT')
    for sid,row in store.data['reads'].items():
        arm,layer,kid=sid.split('/');p=store.data['path_posthoc'][sid]
        raw=None
        try:
            if row['status'] not in GOOD:raise ValueError('primary path read unavailable')
            if adapter.file_sha256(row['path'])!=row['sha256']:raise ValueError('raw identity mismatch')
            raw=load_gzip(row['path']);inf=raw['inference'];obs=np.asarray(inf['projection'],dtype=np.float64)
            mu=state.composite_signs(cfg['keys'][kid]).astype(np.float64)*state.PUBLIC.alpha
            canonical=inf['summary']['canonical_catalog_index'];true_terms=candidate_terms(obs,mu[:44]);winner=None
            if canonical is not None:winner=candidate_terms(obs,mu[np.asarray(catalog[canonical]['taus'])-1])
            costs=np.asarray(inf['path_costs'],dtype=np.float64)
            fixed=dict(opponents[arm+'/'+kid])
            if fixed['status']=='AVAILABLE':
                terms=candidate_terms(obs,mu[np.asarray(fixed['path']['taus'])-1])
                fixed.update(status='COMPARED',terms=terms,true_minus_fixed=true_terms['cost']-terms['cost'])
            p.update(status='EVALUATED',fixed_phase=0,phase_is_not_recovered=True,registered_tau=list(range(1,45)),
                truth_in_top=true_index in inf['summary']['top_catalog_indices'],
                true_rank=1+int(np.sum(costs<true_terms['cost']-state.PUBLIC.tie_atol)),
                true_cost_ties=[i for i,c in zip(inf['valid_catalog_indices'],costs) if abs(float(c)-true_terms['cost'])<=state.PUBLIC.tie_atol],
                truth_unique_top=inf['summary']['top_catalog_indices']==[true_index],
                unique_model_hypothesis=inf['summary']['unique_model_hypothesis'],
                top_class_indices=inf['summary']['top_class_indices'],top_catalog_indices=inf['summary']['top_catalog_indices'],
                canonical_catalog_index=canonical,
                canonical_start_correct=canonical is not None and catalog[canonical]['tau1']==1,
                invented_edit=None if canonical is None else catalog[canonical]['event_type']!='ZERO_EDIT',
                fixed_historical_phase0_opponent=fixed,
                canonical_path=catalog[canonical] if canonical is not None else None,
                canonical_matches=canonical is not None and catalog[canonical]['taus']==list(range(1,45)),
                true_terms=true_terms,winner_terms=winner,
                true_minus_winner=None if winner is None else true_terms['cost']-winner['cost'],
                expected_pilot_present=arm=='OVERLAP_MULTI' and kid=='CORRECT',accepted_payload=False,state_path_accepted=False)
        except Exception as exc:p.update(status='MISSING_READ',error=str(exc))
        for target,text in cfg['messages'].items():
            m=store.data['message_evaluations'][sid+'/'+target]
            bits=(row.get('payload') or {}).get('decoded_bits')
            if bits is not None and row['payload_status']=='READ':
                truth=payload.message_bits(text)
                m.update(status='EVALUATED',diagnostic_bit_errors=sum(a!=b for a,b in zip(bits,truth)),diagnostic_exact=bits==truth,
                    bit_errors=None,accepted_payload=False,note='repeated payload diagnostic; never path selection or accepted recovery')
            else:m.update(status='MISSING_READ')
        store.save()
    # Historical references do not select new paths or replace any failed new observation.
    for arm in ARMS:
        for layer,kind in [('MATCHED_SOURCE_MP4','historical_source_mp4'),('MATCHED_FULL_RESAVED','historical_FULL_mp4')]:
            old=next(r for r in store.data['historical'].values() if r['arm']==arm and r['kind']==kind)
            new=store.data['media'][arm+'/'+layer]
            store.data['historical_comparisons'][arm+'/'+kind]=dict(status='COMPARED' if old['status']=='VERIFIED' and new['status']=='SAVED' else 'MISSING_REFERENCE_OR_NEW',
                old_sha256=old['sha256'],new_sha256=new.get('sha256'),bytes_equal=new.get('sha256')==old['sha256'] if new['status']=='SAVED' and old['status']=='VERIFIED' else None,
                equality_required=False,not_pure_codec_causality=True)
        for kid in KEYS:
            old=next(r for r in store.data['historical'].values() if r['arm']==arm and r['kind']=='FULL_phase0_raw' and r['key_id']==kid)
            now=store.data['reads'][arm+'/MATCHED_FULL_RESAVED/'+kid];report=dict(status='MISSING_REFERENCE_OR_NEW')
            try:
                if old['status']!='VERIFIED' or now['status'] not in GOOD:raise ValueError('historical or new raw unavailable')
                if adapter.file_sha256(old['path'])!=old['sha256']:raise ValueError('historical raw changed')
                historical=load_gzip(old['path']);previous=historical['inference']
                fresh=load_gzip(now['path'])['inference']
                report.update(status='COMPARED',historical_raw_sha256=old['sha256'],new_raw_sha256=now['sha256'],
                    maximum_same_path_cost_difference=float(np.max(np.abs(np.asarray(previous['path_costs'])-np.asarray(fresh['path_costs'])))),
                    previous_summary=previous['summary'],new_summary=fresh['summary'],new_searches=0,not_pure_codec_causality=True)
            except Exception as exc:report['error']=str(exc)
            store.data['historical_comparisons'][arm+'/FULL_raw/'+kid]=report
    if adapter.file_sha256(blind)!=before:raise RuntimeError('posthoc modified blind evidence')
    store.data['blind_sha256']=before;store.save()

def finalize(store,cfg):
    recover_unfinished(store,'upstream or worker incomplete');evaluate_saved(store,cfg)
    c=store.data['counts']
    required=dict(normalized=15,path_reads=30,payload_reads=30,valid_costs=5220,catalog_slots=117450,structural_exclusions=112230,
        path_posthoc=30,message_evaluations=60,rasters=6,mp4_saved=6,mp4_read=6,decode_diagnostics=3)
    calls=all(store.data['calls'][k]['completed']==n for k,n in cfg['planned_calls'].items())
    done=c==required and calls and not store.data.get('interrupted',False)
    store.data['primary_status']='EXECUTION_COMPLETE' if done else 'INCOMPLETE'
    history=all(r['status']=='VERIFIED' for r in store.data['historical'].values())
    store.data['historical_counts']={kind:dict(expected=sum(r['kind']==kind for r in store.data['historical'].values()),
        verified=sum(r['kind']==kind and r['status']=='VERIFIED' for r in store.data['historical'].values()))
        for kind in ('FULL_phase0_tensor','FULL_phase0_raw','historical_source_mp4','historical_FULL_mp4')}
    compared=all(r['status']=='COMPARED' for r in store.data['historical_comparisons'].values()) and len(store.data['historical_comparisons'])==12
    store.data['historical_comparison_status']='COMPLETE' if compared else 'INCOMPLETE'
    store.data['historical_reference_status']='COMPLETE' if history else 'INCOMPLETE'
    store.data['package_status']='COMPLETE' if done and history and compared else 'INCOMPLETE'
    store.data['status']=store.data['primary_status'];store.data['stage']='FINISHED';store.save()
    return done

def stop_worker(child):
    if child.poll() is None:
        try:child.terminate()
        except ProcessLookupError:pass
    try:return child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();return child.wait(timeout=10)

def worker_phase(output):
    command=[sys.executable,'-u','-m',MODULE,'--output',str(output),'--worker']
    child=None;error=None;code=None;halt=False;cleanup=None;start=time.perf_counter()
    try:
        with (Path(output)/'replay.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait()
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';halt=not isinstance(exc,Exception)
        if child is not None:
            try:code=stop_worker(child)
            except BaseException as err:cleanup=str(err);halt=True
    finally:
        if child is not None and child.stdout is not None:
            try:child.stdout.close()
            except Exception:pass
    store=Store(output) # Reload only after child termination; preserve its latest canonical commits.
    store.data['workers']['replay']=dict(command=command,returncode=code,error=error,cleanup_error=cleanup,
        status='COMPLETE' if error is None and code==0 else 'FAILED',elapsed_seconds=time.perf_counter()-start)
    if error or code!=0:store.failure('REPLAY_WORKER',RuntimeError(error or str(code)))
    if halt:store.data['interrupted']=True
    store.checkpoint();return store,halt

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-root',type=Path);parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    cfg=load_config()
    if args.worker:
        store=Store(args.output)
        try:replay_worker(store,cfg)
        except Exception as exc:store.failure('VAE_REPLAY',exc);recover_unfinished(store,str(exc));raise
        return
    store=Store(args.output,create=True,cfg=cfg,input_root=args.input_root or cfg['fixed_input']['root'])
    preflight(store,cfg);terminal_layers(store,cfg)
    if any(r['status']=='VERIFIED' for r in store.data['inputs'].values()):store,halt=worker_phase(args.output)
    else:halt=False;store.data['workers']['replay']=dict(status='NOT_STARTED_NO_VERIFIED_INPUT')
    done=finalize(store,cfg);print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])))
    if halt:raise SystemExit(130)
    if not done:raise SystemExit(1)

if __name__=='__main__':main()
