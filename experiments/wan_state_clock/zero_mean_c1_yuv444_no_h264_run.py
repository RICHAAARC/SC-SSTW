"""Saved Q8 -> materialized raw444 -> RGB24 user-run diagnostic, no new writer."""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.metadata,json,os,signal,subprocess,sys,time
from pathlib import Path
import numpy as np
from main.tube_state import video_overlap_zero_mean_state as state,video_overlap_zero_mean_control as payload
from runtime.wan import zero_mean_c1_yuv444_no_h264 as adapter
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/zero_mean_c1_yuv444_no_h264_v1.json'
MODULE='experiments.wan_state_clock.zero_mean_c1_yuv444_no_h264_run'
ARMS=('OFF','PAYLOAD_MULTI','OVERLAP_MULTI');KEYS=('CORRECT','WRONG');GOOD=('COMPLETE','NO_ENERGY')

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

def candidate_terms(obs,template):
    return dict(cost=float(np.mean((obs-template)**2)),observed_energy=float(np.sum(obs*obs)),
        template_energy=float(np.sum(template*template)),dot=float(np.sum(obs*template)),
        squared_error_by_time_block_age=np.sum((obs-template)**2,axis=-1).tolist())

def stop_worker(child):
    """The fresh child owns a new session; include any live FFmpeg descendants."""
    try:os.killpg(child.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:code=child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        code=child.wait(timeout=10)
    # Parent may have exited before a descendant. End the entire owned group
    # before reopening canonical records; no ffmpeg may keep writing raw files.
    try:os.killpg(child.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    return code


def load_config():
    cfg=json.loads(CONFIG.read_text())
    if cfg['arms']!=list(ARMS) or cfg['geometry']['R']!=44 or cfg['layer']!='YUV444_NO_H264':raise ValueError('frozen roster mismatch')
    for rel,sha in cfg['fixed_input']['inherited_source_files'].items():
        if adapter.file_sha256(ROOT/rel)!=sha:raise ValueError('inherited source identity mismatch: '+rel)
    refs=cfg['fixed_input']['reference_artifacts']
    if len(refs)!=27 or len({(r['root_id'],r['relative_path']) for r in refs})!=27:raise ValueError('reference roster mismatch')
    for layer,source_root in [('RGB8_NO_CODEC','layered'),('MATCHED_SOURCE_MP4','layered'),('YUV420_NO_H264','reference420')]:
        rows=[r for r in refs if r['layer']==layer]
        if (len(rows)!=9 or sum(r['kind']=='normalized' for r in rows)!=3 or sum(r['kind']=='raw' for r in rows)!=6
            or any(r['root_id']!=source_root for r in rows)):raise ValueError('reference layer/kind mismatch')
    expected=adapter.conversion_commands('{yuv444_output}','{rgb24_output}')
    if expected[0]!=cfg['conversion']['segment1_argv']:raise ValueError('first conversion command drift')
    if expected[1]!=[s.replace('{yuv444_input}','{yuv444_output}') for s in cfg['conversion']['segment2_argv']]:raise ValueError('second conversion command drift')
    return cfg

def initial_result(output,cfg,input_root,reference420_root):
    catalog=state.catalog(44);cp=output/'catalog_R44.json.gz';dump_gzip(cp,dict(rows=catalog));shared={}
    for kid,key in cfg['keys'].items():
        family=state.family_receipt(key,44,np.ones((44,4),bool));fp=output/f'family_R44_{kid}.json.gz';dump_gzip(fp,family)
        if len(catalog)!=3915 or len(family['valid_catalog_indices'])!=174:raise ValueError('original family changed')
        shared[kid]=dict(catalog_path=str(cp),catalog_sha256=adapter.file_sha256(cp),family_path=str(fp),family_sha256=adapter.file_sha256(fp),
                         catalog=3915,valid=174,excluded=3741,class_count=len(family['classes']))
    tables=dict(conversions={},normalized={},reads={},path_posthoc={},message_evaluations={})
    for arm in ARMS:
        for stage,ext in [('yuv444','yuv'),('rgb24','rgb')]:
            tables['conversions'][arm+'/'+stage]=dict(status='PENDING',path=str(output/arm/('roundtrip.'+ext)),stage=stage)
        tables['normalized'][arm]=dict(status='PENDING',path=str(output/arm/'normalized.pt'),R=44,full_latent_steps=46)
        for kid in KEYS:
            sid=arm+'/'+kid
            tables['reads'][sid]=dict(status='PENDING',payload_status='PENDING',scored=0,path=str(output/arm/(kid+'.raw.json.gz')),
                shared=shared[kid],accepted_payload=False,state_path_accepted=False)
            tables['path_posthoc'][sid]=dict(status='PENDING',accepted_payload=False,state_path_accepted=False)
            for t in cfg['messages']:tables['message_evaluations'][sid+'/'+t]=dict(status='PENDING',bit_errors=None,accepted_payload=False)
    paths=list(cfg['fixed_input']['inherited_source_files'])+[str(CONFIG.relative_to(ROOT)),str(Path(__file__).relative_to(ROOT)),
        'runtime/wan/zero_mean_c1_yuv444_no_h264.py']
    return dict(status='RUNNING',stage='INITIALIZE',**tables,shared=shared,
        input_identity=dict(status='PENDING'),input_root=str(input_root),
        reference420_identity=dict(status='PENDING'),reference420_root=str(reference420_root),
        inputs={r['arm']:dict(r,status='PENDING') for r in cfg['fixed_input']['required_q8']},
        references={str(i):dict(r,status='PENDING') for i,r in enumerate(cfg['fixed_input']['reference_artifacts'])},
        fixed_opponents={a+'/'+k:dict(status='PENDING') for a in ARMS for k in KEYS},
        reference_posthoc={},calls={k:dict(attempted=0,completed=0) for k in cfg['planned_calls']},
        model_loaded=False,actual_model_calls=False,workers={},failures=[],environment=environment(),model=cfg['model'],
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files={p:adapter.file_sha256(ROOT/p) for p in paths},config_sha256=adapter.file_sha256(CONFIG),
        method_version=state.PUBLIC.method_version,basis_layout={k:state.basis_layout(v) for k,v in cfg['keys'].items()},
        fixed_denominator=cfg['fixed_denominator'],planned_calls=cfg['planned_calls'],evidence_ceiling=cfg['evidence_ceiling'],
        score_status='UNCALIBRATED_DIAGNOSTIC',science_status='NO_AUTOMATIC_SCIENTIFIC_PASS')

class Store:
    def __init__(self,output,*,create=False,cfg=None,input_root=None,reference420_root=None):
        self.output=Path(output);self.path=self.output/'result.json'
        if create:
            reference420_root=reference420_root or cfg['reference420_input']['root']
            for source_root in (input_root,reference420_root):
                if self.output.resolve().is_relative_to(Path(source_root).resolve()):raise ValueError('output must be outside both read-only input roots')
            self.output.mkdir(parents=True,exist_ok=False);self.data=initial_result(self.output,cfg,input_root,reference420_root)
        else:self.data=json.loads(self.path.read_text())
        self.save()
    def save(self):
        d=self.data;sizes=dict(inputs=3,conversions=6,normalized=3,reads=6,path_posthoc=6,message_evaluations=12,references=27,fixed_opponents=6)
        if any(len(d[k])!=n for k,n in sizes.items()):raise ValueError('fixed roster changed')
        d['counts']=dict(yuv444_files=sum(r['status']=='SAVED' and r['stage']=='yuv444' for r in d['conversions'].values()),
            rgb24_files=sum(r['status']=='SAVED' and r['stage']=='rgb24' for r in d['conversions'].values()),
            normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),
            path_reads=sum(r['status'] in GOOD for r in d['reads'].values()),
            payload_reads=sum(r['payload_status']=='READ' for r in d['reads'].values()),
            valid_costs=sum(r['scored'] for r in d['reads'].values()),
            catalog_slots=sum(r['shared']['catalog'] for r in d['reads'].values()),
            structural_exclusions=sum(r['shared']['excluded'] for r in d['reads'].values()),
            path_posthoc=sum(r['status']=='EVALUATED' for r in d['path_posthoc'].values()),
            message_evaluations=sum(r['status']=='EVALUATED' for r in d['message_evaluations'].values()))
        d['reference_counts']={layer:{kind:dict(expected=n,verified=sum(r['status']=='VERIFIED' for r in d['references'].values() if r['layer']==layer and r['kind']==kind))
            for kind,n in [('normalized',3),('raw',6)]} for layer in ('RGB8_NO_CODEC','YUV420_NO_H264','MATCHED_SOURCE_MP4')}
        dump(self.path,d)
    def blind(self):
        dump(self.output/'blind_readouts.json',{k:self.data[k] for k in ('normalized','reads','shared','fixed_denominator','method_version','basis_layout')})
    def checkpoint(self):self.save();self.blind()
    def count(self,kind,completed):
        self.data['calls'][kind]['completed' if completed else 'attempted']+=1
        if kind=='vae_encode' and not completed:self.data['actual_model_calls']=True
        self.save()
    def call(self,kind,fn):
        self.count(kind,False);v=fn();self.count(kind,True);return v
    def failure(self,stage,exc):
        self.data['failures'].append(dict(stage=stage,error=f'{type(exc).__name__}: {exc}'));self.save()
    def conversion_event(self,arm,stage,row):
        self.data['conversions'][arm+'/'+stage].update(row);self.checkpoint()

def preflight(store,cfg):
    root=Path(store.data['input_root']);identity=store.data['input_identity']
    try:
        p=root/'result.json'
        if adapter.file_sha256(p)!=cfg['fixed_input']['result_sha256']:raise ValueError('parent result identity mismatch')
        parent=json.loads(p.read_text())
        if parent['source_sha']!=cfg['fixed_input']['source_sha']:raise ValueError('parent source mismatch')
        identity.update(status='VERIFIED',path=str(p),sha256=adapter.file_sha256(p),source_sha=parent['source_sha'],
                        historical_environment=parent.get('environment'),historical_vae=parent.get('vae_runtime'))
    except Exception as exc:identity.update(status='FAILED',error=str(exc));store.failure('PARENT_INPUT',exc)
    refroot=Path(store.data['reference420_root']);refidentity=store.data['reference420_identity']
    try:
        p=refroot/'result.json'
        if adapter.file_sha256(p)!=cfg['reference420_input']['result_sha256']:raise ValueError('420 parent result identity mismatch')
        refparent=json.loads(p.read_text())
        if refparent['source_sha']!=cfg['reference420_input']['source_sha']:raise ValueError('420 parent source mismatch')
        refidentity.update(status='VERIFIED',path=str(p),sha256=adapter.file_sha256(p),source_sha=refparent['source_sha'])
    except Exception as exc:refidentity.update(status='FAILED',error=str(exc));store.failure('REFERENCE420_INPUT',exc)
    for row in store.data['inputs'].values():
        p=root/row['relative_path']
        try:
            if identity['status']!='VERIFIED':raise ValueError('verified parent required')
            if p.stat().st_size!=row['bytes'] or adapter.file_sha256(p)!=row['sha256']:raise ValueError('Q8 size/hash mismatch')
            row.update(status='VERIFIED',path=str(p))
        except Exception as exc:row.update(status='FAILED',path=str(p),error=str(exc))
    for row in store.data['references'].values():
        rroot,ridentity=(root,identity) if row['root_id']=='layered' else (refroot,refidentity)
        p=rroot/row['relative_path']
        try:
            if ridentity['status']!='VERIFIED':raise ValueError('verified reference parent required')
            if adapter.file_sha256(p)!=row['sha256']:raise ValueError('reference hash mismatch')
            row.update(status='VERIFIED',path=str(p))
        except Exception as exc:row.update(status='MISSING_OR_INVALID',path=str(p),error=str(exc))
    # Do not load the parent's truth/opponent table before the new blind stage.
    store.checkpoint()

def read_normalized(z,key,call):
    received=z[0,:,1:45].numpy().transpose(1,0,2,3)
    inf=call('new_path_infer',lambda:state.infer(received,key,np.ones((44,4),bool)))
    try:bits=call('new_payload_read',lambda:payload.payload_read(z,key,44))
    except Exception as exc:bits=dict(status='FAILED',decoded_bits=None,error=str(exc))
    return dict(inference=inf,payload=bits,fixed_phase=0,truth_used=False,writer_inputs=False)

def score_new(store,arm,cfg):
    normal=store.data['normalized'][arm];z=validate_latent(reopen_tensor(normal))
    for kid,key in cfg['keys'].items():
        row=store.data['reads'][arm+'/'+kid]
        try:
            raw=read_normalized(z,key,store.call);inf=raw['inference']
            if inf['summary']['status'] in GOOD:
                if inf['counts']['scored']!=174 or len(inf['path_costs'])!=174 or not np.isfinite(inf['path_costs']).all():raise ValueError('invalid path costs')
                if len(inf['class_costs'])!=row['shared']['class_count'] or not np.isfinite(inf['class_costs']).all():raise ValueError('invalid class costs')
            raw['normalized_sha256']=normal['sha256'];dump_gzip(Path(row['path']),raw)
            row.update(status=inf['summary']['status'],payload_status=raw['payload']['status'],scored=inf['counts']['scored'],
                sha256=adapter.file_sha256(row['path']),summary=inf['summary'],payload=raw['payload'])
        except Exception as exc:row.update(status='FAILED',payload_status='FAILED',error=str(exc))
        store.checkpoint()

def conversion_worker(store,cfg):
    import torch
    from runtime.wan.generation import load_frozen_vae
    store.data['stage']='VERSIONS';store.save();versions={}
    for exe in ('ffmpeg','ffprobe'):
        p=subprocess.run([exe,'-version'],capture_output=True,text=True,check=True)
        path=store.output/(exe+'_version.txt');path.write_text(p.stdout)
        versions[exe]=dict(path=str(path),sha256=adapter.file_sha256(path),first_line=p.stdout.splitlines()[0])
    store.data['actual_conversion_versions']=versions;store.data['stage']='VAE_LOAD';store.save()
    frozen=load_frozen_vae(cfg,device='cuda' if torch.cuda.is_available() else 'cpu')
    store.data.update(model_loaded=True,vae_runtime=dict(device=str(next(frozen.parameters()).device),dtype=str(next(frozen.parameters()).dtype),
        latents_mean=list(frozen.config.latents_mean),latents_std=list(frozen.config.latents_std),posterior='mode',frames=181,normalization='(raw-mean)/std'))
    store.save()
    for arm in ARMS:
        store.data['stage']='CONVERT_ENCODE_'+arm;store.save()
        try:
            src=store.data['inputs'][arm]
            if src['status']!='VERIFIED':raise ValueError('verified Q8 unavailable')
            q8=reopen_tensor(src)
            if tuple(q8.shape)!=(181,320,512,3) or q8.dtype!=torch.uint8:raise ValueError('Q8 geometry/dtype mismatch')
            pixel=adapter.roundtrip(q8,store.data['conversions'][arm+'/yuv444']['path'],store.data['conversions'][arm+'/rgb24']['path'],
                count=store.count,event=lambda stage,row:store.conversion_event(arm,stage,row))
            del q8
            z=store.call('vae_encode',lambda:adapter.encode_normalized(frozen,pixel.float()/255.));del pixel
            row=store.data['normalized'][arm];row.update(save_tensor(row['path'],z));del z
            store.checkpoint();score_new(store,arm,cfg)
        except Exception as exc:store.failure('ARM_'+arm,exc)
        store.checkpoint()

def recover_unfinished(store,reason):
    for group in ('conversions','normalized','reads'):
        for row in store.data[group].values():
            if row['status'] in ('PENDING','RUNNING','SAVING'):row.update(status='NOT_COMPLETED',error=reason)
            if group=='reads' and row['payload_status']=='PENDING':row['payload_status']='NOT_COMPLETED'
    store.checkpoint()

def projection_metrics(q,mu):
    """Posthoc only: R44 full1408, target active1360; no threshold or path decision."""
    if q.shape!=(44,4,4,2) or mu.shape!=q.shape:raise ValueError('R44 projection required')
    active=mu!=0;energy=float(np.sum(mu*mu));dot=float(np.sum(q*mu));gain=dot/energy
    residual=q-gain*mu
    return dict(shape=list(q.shape),dimensions=1408,active_dimensions=int(active.sum()),boundary_dimensions=int((~active).sum()),
        alpha=state.PUBLIC.alpha,target_energy=energy,target_norm=energy**.5,observation_norm=float(np.linalg.norm(q)),
        dot=dot,signal_gain= gain,orthogonal_residual_full1408_l2=float(np.linalg.norm(residual)),
        sign_agreements=int(np.sum(np.sign(q[active])==np.sign(mu[active]))),sign_denominator=int(active.sum()),
        zero_observation_active=int(np.sum(q[active]==0)),zero_is_not_agreement=True,
        squared_error_by_time_block_age=np.sum((q-mu)**2,axis=-1).tolist())

def describe_inference(inf,key,opponent):
    rows=state.catalog(44);mu=state.composite_signs(key).astype(np.float64)[:44]*state.PUBLIC.alpha
    q=np.asarray(inf['projection'],dtype=np.float64);true_index=0
    costs=np.asarray(inf['path_costs'],dtype=np.float64);true_cost=float(np.mean((q-mu)**2));summary=inf['summary']
    canonical=summary['canonical_catalog_index'];winner=None;fixed=dict(opponent)
    if canonical is not None:
        target=state.composite_signs(key)[np.asarray(rows[canonical]['taus'])-1]*state.PUBLIC.alpha
        winner=candidate_terms(q,target)
    if fixed['status']=='AVAILABLE':
        target=state.composite_signs(key)[np.asarray(fixed['path']['taus'])-1]*state.PUBLIC.alpha
        terms=candidate_terms(q,target);fixed.update(status='COMPARED',terms=terms,true_minus_fixed=true_cost-terms['cost'])
    return dict(status='EVALUATED',fixed_phase=0,phase_is_not_recovered=True,registered_tau=list(range(1,45)),
        true_rank=1+int(np.sum(costs<true_cost-state.PUBLIC.tie_atol)),truth_unique_top=summary['top_catalog_indices']==[true_index],
        truth_in_top=true_index in summary['top_catalog_indices'],true_cost=true_cost,
        true_cost_ties=[i for i,c in zip(inf['valid_catalog_indices'],costs) if abs(float(c)-true_cost)<=state.PUBLIC.tie_atol],
        top_class_indices=summary['top_class_indices'],top_catalog_indices=summary['top_catalog_indices'],
        canonical_catalog_index=canonical,canonical_path=rows[canonical] if canonical is not None else None,
        canonical_start_correct=None if canonical is None else rows[canonical]['tau1']==1,
        invented_edit=None if canonical is None else rows[canonical]['event_type']!='ZERO_EDIT',
        true_minus_winner=None if winner is None else true_cost-winner['cost'],winner_terms=winner,
        fixed_historical_phase0_opponent=fixed,projection=projection_metrics(q,mu),
        accepted_payload=False,state_path_accepted=False,score_status='UNCALIBRATED_DIAGNOSTIC')

def load_parent_opponents(store,cfg):
    try:
        p=Path(store.data['input_identity']['path'])
        if adapter.file_sha256(p)!=cfg['fixed_input']['result_sha256']:raise ValueError('parent changed before posthoc')
        parent=json.loads(p.read_text())
    except Exception as exc:
        for row in store.data['fixed_opponents'].values():row.update(status='MISSING_REFERENCE',error=str(exc))
        return
    rows=state.catalog(44)
    for arm in ARMS:
        for kid in KEYS:
            row=store.data['fixed_opponents'][arm+'/'+kid]
            try:
                old=parent['path_posthoc'][arm+'/RGB8_NO_CODEC/'+kid]['fixed_historical_phase0_opponent'];i=old['catalog_index']
                if old['status']!='COMPARED' or rows[i]!=old['path'] or not rows[i]['structurally_valid']:raise ValueError('invalid fixed old phase0 opponent')
                row.update(status='AVAILABLE',catalog_index=i,path=old['path'],raw_sha256=old['raw_sha256'],
                    parent_result_sha256=cfg['fixed_input']['result_sha256'],role='original FULL phase0 opponent carried by parent posthoc; no old root reread')
            except Exception as exc:row.update(status='MISSING_REFERENCE',error=str(exc))

def evaluate_saved(store,cfg):
    store.data['stage']='POSTHOC';store.checkpoint();blind=store.output/'blind_readouts.json';before=adapter.file_sha256(blind)
    load_parent_opponents(store,cfg)
    for sid,row in store.data['reads'].items():
        arm,kid=sid.split('/');dest=store.data['path_posthoc'][sid]
        try:
            if row['status'] not in GOOD:raise ValueError('new raw unavailable')
            if adapter.file_sha256(row['path'])!=row['sha256']:raise ValueError('new raw identity mismatch')
            raw=load_gzip(row['path']);dest.update(describe_inference(raw['inference'],cfg['keys'][kid],store.data['fixed_opponents'][sid]))
        except Exception as exc:dest.update(status='MISSING_READ',error=str(exc))
        for target,text in cfg['messages'].items():
            m=store.data['message_evaluations'][sid+'/'+target];bits=(row.get('payload') or {}).get('decoded_bits')
            if row['payload_status']=='READ' and bits is not None:
                truth=payload.message_bits(text);m.update(status='EVALUATED',diagnostic_bit_errors=sum(a!=b for a,b in zip(bits,truth)),diagnostic_exact=bits==truth)
            else:m.update(status='MISSING_READ')
        store.save()
    for ref in store.data['references'].values():
        if ref['kind']!='raw':continue
        arm,kid,layer=ref['arm'],ref['key_id'],ref['layer'];sid=arm+'/'+layer+'/'+kid
        report=dict(status='MISSING_REFERENCE',new_infer_calls=0)
        try:
            if ref['status']!='VERIFIED' or adapter.file_sha256(ref['path'])!=ref['sha256']:raise ValueError('reference raw unavailable/changed')
            raw=load_gzip(ref['path']);inf=raw['inference']
            if inf['summary']['status'] not in GOOD or inf['received_length']!=44 or inf['counts']['scored']!=174:raise ValueError('reference incomplete or support changed')
            report.update(describe_inference(inf,cfg['keys'][kid],store.data['fixed_opponents'][arm+'/'+kid]),reference_raw_sha256=ref['sha256'],
                          source='existing raw projection and costs; no new infer/VAE',diagnostic_payload=raw.get('payload'))
            fresh=store.data['path_posthoc'][arm+'/'+kid]
            if fresh['status']=='EVALUATED':report['new_minus_reference_true_cost']=fresh['true_cost']-report['true_cost']
        except Exception as exc:report['error']=str(exc)
        store.data['reference_posthoc'][sid]=report
    if adapter.file_sha256(blind)!=before:raise RuntimeError('posthoc changed blind evidence')
    store.data['blind_sha256']=before;store.save()

def finalize(store,cfg):
    recover_unfinished(store,'worker or input incomplete');evaluate_saved(store,cfg)
    expected=dict(yuv444_files=3,rgb24_files=3,normalized=3,path_reads=6,payload_reads=6,valid_costs=1044,
        catalog_slots=23490,structural_exclusions=22446,path_posthoc=6,message_evaluations=12)
    done=store.data['counts']==expected and all(store.data['calls'][k]['completed']==n for k,n in cfg['planned_calls'].items()) and not store.data.get('interrupted',False)
    refs=all(r['status']=='VERIFIED' for r in store.data['references'].values())
    comparisons=len(store.data['reference_posthoc'])==18 and all(r['status']=='EVALUATED' for r in store.data['reference_posthoc'].values())
    opponents=all(r['status']=='AVAILABLE' for r in store.data['fixed_opponents'].values())
    store.data.update(status='EXECUTION_COMPLETE' if done else 'INCOMPLETE',primary_execution='EXECUTION_COMPLETE' if done else 'INCOMPLETE',
        reference_completeness='COMPLETE' if refs and comparisons else 'INCOMPLETE',posthoc_fixed_opponent_status='COMPLETE' if opponents else 'INCOMPLETE',
        package_completeness='COMPLETE' if done and refs and comparisons and opponents else 'INCOMPLETE',stage='FINISHED')
    store.save();return done

def worker_phase(output):
    command=[sys.executable,'-u','-m',MODULE,'--output',str(output),'--worker'];child=None;error=None;code=None;halt=False;cleanup=None
    try:
        with (Path(output)/'conversion_encode.log').open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait()
            if code!=0:code=stop_worker(child)
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';halt=not isinstance(exc,Exception)
        if child is not None:
            try:code=stop_worker(child)
            except BaseException as err:cleanup=str(err);halt=True
    finally:
        if child is not None and child.stdout is not None:
            try:child.stdout.close()
            except Exception:pass
    store=Store(output)
    store.data['workers']['conversion_encode']=dict(command=command,status='COMPLETE' if error is None and code==0 else 'FAILED',
        error=error,cleanup_error=cleanup,returncode=code)
    if error or code!=0:store.failure('WORKER',RuntimeError(error or str(code)))
    if halt:store.data['interrupted']=True
    store.checkpoint();return store,halt

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--input-root',type=Path);p.add_argument('--reference420-root',type=Path);p.add_argument('--worker',action='store_true')
    args=p.parse_args();cfg=load_config()
    if args.worker:
        store=Store(args.output)
        try:conversion_worker(store,cfg)
        except Exception as exc:store.failure('CONVERSION_ENCODE',exc);recover_unfinished(store,str(exc));raise
        return
    store=Store(args.output,create=True,cfg=cfg,input_root=args.input_root or cfg['fixed_input']['root'],reference420_root=args.reference420_root or cfg['reference420_input']['root']);preflight(store,cfg)
    if any(r['status']=='VERIFIED' for r in store.data['inputs'].values()):store,halt=worker_phase(args.output)
    else:halt=False;store.data['workers']['conversion_encode']=dict(status='NOT_STARTED_NO_INPUT')
    done=finalize(store,cfg);print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])))
    if halt:raise SystemExit(130)
    if not done:raise SystemExit(1)

if __name__=='__main__':main()
