"""Fixed three-arm temporal-pilot bridge; persisted blind search then truth joins."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from main.tube_state import video_temporal_sync_bridge as method
from runtime.wan import video_temporal_sync_bridge as backend

ROOT=Path(__file__).resolve().parents[2]
CONFIG=Path(__file__).parent/'configs/video_temporal_sync_bridge_v1.json'
MODULE='experiments.wan_state_clock.video_temporal_sync_bridge_run'
KEY_IDS=('CORRECT','WRONG')
TARGET_IDS=('REGISTERED','WRONG_MESSAGE')
TABLE_SIZES=dict(generation=3,sources=3,views=6,normalized=15,phase_reads=30,candidates=324,searches=12,evaluations=24)
DENOMINATOR=dict(source_cases=1,arms=3,source_mp4=3,derived_mp4=6,normalized_phase_features=15,phase_payload_reads=30,
    primary_searches=12,pilot_candidates=324,posthoc_evaluations=24,pilot_positives=2,pilot_negatives=10)


def dump(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def environment():
    packages={}
    for name in ('torch','torchvision','diffusers','transformers','accelerate','numpy','imageio','imageio-ffmpeg'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    return dict(python=sys.version,executable=sys.executable,packages=packages)


def load_config():
    cfg=json.loads(CONFIG.read_text())
    if cfg['fixed_denominator']!=DENOMINATOR or cfg['arms']!=list(method.ARMS):raise ValueError('fixed denominator/arms mismatch')
    if cfg['control']['joint_mask_count']!=47104 or cfg['control']['joint_eta']!=5888 or cfg['receiver']['threshold']!=method.PUBLIC.threshold:
        raise ValueError('fixed protocol mismatch')
    method.layout_receipt(cfg['key'],cfg['wrong_key'])
    return cfg


def initial_result(output):
    cfg=load_config();tables={name:{} for name in TABLE_SIZES}
    for arm in method.ARMS:
        tables['generation'][arm]=dict(status='PENDING',steps=[],terminal_path=str(output/arm/'terminal.pt'))
        tables['sources'][arm]=dict(status='PENDING',path=str(output/arm/'source.mp4'))
        for view,length in zip(method.VIEWS,(181,129)):
            av=f'{arm}/{view}';tables['views'][av]=dict(status='PENDING',path=str(output/arm/(view+'.mp4')))
            for g in method.phases(length):
                tables['normalized'][f'{av}/{g}']=dict(status='PENDING',g=g,path=str(output/arm/(view+f'.phase{g}.pt')))
                for key in KEY_IDS:tables['phase_reads'][f'{av}/{g}/{key}']=dict(status='PENDING',g=g,decoded_bits=None,truth_used=False)
            for key in KEY_IDS:
                sid=f'{av}/{key}';tables['searches'][sid]=dict(status='PENDING',decoded_bits=None,truth_used=False)
                for row in method.candidates(length):tables['candidates'][f'{sid}/{row["b"]}']=dict(**row,status='PENDING',score=None)
                for target in TARGET_IDS:tables['evaluations'][f'{sid}/{target}']=dict(status='PENDING',bit_errors=None,exact_bits=None)
    source_paths=('main/tube_state/video_temporal_sync_bridge.py','runtime/wan/video_temporal_sync_bridge.py',
        'runtime/wan/generation.py','runtime/wan/trajectory.py','runtime/wan/vae.py','runtime/wan/io.py',
        'experiments/wan_state_clock/video_temporal_sync_bridge_run.py',
        'experiments/wan_state_clock/configs/video_temporal_sync_bridge_v1.json',
        'experiments/wan_state_clock/requirements-grow-video-reference.txt')
    return dict(status='RUNNING',stage='INITIALIZE',fixed_denominator=DENOMINATOR,**tables,
        calls={},workers={},failures=[],quality={},terminal_diagnostics={f'{a}/{k}':dict(status='PENDING',decoded_bits=None) for a in method.ARMS for k in KEY_IDS},
        diagnostic_denominator=dict(terminal_feature_reads=6,terminal_pilot_scores=6,oracle_saved_feature_references=12),model_loaded={'generation':False,'media':False},
        actual_model_calls=False,source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_files={p:backend.file_sha256(ROOT/p) for p in source_paths},config_sha256=backend.file_sha256(CONFIG),
        environment=environment(),layout=method.layout_receipt(cfg['key'],cfg['wrong_key']),
        evidence_ceiling='One source; dependent controls; uncalibrated pilot gate; full localization trivial; no scientific PASS')


class Store:
    def __init__(self,output,*,create=False):
        self.output=Path(output);self.path=self.output/'result.json'
        if create:self.output.mkdir(parents=True,exist_ok=False);self.data=initial_result(self.output)
        else:self.data=json.loads(self.path.read_text())
        self.save()
    def save(self):
        if any(len(self.data[k])!=n for k,n in TABLE_SIZES.items()):raise ValueError('fixed roster changed')
        d=self.data
        d['counts']=dict(generated=sum(r['status']=='COMPLETE' for r in d['generation'].values()),
            source_mp4=sum(r['status']=='SAVED' for r in d['sources'].values()),derived_mp4=sum(r['status']=='SAVED' for r in d['views'].values()),
            normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),phase_reads=sum(r['status']=='READ' for r in d['phase_reads'].values()),
            candidates=sum(r['status'] in ('SCORED','NO_ENERGY') for r in d['candidates'].values()),
            searches=sum(r['status'] in ('LOCATED','NO_PILOT','AMBIGUOUS') for r in d['searches'].values()),
            evaluated=sum(r['status']=='EVALUATED' for r in d['evaluations'].values()))
        d['diagnostic_counts']=dict(terminal_reads=sum(r['status']=='READ' for r in d['terminal_diagnostics'].values()))
        dump(self.path,d)
    def count(self,kind,completed):
        row=self.data['calls'].setdefault(self.data['stage'],{}).setdefault(kind,dict(attempted=0,completed=0))
        row['completed' if completed else 'attempted']+=1
        if kind.startswith(('transformer_','vae_')) and not completed:self.data['actual_model_calls']=True
        if kind!='pilot_candidate_score':self.save()
    def failure(self,stage,error):
        self.data['failures'].append(dict(stage=stage,error=f'{type(error).__name__}: {error}'));self.save()
    def blind(self):
        dump(self.output/'blind_readouts.json',dict(phase_reads=self.data['phase_reads'],candidates=self.data['candidates'],
            searches=self.data['searches'],normalized=self.data['normalized'],truth_inputs=False,fixed_denominator=DENOMINATOR))
    def event(self,av,kind,slot,row):
        if kind=='normalized':
            import torch
            target=self.data['normalized'][f'{av}/{slot}'];path=Path(target['path']);path.parent.mkdir(parents=True,exist_ok=True)
            temp=path.with_suffix('.tmp');torch.save(row,temp);os.replace(temp,path)
            target.update(status='SAVED',sha256=backend.file_sha256(path),shape=list(row.shape))
        elif kind=='phase':self.data['phase_reads'][f'{av}/{slot[0]}/{slot[1]}'].update(row)
        elif kind=='candidate':
            self.data['candidates'][f'{av}/{slot[0]}/{slot[1]}'].update(row)
            return  # Atomic batch at this key's search boundary; initial324 slots remain on disk.
        elif kind=='search':self.data['searches'][f'{av}/{slot}'].update(row)
        else:raise ValueError(kind)
        # result.json is canonical; blind_readouts.json is its recoverable projection.
        self.save();self.blind()


def recover_unfinished(store,phase,error):
    groups=('generation','terminal_diagnostics') if phase=='generation' else ('sources','views','normalized','phase_reads','candidates','searches')
    for group in groups:
        for row in store.data[group].values():
            if row['status'] in ('PENDING','RUNNING','SAVING'):
                row.update(status='SEARCH_INCOMPLETE' if group=='searches' else 'NOT_COMPLETED',error=error)
    store.save();store.blind()


def generation_worker(store,cfg):
    import torch
    from runtime.wan.generation import prepare_generation
    from runtime.wan import trajectory
    device,dtype=backend.execution_device_dtype();store.data['stage']='GENERATION_LOAD';store.save()
    pipe,initial,prompt,negative,input_dtype=prepare_generation(cfg,load_vae=False,device=device,model_dtype=dtype)
    store.data['model_loaded']['generation']=True;pristine=copy.deepcopy(pipe.scheduler)
    initial_hash=trajectory.fingerprint(initial);history_hash=trajectory.fingerprint(vars(pristine));reference=None
    store.data['generation_setup']=dict(device=device,transformer_dtype=str(dtype),state_control_cfg='float32',initial_sha256=initial_hash,
        pristine_history_sha256=history_hash,model=cfg['model'],seed=cfg['generation']['seed']);store.save()
    for arm in method.ARMS:
        store.data['stage']='GENERATE_'+arm;row=store.data['generation'][arm];row['status']='RUNNING';store.save()
        try:
            def record(value):row['steps'].append(value);store.save()
            terminal,receipt=backend.run_trajectory(pipe,initial,copy.deepcopy(pristine),prompt,negative,input_dtype,arm,
                cfg['key'],method.message_bits(cfg['message']),store.count,record)
            if reference is None:reference=receipt['before_step49']
            if receipt['before_step49']!=reference:raise RuntimeError('step49 z/history/conditional/unconditional mismatch')
            if trajectory.fingerprint(initial)!=initial_hash or trajectory.fingerprint(vars(pristine))!=history_hash:raise RuntimeError('shared initial state changed')
            path=Path(row['terminal_path']);path.parent.mkdir(parents=True,exist_ok=True);torch.save(terminal,path)
            row.update(status='COMPLETE',sha256=backend.file_sha256(path),receipt=receipt);store.save()
            terminal_diagnostics(store,arm,terminal,cfg);del terminal
        except Exception as exc:row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');store.failure(store.data['stage'],exc)
        store.save()
    complete=[r for r in store.data['generation'].values() if r['status']=='COMPLETE']
    store.data['before_step49_identity']=dict(status='MATCH' if len(complete)==3 else 'INCOMPLETE',checked_arms=len(complete))
    store.save()


def terminal_diagnostics(store,arm,terminal,cfg):
    """Separate normalized-terminal observations; never supply the MP4 receiver."""
    store.data['stage']='TERMINAL_DIAGNOSTIC_'+arm;store.save()
    for name,key in (('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])):
        row=store.data['terminal_diagnostics'][f'{arm}/{name}']
        try:
            feature=backend.counted(store.count,'terminal_feature_read',lambda:method.phase_features(terminal,key,45))
            score=backend.counted(store.count,'terminal_pilot_score',lambda:method.score_candidate(feature,key,method.candidates(181)[0]))
            row.update(status='READ',decoded_bits=feature['decoded_bits'],votes=feature['votes'],pilot=score,
                source='saved normalized terminal; excludes first latent; diagnostic only',truth_used=False)
        except Exception as exc:row.update(status='FAILED',error=str(exc))
        store.save()


def media_worker(store,cfg):
    import torch
    from runtime.wan.generation import load_frozen_vae
    from runtime.wan import io,vae as vae_adapter
    keys=dict(CORRECT=cfg['key'],WRONG=cfg['wrong_key'])
    if not any(r['status']=='COMPLETE' for r in store.data['generation'].values()):
        recover_unfinished(store,'media','no completed generation');return
    store.data['stage']='MEDIA_LOAD';store.save();device,_=backend.execution_device_dtype()
    frozen=load_frozen_vae(cfg,device=device);store.data['model_loaded']['media']=True;store.save()
    for arm in method.ARMS:
        generated=store.data['generation'][arm];source=store.data['sources'][arm]
        store.data['stage']='SOURCE_'+arm;store.save()
        if generated['status']=='COMPLETE':
            try:
                if backend.file_sha256(generated['terminal_path'])!=generated['sha256']:raise RuntimeError('terminal identity mismatch')
                terminal=torch.load(generated['terminal_path'],map_location='cpu',weights_only=True)
                rgb=backend.counted(store.count,'vae_decode',lambda:vae_adapter.decode_normalized_latent(frozen,terminal.to(next(frozen.parameters()).device)))
                path=Path(source['path']);path.parent.mkdir(parents=True,exist_ok=True);source['status']='SAVING';store.save()
                backend.counted(store.count,'source_mp4_save',lambda:io.encode_rgb(rgb,path,8,18))
                source.update(status='SAVED',sha256=backend.file_sha256(path));del rgb,terminal
            except Exception as exc:source.update(status='FAILED',error=str(exc));store.failure(store.data['stage'],exc)
            store.save()
        if source['status']!='SAVED':continue
        try:
            if backend.file_sha256(source['path'])!=source['sha256']:raise RuntimeError('source MP4 identity mismatch')
            received=backend.counted(store.count,'source_mp4_read',lambda:io.read_mp4(Path(source['path'])))
            if tuple(received.shape)!=(181,320,512,3):raise ValueError('source MP4 geometry mismatch')
        except Exception as exc:store.failure('SOURCE_READ_'+arm,exc);continue
        for view in method.VIEWS:
            av=f'{arm}/{view}';row=store.data['views'][av];store.data['stage']='DERIVE_'+av
            spec=cfg['views'][view];path=Path(row['path'])
            try:
                row['status']='SAVING';store.save()
                clip=received[spec['start']:spec['start']+spec['length']]
                backend.counted(store.count,'derived_mp4_save',lambda:io.encode_rgb(clip,path,8,18))
                row.update(status='SAVED',sha256=backend.file_sha256(path),frames=spec['length'],second_codec=True)
            except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure(store.data['stage'],exc)
            store.save()
        del received
        for view in method.VIEWS:
            av=f'{arm}/{view}';row=store.data['views'][av]
            if row['status']!='SAVED':continue
            store.data['stage']='PRIMARY_'+av;store.save()
            try:
                if backend.file_sha256(row['path'])!=row['sha256']:raise RuntimeError('derived MP4 identity mismatch')
                backend.read_mp4_search(row['path'],keys,method.PUBLIC,frozen,count=store.count,
                    persist=lambda kind,slot,value:store.event(av,kind,slot,value))
            except Exception as exc:store.failure(store.data['stage'],exc)
    recover_unfinished(store,'media','unavailable input or incomplete media/search')


def evaluate_saved_reads(store,cfg):
    """Posthoc truth only. Oracle uses one registered saved phase; no new search/VAE."""
    path=store.output/'blind_readouts.json';before=backend.file_sha256(path);raw=json.loads(path.read_text())
    store.data['stage']='POSTHOC_EVALUATION';store.save()
    for sid,search in raw['searches'].items():
        arm,view,key_id=sid.split('/');b=0 if view=='FULL_RESAVED181' else 5;g=(-b)%4
        positive=arm=='PILOT_LAST' and key_id=='CORRECT';oracle=dict(status='MISSING_FEATURE',decoded_bits=None)
        # The raw phase features and candidate score were already persisted by
        # the blind receiver. Registering the known origin here causes no FFT,
        # candidate search or VAE call, and cannot alter the primary decision.
        feature=raw['phase_reads'][f'{arm}/{view}/{g}/{key_id}']
        candidate=raw['candidates'][f'{sid}/{b}']
        store.count('oracle_saved_feature_reference',False)
        if feature['status']=='READ':
            oracle=dict(status='READ',decoded_bits=feature['decoded_bits'],registered_b=b,registered_g=g,
                source='persisted phase feature and candidate; posthoc truth only',pilot_score=candidate.get('score'),
                candidate_status=candidate['status'],feature_sha256=feature.get('feature_sha256'),new_fft_calls=0,new_vae_calls=0)
            store.count('oracle_saved_feature_reference',True)
        for target_id,message in (('REGISTERED',cfg['message']),('WRONG_MESSAGE',cfg['wrong_message'])):
            row=store.data['evaluations'][f'{sid}/{target_id}'];truth=method.message_bits(message);bits=search.get('decoded_bits')
            completed=search['status'] in ('LOCATED','NO_PILOT','AMBIGUOUS')
            row.update(status='EVALUATED' if completed else 'MISSING_SEARCH',search_status=search['status'],expected_pilot_present=positive,
                pilot_negative_category=None if positive else ('OFF_UNWATERMARKED' if arm=='OFF' else ('PAYLOAD_ONLY' if arm=='PAYLOAD_LAST' else 'PILOT_WRONG_KEY')),
                gate_pass=search['status']=='LOCATED',accepted_payload=search['status']=='LOCATED' and bits is not None,
                normal_rejection=search['status'] in ('NO_PILOT','AMBIGUOUS'),registered_b=b,localized_correctly=search.get('selected_b')==b if search['status']=='LOCATED' else False,
                nontrivial_localization=view=='CROP5_129',bit_errors=sum(x!=y for x,y in zip(bits,truth)) if bits is not None else None,
                exact_bits=bits==truth if bits is not None else None,oracle=oracle,
                fixed_g0_payload=raw['phase_reads'][f'{arm}/{view}/0/{key_id}'].get('decoded_bits'))
            row['ber']=row['bit_errors']/32 if row['bit_errors'] is not None else None
        store.save()
    if backend.file_sha256(path)!=before:raise RuntimeError('truth evaluation modified blind evidence')
    store.data['blind_sha256']=before;store.save()


def quality_diagnostics(store):
    from runtime.wan import io
    store.data['stage']='QUALITY';store.save()
    for view in method.VIEWS:
        observed={}
        for arm in method.ARMS:
            row=store.data['views'][f'{arm}/{view}']
            if row['status']!='SAVED':continue
            try:
                if backend.file_sha256(row['path'])!=row['sha256']:raise RuntimeError('quality MP4 identity mismatch')
                observed[arm]=backend.counted(store.count,'quality_mp4_read',lambda:io.read_mp4(Path(row['path'])))
            except Exception as exc:store.data['quality'][f'{view}/read/{arm}']=dict(status='FAILED',error=str(exc))
        for arm,reference in (('PAYLOAD_LAST','OFF'),('PILOT_LAST','OFF'),('PILOT_LAST','PAYLOAD_LAST')):
            key=f'{view}/{arm}_vs_{reference}';row=dict(status='MISSING_MP4',threshold=None,diagnostic_only=True)
            if arm in observed and reference in observed:
                try:
                    x,y=observed[arm],observed[reference]
                    if x.shape!=y.shape:raise ValueError('quality geometry mismatch')
                    squared=sum(float((x[i].double()-y[i].double()).square().sum()) for i in range(len(x)))
                    mse=squared/x.numel();row.update(status='MEASURED',rgb_rmse=math.sqrt(mse),psnr_db=-10*math.log10(mse) if mse else None,identical_pixels=mse==0)
                except Exception as exc:row.update(status='FAILED',error=str(exc))
            store.data['quality'][key]=row;store.save()
        del observed


def _stop_worker(child):
    if child.poll() is None:
        try:child.terminate()
        except ProcessLookupError:pass
    try:return child.wait(timeout=10)
    except subprocess.TimeoutExpired:child.kill();return child.wait(timeout=10)


def run_worker_phase(output,phase):
    command=[sys.executable,'-u','-m',MODULE,'--output',str(output),'--worker',phase]
    start=time.perf_counter();child=None;code=None;error=None;cleanup_error=None;halt=None
    try:
        with (output/(phase+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait()
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}'
        if not isinstance(exc,Exception):halt='INTERRUPTED'
        if child is not None:
            try:code=_stop_worker(child)
            except BaseException as cleanup:cleanup_error=str(cleanup);halt=halt or 'CLEANUP_FAILED'
    finally:
        if child is not None and child.stdout is not None:
            try:child.stdout.close()
            except Exception:pass
    store=Store(output)
    status=('START_FAILED' if child is None else 'MONITOR_FAILED') if error else ('COMPLETE' if code==0 else 'NONZERO_EXIT')
    store.data['workers'][phase]=dict(command=command,status=status,returncode=code,elapsed_seconds=time.perf_counter()-start,
        child_started=child is not None,error=error,cleanup_error=cleanup_error,last_child_stage=store.data['stage'])
    if status!='COMPLETE':
        reason=error or f'worker exited {code}';store.failure('WORKER_'+phase.upper(),RuntimeError(reason));recover_unfinished(store,phase,reason)
    store.save();return store,halt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',choices=('generation','media'))
    args=p.parse_args();cfg=load_config()
    if args.worker:
        store=Store(args.output)
        try:(generation_worker if args.worker=='generation' else media_worker)(store,cfg)
        except Exception as exc:store.failure(store.data['stage'],exc);recover_unfinished(store,args.worker,str(exc));raise
        return
    store=Store(args.output,create=True);halt=None
    for phase in ('generation','media'):
        store,halt=run_worker_phase(args.output,phase)
        if halt:
            if phase=='generation':store.data['workers']['media']=dict(status='NOT_STARTED_'+halt)
            recover_unfinished(store,'media',halt);break
    recover_unfinished(store,'generation','generation incomplete');recover_unfinished(store,'media','media incomplete')
    evaluate_saved_reads(store,cfg)
    expected=dict(generated=3,source_mp4=3,derived_mp4=6,normalized=15,phase_reads=30,candidates=324,searches=12,evaluated=24)
    done=store.data['counts']==expected and all(r['status']=='COMPLETE' for r in store.data['workers'].values())
    store.data['status']=halt or ('EXECUTION_COMPLETE' if done else 'INCOMPLETE');store.save()
    if not halt:quality_diagnostics(store)
    store.data['stage']='FINISHED';store.save();print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])))
    if halt=='INTERRUPTED':raise SystemExit(130)
    if not done:raise SystemExit(1)


if __name__=='__main__':main()
