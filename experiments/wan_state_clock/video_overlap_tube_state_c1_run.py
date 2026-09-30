"""C1 user-run three-arm Wan overlap-state candidate; no automatic model execution."""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.metadata,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from main.tube_state import video_overlap_tube_control as method
from main.tube_state import video_overlap_tube_state as state
from runtime.wan import video_overlap_tube_state as backend
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_overlap_tube_state_c1_v1.json'
MODULE='experiments.wan_state_clock.video_overlap_tube_state_c1_run'
KEY_IDS=('CORRECT','WRONG');TARGET_IDS=('REGISTERED','WRONG_MESSAGE')
TABLE_SIZES=dict(generation=3,sources=3,views=15,normalized=60,phase_reads=120,searches=30,evaluations=60)
DENOMINATOR=dict(source_cases=1,arms=3,source_mp4=3,derived_mp4=15,normalized=60,phase_path_searches=120,phase_payload_reads=120,
 primary_searches=30,posthoc=60,catalog_slots=357480,valid_costs=92016,structural_exclusions=265464,pilot_positives=5,pilot_negatives=25)

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
    mechanism=ROOT/'experiments/wan_state_clock/configs/video_overlap_tube_state_v1.json'
    if backend.file_sha256(mechanism)!=cfg['mechanism_config_sha256']:raise ValueError('frozen A/B identity mismatch')
    if cfg['fixed_denominator']!=DENOMINATOR or cfg['arms']!=list(method.ARMS):raise ValueError('fixed roster mismatch')
    if cfg['control']['steps']!=list(range(25,50)) or cfg['control']['pilot_eta']!=174 or cfg['control']['pilot_delta_cap']!=1.:raise ValueError('fixed control mismatch')
    if cfg['views']!={'FULL_RESAVED181':dict(start=0,length=181),**{f'CROP{i}_129':dict(start=i,length=129) for i in range(4,8)}}:raise ValueError('fixed C1 views required')
    method.payload.layout_receipt(cfg['key'],cfg['wrong_key'])
    return cfg


def dump_gzip(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(gzip.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode(),mtime=0));os.replace(temp,path)


def initial_result(output):
    cfg=load_config();tables={name:{} for name in TABLE_SIZES};shared={}
    for R in (44,31):
        path=output/f'catalog_R{R}.json.gz';rows=state.catalog(R);dump_gzip(path,dict(rows=rows))
        for key_id,key in (('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])):
            fp=output/f'family_R{R}_{key_id}.json.gz';family=state.family_receipt(key,R,np.ones((R,4),bool));dump_gzip(fp,family)
            shared[f'{R}/{key_id}']=dict(catalog_path=str(path),catalog_sha256=backend.file_sha256(path),family_path=str(fp),family_sha256=backend.file_sha256(fp),
                catalog_count=len(rows),valid_count=len(family['valid_catalog_indices']),class_count=len(family['classes']))
    for arm in method.ARMS:
        tables['generation'][arm]=dict(status='PENDING',steps=[],terminal_path=str(output/arm/'terminal.pt'))
        tables['sources'][arm]=dict(status='PENDING',path=str(output/arm/'source.mp4'))
        for view,spec in cfg['views'].items():
            av=f'{arm}/{view}';R=method.phase_spec(spec['length'],0)['R'];tables['views'][av]=dict(status='PENDING',path=str(output/arm/(view+'.mp4')))
            for g in range(4):
                tables['normalized'][f'{av}/{g}']=dict(status='PENDING',g=g,path=str(output/arm/(view+f'.phase{g}.pt')))
                for k in KEY_IDS:
                    tables['phase_reads'][f'{av}/{g}/{k}']=dict(status='PENDING',g=g,R=R,payload=None,
                        path=str(output/arm/(view+f'.phase{g}.{k}.raw.json.gz')),shared=shared[f'{R}/{k}'])
            for k in KEY_IDS:
                sid=f'{av}/{k}';tables['searches'][sid]=dict(status='PENDING',accepted_payload=False,state_path_accepted=False,canonical=None)
                for target in TARGET_IDS:tables['evaluations'][f'{sid}/{target}']=dict(status='PENDING',bit_errors=None,accepted_payload=False)
    paths=['main/tube_state/video_overlap_tube_state.py','main/tube_state/video_overlap_tube_control.py','main/tube_state/grow_video_reference.py',
      'runtime/wan/video_overlap_tube_state.py','runtime/wan/video_temporal_sync_bridge.py','main/tube_state/video_temporal_sync_bridge.py','runtime/wan/generation.py','runtime/wan/trajectory.py','runtime/wan/vae.py','runtime/wan/io.py',
      'experiments/wan_state_clock/video_overlap_tube_state_c1_run.py','experiments/wan_state_clock/configs/video_overlap_tube_state_c1_v1.json',
      'experiments/wan_state_clock/configs/video_overlap_tube_state_v1.json','experiments/wan_state_clock/requirements-grow-video-reference.txt']
    return dict(status='RUNNING',stage='INITIALIZE',fixed_denominator=DENOMINATOR,**tables,shared=shared,
      calls={},workers={},failures=[],quality={},terminal_diagnostics={f'{a}/{k}':dict(status='PENDING') for a in method.ARMS for k in KEY_IDS},
      model_loaded={'generation':False,'media':False},actual_model_calls=False,source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      source_files={p:backend.file_sha256(ROOT/p) for p in paths},config_sha256=backend.file_sha256(CONFIG),environment=environment(),
      evidence_ceiling=cfg['evidence_ceiling'])


class Store:
    def __init__(self,output,*,create=False):
        self.output=Path(output);self.path=self.output/'result.json'
        if create:self.output.mkdir(parents=True,exist_ok=False);self.data=initial_result(self.output)
        else:self.data=json.loads(self.path.read_text())
        self.save()
    def save(self):
        d=self.data
        if any(len(d[k])!=n for k,n in TABLE_SIZES.items()):raise ValueError('fixed roster changed')
        phase=list(d['phase_reads'].values())
        d['counts']=dict(generated=sum(r['status']=='COMPLETE' for r in d['generation'].values()),source_mp4=sum(r['status']=='SAVED' for r in d['sources'].values()),
          derived_mp4=sum(r['status']=='SAVED' for r in d['views'].values()),normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),
          phase_searches=sum(r['status'] in ('COMPLETE','NO_ENERGY') for r in phase),
          phase_payload=sum((r.get('payload') or {}).get('status')=='READ' for r in phase),
          path_costs=sum(r.get('scored',0) for r in phase),searches=sum(r['status'] in ('COMPLETE','NO_ENERGY') for r in d['searches'].values()),
          evaluated=sum(r['status']=='EVALUATED' for r in d['evaluations'].values()))
        dump(self.path,d)
    def count(self,kind,completed):
        row=self.data['calls'].setdefault(self.data['stage'],{}).setdefault(kind,dict(attempted=0,completed=0))
        row['completed' if completed else 'attempted']+=1
        if kind.startswith(('transformer_','vae_')) and not completed:self.data['actual_model_calls']=True
        self.save()
    def failure(self,stage,error):
        self.data['failures'].append(dict(stage=stage,error=f'{type(error).__name__}: {error}'));self.save()
    def blind(self):
        dump(self.output/'blind_readouts.json',dict(normalized=self.data['normalized'],phase_reads=self.data['phase_reads'],searches=self.data['searches'],
            shared=self.data['shared'],fixed_denominator=DENOMINATOR,truth_inputs=False))
    def event(self,av,kind,slot,row):
        if kind=='normalized':
            import torch
            target=self.data['normalized'][f'{av}/{slot}'];path=Path(target['path']);path.parent.mkdir(parents=True,exist_ok=True)
            tmp=path.with_suffix('.tmp');torch.save(row,tmp);os.replace(tmp,path)
            target.update(status='SAVED',sha256=backend.file_sha256(path),shape=list(row.shape))
        elif kind=='phase':
            target=self.data['phase_reads'][f'{av}/{slot[0]}/{slot[1]}'];path=Path(target['path']);path.parent.mkdir(parents=True,exist_ok=True)
            dump_gzip(path,row);inf=row.get('inference');scored=0
            if inf is not None and inf['summary']['status'] in ('COMPLETE','NO_ENERGY'):
                if len(inf['path_costs'])!=target['shared']['valid_count'] or not np.isfinite(inf['path_costs']).all():raise ValueError('incomplete path costs')
                if len(inf['class_costs'])!=target['shared']['class_count'] or not np.isfinite(inf['class_costs']).all():raise ValueError('incomplete class costs')
                scored=inf['counts']['scored']
                if scored!=target['shared']['valid_count']:raise ValueError('score count mismatch')
            target.update(status=row['status'],sha256=backend.file_sha256(path),scored=scored,payload=row.get('payload'),summary=inf['summary'] if inf else None,error=row.get('error'))
        elif kind=='search':self.data['searches'][f'{av}/{slot}'].update(row)
        else:raise ValueError(kind)
        self.save();self.blind()  # Canonical-first persistence; blind is recoverable.


def recover_unfinished(store,phase,error):
    groups=('generation','terminal_diagnostics') if phase=='generation' else ('sources','views','normalized','phase_reads','searches')
    for group in groups:
        for row in store.data[group].values():
            if row['status'] in ('PENDING','RUNNING','SAVING'):row.update(status='SEARCH_INCOMPLETE' if group=='searches' else 'NOT_COMPLETED',error=error)
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
            if reference is None:reference=receipt['before_step25']
            if receipt['before_step25']!=reference:raise RuntimeError('step25 z/history/conditional/unconditional mismatch')
            if trajectory.fingerprint(initial)!=initial_hash or trajectory.fingerprint(vars(pristine))!=history_hash:raise RuntimeError('shared initial state changed')
            path=Path(row['terminal_path']);path.parent.mkdir(parents=True,exist_ok=True);torch.save(terminal,path)
            receipt['control_norm_statistics']={k:dict(sum=sum(s.get(k,0.) for s in row['steps']),sum_squares=sum(s.get(k,0.)**2 for s in row['steps']),maximum=max(s.get(k,0.) for s in row['steps'])) for k in ('payload_delta_l2','pilot_delta_l2','merged_delta_l2')}
            row.update(status='COMPLETE',sha256=backend.file_sha256(path),receipt=receipt);store.save()
            terminal_diagnostics(store,arm,terminal,cfg);del terminal
        except Exception as exc:row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');store.failure(store.data['stage'],exc)
        store.save()
    complete=[r for r in store.data['generation'].values() if r['status']=='COMPLETE']
    store.data['before_step25_identity']=dict(status='MATCH' if len(complete)==3 else 'INCOMPLETE',checked_arms=len(complete))
    store.save()


def terminal_diagnostics(store,arm,terminal,cfg):
    store.data['stage']='TERMINAL_DIAGNOSTIC_'+arm;store.save()
    received=terminal[0,:,1:].float().numpy().transpose(1,0,2,3)
    for k,key in (('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])):
        row=store.data['terminal_diagnostics'][f'{arm}/{k}']
        try:
            result=backend.counted(store.count,'terminal_path_search',lambda:state.infer(received,key,np.ones((45,4),bool)))
            payload=backend.counted(store.count,'terminal_payload_read',lambda:method.payload_read(terminal,key,45))
            path=store.output/arm/(k+'.terminal_diagnostic.json.gz');dump_gzip(path,dict(inference=result,payload=payload))
            row.update(status='READ',path=str(path),sha256=backend.file_sha256(path),summary=result['summary'],payload=payload,source='terminal diagnostic only; never MP4 input')
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
    path=store.output/'blind_readouts.json';before=backend.file_sha256(path);raw=json.loads(path.read_text())
    store.data['stage']='POSTHOC_EVALUATION';store.save()
    for sid,search in raw['searches'].items():
        arm,view,key=sid.split('/');b=cfg['views'][view]['start'];g=(-b)%4;R=method.phase_spec(cfg['views'][view]['length'],0)['R']
        truth_tau=list(range((b+g)//4+1,(b+g)//4+1+R));rows=state.catalog(R)
        true_ids={i for i,row in enumerate(rows) if row['taus']==truth_tau and row['structurally_valid']}
        top=search.get('top',[]);canonical=search.get('canonical');feature=raw['phase_reads'][f'{arm}/{view}/{g}/{key}']
        store.count('oracle_saved_reference',False)
        oracle=(feature.get('payload') or {}).get('decoded_bits')
        if feature['status'] in ('COMPLETE','NO_ENERGY'):store.count('oracle_saved_reference',True)
        bits=search.get('canonical_phase_payload_diagnostic')
        for target,message in (('REGISTERED',cfg['message']),('WRONG_MESSAGE',cfg['wrong_message'])):
            truth=method.message_bits(message);row=store.data['evaluations'][f'{sid}/{target}']
            row.update(status='EVALUATED' if search['status'] in ('COMPLETE','NO_ENERGY') else 'MISSING_SEARCH',accepted_payload=False,state_path_accepted=False,
              bit_errors=None,exact_bits=None,registered_g=g,registered_tau=truth_tau,
              truth_in_top=any(t['g']==g and t['catalog_index'] in true_ids for t in top),
              canonical_phase_matches=canonical['g']==g if canonical else None,
              canonical_path_matches=canonical['path']['taus']==truth_tau and canonical['g']==g if canonical else None,
              canonical_event=canonical['path']['event_type'] if canonical else None,
              all_top_require_event=search.get('all_top_require_event'),
              diagnostic_bit_errors=sum(a!=b for a,b in zip(bits,truth)) if bits is not None else None,
              diagnostic_exact=bits==truth if bits is not None else None,oracle_payload=oracle,
              oracle_bit_errors=sum(a!=b for a,b in zip(oracle,truth)) if oracle is not None else None,
              expected_pilot_present=arm=='OVERLAP_MULTI' and key=='CORRECT',
              control_category='PILOT_PRESENT' if arm=='OVERLAP_MULTI' and key=='CORRECT' else 'OFF_UNWATERMARKED' if arm=='OFF' else 'PAYLOAD_ONLY' if arm=='PAYLOAD_MULTI' else 'PILOT_WRONG_KEY',
              claim='posthoc development diagnostic only; canonical fit is not physical edit detection')
        store.save()
    if backend.file_sha256(path)!=before:raise RuntimeError('posthoc modified blind evidence')
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
        for arm,reference in (('PAYLOAD_MULTI','OFF'),('OVERLAP_MULTI','OFF'),('OVERLAP_MULTI','PAYLOAD_MULTI')):
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
    expected=dict(generated=3,source_mp4=3,derived_mp4=15,normalized=60,phase_searches=120,phase_payload=120,path_costs=92016,searches=30,evaluated=60)
    done=store.data['counts']==expected and all(r['status']=='COMPLETE' for r in store.data['workers'].values())
    store.data['status']=halt or ('EXECUTION_COMPLETE' if done else 'INCOMPLETE');store.save()
    if not halt:quality_diagnostics(store)
    if backend.file_sha256(store.output/'blind_readouts.json')!=store.data['blind_sha256']:raise RuntimeError('quality modified blind evidence')
    store.data['stage']='FINISHED';store.save();print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])))
    if halt=='INTERRUPTED':raise SystemExit(130)
    if not done:raise SystemExit(1)


if __name__=='__main__':main()
