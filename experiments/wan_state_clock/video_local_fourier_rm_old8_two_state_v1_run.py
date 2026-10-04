"""User-run OLD8 finite two-state sequence candidate; blind reads then truth join."""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.metadata,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from main.tube_state import video_local_fourier_rm_old8_two_state_v1 as state
from main.tube_state import video_local_fourier_rm_old8_two_state_v1_control as method
from runtime.wan import video_local_fourier_rm_old8_two_state_v1 as backend
from runtime.wan import video_local_fourier_rm_same_raster as media
from runtime.wan import zero_mean_c1_yuv420_no_h264 as raw420
from runtime.wan import trajectory,vae as vae_adapter

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_old8_two_state_v1.json'
BASE_CONFIG=ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_v1.json'
MODULE='experiments.wan_state_clock.video_local_fourier_rm_old8_two_state_v1_run'
ARMS=method.ARMS;CHANNELS=('DIRECT_RGB8','RAW420','MP4');KEYS=('K0','K1');MODES=state.MODES
TARGETS=('REGISTERED','WRONG_MESSAGE');QUALITY_PAIRS=tuple((a,b) for i,a in enumerate(ARMS) for b in ARMS[:i])
FIXED=dict(source_cases=1,arms=6,channels=3,generated=6,generation_steps=300,controlled_steps=125,
 writer_sidecars=150,rasters=6,transport=18,underlying_observations=18,normalized=18,keys=2,keyed_projections=36,
 sequence_score_records=72,sequence_candidate_scores=288,absolute_local_candidate_costs=6336,
 difference_local_candidate_costs=6192,component_state_costs=12240,sequence_posthoc=72,
 local_posthoc=36,payload_reads=36,payload_evaluations=72,quality_pairs=45)
SIZES=dict(generation=6,writer_diagnostics=150,terminal_inputs=6,rasters=6,transport=18,normalized=18,
 observations=18,projections=36,sequence_reads=72,payload_reads=36,sequence_posthoc=72,
 local_posthoc=36,payload_evaluations=72,quality=45)


def sha(path): return media.file_sha256(path)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def gz(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0));os.replace(temp,path)
def environment():
    row=dict(python=sys.version,executable=sys.executable,packages={})
    for name in ('torch','torchvision','diffusers','transformers','accelerate','numpy','huggingface-hub'):
        try:row['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:row['packages'][name]=None
    return row
def observation_id(arm,channel):
    return 'obs_'+hashlib.sha256(('OLD8_TWO_STATE_V1\0'+arm+'\0'+channel).encode()).hexdigest()[:16]


def load_config():
    base=json.loads(BASE_CONFIG.read_text());candidate=json.loads(CONFIG.read_text());cfg={**base,**candidate}
    if cfg['fixed_denominator']!=FIXED or cfg['arms']!=list(ARMS) or cfg['channels']!=list(CHANNELS) or cfg['modes']!=list(MODES):raise ValueError('fixed roster mismatch')
    if cfg['control']!=base['control'] or cfg['model']!=base['model'] or cfg['generation']!=base['generation']:raise ValueError('OLD8 generation/control drift')
    if cfg['receiver']!=dict(g=0,R=44,alpha=state.PUBLIC.alpha,tie_atol=state.PUBLIC.tie_atol,threshold=None,accepted_payload=False,state_sequence_accepted=False):raise ValueError('receiver drift')
    return cfg


class Store:
    def __init__(self,output,*,create=False):
        self.output=Path(output);self.path=self.output/'result.json'
        if not create:self.data=json.loads(self.path.read_text());return
        cfg=load_config();self.output.mkdir(parents=True,exist_ok=False)
        self.data=dict(status='RUNNING',stage='INITIALIZE',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_worktree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),source_files={p:sha(ROOT/p) for p in cfg['source_files']},config_sha256=sha(CONFIG),fixed_denominator=FIXED,method_version=state.PUBLIC.method_version,model=cfg['model'],generation_protocol=cfg['generation'],control_protocol=cfg['control'],selection={kid:state.selection_receipt(key) for kid,key in [('K0',cfg['key']),('K1',cfg['wrong_key'])]},generation={},writer_diagnostics={},terminal_inputs={},rasters={},transport={},normalized={},observations={},projections={},sequence_reads={},payload_reads={},sequence_posthoc={},local_posthoc={},payload_evaluations={},quality={},calls={},workers={},failures=[],actual_generation_calls=False,actual_writer_update_calls=False,actual_vae_calls=False,actual_media_calls=False,evidence_ceiling=cfg['evidence_ceiling'])
        for arm in ARMS:
            folder=self.output/arm;terminal=str(folder/'terminal.pt');self.data['generation'][arm]=dict(status='PENDING',terminal_path=terminal,steps=[],receipt=None)
            self.data['terminal_inputs'][arm]=dict(status='PENDING',path=terminal);self.data['rasters'][arm]=dict(status='PENDING',path=str(folder/'source.rgb8'))
            for index in range(25,50):self.data['writer_diagnostics'][f'{arm}/{index}']=dict(status='PENDING',path=str(folder/'writer'/f'step_{index:02d}.npz'),index=index,role='writer sidecar; never receiver input')
            for channel in CHANNELS:
                aid=arm+'/'+channel;oid=observation_id(arm,channel);self.data['observations'][oid]=dict(arm=arm,channel=channel,truth_join_only=True)
                self.data['transport'][aid]=dict(status='PENDING',events={});self.data['normalized'][aid]=dict(status='PENDING',path=str(folder/(channel+'.pt')),g=0,R=44,observation_id=oid)
                for kid in KEYS:
                    sid=oid+'/'+kid;blind=self.output/'blind'/oid
                    self.data['projections'][sid]=dict(status='PENDING',path=str(blind/(kid+'.q.json.gz')),observation_id=oid,g=0,R=44)
                    self.data['payload_reads'][sid]=dict(status='PENDING',decoded_bits=None,observation_id=oid,truth_used=False)
                    self.data['local_posthoc'][sid]=dict(status='PENDING')
                    for mode in MODES:
                        mid=sid+'/'+mode;self.data['sequence_reads'][mid]=dict(status='PENDING',path=str(blind/(kid+'.'+mode+'.json.gz')),observation_id=oid,mode=mode,summary=None,accepted_payload=False,state_sequence_accepted=False)
                        self.data['sequence_posthoc'][mid]=dict(status='PENDING',accepted_payload=False,state_sequence_accepted=False)
                    for target in TARGETS:self.data['payload_evaluations'][sid+'/'+target]=dict(status='PENDING',accepted_payload=False)
        for channel in CHANNELS:
            for arm,reference in QUALITY_PAIRS:self.data['quality'][channel+'/'+arm+'_vs_'+reference]=dict(status='PENDING',threshold=None,diagnostic_only=True)
        self.save()
    def save(self):
        if any(len(self.data[k])!=v for k,v in SIZES.items()):raise ValueError('fixed roster changed')
        d=self.data;complete=lambda r:r['status'] in ('COMPLETE','NO_ENERGY')
        d['counts']=dict(generated=sum(r['status']=='COMPLETE' for r in d['generation'].values()),generation_steps=sum(len(r['steps']) for r in d['generation'].values()),controlled_steps=sum(s.get('enabled',False) for r in d['generation'].values() for s in r['steps']),writer_sidecars=sum(r['status']=='SAVED' for r in d['writer_diagnostics'].values()),rasters=sum(r['status']=='SAVED' for r in d['rasters'].values()),transport=sum(r['status']=='COMPLETE' for r in d['transport'].values()),normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),keyed_projections=sum(r['status']=='SAVED' for r in d['projections'].values()),sequence_score_records=sum(complete(r) for r in d['sequence_reads'].values()),sequence_candidate_scores=sum(r.get('sequence_scores',0) for r in d['sequence_reads'].values()),absolute_local_candidate_costs=sum(r.get('local_candidate_costs',0) for n,r in d['sequence_reads'].items() if n.endswith('/'+MODES[0])),difference_local_candidate_costs=sum(r.get('local_candidate_costs',0) for n,r in d['sequence_reads'].items() if n.endswith('/'+MODES[1])),component_state_costs=sum(r.get('component_state_costs',0) for r in d['sequence_reads'].values()),sequence_posthoc=sum(r['status'].startswith('EVALUATED') for r in d['sequence_posthoc'].values()),local_posthoc=sum(r['status'].startswith('EVALUATED') for r in d['local_posthoc'].values()),payload_reads=sum(r['status']=='READ' for r in d['payload_reads'].values()),payload_evaluations=sum(r['status']=='EVALUATED' for r in d['payload_evaluations'].values()),quality_pairs=sum(r['status']=='MEASURED' for r in d['quality'].values()))
        dump(self.path,d)
    def count(self,kind,done):
        row=self.data['calls'].setdefault(kind,dict(attempted=0,completed=0));row['completed' if done else 'attempted']+=1
        if kind.startswith(('generation_load','transformer_','native_step')):self.data['actual_generation_calls']=True
        if kind=='local_control':self.data['actual_writer_update_calls']=True
        if kind.startswith('vae_'):self.data['actual_vae_calls']=True
        if kind.startswith(('rgb_to_','raw420_to_','mp4_')):self.data['actual_media_calls']=True
        self.save()
    def call(self,kind,operation):self.count(kind,False);value=operation();self.count(kind,True);return value
    def failure(self,where,exc):self.data['failures'].append(dict(stage=where,error=f'{type(exc).__name__}: {exc}'));self.save()
    def blind_snapshot(self):
        dump(self.output/'blind_receiver_readouts.json',dict(projections=self.data['projections'],sequence_reads=self.data['sequence_reads'],payload_reads=self.data['payload_reads'],selection=self.data['selection'],truth_inputs=False,claim='received observations, key and public finite protocol only; no arm/message/writer/paired/truth inputs'))
    def event(self,aid,stage,row):self.data['transport'][aid]['events'][stage]=row;self.save()
    def writer_event(self,arm,index,arrays,metadata):
        row=self.data['writer_diagnostics'][f'{arm}/{index}'];row.update(metadata);self.count('writer_sidecar_save',False)
        try:
            if metadata['status']!='COMPLETE' or arrays is None:raise ValueError(metadata.get('error','writer observation unavailable'))
            if set(arrays)!=set(backend.DIAGNOSTIC_ARRAYS) or not all(v.dtype==np.float32 and v.shape==(45,4,4,8) and np.isfinite(v).all() for v in arrays.values()):raise ValueError('writer sidecar geometry/finiteness')
            path=Path(row['path']);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
            with temp.open('wb') as f:np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
            os.replace(temp,path);row.update(status='SAVED',sha256=sha(path),array_hashes={k:backend.file_array_sha256(v) for k,v in arrays.items()});self.count('writer_sidecar_save',True)
        except Exception as exc:row.update(status='FAILED',error=str(exc));self.failure(arm+'/WRITER/'+str(index),exc);raise
        self.save()


def generation_worker(store,cfg):
    import torch
    from runtime.wan.generation import prepare_generation
    device,dtype=backend.execution_device_dtype();store.data['stage']='GENERATION_LOAD';store.save()
    pipe,initial,prompt,negative,input_dtype=store.call('generation_load',lambda:prepare_generation(cfg,load_vae=False,device=device,model_dtype=dtype));pristine=copy.deepcopy(pipe.scheduler)
    initial_hash=trajectory.fingerprint(initial);history_hash=trajectory.fingerprint(vars(pristine));before=None
    store.data['generation_setup']=dict(device=device,transformer_dtype=str(dtype),state_control_cfg='float32',initial_sha256=initial_hash,pristine_history_sha256=history_hash,model=cfg['model'],seed=cfg['generation']['seed'],environment=environment());store.save()
    for arm in ARMS:
        row=store.data['generation'][arm];store.data['stage']='GENERATE_'+arm;row['status']='RUNNING';store.save()
        try:
            def record(step):row['steps'].append(step);store.save()
            terminal,receipt=backend.run_trajectory(pipe,initial,copy.deepcopy(pristine),prompt,negative,input_dtype,arm,cfg['key'],method.message_bits(cfg['message']),store.count,record,diagnostic=lambda index,arrays,metadata:store.writer_event(arm,index,arrays,metadata))
            if before is None:before=receipt['before_step25']
            if receipt['before_step25']!=before:raise ValueError('shared before25 mismatch')
            if trajectory.fingerprint(initial)!=initial_hash or trajectory.fingerprint(vars(pristine))!=history_hash:raise ValueError('shared source mutated')
            path=Path(row['terminal_path']);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(terminal,tmp);os.replace(tmp,path)
            receipt['control_norm_statistics']={k:dict(sum=sum(s.get(k,0.) for s in row['steps']),sum_squares=sum(s.get(k,0.)**2 for s in row['steps']),maximum=max(s.get(k,0.) for s in row['steps'])) for k in ('payload_delta_l2','pilot_delta_l2','merged_delta_l2')}
            row.update(status='COMPLETE',sha256=sha(path),receipt=receipt);store.save();del terminal
        except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure(store.data['stage'],exc)
    complete=[r for r in store.data['generation'].values() if r['status']=='COMPLETE'];store.data['before_step25_identity']=dict(status='MATCH' if len(complete)==6 else 'INCOMPLETE',checked_arms=len(complete));settle(store,'generation','generation unavailable')


def blind_receive(store,oid,normalized,key_roster):
    """Receiver boundary: observation, key roster and public protocol only."""
    mask=np.ones((44,4),bool);received=normalized[0,:,1:45].detach().float().cpu().numpy().transpose(1,0,2,3)
    for kid,key in key_roster:
        sid=oid+'/'+kid;qr=store.data['projections'][sid]
        try:
            q=store.call('projection_extract',lambda:state.extract(received,key,mask));gz(qr['path'],dict(q=q.tolist(),availability=mask.tolist(),g=0,R=44,observation_id=oid,truth_inputs=False));qr.update(status='SAVED',sha256=sha(qr['path']),shape=list(q.shape));store.save()
        except Exception as exc:qr.update(status='FAILED',error=str(exc));store.failure(sid+'/Q',exc);continue
        for mode in MODES:
            mid=sid+'/'+mode;row=store.data['sequence_reads'][mid]
            try:
                inf=store.call('sequence_score_'+mode.lower(),lambda:state.score(q,key,mask,mode))
                if inf['summary']['status'] not in ('COMPLETE','NO_ENERGY') or inf['counts']['sequence_scores']!=4:raise ValueError('four sequence scores incomplete')
                gz(row['path'],dict(mode=mode,g=0,R=44,observation_id=oid,truth_inputs=False,inference=inf,q_sha256=qr['sha256']));row.update(status=inf['summary']['status'],sha256=sha(row['path']),summary=inf['summary'],**inf['counts'])
            except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure(mid,exc)
            store.save();store.blind_snapshot()
        pr=store.data['payload_reads'][sid]
        try:pr.update(store.call('payload_read',lambda:method.payload_read(normalized,key,44)))
        except Exception as exc:pr.update(status='FAILED',error=str(exc));store.failure(sid+'/PAYLOAD',exc)
        store.save();store.blind_snapshot()


def media_worker(store,cfg):
    import torch
    from runtime.wan.generation import load_frozen_vae
    if not any(r['status']=='COMPLETE' for r in store.data['generation'].values()):settle(store,'media','no generated terminal');return
    store.data['stage']='VAE_LOAD';store.save();device='cuda' if torch.cuda.is_available() else 'cpu';frozen=store.call('vae_load',lambda:load_frozen_vae(cfg,device=device));store.data['vae_setup']=dict(model=cfg['model'],device=device,dtype=str(next(frozen.parameters()).dtype),environment=environment());store.save()
    for arm in ARMS:
        generated=store.data['generation'][arm];tr=store.data['terminal_inputs'][arm];rr=store.data['rasters'][arm]
        try:
            path=Path(generated['terminal_path'])
            if generated['status']!='COMPLETE' or sha(path)!=generated['sha256']:raise ValueError('saved terminal unavailable')
            terminal=torch.load(path,map_location='cpu',weights_only=True)
            if tuple(terminal.shape)!=(1,16,46,40,64) or not bool(terminal.isfinite().all()):raise ValueError('terminal geometry/finiteness')
            tr.update(status='LOADED',sha256=sha(path),shape=list(terminal.shape),dtype=str(terminal.dtype));store.save();rgb=store.call('vae_decode',lambda:vae_adapter.decode_normalized_latent(frozen,terminal.to(next(frozen.parameters()).device)));q8=vae_adapter.quantize_rgb8_no_codec(rgb);rr.update(store.call('raster_save',lambda:media.save_raster(q8,rr['path'])));store.save();del rgb,q8,terminal
        except Exception as exc:tr.update(status='FAILED',error=str(exc));store.failure(arm+'/DECODE',exc);continue
        for channel in CHANNELS:
            aid=arm+'/'+channel;oid=observation_id(arm,channel);transport=store.data['transport'][aid];norm=store.data['normalized'][aid]
            try:
                source=media.reopen_raster(rr['path'],rr['sha256']);transport.update(status='RUNNING',input_raster_path=rr['path'],input_raster_sha256=rr['sha256'],input_raster_bytes=rr['bytes']);store.save()
                if channel=='DIRECT_RGB8':received=source;transport['events']['rgb24']=dict(status='SAVED',path=rr['path'],sha256=rr['sha256'],bytes=rr['bytes'],source='identical saved raster')
                elif channel=='RAW420':received=raw420.roundtrip(source,store.output/arm/'channel.yuv420',store.output/arm/'raw420.rgb24',count=store.count,event=lambda name,row:store.event(aid,name,row))
                else:del source;received=media.mp4_roundtrip(rr['path'],rr['sha256'],store.output/arm/'channel.mp4',store.output/arm/'mp4.rgb24',count=store.count,event=lambda name,row:store.event(aid,name,row))
                if tuple(received.shape)!=media.SHAPE or received.dtype!=torch.uint8:raise ValueError('received RGB8 geometry')
                transport['status']='COMPLETE';store.save();normalized=store.call('vae_encode',lambda:vae_adapter.reencode_rgb24_readback(frozen,received.float()/255.))
                if tuple(normalized.shape)!=(1,16,46,40,64) or normalized.dtype!=torch.float32 or not bool(normalized.isfinite().all()):raise ValueError('normalized geometry/finiteness')
                path=Path(norm['path']);tmp=path.with_suffix('.tmp');torch.save(normalized.detach().cpu(),tmp);os.replace(tmp,path);norm.update(status='SAVED',sha256=sha(path),shape=list(normalized.shape),dtype=str(normalized.dtype),input_raster_sha256=rr['sha256'],channel_rgb_sha256=transport['events']['rgb24']['sha256']);store.save();store.blind_snapshot();blind_receive(store,oid,normalized.detach().cpu(),(('K0',cfg['key']),('K1',cfg['wrong_key'])))
            except Exception as exc:
                if transport['status']!='COMPLETE':transport.update(status='FAILED',error=str(exc))
                if norm['status']=='PENDING':norm.update(status='FAILED',error=str(exc))
                store.failure(aid,exc)
            finally:
                if 'source' in locals():del source
                if 'received' in locals():del received
                if 'normalized' in locals():del normalized
    settle(store,'media','terminal/channel/read unavailable')


def settle(store,phase,reason):
    groups=('generation','writer_diagnostics') if phase=='generation' else ('terminal_inputs','rasters','transport','normalized','projections','sequence_reads','payload_reads')
    for group in groups:
        for row in store.data[group].values():
            if row['status'] in ('PENDING','RUNNING'):row.update(status='NOT_COMPLETED',error=reason)
    store.save();store.blind_snapshot()


def evaluate(store,cfg):
    store.blind_snapshot();blind=store.output/'blind_receiver_readouts.json';sealed=sha(blind);snapshot=json.loads(blind.read_text())
    for mid,mr in snapshot['sequence_reads'].items():
        out=store.data['sequence_posthoc'][mid];parts=mid.split('/');oid,kid,mode=parts;truth=store.data['observations'][oid];registered=method.STATE_ARMS.get(truth['arm'])
        if mr['status'] not in ('COMPLETE','NO_ENERGY'):out.update(status='MISSING_READ',error=mr.get('error'));continue
        inf=json.loads(gzip.decompress(Path(mr['path']).read_bytes()))['inference'];scores=inf['sequence_scores']
        if kid!='K0' or registered is None:out.update(status='EVALUATED_CONTROL',registered_sequence=None,key_role='WRONG_KEY' if kid=='K1' else 'REGISTERED_KEY',arm_role='CONTROL' if registered is None else 'MARKED',scores=scores,top=inf['summary']['top']);continue
        value=scores[registered];others=[v for k,v in scores.items() if k!=registered]
        out.update(status='EVALUATED_TRUTH',registered_sequence=registered,key_role=kid,true_cost=value,true_rank=1+sum(v<value-state.PUBLIC.tie_atol for v in others),true_delta=min(others)-value,truth_in_top=registered in inf['summary']['top'],unique_truth=inf['summary']['top']==[registered],top=inf['summary']['top'])
    for sid,out in store.data['local_posthoc'].items():
        oid,kid=sid.split('/');truth=store.data['observations'][oid];registered=method.STATE_ARMS.get(truth['arm']);mr=snapshot['sequence_reads'][sid+'/'+MODES[0]]
        if mr['status'] not in ('COMPLETE','NO_ENERGY'):out.update(status='MISSING_READ',error=mr.get('error'));continue
        if kid!='K0' or registered is None:out.update(status='EVALUATED_CONTROL',rows=[]);continue
        inf=json.loads(gzip.decompress(Path(mr['path']).read_bytes()))['inference'];labels=state.start_labels(registered);rows=[]
        for row in inf['component_evidence']:
            if row['status']!='SCORED':rows.append({**row,'truth_label':None});continue
            label=labels[row['source_start']-1];cost=row['costs'][label];other=row['costs']['B' if label=='A' else 'A'];rows.append(dict(received_regular_index=row['received_regular_index'],age=row['age'],source_start=row['source_start'],status='EVALUATED',truth_label=label,true_cost=cost,true_delta=other-cost,truth_in_top=label in row['top'],unique_truth=row['top']==[label],top=row['top']))
        out.update(status='EVALUATED_TRUTH',rows=rows,scored=sum(r['status']=='EVALUATED' for r in rows),truth_in_top=sum(r.get('truth_in_top',False) for r in rows),unique_truth=sum(r.get('unique_truth',False) for r in rows))
    for sid,pr in snapshot['payload_reads'].items():
        for target,message in [('REGISTERED',cfg['message']),('WRONG_MESSAGE',cfg['wrong_message'])]:
            bits=pr['decoded_bits'];out=store.data['payload_evaluations'][sid+'/'+target];truth_bits=method.message_bits(message);out.update(status='EVALUATED' if pr['status']=='READ' else 'MISSING_READ',bit_errors=sum(a!=b for a,b in zip(bits,truth_bits)) if bits is not None else None,exact_bits=bits==truth_bits if bits is not None else None,accepted_payload=False)
    if sha(blind)!=sealed:raise ValueError('truth join changed sealed blind readouts')
    store.data['blind_receiver_sha256']=sealed;store.save()


def quality_diagnostics(store):
    shape=media.SHAPE
    for channel in CHANNELS:
        for arm,reference in QUALITY_PAIRS:
            row=store.data['quality'][channel+'/'+arm+'_vs_'+reference];left=store.data['transport'][arm+'/'+channel];right=store.data['transport'][reference+'/'+channel]
            if left['status']!='COMPLETE' or right['status']!='COMPLETE':row.update(status='MISSING_CHANNEL');store.save();continue
            try:
                def compute():
                    a,b=left['events']['rgb24'],right['events']['rgb24'];x=np.memmap(a['path'],dtype=np.uint8,mode='r',shape=shape);y=np.memmap(b['path'],dtype=np.uint8,mode='r',shape=shape);squared=0.
                    for start in range(0,shape[0],4):delta=(np.asarray(x[start:start+4],dtype=np.float64)-np.asarray(y[start:start+4],dtype=np.float64))/255.;squared+=float(np.square(delta).sum())
                    mse=squared/int(np.prod(shape));del x,y;return dict(status='MEASURED',rgb_rmse=math.sqrt(mse),psnr_db=-10*math.log10(mse) if mse else None,identical_pixels=mse==0,source='persisted RGB24 diagnostic')
                row.update(store.call('quality_compute',compute))
            except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure('QUALITY/'+channel+'/'+arm,exc)
            store.save()


def run_worker_phase(store,phase):
    command=[sys.executable,'-u','-m',MODULE,'--output',str(store.output),'--worker',phase];start=time.perf_counter();child=None;code=None;error=None;cleanup_error=None;interrupted=False
    try:
        with (store.output/(phase+'.log')).open('w') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
            code=child.wait()
    except BaseException as exc:
        error=f'{type(exc).__name__}: {exc}';interrupted=not isinstance(exc,Exception)
        if child is not None:
            try:
                if child.poll() is None:
                    try:child.terminate()
                    except ProcessLookupError:pass
                try:code=child.wait(timeout=10)
                except subprocess.TimeoutExpired:child.kill();code=child.wait(timeout=10)
            except BaseException as cleanup:cleanup_error=str(cleanup)
    finally:
        if child is not None and child.stdout is not None:
            try:child.stdout.close()
            except BaseException as cleanup:cleanup_error=cleanup_error or str(cleanup)
    store=Store(store.output);ok=code==0 and error is None and cleanup_error is None;store.data['workers'][phase]=dict(status='COMPLETE' if ok else 'FAILED',command=command,returncode=code,error=error,cleanup_error=cleanup_error,elapsed_seconds=time.perf_counter()-start)
    if not ok:store.failure('WORKER_'+phase,RuntimeError(error or cleanup_error or f'child exit {code}'))
    settle(store,phase,error or 'worker did not complete');return store,interrupted


def finish(store,cfg):
    expected={k:v for k,v in FIXED.items() if k not in ('source_cases','arms','channels','underlying_observations','keys')}
    calls={k:dict(expected=n,actual=store.data['calls'].get(k,dict(attempted=0,completed=0)),match=store.data['calls'].get(k)==dict(attempted=n,completed=n)) for k,n in cfg['planned_calls'].items()};store.data['call_integrity']=dict(status='MATCH' if all(v['match'] for v in calls.values()) else 'INCOMPLETE',rows=calls)
    writer_ok=all(r.get('receipt',{}).get('writer_diagnostics',{}).get('last_z_post_matches_terminal') is True for r in store.data['generation'].values() if r['status']=='COMPLETE')
    done=not store.data['failures'] and store.data['counts']==expected and store.data['call_integrity']['status']=='MATCH' and writer_ok and all(store.data['workers'].get(p,{}).get('status')=='COMPLETE' for p in ('generation','media'))
    store.data.update(status='EXECUTION_COMPLETE' if done else 'INCOMPLETE',stage='FINISHED',science_status='UNCALIBRATED_OLD8_TWO_STATE_CANDIDATE_NO_AUTOMATIC_PASS');store.save();print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])));return done


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',choices=('generation','media'));args=p.parse_args();cfg=load_config()
    if args.worker:
        store=Store(args.output)
        try:(generation_worker if args.worker=='generation' else media_worker)(store,cfg)
        except Exception as exc:store.failure(store.data['stage'],exc);settle(store,args.worker,str(exc));raise
        return
    store=Store(args.output,create=True);interrupted=False
    for phase in ('generation','media'):
        store,interrupted=run_worker_phase(store,phase)
        if interrupted:break
    settle(store,'generation','generation incomplete');settle(store,'media','media incomplete');evaluate(store,cfg);quality_diagnostics(store);done=finish(store,cfg)
    if interrupted:raise SystemExit(130)
    if not done:raise SystemExit(1)

if __name__=='__main__':main()
