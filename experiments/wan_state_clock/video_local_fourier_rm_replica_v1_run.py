"""User-run fixed two-spatial-copy OLD8 candidate; separate saved-terminal diagnostics."""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.metadata,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from main.tube_state import video_local_fourier_rm_replica_v1_control as method
from main.tube_state import video_local_fourier_rm_replica_v1_state as replica
from experiments.wan_state_clock import video_local_fourier_rm_difference_receiver as receiver
from runtime.wan import video_local_fourier_rm_replica_v1 as backend
from runtime.wan import video_local_fourier_rm_same_raster as media
from runtime.wan import zero_mean_c1_yuv420_no_h264 as raw420
from runtime.wan import trajectory,vae as vae_adapter
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_replica_v1.json'
MODULE='experiments.wan_state_clock.video_local_fourier_rm_replica_v1_run'
ARMS=method.ARMS;CHANNELS=('DIRECT_RGB8','RAW420','MP4');FAMILIES=('OLD8','REPLICA8');KEYS=('CORRECT','WRONG');TARGETS=('REGISTERED','WRONG_MESSAGE');MODES=receiver.MODES
FIXED=dict(source_cases=1,arms=4,channels=3,reader_families=2,generated=4,generation_steps=200,controlled_steps=75,writer_sidecars=100,underlying_observations=12,normalized=12,keys=2,keyed_projections=48,path_score_records=96,valid_paths=174,catalog_per_record=3915,path_costs=16704,difference_edge_costs=272448,absolute_local_state_costs=95040,path_posthoc=96,difference_edge_posthoc=48,payload_reads=24,payload_evaluations=48,catalog_slots=375840,structural_exclusions=359136,quality_pairs=18,group_observation_records=24)
QUALITY_PAIRS=tuple((a,b) for i,a in enumerate(ARMS) for b in ARMS[:i])
TERMINAL_FIXED=dict(inputs=4,group_observation_records=8,keyed_projections=16,path_score_records=32,path_costs=5568,difference_edge_costs=90816,absolute_local_state_costs=31680,path_posthoc=32,difference_edge_posthoc=16,catalog_slots=125280,structural_exclusions=119712)
TERMINAL_SIZES=dict(inputs=4,group_observations=8,projections=16,mode_reads=32,path_posthoc=32,edge_posthoc=16)
REPLICA_CONTROL={'group_count': 2, 'groups': [[[8, 16, 12, 20], [8, 16, 44, 52], [28, 36, 12, 20], [28, 36, 44, 52]], [[8, 16, 20, 28], [8, 16, 52, 60], [28, 36, 20, 28], [28, 36, 52, 60]]], 'logical_basis': 'unchanged OLD8 U_i and key domains copied to both physical groups', 'group_target_alpha': 0.009476225544736292, 'joint_target_l2': 1.0, 'active_coefficients': 11136, 'boundary_coefficients': 384, 'mean_MSE_denominator': 11136, 'eta': 1392.0, 'budget_source_arm': 'STATE_OLD_MULTI', 'budget_target_arm': 'STATE_REPLICA_MULTI', 'budget_steps': [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49], 'budget_quantity': 'same-run OLD applied pilot full physical L2 over45 windows', 'norm_rule': 'joint raw replica direction scaled to OLD norm; positive scale may exceed1', 'zero_OLD_norm': 'zero applied replica pilot', 'zero_new_raw_positive_target': 'explicit failure; no epsilon or fabricated direction', 'group_actual_energy_matched': False, 'total_merged_budget_matched': False, 'read_scale': 1.4142135623730951, 'fusion': 'fixed equal mean of scaled A/B on public support intersection; no single-copy fallback', 'replicas_are_independent_votes': False, 'time_dependent_payload': False}
SIZES=dict(group_observations=24,generation=4,writer_diagnostics=100,terminal_inputs=4,rasters=4,transport=12,normalized=12,projections=48,mode_reads=96,payload_reads=24,path_posthoc=96,edge_posthoc=48,payload_evaluations=48,quality=18)

def sha(path):return media.file_sha256(path)
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def gz(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0));os.replace(temp,path)
def environment():
    row=dict(python=sys.version,executable=sys.executable,packages={})
    for name in ('torch','torchvision','diffusers','transformers','accelerate','numpy','huggingface-hub'):
        try:row['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:row['packages'][name]=None
    return row

def load_config():
    cfg=json.loads(CONFIG.read_text());old=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_v1.json').read_text())
    if any(cfg[k]!=old[k] for k in ('model','generation','control','key','wrong_key','message','wrong_message','precision')):raise ValueError('frozen generation/payload/OLD writer protocol mismatch')
    if cfg['fixed_denominator']!=FIXED or cfg['arms']!=list(ARMS) or cfg['channels']!=list(CHANNELS) or cfg['reader_families']!=list(FAMILIES) or cfg['modes']!=list(MODES):raise ValueError('fixed roster mismatch')
    if cfg['carrier_old']!=old['carrier'] or cfg['replica_control']!=REPLICA_CONTROL or REPLICA_CONTROL['groups']!=[[list(b) for b in g] for g in replica.GROUPS] or REPLICA_CONTROL['group_target_alpha']!=replica.GROUP_ALPHA:raise ValueError('fixed two-copy recipe mismatch')
    if cfg['receiver']!=dict(g=0,R=44,alpha=receiver.PUBLIC.alpha,tie_atol=1e-12,threshold=None,accepted_payload=False,state_path_accepted=False):raise ValueError('frozen receiver mismatch')
    if cfg['method_version']!=replica.VERSION or cfg['terminal_diagnostic_fixed']!=TERMINAL_FIXED or cfg['planned_terminal_diagnostic_calls']!=dict(terminal_tensor_load=4,group_projection_extract=8,projection_extract=16,absolute_control=16,difference_infer=16):raise ValueError('separate terminal diagnostic roster mismatch')
    return cfg


def initial_terminal_diagnostics(output):
    t=dict(fixed_denominator=TERMINAL_FIXED,inputs={},group_observations={},projections={},mode_reads={},path_posthoc={},edge_posthoc={},calls={},counts={},receiver_sha256={},truth_inputs=False,claim='writer-only saved-terminal diagnostic; never primary blind video evidence')
    for arm in ARMS:
        t['inputs'][arm]=dict(status='PENDING',path=str(Path(output)/arm/'terminal.pt'))
        for kid in KEYS:t['group_observations'][arm+'/'+kid]=dict(status='PENDING',path=str(Path(output)/arm/('terminal.'+kid+'.groups.json.gz')),g=0,R=44)
        for family in FAMILIES:
            for kid in KEYS:
                sid=arm+'/'+family+'/'+kid;base=Path(output)/arm/('terminal.'+family+'.'+kid)
                t['projections'][sid]=dict(status='PENDING',path=str(base)+'.q.json.gz',reader_family=family,g=0,R=44);t['edge_posthoc'][sid]=dict(status='PENDING')
                for mode in MODES:
                    mid=sid+'/'+mode;t['mode_reads'][mid]=dict(status='PENDING',path=str(base)+'.'+mode+'.json.gz',reader_family=family,mode=mode,summary=None,accepted_payload=False,state_path_accepted=False);t['path_posthoc'][mid]=dict(status='PENDING',accepted_payload=False,state_path_accepted=False)
    return t

class Store:
    def __init__(self,output,*,create=False):
        self.output=Path(output);self.path=self.output/'result.json'
        if not create:self.data=json.loads(self.path.read_text())
        else:
            cfg=load_config();self.output.mkdir(parents=True,exist_ok=False)
            self.data=dict(status='RUNNING',stage='INITIALIZE',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_worktree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),source_files={p:sha(ROOT/p) for p in cfg['source_files']},config_sha256=sha(CONFIG),fixed_denominator=FIXED,method_version=replica.VERSION,model=cfg['model'],generation_protocol=cfg['generation'],control_protocol=cfg['control'],replica_control_protocol=cfg['replica_control'],environment=environment(),generation={},writer_diagnostics={},group_observations={},terminal_inputs={},rasters={},transport={},normalized={},projections={},mode_reads={},payload_reads={},path_posthoc={},edge_posthoc={},payload_evaluations={},quality={},calls={},workers={},failures=[],shared={},public_readers={},actual_generation_calls=False,actual_writer_update_calls=False,actual_vae_calls=False,actual_media_calls=False,evidence_ceiling=cfg['evidence_ceiling'])
            self.data.update(state_budget_schedule=dict(status='PENDING',source_arm='STATE_OLD_MULTI',target_arm='STATE_REPLICA_MULTI',role='writer only; never receiver input',steps={str(i):dict(status='PENDING',target_l2=None) for i in range(25,50)}),terminal_diagnostics=initial_terminal_diagnostics(self.output))
            for arm in ARMS:
                folder=self.output/arm;terminal=str(folder/'terminal.pt');self.data['generation'][arm]=dict(status='PENDING',terminal_path=terminal,steps=[],receipt=None)
                self.data['terminal_inputs'][arm]=dict(status='PENDING',path=terminal);self.data['rasters'][arm]=dict(status='PENDING',path=str(folder/'source.rgb8'))
                for index in range(25,50):self.data['writer_diagnostics'][f'{arm}/{index}']=dict(status='PENDING',path=str(folder/'writer'/f'step_{index:02d}.npz'),index=index,role='writer sidecar; never receiver input')
                for channel in CHANNELS:
                    aid=arm+'/'+channel;self.data['transport'][aid]=dict(status='PENDING',events={});self.data['normalized'][aid]=dict(status='PENDING',path=str(folder/(channel+'.pt')),g=0,R=44)
                    for kid in KEYS:
                        psid=aid+'/'+kid;self.data['group_observations'][psid]=dict(status='PENDING',path=str(folder/(channel+'.'+kid+'.groups.json.gz')),g=0,R=44);self.data['payload_reads'][psid]=dict(status='PENDING',decoded_bits=None,source='NEW_REPLICA_CHANNEL_READ',truth_used=False)
                        for target in TARGETS:self.data['payload_evaluations'][psid+'/'+target]=dict(status='PENDING',bit_errors=None,accepted_payload=False)
                        for family in FAMILIES:
                            sid=aid+'/'+family+'/'+kid;self.data['projections'][sid]=dict(status='PENDING',path=str(folder/(channel+'.'+family+'.'+kid+'.q.json.gz')),g=0,R=44,reader_family=family);self.data['edge_posthoc'][sid]=dict(status='PENDING')
                            for mode in MODES:
                                mid=sid+'/'+mode;self.data['mode_reads'][mid]=dict(status='PENDING',path=str(folder/(channel+'.'+family+'.'+kid+'.'+mode+'.json.gz')),g=0,R=44,reader_family=family,mode=mode,summary=None,accepted_payload=False,state_path_accepted=False);self.data['path_posthoc'][mid]=dict(status='PENDING',accepted_payload=False,state_path_accepted=False)
            for channel in CHANNELS:
                for arm,reference in QUALITY_PAIRS:self.data['quality'][channel+'/'+arm+'_vs_'+reference]=dict(status='PENDING',threshold=None,diagnostic_only=True)
            catalog=self.output/'catalog_R44.json.gz';gz(catalog,dict(rows=receiver.frozen.catalog(44)))
            for kid,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
                mask=np.ones((44,4),bool);ab=receiver.frozen.family_receipt(key,44,mask);df=receiver.difference_family(key,mask)
                dpayload=dict(availability=mask.tolist(),edge_availability=df['edge_availability'].tolist(),available_dimensions=df['available_dimensions'],valid_catalog_indices=df['valid_catalog_indices'],classes=df['classes'])
                ap=self.output/('family_absolute_'+kid+'.json.gz');dp=self.output/('family_difference_'+kid+'.json.gz');gz(ap,ab);gz(dp,dpayload)
                self.data['shared'][kid]=dict(catalog_path=str(catalog),catalog_sha256=sha(catalog),absolute_family_path=str(ap),absolute_family_sha256=sha(ap),difference_family_path=str(dp),difference_family_sha256=sha(dp),valid_count=174,catalog_count=3915,scope='identical temporal templates for both spatial reader families; no independent-sample claim')
                self.data['public_readers'][kid]={family:dict(blocks=[list(b) for b in carrier.PUBLIC.blocks],basis_layout=carrier.basis_layout(key),carrier_version=carrier.PUBLIC.method_version,blocks_are='logical A blocks; see groups/blocks_all for the full physical reader support' if family=='REPLICA8' else 'all physical OLD8 blocks',groups=[[list(b) for b in g] for g in replica.GROUPS] if family=='REPLICA8' else [[list(b) for b in receiver.frozen.PUBLIC.blocks]],blocks_all=[list(b) for g in replica.GROUPS for b in g] if family=='REPLICA8' else [list(b) for b in receiver.frozen.PUBLIC.blocks]) for family,carrier in [('OLD8',receiver.frozen),('REPLICA8',replica)]}
        self.save()
    def save(self):
        d=self.data
        if any(len(d[k])!=v for k,v in SIZES.items()):raise ValueError('fixed roster changed')
        complete=lambda r:r['status'] in ('COMPLETE','NO_ENERGY')
        d['counts']=dict(group_observation_records=sum(r['status']=='SAVED' for r in d['group_observations'].values()),generated=sum(r['status']=='COMPLETE' for r in d['generation'].values()),generation_steps=sum(len(r['steps']) for r in d['generation'].values()),controlled_steps=sum(s.get('enabled',False) for r in d['generation'].values() for s in r['steps']),writer_sidecars=sum(r['status']=='SAVED' for r in d['writer_diagnostics'].values()),rasters=sum(r['status']=='SAVED' for r in d['rasters'].values()),transport=sum(r['status']=='COMPLETE' for r in d['transport'].values()),normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),keyed_projections=sum(r['status']=='SAVED' for r in d['projections'].values()),path_score_records=sum(complete(r) for r in d['mode_reads'].values()),path_costs=sum(r.get('scored',0) for r in d['mode_reads'].values()),difference_edge_costs=sum(r.get('edge_scored',0) for r in d['mode_reads'].values()),absolute_local_state_costs=sum(r.get('absolute_local_scored',0) for r in d['mode_reads'].values()),payload_reads=sum(r['status']=='READ' for r in d['payload_reads'].values()),path_posthoc=sum(r['status']=='EVALUATED' for r in d['path_posthoc'].values()),difference_edge_posthoc=sum(r['status']=='EVALUATED' for r in d['edge_posthoc'].values()),payload_evaluations=sum(r['status']=='EVALUATED' for r in d['payload_evaluations'].values()),quality_pairs=sum(r['status']=='MEASURED' for r in d['quality'].values()))
        t=d['terminal_diagnostics']
        if any(len(t[k])!=v for k,v in TERMINAL_SIZES.items()) or len(d['state_budget_schedule']['steps'])!=25:raise ValueError('independent diagnostic roster changed')
        t['counts']=dict(group_observation_records=sum(r['status']=='SAVED' for r in t['group_observations'].values()),inputs=sum(r['status']=='LOADED' for r in t['inputs'].values()),keyed_projections=sum(r['status']=='SAVED' for r in t['projections'].values()),path_score_records=sum(complete(r) for r in t['mode_reads'].values()),path_costs=sum(r.get('scored',0) for r in t['mode_reads'].values()),difference_edge_costs=sum(r.get('edge_scored',0) for r in t['mode_reads'].values()),absolute_local_state_costs=sum(r.get('absolute_local_scored',0) for r in t['mode_reads'].values()),path_posthoc=sum(r['status']=='EVALUATED' for r in t['path_posthoc'].values()),difference_edge_posthoc=sum(r['status']=='EVALUATED' for r in t['edge_posthoc'].values()))
        dump(self.path,d)
    def diagnostic_call(self,group,kind,operation):
        if group!='terminal':raise ValueError('only separated terminal diagnostic calls')
        calls=self.data['terminal_diagnostics']['calls'];row=calls.setdefault(kind,dict(attempted=0,completed=0));row['attempted']+=1;self.save();value=operation();row['completed']+=1;self.save();return value
    def count(self,kind,done):
        row=self.data['calls'].setdefault(kind,dict(attempted=0,completed=0));row['completed' if done else 'attempted']+=1
        if kind.startswith(('generation_load','transformer_','native_step')):self.data['actual_generation_calls']=True
        if kind=='local_control':self.data['actual_writer_update_calls']=True
        if kind.startswith('vae_'):self.data['actual_vae_calls']=True
        if kind.startswith(('rgb_to_','raw420_to_','mp4_')):self.data['actual_media_calls']=True
        self.save()
    def call(self,kind,operation):self.count(kind,False);value=operation();self.count(kind,True);return value
    def failure(self,where,exc):self.data['failures'].append(dict(stage=where,error=f'{type(exc).__name__}: {exc}'));self.save()
    def snapshots(self):
        dump(self.output/'receiver_readouts.json',dict(normalized=self.data['normalized'],group_observations=self.data['group_observations'],projections=self.data['projections'],mode_reads=self.data['mode_reads'],shared=self.data['shared'],public_readers=self.data['public_readers'],truth_inputs=False,claim='both public reader families on every observation; no writer sidecar or arm-conditioned reader choice'))
        dump(self.output/'payload_readouts.json',dict(reads=self.data['payload_reads'],truth_inputs=False,claim='new repeated payload reads once per observation/key; never path/family selectors'))
    def event(self,aid,stage,row):self.data['transport'][aid]['events'][stage]=row;self.save()
    def writer_event(self,arm,index,arrays,metadata):
        row=self.data['writer_diagnostics'][f'{arm}/{index}'];row.update(metadata);self.count('writer_sidecar_save',False)
        try:
            if metadata['status']!='COMPLETE' or arrays is None:raise ValueError(metadata.get('error','writer observation unavailable'))
            if set(arrays)!=set(backend.DIAGNOSTIC_ARRAYS) or not all(v.dtype==np.float32 and v.shape==((2,45,4,4,8) if k.endswith('_groups') else (45,4,4,8)) and np.isfinite(v).all() for k,v in arrays.items()):raise ValueError('writer sidecar geometry/finiteness')
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
            schedule=None
            if arm=='STATE_REPLICA_MULTI':
                if store.data['state_budget_schedule']['status']!='SAVED':raise ValueError('same-run OLD norm schedule unavailable')
                schedule=store.call('state_budget_schedule_load',lambda:read_state_budget_schedule(store))
            def record(step):row['steps'].append(step);store.save()
            terminal,receipt=backend.run_trajectory(pipe,initial,copy.deepcopy(pristine),prompt,negative,input_dtype,arm,cfg['key'],method.message_bits(cfg['message']),store.count,record,diagnostic=lambda index,arrays,metadata:store.writer_event(arm,index,arrays,metadata),matched_norm_schedule=schedule)
            if before is None:before=receipt['before_step25']
            if receipt['before_step25']!=before:raise ValueError('shared before25 z/history/branches mismatch')
            if trajectory.fingerprint(initial)!=initial_hash or trajectory.fingerprint(vars(pristine))!=history_hash:raise ValueError('shared initial/native scheduler mutated')
            path=Path(row['terminal_path']);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(terminal,tmp);os.replace(tmp,path)
            receipt['control_norm_statistics']={k:dict(sum=sum(s.get(k,0.) for s in row['steps']),sum_squares=sum(s.get(k,0.)**2 for s in row['steps']),maximum=max(s.get(k,0.) for s in row['steps'])) for k in ('payload_delta_l2','pilot_delta_l2','merged_delta_l2')}
            row.update(status='COMPLETE',sha256=sha(path),receipt=receipt);store.save();del terminal
            if arm=='STATE_OLD_MULTI':
                try:save_state_budget_schedule(store)
                except Exception as exc:
                    store.data['state_budget_schedule'].update(status='FAILED',error=str(exc));store.failure('STATE_BUDGET_SCHEDULE',exc)
        except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure(store.data['stage'],exc)
    complete=[r for r in store.data['generation'].values() if r['status']=='COMPLETE'];store.data['before_step25_identity']=dict(status='MATCH' if len(complete)==4 else 'INCOMPLETE',checked_arms=len(complete));settle(store,'generation','unavailable generation or writer sidecar')


def save_state_budget_schedule(store):
    budget=store.data['state_budget_schedule'];old=store.data['generation']['STATE_OLD_MULTI'];steps=[s for s in old['steps'] if s['index']>=25]
    if old['status']!='COMPLETE' or [s['index'] for s in steps]!=list(range(25,50)):raise ValueError('complete same-run OLD trajectory required for budget')
    for s in steps:
        value=float(s['pilot_delta_l2'])
        if not math.isfinite(value) or value<0:raise ValueError('invalid OLD physical norm schedule')
        budget['steps'][str(s['index'])]=dict(status='RECORDED',target_l2=value,source_step_receipt_sha256=hashlib.sha256(json.dumps(s,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),source_sidecar_sha256=store.data['writer_diagnostics']['STATE_OLD_MULTI/'+str(s['index'])].get('sha256'))
    budget.update(status='SAVED',source_terminal_sha256=old['sha256'],quantity=REPLICA_CONTROL['budget_quantity'],scale_may_exceed_one=True,total_merged_budget_matched=False)
    path=store.output/'state_budget_schedule.json';store.call('state_budget_schedule_save',lambda:dump(path,budget));budget.update(path=str(path),sha256=sha(path));store.save()


def read_state_budget_schedule(store):
    """Writer-only saved norm receipts; never routed into a receiver or scorer."""
    budget=store.data['state_budget_schedule'];old=store.data['generation']['STATE_OLD_MULTI'];path=Path(budget['path'])
    if budget['status']!='SAVED' or not path.is_file() or sha(path)!=budget['sha256']:raise ValueError('saved same-run OLD norm budget missing/changed')
    saved=json.loads(path.read_text())
    if saved['status']!='SAVED' or saved['source_arm']!='STATE_OLD_MULTI' or saved['target_arm']!='STATE_REPLICA_MULTI' or set(saved['steps'])!={str(i) for i in range(25,50)} or saved['source_terminal_sha256']!=old.get('sha256'):raise ValueError('saved OLD budget identity/index mismatch')
    indexed={s['index']:s for s in old['steps']};schedule={}
    for key,row in saved['steps'].items():
        index=int(key);step=indexed[index];value=float(row['target_l2']);fingerprint=hashlib.sha256(json.dumps(step,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if row['status']!='RECORDED' or not math.isfinite(value) or value<0 or value!=step['pilot_delta_l2'] or row['source_step_receipt_sha256']!=fingerprint or row['source_sidecar_sha256']!=store.data['writer_diagnostics']['STATE_OLD_MULTI/'+key].get('sha256'):raise ValueError('saved OLD budget step/norm receipt mismatch')
        schedule[index]=value
    return schedule


def save_group_observation(store,row,received,key,mask,*,terminal=False):
    operation=lambda:replica.extract_groups(received,key,np.stack([mask,mask]))
    result=store.diagnostic_call('terminal','group_projection_extract',operation) if terminal else store.call('group_projection_extract',operation)
    value={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in result.items()}
    value.update(g=0,R=44,truth_inputs=False,groups=[[list(b) for b in g] for g in replica.GROUPS],role='public received A/B projections and fixed fusion; no writer budget or message input')
    gz(row['path'],value);row.update(status='SAVED',sha256=sha(row['path']),raw_shape=list(result['raw_groups'].shape),received_regular_indices=result['received_regular_indices'].tolist(),group_availability=result['group_availability'].tolist(),availability=result['availability'].tolist());store.save();return result


def public_projection(groups,family):
    if family=='OLD8':return groups['raw_groups'][0],groups['group_availability'][0]
    if family=='REPLICA8':return groups['fused'],groups['availability']
    raise ValueError('fixed public family required')


def receive_normalized(store,aid,normalized,cfg):
    mask=np.ones((44,4),bool);received=normalized[0,:,1:45].detach().float().cpu().numpy().transpose(1,0,2,3)
    for kid,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
        gid=aid+'/'+kid;gr=store.data['group_observations'][gid];groups=None
        try:groups=save_group_observation(store,gr,received,key,mask)
        except Exception as exc:gr.update(status='FAILED',error=str(exc));store.failure(gid+'/GROUP_Q',exc)
        for family in FAMILIES:
            sid=aid+'/'+family+'/'+kid;qr=store.data['projections'][sid]
            try:
                if groups is None:raise ValueError('received group projections unavailable')
                q,available=store.call('projection_extract',lambda:public_projection(groups,family));gz(qr['path'],dict(q=q.tolist(),availability=available.tolist(),group_input_path=gr['path'],group_input_sha256=gr['sha256'],g=0,R=44,reader_family=family,truth_inputs=False));qr.update(status='SAVED',sha256=sha(qr['path']),shape=list(q.shape));store.save()
            except Exception as exc:qr.update(status='FAILED',error=str(exc));store.failure(sid+'/Q',exc);continue
            for mode in MODES:
                mid=sid+'/'+mode;row=store.data['mode_reads'][mid]
                try:
                    fn=receiver.absolute_control if mode==MODES[0] else receiver.infer_difference;inf=store.call('absolute_control' if mode==MODES[0] else 'difference_infer',lambda:fn(q,key,available))
                    if inf['summary']['status'] not in ('COMPLETE','NO_ENERGY') or inf['counts']['scored']!=174:raise ValueError('fixed path costs incomplete')
                    gz(row['path'],dict(mode=mode,g=0,R=44,reader_family=family,truth_inputs=False,inference=inf,q_sha256=qr['sha256']));row.update(status=inf['summary']['status'],sha256=sha(row['path']),summary=inf['summary'],scored=174,edge_scored=inf['edge_local']['scored'] if mode==MODES[1] else 0,absolute_local_scored=inf['local_state']['scored'] if mode==MODES[0] else 0)
                except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure(mid,exc)
                store.save();store.snapshots()
        psid=aid+'/'+kid;pr=store.data['payload_reads'][psid]
        try:pr.update(store.call('payload_read',lambda:method.payload_read(normalized,key,44)))
        except Exception as exc:pr.update(status='FAILED',error=str(exc));store.failure(psid+'/PAYLOAD',exc)
        store.save();store.snapshots()


def media_worker(store,cfg):
    import torch
    from runtime.wan.generation import load_frozen_vae
    if not any(r['status']=='COMPLETE' for r in store.data['generation'].values()):settle(store,'media','no generated terminal');return
    store.data['stage']='VAE_LOAD';store.save();device='cuda' if torch.cuda.is_available() else 'cpu';frozen=store.call('vae_load',lambda:load_frozen_vae(cfg,device=device));store.data['vae_setup']=dict(model=cfg['model'],device=device,dtype=str(next(frozen.parameters()).dtype),environment=environment());store.save()
    for arm in ARMS:
        generated=store.data['generation'][arm];tr=store.data['terminal_inputs'][arm];rr=store.data['rasters'][arm];store.data['stage']='DECODE_'+arm;store.save()
        try:
            path=Path(generated['terminal_path'])
            if generated['status']!='COMPLETE' or sha(path)!=generated['sha256']:raise ValueError('new saved terminal unavailable or changed')
            terminal=torch.load(path,map_location='cpu',weights_only=True)
            if tuple(terminal.shape)!=(1,16,46,40,64) or not bool(terminal.isfinite().all()):raise ValueError('full terminal geometry/finiteness')
            tr.update(status='LOADED',sha256=sha(path),shape=list(terminal.shape),dtype=str(terminal.dtype));store.save()
            rgb=store.call('vae_decode',lambda:vae_adapter.decode_normalized_latent(frozen,terminal.to(next(frozen.parameters()).device)));q8=vae_adapter.quantize_rgb8_no_codec(rgb);rr.update(store.call('raster_save',lambda:media.save_raster(q8,rr['path'])));store.save();del rgb,q8,terminal
        except Exception as exc:tr.update(status='FAILED',error=str(exc));store.failure(arm+'/DECODE',exc);continue
        for channel in CHANNELS:
            aid=arm+'/'+channel;transport=store.data['transport'][aid];norm=store.data['normalized'][aid];store.data['stage']='CHANNEL_'+aid;store.save()
            try:
                source=media.reopen_raster(rr['path'],rr['sha256']);transport.update(status='RUNNING',input_raster_path=rr['path'],input_raster_sha256=rr['sha256'],input_raster_bytes=rr['bytes']);store.save()
                if channel=='DIRECT_RGB8':received=source;transport['events']['rgb24']=dict(status='SAVED',path=rr['path'],sha256=rr['sha256'],bytes=rr['bytes'],source='identical saved raster; no media conversion')
                elif channel=='RAW420':received=raw420.roundtrip(source,store.output/arm/'channel.yuv420',store.output/arm/'raw420.rgb24',count=store.count,event=lambda name,row:store.event(aid,name,row))
                else:
                    del source;received=media.mp4_roundtrip(rr['path'],rr['sha256'],store.output/arm/'channel.mp4',store.output/arm/'mp4.rgb24',count=store.count,event=lambda name,row:store.event(aid,name,row))
                if tuple(received.shape)!=media.SHAPE or received.dtype!=torch.uint8 or sha(rr['path'])!=rr['sha256']:raise ValueError('full received RGB8/same-raster identity required')
                transport['status']='COMPLETE';store.save();normalized=store.call('vae_encode',lambda:vae_adapter.reencode_rgb24_readback(frozen,received.float()/255.))
                if tuple(normalized.shape)!=(1,16,46,40,64) or normalized.dtype!=torch.float32 or not bool(normalized.isfinite().all()):raise ValueError('full normalized shape/dtype/finiteness')
                path=Path(norm['path']);tmp=path.with_suffix('.tmp');torch.save(normalized.detach().cpu(),tmp);os.replace(tmp,path);norm.update(status='SAVED',sha256=sha(path),shape=list(normalized.shape),dtype=str(normalized.dtype),input_raster_sha256=rr['sha256'],channel_rgb_sha256=transport['events']['rgb24']['sha256']);store.save();store.snapshots();receive_normalized(store,aid,normalized.detach().cpu(),cfg)
            except Exception as exc:
                if transport['status']!='COMPLETE':transport.update(status='FAILED',error=str(exc))
                if norm['status']=='PENDING':norm.update(status='FAILED',error=str(exc))
                store.failure(aid,exc)
            finally:
                if 'source' in locals():del source
                if 'received' in locals():del received
                if 'normalized' in locals():del normalized
    settle(store,'media','unavailable terminal/channel/read')


def settle(store,phase,reason):
    groups=('generation','writer_diagnostics') if phase=='generation' else ('terminal_inputs','rasters','transport','normalized','group_observations','projections','mode_reads','payload_reads')
    for group in groups:
        for row in store.data[group].values():
            if row['status'] in ('PENDING','RUNNING'):row.update(status='NOT_COMPLETED',error=reason)
    budget=store.data['state_budget_schedule']
    if phase=='generation' and budget['status']=='PENDING':
        budget.update(status='NOT_COMPLETED',error=reason)
        for row in budget['steps'].values():
            if row['status']=='PENDING':row.update(status='NOT_COMPLETED',error=reason)
    store.save();store.snapshots()


def evaluate(store,cfg):
    store.snapshots();paths=[store.output/'receiver_readouts.json',store.output/'payload_readouts.json'];sealed={str(p):sha(p) for p in paths};catalog=receiver.frozen.catalog(44);truth=list(range(1,45));true_id=next(i for i,r in enumerate(catalog) if r['structurally_valid'] and r['taus']==truth)
    for mid,mr in store.data['mode_reads'].items():
        out=store.data['path_posthoc'][mid];inf=None
        if mr['status'] in ('COMPLETE','NO_ENERGY'):
            inf=json.loads(gzip.decompress(Path(mr['path']).read_bytes()))['inference'];ids=inf['valid_catalog_indices'];costs=np.asarray(inf['path_costs']);j=ids.index(true_id);tc=float(costs[j]);canonical=inf['summary']['canonical_catalog_index'];out.update(status='EVALUATED',registered_tau=truth,true_catalog_index=true_id,true_path_cost=tc,true_path_rank=1+int(np.sum(costs<tc-receiver.PUBLIC.tie_atol)),true_path_delta=float(np.delete(costs,j).min()-tc),truth_in_top=true_id in inf['summary']['top_catalog_indices'],canonical_path_matches=canonical==true_id,canonical=None if canonical is None else catalog[canonical],top_catalog_indices=inf['summary']['top_catalog_indices'])
        else:out.update(status='MISSING_READ',error=mr.get('error'))
        if mid.endswith('/'+MODES[1]):
            sid=mid.rsplit('/',1)[0];edge=store.data['edge_posthoc'][sid]
            if inf is None:edge.update(status='MISSING_READ',error=mr.get('error'))
            else:
                pairs=inf['edge_local']['pairs'];rows=[]
                for j,rec in enumerate(inf['edge_local']['rows']):
                    target=[j+1,j+2];idx=pairs.index(target)
                    if rec['status']!='SCORED':rows.append(dict(received_edge_index=j+1,status='NO_SUPPORT',true_transition=target));continue
                    costs=np.asarray(rec['costs']);tc=float(costs[idx]);top=rec['top_pair_indices'];rows.append(dict(received_edge_index=j+1,received_window_pair=[j+1,j+2],true_transition=target,true_pair_index=idx,status='EVALUATED',true_rank=1+int(np.sum(costs<tc-receiver.PUBLIC.tie_atol)),delta=float(np.delete(costs,idx).min()-tc),truth_in_top=idx in top,unique_truth=top==[idx],top_pair_indices=top,top_transitions=[pairs[i] for i in top]))
                edge.update(status='EVALUATED',rows=rows,unique_truth_edges=sum(r.get('unique_truth',False) for r in rows),truth_in_top_edges=sum(r.get('truth_in_top',False) for r in rows))
        store.save()
    for sid,pr in store.data['payload_reads'].items():
        bits=pr['decoded_bits']
        for tid,message in [('REGISTERED',cfg['message']),('WRONG_MESSAGE',cfg['wrong_message'])]:
            target=method.message_bits(message);out=store.data['payload_evaluations'][sid+'/'+tid];out.update(status='EVALUATED' if pr['status']=='READ' else 'MISSING_READ',bit_errors=sum(a!=b for a,b in zip(bits,target)) if bits is not None else None,exact_bits=bits==target if bits is not None else None,accepted_payload=False,claim='new repeated payload; no time-dependent coding or family/path selection')
    if sealed!={str(p):sha(p) for p in paths}:raise ValueError('posthoc changed sealed receiver evidence')
    store.data['receiver_sha256']=sealed;store.save();evaluate_terminal_diagnostics(store)


def terminal_diagnostics(store,cfg):
    """Saved writer terminal only, separate full blind-formula diagnostic tables."""
    import torch
    t=store.data['terminal_diagnostics'];mask=np.ones((44,4),bool)
    for arm in ARMS:
        input_row=t['inputs'][arm];generated=store.data['generation'][arm]
        try:
            def load():
                path=Path(input_row['path'])
                if generated['status']!='COMPLETE' or not path.is_file() or sha(path)!=generated.get('sha256'):raise ValueError('saved terminal missing/changed')
                value=torch.load(path,map_location='cpu',weights_only=True)
                if tuple(value.shape)!=(1,16,46,40,64) or value.dtype!=torch.float32 or not bool(value.isfinite().all()):raise ValueError('terminal geometry/finiteness')
                return value
            value=store.diagnostic_call('terminal','terminal_tensor_load',load);input_row.update(status='LOADED',sha256=sha(input_row['path']),shape=list(value.shape),dtype=str(value.dtype));store.save();received=value[0,:,1:45].numpy().transpose(1,0,2,3)
        except Exception as exc:input_row.update(status='FAILED',error=str(exc));store.failure('TERMINAL_DIAGNOSTIC/'+arm,exc);continue
        for kid,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
            gid=arm+'/'+kid;gr=t['group_observations'][gid];groups=None
            try:groups=save_group_observation(store,gr,received,key,mask,terminal=True)
            except Exception as exc:gr.update(status='FAILED',error=str(exc));store.failure('TERMINAL_DIAGNOSTIC/'+gid+'/GROUP_Q',exc)
            for family in FAMILIES:
                sid=arm+'/'+family+'/'+kid;qr=t['projections'][sid]
                try:
                    if groups is None:raise ValueError('terminal group projections unavailable')
                    q,available=store.diagnostic_call('terminal','projection_extract',lambda:public_projection(groups,family));gz(qr['path'],dict(q=q.tolist(),availability=available.tolist(),group_input_path=gr['path'],group_input_sha256=gr['sha256'],g=0,R=44,reader_family=family,truth_inputs=False,role='writer-terminal diagnostic only; never main blind input'));qr.update(status='SAVED',sha256=sha(qr['path']),shape=list(q.shape));store.save()
                except Exception as exc:qr.update(status='FAILED',error=str(exc));store.failure('TERMINAL_DIAGNOSTIC/'+sid,exc);continue
                for mode in MODES:
                    mid=sid+'/'+mode;row=t['mode_reads'][mid]
                    try:
                        fn=receiver.absolute_control if mode==MODES[0] else receiver.infer_difference;inf=store.diagnostic_call('terminal','absolute_control' if mode==MODES[0] else 'difference_infer',lambda:fn(q,key,available))
                        if inf['summary']['status'] not in ('COMPLETE','NO_ENERGY') or inf['counts']['scored']!=174:raise ValueError('terminal fixed candidate costs incomplete')
                        gz(row['path'],dict(mode=mode,g=0,R=44,reader_family=family,truth_inputs=False,inference=inf,q_sha256=qr['sha256'],role='writer-terminal diagnostic only'));row.update(status=inf['summary']['status'],sha256=sha(row['path']),summary=inf['summary'],scored=174,edge_scored=inf['edge_local']['scored'] if mode==MODES[1] else 0,absolute_local_scored=inf['local_state']['scored'] if mode==MODES[0] else 0)
                    except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure('TERMINAL_DIAGNOSTIC/'+mid,exc)
                    store.save()
        del received,value
    for group in ('inputs','group_observations','projections','mode_reads'):
        for row in t[group].values():
            if row['status'] in ('PENDING','RUNNING'):row.update(status='NOT_COMPLETED',error='saved terminal or diagnostic read unavailable')
    raw=store.output/'terminal_diagnostic_readouts.json';dump(raw,dict(inputs=t['inputs'],group_observations=t['group_observations'],projections=t['projections'],mode_reads=t['mode_reads'],shared=store.data['shared'],public_readers=store.data['public_readers'],truth_inputs=False,role='writer-only terminal raw snapshot, no truth/posthoc'));t['receiver_sha256']={str(raw):sha(raw)};store.save()


def evaluate_terminal_diagnostics(store):
    """Truth join only after main and terminal raw snapshots are sealed."""
    t=store.data['terminal_diagnostics'];sealed=t['receiver_sha256'];catalog=receiver.frozen.catalog(44);truth=list(range(1,45));true_id=next(i for i,r in enumerate(catalog) if r['structurally_valid'] and r['taus']==truth)
    for mid,mr in t['mode_reads'].items():
        out=t['path_posthoc'][mid];inf=None
        if mr['status'] in ('COMPLETE','NO_ENERGY'):
            inf=json.loads(gzip.decompress(Path(mr['path']).read_bytes()))['inference'];ids=inf['valid_catalog_indices'];costs=np.asarray(inf['path_costs']);j=ids.index(true_id);tc=float(costs[j]);canonical=inf['summary']['canonical_catalog_index'];out.update(status='EVALUATED',registered_tau=truth,true_catalog_index=true_id,true_path_cost=tc,true_path_rank=1+int(np.sum(costs<tc-receiver.PUBLIC.tie_atol)),true_path_delta=float(np.delete(costs,j).min()-tc),truth_in_top=true_id in inf['summary']['top_catalog_indices'],canonical_path_matches=canonical==true_id,canonical=None if canonical is None else catalog[canonical],top_catalog_indices=inf['summary']['top_catalog_indices'])
        else:out.update(status='MISSING_READ',error=mr.get('error'))
        if mid.endswith('/'+MODES[1]):
            sid=mid.rsplit('/',1)[0];edge=t['edge_posthoc'][sid]
            if inf is None:edge.update(status='MISSING_READ',error=mr.get('error'))
            else:
                pairs=inf['edge_local']['pairs'];rows=[]
                for j,rec in enumerate(inf['edge_local']['rows']):
                    target=[j+1,j+2];idx=pairs.index(target)
                    if rec['status']!='SCORED':rows.append(dict(received_edge_index=j+1,status='NO_SUPPORT',true_transition=target));continue
                    costs=np.asarray(rec['costs']);tc=float(costs[idx]);top=rec['top_pair_indices'];rows.append(dict(received_edge_index=j+1,received_window_pair=[j+1,j+2],true_transition=target,true_pair_index=idx,status='EVALUATED',true_rank=1+int(np.sum(costs<tc-receiver.PUBLIC.tie_atol)),delta=float(np.delete(costs,idx).min()-tc),truth_in_top=idx in top,unique_truth=top==[idx],top_pair_indices=top,top_transitions=[pairs[i] for i in top]))
                edge.update(status='EVALUATED',rows=rows,unique_truth_edges=sum(r.get('unique_truth',False) for r in rows),truth_in_top_edges=sum(r.get('truth_in_top',False) for r in rows))
        store.save()
    if any(sha(path)!=expected for path,expected in sealed.items()):raise ValueError('terminal posthoc changed sealed raw')
    if any(sha(path)!=expected for path,expected in store.data['receiver_sha256'].items()):raise ValueError('terminal posthoc changed main blind snapshot')
    store.save()


def quality_diagnostics(store):
    """Existing RGB24 bytes only; chunked diagnostic RMSE, no new codec/read calls."""
    shape=media.SHAPE
    for channel in CHANNELS:
        for arm,reference in QUALITY_PAIRS:
            row=store.data['quality'][channel+'/'+arm+'_vs_'+reference];left=store.data['transport'][arm+'/'+channel];right=store.data['transport'][reference+'/'+channel]
            if left['status']!='COMPLETE' or right['status']!='COMPLETE':row.update(status='MISSING_CHANNEL');store.save();continue
            try:
                def compute():
                    a,b=left['events']['rgb24'],right['events']['rgb24']
                    for r in (a,b):
                        if sha(r['path'])!=r['sha256'] or Path(r['path']).stat().st_size!=media.RGB_BYTES:raise ValueError('saved quality RGB24 bytes changed')
                    x=np.memmap(a['path'],dtype=np.uint8,mode='r',shape=shape);y=np.memmap(b['path'],dtype=np.uint8,mode='r',shape=shape);squared=0.
                    for start in range(0,shape[0],4):delta=(np.asarray(x[start:start+4],dtype=np.float64)-np.asarray(y[start:start+4],dtype=np.float64))/255.;squared+=float(np.square(delta).sum())
                    mse=squared/int(np.prod(shape));del x,y
                    return dict(status='MEASURED',rgb_rmse=math.sqrt(mse),psnr_db=-10*math.log10(mse) if mse else None,identical_pixels=mse==0,left_rgb_sha256=a['sha256'],right_rgb_sha256=b['sha256'],source='same-format persisted RGB24; no new VAE/codec calls')
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
    if not ok:store.failure('WORKER_'+phase,RuntimeError(error or cleanup_error or f'child exit{code}'))
    settle(store,phase,error or 'worker did not complete');return store,interrupted


def finish(store,cfg):
    expected=dict(group_observation_records=24,generated=4,generation_steps=200,controlled_steps=75,writer_sidecars=100,rasters=4,transport=12,normalized=12,keyed_projections=48,path_score_records=96,path_costs=16704,difference_edge_costs=272448,absolute_local_state_costs=95040,payload_reads=24,path_posthoc=96,difference_edge_posthoc=48,payload_evaluations=48,quality_pairs=18)
    calls={k:dict(expected=n,actual=store.data['calls'].get(k,dict(attempted=0,completed=0)),match=store.data['calls'].get(k)==dict(attempted=n,completed=n)) for k,n in cfg['planned_calls'].items()};store.data['call_integrity']=dict(status='MATCH' if all(v['match'] for v in calls.values()) else 'INCOMPLETE',rows=calls)
    terminal={k:dict(expected=n,actual=store.data['terminal_diagnostics']['calls'].get(k,dict(attempted=0,completed=0)),match=store.data['terminal_diagnostics']['calls'].get(k)==dict(attempted=n,completed=n)) for k,n in cfg['planned_terminal_diagnostic_calls'].items()}
    store.data['diagnostic_call_integrity']=dict(terminal=terminal)
    terminal_expected={k:v for k,v in TERMINAL_FIXED.items() if k not in ('catalog_slots','structural_exclusions')}
    diagnostic_ok=store.data['terminal_diagnostics']['counts']==terminal_expected and all(v['match'] for v in terminal.values())
    budget=store.data['state_budget_schedule'];budget_ok=budget['status']=='SAVED' and len(budget['steps'])==25 and all(r['status']=='RECORDED' for r in budget['steps'].values())
    writer_ok=all(r.get('receipt',{}).get('writer_diagnostics',{}).get('last_z_post_matches_terminal') is True and r.get('receipt',{}).get('writer_diagnostics',{}).get('last_group_z_post_matches_terminal') is True for r in store.data['generation'].values() if r['status']=='COMPLETE')
    done=diagnostic_ok and budget_ok and not store.data['failures'] and store.data['counts']==expected and store.data['call_integrity']['status']=='MATCH' and writer_ok and all(store.data['workers'].get(p,{}).get('status')=='COMPLETE' for p in ('generation','media'))
    store.data.update(status='EXECUTION_COMPLETE' if done else 'INCOMPLETE',stage='FINISHED',science_status='UNCALIBRATED_FIXED_REPLICA_CANDIDATE_NO_AUTOMATIC_PASS');store.save();print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])));return done


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
    settle(store,'generation','generation incomplete');settle(store,'media','media incomplete');terminal_diagnostics(store,cfg);evaluate(store,cfg);quality_diagnostics(store);done=finish(store,cfg)
    if interrupted:raise SystemExit(130)
    if not done:raise SystemExit(1)

if __name__=='__main__':main()
