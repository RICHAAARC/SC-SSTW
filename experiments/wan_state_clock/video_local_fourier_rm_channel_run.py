"""Fixed user-run VAE/clamp and RGB8 localization; reuse the saved MULTI terminals."""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from main.tube_state import video_local_fourier_rm_control as method
from main.tube_state import video_local_fourier_rm_state as state
from runtime.wan import vae as vae_adapter
from runtime.wan.video_temporal_sync_bridge import file_sha256
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_channel_v1.json'
MODULE='experiments.wan_state_clock.video_local_fourier_rm_channel_run'
ARMS=method.ARMS;KEYS=('CORRECT','WRONG');MESSAGES=('REGISTERED','WRONG_MESSAGE')
NEW=('FLOAT_VAE_ROUNDTRIP','RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC')
REF=('TERMINAL44_REFERENCE','MP4_G0_REFERENCE');STAGES=(REF[0],*NEW,REF[1])
FIXED=dict(source_cases=1,arms=3,new_channels=2,new_normalized=6,reference_inputs=6,
 observation_tensors=12,new_keyed_reads=12,reference_keyed_reads=12,keyed_reads=24,
 posthoc=48,local_posthoc=24,catalog_slots=93960,valid_costs=4176,structural_exclusions=89784,local_state_costs=47520)
SIZES=dict(normalized=6,reference_inputs=6,reads=24,evaluations=48,local_posthoc=24,channel_inputs=3)


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def dump_gzip(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(gzip.compress(json.dumps(value,allow_nan=False,separators=(',',':')).encode(),mtime=0));os.replace(temp,path)


def environment():
    result=dict(python=sys.version,executable=sys.executable,packages={})
    for name in ('torch','torchvision','diffusers','transformers','accelerate','numpy','huggingface-hub'):
        try:result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:result['packages'][name]=None
    return result


def load_config():
    cfg=json.loads(CONFIG.read_text());old=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_v1.json').read_text())
    for name in ('model','key','wrong_key','message','wrong_message','method_version','control','carrier'):
        if cfg[name]!=old[name]:raise ValueError('frozen source protocol changed: '+name)
    if cfg['fixed_denominator']!=FIXED or cfg['channels']!=list(NEW) or cfg['arms']!=list(ARMS):raise ValueError('fixed roster changed')
    return cfg


class Store:
    def __init__(self,output,*,create=False,baseline=None):
        self.output=Path(output);self.path=self.output/'result.json'
        if not create:self.data=json.loads(self.path.read_text())
        else:
            cfg=load_config();self.output.mkdir(parents=True,exist_ok=False)
            self.data=dict(status='RUNNING',stage='INITIALIZE',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                method_version=state.PUBLIC.method_version,fixed_denominator=FIXED,config_sha256=file_sha256(CONFIG),
                source_files={p:file_sha256(ROOT/p) for p in cfg['source_files']},baseline=dict(path=str(baseline),status='PENDING'),
                environment=environment(),model_loaded=False,actual_model_calls=False,model=cfg['model'],
                normalized={},reference_inputs={},reads={},evaluations={},local_posthoc={},channel_inputs={},shared={},
                calls={},workers={},failures=[],comparisons={},evidence_ceiling=cfg['evidence_ceiling'],
                public_state_codebook={k:state.state_code(v).tolist() for k,v in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]},
                basis_layout={k:state.basis_layout(v) for k,v in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]},
                reference_environment_limit=cfg['reference_environment_limit'])
            for arm in ARMS:
                self.data['channel_inputs'][arm]=dict(status='PENDING',terminal_path=str(Path(baseline)/arm/'terminal.pt'),rgb=None)
                for stage in STAGES:
                    tensor=dict(status='PENDING',role='NEW_CHANNEL_DIAGNOSTIC' if stage in NEW else 'OFFLINE_REFERENCE',g=0,R=44)
                    if stage in NEW:tensor['path']=str(self.output/arm/(stage+'.pt'));self.data['normalized'][arm+'/'+stage]=tensor
                    else:tensor['path']=str(Path(baseline)/arm/('terminal.pt' if stage==REF[0] else 'FULL_SOURCE181.phase0.pt'));self.data['reference_inputs'][arm+'/'+stage]=tensor
                    for key in KEYS:
                        sid=f'{arm}/{stage}/{key}'
                        self.data['reads'][sid]=dict(status='PENDING',g=0,R=44,role=tensor['role'],
                            path=str(self.output/arm/(stage+'.'+key+'.raw.json.gz')),payload=None,summary=None,
                            state_path_accepted=False,accepted_payload=False)
                        self.data['local_posthoc'][sid]=dict(status='PENDING')
                        for msg in MESSAGES:self.data['evaluations'][sid+'/'+msg]=dict(status='PENDING',bit_errors=None,state_path_accepted=False,accepted_payload=False)
            for key,value in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
                catalog=self.output/'catalog_R44.json.gz';dump_gzip(catalog,dict(rows=state.catalog(44)))
                family=self.output/('family_R44_'+key+'.json.gz');receipt=state.family_receipt(value,44,np.ones((44,4),bool));dump_gzip(family,receipt)
                self.data['shared'][key]=dict(catalog_path=str(catalog),catalog_sha256=file_sha256(catalog),
                    family_path=str(family),family_sha256=file_sha256(family),catalog_count=3915,valid_count=174,class_count=len(receipt['classes']))
        self.save()
    def save(self):
        d=self.data
        if any(len(d[k])!=n for k,n in SIZES.items()):raise ValueError('fixed table size changed')
        reads=list(d['reads'].values());complete=lambda r:r['status'] in ('COMPLETE','NO_ENERGY')
        d['counts']=dict(new_normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),
            reference_inputs=sum(r['status']=='LOADED' for r in d['reference_inputs'].values()),
            new_reads=sum(complete(r) for r in reads if r['role']=='NEW_CHANNEL_DIAGNOSTIC'),
            reference_reads=sum(complete(r) for r in reads if r['role']=='OFFLINE_REFERENCE'),
            path_reads=sum(complete(r) for r in reads),payload_reads=sum((r.get('payload') or {}).get('status')=='READ' for r in reads),
            valid_costs=sum(r.get('scored',0) for r in reads),local_state_costs=sum(r.get('local_scored',0) for r in reads),
            evaluated=sum(r['status']=='EVALUATED' for r in d['evaluations'].values()),
            local_posthoc=sum(r['status']=='EVALUATED' for r in d['local_posthoc'].values()))
        dump(self.path,d)
    def count(self,kind,completed):
        row=self.data['calls'].setdefault(kind,dict(attempted=0,completed=0));row['completed' if completed else 'attempted']+=1
        if kind.startswith('vae_') and not completed:self.data['actual_model_calls']=True
        self.save()
    def call(self,kind,operation):
        self.count(kind,False);value=operation();self.count(kind,True);return value
    def failure(self,where,exc):
        self.data['failures'].append(dict(stage=where,error=f'{type(exc).__name__}: {exc}'));self.save()
    def receiver_snapshot(self):
        # Terminal/old-MP4 references are separately labelled and never blind inputs.
        new={sid:row for sid,row in self.data['reads'].items() if row['role']=='NEW_CHANNEL_DIAGNOSTIC'}
        refs={sid:row for sid,row in self.data['reads'].items() if row['role']=='OFFLINE_REFERENCE'}
        dump(self.output/'blind_channel_readouts.json',dict(normalized=self.data['normalized'],reads=new,shared=self.data['shared'],
             truth_inputs=False,role='channel diagnostics only; not independent saved-video blind detection'))
        dump(self.output/'reference_readouts.json',dict(inputs=self.data['reference_inputs'],reads=refs,role='offline saved terminal44 and old MP4-g0 reference only'))


def baseline_metadata(store,cfg,baseline):
    path=Path(baseline)/'result.json'
    try:
        raw=json.loads(path.read_text())
        if raw['source_sha']!=cfg['baseline']['source_sha'] or raw['method_version']!=state.PUBLIC.method_version:raise ValueError('baseline source/method mismatch')
        store.data['baseline'].update(status='READ',result_path=str(path),result_sha256=file_sha256(path),source_sha=raw['source_sha'],
             environment=raw.get('environment'),generation_setup=raw.get('generation_setup'),model=cfg['model'])
        store.save();return raw
    except Exception as exc:
        store.data['baseline'].update(status='FAILED',error=str(exc));store.failure('BASELINE_METADATA',exc);return None


def load_saved_tensor(store,baseline,meta,arm,stage):
    import torch
    table='reference_inputs' if stage in REF else 'channel_inputs'
    row=store.data[table][arm+'/'+stage] if stage in REF else store.data[table][arm]
    terminal=stage in (REF[0],*NEW)
    source=meta['generation'][arm] if terminal else meta['normalized'][arm+'/FULL_SOURCE181/0']
    if source['status']!=('COMPLETE' if terminal else 'SAVED'):raise RuntimeError('baseline input unavailable')
    path=Path(baseline)/arm/('terminal.pt' if terminal else 'FULL_SOURCE181.phase0.pt')
    before=file_sha256(path)
    if before!=source['sha256']:raise ValueError('baseline input identity mismatch: '+str(path))
    tensor=torch.load(path,map_location='cpu',weights_only=True)
    if tuple(tensor.shape)!=(1,16,46,40,64) or not bool(tensor.isfinite().all()):raise ValueError('baseline tensor geometry/finiteness')
    if file_sha256(path)!=before:raise ValueError('baseline changed during load')
    row.update(status='LOADED',sha256=before,shape=list(tensor.shape),dtype=str(tensor.dtype),source='saved baseline; local root, not old absolute record path')
    store.save();return tensor


def receiver_read(store,sid,normalized,key):
    """The actual receiver sees normalized observations, key and public support only."""
    row=store.data['reads'][sid];stage=sid.split('/')[1];prefix='new' if stage in NEW else 'reference'
    try:
        if tuple(normalized.shape)!=(1,16,46,40,64) or not bool(normalized.isfinite().all()):raise ValueError('normalized observation geometry/finiteness')
        received=normalized[0,:,1:45].detach().float().cpu().numpy().transpose(1,0,2,3)
        inference=store.call(prefix+'_path_read',lambda:state.infer(received,key,np.ones((44,4),bool)))
        payload=None
        try:payload=store.call(prefix+'_payload_read',lambda:method.payload_read(normalized,key,44))
        except Exception as exc:payload=dict(status='FAILED',decoded_bits=None,error=f'{type(exc).__name__}: {exc}');store.failure(sid+'/PAYLOAD',exc)
        if inference['summary']['status'] not in ('COMPLETE','NO_ENERGY') or inference['counts']['scored']!=174 or inference['local_state']['scored']!=1980:raise ValueError('incomplete state cost set')
        raw=dict(inference=inference,payload=payload,g=0,R=44,truth_inputs=False,role=row['role'])
        dump_gzip(row['path'],raw)
        row.update(status=inference['summary']['status'],sha256=file_sha256(row['path']),summary=inference['summary'],payload=payload,scored=174,local_scored=1980)
    except Exception as exc:row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');store.failure(sid,exc)
    store.save();store.receiver_snapshot()


def offline_references(store,cfg,baseline,meta):
    keys=dict(CORRECT=cfg['key'],WRONG=cfg['wrong_key'])
    for arm in ARMS:
        for stage in REF:
            if meta is None:continue
            try:
                normalized=load_saved_tensor(store,baseline,meta,arm,stage)
                for key_id,key in keys.items():
                    sid=f'{arm}/{stage}/{key_id}';receiver_read(store,sid,normalized,key)
                    # MP4 g0 is replayed from the old normalized tensor; no MP4 decode or VAE call.
                    if stage==REF[1] and store.data['reads'][sid]['status'] in ('COMPLETE','NO_ENERGY'):
                        check=store.data['reads'][sid]
                        try:
                            oldrow=meta['phase_reads'][f'{arm}/FULL_SOURCE181/0/{key_id}']
                            oldpath=Path(baseline)/arm/('FULL_SOURCE181.phase0.'+key_id+'.raw.json.gz')
                            if file_sha256(oldpath)!=oldrow['sha256']:raise ValueError('old raw identity mismatch')
                            old=json.loads(gzip.decompress(oldpath.read_bytes()));new=json.loads(gzip.decompress(Path(check['path']).read_bytes()))
                            check['old_raw_comparison']=dict(status='MATCH' if old['inference']==new['inference'] and old['payload']==new['payload'] else 'DIFFERENT',
                                old_path=str(oldpath),old_sha256=oldrow['sha256'],scope='same saved MP4 g0 tensor, receiver inference and payload exact replay')
                        except Exception as exc:check['old_raw_comparison']=dict(status='UNAVAILABLE',error=str(exc))
                        store.save();store.receiver_snapshot()
                del normalized
            except Exception as exc:
                row=store.data['reference_inputs'][arm+'/'+stage];row.update(status='FAILED',error=str(exc));store.failure(arm+'/'+stage,exc)
    settle_pending(store,groups=('reference_inputs',),roles=('OFFLINE_REFERENCE',),reason='reference input unavailable')


def rgb_receipt(rgb):
    import torch
    from runtime.wan.trajectory import fingerprint
    if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(320,512,3) or len(rgb)!=181 or rgb.dtype!=torch.float32 or not bool(rgb.isfinite().all()):raise ValueError('decoded RGB geometry/dtype/finiteness')
    return dict(shape=list(rgb.shape),dtype=str(rgb.dtype),minimum=float(rgb.min()),maximum=float(rgb.max()),
         fingerprint=fingerprint(rgb),clamped_by_existing_decode=True,persisted_full_rgb=False)


def save_normalized(store,arm,stage,value):
    import torch
    row=store.data['normalized'][arm+'/'+stage]
    if tuple(value.shape)!=(1,16,46,40,64) or value.dtype!=torch.float32 or not bool(value.isfinite().all()):raise ValueError('new normalized geometry/dtype/finiteness')
    path=Path(row['path']);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp')
    torch.save(value.detach().cpu(),temp);os.replace(temp,path)
    row.update(status='SAVED',shape=list(value.shape),dtype=str(value.dtype),sha256=file_sha256(path));store.save();store.receiver_snapshot()


def channel_worker(store,cfg,baseline):
    import torch
    from runtime.wan.generation import load_frozen_vae
    from runtime.wan.trajectory import fingerprint
    meta=baseline_metadata(store,cfg,baseline)
    if meta is None:return
    store.data['stage']='VAE_LOAD';store.save();device='cuda' if torch.cuda.is_available() else 'cpu'
    frozen=store.call('vae_load',lambda:load_frozen_vae(cfg,device=device));store.data['model_loaded']=True
    store.data['vae_setup']=dict(model=cfg['model'],device=device,dtype=str(next(frozen.parameters()).dtype),
         environment=environment(),environment_comparison={name:dict(previous=store.data['baseline'].get('environment',{}).get('packages',{}).get(name),current=value)
           for name,value in environment()['packages'].items()},same_decode_as_old_mp4_not_proven=True)
    store.save();keys=dict(CORRECT=cfg['key'],WRONG=cfg['wrong_key'])
    for arm in ARMS:
        store.data['stage']='CHANNEL_'+arm;store.save();rgb=None
        try:
            terminal=load_saved_tensor(store,baseline,meta,arm,NEW[0]);target_device=next(frozen.parameters()).device
            rgb=store.call('vae_decode',lambda:vae_adapter.decode_normalized_latent(frozen,terminal.to(target_device)))
            store.data['channel_inputs'][arm]['rgb']=rgb_receipt(rgb);store.data['channel_inputs'][arm]['status']='DECODED';store.save();del terminal
        except Exception as exc:
            store.data['channel_inputs'][arm].update(status='FAILED',error=str(exc));store.failure(arm+'/DECODE',exc);continue
        for stage in NEW:
            row=store.data['normalized'][arm+'/'+stage]
            try:
                if stage==NEW[0]:encode_input=rgb  # No uint8 or /255 on this branch.
                else:
                    raster=vae_adapter.quantize_rgb8_no_codec(rgb)
                    encode_input=raster.to(device=rgb.device,dtype=torch.float32)/255.0
                    delta=(encode_input-rgb)
                    quant=dict(shape=list(raster.shape),raster_dtype=str(raster.dtype),encode_dtype=str(encode_input.dtype),
                        raster_fingerprint=fingerprint(raster),encode_fingerprint=fingerprint(encode_input),
                        maximum_abs_error=float(delta.abs().max()),squared_error=sum(float(delta[i:i+4].double().square().sum()) for i in range(0,len(delta),4)),
                        unchanged_pixels=int((delta==0).sum()),pixel_count=delta.numel(),rounding='np.rint(clamp(rgb,0,1)*255).astype(uint8); FP32 /255; zero codec calls')
                    store.data['channel_inputs'][arm]['quantization']=quant;store.save();del raster,delta
                value=store.call('vae_encode',lambda:vae_adapter.reencode_rgb24_readback(frozen,encode_input))
                save_normalized(store,arm,stage,value)
                for key_id,key in keys.items():receiver_read(store,f'{arm}/{stage}/{key_id}',value,key)
                del value
            except Exception as exc:
                if row['status']=='PENDING':row.update(status='FAILED',error=str(exc))
                store.failure(arm+'/'+stage,exc)
            finally:
                if 'encode_input' in locals():del encode_input
        del rgb
    settle_pending(store,groups=('normalized','channel_inputs'),roles=('NEW_CHANNEL_DIAGNOSTIC',),reason='new channel unavailable')


def settle_pending(store,*,groups=(),roles=(),reason):
    for group in groups:
        for row in store.data[group].values():
            if row['status']=='PENDING':row.update(status='NOT_COMPLETED',error=reason)
    for row in store.data['reads'].values():
        if row['role'] in roles and row['status']=='PENDING':row.update(status='NOT_COMPLETED',error=reason)
    store.save();store.receiver_snapshot()


def evaluate_saved(store,cfg):
    store.receiver_snapshot();new=store.output/'blind_channel_readouts.json';refs=store.output/'reference_readouts.json'
    before={str(p):file_sha256(p) for p in (new,refs)};catalog=state.catalog(44);truth=list(range(1,45))
    true_id=next(i for i,r in enumerate(catalog) if r['structurally_valid'] and r['taus']==truth)
    for sid,rec in store.data['reads'].items():
        local=store.data['local_posthoc'][sid];bits=(rec.get('payload') or {}).get('decoded_bits');inf=None
        if rec['status'] in ('COMPLETE','NO_ENERGY'):
            inf=json.loads(gzip.decompress(Path(rec['path']).read_bytes()))['inference'];ids=inf['valid_catalog_indices'];costs=np.asarray(inf['path_costs']);j=ids.index(true_id);cost=float(costs[j])
            rows=[]
            for index,v in enumerate(inf['local_state']['rows']):
                c=np.asarray(v['costs']);tc=float(c[index]);rows.append(dict(received_regular_index=index+1,true_state=index+1,
                    true_rank=1+int(np.sum(c<tc-state.PUBLIC.tie_atol)),delta=float(np.delete(c,index).min()-tc),
                    truth_in_top=index+1 in v['top'],unique_truth=v['top']==[index+1],top=v['top']))
            local.update(status='EVALUATED',R=44,g=0,rows=rows,unique_truth_rows=sum(r['unique_truth'] for r in rows),
                truth_in_top_rows=sum(r['truth_in_top'] for r in rows),true_path_rank=1+int(np.sum(costs<cost-state.PUBLIC.tie_atol)),
                true_path_delta=float(np.delete(costs,j).min()-cost),true_path_cost=cost,true_catalog_index=true_id,
                canonical_path_matches=inf['summary']['canonical_catalog_index']==true_id,truth_in_top=true_id in inf['summary']['top_catalog_indices'],
                role='posthoc known unedited g0 reference; not a blind phase or edit estimate')
        else:local.update(status='MISSING_READ',error=rec.get('error'))
        arm=sid.split('/')[0]
        for msg,value in [('REGISTERED',cfg['message']),('WRONG_MESSAGE',cfg['wrong_message'])]:
            target=method.message_bits(value);row=store.data['evaluations'][sid+'/'+msg]
            row.update(status='EVALUATED' if inf is not None and bits is not None else 'MISSING_READ',
                bit_errors=sum(a!=b for a,b in zip(bits,target)) if bits is not None else None,exact_bits=bits==target if bits is not None else None,
                canonical_path_matches=local.get('canonical_path_matches'),truth_in_top=local.get('truth_in_top'),
                expected_pilot_present=arm=='STATE_MULTI' and sid.endswith('/CORRECT'),
                accepted_payload=False,state_path_accepted=False,claim='posthoc channel diagnostic and repeated payload only')
        store.save()
    for arm in ARMS:
        for key in KEYS:
            available={stage:store.data['local_posthoc'][f'{arm}/{stage}/{key}'] for stage in STAGES}
            store.data['comparisons'][arm+'/'+key]=dict(stages={s:{field:r.get(field) for field in ('status','unique_truth_rows','truth_in_top_rows','true_path_rank','true_path_delta','canonical_path_matches')} for s,r in available.items()},
                boundary='Terminal to FLOAT includes VAE decode, [0,1] clamp and encode. FLOAT to RGB8 uses the same current decoded RGB. RGB8 to old MP4 is a cross-run consistency reference, not codec-only causality.')
    after={str(p):file_sha256(p) for p in (new,refs)}
    if before!=after:raise RuntimeError('posthoc modified saved receiver evidence')
    store.data['receiver_sha256']=before;store.save()


def run_channel_child(store,baseline):
    command=[sys.executable,'-u','-m',MODULE,'--output',str(store.output),'--baseline',str(baseline),'--worker']
    started=time.perf_counter();child=None;code=None;error=None;cleanup_error=None;interrupted=False
    try:
        with (store.output/'channel.log').open('w') as log:
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
            except BaseException as cleanup:cleanup_error=f'{type(cleanup).__name__}: {cleanup}'
    finally:
        if child is not None and child.stdout is not None:
            try:child.stdout.close()
            except BaseException as cleanup:cleanup_error=cleanup_error or f'{type(cleanup).__name__}: {cleanup}'
    store=Store(store.output);store.data['workers']['channel']=dict(status='COMPLETE' if code==0 and error is None and cleanup_error is None else 'FAILED',
        command=command,returncode=code,error=error,cleanup_error=cleanup_error,child_started=child is not None,elapsed_seconds=time.perf_counter()-started)
    if code!=0 or error or cleanup_error:
        store.failure('CHANNEL_CHILD',RuntimeError(error or cleanup_error or f'child exit {code}'))
    settle_pending(store,groups=('normalized','channel_inputs'),roles=('NEW_CHANNEL_DIAGNOSTIC',),reason=error or 'channel child did not complete inputs')
    return store,interrupted


def finish(store,cfg):
    settle_pending(store,groups=('normalized','channel_inputs','reference_inputs'),roles=('NEW_CHANNEL_DIAGNOSTIC','OFFLINE_REFERENCE'),reason='unavailable saved input or channel')
    evaluate_saved(store,cfg)
    expected=dict(new_normalized=6,reference_inputs=6,new_reads=12,reference_reads=12,path_reads=24,payload_reads=24,
        valid_costs=4176,local_state_costs=47520,evaluated=48,local_posthoc=24)
    call_check={kind:dict(expected=n,actual=store.data['calls'].get(kind,dict(attempted=0,completed=0)),
        match=store.data['calls'].get(kind)==dict(attempted=n,completed=n)) for kind,n in cfg['planned_calls'].items()}
    store.data['call_integrity']=dict(status='MATCH' if all(v['match'] for v in call_check.values()) else 'INCOMPLETE',rows=call_check)
    done=store.data['counts']==expected and store.data['workers'].get('channel',{}).get('status')=='COMPLETE' and store.data['call_integrity']['status']=='MATCH'
    store.data.update(status='EXECUTION_COMPLETE' if done else 'INCOMPLETE',stage='FINISHED',science_status='UNCALIBRATED_CHANNEL_DIAGNOSTIC_NO_AUTOMATIC_PASS')
    store.save();print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])),flush=True);return done


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--worker',action='store_true');args=p.parse_args();cfg=load_config()
    baseline=args.baseline or Path(cfg['baseline']['path'])
    if args.worker:
        store=Store(args.output)
        try:channel_worker(store,cfg,baseline)
        except Exception as exc:store.failure(store.data['stage'],exc);settle_pending(store,groups=('normalized','channel_inputs'),roles=('NEW_CHANNEL_DIAGNOSTIC',),reason=str(exc));raise
        return
    store=Store(args.output,create=True,baseline=baseline);meta=baseline_metadata(store,cfg,baseline)
    offline_references(store,cfg,baseline,meta)
    store,interrupted=run_channel_child(store,baseline)
    done=finish(store,cfg)
    if interrupted:raise SystemExit(130)
    if not done:raise SystemExit(1)

if __name__=='__main__':main()
