"""User-run same saved RGB8 media comparison; no generation or writer update."""
from __future__ import annotations
import argparse,gzip,hashlib,importlib.metadata,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from main.tube_state import video_local_fourier_rm_control as payload
from experiments.wan_state_clock import video_local_fourier_rm_difference_receiver as receiver
from runtime.wan import video_local_fourier_rm_same_raster as media
from runtime.wan import zero_mean_c1_yuv420_no_h264 as raw420
from runtime.wan import vae as vae_adapter
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_same_raster_v1.json'
MODULE='experiments.wan_state_clock.video_local_fourier_rm_same_raster_run'
ARMS=('OFF','PAYLOAD_MULTI','STATE_MULTI');CHANNELS=('DIRECT_RGB8','RAW420','MP4');KEYS=('CORRECT','WRONG');TARGETS=('REGISTERED','WRONG_MESSAGE');MODES=receiver.MODES
FIXED=dict(source_cases=1,arms=3,channels=3,underlying_observations=9,normalized=9,keys=2,keyed_projections=18,path_score_records=36,
    valid_paths=174,catalog_per_record=3915,path_costs=6264,difference_edge_costs=102168,absolute_local_state_costs=35640,
    path_posthoc=36,difference_edge_posthoc=18,payload_reads=18,payload_evaluations=36,catalog_slots=140940,structural_exclusions=134676)
SIZES=dict(terminal_inputs=3,rasters=3,transport=9,normalized=9,projections=18,mode_reads=36,payload_reads=18,path_posthoc=36,edge_posthoc=18,payload_evaluations=36)

def sha(path):return media.file_sha256(path)
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

def load_config():
    cfg=json.loads(CONFIG.read_text());old=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_v1.json').read_text())
    if any(cfg[k]!=old[k] for k in ('model','control','carrier','key','wrong_key','message','wrong_message','method_version')):raise ValueError('frozen writer/carrier/model protocol mismatch')
    if cfg['fixed_denominator']!=FIXED or cfg['arms']!=list(ARMS) or cfg['channels']!=list(CHANNELS) or cfg['modes']!=list(MODES):raise ValueError('fixed denominator/roster mismatch')
    if cfg['receiver']!=dict(g=0,R=44,alpha=receiver.PUBLIC.alpha,tie_atol=1e-12,threshold=None,accepted_payload=False,state_path_accepted=False):raise ValueError('frozen receiver mismatch')
    return cfg


class Store:
    def __init__(self,output,*,create=False,baseline=None):
        self.output=Path(output);self.path=self.output/'result.json'
        if not create:self.data=json.loads(self.path.read_text())
        else:
            cfg=load_config();self.output.mkdir(parents=True,exist_ok=False)
            self.data=dict(status='RUNNING',stage='INITIALIZE',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_worktree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),
                source_files={p:sha(ROOT/p) for p in cfg['source_files']},config_sha256=sha(CONFIG),fixed_denominator=FIXED,method_version=receiver.PUBLIC.method_version,
                baseline=dict(path=str(baseline),status='PENDING'),model=cfg['model'],environment=environment(),terminal_inputs={},rasters={},transport={},normalized={},
                projections={},mode_reads={},payload_reads={},path_posthoc={},edge_posthoc={},payload_evaluations={},calls={},workers={},failures=[],shared={},comparisons={},
                actual_generation_calls=False,actual_writer_update_calls=False,actual_vae_calls=False,actual_media_calls=False,evidence_ceiling=cfg['evidence_ceiling'])
            for arm in ARMS:
                folder=self.output/arm;self.data['terminal_inputs'][arm]=dict(status='PENDING',path=str(Path(baseline)/arm/'terminal.pt'))
                self.data['rasters'][arm]=dict(status='PENDING',path=str(folder/'source.rgb8'))
                for channel in CHANNELS:
                    aid=arm+'/'+channel;self.data['transport'][aid]=dict(status='PENDING',events={})
                    self.data['normalized'][aid]=dict(status='PENDING',path=str(folder/(channel+'.pt')),g=0,R=44)
                    for kid in KEYS:
                        sid=aid+'/'+kid;self.data['projections'][sid]=dict(status='PENDING',path=str(folder/(channel+'.'+kid+'.q.json.gz')),g=0,R=44)
                        self.data['payload_reads'][sid]=dict(status='PENDING',decoded_bits=None,source='NEW_SAME_RASTER_CHANNEL_READ',truth_used=False)
                        self.data['edge_posthoc'][sid]=dict(status='PENDING')
                        for target in TARGETS:self.data['payload_evaluations'][sid+'/'+target]=dict(status='PENDING',bit_errors=None,accepted_payload=False)
                        for mode in MODES:
                            mid=sid+'/'+mode;self.data['mode_reads'][mid]=dict(status='PENDING',path=str(folder/(channel+'.'+kid+'.'+mode+'.json.gz')),
                                g=0,R=44,mode=mode,summary=None,accepted_payload=False,state_path_accepted=False)
                            self.data['path_posthoc'][mid]=dict(status='PENDING',accepted_payload=False,state_path_accepted=False)
            catalog=self.output/'catalog_R44.json.gz';gz(catalog,dict(rows=receiver.frozen.catalog(44)))
            for kid,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
                mask=np.ones((44,4),bool);ab=receiver.frozen.family_receipt(key,44,mask);df=receiver.difference_family(key,mask)
                dpayload=dict(availability=mask.tolist(),edge_availability=df['edge_availability'].tolist(),available_dimensions=df['available_dimensions'],valid_catalog_indices=df['valid_catalog_indices'],classes=df['classes'])
                ap=self.output/('family_absolute_'+kid+'.json.gz');dp=self.output/('family_difference_'+kid+'.json.gz');gz(ap,ab);gz(dp,dpayload)
                self.data['shared'][kid]=dict(catalog_path=str(catalog),catalog_sha256=sha(catalog),absolute_family_path=str(ap),absolute_family_sha256=sha(ap),difference_family_path=str(dp),difference_family_sha256=sha(dp),valid_count=174,catalog_count=3915)
        self.save()
    def save(self):
        d=self.data
        if any(len(d[k])!=v for k,v in SIZES.items()):raise ValueError('fixed roster changed')
        complete=lambda r:r['status'] in ('COMPLETE','NO_ENERGY')
        d['counts']=dict(rasters=sum(r['status']=='SAVED' for r in d['rasters'].values()),transport=sum(r['status']=='COMPLETE' for r in d['transport'].values()),
            normalized=sum(r['status']=='SAVED' for r in d['normalized'].values()),keyed_projections=sum(r['status']=='SAVED' for r in d['projections'].values()),
            path_score_records=sum(complete(r) for r in d['mode_reads'].values()),path_costs=sum(r.get('scored',0) for r in d['mode_reads'].values()),
            difference_edge_costs=sum(r.get('edge_scored',0) for r in d['mode_reads'].values()),absolute_local_state_costs=sum(r.get('absolute_local_scored',0) for r in d['mode_reads'].values()),
            payload_reads=sum(r['status']=='READ' for r in d['payload_reads'].values()),path_posthoc=sum(r['status']=='EVALUATED' for r in d['path_posthoc'].values()),
            difference_edge_posthoc=sum(r['status']=='EVALUATED' for r in d['edge_posthoc'].values()),payload_evaluations=sum(r['status']=='EVALUATED' for r in d['payload_evaluations'].values()))
        dump(self.path,d)
    def count(self,kind,done):
        row=self.data['calls'].setdefault(kind,dict(attempted=0,completed=0));row['completed' if done else 'attempted']+=1
        if kind.startswith('vae_'):self.data['actual_vae_calls']=True
        if kind.startswith(('rgb_to_','raw420_to_','mp4_')):self.data['actual_media_calls']=True
        self.save()
    def call(self,kind,operation):
        self.count(kind,False);value=operation();self.count(kind,True);return value
    def failure(self,where,exc):
        self.data['failures'].append(dict(stage=where,error=f'{type(exc).__name__}: {exc}'));self.save()
    def snapshots(self):
        dump(self.output/'receiver_readouts.json',dict(normalized=self.data['normalized'],projections=self.data['projections'],mode_reads=self.data['mode_reads'],shared=self.data['shared'],
            truth_inputs=False,claim='same-raster channel diagnostic only; no writer/terminal/media truth enters inference'))
        dump(self.output/'payload_readouts.json',dict(reads=self.data['payload_reads'],truth_inputs=False,claim='new repeated payload reads; never path selectors'))
    def event(self,aid,stage,row):self.data['transport'][aid]['events'][stage]=row;self.save()


def baseline_meta(store,cfg,baseline):
    path=Path(baseline)/'result.json'
    try:
        meta=json.loads(path.read_text())
        if meta['source_sha']!=cfg['baseline']['source_sha'] or meta['method_version']!=receiver.PUBLIC.method_version:raise ValueError('original baseline source/method mismatch')
        store.data['baseline'].update(status='READ',source_sha=meta['source_sha'],result_sha256=sha(path),generation_setup=meta.get('generation_setup'),environment=meta.get('environment'))
        store.save();return meta
    except Exception as exc:store.data['baseline'].update(status='FAILED',error=str(exc));store.failure('BASELINE',exc);return None

def receive_normalized(store,aid,normalized,cfg):
    mask=np.ones((44,4),bool);received=normalized[0,:,1:45].detach().float().cpu().numpy().transpose(1,0,2,3)
    for kid,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
        sid=aid+'/'+kid;qr=store.data['projections'][sid]
        try:
            q=store.call('projection_extract',lambda:receiver.frozen.extract(received,key,mask))
            gz(qr['path'],dict(q=q.tolist(),availability=mask.tolist(),g=0,R=44,truth_inputs=False));qr.update(status='SAVED',sha256=sha(qr['path']),shape=list(q.shape));store.save()
        except Exception as exc:qr.update(status='FAILED',error=str(exc));store.failure(sid+'/Q',exc);continue
        for mode in MODES:
            mid=sid+'/'+mode;row=store.data['mode_reads'][mid]
            try:
                fn=receiver.absolute_control if mode==MODES[0] else receiver.infer_difference
                inf=store.call('absolute_control' if mode==MODES[0] else 'difference_infer',lambda:fn(q,key,mask))
                if inf['summary']['status'] not in ('COMPLETE','NO_ENERGY') or inf['counts']['scored']!=174:raise ValueError('fixed path cost set incomplete')
                gz(row['path'],dict(mode=mode,g=0,R=44,truth_inputs=False,inference=inf,q_sha256=qr['sha256']))
                row.update(status=inf['summary']['status'],sha256=sha(row['path']),summary=inf['summary'],scored=174,
                    edge_scored=inf['edge_local']['scored'] if mode==MODES[1] else 0,absolute_local_scored=inf['local_state']['scored'] if mode==MODES[0] else 0)
            except Exception as exc:row.update(status='FAILED',error=str(exc));store.failure(mid,exc)
            store.save();store.snapshots()
        pr=store.data['payload_reads'][sid]
        try:pr.update(store.call('payload_read',lambda:payload.payload_read(normalized,key,44)))
        except Exception as exc:pr.update(status='FAILED',error=str(exc));store.failure(sid+'/PAYLOAD',exc)
        store.save();store.snapshots()


def vae_worker(store,cfg,baseline):
    import torch
    from runtime.wan.generation import load_frozen_vae
    meta=baseline_meta(store,cfg,baseline)
    if meta is None:return
    store.data['stage']='VAE_LOAD';store.save();device='cuda' if torch.cuda.is_available() else 'cpu'
    frozen=store.call('vae_load',lambda:load_frozen_vae(cfg,device=device));store.data['vae_setup']=dict(model=cfg['model'],device=device,dtype=str(next(frozen.parameters()).dtype),environment=environment());store.save()
    for arm in ARMS:
        store.data['stage']='DECODE_'+arm;store.save();tr=store.data['terminal_inputs'][arm];rr=store.data['rasters'][arm]
        try:
            old=meta['generation'][arm];path=Path(baseline)/arm/'terminal.pt'
            if old['status']!='COMPLETE' or sha(path)!=old['sha256']:raise ValueError('original saved terminal unavailable or changed')
            terminal=torch.load(path,map_location='cpu',weights_only=True)
            if tuple(terminal.shape)!=(1,16,46,40,64) or not bool(terminal.isfinite().all()):raise ValueError('original full terminal geometry/finiteness')
            tr.update(status='LOADED',sha256=sha(path),shape=list(terminal.shape),dtype=str(terminal.dtype));store.save()
            rgb=store.call('vae_decode',lambda:vae_adapter.decode_normalized_latent(frozen,terminal.to(next(frozen.parameters()).device)))
            q8=vae_adapter.quantize_rgb8_no_codec(rgb);rr.update(store.call('raster_save',lambda:media.save_raster(q8,rr['path'])));store.save();del rgb,q8,terminal
        except Exception as exc:tr.update(status='FAILED',error=str(exc));store.failure(arm+'/DECODE',exc);continue
        for channel in CHANNELS:
            aid=arm+'/'+channel;store.data['stage']='CHANNEL_'+aid;store.save();transport=store.data['transport'][aid];norm=store.data['normalized'][aid]
            try:
                source=media.reopen_raster(rr['path'],rr['sha256']);transport.update(status='RUNNING',input_raster_path=rr['path'],input_raster_sha256=rr['sha256'],input_raster_bytes=rr['bytes']);store.save()
                if channel=='DIRECT_RGB8':received=source;transport['events']['rgb24']=dict(status='SAVED',path=rr['path'],sha256=rr['sha256'],bytes=rr['bytes'],source='the identical saved raster; no media conversion')
                elif channel=='RAW420':received=raw420.roundtrip(source,store.output/arm/'channel.yuv420',store.output/arm/'raw420.rgb24',count=store.count,event=lambda name,row:store.event(aid,name,row))
                else:
                    del source
                    received=media.mp4_roundtrip(rr['path'],rr['sha256'],store.output/arm/'channel.mp4',store.output/arm/'mp4.rgb24',count=store.count,event=lambda name,row:store.event(aid,name,row))
                if tuple(received.shape)!=media.SHAPE or received.dtype!=torch.uint8:raise ValueError('full received uint8 RGB24 required')
                if sha(rr['path'])!=rr['sha256']:raise ValueError('shared saved raster changed during channel')
                transport['status']='COMPLETE';store.save()
                normalized=store.call('vae_encode',lambda:vae_adapter.reencode_rgb24_readback(frozen,received.float()/255.))
                if tuple(normalized.shape)!=(1,16,46,40,64) or normalized.dtype!=torch.float32 or not bool(normalized.isfinite().all()):raise ValueError('full normalized shape/dtype/finiteness')
                p=Path(norm['path']);tmp=p.with_suffix('.tmp');torch.save(normalized.detach().cpu(),tmp);os.replace(tmp,p)
                norm.update(status='SAVED',sha256=sha(p),shape=list(normalized.shape),dtype=str(normalized.dtype),input_raster_sha256=rr['sha256'],channel_rgb_sha256=transport['events']['rgb24']['sha256']);store.save();store.snapshots()
                receive_normalized(store,aid,normalized.detach().cpu(),cfg);del received,normalized
            except Exception as exc:
                if transport['status'] not in ('COMPLETE',):transport.update(status='FAILED',error=str(exc))
                if norm['status']=='PENDING':norm.update(status='FAILED',error=str(exc))
                store.failure(aid,exc)
            finally:
                for name in ('source','received','normalized'):
                    if name in locals():
                        if name=='source':del source
                        elif name=='received':del received
                        else:del normalized
    settle(store,'unavailable terminal or channel')


def settle(store,reason):
    for group in ('terminal_inputs','rasters','transport','normalized','projections','mode_reads','payload_reads'):
        for row in store.data[group].values():
            if row['status'] in ('PENDING','RUNNING'):row.update(status='NOT_COMPLETED',error=reason)
    store.save();store.snapshots()


def evaluate(store,cfg):
    store.snapshots();paths=[store.output/'receiver_readouts.json',store.output/'payload_readouts.json'];sealed={str(p):sha(p) for p in paths}
    catalog=receiver.frozen.catalog(44);truth=list(range(1,45));true_id=next(i for i,r in enumerate(catalog) if r['structurally_valid'] and r['taus']==truth)
    for mid,mr in store.data['mode_reads'].items():
        out=store.data['path_posthoc'][mid];inf=None
        if mr['status'] in ('COMPLETE','NO_ENERGY'):
            inf=json.loads(gzip.decompress(Path(mr['path']).read_bytes()))['inference'];ids=inf['valid_catalog_indices'];costs=np.asarray(inf['path_costs']);j=ids.index(true_id);tc=float(costs[j]);canonical=inf['summary']['canonical_catalog_index']
            out.update(status='EVALUATED',registered_tau=truth,true_catalog_index=true_id,true_path_cost=tc,true_path_rank=1+int(np.sum(costs<tc-receiver.PUBLIC.tie_atol)),true_path_delta=float(np.delete(costs,j).min()-tc),truth_in_top=true_id in inf['summary']['top_catalog_indices'],canonical_path_matches=canonical==true_id,canonical=None if canonical is None else catalog[canonical],top_catalog_indices=inf['summary']['top_catalog_indices'])
        else:out.update(status='MISSING_READ',error=mr.get('error'))
        if mid.endswith('/'+MODES[1]):
            sid=mid.rsplit('/',1)[0];edge=store.data['edge_posthoc'][sid]
            if inf is None:edge.update(status='MISSING_READ',error=mr.get('error'))
            else:
                pairs=inf['edge_local']['pairs'];rows=[]
                for j,rec in enumerate(inf['edge_local']['rows']):
                    target=[j+1,j+2];idx=pairs.index(target)
                    if rec['status']!='SCORED':rows.append(dict(received_edge_index=j+1,status='NO_SUPPORT',true_transition=target));continue
                    c=np.asarray(rec['costs']);tc=float(c[idx]);top=rec['top_pair_indices']
                    rows.append(dict(received_edge_index=j+1,received_window_pair=[j+1,j+2],true_transition=target,true_pair_index=idx,status='EVALUATED',true_rank=1+int(np.sum(c<tc-receiver.PUBLIC.tie_atol)),delta=float(np.delete(c,idx).min()-tc),truth_in_top=idx in top,unique_truth=top==[idx],top_pair_indices=top,top_transitions=[pairs[i] for i in top]))
                edge.update(status='EVALUATED',rows=rows,unique_truth_edges=sum(r.get('unique_truth',False) for r in rows),truth_in_top_edges=sum(r.get('truth_in_top',False) for r in rows))
        store.save()
    for sid,pr in store.data['payload_reads'].items():
        bits=pr['decoded_bits']
        for tid,message in [('REGISTERED',cfg['message']),('WRONG_MESSAGE',cfg['wrong_message'])]:
            target=payload.message_bits(message);out=store.data['payload_evaluations'][sid+'/'+tid]
            out.update(status='EVALUATED' if pr['status']=='READ' else 'MISSING_READ',bit_errors=sum(a!=b for a,b in zip(bits,target)) if bits is not None else None,exact_bits=bits==target if bits is not None else None,accepted_payload=False,claim='new repeated payload; not time-dependent evidence or path selector')
    if sealed!={str(p):sha(p) for p in paths}:raise RuntimeError('posthoc changed saved receiver evidence')
    store.data['receiver_sha256']=sealed;store.save()


def run_child(store,baseline):
    command=[sys.executable,'-u','-m',MODULE,'--output',str(store.output),'--baseline',str(baseline),'--worker'];start=time.perf_counter();child=None;code=None;error=None;cleanup_error=None;interrupted=False
    try:
        with (store.output/'vae_media.log').open('w') as log:
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
    store=Store(store.output);ok=code==0 and error is None and cleanup_error is None
    store.data['workers']['vae_media']=dict(status='COMPLETE' if ok else 'FAILED',command=command,returncode=code,error=error,cleanup_error=cleanup_error,elapsed_seconds=time.perf_counter()-start)
    if not ok:store.failure('VAE_MEDIA_CHILD',RuntimeError(error or cleanup_error or f'child exit {code}'))
    settle(store,error or 'child did not complete');return store,interrupted


def finish(store,cfg):
    expected=dict(rasters=3,transport=9,normalized=9,keyed_projections=18,path_score_records=36,path_costs=6264,difference_edge_costs=102168,absolute_local_state_costs=35640,payload_reads=18,path_posthoc=36,difference_edge_posthoc=18,payload_evaluations=36)
    calls={k:dict(expected=n,actual=store.data['calls'].get(k,dict(attempted=0,completed=0)),match=store.data['calls'].get(k)==dict(attempted=n,completed=n)) for k,n in cfg['planned_calls'].items()}
    store.data['call_integrity']=dict(status='MATCH' if all(v['match'] for v in calls.values()) else 'INCOMPLETE',rows=calls)
    done=not store.data['failures'] and store.data['counts']==expected and store.data['call_integrity']['status']=='MATCH' and store.data['workers'].get('vae_media',{}).get('status')=='COMPLETE'
    store.data.update(status='EXECUTION_COMPLETE' if done else 'INCOMPLETE',stage='FINISHED',science_status='UNCALIBRATED_SAME_RASTER_DIAGNOSTIC_NO_AUTOMATIC_PASS');store.save();print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])));return done


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path);p.add_argument('--worker',action='store_true');a=p.parse_args();cfg=load_config();baseline=a.baseline or Path(cfg['baseline']['path'])
    if a.worker:
        store=Store(a.output)
        try:vae_worker(store,cfg,baseline)
        except Exception as exc:store.failure(store.data['stage'],exc);settle(store,str(exc));raise
        return
    store=Store(a.output,create=True,baseline=baseline);store,halt=run_child(store,baseline);settle(store,'final incomplete rows');evaluate(store,cfg);done=finish(store,cfg)
    if halt:raise SystemExit(130)
    if not done:raise SystemExit(1)

if __name__=='__main__':main()
