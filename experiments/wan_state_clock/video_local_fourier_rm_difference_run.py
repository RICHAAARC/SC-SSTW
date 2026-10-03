"""Fixed saved-projection difference receiver diagnostic; no media/model execution."""
from __future__ import annotations
import argparse,collections,gzip,hashlib,importlib.metadata,json,os,subprocess,sys
from pathlib import Path
import numpy as np
from experiments.wan_state_clock import video_local_fourier_rm_difference_receiver as receiver
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_difference_v1.json'
ARMS=('OFF','PAYLOAD_MULTI','STATE_MULTI');KEYS=('CORRECT','WRONG');TARGETS=('REGISTERED','WRONG_MESSAGE')
NEW=('FLOAT_VAE_ROUNDTRIP','RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC');REF=('TERMINAL44_REFERENCE','MP4_G0_REFERENCE')
STAGES=(REF[0],*NEW,REF[1]);MODES=receiver.MODES
FIXED=dict(source_cases=1,arms=3,stages=4,keys=2,input_projections=24,underlying_observations=12,window_count=44,edges=43,
    valid_paths=174,catalog=3915,path_score_records=48,path_costs=8352,difference_edge_costs=136224,path_posthoc=48,
    difference_edge_posthoc=24,cached_payload_readouts=24,cached_payload_evaluations=48,catalog_slots=187920,structural_exclusions=179568)
SIZES=dict(inputs=24,underlying_observations=12,mode_reads=48,path_posthoc=48,edge_posthoc=24,cached_payload=24,payload_evaluations=48)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def dump_gzip(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_bytes(gzip.compress(json.dumps(value,separators=(',',':'),allow_nan=False).encode(),mtime=0));os.replace(temp,path)
def message_bits(message):
    data=message.encode('ascii')
    if len(data)!=4:raise ValueError('frozen32-bit ASCII message required')
    return np.unpackbits(np.frombuffer(data,dtype=np.uint8)).tolist()
def environment():
    return dict(python=sys.version,executable=sys.executable,numpy=importlib.metadata.version('numpy'),execution='CPU cached-projection only; no Torch/HF/model/media dependency')
def load_config():
    cfg=json.loads(CONFIG.read_text())
    if cfg['fixed_denominator']!=FIXED or cfg['arms']!=list(ARMS) or cfg['stages']!=list(STAGES) or cfg['modes']!=list(MODES):raise ValueError('fixed roster mismatch')
    old=json.loads((ROOT/'experiments/wan_state_clock/configs/video_local_fourier_rm_channel_v1.json').read_text())
    if any(cfg[k]!=old[k] for k in ('key','wrong_key','message','wrong_message','method_version')):raise ValueError('frozen key/code/message protocol mismatch')
    if cfg['receiver']!=dict(g=0,R=44,source_length=45,alpha=receiver.PUBLIC.alpha,tie_atol=1e-12,edge_mask='adjacent public availability intersection',difference='q[j+1]-q[j]',threshold=None,accepted_payload=False,state_path_accepted=False):raise ValueError('frozen difference protocol mismatch')
    return cfg


class Store:
    def __init__(self,output,input_root):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=False);self.path=self.output/'result.json';cfg=load_config()
        self.data=dict(status='RUNNING',stage='INITIALIZE',source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            source_files={p:sha(ROOT/p) for p in cfg['source_files']},source_worktree_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).splitlines(),config_sha256=sha(CONFIG),environment=environment(),fixed_denominator=FIXED,
            input_source=dict(path=str(input_root),status='PENDING',expected_source_sha=cfg['input']['source_sha']),
            method_version=receiver.PUBLIC.method_version,inputs={},underlying_observations={},mode_reads={},path_posthoc={},edge_posthoc={},
            cached_payload={},payload_evaluations={},calls={},failures=[],shared={},evidence_ceiling=cfg['evidence_ceiling'],
            actual_model_calls=False,actual_media_calls=False,comparisons={})
        for arm in ARMS:
            for stage in STAGES:
                role='NEW_CHANNEL_DIAGNOSTIC' if stage in NEW else 'OFFLINE_REFERENCE'
                self.data['underlying_observations'][arm+'/'+stage]=dict(status='PENDING',role=role,expected_keyed_projections=2)
                for key in KEYS:
                    sid=f'{arm}/{stage}/{key}'
                    self.data['inputs'][sid]=dict(status='PENDING',role=role,path=str(Path(input_root)/arm/(stage+'.'+key+'.raw.json.gz')),g=0,R=44)
                    self.data['cached_payload'][sid]=dict(status='PENDING',role='CACHED_EXISTING_PAYLOAD_NOT_NEW_READ',decoded_bits=None)
                    self.data['edge_posthoc'][sid]=dict(status='PENDING')
                    for target in TARGETS:self.data['payload_evaluations'][sid+'/'+target]=dict(status='PENDING',bit_errors=None,accepted_payload=False,readout_source='REUSED')
                    for mode in MODES:
                        mid=sid+'/'+mode;self.data['mode_reads'][mid]=dict(status='PENDING',role=role,mode=mode,g=0,R=44,path=str(self.output/arm/(stage+'.'+key+'.'+mode+'.json.gz')),
                            accepted_payload=False,state_path_accepted=False,summary=None)
                        self.data['path_posthoc'][mid]=dict(status='PENDING',accepted_payload=False,state_path_accepted=False)
        catalog=self.output/'catalog_R44.json.gz';dump_gzip(catalog,dict(rows=receiver.frozen.catalog(44)))
        for kid,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
            mask=np.ones((44,4),bool);absolute=receiver.frozen.family_receipt(key,44,mask);diff=receiver.difference_family(key,mask)
            family=dict(availability=mask.tolist(),edge_availability=diff['edge_availability'].tolist(),available_dimensions=diff['available_dimensions'],
                valid_catalog_indices=diff['valid_catalog_indices'],classes=diff['classes'],grouping='exact masked integer adjacent source-state differences;44-window tau feasible sets retained')
            ap=self.output/('family_absolute_'+kid+'.json.gz');dp=self.output/('family_difference_'+kid+'.json.gz');dump_gzip(ap,absolute);dump_gzip(dp,family)
            self.data['shared'][kid]=dict(catalog_path=str(catalog),catalog_sha256=sha(catalog),absolute_family_path=str(ap),absolute_family_sha256=sha(ap),
                difference_family_path=str(dp),difference_family_sha256=sha(dp),valid_count=174,catalog_count=3915,absolute_classes=len(absolute['classes']),difference_classes=len(family['classes']))
        self.save()
    def save(self):
        d=self.data
        if any(len(d[k])!=n for k,n in SIZES.items()):raise ValueError('fixed table roster changed')
        complete=lambda r:r['status'] in ('COMPLETE','NO_ENERGY')
        d['counts']=dict(input_projections=sum(r['status']=='READ' for r in d['inputs'].values()),underlying_observations=sum(r['status']=='COMPLETE' for r in d['underlying_observations'].values()),
            path_score_records=sum(complete(r) for r in d['mode_reads'].values()),path_costs=sum(r.get('scored',0) for r in d['mode_reads'].values()),
            difference_edge_costs=sum(r.get('edge_scored',0) for r in d['mode_reads'].values()),path_posthoc=sum(r['status']=='EVALUATED' for r in d['path_posthoc'].values()),
            difference_edge_posthoc=sum(r['status']=='EVALUATED' for r in d['edge_posthoc'].values()),cached_payload_readouts=sum(r['status']=='REUSED' for r in d['cached_payload'].values()),
            cached_payload_evaluations=sum(r['status']=='EVALUATED' for r in d['payload_evaluations'].values()),absolute_control_matches=sum(r.get('saved_absolute_comparison',{}).get('status')=='MATCH' for r in d['inputs'].values()))
        dump(self.path,d)
    def count(self,kind,done):
        r=self.data['calls'].setdefault(kind,dict(attempted=0,completed=0));r['completed' if done else 'attempted']+=1;self.save()
    def call(self,kind,operation):
        self.count(kind,False);value=operation();self.count(kind,True);return value
    def fail(self,where,error):
        self.data['failures'].append(dict(stage=where,error=f'{type(error).__name__}: {error}'));self.save()
    def snapshots(self):
        for role,name in [('NEW_CHANNEL_DIAGNOSTIC','receiver_channel_readouts.json'),('OFFLINE_REFERENCE','receiver_reference_readouts.json')]:
            rows={mid:r for mid,r in self.data['mode_reads'].items() if r['role']==role};inputs={sid:r for sid,r in self.data['inputs'].items() if r['role']==role}
            dump(self.output/name,dict(inputs=inputs,mode_reads=rows,shared=self.data['shared'],truth_inputs=False,
                role=role,claim='cached-observation receiver diagnostic only; no new blind-video or payload success'))
        dump(self.output/'cached_payload_readouts.json',dict(readouts=self.data['cached_payload'],truth_inputs=False,role='REUSED unchanged earlier readout; never used for state path selection'))


def process(store,cfg,input_root):
    path=Path(input_root)/'result.json';meta=None
    try:
        meta=json.loads(path.read_text())
        if meta['source_sha']!=cfg['input']['source_sha'] or meta['method_version']!=receiver.PUBLIC.method_version:raise ValueError('fixed input source/method mismatch')
        store.data['input_source'].update(status='READ',result_sha256=sha(path),source_sha=meta['source_sha'],
            fixed_denominator=meta['fixed_denominator'],reference_environment_limit=meta['reference_environment_limit'],baseline=meta['baseline'],vae_setup=meta['vae_setup']);store.save()
    except Exception as exc:meta=None;store.data['input_source'].update(status='FAILED',error=str(exc));store.fail('INPUT_METADATA',exc)
    keys=dict(CORRECT=cfg['key'],WRONG=cfg['wrong_key']);mask=np.ones((44,4),bool)
    for sid,row in store.data['inputs'].items():
        arm,stage,kid=sid.split('/');store.data['stage']='READ_'+sid;store.save()
        try:
            if meta is None:raise RuntimeError('input source metadata unavailable')
            old=meta['reads'][sid];p=Path(input_root)/arm/(stage+'.'+kid+'.raw.json.gz')
            if old['status'] not in ('COMPLETE','NO_ENERGY'):raise RuntimeError('source readout unavailable')
            if sha(p)!=old['sha256']:raise ValueError('saved raw identity mismatch')
            raw=store.call('source_raw_read',lambda:json.loads(gzip.decompress(p.read_bytes())))
            q=np.asarray(raw['inference']['projection'],dtype=np.float64)
            receiver.validate(q,mask)
            if raw['g']!=0 or raw['R']!=44 or q.shape!=(44,4,4,8) or raw['truth_inputs'] is not False:raise ValueError('fixed raw geometry/protocol mismatch')
            row.update(status='READ',sha256=old['sha256'],projection_shape=list(q.shape),projection_fingerprint=hashlib.sha256(q.tobytes()).hexdigest(),public_availability=mask.tolist())
            try:
                payload=raw['payload'];bits=payload.get('decoded_bits') if payload else None
                if payload is None or payload.get('status')!='READ' or not isinstance(bits,list) or len(bits)!=32 or not all(v in (0,1) for v in bits):raise ValueError('cached32-bit payload unavailable')
                store.data['cached_payload'][sid].update(status='REUSED',decoded_bits=bits,readout=payload,source_raw_sha256=old['sha256'],not_new_payload_read=True)
                store.count('cached_payload_reuse',False);store.count('cached_payload_reuse',True);store.save()
            except Exception as exc:
                store.data['cached_payload'][sid].update(status='FAILED',error=str(exc));store.fail(sid+'/CACHED_PAYLOAD',exc)
            for mode in MODES:
                mid=sid+'/'+mode;mr=store.data['mode_reads'][mid]
                try:
                    fn=receiver.absolute_control if mode==MODES[0] else receiver.infer_difference
                    inf=store.call('absolute_control' if mode==MODES[0] else 'difference_infer',lambda:fn(q,keys[kid],mask))
                    if inf['summary']['status'] not in ('COMPLETE','NO_ENERGY') or inf['counts']['scored']!=174:raise ValueError('incomplete fixed path costs')
                    if mode==MODES[0]:
                        fields=('summary','available_dimensions','received_length','valid_catalog_indices','class_costs','path_costs','projection','counts','local_state')
                        comparison={field:inf.get(field)==raw['inference'].get(field) for field in fields}
                        row['saved_absolute_comparison']=dict(status='MATCH' if all(comparison.values()) else 'DIFFERENT',fields=comparison)
                    output=dict(mode=mode,g=0,R=44,truth_inputs=False,role=row['role'],inference=inf,input_projection_fingerprint=row['projection_fingerprint'])
                    dump_gzip(mr['path'],output);mr.update(status=inf['summary']['status'],sha256=sha(mr['path']),summary=inf['summary'],scored=174,
                        edge_scored=inf['edge_local']['scored'] if mode==MODES[1] else 0)
                    store.save();store.snapshots()
                except Exception as exc:mr.update(status='FAILED',error=str(exc));store.fail(mid,exc)
        except Exception as exc:row.update(status='FAILED',error=str(exc));store.fail(sid,exc)
        store.save()
    settle(store,'saved input/readout unavailable');store.snapshots()


def settle(store,reason):
    for group in ('inputs','mode_reads','cached_payload'):
        for row in store.data[group].values():
            if row['status']=='PENDING':row.update(status='NOT_COMPLETED',error=reason)
    for aid,row in store.data['underlying_observations'].items():
        statuses=[store.data['inputs'][aid+'/'+key]['status'] for key in KEYS]
        row.update(status='COMPLETE' if statuses==['READ','READ'] else 'PARTIAL' if 'READ' in statuses else 'NOT_COMPLETED',keyed_input_status=dict(zip(KEYS,statuses)))
    store.save()


def evaluate(store,cfg):
    store.snapshots();paths=[store.output/n for n in ('receiver_channel_readouts.json','receiver_reference_readouts.json','cached_payload_readouts.json')]
    sealed={str(p):sha(p) for p in paths};catalog=receiver.frozen.catalog(44);truth=list(range(1,45));true_id=next(i for i,row in enumerate(catalog) if row['structurally_valid'] and row['taus']==truth)
    for mid,mr in store.data['mode_reads'].items():
        out=store.data['path_posthoc'][mid];inf=None
        if mr['status'] in ('COMPLETE','NO_ENERGY'):
            inf=json.loads(gzip.decompress(Path(mr['path']).read_bytes()))['inference'];ids=inf['valid_catalog_indices'];costs=np.asarray(inf['path_costs']);j=ids.index(true_id);tc=float(costs[j]);canonical=inf['summary']['canonical_catalog_index']
            out.update(status='EVALUATED',registered_tau=truth,true_catalog_index=true_id,true_path_cost=tc,true_path_rank=1+int(np.sum(costs<tc-receiver.PUBLIC.tie_atol)),
                true_path_delta=float(np.delete(costs,j).min()-tc),truth_in_top=true_id in inf['summary']['top_catalog_indices'],canonical_path_matches=canonical==true_id,
                canonical=None if canonical is None else catalog[canonical],top_catalog_indices=inf['summary']['top_catalog_indices'],
                claim='known unedited g0 posthoc only; no physical time synchronization acceptance')
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
                    rows.append(dict(received_edge_index=j+1,received_window_pair=[j+1,j+2],true_transition=target,true_pair_index=idx,status='EVALUATED',
                        true_rank=1+int(np.sum(c<tc-receiver.PUBLIC.tie_atol)),delta=float(np.delete(c,idx).min()-tc),truth_in_top=idx in top,
                        unique_truth=top==[idx],top_pair_indices=top,top_transitions=[pairs[i] for i in top]))
                edge.update(status='EVALUATED',rows=rows,unique_truth_edges=sum(r.get('unique_truth',False) for r in rows),truth_in_top_edges=sum(r.get('truth_in_top',False) for r in rows),
                    role='posthoc original adjacent transitions only; stay templates remain source-position ambiguous')
        store.save()
    for sid,cache in store.data['cached_payload'].items():
        bits=cache['decoded_bits']
        for tid,message in [('REGISTERED',cfg['message']),('WRONG_MESSAGE',cfg['wrong_message'])]:
            target=message_bits(message);out=store.data['payload_evaluations'][sid+'/'+tid]
            out.update(status='EVALUATED' if cache['status']=='REUSED' else 'MISSING_READ',bit_errors=sum(a!=b for a,b in zip(bits,target)) if bits is not None else None,
                exact_bits=bits==target if bits is not None else None,payload_source='REUSED',not_new_payload_recovery=True,accepted_payload=False,
                claim='same cached repeated payload; never used for state path selection')
    for sid in store.data['inputs']:
        a=store.data['path_posthoc'][sid+'/'+MODES[0]];d=store.data['path_posthoc'][sid+'/'+MODES[1]]
        store.data['comparisons'][sid]=dict(absolute={k:a.get(k) for k in ('status','true_path_rank','true_path_delta','canonical_path_matches')},difference={k:d.get(k) for k in ('status','true_path_rank','true_path_delta','canonical_path_matches')},
            note='Different score normalizations; absolute levels and43 correlated differences are paired diagnostics, not independent success counts')
    if sealed!={str(p):sha(p) for p in paths}:raise RuntimeError('posthoc modified receiver/cache evidence')
    store.data['receiver_sha256']=sealed;store.save()


def finish(store,cfg):
    expected=dict(input_projections=24,underlying_observations=12,path_score_records=48,path_costs=8352,difference_edge_costs=136224,
        path_posthoc=48,difference_edge_posthoc=24,cached_payload_readouts=24,cached_payload_evaluations=48,absolute_control_matches=24)
    calls={kind:dict(expected=n,actual=store.data['calls'].get(kind,dict(attempted=0,completed=0)),match=store.data['calls'].get(kind)==dict(attempted=n,completed=n)) for kind,n in cfg['planned_calls'].items()}
    store.data['call_integrity']=dict(status='MATCH' if all(r['match'] for r in calls.values()) else 'INCOMPLETE',rows=calls)
    done=store.data['counts']==expected and store.data['call_integrity']['status']=='MATCH'
    store.data.update(status='EXECUTION_COMPLETE' if done else 'INCOMPLETE',stage='FINISHED',science_status='UNCALIBRATED_DIFFERENCE_DIAGNOSTIC_NO_AUTOMATIC_PASS')
    store.save();print(json.dumps(dict(status=store.data['status'],counts=store.data['counts'])));return done


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--input',type=Path);a=p.parse_args();cfg=load_config()
    source=a.input or Path(cfg['input']['path']);store=Store(a.output,source)
    try:process(store,cfg,source)
    except Exception as exc:store.fail(store.data['stage'],exc);settle(store,str(exc));store.snapshots()
    evaluate(store,cfg)
    if not finish(store,cfg):raise SystemExit(1)

if __name__=='__main__':main()
