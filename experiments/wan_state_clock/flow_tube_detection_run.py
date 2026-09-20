"""Source-independent OFF calibration followed by frozen LAST evaluation. User execution only."""
import argparse,copy,csv,gc,hashlib,json,platform,random,shutil,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import state_clock,terminal_guidance
from runtime.wan import tube_terminal_guidance
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.io import dump,read_mp4,encode_rgb
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,_clear_cache
from .flow_run import quality
from . import flow_tube_detection_protocol as protocol
from . import flow_tube_crop_analysis as crop_analysis

MANIFEST=Path(__file__).parent/'configs/flow_tube_detection.json'
SPATIAL=(320,512,3)
LENGTHS=dict(full181=181,crop0=129,crop4=129,crop5=129,delete90=180,speed125=145,resaved=181)

def load(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
def arms(case):return ('OFF',) if case in protocol.CALIBRATION else protocol.ARMS
def plan(case):
    n=len(arms(case))
    return dict(transformer=100,scheduler_step=49+n,vae_decode=n,mp4_save=4*n,crop_save=3*n,vae_encode=28*n,blind_read=7*n)
def empty(case):
    return dict(status='NOT_RUN',case=case,sources={a:dict(status='NOT_RUN',views={v:dict(status='NOT_RUN',observations={str(g):{'status':'NOT_RUN'} for g in range(4)},statistic={'status':'UNMEASURED'}) for v in protocol.VIEWS},sequence={'status':'UNMEASURED'}) for a in arms(case)},failures=[])
def validate(m):
    if tuple(x['id'] for x in m['calibration'])!=protocol.CALIBRATION or tuple(x['id'] for x in m['evaluation'])!=protocol.EVALUATION:raise ValueError('fixed independent source roster required')
    if m['primary_mode']!=protocol.MODE or tuple(m['views'])!=protocol.VIEWS or tuple(m['sequence_views'])!=protocol.SEQUENCE or m['calibration_sources']!=9:raise ValueError('fixed statistical protocol required')
    rows=m['calibration']+m['evaluation']
    if len({v['seed'] for v in rows})!=11 or len({v['prompt'] for v in rows})!=11:raise ValueError('independent calibration/evaluation sources required')
    if m['calibration_arms']!=['OFF'] or m['evaluation_arms']!=list(protocol.ARMS):raise ValueError('fixed arms required')
def threshold_check(case,path,digest):
    if case in protocol.CALIBRATION:return None
    if path is None or digest is None or sha(path)!=digest:raise ValueError('frozen pre-evaluation threshold hash required')
    frozen=load(path)
    if frozen.get('status')!='FROZEN' or frozen.get('source_denominator')!=9:raise ValueError('invalid frozen calibration record')
    return frozen
def check_rgb(rgb,length):
    if tuple(rgb.shape)!=(length,*SPATIAL) or not torch.isfinite(rgb).all() or bool(((rgb<0)|(rgb>1)).any()):raise ValueError('invalid finite received RGB geometry/range')

def generate(case,output,threshold_path=None,threshold_sha=None):
    threshold_check(case,threshold_path,threshold_sha)  # Before any eval model load.
    m=load(MANIFEST);validate(m);entry=next(v for v in m['calibration']+m['evaluation'] if v['id']==case)
    config=copy.deepcopy(m['base_config']);config['generation'].update(prompt=entry['prompt'],seed=entry['seed'])
    out=Path(output);out.mkdir(parents=True,exist_ok=False);r=empty(case)
    r.update(status='RUNNING',fixed_calls=plan(case),actual_calls={k+'_'+s:0 for k in plan(case) for s in ('attempted','completed')},file_sha256={},threshold_sha256=threshold_sha,scientific_pass=None)
    def save():dump(out/'generation.json',r)
    def count(k,done):r['actual_calls'][k+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,e):r['failures'].append(dict(stage=stage,error=repr(e),traceback=traceback.format_exc()));save()
    book=state_clock.codebook(config['key_utf8'].encode());dump(out/'config.json',config);dump(out/'protocol.json',m);np.savez(out/'book.npz',**book)
    for n in ('config.json','protocol.json','book.npz'):r['file_sha256'][n]=sha(out/n)
    pipe=initial=positive=negative=z=v=snapshot=terminal=None
    save()
    try:
        import diffusers
        r['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        r['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        pipe,initial,positive,negative,dtype=prepare_generation(config,load_vae=False)
        r['resolved_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        z,v,snapshot,precision=tube_terminal_guidance.prepare_last(pipe,initial,positive,negative,dtype,config['generation']['guidance_scale'],count)
        r['precision']=precision;r['history_fingerprint']=tube_terminal_guidance.fingerprint(vars(snapshot))
        dump(out/'schedule.json',dict(config=dict(snapshot.config),sigmas=snapshot.sigmas.tolist(),timesteps=snapshot.timesteps.tolist()))
        clean=z-float(snapshot.sigmas[49])*v
        for arm in arms(case):
            try:
                u=None
                if arm!='OFF':
                    requested,_,evidence=terminal_guidance.correction(clean.cpu().numpy(),book,0 if arm=='LAST_A' else 1)
                    u=torch.from_numpy(requested).to(z.device);r['sources'][arm]['requested_margin']=evidence['minimum_signed_projection_after']
                terminal,step=tube_terminal_guidance.final_step(snapshot,z,v,count,u=u)
                p=out/(arm+'_terminal.pt');torch.save(terminal.cpu(),p);r['file_sha256'][p.name]=sha(p)
                r['sources'][arm].update(status='TERMINAL_PERSISTED',step=step)
            except Exception as e:r['sources'][arm]['status']='FAILED_GENERATION';fail(arm,e)
            finally:terminal=u=None;release();save()
        if tube_terminal_guidance.fingerprint(vars(snapshot))!=r['history_fingerprint']:raise RuntimeError('shared history mutated')
    except Exception as e:fail('generation',e)
    finally:pipe=initial=positive=negative=z=v=snapshot=terminal=None;release()
    r['status']='GENERATION_COMPLETE' if all(s['status']=='TERMINAL_PERSISTED' for s in r['sources'].values()) and not r['failures'] else 'WITH_RETAINED_FAILURES'
    save();return r

def frame_indices(view):
    """Actual attack map used only to construct RGB and post-read geometry reports."""
    if view.startswith('crop'):
        start=int(view[4:]);return list(range(start,start+129))
    if view=='delete90':return list(range(90))+list(range(91,181))
    if view=='speed125':return [int(np.floor(i*1.25+.5)) for i in range(145)]
    if view in ('full181','resaved'):return list(range(181))
    raise ValueError(view)

def transformed(rgb,view):
    return rgb[frame_indices(view)].clone()

def sync_reporting(detection,view):
    """Maps observed RGB supports; never supplies an attack map to the receiver."""
    mapping=frame_indices(view);rank=detection['rankings'][protocol.MODE];best=rank.get('best')
    detail=crop_analysis.class_detail(detection,best['class']) if best else None;windows=[]
    if detail:
        for n,(origin,selected,valid) in enumerate(zip(detail['origins'],detail['selected'],detail['valid'])):
            groups=[]
            for j in selected:
                received=[] if j is None else list(range(origin+1+4*j,origin+5+4*j))
                groups.append(dict(received_rgb_indices=received,source_rgb_indices=[mapping[i] for i in received if i<len(mapping)]))
            windows.append(dict(reference_window=n,valid=valid,origin=origin,selected_groups=selected,actual_supports=groups))
    result=dict(received_to_source_frame_indices=mapping,selected_window_geometry=windows,
                top_tie_observation_classes=len({v['class'] for v in rank['top_ties']}),
                meaning='nominal group RGB supports, not VAE receptive fields or inverse-latent equivalence; truth joins after read')
    if view.startswith('crop'):
        result['crop_complete_window_reference']=crop_analysis.crop_alignment_reporting_only(detection,rank,int(view[4:]))
    elif view in ('full181','resaved'):
        result['nominal_reference']=dict(g=0,scale=[1,1],offset=0,delta=0,boundary=11)
    else:
        result['exact_clock_reference_available']=False
        result['synchronization_success']=None
        result['reason']='single-frame deletion or rounded speed map is not an exact affine clock/VAE inverse; inspect observable support correspondence only'
    return result

def encode_receive(vae,rgb,count,store,fail):
    obs={};rows={str(g):{'status':'NOT_RUN'} for g in range(4)}
    for g in range(4):
        encoded=None
        try:
            groups,tail=divmod(len(rgb)-g-1,4);view=rgb[g:g+1+4*groups]
            count('vae_encode',False);encoded=reencode_rgb24_readback(vae,view).cpu().float();count('vae_encode',True)
            if tuple(encoded.shape)!=(1,16,groups+1,40,64) or not torch.isfinite(encoded).all():raise ValueError('invalid receiver latent')
            store(g,encoded);obs[g]=encoded.numpy();rows[str(g)]=dict(status='COMPLETE',frames_used=len(view),tail_discarded=tail,local_origin=g)
        except Exception as e:rows[str(g)]=dict(status='FAILED',error=repr(e));fail('origin'+str(g),e)
        finally:encoded=None;_clear_cache(vae);release()
    return obs,rows

def media(case,output,threshold_path=None,threshold_sha=None):
    frozen=threshold_check(case,threshold_path,threshold_sha);out=Path(output);r=load(out/'generation.json');config=load(out/'config.json')
    for n in ('config.json','protocol.json','book.npz'):
        if sha(out/n)!=r['file_sha256'].get(n):raise ValueError('source metadata mismatch '+n)
    if load(out/'protocol.json')!=load(MANIFEST):raise ValueError('source/current protocol mismatch')
    if r.get('source_commit')!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip():raise ValueError('generation/media source commit mismatch')
    if r.get('threshold_sha256')!=threshold_sha:raise ValueError('generation/media threshold mismatch')
    book={k:v for k,v in np.load(out/'book.npz').items()};vae=None
    def save():dump(out/'result.json',r)
    def count(k,done):r['actual_calls'][k+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,e):r['failures'].append(dict(stage=stage,error=repr(e),traceback=traceback.format_exc()));save()
    def persisted(p):r['file_sha256'][str(p.relative_to(out))]=sha(p)
    save()
    try:
        vae=load_frozen_vae(config);r['resolved_vae_revision']=getattr(vae.config,'_commit_hash',None)
        for arm in arms(case):
            source=r['sources'][arm];full=z=rgb=None
            try:
                p=out/(arm+'_terminal.pt')
                if sha(p)!=r['file_sha256'].get(p.name):raise ValueError('missing/changed terminal')
                z=torch.load(p,map_location='cpu',weights_only=True);count('vae_decode',False)
                rgb=decode_normalized_latent(vae,z.to(next(vae.parameters()).device)).cpu();count('vae_decode',True);check_rgb(rgb,181)
                p=out/'videos'/arm/'full181.mp4';count('mp4_save',False);encode_rgb(rgb,p,8,18);count('mp4_save',True);persisted(p)
                full=read_mp4(p);check_rgb(full,181)
            except Exception as e:
                source['status']='FAILED_SOURCE_MEDIA';fail(arm+'/base_media',e)
                for row in source['views'].values():row['status']='NOT_RUN_FAILED_SOURCE'
                continue
            finally:z=rgb=None;_clear_cache(vae);release();save()
            for view in protocol.VIEWS:
                row=source['views'][view];received=obs=detection=altered=None
                try:
                    folder=out/'views'/arm/view;folder.mkdir(parents=True,exist_ok=False)
                    if view=='full181':received=full;path=out/'videos'/arm/'full181.mp4'
                    elif view.startswith('crop'):
                        altered=transformed(full,view);path=folder/'rgb8.npy';count('crop_save',False)
                        np.save(path,(altered*255).round().to(torch.uint8).numpy(),allow_pickle=False);count('crop_save',True);persisted(path)
                        pixels=np.load(path,allow_pickle=False)
                        if pixels.dtype!=np.uint8:raise ValueError('crop must be original uint8')
                        received=torch.from_numpy(pixels).float()/255.
                    else:
                        altered=transformed(full,view);path=out/'videos'/arm/(view+'.mp4');count('mp4_save',False)
                        encode_rgb(altered,path,8,18);count('mp4_save',True);persisted(path);received=read_mp4(path)
                    if sha(path)!=r['file_sha256'].get(str(path.relative_to(out))):raise ValueError('saved/read media hash mismatch')
                    check_rgb(received,LENGTHS[view]);row['received_path']=str(path.relative_to(out));row['source_frames']=len(received);row['media_readback_valid']=True
                    if arm!='OFF':
                        offrow=r['sources']['OFF']['views'][view];reference_path=out/offrow.get('received_path','__missing__')
                        if reference_path.is_file() and offrow.get('media_readback_valid') is True:
                            reference=torch.from_numpy(np.load(reference_path,allow_pickle=False)).float()/255. if reference_path.suffix=='.npy' else read_mp4(reference_path)
                            check_rgb(reference,LENGTHS[view]);row['quality_vs_OFF']=quality(reference,received);reference=None
                        else:row['quality_vs_OFF']={'status':'MISSING_OFF_REFERENCE'}
                    def store(g,tensor):
                        p=folder/('g'+str(g)+'.pt');torch.save(tensor,p);persisted(p)
                    obs,observations=encode_receive(vae,received,count,store,lambda stage,e:fail(arm+'/'+view+'/'+stage,e));row['observations']=observations
                    count('blind_read',False);detection=state_clock.read(obs,book);count('blind_read',True)
                    dump(folder/'blind.json',detection);persisted(folder/'blind.json')
                    row['statistic']=protocol.statistic(detection,complete=len(obs)==4)
                    # Existence scoring has completed and been persisted before truth/attack joins.
                    row['sync_reporting_only']=sync_reporting(detection,view)
                    truth=None if arm=='OFF' else (0 if arm=='LAST_A' else 1)
                    row['attribution']=protocol.attribution(row['statistic'],truth)
                    if frozen is not None:row['decision']=protocol.decide(row['statistic'],frozen)
                    row['status']='COMPLETE' if len(obs)==4 else 'PARTIAL_OR_FAILED'
                except Exception as e:row['status']='FAILED';fail(arm+'/'+view,e)
                finally:received=obs=detection=altered=None;release();save()
            source['sequence']=protocol.sequence(source['views'])
            source['sequence']['attribution']=protocol.attribution(source['sequence'],None if arm=='OFF' else (0 if arm=='LAST_A' else 1))
            if frozen is not None:source['sequence']['decision']=protocol.decide(source['sequence'],frozen)
            source['status']='COMPLETE' if all(v['status']=='COMPLETE' for v in source['views'].values()) else 'PARTIAL_OR_FAILED'
            full=None;release();save()
    except Exception as e:fail('media_setup',e)
    finally:vae=None;release()
    r['status']='EXECUTION_COMPLETE' if all(s['status']=='COMPLETE' for s in r['sources'].values()) and not r['failures'] else 'WITH_RETAINED_FAILURES'
    save();return r

def review_package(output,evaluation):
    """Anonymous evaluation base + three attacked complete videos; no auto quality ratings."""
    out=Path(output);folder=out/'blind_review';folder.mkdir(exist_ok=False)
    roster=[(c,a,v) for c in protocol.EVALUATION for a in protocol.ARMS for v in ('full181','delete90','speed125','resaved')]
    random.Random(81357).shuffle(roster);mapping=[];scores=[]
    for i,(case,arm,view) in enumerate(roster,1):
        name='clip%02d.mp4'%i;source=evaluation.get(case,{}).get('sources',{}).get(arm,{}).get('views',{}).get(view,{})
        rel=source.get('received_path');path=out/case/rel if rel else None;status='MISSING'
        if path and path.is_file() and source.get('media_readback_valid') is True:shutil.copyfile(path,folder/name);status='AVAILABLE'
        mapping.append(dict(anonymous_file=name,case=case,arm=arm,view=view,status=status))
        scores.append(dict(anonymous_file=name,availability=status,image_quality='PENDING',subject_integrity='PENDING',motion_smoothness='PENDING',notes=''))
    dump(out/'reporting_only_review_mapping.json',mapping)
    with (folder/'ratings.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(scores[0]));writer.writeheader();writer.writerows(scores)
    (folder/'README.txt').write_text('Review all available anonymous videos. Rate image quality, subject integrity and motion smoothness separately. PENDING is not a pass. Do not open reporting_only_review_mapping.json before blind ratings. No automated perceptual decision.\n')
    return dict(denominator=24,available=sum(v['status']=='AVAILABLE' for v in mapping),rating_status='PENDING_HUMAN_REVIEW')

def run(output):
    m=load(MANIFEST);validate(m);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    r=dict(status='RUNNING',source_denominator=15,view_denominator=105,receiver_encode_denominator=420,calibration={c:empty(c) for c in protocol.CALIBRATION},evaluation={c:empty(c) for c in protocol.EVALUATION},scientific_pass=None)
    r['fixed_calls']={k:sum(plan(c)[k] for c in (*protocol.CALIBRATION,*protocol.EVALUATION)) for k in plan('cal01')}
    dump(out/'protocol.json',m);dump(out/'result.json',r);threshold_path=out/'threshold.json';digest=None
    for split,roster in (('calibration',protocol.CALIBRATION),('evaluation',protocol.EVALUATION)):
        if split=='evaluation':
            frozen=protocol.freeze(r['calibration'])
            with threshold_path.open('x') as f:json.dump(frozen,f,indent=2,allow_nan=False)
            digest=sha(threshold_path);r['threshold']=frozen;r['threshold_sha256']=digest;dump(out/'result.json',r)
        for case in roster:
            for stage in ('generate','media'):
                logfile=out/(case+'_'+stage+'.log');print(case,stage,'started; log:',logfile,flush=True)
                cmd=[sys.executable,'-u','-m','experiments.wan_state_clock.flow_tube_detection_run','--output',str(out/case),'--case-id',case,'--stage',stage]
                if split=='evaluation':cmd+=['--threshold',str(threshold_path),'--threshold-sha',digest]
                try:
                    with logfile.open('w') as f:child=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=False)
                    previous=r[split][case];p=out/case/('generation.json' if stage=='generate' else 'result.json')
                    r[split][case]=load(p)|{k:v for k,v in previous.items() if k.endswith('_exit_code')}|{stage+'_exit_code':child.returncode}
                except Exception as e:r[split][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(e))
                dump(out/'result.json',r)
    if sha(threshold_path)!=digest:raise RuntimeError('threshold mutated during evaluation')
    r['summary']=protocol.summary(r['evaluation']);r['human_review']=review_package(out,r['evaluation'])
    rows=[v for split in ('calibration','evaluation') for v in r[split].values()]
    r['actual_calls']={k+'_'+s:sum(v.get('actual_calls',{}).get(k+'_'+s,0) for v in rows) for k in r['fixed_calls'] for s in ('attempted','completed')}
    r['status']='EXECUTION_COMPLETE' if all(v['status']=='EXECUTION_COMPLETE' and v.get('generate_exit_code')==0 and v.get('media_exit_code')==0 for v in rows) else 'WITH_RETAINED_FAILURES'
    dump(out/'result.json',r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case-id',choices=(*protocol.CALIBRATION,*protocol.EVALUATION));p.add_argument('--stage',choices=('generate','media'));p.add_argument('--threshold');p.add_argument('--threshold-sha');a=p.parse_args()
    if a.case_id and a.stage:r=(generate if a.stage=='generate' else media)(a.case_id,a.output,a.threshold,a.threshold_sha)
    elif not a.case_id and not a.stage:r=run(a.output)
    else:p.error('case and stage must be paired')
    if r['status'] not in ('GENERATION_COMPLETE','EXECUTION_COMPLETE'):raise SystemExit(1)
