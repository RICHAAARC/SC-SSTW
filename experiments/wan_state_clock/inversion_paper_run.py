"""User-run minimal calibration/evaluation workflow. No execution on import."""
import argparse
import copy
import gc
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from main.tube_state import inversion_state as writer
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.flow_generation import continue_steps
from runtime.wan.flow_inversion import public_schedule,invert_received_latent
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,_clear_cache
from runtime.wan.io import dump,read_mp4,encode_rgb
from .inversion_crop_observation_run import encode_lossless,same_schedule
from .flow_run import quality
from . import inversion_paper_protocol as protocol

MANIFEST=Path(__file__).parent/'configs/inversion_paper.json'
RGB=(320,512,3)


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def arms(case):return ('OFF',) if case in protocol.CAL else protocol.ARMS


def plan(n):return dict(generation_transformer=100*n,scheduler_step=50*n,vae_decode=n,vae_encode=3*n,
    mp4_save=n,crop_save=2*n,inversion_transformer=300*n,inverse_update=150*n)


def empty(case,status='NOT_RUN'):
    return dict(status=status,case=case,source_denominator=len(arms(case)),view_denominator=3*len(arms(case)),sources={
        arm:dict(status='NOT_RUN',generation_status='NOT_RUN',views={v:dict(status='NOT_RUN',media_status='NOT_RUN',
            inversion_status='NOT_RUN',readout={'status':'NOT_RUN','decoder_readouts':{d:{'status':'NOT_RUN','candidate_denominator':28,
                'candidates':[dict(shift=s,message=m,status='NOT_SCORED') for s in range(14) for m in (0,1)]} for d in ('STATE','STATIC')}},
            quality={'status':'NOT_RUN'},decisions={}) for v in protocol.VIEWS}) for arm in arms(case)})


def generate_case(config,books,output,result,count,save,fail):
    pipe=initial=prompt=negative=origin=marked=terminal=None
    try:
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False);origin=copy.deepcopy(pipe.scheduler)
        torch.save(initial.cpu(),output/'initial.pt');dump(output/'schedule.json',public_schedule(origin))
        result['generation_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        for arm,row in result['sources'].items():
            try:
                marked=initial.clone() if arm=='OFF' else writer.write(initial,books[arm.split('_')[0]],0 if arm.endswith('_A') else 1)
                torch.save(marked.cpu(),output/'writer'/f'{arm}_initial.pt')
                pipe.scheduler=copy.deepcopy(origin)
                def mapped(k,d):count('generation_transformer' if k=='transformer' else k,d)
                precision={}
                terminal=continue_steps(pipe,pipe.scheduler,marked,prompt,negative,dtype,config['generation']['guidance_scale'],0,50,mapped,precision=precision)
                row['generation_precision']=precision
                torch.save(terminal.cpu(),output/'writer'/f'{arm}_terminal.pt');row['generation_status']='COMPLETE'
            except Exception as exc:row['generation_status']='FAILED';fail(arm+'/generation',exc)
            finally:marked=terminal=None;release();save()
    finally:pipe=initial=prompt=negative=origin=marked=terminal=None;release()


def media_case(config,output,result,count,save,fail):
    vae=latent=rgb=received=encoded=clip=off=None
    try:
        if not any(r['generation_status']=='COMPLETE' for r in result['sources'].values()):return
        vae=load_frozen_vae(config)
        for arm,row in result['sources'].items():
            if row['generation_status']!='COMPLETE':continue
            try:
                latent=torch.load(output/'writer'/f'{arm}_terminal.pt',weights_only=True,map_location='cpu')
                count('vae_decode',False);rgb=decode_normalized_latent(vae,latent.to(next(vae.parameters()).device)).cpu();count('vae_decode',True)
                if tuple(rgb.shape)!=(181,*RGB) or not bool(torch.isfinite(rgb).all()):raise ValueError('full decoded RGB geometry/nonfinite failure')
                full=output/'videos'/f'{arm}_full181.mp4';count('mp4_save',False);encode_rgb(rgb,full,8,18);count('mp4_save',True)
                rgb=read_mp4(full)
                if tuple(rgb.shape)!=(181,*RGB) or not bool(torch.isfinite(rgb).all()):raise ValueError('saved RGB geometry/nonfinite failure')
                for view in protocol.VIEWS:
                    item=row['views'][view]
                    try:
                        path=output/'videos'/f'{arm}_{view}.mp4'
                        if view!='full181':
                            start=int(view[4:]);clip=rgb[start:start+129].clone()
                            count('crop_save',False);encode_lossless(clip,path);count('crop_save',True)
                            received=read_mp4(path)
                            if not torch.equal(received,clip):raise ValueError('lossless crop pixel equality failed')
                        else:received=read_mp4(path)
                        item['mp4_sha256']=sha(path)
                        offpath=output/'videos'/f'OFF_{view}.mp4'
                        try:
                            if offpath.exists():
                                off=read_mp4(offpath);item['quality']={'status':'MEASURED',**quality(off,received)};off=None
                            else:item['quality']={'status':'MISSING_OFF'}
                        except Exception as exc:item['quality']={'status':'FAILED','error':repr(exc)};fail(arm+'/'+view+'/quality',exc)
                        count('vae_encode',False);encoded=reencode_rgb24_readback(vae,received).cpu().float();count('vae_encode',True)
                        expected=(1,16,46 if view=='full181' else 33,40,64)
                        if tuple(encoded.shape)!=expected or not torch.isfinite(encoded).all():raise ValueError('view latent geometry/finite check failed')
                        torch.save(encoded,output/'receiver'/f'{arm}_{view}_encoded.pt');item['media_status']='COMPLETE'
                    except Exception as exc:item['media_status']='FAILED';fail(arm+'/'+view+'/media',exc)
                    finally:received=encoded=clip=off=None;_clear_cache(vae);release();save()
            except Exception as exc:fail(arm+'/media',exc)
            finally:latent=rgb=None;release()
    finally:
        if vae is not None:_clear_cache(vae)
        vae=latent=rgb=received=encoded=clip=off=None;release()


def inverse_case(config,books,output,result,count,save,fail):
    schedule=load(output/'schedule.json')
    for frames,views in [(181,('full181',)),(129,('crop16','crop17'))]:
        pipe=dummy=prompt=negative=received=recovered=None
        try:
            if not any(r['views'][v]['media_status']=='COMPLETE' for r in result['sources'].values() for v in views):continue
            public=dict(model=copy.deepcopy(config['model']),generation=copy.deepcopy(config['generation']))
            public['generation'].update(seed=0,frames=frames,role='paper_receiver_public_conditions')
            pipe,dummy,prompt,negative,dtype=prepare_generation(public,load_vae=False);dummy=None
            if not same_schedule(public_schedule(pipe.scheduler),schedule):raise ValueError('receiver schedule differs')
            result.setdefault('receiver_model_revisions',{})[str(frames)]=getattr(pipe.transformer.config,'_commit_hash',None)
            for arm,row in result['sources'].items():
                for view in views:
                    item=row['views'][view]
                    if item['media_status']!='COMPLETE':continue
                    try:
                        received=torch.load(output/'receiver'/f'{arm}_{view}_encoded.pt',weights_only=True,map_location='cpu')
                        def mapped(k,d):count('inversion_transformer' if k=='transformer' else k,d)
                        item['inverse_steps']=[]
                        recovered=invert_received_latent(pipe.transformer,received.to(next(pipe.transformer.parameters()).device),
                            prompt,negative,dtype,public['generation']['guidance_scale'],schedule['sigmas'],schedule['timesteps'],mapped,item['inverse_steps'].append).cpu().float()
                        path=output/'receiver'/f'{arm}_{view}_recovered.pt';torch.save(recovered,path)
                        item['recovered_sha256']=sha(path);item['inversion_status']='COMPLETE'
                        item['readout']=protocol.score(recovered,books)
                        dump(output/'readouts'/f'{arm}_{view}.json',item['readout']);item['status']='COMPLETE'
                    except Exception as exc:item['inversion_status']='FAILED';fail(arm+'/'+view+'/inverse',exc)
                    finally:received=recovered=None;release();save()
        finally:pipe=dummy=prompt=negative=received=recovered=None;release()


def run_case(case_id,output,threshold_path=None,threshold_sha=None):
    # Eval cannot start any model load before the frozen threshold file is checked.
    frozen=None
    if case_id in protocol.EVAL:
        if threshold_path is None or threshold_sha is None or sha(threshold_path)!=threshold_sha:raise ValueError('frozen pre-evaluation threshold hash required')
        frozen=load(threshold_path)
        if frozen.get('status')!='FROZEN':raise ValueError('threshold freeze record required')
    manifest=load(MANIFEST);split='calibration' if case_id in protocol.CAL else 'evaluation'
    case=next(c for c in manifest[split] if c['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for folder in ('writer','receiver','videos','readouts'):(output/folder).mkdir()
    result=empty(case_id,'RUNNING');result.update(config=config,split=split,stages={},failures=[],fixed_calls=plan(len(arms(case_id))),
        actual_calls={k+'_'+d:0 for k in plan(len(arms(case_id))) for d in ('attempted','completed')},threshold_sha256=threshold_sha,scientific_pass=None)
    def save():dump(output/'result.json',result)
    def count(k,d):result['actual_calls'][k+('_completed' if d else '_attempted')]+=1
    def fail(stage,exc):result['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    books=protocol.books(config['key_utf8'].encode());torch.save(books,output/'books.pt');dump(output/'config.json',config)
    result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    import diffusers
    result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
    save()
    for stage,fn in [('generation',lambda:generate_case(config,books,output,result,count,save,fail)),
                     ('media',lambda:media_case(config,output,result,count,save,fail)),
                     ('inversion',lambda:inverse_case(config,books,output,result,count,save,fail))]:
        start=time.monotonic()
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        try:fn()
        except Exception as exc:fail(stage+'/setup',exc)
        finally:
            release();result['stages'][stage]=dict(wall_seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None);save()
    for arm,row in result['sources'].items():
        for view,item in row['views'].items():
            if item['status']!='COMPLETE':item['status']='MISSING_OR_FAILED'
            if frozen is not None:item['decisions']=protocol.decide(item['readout'],frozen)
            item['reporting_only']=protocol.report(item['readout'],arm,view,books)
        row['status']='COMPLETE' if all(v['status']=='COMPLETE' for v in row['views'].values()) else 'WITH_RETAINED_FAILURES'
    result['status']='EXECUTION_COMPLETE' if not result['failures'] and all(r['status']=='COMPLETE' for r in result['sources'].values()) else 'WITH_RETAINED_FAILURES'
    result['artifact_sha256']={p.name:sha(p) for p in (output/'config.json',output/'books.pt',output/'schedule.json') if p.exists()}
    save();return result


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',source_denominator=12,view_denominator=36,fixed_calls=plan(12),
        calibration={c:empty(c) for c in protocol.CAL},evaluation={c:empty(c) for c in protocol.EVAL},scientific_pass=None)
    dump(output/'protocol.json',load(MANIFEST));dump(output/'result.json',result)
    frozen_path=output/'threshold.json';frozen_hash=None
    for split,cases in [('calibration',protocol.CAL),('evaluation',protocol.EVAL)]:
        if split=='evaluation':
            frozen=protocol.freeze(result['calibration'])
            with frozen_path.open('x') as stream:json.dump(frozen,stream,indent=2,allow_nan=False)
            frozen_hash=sha(frozen_path);result['threshold_sha256']=frozen_hash;result['thresholds']=frozen
            dump(output/'result.json',result)
        for case in cases:
            log=output/(case+'.log');print(split,case,'started; log:',log,flush=True)
            try:
                command=[sys.executable,'-u','-m','experiments.wan_state_clock.inversion_paper_run','--output',str(output/case),'--case-id',case]
                if split=='evaluation':command+=['--threshold',str(frozen_path),'--threshold-sha',frozen_hash]
                with log.open('w') as stream:child=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
                result[split][case]=load(output/case/'result.json')|{'exit_code':child.returncode}
            except Exception as exc:result[split][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(exc))
            dump(output/'result.json',result)
    if sha(frozen_path)!=frozen_hash:raise RuntimeError('frozen threshold changed during evaluation')
    result['summary']=protocol.summary(result['evaluation'])
    rows=[r for split in ('calibration','evaluation') for r in result[split].values()]
    result['actual_calls_observed']={k+'_'+d:sum(r.get('actual_calls',{}).get(k+'_'+d,0) for r in rows) for k in plan(12) for d in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(r['status']=='EXECUTION_COMPLETE' and r.get('exit_code')==0 for r in rows) else 'WITH_RETAINED_FAILURES'
    result['source_sha256']={str(p):sha(p) for p in (Path(__file__),Path(protocol.__file__),Path(writer.__file__),Path(protocol.scorer.__file__),MANIFEST)}
    dump(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-id',choices=protocol.CAL+protocol.EVAL)
    parser.add_argument('--threshold',type=Path);parser.add_argument('--threshold-sha');args=parser.parse_args()
    result=run_case(args.case_id,args.output,args.threshold,args.threshold_sha) if args.case_id else run(args.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
