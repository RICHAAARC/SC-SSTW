"""Fixed initial-noise inversion mechanism baseline; no model run on import."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import platform
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
import torch
from main.tube_state import initial_noise as method
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.flow_generation import continue_steps
from runtime.wan.flow_inversion import public_schedule,invert_received_latent
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,_clear_cache
from runtime.wan.io import dump,encode_rgb,read_mp4
from .flow_run import quality

ARMS=('OFF','A','B')
CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
RGB_SHAPE=(181,320,512,3)
MANIFEST=Path(__file__).parent/'configs/video_inversion.json'
PLAN={'generation':{'transformer':300,'scheduler_step':150},
      'media':{'vae_decode':3,'vae_encode':3,'mp4_save':3},
      'inversion':{'transformer':300,'inverse_update':150}}


def load(path):return json.loads(Path(path).read_text())


def empty_case(status):
    return {'status':status,'video_denominator':3,'videos':{a:{'status':status,
        'generation_status':'NOT_RUN','media_status':'NOT_RUN','inversion_status':'NOT_RUN',
        'decoded':{'status':'NOT_RUN','latent_time_denominator':46,'per_time':[{'time':t,'status':'NOT_RUN'} for t in range(46)]},
        'inverse_steps':[],'quality':{'status':'NOT_RUN'},'posthoc':{'status':'NOT_RUN'}} for a in ARMS}}


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def generation_stage(config,output,result,book,payloads,count,save,fail):
    pipe=base=prompt=negative=origin=marked=terminal=None
    try:
        pipe,base,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        method.check(base);origin=copy.deepcopy(pipe.scheduler)
        schedule=public_schedule(origin);dump(output/'public_schedule.json',schedule)
        result['public_schedule']=schedule
        result['generation_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        torch.save(base.detach().cpu(),output/'writer/base_noise.pt')
        for arm in ARMS:
            item=result['videos'][arm]
            try:
                marked=base.clone() if arm=='OFF' else method.write(base,book,payloads[ARMS.index(arm)-1])
                torch.save(marked.detach().cpu(),output/'writer'/f'{arm}_initial_noise.pt')
                pipe.scheduler=copy.deepcopy(origin)
                precision={}
                terminal=continue_steps(pipe,pipe.scheduler,marked,prompt,negative,dtype,
                    config['generation']['guidance_scale'],0,50,count,precision=precision)
                torch.save(terminal.detach().cpu().float(),output/'writer'/f'{arm}_terminal.pt')
                item.update(generation_status='COMPLETE',generation_precision=precision)
            except Exception as exc:item['generation_status']='FAILED';fail(arm+'/generation',exc)
            finally:marked=terminal=None;release();save()
    finally:pipe=base=prompt=negative=origin=marked=terminal=None;release()


def media_stage(config,output,result,count,save,fail):
    vae=latent=rgb=received=encoded=off=None
    try:
        if not any(v['generation_status']=='COMPLETE' for v in result['videos'].values()):
            for v in result['videos'].values():v['media_status']='MISSING_GENERATION'
            return
        vae=load_frozen_vae(config)
        for arm in ARMS:
            item=result['videos'][arm]
            if item['generation_status']!='COMPLETE':item['media_status']='MISSING_GENERATION';save();continue
            try:
                latent=torch.load(output/'writer'/f'{arm}_terminal.pt',map_location='cpu',weights_only=True)
                count('vae_decode',False)
                rgb=decode_normalized_latent(vae,latent.to(next(vae.parameters()).device)).cpu()
                count('vae_decode',True)
                if tuple(rgb.shape)!=RGB_SHAPE or not bool(torch.isfinite(rgb).all()):raise ValueError('invalid generated RGB')
                path=output/'videos'/f'{arm}.mp4'
                count('mp4_save',False);encode_rgb(rgb,path,8,18);count('mp4_save',True)
                # Receiver input is the actual persisted MP4, never rgb above.
                received=read_mp4(path)
                if tuple(received.shape)!=RGB_SHAPE or not bool(torch.isfinite(received).all()):raise ValueError('invalid MP4 readback')
                if arm=='OFF':off=received
                try:
                    if off is not None:item['quality']={'status':'MEASURED','reference':'same-case OFF persisted MP4',**quality(off,received)}
                    else:item['quality']={'status':'MISSING_OFF_MP4'}
                except Exception as exc:item['quality']={'status':'FAILED','error':repr(exc)};fail(arm+'/quality',exc)
                count('vae_encode',False)
                encoded=reencode_rgb24_readback(vae,received).cpu().float()
                count('vae_encode',True);method.check(encoded)
                torch.save(encoded,output/'receiver'/f'{arm}_mp4_latent.pt')
                item.update(media_status='COMPLETE',mp4_path=str(path))
            except Exception as exc:item['media_status']='FAILED';fail(arm+'/media',exc)
            finally:
                latent=rgb=received=encoded=None
                _clear_cache(vae);release();save()
    finally:
        if vae is not None:_clear_cache(vae)
        vae=latent=rgb=received=encoded=off=None;release()


def receiver_config(config):
    public=copy.deepcopy(config)
    public['generation']['seed']=0
    public['generation']['role']='known_prompt_receiver_fresh_loader_noise_discarded'
    return public


def inversion_stage(public_config,output,result,book,count,save,fail):
    pipe=unused_noise=prompt=negative=received=recovered=None
    try:
        if not any(v['media_status']=='COMPLETE' for v in result['videos'].values()):
            for v in result['videos'].values():v['inversion_status']='MISSING_MP4_LATENT'
            return
        if public_config['generation']['seed']!=0:raise ValueError('receiver loader seed must be independent zero')
        schedule=load(output/'public_schedule.json')
        pipe,unused_noise,prompt,negative,dtype=prepare_generation(public_config,load_vae=False)
        unused_noise=None
        if public_schedule(pipe.scheduler)!=schedule:raise ValueError('public receiver schedule differs from actual forward schedule')
        result['receiver_contract']={'loader_seed':0,'loader_noise_discarded':True,
            'conditioning':'known public prompt/negative/CFG5','writer_state_used':False,
            'resolved_model_revision':getattr(pipe.transformer.config,'_commit_hash',None)}
        for arm in ARMS:
            item=result['videos'][arm]
            if item['media_status']!='COMPLETE':item['inversion_status']='MISSING_MP4_LATENT';save();continue
            try:
                received=torch.load(output/'receiver'/f'{arm}_mp4_latent.pt',map_location='cpu',weights_only=True)
                method.check(received)
                def record(row):item['inverse_steps'].append(row);save()
                recovered=invert_received_latent(pipe.transformer,received.to(next(pipe.transformer.parameters()).device),
                    prompt,negative,dtype,public_config['generation']['guidance_scale'],schedule['sigmas'],schedule['timesteps'],count,record)
                recovered=recovered.detach().cpu().float();method.check(recovered)
                torch.save(recovered,output/'receiver'/f'{arm}_recovered_noise.pt')
                decoded=method.read(recovered,book)
                item.update(inversion_status='COMPLETE',decoded=decoded)
            except Exception as exc:item['inversion_status']='FAILED';fail(arm+'/inversion',exc)
            finally:received=recovered=None;release();save()
    finally:pipe=unused_noise=prompt=negative=received=recovered=None;release()


def posthoc(output,result,payloads):
    # Runs only AFTER receiver outputs are persisted and the receiver model released.
    for arm,item in result['videos'].items():
        if item['inversion_status']!='COMPLETE':item['posthoc']={'status':'MISSING_RECOVERED_NOISE'};continue
        decoded=item['decoded']
        decoded['payload_comparisons_reporting_only']=method.compare_payloads(decoded,payloads)
        decoded['truth_reporting_only']=None if arm=='OFF' else ARMS.index(arm)-1
        decoded['OFF_coincidental_exact_matches']=[r['message'] for r in decoded['payload_comparisons_reporting_only']['aggregate'] if r['exact_payload_match']] if arm=='OFF' else None
        try:
            actual=torch.load(output/'writer'/f'{arm}_initial_noise.pt',map_location='cpu',weights_only=True).double()
            recovered=torch.load(output/'receiver'/f'{arm}_recovered_noise.pt',map_location='cpu',weights_only=True).double()
            values={}
            for name,a,b in [('all',actual,recovered),('channel0',actual[:,0],recovered[:,0])]:
                ac=a-a.mean();bc=b-b.mean();den=float(ac.norm()*bc.norm())
                values[name]={'rmse':float((a-b).square().mean().sqrt()),'correlation':float((ac*bc).sum())/den if den else None}
            item['posthoc']={'status':'MEASURED','initial_vs_recovered_noise':values,'used_for_reconstruction':False}
        except Exception as exc:item['posthoc']={'status':'MISSING_OR_FAILED','error':repr(exc)}


def run_case(case_id,output):
    manifest=load(MANIFEST);case=next(c for c in manifest['development'] if c['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for folder in ('writer','receiver','videos'):(output/folder).mkdir()
    result=empty_case('RUNNING');result.update(case=case,config=config,fixed_calls=PLAN,stages={},failures=[],
        actual_calls={s:{k+'_'+done:0 for k in counts for done in ('attempted','completed')} for s,counts in PLAN.items()},
        quality_tolerance=None,scientific_pass=None,claim='initial-noise inversion mechanism baseline; not secure Gaussian/PRC/SIGMark or exact UniPC inverse')
    def save():dump(output/'result.json',result)
    def fail(stage,exc):result['failures'].append({'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()});save()
    book=method.codebook(config['key_utf8'].encode())
    torch.save(book,output/'codebook.pt');dump(output/'config.json',config)
    result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    import diffusers
    result['environment']={'python':platform.python_version(),'torch':str(torch.__version__),'diffusers':diffusers.__version__}
    for name,fn in [('generation',generation_stage),('media',media_stage),('inversion',inversion_stage)]:
        start=time.monotonic();result['stages'][name]={'status':'RUNNING'}
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        save()
        def count(kind,done):result['actual_calls'][name][kind+('_completed' if done else '_attempted')]+=1;save()
        try:
            if name=='generation':fn(config,output,result,book,manifest['payloads'],count,save,fail)
            elif name=='media':fn(config,output,result,count,save,fail)
            else:fn(receiver_config(config),output,result,book,count,save,fail)
            result['stages'][name]['status']='RETURNED_WITH_RETAINED_ARM_STATUSES'
        except Exception as exc:result['stages'][name]['status']='FAILED_SETUP';fail(name+'/setup',exc)
        finally:
            release();result['stages'][name].update(elapsed_seconds=time.monotonic()-start,
                process_cumulative_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                cuda_stage_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                cuda_stage_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
                cuda_current_allocated_after_release=torch.cuda.memory_allocated() if torch.cuda.is_available() else None)
            save()
    posthoc(output,result,manifest['payloads'])
    for item in result['videos'].values():item['status']='COMPLETE' if all(item[k+'_status']=='COMPLETE' for k in ('generation','media','inversion')) else 'WITH_RETAINED_FAILURES'
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    save();return result


def run_all(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','case_denominator':4,'video_denominator':12,
        'cases':{c:empty_case('NOT_RUN') for c in CASES},'quality_tolerance':None,'scientific_pass':None,
        'fixed_calls':{s:{k:4*v for k,v in counts.items()} for s,counts in PLAN.items()}}
    dump(output/'manifest.json',load(MANIFEST));dump(output/'result.json',result)
    for case in CASES:
        log=output/(case+'.log');print(case,'started; child log:',log,flush=True)
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.video_inversion_run',
                    '--output',str(output/case),'--case-id',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            result['cases'][case]=load(output/case/'result.json')|{'exit_code':child.returncode}
        except Exception as exc:result['cases'][case]=empty_case('FAILED_LAUNCH_OR_RESULT')|{'error':repr(exc)}
        result['cases'][case]['log']=str(log);dump(output/'result.json',result)
        print(case,result['cases'][case]['status'],'child log:',log,flush=True)
    result['actual_calls_observed']={s:{k+'_'+d:sum(v.get('actual_calls',{}).get(s,{}).get(k+'_'+d,0) for v in result['cases'].values()) for k in counts for d in ('attempted','completed')} for s,counts in PLAN.items()}
    result['call_count_case_coverage']=sum('actual_calls' in v for v in result['cases'].values())
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='EXECUTION_COMPLETE' and v.get('exit_code')==0 for v in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-id',choices=CASES)
    args=parser.parse_args();result=run_case(args.case_id,args.output) if args.case_id else run_all(args.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
