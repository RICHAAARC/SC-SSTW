"""Fixed twelve-video four-layer diagnostic. Model execution only through CLI."""
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
from main.tube_state import grow_frequency as method
from runtime.wan.grow_frequency import generate
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.io import dump,encode_rgb,read_mp4
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,quantize_rgb8_no_codec,_clear_cache
from .flow_run import quality

ARMS=('OFF','A','B')
LAYERS=('terminal','float_rgb','rgb8','mp4')
CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
RGB_SHAPE=(181,320,512,3)
PLAN={'transformer':300,'scheduler_step':150,'local_gradient':40,'vae_decode':3,'vae_encode':9,'mp4_save':3}
MANIFEST=Path(__file__).parent/'configs/grow_video_frequency.json'


def load(path):return json.loads(Path(path).read_text())


def empty_layer():return {'status':'NOT_RUN','latent_time_denominator':46,'per_time':[{'time':t,'status':'NOT_RUN'} for t in range(46)]}


def missing_case(status):
    return {'status':status,'video_denominator':3,'videos':{
        arm:{'status':status,'layers':{layer:empty_layer() for layer in LAYERS}} for arm in ARMS}}


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def run_case(case_id,output):
    manifest=load(MANIFEST)
    case=next(c for c in manifest['development'] if c['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result={'status':'RUNNING','case':case,'config':config,'protocol':manifest['frequency'],
        'video_denominator':3,'layers_per_video':4,'latent_time_denominator_per_layer':46,'bits':16,
        'fixed_calls':PLAN,'actual_calls':{k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},
        'videos':{a:{'status':'NOT_RUN','generation_status':'NOT_RUN','steps':[],
                    'layers':{l:empty_layer() for l in LAYERS},'quality':{l:{'status':'NOT_RUN'} for l in LAYERS[1:]}} for a in ARMS},
        'failures':[],'quality_tolerance':None,'scientific_pass':None,
        'claim':'fixed candidate video adaptation; completion, bit recovery and quality are separate; OFF is not FPR'}
    def save():result['elapsed_seconds']=time.monotonic()-started;dump(output/'result.json',result)
    def count(kind,done):result['actual_calls'][kind+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,exc):result['failures'].append({'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()});save()
    def read_layer(arm,layer,z):
        row=result['videos'][arm]['layers'][layer]
        try:
            torch.save(z.detach().cpu().float(),output/'latents'/f'{arm}_{layer}.pt')
            decoded=method.read(z.detach().cpu().float(),book)
            decoded['payload_comparisons_reporting_only']=method.compare_payloads(decoded,book)
            decoded['truth']=None if arm=='OFF' else ARMS.index(arm)-1
            decoded['OFF_coincidental_exact_matches']=[v['message'] for v in decoded['payload_comparisons_reporting_only']['aggregate'] if v['exact_payload_match']] if arm=='OFF' else None
            row.update(decoded)
        except Exception as exc:row['status']='FAILED';fail(arm+'/'+layer,exc)
        save()
    book=method.codebook(config['key_utf8'].encode());dump(output/'codebook.json',book)
    dump(output/'config.json',config);(output/'latents').mkdir();save()
    pipe=initial=prompt=negative=origin=latent=vae=None
    try:
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        import diffusers
        result['environment']={'python':platform.python_version(),'torch':str(torch.__version__),'diffusers':diffusers.__version__}
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        origin=copy.deepcopy(pipe.scheduler)
        if tuple(initial.shape)!=method.SHAPE:raise ValueError('unexpected initial latent geometry')
        torch.save(initial.detach().cpu(),output/'initial_latent.pt')
        result['scheduler']={'class':type(origin).__name__,'config':dict(origin.config),'sigmas':origin.sigmas.tolist()}
        for arm in ARMS:
            item=result['videos'][arm];item['generation_status']='RUNNING';save()
            try:
                pipe.scheduler=copy.deepcopy(origin)
                def record(row):item['steps'].append(row);save()
                latent=generate(pipe,initial.clone(),prompt,negative,dtype,config['generation']['guidance_scale'],book,
                    None if arm=='OFF' else ARMS.index(arm)-1,count,record)
                read_layer(arm,'terminal',latent)
                item['generation_status']='COMPLETE'
            except Exception as exc:item['generation_status']='FAILED';fail(arm+'/generation',exc)
            finally:latent=None;release();save()
    except Exception as exc:fail('generation_setup',exc)
    finally:pipe=initial=prompt=negative=origin=latent=None;release()
    references={}
    try:
        vae=load_frozen_vae(config)
        for arm in ARMS:
            item=result['videos'][arm];images={};latent=rgb=encoded=None
            if item['generation_status']!='COMPLETE' or item['layers']['terminal']['status']!='COMPLETE':
                for layer in LAYERS[1:]:item['layers'][layer]['status']='MISSING_TERMINAL'
                save();continue
            try:
                latent=torch.load(output/'latents'/f'{arm}_terminal.pt',map_location='cpu',weights_only=True)
                count('vae_decode',False)
                rgb=decode_normalized_latent(vae,latent.to(next(vae.parameters()).device)).cpu()
                count('vae_decode',True)
                if tuple(rgb.shape)!=RGB_SHAPE or not torch.isfinite(rgb).all():raise ValueError('invalid decoded RGB')
                images['float_rgb']=rgb
                images['rgb8']=quantize_rgb8_no_codec(rgb).float()/255
                path=output/'videos'/f'{arm}.mp4'
                try:
                    count('mp4_save',False);encode_rgb(rgb,path,8,18);count('mp4_save',True)
                    images['mp4']=read_mp4(path)
                    if images['mp4'].shape!=rgb.shape:raise ValueError('MP4 readback geometry differs')
                except Exception as exc:item['layers']['mp4']['status']='FAILED_SAVE_OR_READBACK';fail(arm+'/mp4',exc);images.pop('mp4',None)
                for layer in LAYERS[1:]:
                    if layer not in images:continue
                    try:
                        count('vae_encode',False)
                        encoded=reencode_rgb24_readback(vae,images[layer]).cpu().float()
                        count('vae_encode',True);read_layer(arm,layer,encoded)
                    except Exception as exc:item['layers'][layer]['status']='FAILED';fail(arm+'/'+layer,exc)
                    finally:encoded=None;_clear_cache(vae);release();save()
                    if arm=='OFF':references[layer]=images[layer]
                    if layer in references:
                        try:item['quality'][layer]={'status':'MEASURED','reference':'same-run OFF same RGB layer',**quality(references[layer],images[layer])}
                        except Exception as exc:item['quality'][layer]={'status':'FAILED','error':repr(exc)};fail(arm+'/'+layer+'/quality',exc)
                    else:item['quality'][layer]={'status':'MISSING_OFF_REFERENCE'}
            except Exception as exc:
                for layer in LAYERS[1:]:
                    if item['layers'][layer]['status']=='NOT_RUN':item['layers'][layer]['status']='FAILED_DECODE'
                fail(arm+'/decode',exc)
            finally:images={};latent=rgb=encoded=None;_clear_cache(vae);release();save()
    except Exception as exc:fail('media_setup',exc)
    finally:
        if vae is not None:_clear_cache(vae)
        vae=None;references={};release()
    for arm,item in result['videos'].items():
        item['expected_controlled_steps']=0 if arm=='OFF' else 20
        item['actual_controlled_steps']=sum(v['controlled'] for v in item['steps'])
        item['control_schedule_complete']=item['generation_status']=='COMPLETE' and item['actual_controlled_steps']==item['expected_controlled_steps']
        item['status']='COMPLETE' if item['control_schedule_complete'] and all(v['status']=='COMPLETE' for v in item['layers'].values()) else 'WITH_RETAINED_FAILURES'
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    result['resources']={'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    save();return result


def run_all(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result={'status':'RUNNING','case_denominator':4,'video_denominator':12,'layer_denominator':48,
        'cases':{c:missing_case('NOT_RUN') for c in CASES},'fixed_calls':{k:v*4 for k,v in PLAN.items()},
        'quality_tolerance':None,'scientific_pass':None}
    dump(output/'manifest.json',load(MANIFEST));dump(output/'result.json',result)
    for case in CASES:
        log=output/(case+'.log')
        print(case,'started; child log:',log,flush=True)
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.grow_frequency_run',
                    '--output',str(output/case),'--case-id',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            result['cases'][case]=load(output/case/'result.json')|{'exit_code':child.returncode,'log':str(log)}
        except Exception as exc:result['cases'][case]=missing_case('FAILED_LAUNCH_OR_RESULT')|{'error':repr(exc),'log':str(log)}
        dump(output/'result.json',result)
        print(case,result['cases'][case]['status'],'child log:',log,flush=True)
    result['actual_calls_observed']={k+'_'+s:sum(v.get('actual_calls',{}).get(k+'_'+s,0) for v in result['cases'].values()) for k in PLAN for s in ('attempted','completed')}
    result['call_count_case_coverage']=sum('actual_calls' in v for v in result['cases'].values())
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='EXECUTION_COMPLETE' and v.get('exit_code')==0 for v in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-id',choices=CASES)
    args=parser.parse_args()
    result=run_case(args.case_id,args.output) if args.case_id else run_all(args.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
