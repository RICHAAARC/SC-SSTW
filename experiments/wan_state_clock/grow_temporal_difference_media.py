"""Fixed twelve-video four-layer diagnostic. Model execution only through CLI."""
from __future__ import annotations
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
from main.tube_state import grow_temporal_difference as method
from runtime.wan.generation import load_frozen_vae
from runtime.wan.io import dump,encode_rgb,read_mp4
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,quantize_rgb8_no_codec,_clear_cache
from .flow_run import quality

ARMS=('OFF','A','B')
LAYERS=('terminal','float_rgb','rgb8','mp4')
CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
RGB_SHAPE=(181,320,512,3)
PLAN={'transformer':0,'scheduler_step':0,'local_gradient':0,'vae_decode':3,'vae_encode':9,'mp4_save':3,'mp4_read':3}
SOURCE_RUN='grow_temporal_difference_20260918T142825573989Z'
SOURCE_COMMIT='a004134f650267485c0b80abdfbb4062fb7c7128'
MANIFEST=Path(__file__).parent/'configs/grow_temporal_difference.json'


def load(path):return json.loads(Path(path).read_text())


def empty_layer():return {'status':'NOT_RUN','latent_time_denominator':46,'pair_denominator':23,'per_pair':[{'pair':k,'times':[2*k,2*k+1],'status':'NOT_RUN'} for k in range(23)]}


def missing_case(status):
    return {'status':status,'video_denominator':3,'videos':{
        arm:{'status':status,'layers':{layer:empty_layer() for layer in LAYERS}} for arm in ARMS}}


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def run_case(case_id,output,source_root,*,expected_source_run=SOURCE_RUN,expected_source_commit=SOURCE_COMMIT,expected_manifest=None):
    manifest=load(MANIFEST) if expected_manifest is None else expected_manifest
    case=next(c for c in manifest['development'] if c['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    source_root=Path(source_root);output=Path(output)
    if output.resolve()==source_root.resolve() or source_root.resolve() in output.resolve().parents:raise ValueError('output must be outside immutable source run')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result={'status':'RUNNING','case':case,'config':config,'protocol':manifest['frequency'],
        'video_denominator':3,'layers_per_video':4,'latent_time_denominator_per_layer':46,'bits':16,
        'fixed_calls':PLAN,'actual_calls':{k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},
        'videos':{a:{'status':'NOT_RUN','generation_status':'NOT_RUN','steps':[],
                    'layers':{l:empty_layer() for l in LAYERS},'quality':{l:{'status':'NOT_RUN'} for l in LAYERS[1:]}} for a in ARMS},
        'file_sha256':{},'source_root':str(source_root),'failures':[],'quality_tolerance':None,'scientific_pass':None,
        'claim':'fixed candidate video adaptation; completion, bit recovery and quality are separate; OFF is not FPR'}
    result['expected_source_run']=expected_source_run;result['expected_source_commit']=expected_source_commit
    result['effective_manifest']=manifest
    result['effective_manifest_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    dump(output/'effective_manifest.json',manifest)
    def save():result['elapsed_seconds']=time.monotonic()-started;dump(output/'result.json',result)
    def count(kind,done):result['actual_calls'][kind+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,exc):result['failures'].append({'stage':stage,'error':repr(exc),'traceback':traceback.format_exc()});save()
    def persist_tensor(path,z):
        torch.save(z.detach().cpu(),path);result['file_sha256'][str(path.relative_to(output))]=hashlib.sha256(path.read_bytes()).hexdigest()
    def read_layer(arm,layer,z):
        row=result['videos'][arm]['layers'][layer]
        try:
            persist_tensor(output/'latents'/f'{arm}_{layer}.pt',z.detach().cpu().float())
            decoded=method.read(z.detach().cpu().float(),book)
            decoded['payload_comparisons_reporting_only']=method.compare_payloads(decoded,book)
            decoded['truth']=None if arm=='OFF' else ARMS.index(arm)-1
            decoded['OFF_coincidental_exact_matches']=[v['message'] for v in decoded['payload_comparisons_reporting_only']['aggregate'] if v['exact_payload_match']] if arm=='OFF' else None
            row.update(decoded)
        except Exception as exc:row['status']='FAILED';fail(arm+'/'+layer,exc)
        save()
    book=method.codebook(config['key_utf8'].encode());dump(output/'codebook.json',book)
    dump(output/'config.json',config);(output/'latents').mkdir();save()
    latent=vae=None
    try:
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        result['media_source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),MANIFEST,Path(method.__file__),Path(method.spatial.__file__),Path(sys.modules[decode_normalized_latent.__module__].__file__),Path(sys.modules[load_frozen_vae.__module__].__file__)]}
        import diffusers
        result['environment']={'python':platform.python_version(),'torch':str(torch.__version__),'diffusers':diffusers.__version__}
        source_result=load(source_root/'result.json');source=load(source_root/case_id/'result.json')
        actual_config=load(source_root/case_id/'config.json')
        result['source_evidence_sha256']={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in
            [('root_result',source_root/'result.json'),('case_result',source_root/case_id/'result.json'),('case_config',source_root/case_id/'config.json')]}
        if source_root.name!=expected_source_run or source['source_commit']!=expected_source_commit:raise ValueError('wrong fixed source run/commit')
        if source['config']!=config or actual_config!=config:raise ValueError('source config differs from fixed roster')
        if source_result['cases'][case_id].get('file_sha256')!=source['file_sha256']:raise ValueError('source root/case hash records disagree')
        result['source_generation_commit']=source['source_commit']
        result['source_model_revision']=source.get('resolved_model_revision')
        result['model_revision_comparison_limit']='missing source model revision prevents exact checkpoint identity claim; no regeneration'
        for arm in ARMS:
            item=result['videos'][arm]
            try:
                path=source_root/case_id/'latents'/f'{arm}_terminal.pt'
                expected=source['file_sha256']['latents/'+arm+'_terminal.pt']
                if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('source terminal SHA256 mismatch')
                latent=torch.load(path,map_location='cpu',weights_only=True)
                if tuple(latent.shape)!=method.SHAPE or not torch.isfinite(latent).all():raise ValueError('invalid source terminal')
                decoded=method.read(latent,book)
                if decoded!= {k:v for k,v in source['videos'][arm]['terminal'].items() if k!='payload_comparisons_reporting_only'}:raise ValueError('terminal readout does not match source evidence')
                item['source_terminal_sha256']=expected;read_layer(arm,'terminal',latent)
                item['generation_status']='VERIFIED_SAVED_TERMINAL'
            except Exception as exc:item['generation_status']='FAILED_SOURCE';fail(arm+'/source',exc)
            finally:latent=None;save()
    except Exception as exc:fail('source_verification',exc)
    references={}
    try:
        if not any(v['generation_status']=='VERIFIED_SAVED_TERMINAL' for v in result['videos'].values()):raise ValueError('no verified source terminals')
        vae=load_frozen_vae(config)
        result['vae_revision']=getattr(vae.config,'_commit_hash',None)
        result['vae_parameter_dtype']=str(next(vae.parameters()).dtype)
        for arm in ARMS:
            item=result['videos'][arm];images={};latent=rgb=encoded=None
            if item['generation_status']!='VERIFIED_SAVED_TERMINAL' or item['layers']['terminal']['status']!='COMPLETE':
                for layer in LAYERS[1:]:item['layers'][layer]['status']='MISSING_TERMINAL'
                save();continue
            try:
                latent=torch.load(output/'latents'/f'{arm}_terminal.pt',map_location='cpu',weights_only=True)
                count('vae_decode',False)
                rgb=decode_normalized_latent(vae,latent.to(next(vae.parameters()).device)).cpu()
                count('vae_decode',True)
                if tuple(rgb.shape)!=RGB_SHAPE or not torch.isfinite(rgb).all():raise ValueError('invalid decoded RGB')
                images['float_rgb']=rgb
                pixels=quantize_rgb8_no_codec(rgb)
                (output/'rgb').mkdir(exist_ok=True)
                persist_tensor(output/'rgb'/f'{arm}_float_rgb.pt',rgb)
                persist_tensor(output/'rgb'/f'{arm}_rgb8.pt',pixels)
                images['rgb8']=pixels.float()/255
                del pixels
                path=output/'videos'/f'{arm}.mp4'
                try:
                    count('mp4_save',False);encode_rgb(rgb,path,8,18);count('mp4_save',True)
                    result['file_sha256'][str(path.relative_to(output))]=hashlib.sha256(path.read_bytes()).hexdigest()
                    count('mp4_read',False);images['mp4']=read_mp4(path);count('mp4_read',True)
                    persist_tensor(output/'rgb'/f'{arm}_mp4_rgb8.pt',quantize_rgb8_no_codec(images['mp4']))
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
        item['status']='COMPLETE' if item['generation_status']=='VERIFIED_SAVED_TERMINAL' and all(v['status']=='COMPLETE' for v in item['layers'].values()) else 'WITH_RETAINED_FAILURES'
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    result['resources']={'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cuda_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None}
    save();return result


def recovery_summary(cases):
    summary={}
    for layer in LAYERS:
        n=errors=erasures=exact=off=off_n=0
        for case in CASES:
            for arm in ARMS:
                row=cases.get(case,{}).get('videos',{}).get(arm,{}).get('layers',{}).get(layer,{})
                if row.get('status')!='COMPLETE':continue
                if arm=='OFF':
                    off_n+=1;off+=bool(row.get('OFF_coincidental_exact_matches'));continue
                n+=1;truth=ARMS.index(arm)-1
                c=row['payload_comparisons_reporting_only']['aggregate'][truth]
                errors+=c['bit_errors_including_erasures'];erasures+=row['aggregate']['bit_erasures'];exact+=bool(c['exact_payload_match'])
        summary[layer]=dict(marked_denominator=8,fixed_bit_denominator=128,completed_marked=n,missing_or_failed_marked=8-n,exact=exact,
            errors_on_observed_bits=errors,observed_bit_denominator=n*16,erasures_on_observed_bits=erasures,
            OFF_case_denominator=4,OFF_completed=off_n,OFF_missing_or_failed=4-off_n,OFF_exact_coincidences=off)
    return summary


def run_all(output,source_root):
    source_root=Path(source_root)
    output=Path(output)
    if output.resolve()==source_root.resolve() or source_root.resolve() in output.resolve().parents:raise ValueError('output must be outside immutable source run')
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
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.grow_temporal_difference_media',
                    '--output',str(output/case),'--case-id',case,'--source-root',str(source_root)],stdout=stream,stderr=subprocess.STDOUT,check=False)
            result['cases'][case]=load(output/case/'result.json')|{'exit_code':child.returncode,'log':str(log)}
        except Exception as exc:result['cases'][case]=missing_case('FAILED_LAUNCH_OR_RESULT')|{'error':repr(exc),'log':str(log)}
        dump(output/'result.json',result)
        print(case,result['cases'][case]['status'],'child log:',log,flush=True)
    result['actual_calls_observed']={k+'_'+s:sum(v.get('actual_calls',{}).get(k+'_'+s,0) for v in result['cases'].values()) for k in PLAN for s in ('attempted','completed')}
    result['call_count_case_coverage']=sum('actual_calls' in v for v in result['cases'].values())
    result['recovery_summary']=recovery_summary(result['cases'])
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='EXECUTION_COMPLETE' and v.get('exit_code')==0 for v in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-id',choices=CASES);parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    result=run_case(args.case_id,args.output,args.source_root) if args.case_id else run_all(args.output,args.source_root)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
