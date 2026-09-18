"""Fixed real-MP4 crops, start-free reception, then explicitly oracle correspondence."""
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
from main.tube_state import inversion_state as state
from runtime.wan.generation import load_frozen_vae,prepare_generation
from runtime.wan.flow_inversion import invert_received_latent,public_schedule
from runtime.wan.vae import reencode_rgb24_readback,_clear_cache,quantize_rgb8_no_codec
from runtime.wan.io import read_mp4,dump
from . import inversion_crop_observation as analysis
from . import inversion_state_holdout_run as holdout

SOURCE_RUN='inversion_state_holdout_20260918T140243514638Z'
SOURCE_COMMIT='847663159f01b5486c616363ce630878ec75ac77'
CASES=holdout.CASES
ARMS=('OFF','A','B')
STARTS=(0,16,17)
RGB_SHAPE=(129,320,512,3)
MANIFEST=Path(__file__).parent/'configs/inversion_crop_observation.json'
PLAN={'mp4_save':9,'vae_encode':9,'transformer':900,'inverse_update':450}


def guard_output(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if output==source or source in output.parents:raise ValueError('output must be independent of the existing source run')


def same_schedule(actual,recorded):
    def canonical(value):
        if isinstance(value,dict):return {k:sorted(v) if k=='_use_default_values' else canonical(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [canonical(v) for v in value]
        return value
    return canonical(actual)==canonical(recorded)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def encode_lossless(rgb,path):
    pixels=quantize_rgb8_no_codec(rgb).numpy();t,h,w,_=pixels.shape
    if t!=129:raise ValueError('crop must contain exactly 129 existing frames')
    path.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{w}x{h}','-r','8','-i','pipe:0','-an','-c:v','libx264rgb','-crf','0','-pix_fmt','rgb24','-n',str(path)],
        input=pixels.tobytes(),capture_output=True,check=True)


def empty_case(status):
    clips={}
    for a,arm in enumerate(ARMS):
        for k,start in enumerate(STARTS):
            rid=f'r{a*3+k:02d}'
            clips[rid]=dict(status=status,source_arm_reporting_only=arm,source_start_reporting_only=start,
                receiver=dict(status=status,clip_path=f'clips/{rid}.mp4',media_status='NOT_RUN',inversion_status='NOT_RUN',
                    raw={'status':'NOT_RUN','local_time_denominator':33,'per_time':[{'local_time':j,'status':'NOT_RUN'} for j in range(33)]},inverse_steps=[]),
                oracle=[dict(status='NOT_RUN',nominal_shift=shift,oracle=True,
                    per_time=[{'local_time':j,'status':'NOT_RUN'} for j in range(33)],
                    core=[dict(window=n,start=a,stop=b,status='NOT_RUN',coverage_status='NOT_RUN',valid=False) for n,(a,b) in enumerate(state.WINDOWS)]) for shift in analysis.MAPS[start]])
    return dict(status=status,clip_denominator=9,oracle_map_denominator=12,clips=clips)


def encode_received_clips(public_config,output,receivers,count,save,fail):
    """Receiver gets opaque local files and public geometry, no attacker map."""
    vae=rgb=encoded=None
    try:
        if not any(r.get('crop_status')=='COMPLETE' for r in receivers.values()):
            for r in receivers.values():r['media_status']='MISSING_CROP'
            return
        vae=load_frozen_vae(public_config)
        for rid,item in receivers.items():
            if item.get('crop_status')!='COMPLETE':item['media_status']='MISSING_CROP';save();continue
            try:
                rgb=read_mp4(output/item['clip_path'])
                if tuple(rgb.shape)!=RGB_SHAPE:raise ValueError('receiver input is not a 129-frame crop')
                count('vae_encode',False);encoded=reencode_rgb24_readback(vae,rgb).cpu().float();count('vae_encode',True)
                analysis.check(encoded);torch.save(encoded,output/'receiver'/f'{rid}_latent.pt')
                item.update(media_status='COMPLETE',encoded_shape=list(encoded.shape),vae_local_phase=0,
                    latent_time_scale=4,first_slice_is_causal_singleton=True,last_slice_full_group=True)
            except Exception as exc:item['media_status']='FAILED';fail(rid+'/vae',exc)
            finally:rgb=encoded=None;_clear_cache(vae);release();save()
    finally:
        if vae is not None:_clear_cache(vae)
        vae=rgb=encoded=None;release()


def invert_received_clips(public_config,schedule,output,receivers,count,save,fail):
    """No crop start, source video, key, pad, truth or writer evidence accepted."""
    pipe=dummy=prompt=negative=received=recovered=None
    try:
        if not any(r['media_status']=='COMPLETE' for r in receivers.values()):
            for r in receivers.values():r['inversion_status']='MISSING_CLIP_LATENT'
            return
        if public_config['generation']['frames']!=129 or public_config['generation']['seed']!=0:raise ValueError('fixed public receiver geometry/seed required')
        pipe,dummy,prompt,negative,dtype=prepare_generation(public_config,load_vae=False);dummy=None
        if not same_schedule(public_schedule(pipe.scheduler),schedule):raise ValueError('public original 50-step schedule differs')
        for rid,item in receivers.items():
            if item['media_status']!='COMPLETE':item['inversion_status']='MISSING_CLIP_LATENT';save();continue
            try:
                received=torch.load(output/'receiver'/f'{rid}_latent.pt',map_location='cpu',weights_only=True)
                analysis.check(received)
                def record(row):item['inverse_steps'].append(row);save()
                recovered=invert_received_latent(pipe.transformer,received.to(next(pipe.transformer.parameters()).device),
                    prompt,negative,dtype,public_config['generation']['guidance_scale'],schedule['sigmas'],schedule['timesteps'],count,record)
                recovered=recovered.detach().cpu().float();analysis.check(recovered)
                path=output/'receiver'/f'{rid}_recovered.pt';torch.save(recovered,path)
                item.update(status='COMPLETE',inversion_status='COMPLETE',raw=analysis.raw_observations(recovered),recovered_sha256=sha(path),
                    receiver_contract={'source_start_used':False,'writer_evidence_used':False,'time_pad_used':False,'loader_seed':0,'dummy_discarded':True,
                        'known_prompt':True,'resolved_model_revision':getattr(pipe.transformer.config,'_commit_hash',None)})
            except Exception as exc:item['inversion_status']='FAILED';fail(rid+'/inversion',exc)
            finally:received=recovered=None;release();save()
    finally:pipe=dummy=prompt=negative=received=recovered=None;release()


def run_case(case_id,source,output):
    source=Path(source);output=Path(output)
    guard_output(source,output)
    if source.name!=SOURCE_RUN:raise ValueError('fixed existing source run required')
    output.mkdir(parents=True,exist_ok=False)
    for folder in ('clips','receiver','oracle'):(output/folder).mkdir()
    result=empty_case('RUNNING');result.update(case_id=case_id,fixed_calls=PLAN,failures=[],stages={},input_sha256={},
        actual_calls={k+'_'+d:0 for k in PLAN for d in ('attempted','completed')},scientific_pass=None,
        source_run=SOURCE_RUN,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()))
    def save():dump(output/'result.json',result)
    def count(k,done):result['actual_calls'][k+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,exc):result['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    save()
    try:
        import diffusers
        result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        config_path=source/case_id/'config.json';schedule_path=source/case_id/'public_schedule.json'
        source_result_path=source/case_id/'result.json'
        source_result=holdout.state.base.load(source_result_path)
        if source_result.get('source_commit')!=SOURCE_COMMIT:raise ValueError('source case provenance differs from fixed published holdout source')
        result['source_case_provenance']={k:source_result.get(k) for k in ('source_commit','source_dirty','status','environment')}
        config=holdout.state.base.load(config_path);schedule=holdout.state.base.load(schedule_path)
        fixed=holdout.validate();case=next(c for c in fixed['holdout'] if c['id']==case_id)
        expected=copy.deepcopy(fixed['base_config']);expected['generation'].update(prompt=case['prompt'],seed=case['seed'])
        if config!=expected:raise ValueError('source config differs from fixed holdout candidate')
        result['input_sha256'].update(config=sha(config_path),schedule=sha(schedule_path),case_result=sha(source_result_path))
        public=dict(model=copy.deepcopy(config['model']),generation=copy.deepcopy(config['generation']))
        public['generation'].update(frames=129,seed=0,role='crop_receiver_without_start')
        dump(output/'receiver_public_config.json',public);dump(output/'public_schedule.json',schedule)
        attacker=[dict(receiver_id=rid,source_mp4=str(source/case_id/'videos'/(r['source_arm_reporting_only']+'.mp4')),
            source_start=r['source_start_reporting_only'],source_arm=r['source_arm_reporting_only'],frames=129,
            clip_path=r['receiver']['clip_path'],status='NOT_RUN') for rid,r in result['clips'].items()]
        for arm in ARMS:
            try:
                path=source/case_id/'videos'/f'{arm}.mp4';result['input_sha256'][arm]=sha(path)
                rgb=read_mp4(path)
                if tuple(rgb.shape)!=(181,RGB_SHAPE[1],RGB_SHAPE[2],3):raise ValueError('source must contain existing 181 frames')
                for rid,row in result['clips'].items():
                    if row['source_arm_reporting_only']!=arm:continue
                    start=row['source_start_reporting_only'];item=row['receiver']
                    try:
                        clip=rgb[start:start+129].clone();target=output/item['clip_path']
                        count('mp4_save',False);encode_lossless(clip,target);count('mp4_save',True)
                        readback=read_mp4(target)
                        if not torch.equal(clip,readback):raise ValueError('lossless crop RGB equality failed')
                        item.update(crop_status='COMPLETE',crop_sha256=sha(target),lossless_rgb_equal=True)
                        next(r for r in attacker if r['receiver_id']==rid).update(status='COMPLETE',
                            source_sha256=result['input_sha256'][arm],clip_sha256=item['crop_sha256'])
                    except Exception as exc:item['crop_status']='FAILED';fail(rid+'/crop',exc)
                    finally:clip=readback=None;save()
            except Exception as exc:fail(arm+'/source',exc)
            finally:rgb=None;release()
        dump(output/'attacker_manifest.json',dict(associations=attacker,claim='not supplied to receiver; current input hashes are not historical generation-time hashes'))
        receivers={rid:row['receiver'] for rid,row in result['clips'].items()}
        for name,fn in [('vae',lambda:encode_received_clips(public,output,receivers,count,save,fail)),
                        ('inversion',lambda:invert_received_clips(public,schedule,output,receivers,count,save,fail))]:
            start=time.monotonic()
            if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
            try:fn()
            except Exception as exc:fail(name+'/setup',exc)
            finally:
                release();result['stages'][name]=dict(elapsed_seconds=time.monotonic()-start,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
                save()
        # Receiver model is gone. This stage alone sees source timing, key and truth.
        book=state.codebook(config['key_utf8'].encode());torch.save(book,output/'oracle'/'codebook.pt')
        for rid,row in result['clips'].items():
            if row['receiver']['inversion_status']!='COMPLETE':continue
            try:
                z=torch.load(output/'receiver'/f'{rid}_recovered.pt',map_location='cpu',weights_only=True)
                truth=None if row['source_arm_reporting_only']=='OFF' else ARMS.index(row['source_arm_reporting_only'])-1
                row['oracle']=analysis.oracle(z,book,row['source_start_reporting_only'],truth)
                dump(output/'oracle'/f'{rid}.json',row['oracle']);row['status']='COMPLETE'
                del z
            except Exception as exc:row['status']='FAILED_ORACLE';fail(rid+'/oracle',exc)
            save()
    except Exception as exc:fail('setup',exc)
    for row in result['clips'].values():
        if row['status']!='COMPLETE':
            row['status']='WITH_RETAINED_FAILURES'
            if row['receiver']['status']!='COMPLETE':row['receiver']['status']='WITH_RETAINED_FAILURES'
    result['status']='EXECUTION_COMPLETE' if not result['failures'] and all(r['status']=='COMPLETE' for r in result['clips'].values()) else 'WITH_RETAINED_FAILURES'
    result['source_sha256']={str(p):sha(p) for p in (Path(__file__),Path(analysis.__file__),MANIFEST)}
    save();return result


def run_all(source,output):
    guard_output(source,output)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',source_run=SOURCE_RUN,source_video_denominator=6,clip_denominator=18,
        local_slice_denominator=594,oracle_map_denominator=24,oracle_core_window_denominator=264,
        marked_clip_denominator=12,marked_oracle_map_denominator=16,marked_oracle_core_denominator=176,
        cases={c:empty_case('NOT_RUN') for c in CASES},fixed_calls={k:v*2 for k,v in PLAN.items()},scientific_pass=None)
    dump(output/'manifest.json',holdout.state.base.load(MANIFEST));dump(output/'result.json',result)
    for case in CASES:
        log=output/(case+'.log');print(case,'started; log:',log,flush=True)
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.inversion_crop_observation_run',
                    '--input',str(source),'--output',str(output/case),'--case-id',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            result['cases'][case]=holdout.state.base.load(output/case/'result.json')|{'exit_code':child.returncode}
        except Exception as exc:result['cases'][case]=empty_case('FAILED_LAUNCH_OR_RESULT')|{'error':repr(exc)}
        dump(output/'result.json',result)
    result['actual_calls_observed']={k+'_'+d:sum(c.get('actual_calls',{}).get(k+'_'+d,0) for c in result['cases'].values()) for k in PLAN for d in ('attempted','completed')}
    result['oracle_summary']=summarize(result['cases'])
    result['status']='EXECUTION_COMPLETE' if all(c['status']=='EXECUTION_COMPLETE' and c.get('exit_code')==0 for c in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result


def summarize(cases):
    """Do not choose phase mapping or pool related windows as independent trials."""
    summaries=[]
    for start in STARTS:
        for shift in analysis.MAPS[start]:
            for arm_group in ('MARKED','OFF'):
                count=4 if arm_group=='MARKED' else 2
                complete_each=7 if (start,shift)==(17,5) else 8
                row=dict(start_reporting_only=start,nominal_shift=shift,arm_group=arm_group,
                    source_video_denominator=count,mapping_denominator=count,core_grid_denominator=11*count,
                    expected_complete_window_denominator=complete_each*count,observed_complete_window_denominator=0,
                    unobserved_expected_complete_windows=complete_each*count,partial_windows_observed=0,missing_windows_observed=0,
                    measured_maps=0,invalid_complete_windows=0,per_video={})
                row['erasures_complete_components']=0
                if arm_group=='MARKED':row.update(exact_complete_windows=0,complete_component_errors=0)
                for case in CASES:
                    for clip in cases.get(case,{}).get('clips',{}).values():
                        arm=clip['source_arm_reporting_only']
                        if clip['source_start_reporting_only']!=start or (arm=='OFF')!=(arm_group=='OFF'):continue
                        mapping=next((v for v in clip['oracle'] if v['nominal_shift']==shift),{})
                        if mapping.get('status')!='MEASURED':continue
                        metrics=mapping['complete_window_report'];row['measured_maps']+=1
                        row['observed_complete_window_denominator']+=metrics['complete_window_denominator']
                        row['unobserved_expected_complete_windows']-=metrics['complete_window_denominator']
                        row['partial_windows_observed']+=mapping['partial_windows'];row['missing_windows_observed']+=mapping['missing_windows']
                        row['invalid_complete_windows']+=metrics['invalid_complete_windows']
                        row['erasures_complete_components']+=metrics['erasures_complete_components']
                        if arm_group=='MARKED':
                            row['exact_complete_windows']+=metrics['exact_windows'];row['complete_component_errors']+=metrics['component_errors']
                        row['per_video'][case+'/'+arm]=dict(complete_windows=metrics,reporting_only=mapping['reporting_only'],
                            oracle_rankings={mode:dict(selected_message=r['selected_message'],message_unique=r['message_unique'],
                                scores=[c['score'] for c in r['candidates']]) for mode,r in mapping['rankings'].items()})
                row['observed_component_denominator']=2*row['observed_complete_window_denominator']
                row['valid_complete_windows']=row['observed_complete_window_denominator']-row['invalid_complete_windows']
                summaries.append(row)
    return dict(groups=summaries,claim='oracle mappings are related diagnostics, not blind synchronization or independent trials; missing-support scores are not directly comparable')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-id',choices=CASES);args=parser.parse_args()
    r=run_case(args.case_id,args.input,args.output) if args.case_id else run_all(args.input,args.output)
    if r['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
