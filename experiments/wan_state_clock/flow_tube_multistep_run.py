"""Fixed full-projection early/final mechanism ablation; budgets are measured, not matched."""
import argparse,copy,gc,hashlib,json,platform,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import projection_margin as carrier,state_clock,terminal_guidance as method,flow_control
from runtime.wan import tube_multistep as runtime
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.io import dump,read_mp4,encode_rgb
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,_clear_cache
from .flow_run import quality,terminal_record
RGB_SHAPE=(181,320,512,3)
ARMS=('OFF','LAST_A','LAST_B','EARLY_LAST_A','EARLY_LAST_B','EARLY_ONLY_A','EARLY_ONLY_B')
CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
MODES=('global_matched','global_state','local_matched','local_without_update','local_state')
PLAN={'transformer':124,'scheduler_step':66,'shadow_step':6,'vae_decode':7,'mp4_save':7,'vae_encode':28}
MANIFEST=Path(__file__).parent/'configs/flow_tube_multistep.json'

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def empty(status):return {'status':status,'videos':{a:{'status':'NOT_RUN','observations':{str(g):{'status':'NOT_RUN'} for g in range(4)}} for a in ARMS},'failures':[]}
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def validate(manifest):
    base=load(MANIFEST.with_name('flow_tube_state_guidance.json'))
    base.update(protocol='flow_tube_multistep_v1',arms=list(ARMS),early_indices=[44,45,46],final_index=49,control='full_projection_no_caps');base.pop('control_index')
    base['base_config']['generation']['role']='fixed_shared_tree_full_projection_multistep'
    base['base_config']['output_drive_parent']='/content/drive/MyDrive/Video-WM/FlowTubeMultistep'
    if manifest!=base or carrier.MARGIN!=1.:raise ValueError('fixed multistep protocol mismatch')


def generate_case(case_id,output):
    manifest=load(MANIFEST);validate(manifest);case=next(v for v in manifest['development'] if v['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=empty('RUNNING');result.update(case=case,config=config,fixed_calls=PLAN,actual_calls={k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},file_sha256={},equivalence={},quality_tolerance=None,scientific_pass=None)
    def save():dump(output/'generation.json',result)
    def count(k,done):result['actual_calls'][k+('_completed' if done else '_attempted')]+=1;save()
    def artifact(name,z):
        path=output/(name+'.pt');torch.save(z.detach().cpu(),path);result['file_sha256'][path.name]=sha(path)
    def store(name,z):
        if not torch.isfinite(z).all():raise FloatingPointError('nonfinite terminal')
        artifact(name+'_terminal',z);dump(output/'terminal_diagnostics'/(name+'.json'),terminal_record(z.detach().cpu().numpy(),book))
        result['videos'][name]['terminal_margin']={str(m):{'minimum_signed_projection':min(flow_control.projection_record(z.detach().cpu().numpy(),book,m)['signed_projection'])} for m in (0,1)}
        result['videos'][name]['status']='TERMINAL_PERSISTED'
    def fail(stage,exc):result['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    book=state_clock.codebook(config['key_utf8'].encode());np.savez(output/'codebook.npz',**book);dump(output/'config.json',config);dump(output/'manifest.json',manifest)
    for name in ('codebook.npz','config.json','manifest.json'):result['file_sha256'][name]=sha(output/name)
    pipe=initial=prompt=negative=z=v=snapshot=off=terminal=u=last=None
    try:
        import diffusers
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        result['source_sha256']={str(p):sha(p) for p in [Path(__file__),MANIFEST,Path(carrier.__file__),Path(state_clock.__file__),Path(method.__file__),Path(runtime.__file__),Path(sys.modules[prepare_generation.__module__].__file__)]}
        result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        result['resolved_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        for name,value in [('initial',initial),('prompt',prompt),('negative',negative)]:artifact(name,value)
        z44,snapshot44,precision=runtime.shared_prefix(pipe,initial,prompt,negative,dtype,config['generation']['guidance_scale'],count)
        result['precision']=precision;result['shared_input_fingerprint']=runtime.fingerprint(z44);result['shared_history_fingerprint']=runtime.fingerprint(vars(snapshot44))
        artifact('prefix44',z44);torch.save(snapshot44,output/'prefix_scheduler.pt');result['file_sha256']['prefix_scheduler.pt']=sha(output/'prefix_scheduler.pt')
        result['scheduler']=dict(config=dict(snapshot44.config),sigmas=snapshot44.sigmas.tolist(),timesteps=snapshot44.timesteps.tolist())
        def continue_branch(message):
            current=z44.clone();scheduler=copy.deepcopy(snapshot44);rows=[]
            for index in range(44,49):
                velocity=runtime.velocity(pipe,current,scheduler,prompt,negative,dtype,config['generation']['guidance_scale'],index,count)
                current,row=runtime.advance(scheduler,current,velocity,book,message,index,count);rows.append(row)
            velocity=runtime.velocity(pipe,current,scheduler,prompt,negative,dtype,config['generation']['guidance_scale'],49,count)
            return current,velocity,scheduler,rows
        def store_path(name,terminal,rows,input49,scheduler49):
            store(name,terminal);item=result['videos'][name];item['steps']=rows;item['cumulative_control']=runtime.cumulative(rows)
            item['pre49_input_fingerprint']=runtime.fingerprint(input49);item['pre49_history_fingerprint']=runtime.fingerprint(vars(scheduler49))
            item['pre_final_margin']=rows[-1]['predicted_clean_before'];item['final_step']=rows[-1]
        try:
            z,v,snapshot,rows=continue_branch(None)
            off,offrow=runtime.final_fork(snapshot,z,v,book,None,count);store_path('OFF',off,rows+[offrow],z,snapshot)
            for m,suffix in enumerate(('A','B')):
                try:
                    last,row=runtime.final_fork(snapshot,z,v,book,m,count,uncontrolled=off)
                    store_path('LAST_'+suffix,last,rows+[row],z,snapshot)
                    direct,_=carrier.write(off.cpu().numpy(),book,m)
                    result['equivalence']['LAST_'+suffix+'_vs_CPU_DIRECT']=method.numerical_equivalence(last.cpu().numpy(),direct)
                except Exception as exc:result['videos']['LAST_'+suffix]['status']='FAILED';fail('LAST_'+suffix,exc)
        except Exception as exc:
            for name in ('OFF','LAST_A','LAST_B'):
                if result['videos'][name]['status']=='NOT_RUN':result['videos'][name]['status']='FAILED'
            fail('baseline_branch',exc)
        finally:z=v=snapshot=off=last=None;release();save()
        for m,suffix in enumerate(('A','B')):
            try:
                z,v,snapshot,rows=continue_branch(m)
                artifact('EARLY_'+suffix+'_pre49',z)
                early,earlyrow=runtime.final_fork(snapshot,z,v,book,None,count)
                store_path('EARLY_ONLY_'+suffix,early,rows+[earlyrow],z,snapshot)
                last,row=runtime.final_fork(snapshot,z,v,book,m,count,uncontrolled=early)
                store_path('EARLY_LAST_'+suffix,last,rows+[row],z,snapshot)
            except Exception as exc:
                for name in ('EARLY_ONLY_'+suffix,'EARLY_LAST_'+suffix):
                    if result['videos'][name]['status']=='NOT_RUN':result['videos'][name]['status']='FAILED'
                fail('EARLY_'+suffix,exc)
            finally:z=v=snapshot=last=early=None;release();save()
        if runtime.fingerprint(vars(snapshot44))!=result['shared_history_fingerprint']:raise RuntimeError('shared prefix history mutated')
    except Exception as exc:fail('generation',exc)
    finally:
        pipe=initial=prompt=negative=z=v=snapshot=off=terminal=u=last=None
        z44=snapshot44=None;release()
    for name in ARMS:
        try:
            terminal=torch.load(output/(name+'_terminal.pt'),map_location='cpu',weights_only=True)
            off=torch.load(output/'OFF_terminal.pt',map_location='cpu',weights_only=True)
            result['videos'][name]['net_terminal_vs_OFF']=runtime.measures(terminal-off)
        except Exception as exc:result['videos'][name]['net_terminal_vs_OFF']={'status':'MISSING','error':repr(exc)}
        finally:terminal=off=None
    result['status']='GENERATION_COMPLETE' if all(v['status']=='TERMINAL_PERSISTED' for v in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    save();return result


def media_case(case_id,output):
    output=Path(output);result=load(output/'generation.json');config=load(output/'config.json');book={k:v for k,v in np.load(output/'codebook.npz').items()}
    if load(output/'manifest.json')!=load(MANIFEST):raise ValueError('source manifest differs')
    for name in ('codebook.npz','config.json','manifest.json'):
        if sha(output/name)!=result['file_sha256'].get(name):raise ValueError('source metadata hash differs: '+name)
    if result.get('source_commit')!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip():raise ValueError('source generation commit differs')
    vae=None;off_precodec_valid=False;off_received_valid=False
    def save():dump(output/'result.json',result)
    def count(k,done):result['actual_calls'][k+('_completed' if done else '_attempted')]+=1;save()
    def fail(stage,exc):result['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    # All available conditions continue through independent media/readout paths.
    try:
        if vae is None and any((output/f'{a}_terminal.pt').exists() for a in ARMS):
            vae = load_frozen_vae(config)
        if vae is not None:
            vae.to('cuda')
        for name in ARMS:
            rgb = z = obs = encoded = None
            saved_this_attempt=False
            item = result['videos'][name]
            path = output/'received_videos'/f'{name}.mp4'
            try:
                terminal_path=output/f'{name}_terminal.pt'
                if sha(terminal_path)!=result['file_sha256'].get(terminal_path.name):raise ValueError('source terminal hash mismatch')
                z = torch.load(terminal_path, map_location='cpu', weights_only=True)
                count('vae_decode', False)
                rgb = decode_normalized_latent(vae, z.to(next(vae.parameters()).device)).cpu()
                count('vae_decode', True)
                if tuple(rgb.shape)!=RGB_SHAPE or not torch.isfinite(rgb).all():raise ValueError('nonfinite or wrong decoded RGB geometry')
                torch.save(rgb, output/f'{name}_precodec_rgb.pt')
                if name=='OFF':off_precodec_valid=True
                if name != 'OFF' and off_precodec_valid:
                    reference = torch.load(output/'OFF_precodec_rgb.pt', weights_only=True)
                    item['precodec_quality_vs_off'] = quality(reference, rgb)
                    del reference
                count('mp4_save', False)
                encode_rgb(rgb, path, 8, 18)
                count('mp4_save', True)
                result['file_sha256'][str(path.relative_to(output))]=sha(path)
                item.update(status='VIDEO_PERSISTED', path=str(path.relative_to(output)));saved_this_attempt=True
            except Exception as exc:
                item['status'] = 'FAILED_MEDIA'
                fail(name+'/media', exc)
            finally:
                rgb = z = None
                if vae is not None:
                    _clear_cache(vae)
                release()
            if not saved_this_attempt:
                for row in item['observations'].values():row.update(status='NOT_RUN_FAILED_MEDIA')
                save();continue
            obs = {}
            try:
                rgb = read_mp4(path)
                if tuple(rgb.shape)!=RGB_SHAPE or not torch.isfinite(rgb).all():raise ValueError('nonfinite or wrong readback RGB geometry')
                if len(rgb) != 181:
                    raise ValueError('normal MP4 requires 181 frames')
                if name=='OFF':off_received_valid=True
                if name != 'OFF' and not off_received_valid:item['saved_quality_vs_off']={'status':'MISSING_OFF_REFERENCE'}
                if name != 'OFF' and off_received_valid:
                    reference = read_mp4(output/'received_videos/OFF.mp4')
                    item['saved_quality_vs_off'] = quality(reference, rgb)
                    del reference
                for g in range(4):
                    row = item['observations'][str(g)]
                    try:
                        count('vae_encode', False)
                        groups, tail = divmod(len(rgb)-g-1, 4)
                        encoded = reencode_rgb24_readback(vae, rgb[g:g+1+4*groups]).cpu().float()
                        count('vae_encode', True)
                        if tuple(encoded.shape) != (1,16,1+groups,40,64) or not torch.isfinite(encoded).all():
                            raise ValueError('receiver latent geometry mismatch')
                        dest = output/'receiver_observations'/name
                        dest.mkdir(parents=True, exist_ok=True)
                        torch.save(encoded, dest/f'g{g}.pt');result['file_sha256'][str((dest/f'g{g}.pt').relative_to(output))]=sha(dest/f'g{g}.pt')
                        obs[g] = encoded.numpy()
                        row.update(status='COMPLETE', frames_used=1+4*groups, tail_discarded=tail)
                    except Exception as exc:
                        row.update(status='FAILED', error=repr(exc))
                        fail(name+f'/g{g}', exc)
                    finally:
                        encoded = None
                        _clear_cache(vae)
                        release()
                        save()
            except Exception as exc:
                fail(name+'/readback', exc)
            try:
                detection = state_clock.read(obs, book)
                dump(output/'detections'/f'{name}.json', detection)
                item['rankings'] = detection['rankings']
                if name != 'OFF':
                    truth = 0 if name.endswith('A') else 1
                    item['reporting_only'] = state_clock.report(detection, truth, 0)
                    item['five_methods_unique_correct'] = all(r['message_unique'] and r['best']['message']==truth for r in detection['rankings'].values()) if len(obs)==4 else None
                item['status'] = 'COMPLETE' if len(obs)==4 else 'PARTIAL_OR_FAILED'
            except Exception as exc:
                fail(name+'/detection', exc)
            rgb = obs = encoded = None
            release()
            save()
    except Exception as exc:
        fail('media_setup', exc)
    finally:
        vae = None
        release()
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    result['receiver_origin_meaning']='internal RGB phase hypotheses from full received MP4; not real temporal-crop attack'
    result['resources']=dict(cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None)
    save();return result


def summary(cases):
    out={}
    for group in ('LAST','EARLY_LAST','EARLY_ONLY'):
        out[group]={}
        for mode in MODES:
            done=correct=0;gaps=[]
            for case in CASES:
                for m,suffix in enumerate(('A','B')):
                    v=cases.get(case,{}).get('videos',{}).get(group+'_'+suffix,{})
                    rank=v.get('rankings',{}).get(mode)
                    if v.get('status')!='COMPLETE' or rank is None:continue
                    done+=1;correct+=int(rank['message_unique'] and rank['best']['message']==m)
                    by=rank.get('best_by_message',{});own=by.get(str(m));other=by.get(str(1-m))
                    if own is not None and other is not None:gaps.append(own['score']-other['score'])
            out[group][mode]=dict(marked_denominator=8,complete=done,missing_or_failed=8-done,unique_correct=correct,margin_available=len(gaps),margin_min=min(gaps) if gaps else None,margin_max=max(gaps) if gaps else None)
    return out


def mechanism_summary(cases):
    rows=[]
    for case in CASES:
        for arm in ARMS:
            v=cases.get(case,{}).get('videos',{}).get(arm,{});c=v.get('cumulative_control',{});final=v.get('final_step',{})
            message=None if arm=='OFF' else str(0 if arm.endswith('A') else 1)
            pre=v.get('pre_final_margin',{}).get('messages',{}).get(message,{})
            rows.append(dict(case=case,arm=arm,status=v.get('status','MISSING'),controlled_steps=c.get('controlled_steps'),
                actual_D_support=c.get('actual_D',{}).get('support'),actual_D_global=c.get('actual_D',{}).get('global'),
                net_terminal_vs_OFF=v.get('net_terminal_vs_OFF'),pre49_correct_min_margin=pre.get('minimum_signed_projection'),pre49_correct_loss=pre.get('loss'),
                final49_controlled=final.get('controlled'),final49_u=final.get('u'),final49_actual_D=final.get('actual_D'),
                final_correct_min_margin=v.get('terminal_margin',{}).get(message,{}).get('minimum_signed_projection'),saved_rgb_psnr_db=v.get('saved_quality_vs_off',{}).get('rgb_psnr_db'),saved_residual_temporal_mse=v.get('saved_quality_vs_off',{}).get('residual_temporal_mse')))
    return rows


def run_all(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);manifest=load(MANIFEST);validate(manifest)
    result=dict(status='RUNNING',video_denominator=28,receiver_encode_denominator=112,cases={c:empty('NOT_RUN') for c in CASES},fixed_calls={k:4*v for k,v in PLAN.items()},scientific_pass=None,quality_tolerance=None)
    dump(output/'manifest.json',manifest);dump(output/'result.json',result)
    for stage in ('generate','media'):
        for case in CASES:
            log=output/(case+'_'+stage+'.log');print(case,stage,'started; log:',log,flush=True)
            try:
                with log.open('w') as f:child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.flow_tube_multistep_run','--output',str(output/case),'--case-id',case,'--stage',stage],stdout=f,stderr=subprocess.STDOUT,check=False)
                p=output/case/('generation.json' if stage=='generate' else 'result.json')
                previous=result['cases'][case];result['cases'][case]=load(p)|{k:v for k,v in previous.items() if k.endswith('_exit_code')}|{stage+'_exit_code':child.returncode}
            except Exception as exc:result['cases'][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(exc))
            dump(output/'result.json',result)
    result['mechanism_summary']=mechanism_summary(result['cases'])
    result['recovery_summary']=summary(result['cases'])
    result['quality_summary']={c:{a:{k:v.get(k) for k in ('status','precodec_quality_vs_off','saved_quality_vs_off')} for a,v in row['videos'].items()} for c,row in result['cases'].items()}
    result['control_summary']={c:{a:{'terminal_margin':v.get('terminal_margin'),'pre_final_margin':v.get('pre_final_margin'),'final_step':v.get('final_step'),'cumulative_control':v.get('cumulative_control'),'net_terminal_vs_OFF':v.get('net_terminal_vs_OFF'),'status':v.get('status')} for a,v in row['videos'].items()} for c,row in result['cases'].items()}
    result['equivalence_summary']={c:row.get('equivalence',{}) for c,row in result['cases'].items()}
    result['OFF_summary']={'denominator':4,'complete':sum(row['videos']['OFF']['status']=='COMPLETE' for row in result['cases'].values()),'claim':'rankings only, no FPR calibration'}
    result['actual_calls_observed']={k+'_'+s:sum(row.get('actual_calls',{}).get(k+'_'+s,0) for row in result['cases'].values()) for k in PLAN for s in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(row['status']=='EXECUTION_COMPLETE' and row.get('media_exit_code')==0 and row.get('generate_exit_code')==0 for row in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case-id',choices=CASES);p.add_argument('--stage',choices=('generate','media'));a=p.parse_args()
    if a.case_id and a.stage:r=generate_case(a.case_id,a.output) if a.stage=='generate' else media_case(a.case_id,a.output)
    elif not a.case_id and not a.stage:r=run_all(a.output)
    else:p.error('case and stage are paired internal child arguments')
    if r['status'] not in ('GENERATION_COMPLETE','EXECUTION_COMPLETE'):raise SystemExit(1)
