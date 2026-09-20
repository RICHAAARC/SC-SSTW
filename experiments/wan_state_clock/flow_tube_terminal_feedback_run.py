"""Fixed single-update terminal-feedback diagnostic, compared with local control."""
import argparse,copy,gc,hashlib,json,math,platform,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import projection_margin as carrier,state_clock,terminal_guidance as method,flow_control
from runtime.wan import tube_terminal_feedback as runtime
from main.tube_state import terminal_feedback as objective
from runtime.wan import tube_retention as retention
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.io import dump,read_mp4,encode_rgb
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,_clear_cache
from .flow_run import quality,terminal_record
RGB_SHAPE=(181,320,512,3)
ARMS=('OFF','LOCAL_A','LOCAL_B','TERMINAL_FEEDBACK_A','TERMINAL_FEEDBACK_B')
CASES=('dev_p0_s0','dev_p1_s0')
MODES=('global_matched','global_state','local_matched','local_without_update','local_state')
PLAN={'transformer':130,'scheduler_step':72,'unit_response_probe_step':4,'terminal_backward':2,'vae_decode':5,'mp4_save':5,'vae_encode':20}
MANIFEST=Path(__file__).parent/'configs/flow_tube_terminal_feedback.json'

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def empty(status):return {'status':status,'videos':{a:{'status':'NOT_RUN','observations':{str(g):{'status':'NOT_RUN'} for g in range(4)}} for a in ARMS},'failures':[]}
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def validate(manifest):
    if tuple(manifest['arms'])!=ARMS or manifest['write_indices']!=[46] or tuple(v['id'] for v in manifest['development'])!=CASES:raise ValueError('fixed feedback roster required')
    if carrier.MARGIN!=1. or manifest['base_config']['generation']['steps']!=50:raise ValueError('fixed tube/native50 required')

def generate_case(case_id,output):
    manifest=load(MANIFEST);validate(manifest);case=next(v for v in manifest['development'] if v['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=empty('RUNNING');result.update(case=case,config=config,fixed_calls=PLAN,actual_calls={k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},file_sha256={},equivalence={},quality_tolerance=None,scientific_pass=None,calls_by_path={})
    scope='OFF_reference'
    def save():dump(output/'generation.json',result)
    def count(k,done):
        key=k+('_completed' if done else '_attempted');result['actual_calls'][key]+=1
        bucket=result['calls_by_path'].setdefault(scope,{});bucket[key]=bucket.get(key,0)+1;save()
    def artifact(name,z):
        path=output/(name+'.pt');torch.save(z.detach().cpu(),path);result['file_sha256'][path.name]=sha(path)
    def fail(stage,e):result['failures'].append(dict(stage=stage,error=repr(e),traceback=traceback.format_exc()));save()
    book=state_clock.codebook(config['key_utf8'].encode());np.savez(output/'codebook.npz',**book);dump(output/'config.json',config);dump(output/'manifest.json',manifest)
    for name in ('codebook.npz','config.json','manifest.json'):result['file_sha256'][name]=sha(output/name)
    def store(name,z):
        if not torch.isfinite(z).all():raise FloatingPointError('nonfinite terminal')
        artifact(name+'_terminal',z);item=result['videos'][name];item['status']='TERMINAL_PERSISTED'
        item['terminal_objective']={str(m):objective.value(z,book,m) for m in (0,1)}
        item['terminal_projection']=runtime.projection_diagnostics(z,book);item['terminal_metrics_status']='COMPLETE'
    pipe=initial=prompt=negative=nodes=snapshots=preview=z=v=s=terminal=gradient=None
    try:
        import diffusers
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        result['source_sha256']={str(p):sha(p) for p in [Path(__file__),MANIFEST,Path(objective.__file__),Path(runtime.__file__),Path(retention.__file__),Path(state_clock.__file__),Path(carrier.__file__)]}
        result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False);device=initial.device;guidance=config['generation']['guidance_scale']
        result['resolved_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        for name,value in [('initial',initial),('prompt',prompt),('negative',negative)]:artifact(name,value)
        nodes,snapshots,precision=runtime.reference(pipe,initial,prompt,negative,dtype,guidance,count);result['precision']=precision
        torch.save(nodes,output/'OFF_nodes.pt');torch.save(snapshots,output/'OFF_snapshots.pt')
        for name in ('OFF_nodes.pt','OFF_snapshots.pt'):result['file_sha256'][name]=sha(output/name)
        result['reference_fingerprints']={str(t):dict(input=runtime.fingerprint(nodes[t]['z']),history=runtime.fingerprint(vars(snapshots[t]))) for t in retention.TIMES}
        result['scheduler']=dict(config=dict(snapshots[49].config),sigmas=snapshots[49].sigmas.tolist(),timesteps=snapshots[49].timesteps.tolist())
        store('OFF',nodes[50]['z']);targets={}
        for message,suffix in enumerate(('A','B')):
            try:
                scope='LAST_oracle_'+suffix
                last,_,row,_=retention.matched_update(snapshots[49],nodes[49]['z'].to(device),nodes[49]['v'].to(device),nodes[50]['z'].to(device),book,message,49,count)
                targets[message]=row['actual_D']['support_rms'];artifact('LAST_oracle_'+suffix,last)
                result.setdefault('LAST_oracle',{})[suffix]=row
            except Exception as e:fail(scope,e)
        z=nodes[46]['z'].to(device);v=nodes[46]['v'].to(device);scope='shared_terminal_preview'
        preview_record=None
        try:
            preview,preview_record=runtime.preview(pipe,snapshots[46],z,v,prompt,negative,dtype,guidance,count)
            result['preview']=preview_record;comparison=method.numerical_equivalence(preview.cpu().numpy(),nodes[50]['z'].numpy())
            comparison['support_relative_rms_error']=runtime.measures(preview.cpu()-nodes[50]['z'])['support_rms']/max(runtime.measures(nodes[50]['z'])['support_rms'],1e-30)
            result['equivalence']['preview_vs_OFF']=comparison;artifact('preview_terminal',preview)
            if not result['equivalence']['preview_vs_OFF']['pass']:raise RuntimeError('unchanged preview differs from OFF reference')
        except Exception as e:preview=None;fail(scope,e)
        for message,suffix in enumerate(('A','B')):
            gradient=None;scope='terminal_leaf_'+suffix
            if preview is not None:
                try:
                    count('terminal_backward',False);gradient,record=objective.terminal_gradient(preview,book,message);count('terminal_backward',True)
                    artifact('terminal_gradient_'+suffix,gradient);result.setdefault('terminal_gradients',{})[suffix]=record|{'norms':runtime.measures(gradient)}
                except Exception as e:fail(scope,e)
            for group in ('LOCAL','TERMINAL_FEEDBACK'):
                name=group+'_'+suffix;scope=name
                try:
                    if message not in targets:raise ValueError('missing positive LAST oracle budget')
                    if gradient is None:
                        # LOCAL still runs when preview/leaf fails; prediction unavailable, not fabricated.
                        if group=='TERMINAL_FEEDBACK':raise ValueError('terminal preview/gradient unavailable')
                        diagnostic_gradient=torch.zeros_like(z)
                    else:diagnostic_gradient=gradient
                    if group=='LOCAL':
                        raw,_,_=method.correction((z-float(snapshots[46].sigmas[46])*v).cpu().numpy(),book,message);direction=torch.from_numpy(raw).to(device)
                    else:direction=-gradient
                    after,s,row,arrays=runtime.matched_direction(snapshots[46],z,v,nodes[47]['z'].to(device),direction,diagnostic_gradient,targets[message],count)
                    if gradient is None:
                        row['predicted_terminal_loss_decrease_identity_tail']=None;row['clean_control_dot_negative_gradient_aux']=None
                    result['videos'][name]['control']=row
                    for label,value in arrays.items():artifact(name+'_'+label,value)
                    terminal,stages=runtime.continue_single(pipe,after,s,nodes,prompt,negative,dtype,guidance,46,book,count)
                    store(name,terminal);item=result['videos'][name];item['short_stages']=stages
                    item['OFF_terminal_objective']=objective.value(nodes[50]['z'],book,message)
                    item['actual_terminal_loss_decrease_vs_OFF']=item['OFF_terminal_objective']-item['terminal_objective'][str(message)]
                    item['net_terminal_vs_OFF']=runtime.measures(terminal.cpu()-nodes[50]['z'])
                except Exception as e:result['videos'][name]['status']='FAILED';fail(name,e)
                finally:after=s=terminal=direction=arrays=None;release();save()
        for t in retention.TIMES:
            if runtime.fingerprint(vars(snapshots[t]))!=result['reference_fingerprints'][str(t)]['history']:raise RuntimeError('OFF reference history mutated')
    except Exception as e:fail('generation',e)
    finally:pipe=initial=prompt=negative=nodes=snapshots=preview=z=v=s=terminal=gradient=None;release()
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
    for group in ('LOCAL','TERMINAL_FEEDBACK'):
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
            out[group][mode]=dict(marked_denominator=4,complete=done,missing_or_failed=4-done,unique_correct=correct,margin_available=len(gaps),margin_min=min(gaps) if gaps else None,margin_max=max(gaps) if gaps else None)
    return out



def receiver_margin(video,message,mode='local_state'):
    if video.get('status')!='COMPLETE':return None
    by=video.get('rankings',{}).get(mode,{}).get('best_by_message',{});own,other=by.get(str(message)),by.get(str(1-message))
    return None if own is None or other is None else own['score']-other['score']

def mechanism_summary(cases):
    rows=[]
    for case in CASES:
        videos=cases.get(case,{}).get('videos',{});off=videos.get('OFF',{})
        for arm in ARMS:
            row=videos.get(arm,{});c=row.get('control',{});message=None if arm=='OFF' else (0 if arm.endswith('A') else 1)
            margin=receiver_margin(row,message) if message is not None else None;offmargin=receiver_margin(off,message) if message is not None else None
            rows.append(dict(case=case,arm=arm,status=row.get('status','MISSING'),actual_D=c.get('actual_D'),target_D_support_rms=c.get('target_D_support_rms'),
                raw_direction=c.get('raw_direction'),applied_unit_scale=c.get('applied_unit_scale'),effective_raw_scale=c.get('effective_raw_scale'),matching_relative_error=c.get('matching_relative_error'),
                predicted_terminal_loss_decrease_identity_tail=c.get('predicted_terminal_loss_decrease_identity_tail'),actual_terminal_loss_decrease_vs_OFF=row.get('actual_terminal_loss_decrease_vs_OFF'),
                local_state_margin=margin,OFF_reference_margin=offmargin,local_state_gain_vs_OFF=None if margin is None or offmargin is None else margin-offmargin,
                saved_quality_vs_OFF=row.get('saved_quality_vs_off'),net_terminal_vs_OFF=row.get('net_terminal_vs_OFF')))
    return rows

def paired_summary(cases):
    rows=[]
    for case in CASES:
        videos=cases.get(case,{}).get('videos',{})
        for message,suffix in enumerate(('A','B')):
            local=videos.get('LOCAL_'+suffix,{});feedback=videos.get('TERMINAL_FEEDBACK_'+suffix,{})
            terminal_complete=local.get('terminal_metrics_status')=='COMPLETE' and feedback.get('terminal_metrics_status')=='COMPLETE'
            a=local.get('terminal_objective',{}).get(str(message)) if terminal_complete else None;b=feedback.get('terminal_objective',{}).get(str(message)) if terminal_complete else None
            ma,mb=receiver_margin(local,message),receiver_margin(feedback,message)
            rows.append(dict(case=case,message=message,LOCAL_status=local.get('status','MISSING'),FEEDBACK_status=feedback.get('status','MISSING'),
                terminal_pair_complete=terminal_complete,media_pair_complete=ma is not None and mb is not None,
                feedback_predicted_terminal_loss_decrease_identity_tail=feedback.get('control',{}).get('predicted_terminal_loss_decrease_identity_tail'),
                feedback_actual_terminal_loss_decrease_vs_OFF=feedback.get('actual_terminal_loss_decrease_vs_OFF') if terminal_complete else None,
                LOCAL_actual_terminal_loss_decrease_vs_OFF=local.get('actual_terminal_loss_decrease_vs_OFF') if terminal_complete else None,
                actual_terminal_loss_decrease_vs_LOCAL=None if a is None or b is None else a-b,
                MP4_local_state_margin_gain_vs_LOCAL=None if ma is None or mb is None else mb-ma))
    return dict(fixed_pair_denominator=4,terminal_complete=sum(r['terminal_pair_complete'] for r in rows),media_complete=sum(r['media_pair_complete'] for r in rows),rows=rows,claim='two correlated development contents; no posterior arm selection, FPR, quality pass or trajectory success')
def run_all(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);manifest=load(MANIFEST);validate(manifest)
    result=dict(status='RUNNING',video_denominator=10,receiver_encode_denominator=40,cases={c:empty('NOT_RUN') for c in CASES},fixed_calls={k:2*v for k,v in PLAN.items()},scientific_pass=None,quality_tolerance=None)
    dump(output/'manifest.json',manifest);dump(output/'result.json',result)
    for stage in ('generate','media'):
        for case in CASES:
            log=output/(case+'_'+stage+'.log');print(case,stage,'started; log:',log,flush=True)
            try:
                with log.open('w') as f:child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.flow_tube_terminal_feedback_run','--output',str(output/case),'--case-id',case,'--stage',stage],stdout=f,stderr=subprocess.STDOUT,check=False)
                p=output/case/('generation.json' if stage=='generate' else 'result.json')
                previous=result['cases'][case];result['cases'][case]=load(p)|{k:v for k,v in previous.items() if k.endswith('_exit_code')}|{stage+'_exit_code':child.returncode}
            except Exception as exc:result['cases'][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(exc))
            dump(output/'result.json',result)
    result['mechanism_summary']=mechanism_summary(result['cases'])
    result['paired_summary']=paired_summary(result['cases'])
    result['recovery_summary']=summary(result['cases'])
    result['quality_summary']={c:{a:{k:v.get(k) for k in ('status','precodec_quality_vs_off','saved_quality_vs_off')} for a,v in row['videos'].items()} for c,row in result['cases'].items()}
    result['equivalence_summary']={c:row.get('equivalence',{}) for c,row in result['cases'].items()}
    result['OFF_summary']={'denominator':2,'complete':sum(row['videos']['OFF']['status']=='COMPLETE' for row in result['cases'].values()),'claim':'rankings only, no FPR calibration'}
    result['actual_calls_observed']={k+'_'+s:sum(row.get('actual_calls',{}).get(k+'_'+s,0) for row in result['cases'].values()) for k in PLAN for s in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(row['status']=='EXECUTION_COMPLETE' and row.get('media_exit_code')==0 and row.get('generate_exit_code')==0 for row in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--case-id',choices=CASES);p.add_argument('--stage',choices=('generate','media'));a=p.parse_args()
    if a.case_id and a.stage:r=generate_case(a.case_id,a.output) if a.stage=='generate' else media_case(a.case_id,a.output)
    elif not a.case_id and not a.stage:r=run_all(a.output)
    else:p.error('case and stage are paired internal child arguments')
    if r['status'] not in ('GENERATION_COMPLETE','EXECUTION_COMPLETE'):raise SystemExit(1)
