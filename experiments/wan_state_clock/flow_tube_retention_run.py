"""Fixed single-write retention diagnostic with LAST native-response RMS reference."""
import argparse,copy,gc,hashlib,json,math,platform,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import projection_margin as carrier,state_clock,terminal_guidance as method,flow_control
from runtime.wan import tube_retention as runtime
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.io import dump,read_mp4,encode_rgb
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,_clear_cache
from .flow_run import quality,terminal_record
RGB_SHAPE=(181,320,512,3)
ARMS=('OFF','T44_A','T44_B','T46_A','T46_B','T49_A','T49_B')
CASES=('dev_p0_s0','dev_p1_s0')
MODES=('global_matched','global_state','local_matched','local_without_update','local_state')
PLAN={'transformer':132,'scheduler_step':72,'unit_response_probe_step':4,'vae_decode':7,'mp4_save':7,'vae_encode':28}
MANIFEST=Path(__file__).parent/'configs/flow_tube_retention.json'

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def empty(status):return {'status':status,'videos':{a:{'status':'NOT_RUN','observations':{str(g):{'status':'NOT_RUN'} for g in range(4)}} for a in ARMS},'failures':[]}
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def validate(manifest):
    base=load(MANIFEST.with_name('flow_tube_state_guidance.json'))
    base.update(protocol='flow_tube_retention_v1',arms=list(ARMS),write_indices=[44,46,49],
        budget='same_case_message_LAST_native_D_support_RMS',short_free_steps=1,
        future_holdout={'enabled':False,'selection_rule':None,'policy':'freeze one shared rule before independent validation; no per-video timing selection'})
    base.pop('control_index');base['development']=[v for v in base['development'] if v['id'] in CASES]
    base['base_config']['generation']['role']='single_write_retention_diagnostic'
    base['base_config']['output_drive_parent']='/content/drive/MyDrive/Video-WM/FlowTubeRetention'
    if manifest!=base or carrier.MARGIN!=1.:raise ValueError('fixed retention protocol mismatch')


def generate_case(case_id,output):
    manifest=load(MANIFEST);validate(manifest);case=next(v for v in manifest['development'] if v['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=empty('RUNNING');result.update(case=case,config=config,fixed_calls=PLAN,actual_calls={k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},file_sha256={},equivalence={},quality_tolerance=None,scientific_pass=None)
    def save():dump(output/'generation.json',result)
    scope='OFF_reference';result['calls_by_path']={}
    def count(k,done):
        key=k+('_completed' if done else '_attempted');result['actual_calls'][key]+=1
        bucket=result['calls_by_path'].setdefault(scope,{});bucket[key]=bucket.get(key,0)+1;save()
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
        result['source_sha256']={str(p):sha(p) for p in [Path(__file__),MANIFEST,Path(carrier.__file__),Path(state_clock.__file__),Path(method.__file__),Path(runtime.__file__),Path(sys.modules['runtime.wan.tube_multistep'].__file__),Path(sys.modules[prepare_generation.__module__].__file__)]}
        result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        result['resolved_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        for name,value in [('initial',initial),('prompt',prompt),('negative',negative)]:artifact(name,value)
        guidance=config['generation']['guidance_scale']
        nodes,snapshots,precision=runtime.reference(pipe,initial,prompt,negative,dtype,guidance,count)
        result['precision']=precision;result['reference_fingerprints']={str(t):{'input':runtime.fingerprint(nodes[t]['z']),'history':runtime.fingerprint(vars(snapshots[t]))} for t in runtime.TIMES}
        torch.save(nodes,output/'OFF_nodes.pt');torch.save(snapshots,output/'OFF_snapshots.pt')
        for name in ('OFF_nodes.pt','OFF_snapshots.pt'):result['file_sha256'][name]=sha(output/name)
        result['scheduler']=dict(config=dict(snapshots[49].config),sigmas=snapshots[49].sigmas.tolist(),timesteps=snapshots[49].timesteps.tolist())
        store('OFF',nodes[50]['z']);device=initial.device
        targets={}
        for index in (49,44,46):
            for m,suffix in enumerate(('A','B')):
                name=f'T{index}_{suffix}'
                try:
                    if index!=49 and m not in targets:raise ValueError('missing LAST same-message diagnostic reference')
                    scope=name
                    z=nodes[index]['z'].to(device);v=nodes[index]['v'].to(device);snapshot=snapshots[index]
                    z,s,row,arrays=runtime.matched_update(snapshot,z,v,nodes[index+1]['z'].to(device),book,m,index,count,targets.get(m) if index!=49 else None)
                    if index==49:
                        targets[m]=row['actual_D']['support_rms'];row['target_D_support_rms']=targets[m];row['matching_relative_error']=0.;row['matching_signed_error']=0.
                        direct,_=carrier.write(nodes[50]['z'].numpy(),book,m)
                        result['equivalence'][name+'_vs_CPU_DIRECT']=method.numerical_equivalence(z.cpu().numpy(),direct)
                    item=result['videos'][name];item['control']=row
                    for label,tensor in arrays.items():artifact(name+'_'+label,tensor)
                    terminal,stages=runtime.continue_single(pipe,z,s,nodes,prompt,negative,dtype,guidance,index,book,count)
                    item['short_stages']=stages;store(name,terminal)
                    item['terminal_projection']=runtime.projection_diagnostics(terminal,book)
                    item['OFF_terminal_projection']=runtime.projection_diagnostics(nodes[50]['z'],book)
                except Exception as exc:result['videos'][name]['status']='FAILED';fail(name,exc)
                finally:z=v=snapshot=terminal=None;release();save()
        for t in runtime.TIMES:
            if runtime.fingerprint(vars(snapshots[t]))!=result['reference_fingerprints'][str(t)]['history']:raise RuntimeError('shared OFF history changed')
    except Exception as exc:fail('generation',exc)
    finally:
        pipe=initial=prompt=negative=z=v=snapshot=off=terminal=u=last=None
        nodes=snapshots=s=arrays=None;release()
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
    for group in ('T44','T46','T49'):
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


def mechanism_summary(cases):
    rows=[]
    for case in CASES:
        off=cases.get(case,{}).get('videos',{}).get('OFF',{})
        for arm in ARMS:
            v=cases.get(case,{}).get('videos',{}).get(arm,{});c=v.get('control',{});message=None if arm=='OFF' else (0 if arm.endswith('A') else 1)
            def gap(diag):
                val=diag.get('nominal_score_A_minus_B')
                return None if val is None or message is None else val*(1 if message==0 else -1)
            def gain(diag,ref):
                a,b=gap(diag),gap(ref);return None if a is None or b is None else a-b
            stages=v.get('short_stages',{})
            denom=c.get('actual_D',{}).get('support_rms')
            def ratio(value):return None if not denom or value is None else value/denom
            retention={stage:{domain:ratio(stages.get(stage,{}).get(domain+'_response',{}).get('support_rms')) for domain in ('state','clean')} for stage in ('immediate_next','one_free_step')}
            retention['terminal_state']=ratio(v.get('net_terminal_vs_OFF',{}).get('support_rms'))
            readout={}
            for mode in MODES:
                def receiver_gap(video):
                    if video.get('status')!='COMPLETE':return None
                    by=video.get('rankings',{}).get(mode,{}).get('best_by_message',{})
                    a,b=by.get(str(message)),by.get(str(1-message)) if message is not None else None
                    return None if a is None or b is None else a['score']-b['score']
                own,ref=receiver_gap(v),receiver_gap(off)
                readout[mode]={'marked_margin':own,'OFF_reference_margin':ref,'gain_vs_OFF':None if own is None or ref is None else own-ref}
            rows.append(dict(case=case,arm=arm,status=v.get('status','MISSING'),index=c.get('index'),target_D_support_rms=c.get('target_D_support_rms'),
                actual_D=c.get('actual_D'),matching_relative_error=c.get('matching_relative_error'),scale=c.get('scale'),u=c.get('u'),delta_velocity=c.get('delta_velocity'),
                current_clean_gain=gain(c.get('controlled_clean',{}),c.get('before_clean',{})),
                immediate_clean_gain=gain(stages.get('immediate_next',{}).get('clean',{}),stages.get('immediate_next',{}).get('OFF_clean',{})),
                short_clean_gain=gain(stages.get('one_free_step',{}).get('clean',{}),stages.get('one_free_step',{}).get('OFF_clean',{})),
                terminal_projection_gain=gain(v.get('terminal_projection',{}),v.get('OFF_terminal_projection',{})),
                net_terminal_vs_OFF=v.get('net_terminal_vs_OFF'),response_norm_ratios=retention,receiver=readout,calls=cases.get(case,{}).get('calls_by_path',{}).get('OFF_reference' if arm=='OFF' else arm),
                saved_rgb_psnr_db=v.get('saved_quality_vs_off',{}).get('rgb_psnr_db'),quality_claim='relative OFF residual only; perceptual quality not measured'))
    return rows


def predictive_diagnostics(rows):
    lookup={(r['case'],r['arm']):r for r in rows}
    slots=[lookup.get((case,f'T{t}_{suffix}'),{}) for case in CASES for t in (44,46) for suffix in ('A','B')]
    reports={}
    for predictor in ('current_clean_gain','short_clean_gain'):
        for outcome in ('terminal_projection_gain','MP4_local_state_gain'):
            report=dict(fixed_pair_denominator=8,valid=0,missing_or_nonfinite=0,same_nonzero_sign=0,opposite_nonzero_sign=0,predictor_zero_only=0,outcome_zero_only=0,both_zero=0)
            for row in slots:
                x=row.get(predictor);y=row.get('terminal_projection_gain') if outcome=='terminal_projection_gain' else row.get('receiver',{}).get('local_state',{}).get('gain_vs_OFF')
                if x is None or y is None or not math.isfinite(x) or not math.isfinite(y):report['missing_or_nonfinite']+=1;continue
                report['valid']+=1
                key='both_zero' if x==0 and y==0 else 'predictor_zero_only' if x==0 else 'outcome_zero_only' if y==0 else 'same_nonzero_sign' if (x>0)==(y>0) else 'opposite_nonzero_sign'
                report[key]+=1
            reports[predictor+'__'+outcome]=report
    return dict(comparisons=reports,excluded_reference_times=[49],base_case_denominator=2,
        claim='fixed eight correlated case/message/time slots; descriptive sign agreement only, no significance or timing selection; exact numeric zero has no tuned threshold')


def run_all(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);manifest=load(MANIFEST);validate(manifest)
    result=dict(status='RUNNING',video_denominator=14,receiver_encode_denominator=56,cases={c:empty('NOT_RUN') for c in CASES},fixed_calls={k:2*v for k,v in PLAN.items()},scientific_pass=None,quality_tolerance=None)
    dump(output/'manifest.json',manifest);dump(output/'result.json',result)
    for stage in ('generate','media'):
        for case in CASES:
            log=output/(case+'_'+stage+'.log');print(case,stage,'started; log:',log,flush=True)
            try:
                with log.open('w') as f:child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.flow_tube_retention_run','--output',str(output/case),'--case-id',case,'--stage',stage],stdout=f,stderr=subprocess.STDOUT,check=False)
                p=output/case/('generation.json' if stage=='generate' else 'result.json')
                previous=result['cases'][case];result['cases'][case]=load(p)|{k:v for k,v in previous.items() if k.endswith('_exit_code')}|{stage+'_exit_code':child.returncode}
            except Exception as exc:result['cases'][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(exc))
            dump(output/'result.json',result)
    result['mechanism_summary']=mechanism_summary(result['cases'])
    result['predictive_diagnostics']=predictive_diagnostics(result['mechanism_summary'])
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
