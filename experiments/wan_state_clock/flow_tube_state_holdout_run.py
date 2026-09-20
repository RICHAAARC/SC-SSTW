"""Fixed independent full-video holdout: original tube/state last-step guidance."""
import argparse,copy,gc,hashlib,json,platform,resource,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from main.tube_state import projection_margin as carrier,state_clock,terminal_guidance as method,flow_control
from runtime.wan import tube_terminal_guidance as runtime
from runtime.wan.generation import prepare_generation,load_frozen_vae
from runtime.wan.io import dump,read_mp4,encode_rgb
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,_clear_cache
from .flow_run import quality,terminal_record
RGB_SHAPE=(181,320,512,3)
ARMS=('OFF','LAST_A','LAST_B')
CASES=('holdout_p0_s0','holdout_p1_s0')
MODES=('global_matched','global_state','local_matched','local_without_update','local_state')
PLAN={'transformer':100,'scheduler_step':52,'vae_decode':3,'mp4_save':3,'vae_encode':12}
MANIFEST=Path(__file__).parent/'configs/flow_tube_state_holdout.json'

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def empty(status):return {'status':status,'videos':{a:{'status':'NOT_RUN','observations':{str(g):{'status':'NOT_RUN'} for g in range(4)}} for a in ARMS},'failures':[]}
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

def validate(manifest):
    base=load(MANIFEST.with_name('velocity_calibration.json'))
    config=copy.deepcopy(base['base_config']);config['generation']['role']='fixed_independent_full_video_last_projection'
    config['output_drive_parent']='/content/drive/MyDrive/Video-WM/FlowTubeStateHoldout'
    expected=dict(protocol='flow_tube_state_holdout_v1',base_config=config,holdout=base['holdout'],arms=list(ARMS),control_index=49,margin=1.,numeric_atol=2e-5,numeric_rtol=2e-4,receiver_origins=[0,1,2,3],quality_tolerance=None)
    if {c['prompt'] for c in base['holdout']}&{c['prompt'] for c in base['development']} or {c['seed'] for c in base['holdout']}&{c['seed'] for c in base['development']}:raise ValueError('holdout overlaps development')
    if manifest!=expected or carrier.MARGIN!=1.:raise ValueError('fixed method manifest mismatch')

def generate_case(case_id,output):
    manifest=load(MANIFEST);validate(manifest);case=next(v for v in manifest['holdout'] if v['id']==case_id)
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
        z,v,snapshot,precision=runtime.prepare_last(pipe,initial,prompt,negative,dtype,config['generation']['guidance_scale'],count)
        result['precision']=precision;result['shared_input_fingerprint']=runtime.fingerprint(z);result['shared_history_fingerprint']=runtime.fingerprint(vars(snapshot))
        artifact('prefix49',z);artifact('velocity49',v);torch.save(snapshot,output/'prefix_scheduler.pt');result['file_sha256']['prefix_scheduler.pt']=sha(output/'prefix_scheduler.pt')
        clean=z-float(snapshot.sigmas[49])*v;artifact('clean49',clean)
        result['scheduler']=dict(config=dict(snapshot.config),sigmas=snapshot.sigmas.tolist(),timesteps=snapshot.timesteps.tolist())
        off,info=runtime.final_step(snapshot,z,v,count);result['videos']['OFF']['step']=info;store('OFF',off)
        result['equivalence']['OFF_vs_clean49']=method.numerical_equivalence(off.cpu().numpy(),clean.cpu().numpy())
        for m,suffix in enumerate(('A','B')):
            direct=None
            try:
                direct,evidence=carrier.write(off.cpu().numpy(),book,m)
                dump(output/'write_evidence'/('CPU_DIRECT_'+suffix+'.json'),evidence)
            except Exception as exc:fail('CPU_DIRECT_'+suffix,exc)
            try:
                delta,projected,evidence=method.correction(clean.cpu().numpy(),book,m);u=torch.from_numpy(delta).to(z.device)
                artifact('LAST_'+suffix+'_u',u);artifact('LAST_'+suffix+'_projected_clean',torch.from_numpy(projected))
                last,info=runtime.final_step(snapshot,z,v,count,u=u);store('LAST_'+suffix,last)
                info.update(u_support_rms=flow_control.rms(delta),u_global_rms=flow_control.rms(delta,False),actual_response_support_rms=flow_control.rms((last-off).cpu().numpy()),
                    actual_response_global_rms=flow_control.rms((last-off).cpu().numpy(),False),delta_velocity_global_rms=float(((v-u/float(snapshot.sigmas[49]))-v).double().square().mean().sqrt()))
                result['videos']['LAST_'+suffix]['step']=info;dump(output/'write_evidence'/('LAST_'+suffix+'.json'),evidence)
                result['equivalence']['LAST_'+suffix+'_vs_projected_clean']=method.numerical_equivalence(last.cpu().numpy(),projected)
                if direct is not None:
                    result['equivalence']['LAST_'+suffix+'_vs_CPU_DIRECT']=method.numerical_equivalence(last.cpu().numpy(),direct)
            except Exception as exc:result['videos']['LAST_'+suffix]['status']='FAILED';fail('LAST_'+suffix,exc)
            finally:terminal=u=last=None;release();save()
        if runtime.fingerprint(vars(snapshot))!=result['shared_history_fingerprint']:raise RuntimeError('shared history mutated')
    except Exception as exc:fail('generation',exc)
    finally:pipe=initial=prompt=negative=z=v=snapshot=off=terminal=u=last=None;release()
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
    for group in ('LAST',):
        out[group]={}
        for mode in MODES:
            done=correct=0
            for case in CASES:
                for m,suffix in enumerate(('A','B')):
                    v=cases.get(case,{}).get('videos',{}).get(group+'_'+suffix,{})
                    rank=v.get('rankings',{}).get(mode)
                    if v.get('status')!='COMPLETE' or rank is None:continue
                    done+=1;correct+=int(rank['message_unique'] and rank['best']['message']==m)
            out[group][mode]=dict(marked_denominator=4,complete=done,missing_or_failed=4-done,unique_correct=correct)
    return out


def message_summary(cases):
    rows=[]
    for case in CASES:
        for arm in ARMS:
            item=cases.get(case,{}).get('videos',{}).get(arm,{})
            for mode in MODES:
                rank=item.get('rankings',{}).get(mode,{})
                by=rank.get('best_by_message',{});a=by.get('0');b=by.get('1')
                complete=item.get('status')=='COMPLETE' and len([v for v in item.get('observations',{}).values() if v.get('status')=='COMPLETE'])==4
                truth=None if arm=='OFF' else (0 if arm=='LAST_A' else 1)
                margin=None
                if truth is not None and a is not None and b is not None:margin=(a['score']-b['score'])*(1 if truth==0 else -1)
                best=rank.get('best')
                rows.append(dict(case=case,arm=arm,mode=mode,status=item.get('status','MISSING'),complete=complete,
                    receiver_origin_denominator=4,completed_origins=sum(v.get('status')=='COMPLETE' for v in item.get('observations',{}).values()),
                    nominal_support_denominator=1760,best_matched_supports=best.get('matched_supports') if best else None,
                    best_score_by_message={str(m):by.get(str(m),{}).get('score') if by.get(str(m)) else None for m in (0,1)},
                    reference_score_0_minus_1=(a['score']-b['score']) if a is not None and b is not None else None,
                    best_message=best.get('message') if best else None,message_unique=rank.get('message_unique'),
                    top_tie_count=len(rank.get('top_ties',[])),best_correct_minus_other_reporting_only=margin,
                    truth_reporting_only=truth,meaning='two-candidate attribution only; no payload-bit recovery or FPR'))
    return rows


def run_all(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);manifest=load(MANIFEST);validate(manifest)
    result=dict(status='RUNNING',video_denominator=6,receiver_encode_denominator=24,cases={c:empty('NOT_RUN') for c in CASES},fixed_calls={k:2*v for k,v in PLAN.items()},scientific_pass=None,quality_tolerance=None)
    dump(output/'manifest.json',manifest);dump(output/'result.json',result)
    for stage in ('generate','media'):
        for case in CASES:
            log=output/(case+'_'+stage+'.log');print(case,stage,'started; log:',log,flush=True)
            try:
                with log.open('w') as f:child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.flow_tube_state_holdout_run','--output',str(output/case),'--case-id',case,'--stage',stage],stdout=f,stderr=subprocess.STDOUT,check=False)
                p=output/case/('generation.json' if stage=='generate' else 'result.json')
                previous=result['cases'][case];result['cases'][case]=load(p)|{k:v for k,v in previous.items() if k.endswith('_exit_code')}|{stage+'_exit_code':child.returncode}
            except Exception as exc:result['cases'][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(exc))
            dump(output/'result.json',result)
    result['message_and_coverage_summary']=message_summary(result['cases'])
    result['recovery_summary']=summary(result['cases'])
    result['quality_summary']={c:{a:{k:v.get(k) for k in ('status','precodec_quality_vs_off','saved_quality_vs_off')} for a,v in row['videos'].items()} for c,row in result['cases'].items()}
    result['control_summary']={c:{a:{'terminal_margin':v.get('terminal_margin'),'step':v.get('step'),'status':v.get('status')} for a,v in row['videos'].items()} for c,row in result['cases'].items()}
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
