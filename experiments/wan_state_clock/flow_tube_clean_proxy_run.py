"""Fixed saved46 hinge/tanh experiment; user-run models, same-batch OFF and three media layers."""
import argparse,copy,gc,hashlib,json,math,platform,resource,shutil,subprocess,sys,time,traceback
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from main.tube_state import state_clock,projection_margin,objective_alignment as objective,terminal_guidance
from runtime.wan import tube_clean_proxy as runtime
from runtime.wan.generation import load_frozen_vae
from runtime.wan.vae import decode_normalized_latent,reencode_rgb24_readback,quantize_rgb8_no_codec,_clear_cache
from runtime.wan.io import dump,encode_rgb,read_mp4
from .flow_run import quality

CASES=('dev_p0_s0','dev_p1_s0');ARMS=('OFF','HINGE_A','HINGE_B','TANH_A','TANH_B')
LAYERS=('floatRGB','RGB8','MP4');MODES=('global_matched','global_state','local_matched','local_without_update','local_state')
SOURCE_COMMIT='9fdfac97126fa681c42423089eb342ce85b676a7'
SOURCE_RUN='flow_tube_response_selection_20260921T013844126172Z'
PROTOCOL=Path(__file__).with_name('CLEAN_PROXY_PROTOCOL.md')
PROTOCOL_SHA='b5c2c8e232eff6f3db0631fc755e155cf12386077bf8e5133af31354d8724124'
PLAN=dict(transformer=30,scheduler_step=20,unit_response_probe_step=4,clean_leaf_backward=4,vae_decode=5,mp4_save=5,vae_encode=60)
RGB_SHAPE=(181,320,512,3)

def load(path):return json.loads(Path(path).read_text())
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
def empty(status):return dict(status=status,videos={a:dict(status='NOT_RUN',layers={l:dict(status='NOT_RUN',observations={str(g):dict(status='NOT_RUN') for g in range(4)}) for l in LAYERS}) for a in ARMS},failures=[])
def guard(source,output):
    source=Path(source).resolve();output=Path(output).resolve()
    if source==output or source in output.parents or output in source.parents:raise ValueError('input/output overlap')
    return source,output

def restore(source,case,result):
    root=source/case;generation=load(root/'generation.json')
    if generation['source_commit']!=SOURCE_COMMIT:raise ValueError('wrong original source')
    result['source_inputs']=[dict(path=str(root/'generation.json'),sha256=sha(root/'generation.json'))]
    files=('config.json','codebook.npz','OFF_nodes.pt','OFF_snapshots.pt','prompt.pt','negative.pt','OFF_terminal.pt')
    for name in files:
        actual=sha(root/name);expected=generation['file_sha256'].get(name)
        result['source_inputs'].append(dict(path=str(root/name),sha256=actual,expected_sha256=expected,matched=actual==expected))
        if actual!=expected:raise ValueError('source artifact hash mismatch: '+name)
    for module in (state_clock,projection_margin):
        suffix='/main/tube_state/'+Path(module.__file__).name
        expected=next(v for k,v in generation['source_sha256'].items() if k.endswith(suffix))
        if sha(module.__file__)!=expected:raise ValueError('old carrier/reader source mismatch')
    config=load(root/'config.json');book={k:v for k,v in np.load(root/'codebook.npz').items()}
    nodes=torch.load(root/'OFF_nodes.pt',map_location='cpu',weights_only=True)
    # Only load trusted source-hash-verified full scheduler objects. Keep serialized device placement.
    snapshots=torch.load(root/'OFF_snapshots.pt',weights_only=False);snapshot=snapshots[46]
    saved=generation['reference_fingerprints']['46']
    if snapshot.step_index!=46 or runtime.fingerprint(vars(snapshot))!=saved['history'] or runtime.fingerprint(nodes[46]['z'])!=saved['input']:raise ValueError('saved46 history/input mismatch')
    if len(snapshot.timesteps)!=50 or not snapshot.predict_x0 or snapshot.config.prediction_type!='flow_prediction' or snapshot.config.thresholding or snapshot.solver_p is not None:raise ValueError('original native50 flow scheduler required')
    targets={m:generation['videos']['LOCAL_'+s]['control']['actual_D']['support_rms'] for m,s in enumerate(('A','B'))}
    if any(not math.isfinite(v) or v<=0 for v in targets.values()):raise ValueError('invalid original LOCAL nativeD target')
    result.update(original_source=SOURCE_COMMIT,original_model_revision=generation.get('resolved_model_revision'),original_environment=generation.get('environment'),
                  original46_fingerprints=saved,native_targets=targets,budget_meaning='historical LOCAL actual native46 D, not terminal residual RMS')
    prompt=torch.load(root/'prompt.pt',map_location='cpu',weights_only=True);negative=torch.load(root/'negative.pt',map_location='cpu',weights_only=True)
    old_off=torch.load(root/'OFF_terminal.pt',map_location='cpu',weights_only=True)
    return config,book,nodes,snapshot,prompt,negative,old_off

def generate_case(case,source,output):
    source,output=guard(source,output);output.mkdir(parents=True,exist_ok=False)
    r=empty('RUNNING');r.update(case=case,protocol_sha256=sha(PROTOCOL),fixed_calls=PLAN,actual_calls={k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},file_sha256={},calls_by_path={},elapsed_seconds_by_path={},scientific_pass=None)
    scope='setup';pipe=nodes=snapshot=prompt=negative=z=v=off_next=terminal=old_off=None
    def save():dump(output/'generation.json',r)
    def fail(stage,exc):r['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    def count(k,done):
        key=k+('_completed' if done else '_attempted');r['actual_calls'][key]+=1
        bucket=r['calls_by_path'].setdefault(scope,{});bucket[key]=bucket.get(key,0)+1;save()
    @contextmanager
    def timed(name):
        nonlocal scope
        scope=name
        if torch.cuda.is_available():torch.cuda.synchronize()
        started=time.perf_counter()
        try:yield
        finally:
            if torch.cuda.is_available():torch.cuda.synchronize()
            r['elapsed_seconds_by_path'][name]=time.perf_counter()-started;save()
    def artifact(name,value):
        torch.save(value.detach().cpu(),output/name);r['file_sha256'][name]=sha(output/name)
    def store(name,terminal):
        if tuple(terminal.shape)!=projection_margin.SHAPE or not torch.isfinite(terminal).all():raise ValueError('invalid formal terminal')
        artifact(name+'_terminal.pt',terminal)
        d=torch.from_numpy(book['directions']).double();c=torch.from_numpy(book['codes']).double()
        r['videos'][name].update(status='TERMINAL_PERSISTED',terminal_metrics_status='COMPLETE',terminal_nominal={str(m):objective.metrics(terminal.cpu(),d,c,m) for m in (0,1)})
    try:
        if r['protocol_sha256']!=PROTOCOL_SHA:raise ValueError('frozen protocol mismatch')
        config,book,nodes,snapshot,prompt,negative,old_off=restore(source,case,r)
        import diffusers
        r['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        r['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        r['source_sha256']={str(p):sha(p) for p in (Path(__file__),Path(runtime.__file__),Path(objective.__file__),PROTOCOL)}
        pipe,dtype,model_record=runtime.load_transformer(config);r['new_model']=model_record
        if model_record.get('resolved_revision'):config=copy.deepcopy(config);config['model']['revision']=model_record['resolved_revision']
        dump(output/'config.json',config);shutil.copyfile(source/case/'codebook.npz',output/'codebook.npz')
        for name in ('config.json','codebook.npz'):r['file_sha256'][name]=sha(output/name)
        device=next(pipe.transformer.parameters()).device;z=nodes[46]['z'].to(device);v=nodes[46]['v'].to(device)
        prompt=prompt.to(device);negative=negative.to(device);guidance=config['generation']['guidance_scale']
        try:
            with timed('formal_OFF'):
                off_next,s,record=runtime.off_step(snapshot,z,v,count);r['videos']['OFF']['control']=record
                terminal,stages=runtime.continue_single(pipe,off_next,s,nodes,prompt,negative,dtype,guidance,46,book,count)
                store('OFF',terminal);r['videos']['OFF']['short_stages']=stages
                eq=terminal_guidance.numerical_equivalence(terminal.cpu().numpy(),old_off.numpy())
                eq.update(status='MATCH' if eq['pass'] else 'NONMATCH',support_difference_rms=runtime.measures(terminal.cpu()-old_off)['support_rms'],meaning='historical output compatibility only, not weight identity; new OFF is the same-batch formal baseline')
                r['historical_OFF_compatibility']=eq
        except Exception as exc:r['videos']['OFF']['status']='FAILED';fail('formal_OFF',exc)
        finally:terminal=s=None;release()
        clean=(z-float(snapshot.sigmas[46])*v).cpu()
        for group in ('HINGE','TANH'):
            for message,suffix in enumerate(('A','B')):
                arm=group+'_'+suffix;raw=q=terminal=arrays=None
                try:
                    if off_next is None:raise ValueError('same-history OFF native response unavailable')
                    with timed('clean_gradient_'+arm):
                        count('clean_leaf_backward',False);raw,record=runtime.clean_direction(clean,book,message,group.lower());count('clean_leaf_backward',True)
                        r['videos'][arm]['clean_gradient']=record
                    with timed('unit_'+arm):
                        q,epsilon,record=runtime.prepare_direction(snapshot,z,v,off_next,raw,r['native_targets'][message],count)
                        record['budget_meaning']='historical same-case/message LOCAL actual native46 D RMS, not terminal residual or newly measured future LAST'
                        if record['status']!='READY':raise ValueError('zero direction is a failed clean-proxy slot, not fallback')
                        r['videos'][arm]['unit_probe']=record
                    with timed('formal_'+arm):
                        terminal,record,stages,arrays=runtime.replay(pipe,snapshot,z,v,off_next,q,epsilon,1,r['native_targets'][message],nodes,prompt,negative,dtype,guidance,book,count)
                        record['budget_meaning']='historical same-case/message LOCAL actual native46 D RMS'
                        record['oracle_mismatch_meaning']='relative to recorded historical LOCAL native46 response target, not a new LAST or terminal residual'
                        r['videos'][arm]['control']=record;r['videos'][arm]['short_stages']=stages
                        for label,value in arrays.items():artifact(arm+'_'+label+'.pt',value)
                        store(arm,terminal)
                except Exception as exc:r['videos'][arm]['status']='FAILED';fail(arm,exc)
                finally:raw=q=terminal=arrays=None;release();save()
        if runtime.fingerprint(vars(snapshot))!=r['original46_fingerprints']['history']:raise ValueError('original46 snapshot mutated')
    except Exception as exc:fail('setup_or_generation',exc)
    finally:pipe=nodes=snapshot=prompt=negative=z=v=off_next=terminal=old_off=None;release()
    r['status']='GENERATION_COMPLETE' if all(v['status']=='TERMINAL_PERSISTED' for v in r['videos'].values()) and not r['failures'] else 'WITH_RETAINED_FAILURES'
    save();return r

def media_case(case,output):
    output=Path(output);r=load(output/'generation.json');vae=None;off_float_valid=off_mp4_valid=False
    def save():dump(output/'result.json',r)
    def fail(stage,exc):r['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    def count(k,done):r['actual_calls'][k+('_completed' if done else '_attempted')]+=1;save()
    try:
        for name in ('config.json','codebook.npz'):
            if sha(output/name)!=r['file_sha256'].get(name):raise ValueError('local metadata hash mismatch')
        if r.get('source_commit')!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip():raise ValueError('new generation/media source mismatch')
        config=load(output/'config.json');book={k:v for k,v in np.load(output/'codebook.npz').items()};vae=load_frozen_vae(config)
        r['new_vae_config_commit_hash']=getattr(vae.config,'_commit_hash',None)
        for arm in ARMS:
            item=r['videos'][arm];rgb=received=None;saved=False
            path=output/'videos'/(arm+'.mp4')
            try:
                terminal_path=output/(arm+'_terminal.pt')
                if sha(terminal_path)!=r['file_sha256'].get(terminal_path.name):raise ValueError('new terminal hash mismatch')
                z=torch.load(terminal_path,map_location='cpu',weights_only=True)
                count('vae_decode',False);rgb=decode_normalized_latent(vae,z.to(next(vae.parameters()).device)).cpu();count('vae_decode',True);z=None
                if tuple(rgb.shape)!=RGB_SHAPE or not torch.isfinite(rgb).all():raise ValueError('invalid floatRGB')
                if arm=='OFF':torch.save(rgb,output/'OFF_floatRGB.pt');off_float_valid=True
                try:
                    count('mp4_save',False);encode_rgb(rgb,path,8,18);count('mp4_save',True);saved=True
                    r['file_sha256'][str(path.relative_to(output))]=sha(path);item['video_path']=str(path.relative_to(output))
                except Exception as exc:fail(arm+'/MP4_save',exc)
            except Exception as exc:item['status']='FAILED_MEDIA';fail(arm+'/decode',exc)
            finally:release()
            for layer in LAYERS:
                row=item['layers'][layer];obs={};pixels=encoded=reference=None
                try:
                    if rgb is None:raise ValueError('decoded floatRGB unavailable')
                    if layer=='floatRGB':pixels=rgb
                    elif layer=='RGB8':pixels=quantize_rgb8_no_codec(rgb).float()/255
                    else:
                        if not saved:raise ValueError('no complete MP4 saved this attempt')
                        pixels=read_mp4(path)
                        if arm=='OFF' and tuple(pixels.shape)==RGB_SHAPE and torch.isfinite(pixels).all():off_mp4_valid=True
                    if tuple(pixels.shape)!=RGB_SHAPE or not torch.isfinite(pixels).all():raise ValueError('invalid actual layer raster')
                    row['actual_raster_dtype']=str(pixels.dtype);row['actual_raster_range']=[float(pixels.min()),float(pixels.max())]
                    if arm!='OFF':
                        if layer=='MP4' and off_mp4_valid:reference=read_mp4(output/'videos/OFF.mp4')
                        elif layer!='MP4' and off_float_valid:
                            reference=torch.load(output/'OFF_floatRGB.pt',map_location='cpu',weights_only=True)
                            if layer=='RGB8':reference=quantize_rgb8_no_codec(reference).float()/255
                        row['quality_vs_new_OFF']=quality(reference,pixels) if reference is not None else {'status':'MISSING_NEW_OFF_REFERENCE'}
                        reference=None
                    for g in range(4):
                        phase=row['observations'][str(g)]
                        try:
                            groups,tail=divmod(len(pixels)-g-1,4)
                            count('vae_encode',False);encoded=reencode_rgb24_readback(vae,pixels[g:g+1+4*groups]).cpu().float();count('vae_encode',True)
                            if tuple(encoded.shape)!=(1,16,1+groups,40,64) or not torch.isfinite(encoded).all():raise ValueError('invalid layer receiver latent')
                            dest=output/'observations'/arm/layer;dest.mkdir(parents=True,exist_ok=True)
                            torch.save(encoded,dest/f'g{g}.pt');r['file_sha256'][str((dest/f'g{g}.pt').relative_to(output))]=sha(dest/f'g{g}.pt')
                            obs[g]=encoded.numpy();phase.update(status='COMPLETE',frames_used=1+4*groups,tail_discarded=tail)
                        except Exception as exc:phase.update(status='FAILED',error=repr(exc));fail(arm+'/'+layer+f'/g{g}',exc)
                        finally:encoded=None;_clear_cache(vae);release();save()
                    detection=state_clock.read(obs,book);dump(output/'detections'/arm/(layer+'.json'),detection)
                    row['rankings']=detection['rankings']
                    if arm!='OFF':row['reporting_only']=state_clock.report(detection,0 if arm.endswith('A') else 1,0)
                    row['status']='COMPLETE' if len(obs)==4 else 'PARTIAL_OR_FAILED'
                except Exception as exc:row.update(status='FAILED',error=repr(exc));fail(arm+'/'+layer,exc)
                finally:pixels=obs=encoded=reference=None;_clear_cache(vae);release();save()
            item['status']='COMPLETE' if saved and all(v['status']=='COMPLETE' for v in item['layers'].values()) else 'PARTIAL_OR_FAILED'
            rgb=received=None;release();save()
    except Exception as exc:fail('media_setup',exc)
    finally:vae=None;release()
    r['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in r['videos'].values()) and not r['failures'] else 'WITH_RETAINED_FAILURES'
    r['resources']=dict(cpu_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
    save();return r

def receiver_record(video,layer,message,mode):
    record=video.get('layers',{}).get(layer,{})
    if record.get('status')!='COMPLETE':return None
    rank=record.get('rankings',{}).get(mode,{})
    by=rank.get('best_by_message',{});own=by.get(str(message));wrong=by.get(str(1-message))
    if own is None or wrong is None:return None
    return dict(correct_score=own['score'],wrong_score=wrong['score'],gap=own['score']-wrong['score'],unique_correct=rank.get('message_unique') and rank['best']['message']==message)

def summarize(cases):
    rows=[]
    for case in CASES:
        videos=cases.get(case,{}).get('videos',{})
        for message,suffix in enumerate(('A','B')):
            off=videos.get('OFF',{});h=videos.get('HINGE_'+suffix,{});t=videos.get('TANH_'+suffix,{})
            row=dict(case=case,message=message,terminal={},media={},controls={k:v.get('control') for k,v in (('HINGE',h),('TANH',t))})
            for name,v in (('OFF',off),('HINGE',h),('TANH',t)):row['terminal'][name]=v.get('terminal_nominal',{}).get(str(message)) if v.get('terminal_metrics_status')=='COMPLETE' else None
            row['terminal_tanh_minus_hinge_gap']=None if row['terminal']['HINGE'] is None or row['terminal']['TANH'] is None else row['terminal']['TANH']['hard_gap']-row['terminal']['HINGE']['hard_gap']
            for layer in LAYERS:
                row['media'][layer]={}
                for mode in MODES:
                    values={k:receiver_record(v,layer,message,mode) for k,v in (('OFF',off),('HINGE',h),('TANH',t))}
                    values['tanh_minus_hinge_gap']=None if values['HINGE'] is None or values['TANH'] is None else values['TANH']['gap']-values['HINGE']['gap']
                    values['gap_gain_vs_new_OFF']={k:None if values[k] is None or values['OFF'] is None else values[k]['gap']-values['OFF']['gap'] for k in ('HINGE','TANH')}
                    row['media'][layer][mode]=values
            rows.append(row)
    return dict(fixed_pair_denominator=4,rows=rows,primary_mode='local_state',claim='two development contents; clean46 proxy comparison, not terminal-gradient transport; no calibrated existence threshold or quality pass')

def run_all(source,output):
    source,output=guard(source,output);output.mkdir(parents=True,exist_ok=False)
    r=dict(status='RUNNING',video_denominator=10,media_layer_denominator=30,receiver_encode_denominator=120,
           fixed_calls={k:2*v for k,v in PLAN.items()},cases={c:empty('NOT_RUN') for c in CASES},protocol_sha256=sha(PROTOCOL),source_run=SOURCE_RUN)
    dump(output/'result.json',r);shutil.copyfile(PROTOCOL,output/'protocol.md')
    for stage in ('generate','media'):
        for case in CASES:
            path=output/(case+'_'+stage+'.log');print(case,stage,'started; log:',path,flush=True)
            try:
                with path.open('w') as log:child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.flow_tube_clean_proxy_run','--source',str(source),'--output',str(output/case),'--case-id',case,'--stage',stage],stdout=log,stderr=subprocess.STDOUT,check=False)
                previous=r['cases'][case];r['cases'][case]=load(output/case/('generation.json' if stage=='generate' else 'result.json'))|{k:v for k,v in previous.items() if k.endswith('_exit_code')}|{stage+'_exit_code':child.returncode}
            except Exception as exc:r['cases'][case].update(status='FAILED_LAUNCH_OR_RESULT',error=repr(exc))
            dump(output/'result.json',r)
    r['paired_summary']=summarize(r['cases'])
    r['actual_calls_observed']={k+'_'+s:sum(c.get('actual_calls',{}).get(k+'_'+s,0) for c in r['cases'].values()) for k in PLAN for s in ('attempted','completed')}
    r['historical_OFF_compatibility']={c:v.get('historical_OFF_compatibility') for c,v in r['cases'].items()}
    r['status']='EXECUTION_COMPLETE' if all(v['status']=='EXECUTION_COMPLETE' and v.get('generate_exit_code')==0 and v.get('media_exit_code')==0 for v in r['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',r);return r

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--case-id',choices=CASES);p.add_argument('--stage',choices=('generate','media'));a=p.parse_args()
    if a.case_id and a.stage:r=generate_case(a.case_id,a.source,a.output) if a.stage=='generate' else media_case(a.case_id,a.output)
    elif not a.case_id and not a.stage:r=run_all(a.source,a.output)
    else:p.error('paired internal child arguments required')
    if r['status'] not in ('GENERATION_COMPLETE','EXECUTION_COMPLETE'):raise SystemExit(1)
