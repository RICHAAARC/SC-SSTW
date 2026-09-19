"""Shared-state step30 local versus input Jacobian direction. No model execution on import."""
import argparse
import copy
import gc
import hashlib
import json
import platform
import subprocess
import sys
import traceback
from pathlib import Path
import torch
from main.tube_state import grow_temporal_difference as method
from runtime.wan import grow_single_step_jacobian as runtime
from runtime.wan.grow_paired_control import observation
from main.tube_state import grow_control_budget as budget
from runtime.wan.grow_control_transfer import fingerprint
from runtime.wan.generation import prepare_generation
from runtime.wan.io import dump

CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
ARMS=runtime.ARMS
MESSAGES={'OFF':None,'LOCAL_A':0,'LOCAL_B':1,'JAC_A':0,'JAC_B':1}
PLAN={'transformer':256,'scheduler_step':130,'local_gradient':2,'input_vjp':2,'unit_response_probe_step':2}
MANIFEST=Path(__file__).parent/'configs/grow_single_step_jacobian.json'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def validate(manifest):
    old=load(MANIFEST.with_name('grow_temporal_difference.json'))
    expected=copy.deepcopy(old);expected['protocol']='grow_single_step_jacobian_v1'
    expected['frequency']['controlled_indices']=[30]
    expected['matching']='single_step_native_response_rms_squared'
    expected['arms']=list(ARMS);expected['messages']=MESSAGES
    if manifest!=expected or method.ETA!=.1 or method.AMPLITUDE!=.5:raise ValueError('fixed single-step protocol mismatch')


def empty(status):
    return dict(status=status,videos={a:dict(status=status,steps=[],terminal={'status':'NOT_RUN',
        'latent_time_denominator':46,'pair_denominator':23,'per_pair':[{'pair':k,'times':[2*k,2*k+1],'status':'NOT_RUN'} for k in range(23)]}) for a in ARMS})


def terminal_summary(cases):
    result={}
    for group in ('LOCAL','JAC'):
        good=done=errors=erasures=0
        for case in CASES:
            for suffix in ('A','B'):
                item=cases.get(case,{}).get('videos',{}).get(group+'_'+suffix,{})
                if item.get('status')!='COMPLETE':continue
                row=item['terminal'];comp=row['payload_comparisons_reporting_only']['aggregate'][0 if suffix=='A' else 1]
                done+=1;good+=int(comp['exact_payload_match']);errors+=comp['bit_errors_including_erasures'];erasures+=row['aggregate']['bit_erasures']
        result[group]=dict(marked_denominator=8,completed=done,missing=8-done,exact=good,errors_observed=errors,erasures_observed=erasures,does_not_control_media=True)
    return result


def run_case(case_id,output):
    manifest=load(MANIFEST);validate(manifest)
    case=next(c for c in manifest['development'] if c['id']==case_id)
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'latents').mkdir()
    result=empty('RUNNING');result.update(case=case,config=config,fixed_calls=PLAN,
        actual_calls={k+'_'+s:0 for k in PLAN for s in ('attempted','completed')},failures=[],
        file_sha256={},tensor_fingerprints={},scientific_pass=None,video_denominator=5,
        historical_comparison={'status':'NOT_ESTABLISHED','required':['source parameters','actual schedule','initial tensor hash','environment'],
            'claim':'no causal carrier advantage inferred without complete matching evidence'})
    def save():dump(output/'result.json',result)
    def count(kind,done):result['actual_calls'][kind+('_completed' if done else '_attempted')]+=1;save()
    def artifact(name,value):
        path=output/'latents'/(name+'.pt');torch.save(value.detach().cpu(),path)
        result['file_sha256'][str(path.relative_to(output))]=hashlib.sha256(path.read_bytes()).hexdigest()
        result['tensor_fingerprints'][name]=fingerprint(value)
    def fail(stage,exc):result['failures'].append(dict(stage=stage,error=repr(exc),traceback=traceback.format_exc()));save()
    book=method.codebook(config['key_utf8'].encode());dump(output/'codebook.json',book);dump(output/'config.json',config)
    pipe=initial=prompt=negative=origin=z=None
    try:
        import diffusers
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),MANIFEST,Path(method.__file__),Path(method.spatial.__file__),Path(budget.__file__),Path(runtime.__file__),
             Path(sys.modules[fingerprint.__module__].__file__)]}
        result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        origin=copy.deepcopy(pipe.scheduler)
        for name,value in [('initial',initial),('prompt',prompt),('negative',negative)]:artifact(name,value)
        result['resolved_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        result['scheduler']=dict(config=dict(origin.config),sigmas=origin.sigmas.tolist(),timesteps=origin.timesteps.tolist())
        dump(output/'schedule.json',result['scheduler'])
        result['schedule_sha256']=hashlib.sha256((output/'schedule.json').read_bytes()).hexdigest()
        checkpoint_stats={};runtime.enable_input_checkpointing(pipe.transformer,checkpoint_stats)
        result['checkpoint_stats']=checkpoint_stats
        result['prefix_steps']=[]
        shared=runtime.prefix(pipe,initial,prompt,negative,dtype,config['generation']['guidance_scale'],count,
            lambda row:result['prefix_steps'].append(row))
        artifact('shared_prefix_step30',shared)
        shared_scheduler=copy.deepcopy(pipe.scheduler)
        torch.save(shared_scheduler,output/'prefix_scheduler.pt')
        result['file_sha256']['prefix_scheduler.pt']=hashlib.sha256((output/'prefix_scheduler.pt').read_bytes()).hexdigest()
        result['prefix_history_fingerprint']=fingerprint(vars(shared_scheduler))
        from runtime.wan.grow_control_transfer import cfg,rms
        velocity=cfg(pipe,shared,prompt,negative,dtype,config['generation']['guidance_scale'],30,count)
        artifact('shared_velocity_step30',velocity)
        post={};local_gradients={}
        # Complete every feasible arm's single step before any tail execution.
        for arm in ARMS:
            item=result['videos'][arm];gradient=None
            try:
                pipe.scheduler=copy.deepcopy(shared_scheduler)
                item['shared_input_fingerprint']=fingerprint(shared)
                item['shared_history_fingerprint']=fingerprint(vars(pipe.scheduler))
                if item['shared_history_fingerprint']!=result['prefix_history_fingerprint']:raise RuntimeError('history clone mismatch')
                kind=arm.split('_')[0]
                if arm!='OFF':
                    if kind=='JAC' and ('LOCAL_'+arm[-1]) not in post:raise RuntimeError('paired LOCAL feasibility missing')
                    gradient,info=runtime.direction(pipe,shared,velocity,prompt,negative,dtype,config['generation']['guidance_scale'],book,MESSAGES[arm],kind,count,checkpoint_stats)
                    item['direction']=info|{'resources':copy.deepcopy(checkpoint_stats['last_direction_resources'])}
                    if kind=='LOCAL':local_gradients[arm[-1]]=gradient.cpu()
                    else:
                        local=local_gradients[arm[-1]].to(gradient.device)
                        dot=float((local.double()*gradient.double()).sum());den=float(local.double().norm()*gradient.double().norm())
                        item['direction'].update(g_clean_rms=rms(local),g_input_rms=rms(gradient),cosine=dot/den if den else None)
                        del local
                    artifact(arm+'_direction',gradient)
                z,row=runtime.one_step(pipe,shared.clone(),velocity,book,MESSAGES[arm],gradient,count,
                    off_next=post['OFF'][0] if arm!='OFF' else None,
                    target_energy=result['videos']['LOCAL_'+arm[-1]]['steps'][0]['actual_response_energy'] if kind=='JAC' else None)
                item['steps'].append(row);item['feasibility']='FINITE_SINGLE_STEP_COMPLETE'
                item['actual_response_energy']=row['actual_response_energy'];item['peak_control_rms']=row['control_rms']
                item['sum_step_control_rms_squared']=row['control_rms']**2
                item['sum_step_delta_velocity_rms_squared']=row['delta_velocity_rms']**2
                artifact(arm+'_step30_after',z)
                post[arm]=(z.detach(),copy.deepcopy(pipe.scheduler))
            except Exception as exc:
                item['status']='FAILED';item['feasibility']='FAILED_NO_FALLBACK'
                item['last_direction_resources']=copy.deepcopy(checkpoint_stats.get('last_direction_resources'))
                item['failure_direction_diagnostics']=copy.deepcopy(checkpoint_stats.get('last_direction_diagnostics'))
                fail(arm+'_single_step',exc)
            finally:
                gradient=z=None;gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache();item['allocated_after_direction_release']=torch.cuda.memory_allocated()
                save()
        local_gradients.clear()
        for arm in ARMS:
            item=result['videos'][arm]
            if arm not in post:continue
            try:
                z,pipe.scheduler=post.pop(arm)
                z=runtime.tail(pipe,z,prompt,negative,dtype,config['generation']['guidance_scale'],count,
                    lambda row:item['steps'].append(row))
                artifact(arm+'_terminal',z)
                decoded=method.read(z.cpu(),book)
                decoded['payload_comparisons_reporting_only']=method.compare_payloads(decoded,book)
                item['terminal']=decoded;item['terminal_candidate_losses']=[float(method.loss(z,book,m)) for m in (0,1)]
                item['actual_controlled_steps']=sum(row['controlled'] for row in item['steps'])
                if len(item['steps'])!=20 or item['actual_controlled_steps']!=(0 if arm=='OFF' else 1):raise RuntimeError('incomplete shared-prefix continuation')
                item['status']='COMPLETE'
            except Exception as exc:item['status']='FAILED';fail(arm+'_tail',exc)
            finally:z=None;gc.collect();save()
    except Exception as exc:fail('setup',exc)
    finally:
        pipe=initial=prompt=negative=origin=z=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    result['terminal_net_contrasts']={}
    for name,left,right in [(a+'_minus_OFF',a,'OFF') for a in ARMS if a!='OFF']+[(g+'_A_minus_B',g+'_A',g+'_B') for g in ('LOCAL','JAC')]:
        try:
            a=torch.load(output/'latents'/f'{left}_terminal.pt',map_location='cpu',weights_only=True)
            b=torch.load(output/'latents'/f'{right}_terminal.pt',map_location='cpu',weights_only=True)
            delta=a-b
            result['terminal_net_contrasts'][name]=dict(status='MEASURED',rms=float(delta.double().square().mean().sqrt()),
                readout=observation(delta,book),meaning='whole trajectory net contrast, not same-history local control response')
            del a,b,delta
        except Exception as exc:result['terminal_net_contrasts'][name]=dict(status='MISSING_OR_FAILED',error=repr(exc))
    save();return result


def run_all(output):
    manifest=load(MANIFEST);validate(manifest)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',case_denominator=4,video_denominator=20,cases={c:empty('NOT_RUN') for c in CASES},
        fixed_calls={k:v*4 for k,v in PLAN.items()},scientific_pass=None)
    dump(output/'manifest.json',manifest);dump(output/'result.json',result)
    for case in CASES:
        log=output/(case+'.log');print(case,'started; log:',log,flush=True)
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.grow_single_step_jacobian_run',
                    '--output',str(output/case),'--case-id',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            result['cases'][case]=load(output/case/'result.json')|{'exit_code':child.returncode}
        except Exception as exc:result['cases'][case]=empty('FAILED_LAUNCH_OR_RESULT')|{'error':repr(exc)}
        dump(output/'result.json',result)
    result['terminal_recovery_summary']=terminal_summary(result['cases'])
    result['actual_calls_observed']={k+'_'+s:sum(c.get('actual_calls',{}).get(k+'_'+s,0) for c in result['cases'].values()) for k in PLAN for s in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(c['status']=='EXECUTION_COMPLETE' and c.get('exit_code')==0 for c in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--case-id',choices=CASES)
    args=parser.parse_args();r=run_case(args.case_id,args.output) if args.case_id else run_all(args.output)
    if r['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
