"""One-case, one-pulse native UniPC transmission diagnostic; no media/model on import."""
from __future__ import annotations
import argparse
import copy
import gc
import hashlib
import inspect
import json
import platform
import resource
import subprocess
import time
import traceback
from pathlib import Path
import torch
from main.tube_state import grow_frequency as method
from runtime.wan import grow_control_transfer as rt
from runtime.wan.flow_generation import continue_steps
from runtime.wan.generation import prepare_generation
from runtime.wan.io import dump

ARMS=('OFF','A','B')
MANIFEST=Path(__file__).parent/'configs/grow_control_transfer.json'
PLAN={'one_step':{'transformer':22,'scheduler_step':13,'local_gradient':2,'response_probe_step':2},
      'tail':{'transformer':234,'scheduler_step':117,'local_gradient':0,'response_probe_step':0}}


def empty_arm():
    return {'status':'NOT_RUN','one_step_status':'NOT_RUN','tail_status':'NOT_RUN',
        'terminal':{'status':'NOT_RUN','latent_time_denominator':46,'per_time':[{'time':i,'status':'NOT_RUN'} for i in range(46)]},
        'observations':[]}


def release():
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    for name in ('snapshots','observations'):(output/name).mkdir()
    manifest=json.loads(MANIFEST.read_text());case=manifest['development'][0]
    if case['id']!='dev_p0_s0' or manifest['diagnostic']['index']!=10:raise ValueError('fixed case/index required')
    config=copy.deepcopy(manifest['base_config']);config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    result={'status':'RUNNING','case':case,'arm_denominator':3,'latent_time_denominator':46,
        'arms':{a:empty_arm() for a in ARMS},'fixed_calls':PLAN,'failures':[], 'stages':{},
        'actual_calls':{stage:{k+'_'+d:0 for k in kinds for d in ('attempted','completed')} for stage,kinds in PLAN.items()},
        'numerical_guard':{'atol_scale':rt.ATOL,'rtol_expected':rt.RTOL,'response_strength_or_sign_gate':False},
        'scientific_pass':None,'claim':'new same-state single-pulse diagnostic; not old-run exact replay',
        'media_calls':0,'transformer_backward_calls':0}
    active='one_step';started=time.monotonic()
    def save():result['elapsed_seconds']=time.monotonic()-started;dump(output/'result.json',result)
    def count(k,done):
        result['actual_calls'][active][k+('_completed' if done else '_attempted')]+=1
        if done and k=='transformer' and result['actual_calls'][active][k+'_completed']%10==0:print(active,'Transformer completed',result['actual_calls'][active][k+'_completed'],flush=True)
        save()
    def fail(where,exc):result['failures'].append({'where':where,'error':repr(exc),'traceback':traceback.format_exc()});save()
    def stage_start(name):
        result['stages'][name]={'status':'RUNNING','started_elapsed':time.monotonic()-started}
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        print(name,'started',flush=True);save()
    def stage_end(name,status):
        row=result['stages'][name];row.update(status=status,elapsed_seconds=time.monotonic()-started-row['started_elapsed'],
            process_cumulative_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None)
        print(name,status,flush=True);save()
    def snapshot(path,z,scheduler):
        value={'latent':rt.tree_cpu(z),'scheduler_state':rt.tree_cpu(scheduler.__dict__)}
        value['fingerprint']=rt.fingerprint(value);torch.save(value,path);return value['fingerprint']
    def observation(index,arm,row):
        row.setdefault('status','MEASURED')
        row['model_input_index']=11 if index==10 else index if index<50 else None
        row['cast_is_actual_model_input']=index<50
        path=output/'observations'/f'{arm}_{index:02d}.json';dump(path,row)
        result.setdefault('contrasts',[]).append({'arm':arm,'index':index,'path':str(path)}) if arm not in ARMS else result['arms'][arm]['observations'].append({'index':index,'path':str(path)})
        save()
    book=method.codebook(config['key_utf8'].encode());dump(output/'codebook.json',book)
    dump(output/'manifest.json',manifest);dump(output/'config.json',config);save()
    pipe=initial=prefix=prompt=negative=common_v=common_clean=None
    branches={};updates={};original=None
    stage_start('one_step')
    try:
        import diffusers
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        result['source_dirty']=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        rt.require_scheduler(pipe.scheduler)
        if tuple(initial.shape)!=method.SHAPE or not rt.finite((initial,prompt,negative)):raise ValueError('invalid fixed latent/conditioning')
        result['environment']={'python':platform.python_version(),'torch':str(torch.__version__),'diffusers':diffusers.__version__,
            'runtime_scheduler_module':type(pipe.scheduler).__module__,
            'scheduler_source_sha256':hashlib.sha256(Path(inspect.getfile(type(pipe.scheduler))).read_bytes()).hexdigest(),
            'model_id':config['model']['id'],'requested_model_revision':config['model'].get('revision'),
            'resolved_transformer_revision':getattr(pipe.transformer.config,'_commit_hash',None),
            'conditioning_saved':True,'input_dtype':str(dtype),'cfg':'completed original model-output arithmetic then float32',
            'cuda_device':torch.cuda.get_device_name() if torch.cuda.is_available() else None}
        torch.save({'prompt':rt.tree_cpu(prompt),'negative':rt.tree_cpu(negative),'guidance_scale':config['generation']['guidance_scale'],
                    'environment':result['environment']},output/'conditioning.pt')
        torch.save(rt.tree_cpu(initial),output/'initial_latent.pt')
        prefix=continue_steps(pipe,pipe.scheduler,initial,prompt,negative,dtype,5.,0,10,count,precision={})
        original=copy.deepcopy(pipe.scheduler)
        if original.step_index!=10 or not rt.finite(original.__dict__):raise ValueError('invalid prefix history')
        result['prefix_fingerprint']=snapshot(output/'snapshots/pre10.pt',prefix,original)
        result['scheduler']={'class':type(original).__name__,'config':dict(original.config),
            'sigmas':original.sigmas.tolist(),'timesteps':original.timesteps.tolist()}
        common_v=rt.cfg(pipe,prefix,prompt,negative,dtype,5.,10,count)
        common_clean=prefix-float(original.sigmas[10])*common_v
        torch.save(rt.tree_cpu(common_v),output/'snapshots/shared_velocity10.pt')
        response=rt.scalar_response(original,count);result['native_scalar_response']=response;save()
        for arm in ARMS:
            row=result['arms'][arm]
            try:
                s=copy.deepcopy(original);z=prefix.clone()
                clone_equal=rt.fingerprint((z,s.__dict__))==rt.fingerprint((prefix,original.__dict__))
                if not clone_equal:raise ValueError('branch snapshot mismatch')
                u,v,clean,local=rt.pulse(z,common_v,float(s.sigmas[10]),book,None if arm=='OFF' else ARMS.index(arm)-1,count)
                count('scheduler_step',False);updated=s.step(v,s.timesteps[10],z,return_dict=False)[0];count('scheduler_step',True)
                checks={'clone_equal':clone_equal,'finite':rt.finite((updated,s.__dict__,u)),
                    'cursor':s.step_index==11,'converted_current':rt.check_delta(s.model_outputs[-1],clean,z,clean)}
                if 'algebra' in local:checks['local_algebra']=local['algebra']
                row.update(one_step_status='COMPLETE',local=local,numerical_checks=checks)
                branches[arm]={'z':updated.detach(),'scheduler':s};updates[arm]=u.detach()
                torch.save({'u':rt.tree_cpu(u),'controlled_velocity':rt.tree_cpu(v),'clean_after':rt.tree_cpu(clean)},output/'snapshots'/f'{arm}_pulse10.pt')
                row['poststep_fingerprint']=snapshot(output/'snapshots'/f'{arm}_post10.pt',updated,s)
            except Exception as exc:row['one_step_status']='FAILED';fail(arm+'/one_step',exc)
            save()
        if len(branches)==3:
            off=branches['OFF'];offz=off['z'];offs=off['scheduler']
            for arm in ('A','B'):
                b=branches[arm];u=updates[arm];s=b['scheduler'];d=b['z']-offz
                checks=result['arms'][arm]['numerical_checks']
                checks.update(native_transmission=rt.check_delta(d,response['K_clean']*u,b['z'],offz),
                    history_delta=rt.check_delta(s.model_outputs[-1]-offs.model_outputs[-1],u,s.model_outputs[-1],offs.model_outputs[-1]),
                    corrected_sample_delta=rt.check_delta(s.last_sample-offs.last_sample,response['K_corrected_sample']*u,s.last_sample,offs.last_sample),
                    older_history_equal=rt.fingerprint(s.model_outputs[:-1])==rt.fingerprint(offs.model_outputs[:-1]),
                    history_structure_equal=(s.step_index==offs.step_index and s.this_order==offs.this_order and s.lower_order_nums==offs.lower_order_nums and rt.fingerprint(s.timestep_list)==rt.fingerprint(offs.timestep_list)))
                observation(10,arm,rt.transmission(b['z'],offz,u,book,dtype))
            ab_u=updates['A']-updates['B'];ab_d=branches['A']['z']-branches['B']['z']
            result['AB_native_transmission']=rt.check_delta(ab_d,response['K_clean']*ab_u,branches['A']['z'],branches['B']['z'])
            spectrum=torch.zeros_like(prefix[:,0:1]);uv=torch.tensor(book['coordinates'],device=prefix.device)
            spectrum[...,uv[:,0],uv[:,1]]=2*method.ETA*method.target(book,0,prefix)
            expected=torch.zeros_like(prefix);expected[:,0:1]=method.idct2(spectrum)
            result['AB_payload_control']=rt.check_delta(ab_u,expected,common_clean)
            observation(10,'AB',rt.transmission(branches['A']['z'],branches['B']['z'],ab_u,book,dtype))
        else:observation(10,'AB',{'status':'MISSING_AB_STATE','reason':'one or more one-step branches failed'})
        def passed(x):return x.get('pass',False) if isinstance(x,dict) else x is True
        gate=len(branches)==3 and all(passed(v) for a in ARMS for v in result['arms'][a]['numerical_checks'].values())
        gate=gate and result['AB_native_transmission']['pass'] and result['AB_payload_control']['pass']
        result['numerical_gate_pass']=bool(gate)
        stage_end('one_step','NUMERICALLY_COMPATIBLE' if gate else 'NUMERICAL_CHECK_FAILED')
    except Exception as exc:
        fail('one_step/setup',exc);result['numerical_gate_pass']=False;stage_end('one_step','FAILED')
    if not (output/'observations/AB_10.json').exists():
        observation(10,'AB',{'status':'MISSING_AB_STATE','reason':'phase-one setup failed before contrast was available'})
    if result['numerical_gate_pass']:
        active='tail';stage_start('tail')
        for arm in ARMS:result['arms'][arm]['tail_status']='RUNNING'
        for index in range(11,50):
            values={}
            for arm in ARMS:
                if result['arms'][arm]['tail_status']!='RUNNING':continue
                try:
                    b=branches[arm];pipe.scheduler=b['scheduler']
                    v=rt.cfg(pipe,b['z'],prompt,negative,dtype,5.,index,count)
                    clean=b['z']-float(b['scheduler'].sigmas[index])*v
                    values[arm]=(v,clean)
                except Exception as exc:result['arms'][arm]['tail_status']='FAILED';fail(arm+f'/cfg{index}',exc)
            for arm in ARMS:
                if arm not in values:continue
                try:
                    b=branches[arm];v,clean=values[arm];s=b['scheduler']
                    if arm!='OFF' and 'OFF' in values:
                        offz=branches['OFF']['z'];offv,offclean=values['OFF'];sigma=float(s.sigmas[index])
                        dz=b['z']-offz;dv=v-offv;dc=clean-offclean
                        row=rt.transmission(b['z'],offz,updates[arm],book,dtype)
                        row.update(predicted_clean_delta=rt.stats(dc,book),velocity_contribution=rt.stats(-sigma*dv,book),
                            clean_delta_decomposition=rt.check_delta(dc,dz-sigma*dv,clean,offclean))
                        observation(index,arm,row)
                except Exception as exc:result['arms'][arm]['tail_status']='FAILED';fail(arm+f'/observation{index}',exc)
            if all(a in values and result['arms'][a]['tail_status']=='RUNNING' for a in ('A','B')):
                try:
                    az,bz=branches['A']['z'],branches['B']['z'];av,ac=values['A'];bv,bc=values['B']
                    sigma=float(branches['A']['scheduler'].sigmas[index]);dz=az-bz;dv=av-bv;dc=ac-bc
                    row=rt.transmission(az,bz,updates['A']-updates['B'],book,dtype)
                    row.update(predicted_clean_delta=rt.stats(dc,book),velocity_contribution=rt.stats(-sigma*dv,book),
                        clean_delta_decomposition=rt.check_delta(dc,dz-sigma*dv,ac,bc))
                    observation(index,'AB',row)
                except Exception as exc:
                    fail(f'AB/observation{index}',exc);observation(index,'AB',{'status':'FAILED_OBSERVATION','reason':repr(exc)})
            else:observation(index,'AB',{'status':'MISSING_AB_STATE','reason':{a:result['arms'][a]['tail_status'] for a in ('A','B')}})
            # Update only after all same-index comparisons; OFF must remain at the same node.
            for arm in ARMS:
                if result['arms'][arm]['tail_status']!='RUNNING':continue
                try:
                    b=branches[arm];s=b['scheduler'];v=values[arm][0]
                    count('scheduler_step',False);z=s.step(v,s.timesteps[index],b['z'],return_dict=False)[0];count('scheduler_step',True)
                    if s.step_index!=index+1 or not rt.finite((z,s.__dict__)):raise ValueError('invalid tail state/history')
                    b['z']=z.detach()
                except Exception as exc:result['arms'][arm]['tail_status']='FAILED';fail(arm+f'/step{index}',exc)
            print('tail step',index,'retained arms',[a for a in ARMS if result['arms'][a]['tail_status']=='RUNNING'],flush=True);save()
        for arm in ARMS:
            row=result['arms'][arm]
            if row['tail_status']!='RUNNING':continue
            try:
                b=branches[arm];row['final_fingerprint']=snapshot(output/'snapshots'/f'{arm}_terminal.pt',b['z'],b['scheduler'])
                decoded=method.read(b['z'],book);decoded['comparisons_reporting_only']=method.compare_payloads(decoded,book)
                decoded['truth_reporting_only']=None if arm=='OFF' else ARMS.index(arm)-1
                row.update(terminal=decoded,tail_status='COMPLETE')
                if arm!='OFF' and result['arms']['OFF']['tail_status']=='COMPLETE':
                    observation(50,arm,rt.transmission(b['z'],branches['OFF']['z'],updates[arm],book,dtype))
            except Exception as exc:row['tail_status']='FAILED';fail(arm+'/terminal',exc)
        if all(result['arms'][a]['tail_status']=='COMPLETE' for a in ('A','B')):
            observation(50,'AB',rt.transmission(branches['A']['z'],branches['B']['z'],updates['A']-updates['B'],book,dtype))
        else:observation(50,'AB',{'status':'MISSING_AB_STATE','reason':{a:result['arms'][a]['tail_status'] for a in ('A','B')}})
        stage_end('tail','RETURNED_WITH_RETAINED_ARM_STATUSES')
    else:
        result['stages']['tail']={'status':'SKIPPED_NUMERICAL_INVALIDITY','reason':'phase-one validity failed; no scientific response threshold'}
        for a in ARMS:result['arms'][a]['tail_status']='SKIPPED_NUMERICAL_INVALIDITY'
        for index in range(11,51):observation(index,'AB',{'status':'SKIPPED_NUMERICAL_INVALIDITY','reason':'phase-one validity failed'})
    for row in result['arms'].values():row['status']='COMPLETE' if row['one_step_status']=='COMPLETE' and row['tail_status']=='COMPLETE' else 'WITH_RETAINED_FAILURES'
    result['status']='EXECUTION_COMPLETE' if all(r['status']=='COMPLETE' for r in result['arms'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    pipe=initial=prefix=prompt=negative=common_v=common_clean=original=None;branches.clear();updates.clear();release()
    save();print(result['status'],'result:',output/'result.json',flush=True);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    result=run(parser.parse_args().output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)


if __name__=='__main__':main()
