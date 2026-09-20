"""Fixed MSE multi/last paired-control preparation. No model execution on import."""
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
from main.tube_state import grow_frequency as method
from runtime.wan.grow_spatial_control import generate as base_generate,CONTROL_INDICES,observation
from main.tube_state import grow_control_budget as budget
from runtime.wan.grow_control_transfer import fingerprint
from runtime.wan.generation import prepare_generation
from runtime.wan.io import dump

CASES=('dev_p0_s0','dev_p0_s1','dev_p1_s0','dev_p1_s1')
ARMS=('OFF','MULTI_A','MULTI_B','LAST_A','LAST_B')
MESSAGES={'OFF':None,'MULTI_A':0,'MULTI_B':1,'LAST_A':0,'LAST_B':1}
PLAN={'transformer':500,'scheduler_step':250,'local_gradient':42,'response_probe_step':42,'unit_response_probe_step':2}
MANIFEST=Path(__file__).parent/'configs/grow_spatial_control.json'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def generate(*args,**kwargs):
    return base_generate(*args,**kwargs)

def validate(manifest):
    old=load(MANIFEST.with_name('grow_paired_control.json'))
    expected=copy.deepcopy(old);expected['protocol']='grow_spatial_control_v1'
    expected['frequency'].pop('temporal_carrier');expected['frequency']['spatial_carrier']='per_latent_time_whole_slice_DCT_no_time_difference'
    expected['frequency']['votes_per_bit']=184
    expected['base_config']['generation']['role']='fixed_spatial_carrier_multi_last_media_completion'
    expected['base_config']['output_drive_parent']='/content/drive/MyDrive/Video-WM/GROWSpatialCarrier'
    if manifest!=expected or method.ETA!=.1 or method.AMPLITUDE!=.5 or CONTROL_INDICES!=tuple(range(30,50)):raise ValueError('fixed spatial paired protocol mismatch')


def empty(status):
    return dict(status=status,videos={a:dict(status=status,steps=[],terminal={'status':'NOT_RUN',
        'latent_time_denominator':46,'per_time':[{'time':k,'status':'NOT_RUN'} for k in range(46)]}) for a in ARMS})


def terminal_summary(cases):
    result={}
    for group in ('MULTI','LAST'):
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
        historical_comparison={'status':'NOT_ESTABLISHED','required':['source parameters','resolved model revision','actual schedule','initial and complete history tensor hashes','FP32/CFG precision','environment'],
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
            [Path(__file__),MANIFEST,Path(method.__file__),Path(budget.__file__),Path(sys.modules[base_generate.__module__].__file__),Path(sys.modules[generate.__module__].__file__),
             Path(sys.modules[fingerprint.__module__].__file__)]}
        result['environment']=dict(python=platform.python_version(),torch=str(torch.__version__),diffusers=diffusers.__version__)
        pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False)
        origin=copy.deepcopy(pipe.scheduler)
        for name,value in [('initial',initial),('prompt',prompt),('negative',negative)]:artifact(name,value)
        result['resolved_model_revision']=getattr(pipe.transformer.config,'_commit_hash',None)
        result['scheduler']=dict(config=dict(origin.config),sigmas=origin.sigmas.tolist(),timesteps=origin.timesteps.tolist())
        dump(output/'schedule.json',result['scheduler'])
        result['schedule_sha256']=hashlib.sha256((output/'schedule.json').read_bytes()).hexdigest()
        for arm in ARMS:
            item=result['videos'][arm]
            try:
                pipe.scheduler=copy.deepcopy(origin)
                def record(row):item['steps'].append(row);save()
                if arm.startswith('LAST') and result['videos']['MULTI_'+arm[-1]]['status']!='COMPLETE':raise RuntimeError('missing paired multi budget')
                z=generate(pipe,initial.clone(),prompt,negative,dtype,config['generation']['guidance_scale'],book,
                    MESSAGES[arm],count,record,
                    lambda name,value:artifact(arm+'_'+name,value),control_indices=(49,) if arm.startswith('LAST') else tuple(manifest['frequency']['controlled_indices']),
                    target_energy=budget.cumulative_energy([x.get('control_induced_delta_rms',0.) for x in result['videos']['MULTI_'+arm[-1]]['steps']]) if arm.startswith('LAST') else None)
                artifact(arm+'_terminal',z)
                decoded=method.read(z.cpu(),book)
                decoded['payload_comparisons_reporting_only']=method.compare_payloads(decoded,book)
                item['terminal']=decoded
                item['spatial_coefficient_rms']=float(method.selected(z,book).double().square().mean().sqrt())
                item['terminal_candidate_losses']=[float(method.loss(z,book,m)) for m in (0,1)]
                item['terminal_writer_objective_losses']=[float(method.loss(z,book,m)) for m in (0,1)]
                item['actual_controlled_steps']=sum(r['controlled'] for r in item['steps'])
                if len(item['steps'])!=50 or item['actual_controlled_steps']!=(0 if arm=='OFF' else (1 if arm.startswith('LAST') else 20)):raise RuntimeError('incomplete control schedule')
                item['cumulative_rms']={key:sum(r.get(key,0.) for r in item['steps']) for key in
                    ('control_rms','effective_clean_control_rms','control_induced_delta_rms')}
                item['actual_response_energy']=budget.cumulative_energy([r.get('control_induced_delta_rms',0.) for r in item['steps']])
                item['sum_step_delta_velocity_rms_squared']=budget.cumulative_energy([r.get('delta_velocity_rms',0.) for r in item['steps']])
                item['peak_delta_velocity_rms']=max(r.get('delta_velocity_rms',0.) for r in item['steps'])
                item['sum_delta_velocity_rms']=sum(r.get('delta_velocity_rms',0.) for r in item['steps'])
                item['peak_control_rms']=max(r.get('control_rms',0.) for r in item['steps'])
                item['peak_actual_response_rms']=max(r.get('control_induced_delta_rms',0.) for r in item['steps'])
                item['sum_step_control_rms_squared']=sum(r.get('control_rms',0.)**2 for r in item['steps'])
                item['control_energy_meaning']='sum of full-latent per-step mean squared local control; not energy of accumulated trajectory displacement'
                item['cumulative_rms_meaning']='sum of per-step RMS values, not RMS of accumulated state displacement'
                item['status']='COMPLETE'
            except Exception as exc:item['status']='FAILED';fail(arm,exc)
            finally:z=None;gc.collect();save()
    except Exception as exc:fail('setup',exc)
    finally:
        pipe=initial=prompt=negative=origin=z=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='COMPLETE' for v in result['videos'].values()) and not result['failures'] else 'WITH_RETAINED_FAILURES'
    result['terminal_net_contrasts']={}
    for name,left,right in [(a+'_minus_OFF',a,'OFF') for a in ARMS if a!='OFF']+[(g+'_A_minus_B',g+'_A',g+'_B') for g in ('MULTI','LAST')]:
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
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.grow_spatial_control_run',
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
