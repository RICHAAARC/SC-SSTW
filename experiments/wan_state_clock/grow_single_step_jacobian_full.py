"""Fixed shared-step local versus input-Jacobian candidate and unconditional four-layer media measurement."""
import argparse,json,subprocess,sys
from pathlib import Path
from runtime.wan.io import dump
from . import grow_single_step_jacobian_run as generation
from . import grow_temporal_difference_media as media
CASES=generation.CASES
MEDIA_PLAN={k:v*5//3 for k,v in media.PLAN.items()}

def load(p):return json.loads(Path(p).read_text())

def media_case(case,root):
    root=Path(root)
    return media.run_case(case,root/'media'/case,root/'generation',expected_source_run='generation',
        expected_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),expected_manifest=load(generation.MANIFEST),arms=generation.ARMS,messages=generation.MESSAGES,compare_readouts=False)

def budget_summary(cases):
    rows=[]
    for case in CASES:
        videos=cases.get(case,{}).get('videos',{})
        for payload in ('A','B'):
            local=videos.get('LOCAL_'+payload,{});jac=videos.get('JAC_'+payload,{})
            match=next((r['energy_matching'] for r in jac.get('steps',[]) if 'energy_matching' in r),{})
            rows.append(dict(case=case,payload=payload,local_status=local.get('status','MISSING'),jac_status=jac.get('status','MISSING'),
                target_E=local.get('actual_response_energy'),actual_JAC_E=jac.get('actual_response_energy'),
                relative_mismatch=match.get('relative_error'),absolute_mismatch=match.get('absolute_error'),scale=match.get('scale'),
                matching_status=match.get('status','MISSING'),local_peak_control_rms=local.get('peak_control_rms'),jac_peak_control_rms=jac.get('peak_control_rms'),
                local_sum_u_rms_squared=local.get('sum_step_control_rms_squared'),jac_sum_u_rms_squared=jac.get('sum_step_control_rms_squared'),
                local_sum_delta_velocity_rms_squared=local.get('sum_step_delta_velocity_rms_squared'),jac_sum_delta_velocity_rms_squared=jac.get('sum_step_delta_velocity_rms_squared')))
    return rows


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',video_denominator=20,marked_denominator=16,layer_denominator=80,scientific_pass=None,quality_tolerance=None,
        stages={'generation':{'status':'NOT_RUN','case_denominator':4,'video_denominator':20,'cases':{c:generation.empty('NOT_RUN') for c in CASES},'fixed_calls':{k:v*4 for k,v in generation.PLAN.items()}},'media':{'status':'NOT_RUN','cases':{c:media.missing_case('NOT_RUN',arms=generation.ARMS) for c in CASES}}},
        media_policy='attempt all twenty source arms regardless of terminal recovery; no terminal selection')
    def save():dump(output/'result.json',result)
    result['fixed_calls']={k:4*(generation.PLAN.get(k,0)+MEDIA_PLAN.get(k,0)) for k in set(generation.PLAN)|set(MEDIA_PLAN)}
    save();dump(output/'manifest.json',load(generation.MANIFEST))
    try:result['stages']['generation']=generation.run_all(output/'generation')
    except Exception as exc:
        try:result['stages']['generation']=load(output/'generation'/'result.json')
        except Exception:pass
        result['stages']['generation'].update(status='FAILED',error=repr(exc))
    save();(output/'media').mkdir()
    stage=result['stages']['media'];stage['fixed_calls']={k:v*4 for k,v in MEDIA_PLAN.items()}
    for case in CASES:
        log=output/'media'/(case+'.log');print(case,'media started; log:',log,flush=True)
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.grow_single_step_jacobian_full','--output',str(output),'--media-case',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            stage['cases'][case]=load(output/'media'/case/'result.json')|{'exit_code':child.returncode,'log':str(log)}
        except Exception as exc:stage['cases'][case]=media.missing_case('FAILED_LAUNCH_OR_RESULT',arms=generation.ARMS)|{'error':repr(exc),'log':str(log)}
        save()
    stage['status']='EXECUTION_COMPLETE' if all(c['status']=='EXECUTION_COMPLETE' and c.get('exit_code')==0 for c in stage['cases'].values()) else 'WITH_RETAINED_FAILURES'
    stage['actual_calls_observed']={k+'_'+s:sum(c.get('actual_calls',{}).get(k+'_'+s,0) for c in stage['cases'].values()) for k in MEDIA_PLAN for s in ('attempted','completed')}
    stage['call_count_case_coverage']=sum('actual_calls' in c for c in stage['cases'].values())
    result['recovery_summary']={g:media.recovery_summary(stage['cases'],arms=('OFF',g+'_A',g+'_B'),messages=generation.MESSAGES) for g in ('LOCAL','JAC')}
    result['budget_comparison']=budget_summary(result['stages']['generation'].get('cases',{}))
    result['direction_diagnostics']={c:{a:{'status':v.get('status'),'feasibility':v.get('feasibility'),'direction':v.get('direction'),'failure_resources':v.get('last_direction_resources'),'failure_diagnostics':v.get('failure_direction_diagnostics'),'allocated_after_release':v.get('allocated_after_direction_release')} for a,v in row.get('videos',{}).items()} for c,row in result['stages']['generation'].get('cases',{}).items()}
    result['quality_summary']={c:{'OFF_status':row.get('videos',{}).get('OFF',{}).get('status','MISSING'),'arms':{a:v.get('quality',{}) for a,v in row.get('videos',{}).items() if a!='OFF'},'failures':row.get('failures',[])} for c,row in stage['cases'].items()}
    kinds=set(generation.PLAN)|set(MEDIA_PLAN)
    result['fixed_calls']={k:4*(generation.PLAN.get(k,0)+MEDIA_PLAN.get(k,0)) for k in kinds}
    result['actual_calls_observed']={k+'_'+s:sum(x.get('actual_calls_observed',{}).get(k+'_'+s,0) for x in result['stages'].values()) for k in kinds for s in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(x['status']=='EXECUTION_COMPLETE' for x in result['stages'].values()) else 'WITH_RETAINED_FAILURES'
    save();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--media-case',choices=CASES);a=p.parse_args()
    result=media_case(a.media_case,a.output) if a.media_case else run(a.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
