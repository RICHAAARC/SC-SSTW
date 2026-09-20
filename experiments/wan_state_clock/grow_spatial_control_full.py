"""Fixed paired-budget multi-step versus final-step candidate and unconditional four-layer media measurement."""
import argparse,json,subprocess,sys
from pathlib import Path
from runtime.wan.io import dump
from . import grow_spatial_control_run as generation
from . import grow_spatial_media as media
CASES=generation.CASES
MEDIA_PLAN={k:v*5//3 for k,v in media.PLAN.items()}

def load(p):return json.loads(Path(p).read_text())

def media_case(case,root):
    root=Path(root)
    return media.run_case(case,root/'media'/case,root/'generation',expected_source_run='generation',
        expected_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),expected_manifest=load(generation.MANIFEST),arms=generation.ARMS,messages=generation.MESSAGES,compare_readouts=True)

def soft_summary(cases,group):
    result={}
    for layer in media.LAYERS:
        n=exact=errors=erasures=0
        for case in CASES:
            for arm in (group+'_A',group+'_B'):
                row=cases.get(case,{}).get('videos',{}).get(arm,{}).get('layers',{}).get(layer,{})
                if row.get('status')!='COMPLETE' or 'hard_soft_comparisons_reporting_only' not in row:continue
                s=row['hard_soft_comparisons_reporting_only']['soft'];n+=1;exact+=int(s['exact']);errors+=s['bit_errors_including_erasures'];erasures+=s['erasures']
        result[layer]=dict(marked_denominator=8,fixed_bit_denominator=128,completed=n,missing=8-n,exact=exact,errors_observed=errors,erasures_observed=erasures,selected=False)
    return result


def budget_summary(cases):
    rows=[]
    for case in CASES:
        videos=cases.get(case,{}).get('videos',{});net=cases.get(case,{}).get('terminal_net_contrasts',{})
        for payload in ('A','B'):
            multi=videos.get('MULTI_'+payload,{});last=videos.get('LAST_'+payload,{})
            match=next((r['energy_matching'] for r in last.get('steps',[]) if 'energy_matching' in r),{})
            rows.append(dict(case=case,payload=payload,multi_status=multi.get('status','MISSING'),last_status=last.get('status','MISSING'),
                multi_net_terminal_rms=net.get('MULTI_'+payload+'_minus_OFF',{}).get('rms'),last_net_terminal_rms=net.get('LAST_'+payload+'_minus_OFF',{}).get('rms'),
                target_E=multi.get('actual_response_energy'),actual_LAST_E=last.get('actual_response_energy'),
                relative_mismatch=match.get('relative_error'),absolute_mismatch=match.get('absolute_error'),scale=match.get('scale'),
                matching_status=match.get('status','MISSING'),multi_peak_control_rms=multi.get('peak_control_rms'),last_peak_control_rms=last.get('peak_control_rms'),
                multi_sum_response_rms=multi.get('cumulative_rms',{}).get('control_induced_delta_rms'),last_sum_response_rms=last.get('cumulative_rms',{}).get('control_induced_delta_rms'),
                multi_peak_response_rms=multi.get('peak_actual_response_rms'),last_peak_response_rms=last.get('peak_actual_response_rms'),
                multi_peak_dv_rms=multi.get('peak_delta_velocity_rms'),last_peak_dv_rms=last.get('peak_delta_velocity_rms'),
                multi_sum_u_rms=multi.get('cumulative_rms',{}).get('control_rms'),last_sum_u_rms=last.get('cumulative_rms',{}).get('control_rms'),
                multi_sum_u_rms_squared=multi.get('sum_step_control_rms_squared'),last_sum_u_rms_squared=last.get('sum_step_control_rms_squared'),
                multi_sum_delta_velocity_rms_squared=multi.get('sum_step_delta_velocity_rms_squared'),last_sum_delta_velocity_rms_squared=last.get('sum_step_delta_velocity_rms_squared')))
    return rows


def attribution_summary(cases):
    out={}
    for group in ('MULTI','LAST'):
        out[group]={}
        for layer in media.LAYERS:
            out[group][layer]={}
            for reader in ('hard','soft'):
                n=correct=ties=0;gaps=[]
                for case in CASES:
                    for arm in (group+'_A',group+'_B'):
                        row=cases.get(case,{}).get('videos',{}).get(arm,{}).get('layers',{}).get(layer,{})
                        if row.get('status')!='COMPLETE':continue
                        rank=row.get('hard_soft_candidates',{}).get(reader)
                        if rank is None:continue
                        n+=1;truth=generation.MESSAGES[arm];correct+=int(rank['top']==truth);ties+=int(rank['tie'])
                        scores=rank['candidate_scores'];gaps.append(scores[truth]-scores[1-truth])
                out[group][layer][reader]=dict(denominator=8,complete=n,missing=8-n,unique_correct=correct,ties=ties,
                    margin_min=min(gaps) if gaps else None,margin_max=max(gaps) if gaps else None,
                    claim='two-candidate attribution only; not exact payload recovery or calibrated presence/FPR')
    return out


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
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.grow_spatial_control_full','--output',str(output),'--media-case',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            stage['cases'][case]=load(output/'media'/case/'result.json')|{'exit_code':child.returncode,'log':str(log)}
        except Exception as exc:stage['cases'][case]=media.missing_case('FAILED_LAUNCH_OR_RESULT',arms=generation.ARMS)|{'error':repr(exc),'log':str(log)}
        save()
    stage['status']='EXECUTION_COMPLETE' if all(c['status']=='EXECUTION_COMPLETE' and c.get('exit_code')==0 for c in stage['cases'].values()) else 'WITH_RETAINED_FAILURES'
    stage['actual_calls_observed']={k+'_'+s:sum(c.get('actual_calls',{}).get(k+'_'+s,0) for c in stage['cases'].values()) for k in MEDIA_PLAN for s in ('attempted','completed')}
    stage['call_count_case_coverage']=sum('actual_calls' in c for c in stage['cases'].values())
    result['recovery_summary']={g:media.recovery_summary(stage['cases'],arms=('OFF',g+'_A',g+'_B'),messages=generation.MESSAGES) for g in ('MULTI','LAST')}
    result['budget_comparison']=budget_summary(result['stages']['generation'].get('cases',{}))
    result['quality_summary']={c:{a:v.get('quality',{}) for a,v in row.get('videos',{}).items()} for c,row in stage['cases'].items()}
    result['terminal_net_contrasts']={c:row.get('terminal_net_contrasts',{}) for c,row in result['stages']['generation'].get('cases',{}).items()}
    result['candidate_attribution_summary']=attribution_summary(stage['cases'])
    result['reader_factorial_summary']={g:{'hard':result['recovery_summary'][g],'soft':soft_summary(stage['cases'],g)} for g in ('MULTI','LAST')}
    kinds=set(generation.PLAN)|set(MEDIA_PLAN)
    result['fixed_calls']={k:4*(generation.PLAN.get(k,0)+MEDIA_PLAN.get(k,0)) for k in kinds}
    result['actual_calls_observed']={k+'_'+s:sum(x.get('actual_calls_observed',{}).get(k+'_'+s,0) for x in result['stages'].values()) for k in kinds for s in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(x['status']=='EXECUTION_COMPLETE' for x in result['stages'].values()) else 'WITH_RETAINED_FAILURES'
    save();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--media-case',choices=CASES);a=p.parse_args()
    result=media_case(a.media_case,a.output) if a.media_case else run(a.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
