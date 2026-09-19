"""Fixed new twelve-video margin candidate and unconditional four-layer media measurement."""
import argparse,json,subprocess,sys
from pathlib import Path
from runtime.wan.io import dump
from . import grow_temporal_margin_run as generation
from . import grow_temporal_difference_media as media
CASES=generation.CASES

def load(p):return json.loads(Path(p).read_text())

def media_case(case,root):
    root=Path(root)
    return media.run_case(case,root/'media'/case,root/'generation',expected_source_run='generation',
        expected_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),expected_manifest=load(generation.MANIFEST))

def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',video_denominator=12,marked_denominator=8,layer_denominator=48,scientific_pass=None,quality_tolerance=None,
        stages={'generation':{'status':'NOT_RUN','case_denominator':4,'video_denominator':12,'cases':{c:generation.empty('NOT_RUN') for c in CASES},'fixed_calls':{k:v*4 for k,v in generation.PLAN.items()}},'media':{'status':'NOT_RUN','cases':{c:media.missing_case('NOT_RUN') for c in CASES}}},
        media_policy='attempt all twelve source arms regardless of terminal recovery; no terminal selection')
    def save():dump(output/'result.json',result)
    result['fixed_calls']={k:4*(generation.PLAN.get(k,0)+media.PLAN.get(k,0)) for k in set(generation.PLAN)|set(media.PLAN)}
    save();dump(output/'manifest.json',load(generation.MANIFEST))
    try:result['stages']['generation']=generation.run_all(output/'generation')
    except Exception as exc:
        try:result['stages']['generation']=load(output/'generation'/'result.json')
        except Exception:pass
        result['stages']['generation'].update(status='FAILED',error=repr(exc))
    save();(output/'media').mkdir()
    stage=result['stages']['media'];stage['fixed_calls']={k:v*4 for k,v in media.PLAN.items()}
    for case in CASES:
        log=output/'media'/(case+'.log');print(case,'media started; log:',log,flush=True)
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.grow_temporal_margin_full','--output',str(output),'--media-case',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            stage['cases'][case]=load(output/'media'/case/'result.json')|{'exit_code':child.returncode,'log':str(log)}
        except Exception as exc:stage['cases'][case]=media.missing_case('FAILED_LAUNCH_OR_RESULT')|{'error':repr(exc),'log':str(log)}
        save()
    stage['status']='EXECUTION_COMPLETE' if all(c['status']=='EXECUTION_COMPLETE' and c.get('exit_code')==0 for c in stage['cases'].values()) else 'WITH_RETAINED_FAILURES'
    stage['actual_calls_observed']={k+'_'+s:sum(c.get('actual_calls',{}).get(k+'_'+s,0) for c in stage['cases'].values()) for k in media.PLAN for s in ('attempted','completed')}
    stage['call_count_case_coverage']=sum('actual_calls' in c for c in stage['cases'].values())
    result['recovery_summary']=media.recovery_summary(stage['cases'])
    kinds=set(generation.PLAN)|set(media.PLAN)
    result['fixed_calls']={k:4*(generation.PLAN.get(k,0)+media.PLAN.get(k,0)) for k in kinds}
    result['actual_calls_observed']={k+'_'+s:sum(x.get('actual_calls_observed',{}).get(k+'_'+s,0) for x in result['stages'].values()) for k in kinds for s in ('attempted','completed')}
    result['status']='EXECUTION_COMPLETE' if all(x['status']=='EXECUTION_COMPLETE' for x in result['stages'].values()) else 'WITH_RETAINED_FAILURES'
    save();return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--media-case',choices=CASES);a=p.parse_args()
    result=media_case(a.media_case,a.output) if a.media_case else run(a.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
