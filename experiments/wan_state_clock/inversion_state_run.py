"""Fixed new state-writing experiment; reuses baseline media and inverse lifetimes."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
import torch
from main.tube_state import inversion_state as method,state_clock,initial_noise
from runtime.wan.io import dump
from . import video_inversion_run as base

CASES,ARMS,PLAN=base.CASES,base.ARMS,base.PLAN
MANIFEST=Path(__file__).parent/'configs/inversion_state.json'
CLAIM='known-time four-phase state observations and fixed-gain observer comparison; no crop/FPR or observer gain claim'


def empty(status):
    result=base.empty_case(status);result['protocol']='inversion_state_v1';result['claim']=CLAIM
    for item in result['videos'].values():
        item['decoded'].update(core_window_denominator=11,
            core=[dict(window=n,start=a,stop=b,status='NOT_RUN') for n,(a,b) in enumerate(method.WINDOWS)],
            boundaries=[dict(time=t,status='NOT_RUN') for t in (0,45)],rankings={})
    return result


def posthoc(output,result,messages):
    book=torch.load(output/'codebook.pt',map_location='cpu',weights_only=True)
    result['claim']=CLAIM
    result['trajectories']={k:book[k].tolist() for k in ('states','steps','drives')}
    dump(output/'state_trajectories.json',result['trajectories'])
    for arm,item in result['videos'].items():
        if item['inversion_status']!='COMPLETE':item['posthoc']={'status':'MISSING_RECOVERED_NOISE'};continue
        truth=None if arm=='OFF' else ARMS.index(arm)-1
        item['posthoc']={'status':'MEASURED',**method.report(item['decoded'],book,truth)}


def run_case(case_id,output):
    manifest=base.load(MANIFEST)
    if manifest['payloads']!=[0,1] or state_clock.GAIN!=.5 or tuple(c['id'] for c in manifest['development'])!=CASES:
        raise ValueError('fixed state protocol mismatch')
    result=base.run_case(case_id,output,manifest_path=MANIFEST,mechanism=method,empty_factory=empty,posthoc_fn=posthoc,claim=CLAIM)
    result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__),Path(base.__file__),Path(method.__file__),Path(state_clock.__file__),Path(initial_noise.__file__),MANIFEST)}
    result['artifact_sha256']={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(output)/'codebook.pt',Path(output)/'config.json',Path(output)/'state_trajectories.json') if p.exists()}
    dump(Path(output)/'result.json',result);return result


def summarize(cases):
    summary=dict(marked_video_denominator=8,off_video_denominator=4,core_window_denominator=88,
        component_denominator=176,complete_marked_videos=0,exact_raw_windows=0,raw_component_errors=0,
        exact_raw_trajectories=0,raw_erasures=0,unique_correct={m:0 for m in ('without_update','with_update')},
        per_video={},scientific_pass=None)
    for case in CASES:
        for arm in ARMS:
            item=cases.get(case,{}).get('videos',{}).get(arm,{})
            post=item.get('posthoc',{})
            key=case+'/'+arm
            summary['per_video'][key]=dict(status=item.get('status','MISSING'),posthoc=post)
            if arm=='OFF' or item.get('status')!='COMPLETE' or post.get('status')!='MEASURED':continue
            summary['complete_marked_videos']+=1
            raw=post['raw_state'];summary['exact_raw_windows']+=raw['exact_windows']
            summary['raw_component_errors']+=raw['component_errors'];summary['raw_erasures']+=raw['erasures']
            summary['exact_raw_trajectories']+=raw['exact_trajectory']
            for mode in summary['unique_correct']:summary['unique_correct'][mode]+=post['modes'][mode]['unique_correct']
    summary['unmeasured_marked_videos']=8-summary['complete_marked_videos']
    summary['unmeasured_core_windows']=11*summary['unmeasured_marked_videos']
    summary['unmeasured_components']=22*summary['unmeasured_marked_videos']
    summary['measured_component_denominator']=22*summary['complete_marked_videos']
    summary['raw_component_error_scope']='observed complete marked videos only; missing components separately retained'
    summary['component_meaning']='two signs per four-phase state, not independent payload bits; two message candidates'
    return summary


def run_all(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=dict(status='RUNNING',case_denominator=4,video_denominator=12,core_window_denominator=132,
        cases={c:empty('NOT_RUN') for c in CASES},quality_tolerance=None,scientific_pass=None,claim=CLAIM,
        fixed_calls={s:{k:4*v for k,v in counts.items()} for s,counts in PLAN.items()})
    dump(output/'manifest.json',base.load(MANIFEST));dump(output/'result.json',result)
    for case in CASES:
        log=output/(case+'.log');print(case,'started; child log:',log,flush=True)
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.inversion_state_run',
                    '--output',str(output/case),'--case-id',case],stdout=stream,stderr=subprocess.STDOUT,check=False)
            result['cases'][case]=base.load(output/case/'result.json')|{'exit_code':child.returncode}
        except Exception as exc:result['cases'][case]=empty('FAILED_LAUNCH_OR_RESULT')|{'error':repr(exc)}
        result['cases'][case]['log']=str(log);dump(output/'result.json',result)
        print(case,result['cases'][case]['status'],'child log:',log,flush=True)
    result['actual_calls_observed']={s:{k+'_'+d:sum(v.get('actual_calls',{}).get(s,{}).get(k+'_'+d,0) for v in result['cases'].values())
        for k in counts for d in ('attempted','completed')} for s,counts in PLAN.items()}
    result['call_count_case_coverage']=sum('actual_calls' in v for v in result['cases'].values())
    result['status']='EXECUTION_COMPLETE' if all(v['status']=='EXECUTION_COMPLETE' and v.get('exit_code')==0 for v in result['cases'].values()) else 'WITH_RETAINED_FAILURES'
    result['state_summary']=summarize(result['cases'])
    dump(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--case-id',choices=CASES)
    args=parser.parse_args();result=run_case(args.case_id,args.output) if args.case_id else run_all(args.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
