"""Fixed-roster staged calibration. Models live only in per-case child processes."""
from __future__ import annotations
import argparse
import copy
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path
from runtime.wan.io import dump
from .calibration_selection import flow_name,select_terminal,select_media,evaluate_holdout_case


def load(path):return json.loads(Path(path).read_text())


def validate(manifest):
    if manifest['protocol']!='velocity_strength_calibration_v1':raise ValueError('unknown calibration protocol')
    strengths=manifest['strengths']
    if not strengths or len(set(strengths))!=len(strengths) or any(not 0<float(rho)<=1 for rho in strengths):raise ValueError('fixed rho list required')
    if manifest['max_media_candidates']!=2 or manifest['selection']['quality_relative_limit']!=1.5:raise ValueError('selection rule differs from protocol')
    dev,hold=manifest['development'],manifest['holdout']
    if not dev or not hold:raise ValueError('both rosters required')
    ids=[c['id'] for c in dev+hold]
    if len(set(ids))!=len(ids) or any(not re.fullmatch(r'[a-zA-Z0-9_-]+',name) for name in ids):raise ValueError('unique safe case IDs required')
    if len({(c['prompt'],c['seed']) for c in dev+hold})!=len(ids):raise ValueError('duplicate content/seed case')
    if set(c['prompt'] for c in dev)&set(c['prompt'] for c in hold) or set(c['seed'] for c in dev)&set(c['seed'] for c in hold):raise ValueError('holdout prompts and seeds must be independent')
    generation=manifest['base_config']['generation']
    if (generation['steps'],generation['frames'],generation['height'],generation['width'],generation['fps'])!=(50,181,320,512,8):raise ValueError('fixed generation geometry/schedule required')


def call_plan(manifest):
    n=len(manifest['development']);h=len(manifest['holdout']);k=len(manifest['strengths'])
    return {'development':{'cases':n,'terminal_strength_rows':n*k*2,'transformer':n*(112+24*k),
        'backward':2*n,'outer_replay_up_to':20*n,'scheduler_step':n*(56+12*k),'shadow_step':n*(12+12*k),
        'response_probe_step':6*n,'media_max_decode_and_save_each':n*7,'media_max_encode':n*28},
        'holdout_if_selected':{'cases':h,'strength_rows':2*h,'transformer':136*h,'backward':2*h,
        'outer_replay_up_to':20*h,'media_decode_and_save_each':5*h,'media_encode':20*h},
        'block_units':'each of three block categories is up to outer_replay_up_to times actual block count, separately counted',
        'media_formula':'per development case: 3+2*C decodes/saves and 4*(3+2*C) encodes for C selected candidates; none if C=0'}


def config_for(manifest,case):
    config=copy.deepcopy(manifest['base_config'])
    config['generation'].update(prompt=case['prompt'],seed=case['seed'],role='fixed_roster_zero_gradient_then_predetermined_positive_strengths')
    config['calibration_case']=case
    config.setdefault('source_snapshot','local calibration source; actual git HEAD/dirty recorded by runner')
    return config


def case_root(root,split,case_id):return Path(root)/split/case_id


def read_cases(root,manifest,split,media=False):
    values={}
    for case in manifest['development' if split=='dev' else 'holdout']:
        path=case_root(root,split,case['id'])/('media' if media else 'terminal')/'result.json'
        try:values[case['id']]=load(path)
        except (OSError,ValueError):values[case['id']]=None
    return values


def strengths_for(manifest,root,split):
    if split=='dev':return manifest['strengths']
    selection=load(Path(root)/'selection.json')
    if selection['status']!='SELECTED':raise ValueError('holdout requires frozen development selection')
    return [selection['rho']]


def child(manifest,root,split,case_id,media):
    roster=manifest['development' if split=='dev' else 'holdout']
    case=next(c for c in roster if c['id']==case_id)
    directory=case_root(root,split,case_id);config=config_for(manifest,case)
    config['output_drive_parent']=str(directory.parent)
    config['generation']['role']='calibration_'+split+('_saved_terminal_media' if media else '_zero_gradient_fixed_strengths')
    config['artifact_paths']=dict(config.get('artifact_paths',{}),source_directory=str(Path.cwd()),launcher_log=str(Path(root)/'logs'/f'{split}_{case_id}_{"media" if media else "terminal"}.log'))
    if not media:
        from .velocity_direction_run import run
        return run(config,directory/'terminal',strengths=strengths_for(manifest,root,split))
    from .calibration_media import run
    terminal=directory/'terminal'
    indices=load(Path(root)/'terminal_selection.json')['selected_indices'] if split=='dev' else [0]
    terminals={'OFF':terminal/'ZERO_A_terminal.pt','TERMINAL_A':terminal/'terminal_reference_0.pt',
        'TERMINAL_B':terminal/'terminal_reference_1.pt'}
    for index in indices:
        for suffix in ('A','B'):terminals[flow_name(index,suffix)]=terminal/(flow_name(index,suffix)+'_terminal.pt')
    return run(config,terminals,directory/'media')


def run_children(manifest,manifest_path,root,split,media,invoke=subprocess.run):
    roster=manifest['development' if split=='dev' else 'holdout']
    stage=('media' if media else 'terminal')
    report={'status':'RUNNING','split':split,'stage':stage,'case_denominator':len(roster),
        'cases':{c['id']:{'status':'NOT_RUN'} for c in roster},'failures':[]}
    report_path=Path(root)/f'{split}_{stage}.json';dump(report_path,report)
    for case in roster:
        row=report['cases'][case['id']];row['status']='RUNNING';dump(report_path,report)
        log=Path(root)/'logs'/f'{split}_{case["id"]}_{stage}.log';log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u','-m','experiments.wan_state_clock.calibration_run',
            '--manifest',str(manifest_path),'--output',str(root),'--stage','media-case' if media else 'case',
            '--split',split,'--case-id',case['id']]
        print(split,case['id'],stage,'started; log:',log,flush=True)
        try:
            # One child per case releases weights and autograd graphs on exit.
            with log.open('w') as stream:
                completed=invoke(command,stdout=stream,stderr=subprocess.STDOUT,check=False)
            row.update(status='COMPLETE' if completed.returncode==0 else 'WITH_RETAINED_FAILURES',exit_code=completed.returncode,log=str(log))
        except Exception as exc:
            row.update(status='FAILED_LAUNCH',error=repr(exc))
            report['failures'].append({'case':case['id'],'traceback':traceback.format_exc()})
        dump(report_path,report)
        print(split,case['id'],stage,row['status'],flush=True)
    report['status']='COMPLETE' if all(row['status']=='COMPLETE' for row in report['cases'].values()) else 'WITH_RETAINED_FAILURES'
    dump(report_path,report);return report


def run_stage(manifest,manifest_path,root,stage):
    if stage=='develop':return run_children(manifest,manifest_path,root,'dev',False)
    if stage=='development-media':
        selected=select_terminal(manifest,read_cases(root,manifest,'dev'))
        dump(Path(root)/'terminal_selection.json',selected)
        if selected['selected_indices']:
            run_children(manifest,manifest_path,root,'dev',True)
        else:
            dump(Path(root)/'dev_media.json',{'status':'NOT_RUN_NO_SELECTION','case_denominator':len(manifest['development']),
                'cases':{c['id']:{'status':'NOT_RUN_NO_SELECTION'} for c in manifest['development']}})
        selection=select_media(manifest,selected,read_cases(root,manifest,'dev',True))
        dump(Path(root)/'selection.json',selection);return selection
    if stage=='holdout':
        selection=load(Path(root)/'selection.json')
        if selection['status']!='SELECTED':
            result={'status':'NOT_RUN_NO_SELECTION','rho':None,'case_denominator':len(manifest['holdout']),
                'message_denominator':2*len(manifest['holdout']),
                'cases':{c['id']:{'status':'NOT_RUN_NO_SELECTION'} for c in manifest['holdout']}}
            dump(Path(root)/'holdout.json',result);return result
        # Selection is written BEFORE looking at or generating holdout evidence.
        run_children(manifest,manifest_path,root,'holdout',False)
        run_children(manifest,manifest_path,root,'holdout',True)
        media_results=read_cases(root,manifest,'holdout',True);terminal_results=read_cases(root,manifest,'holdout')
        cases={}
        for case in manifest['holdout']:
            name=case['id'];media=media_results[name];terminal=terminal_results[name]
            cases[name]={'status':'COMPLETE' if media and media.get('status')=='COMPLETE' and terminal and terminal.get('status')=='EXECUTED_REQUIRES_DIRECTION_REVIEW' else 'WITH_RETAINED_FAILURES',
                'terminal_result':str(case_root(root,'holdout',name)/'terminal/result.json'),
                'media_result':str(case_root(root,'holdout',name)/'media/result.json'),
                'evaluation':evaluate_holdout_case(terminal,media,name)}
        result={'status':'EVALUATED_REQUIRES_REVIEW' if all(value.get('status')=='COMPLETE' for value in cases.values()) else 'WITH_RETAINED_FAILURES',
            'rho':selection['rho'],'case_denominator':len(manifest['holdout']),
            'cases':cases,'message_denominator':2*len(manifest['holdout']),'selection_updated':False,
            'claim':'independent fixed-parameter observations; no tuning, automatic success or FPR claim'}
        dump(Path(root)/'holdout.json',result);return result
    raise ValueError('unknown stage')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('all','develop','development-media','holdout','case','media-case'),default='all')
    parser.add_argument('--split',choices=('dev','holdout'));parser.add_argument('--case-id')
    args=parser.parse_args();manifest=load(args.manifest);validate(manifest)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    frozen=root/'manifest.json'
    if frozen.exists() and load(frozen)!=manifest:raise ValueError('output roster already frozen to another manifest')
    if not frozen.exists():dump(frozen,manifest)
    dump(root/'call_plan.json',call_plan(manifest))
    if args.stage in ('case','media-case'):
        result=child(manifest,root,args.split,args.case_id,args.stage=='media-case')
        if result['failures'] or result['status'] not in ('COMPLETE','EXECUTED_REQUIRES_DIRECTION_REVIEW'):raise SystemExit(1)
    else:
        for stage in ('develop','development-media','holdout') if args.stage=='all' else (args.stage,):
            result=run_stage(manifest,frozen,root,stage)
            print(stage,result['status'],flush=True)


if __name__=='__main__':main()
