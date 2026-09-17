"""Isolated saved-prefix objective comparison; no prefix/baseline/holdout rerun."""
from __future__ import annotations
import argparse
import copy
import subprocess
import sys
import traceback
from pathlib import Path
import torch
from runtime.wan.io import dump
from . import zero_path_run as saved
from .calibration_selection import select_terminal,select_media
from .calibration_run import read_cases
from .velocity_direction_run import endpoint_metrics

ZERO_RUN='velocity_zero_path_20260917T092208999384Z'
STRENGTHS=(.1,.3,1.)


def inputs(source,zero,output):
    source,output=saved.check_paths(source,output)
    zero=Path(zero).resolve()
    if zero.name!=ZERO_RUN:raise ValueError('fixed verified Z0 run required')
    if output==zero or zero in output.parents or output in zero.parents:
        raise ValueError('new output must be separate from original Z0')
    return source,zero,output


def effective_config(original,source,output,case_id,stage):
    config=copy.deepcopy(original)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    config['source_snapshot']=f'{sha}; dirty={dirty}; clipped-margin saved-prefix candidate'
    config['output_drive_parent']=str(output.resolve())
    config['artifact_paths']={'source_directory':str(Path.cwd()),
        'launcher_log':str(output/(stage+'_'+case_id+'.log')),
        'original_run':str(source),'original_config':str(source/'dev'/case_id/'terminal/config.json'),
        'result_directory':str(output/'dev'/case_id/stage)}
    return config


def terminal_case(source,zero,output,case_id):
    source,zero,output=inputs(source,zero,output)
    directory=output/'dev'/case_id/'terminal'
    try:
        original,config,record,payload,book,environment=saved.preflight(source,case_id)
        from main.tube_state import state_clock
        import numpy as np
        expected=state_clock.codebook(config['key_utf8'].encode())
        if set(book)!=set(expected) or any(not np.array_equal(book[k],expected[k]) for k in expected):
            raise ValueError('saved codebook differs from original public key/codebook')
        from .velocity_direction_run import run
        current=effective_config(config,source,output,case_id,'terminal')
        result=run(current,directory,strengths=STRENGTHS,_restored={
            'record':record,'payload':payload,'book':book,'original_run':str(source)})
        dump(directory/'original_config.json',config)
        return result
    except Exception as exc:
        if directory.exists():raise
        directory.mkdir(parents=True,exist_ok=False)
        result={'status':'FAILED_RESTORATION','failures':[{'error':repr(exc),'traceback':traceback.format_exc()}],
            'conditions':{n:{'status':'MISSING_RESTORATION','steps':[]} for n in ('ZERO_A','ZERO_B',*saved.FLOW_NAMES)}}
        dump(directory/'result.json',result);return result


def nominal(detection,message):
    # Reporting only, AFTER the complete unchanged blind search.
    path=next(p for p in detection['candidates'] if p['g']==0 and p['scale']==[1,1]
              and p['offset']==0 and p['delta']==0)
    item=detection['classes'][str(path['class'])]
    correct,wrong=item['scores'][message],item['scores'][1-message]
    return {'path':path,'matched_supports':correct['matched_supports'],
        'correct_matched_score':correct['matched_score'],'wrong_matched_score':wrong['matched_score'],
        'matched_margin':correct['matched_score']-wrong['matched_score'],
        'innovation_correct':correct['observer']['innovation_mean'],
        'innovation_wrong':wrong['observer']['innovation_mean'],
        'innovation_margin_penalty':.05*(correct['observer']['innovation_mean']-wrong['observer']['innovation_mean']),
        'state_margin':correct['state_score']-wrong['state_score'],
        'claim':'oracle location diagnostic only; never passed to reader'}


def media_case(source,zero,output,case_id):
    source,zero,output=inputs(source,zero,output)
    from .calibration_media import run
    original=source/'dev'/case_id
    config=saved.load(original/'terminal/config.json')
    selected=saved.load(output/'terminal_selection.json')['selected_indices']
    directory=output/'dev'/case_id
    terminals={f'FLOW_r{i:02d}_{m}':directory/'terminal'/f'FLOW_r{i:02d}_{m}_terminal.pt'
               for i in selected for m in ('A','B')}
    current=effective_config(config,source,output,case_id,'media')
    result=run(current,terminals,directory/'media',off_reference=original/'media/received_videos/OFF.mp4')
    dump(directory/'media/original_config.json',config)
    return result


def child_all(source,zero,output,stage):
    report={'case_denominator':4,'cases':{c:{'status':'NOT_RUN'} for c in saved.CASES}}
    for case in saved.CASES:
        log=output/(stage+'_'+case+'.log')
        try:
            with log.open('w') as stream:
                child=subprocess.run([sys.executable,'-u','-m','experiments.wan_state_clock.clipped_margin_run','--input-run',str(source),
                    '--zero-run',str(zero),'--output',str(output),'--stage',stage+'-case','--case-id',case],
                    stdout=stream,stderr=subprocess.STDOUT,check=False)
            report['cases'][case]={'status':'COMPLETE' if child.returncode==0 else 'WITH_RETAINED_FAILURES',
                'exit_code':child.returncode,'log':str(log)}
        except Exception as exc:report['cases'][case]={'status':'FAILED_LAUNCH','error':repr(exc)}
        dump(output/(stage+'.json'),report)
    return report


def file_result(path,compute):
    try:return {'status':'MEASURED','path':str(path),'value':compute(path)}
    except Exception as exc:return {'status':'MISSING_OR_FAILED','path':str(path),'error':repr(exc)}


def report(source,zero,output):
    source,zero,output=inputs(source,zero,output)
    report={'case_denominator':4,'flow_denominator_per_method':24,'cases':{},
        'original_selection':saved.optional_json(source/'selection.json'),
        'original_terminal_selection':saved.optional_json(source/'terminal_selection.json'),
        'Z0_run_result':saved.optional_json(zero/'result.json'),
        'holdout':'NOT_RUN_NOT_AUTHORIZED','original_data_modified':False,
        'claim':'paired objective diagnostics; unchanged selector/reader; missing original media rows remain missing'}
    for case in saved.CASES:
        old=source/'dev'/case;new=output/'dev'/case
        item={'baseline_comparisons':{},'flows':{n:{label:{'terminal':{'status':'MISSING_OR_FAILED'},'media':{'status':'MISSING_OR_FAILED'}} for label in ('old_hinge','new_clipped')} for n in saved.FLOW_NAMES},'old_controls':saved.optional_json(old/'media/result.json')}
        report['cases'][case]=item
        try:
            import numpy as np
            with np.load(old/'terminal/codebook.npz',allow_pickle=False) as archive:
                directions=torch.from_numpy(archive['directions'].copy());codes=torch.from_numpy(archive['codes'].copy())
            def tensor(path):
                value=torch.load(path,map_location='cpu',weights_only=True).float()
                if tuple(value.shape)!=saved.carrier.SHAPE or not torch.isfinite(value).all():raise ValueError('invalid terminal')
                return value
            def metrics(path):return endpoint_metrics(tensor(path),directions,codes)
            def difference(path,reference):
                a,b=tensor(path),tensor(reference)
                return {'tensor_difference_rms':saved.measures(a-b),'bitwise_equal':torch.equal(a,b),
                    'terminal_baseline_equal':torch.equal(a,b),
                    'note':'nonidentical zeros: reused controls are historical references, not demonstrated matched controls'}
            for suffix in ('A','B'):
                p=new/'terminal'/('ZERO_'+suffix+'_terminal.pt')
                for label,ref in [('old_OFF',old/'terminal/ZERO_A_terminal.pt'),('verified_Z0',zero/case/'Z0_terminal.pt')]:
                    item['baseline_comparisons'][suffix+'_vs_'+label]=file_result(p,lambda path,ref=ref:difference(path,ref))
            for name in saved.FLOW_NAMES:
                message=0 if name.endswith('A') else 1
                row={};item['flows'][name]=row
                for label,directory in [('old_hinge',old),('new_clipped',new)]:
                    row[label]={'terminal':file_result(directory/'terminal'/(name+'_terminal.pt'),metrics),
                        'media':file_result(directory/'media/detections'/(name+'.json'),
                            lambda path,m=message:{'nominal':nominal(saved.load(path),m),
                                'blind':saved.load(path)['rankings']})}
                row['message']=message
                for label,directory in [('old_hinge',old),('new_clipped',new)]:
                    if row[label]['media']['status']=='MISSING_OR_FAILED' and not (directory/'media/detections'/(name+'.json')).exists():
                        selection_path=(source if label=='old_hinge' else output)/'terminal_selection.json'
                        selection=saved.optional_json(selection_path)
                        chosen=selection.get('value',{}).get('selected_indices')
                        index=int(name.split('_')[1][1:])
                        if chosen is not None and index not in chosen:
                            row[label]['media']['status']='NOT_RUN_ORIGINAL' if label=='old_hinge' else 'NOT_SELECTED_FOR_MEDIA'
                    terminal=row[label]['terminal']
                    if terminal['status']=='MEASURED':
                        scores=terminal['value']['clipped_matched_score_by_message']
                        terminal['correct_minus_wrong_clipped_margin']=scores[message]-scores[1-message]
            item['old_terminal_record']=saved.optional_json(old/'terminal/result.json')
            item['new_terminal_record']=saved.optional_json(new/'terminal/result.json')
        except Exception as exc:item['status']='MISSING_OR_FAILED';item['error']=repr(exc)
    comparability={}
    for case,item in report['cases'].items():
        checks=item['baseline_comparisons']
        values=[checks.get(suffix+'_vs_'+label,{}) for suffix in ('A','B') for label in ('old_OFF','verified_Z0')]
        if any(v.get('status')=='MEASURED' and v['value']['terminal_baseline_equal'] is False for v in values):
            status='NONMATCHING'
        elif all(v.get('status')=='MEASURED' and v['value']['terminal_baseline_equal'] is True for v in values):
            status='MATCHED'
        else:status='UNVERIFIED'
        comparability[case]=status
    statuses=list(comparability.values())
    report['historical_control_comparability']={'status':'NONMATCHING' if 'NONMATCHING' in statuses else
        ('MATCHED' if all(v=='MATCHED' for v in statuses) else 'UNVERIFIED'),'cases':comparability,
        'basis':'exact saved zero terminal equality only; original embedding/model identity not established'}
    report['paired_method_comparison_valid']=all(v=='MATCHED' for v in statuses)
    report['new_selection']=saved.optional_json(output/'selection.json')
    dump(output/'comparison.json',report);return report


def run_stage(source,zero,output,stage):
    source,zero,output=inputs(source,zero,output)
    if stage=='develop':
        output.mkdir(parents=True,exist_ok=False)
        dump(output/'manifest.json',saved.load(source/'manifest.json'))
        result=child_all(source,zero,output,'terminal')
    elif stage=='media':
        manifest=saved.load(output/'manifest.json')
        terminal=select_terminal(manifest,read_cases(output,manifest,'dev'))
        dump(output/'terminal_selection.json',terminal)
        if terminal['selected_indices']:child_all(source,zero,output,'media')
        media=read_cases(output,manifest,'dev',True)
        for case in saved.CASES:
            # Controls are preserved old records, never synthesized into new evidence.
            try:
                controls=saved.load(source/'dev'/case/'media/result.json')['videos']
                if media[case] is not None:
                    media[case]=copy.deepcopy(media[case])
                    media[case]['videos'].update({k:controls[k] for k in ('OFF','TERMINAL_A','TERMINAL_B') if k in controls})
            except (OSError,ValueError,KeyError):pass
        result=select_media(manifest,terminal,media)
        comparison=report(source,zero,output)
        raw=copy.deepcopy(result)
        result['historical_control_comparability']=comparison['historical_control_comparability']
        result['paired_method_comparison_valid']=comparison['paired_method_comparison_valid']
        result['raw_selector_result']=raw
        result['control_provenance']='historical original controls; terminal equality is not model/embedding identity proof'
        if not result['paired_method_comparison_valid']:
            result['status']='PROVISIONAL_'+raw['status']+'_HISTORICAL_CONTROLS'
        dump(output/'selection.json',result)
        comparison['new_selection']=saved.optional_json(output/'selection.json')
        dump(output/'comparison.json',comparison)
    elif stage=='report':result={}
    else:raise ValueError('no automatic holdout or baseline stages')
    if stage!='media':report(source,zero,output)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-run',type=Path,required=True)
    p.add_argument('--zero-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stage',choices=('develop','media','report','terminal-case','media-case'),required=True)
    p.add_argument('--case-id',choices=saved.CASES)
    args=p.parse_args()
    if args.stage.endswith('-case'):
        if args.case_id is None:p.error('--case-id required')
        fn=terminal_case if args.stage=='terminal-case' else media_case
        result=fn(args.input_run,args.zero_run,args.output,args.case_id)
        if result['failures'] or result['status'] not in ('COMPLETE','EXECUTED_REQUIRES_DIRECTION_REVIEW'):raise SystemExit(1)
    else:run_stage(args.input_run,args.zero_run,args.output,args.stage)


if __name__=='__main__':main()
