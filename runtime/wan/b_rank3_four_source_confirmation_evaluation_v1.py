"""Post-only interpretation of the fixed RANK3 Stage 2. No new model calls."""
from collections import Counter
from pathlib import Path
import time
from runtime.wan import b_rank3_four_source_confirmation_v1 as run
from runtime.wan import b_terminal_three_group_evaluation_v1 as old

def assess(decision,bits,positive,frames):
    result=old.assess(decision,bits,positive,frames)
    supplied=decision.get('slots',[])
    for i,row in enumerate(result['slots']):
        raw=supplied[i] if i<len(supplied) else {}
        refs=[old.support(ref,frames,i//8) for ref in raw.get('supporting_refs',[])]
        row['actual_supporting_refs']=refs
        row['actual_supporting_exact']=any(r['actual_correct_group'] and r['exact_claimed_cell'] for r in refs)
        row['actual_supported_correct']=bool(positive and row['emitted'] and row['literal_correct'] and row['actual_supporting_exact'])
    result['counts']['actual_supported_correct']=sum(r['actual_supported_correct'] for r in result['slots'])
    result['counts']['unsupported_positive_outputs']=sum(r['positive'] and r['emitted'] and not r['actual_supporting_exact'] for r in result['slots'])
    return result

def paired(left,right):
    value=old.compare(left,right)
    a,b=left['slots'],right['slots']
    value['added_actual_supported_correct']=[i for i in range(32) if b[i]['actual_supported_correct'] and not a[i]['actual_supported_correct']]
    value['lost_actual_supported_correct']=[i for i in range(32) if a[i]['actual_supported_correct'] and not b[i]['actual_supported_correct']]
    value['right_added_literal_errors']=sum(not b[i]['literal_correct'] for i in value['added_slots'])
    return value


def summarize(tasks,complete):
    totals={};detection={};per_source={};comparisons={}
    for reader in run.READERS:
        for scope in ('positive','null'):
            rows=[t for t in tasks.values() if t['positive']==(scope=='positive')]
            counts=[t['readers'][reader]['counts'] for t in rows]
            totals[reader+'/'+scope]={k:sum(x[k] for x in counts) for k in counts[0]}
            detection[reader+'/'+scope]=dict(planned=len(rows),statuses=dict(Counter(t['readers'][reader]['detection_status'] for t in rows)))
        if reader!='CDC_M':
            comparisons[reader]={task:paired(t['readers']['CDC_M'],t['readers'][reader]) for task,t in tasks.items() if t['positive']}
    for host in run.SOURCES:
        h=host['id'];rows=[t for task,t in tasks.items() if task.startswith(h+'/') and t['positive']]
        per_source[h]=dict(planned=96,positive_tasks=3,
            supported_correct=sum(t['readers']['RANK3']['counts']['actual_supported_correct'] for t in rows),
            detect=sum(t['readers']['RANK3']['detection_status']=='DETECT' for t in rows),
            CDC_M_supported_correct=sum(t['readers']['CDC_M']['counts']['actual_supported_correct'] for t in rows))
    positive,null=totals['RANK3/positive'],totals['RANK3/null']
    null_detect=detection['RANK3/null']['statuses'].get('DETECT',0)
    losses=sum(len(v['lost_actual_supported_correct']) for v in comparisons['RANK3'].values())
    gain=positive['actual_supported_correct']-totals['CDC_M/positive']['actual_supported_correct']
    technical=[task for task,t in tasks.items() if t['readers']['RANK3']['technical_detail_slots'] or t['readers']['RANK3']['detection_status']=='TECHNICAL_UNRESOLVED']
    complete=bool(complete and not technical)
    checks=dict(supported_correct_at_least288=positive['actual_supported_correct']>=288,
        each_source_supported_at_least48=all(v['supported_correct']>=48 for v in per_source.values()),
        positive_detect_at_least9=detection['RANK3/positive']['statuses'].get('DETECT',0)>=9,
        each_source_detect_at_least2=all(v['detect']>=2 for v in per_source.values()),
        CDC_M_supported_losses_zero=losses==0,literal_errors_zero=positive['literal_errors']+null['literal_errors']==0,
        false_attribution_zero=positive['false_attribution']+null['false_attribution']==0,
        outputs_without_actual_support_zero=positive['unsupported_positive_outputs']==0 and null['emitted']==0,
        null_detect_zero=null_detect==0,required_observations_complete=complete)
    stopped=not all(checks[k] for k in ('literal_errors_zero','false_attribution_zero','outputs_without_actual_support_zero','null_detect_zero'))
    met=all(checks.values())
    method='STOP_FIXED_RANK3' if stopped else 'TECHNICAL_UNRESOLVED' if not complete else 'STAGE2_METHOD_CRITERIA_MET' if met else 'STAGE2_METHOD_CRITERIA_NOT_MET'
    contribution='DEMONSTRATED_FIXED_SCOPE' if met and gain>=32 else 'NOT_DEMONSTRATED' if complete else 'TECHNICAL_UNRESOLVED'
    return dict(method_status=method,contribution_status=contribution,checks=checks,totals=totals,detection=detection,
        rank3_per_source=per_source,comparisons_to_CDC_M=comparisons,CDC_M_supported_losses=losses,
        net_actual_supported_gain=gain,incremental_gain_required=32,technical_rank3_tasks=technical,
        next_candidate_allowed=False,method_criteria=run.METHOD_CRITERIA,incremental_claim=run.INCREMENTAL_CLAIM)


def pair_summary(rows):
    paired=[r for r in rows if r['paired']]
    return dict(planned=len(rows),paired=len(paired),missing=len(rows)-len(paired),
        correct_to_error=sum(r['before_error'] is False and r['after_error'] is True for r in paired),
        error_to_correct=sum(r['before_error'] is True and r['after_error'] is False for r in paired),
        retained_error=sum(r['before_error'] is True and r['after_error'] is True for r in paired),
        meaning='true-phase nominal local bit pairs; not full-group recovery or blind receiver success',rows=rows)


def evaluate(out,backend,journal,*,tensor_root=None):
    out=Path(out);tensor_root=Path(tensor_root) if tensor_root is not None else out
    marker=run.read(out/'blind_complete.json')
    if marker.get('status')!='READ_BACK' or marker.get('reader_decisions')!=360:raise ValueError('all 360 blind decisions required before truth')
    run.dump(out/'post_truth_access.json',dict(opened_unix=time.time(),receiver_truth_access='after_all_blind_decisions_read_back',writer_private_payload_separate=True))
    cfg=run.read(out/'protocol.json')['config'];truth=cfg['writer']['bits']
    if len(truth)!=32 or any(type(v) is not int or v not in (0,1) for v in truth):raise ValueError('fixed32 truth required')
    tasks={};native_pairs=[];media_pairs=[]
    for host in cfg['sources']:
        h=host['id']
        for arm in run.ARMS:
            for fixture in run.FIXTURES:
                frames=run.source_frames(host,fixture)
                for key in run.KEYS:
                    task=run.task_id(h,arm,fixture,key);folder=out/'blind'/task
                    plan=run.read(folder/'plan.json');decision=run.read(folder/'decision.json')
                    positive=arm=='PM' and key=='key00';readers={}
                    for name in run.READERS:
                        raw=decision['readers'][name]['aggregate']
                        value=assess(raw,truth,positive,frames)
                        value['detection_status']=decision['readers'][name]['selection']['status']
                        value['technical_detail_slots']=[i for i,row in enumerate(raw['slots']) if row.get('missing_refs') or any(x in row['status'] for x in ('TECHNICAL','UNRESOLVED','UNSUPPORTED','NOT_ENTERED'))]
                        readers[name]=value
                    tasks[task]=dict(positive=positive,readers=readers,
                        local_CDC={w:assess(v,truth,positive,frames) for w,v in decision['local_CDC'].items()},
                        geometry=old.geometry(frames),paths=old.path_assessment(dict(paths={'M3':plan['paths']}),host,fixture)['M3'])
    result=dict(status='POST_PARTIAL',denominators=run.denominators(),tasks=tasks,quality=run.quality_slots(),
        native_audit={},clipping={},media_P0_PM_pair=pair_summary([]),native_P0_PM_pair=pair_summary([]),
        **summarize(tasks,False),quality_pass=None,population_FPR=None,source_authentication=False,bundles=[],
        claim_ceiling=run.CLAIM_CEILING,stop_interpretation='Null DETECT immediately fails this candidate; only fixed saved-data diagnostics continue, no further physical calls or candidate search.')
    run.dump(out/'evaluation.json',result)
    for host in cfg['sources']:
        h=host['id'];native={}
        result['native_audit'][h]=run.read(out/'private'/h/'native/audit.json')
        for arm in run.ARMS:result['clipping'][h+'/'+arm]=run.read(out/'media'/h/arm/'full/preclip.json')
        for arm in ('P0','PM'):
            slot=f'{h}/native/{arm}';dependency=h+'/construct' if arm=='PM' else h+'/generate/P0'
            def project(arm=arm,slot=slot):
                value=backend.payload_project(backend.load_tensor(tensor_root/'private'/h/(arm+'_terminal.pt')),list(range(1,45)),('R0','R1'))
                backend.save_tensor(tensor_root/'private'/h/'native'/arm/'DCT.pt',value);return value
            projected=journal.call(slot+'/DCT',project,[dependency])
            for key,secret in run.KEYS.items():
                native[arm,key]=journal.record(slot+'/'+key+'/bits',lambda key=key,secret=secret:backend.payload_decode(projected,secret,f'native-{h}-{arm}',key),out/'private'/h/'native'/arm/(key+'_bits.json'),[slot+'/DCT'])
        native_pairs.extend(dict(source=h,**v) for v in old.pair_rows(native['P0','key00'],native['PM','key00'],range(1,45),truth,{t:(t-1)//11 for t in range(1,45)}))
        for fixture in run.FIXTURES:
            frames=run.source_frames(host,fixture)
            for window,w in run.WINDOWS.items():
                phase=(-frames[w])%4
                records={arm:run.read(out/'observations'/h/arm/fixture/window/f'p{phase}'/'key00/bits.json') for arm in ('P0','PM')}
                groups={t:((frames[w+phase+4*t-3]+3)//4-1)//11 for t in range(1,11)}
                media_pairs.extend(dict(source=h,fixture=fixture,window=window,true_phase=phase,**v) for v in old.pair_rows(records['P0'],records['PM'],range(1,11),truth,groups))
    result['native_P0_PM_pair']=pair_summary(native_pairs);result['media_P0_PM_pair']=pair_summary(media_pairs)
    run.dump(out/'evaluation.json',result)
    for slot,(h,scope,pair,layer) in run.quality_specs().items():
        host=next(x for x in cfg['sources'] if x['id']==h)
        mapping=list(range(181)) if scope=='full' else run.source_frames(host,scope)
        left,right=pair.split('_to_')
        try:
            result['quality'][slot]=journal.record(slot,lambda:backend.quality(
                backend.load_tensor(tensor_root/'media'/h/left/scope/(layer+'.pt')),
                backend.load_tensor(tensor_root/'media'/h/right/scope/(layer+'.pt')),mapping,pair),out/'quality'/(slot+'.json'))
        except BaseException:
            row=result['quality'][slot];row.update(status=journal.data[slot]['status'],cause=dict(journal.data[slot]))
            for item in row['frames']+row['adjacent']:item['status']=row['status']
            run.dump(out/'quality'/(slot+'.json'),row);run.dump(out/'evaluation.json',result);raise
    # The current post receipt is still STARTED here by design, not missing evidence.
    complete=all(row['status']=='RETURNED' for slot,row in journal.data.items() if slot!='post')
    result.update(summarize(tasks,complete));result['status']='POST_RECORDED'
    return result
