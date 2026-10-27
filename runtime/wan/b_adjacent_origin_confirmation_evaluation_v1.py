"""M3 post-only fixed denominators, actual supporting refs and paired claims."""
from collections import Counter
from pathlib import Path
import time
from runtime.wan import b_adjacent_origin_confirmation_v1 as run
from runtime.wan import b_terminal_three_group_evaluation_v1 as old
from runtime.wan.b_rank3r_complete_repeat_evaluation_v1 import assess
from runtime.wan.b_rank3_four_source_confirmation_evaluation_v1 import paired,pair_summary

def path_truth(decision,frames):
    origins={w:frames[start] for w,start in run.WINDOWS.items()};rows={}
    for reader in ('ADJ','RANK3R'):
        selection=decision['readers'][reader]['selection'];selected=selection.get('selected',{})
        rows[reader]=dict(actual_window_origins=origins,selected_window_origins={w:v.get('origin') for w,v in selected.items()},
            windows_exact=bool(selection.get('detected') and all(selected.get(w,{}).get('origin')==o for w,o in origins.items())),
            exact_seam_certified=False)
    return rows

def summarize(tasks,complete):
    totals={};detection={};comparisons={};per_source={};subsets={};paired_summary={}
    for reader in run.READERS:
        for scope in ('positive','null','wrong_key_PM'):
            rows=[t for name,t in tasks.items() if (t['positive'] if scope=='positive' else not t['positive'] if scope=='null' else name.split('/')[1]=='PM' and name.endswith('/key01'))]
            counts=[t['readers'][reader]['counts'] for t in rows]
            totals[reader+'/'+scope]={k:sum(v[k] for v in counts) for k in counts[0]}
            detection[reader+'/'+scope]=dict(planned=len(rows),statuses=dict(Counter(t['readers'][reader]['detection_status'] for t in rows)))
    for control in ('CDC_M','RANK3R'):
        comparisons[control]={name:paired(t['readers'][control],t['readers']['ADJ']) for name,t in tasks.items() if t['positive']}
        paired_summary[control]={}
        for scope in ('overall','repeat','crop_delete'):
            pairs=[v for name,v in comparisons[control].items() if scope=='overall' or (name.split('/')[2].startswith('REPEAT') if scope=='repeat' else not name.split('/')[2].startswith('REPEAT'))]
            common=sum(p['common_count'] for p in pairs);left=sum(p['left_common_errors'] for p in pairs);right=sum(p['right_common_errors'] for p in pairs)
            added=sum(len(p['added_actual_supported_correct']) for p in pairs);lost=sum(len(p['lost_actual_supported_correct']) for p in pairs)
            paired_summary[control][scope]=dict(added_actual_supported_correct=added,lost_actual_supported_correct=lost,net_gain=added-lost,
                common_emitted_slots=common,control_common_errors=left,main_common_errors=right,
                control_common_BER=left/common if common else None,main_common_BER=right/common if common else None,
                common_BER_defined=bool(common),zero_loss_per_task_slot=lost==0)
    for scope in ('overall','repeat','crop_delete'):
        rows={name:t for name,t in tasks.items() if t['positive'] and (scope=='overall' or (name.split('/')[2].startswith('REPEAT') if scope=='repeat' else not name.split('/')[2].startswith('REPEAT')))}
        subsets[scope]=dict(planned_tasks=len(rows),planned_slots=len(rows)*32,readers={r:dict(
            detected=sum(t['readers'][r]['detection_status']=='DETECT' for t in rows.values()),
            actual_supported_correct=sum(t['readers'][r]['counts']['actual_supported_correct'] for t in rows.values())) for r in run.READERS})
        for host in run.SOURCES:
            selected=[t for name,t in rows.items() if name.startswith(host['id']+'/')]
            per_source.setdefault(host['id'],{})[scope]=dict(planned_tasks=len(selected),planned_slots=len(selected)*32,
                detected=sum(t['readers']['ADJ']['detection_status']=='DETECT' for t in selected),
                actual_supported_correct=sum(t['readers']['ADJ']['counts']['actual_supported_correct'] for t in selected),
                readers={r:dict(detected=sum(t['readers'][r]['detection_status']=='DETECT' for t in selected),actual_supported_correct=sum(t['readers'][r]['counts']['actual_supported_correct'] for t in selected)) for r in run.READERS},
                paired={control:dict(added=sum(len(v['added_actual_supported_correct']) for name,v in comparisons[control].items() if name in rows and name.startswith(host['id']+'/')),
                    lost=sum(len(v['lost_actual_supported_correct']) for name,v in comparisons[control].items() if name in rows and name.startswith(host['id']+'/')),
                    net=sum(len(v['added_actual_supported_correct'])-len(v['lost_actual_supported_correct']) for name,v in comparisons[control].items() if name in rows and name.startswith(host['id']+'/'))) for control in ('CDC_M','RANK3R')})
    technical=[name for name,t in tasks.items() if any(v['technical_detail_slots'] or v['detection_status']=='TECHNICAL_UNRESOLVED' for v in t['readers'].values())]
    complete=bool(complete and not technical);checks={}
    for scope,rules in ((k,run.METHOD_CRITERIA[k]) for k in ('overall','repeat','crop_delete')):
        value=subsets[scope]['readers']['ADJ'];total_min=rules.get('positive_detect_min',rules.get('implied_total_detect_min'))
        checks[scope+'_detect']=value['detected']>=total_min
        checks[scope+'_support']=value['actual_supported_correct']>=rules['supported_correct_min']
        checks[scope+'_each_source_detect']=all(v[scope]['detected']>=rules['positive_detect_min_per_source'] for v in per_source.values())
        checks[scope+'_each_source_support']=all(v[scope]['actual_supported_correct']>=rules['supported_correct_min_per_source'] for v in per_source.values())
    positive,null=totals['ADJ/positive'],totals['ADJ/null']
    checks.update(null_detect_zero=detection['ADJ/null']['statuses'].get('DETECT',0)==0,null_output_zero=null['emitted']==0,
        literal_errors_zero=positive['literal_errors']+null['literal_errors']==0,
        false_attribution_zero=positive['false_attribution']+null['false_attribution']==0,
        unsupported_outputs_zero=positive['unsupported_positive_outputs']==0 and null['emitted']==0,
        loss_vs_CDC_M_zero=paired_summary['CDC_M']['overall']['lost_actual_supported_correct']==0,
        loss_vs_RANK3R_zero=paired_summary['RANK3R']['overall']['lost_actual_supported_correct']==0,required_observations_complete=complete)
    violations=[k for k in ('null_detect_zero','null_output_zero','literal_errors_zero','false_attribution_zero','unsupported_outputs_zero','loss_vs_CDC_M_zero','loss_vs_RANK3R_zero') if not checks[k]]
    met=all(checks.values());gain=paired_summary['CDC_M']['overall']['net_gain'];threshold=run.CONTRIBUTION_CRITERIA['net_actual_supported_correct_gain_over_CDC_M_min']
    method='TECHNICAL_UNRESOLVED' if not complete else 'ADJ_SYSTEM_METHOD_CRITERIA_MET' if met else 'METHOD_NOT_SUPPORTED'
    contribution='TECHNICAL_UNRESOLVED' if not complete else 'ADJ_SYSTEM_CONTRIBUTION_CRITERIA_MET' if met and gain>=threshold else 'CONTRIBUTION_NOT_MET'
    correction_tasks={name:t.get('correction',{}) for name,t in tasks.items() if t.get('correction',{}).get('selected_window') is not None}
    adj_pair=paired_summary['RANK3R']['overall']
    adj_status=('TECHNICAL_UNRESOLVED' if not complete else
        'ADJ_NOT_SUPPORTED' if not met else
        'THIS_BATCH_HAS_NO_NEW_CORRECTION_EVIDENCE' if adj_pair['added_actual_supported_correct']==0 else
        'NEW_CORRECTION_EVIDENCE_IN_FIXED_SOURCES')
    incremental=dict(status=adj_status,corrected_tasks=correction_tasks,
        correction_entered_tasks=sum(bool(t.get('correction',{}).get('entered')) for t in tasks.values()),
        added_actual_supported_correct=adj_pair['added_actual_supported_correct'],
        lost_actual_supported_correct=adj_pair['lost_actual_supported_correct'],
        literal_errors=positive['literal_errors']+null['literal_errors'],
        false_attribution=positive['false_attribution']+null['false_attribution'],
        unsupported_positive_outputs=positive['unsupported_positive_outputs'],null_outputs=null['emitted'],
        automatic_additional_sources=False)
    return dict(method_status=method,contribution_status=contribution,adj_incremental=incremental,checks=checks,observed_method_violations=violations,
        totals=totals,detection=detection,per_source=per_source,subsets=subsets,paired_summary=paired_summary,comparisons=comparisons,
        net_actual_supported_gain_vs_CDC_M=gain,contribution_threshold=threshold,technical_tasks=technical,next_candidate_allowed=False,
        method_criteria=run.METHOD_CRITERIA,contribution_criteria=run.CONTRIBUTION_CRITERIA)

def evaluate(out,backend,journal,*,tensor_root=None):
    out=Path(out);tensor_root=Path(tensor_root) if tensor_root is not None else out
    marker=run.read(out/'blind_complete.json')
    if marker.get('status')!='READ_BACK' or marker.get('reader_decisions')!=360 or marker.get('task_decisions')!=120:raise ValueError('all 360 blind decisions required before truth')
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
                    tasks[task]=dict(positive=positive,readers=readers,correction=plan['readers']['ADJ']['correction'],
                        local_CDC={w:assess(v,truth,positive,frames) for w,v in decision['local_CDC'].items()},
                        geometry=old.geometry(frames),rank_path_truth=path_truth(decision,frames),geometry_label='old three-window geometry M3, not confirmation stage3')
    result=dict(status='POST_PARTIAL',denominators=run.denominators(),tasks=tasks,quality=run.quality_slots(),
        native_audit={},clipping={},media_P0_PM_pair=pair_summary([]),native_P0_PM_pair=pair_summary([]),
        **summarize(tasks,False),quality_pass=None,population_FPR=None,source_authentication=False,bundles=[],
        claim_ceiling=run.CLAIM_CEILING,stop_interpretation='Main ADJ null DETECT immediately fails candidate; only saved-data diagnostics continue, no further physical calls/search.')
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
