"""Post-only paired interpretation; no model or codec calls."""
from pathlib import Path
import time
from runtime.wan import b_gamma2_fixed_validation_v1 as run
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


def summarize(tasks,historical,complete):
    per_pair={};historical_retained={}
    for h,role,_ in run.HOSTS:
        if role!='eval':continue
        for fixture in run.FIXTURES:
            first=tasks[f'{h}/PM1/{fixture}/key00']['aggregates']['CDC_M']
            second=tasks[f'{h}/PM2/{fixture}/key00']['aggregates']['CDC_M']
            per_pair[h+'/'+fixture]=paired(first,second)
            if h=='b2-m3-e01':
                prior=historical[fixture]['slots']
                required=[i for i,r in enumerate(prior) if r['emitted'] and r['literal_correct']]
                historical_retained[fixture]=dict(required=required,retained=[i for i in required if second['slots'][i]['actual_supported_correct']],
                    lost=[i for i in required if not second['slots'][i]['actual_supported_correct']])
    totals={}
    for reader in run.READERS:
        for scope in ('positive','null'):
            records=[t['aggregates'][reader]['counts'] for t in tasks.values() if t['positive']==(scope=='positive')]
            totals[reader+'/'+scope]={k:sum(r[k] for r in records) for k in records[0]}
    per_gamma={}
    for gamma in ('PM1','PM2'):
        positive=[t['aggregates']['CDC_M']['counts'] for q,t in tasks.items() if q.split('/')[1]==gamma and t['positive']]
        null=[t['aggregates']['CDC_M']['counts'] for q,t in tasks.items() if q.split('/')[1] in ('OFF','P0',gamma) and not t['positive']]
        per_gamma[gamma]=dict(positive={k:sum(x[k] for x in positive) for k in positive[0]},null={k:sum(x[k] for x in null) for k in null[0]})
    main=[t['aggregates']['CDC_M']['counts'] for t in tasks.values()]
    technical=sum(x['technical_unread'] for x in main)
    history_count=sum(len(x['required']) for x in historical_retained.values())
    complete=bool(complete and technical==0 and history_count==96)
    checks=dict(e02_gain_each_fixture=all(per_pair['b2-m3-e02/'+f]['added_actual_supported_correct'] for f in run.FIXTURES),
        e01_historical_96_retained=history_count==96 and all(not x['lost'] for x in historical_retained.values()),
        no_loss_of_common_tau_supported_slots=all(not x['lost_actual_supported_correct'] for x in per_pair.values()),
        common_and_added_literal_errors_zero=all(x['left_common_errors']==x['right_common_errors']==x['right_added_literal_errors']==0 for x in per_pair.values()),
        false_attribution_zero=sum(x['false_attribution'] for x in main)==0,
        actual_supporting_attribution_zero=sum(x['unsupported_positive_outputs'] for x in main)==0,
        unique_null_outputs_zero=totals['CDC_M/null']['emitted']==0,
        required_observations_complete=complete)
    support='UNRESOLVED' if not complete else 'SUPPORTED_DEVELOPMENT_ONLY' if all(checks.values()) else 'NOT_SUPPORTED_FIXED_SCOPE'
    return dict(development_support=support,support_checks=checks,paired=per_pair,historical_e01=historical_retained,
        totals=totals,per_gamma=per_gamma,unique_null_denominator=1152,scientific_pass=None)


def evaluate(source,out,backend,journal):
    source,out=Path(source),Path(out)
    if run.read(out/'blind_complete.json').get('status')!='READ_BACK':raise ValueError('all blind plans/decisions must be read back first')
    run.dump(out/'post_truth_access.json',dict(opened_unix=time.time(),blind_record=str(out/'blind_complete.json')))
    cfg=run.read(source/'protocol.json')['config'];bits=cfg['writer']['bits']
    if len(bits)!=32 or any(type(v) is not int or v not in (0,1) for v in bits):raise ValueError('fixed32 payload truth required')
    tasks={};historical={}
    from runtime.wan.b_terminal_three_group_v1 import source_frames
    threshold=run.read(out/'thresholds.json')
    for host in cfg['sources']:
        if host['role']!='eval':continue
        h=host['id']
        for fixture in run.FIXTURES:
            frames=source_frames(host,fixture)
            for arm in run.ARMS:
                for key in run.KEYS:
                    task='/'.join((h,arm,fixture,key));folder=out/'blind'/task
                    plan,decision=run.read(folder/'plan.json'),run.read(folder/'decision.json')
                    positive=arm in ('PM1','PM2') and key=='key00'
                    local={region:{w:assess(decision['local'][region][w],bits,positive,frames) for w in run.WINDOWS} for region in ('R0','R1','CDC')}
                    aggregates={name:assess(decision['aggregates'][name],bits,positive,frames) for name in run.READERS}
                    windows={}
                    for window,w in run.WINDOWS.items():
                        phase=(-frames[w])%4;s=(frames[w]+phase)//4+1
                        loc=plan['locals'][window]
                        member=next((m for m in loc['members'] if m['phase']==phase and m['s']==s),None)
                        windows[window]=dict(D=loc['D'],C_size=len(loc['C']) if loc['C'] is not None else None,
                            maximum=loc['maximum'],maximum_minus_tau=loc['maximum']-threshold['tau'] if loc['maximum'] is not None else None,
                            true_h=dict(phase=phase,s=s),true_h_S=member.get('S') if member else None,
                            true_h_S_minus_tau=member['S']-threshold['tau'] if member and member['S'] is not None else None,
                            true_h_retained=member.get('in_C') if member else None)
                    tasks[task]=dict(positive=positive,local=local,aggregates=aggregates,windows=windows,
                        paths=old.path_assessment(plan,host,fixture),geometry=old.geometry(frames),
                        invariance=decision['aggregates'].get('invariance'),M_vs_E=old.compare(aggregates['CDC_E'],aggregates['CDC_M']))
            if h=='b2-m3-e01':
                previous=run.read(source/'blind'/h/'PM'/fixture/'key00/decision.json')
                historical[fixture]=assess(previous['aggregates']['CDC_M'],bits,True,frames)
    required=[r for r in journal.data.values() if r['kind'] not in ('quality','post')]
    summary=summarize(tasks,historical,all(r['status']=='RETURNED' for r in required))
    result=dict(status='POST_PARTIAL',denominators=run.denominators(),tasks=tasks,quality=run.quality_slots(),historical_quality={},
        native_audit={},clipping={},common_threshold=threshold,**summary,
        quality_pass=None,CFAR=False,population_FPR=None,bundles=[],
        evidence_ceiling='Viewed fixed four sources, two eval sources, fixed keys and fixtures; development paired evidence only, no population FPR or source authentication')
    run.dump(out/'evaluation.json',result)
    for host in cfg['sources']:
        h=host['id'];result['native_audit'][h]=run.read(out/'private'/h/'native_audit.json')
        result['clipping'][h]=dict(PM2=run.read(out/'media'/h/'PM2/full/preclip.json'),PM1=run.read(source/'media'/h/'PM/full/preclip.json'))
        for scope,layers in [('full',('preclip','rgb8','readback'))]+[(f,('rgb8','readback')) for f in run.FIXTURES]:
            frames=list(range(181)) if scope=='full' else source_frames(host,scope)
            for layer in layers:
                slot=f'{h}/quality/{scope}/{layer}';path=out/'quality'/(slot+'.json')
                def quality():
                    return backend.quality(backend.load_tensor(source/'media'/h/'P0'/scope/(layer+'.pt')),
                        backend.load_tensor(out/'media'/h/'PM2'/scope/(layer+'.pt')),frames,'P0_to_PM2')
                try:result['quality'][slot]=journal.call(slot,quality,path)
                except BaseException:
                    state=journal.data[slot];failed=result['quality'][slot]
                    failed.update(status=state['status'],attempts=state['attempts'],error=state.get('error'))
                    for row in failed['frames']+failed['adjacent']:row['status']=state['status']
                    run.dump(path,failed);run.dump(out/'evaluation.json',result);raise
                old_path=source/'quality'/h/'quality'/scope/'P0_to_PM'/(layer+'.json')
                try:result['historical_quality'][slot]=run.read(old_path)
                except (OSError,ValueError) as error:
                    result['historical_quality'][slot]=dict(status='MISSING_OR_UNREADABLE',error=str(error),frame_denominator=len(frames),adjacent_denominator=len(frames)-1)
                run.dump(out/'evaluation.json',result)
    result['status']='POST_RECORDED'
    return result
