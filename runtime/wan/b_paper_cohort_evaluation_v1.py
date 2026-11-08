"""Once-only post and fixed-source summaries; technical unknowns never become zero."""
from collections import Counter
from pathlib import Path
import statistics
import math
import time
from main.tube_state import b_paper_cohort_v1 as p
from . import b_paper_cohort_records_v1 as rec


def bounds_for_slot(row,*,returned,time_ready,truth_available):
    """Classify actual evaluability, not the generic spelling UNRESOLVED."""
    if not returned or not truth_available: return [0,1],'TECHNICAL_NOT_EVALUABLE'
    if row.get('value') is not None:
        refs=row.get('supporting_refs',[])
        if refs and all(r.get('actual_physical_support') for r in refs):
            value=int(row['correct_supported']);return [value,value],'OBSERVED_OUTPUT_AND_SUPPORT'
        return [0,1],'OUTPUT_SUPPORT_RECEIPT_UNKNOWN'
    # Original CDC also erases opposite legal values across local queries;
    # conflict_refs records local conflicts and can be empty in that case.
    if row['status']=='CONFLICT_ERASURE':
        return [0,0],'EXECUTED_CONFLICT_ERASURE'
    causes=set(row.get('cause_set',[]))
    if time_ready and row['status']=='ERASURE_OR_REJECT' and causes and causes<= {'NO_FULL_CELL'}:
        return [0,0],'EXECUTED_NO_FULL_CELL'
    return [0,1],'MISSING_OR_UNCLASSIFIED_TECHNICAL_SUPPORT'


def compare_rows(b,u):
    common=[i for i in range(32) if b[i].get('literal_correct') is not None and u[i].get('literal_correct') is not None]
    lower=sum(x['correct_supported_bounds'][0]-y['correct_supported_bounds'][1] for x,y in zip(b,u))
    upper=sum(x['correct_supported_bounds'][1]-y['correct_supported_bounds'][0] for x,y in zip(b,u))
    def known(r): return r['correct_supported_bounds'][0]==r['correct_supported_bounds'][1]
    return dict(delta_slot_sum_bounds=[lower,upper],common_observed_slots=common,common_count=len(common),
        B_common_BER=sum(b[i]['literal_correct'] is False for i in common)/len(common) if common else None,
        control_common_BER=sum(u[i]['literal_correct'] is False for i in common)/len(common) if common else None,
        added_correct_support=[i for i in range(32) if known(b[i]) and known(u[i]) and b[i]['correct_supported'] and not u[i]['correct_supported']],
        lost_correct_support=[i for i in range(32) if known(b[i]) and known(u[i]) and u[i]['correct_supported'] and not b[i]['correct_supported']],
        unknown_comparison_slots=[i for i in range(32) if not known(b[i]) or not known(u[i])])


def post_once(output):
    output=Path(output)
    if rec.read(output/'blind_terminal.json')['status']!='ALL_BLIND_TERMINAL': raise ValueError('blind closure before truth')
    post=rec.read(output/'post_status.json')
    if post['status']!='NOT_STARTED': return post
    # Started persists before any truth access; a second call cannot repair it.
    post.update(status='STARTED',started=1,started_at=time.time());rec.write(output/'post_status.json',post)
    tick=time.monotonic()
    try:
        value=evaluate(output)
        post.update(status='RETURNED',returned=1,seconds=time.monotonic()-tick)
        rec.write(output/'post_status.json',post);return value
    except BaseException as error:
        post.update(status='FAILED_CONSUMED',error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-tick)
        rec.write(output/'post_status.json',post);raise


def evaluate(output):
    from .b_relative_time_payload_evaluation_v1 import slot_rows,paired_raw
    from .b_joint_time_relative_evaluation_v1 import singleton,edge_pair
    rec.write(output/'truth_access.json',dict(opened_at=time.time(),after_all_blind=True))
    def optional(path):
        try: return rec.read(output/path)
        except (OSError,ValueError): return None
    maps=optional('private/attack_truth.json');writer=optional('private/writer.json')
    message=writer.get('bits') if isinstance(writer,dict) else None
    if message is not None and (len(message)!=32 or any(type(b) is not int or b not in (0,1) for b in message)):
        raise ValueError('actual writer binary32 required')
    operations=rec.read(output/'state.json');raw=rec.read(output/'raw_bank.json')
    time_rows=[];edge_rows=[];time_summary={};payload={};per_view={}
    for item in p.inputs():
        uid,arm,view,n=(item[k] for k in ('id','arm','view','length'))
        mapping=maps.get(view) if isinstance(maps,dict) else None
        if mapping is not None and (len(mapping)!=n or any(type(t) is not int or not 0<=t<181 for t in mapping)):
            raise ValueError('invalid actual editor mapping')
        relative=optional(f'blind/{uid}/relative.json')
        available=operations[f'time/{uid}/relative_compare']['status']=='RETURNED' and relative is not None
        if available and (len(relative['conditional_nodes'])!=n or len(relative['conditional_edges'])!=n-1):
            raise ValueError('fixed time denominator required')
        nodes=relative['conditional_nodes'] if available else [None]*n
        original=relative['original_f42']['nodes'] if available else [None]*n
        own=[]
        for r,(node,old) in enumerate(zip(nodes,original)):
            candidate=node['conditional_candidates'] if node else None
            value=singleton(candidate);old_value=singleton(old['W']) if old else None
            truth=mapping[r] if mapping else None
            evaluable=available and mapping is not None and arm=='NEW'
            status=('CORRECT' if value==truth else 'WRONG' if value is not None else 'UNRESOLVED') if evaluable else (
                'UNWRITTEN_TIME' if available and arm!='NEW' else 'TECHNICAL_UNKNOWN')
            row=dict(input_id=uid,arm=arm,view=view,received_frame=r,status=status,
                output=value,candidates=candidate,truth_n=truth if arm=='NEW' else None,
                correct=(value==truth) if evaluable else None,wrong=(value is not None and value!=truth) if evaluable else None,
                original_correct=(old_value==truth) if evaluable else None,
                original_wrong=(old_value is not None and old_value!=truth) if evaluable else None,
                startup=(truth<4) if evaluable else None,comparison_performed=node.get('comparison_performed') if node else None,
                unwritten_time_proposal=available and arm!='NEW' and value is not None,
                execution_status=operations[f'time/{uid}/relative_compare']['status'])
            time_rows.append(row);own.append(row)
        own_edges=[]
        for r in range(n-1):
            edge=relative['conditional_edges'][r] if available else None
            old=relative['original_f42']['edges'][r] if available else None
            a,b=edge_pair(old),edge_pair(edge);truth=(mapping[r],mapping[r+1]) if mapping else None
            evaluable=available and truth is not None and arm=='NEW'
            status=('CORRECT' if b==truth else 'WRONG' if b is not None else 'UNRESOLVED') if evaluable else (
                'UNWRITTEN_TIME' if available and arm!='NEW' else 'TECHNICAL_UNKNOWN')
            row=dict(input_id=uid,arm=arm,view=view,received_edge=r,status=status,output=b,
                truth_pair=truth if arm=='NEW' else None,correct=b==truth if evaluable else None,
                original_correct=a==truth if evaluable else None,original_wrong=a is not None and a!=truth if evaluable else None,
                true_kind=('REPEAT' if truth[0]==truth[1] else 'GAP' if truth[1]>truth[0]+1 else 'NORMAL') if evaluable else None)
            edge_rows.append(row);own_edges.append(row)
        time_summary[uid]=dict(arm=arm,view=view,planned_nodes=n,planned_edges=n-1,
            node_statuses=dict(Counter(r['status'] for r in own)),edge_statuses=dict(Counter(r['status'] for r in own_edges)),
            empty_candidates=sum(r['candidates']==[] for r in own),unknown_candidates=sum(r['candidates'] is None for r in own),
            mapped=sum(r['output'] is not None for r in own),candidate_sizes=dict(Counter(str(len(r['candidates'])) if isinstance(r['candidates'],list) else 'UNKNOWN' for r in own)))
        plan=optional(f'plans/{uid}.json')
        for strategy in p.strategies(item):
            saved=rec.read(output/'final'/uid/(strategy+'.json'));key=p.key_for(strategy)
            returned=operations[f'{uid}/{strategy}/aggregate']['status']=='RETURNED'
            truth_available=message is not None and mapping is not None and arm!='OFF'
            if returned and truth_available:
                rows=slot_rows(saved,mapping,raw,message,key)
            else:
                rows=[dict(slot=s['slot'],value=s['value'],status=s['status'],cause_set=s.get('cause_set',[]),
                    supporting_refs=[],conflict_refs=s.get('conflict_refs',[]),literal_correct=None,
                    correct_supported=None,wrong_attribution=None,unsupported_output=None) for s in saved['slots']]
            time_ready=bool(plan and (strategy=='U_NEW' or plan['time']['status']=='READY'))
            if strategy=='P_PAIRED':
                partner=optional(f"plans/input{10+p.VIEWS.index(view)}.json")
                time_ready=bool(partner and partner['time']['status']=='READY')
            for row in rows:
                bounds,reason=bounds_for_slot(row,returned=returned,time_ready=time_ready,truth_available=truth_available)
                if arm=='OFF': bounds=[None,None];reason='NO_WRITTEN_IDENTITY_MESSAGE'
                row['correct_supported_bounds']=bounds;row['evaluability_reason']=reason
                if bounds[0]!=bounds[1] or bounds[0] is None: row['correct_supported']=None
                else: row['correct_supported']=bool(bounds[0])
            payload[uid+'/'+strategy]=rows;rec.write(output/'post'/uid/(strategy+'.json'),rows)
    for i,view in enumerate(p.VIEWS):
        b=payload[f'input{10+i}/B_NEW'];u=payload[f'input{10+i}/U_NEW'];paired=payload[f'input{5+i}/P_PAIRED']
        bp=optional(f'plans/input{10+i}.json');pp=optional(f'plans/input{5+i}.json')
        raw_pair=paired_raw(output,bp['strategies']['B_NEW'],pp['strategies']['P_PAIRED']) if bp and pp else dict(status='TECHNICAL_UNKNOWN',rows=[])
        rec.write(output/'post'/view/'paired_raw.json',raw_pair)
        per_view[view]=dict(B_vs_U=compare_rows(b,u),B_vs_P_PAIRED=compare_rows(b,paired),
            paired_raw={k:v for k,v in raw_pair.items() if k!='rows'})
    b=[r for i in range(10,15) for r in payload[f'input{i}/B_NEW']]
    u=[r for i in range(10,15) for r in payload[f'input{i}/U_NEW']]
    delta=[sum(r['B_vs_U']['delta_slot_sum_bounds'][j] for r in per_view.values())/160 for j in (0,1)]
    decisions=rec.read(output/'decisions.json');new=[r for r in time_rows if r['arm']=='NEW']
    source=rec.read(output/'attempt.json')['source']
    def slot_bounds(rows): return [sum(r['correct_supported_bounds'][j] for r in rows) for j in (0,1)]
    result=dict(status='SAVED_OUTPUT_EVALUATION',source=source,denominators=p.DENOMINATORS,
        time=time_summary,NEW_node_statuses=dict(Counter(r['status'] for r in new)),
        NEW_edge_statuses=dict(Counter(r['status'] for r in edge_rows if r['arm']=='NEW')),
        B_correct_supported_bounds=slot_bounds(b),U_correct_supported_bounds=slot_bounds(u),delta_bounds=delta,
        B_wrong_attribution_observed=sum(r.get('wrong_attribution') is True for r in b),
        B_unsupported_output_observed=sum(r.get('unsupported_output') is True for r in b),
        B_literal_errors_observed=sum(r.get('literal_correct') is False for r in b),
        B_literal_comparable=sum(r.get('literal_correct') is not None for r in b),
        B_unknown_slots=sum(r['correct_supported'] is None for r in b),U_unknown_slots=sum(r['correct_supported'] is None for r in u),
        NEW_q0_key00_MATCH=sum(d['verdict']=='MATCH' and d['arm']=='NEW' and d['key_id']=='key00' and d['claim_id']=='q0' for d in decisions),
        NEW_q0_key00_unknown=sum(d['verdict'] is None and d['arm']=='NEW' and d['key_id']=='key00' and d['claim_id']=='q0' for d in decisions),
        claims=dict(Counter(str(d['verdict']) for d in decisions)),views=per_view,scientific_pass=None,
        scope='fixed Wan content panel; conditional recovery, not authentication or population FPR')
    rec.write(output/'node_results.json',time_rows);rec.write(output/'edge_results.json',edge_rows)
    rec.write(output/'evaluation_summary.json',result);return result


def cohort_summary(root,batch):
    """Pure read of all fixed sources; no omitted failures and no automatic TEST-B start."""
    root=Path(root);sources=p.batch_sources(batch) if batch in ('TEST-A','TEST-B') else p.ROSTER if batch=='ALL' else []
    if not sources: raise ValueError('fixed batch required')
    rows=[]
    for source in sources:
        path=root/source['id']/'attempt01';evaluation=None;status='NOT_STARTED';errors=[]
        if path.exists():
            status='CONSUMED_UNKNOWN'
            try:
                attempt=rec.read(path/'attempt.json')
                if not isinstance(attempt,dict) or not isinstance(attempt.get('status'),str): raise ValueError('attempt status object required')
                status=attempt['status']
            except (OSError,ValueError,KeyError,TypeError) as error: errors.append(dict(file='attempt.json',error=str(error)))
            try:
                post=rec.read(path/'post_status.json')
                if not isinstance(post,dict) or not isinstance(post.get('status'),str): raise ValueError('post status object required')
                if post['status']=='RETURNED': evaluation=valid_summary(rec.read(path/'evaluation_summary.json'))
            except (OSError,ValueError,KeyError,TypeError) as error:
                errors.append(dict(file='post_status.json / evaluation_summary.json',error=str(error)));evaluation=None
        value=evaluation or dict(B_correct_supported_bounds=[0,160],U_correct_supported_bounds=[0,160],delta_bounds=[-1,1],
            NEW_node_statuses={'TECHNICAL_UNKNOWN':822},NEW_q0_key00_MATCH=0,NEW_q0_key00_unknown=5,
            B_wrong_attribution_observed=None,B_literal_errors_observed=None)
        rows.append(dict(source=source['id'],status=status,read_errors=errors,evaluation_available=evaluation is not None,**{k:value[k] for k in (
            'B_correct_supported_bounds','U_correct_supported_bounds','delta_bounds','NEW_node_statuses',
            'NEW_q0_key00_MATCH','NEW_q0_key00_unknown','B_wrong_attribution_observed','B_literal_errors_observed')}))
    def distribution(field):
        return {name:[fn([r[field][j] for r in rows]) for j in (0,1)]
                for name,fn in [('mean',statistics.mean),('median',statistics.median),('minimum',min),('maximum',max)]}
    gain=distribution('delta_bounds');lo,hi=gain['mean']
    targets=dict(time90=sum(r['NEW_node_statuses'].get('CORRECT',0)>=740 for r in rows),
                 supported80=sum(r['B_correct_supported_bounds'][0]>=128 for r in rows),
                 q0_four_of_five=sum(r['NEW_q0_key00_MATCH']>=4 for r in rows),
                 time_unknown=sum(r['NEW_node_statuses'].get('TECHNICAL_UNKNOWN',0)>0 for r in rows),
                 support_unknown=sum(r['B_correct_supported_bounds'][0]!=r['B_correct_supported_bounds'][1] for r in rows),
                 q0_unknown=sum(r['NEW_q0_key00_unknown']>0 for r in rows))
    return dict(batch=batch,source_denominator=len(rows),sources=rows,delta=gain,
        B_correct_supported=distribution('B_correct_supported_bounds'),U_correct_supported=distribution('U_correct_supported_bounds'),
        descriptive_targets=targets,targets_are_execution_gates=False,
        gain_claim='FINITE_PANEL_POSITIVE' if lo>0 else 'WITHDRAW_POSITIVE_CLAIM' if hi<=0 else 'INSUFFICIENT_EVIDENCE',
        scientific_pass=None,source_is_summary_unit=True,finite_panel_not_population=True)


def valid_summary(value):
    """Validate only the numeric fields used by this saved-result summary."""
    if not isinstance(value,dict): raise ValueError('evaluation summary object required')
    for name,maximum,minimum in [('B_correct_supported_bounds',160,0),('U_correct_supported_bounds',160,0),('delta_bounds',1,-1)]:
        bounds=value[name]
        if (not isinstance(bounds,list) or len(bounds)!=2 or any(type(x) not in (int,float) or not math.isfinite(x) for x in bounds)
                or not minimum<=bounds[0]<=bounds[1]<=maximum): raise ValueError('invalid fixed-range '+name)
    counts=value['NEW_node_statuses']
    if (not isinstance(counts,dict) or set(counts)-{'CORRECT','WRONG','UNRESOLVED','TECHNICAL_UNKNOWN'}
            or any(type(x) is not int or x<0 for x in counts.values()) or sum(counts.values())!=822):
        raise ValueError('fixed822 NEW node counts required')
    for name in ('NEW_q0_key00_MATCH','NEW_q0_key00_unknown'):
        if type(value[name]) is not int or not 0<=value[name]<=5: raise ValueError('fixed5 '+name+' required')
    if value['NEW_q0_key00_MATCH']+value['NEW_q0_key00_unknown']>5: raise ValueError('q0 counts exceed fixed5')
    for name in ('B_wrong_attribution_observed','B_literal_errors_observed'):
        if value[name] is not None and (type(value[name]) is not int or not 0<=value[name]<=160):
            raise ValueError('fixed160 '+name+' required')
    return value
