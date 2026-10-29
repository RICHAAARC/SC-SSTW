"""Post-only truth evaluation; fixed failures and negative acceptance stay explicit."""
from pathlib import Path
import numpy as np
from main.tube_state import b_framewise_qim_time_recovery_v1 as method
from .b_framewise_qim_time_recovery_v1_records import load,denominators


def classify_indices(nodes,mapping,field):
    counts=dict(correct=0,wrong=0,unresolved=0,not_executed=0)
    details=[]
    for node,truth in zip(nodes,mapping):
        value=node[field];status='unresolved' if value is None else 'correct' if value==truth else 'wrong'
        counts[status]+=1
        interval=node['interval']
        details.append(dict(received_frame=node['received_frame'],truth=truth,predicted=value,status=status,
                            truth_in_interval=None if interval is None else interval[0]<=truth<=interval[1],
                            width=None if interval is None else interval[1]-interval[0]+1))
    return counts,details


def evaluate(output,rows):
    if any(v['status'] in ('STARTED','NOT_STARTED') for v in rows.values()): raise RuntimeError('blind rows not terminal')
    output=Path(output);truth=load(output/'private/attack_truth.json');views=[]
    try:
        resources=load(output/'resources.json')
        if not isinstance(resources,dict): raise ValueError('resource record must be an object')
    except (OSError,ValueError) as error: resources=dict(status='UNRESOLVED_RECEIPT',error=str(error))
    resource_failed=resources.get('status')!='CLOSED' or bool(resources.get('error'))
    for task in truth['views']:
        ident=task['id'];mapping=task['mapping'];length=len(mapping);folder=output/'received'/ident
        result=dict(id=ident,arm=task['arm'],view=task['view'],frames=length,relations=length-1,
                    bit_slots=length*630,lane_slots=length*5,
                    receiver_status=rows[ident+'/receiver']['status'],time_read_status=rows[ident+'/time_read']['status'],
                    frame_counts=dict(correct=0,wrong=0,unresolved=0,not_executed=length),
                    relation_counts=dict(correct=0,wrong=0,unresolved=0,not_executed=length-1),
                    supported=None,base_false_accept=None,local_bits=None,local_labels=None)
        raw=None;decision=None
        try:
            if rows[ident+'/time_read']['status']=='RETURNED':
                with np.load(folder/'raw.npz',allow_pickle=False) as archive: raw={k:archive[k] for k in ('soft','mask')}
            if rows[ident+'/receiver']['status']=='RETURNED': decision=load(folder/'blind.json')
            if decision is not None:
                if len(decision['nodes'])!=length or len(decision['relations'])!=length-1: raise ValueError('blind result length mismatch')
                result.update(supported=decision['qualification']=='SUPPORTED_RELATION_SET',path_status=decision['path_status'],qualification=decision['qualification'])
                if task['arm']=='BASE':
                    result.update(base_false_accept=result['supported'],frame_counts=None,relation_counts=None,
                                  negative_false_labels=sum(bool(x['labels']) for frame in decision['labels'] for x in frame),
                                  negative_compatible_labels=sum(len(x['labels']) for frame in decision['labels'] for x in frame))
                else:
                    result['frame_counts'],result['frame_details']=classify_indices(decision['nodes'],mapping,'output_index')
                    result['raw_frame_counts'],result['raw_frame_details']=classify_indices(decision['nodes'],mapping,'raw_index')
                    counts=dict(correct=0,wrong=0,unresolved=0,not_executed=0);details=[]
                    for r,rel in enumerate(decision['relations']):
                        a,b=mapping[r:r+2];kind='REPEAT' if a==b else 'NORMAL' if b==a+1 else 'GAP'
                        predicted=rel['output_endpoints'];status='unresolved' if predicted is None else 'correct' if predicted==[a,b] else 'wrong'
                        counts[status]+=1
                        details.append(dict(received_left=r,status=status,truth_endpoints=[a,b],output_endpoints=predicted,
                                            kind=rel['kind'],true_kind=kind,kind_correct=None if rel['kind'] in (None,'AMBIGUOUS') else rel['kind']==kind,
                                            delta_range=rel['delta'],true_delta=b-a,missing_indices=rel['output_missing_indices'],repeated_index=rel['output_repeated_index']))
                    result['relation_counts']=counts;result['relation_details']=details
            if task['arm']=='QIM':
                bit_counts=dict(active=0,correct=0,wrong=0,erased=0,not_executed=0)
                label_counts=dict(active=0,correct=0,wrong=0,unresolved=0,not_executed=0,inactive=0,inactive_candidates=0)
                for r,n in enumerate(mapping):
                    for lane in range(5):
                        j=n-(n-lane)%5
                        labels=decision['labels'][r][lane]['labels'] if decision is not None else None
                        if j<0:
                            label_counts['inactive']+=1
                            label_counts['inactive_candidates']+=int(bool(labels));continue
                        bit_counts['active']+=126;label_counts['active']+=1
                        if labels is None: label_counts['not_executed']+=1
                        elif len(labels)!=1: label_counts['unresolved']+=1
                        else: label_counts['correct' if labels==[j] else 'wrong']+=1
                        if raw is None: bit_counts['not_executed']+=126;continue
                        word=int(method.words([j])[0]);bits=np.array([(word>>k)&1 for k in range(62,-1,-1)],dtype=bool)
                        soft=raw['soft'][r,lane];known=raw['mask'][r,lane]&np.isfinite(soft)&(soft!=0)
                        bit_counts['erased']+=int((~known).sum())
                        bit_counts['wrong']+=int((known&((soft>0)!=bits)).sum())
                        bit_counts['correct']+=int((known&((soft>0)==bits)).sum())
                result['local_bits']=bit_counts;result['local_labels']=label_counts
        except (OSError,ValueError,KeyError,TypeError) as error:
            result['evaluation_error']=str(error);result['supported']=None;result['base_false_accept']=None
        views.append(result)
    positives=[v for v in views if v['arm']=='QIM'];negatives=[v for v in views if v['arm']=='BASE']
    false_accepts=[v['id'] for v in negatives if v['base_false_accept'] is True]
    wrong=[v['id'] for v in positives if any((v.get(k) or {}).get('wrong',0)>0 for k in ('frame_counts','relation_counts'))]
    all_rejected=all(v['supported'] is False for v in positives)
    complete=not resource_failed and all(v['receiver_status']=='RETURNED' and not v.get('evaluation_error') for v in views)
    all_exact=all(v['frame_counts']['correct']==v['frames'] and v['relation_counts']['correct']==v['relations'] for v in positives)
    facts=[]
    if false_accepts:facts.append('BASE_FALSE_ACCEPT')
    if wrong:facts.append('WRONG_ATTRIBUTION')
    if all_rejected:facts.append('QIM_ALL_REJECTED')
    if resource_failed:facts.append('RESOURCE_UNRESOLVED')
    if not complete:facts.append('RUN_OR_EVALUATION_FAILURE')
    return dict(denominators=denominators(),views=views,failure_facts=facts,resources=resources,
                outcome='FAILURE_FACTS_RETAINED' if facts else 'COMPLETE_RECOVERY' if all_exact else 'PARTIAL_RECOVERY',
                base_five_rejected=all(v['base_false_accept'] is False for v in negatives),
                all_blind_terminal_before_truth=True,context='Original native target failure and optimality UNPROVEN unchanged',
                claim_ceiling='One source; five fixed positive/negative views; second-generation H264; no edited identity/general rejection proof')
