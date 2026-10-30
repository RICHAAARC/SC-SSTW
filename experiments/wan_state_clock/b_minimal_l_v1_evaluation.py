"""Post-only e02 evaluation. Never reread raw or execute a scientific kernel."""
from pathlib import Path
from .b_minimal_l_v1_records import load, denominators, LENGTHS


def object_file(path):
    value = load(path)
    if not isinstance(value, dict): raise ValueError('saved object required')
    return value


def evaluate(output, rows):
    if any(v['status'] in ('STARTED', 'NOT_STARTED') for v in rows.values()):
        raise RuntimeError('all blind rows must terminate before truth')
    output = Path(output)
    try:
        truth = object_file(output/'private/attack_truth.json')['views']
        if not isinstance(truth,list) or len(truth)!=10: raise ValueError('fixed ten-view mapping required')
        for task,length in zip(truth,LENGTHS):
            if not isinstance(task,dict) or not isinstance(task.get('mapping'),list) or len(task['mapping'])!=length:
                raise ValueError('fixed mapping length')
            if any(type(n)!=int for n in task['mapping']): raise ValueError('integer post mapping required')
    except (OSError,ValueError,KeyError,TypeError) as error:
        return dict(denominators=denominators(),status='TRUTH_UNRESOLVED',error=str(error),
                    views=[dict(id=f'v{i:02}',source_id='b2-m3-e02',frames=n,relations=n-1,
                                truth_unknown_frames=n,truth_unknown_relations=n-1) for i,n in enumerate(LENGTHS)],
                    formal_new_determined=0,new_relation_outputs=0,all_blind_terminal_before_truth=True)
    views = []
    for i, (task, length) in enumerate(zip(truth, LENGTHS)):
        ident = f'v{i:02}'; folder = output/'received'/ident
        result = dict(id=ident, source_id='b2-m3-e02', arm=('QIM','BASE')[i//5], view=task['view'],
                      frames=length, relations=length-1, bit_slots=length*630, lane_slots=length*5,
                      hard_status=rows[ident+'/hard_receiver']['status'], L_status=rows[ident+'/L']['status'],
                      hard=dict(correct=0,wrong=0,unresolved=0,unknown=length),
                      hard_relations=dict(correct=0,wrong=0,unresolved=0,unknown=length-1),
                      truth_excluded_N=None, old_unresolved_L=None, old_determined_candidate_disagreement=None,
                      old_determined_lost=None, old_determined_changed=None,
                      base_qualification=None, soft_scoreable=None, candidate_outputs=None,
                      base_false_candidate_outputs=None, base_soft_evidence='UNRESOLVED',
                      hard_comparisons=None,C_frame_j=None,C_replica_bit_terms=None,
                      C_replica_bit_terms_kind='DENSE_LOGICAL_EQUIVALENT_UPPER_BOUND_NOT_PHYSICAL_OPERATION_COUNT',
                      formal_new_determined=0,new_relation_outputs=0)
        mapping = task['mapping']
        if len(mapping) != length: raise ValueError('fixed mapping length')
        decision = None
        try:
            if result['hard_status'] == 'RETURNED':
                decision = object_file(folder/'blind.json')
                nodes, relations = decision['nodes'], decision['relations']
                if len(nodes)!=length or len(relations)!=length-1: raise ValueError('hard frame/relation length')
                for r, node in enumerate(nodes):
                    if not isinstance(node,dict) or node['received_frame']!=r: raise ValueError('hard node coordinate')
                    value=node['output_index']; interval=node['interval']
                    if value is not None and type(value)!=int: raise ValueError('hard formal index')
                    if interval is not None and (not isinstance(interval,list) or len(interval)!=2 or any(type(x)!=int for x in interval)):
                        raise ValueError('hard interval')
                if any(not isinstance(rel,dict) or 'output_endpoints' not in rel for rel in relations): raise ValueError('hard relation')
                counts=dict(correct=0,wrong=0,unresolved=0,unknown=0)
                excluded=0;details=[]
                for node,n in zip(nodes,mapping):
                    value=node['output_index']; interval=node['interval']
                    status='unresolved' if value is None else 'correct' if value==n else 'wrong'
                    counts[status]+=1
                    outside=interval is None or not interval[0]<=n<=interval[1];excluded+=int(outside)
                    details.append(dict(received_frame=node['received_frame'],truth=n,hard=value,status=status,truth_excluded_N=outside))
                relation_counts=dict(correct=0,wrong=0,unresolved=0,unknown=0)
                for r,rel in enumerate(relations):
                    value=rel['output_endpoints']
                    relation_counts['unresolved' if value is None else 'correct' if value==mapping[r:r+2] else 'wrong']+=1
                result.update(hard=counts,hard_relations=relation_counts,truth_excluded_N=excluded,hard_details=details,
                              qualification=decision['qualification'],hard_comparisons=decision['comparisons'])
                if i>=5: result['base_qualification']=decision['qualification']
        except (OSError,ValueError,KeyError,TypeError) as error:
            decision=None; result['hard_error']=str(error)
            result['hard']=dict(correct=0,wrong=0,unresolved=0,unknown=length)
            result['hard_relations']=dict(correct=0,wrong=0,unresolved=0,unknown=length-1)
        # C accounting is independent of the already saved hard and L evidence.
        try:
            if rows[ident+'/C']['status']=='RETURNED':
                comparisons=object_file(folder/'C.json')['comparisons']
                if type(comparisons)!=int or not 0<=comparisons<=length*denominators()['codewords']:
                    raise ValueError('C comparisons must be an integer within the fixed frame-j bound')
                result['C_frame_j']=comparisons
                # Dense two-replica x 63-bit logical expansion, not measured scalar work.
                result['C_replica_bit_terms']=126*comparisons
        except (OSError,ValueError,KeyError,TypeError) as error:
            result['C_error']=str(error)
        try:
            if result['L_status']=='RETURNED':
                candidate=object_file(folder/'candidates.json')['nodes']
                if len(candidate)!=length: raise ValueError('candidate length')
                for r,node in enumerate(candidate):
                    if not isinstance(node,dict) or node['received_frame']!=r: raise ValueError('candidate coordinate')
                    if node['candidate_index'] is not None and type(node['candidate_index'])!=int: raise ValueError('candidate index')
                    if type(node['best_count'])!=int or not isinstance(node['model_status'],str): raise ValueError('candidate state')
                scoreable=sum(n['model_status']=='DEFINED' for n in candidate)
                outputs=sum(n['candidate_index'] is not None for n in candidate)
                result.update(soft_scoreable=scoreable,candidate_outputs=outputs)
                if i>=5:
                    result.update(base_false_candidate_outputs=outputs,
                                  base_soft_evidence='NO_SOFT_NEGATIVE_EVIDENCE' if not scoreable else 'FINITE_SOURCE_SOFT_NEGATIVE_OBSERVATIONS')
                if decision is not None:
                    counts=dict(correct=0,wrong=0,tie=0,undefined=0,unqualified=0)
                    details=[];disagreement=0
                    for old,new,n in zip(decision['nodes'],candidate,mapping):
                        value=new['candidate_index']
                        if old['output_index'] is not None:
                            disagreement+=int(value is not None and value!=old['output_index']);continue
                        state=('undefined' if new['model_status']!='DEFINED' else 'tie' if new['best_count']!=1
                               else 'unqualified' if value is None else 'correct' if value==n else 'wrong')
                        counts[state]+=1;details.append(dict(received_frame=new['received_frame'],truth=n,candidate=value,status=state))
                    preserved=object_file(folder/'formal_preservation.json')
                    result.update(old_unresolved_L=counts,old_unresolved_L_details=details,
                                  old_determined_candidate_disagreement=disagreement,
                                  old_determined_lost=preserved['lost'],old_determined_changed=preserved['changed'])
        except (OSError,ValueError,KeyError,TypeError) as error: result['L_error']=str(error)
        views.append(result)
    try: resources=object_file(output/'resources.json')
    except (OSError,ValueError) as error: resources=dict(status='UNRESOLVED',error=str(error))
    return dict(source_id='b2-m3-e02',denominators=denominators(),views=views,resources=resources,
                all_blind_terminal_before_truth=True,formal_new_determined=0,new_relation_outputs=0,
                e01='Prior development only; no e01 array read or recomputation',
                claim_ceiling='One fixed independent-QIM-source candidate diagnostic; no reliable release, identity or general FPR claim')
