"""Post-raw payload/time accounting; truth never selects a received observation."""
from collections import Counter
from .public_fpr001_v4 import METHODS, VERIFY_BITS, verify, frame_messages, relation

RELATIONS=('UNKNOWN','adjacent','gap','repeat','reorder','cross-source','synthetic')


def evaluate_view(method,arm,raw,query_record,mapping,queries):
    import numpy as np
    n=len(mapping);bits=np.asarray(raw['bits']);observed=np.asarray(raw['observed'],bool)
    valid=observed & np.asarray(raw['valid'],bool)
    target=verify(method,raw,query_record['query'],planned_frames=n)
    wrong=verify(method,raw,query_record['wrong_query'],planned_frames=n) if arm!='OFF' and query_record['wrong_query'] is not None else None
    # All payload slots remain in the denominator; OFF has no embedded-message truth.
    payload=dict(planned_frames=n,planned_bits=n*METHODS[method],defined_bits=0,undefined_bits=0,
        na_bits=n*METHODS[method] if arm=='OFF' else 0,observed_bits=0,error_bits=0,missing_bits=0,erased_bits=0,
        valid_frames=int(valid.sum()),execution_missing_frames=int((~observed).sum()),
        empty_mask_frames=int((observed & ~valid).sum()),full_message_frames=0,full_owner_frames=0,
        full_message_eligible_frames=0,full_owner_eligible_frames=0)
    actual_owner_correct=np.zeros(n,bool);full=np.zeros(n,bool)
    if arm!='OFF':
        for i,node in enumerate(mapping):
            owner=queries[method+'/'+node['source_id']]['query']
            defined=VERIFY_BITS[method] if method=='frameindex' and node['kind']=='synthetic' else METHODS[method]
            truth=owner if method!='frameindex' or node['kind']=='synthetic' else frame_messages(owner,node['index'],1)[0]
            payload['defined_bits']+=defined;payload['undefined_bits']+=METHODS[method]-defined
            payload['full_owner_eligible_frames']+=1
            payload['full_message_eligible_frames']+=int(defined==METHODS[method])
            if not observed[i]:payload['missing_bits']+=defined
            elif not valid[i]:payload['erased_bits']+=defined
            else:
                err=int(np.count_nonzero(bits[i,:defined]!=truth[:defined]))
                payload['observed_bits']+=defined;payload['error_bits']+=err
                actual_owner_correct[i]=bool(np.array_equal(bits[i,:len(owner)],owner))
                full[i]=defined==METHODS[method] and err==0
                payload['full_owner_frames']+=int(actual_owner_correct[i]);payload['full_message_frames']+=int(full[i])
    payload['ber']=payload['error_bits']/payload['observed_bits'] if payload['observed_bits'] else None
    payload['coverage']=payload['observed_bits']/payload['defined_bits'] if payload['defined_bits'] else None
    payload['native_video_target_agreement']=None
    if arm!='OFF' and target.get('video_bits') is not None:
        error=sum(a!=b for a,b in zip(target['video_bits'],query_record['query']))
        payload['native_video_target_agreement']=dict(bits=METHODS[method],errors=error,ber=error/METHODS[method],
            full_target_message=error==0,scope='agreement with target query; T12 is not a homogeneous-source assertion')
    time_summary={'status':'N/A','planned_nodes':n,'planned_edges':max(0,n-1)};arrays={}
    if method=='frameindex':
        # -1 is no output; owner may be accepted while decoded index is invalid.
        indices=np.array([f['index'] if f['index'] is not None else -1 for f in target['frames']],np.int16)
        owners=np.array([-1 if f['owner_accept'] is None else int(f['owner_accept']) for f in target['frames']],np.int8)
        kinds=np.array([{'target':0,'foreign':1,'synthetic':2}[node['kind']] for node in mapping],np.uint8)
        original=np.array([-1 if node['index'] is None else node['index'] for node in mapping],np.int16)
        counts=Counter();joint=np.zeros(n,bool);exact_owner=np.zeros(n,bool)
        for i,node in enumerate(mapping):
            kind=node['kind'];counts[kind+'_planned']+=1
            if not observed[i]:counts[kind+'_unexecuted']+=1;continue
            if indices[i]<0:
                counts[kind+'_erased']+=1
                if owners[i]==1:counts[kind+'_owner_positive_invalid_index']+=1
                continue
            counts[kind+'_outputs']+=1
            if arm=='OFF':counts[kind+'_off_false_time_outputs']+=1
            elif kind=='target':
                delta=abs(int(indices[i])-node['index'])
                for tol,name in ((0,'exact'),(1,'within1'),(4,'within4')):counts['target_'+name]+=int(delta<=tol)
                counts['target_wrong_index']+=int(delta!=0)
                joint[i]=delta==0 and owners[i]==1;exact_owner[i]=delta==0 and actual_owner_correct[i]
            elif kind=='foreign':counts['foreign_false_target_attribution']+=1
            else:counts['synthetic_false_precise_time']+=1
        truth=[];prediction=[];edges=Counter()
        for i in range(n-1):
            expected=relation(mapping[i],mapping[i+1]);truth.append(RELATIONS.index(expected))
            # The simple baseline has no foreign/synthetic label detector; do not infer it from truth.
            if not observed[i] or not observed[i+1] or indices[i]<0 or indices[i+1]<0:actual='UNKNOWN'
            else:
                delta=int(indices[i+1])-int(indices[i]);actual='adjacent' if delta==1 else 'repeat' if delta==0 else 'gap' if delta>1 else 'reorder'
            prediction.append(RELATIONS.index(actual));edges[expected+'_planned']+=1
            edges[expected+'_unknown' if actual=='UNKNOWN' else expected+'_correct' if actual==expected and arm!='OFF' else expected+'_incorrect']+=1
        time_summary=dict(status='OBSERVED' if observed.all() else 'PARTIAL',planned_nodes=n,planned_edges=max(0,n-1),
            nodes=dict(counts),edges=dict(edges),query_accept_and_exact_index=int(joint.sum()) if arm!='OFF' else None,
            full_owner_and_exact_index=int(exact_owner.sum()) if arm!='OFF' else None,
            original_target_time_truth='N/A' if arm=='OFF' else 'DEFINED',
            edit_localization='adjacent/gap/repeat/reorder edges only; rejection is not foreign/synthetic recognition')
        arrays=dict(index=indices,owner_accept=owners,observed=observed,kind=kinds,original_index=original,
            exact_joint=joint,full_owner_exact_joint=exact_owner,edge_truth=np.array(truth,np.uint8),
            edge_prediction=np.array(prediction,np.uint8))
    def compact(value):
        if value is None:return None
        return {k:v for k,v in value.items() if k not in ('frames','video_bits')}
    return dict(payload=payload,ownership=compact(target),wrong_query=compact(wrong),time=time_summary,
        wrong_query_collision=query_record['wrong_query_collision']),arrays
