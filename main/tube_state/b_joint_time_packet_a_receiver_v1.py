"""Received-only packet diagnostics and complete-domain FP64 hinge evidence.

No hard graph, source mapping, truth, identity, threshold or formal acceptance.
The fixed startup templates are evaluated separately from mature-state ranks.
"""
import numpy as np
from . import b_joint_time_packet_a_v1 as packet

STATE_CHUNK = 2048
EPSILON = 1e-12
ROLES = ('R0', 'R1', 'F')


def inputs(soft, mask):
    soft, mask = np.asarray(soft), np.asarray(mask)
    if (soft.ndim != 4 or soft.shape[1:] != (5,2,63) or not len(soft)
            or soft.dtype.kind != 'f' or mask.shape != soft.shape or mask.dtype != np.bool_):
        raise ValueError('floating [R,5,2,63] soft and boolean observation mask required')
    return soft.astype(np.float64, copy=False), mask


def template(states, coding):
    if coding not in ('OLD', 'NEW'): raise ValueError('explicit OLD or NEW packet semantics required')
    fields = packet.state_fields(states)
    active, js, ages = fields['active'], fields['j'], fields['age']
    messages = np.where(active, js if coding == 'OLD' else 8*js+ages, 0)
    words = packet.encode_raw24_words(messages)
    bits = ((words[...,None] >> np.arange(62,-1,-1,dtype=np.uint64)) & 1).astype(bool)
    return dict(bits=bits, active=active, messages=np.where(active,messages,-1))


def hinge_terms(soft, mask):
    soft, mask = inputs(soft, mask)
    valid = mask & np.isfinite(soft)  # A finite zero is a supported soft observation.
    copies = valid.sum(axis=2)
    v = np.where(valid,soft,0.)
    divisor = np.maximum(copies,1)
    one = np.maximum(0.,-v).sum(axis=2)/divisor
    zero = np.maximum(0.,v).sum(axis=2)/divisor
    if not np.isfinite(one).all() or not np.isfinite(zero).all():
        raise ValueError('nonfinite FP64 replica hinge arithmetic')
    return dict(one=one, zero=zero, support=copies>0)


def score_states(terms, states, coding):
    """All requested states, with per-candidate startup support; no selection."""
    t = template(states,coding)
    active = np.broadcast_to(t['active'][...,None],t['bits'].shape)
    one = (t['bits'] & active).reshape(len(active),315).astype(np.float64)
    zero = (~t['bits'] & active).reshape(len(active),315).astype(np.float64)
    # Two nonnegative FP64 sums avoid subtracting a large common offset.
    numerator = (terms['one'].reshape(-1,315) @ one.T
                 + terms['zero'].reshape(-1,315) @ zero.T)
    denominator = (np.broadcast_to(terms['support'].sum(axis=(1,2))[:,None],numerator.shape).copy()
                   if active.all() else terms['support'].reshape(-1,315).astype(np.int64)
                   @ active.reshape(len(active),315).T.astype(np.int64))
    values = np.full(numerator.shape,np.nan,dtype=np.float64)
    np.divide(numerator,denominator,out=values,where=denominator>0)
    if not np.isfinite(numerator).all() or not np.isfinite(values[denominator>0]).all():
        raise ValueError('nonfinite supported FP64 state score')
    return dict(scores=values,numerator=numerator,denominator=denominator,active=t['active'])


def complete_scores(soft, mask, coding, check=lambda:None):
    """Yield every mature state once; caller saves each chunk before proceeding."""
    terms = hinge_terms(soft,mask)
    for start in range(4,packet.N,STATE_CHUNK):
        check()
        stop = min(start+STATE_CHUNK,packet.N)
        value = score_states(terms,np.arange(start,stop,dtype=np.int64),coding)
        check()
        yield start,stop,value['scores']


def startup_scores(soft, mask, coding):
    result = score_states(hinge_terms(soft,mask),np.arange(4,dtype=np.int64),coding)
    result['partial'] = result.pop('scores')
    result['states'] = np.arange(4,dtype=np.int64)
    return result


def _observation(value, valid, lane, role, coding):
    known = valid & np.isfinite(value) & (value != 0)
    bits = (value>0).astype(np.int8).tolist()
    row = dict(lane=lane,role=role,physical_copy=role!='F',raw_bits=bits,
               known=known.tolist(),erasures=int((~known).sum()),candidate_n=None,
               corrected_positions=[],msg=None)
    if not known.any(): row['status']='ERASURE'; return row
    if not known.all(): row['status']='PARTIAL_ERASURE_UNRESOLVED'; return row
    decoded = packet.decode_raw24(bits)
    row.update(decoded)
    if decoded['msg'] is None: return row
    if coding == 'NEW': row.update(packet.parse_raw24(decoded['msg'],lane))
    else:
        j=decoded['msg']; violations=[]
        if not 0<=j<packet.N: violations.append('J_OUT_OF_DOMAIN')
        if j%5 != lane: violations.append('LANE_MISMATCH')
        row.update(j=j,violations=violations,status='INVALID_PACKET' if violations else 'LEGAL_OLD_J_CANDIDATE',
                   candidate_j=None if violations else j)
    return row


def diagnostics(soft, mask, coding, check=lambda:None):
    soft, mask = inputs(soft,mask)
    if coding not in ('OLD','NEW'): raise ValueError('explicit packet semantics required')
    nodes=[]
    for r in range(len(soft)):
        check();valid=mask[r]&np.isfinite(soft[r]);merged=np.where(valid,soft[r],0.).sum(axis=1)
        lanes=[];constraints=[];observations=[]
        for lane in range(5):
            rows=[_observation(soft[r,lane,k],valid[lane,k],lane,ROLES[k],coding) for k in range(2)]
            rows.append(_observation(merged[lane],valid[lane].any(axis=0),lane,'F',coding))
            observations.extend(rows)
            key='candidate_n' if coding=='NEW' else 'candidate_j'
            candidates=[x.get(key) for x in rows]
            union=sorted({v for v in candidates if v is not None})
            physical={v for v in candidates[:2] if v is not None}
            if union and coding=='NEW': constraints.append(set(union))
            lanes.append(dict(lane=lane,candidate_union=union,roles=dict(zip(ROLES,candidates)),
                              copy_conflict=len(physical)>1,
                              fused_conflict=candidates[2] is not None and any(v!=candidates[2] for v in physical)))
        compatible=sorted(set.intersection(*constraints)) if constraints else None
        if coding=='OLD': status='OLD_LOCAL_J_DIAGNOSTIC_ONLY'
        else:
            status=('NO_VALID_LOCAL_CANDIDATES' if compatible is None else 'CONSTRAINT_EMPTY_SET' if not compatible
                    else 'STRUCTURAL_SINGLETON' if len(compatible)==1 else 'STRUCTURAL_AMBIGUITY')
        nodes.append(dict(received_frame=r,status=status,observations=observations,lanes=lanes,
                          compatible_n=compatible,constraint_count=len(constraints),formal_output=None))
    edges=[]
    if coding=='NEW':
        for r in range(len(nodes)-1):
            left,right=nodes[r]['compatible_n'],nodes[r+1]['compatible_n']
            pairs=[dict(left=a,right=b,delta=b-a,relation='INVERSION' if b<a else 'REPEAT' if b==a else 'NORMAL' if b==a+1 else 'GAP')
                   for a in (left or []) for b in (right or [])]
            if len(pairs)>9: raise ValueError('local t7 three-role endpoint bound exceeded')
            singleton=left is not None and right is not None and len(left)==len(right)==1
            status=('UNRESOLVED' if not pairs else 'INVERSION' if singleton and pairs[0]['delta']<0
                    else 'UNCALIBRATED_EXACT_'+pairs[0]['relation'] if singleton else 'AMBIGUOUS')
            edges.append(dict(received_edge=r,left_candidates=left,right_candidates=right,pairs=pairs,status=status,
                              formal_output=None))
    observations=[o for node in nodes for o in node['observations']]
    return dict(coding=coding,nodes=nodes,edges=edges,formal_output=None,
                logical_counts=dict(planned_role_packets=len(soft)*15,
                    raw24_decode=sum(all(o['known']) for o in observations),
                    packet_parse=sum(o['msg'] is not None for o in observations)),
                F_is_derived_not_third_copy=True,hard_candidates_do_not_restrict_soft_domain=True)
