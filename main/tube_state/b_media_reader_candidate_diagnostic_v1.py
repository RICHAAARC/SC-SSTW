"""Reader comparisons and conditional interval sets; no truth, IO or formal output."""
import numpy as np
from .b_framewise_qim_time_recovery_v1 import DOMAIN,fuse

ROLES=('F','R0','R1')


def normalize(intervals):
    result=[]
    for lo,hi in sorted(intervals):
        if type(lo) is not int or type(hi) is not int or not 0<=lo<=hi<DOMAIN:raise ValueError('invalid public interval')
        if result and lo<=result[-1][1]+1:result[-1][1]=max(result[-1][1],hi)
        else:result.append([lo,hi])
    return result


def intersect(a,b):
    return normalize([[max(x,u),min(y,v)] for x,y in a for u,v in b if max(x,u)<=min(y,v)])


def width(intervals):return None if intervals is None else sum(b-a+1 for a,b in intervals)
def contains(intervals,n):return None if intervals is None or type(n) is not int else any(a<=n<=b for a,b in intervals)
def subset(a,b):return None if a is None or b is None else intersect(a,b)==a


def _message(bundle,role,r,lane):
    try:
        value=bundle[role]
        if value.get('error') or value.get('status') in ('FAILED','INTERRUPTED','RESOURCE_UNRESOLVED'):raise ValueError('failed compatible record')
        row=value['labels'][r][lane];labels=row['labels']
        if type(row.get('lane')) is not int or row['lane']!=lane or not isinstance(labels,list):raise ValueError('lane/message shape')
        if len(labels)>1:raise ValueError('multiple hits remain IMPLEMENTATION_UNRESOLVED')
        if any(type(j) is not int or not 0<=j<DOMAIN or j%5!=lane for j in labels):raise ValueError('invalid public lane label')
        if row.get('status')!=('COMPATIBLE' if labels else 'NO_COMPATIBLE_CODE'):raise ValueError('missing/invalid label status')
        if type(row.get('erasures')) is not int or not 0<=row['erasures']<=63:raise ValueError('invalid erasure count')
        return labels,None
    except (KeyError,TypeError,IndexError,AttributeError,ValueError) as error:return None,f'{role}: {error}'


def candidate_pools(bundle,frames,check=lambda:None):
    """Only complete replica label disagreement triggers one-lane hypotheses."""
    shape_errors=[]
    for role in ROLES:
        value=bundle.get(role)
        labels=value.get('labels') if isinstance(value,dict) else None
        if not isinstance(labels,list) or len(labels)!=frames or any(not isinstance(v,list) or len(v)!=5 for v in labels):
            shape_errors.append(role+' fixed frame/lane shape missing')
    rows=[]
    for r in range(frames):
        check();lanes=[];errors=list(shape_errors)
        for lane in range(5):
            messages={};faults=[]
            for role in ROLES:
                messages[role],error=_message(bundle,role,r,lane)
                if error:faults.append(error)
            errors.extend('lane'+str(lane)+': '+e for e in faults)
            lanes.append(dict(lane=lane,messages=messages,status='IMPLEMENTATION_UNRESOLVED' if faults else 'VALID',errors=faults))
        if errors:
            rows.append(dict(received_frame=r,status='IMPLEMENTATION_UNRESOLVED',lanes=lanes,E=None,C0=None,Cstar=None,
                hypotheses=[dict(lane=l,status='IMPLEMENTATION_UNRESOLVED',intervals=None) for l in range(5)],errors=errors));continue
        constraints=[[[v['messages']['F'][0],min(DOMAIN-1,v['messages']['F'][0]+4)]] if v['messages']['F'] else [[0,DOMAIN-1]] for v in lanes]
        def intersection_without(skip=None):
            value=[[0,DOMAIN-1]]
            for lane,constraint in enumerate(constraints):
                if lane!=skip:value=intersect(value,constraint)
            return value
        c0=intersection_without();trigger=[l for l,v in enumerate(lanes) if v['messages']['R0']!=v['messages']['R1']]
        hypotheses=[dict(lane=l,status='TRIGGERED' if l in trigger else 'NOT_APPLICABLE',intervals=intersection_without(l) if l in trigger else None) for l in range(5)]
        candidate=normalize(c0+[interval for h in hypotheses if h['intervals'] is not None for interval in h['intervals']])
        rows.append(dict(received_frame=r,status='NO_FEASIBLE_PATH' if not candidate else 'NO_DISCRIMINATION' if candidate==[[0,DOMAIN-1]] else 'CONDITIONAL_CANDIDATES',
            lanes=lanes,E=trigger,C0=c0,C0_status='LOCAL_CONFLICT' if not c0 else 'NO_DISCRIMINATION' if c0==[[0,DOMAIN-1]] else 'LOCAL_CONSTRAINT',
            Cstar=candidate,hypotheses=hypotheses,C0_in_Cstar=subset(c0,candidate),errors=[]))
    return dict(rows=rows,frames=frames,lane_slots=frames*5,hypothesis_slots=frames*5,formal_outputs=0)


def edge_sets(left,right):
    pairs=[];kinds=set()
    for a,b in left:
        for c,d in right:
            if a>d:continue
            pairs.append(dict(left=[a,min(b,d)],right=[max(c,a),d],relation='left<=right'))
            if max(a,c)<=min(b,d):kinds.add('REPEAT')
            if max(a,c-1)<=min(b,d-1):kinds.add('NORMAL')
            if d-a>=2:kinds.add('GAP')
    exact=width(left)==width(right)==1 and bool(pairs)
    endpoints=[left[0][0],right[0][0]] if exact else None
    kind=None if not exact else 'REPEAT' if endpoints[1]==endpoints[0] else 'NORMAL' if endpoints[1]==endpoints[0]+1 else 'GAP'
    return dict(status='SHADOW_EXACT' if exact else 'SHADOW_AMBIGUOUS' if pairs else 'NO_FEASIBLE_PATH',endpoint_pairs=pairs,
        possible_kinds=[k for k in ('REPEAT','NORMAL','GAP') if k in kinds],exact_kind=kind,exact_endpoints=endpoints,
        missing_indices=list(range(endpoints[0]+1,endpoints[1])) if exact and kind=='GAP' else None,
        repeated_index=endpoints[0] if exact and kind=='REPEAT' else None,formal_output=False)


def propagate(pools,check=lambda:None):
    """One exact all-nondecreasing propagation; never fills interval holes."""
    rows=pools['rows'];local=[row['Cstar'] for row in rows];n=len(rows)
    status='IMPLEMENTATION_UNRESOLVED' if any(s is None for s in local) else 'NO_FEASIBLE_PATH' if any(not s for s in local) else 'FEASIBLE_SHADOW_SETS'
    lower=[];upper=[]
    if status=='FEASIBLE_SHADOW_SETS':
        bound=0
        for intervals in local:
            check();allowed=[max(a,bound) for a,b in intervals if b>=bound]
            if not allowed:status='NO_FEASIBLE_PATH';break
            bound=min(allowed);lower.append(bound)
        bound=DOMAIN-1
        if status=='FEASIBLE_SHADOW_SETS':
            for intervals in reversed(local):
                check();allowed=[min(b,bound) for a,b in intervals if a<=bound]
                if not allowed:status='NO_FEASIBLE_PATH';break
                bound=max(allowed);upper.append(bound)
            upper.reverse()
    global_sets=[intersect(s,[[lo,hi]]) for s,lo,hi in zip(local,lower,upper)] if status=='FEASIBLE_SHADOW_SETS' else [None if status=='IMPLEMENTATION_UNRESOLVED' else [] for _ in rows]
    nodes=[dict(received_frame=r,status=status if s is None or not s else 'NO_DISCRIMINATION' if s==[[0,DOMAIN-1]] else 'SHADOW_SINGLETON' if width(s)==1 else 'SHADOW_AMBIGUOUS',
                intervals=s,width=width(s),shadow_index=s[0][0] if width(s)==1 else None,formal_output=False) for r,s in enumerate(global_sets)]
    edges=[]
    for r in range(n-1):
        check();value=edge_sets(global_sets[r],global_sets[r+1]) if global_sets[r] is not None and global_sets[r+1] is not None else dict(status='IMPLEMENTATION_UNRESOLVED',endpoint_pairs=None,possible_kinds=None,exact_endpoints=None,exact_kind=None,missing_indices=None,repeated_index=None,formal_output=False)
        edges.append(dict(received_left=r,**value))
    return dict(status=status,nodes=nodes,edges=edges,formal_outputs=0,claim='conditional all-path shadow candidates; singleton is not authentication')


def observation_trace(raw):
    value=dict(raw);soft=raw['soft'];known=raw['mask']&np.isfinite(soft)&(soft!=0)
    merged,fknown=fuse(soft,raw['mask'])
    value.update(known=known,bit=soft>0,boundary_margin_q=np.abs(raw['d0']-raw['d1'])/2,
                 F_soft=merged,F_known=fknown,F_bit=merged>0)
    return value


def compare_observations(left,right):
    arrays={};summary={}
    for key in ('q','d0','d1','soft','boundary_margin_q','F_soft'):
        if left[key].shape!=right[key].shape:raise ValueError('comparison geometry mismatch')
        delta=right[key]-left[key]
        if not np.isfinite(delta).all():raise ValueError('nonfinite comparison difference')
        arrays[key+'_delta']=delta
        summary[key]=dict(exact_equal=bool(np.array_equal(left[key],right[key])),different=int(np.count_nonzero(delta)),max_abs_difference=float(np.abs(delta).max()))
    low=np.minimum(left['q'],right['q']);high=np.maximum(left['q'],right['q'])
    crossings=np.maximum(0,np.ceil((high-8)/16)-np.floor((low-8)/16)-1)
    if not np.isfinite(crossings).all() or np.any(crossings>np.iinfo(np.int64).max):raise ValueError('unrepresentable QIM boundary count')
    arrays['strict_q_boundary_crossings']=crossings.astype(np.int64)
    arrays['left_on_q_boundary']=np.remainder(left['q']-8,16)==0
    arrays['right_on_q_boundary']=np.remainder(right['q']-8,16)==0
    summary['q_boundaries']=dict(coefficients_with_strict_crossing=int(np.count_nonzero(crossings)),strict_crossings=int(crossings.sum()),
        left_endpoint_on_boundary=int(arrays['left_on_q_boundary'].sum()),right_endpoint_on_boundary=int(arrays['right_on_q_boundary'].sum()))
    for prefix in ('','F_'):
        for key in ('bit','known'):
            changed=left[prefix+key]!=right[prefix+key];arrays[prefix+key+'_changed']=changed
            summary[prefix+key+'_changed']=dict(total=int(changed.sum()),per_frame=changed.reshape(len(changed),-1).sum(axis=1).tolist())
        common=left[prefix+'known']&right[prefix+'known']
        changed=common&(left[prefix+'bit']!=right[prefix+'bit']);arrays[prefix+'common_known_bit_changed']=changed
        summary[prefix+'common_known_bit_changed']=int(changed.sum())
    return arrays,summary


def compare_pixels(left,right,check=lambda:None):
    if left.shape!=right.shape:raise ValueError('RGB comparison geometry mismatch')
    rows=[]
    for r,(a,b) in enumerate(zip(left,right)):
        check();delta=b.astype(np.int16)-a.astype(np.int16)
        rows.append(dict(frame=r,different_channels=int(np.count_nonzero(delta)),different_pixels=int(np.count_nonzero(np.any(delta!=0,axis=2))),max_abs_difference=int(np.abs(delta).max())))
    return dict(exact_equal=not any(row['different_channels'] for row in rows),frames=rows)
