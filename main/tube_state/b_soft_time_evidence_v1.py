"""C/L/G node diagnostics conditional on an existing received-only hard graph.

No new formal output or edge/event estimate. No truth, editing label or source length.
"""
import numpy as np
from .b_framewise_qim_time_recovery_v1 import DOMAIN, BLOCK, codebook as public_codebook

EPSILON=1e-12


def codebook(check=lambda:None):
    table=np.empty(DOMAIN,dtype=np.uint64)
    for js,words in public_codebook(check):table[js.astype(np.int64)]=words
    return table


def graph_input(saved,received_length):
    """Extract only public old graph fields; preserve original separately at IO boundary."""
    if not isinstance(saved,dict) or not isinstance(saved.get('nodes'),list) or len(saved['nodes'])!=received_length:
        raise ValueError('saved graph node count must match actual received raw length')
    path=saved.get('path_status');qualification=saved.get('qualification')
    if path not in ('COMPATIBLE_PATH','PARTIAL_PATH_WITH_LOCAL_CONFLICT','NO_COMPATIBLE_PATH'):
        raise ValueError('unknown old graph path status')
    if qualification not in ('SUPPORTED_RELATION_SET','INSUFFICIENT_MULTILANE_EVIDENCE','IMPLEMENTATION_UNRESOLVED'):
        raise ValueError('unknown old qualification')
    statuses=[];bounds=[]
    abnormal=qualification=='IMPLEMENTATION_UNRESOLVED'
    for r,node in enumerate(saved['nodes']):
        if not isinstance(node,dict) or node.get('received_frame')!=r:raise ValueError('invalid old node coordinate')
        status=node.get('status')
        if status not in ('LOCAL_LABEL','LOCAL_CONFLICT','OBSERVATION_UNSUPPORTED','IMPLEMENTATION_UNRESOLVED'):
            raise ValueError('invalid old observation status')
        abnormal |= status=='IMPLEMENTATION_UNRESOLVED'
        interval=node.get('interval')
        if path=='NO_COMPATIBLE_PATH':
            if interval is not None:raise ValueError('infeasible old graph has a feasible interval')
            interval=[-1,-1]
        elif not isinstance(interval,list) or len(interval)!=2 or any(type(n)!=int for n in interval) or not 0<=interval[0]<=interval[1]<DOMAIN:
            raise ValueError('invalid saved public candidate interval')
        bounds.append(interval);statuses.append(status)
    if not received_length:raise ValueError('received length must be positive')
    if path!='NO_COMPATIBLE_PATH' and any(bounds[r][k]>bounds[r+1][k] for r in range(received_length-1) for k in (0,1)):
        raise ValueError('saved exact nondecreasing graph marginals must have nondecreasing bounds')
    return dict(bounds=np.asarray(bounds,dtype=np.int64),statuses=statuses,path_status=path,
                qualification=qualification,abnormal=abnormal)


def costs(soft,mask,graph,table,check=lambda:None):
    soft=np.asarray(soft);mask=np.asarray(mask)
    if soft.ndim!=4 or soft.shape[1:]!=(5,2,63) or mask.shape!=soft.shape or mask.dtype!=np.bool_:
        raise ValueError('soft/mask require [R,5,2,63] with boolean mask')
    if soft.dtype.kind!='f':raise ValueError('soft must be floating point')
    if len(graph['bounds'])!=len(soft):raise ValueError('graph/raw received length mismatch')
    soft=soft.astype(np.float64,copy=False)
    bounds=graph['bounds'];chunks=[];states=[];support=[];comparisons=0
    shifts=np.arange(62,-1,-1,dtype=np.uint64)
    for r,(lo,hi) in enumerate(bounds):
        check();valid=mask[r]&np.isfinite(soft[r]);copies=valid.sum(axis=1)
        count=int((copies>0).sum());support.append(count)
        if graph['path_status']=='NO_COMPATIBLE_PATH':state='NO_COMPATIBLE_PATH'
        elif graph['abnormal']:state='OLD_GRAPH_ABNORMAL'
        elif graph['statuses'][r]=='LOCAL_CONFLICT':state='LOCAL_CONFLICT'
        elif lo<4:state='STARTUP_MODEL_UNDEFINED'
        elif not count:state='NO_SOFT_OBSERVATION'
        else:state='DEFINED'
        states.append(state)
        if lo<0:chunks.append(np.empty(0,dtype=np.float64));continue
        values=np.zeros(hi-lo+1,dtype=np.float64)
        if state!='DEFINED':chunks.append(values);continue
        # Per-replica hinge first; mean valid replicas per bit, then fixed-support bit mean.
        v=np.where(valid,soft[r],0.)
        divisor=np.maximum(copies,1)
        loss_one=np.maximum(0.,-v).sum(axis=1)/divisor
        loss_zero=np.maximum(0.,v).sum(axis=1)/divisor
        if not np.isfinite(loss_one).all() or not np.isfinite(loss_zero).all():raise ValueError('nonfinite FP64 replica loss')
        # Each public j contributes its own lane exactly once; n bundles only look up five j costs.
        first=int(lo)-4;last=int(hi);j_cost=np.empty(last-first+1,dtype=np.float64)
        for start in range(first,last+1,BLOCK):
            check();js=np.arange(start,min(last+1,start+BLOCK),dtype=np.int64);lanes=js%5
            bits=((table[js,None]>>shifts)&np.uint64(1)).astype(bool)
            losses=np.where(bits,loss_one[lanes],loss_zero[lanes])
            j_cost[js-first]=losses.sum(axis=1)/count;comparisons+=len(js)
        ns=np.arange(lo,hi+1,dtype=np.int64)
        for lane in range(5):values+=j_cost[ns-(ns-lane)%5-first]
        if not np.isfinite(values).all():raise ValueError('nonfinite FP64 C')
        chunks.append(values)
    offsets=np.concatenate(([0],np.cumsum([len(x) for x in chunks],dtype=np.int64)))
    return dict(C=np.concatenate(chunks),offsets=offsets,bounds=bounds,states=states,
                support=np.asarray(support),comparisons=comparisons,R=len(soft),graph_abnormal=graph['abnormal'])


def local_rank(cost):
    d=np.empty_like(cost['C']);best=np.zeros(len(d),dtype=bool);minimum=[];status=[]
    for r,(a,b) in enumerate(zip(cost['offsets'][:-1],cost['offsets'][1:])):
        values=cost['C'][a:b]
        if not len(values):minimum.append(None);status.append(cost['states'][r]);continue
        low=float(values.min());delta=values-low;d[a:b]=delta;best[a:b]=delta<=EPSILON;minimum.append(low)
        status.append(cost['states'][r] if cost['states'][r]!='DEFINED' else 'SOFT_MODEL_BEST' if int(best[a:b].sum())==1 else 'MODEL_TIE')
    return dict(dL=d,best=best,minimum=minimum,status=status)


def global_rank(cost,local,check=lambda:None):
    """Sparse interval F/B, each containing current c. No joint path chosen."""
    R=cost['R'];offsets=cost['offsets'];bounds=cost['bounds'];C=cost['C']
    empty=any(a==b for a,b in zip(offsets[:-1],offsets[1:])) or cost['graph_abnormal']
    if empty:
        return dict(Phi=np.full_like(C,np.inf),delta=np.full_like(C,np.inf),dG_C=np.full_like(C,np.inf),
                    native=np.zeros(len(C),bool),aligned=np.zeros(len(C),bool),J_star=None,
                    status=[s if s!='DEFINED' else 'OLD_GRAPH_UNRESOLVED' for s in cost['states']],
                    native_status=[s if s!='DEFINED' else 'OLD_GRAPH_UNRESOLVED' for s in cost['states']],
                    attribution=['OLD_GRAPH_UNRESOLVED']*R,cost_effect=[False]*R,arithmetic_envelope_C=None,negative_residual_count=0)
    F=np.empty_like(C);B=np.empty_like(C)
    for r,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        check();c=C[a:b]/R
        if not r:F[a:b]=c;continue
        p,q=offsets[r-1:r+1];prefix=np.minimum.accumulate(F[p:q]);lo,hi=bounds[r]
        indexes=np.arange(lo,hi+1)-bounds[r-1,0]
        previous=np.where(indexes>=0,prefix[np.clip(indexes,0,len(prefix)-1)],np.inf)
        F[a:b]=c+previous
    for r in range(R-1,-1,-1):
        check();a,b=offsets[r:r+2];c=C[a:b]/R
        if r==R-1:B[a:b]=c;continue
        p,q=offsets[r+1:r+3];suffix=np.minimum.accumulate(B[p:q][::-1])[::-1];lo,hi=bounds[r]
        indexes=np.arange(lo,hi+1)-bounds[r+1,0]
        following=np.where(indexes<len(suffix),suffix[np.clip(indexes,0,len(suffix)-1)],np.inf)
        B[a:b]=c+following
    J=float(F[offsets[-2]:].min())
    if not np.isfinite(J):raise ValueError('old candidate graph has no finite global path')
    Phi=F+B-C/R;delta=Phi-J;aligned_delta=R*delta
    if not np.isfinite(Phi).all() or not np.isfinite(aligned_delta).all():raise ValueError('nonfinite min-marginal arithmetic on feasible old intervals')
    native=delta<=EPSILON;aligned=aligned_delta<=EPSILON;status=[];native_status=[];attribution=[];cost_effect=[]
    # A conservative arithmetic diagnostic conditional on the shared stored C, not a new ranking threshold.
    # Each nonnegative path has <= R additions/divisions; F+B-c-J plus unit conversion
    # is covered by gamma_(8R+32)*R*max(C). Min itself introduces no arithmetic rounding.
    # Use machine eps (twice unit roundoff) for extra slack; no second graph solve.
    eps=np.finfo(np.float64).eps;steps=8*R+32;maximum=float(np.max(np.abs(C)))
    envelope=(steps*eps/(1-steps*eps))*R*maximum
    local_envelope=(4*eps/(1-4*eps))*maximum
    for r,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        prior=cost['states'][r]
        status.append(prior if prior!='DEFINED' else 'SOFT_MODEL_BEST' if int(aligned[a:b].sum())==1 else 'MODEL_TIE')
        native_status.append(prior if prior!='DEFINED' else 'SOFT_MODEL_BEST' if int(native[a:b].sum())==1 else 'MODEL_TIE')
        changed=not np.array_equal(local['best'][a:b],aligned[a:b])
        effect=bool(np.any(np.abs(aligned_delta[a:b]-local['dL'][a:b])>envelope+local_envelope));cost_effect.append(effect)
        robust=False
        if changed and prior=='DEFINED' and int(aligned[a:b].sum())==1:
            selected=int(np.flatnonzero(aligned[a:b])[0]);others=np.delete(aligned_delta[a:b],selected)
            gap=float(others.min()-aligned_delta[a+selected]) if len(others) else np.inf
            stable_local=bool(np.all(np.abs(local['dL'][a:b]-EPSILON)>local_envelope))
            robust=effect and stable_local and gap>2*envelope+EPSILON
        attribution.append('CONSTRAINT_RANKING_DIFFERENCE_AT_STORED_C' if robust else 'NUMERIC_ATTRIBUTION_UNRESOLVED' if changed else 'NO_SET_CHANGE')
    return dict(Phi=Phi,delta=delta,dG_C=aligned_delta,native=native,aligned=aligned,J_star=J,
                status=status,native_status=native_status,attribution=attribution,cost_effect=cost_effect,
                arithmetic_envelope_C=float(envelope) if np.isfinite(envelope) else None,
                negative_residual_count=int((delta<0).sum()))


def row_summary(cost,local,global_result):
    rows=[]
    for r,(a,b) in enumerate(zip(cost['offsets'][:-1],cost['offsets'][1:])):
        lo,hi=cost['bounds'][r];item=dict(received_frame=r,interval=[int(lo),int(hi)] if lo>=0 else None,
            model_status=cost['states'][r],valid_lane_bits=int(cost['support'][r]),C_min=local['minimum'][r],
            L_status=local['status'][r],G_native_status=global_result['native_status'][r],G_aligned_status=global_result['status'][r],
            attribution=global_result['attribution'][r],native_aligned_differ=not np.array_equal(global_result['native'][a:b],global_result['aligned'][a:b]))
        item['constraint_cost_effect_at_stored_C']=global_result['cost_effect'][r]
        item['L_G_native_same_set']=bool(np.array_equal(local['best'][a:b],global_result['native'][a:b]))
        item['L_G_aligned_same_set']=bool(np.array_equal(local['best'][a:b],global_result['aligned'][a:b]))
        for name,key,values in (('L','best',local),('G_native','native',global_result),('G_aligned','aligned',global_result)):
            mask=values[key][a:b];item[name+'_count']=int(mask.sum())
            item[name+'_model_index']=int(lo+np.flatnonzero(mask)[0]) if cost['states'][r]=='DEFINED' and mask.sum()==1 else None
        rows.append(item)
    return dict(nodes=rows,J_star=global_result['J_star'],epsilon=EPSILON,R=cost['R'],
                negative_residual_count=global_result['negative_residual_count'],arithmetic_envelope_C=global_result['arithmetic_envelope_C'],formal_new_determined=0,
                graph_output='NODE_MIN_MARGINALS_ONLY_NOT_A_JOINT_PATH',new_relation_outputs=0,
                numeric_policy='One FP64 solve. Shared-C arithmetic envelope only annotates robust unique ranking differences; all sets use the unchanged epsilon. Unresolved differences remain visible, not graph gains.')
