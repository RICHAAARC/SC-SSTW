"""Pure received-only traces. Reuses the public hard receiver; never selects a role."""
import numpy as np
from . import b_framewise_qim_time_recovery_v1 as hard

ROLES=('F','R0','R1')


def role_mask(mask,role):
    if role not in ROLES:raise ValueError('fixed F/R0/R1 role required')
    result=mask.copy()
    if role!='F':result[:,:,1-int(role[1]),:]=False
    return result


def arithmetic(raw):
    q=raw['q'];distances=[]
    for bit in (0,1):
        low=np.floor((q-16*bit)/32)*32+16*bit
        center=np.where(q-low<=16,low,low+32)
        distances.append(np.abs(q-center))
    d0,d1=distances;soft=(d0*d0-d1*d1)/256
    if not all(np.isfinite(a).all() for a in (d0,d1,soft)):raise FloatingPointError('nonfinite QIM recomputation')
    arrays={};summary={}
    for key,value in dict(d0=d0,d1=d1,soft=soft).items():
        delta=value-raw[key]
        if not np.isfinite(delta).all():raise FloatingPointError('nonfinite arithmetic difference')
        arrays.update({key+'_recomputed':value,key+'_equal':value==raw[key],key+'_delta':delta})
        summary[key]=dict(exact_equal=bool(np.array_equal(value,raw[key])),different=int(np.count_nonzero(delta)),max_abs_difference=float(np.abs(delta).max()))
    arrays['raw_sign_changed']=(soft>0)!=(raw['soft']>0)
    arrays['raw_known_changed']=(raw['mask']&(soft!=0))!=(raw['mask']&(raw['soft']!=0))
    for role in ROLES:
        mask=role_mask(raw['mask'],role)
        saved,known=hard.fuse(raw['soft'],mask);new,new_known=hard.fuse(soft,mask)
        arrays[role+'_sign_changed']=(saved>0)!=(new>0)
        arrays[role+'_known_changed']=known!=new_known
        summary[role]=dict(sign_changes=int(np.count_nonzero(arrays[role+'_sign_changed'])),known_changes=int(np.count_nonzero(known!=new_known)))
    return arrays,summary


def fusion(raw):
    result={}
    for role in ROLES:
        soft,known=hard.fuse(raw['soft'],role_mask(raw['mask'],role))
        result[role]=dict(soft=soft,known=known,bits=soft>0,erasures=63-known.sum(axis=2))
    return result


def compatibility(raw,role,table,check=lambda:None):
    result,merged,known=hard.compatibility(raw['soft'],role_mask(raw['mask'],role),table,check)
    weights=np.uint64(1)<<np.arange(62,-1,-1,dtype=np.uint64)
    for r,frame in enumerate(result['labels']):
        for lane,row in enumerate(frame):
            check();observed=np.sum(weights*known[r,lane],dtype=np.uint64)
            word=np.sum(weights*(merged[r,lane]>0),dtype=np.uint64)
            distances=[]
            js,codes=table[lane]
            for j in row['labels']:
                idx=int(np.searchsorted(js,j))
                if idx>=len(js) or int(js[idx])!=j:raise ValueError('compatible label absent from shared public codebook')
                e=int(((int(codes[idx])^int(word))&int(observed)).bit_count())
                distances.append(dict(j=j,e=e,u=row['erasures'],two_e_plus_u=2*e+row['erasures']))
            row['hit_distances']=distances
    return result


def differences(new,old):
    if old is None:return dict(status='UNKNOWN_SAVED_RECORD',different=None,recomputed=new,saved=None)
    return dict(status='DIFFERENT' if new!=old else 'EQUAL',different=new!=old,recomputed=new,saved=old)


def path_trace(compatible,saved):
    result=hard.interval_paths(compatible['labels']);result.update(compatible)
    previous=saved.get('semantic.json',{}).get('hard',saved.get('blind.json'))
    if not isinstance(previous,dict):previous={}
    def at(rows,i):return rows[i] if isinstance(rows,list) and i<len(rows) else None
    lanes=[]
    for r,frame in enumerate(compatible['labels']):
        old_frame=at(previous.get('labels'),r)
        for lane,row in enumerate(frame):
            new={k:v for k,v in row.items() if k!='hit_distances'}
            lanes.append(dict(received_frame=r,lane=lane,**differences(new,at(old_frame,lane))))
    return dict(recomputed=result,saved_hard=previous,lane_comparison=lanes,
        node_comparison=[dict(received_frame=r,**differences(row,at(previous.get('nodes'),r))) for r,row in enumerate(result['nodes'])],
        edge_comparison=[dict(received_left=r,**differences(row,at(previous.get('relations'),r))) for r,row in enumerate(result['relations'])])


def compare_saved_fusion(trace,saved):
    try:
        soft=np.asarray(saved['fused.npz']['soft']);known=np.asarray(saved['fused.npz']['known'])
        if soft.shape!=trace['soft'].shape or known.shape!=trace['known'].shape or known.dtype!=np.bool_:
            raise ValueError('saved fused array geometry/type mismatch')
        if not np.isfinite(soft).all():raise ValueError('nonfinite saved fused soft')
        return dict(status='COMPARED',soft_equal=bool(np.array_equal(soft,trace['soft'])),known_equal=bool(np.array_equal(known,trace['known'])),
            per_lane_soft_differences=np.count_nonzero(soft!=trace['soft'],axis=2).tolist(),
            per_lane_known_differences=np.count_nonzero(known!=trace['known'],axis=2).tolist())
    except (KeyError,ValueError,TypeError) as error:return dict(status='UNKNOWN',error=str(error))
