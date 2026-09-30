"""Overlapping real-spatial tube states and finite-family uncalibrated inference."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from copy import deepcopy
import numpy as np


@dataclass(frozen=True)
class Protocol:
    source_length: int = 45
    channels: int = 16
    height: int = 40
    width: int = 64
    pilot_channel: int = 4
    blocks: tuple = ((8,12,12,16),(8,12,44,48),(28,32,12,16),(28,32,44,48))
    alpha: float = 1/np.sqrt(1392)
    tie_atol: float = 1e-12
    observed_lengths: tuple = (30,31,32,44,45,46)


PUBLIC=Protocol()


def keyed(domain,key,*indices):
    if not isinstance(key,str) or '\0' in key:raise ValueError('public key must be NUL-free text')
    return hashlib.sha256('\0'.join([domain,key,*map(str,indices)]).encode()).digest()


@lru_cache(maxsize=8)
def state_code(key):
    order=sorted(range(8),key=lambda j:(keyed('VTOS1/state/order',key,j),j))
    signs=np.asarray([2*(keyed('VTOS1/state/sign',key,j)[0]&1)-1 for j in range(8)],dtype=np.int8)
    states=[]
    for t in range(1,46):
        b=[((t-1)>>(5-j))&1 for j in range(6)]
        c=b+[sum(b)%2,(b[0]+b[2]+b[4])%2]
        states.append(signs*np.asarray([2*c[j]-1 for j in order],dtype=np.int8))
    a=np.asarray(states,dtype=np.int8);a.flags.writeable=False;return a


@lru_cache(maxsize=8)
def bases(key):
    H=np.asarray([[(-1)**((r&c).bit_count())/4 for c in range(16)] for r in range(16)],dtype=np.float64)
    result=[]
    for i in range(4):
        order=sorted(range(16),key=lambda c:(keyed('VTOS1/basis/order',key,i,c),c))
        signs=np.asarray([2*(keyed('VTOS1/basis/sign',key,i,q)[0]&1)-1 for q in range(8)])
        result.append(H[:,order[:8]]*signs[None,:])
    a=np.asarray(result);a.flags.writeable=False;return a


def synthesize(key,public=PUBLIC,dtype=np.float64):
    """Scatter every overlapping tube into one full physical latent tensor."""
    if public!=PUBLIC:raise ValueError('frozen public protocol required')
    target=np.zeros((1,16,46,40,64),dtype=dtype)
    S=state_code(key);U=bases(key).astype(dtype);alpha=dtype(public.alpha)
    for start in range(1,46):
        for age in range(4):
            u=start+age
            if u>45:continue
            for i,(h0,h1,w0,w1) in enumerate(public.blocks):
                contribution=U[i,:,2*age:2*age+2]@S[start-1,2*i:2*i+2].astype(dtype)
                target[0,4,u,h0:h1,w0:w1]+=alpha*contribution.reshape(4,4)
    return target


@lru_cache(maxsize=8)
def composite_signs(key):
    S=state_code(key);c=np.zeros((45,4,4,2),dtype=np.int8)
    for u in range(1,46):
        for age in range(4):
            if u-age>=1:c[u-1,:,age,:]=S[u-age-1].reshape(4,2)
    c.flags.writeable=False;return c


def extract(received_tensor,key,availability,public=PUBLIC):
    if public!=PUBLIC:raise ValueError('frozen public protocol required')
    Y=np.asarray(received_tensor)
    if Y.ndim!=4 or Y.shape[1:]!=(16,40,64) or Y.shape[0] not in public.observed_lengths:
        raise ValueError('received tensor must use frozen length and whole-frame latent geometry')
    mask=np.asarray(availability)
    if mask.dtype!=np.bool_ or mask.shape!=(len(Y),4):raise ValueError('boolean received-coordinate mask required')
    values=np.zeros((len(Y),4,4,2),dtype=np.float64)
    for i,(h0,h1,w0,w1) in enumerate(public.blocks):
        selected=Y[mask[:,i],4,h0:h1,w0:w1].reshape(-1,16)
        if not np.isfinite(selected).all():raise ValueError('nonfinite available physical ROI')
        projected=selected.astype(np.float64)@bases(key)[i]
        if not np.isfinite(projected).all():raise ValueError('nonfinite projection')
        values[mask[:,i],i]=projected.reshape(-1,4,2)
    return values


@lru_cache(maxsize=6)
def _catalog(length):
    if length not in PUBLIC.observed_lengths:raise ValueError('length outside frozen structural family')
    rows=[]
    for start in range(1,46):
        events=[('ZERO_EDIT',None,0)]+[(kind,i,d) for kind,d in [('REPEAT',-1),('SKIP',1)] for i in range(2,length+1)]
        for kind,edge,shift in events:
            ts=[start+j+(shift if edge is not None and j+1>=edge else 0) for j in range(length)]
            valid=min(ts)>=1 and max(ts)<=45
            rows.append(dict(id=f't{start:02d}:{kind}:{edge or 0:02d}',tau1=start,event_type=kind,event_i=edge,
                taus=ts,structurally_valid=valid,status='VALID' if valid else 'STRUCTURALLY_EXCLUDED',
                exclusion=None if valid else 'FINITE_SOURCE_SUPPORT'))
    return tuple(rows)


@lru_cache(maxsize=48)
def _family(key,length,mask_bytes):
    mask=np.frombuffer(mask_bytes,dtype=np.bool_).reshape(length,4)
    mask4=np.broadcast_to(mask[:,:,None,None],(length,4,4,2))
    rows=_catalog(length);valid_indices=[i for i,r in enumerate(rows) if r['structurally_valid']]
    paths=np.asarray([rows[i]['taus'] for i in valid_indices],dtype=np.int64)-1
    templates=composite_signs(key)[paths]
    signatures=templates[:,mask4]
    groups={}
    for j,values in enumerate(signatures):groups.setdefault(values.tobytes(),[]).append(j)
    classes=[];class_signs=[];membership=np.empty(len(valid_indices),dtype=np.int64)
    for ci,(signature,members) in enumerate(groups.items()):
        indices=[valid_indices[j] for j in members]
        ts=[rows[j]['taus'] for j in indices]
        possible=[sorted({t[i] for t in ts}) for i in range(length)]
        canonical=min(indices,key=lambda j:(tuple(rows[j]['taus']),rows[j]['event_type'],rows[j]['event_i'] or 0))
        classes.append(dict(index=ci,signature_sha256=hashlib.sha256(signature).hexdigest(),member_catalog_indices=indices,
            canonical_catalog_index=canonical,tau_feasible_sets=possible,
            contains_zero_edit=any(rows[j]['event_type']=='ZERO_EDIT' for j in indices)))
        class_signs.append(signatures[members[0]])
        membership[members]=ci
    signs=np.asarray(class_signs,dtype=np.int8).reshape(len(classes),int(mask4.sum()))
    means=signs.astype(np.float64)*PUBLIC.alpha
    return dict(classes=classes,valid_indices=valid_indices,membership=membership,means=means,mask4=mask4,
                available_dimensions=int(mask4.sum()),length=length,key_sha256=hashlib.sha256(key.encode()).hexdigest())


def _get_family(key,length,availability):
    mask=np.asarray(availability)
    if mask.dtype!=np.bool_ or mask.shape!=(length,4):raise ValueError('boolean public availability mask required')
    return _family(key,length,mask.tobytes())


def family_receipt(key,length,availability):
    f=_get_family(key,length,availability)
    return dict(length=length,key_sha256=f['key_sha256'],availability=np.asarray(availability).tolist(),
                available_dimensions=f['available_dimensions'],valid_catalog_indices=list(f['valid_indices']),classes=deepcopy(f['classes']),
                grouping='exact masked integer composite templates; unavailable indices remain in paths')


def _empty_summary(reason):
    return dict(status='INCOMPLETE',reason=reason,score_status='UNCALIBRATED_DIAGNOSTIC',accepted_payload=False,
        state_path_accepted=False,canonical_catalog_index=None,top_class_indices=[],top_catalog_indices=[],
        min_cost=None,zero_edit_min_cost=None,zero_edit_minus_best=None,unique_model_hypothesis=False,
        unique_is_not_confidence=True)


def infer(received_tensor,key,availability,public=PUBLIC):
    """Only actual received tensor, key and received mask enter; no writer/attack truth."""
    if public!=PUBLIC:raise ValueError('frozen public protocol required')
    length=len(received_tensor);f=_get_family(key,length,availability);rows=_catalog(length)
    summary=_empty_summary('NOT_EVALUATED')
    base=dict(summary=summary,available_dimensions=f['available_dimensions'],received_length=length,
              valid_catalog_indices=list(f['valid_indices']),class_costs=[None]*len(f['classes']),
              path_costs=[None]*len(f['valid_indices']),projection=None,
              counts=dict(catalog=len(rows),scorable=len(f['valid_indices']),structurally_excluded=len(rows)-len(f['valid_indices']),
                          equivalence_classes=len(f['classes']),scored=0))
    try:
        r=extract(received_tensor,key,availability,public)
        if f['available_dimensions']==0:
            summary['reason']='NO_OBSERVATIONS';return base
        obs=r[f['mask4']]
        # Class means share M. Direct residual avoids cancellation at exact zero error.
        with np.errstate(over='raise',invalid='raise'):
            residual=f['means']-obs[None,:]
            costs=np.sum(residual*residual,axis=1,dtype=np.float64)/f['available_dimensions']
            energy=float(np.sum(obs*obs,dtype=np.float64))
        if not np.isfinite(costs).all() or not np.isfinite(energy):raise ValueError('nonfinite score reduction')
        minimum=float(costs.min());top_classes=np.flatnonzero(costs-minimum<=public.tie_atol).tolist()
        top=sorted(i for ci in top_classes for i in f['classes'][ci]['member_catalog_indices'])
        zero_indices=[j for j,i in enumerate(f['valid_indices']) if rows[i]['event_type']=='ZERO_EDIT']
        path_cost=costs[f['membership']]
        zero_min=float(path_cost[zero_indices].min()) if zero_indices else None
        canonical=min(top,key=lambda i:(tuple(rows[i]['taus']),rows[i]['event_type'],rows[i]['event_i'] or 0))
        structural=any(len(f['classes'][i]['member_catalog_indices'])>1 for i in top_classes)
        reason='NO_ENERGY' if energy==0 else ('STRUCTURAL_AMBIGUITY' if structural else 'NUMERICAL_CLASS_TIE' if len(top_classes)>1 else 'UNIQUE_FINITE_MODEL_ONLY')
        summary.update(status='NO_ENERGY' if energy==0 else 'COMPLETE',reason=reason,canonical_catalog_index=None if energy==0 else canonical,
          top_class_indices=top_classes,top_catalog_indices=top,min_cost=minimum,zero_edit_min_cost=zero_min,
          zero_edit_minus_best=None if zero_min is None else zero_min-minimum,
          unique_model_hypothesis=energy>0 and len(top)==1,structural_ambiguity=structural,
          any_top_contains_zero_edit=any(rows[i]['event_type']=='ZERO_EDIT' for i in top),
          all_top_require_event=all(rows[i]['event_type']!='ZERO_EDIT' for i in top),observed_projection_energy=energy,
          top_distinct_class_gap=float(np.partition(costs,1)[1]-minimum) if len(costs)>1 else None)
        base.update(class_costs=costs.tolist(),path_costs=path_cost.tolist(),projection=r.tolist())
        base['counts']['scored']=len(path_cost)
    except (ValueError,TypeError,FloatingPointError,OverflowError) as exc:
        summary.update(reason='INVALID_OBSERVATION',error=f'{type(exc).__name__}: {exc}')
    return base


def catalog(length):
    """Public metadata is detached from the private inference cache."""
    return deepcopy(_catalog(length))


def family(key,length,availability):
    """Inspection copy; production inference uses its private cached arrays."""
    return deepcopy(_get_family(key,length,availability))
