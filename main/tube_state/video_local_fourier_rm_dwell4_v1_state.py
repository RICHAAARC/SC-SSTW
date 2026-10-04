"""Local Fourier RM state carrier; finite blind phase/path diagnostics, not calibrated detection."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
import hashlib
from copy import deepcopy
import numpy as np


@dataclass(frozen=True)
class Protocol:
    method_version: str = "video-local-fourier-rm-dwell4-v1"
    source_length: int = 45
    channels: int = 16
    height: int = 40
    width: int = 64
    pilot_channel: int = 4
    blocks: tuple = ((8,16,12,20),(8,16,44,52),(28,36,12,20),(28,36,44,52))
    alpha: float = 1/np.sqrt(5568)
    tie_atol: float = 1e-12
    observed_lengths: tuple = (31,44,45)


PUBLIC=Protocol()


def keyed(domain,key,*indices):
    if not isinstance(key,str) or '\0' in key:raise ValueError('public key must be NUL-free text')
    return hashlib.sha256('\0'.join([domain,key,*map(str,indices)]).encode()).digest()


@lru_cache(maxsize=8)
def state_code(key):
    """45 of 62 balanced RM(1,5) words; key order never uses measured scores."""
    order=sorted([a for a in range(64) if a&31],key=lambda a:(keyed('VLFRM1/state/word',key,a),a))[:45]
    columns=sorted(range(32),key=lambda j:(keyed('VLFRM1/state/chip',key,j),j))
    out=np.asarray([[(-1)**((a>>5)+((a&31)&x).bit_count()) for x in columns] for a in order],dtype=np.int8)
    out.flags.writeable=False;return out


def fourier_modes():
    """One representative of each real cosine pair; exclude DC and (4,4)."""
    return [(h,w) for h in range(8) for w in range(8)
            if (h,w)<=((-h)%8,(-w)%8) and (h,w) not in ((0,0),(4,4))]


def basis_layout(key):
    modes=fourier_modes();rows=[]
    for i in range(4):
        order=sorted(range(32),key=lambda j:(keyed('VLFRM1/basis/order',key,i,j),j))
        signs=[2*(keyed('VLFRM1/basis/sign',key,i,q)[0]&1)-1 for q in range(32)]
        rows.append(dict(block=i,modes=[modes[j] for j in order],signs=signs,
          factors=[1. if modes[j] in ((0,4),(4,0)) else float(np.sqrt(2)) for j in order]))
    return dict(method_version=PUBLIC.method_version,blocks=rows,
      transform='8x8 fft2 ortho real, nonself conjugate coefficients scaled sqrt2',
      allocation='slot=age*8+chip_in_block; no independent-vote interpretation')


@lru_cache(maxsize=8)
def _bases(key):
    y,x=np.meshgrid(np.arange(8),np.arange(8),indexing='ij');rows=[]
    for row in basis_layout(key)['blocks']:
        columns=[sign*factor*np.cos(2*np.pi*(h*y+w*x)/8)/8
                 for (h,w),sign,factor in zip(row['modes'],row['signs'],row['factors'])]
        rows.append(np.stack(columns,axis=-1).reshape(64,32))
    out=np.asarray(rows);out.flags.writeable=False;return out


def bases(key):return _bases(key).copy()


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
                contribution=U[i,:,8*age:8*age+8]@S[(start-1)//4,8*i:8*i+8].astype(dtype)
                target[0,4,u,h0:h1,w0:w1]+=alpha*contribution.reshape(8,8)
    return target


@lru_cache(maxsize=8)
def composite_signs(key):
    S=state_code(key);c=np.zeros((45,4,4,8),dtype=np.int8)
    for u in range(1,46):
        for age in range(4):
            if u-age>=1:c[u-1,:,age,:]=S[(u-age-1)//4].reshape(4,8)
    c.flags.writeable=False;return c


def extract(received_tensor,key,availability,public=PUBLIC):
    if public!=PUBLIC:raise ValueError('frozen public protocol required')
    Y=np.asarray(received_tensor)
    if Y.ndim!=4 or Y.shape[1:]!=(16,40,64) or Y.shape[0] not in public.observed_lengths:
        raise ValueError('received tensor must use frozen length and whole-frame latent geometry')
    mask=np.asarray(availability)
    if mask.dtype!=np.bool_ or mask.shape!=(len(Y),4):raise ValueError('boolean received-coordinate mask required')
    values=np.zeros((len(Y),4,4,8),dtype=np.float64)
    for i,(h0,h1,w0,w1) in enumerate(public.blocks):
        selected=Y[mask[:,i],4,h0:h1,w0:w1].reshape(-1,64)
        if not np.isfinite(selected).all():raise ValueError('nonfinite available physical ROI')
        projected=selected.astype(np.float64)@bases(key)[i]
        if not np.isfinite(projected).all():raise ValueError('nonfinite projection')
        values[mask[:,i],i]=projected.reshape(-1,4,8)
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
    mask4=np.broadcast_to(mask[:,:,None,None],(length,4,4,8))
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
        base.update(class_costs=costs.tolist(),path_costs=path_cost.tolist(),projection=r.tolist(),
                    local_state=local_state_costs(r,key,np.asarray(availability)))
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


def local_state_costs(projection,key,availability):
    """Full 45-state soft evidence per received row; overlaps are joint features."""
    means=composite_signs(key).astype(np.float64)*PUBLIC.alpha;rows=[];scored=0
    for j,(value,mask) in enumerate(zip(projection,availability)):
        active=np.broadcast_to(mask[:,None,None],(4,4,8));n=int(active.sum())
        if n==0:
            rows.append(dict(received_regular_index=j+1,status='NO_SUPPORT',costs=[None]*45,top=[]));continue
        costs=((means[:,active]-value[active])**2).mean(axis=1)
        minimum=float(costs.min());top=(np.flatnonzero(costs-minimum<=PUBLIC.tie_atol)+1).tolist()
        rows.append(dict(received_regular_index=j+1,status='SCORED',costs=costs.tolist(),top=top,
            min_cost=minimum,available_dimensions=n,observed_energy=float((value[active]**2).sum())))
        scored+=45
    return dict(rows=rows,scored=scored,source_states=45,score_status='UNCALIBRATED_LOCAL_DIAGNOSTIC',
        state_accepted=False,likelihood_model=False)

# Single OLD8 spatial group. Temporal protocol changes templates, never extraction.
VERSION=PUBLIC.method_version
GROUPS=(PUBLIC.blocks,)
GROUP_ALPHA=PUBLIC.alpha
def extract_groups(received_tensor,key,availability):
    mask=np.asarray(availability)
    if mask.dtype!=np.bool_ or mask.shape!=(1,len(received_tensor),4):
        raise ValueError('one public OLD8 spatial group required')
    q=extract(received_tensor,key,mask[0])
    return dict(raw_groups=q[None],group_availability=mask.copy(),q=q,
                availability=mask[0].copy(),received_regular_indices=np.arange(1,len(q)+1),
                fusion='none; one physical q shared by OLD8 and DWELL4 temporal templates')
