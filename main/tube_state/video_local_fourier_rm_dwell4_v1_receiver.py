"""Experimental adjacent-window difference diagnostics; pure NumPy, not a formal core receiver."""
from __future__ import annotations
import hashlib
import numpy as np
from main.tube_state import video_local_fourier_rm_state as frozen
PUBLIC=frozen.PUBLIC
MODES=('ABSOLUTE_CONTROL','ADJACENT_DIFFERENCE')


def validate(q,availability):
    q=np.asarray(q,dtype=np.float64);mask=np.asarray(availability)
    if q.shape!=(44,4,4,8):raise ValueError('frozen44-window projection shape required')
    if mask.dtype!=np.bool_ or mask.shape!=(44,4):raise ValueError('public boolean44x4 availability required')
    mask4=np.broadcast_to(mask[:,:,None,None],q.shape)
    if not np.isfinite(q[mask4]).all():raise ValueError('nonfinite available projection')
    safe=np.where(mask4,q,0.)
    return safe,mask


def transition_pairs():
    """132 public bounded source transitions; stay, +1 and +2 only."""
    return [(u,v) for u in range(1,46) for v in range(u,min(45,u+2)+1)]


def difference_family(key,availability,*,protocol=frozen):
    mask=np.asarray(availability)
    if mask.dtype!=np.bool_ or mask.shape!=(44,4):raise ValueError('public boolean44x4 availability required')
    edge=mask[1:]&mask[:-1];mask4=np.broadcast_to(edge[:,:,None,None],(43,4,4,8))
    catalog=protocol.catalog(44);valid=[i for i,row in enumerate(catalog) if row['structurally_valid']]
    signs=protocol.composite_signs(key);paths=np.asarray([catalog[i]['taus'] for i in valid])-1
    templates=signs[paths[:,1:]]-signs[paths[:,:-1]]
    signatures=templates[:,mask4];groups={}
    for j,sig in enumerate(signatures):groups.setdefault(sig.tobytes(),[]).append(j)
    classes=[];class_signs=[];membership=np.empty(len(valid),dtype=np.int64)
    for ci,(signature,members) in enumerate(groups.items()):
        ids=[valid[j] for j in members];taus=[catalog[i]['taus'] for i in ids]
        canonical=min(ids,key=lambda i:(tuple(catalog[i]['taus']),catalog[i]['event_type'],catalog[i]['event_i'] or 0))
        classes.append(dict(index=ci,signature_sha256=hashlib.sha256(signature).hexdigest(),member_catalog_indices=ids,
            canonical_catalog_index=canonical,tau_feasible_sets=[sorted({t[j] for t in taus}) for j in range(44)],
            contains_zero_edit=any(catalog[i]['event_type']=='ZERO_EDIT' for i in ids)))
        membership[members]=ci;class_signs.append(signatures[members[0]])
    n=int(mask4.sum());means=np.asarray(class_signs,dtype=np.int8).reshape(len(classes),n).astype(np.float64)*PUBLIC.alpha
    return dict(classes=classes,valid_catalog_indices=valid,membership=membership,means=means,edge_availability=edge,
        edge_mask4=mask4,available_dimensions=n,catalog=catalog)


def empty_summary(reason):
    return dict(status='INCOMPLETE',reason=reason,score_status='UNCALIBRATED_DIFFERENCE_DIAGNOSTIC',accepted_payload=False,
        state_path_accepted=False,canonical_catalog_index=None,top_class_indices=[],top_catalog_indices=[],
        min_cost=None,zero_edit_min_cost=None,zero_edit_minus_best=None,unique_model_hypothesis=False,unique_is_not_confidence=True)


def edge_costs(dq,key,edge_availability,*,protocol=frozen):
    pairs=transition_pairs();signs=protocol.composite_signs(key).astype(np.float64)*PUBLIC.alpha
    templates=np.asarray([signs[v-1]-signs[u-1] for u,v in pairs]);rows=[];scored=0
    for j,(obs,mask) in enumerate(zip(dq,edge_availability)):
        support=np.broadcast_to(mask[:,None,None],(4,4,8));n=int(support.sum())
        if n==0:
            rows.append(dict(received_edge_index=j+1,received_window_pair=[j+1,j+2],status='NO_SUPPORT',costs=[None]*132,top_pair_indices=[],available_dimensions=0));continue
        costs=((templates[:,support]-obs[support])**2).mean(axis=1);minimum=float(costs.min());top=np.flatnonzero(costs-minimum<=PUBLIC.tie_atol).tolist()
        rows.append(dict(received_edge_index=j+1,received_window_pair=[j+1,j+2],status='SCORED',costs=costs.tolist(),top_pair_indices=top,
            available_dimensions=n,min_cost=minimum,observed_energy=float((obs[support]**2).sum())))
        scored+=132
    return dict(pairs=[list(p) for p in pairs],rows=rows,scored=scored,
        score_status='UNCALIBRATED_LOCAL_TRANSITION_DIAGNOSTIC',state_accepted=False,
        note='43 correlated differences; not independent votes; paths retain44 source-window coordinates')


def infer_difference(q,key,availability,*,protocol=frozen):
    """Input is cached local q, key and public support only; no truth or message."""
    summary=empty_summary('NOT_EVALUATED');base=dict(summary=summary,received_length=44,edge_length=43,
        valid_catalog_indices=[],class_costs=[],path_costs=[],projection=None,difference_projection=None,
        classes=[],edge_availability=None,available_dimensions=0,edge_local=None,
        counts=dict(catalog=3915,scorable=174,structurally_excluded=3741,equivalence_classes=0,scored=0))
    try:
        obs,mask=validate(q,availability);family=difference_family(key,mask,protocol=protocol);n=family['available_dimensions']
        with np.errstate(over='raise',invalid='raise'):
            dq=obs[1:]-obs[:-1]
            if not np.isfinite(dq).all():raise ValueError('nonfinite difference projection')
            local=edge_costs(dq,key,family['edge_availability'],protocol=protocol)
        base.update(valid_catalog_indices=family['valid_catalog_indices'],classes=family['classes'],
            class_costs=[None]*len(family['classes']),path_costs=[None]*174,projection=obs.tolist(),difference_projection=dq.tolist(),
            edge_availability=family['edge_availability'].tolist(),available_dimensions=n)
        base['counts']['equivalence_classes']=len(family['classes'])
        base['edge_local']=local
        if n==0:summary['reason']='NO_OBSERVATIONS';return base
        values=dq[family['edge_mask4']]
        with np.errstate(over='raise',invalid='raise'):
            residual=family['means']-values[None,:];costs=np.sum(residual*residual,axis=1,dtype=np.float64)/n
            energy=float(np.sum(values*values,dtype=np.float64))
        if not np.isfinite(costs).all() or not np.isfinite(energy):raise ValueError('nonfinite difference reduction')
        minimum=float(costs.min());top_classes=np.flatnonzero(costs-minimum<=PUBLIC.tie_atol).tolist()
        top=sorted(i for ci in top_classes for i in family['classes'][ci]['member_catalog_indices']);catalog=family['catalog']
        zero_indices=[j for j,i in enumerate(family['valid_catalog_indices']) if catalog[i]['event_type']=='ZERO_EDIT']
        path_costs=costs[family['membership']];zero_min=float(path_costs[zero_indices].min())
        canonical=min(top,key=lambda i:(tuple(catalog[i]['taus']),catalog[i]['event_type'],catalog[i]['event_i'] or 0))
        structural=any(len(family['classes'][ci]['member_catalog_indices'])>1 for ci in top_classes)
        summary.update(status='NO_ENERGY' if energy==0 else 'COMPLETE',reason='NO_ENERGY' if energy==0 else 'STRUCTURAL_AMBIGUITY' if structural else 'NUMERICAL_CLASS_TIE' if len(top_classes)>1 else 'UNIQUE_FINITE_MODEL_ONLY',
            canonical_catalog_index=None if energy==0 else canonical,top_class_indices=top_classes,top_catalog_indices=top,min_cost=minimum,
            zero_edit_min_cost=zero_min,zero_edit_minus_best=zero_min-minimum,unique_model_hypothesis=energy>0 and len(top)==1,
            structural_ambiguity=structural,any_top_contains_zero_edit=any(catalog[i]['event_type']=='ZERO_EDIT' for i in top),
            all_top_require_event=all(catalog[i]['event_type']!='ZERO_EDIT' for i in top),observed_projection_energy=energy,
            top_distinct_class_gap=float(np.partition(costs,1)[1]-minimum) if len(costs)>1 else None,
            capture_status='REJECTED_ZERO_ENERGY' if energy==0 else 'UNCALIBRATED_NOT_ACCEPTED')
        base.update(class_costs=costs.tolist(),path_costs=path_costs.tolist());base['counts']['scored']=174
    except (ValueError,TypeError,FloatingPointError,OverflowError) as exc:summary.update(reason='INVALID_OBSERVATION',error=f'{type(exc).__name__}: {exc}')
    return base


def absolute_control(q,key,availability,*,protocol=frozen):
    """Exact old absolute inference algebra from the same q; no latent re-extraction."""
    summary=protocol._empty_summary('NOT_EVALUATED');catalog=protocol.catalog(44)
    base=dict(summary=summary,available_dimensions=0,received_length=44,valid_catalog_indices=[],class_costs=[],path_costs=[],projection=None,
        counts=dict(catalog=3915,scorable=174,structurally_excluded=3741,equivalence_classes=0,scored=0))
    try:
        obs,mask=validate(q,availability);family=protocol.family(key,44,mask);n=family['available_dimensions']
        base.update(available_dimensions=n,valid_catalog_indices=list(family['valid_indices']),class_costs=[None]*len(family['classes']),path_costs=[None]*174)
        base['counts']['equivalence_classes']=len(family['classes'])
        if n==0:summary['reason']='NO_OBSERVATIONS';return base
        values=obs[family['mask4']]
        with np.errstate(over='raise',invalid='raise'):
            residual=family['means']-values[None,:];costs=np.sum(residual*residual,axis=1,dtype=np.float64)/n
            energy=float(np.sum(values*values,dtype=np.float64))
        if not np.isfinite(costs).all() or not np.isfinite(energy):raise ValueError('nonfinite absolute reduction')
        minimum=float(costs.min());top_classes=np.flatnonzero(costs-minimum<=PUBLIC.tie_atol).tolist()
        top=sorted(i for ci in top_classes for i in family['classes'][ci]['member_catalog_indices'])
        path_costs=costs[family['membership']];zero_indices=[j for j,i in enumerate(family['valid_indices']) if catalog[i]['event_type']=='ZERO_EDIT']
        zero_min=float(path_costs[zero_indices].min());canonical=min(top,key=lambda i:(tuple(catalog[i]['taus']),catalog[i]['event_type'],catalog[i]['event_i'] or 0))
        structural=any(len(family['classes'][ci]['member_catalog_indices'])>1 for ci in top_classes)
        summary.update(status='NO_ENERGY' if energy==0 else 'COMPLETE',reason='NO_ENERGY' if energy==0 else 'STRUCTURAL_AMBIGUITY' if structural else 'NUMERICAL_CLASS_TIE' if len(top_classes)>1 else 'UNIQUE_FINITE_MODEL_ONLY',
            canonical_catalog_index=None if energy==0 else canonical,top_class_indices=top_classes,top_catalog_indices=top,min_cost=minimum,
            zero_edit_min_cost=zero_min,zero_edit_minus_best=zero_min-minimum,unique_model_hypothesis=energy>0 and len(top)==1,
            structural_ambiguity=structural,any_top_contains_zero_edit=any(catalog[i]['event_type']=='ZERO_EDIT' for i in top),
            all_top_require_event=all(catalog[i]['event_type']!='ZERO_EDIT' for i in top),observed_projection_energy=energy,
            top_distinct_class_gap=float(np.partition(costs,1)[1]-minimum) if len(costs)>1 else None)
        base.update(class_costs=costs.tolist(),path_costs=path_costs.tolist(),projection=obs.tolist(),local_state=protocol.local_state_costs(obs,key,mask));base['counts']['scored']=174
    except (ValueError,TypeError,FloatingPointError,OverflowError) as exc:summary.update(reason='INVALID_OBSERVATION',error=f'{type(exc).__name__}: {exc}')
    return base
