"""Fixed RANK3 receiver, independent of diagnostics and payload truth.

Full-C CDC remains unchanged; singleton geometry uses an isolated function namespace.
"""
from copy import deepcopy
from types import FunctionType
from main.tube_state import b_m3_time_path_v1 as geometry
from main.tube_state import b_cdc32_aggregation_v1 as cdc
READERS = ('CDC_M', 'UNION_M', 'RANK3', 'RANK2', 'FIXED_ORIGIN')
TAU = 0.49221603782894896
PATHS = {tuple(p['query_source_origins'][w] for w in ('L','M','R')): p for p in geometry.path_domain()}
ENDPOINTS = {(x[0], x[2]) for x in PATHS}

def check_family(family):
    rows, issues, missing, unexpected, submitted = geometry._score_rows(family)
    result = dict(members=rows, issues=issues, missing=missing, unexpected=unexpected, submitted=submitted,
                  status='TECHNICAL_UNRESOLVED', maximum=None, winners=[], selected=None)
    if issues:
        return result
    maximum = max(r['S'] for r in rows)
    winners = [r for r in rows if r['S'] == maximum]
    result.update(maximum=maximum, winners=[dict(phase=r['phase'], source_start=r['source_start']) for r in winners],
                  status='UNIQUE_ARGMAX' if len(winners)==1 else 'METHOD_AMBIGUITY_ERASURE')
    if len(winners)==1:
        result['selected'] = dict(winners[0], origin=geometry.state_to_origin(winners[0]['phase'], winners[0]['source_start']))
    return result

def select(families, reader):
    if reader == 'FIXED_ORIGIN':
        chosen = {w:dict(phase=geometry.origin_to_state(o)[0], source_start=geometry.origin_to_state(o)[1], origin=o)
                  for w, o in geometry.WINDOWS.items()}
        return dict(status='NO_EXISTENCE_DECISION', detected=None, selected=chosen,
                    windows=list(geometry.WINDOWS), exact_seam_certified=False)
    if reader in ('CDC_M', 'UNION_M'):
        local = {w:None if f['issues'] else bool(f['maximum']>TAU) for w, f in families.items()}
        detected = True if any(v is True for v in local.values()) else None if any(v is None for v in local.values()) else False
        candidates = {w:[dict(r, origin=geometry.state_to_origin(r['phase'], r['source_start']))
                         for r in f['members'] if not f['issues'] and r['S']>=TAU] for w, f in families.items()}
        compatible = [p for origins,p in PATHS.items() if all(local[w] is True and origins[i] in {r['origin'] for r in candidates[w]}
                     for i,w in enumerate(('L','M','R')))]
        return dict(status='DETECT' if detected is True else 'TECHNICAL_UNRESOLVED' if detected is None else 'REJECT_ALL',
                    detected=detected, local_D=local, candidates=candidates, windows=['L','M','R'],
                    M3_path_class_count=len(compatible), M3_seam_alias_count=sum(p['seam_multiplicity'] for p in compatible),
                    detection_rule='ANY_LOCAL_DETECTED', tau=TAU, exact_seam_certified=False)
    windows = ('L','M','R') if reader=='RANK3' else ('L','R') if reader=='RANK2' else None
    if windows is None:
        raise ValueError('unknown reader')
    local = {w:families[w] for w in windows}
    if any(f['status']=='TECHNICAL_UNRESOLVED' for f in local.values()):
        return dict(status='TECHNICAL_UNRESOLVED', detected=None, selected={}, windows=list(windows))
    if any(f['status']=='METHOD_AMBIGUITY_ERASURE' for f in local.values()):
        return dict(status='METHOD_AMBIGUITY_ERASURE', detected=False, selected={}, windows=list(windows))
    selected = {w:deepcopy(f['selected']) for w,f in local.items()}
    origins = tuple(selected[w]['origin'] for w in windows)
    accepted = origins in (PATHS if reader=='RANK3' else ENDPOINTS)
    return dict(status='DETECT' if accepted else 'REJECT_GEOMETRY', detected=accepted,
                selected=selected, windows=list(windows), origins=list(origins),
                path_class=deepcopy(PATHS[origins]) if reader=='RANK3' and accepted else None,
                endpoint_projection_only=reader=='RANK2', exact_seam_certified=False)

def _singleton_family(prepared):
    """Independent selection policy; original CDC full-C validation is untouched."""
    mode=prepared['receiver']; gate=prepared['gate']; members=prepared.get('members',[])
    selected=prepared.get('selected')
    if mode=='FIXED_ORIGIN':
        origin=prepared['received_query_start']
        p,s=geometry.origin_to_state(origin)
        chosen=dict(phase=p,source_start=s)
        return [],[chosen],None,None,None,'READY'
    checked=check_family(dict(status='SCORES_COMPLETE',candidates=members))
    if checked['status']=='TECHNICAL_UNRESOLVED' or gate=='TECHNICAL_UNRESOLVED':
        return members,[],checked['maximum'],None,False,'TECHNICAL_UNRESOLVED'
    if gate!='DETECT':
        return members,[],checked['maximum'],None,False,gate
    if checked['status']!='UNIQUE_ARGMAX' or selected is None or any(selected[k]!=checked['selected'][k] for k in ('phase','source_start')):
        raise ValueError('singleton selection contradicts the complete H137 argmax')
    return members,[selected],checked['maximum'],None,True,'READY'

_singleton_builder=FunctionType(cdc.plan_local.__code__,dict(cdc.plan_local.__globals__,_family=_singleton_family), name='singleton_geometry_builder',argdefs=cdc.plan_local.__defaults__)

def plan_singleton_local(prepared,views):
    plan=_singleton_builder(prepared,views)
    plan['mapping_scope']=prepared['receiver']+'_SINGLETON_C' if prepared['receiver']!='FIXED_ORIGIN' else 'FIXED_ORIGIN_NO_EXISTENCE'
    plan['selection_gate']=prepared['gate']
    plan['threshold_used']=False
    plan['singleton_C']=plan.pop('full_local_C')
    return plan

def plan_task(task,record,families,views_by_window):
    """No payload or private labels. Opaque IDs come only from view metadata."""
    result={}
    for reader in READERS:
        selection=record['receivers'][reader]; plans=[]
        for w in selection['windows']:
            views=views_by_window[w]; ids=views[0]
            prepared=dict(query_id=ids['query_id'],input_id=ids['input_id'],key_id=task.split('/')[-1],
                          received_query_start=geometry.WINDOWS[w],members=families[w]['members'])
            if reader in ('CDC_M','UNION_M'):
                raw=dict(status='SCORES_COMPLETE' if not families[w]['issues'] else 'SCORE_UNRESOLVED',candidates=families[w]['members'],
                         input_id=ids['input_id'],key_id=prepared['key_id'])
                local=geometry.prepare_local(raw,dict(status='FROZEN',tau=TAU),ids['query_id'],geometry.WINDOWS[w])
                plan=cdc.plan_local(local,views)
            else:
                prepared.update(receiver=reader,gate=selection['status'],selected=selection.get('selected',{}).get(w),
                                status='READY' if selection['status'] in ('DETECT','NO_EXISTENCE_DECISION') else selection['status'],
                                score_status='SCORES_COMPLETE' if not families[w]['issues'] else 'SCORE_UNRESOLVED')
                plan=plan_singleton_local(prepared,views)
            plan['window']=w
            plans.append(plan)
        result[reader]=dict(selection=selection,locals=cdc.reconcile_plans(plans),
                            message_used=False,physical_calls=0)
    return result
