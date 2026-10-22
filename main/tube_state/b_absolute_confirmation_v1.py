"""Card99 pure role-generic score calibration, Card98 compatibility and B0 shells."""
from . import b_absolute_compatibility_v1 as core

H=core.H
ORIGINS=core.ORIGINS
ARMS=('OFF','P0','PM')
query_metadata=core.query_metadata


def empty_family(reason='NOT_ENTERED'):
    return dict(status=reason,candidates=[dict(phase=p,source_start=s,T=None,F=None,V=None,score_status=reason) for p,s in H],
        phase_local_bits={},score_complete=False,truth_inputs=False,message_inputs=False)


def bind(tables,local_bits=None):
    out=empty_family('SCORE_INCOMPLETE')
    out['phase_local_bits']=local_bits or {}
    for row in out['candidates']:
        t=tables.get(row['phase'],{})
        matches=[x for x in t.get('scores',[]) if x.get('source_start')==row['source_start']]
        if len(matches)==1:
            x=matches[0]
            row.update(T=x.get('centered_min'),F=x.get('F'),V=x.get('V'),score_status=x.get('status','NOT_ENTERED'))
            if t.get('score_call_status','NOT_ENTERED')!='RETURNED':row['score_status']=t.get('score_call_status','NOT_ENTERED')
    out['score_complete']=core.prepare_family(out)['score_complete']
    out['status']='SCORES_COMPLETE' if out['score_complete'] else 'SCORE_INCOMPLETE'
    return out


def calibrate(host_families):
    """Exactly two assigned calibration sources, no raw bits/payload gate/message."""
    out=dict(status='CALIBRATION_UNRESOLVED',tau=None,L=None,hosts={},message_used=False,
        inherited_development_thresholds=False,population_FPR_guarantee=None)
    if len(host_families)!=2:return out
    for h,families in host_families.items():
        if set(families)!=set(query_metadata()):return out
        prepared={q:core.prepare_family(f) for q,f in families.items()}
        if any(not p['score_complete'] for p in prepared.values()):return out
        nulls=[];positives=[]
        for q,meta in query_metadata().items():
            p=prepared[q]
            if meta['arm']=='PM' and meta['key']=='key00':
                phase=(-meta['origin'])%4;s=(meta['origin']+phase)//4+1
                row=next(x for x in p['rows'] if (x['phase'],x['source_start'])==(phase,s))
                positives.append(dict(query=q,true_phase=phase,true_source_start=s,T=row['T']))
            else:nulls.append(dict(query=q,M=p['maximum']))
        out['hosts'][h]=dict(N=max(x['M'] for x in nulls),K=min(x['T'] for x in positives),nulls=nulls,positives=positives)
    out.update(status='FROZEN',tau=max(0,*(v['N'] for v in out['hosts'].values())),L=min(v['K'] for v in out['hosts'].values()))
    return out


def decision(family,thresholds):
    t=dict(status='DEVELOPMENT_THRESHOLDS' if thresholds.get('status')=='FROZEN' else 'CALIBRATION_UNRESOLVED',
        tau_D=thresholds.get('tau'),L_D=thresholds.get('L'))
    d=core.decision(core.prepare_family(family),t)
    d['threshold_mapping']='Card99 fresh tau/L; Card98 decision equations unchanged'
    return d


def unique_max(family):
    p=core.prepare_family(family)
    if not p['score_complete']:return None,'TECHNICAL_UNRESOLVED'
    if len(p['exact_max_ties'])!=1:return None,'EXACT_TIE_UNRESOLVED'
    return p['exact_max_ties'][0],'UNIQUE_MAX'


def b0_shell(family,selected=None,reduced=None,reason='TECHNICAL_UNRESOLVED'):
    p=core.prepare_family(family)
    out=dict(status=reason,detection='UNRESOLVED',location='UNRESOLVED',maximum=p['maximum'],
        exact_max_ties=p['exact_max_ties'],candidate_set=[],location_hypothesis=None,
        message=[dict(slot=i,decoded=None,status=reason) for i in range(32)],authenticated=False,scientific_pass=None)
    if selected is None or reduced is None:return out
    phase=selected['phase'];s=selected['source_start']
    out.update(status='B0_UNAUTHENTICATED_UNIQUE_MAX',detection='DETECTED_PROVISIONAL',location='PROVISIONAL_UNCERTIFIED',
        candidate_set=[selected],location_hypothesis=dict(phase=phase,source_start=s,origin=4*(s-1)-phase))
    if reduced.get('status')!='CONDITIONAL_REDUCED_UNCALIBRATED':return b0_shell(family,reason='TECHNICAL_UNRESOLVED')
    bits=reduced['decoded_bits']
    if len(bits)!=32 or any(v is not None and (type(v) is not int or v not in (0,1)) for v in bits):raise ValueError('invalid B0 reduced bits')
    out['message']=[dict(slot=i,decoded=v,status='B0_CONDITIONAL_GROUP_READ' if v is not None else 'UNREAD_LOCATION') for i,v in enumerate(bits)]
    return out


def gate_b0(base,thresholds):
    import copy
    out=copy.deepcopy(base)
    if thresholds.get('status')!='FROZEN' or not core.finite(base['maximum']):
        return b0_shell(empty_family(),reason='CALIBRATION_OR_SCORE_UNRESOLVED')
    if base['maximum']<=thresholds['tau']:
        out.update(status='REJECT_SYNC',detection='REJECT_SYNC',location='NOT_DETECTED',candidate_set=[],location_hypothesis=None)
        out['message']=[dict(slot=i,decoded=None,status='REJECT_ERASURE') for i in range(32)]
    return out
