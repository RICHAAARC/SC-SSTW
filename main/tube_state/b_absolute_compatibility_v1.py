"""Card98 frozen absolute compatibility/common-cell logic. Standard library only.
No Q, FFT, votes/Counter, model, truth, path search or candidate rescoring.
"""
import math

HOSTS=('b93-c01','b93-c02','b93-v01','b93-v02')
CALIBRATION_HOSTS=HOSTS[:2]
ORIGINS=(0,4,17,62,107,136)
H=tuple((p,s) for p in range(4) for s in range(1 if p==0 else 2,36))


def finite(v):
    return type(v) in (int,float) and math.isfinite(v)


def query_metadata():
    return {f'q{i:02}':dict(arm=a,origin=o,key=k) for i,(a,o,k) in enumerate(
        (a,o,k) for a in ('OFF','P0','PM') for o in ORIGINS for k in ('key00','key01'))}


def prepare_family(family,error=None):
    """Validate/cache saved scores once; never read an external file or truth."""
    original=family.get('candidates',[]) if isinstance(family,dict) else []
    ordered=[(r.get('phase'),r.get('source_start')) for r in original]==list(H)
    rows=[]
    for i,(p,s) in enumerate(H):
        row=original[i] if ordered else {}
        value=row.get('T')
        status=row.get('score_status','UNAVAILABLE')
        rows.append(dict(phase=p,source_start=s,origin=4*(s-1)-p,T=value if finite(value) else None,
            score_status=status,valid=ordered and status=='RETURNED' and finite(value)))
    complete=all(x['valid'] for x in rows)
    maximum=max(x['T'] for x in rows) if complete else None
    ties=[dict(phase=x['phase'],source_start=x['source_start']) for x in rows if x['T']==maximum] if complete else []
    raw=family.get('phase_local_bits',{}) if isinstance(family,dict) else {}
    return dict(score_complete=complete,rows=rows,maximum=maximum,exact_max_ties=ties,
        phase_local_bits=raw if isinstance(raw,dict) else {},read_error=error)


def calibrate(prepared_calibration):
    """Only old c01/c02 scores and frozen supervised role/geometry; no bits."""
    result=dict(status='CALIBRATION_UNRESOLVED',tau_D=None,L_D=None,hosts={},message_used=False,
        independent_confirmation=False,scope='all sources already seen; development only')
    if set(prepared_calibration)!=set(CALIBRATION_HOSTS):return result
    for host in CALIBRATION_HOSTS:
        families=prepared_calibration[host]
        if set(families)!=set(query_metadata()) or any(not f['score_complete'] for f in families.values()):
            return result
        nulls=[];positive=[]
        for qid,meta in query_metadata().items():
            f=families[qid]
            if meta['arm']=='PM' and meta['key']=='key00':
                p=(-meta['origin'])%4;s=(meta['origin']+p)//4+1
                row=next(x for x in f['rows'] if (x['phase'],x['source_start'])==(p,s))
                positive.append(dict(query=qid,true_phase=p,true_source_start=s,T=row['T']))
            else:nulls.append(dict(query=qid,M=f['maximum']))
        result['hosts'][host]=dict(N=max(x['M'] for x in nulls),K=min(x['T'] for x in positive),nulls=nulls,positives=positive)
    result.update(status='DEVELOPMENT_THRESHOLDS',tau_D=max(0,*(v['N'] for v in result['hosts'].values())),
        L_D=min(v['K'] for v in result['hosts'].values()))
    return result


def actual_row_geometry(origin,phase,row):
    """Public geometry helper; actual origin is only supplied by post/case callers."""
    frames=list(range(origin+phase+4*row-3,origin+phase+4*row+1))
    cells=sorted({(f+3)//4 for f in frames})
    guarded=any(j not in range(1,45) for j in cells)
    groups=sorted({(j-1)//11 for j in cells if 1<=j<=44})
    return dict(source_frames=frames,source_cells=cells,
        true_source_cell=cells[0] if len(cells)==1 and not guarded else None,
        true_group=groups[0] if len(groups)==1 and not guarded else None,
        kind='GUARD_OR_OUT_OF_PROTOCOL' if guarded else 'SINGLE_SOURCE_FULL' if len(cells)==1 else 'MIXED_SAME_GROUP' if len(groups)==1 else 'MIXED_GROUP_UNDEFINED')


def physical_cells(origin,group):
    rows=[]
    for j in range(11*group+1,11*group+12):
        count=max(0,min(4*j,origin+44)-max(4*j-3,origin)+1)
        rows.append(dict(source_cell=j,status='FULL' if count==4 else 'PARTIAL' if count else 'ABSENT'))
    return rows


def no_common_reason(candidates,group):
    available=[];partial=[];observed=[]
    for h in candidates:
        physical=physical_cells(h['origin'],group)
        available.append(any(x['status']=='FULL' for x in physical))
        partial.append(any(x['status']=='PARTIAL' for x in physical))
        observed.append(any(11*group+1<=j<=11*group+11 for j in range(h['source_start'],h['source_start']+10)))
    if not any(available):return 'PARTIAL_ONLY_NO_FULL_CELL' if any(partial) else 'STRUCTURALLY_ABSENT_UNDER_C'
    if not all(available):return 'PRESENCE_UNRESOLVED'
    if not all(observed):return 'PRESENT_BUT_UNOBSERVED_UNDER_SOME_H'
    return 'NO_COMMON_CELL_IDENTITY'


def decision(prepared,thresholds):
    """One fixed pass. Does not receive host/query ID, source origin or message."""
    good_thresholds=thresholds.get('status')=='DEVELOPMENT_THRESHOLDS' and finite(thresholds.get('tau_D')) and finite(thresholds.get('L_D'))
    valid=prepared['score_complete'] and good_thresholds
    members=[]
    for row in prepared['rows']:
        included=row['T']>=thresholds['L_D'] if valid else None
        members.append(dict(**row,in_C=included,membership_status='DECIDED' if valid else 'UNRESOLVED'))
    candidates=[x for x in members if x['in_C'] is True]
    if not prepared['score_complete']:detection='UNRESOLVED_SCORE'
    elif not good_thresholds:detection='CALIBRATION_UNRESOLVED'
    else:detection='DETECTED_PROVISIONAL' if prepared['maximum']>thresholds['tau_D'] else 'REJECT_SYNC'
    location='EMPTY_SET_UNRESOLVED' if not candidates else 'SINGLETON_UNCERTIFIED' if len(candidates)==1 else 'SET_UNRESOLVED'
    groups=[];message=[];accesses=0
    left=max(h['source_start'] for h in candidates) if candidates else None
    right=min(h['source_start']+9 for h in candidates) if candidates else None
    for g in range(4):
        common=list(range(max(left,11*g+1),min(right,11*g+11)+1)) if candidates else []
        selected=min(common) if common else None  # Fixed BEFORE reading any bit; never try next j.
        reason='COMMON_CELL_AVAILABLE' if common else no_common_reason(candidates,g) if candidates else 'EMPTY_C'
        attempt=detection=='DETECTED_PROVISIONAL' and selected is not None
        references=[]
        if selected is not None:
            for h in candidates:
                r=selected-h['source_start']+1;p=h['phase']
                reference=dict(phase=p,source_start=h['source_start'],origin_hypothesis=h['origin'],
                    canonical_cell=selected,local_row=r,received_interval_inclusive=[p+4*r-3,p+4*r],
                    source_interval_under_h_inclusive=[4*selected-3,4*selected],raw_lookup_attempted=attempt,
                    values=None,value_status=None)
                if attempt:
                    pack=prepared['phase_local_bits'].get(str(p))
                    if not isinstance(pack,list) or len(pack)!=80:
                        values=[None]*8
                    else:values=[pack[8*(r-1)+b] for b in range(8)]
                    reference['values']=[v if type(v) is int and v in (0,1) else None for v in values]
                    reference['value_status']=['AVAILABLE' if type(v) is int and v in (0,1) else 'UNAVAILABLE' for v in values]
                    accesses+=8
                references.append(reference)
        group=dict(group=g,common_cells=common,selected_cell=selected,selection_reason=reason,
            actual_lookup_attempted=attempt,conditional_references=references)
        group_bits=[]
        for b in range(8):
            if detection=='REJECT_SYNC':status='REJECT_ERASURE';value=None
            elif detection in ('UNRESOLVED_SCORE','CALIBRATION_UNRESOLVED'):status='TECHNICAL_UNRESOLVED';value=None
            elif not candidates:status='ERASE_EMPTY_C';value=None
            elif selected is None:status='ERASE_NO_COMMON_CELL';value=None
            else:
                values=[x['values'][b] for x in references]
                if any(v is None for v in values):status='TECHNICAL_UNRESOLVED';value=None
                elif all(v==values[0] for v in values):status='CONDITIONAL_READ';value=values[0]
                else:status='ERASE_DISAGREEMENT';value=None
            bit=dict(slot=8*g+b,group=g,bit=b,decoded=value,status=status,
                selected_canonical_cell=selected if attempt else None)
            group_bits.append(bit);message.append(bit)
        group['message']=group_bits;groups.append(group)
    singleton=dict(phase=candidates[0]['phase'],source_start=candidates[0]['source_start'],origin=candidates[0]['origin']) if detection=='DETECTED_PROVISIONAL' and len(candidates)==1 else None
    return dict(status='LOGICAL_DECISION_SAVED_INPUT_ONLY',score_complete=prepared['score_complete'],
        source_read_error=prepared['read_error'],maximum=prepared['maximum'],exact_max_ties=prepared['exact_max_ties'],
        detection=detection,location_set_class=location,singleton_candidate=singleton,
        members=members,C=[dict(phase=x['phase'],source_start=x['source_start'],origin=x['origin']) for x in candidates],
        groups=groups,message=message,
        counts=dict(member_slots=137,numeric_member_comparisons=137 if valid else 0,group_slots=4,message_slots=32,
            cached_single_cell_bit_accesses=accesses,selected_canonical_cells=sum(x['actual_lookup_attempted'] for x in groups)),
        truth_used=False,conditional_references_are_not_independent_paths=True,
        nominal_cell_is_not_VAE_receptive_field=True,authenticated=False,scientific_PASS=None)
