"""Card103 pure endpoint-compatible path family. No truth, pixels or filesystem."""
import math
SOURCE_FRAMES=181
RECEIVED_FRAMES=121
ANCHORS={'left':0,'right':76}
LEGACY_TAU=0.5368688672209747
PAIR_COUNT=1891
PATH_COUNT=58621
H=[(p,s) for p in range(4) for s in range(1 if p==0 else 2,36)]

def public_protocol():
    return dict(frame_coordinates='zero_based; canonical_cell and local_row one_based',source_frames=181,received_frames=121,anchors=ANCHORS.copy(),
        phases=list(range(4)),anchor_frames=45,E_frames=41,seam_received_inclusive=[45,76],
        a_plus_d_max=60,pair_count=PAIR_COUNT,path_count=PATH_COUNT,
        limitation='intact endpoint anchors; coarse jump only; middle seam unobserved',
        net_bits_per_source=32,CFAR=False,independent_confirmation=False)

def pairs():
    return [dict(a=a,d=d,b_min=None if d==0 else 45,b_max=None if d==0 else 76,
                 multiplicity=1 if d==0 else 32) for a in range(61) for d in range(61-a)]

def state(origin):
    p=(-origin)%4
    return p,(origin+p)//4+1

def finite(value):
    return type(value) in (float,int) and math.isfinite(value)

def score_map(family):
    """Only numeric anchor scores; no cached bits are read."""
    rows=family.get('candidates',[]) if isinstance(family,dict) else []
    by={(r.get('phase'),r.get('source_start')):r for r in rows}
    valid_domain=len(rows)==137 and set(by)==set(H)
    return {o:(by[state(o)].get('S') if valid_domain and by[state(o)].get('score_status')=='RETURNED'
               and finite(by[state(o)].get('S')) else None) for o in range(137)}

def pair_family(left,right):
    ls,rs=score_map(left),score_map(right)
    rows=[]
    for i,row in enumerate(pairs()):
        x,y=ls[row['a']],rs[row['a']+76+row['d']]
        ok=finite(x) and finite(y)
        rows.append(dict(row,pair_index=i,J=min(x,y) if ok else None,
                         score_status='RETURNED' if ok else 'TECHNICAL_UNRESOLVED'))
    return summarize(rows)

def summarize(rows):
    complete=bool(rows) and all(r['score_status']=='RETURNED' and finite(r['J']) for r in rows)
    return dict(status='SCORES_COMPLETE' if complete else 'TECHNICAL_UNRESOLVED',
        pairs=rows,pair_denominator=len(rows),path_denominator=sum(r['multiplicity'] for r in rows),
        maximum=max(r['J'] for r in rows) if complete else None,
        equal_sync_features_per_pair=80,truth_used=False,message_used=False)

def empty_family(reason='NOT_ENTERED'):
    return summarize([dict(r,pair_index=i,J=None,score_status=reason) for i,r in enumerate(pairs())])

def subset(family,kind):
    rows=[r.copy() for r in family['pairs'] if (r['d']==0 if kind=='continuous' else r['a']==r['d']==0)]
    return summarize(rows)

def calibrate(hosts):
    """Exactly two viewed calibration hosts, each five predeclared null families."""
    out=dict(status='CALIBRATION_UNRESOLVED',tau=None,hosts={},legacy_tau_reference=LEGACY_TAU,
             CFAR=False,positive_floor=False,background_refit=False,development_only=True)
    if len(hosts)!=2:return out
    for host,fs in hosts.items():
        if set(fs)!=set(range(6)):return out
        nulls=[dict(query=q,M=fs[q].get('maximum')) for q in range(6) if q!=4]
        if any(not finite(r['M']) or fs[r['query']]['status']!='SCORES_COMPLETE' or
               fs[r['query']]['pair_denominator']!=PAIR_COUNT or fs[r['query']]['path_denominator']!=PATH_COUNT for r in nulls):
            return out
        out['hosts'][host]=dict(nulls=nulls,N=max(r['M'] for r in nulls))
    out.update(status='FROZEN',tau=max(0,*(x['N'] for x in out['hosts'].values())))
    return out

def available(row):
    result={}
    for anchor,origin in [('left',row['a']),('right',row['a']+76+row['d'])]:
        p,s=state(origin)
        for j in range(s,s+10):
            result[j]=dict(anchor=anchor,phase=p,source_start=s,local_row=j-s+1)
    return result

def prepare(family,thresholds):
    """Freeze all support/cell/reference choices before handing bits to read()."""
    numeric=family['status']=='SCORES_COMPLETE'
    threshold_ok=thresholds.get('status')=='FROZEN' and finite(thresholds.get('tau'))
    tau=thresholds.get('tau') if threshold_ok else None
    detected=numeric and threshold_ok and family['maximum']>tau
    reason='DETECTED_COARSE_SET' if detected else 'REJECT' if numeric and threshold_ok else 'TECHNICAL_UNRESOLVED' if not numeric else 'CALIBRATION_UNRESOLVED'
    members=[dict(r,in_C=(r['J']>=tau if numeric and threshold_ok else None)) for r in family['pairs']]
    C=[r for r in members if r['in_C'] is True]
    support=set.intersection(*(set(available(r)) for r in C)) if C else set()
    groups=[]
    for g in range(4):
        common=sorted(support&set(range(11*g+1,11*g+12)))
        j=common[0] if common and detected else None
        refs=[]
        if j is not None:
            for row in C:
                ref=available(row)[j]
                w=ANCHORS[ref['anchor']];p=ref['phase'];r=ref['local_row']
                refs.append(dict(ref,pair_index=row['pair_index'],a=row['a'],d=row['d'],
                    b_min=row['b_min'],b_max=row['b_max'],path_multiplicity=row['multiplicity'],
                    canonical_cell=j,received_interval=[w+p+4*r-3,w+p+4*r],
                    conditional_source_interval=[4*j-3,4*j]))
        groups.append(dict(group=g,common_cells=common,selected_cell=j,references=refs))
    return dict(status=reason,detection=detected,maximum=family['maximum'],tau=tau,
        pair_denominator=family['pair_denominator'],path_denominator=family['path_denominator'],
        members=members,C=C,C_pairs=len(C),C_paths=sum(r['multiplicity'] for r in C),
        coarse_jump_only=bool(detected and C and all(r['d']>0 for r in C)),
        groups=groups,truth_used=False,cell_choices_frozen_before_bit_access=True,
        middle_observed=False,exact_seam_certified=False,authenticated=False)

def read(prepared,bits):
    """bits[anchor][str(phase)]:80 cached bits. No search/rescore or Counter."""
    out=dict(prepared);groups=[];message=[];accesses=0
    for group in prepared['groups']:
        g=dict(group);rr=[]
        for source in g['references']:
            r=dict(source)
            cache=bits.get(r['anchor'],{}).get(str(r['phase']),[])
            if not isinstance(cache,(list,tuple)):cache=[]
            start=8*(r['local_row']-1)
            r['values']=[cache[start+b] if start+b<len(cache) and type(cache[start+b]) is int and cache[start+b] in (0,1) else None for b in range(8)]
            accesses+=8;rr.append(r)
        g['references']=rr
        for b in range(8):
            vv=[r['values'][b] for r in rr]
            value=vv[0] if vv and None not in vv and len(set(vv))==1 else None
            status='CONDITIONAL_READ' if value is not None else ('TECHNICAL_CACHE' if vv and None in vv else 'DISAGREEMENT' if vv else 'NO_COMMON_OBSERVATION' if prepared['detection'] else prepared['status'])
            message.append(dict(slot=8*g['group']+b,group=g['group'],bit=b,decoded=value,
                status=status,selected_cell=g['selected_cell']))
        groups.append(g)
    out.update(groups=groups,message=message,cached_bit_accesses=accesses,message_slots=32)
    return out

def independent_union(left,right):
    """Parallel reference only: no selection by truth, no voting across anchors."""
    result=[]
    for slot in range(32):
        vals=[x['message'][slot]['decoded'] for x in (left,right) if x['message'][slot]['decoded'] is not None]
        result.append(dict(slot=slot,decoded=vals[0] if vals and len(set(vals))==1 else None,
            status='CONSISTENT_PARALLEL_REFERENCE' if vals and len(set(vals))==1 else 'CONFLICT' if vals else 'ERASURE',
            anchor_values={k:v['message'][slot]['decoded'] for k,v in [('left',left),('right',right)]}))
    return dict(message=result,not_an_additional_receiver=True)
