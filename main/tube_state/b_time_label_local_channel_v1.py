"""Card106 adopted local-channel orchestration arithmetic. Pure stdlib, no truth."""
import copy
from . import b_detectable_time_label_v1 as core
KEYS={'key00':'watermark','key01':'watermark-wrong'}
ARMS=('OFF','P0','PT')
CLEAN_ORIGINS=(0,35,70,105,136)
VIEWS=core.VIEWS

def protocol():
    return dict(status='ADOPTED_FIXED_SEEN_LOCAL_STAGE1',tag_IDs=44,code='RM(1,5)',code_chips=32,pilot_chips=16,
        delta=core.DELTA,alpha=1,code_radius=3,pilot_max_errors=1,carrier_channels=[4,16],
        frame_indices='zero_based; cell/local_row one_based',received_frames=45,phases=list(range(4)),retained_rows=10,
        null_rows_per_host=1200,host_groups=2,expected_local_rows=5760,expected_native_rows=1056,
        no_state_space=True,authentication_claim=False,independent_confirmation=False,
        exact_geometry_certified=False,FP32_single_materialization=True)

def reuse(arm,view,phase):
    if arm not in ('OFF','P0'):return None
    if view=='clean0':return dict(crop=0,origin=0,phase=phase)
    if view=='clean4':return dict(crop=5,origin=136,phase=phase)
    if view=='clean3' and phase in (2,3):return dict(crop=4,origin=107,phase=phase-2)
    return None

def technical_row(reason='NOT_ENTERED'):
    return dict(status='TECHNICAL_UNRESOLVED',technical_failure=True,error=reason,
        pilot_P=None,pilot_H=None,pilot_detected=None,pilot_status='TECHNICAL_UNRESOLVED',
        physical_bits=[None]*48,unmasked_bits=[None]*32,code_status='TECHNICAL_UNRESOLVED',
        code_distances=[None]*64,nearest_ids=None,nearest=None,nearest_distance=None,
        code_id_candidate=None,tag_label=None,truth_used=False,message_used=False,
        authentication_claim=False,exact_geometry_certified=False)

def raw_table(projections,key,expected_rows):
    """Any per-unit core exception becomes a fixed technical row, never deletion."""
    rows=[]
    try:b=core.book(key)
    except Exception as error:return unavailable_table(expected_rows,f'BOOK: {type(error).__name__}: {error}')
    for i in range(expected_rows):
        try:
            item=projections[i]
            if isinstance(item,dict) and item.get('error'):raise ValueError(item['error'])
            q=item['projections'] if isinstance(item,dict) else item
            raw=core.decode_projections(q,b,None)
            raw['technical_failure']=False
        except Exception as error:
            q=None;raw=technical_row(f'{type(error).__name__}: {error}')
        rows.append(dict(row=i+1,projections=q,decoder=raw))
    return dict(status='RAW_UNITS_SAVED',expected_rows=expected_rows,rows=rows,
        technical_rows=sum(r['decoder']['technical_failure'] for r in rows),truth_used=False,message_used=False)

def unavailable_table(n,reason):
    return dict(status='TECHNICAL_UNRESOLVED',expected_rows=n,rows=[
        dict(row=i+1,projections=None,decoder=technical_row(reason)) for i in range(n)],technical_rows=n,
        truth_used=False,message_used=False)

def finish(raw,tau):
    """Threshold application only; no re-projection or repeated64-code distances."""
    value=copy.deepcopy(raw)
    if value.get('technical_failure'):return value
    value['tau']=tau
    if tau is None:
        value.update(status='CALIBRATION_UNRESOLVED',pilot_detected=None,pilot_status='CALIBRATION_UNRESOLVED',tag_label=None);return value
    H,P=value['pilot_H'],value['pilot_P']
    detected=H is not None and H<=core.PILOT_MAX_ERRORS and P>tau
    code=value['code_status']
    value['pilot_detected']=detected
    value['pilot_status']='PILOT_SYMBOL_ERASURE' if H is None else 'PILOT_MISMATCH' if H>1 else 'SOFT_BELOW_THRESHOLD' if P<=tau else 'PILOT_PASS'
    value['status']='ACCEPTED_TAG_CANDIDATE' if detected and code=='LEGAL_ID' else 'DETECTED_LABEL_ERASURE' if detected else 'CODE_ONLY_REJECT' if code=='LEGAL_ID' else 'NO_DETECTION'
    value['tag_label']=value['code_id_candidate']+1 if detected and code=='LEGAL_ID' else None
    return value

def finalize(table,tau):
    rows=[];n=table['expected_rows']
    for i in range(n):
        try:decoder=finish(table['rows'][i]['decoder'],tau)
        except Exception as error:decoder=technical_row(f'{type(error).__name__}: {error}')
        rows.append(dict(row=i+1,decoder=decoder))
    return dict(status='BLIND_UNITS_SAVED',expected_rows=n,rows=rows,
        technical_rows=sum(r['decoder'].get('technical_failure',False) for r in rows),truth_used=False,message_used=False)

def public_interval(phase,row):
    return [phase+4*row-3,phase+4*row]

def null_records(tables):
    rows=[]
    for (arm,view,phase,key),table in tables.items():
        if arm=='PT' and key=='key00':continue
        for r in table['rows']:
            d=r['decoder']
            rows.append({'class':arm+'/'+key,'view':view,'phase':phase,'row':r['row'],'P':d['pilot_P'],
                         'technical_failure':d.get('technical_failure',False)})
    return rows
