"""Fixed local45 observation adapter. No source start, phase or message in selection."""
from collections import Counter
import math
import torch
from . import b_legacy_position_margin_v1 as carrier

KEY_IDS=carrier.KEY_IDS
KEYS=carrier.KEYS
STARTS=tuple(range(1,35))
LOCAL_SHAPE=(1,16,12,40,64)


def validate_local(z):
    if not isinstance(z,torch.Tensor) or tuple(z.shape)!=LOCAL_SHAPE or z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):
        raise ValueError("finite normalized FP32 local latent[1,16,12,40,64] required")


def observe(z,book):
    validate_local(z);value=z.detach().cpu().double();roi=torch.empty((11,4,44),dtype=torch.float64)
    for k,(y0,y1,x0,x1) in enumerate(carrier.ROIS):
        patches=value[0,4:16,1:12,y0:y1,x0:x1].permute(1,0,2,3).reshape(11,768)
        roi[:,k]=(patches@book['directions'][k].reshape(44,768).T)*book['signs'][k]
    pooled=roi.mean(1)
    if not bool(torch.isfinite(roi).all() and torch.isfinite(pooled).all()):raise ValueError("nonfinite local scores")
    rows=[];gaps=[]
    for i,row in enumerate(pooled):
        ties=[j+1 for j in range(44) if float(row[j])==float(row.max())]
        rows.append(dict(local_row=i+1,provisional_best=ties[0] if len(ties)==1 else None,exact_max_ties=ties,reliable_identity=None))
        gaps.append([float(row[j]-torch.cat((row[:j],row[j+1:])).max()) for j in range(44)])
    return dict(status='SCORE_OBSERVED',roi_q=roi.tolist(),pooled_q=pooled.tolist(),competitor_margins=gaps,rows=rows,
                identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',truth_inputs=False)


def unavailable_observation(status):
    return dict(status=status,roi_q=[[[None]*44 for _ in range(4)] for _ in range(11)],pooled_q=[[None]*44 for _ in range(11)],
        competitor_margins=[[None]*44 for _ in range(11)],rows=[dict(local_row=i+1,provisional_best=None,exact_max_ties=None,reliable_identity=None,status=status) for i in range(11)],
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',truth_inputs=False)


def blank_match(status='NOT_ENTERED',error=None):
    return dict(status=status,comparison_status='INCOMPLETE',scores=[dict(source_start=s,status=status,score=None,sum=None,terms=None,
        denominator=11,coordinates=[dict(local_row=i,source_column=s+i-1,row_0based=i-1,column_0based=s+i-2) for i in range(1,12)]) for s in STARTS],
        exact_max_ties=None,provisional_start=None,maximum_score=None,error=error,truth_inputs=False)


def match(local_matrix):
    """Only11x44 received scores; no case ID/parent/phase/message arguments."""
    if not isinstance(local_matrix,list) or len(local_matrix)!=11 or any(not isinstance(row,list) or len(row)!=44 for row in local_matrix):
        raise ValueError('local11x44 Q required')
    if any(type(x) not in (int,float) or not math.isfinite(x) for row in local_matrix for x in row):raise ValueError('finite local Q required')
    result=blank_match()
    for row in result['scores']:
        terms=[local_matrix[c['row_0based']][c['column_0based']] for c in row['coordinates']]
        total=math.fsum(terms);score=total/11
        if not math.isfinite(score):raise ValueError('nonfinite sequence score')
        row.update(status='RETURNED',score=score,sum=total,terms=terms)
    maximum=max(r['score'] for r in result['scores']);ties=[r['source_start'] for r in result['scores'] if r['score']==maximum]
    result.update(status='SCORED',comparison_status='COMPLETE',exact_max_ties=ties,provisional_start=ties[0] if len(ties)==1 else None,maximum_score=maximum)
    return result


def local_map(matched):
    ties=matched['exact_max_ties'] if matched['comparison_status']=='COMPLETE' else []
    rows=[]
    for r in range(1,12):
        js=sorted({s+r-1 for s in ties});gs=sorted({(j-1)//11 for j in js});j=js[0] if len(js)==1 else None
        rows.append(dict(local_row=r,source_cells=js,source_groups=gs,provisional_source_cell=j,
            provisional_group=gs[0] if len(gs)==1 else None,eligible_source_cell=j,eligible_group=None if j is None else (j-1)//11,
            status='CONDITIONAL_CELL' if j is not None else 'GROUP_ONLY' if len(gs)==1 else 'AMBIGUOUS' if js else 'UNREAD',identity_status='UNCALIBRATED'))
    return dict(status='CONDITIONAL_MAP',cells=rows,source_occupancy=[dict(source_cell=j,local_rows=[r['local_row'] for r in rows if r['eligible_source_cell']==j]) for j in range(1,45)],
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',global_path=None,
        collision_scope='this single received clip/key only; do not combine controls')


def unavailable_map(status,unavailable_from=None):
    """Fixed failure rows without invoking the possibly failed mapping operation."""
    cause={} if unavailable_from is None else dict(unavailable_from=unavailable_from)
    rows=[dict(local_row=r,source_cells=[],source_groups=[],provisional_source_cell=None,
        provisional_group=None,eligible_source_cell=None,eligible_group=None,status=status,
        identity_status='UNCALIBRATED',**cause) for r in range(1,12)]
    return dict(status=status,cells=rows,source_occupancy=[dict(source_cell=j,local_rows=[],status=status,**cause) for j in range(1,45)],
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',global_path=None,**cause)


def spectrum(z):
    validate_local(z)
    return torch.fft.fft2(z[0,:4,1:12].float(),dim=(-2,-1),norm='ortho').real


def read_spectrum(value,key):
    if tuple(value.shape)!=(4,11,40,64) or not bool(torch.isfinite(value).all()):raise ValueError('finite shared local spectrum required')
    coords=carrier.grow.coordinates(key)
    x=value[:,:,[c[0] for c in coords],[c[1] for c in coords]]
    signed=(2*(x>0).to(torch.int8)-1).cpu().tolist();zeros=(x==0).to(torch.int8).cpu().tolist();cells=[]
    for r in range(11):
        bits=[]
        for b in range(8):
            seq=[int(signed[c][r][k]>0) for c in range(4) for k in range(b,240,8)];ones=sum(seq)
            bits.append(dict(slot=b,decoded=int(Counter(seq).most_common(1)[0][0]),ones=ones,count=120,tie=ones*2==120,
                             zero_coefficients=sum(zeros[c][r][k] for c in range(4) for k in range(b,240,8))))
        cells.append(dict(local_row=r+1,bits=bits))
    return dict(status='RAW_READ',signed_votes=signed,zero_coefficient_mask=zeros,local_cells=cells,
        raw_bit_slots=88,signed_vote_slots=10560,truth_inputs=False,message_inputs=False)


def unavailable_votes(status):
    return dict(status=status,signed_votes=None,zero_coefficient_mask=None,local_cells=[dict(local_row=r,bits=[dict(slot=b,decoded=None,status=status) for b in range(8)]) for r in range(1,12)],raw_bit_slots=88,signed_vote_slots=10560)


def reduce_payload(raw,mapping):
    if raw['status']!='RAW_READ':return unavailable_payload(raw['status'])
    signed,zeros=raw['signed_votes'],raw['zero_coefficient_mask'];bits=[None]*32;segments=[];local=[]
    for row,values in zip(mapping['cells'],raw['local_cells']):local.append(dict(**row,bits=values['bits']))
    for g in range(4):
        eligible=[r for r in mapping['cells'] if r['eligible_group']==g];rows=[]
        for b in range(8):
            seq=[int(signed[c][r['local_row']-1][k]>0) for c in range(4) for r in eligible for k in range(b,240,8)]
            if seq:
                value=int(Counter(seq).most_common(1)[0][0]);ones=sum(seq);bits[8*g+b]=value
                row=dict(slot=8*g+b,decoded=value,count=len(seq),ones=ones,tie=ones*2==len(seq),status='CONDITIONAL_READ',
                    zero_coefficients=sum(zeros[c][r['local_row']-1][k] for c in range(4) for r in eligible for k in range(b,240,8)))
            else:row=dict(slot=8*g+b,decoded=None,count=0,ones=0,tie=None,status='UNREAD_LOCATION',zero_coefficients=0)
            rows.append(row)
        segments.append(dict(group=g,observed_cells=len(eligible),planned_source_cells=11,cells=eligible,bits=rows))
    return dict(status='CONDITIONAL_REDUCED',decoded_bits=bits,segments=segments,local_cells=local,planned_bits=32,planned_local_bits=88,
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',trusted_full_message=None,truth_inputs=False)


def unavailable_payload(status,unavailable_from=None):
    cause={} if unavailable_from is None else dict(unavailable_from=unavailable_from)
    return dict(status=status,decoded_bits=[None]*32,segments=[dict(group=g,observed_cells=0,planned_source_cells=11,cells=[],status=status,**cause,
        bits=[dict(slot=8*g+b,decoded=None,count=0,status=status,**cause) for b in range(8)]) for g in range(4)],
        local_cells=[dict(local_row=r,provisional_source_cell=None,eligible_source_cell=None,provisional_group=None,eligible_group=None,status=status,**cause,
            bits=[dict(slot=b,decoded=None,status=status,**cause) for b in range(8)]) for r in range(1,12)],
        planned_bits=32,planned_local_bits=88,identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',trusted_full_message=None,truth_inputs=False,**cause)
