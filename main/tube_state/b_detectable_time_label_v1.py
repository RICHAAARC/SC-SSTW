"""Card105 proposed channel core: keyed disjoint QIM, RM time ID, separate pilot.
Pure stdlib reference. Not connected to writer/default/runtime; no real data I/O.
ECC/unique decoding is never authentication or certified temporal geometry.
"""
from dataclasses import dataclass
from functools import cache
import hashlib,hmac,math,struct,sys
DOMAIN='B-TimeTag-QIM-RM48-V1'
DELTA=1/math.sqrt(12)
PILOT_CHIPS=tuple(i for i in range(48) if i%3==2)
DATA_CHIPS=tuple(i for i in range(48) if i%3!=2)
RADIUS=3
PILOT_MAX_ERRORS=1
VIEWS=tuple('clean'+str(i) for i in range(5))+('cross_center',)
NULL_CLASSES=('OFF/key00','OFF/key01','P0/key00','P0/key01','PT/key01')
PROPOSAL_STATUS='PROPOSED_ISOLATED_CHANNEL_CORE'
@dataclass(frozen=True)
class Book:
    vectors:tuple
    dithers:tuple
    pilot:tuple
    mask:tuple

def codeword(identity):
    if type(identity) is not int or not 0<=identity<64:raise ValueError('6-bit identity required')
    bits=tuple((identity>>i)&1 for i in range(6))
    return tuple(bits[0]^sum(bits[i]*((m>>(i-1))&1) for i in range(1,6))%2 for m in range(32))

CODEWORDS=tuple(codeword(i) for i in range(64))

@cache
def book(key):
    if not isinstance(key,str) or not key:raise ValueError('nonempty key required')
    def digest(text):return hmac.new(key.encode('utf-8'),(DOMAIN+'|'+text).encode('ascii'),hashlib.sha256).digest()
    directions=[]
    for chip in range(48):
        order=sorted(range(64),key=lambda n:(digest(f'carrier|{chip}|{n}'),n))
        positive=set(order[:32])
        directions.append(tuple(1/8 if n in positive else -1/8 for n in range(64)))
    dithers=tuple(DELTA*(2*(int.from_bytes(digest(f'dither|{i}')[:3],'big')+.5)/(2**24)-1) for i in range(48))
    pilot=tuple(digest(f'pilot|{i}')[0]&1 for i in range(16))
    mask=tuple(digest(f'mask|{i}')[0]&1 for i in range(32))
    return Book(tuple(directions),dithers,pilot,mask)

def finite(x):return type(x) in (float,int) and math.isfinite(x)

def fp32(x):
    if not finite(x):raise ValueError('finite source/update required')
    value=struct.unpack('f',struct.pack('f',x))[0]
    if not math.isfinite(value):raise ValueError('FP32 overflow')
    return value

def nearest_integer(z):
    """Tie chooses smaller integer. Decoder separately erases a half-integer."""
    lo=math.floor(z);hi=lo+1
    return min((lo,hi),key=lambda n:(abs(z-n),n))

def target_projection(q,bit,dither):
    if not finite(q) or type(bit) is not int or bit not in (0,1) or not finite(dither):
        raise ValueError('finite projection/dither and binary symbol required')
    z=(q-dither-DELTA*bit)/(2*DELTA)
    lo=math.floor(z);hi=lo+1
    k=min((lo,hi),key=lambda n:(abs(dither+DELTA*(2*n+bit)-q),n))
    return dither+DELTA*(2*k+bit),k

def packet(identity,b):
    word=CODEWORDS[identity]
    result=[None]*48
    for i,l in enumerate(PILOT_CHIPS):result[l]=b.pilot[i]
    for i,l in enumerate(DATA_CHIPS):result[l]=word[i]^b.mask[i]
    return tuple(result)

def projections(blocks,b):
    if len(blocks)!=48 or any(len(x)!=64 for x in blocks):raise ValueError('48 disjoint64-coordinate blocks required')
    if not all(finite(x) for row in blocks for x in row):raise ValueError('finite blocks required')
    return tuple(math.fsum(x*u for x,u in zip(row,direction)) for row,direction in zip(blocks,b.vectors))

def embed_packet(blocks,physical_bits,b):
    """Low-level bit-packet primitive; callers own metadata semantics."""
    q=projections(blocks,b)
    if len(physical_bits)!=48:raise ValueError('48 symbols required')
    if any(fp32(x)!=x for row in blocks for x in row):raise ValueError('source blocks must represent actual FP32 values')
    changed=[];records=[]
    for l,(source,u,original,bit,dither) in enumerate(zip(blocks,b.vectors,q,physical_bits,b.dithers)):
        target,k=target_projection(original,bit,dither);coefficient=target-original
        result=tuple(fp32(x+coefficient*v) for x,v in zip(source,u))
        actual=tuple(y-x for x,y in zip(source,result))
        q_after=math.fsum(x*v for x,v in zip(result,u))
        changed.append(result)
        records.append(dict(chip=l,ROI=l//12,channel=4+l%12,bit=bit,k=k,
            q_before=original,target_projection=target,q_after_FP32=q_after,
            coefficient=coefficient,projection_FP32_error=q_after-target,
            ideal_energy=coefficient**2,actual_FP32_energy=math.fsum(x*x for x in actual),
            actual_Linf=max(abs(x) for x in actual)))
    return tuple(changed),dict(status='CONSTRUCTED_METADATA_REFERENCE',records=records,
        ideal_energy=math.fsum(r['ideal_energy'] for r in records),
        actual_FP32_energy=math.fsum(r['actual_FP32_energy'] for r in records),
        ideal_energy_bound=4.,ideal_ROI_bound=1.,actual_Linf=max(r['actual_Linf'] for r in records),
        FP32_materializations=1,FP32_correction_attempts=0,payload_rewritten=False,
        alpha=1.,delta=DELTA,authentication_claim=False)

def encode_label(blocks,label,key):
    if type(label) is not int or not 1<=label<=44:raise ValueError('writer label1..44 required')
    b=book(key);changed,receipt=embed_packet(blocks,packet(label-1,b),b)
    receipt['writer_label']=label
    return changed,receipt

def demodulate(q,dither):
    r=(q-dither)/DELTA
    if not math.isfinite(r):raise ValueError('nonfinite dimensionless projection')
    lo=math.floor(r);fraction=r-lo
    tolerance=32*sys.float_info.epsilon*max(1.,abs(r))
    if abs(fraction-.5)<=tolerance:return None,r
    return nearest_integer(r)%2,r

def decode_projections(q,b,tau):
    """No expected identity, phase, geometry, message, host or source inputs."""
    if tau is not None and (not finite(tau) or not 0<=tau<=1):raise ValueError('finite frozen pilot threshold0..1 or unresolved None')
    if len(q)!=48 or not all(finite(x) for x in q):raise ValueError('48 finite projections required')
    physical=[];scaled=[]
    for x,dither in zip(q,b.dithers):
        bit,r=demodulate(x,dither);physical.append(bit);scaled.append(r)
    P=math.fsum((1-2*b.pilot[i])*math.cos(math.pi*scaled[l]) for i,l in enumerate(PILOT_CHIPS))/16
    Hp=None if any(physical[l] is None for l in PILOT_CHIPS) else sum(physical[l]!=b.pilot[i] for i,l in enumerate(PILOT_CHIPS))
    data=tuple(None if physical[l] is None else physical[l]^b.mask[i] for i,l in enumerate(DATA_CHIPS))
    distances=[None]*64;nearest=None;distance=None;identity=None
    if None in data:code_status='SYMBOL_ERASURE'
    else:
        distances=[sum(x!=y for x,y in zip(data,c)) for c in CODEWORDS]
        distance=min(distances);nearest=[i for i,d in enumerate(distances) if d==distance]
        if len(nearest)!=1:code_status='CODE_TIE'
        elif distance>RADIUS:code_status='OUTSIDE_RADIUS'
        elif nearest[0]>=44:code_status='RESERVED_ID'
        else:code_status='LEGAL_ID';identity=nearest[0]
    detected=None if tau is None else Hp is not None and Hp<=PILOT_MAX_ERRORS and P>tau
    pilot_status='CALIBRATION_UNRESOLVED' if tau is None else 'PILOT_SYMBOL_ERASURE' if Hp is None else 'PILOT_MISMATCH' if Hp>PILOT_MAX_ERRORS else 'SOFT_BELOW_THRESHOLD' if P<=tau else 'PILOT_PASS'
    accepted=detected is True and code_status=='LEGAL_ID'
    status='CALIBRATION_UNRESOLVED' if tau is None else 'ACCEPTED_TAG_CANDIDATE' if accepted else 'DETECTED_LABEL_ERASURE' if detected else 'CODE_ONLY_REJECT' if code_status=='LEGAL_ID' else 'NO_DETECTION'
    return dict(status=status,pilot_P=P,pilot_H=Hp,pilot_detected=detected,pilot_status=pilot_status,
        physical_bits=physical,unmasked_bits=data,code_status=code_status,
        code_distances=distances,nearest_ids=nearest,nearest=(nearest[0] if nearest is not None and len(nearest)==1 else None),nearest_distance=distance,
        code_id_candidate=identity,tag_label=(identity+1 if accepted else None),
        code_radius=RADIUS,pilot_max_errors=PILOT_MAX_ERRORS,tau=tau,
        truth_used=False,message_used=False,authentication_claim=False,
        exact_geometry_certified=False)

def decode_blocks(blocks,key,tau):
    b=book(key);return decode_projections(projections(blocks,b),b,tau)

def null_domain():
    return {(c,v,p,r) for c in NULL_CLASSES for v in VIEWS for p in range(4) for r in range(1,11)}

def calibrate_complete_nulls(host_records):
    """All1200 pilot P per host; no Hp, valid-code or geometry screening."""
    result=dict(status='CALIBRATION_UNRESOLVED',tau=None,host_groups=2,
        expected_null_rows_per_host=1200,hosts={},ID_screening_used=False,Hp_screening_used=False,
        CFAR=False,population_FPR_guarantee=False)
    if len(host_records)!=2:
        result['reason']='TWO_SOURCE_GROUPS_REQUIRED';return result
    expected=null_domain()
    for host,records in host_records.items():
        keys=[(r.get('class'),r.get('view'),r.get('phase'),r.get('row')) for r in records]
        if len(records)!=1200 or len(set(keys))!=1200 or set(keys)!=expected:
            result['reason']='INCOMPLETE_FIXED_NULL_DOMAIN';return result
        if any(r.get('technical_failure',False) or not finite(r.get('P')) or not -1<=r['P']<=1 for r in records):
            result['reason']='NULL_NUMERIC_UNRESOLVED';return result
        maximum=max(r['P'] for r in records)
        result['hosts'][host]=dict(N=maximum,null_rows=1200,
            all_maxima=[dict(r) for r in records if r['P']==maximum])
    result.update(status='FROZEN',tau=max(0.,*(r['N'] for r in result['hosts'].values())))
    return result
