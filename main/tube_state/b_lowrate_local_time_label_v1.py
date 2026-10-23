"""Fixed low-rate local tag: received-only RM, separate periodic pilot, new balanced PN."""
from functools import cache
import hashlib,hmac,math
import numpy as np
from .b_lfg_posthoc_reference_v1 import haar,inverse,lll
DOMAIN='B-Lowrate-Local-Time-Label-V1'
ALPHA=.15
KEYS={'key00':'watermark','key01':'watermark-wrong'}
VALID_IDS=tuple(range(1,22))

# Information integer: b0 intercept, b1..b5 slopes; x=0..31 with LSB-first x bits.
# Physical chip i indexes HMAC-ranked Walsh row i; mask_i is first digest byte LSB.
def word(i):
    if type(i) is not int or i not in range(64):raise ValueError('six-bit integer required')
    return [(i&1)^((sum(((i>>b)&1)*((x>>(b-1))&1) for b in range(1,6)))&1) for x in range(32)]
WORDS=tuple(tuple(word(i)) for i in range(64))

def digest(key,label):return hmac.new(key.encode(),(DOMAIN+'|'+label).encode(),hashlib.sha256).digest()
def rank(key,label,n):return sorted(range(n),key=lambda i:(digest(key,label+'|'+str(i)),i))

@cache
def book(key):
    if not isinstance(key,str) or not key:raise ValueError('nonempty key')
    result={}
    for role,order,n,ch in [('pilot',32,16,range(4,8)),('data',64,32,range(8,16))]:
        rows=[i+1 for i in rank(key,role+'|rows',order-1)[:n]]
        base=np.array([[1 if (int(r&x).bit_count())%2==0 else -1 for x in range(order)] for r in rows],dtype=np.int8)
        base=np.tile(base,(1,640//order));pn=[]
        for c in ch:pn.append(base[:,rank(key,role+'|channel'+str(c)+'|position',640)])
        result[role]=np.stack(pn,axis=1) # N,C,640, channelwise balanced/orthogonal
    result['mask']=[digest(key,'data|mask|'+str(i))[0]&1 for i in range(32)]
    order=rank(key,'pilot|balanced',16);result['pilot_bits']=[int(i in order[:8]) for i in range(16)]
    return result

def initialize():
    for key in KEYS.values():book(key)
    return dict(keys=2,HMAC=15644,per_key_HMAC=7822,int8_PN_elements_per_key=204800,PN='nonDC Walsh rows, keyed row/channel permutations',statistical_independence_claim=False)

def cost(v):
    a=np.asarray(v,dtype=np.float64);e=float(np.square(a).sum())
    return dict(energy=e,L2=math.sqrt(e),RMS=math.sqrt(e/a.size),Linf=float(np.max(np.abs(a))))

def construct(source,key):
    z=np.asarray(source)
    if z.shape!=(16,46,40,64) or z.dtype!=np.float32 or not np.isfinite(z).all():raise ValueError('actual normalized finite FP32[16,46,40,64]')
    b=book(key);bands=haar(z);low=bands['LLL'];sigma=low.std(axis=(1,2,3),ddof=0)
    delta=np.zeros_like(low)
    for g in VALID_IDS:
        bits=[a^m for a,m in zip(word(g),b['mask'])]
        pilot=[x^(g%2) for x in b['pilot_bits']]
        for role,cs,values in [('pilot',slice(4,8),pilot),('data',slice(8,16),bits)]:
            w=np.tensordot(2*np.asarray(values,dtype=np.float64)-1,b[role].astype(np.float64),axes=(0,0))/math.sqrt(len(values))
            delta[cs,g]=ALPHA*sigma[cs,None,None]*w.reshape(-1,20,32)
    lifted=inverse({name:delta if name=='LLL' else np.zeros_like(low) for name in bands})
    marked=z.copy();marked[4:16,2:44]=(z[4:16,2:44].astype(np.float64)+lifted[4:16,2:44]).astype(np.float32)
    if not np.isfinite(marked).all():raise ValueError('nonfinite marked FP32')
    actual=marked.astype(np.float64)-z;after=haar(marked)
    return marked,dict(status='CONSTRUCTED',alpha=ALPHA,sigma_channels=sigma.tolist(),ideal_budget_energy=float(ALPHA**2*21*640*np.square(sigma[4:]).sum()),ideal_LLL_delta=cost(delta),actual_FP32_delta=cost(actual),actual_LLL_delta=cost(after['LLL']-low),pilot_ideal_LLL_delta=cost(delta[4:8]),data_ideal_LLL_delta=cost(delta[8:16]),pilot_actual_FP32_delta=cost(actual[4:8]),data_actual_FP32_delta=cost(actual[8:16]),Haar_full_forward_calls=2,Haar_inverse_calls=1,other7_FP32_drift={n:cost(after[n]-bands[n]) for n in bands if n!='LLL'},FP32_materialization_error=cost(actual-lifted),channel_costs=[cost(v) for v in actual],payload_channels_unchanged=bool(np.array_equal(marked[:4].view(np.uint32),z[:4].view(np.uint32))),boundary_times_unchanged=bool(np.array_equal(marked[:,[0,1,44,45]].view(np.uint32),z[:,[0,1,44,45]].view(np.uint32))),source_unchanged=True,FP32_materializations=1,correction_attempts=0,new_user_payload=False)

def observe(low,key,rows):
    low=np.asarray(low,dtype=np.float64)
    if low.ndim!=4 or low.shape[0]!=16 or low.shape[2:]!=(20,32) or not np.isfinite(low).all() or any(type(j) is not int or not 0<=j<low.shape[1] for j in rows):raise ValueError('finite received LLL[16,T,20,32] and public rows')
    b=book(key);out=[]
    for j in rows:
        p=np.tensordot(b['pilot'].astype(np.float64),low[4:8,j].reshape(4,640),axes=((1,2),(0,1)))/2560
        d=np.tensordot(b['data'].astype(np.float64),low[8:16,j].reshape(8,640),axes=((1,2),(0,1)))/5120
        if not np.isfinite(p).all() or not np.isfinite(d).all():raise ValueError('nonfinite correlations')
        out.append(dict(row=j,status='OBSERVED',pilot_scores=p.tolist(),data_scores=d.tolist()))
    return dict(status='OBSERVED',rows=out,truth_inputs=False,message_inputs=False)

def unavailable(rows,reason='NOT_ENTERED'):
    return dict(status='TECHNICAL_UNRESOLVED',error=reason,rows=[dict(row=j,status='TECHNICAL_UNRESOLVED',pilot_scores=[None]*16,data_scores=[None]*32) for j in rows],truth_inputs=False,message_inputs=False)

def validate_raw(table,rows):
    if [r['row'] for r in table['rows']]!=list(rows):raise ValueError('fixed row geometry')
    for r in table['rows']:
        if len(r['pilot_scores'])!=16 or len(r['data_scores'])!=32:raise ValueError('48 score slots')
        if r['status']=='OBSERVED' and not all(type(x) in (int,float) and math.isfinite(x) for x in r['pilot_scores']+r['data_scores']):raise ValueError('observed scores must be finite')
    if table['status']=='OBSERVED' and any(r['status']!='OBSERVED' for r in table['rows']):raise ValueError('inconsistent table availability')
    return table

def decode_data(scores,key):
    out=dict(status='TECHNICAL_UNRESOLVED',raw_bits=[None]*32,unmasked_bits=[None]*32,distances=None,nearest_ids=[],nearest_distance=None,decoded_id=None,RM_success=False,authentication=False)
    if len(scores)!=32:raise ValueError('32 data scores')
    if any(x is None for x in scores):return out
    if not all(type(x) in (int,float) and math.isfinite(x) for x in scores):raise ValueError('finite data scores')
    raw=[int(x>0) if x!=0 else None for x in scores];out['raw_bits']=raw
    if None in raw:out['status']='ERASURE';return out
    bits=[x^m for x,m in zip(raw,book(key)['mask'])];dist=[sum(x!=y for x,y in zip(bits,w)) for w in WORDS];d=min(dist);ids=[i for i,x in enumerate(dist) if x==d]
    out.update(unmasked_bits=bits,distances=dist,nearest_ids=ids,nearest_distance=d)
    if len(ids)!=1:out['status']='TIE'
    elif d>7:out['status']='OUTSIDE_RADIUS'
    elif ids[0] not in VALID_IDS:out['status']='RESERVED_ID'
    else:out.update(status='VALID_RM',decoded_id=ids[0],RM_success=True)
    return out

def pilot(row,key,parity):
    scores=row['pilot_scores'];expected=[x^parity for x in book(key)['pilot_bits']]
    if row['status']!='OBSERVED' or any(x is None for x in scores):return dict(status='TECHNICAL_UNRESOLVED',passed=False,mismatches=None,erasures=None)
    signed=[x*(2*b-1) for x,b in zip(scores,expected)]
    return dict(status='PILOT_PRESENT' if all(x>0 for x in signed) else 'PILOT_REFUSED',passed=all(x>0 for x in signed),mismatches=sum(x<0 for x in signed),erasures=sum(x==0 for x in signed),signed_scores=signed)

def final_row(raw,decoded,key,parity):
    evidence=pilot(raw,key,parity);i=decoded['decoded_id'];clock=i is not None and (i-raw['row'])%2==(parity-raw['row'])%2
    accepted=evidence['passed'] and decoded['RM_success'] and clock
    return dict(row=raw['row'],status='ACCEPTED' if accepted else 'REFUSED',accepted=accepted,accepted_id=i if accepted else None,pilot=evidence,data=decoded,clock_consistent=clock)

def canonical(table,key):
    validate_raw(table,VALID_IDS)
    return dict(status=table['status'],rows=[final_row(r,decode_data(r['data_scores'],key),key,r['row']%2) for r in table['rows']],truth_inputs=False,message_inputs=False,authentication=False)

def bind_family(tables,key):
    if set(tables)!=set(range(8)):raise ValueError('all8 phase slots required')
    decoded={};hyp=[];missing=[]
    for p in range(8):
        validate_raw(tables[p],range(1,5));decoded[p]=[decode_data(r['data_scores'],key) for r in tables[p]['rows']]
        if tables[p]['status']!='OBSERVED':missing.append(p)
        for h in range(2):
            logical=[final_row(r,d,key,(r['row']+h)%2) for r,d in zip(tables[p]['rows'],decoded[p])]
            signed=[v for r in logical for v in r['pilot'].get('signed_scores',[])]
            statistic=sum(signed)/64 if len(signed)==64 else None
            hyp.append(dict(phase=p,clock=h,pilot_passes=sum(r['pilot']['passed'] for r in logical),pilot_statistic=statistic,rows=logical))
    maximum=max(r['pilot_statistic'] for r in hyp) if not missing else None;ties=[r for r in hyp if r['pilot_statistic']==maximum] if not missing else []
    selected=ties[0] if not missing and len(ties)==1 and ties[0]['pilot_passes']>=3 else None
    empty=[dict(row=j,status='NO_SELECTED_PHASE',accepted=False,accepted_id=None) for j in range(1,5)]
    result=dict(status='INCOMPLETE_PHASE_FAMILY' if missing else 'PILOT_PHASE_UNRESOLVED',hypotheses=hyp,maximum=maximum,max_ties=ties,missing_phases=missing,selected_phase=None,selected_clock=None,rows=empty,data_decodes={str(p):v for p,v in decoded.items()},provisional_origin=None,origin_proposals=[],truth_inputs=False,message_inputs=False,authentication=False,origin_status='CONDITIONAL_UNCALIBRATED',phase_statistic='all64 expected-sign times raw-correlation mean')
    if selected is None:return result
    p,h=selected['phase'],selected['clock'];rows=[final_row(r,d,key,(r['row']+h)%2) for r,d in zip(tables[p]['rows'],decoded[p])];origins=[8*(r['accepted_id']-r['row'])-p for r in rows if r['accepted']]
    result.update(status='ORIGIN_INSUFFICIENT' if len(origins)<2 else 'ORIGIN_CONFLICT',selected_phase=p,selected_clock=h,rows=rows,origin_proposals=origins)
    if len(origins)>=2 and len(set(origins))==1:
        if 0<=origins[0]<=129:result.update(status='PROVISIONAL_ORIGIN',provisional_origin=origins[0])
        else:result['status']='ORIGIN_OUT_OF_RANGE'
    return result
