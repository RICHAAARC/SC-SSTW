"""Fixed joint (tube start, age) packet A. Pure public algebra, no formal receiver.

Old framewise encode/decode retain their old j-domain semantics unchanged.
Exact batch decoding is for noiseless public codewords; the scalar raw24 entry
reuses the original GF/BCH kernel for bounded hard-error diagnostics.
"""
from functools import lru_cache
import numpy as np
from . import b_framewise_qim_v1 as legacy
from . import b_framewise_qim_time_recovery_v1 as old_time

N = 92672
LANES, COPIES, CODE_BITS, INFO_BITS = 5, 2, 63, 24
LEGAL_PAIRS = 5*N-10
MAX_LEGAL_MESSAGE = 8*(N-1)
ROIS, RMS, LINF = legacy.ROIS, legacy.RMS, legacy.LINF


@lru_cache(None)
def generator(): return legacy.generator()


def _integer(value, name, upper):
    if type(value) is not int or not 0 <= value < upper:
        raise ValueError(name+' must be an ordinary integer in [0,'+str(upper)+')')
    return value


def bits_to_word(bits):
    if len(bits) != 63 or any(type(b) is not int or b not in (0,1) for b in bits):
        raise ValueError('63 integer binary bits required')
    word=0
    for bit in bits: word=(word<<1)|bit
    return word


def word_to_bits(word):
    _integer(word,'63-bit word',1<<63)
    return [(word>>k)&1 for k in range(62,-1,-1)]


def _remainder(word):
    g=generator()
    while word.bit_length()>=40: word ^= g<<(word.bit_length()-40)
    return word


def encode_raw24(msg):
    _integer(msg,'raw24 message',1<<24)
    shifted=msg<<39
    return word_to_bits(shifted^_remainder(shifted))


def decode_raw24(bits):
    """Original t=7 BCH algebra before any application-domain interpretation."""
    word=bits_to_word(bits)
    if _remainder(word)==0:
        return dict(status='RAW24_BCH_CANDIDATE_UNCALIBRATED',msg=word>>39,corrected_positions=[])
    # Same Berlekamp-Massey/Chien kernel as the original diagnostic, using its GF.
    s=legacy.syndromes(bits);c=[1]+[0]*14;old=c.copy();degree=0;shift=1;last=1
    for n in range(14):
        discrepancy=s[n]
        for k in range(1,degree+1):discrepancy ^= legacy._mul(c[k],s[n-k])
        if not discrepancy:shift+=1;continue
        saved=c.copy();factor=legacy._div(discrepancy,last)
        for k in range(15-shift):c[k+shift] ^= legacy._mul(factor,old[k])
        if 2*degree<=n:degree,old,last,shift=n+1-degree,saved,discrepancy,1
        else:shift+=1
    errors=[]
    for power in range(63):
        value=0
        for k in range(degree+1):value ^= legacy._mul(c[k],legacy._powers()[(-power*k)%63])
        if value==0:errors.append(62-power)
    if degree>7 or len(errors)!=degree:return dict(status='UNRESOLVED_CODE',msg=None)
    corrected=list(bits)
    for k in errors:corrected[k]^=1
    if any(legacy.syndromes(corrected)):return dict(status='UNRESOLVED_CODE',msg=None)
    return dict(status='RAW24_BCH_CANDIDATE_UNCALIBRATED',msg=bits_to_word(corrected)>>39,corrected_positions=errors)


def _array(values, upper, name):
    a=np.asarray(values)
    if a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>=upper):
        raise ValueError(name+' requires bounded integer arrays')
    return a.astype(np.uint64,copy=False)


def encode_raw24_words(messages):
    messages=_array(messages,1<<24,'raw24 messages')
    code=messages<<np.uint64(39);rem=code.copy();g=generator()
    for power in range(62,38,-1):
        rem ^= ((rem>>np.uint64(power))&np.uint64(1))*np.uint64(g<<(power-39))
    return code^rem


def decode_raw24_words_exact(words):
    """Noiseless code membership plus systematic extraction, no error filling."""
    words=_array(words,1<<63,'codewords');rem=words.copy();g=generator()
    for power in range(62,38,-1):
        rem ^= ((rem>>np.uint64(power))&np.uint64(1))*np.uint64(g<<(power-39))
    return dict(msg=words>>np.uint64(39),code_valid=rem==0,remainder=rem)


def parse_raw24_arrays(messages, lanes):
    messages=_array(messages,1<<24,'raw24 messages');lanes=_array(lanes,5,'observed lanes')
    messages,lanes=np.broadcast_arrays(messages,lanes)
    j=messages>>np.uint64(3);age=messages&np.uint64(7);n=j+age
    invalid_age=age>4;invalid_j=j>=N;invalid_n=n>=N;wrong_lane=j%5!=lanes
    valid=~(invalid_age|invalid_j|invalid_n|wrong_lane)
    return dict(j=j,age=age,n=n,lane=lanes,valid=valid,invalid_age=invalid_age,
                invalid_j=invalid_j,invalid_n=invalid_n,wrong_lane=wrong_lane)


def parse_raw24(msg, lane):
    _integer(msg,'raw24 message',1<<24);_integer(lane,'observed lane',5)
    j,age=msg>>3,msg&7;n=j+age;violations=[]
    if age>4:violations.append('INVALID_AGE')
    if j>=N:violations.append('J_OUT_OF_DOMAIN')
    if n>=N:violations.append('N_OUT_OF_DOMAIN')
    if j%5!=lane:violations.append('LANE_MISMATCH')
    return dict(status='INVALID_PACKET' if violations else 'LEGAL_LOCAL_CANDIDATE',msg=msg,j=j,age=age,n=n,
                lane=lane,violations=violations,candidate_n=None if violations else n)


def message(j, age):
    _integer(j,'tube start',N);_integer(age,'age',5)
    if j+age>=N:raise ValueError('tube/age crosses public source domain; no wrap')
    return 8*j+age


def encode_packet(j, age):return encode_raw24(message(j,age))


def state_fields(indices):
    ns=_array(indices,N,'states').reshape(-1,1);lane=np.arange(5,dtype=np.uint64)[None,:]
    # Work signed for startup: negative j denotes unwritten, never word zero.
    js=ns.astype(np.int64)-((ns.astype(np.int64)-lane.astype(np.int64))%5)
    active=js>=0;ages=ns.astype(np.int64)-js
    return dict(j=js,age=ages,active=active)


def popcount(words):
    words=np.ascontiguousarray(words,dtype=np.uint64)
    return old_time.POPCOUNT[words.view(np.uint8).reshape(words.shape+(8,))].sum(axis=-1,dtype=np.uint16)


def gf2_rank(words):
    pivots={}
    for value in words:
        value=int(value)
        while value:
            pivot=value.bit_length()-1
            if pivot in pivots:value^=pivots[pivot]
            else:pivots[pivot]=value;break
    return len(pivots)


def parse_observations(observations):
    """Partial local diagnostics only. Per-lane union then cross-lane intersection.

    Invalid/erased copies do not erase other candidates. Copy conflicts survive
    in the per-lane output even when other observed lanes narrow the intersection.
    There is no presence decision, formal acceptance, or argmax.
    """
    records=[];seen=set();by_lane={lane:[] for lane in range(5)};decode_calls=0;parse_calls=0
    for index,obs in enumerate(observations):
        row=dict(observation_index=index,lane=obs.get('lane'),replica=obs.get('replica'),candidate_n=None)
        try:
            lane=_integer(obs['lane'],'lane',5);replica=_integer(obs['replica'],'replica',2)
            if (lane,replica) in seen:raise ValueError('duplicate physical copy slot')
            seen.add((lane,replica));bits=obs['bits'];known=obs['known']
            if len(bits)!=63 or len(known)!=63 or any(type(v) is not bool for v in known):
                raise ValueError('63 bits and boolean observation mask required')
            if not any(known):row.update(status='ERASURE',erasures=63)
            elif not all(known):row.update(status='PARTIAL_ERASURE_UNRESOLVED',erasures=63-sum(known))
            else:
                decode_calls+=1
                decoded=decode_raw24(bits);row['raw24']=decoded
                if decoded['msg'] is None:row['status']='UNRESOLVED_CODE'
                else:
                    parse_calls+=1;row.update(parse_raw24(decoded['msg'],lane))
            by_lane[lane].append(row)
        except (ValueError,KeyError,TypeError) as error:
            row.update(status='INVALID_OBSERVATION',error=type(error).__name__+': '+str(error))
        records.append(row)
    lanes=[];constraints=[];conflicts=[]
    for lane in range(5):
        candidates=sorted({v['candidate_n'] for v in by_lane[lane] if v['candidate_n'] is not None})
        status='COPY_CONFLICT' if len(candidates)>1 else 'LOCAL_CANDIDATE' if candidates else 'NO_VALID_CANDIDATE'
        if len(candidates)>1:conflicts.append(lane)
        if candidates:constraints.append(set(candidates))
        lanes.append(dict(lane=lane,status=status,candidate_n=candidates,observed_copies=len(by_lane[lane])))
    compatible=sorted(set.intersection(*constraints)) if constraints else None
    status=('NO_VALID_LOCAL_CANDIDATES' if compatible is None else 'CONSTRAINT_EMPTY_SET' if not compatible
            else 'STRUCTURAL_SINGLETON' if len(compatible)==1 else 'STRUCTURAL_AMBIGUITY')
    return dict(status=status,observations=records,lanes=lanes,copy_conflict_lanes=conflicts,compatible_n=compatible,
                raw24_decode_calls=decode_calls,packet_field_parse_calls=parse_calls,
                formal_output=None,presence_claim=False,claim_ceiling='Ideal local packet/constraint diagnostic only; not a calibrated receiver')
