"""LFG public-formula Post-hoc K0 reference; normalized latent, even orthogonal Haar.
No runtime, writer-message fixture, model or historical QIM route import.
"""
from functools import cache
import hashlib,hmac,math
import numpy as np
N=512
ALPHA=0.15
DOMAIN='LFG-WM-Posthoc-Reference-V1'
KEYS={'key00':'watermark','key01':'watermark-wrong'}

def validate(value):
    z=np.asarray(value)
    if z.ndim!=4 or not z.shape[0] or any(d<=0 or d%2 for d in z.shape[1:]) or not np.isfinite(z).all():raise ValueError('finite[C,T,H,W] with positive even T/H/W required; no padding/cropping')
    return z.astype(np.float64,copy=False)

def haar(z):
    z=validate(z);bands={'':z}
    for axis in (1,2,3):
        nxt={}
        for name,v in bands.items():
            a=np.take(v,range(0,v.shape[axis],2),axis=axis);b=np.take(v,range(1,v.shape[axis],2),axis=axis)
            nxt[name+'L']=(a+b)/math.sqrt(2);nxt[name+'H']=(a-b)/math.sqrt(2)
        bands=nxt
    return bands

def inverse(bands):
    if set(bands)!={a+b+c for a in 'LH' for b in 'LH' for c in 'LH'}:raise ValueError('eight Haar subbands required')
    current={k:np.asarray(v,dtype=np.float64) for k,v in bands.items()}
    if len({v.shape for v in current.values()})!=1 or any(not np.isfinite(v).all() for v in current.values()):raise ValueError('matching finite Haar subbands required')
    for axis in (3,2,1):
        nxt={}
        for name in sorted({k[:-1] for k in current}):
            low,high=current[name+'L'],current[name+'H'];shape=list(low.shape);shape[axis]*=2;v=np.empty(shape,dtype=np.float64)
            a=[slice(None)]*4;b=a[:];a[axis]=slice(0,None,2);b[axis]=slice(1,None,2)
            v[tuple(a)]=(low+high)/math.sqrt(2);v[tuple(b)]=(low-high)/math.sqrt(2);nxt[name]=v
        current=nxt
    return current['']

def lll(z):
    v=validate(z)
    for axis in (1,2,3):v=(np.take(v,range(0,v.shape[axis],2),axis=axis)+np.take(v,range(1,v.shape[axis],2),axis=axis))/math.sqrt(2)
    return v

@cache
def packed_book(key,shape):
    if not isinstance(key,str) or not key or len(shape)!=4 or any(type(d) is not int or d<=0 for d in shape):raise ValueError('nonempty key and4D positive LLL shape required')
    C,T,H,W=shape;size=math.prod(shape);rows=[]
    for i in range(N):
        label=f'{DOMAIN}|C{C}|T{T}|H{H}|W{W}|i{i}'.encode('ascii')
        seed=hmac.new(key.encode('utf-8'),label,hashlib.sha256).digest()
        rows.append(hashlib.shake_256(seed).digest((size+7)//8))
    return tuple(rows)

def pn_rows(key,shape):
    size=math.prod(shape)
    for packed in packed_book(key,tuple(shape)):
        yield (2*np.unpackbits(np.frombuffer(packed,dtype=np.uint8),bitorder='big')[:size].astype(np.int8)-1).reshape(shape)

def initialize_books(shape=(16,23,20,32)):
    for key in KEYS.values():packed_book(key,tuple(shape))
    return dict(shape=list(shape),N=N,M=math.prod(shape),keys=2,HMAC=1024,SHAKE=1024,packed_bytes=2*N*((math.prod(shape)+7)//8),selection=False)

def spread(bits,key,shape):
    if len(bits)!=N or any(type(x) is not int or x not in (0,1) for x in bits):raise ValueError('512 physical coded bits required')
    W=np.zeros(shape,dtype=np.float64)
    for bit,pn in zip(bits,pn_rows(key,tuple(shape))):W+=(2*bit-1)*pn
    return W/math.sqrt(N)

def construct(source,bits,key):
    if np.asarray(source).dtype!=np.float32:raise ValueError('actual FP32 source required')
    z=validate(source);bands=haar(z);low=bands['LLL'];means=low.mean(axis=(1,2,3),keepdims=True)
    sigma=np.sqrt(((low-means)**2).mean(axis=(1,2,3),keepdims=True))
    W=spread(bits,key,low.shape);ideal_delta=ALPHA*sigma*W
    changed=dict(bands);changed['LLL']=low+ideal_delta
    ideal=inverse(changed);marked=ideal.astype(np.float32)
    if not np.isfinite(marked).all():raise ValueError('nonfinite FP32 marked latent')
    actual=marked.astype(np.float64)-z;error=marked.astype(np.float64)-ideal
    actual_bands=haar(marked)
    def cost(v):return dict(energy=float(np.square(v).sum()),L2=float(np.sqrt(np.square(v).sum())),RMS=float(np.sqrt(np.square(v).mean())),Linf=float(np.max(np.abs(v))))
    return marked,dict(status='CONSTRUCTED_LFG_POSTHOC_REFERENCE',alpha=ALPHA,K=0,N=N,M=low.size,LLL_shape=list(low.shape),coordinate='normalized Wan latent',sigma_ddof=0,LLL_channel_means=means.reshape(-1).tolist(),sigma_channels=sigma.reshape(-1).tolist(),zero_sigma_channels=np.where(sigma.reshape(-1)==0)[0].tolist(),
        ideal_LLL_delta=cost(ideal_delta),actual_LLL_delta=cost(actual_bands['LLL']-bands['LLL']),actual_other7_drift={name:cost(actual_bands[name]-bands[name]) for name in bands if name!='LLL'},actual_FP32_delta=cost(actual),FP32_materialization_error=cost(error),channel_costs=[cost(v) for v in actual],
        original_payload_channels_cost=cost(actual[:4]),guard0_cost=cost(actual[:,0]),guard45_or_last_cost=cost(actual[:,-1]),
        Haar_full_forward_calls=2,Haar_inverse_calls=1,FP32_materializations=1,correction_attempts=0,other7_ideal_subbands_preserved=True,payload_channels_forced_unchanged=False,guards_forced_unchanged=False,W_normalization='1/sqrt(512)',authentication=False)

def score_from_lll(low,key):
    low=np.asarray(low,dtype=np.float64)
    if low.ndim!=4 or not np.isfinite(low).all():raise ValueError('finite4D received LLL required')
    values=[float(np.sum(low*pn,dtype=np.float64)/low.size) for pn in pn_rows(key,low.shape)]
    if not all(math.isfinite(x) for x in values):raise ValueError('nonfinite correlations')
    return dict(status='CORRELATIONS_SAVED',scores=values,raw_codeword_bits=[int(x>0) for x in values],zero_scores=sum(x==0 for x in values),N=N,M=low.size,LLL_shape=list(low.shape),coordinate='normalized Wan latent',decision='r>0 is1 else0',truth_inputs=False,message_inputs=False,authentication=False)

def unavailable(reason):return dict(status='TECHNICAL_UNRESOLVED',error=reason,scores=[None]*512,raw_codeword_bits=[None]*512,zero_scores=None,N=N,truth_inputs=False,message_inputs=False,authentication=False)
