"""Fixed pilot and shortened BCH payload. Key PRF is coding, never an IO gate."""
from functools import cache
import hashlib
import hmac
import random

N=63;T=5;FIELD_POLY=0x43;INFO_BITS=32;CODE_BITS=59;PARITY_BITS=27
EVAL_KEYS={'key00':'local-vae-v1-eval-owner-00','key01':'local-vae-v1-eval-wrong-01'}
FILLER=[i%2 for i in range(160)]

def bits_to_int(bits):
    out=0
    for bit in bits:
        if type(bit) is not int or bit not in (0,1):raise ValueError('binary integer bits required')
        out=(out<<1)|bit
    return out

def int_to_bits(value,n):return [(value>>(n-1-i))&1 for i in range(n)]

@cache
def field():
    exp=[];log=[-1]*64;x=1
    for i in range(N):
        if log[x]!=-1:raise ValueError('nonprimitive field polynomial')
        log[x]=i;exp.append(x);x<<=1
        if x&64:x^=FIELD_POLY
    if x!=1 or len(set(exp))!=N:raise ValueError('invalid GF64 cycle')
    return tuple(exp+exp),tuple(log)

def mul(a,b):
    if not a or not b:return 0
    e,l=field();return e[l[a]+l[b]]

def div(a,b):
    if not b:raise ZeroDivisionError('GF division')
    if not a:return 0
    e,l=field();return e[(l[a]-l[b])%N]

@cache
def generator():
    e,_=field();roots=set()
    for j in range(1,11):
        x=j
        while x not in roots:roots.add(x);x=2*x%N
    p=[1]
    for j in sorted(roots):
        q=[0]*(len(p)+1)
        for i,c in enumerate(p):q[i]^=mul(c,e[j]);q[i+1]^=c
        p=q
    if len(p)!=28 or any(x not in (0,1) for x in p):raise ValueError('BCH degree27 required')
    return sum(v<<i for i,v in enumerate(p))

def remainder(x,g):
    while x and x.bit_length()>=g.bit_length():x^=g<<(x.bit_length()-g.bit_length())
    return x

def encode(message):
    if len(message)!=32:raise ValueError('32 information bits required')
    x=bits_to_int(message)<<27
    # Four leading zero mother information bits are shortened, not transmitted.
    return int_to_bits(x^remainder(x,generator()),59)

def syndromes(bits):
    e,_=field();result=[]
    for j in range(1,11):
        q=0
        for bit in bits:q=mul(q,e[j])^bit
        result.append(q)
    return result

def decode(bits):
    if len(bits)!=59:raise ValueError('59 coded bits required')
    bits=list(bits);bits_to_int(bits);s=syndromes(bits)
    C=[1]+[0]*10;B=C[:];L=0;shift=1;previous=1
    for n in range(10):
        discrepancy=s[n]
        for i in range(1,L+1):discrepancy^=mul(C[i],s[n-i])
        if not discrepancy:shift+=1;continue
        saved=C[:];ratio=div(discrepancy,previous)
        for j in range(11-shift):C[j+shift]^=mul(ratio,B[j])
        if 2*L<=n:L=n+1-L;B=saved;previous=discrepancy;shift=1
        else:shift+=1
    failed=dict(status='ERASURE',message=None,raw_bits=bits,corrected_bits=None,corrected_positions=[],authentication=False)
    if L>5:return failed
    e,_=field();roots=[]
    if L:
        for power in range(63):
            z=e[(-power)%63];q=0
            for c in reversed(C[:L+1]):q=mul(q,z)^c
            if q==0:roots.append(power)
    if len(roots)!=L or any(i>=59 for i in roots):return failed
    positions=sorted(58-i for i in roots);corrected=bits[:]
    for i in positions:corrected[i]^=1
    if len(positions)>5 or any(syndromes(corrected)) or remainder(bits_to_int(corrected),generator()):return failed
    if encode(corrected[:32])!=corrected:return failed
    return dict(status='OUTPUT',message=corrected[:32],raw_bits=bits,corrected_bits=corrected,
        corrected_positions=positions,information_observation=['ECC_RECONSTRUCTED' if i in positions else 'DIRECT_OBSERVED' for i in range(32)],authentication=False)

def rng(key,context):
    seed=hmac.new(key.encode(),context.encode(),hashlib.sha256).digest()
    return random.Random(int.from_bytes(seed,'big'))

@cache
def transform(key,context,n):
    r=rng(key,context);permutation=list(range(n));r.shuffle(permutation)
    mask=[r.randrange(2) for _ in range(n)]
    return tuple(permutation),tuple(mask)

def scramble(bits,key,context):
    p,mask=transform(key,context,len(bits))
    return [bits[p[i]]^mask[i] for i in range(len(bits))]

def unscramble_logits(values,key,block):
    if len(values)!=64 or type(block) is not int or block not in range(23):raise ValueError('64 logits and source block0..22 required')
    p,mask=transform(key,'payload/'+str(block),64);out=[None]*64
    for i in range(64):out[p[i]]=float(values[i])*(-1 if mask[i] else 1)
    return out[:59]

def pilot_word(label):
    if type(label) is not int or label not in range(256):raise ValueError('8-bit pilot label required')
    coefficients=[(label>>i)&1 for i in range(8)];word=[]
    for position in range(32):
        x=[(position>>i)&1 for i in range(5)]
        basis=[1,*x,(x[0]&x[1])^(x[2]&x[3]),(x[0]&x[2])^(x[1]&x[4])]
        word.append(sum(a*b for a,b in zip(coefficients,basis))%2)
    return word

@cache
def pilots(key):return tuple(tuple(scramble(pilot_word(t),key,'pilot')) for t in range(181))

def condition(message,key,frame):
    if type(frame) is not int or frame not in range(181):raise ValueError('legal source frame0..180 required')
    payload=scramble(encode(message)+[0]*5,key,'payload/'+str(frame//8))
    return list(pilots(key)[frame])+payload

def native_message(message,key,frame):return condition(message,key,frame)+FILLER

def parameters():
    return dict(pilot=[32,8,12],pilot_coefficient_order='LSB first label; positions x0 LSB',legal_frames=181,
        mother_BCH=[63,36,11],shortened_leading_information=4,payload=[59,32,11],parity=27,padding=5,t=5,
        BCH_bit_order='MSB first information then parity',condition_bits=96,native_bits=256,capacity=32)
