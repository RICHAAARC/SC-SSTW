"""Formula-reference BCH(512,432,t8), standard shortening of binary (1023,943).
Pure stdlib, MSB-first; no message truth, soft-list search or authentication.
"""
from functools import cache
N=1023
T=8
FIELD_POLY=0x409
INFO_BITS=432
CODE_BITS=512
PARITY_BITS=80
SHORTENED=511

@cache
def _field():
    exponent=[];log=[-1]*1024;x=1
    for i in range(N):
        if log[x]!=-1:raise ValueError('GF polynomial does not give primitive cycle')
        log[x]=i;exponent.append(x);x<<=1
        if x&1024:x^=FIELD_POLY
    if x!=1 or len(set(exponent))!=N:raise ValueError('invalid GF cycle')
    return tuple(exponent+exponent),tuple(log)

def _mul(a,b):
    if not a or not b:return 0
    e,l=_field();return e[l[a]+l[b]]

def _div(a,b):
    if not b:raise ZeroDivisionError('GF division by zero')
    if not a:return 0
    e,l=_field();return e[(l[a]-l[b])%N]

@cache
def generator():
    e,_=_field();roots=set()
    for j in range(1,2*T+1):
        x=j
        while x not in roots:roots.add(x);x=2*x%N
    polynomial=[1] # GF coefficients in ascending order
    for j in sorted(roots):
        q=[0]*(len(polynomial)+1)
        for i,c in enumerate(polynomial):q[i]^=_mul(c,e[j]);q[i+1]^=c
        polynomial=q
    if len(polynomial)!=81 or any(x not in (0,1) for x in polynomial):raise ValueError('BCH generator degree/field invalid')
    return sum(v<<i for i,v in enumerate(polynomial))

def _remainder(value,divisor):
    while value and value.bit_length()>=divisor.bit_length():value^=divisor<<(value.bit_length()-divisor.bit_length())
    return value

def bits_to_int(bits):
    value=0
    for bit in bits:
        if type(bit) is not int or bit not in (0,1):raise ValueError('binary Python integer bits required')
        value=(value<<1)|bit
    return value

def int_to_bits(value,n):return [(value>>(n-1-i))&1 for i in range(n)]

def encode(message):
    if not isinstance(message,bytes) or len(message)!=54:raise ValueError('exact54-byte ordinary test message required')
    # 511 leading zero mother information symbols are shortened, never transmitted.
    shifted=int.from_bytes(message,'big')<<PARITY_BITS
    return int_to_bits(shifted^_remainder(shifted,generator()),CODE_BITS)

def _syndromes(bits):
    e,_=_field();s=[]
    for j in range(1,2*T+1):
        q=0
        for bit in bits:q=_mul(q,e[j])^bit
        s.append(q)
    return s

def _locator(syndromes):
    C=[1]+[0]*(2*T);B=C[:];L=0;shift=1;previous=1
    for n in range(2*T):
        discrepancy=syndromes[n]
        for i in range(1,L+1):discrepancy^=_mul(C[i],syndromes[n-i])
        if not discrepancy:shift+=1;continue
        saved=C[:];ratio=_div(discrepancy,previous)
        for j in range(2*T+1-shift):C[j+shift]^=_mul(ratio,B[j])
        if 2*L<=n:L=n+1-L;B=saved;previous=discrepancy;shift=1
        else:shift+=1
    return C[:L+1],L

def decode(bits):
    if len(bits)!=CODE_BITS:raise ValueError('512 received coded bits required')
    bits=list(bits);bits_to_int(bits);syndromes=_syndromes(bits)
    def failed(status,**kw):
        return dict(status=status,BCH_success=False,raw_codeword_bits=bits,information_bits=[None]*INFO_BITS,corrected_codeword_bits=None,correction_count=None,syndromes=syndromes,authentication=False,**kw)
    locator,degree=_locator(syndromes)
    if degree>T:return failed('BCH_UNCORRECTABLE',locator=locator,locator_degree=degree)
    roots=[];e,_=_field()
    if degree:
        for power in range(N):
            z=e[(-power)%N];q=0
            for c in reversed(locator):q=_mul(q,z)^c
            if q==0:roots.append(power)
    if len(roots)!=degree or any(x>=CODE_BITS for x in roots):return failed('BCH_ROOTS_OR_SHORTENING_UNRESOLVED',locator=locator,locator_degree=degree,root_powers=roots)
    positions=sorted(CODE_BITS-1-x for x in roots);corrected=bits[:]
    for i in positions:corrected[i]^=1
    if any(_syndromes(corrected)) or _remainder(bits_to_int(corrected),generator()):return failed('BCH_POSTCHECK_FAILED',locator=locator,locator_degree=degree)
    information=corrected[:INFO_BITS];message=bits_to_int(information).to_bytes(54,'big')
    return dict(status='BCH_SUCCESS',BCH_success=True,raw_codeword_bits=bits,information_bits=information,corrected_codeword_bits=corrected,correction_count=len(positions),correction_positions=positions,syndromes=syndromes,locator=locator,locator_degree=degree,message_hex=message.hex(),authentication=False)

def unavailable(reason):return dict(status='TECHNICAL_UNRESOLVED',error=reason,BCH_success=None,raw_codeword_bits=[None]*512,information_bits=[None]*432,corrected_codeword_bits=None,correction_count=None,authentication=False)

def parameters():return dict(mother_n=N,mother_k=943,t=T,GF_polynomial=hex(FIELD_POLY),generator_degree=generator().bit_length()-1,shortened_leading_info=SHORTENED,code_bits=512,information_bits=432,bitorder='MSB-first',author_implementation=False)
