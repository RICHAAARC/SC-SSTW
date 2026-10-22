"""Isolated adopted full-only carrier family. No model, tensor, truth or I/O."""
from functools import cache
import hashlib,hmac,math
from . import b_detectable_time_label_v1 as core
from . import b_time_label_local_channel_v1 as legacy
FAMILIES=('RANDOM64','BLOCK2-16')
KEYS=legacy.KEYS
HOSTS=('b101-s01','b101-s03','b101-s02','b101-s04')
ARMS=('OFF','P0','RANDOM','BLOCK2')
LAYERS=('native','full')
TAU=0.4959127627330064
CARRIER_DOMAIN='B-TimeTag-QIM-Block2Carrier-V1'

@cache
def book(family,key):
    if family not in FAMILIES:raise ValueError('unknown carrier family')
    base=core.book(key)
    if family=='RANDOM64':return base
    vectors=[]
    for chip in range(48):
        def digest(s):
            return hmac.new(key.encode('utf-8'),(CARRIER_DOMAIN+f'|carrier|{chip}|{s}').encode('ascii'),hashlib.sha256).digest()
        positive=set(sorted(range(16),key=lambda s:(digest(s),s))[:8])
        vectors.append(tuple(1/8 if 4*(y//2)+x//2 in positive else -1/8 for y in range(8) for x in range(8)))
    return core.Book(tuple(vectors),base.dithers,base.pilot,base.mask)

def family_for(arm):
    if arm=='RANDOM':return ('RANDOM64',)
    if arm=='BLOCK2':return ('BLOCK2-16',)
    if arm in ('OFF','P0'):return FAMILIES
    raise ValueError('unknown arm')

def tables():
    return [(layer,h,arm,family,key) for layer in LAYERS for h in HOSTS for arm in ARMS for family in family_for(arm) for key in KEYS]

def initialize_books():
    for family in FAMILIES:
        for key in KEYS.values():book(family,key)
    rows=[]
    for family in FAMILIES:
        a,b=(book(family,KEYS[k]) for k in ('key00','key01'))
        for chip,(u,v) in enumerate(zip(a.vectors,b.vectors)):
            q=math.fsum(x*y for x,y in zip(u,v))
            rows.append(dict(family=family,chip=chip,dot=q,same_direction=u==v,opposite_direction=all(x==-y for x,y in zip(u,v))))
    return dict(rows=rows,dots_64D=96,coordinate_products=6144,selection=False,calibration=False,authentication=False)

def raw_table(projections,family,key):
    b=book(family,key);rows=[]
    for i in range(44):
        try:
            item=projections[i]
            if isinstance(item,dict) and item.get('error'):raise ValueError(item['error'])
            q=item['projections'] if isinstance(item,dict) else item
            d=core.decode_projections(q,b,None);d['technical_failure']=False
        except Exception as error:
            q=None;d=legacy.technical_row(f'{type(error).__name__}: {error}')
        rows.append(dict(row=i+1,projections=q,decoder=d))
    return dict(expected_rows=44,rows=rows,technical_rows=sum(r['decoder']['technical_failure'] for r in rows),truth_used=False,message_used=False)

def unavailable(reason):return legacy.unavailable_table(44,reason)
def finalize(table):return legacy.finalize(table,TAU)
