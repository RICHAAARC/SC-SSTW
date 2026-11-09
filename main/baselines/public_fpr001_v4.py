"""Adopted UCF/40-view random-query protocol. Pure method/plan functions only."""
from collections import Counter
from itertools import permutations
import math
import re

EXPERIMENT = 'public-native-and-frameindex-fpr001-v4'
METHODS = dict(wam=32, pixelseal=256, videoseal=256, frameindex=256)
BACKENDS = ('wam', 'pixelseal', 'videoseal')
SHAPE = [181, 320, 512, 3]
SEED = 2026110601
LENGTHS = (181,129,129,129,129,163,127,163,127,199,199,181,181,199,145,227)
THRESHOLDS = dict(wam=30, pixelseal=161, videoseal=161, frameindex=165)
VERIFY_BITS = dict(wam=32, pixelseal=256, videoseal=256, frameindex=248)
OUTPUT = '/content/drive/MyDrive/Video-WM_Result/baselines/'+EXPERIMENT+'/attempt01'
CACHE = '/content/drive/MyDrive/Video-WM_Cache/baseline-v4'


def views():
    return ([dict(id=f'{generation}/T{i:02}', operation=f'T{i:02}', generation=generation, frames=n)
             for i,n in enumerate(LENGTHS) for generation in ('G1','G2')]
            + [dict(id=f'S/S{i:02}', operation=f'S{i:02}', generation='G2', frames=181) for i in range(8)])


def batches(n, size): return [(start,min(size,n-start)) for start in range(0,n,size)]
def backend(method): return 'pixelseal' if method=='frameindex' else method
def read_batch(method): return 8 if backend(method)=='wam' else 32
def writer_batches(method): return batches(181,8) if method in ('wam','frameindex') else batches(181,128 if method=='videoseal' else 181)


def validate_config(c):
    expected = dict(experiment=EXPERIMENT,shape=SHAPE,fps=25,methods=list(METHODS),dev_sources=64,
        cal_sources=500,test_sources=500,pack_size=16,edit_seed=SEED,alpha=.001,
        thresholds=THRESHOLDS,bootstrap_replicates=2000)
    for key,value in expected.items():
        if c.get(key)!=value:raise ValueError(f'fixed v4 method setting {key} requires {value!r}')
    return c


def validate_sources(rows):
    if len(rows)!=1064:raise ValueError('fixed DEV64/CAL500/TEST500 roster required')
    ids,groups=set(),set()
    for i,s in enumerate(rows,1):
        split='DEV' if i<=64 else 'CAL' if i<=564 else 'TEST'
        group=(s['action'],int(s['group']))
        if (s['global_ordinal']!=i or s['split']!=split or not re.fullmatch(r'[A-Za-z0-9_-]+',s['id'])
                or s['id'] in ids or group in groups or int(s['header_frames'])<181 or float(s['fps'])!=25
                or min(int(s['width']),int(s['height']))<=0 or not s['url'].startswith('https://')):
            raise ValueError('invalid fixed roster/group/real-frame format: '+s['id'])
        ids.add(s['id']);groups.add(group)
    return rows


def allocate_queries(sources, random_bytes):
    """Caller supplies real random bytes; collisions are retained, never redrawn."""
    def bits(n):
        value=random_bytes((n+7)//8)
        if len(value)!=(n+7)//8:raise ValueError('random byte source returned wrong length')
        return [(byte>>bit)&1 for byte in value for bit in range(7,-1,-1)][:n]
    output={}
    for source in sources:
        q={method:bits(n) for method,n in METHODS.items() if method!='frameindex'}
        q['frameindex']=q['pixelseal'][:248]
        for method in METHODS:
            wrong=bits(VERIFY_BITS[method]) if source['split']!='CAL' else None
            output[method+'/'+source['id']]=dict(query=q[method],wrong_query=wrong,
                wrong_query_collision=(wrong==q[method]) if wrong is not None else None,
                frameindex_owner_from='pixelseal first248' if method=='frameindex' else None)
    return output


def frame_messages(owner, start, n):
    if len(owner)!=248 or start<0 or start+n>181:raise ValueError('248 owner bits and original0..180 indices required')
    return [list(owner)+[(index>>bit)&1 for bit in range(7,-1,-1)] for index in range(start,start+n)]


def source_plan(source, donors):
    import numpy as np
    ordinal=source['global_ordinal'];sid=source['id']
    def rng(op):return np.random.Generator(np.random.PCG64(np.random.SeedSequence([SEED,0,ordinal,op])))
    def target(index):return dict(kind='target',source_id=sid,index=int(index))
    a=int(rng(1).integers(0,50));drop=rng(5).permutation(181);c=int(rng(7).integers(27,155))
    b=int(rng(10).integers(0,164));blocks=[list(range(i,j)) for i,j in ((0,45),(45,90),(90,135),(135,181))]
    orders=[p for p in permutations(range(4)) if p!=(0,1,2,3)];order=orders[int(rng(11).integers(23))]
    candidates=[s['id'] for s in donors if s['id']!=sid]
    if not candidates:raise ValueError('independent DEV donor required')
    random=rng(12);donor=candidates[int(random.integers(len(candidates)))];begin=int(random.integers(164));donor_begin=int(random.integers(164))
    indices={0:list(range(181)),**{r+1:list(range(a+r,a+r+129)) for r in range(4)},
        5:[i for i in range(181) if i not in set(drop[:18])],6:[i for i in range(181) if i not in set(drop[:54])],
        7:[i for i in range(181) if not c-9<=i<c+9],8:[i for i in range(181) if not c-27<=i<c+27],
        9:[i for i in range(181) for _ in range(2 if i in set(drop[:18]) else 1)],
        10:list(range(b+18))+list(range(b,b+18))+list(range(b+18,181)),
        11:[i for k in order for i in blocks[k]],14:[5*j//4 for j in range(145)],15:[4*j//5 for j in range(227)]}
    mappings={f'T{k:02}':[target(i) for i in values] for k,values in indices.items()}
    mappings['T12']=[dict(kind='foreign',source_id=donor,index=donor_begin+i-begin) if begin<=i<begin+18 else target(i) for i in range(181)]
    gaps=set(int(x) for x in rng(13).choice(180,18,replace=False));interpolated=[]
    for i in range(181):
        interpolated.append(target(i))
        if i in gaps:interpolated.append(dict(kind='synthetic',source_id=sid,index=None,parents=[i,i+1],weights=[.5,.5]))
    mappings['T13']=interpolated
    for mapping in mappings.values():
        copies=Counter()
        for row in mapping:
            if row['index'] is not None:
                key=(row['source_id'],row['index']);row['copy_ordinal']=copies[key];copies[key]+=1
    return dict(source_id=sid,global_ordinal=ordinal,domain_id=0,seed_base=SEED,donor=donor,
        maps=mappings,trim_start=a,trim_mod8=[(a+r)%8 for r in range(4)],
        noise_seed_sequence=[SEED,0,ordinal,106],all_views=views())


def relation(left,right):
    if left is None or right is None:return 'UNKNOWN'
    if left['kind']=='synthetic' or right['kind']=='synthetic':return 'synthetic'
    if left['source_id']!=right['source_id']:return 'cross-source'
    delta=right['index']-left['index']
    return 'adjacent' if delta==1 else 'repeat' if delta==0 else 'gap' if delta>1 else 'reorder'


def verify(method, raw, query, *, planned_frames):
    """Post-raw query verification. No truth/map/clean pixels or score-selected retries."""
    import numpy as np
    observed=np.asarray(raw['observed'],dtype=bool);valid=np.asarray(raw['valid'],dtype=bool)
    bits=np.asarray(raw['bits']);q=np.asarray(query)
    if observed.shape!=(planned_frames,) or valid.shape!=observed.shape or bits.shape!=(planned_frames,METHODS[method]):
        raise ValueError('fixed received-frame slots required')
    if q.shape!=(VERIFY_BITS[method],) or not np.isin(q,[0,1]).all() or not np.isin(bits[observed & valid],[0,1]).all():
        raise ValueError('binary native/query bits required')
    complete=bool(observed.all());threshold=THRESHOLDS[method];frames=[]
    if method in ('pixelseal','videoseal'):
        score=None;video_bits=None
        if complete and bool(valid.all()):
            logits=np.asarray(raw['logits'])
            if logits.shape!=(planned_frames,257) or logits.dtype!=np.float32 or not np.isfinite(logits).all():
                raise ValueError('all original FP32 logits required for native mean')
            # Match native torch FP32 mean, in the original received order.
            import torch
            video_bits=(torch.from_numpy(logits[:,1:].copy()).mean(0)>0).to(torch.uint8).tolist()
            score=sum(a==b for a,b in zip(video_bits,query))
        return dict(decision=None if score is None else score>=threshold,score=score,score_lower_bound=None,
            complete=complete,accepted_from_partial=False,threshold=threshold,frames=frames,video_bits=video_bits)
    known=observed & valid;scores=(bits[:,:len(q)]==q).sum(1);accepted=known & (scores>=threshold)
    witness=np.flatnonzero(accepted).tolist();maximum=int(scores[known].max()) if known.any() else -1
    for i in range(planned_frames):
        owner=bool(accepted[i]) if observed[i] else None
        index=None;index_valid=None
        if method=='frameindex' and known[i]:
            decoded=sum(int(bit)<<(7-j) for j,bit in enumerate(bits[i,248:]))
            index_valid=decoded<=180
            if owner and index_valid:index=decoded
        frames.append(dict(received_frame=i,observed=bool(observed[i]),valid=bool(valid[i]) if observed[i] else None,
            owner_accept=owner,score=int(scores[i]) if known[i] else None,index=index,index_valid=index_valid))
    return dict(decision=True if witness else False if complete else None,complete=complete,
        accepted_from_partial=bool(witness) and not complete,score=maximum if complete else None,
        score_lower_bound=maximum if not complete else None,positive_witness_frames=witness,
        threshold=threshold,frames=frames,video_bits=None)


def family_event(rows):
    if any(row['decision'] is True for row in rows):return True
    return False if len(rows)==40 and all(row['complete'] and row['decision'] is False for row in rows) else None


def all_robust(rows):
    if any(row['complete'] and row['decision'] is False for row in rows):return False
    return True if len(rows)==40 and all(row['complete'] and row['decision'] is True for row in rows) else None


def nominal_bounds():
    return {m:dict(bits=VERIFY_BITS[m],candidates=6664 if m in ('wam','frameindex') else 40,
        threshold=THRESHOLDS[m],upper=(6664 if m in ('wam','frameindex') else 40)*
            sum(math.comb(VERIFY_BITS[m],j) for j in range(THRESHOLDS[m],VERIFY_BITS[m]+1))/2**VERIFY_BITS[m]) for m in METHODS}


def physical_budget():
    base=dict(download=1064,source_probe=1064,normalize=1064,video_encode=83000,video_probe=83000,
        video_decode=144032,jpeg_encode=600920,jpeg_decode=600920,
        detect_wam=1387056,detect_pixelseal=508544,detect_videoseal=377696,
        embed_wam=12972,embed_pixelseal=564,embed_videoseal=1128,embed_frameindex=12972,
        lpips=81216,model_load=204,lpips_load=1)
    recovery={k:math.ceil(v*.02) for k,v in base.items()};recovery.update(model_load=192,lpips_load=64)
    return dict(base=base,recovery=recovery,extra_per_unit=2,
        model_loads_base=204,extra_runtimes=64,extra_loads_per_runtime=3,lpips_loads_per_runtime=1,
        physical_views=217920,logical_views=260480,detector_forwards=2273296,detector_frames=36305472,
        frames_per_unit=6664,edges_per_unit=6624,quality_frame_pairs=1225008,lpips_alex_forwards=162432,
        video_decode_min_expected=137264,video_decode_upper=144032,video_streams=3320,
        DiT=0,Wan_generation=0,VAE=0,inversion=0,training=0)
