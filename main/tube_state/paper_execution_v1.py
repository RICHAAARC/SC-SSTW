"""Static six-worker ownership; scheduling does not add sources or calls."""
from collections import Counter
from .paper_suite_v1 import SOURCES
from main.baselines import public_compact12_v1 as public

MAIN_SHARDS=2
PUBLIC_SHARDS=4


def main_sources(shard):
    if type(shard) is not int or not 0<=shard<MAIN_SHARDS:raise ValueError('main shard must be0 or1')
    return tuple(SOURCES[shard::MAIN_SHARDS])


def public_packs(sources):
    packs=[]
    for split in ('CAL','TEST'):
        selected=[s for s in sources if s['split']==split]
        for start in range(0,len(selected),16):
            packs.append(dict(id=f'{split}-{start//16:02}',ordinal=len(packs),sources=selected[start:start+16]))
    return packs


def owned_packs(sources,shard):
    if type(shard) is not int or not 0<=shard<PUBLIC_SHARDS:raise ValueError('public shard must be0..3')
    return [p for p in public_packs(sources) if p['ordinal']%PUBLIC_SHARDS==shard]


def share(total,shard):return total//4+int(shard<total%4)


def inherited_spend(sources,events):
    owners={s['id']:pack['ordinal']%4 for pack in public_packs(sources) for s in pack['sources']}
    packowners={pack['id']:pack['ordinal']%4 for pack in public_packs(sources)}
    output=[dict(base=Counter(),recovery=Counter()) for _ in range(4)]
    for row in events:
        if row.get('status')=='UNREADABLE_EVENT':raise ValueError('existing compact physical ledger unreadable; consumed budget unknown')
        if row.get('status')!='STARTED':continue
        unit=row['unit'].split('/');operation=row['operation']
        if unit[0]=='loaders':owner=0 if row['category']=='lpips_load' else packowners[operation.split('/')[0]]
        else:owner=owners[unit[1]]
        output[owner]['recovery' if row['recovery'] else 'base'][row['category']]+=row['count']
    return output


def shard_budget(sources,shard,spent=None,parent_runtimes=0):
    packs=owned_packs(sources,shard);selected=[s for p in packs for s in p['sources']]
    c=sum(s['split']=='CAL' for s in selected);t=sum(s['split']=='TEST' for s in selected)
    base=dict(download=c+t,source_probe=c+t,normalize=c+t,video_encode=12*(c+5*t),video_probe=12*(c+5*t),
        video_decode=34*c+82*t,jpeg_encode=0,jpeg_decode=0,
        detect_wam=236*(c+2*t),detect_pixelseal=64*(c+3*t),detect_videoseal=64*(c+2*t),
        embed_wam=23*t,embed_pixelseal=t,embed_videoseal=2*t,embed_frameindex=23*t,
        lpips=144*t,model_load=3*len(packs),lpips_load=int(shard==0))
    global_limits=public.physical_budget();spent=spent or [dict(base={},recovery={}) for _ in range(4)]
    recovery={}
    for key,total in global_limits['recovery'].items():
        used=sum(s['recovery'].get(key,0) for s in spent)
        if used>total:raise ValueError('existing global recovery budget exhausted: '+key)
        recovery[key]=spent[shard]['recovery'].get(key,0)+share(total-used,shard)
    if parent_runtimes>65:raise ValueError('existing global runtime allowance exhausted')
    return dict(base=base,recovery=recovery,extra_per_unit=2,runtime_limit=share(65-parent_runtimes,shard),
        source_ids=[s['id'] for s in selected],pack_ids=[p['id'] for p in packs],
        inherited=dict(base=dict(spent[shard]['base']),recovery=dict(spent[shard]['recovery'])),
        global_method_limits=global_limits,shard=shard,shards=4,
        scope='disjoint full packs; all shard base/recovery limits sum to the original global budget')
