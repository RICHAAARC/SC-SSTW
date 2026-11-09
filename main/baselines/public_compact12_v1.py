"""Frozen compact public measurement protocol; no runtime dependencies."""
import math
from .public_fpr001_v4 import (METHODS, BACKENDS, SHAPE, SEED, THRESHOLDS, VERIFY_BITS,
    views as original_views, validate_sources as original_sources)

EXPERIMENT='public-native-frameindex-compact12-v1'
OUTPUT='/content/drive/MyDrive/Video-WM_Result/baselines/'+EXPERIMENT+'/attempt01'
CACHE='/content/drive/MyDrive/Video-WM_Cache/baseline-compact12-v1'
OLD_OUTPUT='/content/drive/MyDrive/Video-WM_Result/baselines/public-native-and-frameindex-fpr001-v4/attempt01'
OLD_CACHE='/content/drive/MyDrive/Video-WM_Cache/baseline-v4'
VIEW_IDS=('G1/T00','G2/T00','G2/T01','G2/T02','G2/T03','G2/T04','G2/T06','G2/T08','G2/T10','G2/T14','S/S00','S/S02')


def views():
    lookup={v['id']:v for v in original_views()}
    return [dict(lookup[k]) for k in VIEW_IDS]


def selected_sources(roster):
    original_sources(roster)
    return [s for s in roster if 65<=s['global_ordinal']<=96 or s['split']=='TEST']


def validate_config(value):
    fixed=dict(experiment=EXPERIMENT,shape=SHAPE,fps=25,methods=list(METHODS),
        cal_sources=32,test_sources=500,dev_execution_sources=0,dev_reference_sources=64,
        view_ids=list(VIEW_IDS),edit_seed=SEED,pack_size=16,thresholds=THRESHOLDS,
        bootstrap_replicates=2000,calibration='independent null sanity; no fitted thresholds')
    for key,expected in fixed.items():
        if value.get(key)!=expected:raise ValueError('fixed compact method setting '+key+' requires '+repr(expected))
    return value


def family_event(rows,robust=False):
    if len(rows)!=12:raise ValueError('all twelve fixed rows required')
    if robust:
        if any(r['decision'] is False and r['complete'] for r in rows):return False
        return True if all(r['decision'] is True and r['complete'] for r in rows) else None
    if any(r['decision'] is True for r in rows):return True
    return False if all(r['decision'] is False and r['complete'] for r in rows) else None


def nominal_bounds():
    result={}
    for m in METHODS:
        n=VERIFY_BITS[m];candidates=1838 if m in ('wam','frameindex') else 12
        result[m]=dict(bits=n,threshold=THRESHOLDS[m],candidates=candidates,
            upper=candidates*sum(math.comb(n,j) for j in range(THRESHOLDS[m],n+1))/2**n,
            scope='one independent uniform query per source and method; fixed twelve views; no extracted-bit iid assumption')
    return result


def physical_budget():
    base=dict(download=532,source_probe=532,normalize=532,video_encode=30384,video_probe=30384,
        video_decode=42088,jpeg_encode=0,jpeg_decode=0,detect_wam=243552,
        detect_pixelseal=98048,detect_videoseal=66048,embed_wam=11500,
        embed_pixelseal=500,embed_videoseal=1000,embed_frameindex=11500,
        lpips=72000,model_load=102,lpips_load=1)
    recovery={k:math.ceil(v*.02) for k,v in base.items()};recovery.update(model_load=192,lpips_load=64)
    return dict(base=base,recovery=recovery,extra_per_unit=2,extra_runtimes=64,
        model_loads_base=102,extra_loads_per_runtime=3,lpips_loads_per_runtime=1,
        physical_views=43152,logical_views=49536,detector_forwards=407648,detector_frames=6609448,
        frames_per_unit=1838,edges_per_unit=1826,quality_frame_pairs=1086000,
        quality_stages=6000,lpips_alex_forwards=144000,DiT=0,Wan_generation=0,VAE=0,inversion=0,training=0,
        reuse='imported old work is separately recorded; these are maximum new calls before reuse')
