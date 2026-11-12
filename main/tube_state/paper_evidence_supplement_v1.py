"""Frozen evidence arithmetic and imaging-quality layout; no IO or query draws."""
from collections import Counter
import math
from . import paper_suite_v1 as prior
from .paper_fpr_claims_v1 import VERDICTS, uniform_query_union

SOURCES=prior.SOURCES
VIEWS=prior.VIEWS
METHODS=prior.METHODS
ARMS=('OFF','P0','NEW')
KEYS=('key00','key01')
INDICES=tuple(180*j//15 for j in range(16))
STREAMS=('main_OFF','main_P0','main_NEW',*METHODS)
TITLE='固定16帧采样的MUSIQ-SPAQ成像质量'
ROOT='/content/drive/MyDrive/Video-WM_Result'
QUALITY_BUDGET=dict(model_load=1,logical_streams=84,media_decode_max=84,frame_slots=1344,
    new_sources=0,new_attacks=0,new_queries=0,watermark_receiver=0,encoding=0,training=0)


def streams(root=ROOT):
    result=[]
    for sid in SOURCES:
        for name in STREAMS:
            relative=(f'main_method/paper_cohort_v1/{sid}/attempt01/media/input{dict(main_OFF=0,main_P0=5,main_NEW=10)[name]}/received.mp4'
                if name.startswith('main_') else f'baselines/wan-cohort-five-view-v1/{sid}/attempt01/media/{name}/FULL.mp4')
            result.append(dict(id=f'{sid}/{name}',source=sid,stream=name,view='FULL',media=root.rstrip('/')+'/'+relative,
                planned_indices=list(INDICES),planned_frames=16,expected_media_frames=181))
    return result


def family_counts(evidences,*,actual_message=None,arm):
    """The accept set is formed without accepting a query argument."""
    counted=prior.nominal_family(evidences)
    words=set(counted['known_words']);unknown=counted['unknown_inputs']
    states=Counter(prior.evidence_word(e)[1] for e in evidences)
    actual_valid=isinstance(actual_message,list) and len(actual_message)==32 and all(type(b) is int and b in (0,1) for b in actual_message)
    denominator=2**32 if arm=='OFF' else 2**32-1
    if arm=='OFF': filtered=words
    elif actual_valid: filtered=words-{''.join(map(str,actual_message))}
    else: filtered=None
    return dict(distinct_words=sorted(words),d=len(words),u=unknown,slot_states=dict(states),
        uniform32_lower=len(words)/2**32,uniform32_upper=min(10,len(words)+unknown)/2**32,
        uniform32_exact=len(words)/2**32 if unknown==0 else None,
        wrong_query_denominator=denominator,wrong_query_distinct=None if filtered is None else len(filtered),
        wrong_query_lower=0. if filtered is None else len(filtered)/denominator,
        wrong_query_upper=(10 if filtered is None else min(10,len(filtered)+unknown))/denominator,
        wrong_query_status='TRUTH_UNKNOWN' if filtered is None else 'BOUNDED' if unknown else 'EXACT_CONDITIONAL_MASS',
        probability_model='Q conditionally independent uniform given all frozen query-free evidence E',
        fixed_q0_q1_probability=None,query_draws_performed=0)


def protocol_bounds():
    return dict(model_families=36,model_claim_opportunities=360,new_query_draws=0,
        single_claim_upper=uniform_query_union(1),family10_upper=uniform_query_union(10),
        package_uniform32_upper=uniform_query_union(360),package_mixed_wrong_query_upper=360/(2**32-1),
        target_reference=1e-6,empirical_population_fpr=None,scientific_pass=None,
        scope='conditional random-query acceptance / conservative wrong-message bound; fixed q0/q1 are a separate event',
        execution_note='36 shared-query families are a probability model, not 36 executed random draws')


def clip_summary(stream,frames):
    by_index={r.get('frame_index'):r for r in frames if isinstance(r,dict)}
    valid=[];unknown=[]
    for index in INDICES:
        row=by_index.get(index,{})
        score=row.get('score')
        if row.get('status')=='RETURNED' and type(score) in (int,float) and math.isfinite(score): valid.append(float(score))
        else: unknown.append(index)
    complete=len(valid)==16 and len(frames)==16 and len(by_index)==16
    return dict(stream,status='COMPLETE' if complete else 'INCOMPLETE',valid_frames=len(valid),
        unknown_frames=16-len(valid),missing_indices=unknown,
        complete_mean=sum(valid)/16 if complete else None,partial_mean=sum(valid)/len(valid) if valid else None,
        score_scale='raw MUSIQ-SPAQ score, not VBench total score',scientific_pass=None)


def paired_scores(clips,baseline_references=()):
    lookup={(row['source'],row['stream']):row for row in clips};pairs=[]
    references={(r.get('source'),r.get('method')):r for r in baseline_references if isinstance(r,dict)}
    for sid in SOURCES:
        for left,right in [('main_NEW','main_P0'),('main_P0','main_OFF'),('main_NEW','main_OFF'),*((m,'main_OFF') for m in METHODS)]:
            a,b=lookup.get((sid,left)),lookup.get((sid,right));reference=references.get((sid,left))
            allowed=left.startswith('main_') or bool(reference and b and reference.get('off_media')==b['media'])
            complete=bool(allowed and a and b and a['status']==b['status']=='COMPLETE'
                and a['planned_indices']==b['planned_indices']==list(INDICES))
            pairs.append(dict(source=sid,left=left,right=right,complete=complete,
                difference=a['complete_mean']-b['complete_mean'] if complete else None,
                reference_status='DECLARED_MAIN_PAIR' if left.startswith('main_') else 'SAVED_REFERENCE_MATCH' if allowed else 'REFERENCE_UNKNOWN_OR_OUTSIDE_84',
                reason=None if complete else 'paired complete fixed indices and eligible actual OFF reference required'))
    return pairs


def distribution(values):
    values=sorted(float(v) for v in values if type(v) in (int,float) and math.isfinite(v))
    n=len(values)
    return dict(valid_count=n,mean=sum(values)/n if n else None,minimum=values[0] if n else None,
        maximum=values[-1] if n else None,median=(values[(n-1)//2]+values[n//2])/2 if n else None)
