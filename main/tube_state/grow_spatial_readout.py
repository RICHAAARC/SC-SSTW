"""Fixed hard versus raw-soft scoring; no truth-dependent selection."""
import math

def score(aggregate):
    votes=aggregate['vote_sums'];means=aggregate['coefficient_mean_diagnostic'];n=aggregate['votes_per_bit']
    if n!=184 or len(votes)!=16 or len(means)!=16:raise ValueError('requires fixed 16-bit/184-vote carrier')
    if not all(math.isfinite(float(x)) for x in votes+means):raise ValueError('nonfinite score')
    sign=lambda x:(x>0)-(x<0)
    hard=[sign(x) for x in votes];soft=[sign(x) for x in means]
    if hard!=aggregate['signs']:raise ValueError('inconsistent saved hard votes')
    return dict(hard=dict(signs=hard,scores=[x/n for x in votes],erasures=hard.count(0)),
        soft=dict(signs=soft,scores=list(means),erasures=soft.count(0)),
        disagreement_bits=[i for i,(a,b) in enumerate(zip(hard,soft)) if a!=b],
        rule='both fixed threshold zero; no switching or calibrated presence decision')

def compare(scored,payload):
    if len(payload)!=16 or any(x not in (-1,1) for x in payload):raise ValueError('fixed signed payload required')
    return {name:dict(bit_errors_including_erasures=sum(a!=b for a,b in zip(scored[name]['signs'],payload)),
        exact=scored[name]['signs']==payload,erasures=scored[name]['erasures']) for name in ('hard','soft')}


def candidate_scores(scored,payloads):
    if len(payloads)!=2 or any(len(p)!=16 or any(x not in (-1,1) for x in p) for p in payloads):raise ValueError('fixed public A/B codebook required')
    result={}
    for decoder in ('hard','soft'):
        values=[sum(a*b for a,b in zip(scored[decoder]['scores'],p))/16 for p in payloads]
        tie=values[0]==values[1]
        result[decoder]=dict(candidate_scores=values,top=None if tie else (0 if values[0]>values[1] else 1),tie=tie,gap=abs(values[0]-values[1]),
            meaning='two-candidate continuous attribution, not 16-bit recovery or presence detection')
    return result
