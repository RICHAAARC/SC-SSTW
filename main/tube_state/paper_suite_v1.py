"""Frozen formal measurement scope. No models, files or truth-dependent selection."""
from . import b_paper_cohort_v1 as cohort
from .b_joint_time_relative_validation_v1 import construction_maps

SOURCES = tuple(s['id'] for s in cohort.ROSTER)
VIEWS = cohort.VIEWS
LENGTHS = cohort.LENGTHS
METHODS = ('wam', 'pixelseal', 'videoseal', 'frameindex')
BACKENDS = ('wam', 'pixelseal', 'videoseal')
ROOT = '/content/drive/MyDrive/Video-WM_Result'
COHORT_ROOT = ROOT + '/main_method/paper_cohort_v1'
PUBLIC_ROOT = ROOT + '/baselines/public-native-and-frameindex-fpr001-v4/attempt01'
WAN_ROOT = ROOT + '/baselines/wan-cohort-five-view-v1'
QUERY_ROOT = ROOT + '/main_method/random-query-family-v1/attempt01'
REPORT_ROOT = ROOT + '/paper_suite_v1'
WAN_BUDGET = dict(model_load=3, codec_probe=26, codec_decode=26, codec_encode=20,
                  writer=49, reader=350, quality=20, post=40)
QUERY_BUDGET = dict(sources=12, random_draws=36, evidence_inputs=360,
                    claims=360, bit_comparisons_max=11520, families=36)
COMPARISON_SCOPE = ('Same Wan source, five edits and second-generation final media; '
    'generation-time main writer versus post-first-media external writers; '
    'different native capacities, no equal-capacity or universal-FPR ranking')


def backend(method):
    return 'pixelseal' if method == 'frameindex' else method


def batches(n, size):
    return [(i, min(size, n-i)) for i in range(0, n, size)]


def writer_batches(method):
    return batches(181, 8 if method in ('wam','frameindex') else 128 if method=='videoseal' else 181)


def wan_plan():
    rows = {}
    def add(key, kind):
        rows[key] = dict(kind=kind, status='NOT_STARTED', started=0, returned=0)
    for view in ('canonical', *VIEWS):
        for kind in ('codec_probe','codec_decode'): add('OFF/'+view+'/'+kind, kind)
    for name in BACKENDS:
        add(name+'/load','model_load')
        for view,n in zip(VIEWS,LENGTHS):
            for start,_ in batches(n,8 if name=='wam' else 32): add(name+'/OFF/'+view+'/read'+str(start),'reader')
    for method in METHODS:
        for start,_ in writer_batches(method): add(method+'/write'+str(start),'writer')
        for view,n in zip(VIEWS,LENGTHS):
            for kind in ('codec_encode','codec_probe','codec_decode','quality'): add(method+'/'+view+'/'+kind,kind)
            for start,_ in batches(n,8 if method=='wam' else 32): add(method+'/WM/'+view+'/read'+str(start),'reader')
            for arm in ('OFF','WM'): add(method+'/'+arm+'/'+view+'/post','post')
    return rows


def family_event(values, size=5):
    if len(values)!=size: raise ValueError('fixed family size required')
    return True if any(v is True for v in values) else False if all(v is False for v in values) else None


def evidence_word(evidence):
    """Accepted message depends only on the frozen evidence, before random q."""
    if not isinstance(evidence,dict) or evidence.get('status')!='READY': return None, 'UNKNOWN'
    slots=evidence.get('slots')
    if not isinstance(slots,list) or len(slots)!=32: return None, 'UNKNOWN'
    for i,row in enumerate(slots):
        if not isinstance(row,dict) or row.get('slot')!=i or type(row.get('valid')) is not bool:
            return None, 'UNKNOWN'
        if row['valid'] and (type(row.get('value')) is not int or row['value'] not in (0,1)):
            return None, 'UNKNOWN'
    if not all(row['valid'] for row in slots): return None, 'NO_ACCEPTABLE_WORD'
    return ''.join(str(row['value']) for row in slots), 'WORD'


def nominal_family(evidences):
    if len(evidences)!=10: raise ValueError('five views times two keys required')
    words=set();unknown=0
    for evidence in evidences:
        word,status=evidence_word(evidence)
        if status=='WORD': words.add(word)
        unknown += status=='UNKNOWN'
    return dict(known_words=sorted(words), unknown_inputs=unknown,
        probability_lower=len(words)/2**32, probability_upper=min(10,len(words)+unknown)/2**32,
        exact_probability=len(words)/2**32 if not unknown else None,
        analytic_family_bound=10/2**32, target=0.001,
        scope='one independent uniform32 query; fixed five-view two-key family; not population certification')


def rate_summary(values, not_applicable=None):
    """Source-unit descriptive rate; CP is explicitly an iid-Bernoulli reference."""
    from scipy.stats import beta
    n=len(values);na=list(not_applicable or [False]*n)
    if len(na)!=n or any(type(v) is not bool for v in na):raise ValueError('fixed N/A mask required')
    eligible=[v for v,a in zip(values,na) if not a];m=len(eligible)
    yes=sum(v is True for v in eligible);no=sum(v is False for v in eligible);unknown=m-yes-no
    def upper(k,total): return 1. if k==total or not total else float(beta.ppf(.95,k+1,total-k))
    def interval(k,total):
        if not total:return [None,None]
        return [0. if k==0 else float(beta.ppf(.025,k,total-k+1)),
                1. if k==total else float(beta.ppf(.975,k+1,total-k))]
    return dict(planned_sources=n,eligible_sources=m,not_applicable=n-m,positive=yes,negative=no,unknown=unknown,
        rate_bounds=[yes/n,(yes+unknown)/n] if n else [None,None],
        eligible_rate_bounds=[yes/m,(yes+unknown)/m] if m else [None,None],
        known_decision_denominator=yes+no,known_decision_rate=yes/(yes+no) if yes+no else None,
        iid_reference_two_sided95=interval(yes,yes+no),
        iid_reference_one_sided95_upper=upper(yes,yes+no) if yes+no else None,
        conservative_iid_reference_upper=upper(yes+unknown,m) if m else None,
        scope='fixed prompt panel; views/keys/claims are not independent source samples; no population certification')
