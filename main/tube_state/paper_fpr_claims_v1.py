"""Probability spaces and saved-decision summaries; no new receiver or query."""
from collections import Counter
from . import paper_suite_v1 as p

PANELS=(('ALL',p.SOURCES),('TEST-A',p.SOURCES[:4]),('TEST-B',p.SOURCES[4:]))
PANEL_ROLE={'ALL':'ORIGINAL_FORMAL_TWELVE_SOURCE_PANEL','TEST-A':'EXISTING_OBSERVED_FORMAL_EVIDENCE',
    'TEST-B':'PREDECLARED_UNOBSERVED_EIGHT_SOURCE_VALIDATION_AT_INCREMENT_DESIGN'}
VERDICTS=('MATCH','MISMATCH','UNRESOLVED','INPUT_UNRESOLVED')
TARGETS={'one_percent':.01,'point_one_percent':.001,'one_in_million':1e-6}
DERIVED_ROWS=48
ADDITIONAL_SCIENTIFIC_BUDGET=dict(sources=0,model=0,GPU=0,codec=0,receiver=0,post=0,query_draws=0,query_claims=0,bit_comparisons=0)


def at_least(verdicts,k):
    if not 1<=k<=len(verdicts):raise ValueError('fixed positive family threshold required')
    accepted=sum(v=='MATCH' for v in verdicts)
    unknown=sum(v=='INPUT_UNRESOLVED' for v in verdicts)
    return True if accepted>=k else False if accepted+unknown<k else None


def uniform_query_union(max_words):
    if type(max_words) is not int or max_words<0:raise ValueError('nonnegative word opportunities required')
    return min(1.,max_words/2**32)


def saved_family_mass(row):
    words=row.get('known_words');unknown=row.get('unknown_inputs')
    valid=(isinstance(words,list) and all(isinstance(w,str) and len(w)==32 and set(w)<={'0','1'} for w in words)
        and type(unknown) is int and 0<=unknown<=10)
    distinct=set(words) if valid else set()
    valid=valid and len(distinct)+unknown<=10
    if not valid:return dict(lower=0.,upper=10/2**32,exact=None,status='TECHNICAL_UNKNOWN',reason='invalid or missing saved accept-set counts')
    return dict(lower=len(distinct)/2**32,upper=(len(distinct)+unknown)/2**32,
        exact=len(distinct)/2**32 if not unknown else None,status='SAVED_EXACT' if not unknown else 'SAVED_BOUNDS',reason=None)


def unique(rows,**keys):
    matches=[r for r in rows if isinstance(r,dict) and all(r.get(k)==v for k,v in keys.items())]
    return matches[0] if len(matches)==1 else {}


def role(arm,key,claim):
    if arm=='OFF':return 'UNWATERMARKED_FIXED_CLAIM_EMPIRICAL_FPR'
    if key=='key01':return 'SPECIFIC_WRONG_KEY_CONTROL'
    if claim=='q1':return 'SPECIFIC_WRONG_MESSAGE_CONTROL'
    return 'MARKED_CORRECT_QUERY_TPR' if arm=='NEW' else 'IDENTITY_PRESENT_TIME_ABSENT_CONTROL_NOT_IDENTITY_NEGATIVE'


def statistics(values,verdicts):
    return dict(p.rate_summary(values),verdict_counts=dict(Counter(verdicts)),
        contradiction_rejections=sum(v=='MISMATCH' for v in verdicts),
        logical_unresolved=sum(v=='UNRESOLVED' for v in verdicts),
        technical_unknown=sum(v=='INPUT_UNRESOLVED' for v in verdicts))


def measurement_rows(fixed_claims,random_families):
    """Exactly48 arithmetic summaries of old saved decisions. Never matches bits."""
    rows=[]
    for panel,sids in PANELS:
        common=dict(panel=panel,panel_role=PANEL_ROLE[panel],source_unit='predeclared prompt-seed, not views/arms/keys',
            population_fpr_certified=False,scientific_pass=None)
        for arm in p.cohort.ARMS:
            for key in p.cohort.KEYS:
                for claim in p.cohort.CLAIMS:
                    groups=[]
                    for sid in sids:
                        group=[]
                        for view in p.VIEWS:
                            row=unique(fixed_claims,source=sid,arm=arm,key=key,claim=claim,view=view)
                            verdict=row.get('verdict');group.append(verdict if verdict in VERDICTS else 'INPUT_UNRESOLVED')
                        groups.append(group)
                    flat=[v for g in groups for v in g];events={}
                    for i,view in enumerate(p.VIEWS):
                        one=[g[i] for g in groups];events[view]=statistics([at_least([v],1) for v in one],one)
                    for event,k in [('ANY5',1),('ALL5',5),('AT_LEAST4_OF5',4)]:
                        events[event]=statistics([at_least(g,k) for g in groups],flat)
                    rows.append(dict(common,row_kind='FIXED_CLAIM',arm=arm,key=key,claim=claim,
                        case_role=role(arm,key,claim),primary=key=='key00' and claim=='q0' and arm in ('OFF','NEW'),
                        probability_space='finite content panel with fixed q0/q1 and fixed key; no random-query theorem for this event',
                        theoretical_bound=None,pvalue=None,planned_sources=len(sids),planned_claims=5*len(sids),events=events,
                        semantic_correctness='read separate saved main_time/main_payload_contributions; MATCH is not time authentication'))
        for arm in p.cohort.ARMS:
            families=[unique(random_families,source=sid,arm=arm) for sid in sids]
            events=[f.get('false_accept_event') if type(f.get('false_accept_event')) is bool else None for f in families]
            na=[f.get('correct_query_collision') is True for f in families]
            masses=[saved_family_mass(f) for f in families]
            verdicts=Counter()
            for family in families:
                saved=family.get('verdict_counts')
                if not (isinstance(saved,dict) and set(saved)<=set(VERDICTS) and all(type(v) is int and v>=0 for v in saved.values()) and sum(saved.values())==10):
                    saved={'INPUT_UNRESOLVED':10}
                verdicts.update(saved)
            rows.append(dict(common,row_kind='RANDOM_QUERY_ARM',arm=arm,family='ANY10_VIEWS_KEYS',
                probability_space='fresh uniform32 query independent of frozen evidence; collisions with actual owner N/A for wrong-query rate',
                events=p.rate_summary(events,na),verdict_counts=dict(verdicts),planned_queries=len(sids),planned_claims=10*len(sids),
                family_probability_records=[dict(source=sid,**mass) for sid,mass in zip(sids,masses)],
                protocol_single_family_upper=10/2**32,
                scope='false-message empirical events and all-query theoretical acceptance mass are distinct'))
        families=[unique(random_families,source=sid,arm=arm) for sid in sids for arm in p.cohort.ARMS]
        masses=[saved_family_mass(f) for f in families];upper=uniform_query_union(30*len(sids))
        rows.append(dict(common,row_kind='RANDOM_PACKAGE_UNION',scope='RANDOM_QUERY_PACKAGE_ONLY_EXCLUDES_FIXED_CLAIMS_AND_EXTERNAL_BASELINES',
            probability_space='any acceptance among the existing random-query subset, not fixed-q0 population FPR',
            planned_sources=len(sids),planned_queries=3*len(sids),planned_claims=30*len(sids),
            protocol_union_upper=upper,saved_mass_union_upper=min(1.,sum(m['upper'] for m in masses)),
            saved_unknown_families=sum(m['exact'] is None for m in masses),whole_package_exact_probability=None,
            method='union bound; no independence among views, keys, or output bits needed',
            nominal_targets=TARGETS,within_nominal_targets={k:upper<=v for k,v in TARGETS.items()},
            thresholds_change_receiver=False))
    assert len(rows)==DERIVED_ROWS
    return rows


def protocol_rows():
    return [dict(probability_model='INDEPENDENT_UNIFORM32_QUERY',implemented=True,
        assumptions='query conditionally uniform given all frozen query-free evidence; at most one complete32 word per view-key; no missing-bit acceptance',
        single_claim_upper=1/2**32,single_family_upper=10/2**32,full_existing_random_package_union_upper=360/2**32,
        repeated_query_rule='J fresh independent-of-evidence queries: upper min(1,10*J/2^32); expand all actually searched identities/media/keys',
        exclusions='query chosen from evidence; unbounded/adaptive claim search; fixed identity population guarantee; precise time authentication'),
        dict(probability_model='FIXED_IDENTITY_FIXED_KEY_CONTENT_DISTRIBUTION',implemented=True,theoretical_bound=None,
            assumptions='no output-bit iid fair null adopted; fixed panel measurements with iid-reference intervals only',
            q0='A6D39C5E',q1='26D39C5E',primary_key='key00',population_fpr_certified=False),
        dict(probability_model='RANDOM_REGISTRATION',implemented=False,theoretical_scope='ex ante over uniform registered message independent of OFF evidence; same counting proof',
            exclusions='not performed by fixed-message C4; not guarantee for every realized identity or identity-adaptive content'),
        dict(probability_model='IID_FAIR_OUTPUT_BINOMIAL',implemented=False,theoretical_bound=None,
            reason='bit balance/pairwise correlation/whitening alone do not prove high-order independence or extreme tails')]
