"""Card101 post-only reporting; no scorer/threshold/bit vote calls."""
from runtime.wan.b_absolute_compatibility_d1_evaluation_v1 import evaluate_query,totals
from runtime.wan.b_absolute_confirmation_evaluation_v1 import evaluate as physical_post


def evaluate_saved(decisions,old,bits,hosts):
    reports={};allqueries={}
    for h in hosts:
        queries={}
        for q,d in decisions[h].items():
            v=evaluate_query(d,old['hosts'][h]['B0_post'][q],bits)
            v.update(independent_confirmation=False,scope='all four old sources already seen; CPU development only')
            previous=old['hosts'][h]['queries'][q]['message']
            previous_correct={x['slot'] for x in previous if x['literal_correct'] is True}
            current_correct={x['slot'] for x in v['message'] if x['literal_correct'] is True}
            v['against_frozen_Card100']=dict(correct_retained=sorted(previous_correct&current_correct),
                correct_lost=sorted(previous_correct-current_correct),correct_added=sorted(current_correct-previous_correct))
            queries[q]=v
        reports[h]=dict(role='seen_development',queries=queries,totals=totals(queries))
        allqueries.update({h+'/'+q:v for q,v in queries.items()})
    return dict(status='SEEN_Q_CPU_DEVELOPMENT_POST',hosts=reports,development72=totals(allqueries),
        historical_B0_has_old_score_and_threshold=True,physical_payload_not_reread=True,
        old_physical_pair_records={h:old['hosts'][h]['paired_P0_PM_same_true_phase_physical_payload'] for h in hosts},
        independent_confirmation=False,CFAR=False,role_randomization_rank_guarantee=None,scientific_PASS=None)


def evaluate_user(decisions,baselines,families,thresholds,bits,roles):
    v=physical_post(decisions,baselines,families,dict(tau=thresholds['tau_N'],L=None),bits,roles)
    v.update(status='BACKGROUND_NORMALIZED_PHYSICAL_POST',thresholds=dict(tau_N=thresholds['tau_N']),
        finite_roster_rank_claim=None,role_randomization_rank_guarantee=None,CFAR=False,
        training_OFF_reused_in_null=True,score_kind='Q40_normalized_matched_S',scientific_PASS=None)
    return v
