"""Card99 post-only comparison; never invoked before all confirmation blind saves."""
from main.tube_state import b_absolute_confirmation_v1 as method
from runtime.wan.b_absolute_compatibility_d1_evaluation_v1 import evaluate_query,totals
from runtime.wan.b_fixed_detection_confidence_evaluation_v1 import assess,paired_payload


def evaluate(decisions,baselines,families,thresholds,bits,roles):
    if len(bits)!=32 or any(type(x) is not int or x not in (0,1) for x in bits):raise ValueError('post-only32 binary bits required')
    hosts={};allqueries={}
    for h in roles['confirmation']:
        previous={};queries={}
        for q,meta in method.query_metadata().items():
            base=baselines[h][q]
            forced=assess(base['ungated'],families[h][q],meta,bits)
            gated=assess(base['gated'],families[h][q],meta,bits)
            previous[q]=dict(forced_min=forced,selected=gated)
            value=evaluate_query(decisions[h][q],previous[q],bits)
            value['independent_confirmation']=True
            value['scope']='fixed four-source roster with random role split; not population sampling'
            queries[q]=value
        paired={}
        for origin in method.ORIGINS:
            lookup={v['selected']['metadata_post_only']['arm']:v['selected'] for v in previous.values() if v['selected']['metadata_post_only']['origin']==origin and v['selected']['metadata_post_only']['key']=='key00'}
            paired[str(origin)]=paired_payload(lookup['P0'],lookup['PM'])
        strata={a+'/'+key:totals({q:v for q,v in queries.items() if v['metadata_post_only']['arm']==a and v['metadata_post_only']['key']==key}) for a in method.ARMS for key in ('key00','key01')}
        selected_cells=set();emitted_cells=set();read_source_bits=set()
        for v in queries.values():
            if not v['positive']:continue
            for cell in v['fixed_true_phase_cells']:
                if cell['actually_selected_for_lookup']:selected_cells.add(cell['source_cell'])
                for b in cell['bits']:
                    if b['source_exact_read']:
                        emitted_cells.add(cell['source_cell']);read_source_bits.add((cell['source_cell'],b['bit']))
        hosts[h]=dict(role='random_assigned_confirmation',queries=queries,totals=totals(queries),by_arm_key=strata,
            B0_post=previous,paired_P0_PM_same_true_phase_physical_payload=paired,
            unique_source_cell_union_selected=len(selected_cells),unique_source_cell_union_with_output=len(emitted_cells),unique_exact_source_bit_union_read=len(read_source_bits),
            fixed_source_cell_union_denominator=42,fixed_source_bit_union_denominator=336)
        allqueries.update({h+'/'+q:v for q,v in queries.items()})
    return dict(status='POST_ONLY_UNSEEN_SOURCE_RANDOM_ROLE_CONFIRMATION',hosts=hosts,confirmation=totals(allqueries),
        roles=roles,thresholds=dict(tau=thresholds['tau'],L=thresholds['L']),
        fixed_confirmation_denominators=dict(hosts=2,queries=72,null_queries=60,positive_queries=12,H=9864,message_slots=2304,
            positive_message_slots=384,physical_present=160,physical_missing=224,true_phase_nominal_supported=144,present_unobserved=16,
            cell_views=120,raw_bit_views=960,unique_source_cell_union=84,unique_source_bit_union=672,net_bits_per_source=32),
        no_output_filtered_on_true_membership=True,conditional_alternatives_not_independent_paths=True,
        physical_P0_PM_payload_pairs_are_separate_from_logical_B0_comparison=True,
        FPR_population=None,conditional_confidence_guarantee=None,authenticated=False,scientific_PASS=None,
        finite_roster_rank_claim='only under complete role-invariant potential observations; not conditional on acceptance',
        no_scientific_PASS_from_all_reject_or_all_erase=True)
