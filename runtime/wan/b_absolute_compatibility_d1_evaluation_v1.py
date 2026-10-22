"""Card98 post-only fixed-denominator evaluation, no receiver calls or raw votes."""
from main.tube_state import b_absolute_compatibility_v1 as method


def compare_message(base,new):
    a={r['slot']:r for r in base['message']}
    b={r['slot']:r for r in new}
    ra={i for i,r in a.items() if r['embedded_keyed_target_available'] and r['decoded'] is not None}
    rb={i for i,r in b.items() if r['keyed_target_available'] and r['decoded'] is not None}
    ca={i for i,r in a.items() if r['correct'] is True}
    cb={i for i,r in b.items() if r['literal_correct'] is True}
    common=ra & rb
    return dict(common_indices=sorted(common),common_count=len(common),
        old_errors=len(common-ca) if common else None,new_errors=len(common-cb) if common else None,
        old_common_BER=len(common-ca)/len(common) if common else None,
        new_common_BER=len(common-cb)/len(common) if common else None,
        correct_retained=sorted(ca & cb),correct_lost=sorted(ca-cb),correct_added=sorted(cb-ca),
        readable_lost=sorted(ra-rb),readable_added=sorted(rb-ra),
        note='literal keyed message comparison; source support classifications remain separate')


def physical_status(origin,j):
    overlap=max(0,min(4*j,origin+44)-max(4*j-3,origin)+1)
    return 'FULL' if overlap==4 else 'PARTIAL' if overlap else 'ABSENT'


def evaluate_query(d,old,truth):
    previous=old['selected'];meta=previous['metadata_post_only'];a=meta['origin']
    ptrue=(-a)%4;strue=(a+ptrue)//4+1
    C={(h['phase'],h['source_start']) for h in d['C']}
    covered=(ptrue,strue) in C if all(x['in_C'] is not None for x in d['members']) else None
    positive=meta['arm']=='PM' and meta['key']=='key00'
    keyed=meta['arm'] in ('P0','PM') and meta['key']=='key00'
    off=meta['arm']=='OFF';wrongkey=meta['arm']!='OFF' and meta['key']=='key01'
    physical_groups=[]
    for g in range(4):
        statuses=[physical_status(a,j) for j in range(11*g+1,11*g+12)]
        physical_groups.append(dict(group=g,present=any(x=='FULL' for x in statuses),
            partial_only=not any(x=='FULL' for x in statuses) and any(x=='PARTIAL' for x in statuses)))
    groups=[];reference_counts={}
    for group in d['groups']:
        g=group['group'];refs=[]
        for r in group['conditional_references']:
            geo=method.actual_row_geometry(a,r['phase'],r['local_row'])
            entry=dict(**r,actual_geometry_post_only=geo,is_true_state_reference=(r['phase'],r['source_start'])==(ptrue,strue))
            refs.append(entry)
            if r['raw_lookup_attempted']:
                reference_counts[geo['kind']]=reference_counts.get(geo['kind'],0)+1
        used=[r for r in refs if r['raw_lookup_attempted']]
        same_group=bool(used) and all(r['actual_geometry_post_only']['true_group']==g for r in used)
        cross=any(r['actual_geometry_post_only']['true_group'] is None for r in used)
        other_groups=sorted({r['actual_geometry_post_only']['true_group'] for r in used if r['actual_geometry_post_only']['true_group'] is not None})
        if not used:classification='NOT_READ'
        elif covered:classification='TRUE_STATE_REFERENCE_COVERED_OTHER_REFERENCES_RETAINED'
        elif same_group:classification='GROUP_COMPATIBLE_BUT_CANONICAL_LABEL_UNSUPPORTED'
        elif cross:classification='CROSS_GROUP_OR_GUARD_REFERENCE_UNDEFINED'
        elif len(other_groups)==1:classification='WRONG_GROUP_LABEL'
        else:classification='DIFFERENT_ACTUAL_GROUP_REFERENCES_UNRESOLVED'
        groups.append(dict(group=g,common_cells=group['common_cells'],selected_cell=group['selected_cell'],
            actual_lookup_attempted=group['actual_lookup_attempted'],selection_reason=group['selection_reason'],
            conditional_references=refs,output_support_class=classification,
            conditional_references_are_not_multiple_asserted_true_paths=True))
    messages=[]
    for row in d['message']:
        g=row['group'];value=row['decoded'];pg=physical_groups[g];support=groups[g]['output_support_class']
        read=value is not None
        messages.append(dict(slot=row['slot'],group=g,bit=row['bit'],decoded=value,read_status=row['status'],
            selected_canonical_cell=row['selected_canonical_cell'],geometry_present=pg['present'],
            keyed_target_available=keyed and pg['present'],
            literal_correct=None if not keyed or not pg['present'] or not read else value==truth[row['slot']],
            support_class=support,true_state_in_C=covered,
            canonical_label_unsupported=read and not covered,
            cross_group_undefined_claim=read and not covered and support=='CROSS_GROUP_OR_GUARD_REFERENCE_UNDEFINED',
            wrong_group_label_claim=read and not covered and support=='WRONG_GROUP_LABEL',
            group_compatible_but_unverified_cell=read and not covered and support=='GROUP_COMPATIBLE_BUT_CANONICAL_LABEL_UNSUPPORTED',
            OFF_no_target_output=read and off,wrong_key_attribution=read and wrongkey,
            false_missing_group_output=read and not pg['present'],partial_only_false_group_output=read and pg['partial_only']))
    source_rows=[]
    for j in range(1,45):
        matching=[g for g in groups if g['actual_lookup_attempted'] and g['selected_cell']==j]
        g=(j-1)//11;emitted=[m for m in messages if m['selected_canonical_cell']==j and m['decoded'] is not None]
        source_rows.append(dict(source_cell=j,group=g,physical_status=physical_status(a,j),
            common_canonical_support=any(j in x['common_cells'] for x in groups),
            actual_lookup_attempted=bool(matching),emitted_bit_claims=len(emitted),
            true_state_reference_available=bool(matching) and covered,
            emitted_canonical_label_supported=bool(emitted) and covered,
            emitted_canonical_label_unsupported=bool(emitted) and not covered,
            claim_to_absent=bool(emitted) and physical_status(a,j)=='ABSENT',
            claim_to_partial=bool(emitted) and physical_status(a,j)=='PARTIAL'))
    # Keep all10 true-phase cells/80 original bit slots; never expand a group read
    # onto the unselected cells that happen to repeat the same payload.
    fixed_cells=[];old_correct=set();new_correct=set();old_read=set();new_exact_read=set()
    raw=previous['raw_phase_bits'][str(ptrue)]['rows']
    for r in range(1,11):
        j=strue+r-1;g=(j-1)//11
        chosen=d['groups'][g]['actual_lookup_attempted'] and d['groups'][g]['selected_cell']==j
        bits=[]
        for bit in range(8):
            oldbit=raw[(r-1)*8+bit]
            emitted=d['message'][8*g+bit] if chosen else None
            v=emitted['decoded'] if emitted else None
            available=keyed and oldbit['decoded'] is not None and oldbit['reference_defined']
            exact_read=v is not None and covered and chosen
            if available:old_read.add((j,bit))
            if available and oldbit['correct'] is True:old_correct.add((j,bit))
            if keyed and exact_read:new_exact_read.add((j,bit))
            correct=None if not keyed or v is None else v==truth[8*g+bit]
            if keyed and exact_read and correct is True:new_correct.add((j,bit))
            status='NOT_SELECTED_UNREAD' if not chosen else 'SELECTED_BUT_ERASED' if v is None else 'EMITTED_TRUE_REFERENCE_COVERED' if covered else 'EMITTED_CANONICAL_LABEL_UNSUPPORTED'
            bits.append(dict(bit=bit,original_decoded=oldbit['decoded'],original_correct=oldbit['correct'],
                new_decoded=v,new_literal_correct=correct,source_exact_read=exact_read,status=status))
        fixed_cells.append(dict(local_row=r,source_cell=j,group=g,
            common_canonical_support=any(j in x['common_cells'] for x in d['groups']),
            actually_selected_for_lookup=chosen,true_state_in_C=covered,bits=bits,
            emitted_bits=sum(x['new_decoded'] is not None for x in bits),
            exact_read_bits=sum(x['source_exact_read'] for x in bits)))
    common=old_read & new_exact_read
    singleton=d['singleton_candidate']
    readable=[m for m in messages if m['keyed_target_available'] and m['decoded'] is not None]
    errors=sum(m['literal_correct'] is False for m in readable)
    output=dict(metadata_post_only=meta,positive=positive,C_size=len(C),true_state_in_C=covered,
        detection=d['detection'],location_set_class=d['location_set_class'],singleton_candidate=singleton,
        singleton_origin_correct=None if singleton is None else singleton['origin']==a,
        message=messages,groups=groups,source_rows=source_rows,fixed_true_phase_cells=fixed_cells,
        counts=dict(message_slots=32,geometry_present=sum(m['geometry_present'] for m in messages),
            read=sum(m['decoded'] is not None for m in messages),erased=sum(m['decoded'] is None for m in messages),
            technical_unresolved=sum(m['read_status']=='TECHNICAL_UNRESOLVED' for m in messages),
            readable_present_keyed=len(readable),literal_present_errors=errors if keyed else None,
            literal_present_BER=errors/len(readable) if readable else None,
            common_canonical_support_cells=sum(x['common_canonical_support'] for x in source_rows),
            actual_selected_cell_labels=sum(x['actual_lookup_attempted'] for x in source_rows),
            actual_selected_true_phase_cells=sum(x['actually_selected_for_lookup'] for x in fixed_cells),
            actual_lookup_true_reference_cells=sum(x['actually_selected_for_lookup'] and covered for x in fixed_cells),
            emitted_exact_source_bits=len(new_exact_read),unselected_true_phase_cell_views=sum(not x['actually_selected_for_lookup'] for x in fixed_cells),
            original_raw_bit_slots=80,source_exact_unread_bit_slots=80-len(new_exact_read),
            OFF_no_target_outputs=sum(m['OFF_no_target_output'] for m in messages),
            wrong_key_attributions=sum(m['wrong_key_attribution'] for m in messages),
            false_missing_group_outputs=sum(m['false_missing_group_output'] for m in messages),
            unsupported_canonical_label_bits=sum(m['canonical_label_unsupported'] for m in messages),
            cross_group_undefined_claim_bits=sum(m['cross_group_undefined_claim'] for m in messages),
            wrong_group_label_bits=sum(m['wrong_group_label_claim'] for m in messages),
            group_compatible_unverified_cell_bits=sum(m['group_compatible_but_unverified_cell'] for m in messages)),
        conditional_reference_geometry_counts=reference_counts,
        against_old_gated=compare_message(old['selected'],messages),
        against_old_forced_min=compare_message(old['forced_min'],messages),
        source_bit_selection_cost=dict(original_read=len(old_read),new_exact_read=len(new_exact_read),common_count=len(common),
            old_common_errors=len(common-old_correct) if common else None,new_common_errors=len(common-new_correct) if common else None,
            old_common_BER=len(common-old_correct)/len(common) if common else None,
            new_common_BER=len(common-new_correct)/len(common) if common else None,
            correct_retained=len(old_correct & new_correct),correct_lost=len(old_correct-new_correct),correct_added=len(new_correct-old_correct),
            unsupported_output_claims_are_reported_separately_not_dropped=True),
        authenticated=False,independent_confirmation=False,new_physical_BER=False)
    return output


def totals(queries):
    rows=list(queries.values());pos=[x for x in rows if x['positive']];null=[x for x in rows if not x['positive']]
    sum_count=lambda name,subset:sum(x['counts'][name] for x in subset)
    return dict(query_slots=len(rows),null_query_slots=len(null),positive_query_slots=len(pos),
        null_detected=sum(x['detection']=='DETECTED_PROVISIONAL' for x in null),null_rejected=sum(x['detection']=='REJECT_SYNC' for x in null),
        positive_detected=sum(x['detection']=='DETECTED_PROVISIONAL' for x in pos),
        positive_true_in_C=sum(x['true_state_in_C'] is True for x in pos),positive_true_excluded=sum(x['true_state_in_C'] is False for x in pos),
        positive_set_unresolved=sum(x['true_state_in_C'] is None for x in pos),
        positive_singleton_wrong=sum(x['singleton_origin_correct'] is False for x in pos),
        positive_multiset=sum(x['C_size']>1 for x in pos),positive_empty_C=sum(x['C_size']==0 and x['true_state_in_C'] is not None for x in pos),
        all_message_slots=sum_count('message_slots',rows),all_output_bits=sum_count('read',rows),all_erasures=sum_count('erased',rows),
        positive_message_slots=sum_count('message_slots',pos),positive_read=sum_count('read',pos),positive_erasures=sum_count('erased',pos),
        positive_literal_errors=sum(x['counts']['literal_present_errors'] or 0 for x in pos),
        positive_common_canonical_support_cells=sum_count('common_canonical_support_cells',pos),
        positive_actual_selected_cell_labels=sum_count('actual_selected_cell_labels',pos),
        positive_actual_selected_true_phase_cells=sum_count('actual_selected_true_phase_cells',pos),
        positive_actual_lookup_true_reference_cells=sum_count('actual_lookup_true_reference_cells',pos),
        positive_original_cell_view_denominator=10*len(pos),positive_original_raw_bit_denominator=80*len(pos),
        positive_new_exact_source_bits=sum_count('emitted_exact_source_bits',pos),positive_exact_source_unread_bits=sum_count('source_exact_unread_bit_slots',pos),
        positive_source_correct_retained=sum(x['source_bit_selection_cost']['correct_retained'] for x in pos),
        positive_source_correct_lost=sum(x['source_bit_selection_cost']['correct_lost'] for x in pos),
        positive_source_correct_added=sum(x['source_bit_selection_cost']['correct_added'] for x in pos),
        positive_message_correct_retained_vs_old=sum(len(x['against_old_gated']['correct_retained']) for x in pos),
        positive_message_correct_lost_vs_old=sum(len(x['against_old_gated']['correct_lost']) for x in pos),
        positive_message_correct_added_vs_old=sum(len(x['against_old_gated']['correct_added']) for x in pos),
        OFF_no_target_outputs=sum_count('OFF_no_target_outputs',rows),wrong_key_attributions=sum_count('wrong_key_attributions',rows),
        false_missing_group_outputs=sum_count('false_missing_group_outputs',rows),
        unsupported_canonical_label_bits=sum_count('unsupported_canonical_label_bits',rows),
        cross_group_undefined_claim_bits=sum_count('cross_group_undefined_claim_bits',rows),
        wrong_group_label_bits=sum_count('wrong_group_label_bits',rows),
        technical_message_slots=sum_count('technical_unresolved',rows))


def evaluate(decisions,old_evaluation,truth):
    if len(truth)!=32 or any(type(x) is not int or x not in (0,1) for x in truth):raise ValueError('post-only32 binary message required')
    hosts={};all_queries={};confirmation={}
    for host in method.HOSTS:
        queries={qid:evaluate_query(d,old_evaluation['hosts'][host]['queries'][qid],truth) for qid,d in decisions[host].items()}
        strata={arm+'/'+key:totals({q:v for q,v in queries.items() if v['metadata_post_only']['arm']==arm and v['metadata_post_only']['key']==key}) for arm in ('OFF','P0','PM') for key in ('key00','key01')}
        hosts[host]=dict(by_arm_key=strata,role='seen_old_calibration' if host in method.CALIBRATION_HOSTS else 'seen_old_confirmation_now_development',queries=queries,totals=totals(queries))
        all_queries.update({host+'/'+q:v for q,v in queries.items()})
        if host not in method.CALIBRATION_HOSTS:confirmation.update({host+'/'+q:v for q,v in queries.items()})
    return dict(status='POST_ONLY_SEEN_DATA_DEVELOPMENT',hosts=hosts,all144=totals(all_queries),old_confirmation72=totals(confirmation),
        fixed_old_confirmation_denominators=dict(PM_cell_views=120,PM_raw_bit_views=960,PM_message_slots=384,physical_present=160,physical_missing=224),
        source_support_actual_lookup_and_message_coverage_are_distinct=True,
        alternatives_are_conditional_references_not_multiple_true_path_claims=True,
        no_output_rows_filtered_on_true_state_membership=True,new_physical_BER=False,independent_confirmation=False,scientific_PASS=None)
