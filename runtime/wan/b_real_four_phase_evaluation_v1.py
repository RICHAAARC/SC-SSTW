"""Card89 post-only phase-specific frame support, source/group and message evaluation."""
from collections import Counter
from main.tube_state import b_phase_centered_saved_v1 as cpu


def geometry(start, stop, phase):
    if type(start) is not int or stop-start != 45 or not 0 <= start <= 136 or phase not in range(4):
        raise ValueError('post-only fixed45-frame received geometry required')
    visible = set(range(start, stop))
    retained = set(range(start+phase+1, start+phase+41))
    sources=[]
    for j in range(1,45):
        canonical=set(range(4*j-3,4*j+1))
        a,b=sorted(canonical & visible),sorted(canonical & retained)
        sources.append(dict(source_cell=j,group=(j-1)//11,visible_frames=a,
            physical_status='FULL' if len(a)==4 else 'PARTIAL' if a else 'ABSENT',
            retained_nominal_frames=b,retained_nominal_status='FULL' if len(b)==4 else 'PARTIAL' if b else 'ABSENT',
            physically_present_but_observation_not_full=len(a)==4 and len(b)!=4))
    rows=[]
    for r in range(1,11):
        frames=list(range(start+phase+4*r-3,start+phase+4*r+1))
        counts=Counter((f+3)//4 for f in frames)
        js=sorted(counts);gs=sorted({(j-1)//11 for j in js})
        rows.append(dict(local_row=r,phase=phase,nominal_frames=frames,
            received_frame_interval_half_open=[phase+4*r-3,phase+4*r+1],
            source_cells=js,frame_counts={str(j):counts[j] for j in js},groups=gs,
            true_source_cell=js[0] if len(js)==1 else None,true_group=gs[0] if len(gs)==1 else None,
            truth_kind='SINGLE_CELL' if len(js)==1 else 'GROUP_ONLY' if len(gs)==1 else 'MIXED_GROUP_UNDEFINED'))
    groups=[]
    for g in range(4):
        a=[x['source_cell'] for x in sources if x['group']==g and x['physical_status']=='FULL']
        b=[x['source_cell'] for x in sources if x['group']==g and x['retained_nominal_status']=='FULL']
        groups.append(dict(group=g,structurally_present=bool(a),physical_full_cells=a,
            retained_nominal_full_cells=b,defined_reference_local_rows=[x['local_row'] for x in rows if x['true_group']==g],
            present_but_no_retained_nominal_full_cell=bool(a) and not bool(b)))
    return dict(source_cells=sources,observed_cells=rows,groups=groups,phase=phase,
        first_context_source_frame=start+phase,physical_received_frames=[start,stop],
        causal_history='reset at own phase view start; not full-history',frame_membership_is_not_linear_q_mixture=True)

def raw_evaluation(votes, geo, bits):
    rows = []
    for r, g in enumerate(geo['observed_cells']):
        for b in range(8):
            value = None if votes is None else votes['local_cells'][r]['bits'][b]['decoded']
            refg = g['true_group']
            rows.append(dict(local_row=r+1, slot=b, decoded=value, truth_kind=g['truth_kind'],
                reference_undefined=refg is None, true_group=refg,
                correct=None if value is None or refg is None or bits is None else value == bits[8*refg+b]))
    return dict(rows=rows, planned=80, unread=sum(x['decoded'] is None for x in rows),
        single_cell_references=sum(x['truth_kind'] == 'SINGLE_CELL' for x in rows),
        single_cell_errors=None if bits is None else sum(x['truth_kind'] == 'SINGLE_CELL' and x['correct'] is False for x in rows),
        group_only_references=sum(x['truth_kind'] == 'GROUP_ONLY' for x in rows),
        group_only_errors=None if bits is None else sum(x['truth_kind'] == 'GROUP_ONLY' and x['correct'] is False for x in rows),
        undefined=sum(x['reference_undefined'] for x in rows))


def candidate_evaluation(item, geo, bits):
    observed = []
    mixed = any(x['true_source_cell'] is None for x in geo['observed_cells'])
    for source, hypothesis in zip(geo['observed_cells'], item['cells']):
        j, g, refj, refg = hypothesis['source_cell'], hypothesis['group'], source['true_source_cell'], source['true_group']
        observed.append(dict(local_row=source['local_row'], claimed_source_cell=j, claimed_group=g,
            true_source_cell=refj, true_group=refg,
            exact_cell_correct=None if refj is None else j == refj,
            group_correct=None if refg is None else g == refg,
            cell_wrong_but_group_correct=None if refj is None else j != refj and g == refg,
            mixed_single_cell_unsupported=refj is None,
            mixed_group_only_compatible=refj is None and refg is not None and g == refg,
            mixed_group_only_wrong=refj is None and refg is not None and g != refg,
            group_reference_undefined=refg is None))
    sources = []
    for source in geo['source_cells']:
        j = source['source_cell']
        claims = [x['local_row'] for x in observed if x['claimed_source_cell'] == j]
        exact = [x['local_row'] for x in observed if x['claimed_source_cell'] == j and x['exact_cell_correct'] is True]
        sources.append(dict(**source, claimed_local_rows=claims, exact_correct_local_rows=exact,
            conditional_false_source_coverage=None if mixed else bool(claims) and not bool(exact),
            claim_to_absent=bool(claims) and source['physical_status'] == 'ABSENT',
            claim_to_partial=bool(claims) and source['physical_status'] == 'PARTIAL',
            mixed_identity_unverifiable=mixed and bool(claims)))
    messages = []
    for b, value in enumerate(item['decoded_bits']):
        group = geo['groups'][b//8]
        exists = group['structurally_present']
        messages.append(dict(slot=b, decoded=value, structurally_present=exists,
            correct_present=None if not exists or value is None or bits is None else value == bits[b],
            present_unread=exists and value is None,
            structurally_missing_unread=not exists and value is None,
            physically_present_but_no_retained_full_source_cell=group['present_but_no_retained_nominal_full_cell'],
            false_missing_group_coverage=not exists and value is not None,
            chance_match_missing_source=None if exists or value is None or bits is None else value == bits[b]))
    readable = [x for x in messages if x['structurally_present'] and x['decoded'] is not None]
    correct_cells = [x['claimed_source_cell'] for x in observed if x['exact_cell_correct'] is True]
    return dict(source_start=item['source_start'], numeric_status=item['status'],
        unavailable_from=item.get('unavailable_from'), hypothesis_only=True,
        observed_cells=observed, source_cells=sources, message=messages,
        planned_cells=10, planned_message_bits=32, planned_source_rows=44,
        singleton_rows=sum(x['true_source_cell'] is not None for x in observed),
        exact_correct_cells=None if mixed else len(correct_cells),
        exact_wrong_cells=None if mixed else sum(x['exact_cell_correct'] is False for x in observed),
        exact_source_evaluable=not mixed,
        exact_correct_source_cells=None if mixed else correct_cells,
        cell_wrong_but_group_correct_rows=sum(x['cell_wrong_but_group_correct'] is True for x in observed),
        group_correct_rows=sum(x['group_correct'] is True for x in observed),
        group_wrong_rows=sum(x['group_correct'] is False for x in observed),
        group_correct_local_rows=[x['local_row'] for x in observed if x['group_correct'] is True],
        group_wrong_local_rows=[x['local_row'] for x in observed if x['group_correct'] is False],
        mixed_unsupported_single_cell_claims=sum(x['mixed_single_cell_unsupported'] for x in observed),
        mixed_group_only_compatible_rows=sum(x['mixed_group_only_compatible'] for x in observed),
        mixed_group_only_wrong_rows=sum(x['mixed_group_only_wrong'] for x in observed),
        group_undefined_rows=sum(x['group_reference_undefined'] for x in observed),
        absent_source_claims=sum(x['claim_to_absent'] for x in sources),
        partial_source_claims=sum(x['claim_to_partial'] for x in sources),
        false_source_coverage=None if mixed else sum(x['conditional_false_source_coverage'] for x in sources),
        structurally_present_bits=sum(x['structurally_present'] for x in messages),
        structurally_missing_bits=sum(not x['structurally_present'] for x in messages),
        readable_present_bits=len(readable), present_errors=None if bits is None else sum(x['correct_present'] is False for x in readable),
        present_BER=None if not readable or bits is None else sum(x['correct_present'] is False for x in readable)/len(readable),
        present_unread_bits=sum(x['present_unread'] for x in messages),
        missing_group_false_coverage=sum(x['false_missing_group_coverage'] for x in messages),
        present_but_no_retained_full_source_cell_bits=sum(x['physically_present_but_no_retained_full_source_cell'] for x in messages),
        identity_status='UNCALIBRATED', phase_status='UNCALIBRATED', authenticated=False)


def attach_support(summary, phase, geometry_value, variant, truth_available):
    summary['phase']=phase
    summary['support_scope']='received-variant-'+str(variant)
    summary['truth_available']=truth_available
    for row, actual in zip(summary['observed_cells'],geometry_value['observed_cells']):
        row.update(phase=phase,nominal_frames=actual['nominal_frames'],
            received_frame_interval_half_open=actual['received_frame_interval_half_open'])
    summary['group_correct_frame_supports']=[row['received_frame_interval_half_open']
        for row in summary['observed_cells'] if row['group_correct'] is True]
    return summary


def previous_summary(old, name, variant, received_variant):
    if old is None:
        return None
    value=old['tables'][name]['variants'][variant]['summary']
    if value is None:
        return None
    result=dict(value)
    result.update(phase=0,support_scope='received-variant-'+str(received_variant),truth_available=True,
        group_correct_frame_supports=[[4*r-3,4*r+1] for r in value['group_correct_local_rows']])
    return result


def compare(a,b):
    if a is None or b is None:
        return dict(status='UNRESOLVED_NO_COMPLETE_UNIQUE_CONDITIONAL_SUMMARY',
            exact_source_coverage_status='UNRESOLVED',correct_source_retained=None,
            correct_source_lost=None,correct_source_added=None,authenticated=False)
    readable=lambda x:{r['slot'] for r in x['message'] if r['structurally_present'] and r['decoded'] is not None}
    correct=lambda x:{r['slot'] for r in x['message'] if r['correct_present'] is True}
    ra,rb=readable(a),readable(b)
    common=ra & rb
    available=a['truth_available'] and b['truth_available']
    ca,cb=(correct(a),correct(b)) if available else (None,None)
    source_ok=a['exact_source_evaluable'] and b['exact_source_evaluable']
    ja,jb=(set(a['exact_correct_source_cells']),set(b['exact_correct_source_cells'])) if source_ok else (None,None)
    support_ok=a['phase']==b['phase'] and a['support_scope']==b['support_scope']
    ga,gb=({tuple(x) for x in a['group_correct_frame_supports']},{tuple(x) for x in b['group_correct_frame_supports']}) if support_ok else (None,None)
    return dict(status='POST_ONLY_CONDITIONAL_COMPARISON',common_readable_indices=sorted(common),
        a_common_errors=len(common-ca) if available else None,b_common_errors=len(common-cb) if available else None,
        a_common_BER=len(common-ca)/len(common) if available and common else None,
        b_common_BER=len(common-cb)/len(common) if available and common else None,
        correct_message_retained=sorted(ca & cb) if available else None,
        correct_message_lost=sorted(ca-cb) if available else None,correct_message_added=sorted(cb-ca) if available else None,
        readable_added=sorted(rb-ra),readable_lost=sorted(ra-rb),
        exact_source_coverage_status='EVALUABLE_SINGLE_SOURCE' if source_ok else 'UNDEFINED_MIXED_IDENTITY',
        correct_source_retained=sorted(ja & jb) if source_ok else None,
        correct_source_lost=sorted(ja-jb) if source_ok else None,correct_source_added=sorted(jb-ja) if source_ok else None,
        group_support_comparison_status='SAME_PHASE_FRAME_SUPPORT' if support_ok else 'UNRESOLVED_DIFFERENT_PHASE_SUPPORT',
        group_correct_support_retained=sorted(ga & gb) if support_ok else None,
        group_correct_support_lost=sorted(ga-gb) if support_ok else None,group_correct_support_added=sorted(gb-ga) if support_ok else None,
        a_group_correct_frame_supports=a['group_correct_frame_supports'],b_group_correct_frame_supports=b['group_correct_frame_supports'],
        a_missing_group_false_coverage=a['missing_group_false_coverage'],b_missing_group_false_coverage=b['missing_group_false_coverage'],
        a_absent_source_claims=a['absent_source_claims'],b_absent_source_claims=b['absent_source_claims'],
        a_partial_source_claims=a['partial_source_claims'],b_partial_source_claims=b['partial_source_claims'],
        raw_physical_BER_gain_claim=False,authenticated=False)


def evaluate(families,tables,votes,metadata,truth,previous):
    if metadata is None:
        return dict(status='POST_METADATA_UNAVAILABLE',tables={},planned_queries=8,scientific_pass=None)
    bits=None if truth is None else truth.get('bits')
    if truth is not None and (not isinstance(bits,list) or len(bits)!=32 or any(type(x) is not int or x not in (0,1) for x in bits)):
        raise ValueError('post-only32 binary truth required when available')
    output={}
    for name,family in families.items():
        case,key=name.split('/')
        meta=metadata[case]
        views={p:geometry(meta['start'],meta['stop'],p) for p in range(4)}
        true_phase=(-meta['start'])%4
        true_start=(meta['start']+true_phase)//4+1
        evaluated=[]
        lookup={}
        for h in family['candidates']:
            p,s=h['phase'],h['source_start']
            observation=h['candidate_ref']['observation']
            candidate=next(x for x in tables[observation]['candidates'] if x['source_start']==s)
            result=candidate_evaluation(candidate,views[p],bits)
            attach_support(result,p,views[p],meta['variant'],bits is not None)
            result.update(public_origin_hypothesis=4*(s-1)-p,
                true_phase_post_only=true_phase,phase_guess_correct=p==true_phase,
                origin_guess_correct=4*(s-1)-p==meta['start'],candidate_ref=h['candidate_ref'])
            lookup[p,s]=result
            evaluated.append(result)
        variants={}
        for variant in cpu.VARIANTS:
            maxima=family['maxima'][variant]
            ties=maxima['exact_max_ties']
            winners=[lookup[x['phase'],x['source_start']] for x in ties] if ties else []
            complete=family['status']=='COMPLETE_UNCALIBRATED' and maxima['status']=='COMPLETE'
            summary=winners[0] if complete and len(winners)==1 and winners[0]['numeric_status']=='CONDITIONAL_REDUCED_UNCALIBRATED' else None
            target=next(x for x in family['candidates'] if x['phase']==true_phase and x['source_start']==true_start)
            target_score=target[variant]
            all_scores=all(cpu.finite_number(x[variant]) and x['score_status']=='RETURNED' for x in family['candidates'])
            prior=previous_summary(previous,name,variant,meta['variant'])
            variants[variant]=dict(status='UNIQUE_UNCALIBRATED_CONDITIONAL_HYPOTHESIS' if summary else
                'EXACT_TIE_ALL_ALTERNATIVES_RETAINED' if complete else maxima['status'],
                maximum=maxima['maximum'],exact_max_ties=ties,winner_evaluations=winners,summary=summary,
                true_phase_post_only=true_phase,true_source_start_post_only=true_start,
                true_state_score=target_score if cpu.finite_number(target_score) else None,
                true_state_rank=1+sum(x[variant]>target_score for x in family['candidates']) if all_scores else None,
                compared_to_Card88_p0=compare(prior,summary),accepted_count=None,rejected_count=None,
                trusted_full_message=None,identity_status='UNCALIBRATED',phase_status='UNCALIBRATED')
        output[name]=dict(case=case,key=key,arm=meta['arm'],received_variant=meta['variant'],
            family_status=family['status'],incomplete_phases=family['incomplete_phases'],
            candidate_denominator=137,logical_message_rows=4384,
            phase_observations={str(p):dict(geometry_post_only=views[p],
                raw=raw_evaluation(votes.get(meta['observation_ids'][p]+'/'+key),views[p],bits)) for p in range(4)},
            all_candidate_evaluations=evaluated,variants=variants)
    pairs={}
    for received in (0,1):
        for key in ('key00','key01'):
            a=next((t for t in output.values() if t['arm']=='P0' and t['received_variant']==received and t['key']==key),None)
            b=next((t for t in output.values() if t['arm']=='PM' and t['received_variant']==received and t['key']==key),None)
            pairs[f'variant{received}/{key}']={v:compare(a['variants'][v]['summary'],b['variants'][v]['summary']) for v in cpu.VARIANTS} if a and b else dict(status='UNAVAILABLE_PAIR')
    return dict(status='POST_ONLY' if bits is not None else 'POST_TRUTH_UNAVAILABLE',tables=output,paired_arms=pairs,
        fixed_denominators=dict(queries=8,candidates=1096,logical_message_rows=35072,unique_candidates=688,
            per_variant_message_slots=256,per_variant_source_rows=352,logical_phase_local_bits=2560),
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',accepted_count=None,rejected_count=None,
        trusted_full_messages=None,FPR=None,scientific_pass=None)
