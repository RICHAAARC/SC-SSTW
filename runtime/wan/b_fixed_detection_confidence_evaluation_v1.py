"""Card94 post-marker geometry and target-aware empirical evaluation; stdlib only."""
from main.tube_state import b_fixed_detection_confidence_v1 as method


def geometry(origin, phase):
    visible = set(range(origin, origin+45))
    retained = set(range(origin+phase+1, origin+phase+41))
    sources = []
    for j in range(1, 45):
        support = set(range(4*j-3, 4*j+1))
        n, m = len(support & visible), len(support & retained)
        sources.append(dict(source_cell=j, group=(j-1)//11,
            physical_status='FULL' if n == 4 else 'PARTIAL' if n else 'ABSENT',
            retained_status='FULL' if m == 4 else 'PARTIAL' if m else 'ABSENT'))
    observed = []
    for r in range(1, 11):
        frames = list(range(origin+phase+4*r-3, origin+phase+4*r+1))
        js = sorted({(f+3)//4 for f in frames})
        gs = sorted({(j-1)//11 for j in js if 1 <= j <= 44})
        guarded = any(j not in range(1,45) for j in js)
        observed.append(dict(local_row=r, frames=frames, source_cells=js,
            true_source_cell=js[0] if len(js)==1 and not guarded else None,
            true_group=gs[0] if len(gs)==1 and not guarded else None,
            truth_kind='GUARD_OR_OUT_OF_PROTOCOL' if guarded else 'SINGLE_SOURCE_FULL' if len(js)==1 else 'MIXED_SAME_GROUP' if len(gs)==1 else 'MIXED_GROUP_UNDEFINED'))
    groups = [dict(group=g, physical_present=any(x['group']==g and x['physical_status']=='FULL' for x in sources),
        nominal_supported=any(x['group']==g and x['retained_status']=='FULL' for x in sources)) for g in range(4)]
    return dict(sources=sources, observed=observed, groups=groups)


def assess(decision, family, meta, bits):
    phase = (-meta['origin']) % 4
    true = geometry(meta['origin'], phase)
    has_target = meta['arm'] in ('P0', 'PM') and meta['key']=='key00'
    positive = meta['arm']=='PM' and meta['key']=='key00'
    off = meta['arm']=='OFF'
    wrong_key = meta['arm']!='OFF' and meta['key']=='key01'
    messages = []
    for b, row in enumerate(decision['message']):
        group = true['groups'][b//8]
        value = row['decoded']
        available = group['physical_present']
        messages.append(dict(slot=b, decoded=value, read_status=row['status'],
            geometry_group_supported=available, true_state_nominal_supported=group['nominal_supported'],
            present_but_nominal_unobserved=available and not group['nominal_supported'],
            embedded_keyed_target_available=has_target and available,
            correct=None if not has_target or not available or value is None else value==bits[b],
            false_missing_group_output=not available and value is not None,
            OFF_no_target_output=off and value is not None,
            wrong_key_attribution=wrong_key and value is not None,
            present_unread=available and value is None))
    location = decision.get('location_hypothesis')
    observed, exact_correct, source_evaluable = [], [], True
    if location is not None:
        actual = geometry(meta['origin'], location['phase'])
        for r, row in enumerate(actual['observed']):
            j = location['source_start']+r
            full = row['true_source_cell'] is not None
            observed.append(dict(**row, claimed_source_cell=j, exact_correct=j==row['true_source_cell'] if full else None,
                claimed_group=(j-1)//11, group_correct=(j-1)//11==row['true_group'] if row['true_group'] is not None else None))
            source_evaluable &= full
            if full and j == row['true_source_cell']:
                exact_correct.append(j)
    source_rows = []
    for row in true['sources']:
        claims = [r for r in observed if r['claimed_source_cell']==row['source_cell']]
        source_rows.append(dict(**row, claimed=bool(claims), exact_correct=row['source_cell'] in exact_correct,
            false_coverage=None if not source_evaluable else bool(claims) and row['source_cell'] not in exact_correct,
            claim_to_partial=bool(claims) and row['physical_status']=='PARTIAL',
            claim_to_absent=bool(claims) and row['physical_status']=='ABSENT'))
    raw = {}
    for p in range(4):
        geo = geometry(meta['origin'], p)
        values = family.get('phase_local_bits', {}).get(str(p), [None]*80)
        rows = []
        for r, source in enumerate(geo['observed']):
            for bit in range(8):
                value = values[r*8+bit] if len(values)==80 else None
                g = source['true_group']
                rows.append(dict(local_row=r+1, slot=bit, decoded=value, truth_kind=source['truth_kind'], frame_support=source['frames'],
                    reference_defined=has_target and g is not None,
                    correct=None if not has_target or g is None or value is None else value==bits[8*g+bit]))
        raw[str(p)] = dict(planned=80, rows=rows,
            defined_read=sum(x['correct'] is not None for x in rows),
            errors=sum(x['correct'] is False for x in rows), undefined=sum(not x['reference_defined'] for x in rows))
    selected = {(x['phase'], x['source_start']) for x in decision['candidate_set']}
    true_state = (phase, (meta['origin']+phase)//4+1)
    readable = [x for x in messages if x['embedded_keyed_target_available'] and x['decoded'] is not None]
    errors = sum(x['correct'] is False for x in readable)
    return dict(status=decision['status'], detection=decision['detection'], location=decision['location'],
        metadata_post_only=meta, is_sync_positive=positive, keyed_payload_target_exists=has_target,
        null_sync_detection=not positive and decision['detection']=='DETECTED_PROVISIONAL',
        null_singleton_location_accept=not positive and decision['location']=='ACCEPT_LOCATION_PROVISIONAL',
        positive_missed_detection=positive and decision['detection']=='REJECT_SYNC',
        technical_unresolved=decision['status'] in ('UNRESOLVED_TECHNICAL','PAYLOAD_UNRESOLVED','CALIBRATION_UNRESOLVED'),
        true_state_in_C=true_state in selected if selected else None,
        C_size=len(selected), location_hypothesis=location,
        phase_correct=None if location is None else location['phase']==phase,
        origin_correct=None if location is None else location['origin']==meta['origin'],
        message=messages, message_denominator=32,
        geometry_present=sum(x['geometry_group_supported'] for x in messages),
        geometry_missing=sum(not x['geometry_group_supported'] for x in messages),
        nominal_supported=sum(x['true_state_nominal_supported'] for x in messages),
        physical_present_nominal_unobserved=sum(x['present_but_nominal_unobserved'] for x in messages),
        readable_present_keyed_bits=len(readable), present_errors=errors if has_target else None,
        present_BER=errors/len(readable) if readable else None,
        OFF_no_target_outputs=sum(x['OFF_no_target_output'] for x in messages),
        wrong_key_attributions=sum(x['wrong_key_attribution'] for x in messages),
        false_missing_group_outputs=sum(x['false_missing_group_output'] for x in messages),
        erasures=sum(x['decoded'] is None for x in messages), source_rows=source_rows,
        observed_cell_rows=observed, exact_source_evaluable=source_evaluable,
        exact_correct_source_cells=sorted(set(exact_correct)) if source_evaluable else None,
        mixed_or_guard_unsupported_claims=sum(x['exact_correct'] is None for x in observed),
        wrong_exact_source_cells=sum(x['exact_correct'] is False for x in observed), raw_phase_bits=raw,
        authenticated=False, empirical_only=True)


def compare(a, b):
    readable = lambda t: {x['slot'] for x in t['message'] if x['embedded_keyed_target_available'] and x['decoded'] is not None}
    correct = lambda t: {x['slot'] for x in t['message'] if x['correct'] is True}
    ra, rb, ca, cb = readable(a), readable(b), correct(a), correct(b)
    common = ra & rb
    source_ok = a['exact_source_evaluable'] and b['exact_source_evaluable']
    sa = set(a['exact_correct_source_cells'] or [])
    sb = set(b['exact_correct_source_cells'] or [])
    return dict(common_readable=sorted(common), a_common_errors=len(common-ca) if common else None,
        b_common_errors=len(common-cb) if common else None,
        a_common_BER=len(common-ca)/len(common) if common else None,
        b_common_BER=len(common-cb)/len(common) if common else None,
        correct_retained=sorted(ca & cb), correct_lost=sorted(ca-cb), correct_added=sorted(cb-ca),
        readable_lost=sorted(ra-rb), readable_added=sorted(rb-ra),
        exact_source_comparison='EVALUABLE' if source_ok else 'UNDEFINED_MIXED_IDENTITY',
        correct_source_retained=sorted(sa & sb) if source_ok else None,
        correct_source_lost=sorted(sa-sb) if source_ok else None,
        correct_source_added=sorted(sb-sa) if source_ok else None,
        a_false_missing_group_outputs=a['false_missing_group_outputs'], b_false_missing_group_outputs=b['false_missing_group_outputs'],
        a_OFF_no_target_outputs=a['OFF_no_target_outputs'], b_OFF_no_target_outputs=b['OFF_no_target_outputs'],
        a_wrong_key_attributions=a['wrong_key_attributions'], b_wrong_key_attributions=b['wrong_key_attributions'])


def paired_payload(a, b):
    """Same host/origin/TRUE phase original votes, independent of selected locations."""
    ma, mb = a['metadata_post_only'], b['metadata_post_only']
    if ma['origin'] != mb['origin'] or ma['key'] != 'key00' or mb['key'] != 'key00':
        return dict(status='UNRESOLVED_DIFFERENT_INPUT_SUPPORT')
    phase = str((-ma['origin']) % 4)
    left, right = a['raw_phase_bits'][phase]['rows'], b['raw_phase_bits'][phase]['rows']
    key = lambda x: (tuple(x['frame_support']), x['slot'])
    la = {key(x): x for x in left if x['reference_defined'] and x['decoded'] is not None}
    lb = {key(x): x for x in right if x['reference_defined'] and x['decoded'] is not None}
    common = sorted(la.keys() & lb.keys())
    ae, be = sum(la[k]['correct'] is False for k in common), sum(lb[k]['correct'] is False for k in common)
    return dict(status='POST_SAME_TRUE_PHASE_PHYSICAL_PAYLOAD_PAIR', true_phase=int(phase),
        raw_bit_denominator=80, common_raw_read=len(common), P0_common_errors=ae if common else None,
        PM_common_errors=be if common else None, P0_common_BER=ae/len(common) if common else None,
        PM_common_BER=be/len(common) if common else None,
        raw_correct_retained=sum(la[k]['correct'] is True and lb[k]['correct'] is True for k in common),
        raw_correct_lost=sum(la[k]['correct'] is True and lb[k]['correct'] is False for k in common),
        raw_correct_added=sum(la[k]['correct'] is False and lb[k]['correct'] is True for k in common),
        common_frame_bit_supports=[dict(frames=list(k[0]), slot=k[1]) for k in common],
        readable_lost=len(la.keys()-lb.keys()), readable_added=len(lb.keys()-la.keys()),
        gated_group_message_comparison=compare(a, b), physical_pair_is_not_selected_peak_comparison=True)


def evaluate(families, decisions, metadata, bits):
    if len(bits)!=32 or any(type(x) is not int or x not in (0,1) for x in bits):
        raise ValueError('post-only32 binary truth required')
    hosts = {}
    for host, fs in families.items():
        queries = {}
        for qid, meta in metadata.items():
            selected = assess(decisions[host][qid], fs[qid], meta, bits)
            forced = assess(method.forced_min(fs[qid]), fs[qid], meta, bits)
            queries[qid] = dict(selected=selected, forced_min=forced, comparison=compare(forced, selected))
        positives = [q['selected'] for q in queries.values() if q['selected']['is_sync_positive']]
        sources = set().union(*(set(q['exact_correct_source_cells'] or []) for q in positives))
        message_union = sorted({r['slot'] for q in positives for r in q['message'] if r['correct'] is True})
        paired = {}
        for origin in method.ORIGINS:
            lookup = {q['selected']['metadata_post_only']['arm']: q['selected'] for q in queries.values() if q['selected']['metadata_post_only']['origin']==origin and q['selected']['metadata_post_only']['key']=='key00'}
            paired[str(origin)] = paired_payload(lookup['P0'], lookup['PM'])
        hosts[host] = dict(paired_P0_PM_same_true_phase_payload=paired, role='confirmation' if host in ('b93-v01','b93-v02') else 'calibration_reuse_not_independent',
            query_denominator=36, null_query_denominator=30, positive_query_denominator=6,
            output_message_denominator=1152, queries=queries,
            null_detections=sum(q['selected']['null_sync_detection'] for q in queries.values()),
            null_singleton_accepts=sum(q['selected']['null_singleton_location_accept'] for q in queries.values()),
            positive_singleton_wrong_origin=sum(q['origin_correct'] is False for q in positives),
            positive_true_state_excluded=sum(q['true_state_in_C'] is False for q in positives),
            positive_geometry_present=sum(q['geometry_present'] for q in positives),
            positive_geometry_missing=sum(q['geometry_missing'] for q in positives),
            positive_nominal_supported=sum(q['nominal_supported'] for q in positives),
            positive_present_unobserved=sum(q['physical_present_nominal_unobserved'] for q in positives),
            positive_readable_present=sum(q['readable_present_keyed_bits'] for q in positives),
            positive_message_correct_union=message_union, positive_exact_source_correct_union=sorted(sources),
            technical_unresolved_queries=sum(q['selected']['technical_unresolved'] for q in queries.values()),
            utility_status='NO_USABLE_POSITIVE_OUTPUT' if not any(q['readable_present_keyed_bits'] for q in positives) else 'PARTIAL_EMPIRICAL_OUTPUT',
            planned_unique_marked_source_cells=42, planned_nominal_cell_views=60, net_payload_bits=32)
    confirm = [hosts[h] for h in ('b93-v01','b93-v02')]
    return dict(status='POST_ONLY_EMPIRICAL', hosts=hosts,
        confirmation=dict(host_denominator=2, null_queries=60, positive_queries=12, all_queries=72,
            candidate_denominator=9864, output_message_slots=2304, positive_message_slots=384,
            positive_physical_present=160, positive_missing=224, positive_nominal_supported=144, positive_unobserved=16,
            hosts_with_null_detection=sum(h['null_detections']>0 for h in confirm),
            hosts_with_null_singleton_accept=sum(h['null_singleton_accepts']>0 for h in confirm),
            hosts_with_usable_positive=sum(h['positive_readable_present']>0 for h in confirm)),
        FPR_population=None, confidence_guarantee=None, authenticated=False, scientific_pass=None)
