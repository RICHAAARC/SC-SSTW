"""Post-blind evaluation only; this module is never imported by the receiver core."""
from main.tube_state import blind_temporal_segmented_v1 as core


def true_parameters(view):
    return (37, 0, 0) if view == 'CROP_MID' else (0, 43, 45) if view == 'DELETE43_PHASE' else None


def unavailable_evaluation(n, status):
    return dict(status=status, frames=[dict(received=i, status=status, exact=None, absolute_error=None) for i in range(n)],
        source_cells=[dict(source_t=t, status=status) for t in range(1, 45)],
        source_states=[dict(state=k, status=status) for k in range(22)],
        bits=[dict(bit=i, status=status, decoded=None, correct=None, structurally_present=None, false_coverage=None) for i in range(32)],
        denominators=dict(frames=n, cells=44, states=22, bits=32, structurally_present=24, structurally_missing=8))


def evaluate(plan, payload, truth_path, truth_bits, *, pair_available_starts=()):
    if len(truth_bits) != 32 or any(type(b) is not int or b not in (0, 1) for b in truth_bits):
        raise ValueError('32 binary segmented truth bits required')
    n = len(truth_path)
    pair_available = set(map(int, pair_available_starts))
    estimated = plan['frame_consensus']
    inverse = {s: i for i, s in enumerate(truth_path)}
    full_cells, cell_rows, state_rows = {}, [], []
    for t in range(1, 45):
        source = list(range(4*t-3, 4*t+1))
        received = [inverse[s] for s in source if s in inverse]
        complete = len(received) == 4 and received == list(range(received[0], received[0]+4))
        if complete:
            full_cells[t] = received[0]
        assigned = [u for u in plan['units'] if u['source_t'] == t]
        correct = [u for u in assigned if complete and u['received_start'] == received[0]]
        cell_rows.append(dict(source_t=t, segment=(t-1)//11, true_status='FULL' if complete else 'PARTIAL' if received else 'ABSENT',
            true_received_positions=received, assigned_units=assigned, mapping_correct=len(correct),
            wrong_source_attribution=len(assigned)-len(correct), status='MAPPED' if assigned else plan['status'] if plan['status'] == 'INCOMPLETE' else 'UNASSIGNED'))
    for k in range(22):
        source = list(range(8*k+1, 8*k+9))
        received = [inverse[s] for s in source if s in inverse]
        complete = len(received) == 8 and received == list(range(received[0], received[0]+8))
        inferred_a = [a for a in range(n-7) if estimated[a:a+8] == source]
        observed_a = [a for a in inferred_a if a in pair_available]
        state_rows.append(dict(state=k, true_status='FULL' if complete else 'PARTIAL' if received else 'ABSENT',
            true_received_positions=received, inferred_starts=inferred_a,
            correct_mapping=bool(complete and received[0] in inferred_a),
            wrong_mappings=[a for a in inferred_a if not complete or a != received[0]],
            geometry='COMPLETE_SOURCE_BLOCK' if inferred_a else 'NO_CONSENSUS_COMPLETE_BLOCK',
            actual_pilot_starts=observed_a,
            observation='COMPLETE_PILOT_AVAILABLE' if observed_a else 'PILOT_UNAVAILABLE' if inferred_a else 'NO_CONSENSUS_COMPLETE_PILOT'))
    missing_frame_status = 'UNRESOLVED' if plan['status'] == 'UNRESOLVED' else plan['status'] if plan['status'] in ('INCOMPLETE','FAILED','INTERRUPTED','NOT_ENTERED') else 'UNAVAILABLE'
    frames = [dict(received=i, truth=t, estimated=e, status=missing_frame_status if e is None else 'MAPPED',
                   exact=None if e is None else e == t, absolute_error=None if e is None else abs(e-t)) for i, (e, t) in enumerate(zip(estimated, truth_path))]
    unit_attribution = []
    for u in plan['units']:
        a, t = u['received_start'], u['source_t']
        actual = truth_path[a:a+4]
        true_t = (actual[0]+3)//4 if actual == list(range(actual[0], actual[0]+4)) and (actual[0]+3)%4 == 0 else None
        true_g = (true_t-1)//11 if true_t is not None and 1 <= true_t <= 44 else None
        unit_attribution.append(dict(**u, actual_source_frames=actual, actual_source_t=true_t, actual_segment=true_g,
            wrong_source_unit=true_t != t, wrong_segment=true_g != u['segment'],
            actual_complete_pilot_available=u.get('complete_pilot_received_start') in pair_available,
            true_context_flags=dict(missing_source_prefix=truth_path[0] > 0,
                after_internal_gap=any(truth_path[i] != truth_path[i-1]+1 for i in range(1, a+1)))))
    bits = []
    for i, decoded in enumerate(payload['decoded_bits']):
        g = i//8
        structural = any((t-1)//11 == g for t in full_cells)
        seg = payload['segments'][g]
        b = seg['bits'][i%8]
        selected = [u for u in unit_attribution if u['segment'] == g]
        used_starts = {u['received_start'] for u in seg.get('used_units', [])}
        available = [u for u in selected if u['received_start'] in used_starts]
        bits.append(dict(bit=i, decoded=decoded, truth=truth_bits[i], correct=None if decoded is None or not structural else decoded == truth_bits[i],
            message_agreement=None if decoded is None else decoded == truth_bits[i],
            scoring_status='SCORED' if structural and decoded is not None else 'NOT_SCORED_STRUCTURAL_MISSING' if not structural else 'UNREADABLE',
            status=b['status'], structurally_present=structural, false_coverage=bool(not structural and decoded is not None),
            vote_count=b['count'], wrong_source_unit_votes=120*sum(u['wrong_source_unit'] for u in available),
            wrong_segment_votes=120*sum(u['wrong_segment'] for u in available), tie=b.get('tie'),
            zero_coefficients=b.get('zero_coefficients')))
    errors = sum(b['correct'] is False for b in bits)
    read = sum(b['decoded'] is not None for b in bits)
    valid_read = sum(b['structurally_present'] and b['decoded'] is not None for b in bits)
    return dict(status='EVALUATED', path_status=plan['status'], payload_status=payload['status'],
        denominators=dict(frames=n, cells=44, states=22, bits=32, structurally_present=24, structurally_missing=8),
        frames=frames, source_cells=cell_rows, source_states=state_rows, unit_attribution=unit_attribution, bits=bits,
        summary=dict(read_bits=read, valid_covered_bits=valid_read, correct_bits=valid_read-errors, errors=errors,
            unread_output_bits=32-read, structural_missing_bits=8, present_unread_bits=24-valid_read,
            false_coverage_bits=sum(b['false_coverage'] for b in bits),
            unresolved_frames=sum(f['status'] == 'UNRESOLVED' for f in frames),
            incomplete_frames=sum(f['status'] == 'INCOMPLETE' for f in frames),
            unavailable_frames=sum(f['estimated'] is None and f['status'] not in ('INCOMPLETE','UNRESOLVED') for f in frames),
            mapped_frames=sum(f['estimated'] is not None for f in frames),
            wrong_frames=sum(f['exact'] is False for f in frames), wrong_source_units=sum(u['wrong_source_unit'] for u in unit_attribution),
            wrong_segment_units=sum(u['wrong_segment'] for u in unit_attribution)))


def compare(base, candidate):
    b, c = base['bits'], candidate['bits']
    present = [i for i in range(32) if b[i]['structurally_present'] and c[i]['structurally_present']]
    common = [i for i in present if b[i]['decoded'] is not None and c[i]['decoded'] is not None]
    return dict(denominator=32, common_bits=common,
        baseline_common_ber=None if not common else sum(b[i]['correct'] is False for i in common)/len(common),
        candidate_common_ber=None if not common else sum(c[i]['correct'] is False for i in common)/len(common),
        added_coverage=[i for i in present if b[i]['decoded'] is None and c[i]['decoded'] is not None],
        lost_coverage=[i for i in present if b[i]['decoded'] is not None and c[i]['decoded'] is None],
        introduced_errors=[i for i in common if b[i]['correct'] is True and c[i]['correct'] is False],
        repaired_errors=[i for i in common if b[i]['correct'] is False and c[i]['correct'] is True],
        baseline_false_coverage=[i for i in range(32) if b[i]['false_coverage']],
        candidate_false_coverage=[i for i in range(32) if c[i]['false_coverage']])
