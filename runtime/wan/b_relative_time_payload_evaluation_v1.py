"""Once-only post of frozen slots/refs; no oracle planner or payload decoder."""
from pathlib import Path
from collections import Counter
import time
from . import b_relative_time_payload_records_v1 as rec


def reference(ref, mapping, raw, slot, key):
    four, support = ref['nominal_received_cell'], ref['encoder_input_support']
    def actual(indices):
        return [mapping[r] for r in indices] if all(type(r) is int and 0 <= r < len(mapping) for r in indices) else None
    frames, context = actual(four), actual(support)
    declared = ref['canonical_cell']
    true_t = (frames[0]+3)//4 if frames and len(frames) == 4 else None
    full = true_t in range(1,45) and frames == list(range(4*true_t-3, 4*true_t+1))
    exact = bool(full and declared == true_t)
    group = bool(full and (true_t-1)//11 == slot//8)
    unit = raw.get(ref['view_id'], {})
    physical = (unit.get('E41') == unit.get('project') == 'RETURNED' and unit.get('keys', {}).get(key) == 'RETURNED'
                and len(support) == 41 and support == unit.get('encoder_input_support')
                and support == list(range(support[0], support[0]+41)))
    visible = ref['context']['visible_mapping']
    kind = ('UNKNOWN' if context is None else 'CONTIGUOUS' if all(b-a == 1 for a,b in zip(context,context[1:])) else 'MIXED')
    return dict(evidence_id=ref['evidence_id'], declared_cell=declared, true_cell=true_t if full else None,
                four_frame_exact=exact, group_exact=group, same_group_wrong_cell=bool(group and not exact),
                correct_geometry_support=bool(exact and group and physical), actual_physical_support=physical,
                received_four=four, true_four=frames, received_41=support, true_41=context,
                query=ref['query'], phase=ref['phase'], start=ref['start'], local_row=ref['local_row'],
                region=ref['observed_region_id'], declared_context=ref['context']['status'], actual_context=kind,
                context_visible_wrong=None if context is None else sum(a is not None and a != b for a,b in zip(visible,context)),
                context_unknown=sum(v is None for v in visible), native_181_context_equivalent=False)


def slot_rows(saved, mapping, raw, message, key):
    result = []
    for slot in saved['slots']:
        i, value = slot['slot'], slot['value']
        if value is not None and (type(value) is not int or value not in (0,1)):
            raise ValueError('illegal saved output bit')
        selected = [reference(r, mapping, raw, i, key) for r in slot['selected_refs']]
        support = [reference(r, mapping, raw, i, key) for r in slot['supporting_refs']]
        literal = None if value is None else value == message[i]
        wrong = bool(value is not None and any(not r['four_frame_exact'] or not r['group_exact'] for r in support))
        unsupported = bool(value is not None and (not support or any(not r['actual_physical_support'] for r in support)))
        result.append(dict(slot=i, value=value, status=slot['status'], literal_correct=literal,
            correct_supported=literal is True and bool(support) and all(r['correct_geometry_support'] for r in support),
            wrong_attribution=wrong, unsupported_output=unsupported,
            any_selected_wrong_geometry=any(not r['four_frame_exact'] or not r['group_exact'] for r in selected),
            same_group_wrong_cell=any(r['same_group_wrong_cell'] for r in support),
            selected_refs=selected, supporting_refs=support, cause_set=slot.get('cause_set', []),
            missing_refs=slot.get('missing_refs', []), conflict_refs=slot.get('conflict_refs', [])))
    return result


def summary(rows):
    supported = [r for row in rows for r in row['supporting_refs']]
    return dict(planned_slots=32, outputs=sum(r['value'] is not None for r in rows),
        literal_errors=sum(r['literal_correct'] is False for r in rows),
        correct_supported=sum(r['correct_supported'] for r in rows),
        wrong_attribution=sum(r['wrong_attribution'] for r in rows),
        same_group_wrong_cell=sum(r['same_group_wrong_cell'] for r in rows),
        unsupported_output=sum(r['unsupported_output'] for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)),
        supporting_contexts=dict(Counter(r['actual_context'] for r in supported)),
        unique_supporting_evidence=len({r['evidence_id'] for r in supported}),
        full32_supported=all(r['correct_supported'] for r in rows))


def comparison(b, control):
    common = [i for i in range(32) if b[i]['value'] is not None and control[i]['value'] is not None]
    return dict(common_count=len(common), common_slots=common,
        B_common_BER=sum(b[i]['literal_correct'] is False for i in common)/len(common) if common else None,
        control_common_BER=sum(control[i]['literal_correct'] is False for i in common)/len(common) if common else None,
        wrong_to_correct=[i for i in common if control[i]['literal_correct'] is False and b[i]['literal_correct'] is True],
        correct_to_wrong=[i for i in common if control[i]['literal_correct'] is True and b[i]['literal_correct'] is False],
        added_correct_support=[i for i in range(32) if b[i]['correct_supported'] and not control[i]['correct_supported']],
        lost_correct_support=[i for i in range(32) if control[i]['correct_supported'] and not b[i]['correct_supported']],
        coverage_is_not_common_BER=True)


def ref_comparison(b, control):
    result = []
    for a, c in zip(b['slots'], control['slots']):
        x = {r['evidence_id'] for r in a['selected_refs']}; y = {r['evidence_id'] for r in c['selected_refs']}
        address = lambda r: (r['start'], r['phase'], r['local_row'], tuple(r['nominal_received_cell']))
        result.append(dict(slot=a['slot'], same_evidence_ids=sorted(x&y), only_B=sorted(x-y), only_control=sorted(y-x),
            B_value=a['value'], control_value=c['value'],
            physical_addresses_changed={address(r) for r in a['selected_refs']} != {address(r) for r in c['selected_refs']},
            empty_intersection=not bool(x&y)))
    return dict(rows=result, slots_with_shared_refs=sum(bool(r['same_evidence_ids']) for r in result),
                slots_with_different_refs=sum(bool(r['only_B'] or r['only_control']) for r in result),
                claim='FINITE_STRATEGY_COMPARISON_NOT_PURE_PHASE_CAUSALITY')


def paired_raw(output, bplans, pplans):
    """Post-only saved key00 value comparison at matched physical position addresses."""
    rows, cache = [], {}
    def value(ref, bit):
        try:
            uid = ref['view_id']
            if uid not in cache: cache[uid] = rec.read(output/'bank'/uid/'key00_cache.json')
            return cache[uid][ref['evidence_id']]['key00'][str(bit)]
        except (OSError, ValueError, KeyError) as error:
            return dict(value=None, status='UNREADABLE', error=str(error))
    def index(plans):
        found = {}
        for plan in plans:
            for group in plan['groups']:
                for ref in group['references']:
                    for bit in range(8):
                        addr = (ref['query'], ref['query_start'], ref['start'], ref['phase'], ref['local_row'], ref['observed_region_id'], bit)
                        found[addr] = ref
        return found
    new, base = index(bplans), index(pplans)
    for address in sorted(set(new)|set(base)):
        a, b = new.get(address), base.get(address)
        av = value(a,address[-1]) if a else dict(value=None,status='NO_REF')
        bv = value(b,address[-1]) if b else dict(value=None,status='NO_REF')
        rows.append(dict(address=list(address), NEW_eid=a['evidence_id'] if a else None,
            BASE_eid=b['evidence_id'] if b else None, NEW=av, BASE=bv,
            equal=None if av['value'] is None or bv['value'] is None else av['value'] == bv['value']))
    return dict(rows=rows, address_count=len(rows), common_values=sum(r['equal'] is not None for r in rows),
                different_values=sum(r['equal'] is False for r in rows), independent_source_claim=False)


def evaluate(output, source, *, retain_incomplete=False, source_scope='one seen content; five related views'):
    output, source = Path(output), Path(source)
    if rec.read(output/'blind_terminal.json')['status'] != 'ALL_BLIND_SLOTS_TERMINAL':
        raise ValueError('complete blind closure before truth')
    with (output/'post_started.json').open('x') as stream:
        stream.write('{"consumed":true}')
    rec.write(output/'truth_access.json', dict(opened_at=time.time(), after_blind_terminal=True))
    maps = rec.read(source/'private/attack_truth.json')
    message = rec.read(source/'protocol.json')['writer']['bits']
    if len(message) != 32 or any(type(b) is not int or b not in (0,1) for b in message):
        raise ValueError('actual post writer message must be 32 binary integers')
    raw = rec.read(output/'raw_bank.json')
    if retain_incomplete and not (output/'plans.json').is_file():
        # Saved partial plans only: never construct new geometry during post.
        plans = {item['id']: rec.read(output/'plans'/f"{item['id']}.json")
                 for item in rec.inputs() if (output/'plans'/f"{item['id']}.json").is_file()}
    else:
        plans = rec.read(output/'plans.json')
    totals, per_view, counterexamples = {}, {}, []
    for i, (view,n) in enumerate(zip(rec.VIEWS,rec.LENGTHS)):
        mapping = maps[view]
        if len(mapping) != n or any(type(t) is not int or not 0 <= t < 181 for t in mapping):
            raise ValueError('complete valid post mapping required: '+view)
        saved, rows = {}, {}
        for strategy in rec.STRATEGIES:
            stem = f'input{i+5 if strategy == "P_BASE" else i}'
            saved[strategy] = rec.read(output/'final'/stem/f'{strategy}.json')
            rows[strategy] = slot_rows(saved[strategy], mapping, raw, message, 'key01' if strategy=='K_NEW' else 'key00')
            rec.write(output/'post'/view/f'{strategy}.json', rows[strategy])
            totals[view+'/'+strategy] = summary(rows[strategy])
            if strategy == 'B_NEW':
                counterexamples.extend(dict(view=view,slot=r['slot'],literal=r['literal_correct'],wrong_attribution=r['wrong_attribution'],
                    unsupported=r['unsupported_output']) for r in rows[strategy]
                    if r['literal_correct'] is False or r['wrong_attribution'] or r['unsupported_output'])
        if f'input{i}' in plans and f'input{i+5}' in plans:
            paired = paired_raw(output, plans[f'input{i}']['strategies']['B_NEW'], plans[f'input{i+5}']['strategies']['P_BASE'])
        else:
            paired = dict(status='DEPENDENCY_UNRESOLVED', rows=[], reason='saved paired plans unavailable')
        rec.write(output/'post'/view/'paired_raw.json', paired)
        per_view[view] = dict(comparisons={name: comparison(rows['B_NEW'],rows[name]) for name in ('U_NEW','W_NEW','P_BASE')},
                             paired_raw_counts={k:v for k,v in paired.items() if k!='rows'})
        for name in ('U_NEW','W_NEW'):
            compared = ref_comparison(saved['B_NEW'],saved[name])
            rec.write(output/'post'/view/f'B_vs_{name}_refs.json', compared)
            per_view[view][f'B_vs_{name}_refs'] = {k:v for k,v in compared.items() if k!='rows'}
    technical = len(plans) != 10 or any(value['time']['status'] != 'READY' for value in plans.values()) or any(
        any(name in ('TECHNICAL_UNREAD','PARTIAL_EVIDENCE_READ','NOT_ENTERED','UNRESOLVED_RECEIPT') and count
            for name,count in row['statuses'].items()) for row in totals.values())
    supported = sum(totals[view+'/B_NEW']['correct_supported'] for view in rec.VIEWS)
    status = ('COUNTEREXAMPLE' if counterexamples else 'EVIDENCE_INCOMPLETE' if technical else
              'FINITE_CONDITIONAL_READBACK' if supported else 'NO_SUPPORTED_READBACK')
    result = dict(status=status, evidence_complete=not technical, B_correct_supported=supported, scientific_pass=None,
        denominators=rec.DENOMINATORS, rows=totals, views=per_view, B_counterexamples=counterexamples,
        limits=[source_scope, 'no identity existence test or FPR',
                'not pure phase/context causality', 'repeated payload does not authenticate within-group position',
                'native exact-target failure and solver optimality unproved retained', 'no clean OFF total quality'],
        message_source='original saved protocol.writer.bits; post only', post_calls=1)
    rec.write(output/'evaluation_summary.json', result)
    return result
