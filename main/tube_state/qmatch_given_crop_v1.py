"""Given-map identity component only; no blind time estimation or query planning."""
from . import b_time_payload_cdc32_v1 as geometry
from . import b_cdc32_aggregation_v1 as cdc
from . import qmatch_v1 as qmatch

SOURCES = (('b-rank3r-m01', 2), ('b-rank3r-m02', 11), ('b-rank3r-m03', 17), ('b-rank3r-m04', 6))
WINDOWS = (('L', 0), ('M', 53), ('R', 106))
LIMITS = dict(bits_files=192, raw_R0_R1_slots=30720, plans=16, local_calls=48,
              aggregate_calls=16, local_slots=1536, final_slots=512, decisions=32,
              max_slot_query_comparisons=1024, new_physical_calls=0)


def units():
    return [dict(source=source, a=a, arm=arm, key_id=key, view='CROP', length=151,
                 input_id=f'{source}/{arm}/CROP', geometry_scope='GIVEN_CROP_MAP')
            for source, a in SOURCES for arm in ('OFF', 'P0') for key in ('key00', 'key01')]


def bank(unit):
    return [dict(window=window, query=q, phase=p, start=w+p,
                 stem=f"{unit['input_id']}/{window}/p{p}", key_id=unit['key_id'],
                 file=f"observations/{unit['input_id']}/{window}/p{p}/{unit['key_id']}/bits.json")
            for q, (window, w) in enumerate(WINDOWS) for p in range(4)]


def given_plans(unit, headers, operations):
    """Only geometry/header/operation metadata. Values and claims are not inputs."""
    time = dict(status='READY', mapping=list(range(unit['a'], unit['a']+151)))
    entries = {(u['query'], u['phase']): u for u in bank(unit)}
    plans = []
    for query in range(3):
        plan = geometry._plan(unit['input_id'], 151, query, time, 'B')
        plan.update(key_id=unit['key_id'], arm=unit['arm'], geometry_scope='GIVEN_CROP_MAP',
                    mapping_scope='TIME_CONDITIONAL', alignment_source='PREDECLARED_GIVEN_CROP_MAP',
                    given_crop_start=unit['a'])
        for row in plan['geometry_rows']:
            entry = entries[query, row['phase']]
            row['view_id'] = headers.get(entry['file'], {}).get('view_id')
        for group in plan['groups']:
            for ref in group['references']:
                entry = entries[query, ref['phase']]
                header = headers.get(entry['file'], {})
                uid = header.get('view_id')
                addresses = [entry['stem']+'/'+suffix for suffix in ('E', 'DCT', unit['key_id']+'/bits')]
                returned = bool(uid) and all(isinstance(operations.get(a), dict)
                                            and operations[a].get('status') == 'RETURNED' for a in addresses)
                ref.update(view_id=uid, evidence_id=f"{uid}/row{ref['local_row']}/{ref['observed_region_id']}" if uid else None,
                           candidate_id='GIVEN_CROP_MAP', geometry_scope='GIVEN_CROP_MAP',
                           status='RETURNED' if returned else 'TECHNICAL_UNREAD',
                           cause_set=[] if returned else ['TECHNICAL_CACHE'],
                           original_bits_file=entry['file'], original_operations=addresses,
                           original_operation_statuses={a: operations.get(a, {}).get('status', 'MISSING')
                                                        if isinstance(operations.get(a, {}), dict) else 'INVALID' for a in addresses})
        plans.append(plan)
    return cdc.reconcile_plans(plans)


def evidence_from_aggregate(final, unit, *, input_errors=(), technical_errors=()):
    """Reuse CDC's returned hard-bit semantics, without voting or query access."""
    errors = list(input_errors)
    evidence = dict(scope=qmatch.SCOPE, geometry_scope='GIVEN_CROP_MAP', unit=dict(unit),
                    input_errors=errors, technical_errors=list(technical_errors), slots=[])
    if not isinstance(final, dict) or final.get('status') != 'RETURNED' or not qmatch._ordered32(final.get('slots')):
        errors.append(dict(kind='FINAL_STRUCTURE', reason='returned ordered32 aggregate required'))
        return dict(evidence, status='INPUT_UNRESOLVED')
    for i, row in enumerate(final['slots']):
        status, value = row.get('status'), row.get('value')
        conflict = status == 'CONFLICT_ERASURE' or bool(row.get('conflict_refs')) or 'BIT_CONFLICT' in row.get('cause_set', [])
        supports = row.get('supporting_refs', [])
        valid = bool(not conflict and qmatch._binary(value) and status in qmatch.READS and supports)
        provenance = []
        for ref in supports if valid else []:
            if (ref.get('status') != 'RETURNED' or not ref.get('view_id')
                    or ref.get('evidence_id') != f"{ref['view_id']}/row{ref['local_row']}/{ref['observed_region_id']}"
                    or len(ref.get('original_operations', [])) != 3
                    or set(ref.get('original_operation_statuses', {})) != set(ref.get('original_operations', []))
                    or any(s != 'RETURNED' for s in ref.get('original_operation_statuses', {}).values())):
                errors.append(dict(kind='SUPPORT_RECEIPT', slot=i, reason='asserted support lacks actual original address/returns'))
            provenance.append(dict(evidence_id=ref['evidence_id'], view_id=ref['view_id'],
                                   local_row=ref['local_row'], local_bit=i % 8, region=ref['observed_region_id'],
                                   canonical_cell=ref['canonical_cell'], key_id=unit['key_id'],
                                   original_bits_file=ref['original_bits_file'], original_operations=ref['original_operations']))
        evidence['slots'].append(dict(slot=i, value=value if valid else None, valid=valid,
            saved_status=status, conflict=conflict,
            technical_missing=status in ('TECHNICAL_UNREAD', 'PARTIAL_EVIDENCE_READ') or bool(row.get('missing_refs')),
            support_kind=('PARTIAL' if status == 'PARTIAL_EVIDENCE_READ' else 'NORMAL') if valid else None,
            supporting_refs=provenance))
    return dict(evidence, status='INPUT_UNRESOLVED' if errors else 'READY')
