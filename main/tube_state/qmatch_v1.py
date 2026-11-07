"""Single external claim against saved CDC32 evidence; no IO or decoding.

Validity means legal saved support, not correct time, source, or bit truth.
Preparing evidence never receives a claim. Matching never selects evidence.
"""

SCOPE = 'CONDITIONAL_MESSAGE_MATCH'
VIEWS = ('FULL', 'CROP', 'DELETE1', 'REPEAT1', 'DELETE3')
LENGTHS = (181, 101, 180, 182, 178)
SCOPES = dict(B_NEW='RELATIVE_TIME_CONDITIONAL', U_NEW='RECEIVED_CLOCK_FOUR_PHASE',
              W_NEW='FIXED_PLUS_ONE_TIME_SHIFT', K_NEW='RELATIVE_TIME_CONDITIONAL',
              P_BASE='PAIRED_GIVEN_TIME')
READS = {'CONDITIONAL_READ', 'PARTIAL_EVIDENCE_READ'}


def fixed_units():
    return [dict(source_package=source, view=view, length=length,
                 input_id=f'input{i + (5 if strategy == "P_BASE" else 0)}',
                 strategy=strategy, key_id='key01' if strategy == 'K_NEW' else 'key00')
            for source in ('r01', 'r02') for i, (view, length) in enumerate(zip(VIEWS, LENGTHS))
            for strategy in SCOPES]


def _ordered32(rows):
    return (isinstance(rows, list) and len(rows) == 32 and
            all(isinstance(r, dict) and type(r.get('slot')) is int and r['slot'] == i
                for i, r in enumerate(rows)))


def _binary(value):
    return type(value) is int and value in (0, 1)


def _returned(operations, address, kind):
    row = operations.get(address)
    return (isinstance(row, dict) and row.get('kind') == kind and row.get('status') == 'RETURNED'
            and type(row.get('started')) is int and row['started'] == 1
            and type(row.get('returned')) is int and row['returned'] == 1)


def _address(ref):
    return tuple(ref.get(k) for k in ('input_id', 'query_id', 'view_id', 'local_row',
                                     'observed_region_id', 'evidence_id', 'group'))


def _support(ref, unit, slot, operations):
    """Validate actual saved reference/operation addresses, not time authenticity."""
    stem, key, strategy = (unit[k] for k in ('input_id', 'key_id', 'strategy'))
    q, phase, start, row, group = (ref.get(k) for k in ('query', 'phase', 'start', 'local_row', 'group'))
    if not (all(type(x) is int for x in (q, phase, start, row, group)) and q in range(3)
            and phase in range(4) and row in range(1, 11) and group == slot // 8):
        raise ValueError('illegal support coordinates')
    query_start = (0, (unit['length'] - 45) // 2, unit['length'] - 45)[q]
    uid = f'{stem}/E41/a{query_start + phase}'
    region = ref.get('observed_region_id')
    if (region not in ('R0', 'R1') or ref.get('carrier_region_id') != region
            or ref.get('input_id') != stem or ref.get('candidate_id') != strategy
            or ref.get('query_id') != f'{stem}/q{q}' or ref.get('view_id') != uid
            or ref.get('query_start') != query_start or start != query_start + phase
            or ref.get('evidence_id') != f'{uid}/row{row}/{region}'
            or ref.get('encoder_input_support') != list(range(start, start + 41))
            or ref.get('status') != 'RETURNED'):
        raise ValueError('support scope/address/status mismatch')
    cell = ref.get('canonical_cell')
    if type(cell) is not int or cell not in range(1, 45) or (cell - 1) // 11 != group:
        raise ValueError('support canonical group mismatch')
    receipt = ref.get('physical_receipt')
    if not isinstance(receipt, dict):
        raise ValueError('missing embedded support receipt')
    if (any(receipt.get(k) != ref.get(k) for k in
            ('view_id', 'input_id', 'start', 'phase', 'query', 'encoder_input_support'))
            or any(receipt.get(k) != 'RETURNED' for k in ('status', 'E41', 'project'))
            or not isinstance(receipt.get('keys'), dict) or receipt['keys'].get(key) != 'RETURNED'):
        raise ValueError('embedded E41/project/key receipt mismatch')
    addresses = [f'{uid}/{op}' for op in ('E41', 'project', key)] + [f'{stem}/{strategy}/q{q}/local']
    for address, kind in zip(addresses, ('E41', 'project', 'bits', 'local')):
        if not _returned(operations, address, kind):
            raise ValueError('missing or nonreturned operation: ' + address)
    return dict(evidence_id=ref['evidence_id'], input_id=stem, local_query_id=ref['query_id'],
                key_id=key, view_id=uid, local_row=row, local_bit=slot % 8,
                region=region, canonical_cell=cell, operations=addresses)


def prepare_saved(final, operations, unit, *, input_errors=()):
    """Inspect saved records once, independently of q; never re-aggregate bits."""
    errors = [dict(e) for e in input_errors]
    result = dict(scope=SCOPE, unit=dict(unit), input_errors=errors, slots=[])
    if unit not in fixed_units():
        errors.append(dict(kind='FINAL_STRUCTURE', reason='unit outside fixed source/view/strategy roster'))
    if not isinstance(final, dict):
        errors.append(dict(kind='FINAL_STRUCTURE', reason='final object unavailable'))
    if not isinstance(operations, dict):
        errors.append(dict(kind='RECEIPT_INPUT', reason='state operation object unavailable'))
    if errors:
        return dict(result, status='INPUT_UNRESOLVED')
    stem, strategy, key = (unit[k] for k in ('input_id', 'strategy', 'key_id'))
    aggregate = f'{stem}/{strategy}/aggregate'
    required = dict(input_id=stem, strategy=strategy, key_id=key, kind='final',
                    operation=aggregate, path=f'final/{stem}/{strategy}.json', status='RETURNED',
                    mapping_scope=SCOPES[strategy], source_relation='SINGLE_INPUT_CONDITIONAL', planned_slots=32)
    if any(final.get(k) != v for k, v in required.items()) or not _ordered32(final.get('slots')):
        errors.append(dict(kind='FINAL_STRUCTURE', reason='final scope/status/ordered32 structure invalid'))
    if not _returned(operations, aggregate, 'aggregate'):
        errors.append(dict(kind='RECEIPT_INPUT', reason='aggregate operation not returned: ' + aggregate))
    locals_ = final.get('per_query_slots')
    if not isinstance(locals_, list) or len(locals_) != 3:
        errors.append(dict(kind='FINAL_STRUCTURE', reason='three saved local read plans required'))
    else:
        for q, local in enumerate(locals_):
            if (not isinstance(local, dict) or local.get('query_id') != f'{stem}/q{q}'
                    or local.get('input_id') != stem or local.get('key_id') != key
                    or local.get('mapping_scope') != SCOPES[strategy] or not _ordered32(local.get('slots'))):
                errors.append(dict(kind='FINAL_STRUCTURE', reason=f'local q{q} scope/ordered32 invalid'))
    if errors:
        return dict(result, status='INPUT_UNRESOLVED')
    for i, row in enumerate(final['slots']):
        parts = [local['slots'][i] for local in locals_]
        fields = ('selected_refs', 'supporting_refs', 'conflict_refs', 'missing_refs', 'cause_set')
        if not isinstance(row.get('status'), str) or any(not isinstance(row.get(k), list) for k in fields) or any(
                any(not isinstance(p.get(k), list) for k in ('selected_refs', 'conflict_refs', 'missing_refs', 'cause_set'))
                or not isinstance(p.get('status'), str) or type(p.get('technical_failure')) is not bool for p in parts):
            errors.append(dict(kind='FINAL_STRUCTURE', reason=f'slot {i} malformed status/reference/failure fields'))
            continue
        all_refs = row['selected_refs'] + row['supporting_refs'] + [r for p in parts for r in p['selected_refs']]
        if any(not isinstance(r, dict) for r in all_refs):
            errors.append(dict(kind='FINAL_STRUCTURE', reason=f'slot {i} malformed reference object'))
            continue
        # Failed/technical local records may contain stale digits, not observed bit conflicts.
        known_values = {p['value'] for q, p in enumerate(parts) if _binary(p.get('value'))
                        and p.get('status') == 'LOCAL_CONDITIONAL_READ' and not p['technical_failure']
                        and _returned(operations, f'{stem}/{strategy}/q{q}/local', 'local')}
        conflict = (row.get('status') == 'CONFLICT_ERASURE' or bool(row['conflict_refs'])
                    or 'BIT_CONFLICT' in row['cause_set'] or known_values == {0, 1}
                    or any(p.get('status') == 'LOCAL_CONFLICT' or p['conflict_refs']
                           or 'BIT_CONFLICT' in p['cause_set'] for p in parts))
        missing_local_receipts = [f'{stem}/{strategy}/q{q}/local' for q in range(3)
                                  if not _returned(operations, f'{stem}/{strategy}/q{q}/local', 'local')]
        technical = (row.get('status') in ('TECHNICAL_UNREAD', 'NOT_ENTERED', 'UNRESOLVED_RECEIPT')
                     or bool(row['missing_refs']) or bool(missing_local_receipts)
                     or any(p['technical_failure'] or p['missing_refs'] for p in parts))
        reasons, provenance = [], []
        value = row.get('value')
        if not conflict and row.get('status') in READS and _binary(value):
            if not row['supporting_refs']:
                reasons.append('EMPTY_SUPPORT')
            for ref in row['supporting_refs']:
                # Validate every asserted support; a missing required receipt affects this input unit.
                try:
                    support = _support(ref, unit, i, operations)
                    if not any(_address(r) == _address(ref) for r in row['selected_refs']):
                        raise ValueError('support absent from saved selected references')
                    local = parts[ref['query']]
                    if (local.get('status') != 'LOCAL_CONDITIONAL_READ' or not _binary(local.get('value'))
                            or local.get('value') != value
                            or local['technical_failure'] or not any(_address(r) == _address(ref) for r in local['selected_refs'])):
                        raise ValueError('support absent from legal saved local read')
                    provenance.append(support)
                except (KeyError, TypeError, ValueError) as error:
                    errors.append(dict(kind='SUPPORT_RECEIPT', slot=i, reason=str(error)))
        elif not conflict and (value is not None or row.get('status') in READS):
            reasons.append('ILLEGAL_SAVED_BIT_OR_READ_STATUS')
        if row.get('status') not in READS | {'CONFLICT_ERASURE', 'TECHNICAL_UNREAD', 'ERASURE_OR_REJECT',
                                       'NOT_ENTERED', 'UNRESOLVED_RECEIPT'}:
            reasons.append('UNKNOWN_SLOT_STATUS')
        valid = bool(not conflict and not reasons and row.get('status') in READS and _binary(value) and provenance)
        result['slots'].append(dict(slot=i, value=value if valid else None, valid=valid,
                                    saved_status=row.get('status'), conflict=conflict,
                                    technical_missing=bool(technical or reasons), reasons=reasons,
                                    missing_local_receipts=missing_local_receipts,
                                    support_kind=('PARTIAL' if row['status'] == 'PARTIAL_EVIDENCE_READ' else 'NORMAL') if valid else None,
                                    supporting_refs=provenance if valid else []))
    return dict(result, status='INPUT_UNRESOLVED' if errors else 'READY')


def match_claim(evidence, claim_id, bits):
    """One call, one external 32-bit claim. No search, OR, or early selection."""
    if not isinstance(claim_id, str) or not claim_id or not isinstance(bits, (list, tuple)) or len(bits) != 32 or not all(map(_binary, bits)):
        raise ValueError('one named ordinary binary 32-bit claim required')
    result = dict(scope=SCOPE, unit=dict(evidence['unit']), claim_id=claim_id,
                  input_errors=evidence['input_errors'], verdict='INPUT_UNRESOLVED', comparisons=0,
                  valid_count=None, matching_slots=[], contradictory_slots=[], conflict_slots=[],
                  technical_missing_slots=[], unresolved_slots=[], normal_support_slots=[],
                  partial_support_slots=[], mismatch_basis='NONE')
    if evidence['status'] != 'READY':
        return result
    rows = evidence['slots']
    valid = [r for r in rows if r['valid']]
    same, different = [], []
    for row in valid:
        (same if row['value'] == bits[row['slot']] else different).append(row['slot'])
    kinds = {r['support_kind'] for r in valid if r['slot'] in different}
    basis = 'MIXED_NORMAL_AND_PARTIAL' if len(kinds) == 2 else 'NORMAL_ONLY' if kinds == {'NORMAL'} else 'PARTIAL_ONLY' if kinds else 'NONE'
    return dict(result, verdict='MISMATCH' if different else 'MATCH' if len(valid) == 32 else 'UNRESOLVED',
                comparisons=len(valid), valid_count=len(valid), matching_slots=same, contradictory_slots=different,
                conflict_slots=[r['slot'] for r in rows if r['conflict']],
                technical_missing_slots=[r['slot'] for r in rows if r['technical_missing']],
                unresolved_slots=[r['slot'] for r in rows if not r['valid']],
                normal_support_slots=[r['slot'] for r in valid if r['support_kind'] == 'NORMAL'],
                partial_support_slots=[r['slot'] for r in valid if r['support_kind'] == 'PARTIAL'],
                mismatch_basis=basis)
