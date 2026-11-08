"""Received-only cell planning. No IO, tensors, payload, truth or H137 scoring."""
from copy import deepcopy
from . import b_cdc32_aggregation_v1 as cdc

ARMS = ('B', 'W', 'U')
SCOPES = dict(B='TIME_CONDITIONAL', W='GROUP_PERMUTATION_CONTROL', U='RECEIVED_CLOCK_BASELINE')


def bank(input_id, length):
    if not isinstance(input_id, str) or not input_id or type(length) is not int or length < 45:
        raise ValueError('anonymous input id and received length >=45 required')
    return [dict(query=q, query_start=w, phase=p, start=w+p, view_id=f'{input_id}/E41/a{w+p}',
                 encoder_input_support=list(range(w+p, w+p+41)))
            for q, w in enumerate((0, (length-45)//2, length-45)) for p in range(4)]


def saved_time(submission, length):
    """Allowlist final received-side positions only; never copy old L/H/truth."""
    try:
        if not isinstance(submission, dict): raise ValueError('submission object required')
        status = submission.get('status')
        if status != 'CONDITIONAL_OUTPUTS_SAVED':
            return dict(status='UPSTREAM_UNSUPPORTED', upstream_status=status, mapping=[None]*length,
                        error=str(submission.get('error', 'upstream did not save conditional outputs')))
        nodes = submission['final_nodes']
        if not isinstance(nodes, list) or len(nodes) != length: raise ValueError('complete final positions required')
        mapping = []
        for r, row in enumerate(nodes):
            if not isinstance(row, dict) or type(row.get('received_frame')) is not int or row['received_frame'] != r:
                raise ValueError('invalid received coordinate')
            if 'output_index' not in row: raise ValueError('missing final index field')
            value, origin = row['output_index'], row.get('origin')
            if value is not None and (type(value) is not int or not 0 <= value < 92672 or origin not in ('OLD', 'NEW')):
                raise ValueError('invalid explicit final index/origin')
            if value is None and origin != 'UNRESOLVED': raise ValueError('invalid unresolved final position')
            mapping.append(value)
        known = [x for x in mapping if x is not None]
        if any(a > b for a, b in zip(known, known[1:])): raise ValueError('submitted final positions invert')
        return dict(status='READY', upstream_status=status, mapping=mapping)
    except (ValueError, KeyError, TypeError) as error:
        return dict(status='SEMANTIC_UNRESOLVED', mapping=[None]*length, error=str(error))


def context(mapping, support):
    values = [mapping[r] for r in support]
    unknown = [r for r, n in zip(support, values) if n is None]
    deltas = [dict(received_left=a, delta=y-x) for a, x, y in zip(support, values, values[1:])
              if x is not None and y is not None]
    jumps = [d for d in deltas if d['delta'] != 1]
    return dict(status='CONTEXT_MAPPING_MIXED' if jumps else 'CONTEXT_MAPPING_UNKNOWN' if unknown else 'CONTEXT_MAPPING_CONTIGUOUS',
                visible_mapping=values, unknown_received=unknown, known_adjacent_deltas=deltas,
                known_nonunit_deltas=jumps, has_unknown=bool(unknown), has_known_jump=bool(jumps),
                independent_E41_reset=True, native_181_context_equivalent=False)


def geometry(values):
    if any(v is None for v in values): return None, 'UNKNOWN_TIME_OUTPUT'
    t = (values[0]+3)//4
    if t not in range(1, 45): return None, 'GUARD_OR_OUTSIDE_WRITER_CELLS'
    if values != list(range(4*t-3, 4*t+1)): return None, 'NON_FULL_OR_MIXED_CELL'
    return t, 'CELL_GEOMETRY_FULL'


def _plan(input_id, length, query, time, arm):
    mapping = time['mapping']; units = [u for u in bank(input_id, length) if u['query'] == query and (arm != 'U' or u['phase'] == 0)]
    rows = []
    for unit in units:
        ctx = context(mapping, unit['encoder_input_support'])
        for row in range(1, 11):
            support = list(range(unit['start']+4*row-3, unit['start']+4*row+1))
            visible = [mapping[r] for r in support]
            t, reason = geometry(visible) if time['status'] == 'READY' else (None, time['status'])
            rows.append(dict(**unit, local_row=row, nominal_received_cell=support, visible_cell_mapping=visible,
                             geometry_cell=t, geometry_status=reason, context=ctx))
    groups = []
    for g in range(4):
        candidates = [r for r in rows if r['geometry_cell'] is not None and (r['geometry_cell']-1)//11 == g]
        t = min((r['geometry_cell'] for r in candidates), default=None)
        refs, seen = [], set()
        for r in sorted(candidates, key=lambda r:(r['start'], r['local_row'])):
            if r['geometry_cell'] != t: continue
            for region in cdc.REGIONS:
                eid = cdc.evidence_id(r['view_id'], r['local_row'], region)
                if eid in seen: continue
                seen.add(eid)
                refs.append(dict(**deepcopy(r), evidence_id=eid, input_id=input_id, query_id=f'{input_id}/q{query}',
                    candidate_id='FROZEN_TIME' if arm == 'B' else 'RECEIVED_CLOCK_P0', canonical_cell=t,
                    group=g, observed_region_id=region, carrier_region_id=region, region_slice=cdc.REGION_SLICES[region],
                    status='PLANNED', cause_set=[], qualification='CELL_GEOMETRY_FULL'))
        status = 'READY' if refs else time['status'] if time['status'] != 'READY' else 'NO_FULL_CELL'
        groups.append(dict(group=g, selected_canonical_cell=t, references=refs, status=status,
                           cause_set=[] if refs else [status], failure_details={k:v for k,v in time.items() if k != 'mapping'}))
    return dict(input_id=input_id, query_id=f'{input_id}/q{query}', key_id='key00', arm=arm,
                status=time['status'], mapping_scope=SCOPES[arm], existence_test='NOT_EVALUATED',
                source_authenticated=False, references_frozen_before_bits=True, groups=groups, geometry_rows=rows)


def blind_plans(input_id, length, submission):
    time = saved_time(submission, length)
    b = cdc.reconcile_plans([_plan(input_id, length, q, time, 'B') for q in range(3)])
    return dict(B=b, W=permuted(b), U=unaligned_plans(input_id, length))


def unaligned_plans(input_id, length):
    # No submission argument; fixed p0, identity received clock, no floor rescue.
    time = dict(status='READY', mapping=list(range(length)))
    return cdc.reconcile_plans([_plan(input_id, length, q, time, 'U') for q in range(3)])


def permuted(b):
    result = deepcopy(b)
    for plan in result:
        plan.update(arm='W', mapping_scope=SCOPES['W'], control_transform='t=1+((t-1+11)%44); g=(g+1)%4')
        for group in plan['groups']:
            group['group'] = (group['group']+1)%4
            if group['selected_canonical_cell'] is not None:
                group['selected_canonical_cell'] = 1+(group['selected_canonical_cell']-1+11)%44
            for ref in group['references']:
                ref.update(original_geometry_cell=ref['geometry_cell'], canonical_cell=1+(ref['canonical_cell']-1+11)%44,
                           group=group['group'], control_transform=plan['control_transform'])
        plan['groups'].sort(key=lambda g:g['group'])
    return cdc.reconcile_plans(result)


def attach_receipts(plan, receipts):
    """Attach physical outcomes to frozen refs, without selecting replacements."""
    value = deepcopy(plan)
    for group in value['groups']:
        for ref in group['references']:
            receipt = receipts.get(ref['view_id'], {})
            ref['status'] = 'RETURNED' if receipt.get('status') == 'RETURNED' else receipt.get('status', 'NOT_ENTERED')
            ref['physical_receipt'] = deepcopy(receipt)
            if ref['status'] != 'RETURNED': ref['cause_set'] = [ref['status']]
    return value
