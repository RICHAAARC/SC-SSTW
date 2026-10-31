"""Post-only exact endpoint evaluation; full denominators, unknowns and failures."""
from pathlib import Path
from collections import Counter
from . import b_conditional_time_output_v1_records as records


def _index(value):
    return type(value) is int and 0 <= value < 92672


def _node_status(value, truth):
    return 'UNKNOWN' if not _index(truth) else 'CORRECT' if value == truth else 'WRONG'


def _edge_status(row, a, b, old=False):
    if not _index(a) or not _index(b):
        return 'UNKNOWN', 'UNKNOWN'
    delta = b-a
    kind = 'REPEAT' if delta == 0 else 'NORMAL' if delta == 1 else 'GAP'
    predicted_delta = row.get('delta')
    delta_correct = predicted_delta == ([delta, delta] if old else delta)
    correct = (row.get('output_endpoints') == [a, b] and delta_correct and row.get('kind') == kind
               and row.get('output_missing_indices') == (list(range(a+1, b)) if delta > 1 else None)
               and row.get('output_repeated_index') == (a if delta == 0 else None))
    return 'CORRECT' if correct else 'WRONG', 'CORRECT' if delta_correct else 'WRONG'


def evaluate_view(bundle, submission, truth, frames):
    old = bundle['hard']
    if len(old['nodes']) != frames or len(old['relations']) != frames-1:
        raise ValueError('incomplete original semantic rows')
    truth = truth if isinstance(truth, list) and len(truth) == frames else [None]*frames
    final = submission['final_nodes']
    final_edges = submission['final_relations']
    preservation = submission['old_hard'] == old and submission['original_L'] == bundle['L']
    structural_errors = []
    if len(final) != frames or len(final_edges) != frames-1:
        structural_errors.append('incomplete final rows')
    new_nodes = {row['received_frame']: row for row in submission['new_nodes']}
    new_edges = {row['received_left']: row for row in submission['new_relations']}
    if len(new_nodes) != len(submission['new_nodes']) or len(new_edges) != len(submission['new_relations']):
        structural_errors.append('duplicate new coordinates')
    if any(type(r) is not int or not 0 <= r < frames for r in new_nodes) or any(type(r) is not int or not 0 <= r < frames-1 for r in new_edges):
        structural_errors.append('out-of-range new coordinates')
    node_rows, edge_rows = [], []
    for r, node in enumerate(old['nodes']):
        value = node['output_index']
        new = new_nodes.get(r)
        if value is not None and (r >= len(final) or final[r].get('output_index') != value or final[r].get('origin') != 'OLD'):
            preservation = False
        if new is not None:
            if value is not None or new.get('origin') != 'NEW' or new.get('interval') != node['interval'] or r >= len(final) or new != final[r]:
                structural_errors.append('new/final node contradiction at '+str(r))
        elif value is None and r < len(final) and final[r].get('output_index') is not None:
            structural_errors.append('unlisted final node at '+str(r))
        node_rows.append(dict(received_frame=r, original_interval=node['interval'], truth=truth[r],
            old_index=value, new_index=new['output_index'] if new else None,
            old_status=_node_status(value, truth[r]) if value is not None else 'UNRESOLVED',
            new_status=_node_status(new['output_index'], truth[r]) if new else 'NOT_ADDED'))
    for r, edge in enumerate(old['relations']):
        exact_old = edge['output_endpoints'] is not None
        new = new_edges.get(r)
        if exact_old and (r >= len(final_edges) or final_edges[r] != edge):
            preservation = False
        if new is not None:
            if exact_old or r >= len(final_edges) or final_edges[r] != new:
                structural_errors.append('new/final edge contradiction at '+str(r))
            if r+1 >= len(final) or new.get('output_endpoints') != [final[r].get('output_index'), final[r+1].get('output_index')]:
                structural_errors.append('new edge has no explicit adjacent endpoints at '+str(r))
        elif not exact_old and r < len(final_edges) and final_edges[r].get('output_endpoints') is not None:
            structural_errors.append('unlisted final edge at '+str(r))
        old_status, _ = _edge_status(edge, truth[r], truth[r+1], old=True) if exact_old else ('UNRESOLVED', 'UNRESOLVED')
        new_status, delta_status = _edge_status(new, truth[r], truth[r+1]) if new else ('NOT_ADDED', 'NOT_ADDED')
        edge_rows.append(dict(received_left=r, old_status=old_status, new_status=new_status,
                              new_delta_only_status=delta_status, new_relation=new,
                              truth_endpoints=[truth[r], truth[r+1]]))
    counts = {}
    for name, rows, field in (('old_nodes', node_rows, 'old_status'), ('new_nodes', node_rows, 'new_status'),
                              ('old_relations', edge_rows, 'old_status'), ('new_relations', edge_rows, 'new_status')):
        counts[name] = dict(Counter(row[field] for row in rows))
    correct_edges = {row['received_left'] for row in edge_rows if row['new_status'] == 'CORRECT'}
    witnesses = [row['received_frame'] for row in node_rows if row['new_status'] == 'CORRECT'
                 and isinstance(row['original_interval'], list) and row['original_interval'][1] > row['original_interval'][0]
                 and (row['received_frame'] in correct_edges or row['received_frame']-1 in correct_edges)]
    event_support = dict(Counter(row['new_relation']['kind'] for row in edge_rows if row['new_status'] == 'CORRECT'))
    return dict(nodes=node_rows, relations=edge_rows, counts=counts, preservation=preservation,
                structural_errors=structural_errors, witnesses=witnesses, event_support=event_support,
                truth_complete=all(_index(n) for n in truth), submission_status=submission['status'],
                candidate_attrition={key: submission.get(key) for key in ('S', 'S_H', 'T', 'inversion_participants', 'proposed', 'attrition', 'error')})


def evaluate(output, rows, truth_path):
    output = Path(output)
    terminal = records.load(output/'blind_terminal.json')
    attempt = records.load(output/'attempt.json')
    expected = records.plan(attempt['entry'])
    if (terminal.get('status') != 'ALL_BLIND_ROWS_TERMINAL' or terminal.get('rows') != rows
            or set(rows) != set(expected) or any(row.get('status') not in records.TERMINAL for row in rows.values())):
        raise ValueError('all fixed blind rows must be terminal and saved before opening truth')
    truth_error = None
    try:
        truth = records.load(truth_path)['views'] if truth_path else {}
        if not isinstance(truth, dict):
            raise ValueError('post truth views object required')
    except (OSError, ValueError, KeyError, TypeError) as error:
        truth, truth_error = {}, f'{type(error).__name__}: {error}'
    views, totals = {}, {k: Counter() for k in ('old_nodes', 'new_nodes', 'old_relations', 'new_relations')}
    complete = all(row['status'] == 'RETURNED' and row.get('kind') == expected[key]['kind'] for key, row in rows.items())
    resource = records.load(output/'resources.json')
    complete &= resource.get('status') == 'CLOSED' and not resource.get('error')
    preserved, structural, witnesses, events = True, [], [], Counter()
    for view, frames in zip(records.VIEWS, records.LENGTHS):
        folder = output/'received'/view
        operation = view+('/semantic_submit' if attempt['entry'] == 'semantic' else '/submit')
        try:
            # A genuinely called operation may leave valid saved output even after late failure.
            if rows[operation]['status'] == 'DEPENDENCY_UNRESOLVED':
                raise ValueError('submission was not called; orphan files are not evidence')
            value = evaluate_view(records.load(folder/'semantic.json'), records.load(folder/'submission.json'), truth.get(view), frames)
            complete &= value['truth_complete'] and value['submission_status'] == 'CONDITIONAL_OUTPUTS_SAVED'
            preserved &= value['preservation']
            structural.extend(view+': '+e for e in value['structural_errors'])
            witnesses.extend(dict(view=view, received_frame=r) for r in value['witnesses'])
            events.update(value['event_support'])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            complete = False
            preserved = False
            value = dict(error=f'{type(error).__name__}: {error}',
                         nodes=[dict(received_frame=r, old_status='UNKNOWN', new_status='UNKNOWN') for r in range(frames)],
                         relations=[dict(received_left=r, old_status='UNKNOWN', new_status='UNKNOWN') for r in range(frames-1)],
                         counts={k: {'UNKNOWN': frames if k.endswith('nodes') else frames-1} for k in totals})
        views[view] = value
        for key in totals:
            totals[key].update(value['counts'][key])
    complete = bool(complete and preserved and not structural)
    wrong = totals['new_nodes']['WRONG']+totals['new_relations']['WRONG']
    unknown = totals['new_nodes']['UNKNOWN']+totals['new_relations']['UNKNOWN']
    status = ('COUNTEREXAMPLE' if wrong else 'EVIDENCE_INCOMPLETE' if not complete or unknown
              else 'FINITE_ADDED_SUPPORT' if witnesses else 'NO_ADDED_GAIN_OR_TEST_POWER')
    return dict(status=status, evidence_complete=complete, views=views, counts={k: dict(v) for k, v in totals.items()},
                old_outputs_preserved=preserved, structural_errors=structural, nondegenerate_node_edge_witnesses=witnesses,
                correct_new_event_counts=dict(events), truth_read_error=truth_error,
                failed_rows=[k for k, row in rows.items() if row['status'] != 'RETURNED'], denominators=records.denominators(),
                claim_ceiling='Only the new portion of this fixed single-source batch; old truth errors remain separately reported; no identity/null/general reliability claim')
