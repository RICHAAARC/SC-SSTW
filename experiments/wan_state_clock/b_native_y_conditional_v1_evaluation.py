"""Single post-only four-layer comparison; no receiver input or selection."""
from pathlib import Path
from collections import Counter
import json
from . import b_native_y_conditional_v1_records as rec
from .b_native_y_conditional_v1_experiment import serial
from .b_native_y_diagnostic_v1_evaluation import new_error, added_correct
from .b_time_layer_diagnostic_v1_evaluation import at, obj, contains, edge_report
from .b_conditional_time_output_v1_evaluation import evaluate_view

LAYERS = ('historical_H', 'historical_final', 'Y_H', 'Y_conditional_final')
KNOWN = ('CORRECT', 'WRONG', 'UNRESOLVED')


def node_status(row, truth, coordinate, missing='UNKNOWN'):
    if row is None:
        return missing
    if (not isinstance(row, dict) or type(truth) is not int or 'output_index' not in row
            or type(row.get('received_frame')) is not int or row['received_frame'] != coordinate):
        return 'UNKNOWN'
    n = row['output_index']
    if n is None:
        return 'UNRESOLVED'
    if type(n) is not int or not 0 <= n < 92672:
        return 'UNKNOWN'
    return 'CORRECT' if n == truth else 'WRONG'


def edge_status(row, a, b, coordinate, missing='UNKNOWN'):
    if row is None:
        return dict(status=missing)
    required = ('output_endpoints', 'delta', 'kind', 'output_missing_indices', 'output_repeated_index')
    if (not isinstance(row, dict) or any(k not in row for k in required)
            or type(row.get('received_left')) is not int or row['received_left'] != coordinate):
        return dict(status='UNKNOWN')
    pair = row['output_endpoints']
    if pair is not None and (not isinstance(pair, list) or len(pair) != 2
                            or any(type(n) is not int or not 0 <= n < 92672 for n in pair)):
        return dict(status='UNKNOWN')
    def index(n):
        return type(n) is int and 0 <= n < 92672
    delta = row['delta']
    delta_valid = (index(delta) or (isinstance(delta, list) and len(delta) == 2
                   and all(index(n) for n in delta) and delta[0] <= delta[1])
                   or (delta is None and pair is None))
    missing_indices, repeated = row['output_missing_indices'], row['output_repeated_index']
    if (not delta_valid or row['kind'] not in (None, 'NORMAL', 'GAP', 'REPEAT', 'AMBIGUOUS')
            or not (missing_indices is None or isinstance(missing_indices, list) and all(index(n) for n in missing_indices))
            or not (repeated is None or index(repeated))):
        return dict(status='UNKNOWN')
    # Format only: legal numeric/event values that disagree with truth stay WRONG.
    return edge_report(row, a, b)


def comparison(old, new):
    known = old in KNOWN and new in KNOWN
    return dict(transition=old+'->'+new,
                correct_support_preserved=(new == 'CORRECT') if known and old == 'CORRECT' else None,
                added_correct=added_correct(old, new), new_wrong=new_error(old, new))


def cost_row(unit, r):
    cost, local = unit.get('cost'), unit.get('local')
    if cost is None:
        return dict(status='UNKNOWN')
    a, b = map(int, cost['offsets'][r:r+2])
    lo, hi = map(int, cost['bounds'][r])
    return dict(status=cost['states'][r], interval=[lo, hi], candidates=list(range(lo, hi+1)) if lo >= 0 else [],
                C=serial(cost['C'][a:b]), dL=serial(local['dL'][a:b]) if local is not None else None,
                best=serial(local['best'][a:b]) if local is not None else None,
                L_status=local['status'][r] if local is not None else 'UNKNOWN',
                candidate=at(obj(unit.get('candidates')).get('nodes'), r),
                attrition=at(obj(unit.get('submission')).get('attrition'), r))


def evaluate(output, config, data, rows, check=lambda: None):
    output = Path(output)
    rec.require_blind(output, rows)
    if any(rows[v+'/historical_public_load']['status'] not in rec.TERMINAL for v in rec.VIEWS):
        raise ValueError('five historical loads must be terminal before truth')
    truth, truth_error = {}, None
    try:
        saved = rec.load(config['truth'])
        for view, n in zip(rec.VIEWS, rec.LENGTHS):
            source = obj(obj(saved.get('views')).get(view)).get('nodes')
            mapping = [obj(at(source, r)).get('truth') for r in range(n)]
            truth[view] = [v if type(v) is int and 0 <= v < 181 else None for v in mapping]
    except (OSError, ValueError, TypeError, KeyError) as error:
        truth_error = str(error)
    technical = bool(truth_error) or any(row['status'] != 'RETURNED' for k, row in rows.items() if k != 'post')
    report = dict(status='POST_IN_PROGRESS', counts={}, views={}, profiles={}, targets=dict(nodes=[], edges=[]),
                  r5=[], startup=[], true_events=[], false_event_transitions=[], structural_errors=[],
                  confirmed_legal_new_wrong=dict(nodes=0, edges=0), counterexample_stop=False,
                  truth_error=truth_error, denominators=rec.denominators(), scientific_pass=None,
                  source_control='Prior 37 source 894->895 correct F referenced only; not read in fixed33',
                  prior_limits='B160, native target tolerance failure, solver optimality unproved remain unchanged',
                  claim_ceiling='same-c02 development shadow integration; no formal replacement, FPR, identity, or general recovery proof')
    totals = Counter()
    folder = output/'post'
    folder.mkdir(exist_ok=True)

    def line(stream, value):
        check()
        stream.write(json.dumps(value, allow_nan=False)+'\n')

    def progress():
        report['counts'] = dict(totals)
        rec.dump(output/'post_progress.json', report)

    with (folder/'nodes.jsonl').open('w') as nodes, (folder/'edges.jsonl').open('w') as edges:
        for view, n in zip(rec.VIEWS, rec.LENGTHS):
            unit = data.get(view, {})
            public = obj(unit.get('historical'))
            old_h = obj(obj(public.get('semantic.json')).get('hard', public.get('blind.json')))
            old_final = obj(public.get('submission.json'))
            y_h, y_final = obj(unit.get('hard')), obj(unit.get('submission'))
            mapping = truth.get(view, [None]*n)
            node_sets = dict(zip(LAYERS, (old_h.get('nodes'), old_final.get('final_nodes'), y_h.get('nodes'), y_final.get('final_nodes'))))
            edge_sets = dict(zip(LAYERS, (old_h.get('relations'), old_final.get('final_relations'), y_h.get('relations'), y_final.get('final_relations'))))
            missing = {k: 'UNKNOWN' for k in LAYERS}
            for layer, op in (('Y_H', 'saved_Y_H_load'), ('Y_conditional_final', 'conditional_submit')):
                if rows[view+'/'+op]['started'] == 0:
                    missing[layer] = 'NOT_EXECUTED'
            counts = Counter()
            complete = not public.get('errors') and all(t is not None for t in mapping)
            for layer in LAYERS:
                complete &= isinstance(node_sets[layer], list) and len(node_sets[layer]) == n
                complete &= isinstance(edge_sets[layer], list) and len(edge_sets[layer]) == n-1
            report['profiles'][view] = unit.get('profile')
            structure = dict(preservation=None, structural_errors=[])
            if y_h and y_final and unit.get('candidates'):
                try:
                    examined = evaluate_view(dict(hard=y_h, L=unit['candidates']), y_final, mapping, n)
                    structure = {k: examined[k] for k in ('preservation', 'structural_errors')}
                    if not structure['preservation']:
                        structure['structural_errors'].append('supplied Y-H or original L not preserved')
                except (ValueError, TypeError, KeyError, IndexError) as error:
                    structure['structural_errors'].append(str(error))
            report['structural_errors'].extend(view+': '+e for e in structure['structural_errors'])
            complete &= y_final.get('status') == 'CONDITIONAL_OUTPUTS_SAVED'
            for r, t in enumerate(mapping):
                saved_nodes = {layer: at(node_sets[layer], r) for layer in LAYERS}
                statuses = {layer: node_status(value, t, r, missing[layer]) for layer, value in saved_nodes.items()}
                after = statuses['Y_conditional_final']
                comparisons = {layer: comparison(statuses[layer], after) for layer in LAYERS[:-1]}
                evidence = cost_row(unit, r)
                value = dict(view=view, received_frame=r, truth=t, layers=saved_nodes, statuses=statuses,
                             comparisons=comparisons, Y_C_L=evidence,
                             historical_L_added_correct=added_correct(statuses['historical_H'], statuses['historical_final']))
                for layer, status in statuses.items():
                    counts['node/'+layer+'/'+status] += 1
                if obj(saved_nodes['Y_conditional_final']).get('origin') == 'NEW':
                    counts['node_added/'+after] += 1
                for layer, comp in comparisons.items():
                    counts['node_transition/'+layer+'/'+comp['transition']] += 1
                    if comp['new_wrong']:
                        counts['node_new_wrong/'+layer] += 1
                if evidence['status'] == 'STARTUP_MODEL_UNDEFINED':
                    report['startup'].append(dict(view=view, received_frame=r, final_status=after,
                        candidate=evidence.get('candidate'), output_index=obj(saved_nodes['Y_conditional_final']).get('output_index')))
                    counts['startup/'+after] += 1
                if (view, r) in (('FULL', 141), ('REPEAT1', 32)):
                    report['targets']['nodes'].append(value)
                if r == 5 and view in ('FULL', 'DELETE1', 'REPEAT1', 'DELETE3'):
                    entry = dict(view=view, statuses=statuses, final_index=obj(saved_nodes['Y_conditional_final']).get('output_index'),
                        wrong4_possible=contains(obj(saved_nodes['Y_H']).get('interval'), 4))
                    entry['corrected'] = None if after not in KNOWN or entry['wrong4_possible'] is None else bool(
                        after == 'CORRECT' and entry['final_index'] == 5 and entry['wrong4_possible'] is False)
                    report['r5'].append(entry)
                if any(status not in KNOWN for status in statuses.values()):
                    complete = False
                line(nodes, value)
            for r in range(n-1):
                a, b = mapping[r:r+2]
                layers = {layer: edge_status(at(edge_sets[layer], r), a, b, r, missing[layer]) for layer in LAYERS}
                after = layers['Y_conditional_final']['status']
                comparisons = {layer: comparison(layers[layer]['status'], after) for layer in LAYERS[:-1]}
                value = dict(view=view, received_left=r, truth_endpoints=[a,b], layers=layers, comparisons=comparisons,
                    historical_L_added_correct=added_correct(layers['historical_H']['status'], layers['historical_final']['status']))
                for layer, item in layers.items():
                    counts['edge/'+layer+'/'+item['status']] += 1
                if any(obj(e).get('received_left') == r for e in y_final.get('new_relations', [])):
                    counts['edge_added/'+after] += 1
                for layer, comp in comparisons.items():
                    counts['edge_transition/'+layer+'/'+comp['transition']] += 1
                    if comp['new_wrong']:
                        counts['edge_new_wrong/'+layer] += 1
                if type(a) is int and type(b) is int and b-a != 1:
                    report['true_events'].append(value)
                if layers['historical_H'].get('false_GAP') or layers['historical_H'].get('false_REPEAT'):
                    report['false_event_transitions'].append(value)
                if (view, r) in (('FULL',140),('FULL',141),('REPEAT1',31),('REPEAT1',32)):
                    report['targets']['edges'].append(value)
                if any(item['status'] not in KNOWN for item in layers.values()):
                    complete = False
                line(edges, value)
            totals.update(counts)
            if structure['preservation'] is True and not structure['structural_errors'] and y_final.get('status') == 'CONDITIONAL_OUTPUTS_SAVED':
                report['confirmed_legal_new_wrong']['nodes'] += counts['node_added/WRONG']
                report['confirmed_legal_new_wrong']['edges'] += counts['edge_added/WRONG']
                report['counterexample_stop'] = any(report['confirmed_legal_new_wrong'].values())
            report['views'][view] = dict(complete=bool(complete), counts=dict(counts),
                submission_status=y_final.get('status'), preservation=structure['preservation'],
                public_errors=public.get('errors', {}),
                filters={k: y_final.get(k) for k in ('S','S_H','T','inversion_participants','proposed','attrition','extension_check','error')},
                C_candidate_slots=len(unit['cost']['C']) if 'cost' in unit else None,
                C_j_scores=unit.get('cost', {}).get('comparisons'))
            technical |= not complete
            progress()
    profiles = [obj(p).get('status') for p in report['profiles'].values()]
    report['model_applicability'] = ('CONFLICT' if 'MODEL_APPLICABILITY_CONFLICT' in profiles else
        'UNCONFIRMED' if any(s != 'METADATA_CONSISTENT' for s in profiles) else 'METADATA_CONSISTENT_NOT_PROOF')
    for row in report['r5']:
        if row['final_index'] == 4:
            report['structural_errors'].append(row['view']+': r5 returned wrong4 after adopted Y correction')
    support_loss = any(totals[f'{kind}_transition/{layer}/CORRECT->{status}'] for kind in ('node','edge')
        for layer in ('historical_H','historical_final','Y_H') for status in ('WRONG','UNRESOLVED'))
    exact_target = (totals['node/Y_conditional_final/CORRECT'] == 806 and totals['edge/Y_conditional_final/CORRECT'] == 801
        and totals['node/Y_conditional_final/UNRESOLVED'] == totals['edge/Y_conditional_final/UNRESOLVED'] == 16
        and len(report['startup']) == 16 and not support_loss)
    report['outcome'] = ('ENGINEERING_ANOMALY' if report['structural_errors'] else
                         'COUNTEREXAMPLE_STOP' if report['counterexample_stop'] else 'EVIDENCE_INCOMPLETE' if technical else
                         'CONDITIONAL_EXACT_RECOVERY' if exact_target else 'PARTIAL_OR_UNRECOVERED')
    report['all_four_r5_corrected'] = None if any(x['corrected'] is None for x in report['r5']) else bool(
        len(report['r5']) == 4 and all(x['corrected'] for x in report['r5']))
    report['status'] = 'EVIDENCE_INCOMPLETE' if technical or report['structural_errors'] else 'DIAGNOSTIC_COMPLETE'
    progress()
    rec.dump(output/'evaluation_summary.json', report)
    return report
