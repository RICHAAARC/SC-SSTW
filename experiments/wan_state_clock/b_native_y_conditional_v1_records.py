"""Fixed33 saved-Y plan and stdlib-only consumed recovery."""
from pathlib import Path
from collections import Counter
from . import b_time_layer_diagnostic_v1_records as prior
from .b_minimal_l_v1_records import load, dump, output_state

VIEWS = prior.VIEWS
LENGTHS = prior.LENGTHS
LIMITS = dict(rss_bytes=8*1024**3, total_wall_seconds=6300, per_view_seconds=600)
TERMINAL = ('RETURNED', 'FAILED', 'INTERRUPTED', 'RESOURCE_UNRESOLVED', 'DEPENDENCY_UNRESOLVED')


def plan():
    kinds = {'saved_codebook_load': 'saved_codebook_load'}
    for view in VIEWS:
        for op in ('saved_Y_raw_load', 'saved_Y_H_load', 'C', 'L', 'conditional_submit'):
            kinds[view+'/'+op] = op
    kinds['blind_save'] = 'blind_terminal_save'
    for view in VIEWS:
        kinds[view+'/historical_public_load'] = 'historical_public_load'
    kinds['post'] = 'post'
    return {k: dict(kind=v, planned=1, started=0, returned=0, status='NOT_STARTED') for k, v in kinds.items()}


def blind_keys():
    return [k for k, v in plan().items() if v['kind'] not in ('blind_terminal_save', 'historical_public_load', 'post')]


def budgets():
    return dict(Counter(row['kind'] for row in plan().values()))


def denominators():
    return dict(operations=33, blind_operations=26, views=5, nodes=822, edges=817, lanes=4110,
                input_replica_bit_slots=517860, C_candidate_slots_expected=876, C_j_scores_expected=4032,
                startup_nodes_expected=16, independent_new_sources=0,
                forbidden_budget={k: 0 for k in ('reader', 'codec', 'BCH_scan', 'H_solve', 'codebook_rebuild',
                    'G', 'E41', 'VAE', 'model', 'GPU', 'writer', 'solver', 'training', 'payload')})


def skeleton():
    return {view: dict(frames=n, edges=n-1, lanes=n*5, input_replica_bit_slots=n*630,
                      C_candidate_slots_expected=slots, C_j_scores_expected=scores)
            for view, n, slots, scores in zip(VIEWS, LENGTHS, (195,101,193,196,191), (886,505,880,891,870))}


def preflight(config, output):
    try:
        roots = [Path(config[k]).expanduser().resolve() for k in ('new_y_root', 'historical_public_root')]
        roots.append(Path(config['truth']).expanduser().resolve().parent)
        output = Path(output).expanduser().resolve()
        for target in (output, output.with_name(output.name+'-logs')):
            if any(target == source or target.is_relative_to(source) or source.is_relative_to(target) for source in roots):
                raise ValueError('output/log overlaps protected input')
        return dict(status='PREFLIGHT_READY', protected_roots=list(map(str, roots)), physical_attempt_reserved=False)
    except (KeyError, OSError, ValueError, TypeError) as error:
        return dict(status='PREFLIGHT_UNRESOLVED', error=str(error), computations_this_invocation=0)


def recover(output):
    saved, errors, rows = {}, {}, plan()
    for name in ('attempt', 'state', 'counts', 'resources', 'blind_terminal', 'blind_summary', 'post_progress', 'evaluation_summary', 'result'):
        try:
            value = load(Path(output)/(name+'.json'))
            if not isinstance(value, dict):
                raise ValueError('saved metadata object required')
            saved[name] = value
        except (OSError, ValueError) as error:
            errors[name] = str(error)
    for key, row in rows.items():
        actual = saved.get('state', {}).get(key)
        if prior.valid_receipt(actual, row):
            row.update(actual)
            if row['status'] == 'STARTED':
                row.update(status='UNFINISHED_RECEIPT', error='consumed; no resume')
        else:
            row.update(status='UNRESOLVED_RECEIPT', started=None, returned=None)
    return dict(status='READ_ONLY_EXISTING_DIAGNOSTIC', computations_this_invocation=0, post_calls_this_invocation=0,
                saved=saved, rows=rows, read_errors=errors, fixed_slots=skeleton(), denominators=denominators())


def summary(result, output):
    snapshot = recover(output)
    saved = snapshot['saved']
    resource = saved.get('resources', {})
    diagnostic = saved.get('evaluation_summary') or saved.get('post_progress') or {}
    recorded = result.get('evaluation_status', saved.get('result', {}).get('evaluation_status'))
    return dict(status=result.get('status'), output=str(output),
                computations_this_invocation=result.get('computations_this_invocation'),
                post_calls_this_invocation=result.get('post_calls_this_invocation'),
                evaluation_status='EVIDENCE_INCOMPLETE' if not resource or resource.get('error') or resource.get('completion_errors') else recorded,
                recorded_evaluation_status=recorded, denominators=denominators(), counts=saved.get('counts'),
                receipt_statuses=dict(Counter(r['status'] for r in snapshot['rows'].values())),
                resources=resource, errors=saved.get('result', {}).get('errors', {}),
                read_errors=snapshot['read_errors'], blind=saved.get('blind_summary'), findings=diagnostic.get('counts'),
                post_status=diagnostic.get('status'), scientific_pass=None)


def require_blind(output, rows):
    terminal = load(Path(output)/'blind_terminal.json')
    expected = set(blind_keys())
    saved = terminal.get('rows', {})
    if (terminal.get('status') != 'ALL_BLIND_DIAGNOSTICS_TERMINAL' or set(saved) != expected
            or rows['blind_save']['status'] != 'RETURNED'
            or any(saved[k] != rows[k] or saved[k].get('status') not in TERMINAL for k in expected)):
        raise ValueError('all 26 blind terminal receipts and successful blind save required')


class Journal(prior.Journal):
    def __init__(self, output, check):
        self.output, self.rows, self.check = Path(output), plan(), check
        self.save()

    def save(self):
        dump(self.output/'state.json', self.rows)
        counts = {k: dict(planned=v, started=0, returned=0) for k, v in budgets().items()}
        for row in self.rows.values():
            for field in ('started', 'returned'):
                counts[row['kind']][field] += row[field]
        dump(self.output/'counts.json', counts)

    def terminalize(self, include_post=False):
        keys = self.rows if include_post else blind_keys()
        for key in keys:
            row = self.rows[key]
            if row['status'] == 'NOT_STARTED':
                row.update(status='DEPENDENCY_UNRESOLVED', error='fixed item not executed; no refill')
            elif row['status'] == 'STARTED':
                row.update(status='INTERRUPTED', error='unfinished fixed item')
        self.save()
