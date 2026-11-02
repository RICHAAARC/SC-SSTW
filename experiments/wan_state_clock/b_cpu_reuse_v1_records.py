"""Consumed-attempt receipts shared by the adopted 93/134 CPU adapters."""
from pathlib import Path
from collections import Counter
import json
import time
from .b_minimal_l_v1_records import load, dump, output_state
from .b_time_layer_diagnostic_v1_records import Journal as PriorJournal, valid_receipt

LIMITS = dict(rss_bytes=8*1024**3, total_wall_seconds=6300, per_view_seconds=600, codec_seconds=300)
TERMINAL = ('RETURNED', 'FAILED', 'INTERRUPTED', 'RESOURCE_UNRESOLVED', 'DEPENDENCY_UNRESOLVED', 'N/A_NOT_REFERENCED')


def make_plan(pairs):
    return {k: dict(kind=v, planned=1, started=0, returned=0, status='NOT_STARTED') for k, v in pairs}


def blind_keys(rows):
    return [k for k, v in rows.items() if v['kind'] not in ('blind_save', 'historical_load', 'post')]


def recover(output, spec):
    saved, errors, rows = {}, {}, spec.plan()
    for name in ('attempt', 'state', 'counts', 'resources', 'fixed_slots', 'blind_summary', 'post_status', 'result'):
        try:
            value = load(Path(output)/(name+'.json'))
            if not isinstance(value, dict): raise ValueError('metadata object required')
            saved[name] = value
        except (OSError, ValueError) as error: errors[name] = str(error)
    for key, row in rows.items():
        value = saved.get('state', {}).get(key)
        na = (isinstance(value, dict) and value.get('kind') == row['kind'] == 'cache_load'
              and value.get('status') == 'N/A_NOT_REFERENCED' and value.get('started') == value.get('returned') == 0)
        if na or valid_receipt(value, row):
            row.update(value)
            if row['status'] == 'STARTED': row.update(status='UNFINISHED_RECEIPT', error='consumed; no resume')
        else: row.update(status='UNRESOLVED_RECEIPT', started=None, returned=None)
    return dict(status='READ_ONLY_EXISTING_DIAGNOSTIC', computations_this_invocation=0, post_calls_this_invocation=0,
                saved=saved, rows=rows, read_errors=errors, denominators=spec.denominators())


def require_blind(output, rows):
    value = load(Path(output)/'blind_terminal.json')
    keys = blind_keys(rows)
    if (rows['blind_save']['status'] != 'RETURNED' or value.get('status') != 'ALL_BLIND_SLOTS_TERMINAL'
            or set(value.get('rows', {})) != set(keys)
            or any(value['rows'][k] != rows[k] or rows[k]['status'] not in TERMINAL for k in keys)):
        raise ValueError('complete saved blind receipts required before historical/truth access')


def post_status(output, report):
    """Compact receipt only: no bit values, refs, plans or truth."""
    dump(Path(output)/'post_status.json', {k: report.get(k) for k in
        ('status','outcome','counterexample_stop','evidence_complete','counts','B_correct_supported')})


class Journal(PriorJournal):
    def __init__(self, output, rows):
        self.output, self.rows, self.check = Path(output), rows, lambda: None
        self.save()

    def save(self):
        dump(self.output/'state.json', self.rows)
        counts = {}
        for row in self.rows.values():
            c = counts.setdefault(row['kind'], dict(planned=0, started=0, returned=0, not_referenced=0))
            for field in ('planned', 'started', 'returned'): c[field] += row[field]
            c['not_referenced'] += row['status'] == 'N/A_NOT_REFERENCED'
        dump(self.output/'counts.json', counts)

    def terminalize(self, include_post=False):
        for key in self.rows if include_post else blind_keys(self.rows):
            row = self.rows[key]
            if row['status'] == 'NOT_STARTED': row.update(status='DEPENDENCY_UNRESOLVED', error='fixed item not executed; no refill')
            elif row['status'] == 'STARTED': row.update(status='INTERRUPTED', error='unfinished fixed item')
        self.save()


class Attempt:
    def __init__(self, output, spec):
        self.output, self.spec = output, spec
        self.started = time.monotonic()
        self.journal = Journal(output, spec.plan())
        self.backend = None
        self.errors, self.evaluation = {}, {}
        self.interrupted = False

    def attach(self, backend):
        self.backend = backend
        backend.started = self.started
        self.journal.check = backend.check

    def call(self, key, fn):
        try: return self.journal.call(key, fn)
        except Exception as error:
            self.errors[key] = f'{type(error).__name__}: {error}'
            if type(error).__name__ in ('ResourceExceeded', 'RawLimitExceeded'): raise
            return None

    def failed(self, error):
        self.interrupted |= not isinstance(error, Exception)
        self.errors['attempt'] = f'{type(error).__name__}: {error}'

    def seal(self, summary, **payload):
        self.journal.terminalize()
        def save():
            dump(self.output/'blind_summary.json', summary)
            dump(self.output/'blind_terminal.json', dict(status='ALL_BLIND_SLOTS_TERMINAL',
                 rows={k: self.journal.rows[k] for k in blind_keys(self.journal.rows)},
                 errors=dict(self.errors), denominators=self.spec.denominators(), **payload))
        self.call('blind_save', save)
        require_blind(self.output, self.journal.rows)

    def finish(self):
        resources = {}
        try:
            for key, fn in (
                ('terminal_state_save',lambda:self.journal.terminalize(include_post=True)),
                ('fixed_slots_save',lambda:dump(self.output/'fixed_slots.json',self.spec.slot_receipts(self.journal.rows)))):
                try: fn()
                except BaseException as error:
                    self.errors[key]=f'{type(error).__name__}: {error}'
                    self.interrupted |= not isinstance(error,Exception)
            if not self.evaluation and (self.output/'post_status.json').is_file():
                try: self.evaluation=load(self.output/'post_status.json')
                except (OSError,ValueError) as error:self.errors['partial_post_receipt']=str(error)
            if self.backend is not None:
                try:self.backend.check()
                except BaseException as error:
                    self.errors['completion_resource']=str(error)
                    self.interrupted |= not isinstance(error,Exception)
        finally:
            if self.backend is not None:
                try:
                    resources=self.backend.close()
                    if not isinstance(resources,dict):raise ValueError('resource close object required')
                except BaseException as error:
                    resources={};self.errors['resource_close']=f'{type(error).__name__}: {error}'
                    self.interrupted |= not isinstance(error,Exception)
        elapsed = time.monotonic()-self.started
        if resources.get('error'): self.errors['resource_error'] = resources['error']
        if resources.get('peak_resident_bytes', 0) > LIMITS['rss_bytes']: self.errors['resource_peak'] = 'RSS limit exceeded'
        if elapsed > LIMITS['total_wall_seconds']: self.errors['resource_total'] = '6300s total exceeded'
        resources.update(limits=LIMITS, total_wall_seconds=elapsed, completion_errors=dict(self.errors))
        rows = self.journal.rows
        result = dict(status='INTERRUPTED' if self.interrupted else 'FIXED_DIAGNOSTIC_TERMINATED',
            evaluation_status='EVIDENCE_INCOMPLETE' if self.errors else self.evaluation.get('status', 'EVIDENCE_INCOMPLETE'),
            counterexample_stop=bool(self.evaluation.get('counterexample_stop')), errors=self.errors,
            denominators=self.spec.denominators(), scientific_pass=None,
            computations_this_invocation=sum(v['started'] for v in rows.values()), post_calls_this_invocation=rows['post']['started'])
        try: dump(self.output/'resources.json', resources)
        except Exception as error:
            self.errors['resources_save'] = str(error)
            result.update(status='FINAL_RECORD_SAVE_FAILED', evaluation_status='EVIDENCE_INCOMPLETE')
        dump(self.output/'result.json', result)
        return result


def open_attempt(config, output, spec, input_paths):
    output = Path(output or config['output']).expanduser().resolve()
    if output_state(output) == 'CONSUMED': return recover(output, spec)
    if output_state(output) == 'CONFLICT': return dict(status='OUTPUT_CONFLICT', computations_this_invocation=0)
    for path in input_paths:
        source = Path(path).expanduser().resolve()
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            return dict(status='PREFLIGHT_UNRESOLVED', error='output overlaps protected input', computations_this_invocation=0)
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED', plan=spec.plan(), denominators=spec.denominators(),
                           limits=LIMITS, rerun='METADATA_ONLY_NO_POST'), stream, indent=2)
    except FileExistsError: return recover(output, spec)
    return Attempt(output, spec)


def cli(spec, run, default_config, argv=None):
    import argparse
    import signal
    parser = argparse.ArgumentParser(description='One consumed CPU attempt; repeated output is metadata-only.')
    parser.add_argument('--config', default=str(default_config)); parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.output and output_state(args.output) == 'CONSUMED': result = recover(args.output, spec)
    else:
        config = load(args.config)
        def stop(signum, frame): raise KeyboardInterrupt('fixed CPU attempt terminated')
        signal.signal(signal.SIGTERM, stop)
        result = run(config, args.output)
    if 'saved' in result:
        display = {k: v for k, v in result.items() if k not in ('saved', 'rows')}
        display['receipt_statuses'] = dict(Counter(v['status'] for v in result['rows'].values()))
    else: display = result
    print(json.dumps(display, ensure_ascii=False, allow_nan=False))
    return result
