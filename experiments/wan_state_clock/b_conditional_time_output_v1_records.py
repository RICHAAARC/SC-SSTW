"""Fixed mutually exclusive entry plans; standard-library recovery before science."""
from pathlib import Path
from collections import Counter
from . import b_minimal_l_v1_records as common

load, dump, output_state = common.load, common.dump, common.output_state
VIEWS = ('FULL', 'CROP', 'DELETE1', 'REPEAT1', 'DELETE3')
LENGTHS = (181, 101, 180, 182, 178)
ENTRIES = ('semantic', 'raw', 'media')
TERMINAL = ('RETURNED', 'FAILED', 'INTERRUPTED', 'RESOURCE_UNRESOLVED', 'DEPENDENCY_UNRESOLVED')


def denominators():
    return dict(content_sources=1, correlated_views=5, received_frames=822,
                relations=817, time_bit_slots=517860, lane_slots=4110)


def plan(entry):
    if entry not in ENTRIES:
        raise ValueError('choose exactly one semantic/raw/media entry')
    rows = {}
    def add(key, kind, frames=None):
        rows[key] = dict(kind=kind, status='NOT_STARTED')
        if frames is not None:
            rows[key].update(frames=frames, relations=frames-1, bit_slots=frames*630, lane_slots=frames*5)
    if entry != 'semantic':
        add('codebook', 'codebook')
    if entry == 'media':
        for op in ('probe', 'decode'):
            add('source/'+op, 'codec_'+op)
    for view, frames in zip(VIEWS, LENGTHS):
        ops = ('semantic_submit',) if entry == 'semantic' else (
            ('raw_read',) if entry == 'raw' else ('encode', 'probe', 'decode', 'time_read')) + ('hard', 'C', 'L', 'submit')
        for op in ops:
            add(view+'/'+op, 'codec_'+op if op in ('encode', 'probe', 'decode') else op, frames)
    return rows


def budgets(entry):
    result = dict(Counter(row['kind'] for row in plan(entry).values()))
    for kind in ('codebook', 'codec_encode', 'codec_probe', 'codec_decode', 'raw_read', 'time_read',
                 'hard', 'C', 'L', 'submit', 'semantic_submit', 'G', 'A', 'B', 'model', 'GPU', 'writer', 'training'):
        result.setdefault(kind, 0)
    return result


def input_paths(config):
    entry = config.get('entry')
    if entry == 'media':
        return [config.get('media')]
    values = config.get('views')
    if not isinstance(values, dict) or set(values) != set(VIEWS):
        raise ValueError('all five fixed view inputs required')
    paths = []
    for view in VIEWS:
        value = values[view]
        if entry == 'semantic' and isinstance(value, dict):
            if set(value) != {'hard', 'L'}:
                raise ValueError('semantic pair requires hard and L file paths')
            paths.extend(value.values())
        else:
            paths.append(value)
    return paths


def preflight(config, output):
    if config.get('entry') not in ENTRIES or not output:
        return dict(status='UNCONFIGURED', physical_calls_this_invocation=0, denominators=denominators())
    try:
        paths = input_paths(config)
        if any(not isinstance(p, str) or not p.strip() for p in paths):
            return dict(status='UNCONFIGURED', physical_calls_this_invocation=0, denominators=denominators())
        output = Path(output).expanduser().resolve()
        inputs = [Path(p).expanduser().resolve() for p in paths]
        if config.get('truth'):
            inputs.append(Path(config['truth']).expanduser().resolve())
        for source in inputs:
            protected = source.parent
            if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
                raise ValueError('output overlaps read-only input directory')
        missing = [str(p) for p in inputs[:len(paths)] if not p.is_file()]
        if missing:
            raise ValueError('input files unavailable: '+repr(missing))
        return dict(status='PREFLIGHT_READY', entry=config['entry'], plan=plan(config['entry']),
                    budgets=budgets(config['entry']), denominators=denominators(), input_paths=[str(p) for p in inputs])
    except (ValueError, TypeError, OSError) as error:
        return dict(status='PREFLIGHT_UNRESOLVED', error=str(error), physical_calls_this_invocation=0,
                    denominators=denominators())


class Journal(common.Journal):
    def __init__(self, output, entry):
        self.output = Path(output)
        self.rows = plan(entry)
        self.counts = {k: dict(planned=v, started=0, returned=0) for k, v in budgets(entry).items()}
        self.save()

    def terminalize(self, reason):
        for row in self.rows.values():
            if row['status'] == 'NOT_STARTED':
                row.update(status='DEPENDENCY_UNRESOLVED', error=reason)
            elif row['status'] == 'STARTED':
                row.update(status='INTERRUPTED', error=reason)
        self.save()


def recover(output):
    saved, errors = {}, {}
    for name in ('attempt', 'state', 'counts', 'resources', 'blind_terminal', 'evaluation', 'result'):
        try:
            value = load(Path(output)/(name+'.json'))
            if not isinstance(value, dict):
                raise ValueError('saved object required')
            saved[name] = value
        except (OSError, ValueError) as error:
            errors[name] = str(error)
    entry = saved.get('attempt', {}).get('entry')
    rows = plan(entry) if entry in ENTRIES else {}
    for key, row in rows.items():
        actual = saved.get('state', {}).get(key)
        if not isinstance(actual, dict) or actual.get('kind') != row['kind'] or actual.get('status') not in TERMINAL:
            row.update(status='UNRESOLVED_RECEIPT', error='missing, interrupted or invalid saved operation receipt')
            errors[key] = row['error']
        else:
            row.update(status=actual['status'], **({'error': actual['error']} if 'error' in actual else {}))
    if not rows:
        errors['plan'] = 'saved unique entry unavailable'
    return dict(status='READ_ONLY_EXISTING_ATTEMPT', physical_calls_this_invocation=0,
                saved=saved, rows=rows, read_errors=errors, denominators=denominators())
