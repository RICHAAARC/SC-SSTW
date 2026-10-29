"""Saved-candidate fixed ledger, structural preflight and read-only recovery (stdlib)."""
from pathlib import Path
import ast
import json
import struct
import time
import zipfile
from .b_framewise_qim_v1_records import load, dump

STARTS = (0, *range(1, 181, 4))
LAYERS = ('preclip', 'clip', 'rgb8', 'h264')
BUDGETS = dict(model_load=1, K=1, BASE_constraint_P=1, Aa_constraint_P=1,
               head_replay=46, time_reads=4, E41=8, codec_encode=1, codec_probe=1,
               codec_decode=1, D181=0, baseline_head=0, solver=0, KT=0, DiT=0,
               training=0, BASE_new_physical=0)


def denominators():
    return dict(sources=1, new_arms=1, frames=181, time_layers=4,
                time_read_slots=456120, time_slots_per_layer=114030,
                active_time_slots_per_layer=112770, identity_views=8,
                identity_local_bit_slots=1280, paired_identity_slots=1280,
                complete_cell_eligible_slots=320)


def plan():
    rows = {name:dict(kind=name, slots=0, status='NOT_STARTED')
            for name in ('linear_constraints', 'native', 'media')}
    rows.update({f'QIM/{layer}':dict(kind='time_read', slots=114030, status='NOT_STARTED') for layer in LAYERS})
    rows.update({f'QIM/identity/{start}/{phase}':dict(kind='identity_read', slots=160, status='NOT_STARTED')
                 for start in (0, 40) for phase in range(4)})
    return rows


def output_state(output):
    output = Path(output)
    if (output/'attempt.json').exists(): return 'CONSUMED'
    if output.exists() and (not output.is_dir() or any(output.iterdir())): return 'CONFLICT'
    return 'NEW'


def resolve_input_output(saved, output, truth_source=None):
    saved, output = Path(saved).resolve(), Path(output).resolve()
    protected = [saved]+([Path(truth_source).resolve()] if truth_source is not None else [])
    if any(output == source or output.is_relative_to(source) for source in protected):
        raise ValueError('new output must be outside read-only inputs')
    return saved, output


def validate_paths(saved, output, truth_source=None):
    saved, output = resolve_input_output(saved, output, truth_source)
    protected = [saved]+([Path(truth_source).resolve()] if truth_source is not None else [])
    writable = [output]+[output.with_name(output.name+suffix).resolve() for suffix in ('-preflight-v1','-logs','-entry-cache')]
    for path in writable:
        if any(path == source or path.is_relative_to(source) for source in protected):
            raise ValueError('write destination must be outside read-only inputs: '+str(path))
    return saved, output


def validate_config(cfg):
    expected = dict(source_id='b2-m3-e01', terminal_relative='private/b2-m3-e01/P0_terminal.pt',
                    frames=181, global_start=0, identity_starts=[0, 40], identity_phases=[0, 1, 2, 3])
    if not isinstance(cfg, dict): raise ValueError('saved protocol must be an object')
    for key, value in expected.items():
        if type(cfg.get(key)) is not type(value) or cfg[key] != value:
            raise ValueError('fixed saved method field: '+key)
    if not isinstance(cfg.get('model_id'), str) or not cfg['model_id']: raise ValueError('usable model id/path required')
    return cfg


def npy_header(path):
    with Path(path).open('rb') as stream:
        if stream.read(6) != b'\x93NUMPY': raise ValueError('invalid npy header')
        version = tuple(stream.read(2))
        if version not in ((1, 0), (2, 0), (3, 0)): raise ValueError('unsupported npy header')
        size_bytes = 2 if version == (1, 0) else 4
        size = int.from_bytes(stream.read(size_bytes), 'little')
        if not 0 < size <= 65536: raise ValueError('invalid npy header length')
        header = ast.literal_eval(stream.read(size).decode('utf-8' if version == (3, 0) else 'latin1'))
        if not isinstance(header, dict): raise ValueError('invalid npy metadata')
        return header


def tensor_container(path):
    # Inspect only the format container, never unpickle a tensor before reservation.
    with zipfile.ZipFile(path) as archive:
        metadata = [name for name in archive.namelist() if name.endswith('/data.pkl')]
        if len(metadata) != 1: raise ValueError('saved torch tensor metadata unavailable')
        with archive.open(metadata[0]) as stream:
            if not stream.read(1): raise ValueError('empty tensor metadata')
    return dict(format='torch_zip', tensor_shape_dtype_finite='UNVERIFIED_UNTIL_RESERVED_READ')


def preflight(saved, output, truth_source=None):
    saved, output = validate_paths(saved, output, truth_source)
    destination = output.with_name(output.name+'-preflight-v1')
    errors, inspected, cfg = {}, {}, None
    paths = ['protocol.json', 'coefficients.pt', 'fixed_targets.npy', 'BASE_preclip.npy']
    paths += [f'features/{start:03}.pt' for start in STARTS]
    for relative in paths:
        path = saved/relative
        try:
            if relative == 'protocol.json': cfg = validate_config(load(path)); inspected[relative] = 'fixed method fields checked'
            elif path.suffix == '.pt': inspected[relative] = tensor_container(path)
            else:
                header = npy_header(path)
                shape, dtype = ((181, 10, 63), '<f8') if relative == 'fixed_targets.npy' else ((181, 320, 512, 3), '<f4')
                if header.get('shape') != shape or header.get('descr') != dtype or header.get('fortran_order') is not False:
                    raise ValueError('fixed array shape/dtype/layout header mismatch')
                inspected[relative] = dict(shape=list(shape), dtype=dtype, finite='UNVERIFIED_UNTIL_RESERVED_READ')
        except (OSError, ValueError, SyntaxError, zipfile.BadZipFile, EOFError) as error:
            errors[relative] = f'{type(error).__name__}: {error}'
    state = output_state(output)
    if state != 'NEW': errors['new_output'] = 'output is '+state+'; no overwrite'
    result = dict(status='PREFLIGHT_READY' if not errors else 'PREFLIGHT_UNRESOLVED',
                  errors=errors, inspected=inspected, feature_starts=list(STARTS),
                  config=cfg, denominators=denominators(), rows=plan(),
                  counts={key:dict(planned=n, started=0, returned=0) for key, n in BUDGETS.items()},
                  physical_attempt_reserved=False, old_attempt_read_only=True)
    dump(destination/'preflight.json', result)
    return result


def recover(output):
    saved, errors = {}, {}
    for name in ('attempt', 'state', 'counts', 'linear', 'native', 'evaluation', 'result'):
        try:
            value = load(Path(output)/(name+'.json'))
            if not isinstance(value, dict): raise ValueError('saved object required')
            saved[name] = value
        except (OSError, ValueError) as error: errors[name] = f'{type(error).__name__}: {error}'
    rows = plan()
    for key, row in rows.items():
        value = saved.get('state', {}).get(key)
        if not isinstance(value, dict) or not isinstance(value.get('status'), str) or value['status'] not in ('NOT_STARTED','STARTED','RETURNED','FAILED','INTERRUPTED'):
            row.update(status='UNRESOLVED_RECEIPT')
        else:
            kind, slots = row['kind'], row['slots']; row.update(value); row.update(kind=kind, slots=slots)
            if row['status'] == 'STARTED': row['status'] = 'UNRESOLVED_INTERRUPTED'
    return dict(status='READ_ONLY_EXISTING_ATTEMPT', saved=saved, rows=rows,
                denominators=denominators(), read_errors=errors, physical_calls_this_invocation=0)


class Journal:
    def __init__(self, output):
        self.output = Path(output); self.rows = plan()
        self.counts = {key:dict(planned=n, started=0, returned=0) for key, n in BUDGETS.items()}
        dump(self.output/'state.json', self.rows); dump(self.output/'counts.json', self.counts)

    def event(self, kind, done):
        row = self.counts[kind]
        if not done and row['started'] >= row['planned']: raise RuntimeError('fixed budget: '+kind)
        if done and row['returned'] >= row['started']: raise RuntimeError('return without start: '+kind)
        row['returned' if done else 'started'] += 1
        dump(self.output/'counts.json', self.counts)

    def call(self, kind, fn):
        self.event(kind, False); value = fn(); self.event(kind, True); return value

    def operation(self, key, fn):
        row = self.rows[key]; row['status'] = 'STARTED'; dump(self.output/'state.json', self.rows)
        try:
            value = fn(); row['status'] = 'RETURNED'; return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error, Exception) else 'INTERRUPTED', error=f'{type(error).__name__}: {error}')
            raise
        finally: dump(self.output/'state.json', self.rows)
