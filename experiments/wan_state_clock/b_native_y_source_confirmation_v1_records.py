"""Fixed n01 package metadata: A8457+B57+time96+payload256; three posts."""
from pathlib import Path
from collections import Counter
from .b_minimal_l_v1_records import load, dump, output_state
from . import b_minimal_l_materialization_v1_records as material
from . import b_time_payload_cdc32_v1_records as payload
from .b_cpu_reuse_v1_records import Journal as ItemJournal, make_plan
from .b_new_source_chain_confirmation_v1_records import Journal as PhaseJournal

SOURCE_ID = 'b-rank3-n01'
TERMINAL = 'private/b-rank3-n01/P0_terminal.pt'
PHASES = ('A', 'B', 'time', 'payload', 'time_post', 'payload_post', 'summary')
FOLDERS = dict(A='time_chain/materialization/A', B='time_chain/materialization/B',
               time='time_chain/receiver', payload='payload')
VIEWS = ('FULL', 'CROP', 'DELETE1', 'REPEAT1', 'DELETE3')
LENGTHS = (181, 101, 180, 182, 178)
IDS = tuple(f'v{i:02}' for i in range(10))
TIME_OPS = ('probe', 'native_Y_decode', 'Y_reader', 'F_compatibility', 'H', 'C', 'L', 'submit')


def units():
    return [dict(id=IDS[i], arm='QIM' if i < 5 else 'BASE', view=VIEWS[i % 5], frames=LENGTHS[i % 5],
                 folder=('received/' if i < 5 else 'base_received/')+VIEWS[i % 5]) for i in range(10)]


def mappings():
    return dict(FULL=list(range(181)), CROP=list(range(38, 139)),
                DELETE1=[n for n in range(181) if n != 44],
                REPEAT1=list(range(45))+list(range(44, 181)),
                DELETE3=[n for n in range(181) if n not in (43, 44, 45)])


def time_plan():
    return make_plan([(f'source/{arm}/{op}', 'codec_'+op) for arm in ('QIM', 'BASE') for op in ('probe', 'decode')]
                     + [(u+'/encode', 'codec_encode') for u in IDS] + [('codebook', 'codebook')]
                     + [(u+'/'+op, 'codec_probe' if op == 'probe' else 'codec_decode' if op == 'native_Y_decode' else op)
                        for u in IDS for op in TIME_OPS] + [('blind_save', 'blind_save')])


def time_budgets():
    return dict(Counter(r['kind'] for r in time_plan().values()))


def budgets():
    return dict(materialization_A=dict(material.A_BUDGETS), materialization_B=dict(material.B_BUDGETS),
                time=time_budgets(), payload=payload.budgets(),
                stage_arithmetic_upper_bounds=dict(A=8457, B=57, time=96, payload=256),
                heterogeneous_arithmetic_upper_bound=8866, arithmetic_is_model_inference_count=False,
                codec=dict(encode=12, probe=19, decode=19, total=50),
                physical_model_scale=dict(model_load=3, D181=1, baseline_head_in_D181=46, head_replay=46, E41=60, DiT=0),
                post=dict(time=1, payload=1, summary=1), wrapper_phases_add_physical_calls=0)


def denominators():
    return dict(source_id=SOURCE_ID, independent_sources=1,
                time=dict(views=10, nodes=1644, edges=1634, lanes=8220, replica_bit_slots=1035720,
                          per_arm_nodes=822, per_arm_edges=817), payload=payload.denominators(),
                materialization=material.denominators(dict(source_id=SOURCE_ID, frames=181), 44))


def plan():
    return {k: dict(planned=1, started=0, returned=0, status='NOT_STARTED') for k in PHASES}


def preflight(config, output):
    try:
        if config.get('source_id') != SOURCE_ID or config.get('terminal_relative') != TERMINAL:
            raise ValueError('fixed n01 original P0 required; no fallback')
        source, output = Path(config['source']).expanduser().resolve(), Path(output).expanduser().resolve()
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise ValueError('source and output overlap')
        if not isinstance(config.get('model_id'), str) or not config['model_id'].strip():
            raise ValueError('usable model location required')
        return dict(status='PREFLIGHT_READY', input_paths=[str(source/TERMINAL)], protected_roots=[str(source)],
                    source_id=SOURCE_ID, budgets=budgets(), denominators=denominators(), physical_attempt_reserved=False)
    except (KeyError, ValueError, TypeError) as error:
        return dict(status='PREFLIGHT_UNRESOLVED', error=str(error), physical_calls_this_invocation=0)


def read_metadata(output):
    saved, errors = {}, {}
    for name in ('attempt', 'state', 'counts', 'resources', 'fixed_slots', 'blind_summary', 'post_status', 'result', 'disk_cache_capacity'):
        path = Path(output)/(name+'.json')
        if name == 'disk_cache_capacity' and not path.exists():
            continue
        try:
            value = load(path)
            if not isinstance(value, dict):
                raise ValueError('metadata object required')
            saved[name] = value
        except (OSError, ValueError) as error:
            errors[name] = str(error)
    return saved, errors


def recover(output):
    output = Path(output)
    saved, errors = read_metadata(output)
    stages = {}
    for stage, folder in FOLDERS.items():
        stages[stage], stage_errors = read_metadata(output/folder)
        errors.update({stage+'/'+k: v for k, v in stage_errors.items()})
    return dict(status='READ_ONLY_EXISTING_PACKAGE', physical_calls_this_invocation=0, post_calls_this_invocation=0,
                saved=saved, stages=stages, read_errors=errors, denominators=denominators(), budgets=budgets())


class Journal(PhaseJournal):
    def __init__(self, output):
        self.output, self.rows = Path(output), plan()
        self.save()


def time_fixed_slots(rows):
    return {u['id']: dict(nodes=u['frames'], edges=u['frames']-1, arm=u['arm'], view=u['view'],
                         operations={op: rows[u['id']+'/'+op]['status'] for op in TIME_OPS}) for u in units()}


def require_time_blind(output, rows):
    value = load(Path(output)/'blind_terminal.json')
    keys = set(rows)-{'blind_save'}
    if (rows['blind_save']['status'] != 'RETURNED' or value.get('status') != 'ALL_BLIND_ROWS_TERMINAL'
            or set(value.get('rows', {})) != keys or any(value['rows'][k] != rows[k] for k in keys)
            or any(rows[k]['status'] in ('NOT_STARTED', 'STARTED') for k in keys)):
        raise ValueError('complete time blind terminal required')


def require_whole_blind(output):
    value = load(Path(output)/'blind_terminal.json')
    if value.get('status') != 'ALL_CHAIN_BLIND_SLOTS_TERMINAL':
        raise ValueError('whole blind terminal required before post')
    if set(value.get('phases', {})) != {'A', 'B', 'time', 'payload'}:
        raise ValueError('complete whole blind phase rows required')
    if any(r.get('status') in ('NOT_STARTED', 'STARTED') for r in value['phases'].values()):
        raise ValueError('unfinished physical phase at post barrier')
    return value
