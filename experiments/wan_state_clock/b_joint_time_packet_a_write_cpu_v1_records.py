"""Stage P receipts only. No scientific imports or execution on consumed reentry."""
from pathlib import Path
from datetime import datetime, timezone
import json
import math
import time

SOURCES = ('n01', 'e02')
SOURCE_IDS = dict(n01='b-rank3-n01', e02='b2-m3-e02')
MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
OPERATIONS = ('q0_load', 'target', 'solver', 'prediction', 'handoff')


def plain(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return 'NONFINITE:'+repr(value)
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    return value


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(plain(value), ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temporary.replace(path)


def load(path): return json.loads(Path(path).read_text())


def plan(kind='solve'):
    keys = ['head_prepare'] if kind == 'head' else ['head_load']+[s+'/'+op for s in SOURCES for op in OPERATIONS]
    return {key: dict(planned=1, started=0, returned=0, status='NOT_STARTED') for key in keys}


def budgets():
    return dict(q0_load=2, target=2, solver=2, K=8402, KT=8400,
                per_source=dict(solver_K=4200, solver_KT=4200, prediction_K=1),
                model_forward=0, D181=0, E41=0, DiT=0, codec=0, training=0)


def output_files():
    common = ['attempt.json', 'operation_plan.json', 'file_plan.json', 'operations.json',
              'operator_events.jsonl', 'result.json', 'resources.json', 'cpu_handoff.json', 'head.pt']
    source = ['input.json', 'q0.npy', 'schedule.npz', 'fixed_targets.npy', 'solver_progress.json',
              'solver.json', 'coefficients.pt', 'prediction.pt', 'linear.json', 'handoff.json']
    return dict(common=common, per_source={s: [s+'/'+f for f in source] for s in SOURCES},
                missing_files='retained as missing; no later rerun or regeneration')


def operation_catalog():
    return dict(operations=list(plan()), bounds=budgets(),
        operator_ids={s: dict(solver_K=[f'{s}/solver/K/{i:04}' for i in range(1,4201)],
                             solver_KT=[f'{s}/solver/KT/{i:04}' for i in range(1,4201)],
                             prediction_K=[s+'/prediction/K/0001']) for s in SOURCES},
        unused_operator_slots='NOT_STARTED; numerical early stop does not spend remaining cap',
        physical_slots_per_source=181*5*2*63, active_target_slots_per_source=(181*5-10)*2*63,
        source_frames=181, order='replica-major: tile=5*replica+lane',
        solver_limits=dict(iterations=4000, check_every=20, seconds=7200, estimated_live_bytes=16*1024**3))


def recover(output):
    output = Path(output); saved, errors = {}, {}
    for name in ('attempt', 'operations', 'resources', 'result', 'cpu_handoff'):
        try: saved[name] = load(output/(name+'.json'))
        except (OSError, ValueError) as error: errors[name] = str(error)
    return dict(status='READ_ONLY_CONSUMED_STAGE', saved=saved, read_errors=errors,
                physical_calls_this_invocation=0, scientific_reads_this_invocation=0)


def reserve(output, kind, inputs):
    output = Path(output)
    if (output/'attempt.json').exists(): return False
    if output.exists() and any(output.iterdir()): raise ValueError('new stage output is not empty')
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='CONSUMED_ON_START', stage=kind, inputs=inputs,
                started_utc=datetime.now(timezone.utc).isoformat(), retry=False,
                rerun='METADATA_ONLY', budgets=budgets() if kind=='solve' else dict(head_prepare=1, forward=0)), stream, indent=2)
    except FileExistsError: return False
    return True


class BudgetExceeded(RuntimeError): pass


class Journal:
    def __init__(self, output, kind='solve'):
        self.output = Path(output); self.rows = plan(kind); self.counts = {}; self.current = None
        self.started = time.monotonic(); self.cpu = time.process_time(); self.budget_error = None
        self.persist()

    def persist(self):
        dump(self.output/'operations.json', dict(rows=self.rows, operator_counts=self.counts, budget_error=self.budget_error))

    def call(self, key, function):
        row = self.rows[key]
        if row['started']: raise RuntimeError('operation already consumed: '+key)
        row.update(started=1, status='STARTED'); self.current = key; self.persist()
        try:
            value = function(); row.update(returned=1, status='RETURNED'); return value
        except BaseException as error:
            row.update(status='INTERRUPTED' if not isinstance(error, Exception) else 'FAILED',
                       error=type(error).__name__+': '+str(error))
            raise
        finally:
            self.current = None; self.persist()

    def event(self, source, kind, done):
        if kind not in ('K', 'KT') or self.current not in (source+'/solver', source+'/prediction'):
            raise ValueError('unplanned operator call')
        phase = self.current.rsplit('/',1)[1]
        cap = 4200 if phase == 'solver' else 1 if kind == 'K' else 0
        key = self.current+'/'+kind
        row = self.counts.setdefault(key, dict(planned=cap, started=0, returned=0))
        if not done:
            if row['started'] >= cap:
                self.budget_error = key+' exhausted'; self.persist(); raise BudgetExceeded(self.budget_error)
            row['started'] += 1
        elif row['returned'] >= row['started']:
            raise ValueError('operator return without start')
        else: row['returned'] += 1
        occurrence = row['returned'] if done else row['started']
        with (self.output/'operator_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(id=key+f'/{occurrence:04}', state='RETURNED' if done else 'STARTED'))+'\n')

    def terminalize(self):
        for row in self.rows.values():
            if row['status'] == 'STARTED': row['status'] = 'INTERRUPTED'
            if row['status'] == 'NOT_STARTED': row['reason'] = 'consumed stage ended; no later execution'
        self.persist()
        import resource
        dump(self.output/'resources.json', dict(wall_seconds=time.monotonic()-self.started,
            cpu_seconds=time.process_time()-self.cpu, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            solver_enforcement='unchanged 7200s cooperative deadline and 16GiB estimated live arrays',
            forward=0, GPU=0, codec=0, DiT=0, training=0))

