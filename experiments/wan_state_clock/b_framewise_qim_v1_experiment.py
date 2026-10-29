"""Fixed e01 experiment, journal, raw persistence and post-evaluation. No report dependency."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as p
from experiments.wan_state_clock.b_framewise_qim_v1_records import (
    ARMS, LAYERS, BUDGETS, load, dump, plan, denominators, has_existing_output, recover)


class Journal:
    def __init__(self, output):
        self.output = Path(output); self.rows = plan()
        self.counts = {k:dict(planned=n, started=0, returned=0) for k,n in BUDGETS.items()}
        dump(self.output/'state.json', self.rows); dump(self.output/'counts.json', self.counts)
    def event(self, kind, done):
        row = self.counts[kind]
        if not done and row['started'] >= row['planned']: raise RuntimeError('fixed call budget: '+kind)
        if done and row['returned'] >= row['started']: raise RuntimeError('return without start: '+kind)
        row['returned' if done else 'started'] += 1
        dump(self.output/'counts.json', self.counts)
    def operation(self, key, fn):
        row = self.rows[key]; row['status'] = 'STARTED'; dump(self.output/'state.json', self.rows)
        try:
            value = fn(); row['status'] = 'RETURNED'; return value
        except BaseException as e:
            row.update(status='FAILED' if isinstance(e, Exception) else 'INTERRUPTED', error=f'{type(e).__name__}: {e}')
            raise
        finally: dump(self.output/'state.json', self.rows)


def save_read(output, key, raw):
    """Persist and reread raw observed tensors before fusion/any truth assessment."""
    path = Path(output)/'raw'/(key+'.npz'); path.parent.mkdir(parents=True, exist_ok=True)
    expected = (181,5,2,63)
    if set(raw) != {'q','d0','d1','soft','mask'} or any(x.shape != expected for x in raw.values()):
        raise ValueError('fixed raw q/d0/d1/soft/mask shape')
    if raw['mask'].dtype != np.bool_: raise ValueError('boolean observation mask required')
    if any(not np.isfinite(raw[k][raw['mask']]).all() for k in ('q','d0','d1','soft')):
        raise ValueError('observed nonfinite raw read')
    np.savez(path, **raw)
    with np.load(path, allow_pickle=False) as saved:
        merged = p.merge_replicas(torch.from_numpy(saved['soft']), torch.from_numpy(saved['mask']))
    dump(Path(output)/'reads'/(key+'.json'), dict(candidates=merged['candidates'], presence='UNCALIBRATED',
        frame_axis='received_frame', truth_inputs=False, fusion='same_received_frame_lane_bit_sum'))


def process_arm(backend, journal, preclip, arm):
    for layer in LAYERS[:3]:
        key = f'{arm}/{layer}'
        journal.operation(key, lambda key=key,layer=layer: save_read(journal.output,key,backend.read_layer(preclip,layer)))
    media = backend.media(preclip, arm)
    key = f'{arm}/h264'
    journal.operation(key, lambda: save_read(journal.output,key,backend.read_layer(media,'h264')))
    for start in (0,40):
        for phase in range(4):
            key = f'{arm}/identity/{start}/{phase}'
            journal.operation(key, lambda key=key,start=start,phase=phase: dump(journal.output/'identity'/(key+'.json'),
                backend.identity(media,start,phase,key)))


def post_evaluate(output, source, rows):
    """Known-position post diagnostics only. Raw reads are already durable."""
    dump(Path(output)/'post_truth_access.json',dict(unix=time.time(), meaning='after fixed raw/identity rows terminalized'))
    truth = load(Path(source)/'protocol.json')['config']['writer']['bits']
    if len(truth) != 32 or any(type(x) is not int or x not in (0,1) for x in truth): raise ValueError('32 saved identity truth bits required')
    js,bits = p.schedule(); truth_code = bits.reshape(181,2,5,63).transpose(1,2).numpy()
    active = (js >= 0).reshape(181,2,5).transpose(1,2).numpy()[...,None]
    time_rows = []
    for arm in ARMS:
        for layer in LAYERS:
            key = f'{arm}/{layer}'; row = dict(key=key, status=rows[key]['status'], fixed_slots=114030)
            if rows[key]['status'] == 'RETURNED':
                with np.load(Path(output)/'raw'/(key+'.npz'),allow_pickle=False) as raw:
                    known = raw['mask'] & (raw['soft'] != 0) & active
                    row.update(active_slots=int(np.broadcast_to(active,raw['mask'].shape).sum()),
                        observed_active_slots=int(known.sum()), bit_errors=int(((raw['soft']>0) != truth_code)[known].sum()),
                        erased_active_slots=int((np.broadcast_to(active,known.shape)&~known).sum()))
                candidates = load(Path(output)/'reads'/(key+'.json'))['candidates']
                wrong = [c for c in candidates if c['j'] is not None and c['j'] != int(js[c['received_frame'],c['lane']])]
                row.update(candidate_rows=905, wrong_origin_candidate_count=len(wrong),
                    temporal_subsets={label:[c for c in candidates if c['received_frame'] in frames]
                        for label,frames in [('inside_chunk_1_to_4',range(1,5)),('chunk_seam_4_to_5',(4,5))]},
                    source_authentication=False, presence_calibrated=False)
            time_rows.append(row)
    identity = []; paired = {}
    for arm in ARMS:
        for start in (0,40):
            for phase in range(4):
                key = f'{arm}/identity/{start}/{phase}'; saved = None
                if rows[key]['status'] == 'RETURNED': saved = load(Path(output)/'identity'/(key+'.json'))
                for region in ('R0','R1'):
                    for local_row in range(1,11):
                        first = start+phase+1+4*(local_row-1)
                        full = first % 4 == 1
                        group = min(3, (first//4)//11) if full else None
                        for bit in range(8):
                            entry = saved['regions'][region]['local_cells'][local_row-1]['bits'][bit] if saved else None
                            value = entry['value'] if entry and entry['status']=='AVAILABLE' else None
                            ident = (start,phase,region,local_row,bit)
                            row = dict(arm=arm,start=start,phase=phase,region=region,local_row=local_row,bit=bit,
                                value=value,status=entry['status'] if entry else rows[key]['status'],
                                common_complete_cell_eligible=full,group=group,
                                correct=value==truth[group*8+bit] if full and value is not None else None)
                            identity.append(row); paired.setdefault(ident,{})[arm] = row
    common = [v for v in paired.values() if v['BASE']['common_complete_cell_eligible'] and all(v[a]['value'] is not None for a in ARMS)]
    comparison = dict(fixed_paired_slots=1280, common_complete_observed_slots=len(common),
        base_common_errors=sum(v['BASE']['correct'] is False for v in common), qim_common_errors=sum(v['QIM']['correct'] is False for v in common),
        base_common_ber=sum(v['BASE']['correct'] is False for v in common)/len(common) if common else None,
        qim_common_ber=sum(v['QIM']['correct'] is False for v in common)/len(common) if common else None,
        added_support=sum(v['BASE']['value'] is None and v['QIM']['value'] is not None for v in paired.values()),
        lost_support=sum(v['BASE']['value'] is not None and v['QIM']['value'] is None for v in paired.values()))
    return dict(time=time_rows,identity=identity,identity_comparison=comparison,
        claim_ceiling='known-position physical diagnostics only; no blind synchronization, presence calibration, identity authentication or scientific PASS')


def run(cfg, source, output, backend_factory=None):
    output, source = Path(output), Path(source)
    if has_existing_output(output): return recover(output)
    if (cfg['source_id'] != 'b2-m3-e01' or cfg['terminal_relative'] != 'private/b2-m3-e01/P0_terminal.pt'
            or cfg['frames'] != 181 or cfg['global_start'] != 0 or cfg['identity_starts'] != [0,40] or cfg['identity_phases'] != [0,1,2,3]):
        raise ValueError('fixed e01/181/global0 package semantics')
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation prevents a concurrent invocation from starting another attempt.
    try:
        with (output/'attempt.json').open('x') as f: json.dump(dict(status='STARTED',unix=time.time(),retry=False),f)
    except FileExistsError: return recover(output)
    journal = Journal(output); dump(output/'protocol.json',cfg)
    backend = None; stop = None
    try:
        terminal = source/cfg['terminal_relative']
        missing = [str(x) for x in (terminal,source/'protocol.json') if not x.is_file()]
        if missing: raise FileNotFoundError('fixed input missing: '+', '.join(missing))
        if backend_factory is None:
            from runtime.wan.b_framewise_qim_v1_backend import Backend
            backend_factory = Backend
        backend = backend_factory(cfg,output,journal.event)
        baseline = backend.capture(terminal)
        process_arm(backend,journal,baseline,'BASE')
        answer = backend.solve(baseline,lambda r:dump(output/'solver_progress.json',r)); dump(output/'solver.json',answer)
        if answer['status'] != 'SOLVED_APPROXIMATE_KKT': raise ArithmeticError('fixed solver result: '+answer['status'])
        changed,native = backend.apply(baseline); dump(output/'native.json',native)
        # Even a native tolerance failure retains the materialized reads; no fallback.
        process_arm(backend,journal,changed,'QIM')
        if native['status'] != 'NATIVE_RESIDUAL_WITHIN_TOLERANCE': stop = dict(kind='NUMERICALLY_UNRESOLVED',error='native residual/target tolerance')
    except BaseException as e:
        stop = dict(kind='INPUT_MISSING' if isinstance(e,FileNotFoundError) else 'RESOURCE_UNRESOLVED' if isinstance(e,MemoryError)
                    else 'INTERRUPTED' if not isinstance(e,Exception) else 'UNRESOLVED', error=f'{type(e).__name__}: {e}')
    finally:
        if backend is not None:
            try: backend.close()
            except Exception as e:
                if stop is None: stop = dict(kind='UNRESOLVED_RELEASE',error=str(e))
    for row in journal.rows.values():
        if row['status']=='NOT_STARTED': row.update(status='NOT_STARTED',cause=stop)
    dump(output/'state.json',journal.rows)
    # Do not interpret missing observations as errors=0. Truth is opened only after all reads end.
    evaluation_status = 'POST_RECORDED'
    try: dump(output/'evaluation.json',post_evaluate(output,source,journal.rows))
    except Exception as e:
        evaluation_status = 'POST_UNRESOLVED'
        error = f'{type(e).__name__}: {e}'
        dump(output/'evaluation.json',dict(status=evaluation_status,error=error,denominators=denominators()))
        if stop is None: stop = dict(kind=evaluation_status,error=error)
    result = dict(status='COMPLETED_FIXED_RECORDS' if stop is None else 'STOPPED_FIXED_ATTEMPT',stop=stop,
                  evaluation_status=evaluation_status,denominators=denominators(),counts=journal.counts,scientific_pass=None,
                  physical_execution_authorized_by_this_code=False)
    dump(output/'result.json',result)
    return result
