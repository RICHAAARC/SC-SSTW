"""One fixed future attempt; resumptions read receipts, never rerun physical work."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as p

ARMS = ('BASE','QIM')
LAYERS = ('preclip','clip','rgb8','h264')
BUDGETS = dict(D181=1, baseline_head=46, head_replay=46, K=4201, KT=4201,
               E41=16, codec_encode=2, codec_probe=2, codec_decode=2, DiT=0, training=0)


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def dump(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


def plan():
    rows = {f'{a}/{l}':dict(kind='time_read', slots=181*5*2*63, status='NOT_STARTED') for a in ARMS for l in LAYERS}
    rows.update({f'{a}/identity/{s}/{phase}':dict(kind='identity_read', slots=160, status='NOT_STARTED')
                 for a in ARMS for s in (0,40) for phase in range(4)})
    return rows


def denominators():
    return dict(sources=1, arms=2, frames=181, time_layers=4, time_read_slots=912240,
                identity_views=16, identity_local_bit_slots=2560, paired_identity_slots=1280)


def has_existing_output(output):
    output = Path(output)
    return output.exists() and any(output.iterdir())


def recover(output):
    output = Path(output); errors = {}; saved = {}
    for name in ('attempt','state','counts','solver','solver_progress','native','evaluation','result'):
        try:
            value = load(output/(name+'.json'))
            if not isinstance(value,dict): raise ValueError('saved record must be an object')
            saved[name] = value
        except (OSError, ValueError) as e: errors[name] = f'{type(e).__name__}: {e}'
    rows = plan()
    state = saved.get('state')
    legal = {'NOT_STARTED','STARTED','RETURNED','FAILED','INTERRUPTED'}
    for key, row in rows.items():
        value = state.get(key) if isinstance(state,dict) else None
        if not isinstance(value,dict) or not isinstance(value.get('status'),str) or value['status'] not in legal:
            row.update(status='UNRESOLVED_RECEIPT',cause='missing or invalid saved row')
        else:
            row.update(value)
            row.update(kind=plan()[key]['kind'],slots=plan()[key]['slots'])
    for row in rows.values():
        if row['status'] == 'STARTED': row['status'] = 'UNRESOLVED_INTERRUPTED'
    return dict(status='READ_ONLY_EXISTING_ATTEMPT', physical_calls_this_invocation=0,
                denominators=denominators(), rows=rows, saved=saved, read_errors=errors)


def recovery_summary(snapshot):
    """Summarize saved receipts only; no raw arrays, scoring, truth or scientific imports."""
    def count(value, maximum):
        return type(value) is int and 0 <= value <= maximum
    def time_row_valid(row):
        status = row.get('status')
        if not isinstance(status,str) or status not in ('RETURNED','NOT_STARTED','FAILED','INTERRUPTED'): return False
        if row.get('fixed_slots') != 114030: return False
        if status != 'RETURNED': return True
        fields = ('active_slots','observed_active_slots','bit_errors','erased_active_slots')
        if any(not count(row.get(k),114030) for k in fields): return False
        return (row['bit_errors'] <= row['observed_active_slots'] <= row['active_slots']
                and row['erased_active_slots']==row['active_slots']-row['observed_active_slots']
                and count(row.get('wrong_origin_candidate_count'),905))
    def identity_row_valid(row):
        required = ('start','phase','region','local_row','bit','value','status','common_complete_cell_eligible','group','correct')
        if any(k not in row for k in required): return False
        if (type(row['start']) is not int or row['start'] not in (0,40)
                or type(row['phase']) is not int or row['phase'] not in range(4)
                or row['region'] not in ('R0','R1')
                or type(row['local_row']) is not int or row['local_row'] not in range(1,11)
                or type(row['bit']) is not int or row['bit'] not in range(8)): return False
        status = row['status']
        if not isinstance(status,str) or status not in ('AVAILABLE','NOT_STARTED','NOT_ENTERED','FAILED','INTERRUPTED','UNRESOLVED','UNSUPPORTED'): return False
        available = status=='AVAILABLE'
        if (available and not count(row['value'],1)) or (not available and row['value'] is not None): return False
        full = row['common_complete_cell_eligible']
        if type(full) is not bool: return False
        if (full and not count(row['group'],3)) or (not full and row['group'] is not None): return False
        return type(row['correct']) is bool if full and available else row['correct'] is None
    def comparison_valid(row):
        if not isinstance(row,dict) or row.get('fixed_paired_slots')!=1280: return False
        if any(not count(row.get(k),1280) for k in ('common_complete_observed_slots','added_support','lost_support')): return False
        common = row['common_complete_observed_slots']
        for arm in ('base','qim'):
            errors, ber = row.get(arm+'_common_errors'), row.get(arm+'_common_ber')
            if not count(errors,common): return False
            if common==0:
                if ber is not None: return False
            elif type(ber) not in (int,float) or not 0 <= ber <= 1 or ber != errors/common: return False
        return True
    saved = snapshot.get('saved',{})
    result = saved.get('result',{}); evaluation = saved.get('evaluation',{})
    rows = snapshot.get('rows',{})
    counts = saved.get('counts',{})
    calls = {}
    for kind, planned in BUDGETS.items():
        row = counts.get(kind,{})
        started = row.get('started') if isinstance(row,dict) else None
        returned = row.get('returned') if isinstance(row,dict) else None
        valid = type(started) is int and type(returned) is int and 0 <= returned <= started <= planned
        calls[kind] = dict(planned=planned,started=started if valid else None,
            returned=returned if valid else None,status='RECORDED' if valid else 'UNRESOLVED_RECEIPT')
    state_counts = {}
    for row in rows.values():
        status = row.get('status','UNRESOLVED_RECEIPT')
        state_counts[status] = state_counts.get(status,0)+1
    layer_rows = evaluation.get('time',[])
    if not isinstance(layer_rows,list): layer_rows = []
    layers = []
    for arm in ARMS:
        for layer in LAYERS:
            key = f'{arm}/{layer}'
            matches = [x for x in layer_rows if isinstance(x,dict) and x.get('key')==key]
            candidate = matches[0] if len(matches)==1 else {}
            row = candidate if time_row_valid(candidate) else {}
            layers.append(dict(key=key,fixed_slots=181*5*2*63,read_status=rows.get(key,{}).get('status','UNRESOLVED_RECEIPT'),
                evaluation_status=row.get('status','POST_UNRESOLVED'),
                **{field:row.get(field) if row.get('status')=='RETURNED' else None for field in ('active_slots','observed_active_slots','bit_errors','erased_active_slots','wrong_origin_candidate_count')}))
    identity_rows = evaluation.get('identity',[])
    if not isinstance(identity_rows,list): identity_rows = []
    identity = []
    for arm in ARMS:
        selected = [x for x in identity_rows if isinstance(x,dict) and x.get('arm')==arm]
        complete = len(selected)==1280 and all(identity_row_valid(x) for x in selected)
        if complete:
            coordinates = {(x['start'],x['phase'],x['region'],x['local_row'],x['bit']) for x in selected}
            complete = len(coordinates)==1280
        known = [x for x in selected if x.get('status')=='AVAILABLE' and type(x.get('value')) is int and x['value'] in (0,1)]
        scored = sum(type(x.get('correct')) is bool for x in selected) if complete else None
        identity.append(dict(arm=arm,fixed_slots=1280,recorded_rows=len(selected),
            observed_bits=len(known) if complete else None,
            erased_or_unread_bits=1280-len(known) if complete else None,
            full_cell_scored_bits=scored,
            full_cell_errors=sum(x.get('correct') is False for x in selected) if scored else None,
            status='SAVED_ROWS' if complete else 'POST_UNRESOLVED'))
    comparison = evaluation.get('identity_comparison')
    comparison_ok = comparison_valid(comparison)
    unresolved = any(x in state_counts for x in ('UNRESOLVED_RECEIPT','UNRESOLVED_INTERRUPTED','STARTED'))
    final_status = result.get('status')
    status = ('UNRESOLVED_RECORDS' if unresolved or any(c['status']!='RECORDED' for c in calls.values())
              else final_status if final_status in ('COMPLETED_FIXED_RECORDS','STOPPED_FIXED_ATTEMPT') else 'UNRESOLVED_FINAL_RECORD')
    if status=='COMPLETED_FIXED_RECORDS' and any(row.get('status')!='RETURNED' for row in rows.values()):
        status='UNRESOLVED_RECORDS'
    if status=='COMPLETED_FIXED_RECORDS' and (result.get('evaluation_status')!='POST_RECORDED'
            or evaluation.get('status')=='POST_UNRESOLVED'
            or any(row['evaluation_status']=='POST_UNRESOLVED' for row in layers)
            or any(row['status']=='POST_UNRESOLVED' for row in identity)
            or not comparison_ok):
        status='UNRESOLVED_POST_RECORD'
    solver = saved.get('solver',saved.get('solver_progress',{}))
    metrics = solver.get('metrics',{})
    if not isinstance(metrics,dict): metrics = {}
    budget = metrics.get('budget',{})
    if not isinstance(budget,dict): budget = {}
    solver_summary = {k:solver.get(k) for k in ('status','reason','iterations','estimated_bytes')}
    solver_summary['metrics'] = {k:metrics.get(k) for k in ('split','target','a_stationarity','r_stationarity')}
    solver_summary['budget'] = {k:budget.get(k) for k in ('linf','outside','excess')}
    return dict(status=status,saved_result_status=final_status,stop=result.get('stop'),
        evaluation_status=result.get('evaluation_status','POST_UNRESOLVED'),denominators=denominators(),
        state_counts=state_counts,calls=calls,
        solver=solver_summary,
        native_status=saved.get('native',{}).get('status','UNRESOLVED'),time_layers=layers,identity=identity,
        identity_comparison=comparison if comparison_ok else None,read_errors=snapshot.get('read_errors',{}),
        physical_calls_this_invocation=0,claim_ceiling='saved-record display only; no media/identity/synchronization/quality scientific PASS')


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
