"""One consumed attempt, five fixed views, one selected input level. No fallback."""
from pathlib import Path
import json
import time
from . import b_conditional_time_output_v1_records as records


def attack_mappings():
    """Attack/post-evaluation side only, never passed to a receiver or submitter."""
    return dict(zip(records.VIEWS, [list(range(181)), list(range(38, 139)),
        [n for n in range(181) if n != 44], list(range(45))+[44]+list(range(45, 181)),
        [n for n in range(181) if n not in (43, 44, 45)]]))


def _load_semantic(value, frames):
    if isinstance(value, dict):
        bundle = dict(hard=records.load(value['hard']), L=records.load(value['L']))
    else:
        bundle = records.load(value)
    if not isinstance(bundle, dict) or not isinstance(bundle.get('hard'), dict) or not isinstance(bundle.get('L'), dict):
        raise ValueError('full hard/L semantic object required')
    if len(bundle['hard'].get('nodes', [])) != frames or len(bundle['L'].get('nodes', [])) != frames:
        raise ValueError('semantic received length differs from fixed view')
    return bundle


def _load_raw(path, frames):
    import numpy as np
    with np.load(path, allow_pickle=False) as saved:
        raw = {key: saved[key] for key in ('q', 'd0', 'd1', 'soft', 'mask')}
    for key, value in raw.items():
        if value.shape != (frames, 5, 2, 63):
            raise ValueError('raw '+key+' geometry mismatch')
        if key == 'mask':
            if value.dtype != np.bool_:
                raise ValueError('boolean observation mask required')
        else:
            if value.dtype.kind != 'f':
                raise ValueError('floating raw '+key+' required')
            # The original hard fusion operates in its input dtype; normalize before it.
            with np.errstate(over='ignore', invalid='ignore'):
                raw[key] = value.astype(np.float64, copy=False)
            if not np.isfinite(raw[key]).all():
                raise ValueError('finite FP64 raw '+key+' required')
    return raw


def run(config, output=None, backend_factory=None, evaluator=None):
    output = output or config.get('output')
    if output:
        output = Path(output).expanduser().resolve()
        state = records.output_state(output)
        if state == 'CONSUMED':
            return records.recover(output)
        if state == 'CONFLICT':
            return dict(status='OUTPUT_CONFLICT', physical_calls_this_invocation=0, denominators=records.denominators())
    preflight = records.preflight(config, output)
    if preflight['status'] != 'PREFLIGHT_READY':
        return preflight
    entry = config['entry']
    output.mkdir(parents=True, exist_ok=True)
    attempt = dict(status='RESERVED_CONSUMED', entry=entry, budgets=records.budgets(entry),
                   plan=records.plan(entry), denominators=records.denominators(), input_paths=preflight['input_paths'],
                   source_note=config.get('source_note'), resource_limits=dict(rss_bytes=8*1024**3,
                       codebook_seconds=300, view_receiver_seconds=600, total_receiver_seconds=3300, codec_call_seconds=300))
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(attempt, stream, ensure_ascii=False, allow_nan=False, indent=2)
    except FileExistsError:
        return records.recover(output)
    journal = records.Journal(output, entry)
    errors, backend = {}, None
    wall_start = time.monotonic()
    interrupted = False
    try:
        if backend_factory is None:
            from runtime.wan.b_conditional_time_output_v1 import Backend
            backend_factory = Backend
        backend = backend_factory()
        table = source_rgb = None
        if entry != 'semantic':
            try:
                table = journal.call('codebook', backend.codebook)
            except Exception as error:
                errors['codebook'] = f'{type(error).__name__}: {error}'
        if entry == 'media':
            try:
                def source_probe():
                    records.dump(output/'source_probe.json', backend.probe(config['media'], 181))
                journal.call('source/probe', source_probe)
                source_rgb = journal.call('source/decode', lambda: backend.decode(config['media'], 181))
            except Exception as error:
                errors['source'] = f'{type(error).__name__}: {error}'
        for view, frames in zip(records.VIEWS, records.LENGTHS):
            folder = output/'received'/view
            raw = hard = cost = local = pixels = None
            try:
                folder.mkdir(parents=True, exist_ok=True)
                if entry == 'semantic':
                    def semantic_submit():
                        bundle = _load_semantic(config['views'][view], frames)
                        records.dump(folder/'semantic.json', bundle)
                        records.dump(folder/'submission.json', backend.semantic(bundle['hard'], bundle['L']))
                    with backend.receiving_view():
                        journal.call(view+'/semantic_submit', semantic_submit)
                    continue
                import numpy as np
                if entry == 'media':
                    if source_rgb is None:
                        continue
                    mapping = attack_mappings()[view]
                    journal.call(view+'/encode', lambda: backend.encode(source_rgb[mapping], folder/'received.mp4'))
                    def view_probe():
                        records.dump(folder/'probe.json', backend.probe(folder/'received.mp4', frames))
                    journal.call(view+'/probe', view_probe)
                    pixels = journal.call(view+'/decode', lambda: backend.decode(folder/'received.mp4', frames))
                def read_and_save():
                    nonlocal raw
                    if entry == 'raw':
                        raw = _load_raw(config['views'][view], frames)
                    else:
                        raw = backend.read(pixels)
                    np.savez(folder/'raw.npz', **raw)
                    # Save and close observations before semantic fusion, using the same format validator.
                    raw = _load_raw(folder/'raw.npz', frames)
                def hard_and_save():
                    nonlocal hard
                    hard, merged, known = backend.receive(raw, table)
                    np.savez(folder/'fused.npz', soft=merged, known=known)
                    records.dump(folder/'blind.json', hard)
                def cost_and_save():
                    nonlocal cost
                    cost = backend.C(raw, hard, table)
                    np.savez(folder/'C.npz', **{k: cost[k] for k in ('C', 'bounds', 'offsets', 'support')})
                    records.dump(folder/'C.json', {k: cost[k] for k in ('states', 'R', 'comparisons', 'graph_abnormal')})
                def local_and_save():
                    nonlocal local
                    rank = backend.L(cost)
                    np.savez(folder/'L.npz', dL=rank['dL'], best=rank['best'])
                    records.dump(folder/'L.json', dict(status=rank['status'], minimum=rank['minimum']))
                    local = backend.candidates(hard, cost, rank)
                    records.dump(folder/'candidates.json', local)
                def submit_and_save():
                    records.dump(folder/'semantic.json', dict(hard=hard, L=local))
                    records.dump(folder/'submission.json', backend.semantic(hard, local))
                with backend.receiving_view():
                    journal.call(view+('/raw_read' if entry == 'raw' else '/time_read'), read_and_save)
                    pixels = None
                    if table is not None:
                        journal.call(view+'/hard', hard_and_save)
                        journal.call(view+'/C', cost_and_save)
                        journal.call(view+'/L', local_and_save)
                        journal.call(view+'/submit', submit_and_save)
            except Exception as error:
                errors[view] = f'{type(error).__name__}: {error}'
                if type(error).__name__ == 'ResourceExceeded':
                    for key in reversed(journal.rows):
                        if key.startswith(view+'/') and journal.rows[key]['status'] != 'NOT_STARTED':
                            journal.rows[key].update(status='RESOURCE_UNRESOLVED', error=errors[view])
                            journal.save()
                            break
            finally:
                raw = hard = cost = local = pixels = None
        source_rgb = None
    except BaseException as error:
        interrupted = not isinstance(error, Exception)
        errors['attempt'] = f'{type(error).__name__}: {error}'
    finally:
        resources = dict(status='NOT_STARTED', error='backend unavailable')
        if backend is not None:
            try:
                resources = backend.close()
                failure = resources.get('error')
                if resources.get('peak_resident_bytes', 0) > 8*1024**3 or resources.get('receiver_seconds', 0) > 3300:
                    failure = failure or 'fixed RSS/receiver resource limit exceeded'
                resources.update(status='RESOURCE_UNRESOLVED' if failure else 'CLOSED', error=failure)
                if failure:
                    errors['resources'] = str(failure)
            except Exception as error:
                errors['resources'] = f'{type(error).__name__}: {error}'
                resources = dict(status='CLOSE_FAILED', error=errors['resources'])
        records.dump(output/'resources.json', resources)
    journal.terminalize('fixed attempt ended; failed/unstarted calls are consumed, no retry')
    records.dump(output/'blind_terminal.json', dict(status='ALL_BLIND_ROWS_TERMINAL', rows=journal.rows))
    # Only now construct/open post-evaluation truth. No truth is passed to method/runtime.
    truth_path = config.get('truth')
    if entry == 'media' and not truth_path:
        truth_path = output/'private/attack_truth.json'
        records.dump(truth_path, dict(views=attack_mappings()))
    evaluation_status = 'EVIDENCE_INCOMPLETE'
    try:
        if evaluator is None:
            from .b_conditional_time_output_v1_evaluation import evaluate
            evaluator = evaluate
        evaluation = evaluator(output, journal.rows, truth_path)
        records.dump(output/'evaluation.json', evaluation)
        evaluation_status = evaluation.get('status', 'EVIDENCE_INCOMPLETE')
    except Exception as error:
        errors['evaluation'] = f'{type(error).__name__}: {error}'
    result = dict(status='INTERRUPTED' if interrupted else 'FIXED_ATTEMPT_TERMINATED', entry=entry,
                  errors=errors, denominators=records.denominators(), full_wall_seconds=time.monotonic()-wall_start,
                  failed_rows=[k for k, row in journal.rows.items() if row['status'] != 'RETURNED'],
                  evaluation_status=evaluation_status,
                  claim_ceiling='Conditional outputs only; five correlated edits of one source, no identity/null/general reliability claim')
    records.dump(output/'result.json', result)
    return result
