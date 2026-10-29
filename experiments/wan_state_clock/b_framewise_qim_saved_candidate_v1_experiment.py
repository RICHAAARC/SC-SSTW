"""Independent fixed saved-candidate experiment. Imports science only after reservation."""
from pathlib import Path
import json
import time
from .b_framewise_qim_saved_candidate_v1_records import (
    resolve_input_output, output_state, preflight, recover, Journal, dump, denominators, LAYERS)


def run(saved, truth_source, output, model_id=None, backend_factory=None):
    saved, output = resolve_input_output(saved, output, truth_source)
    if output_state(output) == 'CONSUMED': return recover(output)
    check = preflight(saved, output, truth_source)
    if check['status'] != 'PREFLIGHT_READY': return check
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='STARTED', unix=time.time(), retry=False, kind='saved_candidate_native_v1'), stream)
    except FileExistsError: return recover(output)
    journal = Journal(output)
    cfg = dict(check['config'])
    if model_id is not None: cfg['model_id'] = str(model_id)
    dump(output/'protocol.json', cfg)
    backend, stop, native = None, None, None
    try:
        if backend_factory is None:
            from runtime.wan.b_framewise_qim_saved_candidate_v1_backend import Backend
            backend_factory = Backend
        backend = backend_factory(cfg, output, journal.event)
        backend.restore_saved(saved)
        linear = journal.operation('linear_constraints', backend.check_candidate)
        dump(output/'linear.json', linear)
        if linear.get('status') != 'DIRECT_CONSTRAINTS_WITHIN_TOLERANCE':
            raise ArithmeticError('same saved candidate failed direct constraints; no head/re-solve/projection')
        preclip, native = journal.operation('native', backend.apply_saved)
        dump(output/'native.json', native)
        if native.get('finite_complete') is not True or native.get('frames') != 181:
            raise ValueError('native nonfinite/incomplete; dependent reads stopped')
        if native.get('status') != 'NATIVE_RESIDUAL_WITHIN_TOLERANCE':
            stop = dict(kind='NATIVE_NUMERICALLY_UNRESOLVED', error='native finite target/budget failure retained; fixed reads continue')
        # Existing raw-before-fusion implementation, imported only inside a reserved run.
        from .b_framewise_qim_v1_experiment import save_read
        def read(path, layer):
            raw = journal.call('time_reads', lambda: backend.read_layer(path, layer))
            save_read(output, 'QIM/'+layer, raw)
        for layer in LAYERS[:3]:
            journal.operation('QIM/'+layer, lambda layer=layer: read(preclip, layer))
        media = journal.operation('media', lambda: backend.media(preclip, 'QIM'))
        journal.operation('QIM/h264', lambda: read(media, 'h264'))
        for start in (0, 40):
            for phase in range(4):
                key = f'QIM/identity/{start}/{phase}'
                journal.operation(key, lambda key=key, start=start, phase=phase:
                    dump(output/'identity'/(key+'.json'), backend.identity(media, start, phase, key)))
    except BaseException as error:
        stop = dict(kind='INTERRUPTED' if not isinstance(error, Exception) else 'UNRESOLVED',
                    error=f'{type(error).__name__}: {error}', previous_stop=stop)
    finally:
        if backend is not None:
            try: backend.close()
            except Exception as error:
                stop = dict(kind='RELEASE_UNRESOLVED', error=str(error), previous_stop=stop)
    for row in journal.rows.values():
        if row['status'] == 'NOT_STARTED': row['cause'] = stop
    dump(output/'state.json', journal.rows)
    try:
        from .b_framewise_qim_saved_candidate_v1_evaluation import post_evaluate
        evaluation = post_evaluate(output, saved, truth_source, journal.rows)
    except Exception as error:
        evaluation = dict(status='POST_UNRESOLVED', error=f'{type(error).__name__}: {error}', denominators=denominators())
    dump(output/'evaluation.json', evaluation)
    result = dict(status='COMPLETED_FIXED_RECORDS' if stop is None else 'STOPPED_FIXED_ATTEMPT', stop=stop,
                  evaluation_status=evaluation['status'], native_status=native.get('status') if native else 'UNMEASURED',
                  denominators=denominators(), counts=journal.counts, optimality='UNPROVEN',
                  old_attempt_read_only=True, scientific_pass=None)
    dump(output/'result.json', result)
    return result
