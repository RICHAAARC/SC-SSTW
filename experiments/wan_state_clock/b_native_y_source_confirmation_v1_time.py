"""Two source media prep14 and ten native-Y blind views82. No post or history."""
from pathlib import Path
from collections import Counter
import json
import time
from . import b_native_y_source_confirmation_v1_records as rec


def run(config, output, *, backend_factory=None):
    output = Path(output)
    if rec.output_state(output) == 'CONSUMED':
        saved, errors = rec.read_metadata(output)
        return dict(status='READ_ONLY_EXISTING_TIME', saved=saved, errors=errors, computations_this_invocation=0)
    if rec.output_state(output) == 'CONFLICT':
        return dict(status='OUTPUT_CONFLICT', computations_this_invocation=0)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'attempt.json').open('x') as stream:
        json.dump(dict(status='RESERVED_CONSUMED', plan=rec.time_plan(), budgets=rec.time_budgets(),
                       denominators=rec.denominators()['time'], rerun='METADATA_ONLY'), stream, indent=2)
    journal = rec.ItemJournal(output, rec.time_plan())
    start = time.monotonic()
    backend = None
    errors, data = {}, {}
    interrupted = False
    table = None
    blind_saved = False

    def call(key, function):
        try:
            return journal.call(key, function)
        except Exception as error:
            errors[key] = f'{type(error).__name__}: {error}'
            if type(error).__name__ in ('ResourceExceeded', 'RawLimitExceeded'):
                raise
            return None

    try:
        import numpy as np
        if backend_factory is None:
            from runtime.wan.b_native_y_source_confirmation_v1 import Backend
            backend_factory = Backend
        backend = backend_factory()
        backend.started = start
        journal.check = backend.check
        # Private fixture construction owns these mappings; reader receives pixels only.
        mappings = rec.mappings()
        rec.dump(output/'private/attack_truth.json', dict(views=mappings, source_id=rec.SOURCE_ID, reader_input=False))
        for arm in ('QIM', 'BASE'):
            path = Path(config['media'][arm])
            pixels = None
            try:
                probe = call(f'source/{arm}/probe', lambda: backend.source_probe(path))
                if probe is not None:
                    rec.dump(output/'source'/arm/'probe.json', probe)
                    pixels = call(f'source/{arm}/decode', lambda: backend.source_decode(path))
                if pixels is not None:
                    for unit in (u for u in rec.units() if u['arm'] == arm):
                        folder = output/unit['folder']
                        folder.mkdir(parents=True, exist_ok=True)
                        call(unit['id']+'/encode', lambda unit=unit, folder=folder:
                             backend.encode_view(pixels[mappings[unit['view']]], folder/'received.mp4'))
            except Exception as error:
                if type(error).__name__ != 'RawLimitExceeded':
                    raise
                errors['source/'+arm+'/resource'] = str(error)
            finally:
                pixels = None
        table = call('codebook', lambda: backend.make_codebook(output/'shared_codebook.npz'))
        for unit in rec.units():
            uid, n = unit['id'], unit['frames']
            folder = output/unit['folder']
            folder.mkdir(parents=True, exist_ok=True)
            saved, temporary = data.setdefault(uid, {}), {}
            try:
                backend.begin_view()
                if journal.rows[uid+'/encode']['status'] != 'RETURNED':
                    continue
                media = folder/'received.mp4'

                def persist(key, filename, fn):
                    value = fn()
                    rec.dump(folder/filename, value)
                    saved[key] = value
                    return value
                call(uid+'/probe', lambda: persist('probe', 'probe.json', lambda: backend.probe(media, n)))
                if 'probe' in saved:
                    def decode():
                        y = backend.decode(media, n, saved['probe'])
                        np.save(folder/'native_Y.npy', y, allow_pickle=False)
                        temporary['y'] = y
                    call(uid+'/native_Y_decode', decode)
                if 'y' in temporary:
                    def read():
                        raw = backend.read(temporary.pop('y'))
                        np.savez(folder/'readout.npz', **raw)
                        temporary['raw'] = raw
                    call(uid+'/Y_reader', read)
                if 'raw' in temporary and table is not None:
                    call(uid+'/F_compatibility', lambda: persist('compatible', 'compatibility_F.json',
                         lambda: backend.compatible(temporary['raw'], table[0])))
                if 'compatible' in saved:
                    def H():
                        hard, graph = backend.H(saved['compatible'], n)
                        rec.dump(folder/'Y_H.json', hard)
                        saved['hard'], temporary['graph'] = hard, graph
                    call(uid+'/H', H)
                if 'graph' in temporary and 'raw' in temporary and table is not None:
                    def C():
                        cost = backend.C(temporary['raw'], temporary['graph'], table[1])
                        np.savez(folder/'C.npz', **{k: cost[k] for k in ('C', 'offsets', 'bounds', 'support')})
                        summary = dict(states=cost['states'], state_counts=dict(Counter(cost['states'])), R=n,
                                       comparisons=cost['comparisons'], candidate_slots=len(cost['C']),
                                       defined_rows=sum(s == 'DEFINED' for s in cost['states']), graph_abnormal=cost['graph_abnormal'])
                        rec.dump(folder/'C.json', summary)
                        temporary['cost'], saved['C_summary'] = cost, summary
                    call(uid+'/C', C)
                if 'cost' in temporary:
                    def L():
                        local, candidates = backend.L(temporary['graph'], temporary['cost'])
                        np.savez(folder/'L.npz', dL=local['dL'], best=local['best'])
                        rec.dump(folder/'L.json', dict(minimum=local['minimum'], status=local['status']))
                        rec.dump(folder/'candidates.json', candidates)
                        saved['candidates'] = candidates
                    call(uid+'/L', L)
                if 'candidates' in saved:
                    call(uid+'/submit', lambda: persist('submission', 'submission.json',
                         lambda: backend.submit(saved['hard'], saved['candidates'])))
            except Exception as error:
                if type(error).__name__ != 'RawLimitExceeded':
                    raise
                errors[uid+'/resource'] = str(error)
            finally:
                temporary.clear()
                backend.end_view()
        journal.terminalize()

        def seal():
            rec.dump(output/'fixed_slots.json', rec.time_fixed_slots(journal.rows))
            rec.dump(output/'blind_summary.json', dict(denominators=rec.denominators()['time'], errors=dict(errors),
                     units={u: dict(C=v.get('C_summary'), profile=v.get('probe', {}).get('profile'),
                                    qualification=v.get('hard', {}).get('qualification'),
                                    submission_status=v.get('submission', {}).get('status')) for u, v in data.items()}))
            rec.dump(output/'blind_terminal.json', dict(status='ALL_BLIND_ROWS_TERMINAL',
                     rows={k: v for k, v in journal.rows.items() if k != 'blind_save'},
                     denominators=rec.denominators()['time'], errors=dict(errors)))
        call('blind_save', seal)
        rec.require_time_blind(output, journal.rows)
        blind_saved = True
    except BaseException as error:
        interrupted = not isinstance(error, Exception)
        errors['attempt'] = f'{type(error).__name__}: {error}'
    finally:
        resources = {}
        try:
            for key, fn in (('terminal_state_save', lambda: journal.terminalize(include_post=True)),
                            ('fixed_slots_save', lambda: rec.dump(output/'fixed_slots.json', rec.time_fixed_slots(journal.rows)))):
                try:
                    fn()
                except BaseException as error:
                    errors[key] = f'{type(error).__name__}: {error}'
                    interrupted |= not isinstance(error, Exception)
            if backend is not None:
                try:
                    backend.end_view()
                    backend.check()
                except BaseException as error:
                    errors['completion_resource'] = str(error)
        finally:
            if backend is not None:
                try:
                    resources = backend.close()
                    if not isinstance(resources, dict):
                        raise ValueError('resource close must return an object')
                except BaseException as error:
                    resources = {}
                    errors['resource_close'] = str(error)
                    interrupted |= not isinstance(error, Exception)
        elapsed = time.monotonic()-start
        peak = resources.get('peak_resident_bytes', 0)
        if type(peak) not in (int, float) or not 0 <= peak <= 8*1024**3:
            errors['resource_final'] = 'invalid or exceeded 8GiB RSS receipt'
        if resources.get('error'):
            errors['resource_close_report'] = resources['error']
        if elapsed > 6300:
            errors['resource_total'] = 'time stage 6300 seconds exceeded'
        resources.update(total_wall_seconds=elapsed, completion_errors=dict(errors))
        try:
            rec.dump(output/'resources.json', resources)
        except BaseException as error:
            errors['resources_save'] = str(error)
    result = dict(status='INTERRUPTED' if interrupted else 'FIXED_TIME_ATTEMPT_TERMINATED', errors=errors,
                  blind_terminal_saved=blind_saved, budgets=rec.time_budgets(), denominators=rec.denominators()['time'],
                  scientific_pass=None, post_calls_this_invocation=0,
                  computations_this_invocation=sum(v['started'] for v in journal.rows.values()))
    try:
        rec.dump(output/'result.json', result)
    except Exception as error:
        errors['result_save'] = str(error)
        result['status'] = 'FINAL_RECORD_SAVE_FAILED'
    return result
