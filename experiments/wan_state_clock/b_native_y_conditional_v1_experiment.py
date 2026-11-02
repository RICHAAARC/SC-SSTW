"""One fixed33 saved-Y attempt: 26 blind, save, five historical loads, post."""
from pathlib import Path
import json
import time
from . import b_native_y_conditional_v1_records as rec


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value.tolist() if hasattr(value, 'tolist') else value


def run(config, output=None, *, backend_factory=None, evaluator=None):
    output = Path(output or config['output']).expanduser().resolve()
    if rec.output_state(output) == 'CONSUMED':
        return rec.recover(output)
    if rec.output_state(output) == 'CONFLICT':
        return dict(status='OUTPUT_CONFLICT', computations_this_invocation=0)
    preflight = rec.preflight(config, output)
    if preflight['status'] != 'PREFLIGHT_READY':
        return preflight
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED', plan=rec.plan(), budgets=rec.budgets(),
                           denominators=rec.denominators(), fixed_slots=rec.skeleton(), limits=rec.LIMITS,
                           rerun='METADATA_ONLY_NO_POST'), stream, indent=2)
    except FileExistsError:
        return rec.recover(output)
    start = time.monotonic()
    backend = journal = None
    errors, data, shared, evaluation, resources = {}, {}, {}, {}, {}
    global_stop = interrupted = False
    try:
        journal = rec.Journal(output, lambda: None)
        if backend_factory is None:
            from runtime.wan.b_native_y_conditional_v1 import Backend
            backend_factory = Backend
        backend = backend_factory()
        backend.started = start
        journal.check = backend.check
        source = Path(config['new_y_root']).expanduser().resolve()

        def attempt(key, fn):
            try:
                return journal.call(key, fn)
            except Exception as error:
                errors[key] = f'{type(error).__name__}: {error}'
                if type(error).__name__ in ('ResourceExceeded', 'RawLimitExceeded'):
                    raise
                return None

        def codebook():
            table = backend.codebook(source/'shared_codebook.npz')
            rec.dump(output/'codebook_reference.json', dict(path=str(source/'shared_codebook.npz'),
                     status='LOADED_VALIDATED', entries=len(table), rebuilt=False))
            shared['table'] = table
        attempt('saved_codebook_load', codebook)
        for view, n in zip(rec.VIEWS, rec.LENGTHS):
            unit = data.setdefault(view, {})
            folder = output/'units'/view
            original = source/'units'/view
            raw_holder = {}
            backend.begin_view()
            try:
                def raw_load():
                    raw = backend.raw(original/'readout.npz', n)
                    rec.dump(folder/'raw_reference.json', dict(path=str(original/'readout.npz'),
                             keys=['q','d0','d1','soft','mask'], frames=n, validation='ORIGINAL_VALUES_UNCHANGED',
                             input_replica_bit_slots=n*630, valid_mask_slots=int(raw['mask'].sum()),
                             zero_soft_valid_slots=int(((raw['soft'] == 0) & raw['mask']).sum())))
                    raw_holder['raw'] = raw
                attempt(view+'/saved_Y_raw_load', raw_load)

                def hard_load():
                    hard, graph = backend.hard(original/'path.json', n)
                    rec.dump(folder/'Y_H.json', hard)
                    unit.update(hard=hard, graph=graph)
                    # Optional profile metadata is record-only and never changes method input.
                    try:
                        profile = rec.load(original/'probe.json').get('profile')
                        rec.dump(folder/'profile.json', dict(profile=profile))
                        unit['profile'] = profile
                    except (OSError, ValueError, TypeError, AttributeError) as error:
                        unit['profile_note'] = str(error)
                attempt(view+'/saved_Y_H_load', hard_load)
                if 'raw' in raw_holder and 'graph' in unit and 'table' in shared:
                    def C():
                        value = backend.C(raw_holder['raw'], unit['graph'], shared['table'])
                        rec.dump(folder/'C.json', serial(value))
                        unit['cost'] = value
                    attempt(view+'/C', C)
                if 'cost' in unit:
                    def L():
                        local, candidates = backend.L(unit['graph'], unit['cost'])
                        rec.dump(folder/'L.json', serial(local))
                        rec.dump(folder/'candidates.json', candidates)
                        unit.update(local=local, candidates=candidates)
                    attempt(view+'/L', L)
                if 'candidates' in unit:
                    def submit():
                        value = backend.submit(unit['hard'], unit['candidates'])
                        rec.dump(folder/'submission.json', value)
                        unit['submission'] = value
                    attempt(view+'/conditional_submit', submit)
            except Exception as error:
                if type(error).__name__ != 'RawLimitExceeded':
                    raise
                errors[view+'/view_resource'] = str(error)
            finally:
                raw_holder.clear()
                backend.end_view()
    except BaseException as error:
        global_stop = True
        interrupted = not isinstance(error, Exception)
        errors['attempt'] = f'{type(error).__name__}: {error}'
    try:
        if journal is not None:
            journal.terminalize()
        if not global_stop:
            def blind_save():
                summary = dict(units={v: dict(profile=u.get('profile'), profile_note=u.get('profile_note'),
                    C_candidate_slots=len(u['cost']['C']) if 'cost' in u else None,
                    C_j_scores=u.get('cost', {}).get('comparisons'),
                    submission_status=u.get('submission', {}).get('status')) for v, u in data.items()},
                    errors=dict(errors), denominators=rec.denominators())
                rec.dump(output/'blind_summary.json', summary)
                rec.dump(output/'blind_terminal.json', dict(status='ALL_BLIND_DIAGNOSTICS_TERMINAL',
                    rows={k: journal.rows[k] for k in rec.blind_keys()}, errors=dict(errors), fixed_slots=rec.skeleton()))
            journal.call('blind_save', blind_save)
            rec.require_blind(output, journal.rows)
            for view in rec.VIEWS:
                def historical_load():
                    path = Path(config['historical_public_root']).expanduser().resolve()/view/'saved_public.json'
                    value = backend.public(path)
                    rec.dump(output/'historical'/view/'saved_public.json', value)
                    data[view]['historical'] = value
                attempt(view+'/historical_public_load', historical_load)

            def post():
                nonlocal evaluation
                selected = evaluator
                if selected is None:
                    from .b_native_y_conditional_v1_evaluation import evaluate as selected
                evaluation = selected(output, config, data, journal.rows, backend.check)
                return evaluation
            journal.call('post', post)
    except BaseException as error:
        errors['completion'] = f'{type(error).__name__}: {error}'
        interrupted |= not isinstance(error, Exception)
    try:
        if journal is not None:
            journal.terminalize(include_post=True)
    except BaseException as error:
        errors['terminal_save'] = f'{type(error).__name__}: {error}'
        interrupted |= not isinstance(error, Exception)
    rows = journal.rows if journal is not None else rec.plan()
    result = dict(status='INTERRUPTED' if interrupted else 'FIXED_DIAGNOSTIC_TERMINATED',
        evaluation_status='EVIDENCE_INCOMPLETE' if errors else evaluation.get('status', 'EVIDENCE_INCOMPLETE'),
        errors=errors, denominators=rec.denominators(), budgets=rec.budgets(), scientific_pass=None,
        computations_this_invocation=sum(r['started'] for r in rows.values()), post_calls_this_invocation=rows['post']['started'])
    try:
        rec.dump(output/'result.json', result)
        if backend is not None:
            backend.check()
    except Exception as error:
        errors['final_save_or_resource'] = f'{type(error).__name__}: {error}'
    finally:
        if backend is not None:
            try:
                resources = backend.close()
                if not isinstance(resources, dict):
                    raise ValueError('resource close object required')
            except Exception as error:
                resources = dict(error=f'{type(error).__name__}: {error}')
    elapsed = time.monotonic()-start
    if resources.get('error'):
        errors['resource_close'] = resources['error']
    if resources.get('peak_resident_bytes', 0) > rec.LIMITS['rss_bytes']:
        errors['resource_peak'] = 'closed peak exceeds 8 GiB'
    if elapsed > rec.LIMITS['total_wall_seconds']:
        errors['resource_total_wall'] = 'whole attempt 6300 seconds exceeded on close'
    if errors:
        result['evaluation_status'] = 'EVIDENCE_INCOMPLETE'
    resources.update(limits=rec.LIMITS, total_wall_seconds=elapsed, completion_errors=dict(errors))
    save_error = None
    try:
        rec.dump(output/'resources.json', resources)
    except Exception as error:
        save_error = error
        errors['resources_save'] = f'{type(error).__name__}: {error}'
        result.update(status='FINAL_RECORD_SAVE_FAILED', evaluation_status='EVIDENCE_INCOMPLETE')
    rec.dump(output/'result.json', result)
    if save_error is not None:
        raise OSError('resource receipt save failed; consumed with no refill') from save_error
    return result
