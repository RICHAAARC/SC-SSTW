"""Fixed saved-RGB receiver task. One attempt; all blind outputs precede truth."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from main.tube_state import blind_temporal_segmented_v1 as core
from runtime.wan import local_joint_positive_projection_margin_v1 as shared
from runtime.wan import blind_temporal_segmented_evaluation_v1 as evaluation

dump = shared.dump
ARMS = ('P', 'S')
VIEWS = ('CROP_MID', 'DELETE43_PHASE')
MODES = ('EST', 'UNALIGNED', 'ORACLE')
LENGTHS = (89, 138)


def slots():
    return [(f'{arm}/{view}/{key}', arm, view, n, key)
            for arm in ARMS for view, n in zip(VIEWS, LENGTHS) for key in core.KEYS]


def fixed_denominator():
    return dict(receiver_rows=8, paired_windows=852, roi_scalars=74976, pooled_scalars=18744,
        received_payload_units=884, signed_votes=848640, zero_masks=848640, payload_tables=24,
        planned_bits=768, structurally_present_bits=576, structurally_missing_bits=192,
        per_mode=dict(planned_bits=256, structurally_present_bits=192, structurally_missing_bits=64,
                      source_states=176, source_cells=352, received_frames=908))


def execution_artifacts(output):
    output = Path(output)
    names = ('attempt.json', 'result.json', 'checkpoint.json', 'ledger.json', 'protocol.json',
             'plans_completion.json', 'blind_completion.json', 'evaluation.json', 'evaluation_truth.json')
    found = [n+s for n in names for s in ('', '.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ('blind', 'features', 'observations', 'post_blind', 'prepared'):
        root = output/folder
        if root.is_dir():
            found.extend(str(p.relative_to(output)) for p in root.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def placeholder_evaluations(status='NOT_ENTERED'):
    """Stdlib-only fixed display skeleton, also embedded in the notebook."""
    out = {}
    for arm in ('P', 'S'):
        for view, n in (('CROP_MID', 89), ('DELETE43_PHASE', 138)):
            for key in ('CORRECT', 'WRONG'):
                for mode in ('EST', 'UNALIGNED', 'ORACLE'):
                    out[f'{arm}/{view}/{key}/{mode}'] = dict(status=status,
                        frames=[dict(received=i, status=status, exact=None, absolute_error=None) for i in range(n)],
                        source_cells=[dict(source_t=t, status=status) for t in range(1, 45)],
                        source_states=[dict(state=k, status=status) for k in range(22)],
                        bits=[dict(bit=i, status=status, decoded=None, correct=None, structurally_present=None,
                                   false_coverage=None) for i in range(32)],
                        denominators=dict(frames=n, cells=44, states=22, bits=32, structurally_present=24, structurally_missing=8))
    return out


def read_saved_attempt(output, artifacts):
    """Read only; missing rows remain unavailable without truth or new inference."""
    output = Path(output)
    saved, errors, name_used = {}, {}, None
    for name in ('result.json', 'checkpoint.json', 'result.json.tmp', 'checkpoint.json.tmp'):
        if not (output/name).is_file():
            continue
        try:
            candidate = json.loads((output/name).read_text())
            if not isinstance(candidate, dict):
                raise ValueError('saved record is not an object')
            saved, name_used = candidate, name
            break
        except (OSError, ValueError) as exc:
            errors[name] = f'{type(exc).__name__}: {exc}'
    tables = placeholder_evaluations('UNAVAILABLE_SAVED_ATTEMPT')
    stored_tables = saved.get('evaluation', {})
    if not isinstance(stored_tables, dict):
        errors['evaluation'] = 'invalid saved evaluation object; fixed unavailable rows displayed'
        stored_tables = {}
    for slot, table in stored_tables.items():
        if slot not in tables:
            continue
        expected = tables[slot]
        if isinstance(table, dict) and all(isinstance(table.get(field), list) and len(table[field]) == len(expected[field])
                for field in ('bits', 'frames', 'source_cells', 'source_states')):
            tables[slot] = table
        else:
            errors['evaluation/'+slot] = 'incomplete saved table; fixed unavailable rows displayed'
    calls, counts = None, saved.get('counts')
    ledger_path = output/'ledger.json'
    if ledger_path.is_file():
        try:
            calls = json.loads(ledger_path.read_text())
            if not isinstance(calls, dict) or not all(isinstance(row, dict) and 'kind' in row and 'status' in row for row in calls.values()):
                raise ValueError('invalid saved ledger rows')
            counts = {}
            for row in calls.values():
                group = counts.setdefault(row['kind'], dict(budget=0, started=0, returned=0, failed=0, interrupted=0, active=0))
                group['budget'] += 1
                group['started'] += row['status'] in ('STARTED', 'RETURNED', 'FAILED', 'INTERRUPTED')
                group['returned'] += row['status'] == 'RETURNED'
                group['failed'] += row['status'] == 'FAILED'
                group['interrupted'] += row['status'] == 'INTERRUPTED'
                group['active'] += row['status'] == 'STARTED'
        except (OSError, ValueError, TypeError) as exc:
            errors['ledger.json'] = f'{type(exc).__name__}: {exc}'
            calls, counts = None, saved.get('counts')
    return dict(status='EXISTING_ATTEMPT_NOT_REEXECUTED' if name_used else 'EXISTING_ATTEMPT_WITHOUT_CHECKPOINT',
        saved=saved, saved_record=name_used, existing_artifacts=artifacts, read_errors=errors,
        evaluation=tables, fixed_denominator=fixed_denominator(), counts=counts, calls=calls,
        count_source='ledger.json' if calls is not None else name_used,
        interpretation='Saved STARTED/unavailable rows stay spent; no model, FFT, inference or truth read')


def units():
    out = {'vae': 'vae_load'}
    out.update({f'source/{arm}': 'saved_rgb_load' for arm in ARMS})
    out.update({f'codebook/{key}': 'state_codebook' for key in core.KEYS})
    for arm in ARMS:
        for view in VIEWS:
            name = f'{arm}/{view}'
            out[name+'/edit'] = 'edit_construct'
            for p in range(4):
                bank = f'{name}/p{p}'
                out[bank+'/prepare'] = 'phase_prepare'
                out[bank+'/encode'] = 'posterior_encode'
                out[bank+'/fft'] = 'fft'
                for key in core.KEYS:
                    out[bank+'/'+key+'/pair'] = 'pair_extract'
                    out[bank+'/'+key+'/votes'] = 'vote_extract'
            for key in core.KEYS:
                slot = f'{name}/{key}'
                out[slot+'/path'] = 'blind_path_and_plan'
                out[slot+'/UNALIGNED/plan'] = 'baseline_plan'
                out[slot+'/ORACLE/plan'] = 'oracle_plan'
                for mode in MODES:
                    out[slot+'/'+mode+'/payload'] = 'payload_reduce'
                    out[slot+'/'+mode+'/evaluate'] = 'evaluation'
    return out


class Journal(shared.Journal):
    def __init__(self, output, backend):
        self.output, self.backend = output, backend
        self.data = {name: dict(kind=kind, status='NOT_ENTERED') for name, kind in units().items()}
        self.save()


class WanBackend(shared.WanBackend):
    def load_rgb(self, path):
        return shared.validate_rgb(torch.load(path, map_location='cpu', weights_only=True), pixels=True)

    def encode(self, rgb):
        from runtime.wan.vae import reencode_rgb24_readback
        if rgb.dtype != torch.uint8 or rgb.shape[1:] != (320, 512, 3):
            raise ValueError('uint8 received RGB24 phase input required')
        return reencode_rgb24_readback(self.vae, rgb.float()/255).detach().cpu().float()


def construct_received(rgb, view):
    if view == 'CROP_MID':
        return rgb[37:126]
    if view == 'DELETE43_PHASE':
        return torch.cat((rgb[:45], rgb[88:]), dim=0)
    raise ValueError('unknown fixed preparation view')


def run(source_root, output, truth_record, model_id=shared.DEFAULT_MODEL_ID, *, backend=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    old = execution_artifacts(output)
    if old:
        return read_saved_attempt(output, old)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(started_unix=time.time(), source_root=str(source_root), one_attempt=True), stream)
    except FileExistsError:
        return read_saved_attempt(output, execution_artifacts(output))
    # Backend construction and all possible model calls occur after durable attempt.
    result = dict(status='RUNNING', fixed_denominator=fixed_denominator(), scientific_pass=None,
        scope='one saved host, two fixed edits; oracle is same-bank correspondence control, not an accuracy upper bound')
    ledger = Journal(output, backend)
    plans = {(slot, mode): core.unit_plan(core.unavailable_path(n)) for slot, _, _, n, _ in slots() for mode in MODES}
    payloads = {(slot, mode): core.unavailable_payload() for slot, _, _, n, _ in slots() for mode in MODES}
    evaluated = {slot+'/'+mode: evaluation.unavailable_evaluation(n, 'NOT_ENTERED') for slot, _, _, n, _ in slots() for mode in MODES}
    paths = {slot: core.unavailable_path(n) for slot, _, _, n, _ in slots()}
    books, banks, pairs, votes = {}, {}, {}, {}
    inputs = {arm: dict(path=str(Path(source_root)/'media'/arm/'mp4_rgb8.pt'), status='NOT_ENTERED') for arm in ARMS}
    started = time.perf_counter()

    def checkpoint():
        result['counts'] = ledger.counts()
        result['evaluation'] = evaluated
        dump(output/'checkpoint.json', result)

    def tensor_save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)
        return value

    def npz_save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **value)
        return value

    def persist_blind():
        for slot, _, _, _, _ in slots():
            dump(output/'blind'/slot/'path.json', paths[slot])
            for mode in ('EST', 'UNALIGNED'):
                dump(output/'blind'/slot/mode/'plan.json', plans[slot, mode])
                dump(output/'blind'/slot/mode/'payload.json', payloads[slot, mode])

    dump(output/'protocol.json', dict(status='USER_ADOPTED_CARD57_CARD58', fixed_denominator=fixed_denominator(),
        expected_physical_calls=dict(VAE_load=1, E=16, FFT=16, D=0, codec=0, media_decode=0, generation=0, gradient=0, path_after_E=0),
        fixed_views=dict(CROP_MID='source[37:126]', DELETE43_PHASE='source[:45]+source[88:]'),
        blind_inputs='received RGB and N, key, public source181/22 states/4x8 geometry; no view or truth in core',
        candidate_score='source-k ordered float64 math.fsum, floor(N/8), ALL exact maximum scores',
        full_frame_uniqueness_promised=False, oracle_accuracy_upper_bound=False))
    checkpoint()
    try:
        backend = backend or WanBackend(model_id)
        ledger.backend = backend
        if getattr(backend, 'device', 'cpu') == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        for key, value in core.KEYS.items():
            b = ledger.call('codebook/'+key, lambda value=value: core.public_codebook(value))
            if b is not None:
                books[key] = b
        loaded = ledger.call('vae', backend.load)
        for arm in ARMS:
            rgb = ledger.call('source/'+arm, lambda arm=arm: backend.load_rgb(Path(source_root)/'media'/arm/'mp4_rgb8.pt'))
            inputs[arm].update(status=ledger.data['source/'+arm]['status'])
            if rgb is None:
                continue
            inputs[arm].update(shape=list(rgb.shape), dtype=str(rgb.dtype))
            for view, n in zip(VIEWS, LENGTHS):
                name = f'{arm}/{view}'
                anonymous_id = f'received_{2*ARMS.index(arm)+VIEWS.index(view)}'
                received = ledger.call(name+'/edit', lambda: tensor_save(
                    output/'prepared'/(anonymous_id+'.pt'), construct_received(rgb, view)))
                if received is None:
                    continue
                if len(received) != n:
                    raise ValueError('fixed edited received length mismatch')
                for p in range(4):
                    bank_id = f'{name}/p{p}'
                    def prepare():
                        phase, record = core.prepare_phase(received, p)
                        dump(output/'observations'/bank_id/'phase.json', record)
                        return phase
                    phase = ledger.call(bank_id+'/prepare', prepare)
                    if phase is None or loaded is None:
                        continue
                    def encode():
                        z = backend.encode(phase)
                        core.validate_bank(z, n, p)
                        return tensor_save(output/'features'/bank_id/'normalized.pt', z)
                    z = ledger.call(bank_id+'/encode', encode)
                    del phase
                    if z is None:
                        continue
                    banks[arm, view, p] = z
                    for key in core.KEYS:
                        if key not in books:
                            continue
                        obs = ledger.call(bank_id+'/'+key+'/pair', lambda key=key: npz_save(
                            output/'observations'/bank_id/key/'pairs.npz', core.pair_observations(z, n, p, books[key])))
                        if obs is not None:
                            pairs[arm, view, p, key] = obs
                    checkpoint()
                del received
            del rgb
        # Every path and explicit unit plan is persisted before the first FFT.
        for slot, arm, view, n, key in slots():
            def choose():
                obs = {p: pairs[arm, view, p, key] for p in range(4) if (arm, view, p, key) in pairs}
                if len(obs) != 4:
                    path = core.unavailable_path(n)
                    return path, core.unit_plan(path)
                q = core.combine_pairs(obs, n)
                path, arrays = core.select_path(q, n)
                npz_save(output/'blind'/slot/'candidates.npz', arrays)
                return path, core.unit_plan(path, arrays['parameters'][arrays['maximum_indices']])
            chosen = ledger.call(slot+'/path', choose)
            if chosen is not None:
                paths[slot], plans[slot, 'EST'] = chosen
            baseline = ledger.call(slot+'/UNALIGNED/plan', lambda n=n: core.direct_plan(n, (0, 0, 0), phase_zero_only=True))
            if baseline is not None:
                plans[slot, 'UNALIGNED'] = baseline
        persist_blind()
        dump(output/'plans_completion.json', dict(receiver_rows=8, truth_used=False, message_used=False))
        for arm in ARMS:
            for view, n in zip(VIEWS, LENGTHS):
                for p in range(4):
                    if (arm, view, p) not in banks:
                        continue
                    bank_id = f'{arm}/{view}/p{p}'
                    spectrum = ledger.call(bank_id+'/fft', lambda: core.fft_bank(banks[arm, view, p], n, p))
                    if spectrum is None:
                        continue
                    for key, value in core.KEYS.items():
                        raw = ledger.call(bank_id+'/'+key+'/votes', lambda value=value, key=key: npz_save(
                            output/'observations'/bank_id/key/'votes.npz', core.extract_votes(spectrum, n, p, value)))
                        if raw is not None:
                            votes[arm, view, p, key] = raw
                    del spectrum
                    checkpoint()
        for slot, arm, view, n, key in slots():
            available = {p: votes[arm, view, p, key] for p in range(4) if (arm, view, p, key) in votes}
            for mode in ('EST', 'UNALIGNED'):
                raw = ledger.call(slot+'/'+mode+'/payload', lambda: core.reduce_votes(plans[slot, mode], available))
                payloads[slot, mode] = raw if raw is not None else core.unavailable_payload(ledger.data[slot+'/'+mode+'/payload']['status'])
    except BaseException as exc:
        result['fatal_error'] = f'{type(exc).__name__}: {exc}'
    for arm in ARMS:
        inputs[arm]['status'] = ledger.data['source/'+arm]['status']
        inputs[arm]['call'] = ledger.data['source/'+arm]
        dump(output/'observations'/'inputs'/(arm+'.json'), inputs[arm])
    # Persist the complete fixed blind table even after failure or interruption.
    for slot, _, _, _, _ in slots():
        for mode in ('EST', 'UNALIGNED'):
            status = ledger.data[slot+'/'+mode+'/payload']['status']
            if status in ('FAILED', 'INTERRUPTED'):
                payloads[slot, mode] = core.unavailable_payload(status)
    persist_blind()
    dump(output/'blind_completion.json', dict(receiver_rows=8, payload_tables=16, truth_used=False,
        completed_unix=time.time(), fatal_error=result.get('fatal_error')))
    checkpoint()
    # Only this post-blind stage opens message truth and materializes true paths.
    try:
        true_maps = {f'received_{2*ARMS.index(arm)+VIEWS.index(view)}': dict(arm=arm, view=view,
            source_path=core.path_frames(n, evaluation.true_parameters(view))) for arm in ARMS for view,n in zip(VIEWS,LENGTHS)}
        dump(output/'post_blind'/'oracle_maps.json', dict(read_after_blind=True, received=true_maps))
        truth_bits = json.loads(Path(truth_record).read_text())['fixed_method_semantics']['bits']
        if len(truth_bits) != 32 or any(type(b) is not int or b not in (0, 1) for b in truth_bits):
            raise ValueError('32 binary truth bits required')
        dump(output/'evaluation_truth.json', dict(bits=truth_bits, source=str(truth_record), read_after_blind=True))
        for slot, arm, view, n, key in slots():
            params = evaluation.true_parameters(view)
            oracle = ledger.call(slot+'/ORACLE/plan', lambda: core.direct_plan(n, params))
            if oracle is not None:
                oracle['truth_inputs'] = True
                plans[slot, 'ORACLE'] = oracle
            available = {p: votes[arm, view, p, key] for p in range(4) if (arm, view, p, key) in votes}
            raw = ledger.call(slot+'/ORACLE/payload', lambda: core.reduce_votes(plans[slot, 'ORACLE'], available))
            payloads[slot, 'ORACLE'] = raw if raw is not None else core.unavailable_payload(ledger.data[slot+'/ORACLE/payload']['status'])
            for mode in MODES:
                if mode != 'ORACLE':
                    # Evaluate what was actually persisted, never rewrite blind results using truth.
                    plan = json.loads((output/'blind'/slot/mode/'plan.json').read_text())
                    pay = json.loads((output/'blind'/slot/mode/'payload.json').read_text())
                else:
                    plan, pay = plans[slot, mode], payloads[slot, mode]
                available_pair_starts = [int(a) for p in range(4) if (arm,view,p,key) in pairs for a in pairs[arm,view,p,key]['starts']]
                value = ledger.call(slot+'/'+mode+'/evaluate', lambda: evaluation.evaluate(plan, pay, core.path_frames(n, params), truth_bits,
                    pair_available_starts=available_pair_starts))
                evaluated[slot+'/'+mode] = value if value is not None else evaluation.unavailable_evaluation(n, ledger.data[slot+'/'+mode+'/evaluate']['status'])
                if mode == 'ORACLE':
                    dump(output/'post_blind'/slot/'plan.json', plan)
                    dump(output/'post_blind'/slot/'payload.json', pay)
    except BaseException as exc:
        result['evaluation_error'] = f'{type(exc).__name__}: {exc}'
    for slot, _, _, n, _ in slots():
        dump(output/'post_blind'/slot/'plan.json', plans[slot, 'ORACLE'])
        dump(output/'post_blind'/slot/'payload.json', payloads[slot, 'ORACLE'])
        for mode in MODES:
            name = slot+'/'+mode
            status = ledger.data[name+'/evaluate']['status']
            if status in ('FAILED', 'INTERRUPTED'):
                evaluated[name] = evaluation.unavailable_evaluation(n, status)
    comparisons = {}
    for slot, _, _, _, _ in slots():
        for base in ('UNALIGNED', 'ORACLE'):
            comparisons[slot+'/'+base+'_to_EST'] = evaluation.compare(evaluated[slot+'/'+base], evaluated[slot+'/EST'])
    for view in VIEWS:
        for key in core.KEYS:
            for mode in MODES:
                comparisons[f'P_to_S/{view}/{key}/{mode}'] = evaluation.compare(evaluated[f'P/{view}/{key}/{mode}'], evaluated[f'S/{view}/{key}/{mode}'])
    result.update(evaluation=evaluated, comparisons=comparisons, counts=ledger.counts(),
        seconds=time.perf_counter()-started, paths={s: dict(status=p['status'], maximum_count=p['maximum_count']) for s, p in paths.items()})
    if getattr(backend, 'device', 'cpu') == 'cuda':
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    result['status'] = 'EXECUTION_COMPLETE' if not result.get('fatal_error') and not result.get('evaluation_error') and all(r['status'] == 'RETURNED' for r in ledger.data.values()) else 'INCOMPLETE'
    dump(output/'evaluation.json', evaluated)
    dump(output/'result.json', result)
    checkpoint()
    return result
