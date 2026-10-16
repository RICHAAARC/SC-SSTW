"""Fixed saved-observation clip diagnosis with real per-operation Wan reads.

No generator, writer, framewise encoder, or media encoder is reachable here.
The unchanged selector receives only the edited observation matrices and its
receiver name. Source maps/message truth enter only editing or posthoc reports.
"""
from __future__ import annotations
from collections import Counter
import io
import json
from pathlib import Path
import time

from experiments.paper_results_v1.resume_state import StateJournal, atomic_bytes, read_state, state_exists, unit_attempt
from experiments.paper_results_v1.raw8_payload_v1 import atomic_json, read_json, source_path, _validate_read
from experiments.paper_results_v1.residual_joint_v1 import (
    RGB_SHAPE, _load_rgb_path, _atomic_torch, _load_torch,
    _default_wan_encode, _default_payload_read,
)

SPEC_PATH = Path(__file__).with_name('local_clip_diagnostic_v1.adopted.json')
SPEC = read_json(SPEC_PATH)
SCHEMA = SPEC['schema']
CASES = tuple(SPEC['cases'])
GROUPS = tuple(SPEC['views'])
RECEIVERS = tuple(SPEC['receivers'])
KEY = SPEC['key']
SOURCE_DEFAULT = '/content/drive/MyDrive/Video-WM/Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1'
TERMINAL = {'READ', 'UNSUPPORTED', 'UNRESOLVED'}
CALLS = ('vae_loads', 'wan_encode_clips', 'payload_read_calls', 'media_decodes', 'selector_calls', 'score_recomputes')
METHOD_FIELDS = ('cases', 'source_frames', 'key', 'receivers', 'crop_start', 'crop_length',
                 'edit_positions_local', 'views', 'delete_each', 'duplicate_extra_copies_each',
                 'payload_hex_posthoc_only', 'planned_rows', 'planned_bits')


def _check_method_record(output):
    """Compare known method values, never source identity or optional provenance."""
    current = {k: SPEC[k] for k in METHOD_FIELDS}
    records = [output / 'fixed_method_semantics.json', output / 'fixed_diagnostic_spec.json']
    known = set()
    unreadable = []
    for path in records:
        if not path.is_file():
            continue
        try:
            prior = read_json(path)
            if not isinstance(prior, dict):
                raise ValueError('optional method record is not a JSON object')
        except (OSError, ValueError, TypeError) as exc:
            unreadable.append(dict(record=path.name, reason=reason(exc)))
            continue
        differences = [k for k in METHOD_FIELDS if k in prior and prior[k] != current[k]]
        if differences:
            raise ValueError('fixed method values differ: ' + ', '.join(differences))
        known.update(k for k in METHOD_FIELDS if k in prior)
    return sorted(set(METHOD_FIELDS) - known), unreadable


def reason(exc):
    return f'{type(exc).__name__}: {exc}'


def view_source_indices(group):
    """Attack constructor/posthoc truth only; never passed to the selector."""
    base = list(range(SPEC['crop_start'], SPEC['crop_start'] + SPEC['crop_length']))
    edit = set(SPEC['edit_positions_local'])
    if group == 'CROP_MID':
        return base
    if group == 'CROP_DELETE':
        return [value for i, value in enumerate(base) if i not in edit]
    if group == 'CROP_DUPLICATE':
        return [value for i, value in enumerate(base) for _ in range(2 if i in edit else 1)]
    raise ValueError('unknown fixed view')


def planned_rows():
    return [dict(slot_id=f'{g.lower()}/{case}/{receiver.lower()}', case_id=case,
                 group=g, receiver=receiver, received_frames=len(view_source_indices(g)),
                 status='PENDING', reason=None, planned_bits=32, missing_bits=32)
            for case in CASES for g in GROUPS for receiver in RECEIVERS]


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.path = self.output / 'local_clip_state.json'
        self.data = read_state(self.path)
        expected = [(r['slot_id'], r['received_frames']) for r in planned_rows()]
        if self.data.get('schema') != SCHEMA or [(r.get('slot_id'), r.get('received_frames')) for r in self.data.get('rows', [])] != expected:
            raise ValueError('local clip fixed method/24-row denominator differs')
        unavailable, unreadable = _check_method_record(self.output)
        if unavailable or unreadable:
            self.data.setdefault('method_record_notes', []).append(dict(
                status='PRIOR_METHOD_RECORD_UNAVAILABLE', fields=unavailable,
                unreadable_records=unreadable,
                reason='current values recorded; prior optional method values were unavailable'))
        self.journal = StateJournal(self.path)
        self.journal.seed_cache(self.data)
        for collection in ('rows', 'invocations', *CALLS):
            for row in self.data[collection]:
                if row.get('status') == 'RUNNING':
                    row.update(status='INTERRUPTED', reason='previous process ended before durable completion')
        for phase, status in list(self.data['phases'].items()):
            if status == 'RUNNING':
                self.data['phases'][phase] = 'INTERRUPTED'
        self.save()
        self.save_method_record('fixed_method_semantics.json', {k: SPEC[k] for k in METHOD_FIELDS})

    def save(self):
        self.journal.save(self.data)

    def save_method_record(self, name, record):
        try:
            atomic_json(self.output / name, record)
        except OSError as exc:
            self.data.setdefault('method_record_notes', []).append(dict(
                status='OPTIONAL_METHOD_RECORD_WRITE_FAILED', record=name, reason=reason(exc)))
            self.save()  # Necessary execution-state failures still propagate.


def initialize(output, source_run=SOURCE_DEFAULT, model_path='', device='cuda'):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'local_clip_state.json'
    if not state_exists(path):
        _check_method_record(output)
        atomic_json(path, dict(schema=SCHEMA, rows=planned_rows(), phases={}, unit_attempts={},
                              source_run=str(source_run), model_path=str(model_path), device=device,
                              key=KEY, invocations=[], cache_repairs=[], **{k: [] for k in CALLS}))
    store = Store(output)
    store.data.update(source_run=str(source_run), model_path=str(model_path), device=device)
    store.save()
    store.save_method_record('fixed_diagnostic_spec.json', SPEC)
    report(store)
    return store


def _call(store, collection, **fields):
    row = dict(status='RUNNING', started_at=time.time(), invocation=len(store.data['invocations']), **fields)
    store.data[collection].append(row)
    store.save()
    return row


def _end(store, call, error=None):
    call.update(status='COMPLETE' if error is None else 'FAILED' if isinstance(error, Exception) else 'INTERRUPTED',
                finished_at=time.time())
    if error is not None:
        call['reason'] = reason(error)
    store.save()


def _paths(store, row):
    read_owner = row.get('read_owner', row['slot_id'])
    latent_owner = row.get('latent_owner', row['slot_id'])
    read_folder = store.output / 'payload' / read_owner
    return read_folder / 'read.json', read_folder / 'votes.npz', store.output / 'payload' / latent_owner / 'latent.pt'


def _durable_read(store, row):
    if not row.get('operation') or row['operation']['status'] != 'SUPPORTED':
        return False
    try:
        record, votes, _ = _paths(store, row)
        _validate_read(read_json(record), votes, row['operation'])
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _latent(store, row):
    return _load_torch(_paths(store, row)[2], row['operation']['output_frames'])


def _validate_selection(selection, row):
    from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span
    if selection.get('receiver') != row['receiver'] or any(selection.get(k) is not False for k in ('truth_inputs', 'attack_inputs', 'message_inputs')):
        raise ValueError('saved selection has invalid blind receiver inputs')
    estimate = selection['estimate']
    if estimate['status'] == 'UNRESOLVED':
        if estimate.get('path') is not None or selection.get('operation') is not None:
            raise ValueError('unresolved path has an executable operation')
        return
    path = estimate.get('path')
    if estimate['status'] != 'ESTIMATED' or not isinstance(path, list) or len(path) != row['received_frames']:
        raise ValueError('saved selected path has wrong received geometry')
    if selection['operation'] != decode_visible_span(path):
        raise ValueError('saved selected operation differs from unchanged span semantics')


def _source_record(store, case):
    root = Path(store.data['source_run'])
    state = read_json(root / 'joint/run_state/residual_joint_state.json')
    if not isinstance(state, dict) or not isinstance(state.get('rows'), list):
        raise ValueError('source state must be an object with a rows list')
    store.data['source_format_label'] = state.get('schema')  # Optional locator only.
    if state.get('key') != KEY:
        raise ValueError('source clock key differs from fixed K0')
    if any(not isinstance(r, dict) or not isinstance(r.get('case_id'), str)
           or not isinstance(r.get('group'), str) for r in state['rows']):
        raise ValueError('source rows require object entries with string case_id and group')
    matches = [r for r in state['rows'] if r['case_id'] == case and r['group'] == 'CLEAN_MP4']
    if not matches:
        raise FileNotFoundError(f'no CLEAN_MP4 source row for {case}')
    if len(matches) != 1:
        raise ValueError(f'expected one CLEAN_MP4 source row for {case}; found {len(matches)}')
    return matches[0]


def _media_locators(store, row):
    origin = _source_record(store, row['case_id'])
    root = Path(store.data['source_run'])
    row['media'] = {key: str(source_path(root, origin['media'][key])) for key in ('received_path', 'mp4_path')}


def _numeric_scores(path, frames):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        signed, rho = data['signed_projection'].copy(), data['rho'].copy()
    if signed.shape != (frames, 181) or rho.shape != signed.shape or not np.isfinite(signed).all() or not np.isfinite(rho).all() or (rho <= 0).any():
        raise ValueError('score arrays must be finite [received_frames,181], with positive rho')
    return signed, rho


def _view_scores(store, row):
    import numpy as np
    destination = store.output / 'observations' / row['group'].lower() / row['case_id'] / 'scores.npz'
    if destination.is_file():
        try:
            signed, rho = _numeric_scores(destination, row['received_frames'])
            return signed, rho, destination
        except Exception as exc:
            store.data['cache_repairs'].append(dict(path=str(destination), reason=reason(exc), at=time.time()))
            store.save()
    root = Path(store.data['source_run'])
    origin = _source_record(store, row['case_id'])
    blind_path = source_path(root, origin['clock']['selector_record'])
    scores_path = blind_path.with_name('scores.npz')
    try:
        signed, rho = _numeric_scores(scores_path, 181)
        provenance = dict(kind='SAVED_CLEAN_SCORES', path=str(scores_path), new_framewise_model_calls=0)
    except (OSError, ValueError, KeyError, TypeError) as score_error:
        from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import score_framewise
        latent_path = source_path(root, origin['clock']['latent_path'])
        latent = np.load(latent_path, allow_pickle=False)
        call = _call(store, 'score_recomputes', case_id=row['case_id'], source=str(latent_path), original_score_error=reason(score_error))
        try:
            evidence = score_framewise(latent, KEY)
            signed, rho = evidence['signed_projection'], evidence['rho']
        except BaseException as exc:
            _end(store, call, exc)
            raise
        else:
            _end(store, call)
        provenance = dict(kind='SAVED_CLEAN_FRAMEWISE_LATENT_CPU_SCORE', path=str(latent_path), new_framewise_model_calls=0)
    indices = view_source_indices(row['group'])
    signed, rho = signed[indices], rho[indices]
    buffer = io.BytesIO()
    np.savez_compressed(buffer, signed_projection=signed, rho=rho)
    atomic_bytes(destination, buffer.getvalue())
    atomic_json(destination.with_suffix('.json'), dict(provenance=provenance, received_frames=len(indices),
                observation_reuse='row selection of saved per-frame observations; not a new Wan encode',
                centering='computed after row selection by unchanged receive_scores'))
    return signed, rho, destination


def prepare(store, *, selector=None):
    from main.tube_state.receiver_controls_v1 import receive_scores
    selector = selector or receive_scores
    store.data['phases']['prepare'] = 'RUNNING'
    store.save()
    for row in store.data['rows']:
        if row['status'] == 'READ' and _durable_read(store, row):
            continue
        if row['status'] in ('UNSUPPORTED', 'UNRESOLVED'):
            continue
        if row['status'] not in ('PENDING', 'PREPARED'):
            row.setdefault('history', []).append(dict(status=row['status'], reason=row.get('reason'), at=time.time()))
        selection_path = store.output / 'selections' / row['slot_id'] / 'selection.json'
        try:
            if selection_path.is_file():
                selection = read_json(selection_path)
                _validate_selection(selection, row)
            else:
                signed, rho, observation_path = _view_scores(store, row)
                call = _call(store, 'selector_calls', slot_id=row['slot_id'])
                try:
                    # Only the received matrices and fixed receiver enter selection.
                    selection = selector(signed, rho, row['receiver'])
                    selection.update(receiver=row['receiver'], truth_inputs=False, attack_inputs=False,
                                     message_inputs=False, observation_path=str(observation_path))
                    _validate_selection(selection, row)
                    atomic_json(selection_path, selection)
                except BaseException as exc:
                    _end(store, call, exc)
                    raise
                else:
                    _end(store, call)
            row.update(selection_record=str(selection_path), operation=selection.get('operation'),
                       status=selection['status'] if selection['status'] != 'SUPPORTED' else 'PREPARED',
                       reason=selection.get('reason'), missing_bits=32)
            if row['status'] == 'PREPARED':
                # Own durable latent/read can resume even when original source files vanish.
                if _durable_read(store, row):
                    row.update(status='READ', missing_bits=0)
                else:
                    try:
                        _latent(store, row)
                    except Exception:
                        if not all(row.get('media', {}).get(k) for k in ('received_path', 'mp4_path')):
                            _media_locators(store, row)
        except BaseException as exc:
            row.update(status='MISSING' if isinstance(exc, (FileNotFoundError, StopIteration)) else 'FAILED' if isinstance(exc, Exception) else 'INTERRUPTED',
                       reason=reason(exc), missing_bits=32)
            store.save()
            if not isinstance(exc, Exception):
                store.data['phases']['prepare'] = 'INTERRUPTED'
                store.save()
                raise
        store.save()
    store.data['phases']['prepare'] = 'COMPLETE'
    store.save()
    report(store)
    return store


def _received_rgb(store, row):
    import torch
    try:
        full = _load_rgb_path(row['media']['received_path'], RGB_SHAPE)
    except (OSError, ValueError):
        cache = store.output / 'media_cache' / row['case_id'] / 'published.rgb8'
        try:
            full = _load_rgb_path(cache, RGB_SHAPE)
        except (OSError, ValueError):
            invocation = len(store.data['invocations'])
            previous_failure = next((call for call in reversed(store.data['media_decodes'])
                if call.get('case_id') == row['case_id'] and call.get('invocation') == invocation
                and call.get('status') == 'FAILED'), None)
            if previous_failure:
                # One physical decode attempt per source in this traversal.
                # A later explicit Run all has a new invocation and may retry.
                error_type = FileNotFoundError if previous_failure.get('error_type') == 'FileNotFoundError' else RuntimeError
                raise error_type('same-source decode already failed in this traversal: ' + previous_failure['reason'])
            from runtime.wan.variable_rgb_media import decode_published
            call = _call(store, 'media_decodes', case_id=row['case_id'], source=row['media']['mp4_path'])
            try:
                with unit_attempt(store, 'decode/' + row['case_id'], operation='decode_existing_candidate_mp4'):
                    decode_published(Path(row['media']['mp4_path']), cache, expected_frames=181)
                    full = _load_rgb_path(cache, RGB_SHAPE)
            except BaseException as exc:
                call['error_type'] = type(exc).__name__
                _end(store, call, exc)
                raise
            else:
                _end(store, call)
    return full[torch.tensor(view_source_indices(row['group']))].contiguous()


def _matching_rows(store, row):
    for other in store.data['rows']:
        if other is row or other['case_id'] != row['case_id'] or other['group'] != row['group']:
            continue
        if not other.get('operation') or other['operation']['status'] != 'SUPPORTED':
            continue
        if (other['operation']['received_index_map'] == row['operation']['received_index_map']
                and other['operation']['output_frames'] == row['operation']['output_frames']):
            yield other


def _loader(store):
    from runtime.wan.generation import load_frozen_vae
    return load_frozen_vae({'model': {'id': store.data['model_path']}}, device=store.data['device'])


def run_wan(store, *, load_vae=None, encode=None, payload_read=None, load_rgb=None):
    import numpy as np
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import operate_map
    load_vae = load_vae or (lambda: _loader(store))
    encode = encode or _default_wan_encode
    payload_read = payload_read or _default_payload_read
    store.data['invocations'].append(dict(status='RUNNING', started_at=time.time()))
    store.data['phases']['wan'] = 'RUNNING'
    store.save()
    vae = None
    load_failure = None
    try:
        with store.journal.periodic():
            for row in store.data['rows']:
                if row['status'] != 'PREPARED':
                    continue
                row.update(status='RUNNING', reason=None)
                store.save()
                op = row['operation']
                try:
                    # Share actual identical input only within the same source/view.
                    # Fully equal operations may also share the completed reader record.
                    matches = list(_matching_rows(store, row))
                    reader_owner = next((other for other in matches if other['operation'] == op and _durable_read(store, other)), None)
                    if reader_owner:
                        row.update(read_owner=reader_owner.get('read_owner', reader_owner['slot_id']),
                                   latent_owner=reader_owner.get('latent_owner', reader_owner['slot_id']),
                                   status='READ', missing_bits=0, reuse='IDENTICAL_OPERATION_SAME_VIEW')
                        store.save()
                        continue
                    latent = None
                    for candidate in [row, *matches]:
                        try:
                            latent = _latent(store, candidate)
                        except Exception:
                            continue
                        if candidate is not row:
                            row['latent_owner'] = candidate.get('latent_owner', candidate['slot_id'])
                            row['reuse'] = 'IDENTICAL_RGB_MAP_SAME_VIEW_OWN_READER'
                        break
                    row.pop('read_owner', None)
                    if latent is None:
                        row.pop('latent_owner', None)
                        received = load_rgb(row) if load_rgb else _received_rgb(store, row)
                        corrected = operate_map(received, op['received_index_map'])
                        if vae is None:
                            if load_failure:
                                raise RuntimeError('VAE load already failed in this traversal: ' + load_failure)
                            call = _call(store, 'vae_loads')
                            try:
                                with unit_attempt(store, 'vae-load', operation='load_wan_vae'):
                                    vae = load_vae()
                            except BaseException as exc:
                                load_failure = reason(exc)
                                _end(store, call, exc)
                                raise
                            else:
                                _end(store, call)
                        call = _call(store, 'wan_encode_clips', slot_id=row['slot_id'], output_frames=op['output_frames'], R=op['wan_support'])
                        def observer(event, error):
                            call.setdefault('events', []).append(dict(event=event, at=time.time(), error=reason(error) if error else None))
                            store.save()
                        try:
                            with unit_attempt(store, 'encode/' + row['slot_id'], operation='encode_blind_corrected_clip_rgb'):
                                latent = encode(vae, corrected, observer).detach().cpu().float().contiguous()
                                if tuple(latent.shape) != (1, 16, (op['output_frames'] - 1)//4 + 1, 40, 64) or not bool(latent.isfinite().all()):
                                    raise ValueError('Wan output has invalid geometry/numerics')
                                _atomic_torch(_paths(store, row)[2], latent)
                        except BaseException as exc:
                            _end(store, call, exc)
                            raise
                        else:
                            _end(store, call)
                    call = _call(store, 'payload_read_calls', slot_id=row['slot_id'], output_frames=op['output_frames'], R=op['wan_support'])
                    try:
                        with unit_attempt(store, 'read/' + row['slot_id'], operation='original_payload_reader'):
                            result = payload_read(latent, KEY, op['output_frames'], op['source_coordinate_map'])
                            signed, zeros = result.pop('signed_votes'), result.pop('zero_mask')
                            record_path, vote_path, latent_path = _paths(store, row)
                            buffer = io.BytesIO()
                            np.savez_compressed(buffer, signed_votes=signed, zero_mask=zeros)
                            atomic_bytes(vote_path, buffer.getvalue())
                            result.update(vote_sidecar={'path': str(vote_path)}, input_kind='LOCAL_CLIP_BLIND_OPERATION', truth_inputs=False)
                            _validate_read(result, vote_path, op)
                            atomic_json(record_path, result)
                            row.update(status='READ', reason=None, missing_bits=0,
                                       read_record=str(record_path), latent_path=str(latent_path))
                            store.save()
                    except BaseException as exc:
                        _end(store, call, exc)
                        raise
                    else:
                        _end(store, call)
                except BaseException as exc:
                    row.update(status='MISSING' if isinstance(exc, FileNotFoundError) else 'FAILED' if isinstance(exc, Exception) else 'INTERRUPTED',
                               reason=reason(exc), missing_bits=32)
                    store.save()
                    if not isinstance(exc, Exception):
                        raise
    except BaseException:
        store.data['phases']['wan'] = 'INTERRUPTED'
        store.data['invocations'][-1]['status'] = 'INTERRUPTED'
        raise
    else:
        store.data['phases']['wan'] = 'COMPLETE'
        store.data['invocations'][-1]['status'] = 'COMPLETE'
    finally:
        store.data['invocations'][-1]['finished_at'] = time.time()
        vae = None
        store.save()
        report(store)
    return store


def _path_metrics(store, row, joined):
    if not joined:
        return dict(status='NOT_JOINED', reason='fixed prepare traversal has not completed')
    if not row.get('selection_record'):
        return dict(status='MISSING', reason='no completed blind selection')
    try:
        selection = read_json(row['selection_record'])
        _validate_selection(selection, row)
        estimate, op = selection['estimate'], selection.get('operation')
        base = dict(status=estimate['status'], operation_status=op['status'] if op else None,
                    score_gap=estimate.get('score_gap'), exact_tie=estimate.get('exact_tie'),
                    operation=op, received_frames=row['received_frames'])
        if estimate['status'] != 'ESTIMATED':
            return base
        truth, path = view_source_indices(row['group']), estimate['path']
        errors = [abs(a-b) for a,b in zip(path, truth)]
        selected = op['received_index_map']
        return dict(**base, estimated_start=path[0], estimated_end=path[-1],
                    truth_start=truth[0], truth_end=truth[-1], start_error=abs(path[0]-truth[0]),
                    end_error=abs(path[-1]-truth[-1]), interval_exact=(path[0],path[-1]) == (truth[0],truth[-1]),
                    path_exact=not any(errors), support_error_sum=sum(errors),
                    support_error_mean=sum(errors)/len(errors), support_error_max=max(errors),
                    support_correct_frames=errors.count(0), wrong_correspondences=sum(e>0 for e in errors),
                    support_correct_fraction=errors.count(0)/len(errors),
                    correctly_located_distinct_source_frames=len({s for s,e in zip(truth, errors) if e == 0}),
                    selected_distinct_received_frames=len(set(selected)),
                    selected_distinct_true_source_frames=len({truth[i] for i in selected}),
                    received_distinct_true_source_frames=len(set(truth)),
                    support_semantics='single-source frames; duplicated observations are dependent; nearest copies are not restored missing frames')
    except Exception as exc:
        return dict(status='FAILED', reason=reason(exc))


def report(store):
    # Truth never enters execution state or selection. Payload truth is joined
    # only after the fixed traversal terminates; path truth follows all choices.
    pending_prepare = {'PENDING', 'RUNNING', 'INTERRUPTED'}
    path_joined = (store.data['phases'].get('prepare') == 'COMPLETE'
                   and not any(r['status'] in pending_prepare for r in store.data['rows']))
    joined = (path_joined and store.data['phases'].get('wan') == 'COMPLETE'
              and not any(r['status'] in pending_prepare | {'PREPARED'} for r in store.data['rows']))
    truth = [int(b) for b in f"{int(SPEC['payload_hex_posthoc_only'], 16):032b}"] if joined else None
    rows = []
    for source in store.data['rows']:
        row = {k: source.get(k) for k in ('slot_id','case_id','group','receiver','status','reason','received_frames','planned_bits','missing_bits','read_owner','latent_owner','reuse')}
        row.update(path=_path_metrics(store, source, path_joined), decoded_bits=None, bit_errors=None, message_exact=None)
        if source['status'] == 'READ':
            try:
                record, votes, _ = _paths(store, source)
                result = read_json(record)
                _validate_read(result, votes, source['operation'])
                row['decoded_bits'] = result['decoded_bits']
                if truth is not None:
                    row['bit_errors'] = sum(a != b for a,b in zip(result['decoded_bits'], truth))
                    row['message_exact'] = row['bit_errors'] == 0
            except Exception as exc:
                row.update(status='MISSING', reason=reason(exc), missing_bits=32)
        rows.append(row)
    by_receiver = {}
    for receiver in RECEIVERS:
        arm = [r for r in rows if r['receiver'] == receiver]
        evaluated = [r for r in arm if r['bit_errors'] is not None]
        by_receiver[receiver] = dict(planned_rows=12, planned_bits=384,
            status_counts=dict(Counter(r['status'] for r in arm)), read_rows=sum(r['decoded_bits'] is not None for r in arm),
            evaluated_rows=len(evaluated), bit_errors=sum(r['bit_errors'] for r in evaluated) if evaluated else None,
            exact_messages=sum(r['message_exact'] is True for r in arm) if evaluated else None,
            missing_bits=sum(r['missing_bits'] for r in arm),
            exact_paths=sum(r['path'].get('path_exact') is True for r in arm),
            path_evaluated_rows=sum(r['path'].get('status') == 'ESTIMATED' for r in arm),
            support_error_sum=sum(r['path'].get('support_error_sum',0) for r in arm))
    paired = []
    for case in CASES:
        for group in GROUPS:
            pair = [r for r in rows if r['case_id'] == case and r['group'] == group]
            valid = all(r['bit_errors'] is not None for r in pair)
            paired.append(dict(case_id=case, group=group, complete=valid,
                               error_reduction=pair[0]['bit_errors']-pair[1]['bit_errors'] if valid else None))
    ready = [r for r in store.data['rows'] if r['status'] == 'PREPARED']
    unique_inputs = set()
    for r in ready:
        reusable = False
        for candidate in [r, *_matching_rows(store, r)]:
            try:
                _latent(store, candidate)
                reusable = True
                break
            except Exception:
                pass
        if not reusable:
            unique_inputs.add((r['case_id'], r['group'], tuple(r['operation']['received_index_map'])))
    cumulative = {k:dict(Counter(c['status'] for c in store.data[k])) for k in CALLS}
    latest = {k:dict(Counter(c['status'] for c in store.data[k] if c.get('invocation') == len(store.data['invocations']))) for k in CALLS}
    complete_pairs = sum(p['complete'] for p in paired)
    reduction = sum(p['error_reduction'] for p in paired) if complete_pairs == 12 else None
    summary = dict(schema=SCHEMA, independent_sources=4, related_views=12, planned_rows=24, planned_bits=768,
        status_counts=dict(Counter(r['status'] for r in rows)), missing_bits=sum(r['missing_bits'] for r in rows),
        by_receiver=by_receiver, complete_pairs=complete_pairs, paired_error_reduction=reduction,
        paired_ber_gain=reduction/384 if reduction is not None else None,
        truth_join='POSTHOC_AFTER_FIXED_TRAVERSAL' if joined else 'NOT_JOINED',
        path_truth_join='POSTHOC_AFTER_FIXED_PREPARE_TRAVERSAL' if path_joined else 'NOT_JOINED',
        pending_technical_rows=sum(r['status'] not in TERMINAL for r in rows),
        needs_wan_model=bool(unique_inputs), pending_unique_wan_inputs=len(unique_inputs),
        cumulative_calls=cumulative, latest_invocation_calls=latest,
        new_generation=0, new_writer_calls=0, new_framewise_encodes=0, new_media_encodes=0, original_raw8_runs=0,
        interpretation='Fixed one-start/one-length residual diagnosis. Saved-score path analysis and real Wan payload reads are separate. Retain every failed/missing/unsupported/unresolved row. Nearest copies do not restore missing frames; repeated views/frames are not independent evidence. No general synchronization, FPR or BER-gain claim.')
    atomic_json(store.output/'evaluation_report.json', dict(summary=summary, rows=rows, pairs=paired))
    atomic_json(store.output/'summary.json', summary)
    return summary


def run_all(store, **backends):
    prepare(store)
    return run_wan(store, **backends)
