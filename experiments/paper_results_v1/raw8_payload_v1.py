"""Eight saved RAW_U operations -> eight real Wan inputs, with durable resume.

No selector, writer, generation or media encode is called. Truth and existing
CENTERED_U reads are consumed only by the post-traversal report function.
"""
from __future__ import annotations
from collections import Counter
import copy
import json
from pathlib import Path
import time

from experiments.paper_results_v1.resume_state import StateJournal, atomic_bytes, read_state, state_exists, unit_attempt
from experiments.paper_results_v1.residual_joint_v1 import (
    DELETE_POSITIONS, RGB_SHAPE, _atomic_torch, _load_torch, _load_rgb_path,
    _default_wan_encode, _default_payload_read,
)

SCHEMA = 'paper-results-v1-raw8-payload-v1'
CASES = tuple(f'independent_20261015_{i:02d}' for i in range(1, 5))
GROUPS = ('CLEAN_MP4', 'DELETE_G0')
KEY = 'watermark'
SOURCE_DEFAULT = '/content/drive/MyDrive/Video-WM/Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1'
OPERATIONS = Path(__file__).with_name('raw8_saved_operations.json')
TERMINAL = {'READ', 'UNSUPPORTED', 'UNRESOLVED'}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode())


def reason(exc):
    return f'{type(exc).__name__}: {exc}'


def slots():
    return [f'{g.lower()}/{c}' for g in GROUPS for c in CASES]


def source_path(root, saved):
    """Rebase the known task-relative locator; never search other runs."""
    text = str(saved)
    marker = 'resumable-independent-four-v1/'
    if marker in text:
        relative = text.split(marker, 1)[1]
        result = (Path(root) / relative).resolve()
        if Path(root).resolve() not in result.parents:
            raise ValueError('source locator escapes the specified run')
        return result
    return Path(saved)


class Store:
    def __init__(self, output):
        self.output = Path(output).resolve()
        self.path = self.output / 'raw8_state.json'
        self.data = read_state(self.path)
        if self.data.get('schema') != SCHEMA or [r.get('slot_id') for r in self.data.get('rows', [])] != slots():
            raise ValueError('RAW8 fixed eight-row schema/denominator differs')
        self.journal = StateJournal(self.path); self.journal.seed_cache(self.data)
        changed = False
        for collection in ('rows', 'vae_loads', 'wan_encode_clips', 'payload_read_calls', 'media_decodes', 'invocations'):
            for row in self.data[collection]:
                if row.get('status') == 'RUNNING':
                    row.update(status='INTERRUPTED', reason='previous process ended before durable completion')
                    changed = True
        for phase, status in list(self.data['phases'].items()):
            if status == 'RUNNING':
                self.data['phases'][phase] = 'INTERRUPTED'; changed = True
        if changed:
            self.save()

    def save(self):
        self.journal.save(self.data)


def initialize(output, source_run=SOURCE_DEFAULT, operations=OPERATIONS, model_path='', device='cuda'):
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    path = output / 'raw8_state.json'
    if not state_exists(path):
        # Create every slot before any individual input is read.
        rows = [dict(slot_id=f'{g.lower()}/{c}', case_id=c, group=g, receiver='RAW_U',
                     key_label='K0', received_frames=181 if g == 'CLEAN_MP4' else 163,
                     planned_bits=32, missing_bits=32, status='PENDING', reason=None)
                for g in GROUPS for c in CASES]
        atomic_json(path, dict(schema=SCHEMA, rows=rows, phases={}, unit_attempts={},
                              source_run=str(Path(source_run)), operations_input=str(operations),
                              model_path=str(model_path), device=device, key=KEY,
                              vae_loads=[], wan_encode_clips=[], payload_read_calls=[],
                              media_decodes=[], cache_repairs=[], invocations=[]))
    store = Store(output)
    # Paths/device are locators, not immutable identity or version gates.
    store.data.update(source_run=str(Path(source_run)), operations_input=str(operations),
                      model_path=str(model_path), device=device)
    store.save(); report(store)
    return store


def validate_operation(record, row):
    from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span
    if (record.get('receiver') != 'RAW_U' or record.get('key_label') != 'K0'
            or record.get('slot_id') != row['slot_id']
            or record.get('received_frames') != row['received_frames']
            or any(record.get(k) is not False for k in ('truth_inputs', 'message_inputs', 'attack_inputs'))):
        raise ValueError('saved operation is not the fixed blind RAW_U slot')
    estimate = record['estimate']; operation = record.get('operation')
    if estimate.get('status') == 'UNRESOLVED':
        if estimate.get('path') is not None or operation is not None:
            raise ValueError('unresolved selection has an executable operation')
        return 'UNRESOLVED'
    path = estimate.get('path')
    if estimate.get('status') != 'ESTIMATED' or not isinstance(path, list) or len(path) != row['received_frames']:
        raise ValueError('saved path geometry is invalid')
    # Verify the actual mapping semantics, without choosing/recomputing a path.
    expected = decode_visible_span(path)
    if operation != expected:
        raise ValueError('saved operation does not implement its saved blind path')
    return operation['status']


def _paths(store, row):
    folder = store.output / 'raw_reads' / row['slot_id']
    return folder / 'read.json', folder / 'votes.npz', folder / 'raw_latent.pt'


def _validate_read(record, vote_path, operation):
    import numpy as np
    bits = record.get('decoded_bits')
    if (record.get('status') != 'READ' or not isinstance(bits, list) or len(bits) != 32
            or any(type(b) is not int or b not in (0, 1) for b in bits)
            or record.get('R') != operation['wan_support'] or record.get('output_frames') != operation['output_frames']):
        raise ValueError('raw read has invalid bits/support')
    with np.load(vote_path, allow_pickle=False) as z:
        signed, zeros = z['signed_votes'], z['zero_mask']
        if signed.shape != (4, record['R'], 240) or not np.isin(signed, [-1, 0, 1]).all() or not np.array_equal(signed == 0, zeros):
            raise ValueError('raw vote sidecar shape/sign/zero mismatch')
        decoded = []; votes = []
        for bit in range(32):
            ch, rem = divmod(bit, 8)
            ordered = (signed[ch, :, rem::8] > 0).astype(int).reshape(-1).tolist()
            decoded.append(Counter(ordered).most_common(1)[0][0])
            votes.append(dict(ones=sum(ordered), zeros=len(ordered)-sum(ordered), count=len(ordered)))
    if decoded != bits or votes != record['votes']:
        raise ValueError('raw saved bits/counts do not follow original vote ordering')


def _read_is_durable(store, row):
    if 'operation' not in row or row['operation'] is None or row['operation']['status'] != 'SUPPORTED':
        return False
    record, votes, _ = _paths(store, row)
    try:
        _validate_read(read_json(record), votes, row['operation'])
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def prepare(store):
    store.data['phases']['prepare'] = 'RUNNING'; store.save()
    saved = store.output / 'saved_raw_operations.json'
    pending = [r for r in store.data['rows'] if r['status'] not in ('UNSUPPORTED', 'UNRESOLVED')]
    try:
        # Once copied, the same saved operations are used on resume.
        records = read_json(saved if saved.is_file() else store.data['operations_input'])['rows']
        if len(records) != 8 or [r.get('slot_id') for r in records] != slots():
            raise ValueError('saved RAW operation file must contain the fixed eight slots')
        by_slot = {r['slot_id']: r for r in records}
        for row in store.data['rows']:
            validate_operation(by_slot[row['slot_id']], row)
        if not saved.is_file():
            atomic_json(saved, dict(schema=SCHEMA, rows=records, truth_inputs=False))
    except Exception as exc:
        for row in pending:
            if row['status'] != 'READ':
                row.update(status='MISSING' if isinstance(exc, FileNotFoundError) else 'FAILED', reason=reason(exc))
        store.data['phases']['prepare'] = 'COMPLETE'; store.save(); report(store); return store
    for row in pending:
        if row['status'] == 'READ' and _read_is_durable(store, row):
            continue
        if row['status'] in ('FAILED', 'MISSING', 'INTERRUPTED', 'READ'):
            row.setdefault('history', []).append(dict(status=row['status'], reason=row.get('reason'), at=time.time()))
        try:
            selection = copy.deepcopy(by_slot[row['slot_id']])
            status = validate_operation(selection, row)
            # This save is prior to reading reference payloads or truth.
            selector_path = store.output / 'selectors' / row['slot_id'] / 'raw_u.json'
            atomic_json(selector_path, selection)
            row.update(operation=selection['operation'], selector_record=str(selector_path),
                       status=status if status in ('UNSUPPORTED', 'UNRESOLVED') else 'PREPARED', reason=None)
            if status != 'SUPPORTED':
                store.save(); continue
            if _read_is_durable(store, row):
                row.update(status='READ', missing_bits=0); store.save(); continue
            # A committed own RAW latent contains everything needed by the
            # original reader. Source media locators are only needed to encode.
            try:
                _load_torch(_paths(store, row)[2], row['operation']['output_frames'])
            except Exception:
                pass
            else:
                row.update(status='PREPARED', missing_bits=32); store.save(); continue
            root = Path(store.data['source_run'])
            source = read_json(root / 'joint/run_state/residual_joint_state.json')
            origin = next(r for r in source['rows'] if r['slot_id'] == row['slot_id'])
            if source.get('key') != KEY:
                raise ValueError('saved source K0 differs from the adopted key')
            # Only media locators are copied; no centered operation/latent/vote enters execution.
            media = origin['media']
            mp4_field = 'mp4_path' if row['group'] == 'CLEAN_MP4' else 'source_mp4'
            row['media'] = dict(received_path=str(source_path(root, media['received_path'])),
                                mp4_path=str(source_path(root, media[mp4_field])))
            row.update(status='PREPARED', missing_bits=32)
        except Exception as exc:
            row.update(status='MISSING' if isinstance(exc, (FileNotFoundError, StopIteration)) else 'FAILED', reason=reason(exc))
        store.save()
    store.data['phases']['prepare'] = 'COMPLETE'; store.save(); report(store); return store


def _recorded_call(store, collection, invocation, **fields):
    row = dict(status='RUNNING', started_at=time.time(), invocation=invocation, **fields)
    store.data[collection].append(row); store.save(); return row


def _received_rgb(store, row, invocation):
    import numpy as np
    import torch
    shape = (row['received_frames'], 320, 512, 3)
    try:
        return _load_rgb_path(row['media']['received_path'], shape)
    except (OSError, ValueError):
        pass
    cache = store.output / 'media_cache' / row['case_id'] / 'published_rgb8.rgb8'
    try:
        full = _load_rgb_path(cache, RGB_SHAPE)
    except (OSError, ValueError):
        from runtime.wan.variable_rgb_media import decode_published
        call = _recorded_call(store, 'media_decodes', invocation, case_id=row['case_id'], source=row['media']['mp4_path'])
        try:
            with unit_attempt(store, 'decode/' + row['case_id'], operation='decode_existing_candidate_mp4'):
                decode_published(Path(row['media']['mp4_path']), cache, expected_frames=181)
            full = _load_rgb_path(cache, RGB_SHAPE)
            call.update(status='COMPLETE', finished_at=time.time())
        except BaseException as exc:
            call.update(status='FAILED' if isinstance(exc, Exception) else 'INTERRUPTED', reason=reason(exc), finished_at=time.time()); raise
        finally:
            store.save()
    if row['group'] == 'CLEAN_MP4':
        return full
    keep = [i for i in range(181) if i not in DELETE_POSITIONS]
    return full[torch.tensor(keep)].contiguous()


def _loader(store):
    from runtime.wan.generation import load_frozen_vae
    return load_frozen_vae({'model': {'id': store.data['model_path']}}, device=store.data['device'])


def run_wan(store, *, load_vae=None, encode=None, payload_read=None, load_rgb=None):
    from runtime.wan.video_trajectory_temporal_edit_receiver_v1 import operate_map
    import numpy as np
    load_vae = load_vae or (lambda: _loader(store))
    encode = encode or _default_wan_encode
    payload_read = payload_read or _default_payload_read
    store.data['invocations'].append(dict(started_at=time.time(), status='RUNNING'))
    invocation = len(store.data['invocations'])
    store.data['phases']['wan'] = 'RUNNING'; store.save()
    vae = None; load_failure = None
    try:
        with store.journal.periodic():
            for row in store.data['rows']:
                if row['status'] != 'PREPARED':
                    continue
                row.update(status='RUNNING', reason=None); store.save()
                record_path, vote_path, latent_path = _paths(store, row)
                operation = row['operation']; output_frames = operation['output_frames']
                try:
                    latent = None
                    if latent_path.is_file():
                        try:
                            latent = _load_torch(latent_path, output_frames)
                        except Exception as exc:
                            store.data['cache_repairs'].append(dict(slot_id=row['slot_id'], path=str(latent_path), reason=reason(exc), at=time.time()))
                            store.save()
                    if latent is None:
                        if vae is None:
                            if load_failure:
                                raise RuntimeError('VAE load already failed in this traversal: ' + load_failure)
                            call = _recorded_call(store, 'vae_loads', invocation)
                            try:
                                with unit_attempt(store, 'vae-load', operation='load_wan_vae'):
                                    vae = load_vae()
                                call.update(status='COMPLETE', finished_at=time.time())
                            except BaseException as exc:
                                load_failure = reason(exc)
                                call.update(status='FAILED' if isinstance(exc, Exception) else 'INTERRUPTED', reason=load_failure, finished_at=time.time()); raise
                            finally:
                                store.save()
                        received = load_rgb(row) if load_rgb else _received_rgb(store, row, invocation)
                        corrected = operate_map(received, operation['received_index_map'])
                        call = _recorded_call(store, 'wan_encode_clips', invocation, slot_id=row['slot_id'], output_frames=output_frames, R=operation['wan_support'])
                        def observer(event, error):
                            call.setdefault('events', []).append(dict(event=event, at=time.time(), error=reason(error) if error else None)); store.save()
                        try:
                            with unit_attempt(store, 'encode/' + row['slot_id'], operation='encode_saved_raw_corrected_rgb'):
                                latent = encode(vae, corrected, observer).detach().cpu().float().contiguous()
                                if tuple(latent.shape) != (1, 16, (output_frames-1)//4+1, 40, 64) or not bool(latent.isfinite().all()):
                                    raise ValueError('Wan output shape/nonfinite mismatch')
                                _atomic_torch(latent_path, latent)
                            call.update(status='COMPLETE', finished_at=time.time())
                        except BaseException as exc:
                            call.update(status='FAILED' if isinstance(exc, Exception) else 'INTERRUPTED', reason=reason(exc), finished_at=time.time()); raise
                        finally:
                            store.save()
                    call = _recorded_call(store, 'payload_read_calls', invocation, slot_id=row['slot_id'], output_frames=output_frames, R=operation['wan_support'])
                    try:
                        with unit_attempt(store, 'read/' + row['slot_id'], operation='original_raw_payload_read'):
                            result = payload_read(latent, KEY, output_frames, operation['source_coordinate_map'])
                            signed, zeros = result.pop('signed_votes'), result.pop('zero_mask')
                            import io
                            buffer = io.BytesIO(); np.savez_compressed(buffer, signed_votes=signed, zero_mask=zeros)
                            atomic_bytes(vote_path, buffer.getvalue())
                            result.update(vote_sidecar={'path': str(vote_path)}, input_kind='OWN_RAW_U_ENCODE', truth_inputs=False)
                            _validate_read(result, vote_path, operation)
                            atomic_json(record_path, result)
                            row.update(status='READ', reason=None, missing_bits=0, read_record=str(record_path), latent_path=str(latent_path))
                            store.save()
                        call.update(status='COMPLETE', finished_at=time.time())
                    except BaseException as exc:
                        call.update(status='FAILED' if isinstance(exc, Exception) else 'INTERRUPTED', reason=reason(exc), finished_at=time.time()); raise
                    finally:
                        store.save()
                except BaseException as exc:
                    row.update(status='MISSING' if isinstance(exc, FileNotFoundError) else 'FAILED' if isinstance(exc, Exception) else 'INTERRUPTED', reason=reason(exc), missing_bits=32)
                    store.save()
                    if not isinstance(exc, Exception):
                        raise
    except BaseException:
        store.data['phases']['wan'] = 'INTERRUPTED'; store.data['invocations'][-1]['status'] = 'INTERRUPTED'; raise
    else:
        store.data['phases']['wan'] = 'COMPLETE'; store.data['invocations'][-1]['status'] = 'COMPLETE'
    finally:
        store.data['invocations'][-1]['finished_at'] = time.time()
        vae = None
        store.save(); report(store)
    return store


def report(store):
    # Execution state never stores message truth or paired error counts.
    joined = store.data['phases'].get('wan') == 'COMPLETE' and all(r['status'] not in ('PENDING', 'PREPARED', 'RUNNING', 'INTERRUPTED') for r in store.data['rows'])
    truth = None; centered = {}; reference_error = None
    if joined:
        try:
            root = Path(store.data['source_run'])
            truth = read_json(root/'effective_independent_four_config.json')['payload_bits']
            if len(truth) != 32 or any(type(b) is not int or b not in (0, 1) for b in truth):
                raise ValueError('saved target message is not 32 binary integers')
            reference = read_json(root/'evaluation_report.json')
            centered = {r['slot_id'].removeprefix('wm_k0_reused/'):r for r in reference['rows'] if r['arm']=='WM_K0_REUSED'}
        except Exception as exc:
            truth = None; reference_error = reason(exc)
    rows = []
    for source in store.data['rows']:
        row = {k:source.get(k) for k in ('slot_id','case_id','group','status','reason','received_frames','planned_bits','missing_bits')}
        op = source.get('operation') or {}
        row.update(output_frames=op.get('output_frames'), R=op.get('wan_support'), raw_bits=None, raw_errors=None,
                   centered_status='NOT_JOINED' if not joined else 'MISSING', centered_bits=None, centered_errors=None, error_reduction=None)
        if source['status'] == 'READ':
            try:
                record, votes, _ = _paths(store, source)
                result = read_json(record); _validate_read(result, votes, op)
                row['raw_bits'] = result['decoded_bits']
            except Exception as exc:
                row.update(status='MISSING', reason=reason(exc), missing_bits=32)
        if joined and truth is not None:
            if row['raw_bits'] is not None:
                row['raw_errors'] = sum(a != b for a,b in zip(row['raw_bits'], truth))
            prior = centered.get(source['slot_id'])
            bits = prior.get('decoded_bits') if prior else None
            if prior and prior.get('status') == 'EVALUATED' and isinstance(bits,list) and len(bits)==32 and all(type(b) is int and b in (0,1) for b in bits):
                row.update(centered_status='EVALUATED', centered_bits=bits, centered_errors=sum(a!=b for a,b in zip(bits,truth)))
                if row['raw_errors'] is not None:
                    row['error_reduction'] = row['raw_errors'] - row['centered_errors']
        rows.append(row)
    pair_complete = sum(r['error_reduction'] is not None for r in rows)
    cumulative = {k:dict(Counter(r['status'] for r in store.data[k])) for k in ('vae_loads','wan_encode_clips','payload_read_calls','media_decodes')}
    last = len(store.data['invocations'])
    latest = {k:dict(Counter(r['status'] for r in store.data[k] if r.get('invocation')==last)) for k in cumulative}
    pending = [r for r in store.data['rows'] if r['status'] not in TERMINAL]
    needs_model = False
    for row in pending:
        if row['status'] not in ('PREPARED','RUNNING') or not row.get('operation'):
            continue
        try:
            _load_torch(_paths(store,row)[2], row['operation']['output_frames'])
        except Exception:
            needs_model = True
    summary = dict(schema=SCHEMA, planned_rows=8, planned_bits=256, status_counts=dict(Counter(r['status'] for r in rows)),
                   raw_read_rows=sum(r['raw_bits'] is not None for r in rows), missing_bits=sum(r['missing_bits'] for r in rows),
                   raw_evaluated_rows=sum(r['raw_errors'] is not None for r in rows),
                   raw_exact_rows=sum(r['raw_errors']==0 for r in rows) if truth is not None else None,
                   raw_bit_errors=sum(r['raw_errors'] for r in rows if r['raw_errors'] is not None) if truth is not None else None,
                   paired_rows=pair_complete, paired_error_reduction=sum(r['error_reduction'] for r in rows) if pair_complete==8 else None,
                   truth_join='POSTHOC_AFTER_FIXED_RAW_TRAVERSAL' if joined else 'NOT_JOINED', reference_error=reference_error,
                   pending_technical_rows=len(pending), needs_wan_model=needs_model, cumulative_calls=cumulative, latest_invocation_calls=latest,
                   generated_videos=0, media_encode_calls=0, selector_calls=0, centered_latent_reuses=0,
                   interpretation='Eight queries on four sources. Only real paired reads establish payload gain; both zero imply path improvement plus payload preservation. Missing/unsupported/failed retained. No population FPR or quality PASS.')
    if pair_complete==8:
        summary['paired_conclusion'] = ('BOTH_READERS_ZERO_ERROR_PATH_IMPROVEMENT_ONLY' if all(r['raw_errors']==0 and r['centered_errors']==0 for r in rows)
                                        else 'CENTERED_REMOVES_RAW_ERRORS_THIS_BATCH' if sum(r['raw_errors'] for r in rows)>0 and all(r['centered_errors']==0 for r in rows)
                                        else 'RETAIN_FULL_PAIRED_RESULTS_NO_AUTOMATIC_SUCCESS')
    else:
        summary['paired_conclusion'] = 'INCOMPLETE_FIXED_EIGHT_PAIR_COMPARISON'
    atomic_json(store.output/'evaluation_report.json', dict(summary=summary, rows=rows))
    atomic_json(store.output/'summary.json', summary)
    return summary


def run_all(store, **backends):
    prepare(store)
    return run_wan(store, **backends)
