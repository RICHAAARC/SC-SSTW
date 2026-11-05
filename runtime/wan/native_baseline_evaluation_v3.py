"""Saved-native-only evaluation. No model or codec calls."""
from collections import Counter
import csv
import os
from pathlib import Path
from main.baselines.native_paper_fpr1_v3 import (METHODS, CHANNELS, CONDITIONS, tasks, view_score,
    calibrate, family_event, rate_summary, action_sensitivity, source_maps)
from runtime.wan.native_baseline_storage_v1 import read_json, write_json, now
from runtime.wan.native_baseline_storage_v3 import read_native, unit_ledger, state

STATUS_CODES = {'NOT_STARTED': 0, 'RETURNED': 1, 'FAILED': 2, 'INTERRUPTED_CONSUMED': 3,
                'OUTPUT_MISSING': 4, 'OUTPUT_INVALID': 5, 'DEPENDENCY_FAILED': 6}


def evaluate_unit(output, source, method):
    import numpy as np
    output = Path(output); folder = output/'units'/method/source['id']
    if state(folder) in ('NEVER_STARTED', 'INTERRUPTED_CONSUMED'):
        # A dead worker has ended blind work. Recovery is read-only physical accounting.
        pass
    elif not (folder/'blind_closed.json').exists():
        raise ValueError('Native evidence not closed before query access')
    records, errors = read_native(folder)
    ledger = unit_ledger(folder)
    grouped = {}
    for meta, arrays in records:
        grouped.setdefault(meta['task'], []).append((meta['start'], arrays))
    query = read_json(output/'queries.json')[f"{method}/{source['id']}"]
    maps = source_maps(source)
    rows, frame_arrays, offset = [], [], 0
    for t in tasks(source, method):
        n, k = t['frames'], t['nbits']
        bits = np.zeros((n, k), dtype=np.uint8); valid = np.zeros(n, dtype=bool)
        saved = np.zeros(n, dtype=bool); selected = np.zeros(n, dtype=np.uint32)
        center = np.zeros(n, dtype=np.uint32); video_bits = None
        invalid = False
        for start, a in grouped.get(t['id'], []):
            try:
                b = np.asarray(a['bits']); length = len(b); end = start+length
                if start < 0 or end > n or b.shape != (length, k) or length == 0 or saved[start:end].any():
                    raise ValueError('Invalid/duplicate native frame range')
                if not np.isin(b, [0, 1]).all():
                    raise ValueError('Non-binary native bits')
                v = np.asarray(a['valid'], dtype=bool) if method == 'wam' else np.ones(length, dtype=bool)
                if v.shape != (length,):
                    raise ValueError('Invalid native validity shape')
                if method == 'wam':
                    for name in ('means', 'sums'):
                        if a[name].shape != (length, 32) or not np.isfinite(a[name]).all():
                            raise ValueError('Invalid WAM diagnostic values')
                    nc, ic = a['selected_count'], a['center_count']
                    if (nc.shape != (length,) or ic.shape != (length,) or (nc > 65536).any()
                            or (ic > 16384).any() or (ic > nc).any() or not np.array_equal(v, nc > 0)
                            or not np.array_equal(a['mask_shape'], [256, 256])):
                        raise ValueError('Invalid native mask statistics')
                    selected[start:end] = nc; center[start:end] = ic
                else:
                    if (a['logits'].shape != (n, 257) or not np.isfinite(a['logits']).all()
                            or a['logits'].dtype != np.float32 or start != 0 or length != n
                            or a['video_bits'].shape != (256,) or not np.isin(a['video_bits'], [0, 1]).all()):
                        raise ValueError('Invalid native PixelSeal logits/video output')
                    video_bits = a['video_bits'].tolist()
                bits[start:end] = b; valid[start:end] = v; saved[start:end] = True
            except (ValueError, KeyError, TypeError, IndexError) as e:
                invalid = True; errors.append(dict(task=t['id'], error=str(e)))
        statuses = []
        for f in range(n):
            batch = f//8*8 if method == 'wam' else 0
            call = ledger.get(f"detect/{t['id']}/{batch}", {}).get('status', 'NOT_STARTED')
            if call == 'RETURNED' and not saved[f]:
                call = 'OUTPUT_MISSING'
            statuses.append(call)
        readable = valid & saved & np.asarray([s == 'RETURNED' for s in statuses])
        complete = not invalid and saved.all() and all(s == 'RETURNED' for s in statuses)
        score = view_score(method, bits, readable, query['query'], complete=bool(complete), video_bits=video_bits)
        wrong_score = (view_score(method, bits, readable, query['wrong_query'], complete=bool(complete),
                                  video_bits=video_bits) if query['wrong_query'] is not None else None)
        correct = (bits == np.asarray(query['query'])).sum(1).astype(np.int16)
        correct[~readable] = 0
        wrong = np.where(readable, k-correct, 0).astype(np.int16)
        unknown = np.where(readable, 0, k).astype(np.int16)
        iou = np.full(n, np.nan)
        if method == 'wam':
            known = saved & np.asarray([s == 'RETURNED' for s in statuses])
            iou[known] = (center[known].astype(float)/(selected[known]+16384-center[known])
                          if t['arm'] == 'WM' else (selected[known] == 0).astype(float))
        mapping = maps['conditions'][t['condition']]
        frame_arrays.append(dict(status=np.asarray([STATUS_CODES.get(s, 5) for s in statuses], dtype=np.uint8),
            readable=readable, correct=correct, wrong=wrong, unknown=unknown,
            source_frame=np.asarray(mapping['indices'], dtype=np.int16),
            copy_ordinal=np.asarray([x['copy_ordinal'] for x in mapping['instances']], dtype=np.int16),
            localization_iou=iou))
        read_bits = int(readable.sum())*k
        video_correct = None if video_bits is None or not complete else sum(a == b for a, b in zip(video_bits, query['query']))
        rows.append(dict(t, complete=bool(complete), score=score, wrong_query_score=wrong_score,
            frame_offset=offset, planned_frames=n, planned_bits=n*k, correct=int(correct.sum()),
            wrong=int(wrong.sum()), unknown=int(unknown.sum()), coverage=read_bits/(n*k),
            readable_ber=int(wrong.sum())/read_bits if read_bits else None,
            complete_messages_correct=int(((correct == k) & readable).sum()),
            complete_messages_error=int(((correct != k) & readable).sum()),
            complete_messages_unknown=int((~readable).sum()), statuses=dict(Counter(statuses)),
            native_video_correct_bits=video_correct,
            native_video_ber=None if video_correct is None else (k-video_correct)/k,
            native_video_exact=None if video_correct is None else video_correct == k,
            native_video_status='N/A' if method == 'wam' else 'RETURNED' if video_correct is not None else 'UNKNOWN',
            localization_iou_mean=float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
            unique_retained_source_frames=mapping['unique_retained_source_frames'],
            not_received=mapping['not_received'], duplicate_instances=mapping['duplicate_instances']))
        offset += n
    destination = output/'post'/method/source['id']; destination.mkdir(parents=True, exist_ok=True)
    with (destination/'frames.npz.pending').open('wb') as f:
        np.savez(f, **{k: np.concatenate([a[k] for a in frame_arrays]) for k in frame_arrays[0]})
        f.flush(); os.fsync(f.fileno())
    os.replace(destination/'frames.npz.pending', destination/'frames.npz')
    value = dict(source=source, method=method, rows=rows, read_errors=errors, status_codes=STATUS_CODES,
                 frame_fields='frame_offset + received_index; source identity/view from rows; time=index/25',
                 evaluated_at=now(), query_access_after_unit_closed=True)
    write_json(destination/'evaluation.json', value)
    return value


def evaluated_or_unknown(output, source, method):
    """One unreadable unit cannot silently remove its rows or stop other sources."""
    try:
        return evaluate_unit(output, source, method)
    except (OSError, ValueError, KeyError, TypeError, EOFError) as error:
        mapping = source_maps(source); rows = []
        for t in tasks(source, method):
            n, k = t['frames'], t['nbits']; m = mapping['conditions'][t['condition']]
            rows.append(dict(t, complete=False, score=None, wrong_query_score=None,
                frame_offset=None, planned_frames=n, planned_bits=n*k, correct=0, wrong=0,
                unknown=n*k, coverage=0., readable_ber=None, complete_messages_correct=0,
                complete_messages_error=0, complete_messages_unknown=n, statuses={'OUTPUT_INVALID': n},
                native_video_correct_bits=None, native_video_ber=None, native_video_exact=None,
                native_video_status='N/A' if method == 'wam' else 'UNKNOWN', localization_iou_mean=None,
                unique_retained_source_frames=m['unique_retained_source_frames'], not_received=m['not_received'],
                duplicate_instances=m['duplicate_instances']))
        value = dict(source=source, method=method, rows=rows, read_errors=[str(error)],
                     physical_retry=False, native_artifacts_retained=True)
        write_json(Path(output)/'post'/method/source['id']/'evaluation_failure.json', value)
        return value


def calibration(output, sources):
    output = Path(output); result = {}
    for method, k in METHODS.items():
        path = output/f'calibration_{method}.json'
        if path.exists():
            result[method] = read_json(path)
            continue
        scores = []
        for source in sources:
            if source['split'] == 'CAL':
                evaluated = evaluated_or_unknown(output, source, method)
                scores.append([r['score'] for r in evaluated['rows']])
        result[method] = calibrate(scores, k)
        write_json(path, result[method])
    return result


def final_evaluation(output, sources):
    output = Path(output)
    calibrated = {m: read_json(output/f'calibration_{m}.json') for m in METHODS}
    test = [s for s in sources if s['split'] == 'TEST']; actions = [s['action'] for s in test]
    comparisons, native, verifications, summaries = [], [], [], []
    for method in METHODS:
        c = calibrated[method]; by_condition = {}; families = {'OFF': [], 'WM': []}
        wrong_families = {'OFF': [], 'WM': []}
        for source in test:
            evaluated = evaluated_or_unknown(output, source, method)
            rows = evaluated['rows']; by_id = {r['id']: r for r in rows}
            events = {'OFF': [], 'WM': []}; wrong_events = {'OFF': [], 'WM': []}
            for r in rows:
                accept = None if c['status'] != 'CALIBRATED' or r['score'] is None else r['score'] > c['tau']
                wrong_accept = None if c['status'] != 'CALIBRATED' or r['wrong_query_score'] is None else r['wrong_query_score'] > c['tau']
                v = dict(source=source['id'], action=source['action'], method=method, arm=r['arm'],
                         channel=r['channel'], condition=r['condition'], score=r['score'], accept=accept,
                         wrong_query_score=r['wrong_query_score'], wrong_query_accept_descriptive=wrong_accept,
                         verification=c['status'], native_complete=r['complete'])
                verifications.append(v); native.append(r)
                key = r['arm'], r['channel'], r['condition']
                by_condition.setdefault(key, []).append(accept)
                events[r['arm']].append(accept); wrong_events[r['arm']].append(wrong_accept)
                identity = by_id[f"{r['arm']}/{r['channel']}/IDENTITY"]
                g1 = by_id[f"{r['arm']}/{CHANNELS[0]}/{r['condition']}"]
                diff = lambda a, b: None if a is None or b is None else a-b
                comparisons.append(dict(source=source['id'], method=method, task=r['id'],
                    ber_minus_identity=diff(r['readable_ber'], identity['readable_ber']),
                    coverage_minus_identity=r['coverage']-identity['coverage'],
                    g2_minus_g1_ber=diff(r['readable_ber'], g1['readable_ber']) if r['channel'] == CHANNELS[1] else None))
            for arm in families:
                families[arm].append(family_event(events[arm]))
                wrong_families[arm].append(family_event(wrong_events[arm]))
        for (arm, channel, condition), events in by_condition.items():
            summaries.append(dict(method=method, arm=arm, channel=channel, condition=condition,
                measure='FPR' if arm == 'OFF' else 'TPR', verification=calibrated[method]['status'],
                **rate_summary(events), action_sensitivity=action_sensitivity(events, actions)))
        for arm, events in families.items():
            summaries.append(dict(method=method, arm=arm, channel='FAMILY_22', condition='ANY_VIEW',
                measure='FPR' if arm == 'OFF' else 'TPR', verification=calibrated[method]['status'],
                **rate_summary(events), action_sensitivity=action_sensitivity(events, actions),
                wrong_query_descriptive=rate_summary(wrong_families[arm])))
    for filename, rows in [('verification.csv', verifications), ('native_messages.csv', native), ('comparisons.csv', comparisons)]:
        with (output/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = dict(status='SAVED_OUTPUT_EVALUATION', calibration=calibrated, summaries=summaries,
        scientific_pass=None, unsupported=dict(wam_video_message='N/A', source_time='N/A', gap='N/A',
        repeat='N/A', joint_identity_time='N/A'),
        evidence_scope='UCF101 external natural-video, one preassigned full query; native 32/256 bits/support differ',
        caveats=['nominal marginal 1% family target, not population certification',
                 'group split reduces dependence; no proof of absolute independence or pretraining disjointness',
                 'not Wan same-source comparison; no presence/open-set identity or long-video claim',
                 'wrong-query results descriptive only; not separately calibrated'])
    write_json(output/'evaluation.json', result)
    return result
