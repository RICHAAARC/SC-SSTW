"""One fixed key-derived 11-window marker; blind time search and existence only."""
from __future__ import annotations
import hashlib
import math
import numpy as np
from . import projection_margin as carrier, state_clock

PROTOCOL_ID = 'SC-SSTW-Fixed-Key-11Window-V1'
FIXED_VIEWS = ('FULL', 'DELETE90', 'SPEED5_4')
WINDOWS = 11
SUPPORTS = 160
REFERENCE_PATHS = {
    'IDENTITY': dict(g=0, scale=[1, 1], offset=0, boundary=11, delta=0),
    'DELETE90_REFERENCE': dict(g=0, scale=[1, 1], offset=0, boundary=6, delta=1),
    'SPEED5_4_REFERENCE': dict(g=0, scale=[5, 4], offset=0, boundary=11, delta=0),
}


def key_identifier(key: bytes) -> str:
    return hashlib.sha256(b'SC-SSTW/fixed-key-v1\0' + key).hexdigest()[:16]


def codebook(key: bytes) -> dict:
    base = carrier.codebook(key)
    states, steps, _ = state_clock.trajectory(key, 0)
    polarity = np.array([carrier.sign(key, 'state_clock/polarity/' + state_clock.CONTEXT_DIGEST, i) for i in range(1760)])
    code = states[np.arange(1760) // 160, np.arange(1760) % 2] * polarity * base['sync']
    return dict(directions=base['directions'], sync=base['sync'], polarity=polarity,
                code=code, states=states, steps=np.asarray(steps),
                receiver_protocol_id=PROTOCOL_ID, key_id=key_identifier(key))


def torch_projections(z, directions):
    if tuple(z.shape) != carrier.SHAPE:
        raise ValueError('fixed marker latent shape mismatch')
    x = z[0, :, 1:45].permute(1, 0, 2, 3)
    blocks = x.reshape(11, 4, 16, 10, 4, 16, 4).permute(0, 3, 5, 1, 2, 4, 6).reshape(1760, 1024)
    return (blocks.double() * directions.double()).sum(dim=1)


def loss(z, book):
    import torch
    p = torch_projections(z, torch.as_tensor(book['directions'], device=z.device))
    code = torch.as_tensor(book['code'], dtype=torch.float64, device=z.device)
    return -(p.tanh() * code).mean()


def nominal_record(z, book, *, include_state=False):
    import torch
    with torch.no_grad():
        p = torch_projections(z.detach().cpu(), torch.as_tensor(book['directions'])).numpy()
    clipped = np.clip(p, -1, 1) * book['code']
    tanh = np.tanh(p) * book['code']
    result = dict(clipped_score=float(clipped.mean()), tanh_score=float(tanh.mean()),
                clipped_window_evidence=clipped.reshape(11, 160).mean(axis=1).tolist(),
                tanh_window_evidence=tanh.reshape(11, 160).mean(axis=1).tolist(),
                meaning='fixed nominal latent marker evidence, not calibrated video detection')
    if include_state:
        result['identity_state'] = score_path({0: z.detach().cpu().numpy()}, book, REFERENCE_PATHS['IDENTITY'])
    return result


def emission(observations, book, window, g, num, den, offset):
    z = observations.get(g)
    selected = [None] * 4 if z is None else carrier.allocation(z.shape[2] - 1, g, num, den, offset)[4*window:4*window+4]
    groups = sum(j is not None for j in selected)
    base = dict(window=window, phase=g, selected=selected, group_count=groups,
                valid=groups >= 3, observed_components=160 * groups if groups >= 3 else 0)
    if groups < 3:
        return dict(base, support_kind='INVALID', signed_evidence=None, q=[0., 0.], q_norm=0.)
    sl = slice(window*160, (window+1)*160)
    if groups == 4:
        data = np.take(z[0], [j+1 for j in selected], axis=1).transpose(1, 0, 2, 3)
        data = data.reshape(4, 16, 10, 4, 16, 4).transpose(2, 4, 0, 1, 3, 5).reshape(160, 1024)
        projection = np.einsum('ij,ij->i', data.astype(np.float64), book['directions'][sl].astype(np.float64))
    else:
        directions = book['directions'][sl].astype(np.float64).reshape(160, 4, 256)
        projection = np.zeros(160, dtype=np.float64)
        for slot, j in enumerate(selected):
            if j is None:
                continue
            component = z[0, :, j+1].reshape(16, 10, 4, 16, 4).transpose(1, 3, 0, 2, 4).reshape(160, 256)
            projection += np.einsum('ij,ij->i', component.astype(np.float64), directions[:, slot])
    clipped = np.clip(projection, -1., 1.)
    decoded = clipped * book['sync'][sl] * book['polarity'][sl]
    q = [float(decoded[::2].mean()), float(decoded[1::2].mean())]
    return dict(base, support_kind='FULL' if groups == 4 else 'PARTIAL3',
                signed_evidence=float(np.mean(clipped * book['code'][sl])), q=q,
                q_norm=float(np.linalg.norm(q)), projection_rms=float(np.sqrt(np.mean(projection**2))),
                clipped_fraction=float(np.mean(np.abs(projection) >= 1)))


def score_path(observations, book, path, cache=None):
    cache = {} if cache is None else cache
    windows = []
    for window in range(11):
        delta = path['delta'] if window >= path['boundary'] else 0
        g = (path['g'] - delta) % 4
        args = (window, g, *path['scale'], path['offset'] + delta)
        if args not in cache:
            cache[args] = emission(observations, book, *args)
        windows.append(cache[args])
    valid = [w['valid'] for w in windows]
    weights = [w['observed_components'] for w in windows if w['valid']]
    if not weights:
        return dict(status='INVALID', path=dict(path), windows=windows, score=None)
    matched = float(np.average([w['signed_evidence'] for w in windows if w['valid']], weights=weights))
    observer = state_clock.observe(np.asarray([w['q'] for w in windows]), valid, book['states'], book['steps'])
    penalty = state_clock.EDIT_COST if path['delta'] else 0.
    score = matched - state_clock.INNOVATION_WEIGHT * observer['innovation_mean'] - penalty
    return dict(status='SCORED' if math.isfinite(score) else 'INVALID', path=dict(path),
                score=score, matched_score=matched, state_innovation_mean=observer['innovation_mean'],
                innovation_by_window=observer['innovation_by_window'], event_cost=penalty,
                windows=windows, matched_windows=[i for i, v in enumerate(valid) if v],
                matched_components=sum(weights), full_blocks=160*sum(w['group_count'] == 4 for w in windows),
                partial_blocks=160*sum(w['group_count'] == 3 for w in windows))


def _rank(row):
    p = row['path']
    return (-row['score'], abs(p['delta']), abs(p['offset']), abs(p['scale'][0]/p['scale'][1]-1), p['g'], p['boundary'], p['delta'], p['offset'], *p['scale'])


def read(observations, book):
    invalid = [g for g, z in observations.items() if z.ndim != 5 or z.shape[:2] != (1, 16) or z.shape[3:] != (40, 64) or not np.isfinite(z).all()]
    if invalid:
        return dict(status='INVALID', reason='invalid or nonfinite observation', invalid_phases=invalid,
                    best=None, existence_statistic=None, receiver_protocol_id=PROTOCOL_ID, key_id=book['key_id'])
    cache, best, attempted, scored = {}, None, 0, 0
    for path in state_clock.clock_paths():
        attempted += 1
        row = score_path(observations, book, path, cache)
        if row['status'] == 'SCORED':
            scored += 1
            if best is None or _rank(row) < _rank(best):
                best = row
    # Fixed diagnostics are computed after blind maximization, and never enter its score.
    diagnostics = {name: score_path(observations, book, path, cache) for name, path in REFERENCE_PATHS.items()}
    return dict(status='SCORED' if best is not None else 'INVALID', best=best,
                existence_statistic=None if best is None else best['score'],
                attempted_path_count=attempted, scored_path_count=scored, template_count=1,
                receiver_protocol_id=PROTOCOL_ID, key_id=book['key_id'],
                fixed_path_diagnostics=diagnostics)


def source_max_statistic(detections):
    missing = [v for v in FIXED_VIEWS if detections.get(v, {}).get('status') != 'SCORED']
    binding = {(detections[v].get('receiver_protocol_id'), detections[v].get('key_id')) for v in FIXED_VIEWS if v in detections}
    if missing or len(binding) != 1:
        return dict(status='INVALID', statistic=None, winning_view=None, missing_or_failed_views=missing)
    if any(not math.isfinite(detections[v]['existence_statistic']) for v in FIXED_VIEWS):
        return dict(status='INVALID', statistic=None, winning_view=None, reason='nonfinite fixed family')
    winner = max(FIXED_VIEWS, key=lambda v: (detections[v]['existence_statistic'], v))
    pid, kid = next(iter(binding))
    return dict(status='SCORED', statistic=detections[winner]['existence_statistic'], winning_view=winner,
                receiver_protocol_id=pid, key_id=kid)


def freeze_calibration(source_rows, guard=1e-6, *, protocol_id=PROTOCOL_ID, key_id=None):
    ok = len(source_rows) == 2 and all(r.get('status') == 'SCORED' and r.get('receiver_protocol_id') == protocol_id and r.get('key_id') == key_id and math.isfinite(r.get('statistic', math.nan)) for r in source_rows)
    if not ok:
        return dict(status='UNCALIBRATED', threshold=None, sources=source_rows, receiver_protocol_id=protocol_id, key_id=key_id)
    if not math.isfinite(guard) or guard <= 0:
        raise ValueError('positive finite calibration guard required')
    return dict(status='FROZEN', threshold=max(r['statistic'] for r in source_rows)+guard, guard=guard,
                sources=source_rows, receiver_protocol_id=protocol_id, key_id=key_id,
                source_count=2, empirical_rank_resolution='1/3', claim='functional independent-OFF calibration only; no low-FPR estimate')


def decide(detection, calibration):
    if detection.get('status') != 'SCORED':
        return dict(status='INVALID', detected=None)
    if not calibration or calibration.get('status') != 'FROZEN' or any(calibration.get(k) != detection.get(k) for k in ('receiver_protocol_id', 'key_id')):
        return dict(status='UNCALIBRATED', detected=None)
    score, threshold = detection.get('existence_statistic'), calibration.get('threshold')
    if not isinstance(score, (float, int)) or not isinstance(threshold, (float, int)) or not math.isfinite(score) or not math.isfinite(threshold):
        return dict(status='INVALID', detected=None)
    return dict(status='DETECTED' if score > threshold else 'REJECTED', detected=score > threshold,
                statistic=score, threshold=threshold)


def alignment_report(detection, view):
    """Post-search allocation diagnostics. They do not change score or decision."""
    name = {'FULL':'IDENTITY', 'DELETE90':'DELETE90_REFERENCE', 'SPEED5_4':'SPEED5_4_REFERENCE'}[view]
    reference = detection.get('fixed_path_diagnostics', {}).get(name)
    best = detection.get('best')
    if not best or not reference:
        return dict(status='UNAVAILABLE', reference=name)
    excluded = [5] if view == 'DELETE90' else []
    rows = []
    for a, b in zip(best['windows'], reference['windows']):
        index = a['window']
        pairs = [(a['phase']+2.5+4*x, b['phase']+2.5+4*y) for x,y in zip(a['selected'], b['selected']) if x is not None and y is not None]
        comparable = a['valid'] and b['valid'] and index not in excluded
        rows.append(dict(window=index, excluded_event_window=index in excluded, comparable=comparable,
                         reference_eligible=b['valid'] and index not in excluded, best_missing=not a['valid'],
                         best_groups=a['group_count'], reference_groups=b['group_count'],
                         exact_allocation_match=comparable and a['phase']==b['phase'] and a['selected']==b['selected'],
                         mean_abs_received_center_difference=None if not comparable or not pairs else float(np.mean([abs(x-y) for x,y in pairs]))))
    return dict(status='REPORTED', reference=name, windows=rows, excluded_from_alignment_only=excluded,
                reference_eligible_windows=sum(r['reference_eligible'] for r in rows),
                best_missing_reference_windows=[r['window'] for r in rows if r['reference_eligible'] and r['best_missing']],
                best_missing_reference_count=sum(r['reference_eligible'] and r['best_missing'] for r in rows),
                comparable_windows=sum(r['comparable'] for r in rows), exact_allocation_matches=sum(r['exact_allocation_match'] for r in rows),
                nominal_scale_match=best['path']['scale']==reference['path']['scale'],
                meaning='allocation agreement with a predeclared group-level reference; speed scale match is not exact video-frame synchronization; event exclusion never affects scoring')
