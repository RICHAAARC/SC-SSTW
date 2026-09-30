"""Truth-free fixed-phase, zero/one regular-window-edit pilot path diagnostic."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import math
import numpy as np
from main.tube_state import video_temporal_sync_bridge as carrier


@dataclass(frozen=True)
class PathProtocol:
    source_min: int = 1
    source_max: int = 45
    max_nonunit_edges: int = 1
    tie_atol: float = 1e-12


PUBLIC = PathProtocol()


def path_id(b, event_type, event_i):
    return f"b{b:02d}:{event_type}:{event_i or 0:02d}"


def path_taus(row):
    offset = {'ZERO_EDIT': 0, 'REPEAT': -1, 'SKIP': 1}[row['event_type']]
    return tuple(row['tau1'] + i - 1 + (offset if row['event_i'] is not None and i >= row['event_i'] else 0)
                 for i in range(1, row['R'] + 1))


def canonical_key(row):
    return row['g'], path_taus(row), row['b'], row['event_type'], row['event_i'] or 0


def catalog(length, public=PUBLIC):
    if public != PUBLIC:
        raise ValueError('fixed path protocol required')
    rows = []
    for start in carrier.candidates(length):
        events = [('ZERO_EDIT', None)] + [(kind, i) for kind in ('REPEAT', 'SKIP') for i in range(2, start['R'] + 1)]
        for kind, i in events:
            row = dict(id=path_id(start['b'], kind, i), b=start['b'], g=start['g'], tau1=start['a'] + 1,
                       R=start['R'], event_type=kind, event_i=i)
            ts = path_taus(row)
            valid = min(ts) >= public.source_min and max(ts) <= public.source_max
            row.update(structurally_valid=valid, status='PENDING' if valid else 'STRUCTURALLY_EXCLUDED',
                       exclusion=None if valid else 'FINITE_SOURCE_SUPPORT', class_id=None, score=None)
            rows.append(row)
    return rows


def equivalence_classes(rows):
    groups = {}
    for row in rows:
        if row['structurally_valid']:
            signature = row['g'], tuple(carrier.temporal_sign(t) for t in path_taus(row))
            groups.setdefault(signature, []).append(row)
    result = []
    for (g, signs), members in sorted(groups.items()):
        token = bytes([g] + [int(s > 0) for s in signs])
        cid = hashlib.sha256(token).hexdigest()
        taus = [path_taus(m) for m in members]
        possible = [sorted({ts[i] for ts in taus}) for i in range(len(signs))]
        representative = min(members, key=canonical_key)
        for m in members:
            m['class_id'] = cid
        result.append(dict(id=cid, g=g, emitted_signs=list(signs), member_ids=[m['id'] for m in members],
                           canonical_path_id=representative['id'], b_set=sorted({m['b'] for m in members}),
                           events=[dict(event_type=k, event_indices=sorted({m['event_i'] or 0 for m in members if m['event_type'] == k}))
                                   for k in sorted({m['event_type'] for m in members})],
                           tau_feasible_sets=possible, tau_ranges=[[v[0], v[-1]] for v in possible],
                           contains_zero_edit=any(m['event_type'] == 'ZERO_EDIT' for m in members),
                           status='PENDING', score=None))
    assert len({c['id'] for c in result}) == len(result)
    return result


def _observations(pilot_by_phase, length, key):
    if not isinstance(pilot_by_phase, dict) or any(type(g) is not int for g in pilot_by_phase):
        raise ValueError('only an integer-phase to pilot-matrix mapping is accepted')
    expected = carrier.phases(length)
    if set(pilot_by_phase) - set(expected):
        raise ValueError('unknown phase; metadata/truth is not accepted')
    R = carrier.candidates(length)[0]['R']
    pn = np.asarray(carrier.pilot_layout(key)[1], dtype=np.float64)
    phases = {}
    for g in expected:
        if g not in pilot_by_phase:
            phases[g] = dict(status='MISSING_PHASE', error='required public phase missing')
            continue
        try:
            X = np.asarray(pilot_by_phase[g], dtype=np.float64)
            if X.shape != (R, 64) or not np.isfinite(X).all():
                raise ValueError('pilot matrix must be finite with exact primary R by64 shape')
            with np.errstate(over='raise', invalid='raise'):
                energy = float(np.sum(X * X, dtype=np.float64))
                norm2 = 64 * R * energy
                if not math.isfinite(norm2):
                    raise FloatingPointError('nonfinite normalizer')
                projected = np.sum(X * pn[None, :], axis=1, dtype=np.float64)
                emissions = np.zeros(R, dtype=np.float64) if energy == 0 else projected / math.sqrt(norm2)
                if not np.isfinite(emissions).all():
                    raise FloatingPointError('nonfinite emission')
            phases[g] = dict(status='NO_ENERGY' if energy == 0 else 'SCORED', energy=energy, R=R,
                             normalizer=math.sqrt(norm2), emissions=emissions.tolist())
        except (ValueError, TypeError, OverflowError, FloatingPointError) as exc:
            phases[g] = dict(status='INVALID_PHASE', error=f'{type(exc).__name__}: {exc}')
    return phases


def dynamic_program(phases, length, public=PUBLIC):
    """Complete numeric-optimal predecessor DAG; no edit penalty or truth input."""
    if public != PUBLIC:
        raise ValueError('fixed path protocol required')
    nodes, previous, terminal = {}, {}, []
    R = carrier.candidates(length)[0]['R']
    for start in carrier.candidates(length):
        g, tau = start['g'], start['a'] + 1
        if phases[g]['status'] not in ('SCORED', 'NO_ENERGY'):
            continue
        nid = f'{g}:1:{tau}:0'
        node = dict(id=nid, g=g, i=1, tau=tau, edits=0,
                    score=phases[g]['emissions'][0] * carrier.temporal_sign(tau), predecessors=[], initial_b=start['b'])
        nodes[nid] = node
        previous[(g, tau, 0)] = node
    for i in range(2, R + 1):
        options = {}
        for (g, tau, used), prev in previous.items():
            for delta in (0, 1, 2):
                next_tau, edits = tau + delta, used + int(delta != 1)
                if edits > public.max_nonunit_edges or not public.source_min <= next_tau <= public.source_max:
                    continue
                value = prev['score'] + phases[g]['emissions'][i - 1] * carrier.temporal_sign(next_tau)
                options.setdefault((g, next_tau, edits), []).append((value, prev['id'], delta))
        current = {}
        for (g, tau, used), incoming in sorted(options.items()):
            best = max(x[0] for x in incoming)
            predecessors = [dict(node_id=nid, delta=delta) for score, nid, delta in incoming if best - score <= public.tie_atol]
            nid = f'{g}:{i}:{tau}:{used}'
            node = dict(id=nid, g=g, i=i, tau=tau, edits=used, score=best, predecessors=predecessors, initial_b=None)
            nodes[nid] = node
            current[(g, tau, used)] = node
        previous = current
    if previous:
        best = max(n['score'] for n in previous.values())
        terminal = sorted(n['id'] for n in previous.values() if best - n['score'] <= public.tie_atol)
    else:
        best = None
    traced = set()
    def walk(nid, event_type='ZERO_EDIT', event_i=None):
        node = nodes[nid]
        if node['i'] == 1:
            traced.add(path_id(node['initial_b'], event_type, event_i))
            return
        for pred in node['predecessors']:
            kind = event_type if pred['delta'] == 1 else ('REPEAT' if pred['delta'] == 0 else 'SKIP')
            ei = event_i if pred['delta'] == 1 else node['i']
            walk(pred['node_id'], kind, ei)
    for nid in terminal:
        walk(nid)
    return dict(status='COMPLETE' if all(p['status'] in ('SCORED', 'NO_ENERGY') for p in phases.values()) else 'PARTIAL',
                nodes=list(nodes.values()), optimal_terminal_node_ids=terminal, max_score=best,
                locally_tied_backtrace_path_ids=sorted(traced), tie_atol=public.tie_atol,
                graph_semantics='all locally numerical-optimal predecessors; catalog scores filter cumulative tolerance at final boundary')


def analyze(pilot_by_phase, received_frame_count, key, public=PUBLIC):
    """Only received pilot arrays, length, public key/protocol enter this API."""
    if public != PUBLIC:
        raise ValueError('fixed path protocol required')
    rows = catalog(received_frame_count, public)
    classes = equivalence_classes(rows)
    phases = _observations(pilot_by_phase, received_frame_count, key)
    by_id = {r['id']: r for r in rows}
    for cls in classes:
        phase = phases[cls['g']]
        cls['status'] = phase['status']
        if phase['status'] in ('SCORED', 'NO_ENERGY'):
            cls['score'] = float(np.sum(np.asarray(phase['emissions']) * np.asarray(cls['emitted_signs']), dtype=np.float64))
        for pid in cls['member_ids']:
            by_id[pid].update(status=cls['status'], score=cls['score'])
    dp = dynamic_program(phases, received_frame_count, public)
    valid = [r for r in rows if r['status'] in ('SCORED', 'NO_ENERGY')]
    complete = len(valid) == sum(r['structurally_valid'] for r in rows)
    zero = [r for r in valid if r['event_type'] == 'ZERO_EDIT']
    zero_max = max((r['score'] for r in zero), default=None)
    summary = dict(status='COMPLETE' if complete else 'INCOMPLETE', score_status='UNCALIBRATED_DIAGNOSTIC',
                   accepted_payload=False, truth_used=False, decision_threshold=None, canonical_path_id=None,
                   top_class_ids=[], top_path_ids=[], max_score=None, top_class_gap=None,
                   zero_edit_max_score=zero_max if complete else None, path_minus_zero_edit=None,
                   any_top_class_contains_zero_edit=None, all_top_paths_require_event=None,
                   synchronization_status='INCOMPLETE', unique_sync_claim=False,
                   numerical_winner_is_not_confidence=True, trivial_full=received_frame_count == 181)
    if complete:
        best = max(r['score'] for r in valid)
        top_classes = [c for c in classes if best - c['score'] <= public.tie_atol]
        top = [r for r in valid if best - r['score'] <= public.tie_atol]
        top_ids = {r['id'] for r in top}
        traced = set(dp['locally_tied_backtrace_path_ids'])
        dp['enumeration_max_abs_error'] = abs(best - dp['max_score'])
        dp['final_numerical_top_path_ids'] = sorted(pid for pid in traced if best - by_id[pid]['score'] <= public.tie_atol)
        dp['enumeration_top_paths_match'] = set(dp['final_numerical_top_path_ids']) == top_ids
        if dp['enumeration_max_abs_error'] > public.tie_atol or not dp['enumeration_top_paths_match']:
            summary.update(status='INCOMPLETE', synchronization_status='DP_ENUMERATION_DISAGREEMENT')
        else:
            canonical = min(top, key=canonical_key)
            sorted_classes = sorted(classes, key=lambda c: -c['score'])
            structural = any(len(c['member_ids']) > 1 for c in top_classes)
            no_energy = all(p['status'] == 'NO_ENERGY' for p in phases.values())
            sync = ('NO_ENERGY' if no_energy else 'STRUCTURALLY_AMBIGUOUS' if structural else
                    'NUMERICAL_CLASS_TIE' if len(top_classes) > 1 else 'UNIQUE_NUMERICAL_PATH_UNCALIBRATED')
            summary.update(canonical_path_id=canonical['id'], canonical_event_type=canonical['event_type'],
                           canonical_event_i=canonical['event_i'], top_class_ids=[c['id'] for c in top_classes],
                           top_path_ids=sorted(top_ids), max_score=best,
                           top_class_gap=best - sorted_classes[1]['score'] if len(sorted_classes) > 1 else None,
                           path_minus_zero_edit=best - zero_max,
                           any_top_class_contains_zero_edit=any(c['contains_zero_edit'] for c in top_classes),
                           all_top_paths_require_event=all(r['event_type'] != 'ZERO_EDIT' for r in top),
                           synchronization_status=sync, structural_ambiguity=structural,
                           top_distinct_class_count=len(top_classes))
    return dict(received_frame_count=received_frame_count, summary=summary, phases=phases,
                catalog=rows, equivalence_classes=classes, dp=dp,
                counts=dict(catalog=len(rows), structurally_excluded=sum(not r['structurally_valid'] for r in rows),
                            scorable=sum(r['structurally_valid'] for r in rows), scored=len(valid),
                            zero_edit=sum(r['event_type'] == 'ZERO_EDIT' for r in rows), equivalence_classes=len(classes)))
