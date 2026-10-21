"""Card94 fixed host-family calibration and blind set-valued reading (stdlib only)."""
import math
from . import b_phase_centered_saved_v1 as arithmetic

ORIGINS = (0, 4, 17, 62, 107, 136)
ARMS = ('OFF', 'P0', 'PM')
KEY_IDS = ('key00', 'key01')
H = tuple((p, s) for p in range(4) for s in range(1 if p == 0 else 2, 36))


def query_metadata():
    return {f'q{i:02}': dict(arm=a, origin=o, key=k)
            for i, (a, o, k) in enumerate((a, o, k) for a in ARMS for o in ORIGINS for k in KEY_IDS)}


def cells(s):
    return [dict(local_row=r, source_cell=s+r-1, group=(s+r-2)//11) for r in range(1,11)]


def empty_family(reason='NOT_ENTERED'):
    return dict(status=reason, candidates=[dict(phase=p, source_start=s, T=None,
        score_status=reason, payload_status=reason, cells=cells(s),
        decoded_bits=[None]*32) for p, s in H], phase_local_bits={},
        hypothesis_denominator=137, truth_inputs=False, message_inputs=False)


def bind(tables, local_bits=None):
    """Accept only own four phase tables; no origin/arm/source/truth inputs."""
    out = empty_family('INCOMPLETE_PHASE_FAMILY')
    out['phase_local_bits'] = local_bits or {}
    for row in out['candidates']:
        p, s = row['phase'], row['source_start']
        table = tables.get(p, {})
        scores = [x for x in table.get('scores', []) if x.get('source_start') == s]
        payloads = [x for x in table.get('candidates', []) if x.get('source_start') == s]
        if len(scores) == len(payloads) == 1:
            score, payload = scores[0], payloads[0]
            row.update(T=score.get('centered_min'), F=score.get('F'), V=score.get('V'),
                raw_pooled=score.get('raw_pooled'), centered_pooled=score.get('centered_pooled'),
                score_status=score.get('status'), payload_status=payload.get('status'),
                decoded_bits=payload.get('decoded_bits'), cells=payload.get('cells'))
            if table.get('score_call_status', 'RETURNED') != 'RETURNED':
                row['score_status'] = table['score_call_status']
            if table.get('payload_call_status', 'RETURNED') != 'RETURNED':
                row['payload_status'] = table['payload_call_status']
    if valid_family(out):
        out['status'] = 'COMPLETE'
    return out


def score_complete(family):
    rows = family.get('candidates', [])
    if [(x.get('phase'), x.get('source_start')) for x in rows] != list(H):
        return False
    for row in rows:
        if row.get('score_status') != 'RETURNED' or not arithmetic.finite_number(row.get('T')):
            return False
    return True


def valid_family(family):
    if not score_complete(family):
        return False
    for row in family['candidates']:
        if row.get('payload_status') != 'CONDITIONAL_REDUCED_UNCALIBRATED':
            return False
        expected = cells(row['source_start'])
        if row.get('cells') != expected:
            return False
        bits = row.get('decoded_bits')
        if not isinstance(bits, list) or len(bits) != 32:
            return False
        groups = {x['group'] for x in expected}
        if any((type(v) is not int or v not in (0, 1)) if i//8 in groups else v is not None
               for i, v in enumerate(bits)):
            return False
    return True


def _output(status):
    return dict(status=status, detection='UNRESOLVED', location='UNRESOLVED', maximum=None,
        exact_max_ties=[], candidate_set=[], location_hypothesis=None,
        message=[dict(slot=i, decoded=None, status='TECHNICAL_UNRESOLVED') for i in range(32)],
        authenticated=False, scientific_pass=None)


def from_set(family, selected, status):
    out = _output(status)
    rows = family['candidates']
    out['maximum'] = max(x['T'] for x in rows)
    out['exact_max_ties'] = [dict(phase=x['phase'], source_start=x['source_start'])
                            for x in rows if x['T'] == out['maximum']]
    out['candidate_set'] = [dict(phase=x['phase'], source_start=x['source_start']) for x in selected]
    if not selected:
        return out
    singleton = len(selected) == 1
    out.update(detection='DETECTED_PROVISIONAL',
        location='ACCEPT_LOCATION_PROVISIONAL' if singleton else 'UNRESOLVED_LOCATION')
    if singleton:
        x = selected[0]
        out['location_hypothesis'] = dict(phase=x['phase'], source_start=x['source_start'],
            origin=4*(x['source_start']-1)-x['phase'])
    for i, item in enumerate(out['message']):
        supported = all(any(c['group'] == i//8 for c in x['cells']) for x in selected)
        values = [x['decoded_bits'][i] for x in selected]
        agree = supported and all(type(v) is int and v in (0, 1) for v in values) and len(set(values)) == 1
        item.update(decoded=values[0] if agree else None,
            status=('CONDITIONAL_GROUP_READ' if singleton else 'CONDITIONAL_GROUP_READ_WITH_LOCATION_UNRESOLVED')
                   if agree else 'ERASE_AMBIGUOUS', all_candidates_support=supported)
    return out


def decide(family, thresholds):
    if not score_complete(family):
        return _output('CALIBRATION_UNRESOLVED' if thresholds.get('status') != 'FROZEN' else 'UNRESOLVED_TECHNICAL')
    maximum = max(x['T'] for x in family['candidates'])
    ties = [dict(phase=x['phase'], source_start=x['source_start']) for x in family['candidates'] if x['T'] == maximum]
    numeric_thresholds = thresholds.get('status') == 'FROZEN' and all(arithmetic.finite_number(thresholds.get(k)) for k in ('tau','lambda')) and thresholds['tau'] >= 0 and thresholds['lambda'] >= 0
    if not valid_family(family):
        out = _output('PAYLOAD_UNRESOLVED')
        out.update(maximum=maximum, exact_max_ties=ties)
        if numeric_thresholds:
            out['score_only_detection'] = 'DETECTED_PROVISIONAL' if maximum > thresholds['tau'] else 'REJECT_SYNC'
        return out
    if not numeric_thresholds:
        out = _output('CALIBRATION_UNRESOLVED')
        out.update(maximum=maximum, exact_max_ties=ties)
        return out
    if maximum <= thresholds['tau']:
        out = from_set(family, [], 'REJECT_SYNC')
        out['detection'] = 'REJECT_SYNC'
        out['location'] = 'NOT_DETECTED'
        for row in out['message']:
            row['status'] = 'REJECT_ERASURE'
        return out
    # Do not filter T(h)>tau: the calibrated gap set must retain its full boundary.
    selected = [x for x in family['candidates'] if maximum-x['T'] <= thresholds['lambda']]
    return from_set(family, selected, 'DECIDED_PROVISIONAL') if selected else _output('UNRESOLVED_EMPTY_SET')


def forced_min(family):
    if family.get('status') != 'COMPLETE' or not valid_family(family):
        return _output('UNRESOLVED_TECHNICAL')
    maximum = max(x['T'] for x in family['candidates'])
    selected = [x for x in family['candidates'] if x['T'] == maximum]
    out = from_set(family, selected, 'UNAUTHENTICATED_FORCED_MIN_REFERENCE')
    if len(selected) != 1:
        out['location_hypothesis'] = None
        for row in out['message']:
            row.update(decoded=None, status='EXACT_TIE_UNRESOLVED')
    return out


def calibrate(host_families, metadata):
    """Supervised calibration AFTER saved blind completion; no message argument."""
    result = dict(status='CALIBRATION_UNRESOLVED', tau=None, **{'lambda': None}, hosts={},
        scope='fixed-roster empirical; no population or conditional confidence guarantee', message_used=False)
    if set(host_families) != {'b93-c01', 'b93-c02'} or metadata != query_metadata():
        return result
    for host, families in host_families.items():
        if set(families) != set(metadata) or any(f.get('status') != 'COMPLETE' or not valid_family(f) for f in families.values()):
            return result
        nulls, gaps = [], []
        for qid, meta in metadata.items():
            family = families[qid]
            maximum = max(x['T'] for x in family['candidates'])
            if meta['arm'] == 'PM' and meta['key'] == 'key00':
                p = (-meta['origin']) % 4
                s = (meta['origin']+p)//4+1
                true = next(x for x in family['candidates'] if (x['phase'], x['source_start']) == (p, s))
                gaps.append(dict(query=qid, gap=maximum-true['T'], true_phase=p, true_source_start=s))
            else:
                nulls.append(dict(query=qid, maximum=maximum))
        if len(nulls) != 30 or len(gaps) != 6:
            return result
        result['hosts'][host] = dict(N=max(x['maximum'] for x in nulls), A=max(x['gap'] for x in gaps), nulls=nulls, positives=gaps)
    result.update(status='FROZEN', tau=max(0, *(v['N'] for v in result['hosts'].values())))
    result['lambda'] = max(v['A'] for v in result['hosts'].values())
    return result
