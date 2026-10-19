"""Adopted Card57 receiver. No truth, edit names, writer state or runtime inputs."""
from collections import Counter
import math
import numpy as np
import torch
from main.tube_state.local_segmented_terminal_sync_v1 import public_codebook, KEYS, ROIS
from main.tube_state.grow_video_reference import coordinates


def phase_plan(n, p):
    if not 8 <= n <= 181 or p not in range(4):
        raise ValueError("public source181, four phases and at least eight received frames required")
    length = 1 + 4*((n+p-1)//4)
    return dict(received_frames=n, phase=p, encoded_frames=length,
                latent_times=(length-1)//4+1, synthetic_prefix=p,
                tail_discarded=n+p-length,
                received_indices=[0]*p + list(range(length-p)))


def prepare_phase(rgb, p):
    plan = phase_plan(len(rgb), p)
    return rgb[plan['received_indices']], plan


def bank_address(a):
    p = (1-a) % 4
    return p, (a+p+3)//4


def validate_bank(z, n, p):
    shape = (1, 16, phase_plan(n, p)['latent_times'], 40, 64)
    if not isinstance(z, torch.Tensor) or tuple(z.shape) != shape or not z.is_floating_point() or not bool(torch.isfinite(z).all()):
        raise ValueError(f"finite normalized phase latent {shape} required")


def pair_observations(z, n, p, book):
    validate_bank(z, n, p)
    directions = book['directions']
    if directions.shape != (4, 22, 12, 2, 8, 8):
        raise ValueError('original 22 paired pilot directions required')
    starts = [a for a in range(n-7) if bank_address(a)[0] == p]
    raw = np.empty((len(starts), 4, 22), dtype=np.float64)
    z = z.detach().cpu().double()
    for i, a in enumerate(starts):
        _, r = bank_address(a)
        for roi, (y0, y1, x0, x1) in enumerate(ROIS):
            raw[i, roi] = (directions[roi].reshape(22, -1) @
                z[0, 4:16, r:r+2, y0:y1, x0:x1].reshape(-1)).numpy()
    pooled = raw.mean(axis=1, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite paired pilot')
    return dict(starts=np.array(starts, dtype=np.int16), roi_scores=raw, pooled_scores=pooled)


def combine_pairs(observations, n):
    if set(observations) != set(range(4)):
        raise ValueError('all four phase pilot observations required')
    q = np.full((n-7, 22), np.nan, dtype=np.float64)
    seen = []
    for item in observations.values():
        q[item['starts']] = item['pooled_scores']
        seen.extend(item['starts'].tolist())
    if sorted(seen) != list(range(n-7)) or not np.isfinite(q).all():
        raise ValueError('missing, duplicate or invalid received pilot windows')
    return q


def path_parameters(n):
    m = 181-n
    for s in range(m+1):
        yield s, 0, 0
    for d in range(1, m+1):
        for s in range(m-d+1):
            for j in range(1, n):
                yield s, d, j


def path_frames(n, parameters):
    s, d, j = map(int, parameters)
    return [s+i+(d if d and i >= j else 0) for i in range(n)]


def pilot_terms(n, parameters):
    """Every complete source block, in increasing source k; no selective support."""
    s, d, j = map(int, parameters)
    terms = []
    for k in range(22):
        b = 8*k+1
        a = b-s
        if 0 <= a and a+7 < (j if d else n):
            terms.append((a, k))
        if d:
            a = b-s-d
            if j <= a and a+7 < n:
                terms.append((a, k))
    return terms


def unavailable_path(n, status='INCOMPLETE'):
    return dict(status=status, received_frames=n, truth_used=False, message_used=False,
                maximum_count=0, frame_consensus=[None]*n, maximum_score=None,
                complete_phase_observations=False, candidates_evaluated=0)


def score_maxima(raw_sums, divisor):
    scores = np.asarray(raw_sums, dtype=np.float64)/divisor
    maximum = float(scores.max())
    return scores, maximum, np.flatnonzero(scores == maximum)


def select_path(q, n):
    """Normative float64 fsum/divisor, exhaustive finite family, exact ALL maxima.

    Integer interval filtering reuses the list of possible terms for an offset.
    It never uses a floating prefix sum, approximate tie or top-K truncation.
    """
    q = np.asarray(q)
    if q.dtype != np.float64 or q.shape != (n-7, 22) or not np.isfinite(q).all():
        raise ValueError('complete finite float64 Q required')
    m = 181-n
    count = m+1+(n-1)*m*(m+1)//2
    params = np.empty((count, 3), dtype=np.int16)
    sums = np.empty(count, dtype=np.float64)
    support = np.empty(count, dtype=np.int16)
    offsets = {s: [(8*k+1-s, k, float(q[8*k+1-s, k])) for k in range(22)
                   if 0 <= 8*k+1-s <= n-8] for s in range(m+1)}
    for idx, (s, d, j) in enumerate(path_parameters(n)):
        terms = ([v for a, k, v in offsets[s] if a+7 < j] +
                 [v for a, k, v in offsets[s+d] if a >= j]) if d else [v for a, k, v in offsets[s]]
        # Before-gap source blocks precede after-gap blocks: this is source-k order.
        params[idx] = s, d, j
        sums[idx] = math.fsum(terms)
        support[idx] = len(terms)
    scores, maximum, ids = score_maxima(sums, n//8)
    maxima = params[ids]
    consensus = []
    for i in range(n):
        values = maxima[:, 0].astype(np.int32)+i+np.where((maxima[:, 1] > 0)&(i >= maxima[:, 2]), maxima[:, 1], 0)
        consensus.append(int(values[0]) if np.all(values == values[0]) else None)
    result = dict(status='READ' if len(ids) == 1 else 'UNRESOLVED', received_frames=n,
        truth_used=False, message_used=False, complete_phase_observations=True,
        candidates_evaluated=count, common_denominator=n//8, maximum_score=maximum,
        maximum_count=len(ids), frame_consensus=consensus,
        minimum_support=int(support.min()), maximum_support=int(support.max()),
        maximum_support_counts=sorted(set(map(int, support[ids]))),
        candidate_columns=['source_start', 'gap_length', 'received_gap_position'],
        raw_candidate_file='candidates.npz', full_maximum_set_field='maximum_indices')
    # Exact integer occupancy only; no numerical scoring shortcut.
    s, d, j = (params[:, i].astype(np.int32) for i in range(3))
    left_end = s+np.where(d > 0, j, n)-1
    right_start, right_end = s+j+d, s+n+d-1
    absent = np.zeros(count, dtype=np.int16)
    for k in range(22):
        lo, hi = 8*k+1, 8*k+8
        present = ((hi >= s)&(lo <= left_end)) | ((d > 0)&(hi >= right_start)&(lo <= right_end))
        absent += ~present
    arrays = dict(parameters=params, raw_sums=sums, scores=scores, support_counts=support,
                  partial_state_counts=22-support-absent, absent_state_counts=absent, maximum_indices=ids)
    return result, arrays


def _unanimous(values):
    values = np.asarray(values, dtype=bool)
    return bool(values[0]) if len(values) and np.all(values == values[0]) else 'unknown'


def unit_plan(path_result, maxima=None, *, phase_zero_only=False):
    n = path_result['received_frames']
    path = path_result['frame_consensus']
    units, positions = [], []
    used = set()
    for a in range(n-3):
        p, r = bank_address(a)
        window = path[a:a+4]
        status = 'UNRESOLVED'
        t = None
        if all(v is not None for v in window):
            start = window[0]
            if window == list(range(start, start+4)) and (start+3)%4 == 0:
                t = (start+3)//4
                status = 'MAPPED' if 1 <= t <= 44 else 'GUARD'
            else:
                status = 'NOT_FULL_SOURCE_CELL'
        if phase_zero_only and p != 0:
            status = 'UNALIGNED_OTHER_PHASE'
        if path_result['status'] == 'INCOMPLETE':
            status = 'INCOMPLETE'
        positions.append(dict(received_start=a, status=status, source_t=t))
        if status != 'MAPPED':
            continue
        if t in used:
            raise ValueError('source cell cannot receive duplicated phase votes')
        used.add(t)
        pp = phase_plan(n, p)
        flags = dict(bank_synthetic_prefix=pp['synthetic_prefix'], bank_tail_discarded=pp['tail_discarded'],
                     missing_source_prefix='unknown', after_internal_gap='unknown')
        if maxima is not None and len(maxima):
            flags.update(missing_source_prefix=_unanimous(maxima[:, 0] > 0),
                after_internal_gap=_unanimous((maxima[:, 1] > 0)&(a >= maxima[:, 2])))
        k = (t-1)//2
        pilot_start = 8*k+1
        pilot_a = path.index(pilot_start) if pilot_start in path else None
        full_pilot = pilot_a is not None and path[pilot_a:pilot_a+8] == list(range(pilot_start, pilot_start+8))
        units.append(dict(received_start=a, bank=p, latent_index=r, source_t=t, segment=(t-1)//11,
                          mapping_status='CONSENSUS' if path_result['status'] == 'UNRESOLVED' else 'MAPPED',
                          pilot_geometry='COMPLETE_SOURCE_BLOCK' if full_pilot else 'BOUNDARY_UNIT_INFERRED_FROM_PATH',
                          complete_pilot_received_start=pilot_a if full_pilot else None,
                          context_flags=flags))
    return dict(status=path_result['status'], received_frames=n, units=units, received_positions=positions,
                frame_consensus=path, truth_inputs=False, message_inputs=False,
                phase_zero_only=phase_zero_only, source_cell_denominator=44)


def direct_plan(n, parameters, *, phase_zero_only=False):
    params = np.asarray([parameters], dtype=np.int16)
    return unit_plan(dict(received_frames=n, frame_consensus=path_frames(n, parameters), status='READ'),
                     params, phase_zero_only=phase_zero_only)


def fft_bank(z, n, p):
    validate_bank(z, n, p)
    return torch.fft.fft2(z.detach().cpu().float(), dim=(-2, -1), norm='ortho').real


def extract_votes(spectrum, n, p, key):
    validate_bank(spectrum, n, p)
    starts = [a for a in range(n-3) if bank_address(a)[0] == p]
    times = [bank_address(a)[1] for a in starts]
    coords = coordinates(key)
    h, w = [v[0] for v in coords], [v[1] for v in coords]
    values = torch.stack([spectrum[0, ch, times][:, h, w] for ch in range(4)])
    return dict(starts=np.asarray(starts, dtype=np.int16),
                signed_votes=(2*(values > 0).to(torch.int8)-1).numpy(),
                zero_coefficient_mask=(values == 0).to(torch.int8).numpy())


def unavailable_payload(status='NOT_ENTERED'):
    return dict(status=status, net_payload_bits=32, decoded_bits=[None]*32,
        segments=[dict(segment=g, status=status, cells=[], used_units=[], missing_bank_units=[],
            bits=[dict(bit_index=8*g+b, decoded=None, count=0, status=status) for b in range(8)]) for g in range(4)])


def reduce_votes(plan, banks):
    for p, bank in banks.items():
        shape = (4, len(bank['starts']), 240)
        v, z = bank['signed_votes'], bank['zero_coefficient_mask']
        expected = [a for a in range(plan['received_frames']-3) if bank_address(a)[0] == p]
        if list(bank['starts']) != expected or v.shape != shape or z.shape != shape or v.dtype != np.int8 or z.dtype != np.int8:
            raise ValueError('invalid signed vote bank geometry or dtype')
        if not np.isin(v, [-1, 1]).all() or not np.isin(z, [0, 1]).all() or np.any((z == 1)&(v != -1)):
            raise ValueError('strict-zero signed votes required')
    if plan['status'] == 'INCOMPLETE':
        return unavailable_payload('INCOMPLETE')
    segments, bits = [], [None]*32
    for g in range(4):
        units = sorted([u for u in plan['units'] if u['segment'] == g], key=lambda u: u['received_start'])
        available = [u for u in units if u['bank'] in banks]
        missing = [u for u in units if u['bank'] not in banks]
        status = 'PARTIAL_FAILED' if missing and available else 'FAILED' if missing else 'READ' if units else 'MISSING_OR_NO_CONSENSUS'
        row = dict(segment=g, status=status, cells=units, used_units=available, missing_bank_units=missing, bits=[])
        for slot in range(8):
            seq, zero_count = [], 0
            for ch in range(4):
                for unit in available:
                    bank = banks[unit['bank']]
                    idx = int(np.flatnonzero(bank['starts'] == unit['received_start'])[0])
                    seq.extend((bank['signed_votes'][ch, idx, slot::8] > 0).astype(int).tolist())
                    zero_count += int(bank['zero_coefficient_mask'][ch, idx, slot::8].sum())
            decoded = int(Counter(seq).most_common(1)[0][0]) if seq else None
            ones, count = sum(seq), len(seq)
            item = dict(bit_index=8*g+slot, decoded=decoded, count=count, ones=ones, zeros=count-ones,
                        tie=bool(count and ones*2 == count), zero_coefficients=zero_count, status=status)
            row['bits'].append(item)
            bits[8*g+slot] = decoded
        segments.append(row)
    failed = any(s['missing_bank_units'] for s in segments)
    used = any(s['used_units'] for s in segments)
    return dict(status=('PARTIAL_FAILED' if used else 'FAILED') if failed else 'READ',
                decoded_bits=bits, segments=segments, net_payload_bits=32, truth_inputs=False,
                path_status=plan['status'], vote_order='channel, ascending received unit, coordinate slot::8; Counter encounter ties; strict >0')
