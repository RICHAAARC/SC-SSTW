"""Card89 E41 physical observations and blind four-phase logical binding."""
from collections import Counter
import math
import torch
from . import b_legacy_position_margin_v1 as carrier
from . import b_phase_centered_saved_v1 as centered

KEY_IDS = carrier.KEY_IDS
KEYS = carrier.KEYS
LOCAL_SHAPE = (1, 16, 11, 40, 64)


def validate_local(z):
    if not isinstance(z, torch.Tensor) or tuple(z.shape) != LOCAL_SHAPE or z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
        raise ValueError('finite normalized FP32 E41 latent[1,16,11,40,64] required')


def observe(z, book):
    validate_local(z)
    value = z.detach().cpu().double()
    roi = torch.empty((10, 4, 44), dtype=torch.float64)
    for k, (y0, y1, x0, x1) in enumerate(carrier.ROIS):
        patches = value[0, 4:16, 1:11, y0:y1, x0:x1].permute(1, 0, 2, 3).reshape(10, 768)
        roi[:, k] = (patches @ book['directions'][k].reshape(44, 768).T) * book['signs'][k]
    if not bool(torch.isfinite(roi).all()):
        raise ValueError('nonfinite new ROI Q')
    return dict(status='NEW_E41_OBSERVED', roi_q=roi.tolist(), pooled_q=roi.mean(1).tolist(),
        retained_rows=list(range(1, 11)), truth_inputs=False)


def spectrum(z):
    validate_local(z)
    return torch.fft.fft2(z[0, :4, 1:11].float(), dim=(-2, -1), norm='ortho').real


def read_spectrum(value, key):
    if tuple(value.shape) != (4, 10, 40, 64) or not bool(torch.isfinite(value).all()):
        raise ValueError('finite shared E41 spectrum[4,10,40,64] required')
    coords = carrier.grow.coordinates(key)
    x = value[:, :, [c[0] for c in coords], [c[1] for c in coords]]
    signed = (2*(x > 0).to(torch.int8)-1).cpu().tolist()
    zeros = (x == 0).to(torch.int8).cpu().tolist()
    cells = []
    for r in range(10):
        bits = []
        for b in range(8):
            seq = [int(signed[c][r][k] > 0) for c in range(4) for k in range(b, 240, 8)]
            bits.append(dict(slot=b, decoded=Counter(seq).most_common(1)[0][0], ones=sum(seq),
                count=120, tie=sum(seq)*2 == 120,
                zero_coefficients=sum(zeros[c][r][k] for c in range(4) for k in range(b, 240, 8))))
        cells.append(dict(local_row=r+1, bits=bits))
    return dict(status='NEW_E41_RAW', signed_votes=signed, zero_coefficient_mask=zeros,
        local_cells=cells, raw_bit_slots=80, signed_vote_slots=9600,
        truth_inputs=False, message_inputs=False)


def phase_starts(p):
    return range(1 if p == 0 else 2, 36)


def bind_family(phase_tables, references):
    """Only own received input/key's four views; no origin, role or truth arguments."""
    rows = []
    missing = []
    for p in range(4):
        table = phase_tables[p]
        if table['status'] != 'NUMERIC_COMPLETE_UNCALIBRATED':
            missing.append(dict(phase=p, observation=references[p], status=table['status'],
                unavailable_from=table.get('unavailable_from')))
        scores = {x['source_start']:x for x in table['scores']}
        payloads = {x['source_start']:x for x in table['candidates']}
        for s in phase_starts(p):
            score = scores[s]
            payload = payloads[s]
            rows.append(dict(phase=p, source_start=s, public_origin_hypothesis=4*(s-1)-p,
                score_status=score['status'], payload_status=payload['status'],
                raw_pooled=score['raw_pooled'], centered_pooled=score['centered_pooled'], centered_min=score['centered_min'],
                candidate_ref=dict(observation=references[p], source_start=s),
                score_unavailable_from=score.get('unavailable_from'), payload_unavailable_from=payload.get('unavailable_from')))
    maxima = {}
    for variant in centered.VARIANTS:
        complete_scores = all(x['score_status'] == 'RETURNED' and centered.finite_number(x[variant]) for x in rows)
        maximum = max(x[variant] for x in rows) if complete_scores else None
        ties = [dict(phase=x['phase'], source_start=x['source_start'], candidate_ref=x['candidate_ref'])
                for x in rows if x[variant] == maximum] if complete_scores else None
        bad_payload = [x for x in rows if x['payload_status'] != 'CONDITIONAL_REDUCED_UNCALIBRATED']
        status = 'NUMERIC_UNAVAILABLE' if not complete_scores else 'PAYLOAD_NUMERIC_UNAVAILABLE' if bad_payload else 'INCOMPLETE_PHASE_FAMILY' if missing else 'COMPLETE'
        maxima[variant] = dict(status=status, maximum=maximum, exact_max_ties=ties,
            singleton_conditional_ref=ties[0] if status == 'COMPLETE' and len(ties) == 1 else None,
            authenticated=False, threshold=None)
    return dict(status='COMPLETE_UNCALIBRATED' if not missing and all(x['status'] == 'COMPLETE' for x in maxima.values()) else 'INCOMPLETE_PHASE_FAMILY',
        hypothesis_denominator=137, phase_views=4, candidates=rows, maxima=maxima,
        incomplete_phases=missing, truth_inputs=False, message_inputs=False,
        identity_status='UNCALIBRATED', phase_status='UNCALIBRATED', trusted_message=None,
        accepted_count=None, rejected_count=None, FPR=None)
