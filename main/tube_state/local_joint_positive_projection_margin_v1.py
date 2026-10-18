"""Fixed host-aware terminal projection and message-free clean-grid receiver."""
from __future__ import annotations

import hashlib
import hmac
import math

import torch

LATENT_SHAPE = (1, 16, 46, 40, 64)
VIDEO_SHAPE = (181, 320, 512, 3)
ROIS = ((8, 16, 12, 20), (8, 16, 44, 52), (28, 36, 12, 20), (28, 36, 44, 52))
STATES = tuple(range(22))
ARMS = ("PAYLOAD_ONLY", "PROJECTION_MARGIN")
LAYERS = ("terminal", "float_rgb", "rgb8", "mp4")
KEYS = {"CORRECT": "watermark", "WRONG": "watermark-wrong"}
DOMAIN = "B-JPM-v1"
GAMMA_S = 1 / math.sqrt(2)
GAMMA_P = 0.5


def validate_latent(value):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != LATENT_SHAPE:
        raise ValueError("expected normalized latent [1,16,46,40,64]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError("normalized latent must be finite floating point")


def public_codebook(key):
    """Key/public geometry only. No message, sample identity, host or targets."""
    if not isinstance(key, str) or not key:
        raise ValueError("nonempty receiver key required")
    def digest(text):
        return hmac.new(key.encode("utf-8"), text.encode("ascii"), hashlib.sha256).digest()
    def sign(text):
        return 1 if digest(text)[0] & 1 == 0 else -1
    basis = [(c, a, b) for c in range(4, 16) for a in range(8) for b in range(8) if (a, b) != (0, 0)]
    directions = torch.zeros((4, 54, 12, 2, 8, 8), dtype=torch.float64)
    selected = []
    for r in range(4):
        indices = sorted(range(756), key=lambda i: (digest(f"{DOMAIN}|roi={r}|basis={i}|perm"), i))[:54]
        selected.append(indices)
        for j, i in enumerate(indices):
            c, a, b = basis[i]
            polarity = sign(f"{DOMAIN}|roi={r}|basis={i}|sign")
            for y in range(8):
                for x in range(8):
                    value = polarity * (1 - 2 * (((a & y).bit_count() + (b & x).bit_count()) % 2)) / math.sqrt(128)
                    directions[r, j, c-4, 0, y, x] = value
                    directions[r, j, c-4, 1, y, x] = -value
    permutations = [sorted(range(32), key=lambda l: (digest(f"{DOMAIN}|state={k}|bit={l}|payload-perm"), l)) for k in STATES]
    masks = [[sign(f"{DOMAIN}|state={k}|bit={l}|payload-sign") for l in range(32)] for k in STATES]
    return dict(key=key, domain=DOMAIN, directions=directions, selected_basis=selected,
                permutations=permutations, masks=masks)


def protocol_record():
    return dict(domain=DOMAIN, gamma_s=GAMMA_S, gamma_p=GAMMA_P,
                gamma_s_definition="1/sqrt(2)", gamma_p_definition="1/2",
                unit_normal_distance=1.0, calibrated_to_media=False, norm_cap=None,
                latent_shape=list(LATENT_SHAPE), video_shape=list(VIDEO_SHAPE), rois=ROIS,
                arms=ARMS, layers=LAYERS, keys=KEYS,
                receiver_scope="complete clean 181-frame public grid; unknown start/edits not tested",
                aggregation="equal soft sum divided by fixed 22; erased positions have no evidence",
                gamma_shortfall_action="record actual FP32 margins; no retry; finite candidate proceeds to media")


def read_local(received, key, public):
    """Blind interface: observation, current key and message-free public book."""
    validate_latent(received)
    if public["key"] != key or public["domain"] != DOMAIN:
        raise ValueError("public receiver codebook/key mismatch")
    z = received.detach().to(device="cpu", dtype=torch.float64)
    scores = torch.empty((22, 4, 54), dtype=torch.float64)
    for pos in STATES:
        for r, (y0, y1, x0, x1) in enumerate(ROIS):
            block = z[0, 4:16, 1+2*pos:3+2*pos, y0:y1, x0:x1].reshape(-1)
            scores[pos, r] = public["directions"][r].reshape(54, -1) @ block
    pooled = scores.mean(dim=1)
    if not bool(torch.isfinite(pooled).all()):
        raise ValueError("nonfinite local scores")
    states, local = [], []
    for pos, row in enumerate(pooled):
        pilot = row[:22]
        maximum = float(pilot.max())
        winners = torch.nonzero(pilot == maximum).flatten().tolist()
        ordered = pilot.sort(descending=True).values
        label = winners[0] if len(winners) == 1 else None
        states.append(dict(position=pos, label=label, status="READ" if label is not None else "UNRESOLVED",
                           tied_candidates=winners, top_score=maximum, runner_up=float(ordered[1]),
                           top_gap=float(ordered[0]-ordered[1])))
        soft = [None] * 32
        if label is not None:
            for l, bit in enumerate(public["permutations"][label]):
                soft[bit] = public["masks"][label][l] * float(row[22+l])
        local.append(dict(position=pos, inferred_state=label, soft_bits=soft,
                          bits=[None if x is None or x == 0 else int(x > 0) for x in soft]))
    aggregate = []
    for bit in range(32):
        support = [p for p in STATES if local[p]["soft_bits"][bit] is not None]
        soft_sum = math.fsum(local[p]["soft_bits"][bit] for p in support)
        decoded = None if not support or soft_sum == 0 else int(soft_sum > 0)
        aggregate.append(dict(bit=bit, decoded=decoded, status="UNRESOLVED" if decoded is None else "READ",
                              soft_sum=soft_sum, fixed_denominator_mean=soft_sum/22,
                              support_count=len(support), support_positions=support,
                              erased_positions=[p for p in STATES if p not in support], position_denominator=22))
    return dict(status="READ", truth_used=False, roi_scores=scores.tolist(), pooled_scores=pooled.tolist(),
                states=states, local_payload=local, payload=aggregate,
                public_permutations=public["permutations"], public_masks=public["masks"],
                geometry="clean complete public grid; indices locate observations, not decoded states")


def project_state(a, target):
    """One sorted active-set reduction for the coupled 21-inequality cone."""
    values = a.detach().double().clone()
    competitors = sorted((j for j in STATES if j != target), key=lambda j: (-float(a[j]), j))
    total, count = float(a[target]), 1
    active = []
    for j in competitors:
        shifted = float(a[j]) + GAMMA_S
        if shifted <= total/count:
            break
        total += shifted
        count += 1
        active.append(j)
    level = total/count
    values[target] = level
    for j in competitors:
        values[j] = min(float(a[j]), level-GAMMA_S)
    return values, active


def delta_statistics(base, marked):
    delta = marked.detach().cpu().double() - base.detach().cpu().double()
    mask = torch.zeros(LATENT_SHAPE, dtype=torch.bool)
    for y0, y1, x0, x1 in ROIS:
        mask[0, 4:16, 1:45, y0:y1, x0:x1] = True
    def stats(x):
        return dict(coordinates=x.numel(), l2=float(torch.linalg.vector_norm(x)),
                    rms=float(x.square().mean().sqrt()), linf=float(x.abs().max()),
                    changed_coordinates=int(torch.count_nonzero(x)))
    return dict(global_delta=stats(delta), support=stats(delta[mask]), outside_support=stats(delta[~mask]),
                legacy_channels_0_3=stats(delta[:, :4]),
                times=[dict(latent_time=t, **stats(delta[:, :, t])) for t in range(46)],
                coordinate_system="actual FP32 candidate minus FP32 source in normalized latent coordinates")


def write_projection(base, host_blind, bits, public):
    """One continuous host-aware projection, one FP32 materialization, no cap."""
    validate_latent(base)
    if len(bits) != 32 or any(type(b) is not int or b not in (0, 1) for b in bits):
        raise ValueError("writer requires exactly 32 binary message bits")
    if host_blind.get("status") != "READ":
        raise ValueError("writer requires the existing correct-key host score table")
    host = torch.tensor(host_blind["pooled_scores"], dtype=torch.float64)
    if host.shape != (22, 54) or not bool(torch.isfinite(host).all()):
        raise ValueError("invalid host score table")
    source = base.detach().cpu().float()
    out = source.double().clone()
    coefficients, records = [], []
    for k in STATES:
        updated, active = project_state(host[k, :22], k)
        signs = torch.tensor([(2*bits[public["permutations"][k][l]]-1)*public["masks"][k][l]
                              for l in range(32)], dtype=torch.float64)
        payload_delta = signs * (GAMMA_P-signs*host[k, 22:]).clamp_min(0)
        delta = torch.cat((updated-host[k, :22], payload_delta))
        coefficients.append(delta)
        records.append(dict(position=k, active_state_competitors=active,
                            state_score_correction=delta[:22].tolist(), payload_score_correction=delta[22:].tolist()))
        for r, (y0, y1, x0, x1) in enumerate(ROIS):
            # w=(1/4) sum u, so 4*Delta*w adds Delta*u to each ROI.
            out[0, 4:16, 1+2*k:3+2*k, y0:y1, x0:x1] += (delta @ public["directions"][r].reshape(54, -1)).reshape(12, 2, 8, 8)
    candidate = out.float()
    validate_latent(candidate)
    coeff = torch.stack(coefficients)
    return candidate, dict(status="CONSTRUCTED", protocol=protocol_record(), rows=records,
        continuous_distance_squared_display=float(4*coeff.square().sum()),
        continuous_state_distance_display=float((4*coeff[:, :22].square().sum()).sqrt()),
        continuous_payload_distance_display=float((4*coeff[:, 22:].square().sum()).sqrt()),
        actual_fp32=delta_statistics(source, candidate),
        rounding="FP64 computation then a single nearest FP32 materialization; no ULP adjustment or retry",
        actual_constraint_check="after persisted blind observations; target shortfall does not suppress media")


def evaluate_bits(rows, bits, status="READ"):
    result = []
    for i in range(32):
        decoded = rows[i].get("decoded") if status == "READ" else None
        result.append(dict(bit=i, target=bits[i], decoded=decoded, status=("READ" if decoded is not None else "UNRESOLVED") if status == "READ" else status,
                           correct=None if decoded is None else decoded == bits[i]))
    observed = [r for r in result if r["decoded"] is not None]
    errors = sum(not r["correct"] for r in observed)
    return dict(status="EVALUATED" if status == "READ" else status, denominator=32, rows=result,
                read_bits=len(observed), unresolved_bits=32-len(observed) if status == "READ" else None,
                unavailable_bits=0 if status == "READ" else 32, bit_errors=errors if status == "READ" else None,
                ber_on_read_bits=None if not observed else errors/len(observed),
                error_fraction_fixed32=errors/32 if status == "READ" else None, coverage=len(observed)/32)


def evaluate_local(raw, targets, bits):
    """Called only after all available blind outputs are saved; no feedback."""
    status = raw.get("status", "NOT_ENTERED")
    if status != "READ":
        return dict(status=status, state_denominator=22, state_competitor_denominator=462, chip_denominator=704,
                    states=[dict(position=k, status=status, competitors=[dict(competitor=j, status=status) for j in STATES if j != k]) for k in STATES],
                    chips=[dict(position=k, chip=l, status=status) for k in STATES for l in range(32)],
                    local_decoded_bit_denominator=704,
                    local_decoded_bits=[dict(position=k, bit=b, status=status, decoded=None, correct=None) for k in STATES for b in range(32)],
                    payload=evaluate_bits([], bits, status), exact_states=None, unresolved_states=None, unavailable_states=22)
    states, chips = [], []
    for pos, target in enumerate(targets):
        row = raw["pooled_scores"][pos]
        decision = raw["states"][pos]
        competitors = [dict(competitor=j, gap=row[target]-row[j],
                            target_shortfall=GAMMA_S-(row[target]-row[j])) for j in STATES if j != target]
        states.append(dict(position=pos, target=target, label=decision["label"], status=decision["status"],
                           exact=decision["label"] == target, competitors=competitors,
                           true_min_gap=min(r["gap"] for r in competitors)))
        for l in range(32):
            bit = raw["public_permutations"][target][l]
            sign = (2*bits[bit]-1)*raw["public_masks"][target][l]
            margin = sign*row[22+l]
            chips.append(dict(position=pos, chip=l, message_bit=bit, target_sign=sign,
                              signed_margin=margin, target_shortfall=GAMMA_P-margin,
                              status="UNRESOLVED" if margin == 0 else "SCORED", positive=margin > 0,
                              meaning="truth-state geometry after blind persistence; not blind decoder input"))
    state_gaps = [c["gap"] for s in states for c in s["competitors"]]
    chip_gaps = [c["signed_margin"] for c in chips]
    decoded_rows = [dict(position=k, bit=b, inferred_state=raw["local_payload"][k]["inferred_state"],
                         decoded=raw["local_payload"][k]["bits"][b], target=bits[b],
                         status="UNRESOLVED" if raw["local_payload"][k]["bits"][b] is None else "READ",
                         correct=None if raw["local_payload"][k]["bits"][b] is None else raw["local_payload"][k]["bits"][b] == bits[b])
                    for k in STATES for b in range(32)]
    return dict(status="EVALUATED", state_denominator=22, state_competitor_denominator=462,
                chip_denominator=704, states=states, chips=chips,
                local_decoded_bit_denominator=704, local_decoded_bits=decoded_rows,
                local_decoded_correct=sum(r["correct"] is True for r in decoded_rows),
                local_decoded_errors=sum(r["correct"] is False for r in decoded_rows),
                local_decoded_erasures=sum(r["decoded"] is None for r in decoded_rows),
                exact_states=sum(s["exact"] for s in states),
                unresolved_states=sum(s["label"] is None for s in states),
                min_state_gap=min(state_gaps), min_payload_margin=min(chip_gaps),
                all_state_gaps_positive=all(g > 0 for g in state_gaps),
                all_chip_margins_positive=all(g > 0 for g in chip_gaps),
                state_target_shortfalls=sum(g < GAMMA_S for g in state_gaps),
                payload_target_shortfalls=sum(g < GAMMA_P for g in chip_gaps),
                payload=evaluate_bits(raw["payload"], bits))
