"""OLD8 two-state finite-sequence carrier and blind soft scorer.

Only the temporal word schedule changes. Spatial support, Fourier basis,
channel, four-age overlap and amplitude are the original OLD8 construction.
Scores are uncalibrated diagnostics and never accept a state.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import numpy as np

from main.tube_state import video_local_fourier_rm_state as old


PUBLIC = replace(old.PUBLIC, method_version="video-local-fourier-rm-old8-two-state-v1")
SEQUENCES = ("AA", "BB", "AB", "BA")
SWITCH_START = 23
MODES = ("ABSOLUTE_CONTROL", "ADJACENT_DIFFERENCE")


def selected_words(key):
    """First two words in the existing keyed OLD8 order; no response selection."""
    words = old.state_code(key)
    a, b = words[0].copy(), words[1].copy()
    if np.array_equal(a, b):
        raise ValueError("public OLD8 A/B words must differ")
    return {"A": a, "B": b}


def selection_receipt(key):
    words = selected_words(key)
    return dict(rule="A=state_code(key)[0]; B=state_code(key)[1]", response_selected=False,
                key_sha256=hashlib.sha256(key.encode()).hexdigest(),
                a_sha256=hashlib.sha256(words["A"].tobytes()).hexdigest(),
                b_sha256=hashlib.sha256(words["B"].tobytes()).hexdigest(),
                hamming=int(np.count_nonzero(words["A"] != words["B"])), switch_start=SWITCH_START)


def start_labels(sequence):
    if sequence not in SEQUENCES:
        raise ValueError("one of the four public sequences required")
    if sequence == "AA": return ("A",) * 45
    if sequence == "BB": return ("B",) * 45
    left, right = sequence
    return (left,) * (SWITCH_START - 1) + (right,) * (46 - SWITCH_START)


def start_signs(key, sequence):
    words = selected_words(key)
    out = np.asarray([words[label].reshape(4, 8) for label in start_labels(sequence)], dtype=np.int8)
    out.flags.writeable = False
    return out


def composite_signs(key, sequence):
    starts = start_signs(key, sequence)
    out = np.zeros((45, 4, 4, 8), dtype=np.int8)
    for u in range(1, 46):
        for age in range(4):
            if u - age >= 1: out[u - 1, :, age, :] = starts[u - age - 1]
    out.flags.writeable = False
    return out


def basis_layout(key):
    row = old.basis_layout(key); row["method_version"] = PUBLIC.method_version; return row


def bases(key): return old.bases(key)


def synthesize(key, sequence, public=PUBLIC, dtype=np.float64):
    if public != PUBLIC: raise ValueError("fixed public protocol required")
    target = np.zeros((1, 16, 46, 40, 64), dtype=dtype)
    signs = start_signs(key, sequence); basis = bases(key).astype(dtype); alpha = dtype(public.alpha)
    for start in range(1, 46):
        for age in range(4):
            u = start + age
            if u > 45: continue
            for block, (h0, h1, w0, w1) in enumerate(public.blocks):
                value = basis[block, :, 8 * age:8 * age + 8] @ signs[start - 1, block].astype(dtype)
                target[0, 4, u, h0:h1, w0:w1] += alpha * value.reshape(8, 8)
    return target


def extract(received_tensor, key, availability, public=PUBLIC):
    if public != PUBLIC: raise ValueError("fixed public protocol required")
    return old.extract(received_tensor, key, availability, old.PUBLIC)


def _empty(mode, reason, error=None):
    length = 44 if mode == MODES[0] else 43
    rows = [dict(received_index=i + 1, status=reason, available_dimensions=0,
                 candidate_costs={name: None for name in SEQUENCES}, top=[]) for i in range(length)]
    components = [dict(received_regular_index=u, age=age, source_start=u - age,
                       status="STRUCTURALLY_EXCLUDED" if u <= age else reason,
                       available_dimensions=0, costs={"A": None, "B": None}, top=[])
                  for u in range(1, 45) for age in range(4)] if mode == MODES[0] else []
    summary = dict(status="INCOMPLETE", reason=reason, score_status="UNCALIBRATED_FINITE_SEQUENCE_DIAGNOSTIC",
                   accepted_payload=False, state_sequence_accepted=False, top=[], canonical=None,
                   min_cost=None, top_distinct_gap=None, unique_model_hypothesis=False)
    if error is not None: summary["error"] = error
    return dict(mode=mode, summary=summary, sequence_scores={name: None for name in SEQUENCES},
                local_rows=rows, component_evidence=components, available_dimensions=0,
                counts=dict(sequence_candidates=4, sequence_scores=0, local_candidate_costs=0,
                            component_state_costs=0), truth_inputs=False)


def _component_evidence(q, key, availability):
    words = selected_words(key)
    means = {label: word.reshape(4, 8).astype(np.float64) * PUBLIC.alpha for label, word in words.items()}
    rows = []; scored = 0
    for u in range(1, 45):
        for age in range(4):
            start = u - age
            if start < 1:
                rows.append(dict(received_regular_index=u, age=age, source_start=start,
                                 status="STRUCTURALLY_EXCLUDED", available_dimensions=0,
                                 costs={"A": None, "B": None}, top=[])); continue
            support = np.broadcast_to(availability[u - 1, :, None], (4, 8)); n = int(support.sum())
            if n == 0:
                rows.append(dict(received_regular_index=u, age=age, source_start=start,
                                 status="NO_SUPPORT", available_dimensions=0,
                                 costs={"A": None, "B": None}, top=[])); continue
            obs = q[u - 1, :, age, :]
            costs = {label: float(np.mean((mean[support] - obs[support]) ** 2)) for label, mean in means.items()}
            minimum = min(costs.values()); top = [label for label in ("A", "B") if costs[label] - minimum <= PUBLIC.tie_atol]
            rows.append(dict(received_regular_index=u, age=age, source_start=start, status="SCORED",
                             available_dimensions=n, costs=costs, top=top, b_minus_a=costs["B"] - costs["A"],
                             unique=len(top) == 1, observed_energy=float(np.sum(obs[support] ** 2))))
            scored += 2
    return rows, scored


def score(q, key, availability, mode):
    """Blindly score all four fixed sequences; truth/message/arm are absent."""
    if mode not in MODES: raise ValueError("fixed score mode required")
    try:
        q = np.asarray(q, dtype=np.float64); mask = np.asarray(availability)
        if q.shape != (44, 4, 4, 8) or mask.dtype != np.bool_ or mask.shape != (44, 4):
            raise ValueError("fixed q44 and boolean44x4 support required")
        mask4 = np.broadcast_to(mask[:, :, None, None], q.shape)
        if not np.isfinite(q[mask4]).all(): raise ValueError("nonfinite available projection")
        safe = np.where(mask4, q, 0.0)
        templates = {name: composite_signs(key, name)[:44].astype(np.float64) * PUBLIC.alpha for name in SEQUENCES}
        if mode == MODES[0]:
            values, support, local_support, local_templates = safe, mask4, mask, templates
        else:
            values = safe[1:] - safe[:-1]; local_support = mask[1:] & mask[:-1]
            support = np.broadcast_to(local_support[:, :, None, None], values.shape)
            local_templates = {name: value[1:] - value[:-1] for name, value in templates.items()}
        n = int(support.sum())
        if n == 0: return _empty(mode, "NO_OBSERVATIONS")
        sequence_scores = {name: float(np.mean((value[support] - values[support]) ** 2))
                           for name, value in local_templates.items()}
        if not all(np.isfinite(v) for v in sequence_scores.values()): raise ValueError("nonfinite sequence score")
        minimum = min(sequence_scores.values()); top = [name for name in SEQUENCES if sequence_scores[name] - minimum <= PUBLIC.tie_atol]
        ordered = sorted(sequence_scores.values()); local_rows = []; local_scored = 0
        for index, (obs, row_mask) in enumerate(zip(values, local_support)):
            row_support = np.broadcast_to(row_mask[:, None, None], obs.shape); rn = int(row_support.sum())
            if rn == 0:
                local_rows.append(dict(received_index=index + 1, status="NO_SUPPORT", available_dimensions=0,
                                       candidate_costs={name: None for name in SEQUENCES}, top=[])); continue
            costs = {name: float(np.mean((template[index][row_support] - obs[row_support]) ** 2))
                     for name, template in local_templates.items()}
            row_min = min(costs.values()); row_top = [name for name in SEQUENCES if costs[name] - row_min <= PUBLIC.tie_atol]
            local_rows.append(dict(received_index=index + 1, status="SCORED", available_dimensions=rn,
                                   candidate_costs=costs, top=row_top, unique=len(row_top) == 1,
                                   observed_energy=float(np.sum(obs[row_support] ** 2)))); local_scored += 4
        components, component_scored = _component_evidence(safe, key, mask) if mode == MODES[0] else ([], 0)
        energy = float(np.sum(values[support] ** 2))
        summary = dict(status="NO_ENERGY" if energy == 0 else "COMPLETE",
                       reason="NO_ENERGY" if energy == 0 else "FINITE_SEQUENCE_TIE" if len(top) > 1 else "UNIQUE_FINITE_MODEL_ONLY",
                       score_status="UNCALIBRATED_FINITE_SEQUENCE_DIAGNOSTIC", accepted_payload=False,
                       state_sequence_accepted=False, top=top, canonical=top[0] if energy > 0 and len(top) == 1 else None,
                       min_cost=minimum, top_distinct_gap=ordered[1] - ordered[0],
                       unique_model_hypothesis=energy > 0 and len(top) == 1, observed_projection_energy=energy,
                       interpretation="ABS is the primary four-sequence diagnostic" if mode == MODES[0]
                                      else "predeclared adjacent-difference diagnostic; static AA/BB need not be unique")
        return dict(mode=mode, summary=summary, sequence_scores=sequence_scores, local_rows=local_rows,
                    component_evidence=components, available_dimensions=n,
                    counts=dict(sequence_candidates=4, sequence_scores=4, local_candidate_costs=local_scored,
                                component_state_costs=component_scored), truth_inputs=False)
    except (ValueError, TypeError, FloatingPointError, OverflowError) as exc:
        return _empty(mode, "INVALID_OBSERVATION", f"{type(exc).__name__}: {exc}")
