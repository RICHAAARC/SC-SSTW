"""Uniform-budget tanh control at T46 or at T44+T46 on live native histories."""
import copy
import math

import torch

from main.tube_state import objective_alignment as objective
from .flow_step import measures
from .tube_clean_proxy import clean_direction, load_transformer
from .tube_multistep import fingerprint, velocity
from .tube_retention import native_step

CONTROL_STEPS = (44, 46)


def nominal_objective_record(z, directions, codes, message):
    raw = objective.metrics(z.detach().cpu(), directions, codes, message)
    hard = raw["hard_clipped_message_scores"]
    tanh = raw["tanh_message_scores"]
    return dict(
        tanh_correct_score=tanh[message],
        tanh_wrong_score=tanh[1 - message],
        tanh_gap=tanh[message] - tanh[1 - message],
        hard_correct_score=hard[message],
        hard_wrong_score=hard[1 - message],
        hard_gap=hard[message] - hard[1 - message],
        nominal_rank=raw["nominal_rank"],
        meaning="absolute target/wrong nominal latent scores; not blind media receiver evidence",
    )


def tanh_direction(z, v, sigma, book, message, index):
    """Compute temperature-1 tanh direction on a detached CPU clean leaf."""
    clean = (z - float(sigma) * v).detach().cpu()
    raw, record = clean_direction(clean, book, message, "tanh")
    record.update(
        index=index,
        temperature=1,
        meaning=f"CPU detached current clean{index} leaf; no terminal gradient transport, model/VAE graph or receiver derivative",
    )
    return raw, record


@torch.no_grad()
def zero_step(snapshot, z, v, index, count, kind="scheduler_step"):
    """Advance a deepcopy; the supplied complete history is never mutated."""
    before = fingerprint(vars(snapshot))
    if snapshot.step_index != index:
        raise ValueError("native cursor mismatch")
    after_snapshot = copy.deepcopy(snapshot)
    after = native_step(after_snapshot, z, v, index, count, kind)
    if fingerprint(vars(snapshot)) != before:
        raise RuntimeError("zero step polluted source history")
    return after, after_snapshot


@torch.no_grad()
def prepare_direction(snapshot, z, v, zero_next, raw, target, index, count):
    """Measure a same-history unit response and solve epsilon for fixed native RMS."""
    before = fingerprint(vars(snapshot))
    if index not in CONTROL_STEPS or snapshot.step_index != index:
        raise ValueError("invalid control cursor")
    sigma = float(snapshot.sigmas[index])
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite sigma required")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("positive fixed native response budget required")
    raw = raw.detach().to(device=z.device, dtype=torch.float32)
    norm = measures(raw)["support_rms"]
    if not math.isfinite(norm) or norm <= 0 or not torch.isfinite(raw).all():
        raise ValueError("zero/nonfinite tanh direction; no fallback")
    q = raw / norm
    probe, _ = zero_step(snapshot, z, v - q / sigma, index, count, "unit_response_probe_step")
    unit_D = measures(probe - zero_next)
    response = unit_D["support_rms"]
    if not math.isfinite(response) or response <= 0:
        raise ValueError("zero/nonfinite unit native response; no fallback")
    epsilon = target / response
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("invalid fixed-budget epsilon")
    if fingerprint(vars(snapshot)) != before:
        raise RuntimeError("unit probe polluted source history")
    return q, epsilon, dict(
        status="READY",
        index=index,
        sigma=sigma,
        raw_direction=measures(raw),
        unit_direction=measures(q),
        unit_D=unit_D,
        epsilon=epsilon,
        target_D_support_rms=target,
        input_fingerprint=fingerprint(z),
        history_fingerprint=before,
        budget_meaning="fixed historical-oracle-provenance constant split; measured only on this live same-history unit response",
    )


@torch.no_grad()
def controlled_step(snapshot, z, v, zero_next, q, epsilon, target, index, book, message, count):
    """Apply one formal control from a fresh deepcopy and retain its local response."""
    before = fingerprint(vars(snapshot))
    if snapshot.step_index != index:
        raise ValueError("formal control cursor mismatch")
    sigma = float(snapshot.sigmas[index])
    clean_before = z - sigma * v
    u = epsilon * q
    controlled_v = v - u / sigma
    if not torch.isfinite(controlled_v).all():
        raise FloatingPointError("nonfinite controlled velocity")
    after, after_snapshot = zero_step(snapshot, z, controlled_v, index, count)
    actual_D = measures(after - zero_next)
    relative_error = abs(actual_D["support_rms"] - target) / target
    directions = torch.from_numpy(book["directions"]).double()
    codes = torch.from_numpy(book["codes"]).double()
    row = dict(
        index=index,
        sigma=sigma,
        target_D_support_rms=target,
        epsilon=epsilon,
        u=measures(u),
        delta_velocity=measures(controlled_v - v),
        actual_D=actual_D,
        matching_signed_error=actual_D["support_rms"] - target,
        matching_relative_error=relative_error,
        input_fingerprint=fingerprint(z),
        history_fingerprint=before,
        zero_next_fingerprint=fingerprint(zero_next),
        controlled_next_fingerprint=fingerprint(after),
        clean_before=nominal_objective_record(clean_before, directions, codes, message),
        clean_after=nominal_objective_record(z - sigma * controlled_v, directions, codes, message),
        response_meaning="same-input complete-history controlled-minus-zero native next-state response",
    )
    if fingerprint(vars(snapshot)) != before:
        raise RuntimeError("formal control polluted source history")
    return after, after_snapshot, row, dict(applied_u=u.cpu(), actual_D=(after - zero_next).cpu())


def cumulative(rows):
    values = [row["actual_D"] for row in rows]
    result = {}
    for space in ("support", "global"):
        rms = [row[space + "_rms"] for row in values]
        result[space] = dict(
            sum_rms=sum(rms),
            peak_rms=max(rms, default=0.0),
            sum_rms_squared=sum(value * value for value in rms),
        )
    return dict(
        controlled_steps=len(rows),
        actual_D=result,
        meaning="sum of per-step same-history native response RMS; not equal energy or terminal displacement",
    )


@torch.no_grad()
def free_step(pipe, snapshot, z, prompt, negative, dtype, guidance, index, count):
    """One uncontrolled live-model native step."""
    v = velocity(pipe, z, snapshot, prompt, negative, dtype, guidance, index, count)
    return (*zero_step(snapshot, z, v, index, count), v)
