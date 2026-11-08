"""DCT2 local latent identity: fixed writer and separate regional read caches.

No model, IO, source-clock inference, state selection or cross-region fusion.
An E41 cache is local to its actual encoding view, not an RGB-crop decoder.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import math

import numpy as np
import torch

LATENT_SHAPE = (1, 16, 46, 40, 64)
LOCAL_SHAPE = (1, 16, 11, 40, 64)
REGIONS = {"R0": (8, 24, 8, 24), "R1": (8, 24, 40, 56), "Rc": (8, 24, 24, 40)}
PAYLOAD_REGIONS = ("R0", "R1")
MODES = tuple((u, v) for u in range(8, 12) for v in range(8, 12))
BETA = 1 / math.sqrt(2)
TARGET_COUNT = 5888
ETA = 368.0
VOTES_PER_BIT = 8


@lru_cache(maxsize=1)
def _canonical_basis():
    # One canonical CPU construction, shared by writer/read and both keys.
    axis = torch.arange(16, dtype=torch.float64) + 0.5
    rows = {u: math.sqrt(2 / 16) * torch.cos(math.pi * axis * u / 16)
            for u in range(8, 12)}
    return torch.stack([rows[u][:, None] * rows[v][None, :] for u, v in MODES])


def basis(*, device=None, dtype=torch.float32):
    """Sixteen unit DCT-II rows in canonical (u outer, v inner) order."""
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("DCT basis requires float32 or float64")
    return _canonical_basis().to(device=device, dtype=dtype, copy=True)


@lru_cache(maxsize=64)
def coordinates(key, region_id):
    """Immutable keyed mode order. Rc uses a named R0/R1 template only."""
    if not isinstance(key, str) or not key:
        raise ValueError("nonempty key required")
    if region_id not in PAYLOAD_REGIONS:
        raise ValueError("key layout requires R0 or R1")
    rho = PAYLOAD_REGIONS.index(region_id)
    modes = list(MODES)
    np.random.default_rng(sum(map(ord, key)) + 104729 * (rho + 1)).shuffle(modes)
    return tuple(modes)


def source_group(time):
    """Writer-only time mapping. Receivers never call this to assign a group."""
    if type(time) is not int or not 0 <= time <= 45:
        raise ValueError("writer latent time must be 0..45")
    return 0 if time == 0 else 3 if time == 45 else (time - 1) // 11


def _validate_tensor(value, shape):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
        raise ValueError(f"normalized latent shape must be {shape}")
    if value.dtype != torch.float32:
        raise ValueError("normalized latent must be FP32")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("nonfinite normalized latent")


def build_target(latent, key, bits):
    """Writer target [region,channel,time,canonical-mode], including guards."""
    _validate_tensor(latent, LATENT_SHAPE)
    if len(bits) != 32 or any(type(b) is not int or b not in (0, 1) for b in bits):
        raise ValueError("exactly 32 binary writer bits required")
    groups = torch.tensor([source_group(t) for t in range(46)], device=latent.device)
    message = latent.new_tensor(bits)
    target = latent.new_empty((2, 4, 46, 16))
    for rho, region in enumerate(PAYLOAD_REGIONS):
        inverse = [0] * 16
        for j, mode in enumerate(coordinates(key, region)):
            inverse[MODES.index(mode)] = j % 8
        slots = torch.tensor(inverse, device=latent.device)
        values = BETA * (2 * message[8 * groups[:, None] + slots[None, :]] - 1)
        target[rho] = values.unsqueeze(0).expand(4, -1, -1)
    return target


def _project_patch(value, region, times, directions):
    y0, y1, x0, x1 = REGIONS[region]
    patch = value[0, :4, times, y0:y1, x0:x1]
    return torch.einsum("ctij,mij->ctm", patch, directions)


def local_delta(clean, target):
    """One FP32 local leaf, one joint loss and backward; no tail gradient."""
    _validate_tensor(clean, LATENT_SHAPE)
    if (not isinstance(target, torch.Tensor) or tuple(target.shape) != (2, 4, 46, 16)
            or target.dtype != torch.float32 or target.device != clean.device):
        raise ValueError("FP32 DCT target [2,4,46,16] on the clean device required")
    if not bool(torch.isfinite(target).all()):
        raise FloatingPointError("nonfinite DCT target")
    directions = basis(device=clean.device)
    with torch.enable_grad():
        leaf = clean.detach().clone().requires_grad_(True)
        q = torch.stack([_project_patch(leaf, r, slice(0, 46), directions)
                         for r in PAYLOAD_REGIONS])
        squared = (q - target.detach()).square()
        loss = squared.sum() / TARGET_COUNT
        gradient = torch.autograd.grad(loss, leaf, create_graph=False)[0]
        delta = (-ETA * gradient).detach()
    if not bool(torch.isfinite(delta).all() and torch.isfinite(loss)):
        raise FloatingPointError("nonfinite local DCT loss/gradient")
    regions = []
    for rho, region in enumerate(PAYLOAD_REGIONS):
        y0, y1, x0, x1 = REGIONS[region]
        part = delta[0, :4, :, y0:y1, x0:x1].double()
        regions.append(dict(region=region, target_count=2944,
                            loss=float(squared[rho].mean().detach()),
                            delta_l2=float(part.square().sum().sqrt()),
                            delta_rms=float(part.square().mean().sqrt())))
    return delta, dict(status="RETURNED", carrier="DCT2", target_count=TARGET_COUNT,
                       effective_target_count=5632, guard_target_count=256,
                       eta=ETA, beta=BETA, unit_direction_update=0.125,
                       loss=float(loss.detach()), regions=regions,
                       guard_loss=float(squared[:, :, [0, 45]].mean().detach()),
                       gradient_rms=float(gradient.detach().double().square().mean().sqrt()),
                       conditional_clean_delta_rms=float(delta.double().square().mean().sqrt()),
                       projection_region_batches=2, target_coefficients=TARGET_COUNT,
                       local_backwards=1, writer_fft_calls=0, denoiser_backward=False,
                       tail_backward=False, guard_times=[0, 45])


def project(normalized, regions=None, rows=None, *, native=False, include_control=False):
    """Project once and share canonical coefficients across keys/templates.

    Returns CPU tensors; use decode() for JSON-ready keyed caches. This strict
    physical projection raises actual input errors. unavailable_projection()
    represents an upstream failed/not-entered call without inventing numbers.
    """
    if regions is not None:
        regions = tuple(regions)
        if regions not in (PAYLOAD_REGIONS, PAYLOAD_REGIONS + ("Rc",)):
            raise ValueError("project both R0/R1, optionally followed by diagnostic Rc")
        include_control = regions == PAYLOAD_REGIONS + ("Rc",)
    if rows is not None:
        rows = list(rows)
        if rows not in (list(range(1, 11)), list(range(1, 45))):
            raise ValueError("fixed effective E41/native rows required; no guard read")
        native = rows == list(range(1, 45))
    if type(native) is not bool or type(include_control) is not bool:
        raise ValueError("native/include_control must be booleans")
    _validate_tensor(normalized, LATENT_SHAPE if native else LOCAL_SHAPE)
    n = 44 if native else 10
    names = PAYLOAD_REGIONS + (("Rc",) if include_control else ())
    directions = basis(device=normalized.device)
    with torch.no_grad():
        q = {region: _project_patch(normalized, region, slice(1, n + 1), directions)
             .detach().cpu() for region in names}
    return dict(status="PROJECTED", cause_set=[], kind="NATIVE" if native else "E41",
                local_rows=list(range(1, n + 1)), include_control=include_control,
                coefficients=q, projection_region_batches=len(names),
                coefficient_slots=len(names) * 4 * n * 16,
                truth_inputs=False, message_inputs=False, guard_read=False)


def unavailable_projection(*, native=False, include_control=False,
                           status="UNRESOLVED", cause="UPSTREAM_UNAVAILABLE"):
    """Full planned projection shape, with no fabricated successful zeros."""
    if type(native) is not bool or type(include_control) is not bool:
        raise ValueError("native/include_control must be booleans")
    if status not in ("UNRESOLVED", "FAILED", "NOT_ENTERED", "UNSUPPORTED", "INTERRUPTED"):
        raise ValueError("explicit unavailable status required")
    if not isinstance(cause, str) or not cause:
        raise ValueError("unavailability cause required")
    n = 44 if native else 10
    names = PAYLOAD_REGIONS + (("Rc",) if include_control else ())
    return dict(status=status, cause_set=[cause], kind="NATIVE" if native else "E41",
                local_rows=list(range(1, n + 1)), include_control=include_control,
                coefficients={r: None for r in names}, projection_region_batches=0,
                coefficient_slots=len(names) * 4 * n * 16,
                truth_inputs=False, message_inputs=False, guard_read=False)


def _coefficient_array(value, n):
    if value is None:
        return None, "MISSING_COEFFICIENT_CACHE"
    try:
        q = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError):
        return None, "INVALID_COEFFICIENT_CACHE"
    if tuple(q.shape) != (4, n, 16) or not q.is_floating_point():
        return None, "INVALID_COEFFICIENT_SHAPE_OR_DTYPE"
    return q, None


def _regional_cache(value, *, key, key_id, observed, template, rows, view_id,
                    inherited_status, inherited_causes):
    n = len(rows)
    q, error = _coefficient_array(value, n)
    order = [MODES.index(mode) for mode in coordinates(key, template)]
    diagnostic = observed == "Rc"
    row_records = []
    # Raw arrays stay in canonical mode order. Entire bad rows are null.
    raw = [[[None] * 16 for _ in rows] for _ in range(4)]
    signed = [[[None] * 16 for _ in rows] for _ in range(4)]
    zeros = [[[None] * 16 for _ in rows] for _ in range(4)]
    for ri, row in enumerate(rows):
        causes = list(inherited_causes)
        if error:
            causes.append(error)
        elif not bool(torch.isfinite(q[:, ri]).all()):
            causes.append("NONFINITE_COEFFICIENT_ROW")
        causes = list(dict.fromkeys(causes))
        valid = not causes
        status = "AVAILABLE" if valid else (inherited_status if inherited_status in
                 ("FAILED", "NOT_ENTERED", "UNSUPPORTED", "INTERRUPTED") else "UNRESOLVED")
        evidence = f"{view_id}/row{row}/{observed}" if view_id is not None else None
        if valid:
            for c in range(4):
                raw[c][ri] = q[c, ri].tolist()
                signed[c][ri] = (2 * (q[c, ri] > 0).to(torch.int8) - 1).tolist()
                zeros[c][ri] = (q[c, ri] == 0).to(torch.int8).tolist()
        bits = []
        for b in range(8):
            refs = [dict(evidence_id=evidence, channel=c, mode_index=order[j],
                         mode=list(MODES[order[j]]), keyed_index=j)
                    for c in range(4) for j in (b, b + 8)]
            seq = [int(q[c, ri, order[j]] > 0) for c in range(4)
                   for j in (b, b + 8)] if valid else None
            ones = sum(seq) if valid else None
            bit = dict(local_bit=b, key=key_id, evidence_id=evidence,
                       value=Counter(seq).most_common(1)[0][0] if valid else None,
                       ones=ones, count=VOTES_PER_BIT, planned_count=VOTES_PER_BIT,
                       returned_count=VOTES_PER_BIT if valid else 0,
                       tie=ones == 4 if valid else None,
                       zero_coefficients=sum(int(q[c, ri, order[j]] == 0)
                                             for c in range(4) for j in (b, b + 8)) if valid else None,
                       coefficient_refs=refs, status=status, cause_set=causes,
                       diagnostic_only=diagnostic)
            bits.append(bit)
        row_records.append(dict(local_row=row, evidence_id=evidence,
                                observed_region_id=observed,
                                observed_region_slice=list(REGIONS[observed]),
                                carrier_region_id=template,
                                template_region_id=template if diagnostic else None,
                                channels=[0, 4], coordinate_space="normalized_latent",
                                diagnostic_only=diagnostic, bits=bits,
                                status=status, cause_set=causes))
    returned = sum(bit["value"] is not None for row in row_records for bit in row["bits"])
    return dict(observed_region_id=observed, carrier_region_id=template,
                diagnostic_only=diagnostic, coefficients=raw, signed_votes=signed,
                zero_coefficient_mask=zeros, local_cells=row_records,
                planned_bits=n * 8, returned_bits=returned,
                status="AVAILABLE" if returned == n * 8 else "INCOMPLETE",
                cause_set=list(dict.fromkeys(c for row in row_records for c in row["cause_set"])))


def read_coefficients(coefficients, key, carrier_region_id, rows, *,
                      observed_region_id=None, key_id=None):
    """Thin integrator API: regional bits without assigning physical IDs.

    rows must be 1..10 or 1..44. An Rc observation must explicitly say Rc;
    carrier_region_id then selects its R0/R1 template, not production support.
    The common runtime attaches actual view/evidence IDs to these records.
    """
    rows = list(rows)
    if rows not in (list(range(1, 11)), list(range(1, 45))):
        raise ValueError("fixed effective rows required; no guard read")
    if carrier_region_id not in PAYLOAD_REGIONS:
        raise ValueError("carrier template must be R0 or R1")
    observed = observed_region_id or carrier_region_id
    if observed not in (carrier_region_id, "Rc"):
        raise ValueError("actual region cannot be renamed as another production replica")
    if key_id is not None and (not isinstance(key_id, str) or not key_id):
        raise ValueError("key_id must be a nonempty label")
    return _regional_cache(coefficients, key=key, key_id=key_id or key,
                           observed=observed, template=carrier_region_id, rows=rows,
                           view_id=None, inherited_status="PROJECTED", inherited_causes=[])


def decode(projection, key, *, view_id, key_id=None):
    """Keyed local bits only; no source clock, group, message or C arguments.

    Missing regions/rows keep their planned slots. Rc remains a separate
    diagnostic dictionary, never a third production replica.
    """
    if not isinstance(view_id, str) or not view_id:
        raise ValueError("nonempty actual encoding view_id required")
    if key_id is not None and (not isinstance(key_id, str) or not key_id):
        raise ValueError("key_id must be a nonempty label")
    coordinates(key, "R0")  # Validate key even for entirely unavailable inputs.
    kind = projection.get("kind")
    n = 44 if kind == "NATIVE" else 10 if kind == "E41" else None
    if n is None or projection.get("local_rows") != list(range(1, n + 1)):
        raise ValueError("projection must retain the fixed effective local rows")
    if type(projection.get("include_control")) is not bool:
        raise ValueError("projection requires explicit include_control")
    coefficients = projection.get("coefficients")
    if not isinstance(coefficients, dict):
        coefficients = {}
    causes = list(projection.get("cause_set", []))
    status = projection.get("status", "UNRESOLVED")
    if status != "PROJECTED" and not causes:
        causes = ["PROJECTION_UNAVAILABLE"]
    kwargs = dict(key=key, key_id=key_id or key, rows=projection["local_rows"],
                  view_id=view_id, inherited_status=status, inherited_causes=causes)
    regions = {region: _regional_cache(coefficients.get(region), observed=region,
                                       template=region, **kwargs) for region in PAYLOAD_REGIONS}
    control = {template: _regional_cache(coefficients.get("Rc"), observed="Rc",
                                         template=template, **kwargs)
               for template in PAYLOAD_REGIONS} if projection["include_control"] else {}
    returned = sum(r["returned_bits"] for r in regions.values())
    return dict(status="AVAILABLE" if returned == 2 * n * 8 else "INCOMPLETE",
                cause_set=list(dict.fromkeys(c for r in regions.values() for c in r["cause_set"])),
                kind=kind, view_id=view_id, key=key_id or key, local_rows=projection["local_rows"],
                regions=regions, control_diagnostics=control,
                required_replica_ids=list(PAYLOAD_REGIONS), planned_bits=2 * n * 8,
                returned_bits=returned, diagnostic_planned_bits=2 * n * 8 if control else 0,
                truth_inputs=False, message_inputs=False, group_assigned=False,
                guard_read=False, vote_order="channel, keyed indices b then b+8; strict >0; Counter encounter tie")
