"""Fixed raw-scalar 1:1 sum of the two saved BASE gradients."""
from __future__ import annotations

import math

VIEWS = ("BASE", "SUM_STEP")
NONPOSITIVE_ROSTER = (0, 1, 4, 5, 7, 9, 12, 13, 16, 17, 18, 20, 27, 29, 30, 31)
FIXED_WRONG_OFFSET = 16
STEP = 0.10
CAP = 1.0
MU = 0.02
GAMMA = 0.01
CEILING = (
    "One fixed candidate from the saved BASE and the raw scalar-weight 1:1 sum of the "
    "saved aggregate-payload and state-G16 masked gradients. The summed direction alone "
    "is normalized, then one 0.10 step is decoded once. No replacement VJP, per-gradient "
    "normalization, scan, projection, second step, rollback, encode, DiT/native/scheduler "
    "work or codec. The finite endpoint reports all32 margins, all22 correlations, actual "
    "max-wrong G and clamped-float RGB quality. It does not establish blind recovery, "
    "codec survival, FPR, independent-sample behavior, generalization or scientific PASS."
)


def construct_candidate(base, gradients, support):
    """Return BASE + 0.10*unit(g_A+g_G), preserving raw scalar units."""
    import torch

    if set(gradients) != {"aggregate_payload", "state_G16"}:
        raise ValueError("exactly the two adopted saved scalar gradients are required")
    support.validate_for(tuple(base.shape))
    if not bool(torch.isfinite(base).all()):
        raise ValueError("BASE must be finite")

    def masked(value):
        result = torch.zeros_like(value)
        start, end = support.time_range
        for y0, y1, x0, x1 in support.rois:
            result[:, :, start:end, y0:y1, x0:x1] = value[:, :, start:end, y0:y1, x0:x1]
        return result
    checked = {}
    for name in ("aggregate_payload", "state_G16"):
        value = gradients[name]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(base.shape):
            raise ValueError(name + " gradient must share the BASE tensor shape")
        value = value.detach().cpu().double()
        if not bool(torch.isfinite(value).all()):
            raise ValueError(name + " gradient must be finite")
        if torch.count_nonzero(value - masked(value)):
            raise ValueError(name + " gradient must retain the adopted support mask")
        checked[name] = value
    summed = checked["aggregate_payload"] + checked["state_G16"]
    summed_norm = float(summed.norm())
    if not summed_norm > 0:
        raise ValueError("saved raw-weight gradient sum has zero norm")
    direction = summed / summed_norm
    slopes = {name: float((value * direction).sum()) for name, value in checked.items()}
    candidate = (base.detach().cpu().double() + STEP * direction).to(base.dtype)
    displacement = candidate.double() - base.detach().cpu().double()
    radius = float(displacement.norm())
    if torch.count_nonzero(displacement - masked(displacement)):
        raise ValueError("stored candidate displacement leaked outside the adopted support")
    if radius > CAP:
        raise ValueError("stored candidate exceeds the original BASE radius 1")
    return candidate, dict(
        status="FIXED_GRADIENT_SUM_CANDIDATE",
        scalar_weights={"aggregate_payload": 1.0, "state_G16": 1.0},
        per_gradient_normalization=False,
        summed_direction_normalized=True,
        step=STEP,
        endpoint_projection=False,
        scan=False,
        second_step=False,
        rollback=False,
        gradient_l2={name: float(value.norm()) for name, value in checked.items()},
        summed_gradient_l2=summed_norm,
        raw_unit_direction_slopes=slopes,
        first_order_predicted_changes={name: STEP * value for name, value in slopes.items()},
        stored_BASE_distance=radius,
        radius_cap=CAP,
        ceiling=CEILING,
    )


def summarize(values):
    """Descriptive fixed readout summary; MU/GAMMA are never gates."""
    margins = list(values["margins"])
    correlations = list(values["correlations"])
    if len(margins) != 32 or len(correlations) != 22:
        raise ValueError("complete all32 margins and all22 correlations required")
    try:
        finite = all(math.isfinite(float(value)) for value in margins + correlations)
    except (TypeError, ValueError, OverflowError):
        finite = False
    if not finite:
        raise ValueError("finite complete readout required")
    maximum = max(correlations[1:])
    offsets = [index for index in range(1, 22) if correlations[index] == maximum]
    state_gap = correlations[0] - maximum
    g16 = correlations[0] - correlations[FIXED_WRONG_OFFSET]
    mean_a = sum(margins[index] for index in NONPOSITIVE_ROSTER) / len(NONPOSITIVE_ROSTER)
    return dict(
        margins=margins,
        correlations=correlations,
        state_gap=state_gap,
        actual_max_wrong_correlation=maximum,
        actual_max_wrong_offsets=offsets,
        fixed_G16=g16,
        fixed_mean_A=mean_a,
        positive_count=sum(value > 0 for value in margins),
        nonpositive_count=sum(value <= 0 for value in margins),
        margin_minus_mu=[value - MU for value in margins],
        state_gap_minus_gamma=state_gap - GAMMA,
        mu=MU,
        gamma=GAMMA,
        thresholds_are_descriptive_only=True,
    )
