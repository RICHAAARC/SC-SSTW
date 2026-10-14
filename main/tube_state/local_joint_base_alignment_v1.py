"""Two fixed BASE-point scalar derivatives and three-direction alignment report."""
from __future__ import annotations

from main.tube_state.local_joint_readout_m0_v1 import tensor_readout

UNITS = ("aggregate_payload", "state_G16")
NONPOSITIVE_ROSTER = (0, 1, 4, 5, 7, 9, 12, 13, 16, 17, 18, 20, 27, 29, 30, 31)
FIXED_WRONG_OFFSET = 16
DIRECTIONS = ("old_bridge", "posterior_bridge", "base_to_cont_final")
CEILING = (
    "Two serial independent decoder VJPs at one saved BASE point: the mean of a fixed "
    "16-bit message-aligned margin roster and correlation[0]-correlation[16]. Three fixed "
    "saved directions are projected on CPU. No candidate, update, scan, per-bit Jacobian, "
    "ordinary decode, encode, DiT/native/scheduler work or codec. Zero or adverse gradients "
    "remain complete observations. This known-message one-point diagnostic does not establish "
    "individual-bit common improvement, positive-bit protection, finite-step behavior, blind "
    "decoding, media survival, generalization or scientific PASS."
)


def values55(read):
    margins = read["margins"].detach().cpu().double().reshape(-1)
    correlations = read["correlations"].detach().cpu().double().reshape(-1)
    if len(margins) != 32 or len(correlations) != 22:
        raise ValueError("readout must retain 32 margins and 22 correlations")
    if not bool(margins.isfinite().all()) or not bool(correlations.isfinite().all()):
        raise ValueError("readout values must be finite")
    return dict(margins=margins.tolist(), correlations=correlations.tolist(),
                state_gap=float(correlations[0]-correlations[1:].max()))


def scalar(name, read):
    """Return the adopted scalar without selecting from the fresh readout."""
    if name == "aggregate_payload":
        return read["margins"].reshape(-1)[list(NONPOSITIVE_ROSTER)].mean()
    if name == "state_G16":
        return read["correlations"].reshape(-1)[0] - read["correlations"].reshape(-1)[FIXED_WRONG_OFFSET]
    raise ValueError("unknown BASE alignment scalar")


def readout_report(name, read):
    values = values55(read)
    correlations = values["correlations"]
    maximum = max(correlations[1:])
    fresh_offsets = [i for i in range(1, 22) if correlations[i] == maximum]
    fresh_nonpositive = [i for i, value in enumerate(values["margins"]) if value <= 0]
    return dict(
        name=name,
        values=values,
        scalar_value=float(scalar(name, read).detach()),
        scalar_definition=(
            "mean(message_aligned_margin[i] for i in fixed historical roster)"
            if name == "aggregate_payload" else "correlation[0]-correlation[16]"
        ),
        fixed_historical_nonpositive_roster=list(NONPOSITIVE_ROSTER),
        fresh_nonpositive_bits=fresh_nonpositive,
        fresh_roster_matches_historical=fresh_nonpositive == list(NONPOSITIVE_ROSTER),
        fixed_wrong_offset=FIXED_WRONG_OFFSET,
        fresh_active_wrong_offsets=fresh_offsets,
        fresh_active_includes_fixed_16=FIXED_WRONG_OFFSET in fresh_offsets,
        fixed_definition_unchanged_by_fresh_readout=True,
    )


def _unit(value):
    norm = float(value.norm())
    return (None if norm == 0 else value / norm), norm


def analyze(gradients, directions):
    """Project two unnormalized masked gradients onto three unit directions.

    Direction normalization only defines a per-unit-latent displacement slope. The
    gradients are never normalized; raw scalar slopes and dimensionless cosines are
    stored separately.
    """
    import torch

    if set(gradients) != set(UNITS) or set(directions) != set(DIRECTIONS):
        raise ValueError("exactly two fixed gradients and three fixed directions required")
    shape = tuple(next(iter(gradients.values())).shape)
    for collection in (gradients, directions):
        for name, value in collection.items():
            if not torch.is_tensor(value) or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} must be a finite tensor with the shared shape")
    direction_units, direction_norms = {}, {}
    for name in DIRECTIONS:
        direction_units[name], direction_norms[name] = _unit(directions[name].detach().cpu().double())
    pairwise = {}
    for left in DIRECTIONS:
        pairwise[left] = {}
        for right in DIRECTIONS:
            a, b = direction_units[left], direction_units[right]
            pairwise[left][right] = None if a is None or b is None else float((a*b).sum())
    rows = {}
    gradient_vectors = {name: gradients[name].detach().cpu().double() for name in UNITS}
    gradient_norms = {name: float(value.norm()) for name, value in gradient_vectors.items()}
    gradient_gram = {left: {right: float((gradient_vectors[left]*gradient_vectors[right]).sum())
                            for right in UNITS} for left in UNITS}
    gradient_cosines = {left: {right: (
        None if gradient_norms[left] == 0 or gradient_norms[right] == 0 else
        gradient_gram[left][right]/(gradient_norms[left]*gradient_norms[right]))
        for right in UNITS} for left in UNITS}
    for unit_name in UNITS:
        gradient = gradient_vectors[unit_name]
        gradient_norm = gradient_norms[unit_name]
        rows[unit_name] = {}
        for direction_name in DIRECTIONS:
            direction = direction_units[direction_name]
            slope = None if direction is None else float((gradient*direction).sum())
            rows[unit_name][direction_name] = dict(
                raw_unit_direction_slope=slope,
                masked_gradient_l2=gradient_norm,
                cosine=None if gradient_norm == 0 or direction is None else slope/gradient_norm,
                direction_l2_before_unit_normalization=direction_norms[direction_name],
                unit_direction_status=("UNDEFINED_ZERO_DIRECTION" if direction is None else "DEFINED"),
                scalar_units_preserved=True,
            )
    return dict(
        status="COMPLETE_DIAGNOSTIC",
        directions={name: dict(masked_l2=direction_norms[name]) for name in DIRECTIONS},
        unit_direction_cosines=pairwise,
        gradient_geometry=dict(masked_l2=gradient_norms, gram=gradient_gram,
                               pairwise_cosines=gradient_cosines),
        projections=rows,
        gradient_normalization=False,
        direction_normalization=True,
        no_sign_or_success_gate=True,
        ceiling=CEILING,
    )
