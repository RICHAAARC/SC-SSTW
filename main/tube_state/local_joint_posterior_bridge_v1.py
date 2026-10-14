"""One fixed residual-compensated bridge from the original saved BASE.

The saved ``raw`` and ``reconstruction_residual`` tensors are consumed under
their completed terminal-bridge field semantics.  This module performs no
encode, gradient, solver search, or endpoint projection.
"""
from __future__ import annotations

VIEWS = ("BASE", "CAPPED", "POSTERIOR_BRIDGE")
RGB_VIEWS = ("BASE", "POSTERIOR_BRIDGE")
CAP = 1.0
LAMBDA = 1.0
NUMERICAL_RADIUS = 0.99998
CEILING = (
    "One known saved JOINT/BASE terminal, one known message, and the completed terminal-bridge "
    "raw/reconstruction_residual fields define one fixed residual-compensated masked direction. "
    "Only POSTERIOR_BRIDGE is newly decoded. No encode, VJP, solver, scan, second candidate, "
    "endpoint projection, DiT/native/scheduler step or codec. Full32/G are descriptive outcomes, "
    "not delivery gates. This run does not establish blind decoding, codec survival, FPR, "
    "generalization, global radius-1 feasibility or scientific PASS."
)


def _finite_fp32_tensor(value, shape, name):
    import torch

    if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape) or value.dtype != torch.float32:
        raise ValueError(f"{name} must be an FP32 tensor with shape {tuple(shape)}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


def masked(value, support):
    """Apply the adopted public support without importing the runtime layer."""
    import torch

    result = torch.zeros_like(value)
    t0, t1 = support.time_range
    for y0, y1, x0, x1 in support.rois:
        result[:, :, t0:t1, y0:y1, x0:x1] = value[:, :, t0:t1, y0:y1, x0:x1]
    return result


def construct_candidate(base, raw, reconstruction_residual, support):
    """Construct the sole FP32 endpoint and return its CPU geometry report."""
    import torch

    shape = tuple(base.shape)
    for name, value in (("base", base), ("raw", raw),
                        ("reconstruction_residual", reconstruction_residual)):
        _finite_fp32_tensor(value, shape, name)
    support.validate_for(shape)

    old_direction = masked(raw.double(), support)
    direction = masked(raw.double() - LAMBDA * reconstruction_residual.double(), support)
    old_norm = old_direction.norm()
    direction_norm = direction.norm()
    if not bool(torch.isfinite(direction_norm)) or float(direction_norm) == 0:
        raise ValueError("residual-compensated masked direction must have nonzero finite norm")

    unit = direction / direction_norm
    terminal = (base.double() + NUMERICAL_RADIUS * unit).float()
    actual = terminal.double() - base.double()
    outside = actual - masked(actual, support)
    radius = float(actual.norm())
    if not bool(torch.isfinite(terminal).all()):
        raise ValueError("candidate contains nonfinite values")
    if torch.count_nonzero(outside):
        raise ValueError("candidate changes values outside the adopted support")
    if radius > CAP:
        raise ValueError(f"stored FP32 candidate exceeds BASE radius 1: {radius}")

    if float(old_norm) == 0:
        cosine = None
        distance = None
    else:
        old_unit = old_direction / old_norm
        cosine = float((old_unit * unit).sum())
        distance = float((old_unit - unit).norm())

    t0, t1 = support.time_range
    roi_norms = []
    for index, (y0, y1, x0, x1) in enumerate(support.rois):
        roi_norms.append(dict(index=index,
            old=float(old_direction[:, :, t0:t1, y0:y1, x0:x1].norm()),
            new=float(direction[:, :, t0:t1, y0:y1, x0:x1].norm())))
    channel_norms = [dict(channel=index, old=float(old_direction[:, index].norm()),
                          new=float(direction[:, index].norm()))
                     for index in range(shape[1])]
    report = dict(
        status="FIXED_POSTERIOR_BRIDGE_CANDIDATE",
        formula="z + 0.99998 * normalize(M(raw - reconstruction_residual))",
        start="original saved JOINT/BASE terminal",
        lambda_residual=LAMBDA,
        cap=CAP,
        numerical_radius=NUMERICAL_RADIUS,
        scan=False,
        endpoint_projection=False,
        source_semantics=(
            "raw and reconstruction_residual are reused under their completed-run field names; "
            "this pair does not independently rederive the encoder identities"
        ),
        independent_algebraic_source_validation=False,
        raw_l2=float(raw.double().norm()),
        reconstruction_residual_l2=float(reconstruction_residual.double().norm()),
        old_masked_direction_l2=float(old_norm),
        new_masked_direction_l2=float(direction_norm),
        old_new_unit_cosine=cosine,
        old_new_unit_l2_distance=distance,
        roi_norms=roi_norms,
        channel_norms=channel_norms,
        fp32=dict(BASE_radius=radius, outside_mask_max=float(outside.abs().max()),
                  finite=True, within_cap=radius <= CAP),
        ceiling=CEILING,
    )
    return terminal, report
