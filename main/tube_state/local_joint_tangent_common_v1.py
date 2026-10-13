"""One fixed straight step in a saved pure tangent common direction."""
from __future__ import annotations

import math

BITS = (4, 12, 18, 20, 27, 30)
VIEWS = ("BASE", "START_FINAL", "TANGENT")
CEILING = ("One seen saved CONT_FINAL and its saved pure tangent common direction; one fixed "
           "straight step inside the original BASE radius 1. Tangency is at START only, not "
           "constant-radius motion. All 32 margins, state and float quality are observations, "
           "not protected quantities. No new VJP, search, projection, rollback, additional step, "
           "cap increase, codec, blind decoding, FPR, trajectory/CFG/native causality or scientific PASS.")


def candidate(base, start, direction, step_size):
    import torch
    if not math.isfinite(step_size) or step_size <= 0:
        raise ValueError("one finite positive fixed step required")
    if base.dtype != torch.float32 or start.dtype != torch.float32 or direction.dtype != torch.float64:
        raise ValueError("saved FP32 BASE/START and FP64 direction required")
    if base.shape != start.shape or start.shape != direction.shape:
        raise ValueError("BASE/START/direction shapes differ")
    if not all(bool(torch.isfinite(x).all()) for x in (base,start,direction)):
        raise ValueError("BASE/START/direction must be finite")
    delta = start.double()-base.double()
    radius, norm = float(delta.norm()), float(direction.norm())
    eps = torch.finfo(torch.float64).eps
    arithmetic_bound = 8*(direction.numel()+16)*eps
    radial_dot = float((delta*direction).sum())
    if radius == 0 or radius > 1:
        raise ValueError("START must define a radial axis inside the original BASE radius 1")
    if abs(norm-1) > arithmetic_bound or abs(radial_dot) > arithmetic_bound*radius:
        raise ValueError("saved direction must be unit length and tangent at actual START")
    planned = step_size*direction
    terminal = (start.double()+planned).to(start.dtype)
    actual = terminal.double()-start.double()
    actual_delta = terminal.double()-base.double()
    if float(actual_delta.norm()) > 1:
        raise ValueError("fixed straight candidate exceeds original BASE radius 1; no automatic projection or smaller step")
    return terminal, dict(status="DEFINED",step_size=step_size,start_radius=radius,
        actual_radius=float(actual_delta.norm()),direction_norm=norm,
        start_delta_dot_direction=radial_dot,planned_increment_l2=float(planned.norm()),
        actual_increment_l2=float(actual.norm()),addition_rounding_l2=float((actual-planned).norm()),
        actual_increment_dot_start_delta=float((actual*delta).sum()),
        no_projection=True, no_radius_fill=True, tangent_at="START_FINAL",
        direction_semantics="saved FP64 tangent_common; reused verbatim",
        ceiling=CEILING)
