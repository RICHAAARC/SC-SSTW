"""Same-chain quality with explicit C73 guard-inclusive nominal rectangles."""
import math
import torch
from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame, _stats, _combine, nominal_mask
from main.tube_state.c73_identity_v1 import ROIS


def distortion(base, changed, *, includes_guard, meaning):
    if tuple(base.shape) != tuple(changed.shape) or tuple(base.shape) != (181, 320, 512, 3):
        raise ValueError("three-arm quality requires matching full181 RGB")
    frames, adjacent = [], []
    previous = previous_mask = None
    for t in range(181):
        left, right = _frame(base[t]), _frame(changed[t])
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError("nonfinite quality observation")
        delta = right-left
        mask = nominal_mask(1 if includes_guard else t, 320, 512)
        frames.append(dict(frame=t, whole=_stats(delta), outside_nominal_support=_stats(delta[~mask])))
        if previous is not None:
            residual = delta-previous
            adjacent.append(dict(frame_before=t-1, frame_after=t, whole=_stats(residual),
                                 outside_union_nominal_support=_stats(residual[~(mask | previous_mask)])))
        previous, previous_mask = delta, mask
    whole = _combine([x["whole"] for x in frames])
    return dict(status="SCORED", frame_denominator=181, adjacent_denominator=180, frames=frames,
                adjacent=adjacent, whole=whole, meaning=meaning,
                outside_nominal_support=_combine([x["outside_nominal_support"] for x in frames]),
                adjacent_residual=_combine([x["whole"] for x in adjacent]),
                adjacent_outside_union_nominal_support=_combine([x["outside_union_nominal_support"] for x in adjacent]),
                mask_meaning="four public ROI rectangles times8; all181 frames including guards" if includes_guard else "legacy nominal four ROI rectangles times8 on frames1..176",
                mask_is_actual_causal_support=False,
                psnr_db_range1=None if whole["rms"] == 0 else -20*math.log10(whole["rms"]), quality_pass=None)


def preclip_record(value):
    if tuple(value.shape) != (181, 320, 512, 3) or not bool(torch.isfinite(value).all()):
        raise ValueError("finite full181 preclip RGB required")
    rows = []
    for t, frame in enumerate(value):
        below, above = frame < 0, frame > 1
        rows.append(dict(frame=t, minimum=float(frame.min()), maximum=float(frame.max()),
                         below_zero=int(below.sum()), above_one=int(above.sum()),
                         clipping_squared_distance=float((frame-frame.clamp(0, 1)).double().square().sum())))
    return dict(status="MEASURED", frame_denominator=181, rows=rows,
                minimum=min(x["minimum"] for x in rows), maximum=max(x["maximum"] for x in rows),
                below_zero=sum(x["below_zero"] for x in rows), above_one=sum(x["above_one"] for x in rows),
                clipping_squared_distance=math.fsum(x["clipping_squared_distance"] for x in rows))
