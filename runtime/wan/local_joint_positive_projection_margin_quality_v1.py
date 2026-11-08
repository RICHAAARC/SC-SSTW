"""Observed paired/processing distortion; no perceptual acceptance threshold."""
import math
import torch

from main.tube_state.local_joint_positive_projection_margin_v1 import ROIS


def nominal_mask(frame, height, width):
    mask = torch.zeros((height, width, 3), dtype=torch.bool)
    if 1 <= frame < 177:
        for y0, y1, x0, x1 in ROIS:
            mask[y0*8:y1*8, x0*8:x1*8] = True
    return mask


def _frame(value):
    return value.detach().cpu().double() / (255 if value.dtype == torch.uint8 else 1)


def _stats(delta):
    n = delta.numel()
    ss = float(delta.square().sum())
    return dict(coordinates=n, sum_squared=ss, l2=math.sqrt(ss),
                rms=None if not n else math.sqrt(ss/n),
                linf=None if not n else float(delta.abs().max()),
                changed_coordinates=int(torch.count_nonzero(delta)))


def _combine(rows):
    n = sum(r["coordinates"] for r in rows)
    ss = math.fsum(r["sum_squared"] for r in rows)
    return dict(coordinates=n, sum_squared=ss, l2=math.sqrt(ss), rms=None if not n else math.sqrt(ss/n),
                linf=max((r["linf"] for r in rows if r["linf"] is not None), default=None),
                changed_coordinates=sum(r["changed_coordinates"] for r in rows))


def distortion(base, changed, *, meaning):
    if tuple(base.shape) != tuple(changed.shape) or base.ndim != 4 or base.shape[-1] != 3:
        raise ValueError("matching [T,H,W,3] observations required for distortion")
    frames, adjacent = [], []
    previous = previous_mask = None
    for t in range(base.shape[0]):
        left, right = _frame(base[t]), _frame(changed[t])
        if not bool(torch.isfinite(left).all() and torch.isfinite(right).all()):
            raise ValueError("nonfinite quality observation")
        delta = right-left
        mask = nominal_mask(t, *delta.shape[:2])
        frames.append(dict(frame=t, whole=_stats(delta), outside_nominal_support=_stats(delta[~mask]),
                           base_range=[float(left.min()), float(left.max())],
                           changed_range=[float(right.min()), float(right.max())]))
        if previous is not None:
            residual = delta-previous
            adjacent.append(dict(frame_before=t-1, frame_after=t, whole=_stats(residual),
                                 outside_union_nominal_support=_stats(residual[~(mask | previous_mask)])))
        previous, previous_mask = delta, mask
    whole = _combine([r["whole"] for r in frames])
    return dict(status="SCORED", meaning=meaning, frame_denominator=len(frames), adjacent_denominator=len(adjacent),
                frames=frames, adjacent=adjacent, whole=whole,
                outside_nominal_support=_combine([r["outside_nominal_support"] for r in frames]),
                adjacent_residual=_combine([r["whole"] for r in adjacent]),
                adjacent_outside_union_nominal_support=_combine([r["outside_union_nominal_support"] for r in adjacent]),
                psnr_db_range1=None if whole["rms"] == 0 else -20*math.log10(whole["rms"]),
                psnr_definition="range=1 including preclip; not a claim that preclip is bounded to [0,1]",
                mask_meaning="public nominal ROI times 8 pixels and [1,177) frames; not VAE causal support",
                quality_pass=None)
