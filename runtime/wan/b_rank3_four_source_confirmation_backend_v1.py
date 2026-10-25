"""Reuse the successful generation, FP32 VAE, DCT and codec adapter."""
import math
import torch
from main.tube_state import b_legacy_position_margin_v1 as state
from runtime.wan.b_terminal_three_group_backend_v1 import WanBackend as Base

class WanBackend(Base):
    def quality(self, left, right, source_frames, pair):
        """Direct full RGB difference; nominal support is payload/state union."""
        if pair != 'OFF_to_PM':return super().quality(left,right,source_frames,pair)
        from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame, _stats, _combine
        if tuple(left.shape) != tuple(right.shape) or tuple(left.shape)[1:] != (320, 512, 3):
            raise ValueError('matching actual RGB arrays required')
        if len(source_frames) != len(left) or len(left) not in (151, 181):
            raise ValueError('complete post-only frame map required')
        regions = tuple(dict.fromkeys(((8,24,8,24),(8,24,40,56))+tuple(state.ROIS)))
        frames, adjacent = [], []
        previous = previous_mask = None
        for t, source in enumerate(source_frames):
            a, b = _frame(left[t]), _frame(right[t])
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError('nonfinite quality input')
            delta = b-a
            mask = torch.zeros((320, 512, 3), dtype=torch.bool)
            for y0, y1, x0, x1 in regions:
                if (y0,y1,x0,x1) in ((8,24,8,24),(8,24,40,56)) or 1 <= source <= 176:
                    mask[y0*8:y1*8, x0*8:x1*8] = True
            frames.append(dict(frame=t, source_frame=source, whole=_stats(delta),
                               outside_nominal_support=_stats(delta[~mask]),
                               base_range=[float(a.min()), float(a.max())],
                               changed_range=[float(b.min()), float(b.max())]))
            if previous is not None:
                adjacent.append(dict(frame_before=t-1, frame_after=t,
                                     source_before=source_frames[t-1], source_after=source,
                                     crosses_deletion=source != source_frames[t-1]+1,
                                     whole=_stats(delta-previous),
                                     outside_union_nominal_support=_stats((delta-previous)[~(mask | previous_mask)])))
            previous, previous_mask = delta, mask
        whole = _combine([r['whole'] for r in frames])
        return dict(status='SCORED', frame_denominator=len(frames), adjacent_denominator=len(adjacent),
                    frames=frames, adjacent=adjacent, whole=whole,
                    outside_nominal_support=_combine([r['outside_nominal_support'] for r in frames]),
                    adjacent_residual=_combine([r['whole'] for r in adjacent]),
                    adjacent_outside_union_nominal_support=_combine([r['outside_union_nominal_support'] for r in adjacent]),
                    psnr_db_range1=None if whole['rms'] == 0 else -20*math.log10(whole['rms']),
                    nominal_regions=[list(r) for r in regions], mask_is_actual_causal_support=False,
                    quality_pass=None)
