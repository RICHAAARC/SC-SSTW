"""Only existing-media/E41 operations of the already successful B adapter."""
import math
import torch
from runtime.wan.b_terminal_three_group_backend_v1 import WanBackend as Base

class WanBackend(Base):
    def codec_quality(self,edited,readback):
        """Whole RGB generation residual. No watermark mask and no extra decode."""
        from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame,_stats,_combine
        self.validate_pixels(edited,151);self.validate_pixels(readback,151)
        frames=[];adjacent=[];previous=None
        for index in range(151):
            a,b=_frame(edited[index]),_frame(readback[index]);delta=b-a
            if not bool(torch.isfinite(delta).all()):raise ValueError('nonfinite codec quality')
            frames.append(dict(frame=index,whole=_stats(delta)))
            if previous is not None:adjacent.append(dict(frame_before=index-1,frame_after=index,whole=_stats(delta-previous)))
            previous=delta
        whole=_combine([r['whole'] for r in frames])
        return dict(status='SCORED',comparison='edited_uint8_vs_second_codec_readback',frame_denominator=151,adjacent_denominator=150,
            frames=frames,adjacent=adjacent,whole=whole,adjacent_residual=_combine([r['whole'] for r in adjacent]),
            psnr_db_range1=None if whole['rms']==0 else -20*math.log10(whole['rms']),support_mask=None,quality_pass=None)
