"""Frozen Wan/shared payload APIs only; no QIM execution or model work on import."""
from pathlib import Path
from runtime.wan.b_full_context_local_read_backend_v1 import FullBackend
from runtime.wan import b_lfg_posthoc_tensor_v1 as tensor
from main.tube_state import b_lfg_posthoc_reference_v1 as core
from main.tube_state import b_legacy_position_margin_v1 as payload
from main.tube_state.b_lfg_posthoc_plan_v1 import CRF,LLL_SHAPE

class LFGBackend(FullBackend):
    def initialize_PN(self):return core.initialize_books(LLL_SHAPE)
    def construct_message(self,z,bits,key):return tensor.construct(z,bits,key)
    def message_lll(self,z):return tensor.received_lll(z)
    def message_scores(self,low,key):return core.score_from_lll(low,key)
    def payload_spectrum(self,z):return payload.payload_spectrum(z)
    def payload_votes(self,spectrum,key):return payload.payload_from_spectrum(spectrum,key)
    def save_mp4(self,pixels,path):
        from runtime.wan.io import encode_rgb8
        path=Path(path);encode_rgb8(pixels,path,fps=8,crf=CRF)
        return dict(path=str(path),bytes=path.stat().st_size,fps=8,crf=CRF,codec='libx264',pixel_format='yuv420p',input='same saved uint8 RGB24; no requantization')

    def quality(self,left,right,source_frames,meaning):
        import math,torch
        from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame,_stats,_combine
        if tuple(left.shape)!=tuple(right.shape) or tuple(left.shape)!=(181,320,512,3) or source_frames!=list(range(181)):raise ValueError('matching complete181 quality arrays required')
        frames=[];adjacent=[];previous=None
        for t in range(181):
            delta=_frame(right[t])-_frame(left[t])
            if not bool(torch.isfinite(delta).all()):raise ValueError('nonfinite paired quality')
            frames.append(dict(frame=t,whole=_stats(delta)))
            if previous is not None:adjacent.append(dict(frame_before=t-1,frame_after=t,whole=_stats(delta-previous)))
            previous=delta
        whole=_combine([r['whole'] for r in frames])
        return dict(status='SCORED',meaning=meaning,frame_denominator=181,adjacent_denominator=180,frames=frames,adjacent=adjacent,whole=whole,adjacent_residual=_combine([r['whole'] for r in adjacent]),psnr_db_range1=-20*math.log10(whole['rms']) if whole['rms'] else None,spatial_support='global LLL footprint; no historical ROI mask',quality_pass=None)
