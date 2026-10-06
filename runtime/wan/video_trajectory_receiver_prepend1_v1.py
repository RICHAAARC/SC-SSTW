"""Isolated public clip operation; unchanged origin Wan FP32-mode receiver."""
from main.tube_state import video_trajectory_receiver_prepend1_v1 as method
from runtime.wan.video_trajectory_receiver_origin_v1 import ReceiverBackend as OriginBackend
from runtime.wan.video_trajectory_receiver_origin_v1 import read_detailed

class ReceiverBackend(OriginBackend):
    @staticmethod
    def operate_clip(clip,operation):
        import torch
        if clip.dtype!=torch.uint8 or clip.ndim!=4 or clip.shape[0]!=177 or clip.shape[-1]!=3:
            raise ValueError("fixed received 177-frame uint8 RGB clip required")
        indices=torch.tensor(method.received_index_map(operation),device=clip.device)
        return clip.index_select(0,indices).contiguous().clone()

    read=staticmethod(read_detailed)
