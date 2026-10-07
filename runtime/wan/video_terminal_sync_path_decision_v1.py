"""Receiver-only adapter; no experiment imports or receiver Wan/payload calls."""
from runtime.wan.video_trajectory_receiver_estimated_align_v1 import FramewiseBackend as Base
from main.tube_state import video_trajectory_internal_single_deletion_v1 as deletion
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as prior
class FramewiseBackend(Base):
    @staticmethod
    def score(latent,key):
        if len(latent)==177:return deletion.score(latent,key)
        if len(latent)==89:return prior.sync.score_received_latent(latent,key,prior.PUBLIC)
        raise ValueError("fixed177/89 received length required")

class InputBackend:
    @staticmethod
    def read_source(spec):
        from runtime.wan.rgb8_source import read_rgb8_source
        return read_rgb8_source(spec["path"],expected_sha256=spec["sha256"],shape=tuple(spec["shape"]))
    @staticmethod
    def construct(full,indices):
        import torch
        if full.dtype!=torch.uint8 or full.device.type!="cpu" or tuple(full.shape)!=(181,320,512,3) or len(indices) not in (177,89) or any(type(i) is not int or i not in range(181) for i in indices):raise ValueError("fixed full/cut geometry")
        return full.index_select(0,torch.tensor(indices)).contiguous().clone()
    @staticmethod
    def receipt(rgb):
        import hashlib
        raw=rgb.contiguous().numpy().tobytes()
        return dict(sha256=hashlib.sha256(raw).hexdigest(),shape=list(rgb.shape),bytes=len(raw))
