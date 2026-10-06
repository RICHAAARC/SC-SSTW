"""Thin single-deletion receiver adapters; no experiment imports."""
from runtime.wan.video_trajectory_receiver_estimated_align_v1 import FramewiseBackend as BaseFramewise,WanBackend as BaseWan
from main.tube_state import video_trajectory_internal_single_deletion_v1 as method
class FramewiseBackend(BaseFramewise):
    score=staticmethod(method.score)
class WanBackend(BaseWan):
    @staticmethod
    def operate_map(rgb,indices):
        import torch
        if rgb.dtype!=torch.uint8 or len(rgb)!=177 or len(indices)!=177 or any(type(x) is not int or x<0 or x>=177 for x in indices):
            raise ValueError("fixed received177 input map required")
        return rgb.index_select(0,torch.tensor(indices,device=rgb.device)).contiguous().clone()
    @staticmethod
    def cache_latent(z):return z.detach().cpu().clone()
    def restore_latent(self,z):return z.to(self.device)
