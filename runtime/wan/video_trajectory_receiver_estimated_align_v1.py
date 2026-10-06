"""Fresh framewise sync and isolated length-aware detailed Wan reads."""
import hashlib
from main.tube_state import grow_video_reference as layout
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as method
from runtime.wan.video_trajectory_receiver_origin_v1 import ReceiverBackend as OriginBackend
from runtime.wan.video_trajectory_payload_gt_v1 import read_payload

def read_detailed(normalized,key,frames):
    import torch
    times=(frames-1)//4+1;R=method.support(frames)
    if tuple(normalized.shape)!=(1,16,times,40,64) or not bool(torch.isfinite(normalized).all()):
        raise ValueError("finite normalized latent with public length required")
    original=read_payload(normalized,key,frames)
    spectrum=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm="ortho").real
    coords=layout.coordinates(key);h=[x[0] for x in coords];w=[x[1] for x in coords]
    selected=torch.stack([spectrum[0,ch,1:R+1,h,w] for ch in range(4)])
    detail=method.detailed_votes((2*(selected>0).to(torch.int8)-1).cpu().numpy(),coords,
                                (selected==0).to(torch.int8).cpu().numpy(),frames)
    if original["decoded_bits"]!=[x["decoded"] for x in detail["bit_rows"]] or original["votes"]!=[
        {k:x[k] for k in ("ones","zeros","count")} for x in detail["bit_rows"]]:
        raise RuntimeError("detailed evidence mismatches unchanged canonical reader")
    detail.update(original_readout=original,original_reader_match=True,key_id=hashlib.sha256(key.encode()).hexdigest())
    return detail

class WanBackend(OriginBackend):
    @staticmethod
    def operate_clip(rgb,phase):
        import torch
        if rgb.dtype!=torch.uint8 or rgb.ndim!=4 or rgb.shape[-1]!=3:raise ValueError("uint8 received RGB required")
        indices=torch.tensor(method.phase_map(int(rgb.shape[0]),phase),device=rgb.device)
        return rgb.index_select(0,indices).contiguous().clone()
    read=staticmethod(read_detailed)

class FramewiseBackend:
    def __init__(self,config):
        import torch
        from runtime.wan.framewise_autoencoder_kl import load_frozen_framewise_vae
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.vae=load_frozen_framewise_vae(device=self.device)
        self.batch_frames=config["batch_frames"]
    def encode(self,rgb):
        from runtime.wan.video_trajectory_payload_framewise_sync_v1 import encode_received_video
        return encode_received_video(rgb,self.vae,batch_frames=self.batch_frames,public=method.PUBLIC)
    def score(self,latent,key):
        return method.sync.score_received_latent(latent,key,method.PUBLIC)
    def close(self):
        import gc,torch
        self.vae=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
