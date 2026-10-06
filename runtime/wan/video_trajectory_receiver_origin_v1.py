"""Wan-only receiver evidence, preserving the original payload reader."""
import hashlib
from main.tube_state import grow_video_reference as layout
from main.tube_state import payload_reader as original
from main.tube_state import video_trajectory_receiver_origin_v1 as method

def read_detailed(normalized,key):
    import torch
    if tuple(normalized.shape)!=(1,16,45,40,64) or not bool(torch.isfinite(normalized).all()):
        raise ValueError("fixed finite normalized [1,16,45,40,64] required")
    # This is the one method read. The additional FFT only records its same votes.
    reference=original.payload_read(normalized,key,44)
    spectrum=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm="ortho").real
    coords=layout.coordinates(key); h=[v[0] for v in coords];w=[v[1] for v in coords]
    selected=torch.stack([spectrum[0,ch,1:45,h,w] for ch in range(4)])
    detail=method.detailed_votes((2*(selected>0).to(torch.int8)-1).cpu().numpy(),
                                coords,(selected==0).to(torch.int8).cpu().numpy())
    expected_votes=[dict(ones=x["ones"],zeros=x["zeros"],count=x["count"]) for x in detail["bit_rows"]]
    if reference["votes"]!=expected_votes or reference["decoded_bits"]!=[x["decoded"] for x in detail["bit_rows"]]:
        raise RuntimeError("detailed votes do not reproduce unchanged Counter reader")
    detail["key_id"]=hashlib.sha256(key.encode()).hexdigest()
    detail["original_reader_match"]=True
    detail["original_readout"]=reference
    return detail

class ReceiverBackend:
    def __init__(self,model,device=None):
        import torch
        from runtime.wan.generation import load_frozen_vae
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vae=load_frozen_vae({"model":model},device=self.device)

    @staticmethod
    def read_source(spec):
        from runtime.wan.rgb8_source import read_rgb8_source
        return read_rgb8_source(spec["path"],expected_sha256=spec["sha256"],shape=tuple(spec["shape"]))

    @staticmethod
    def slice_source(source,start):
        import torch
        if source.dtype!=torch.uint8 or int(source.shape[0])!=181 or start not in (0,1):
            raise ValueError("fixed verified FULL181 uint8 and start0/1 required")
        return source[start:start+177].contiguous().clone()

    @staticmethod
    def pixel_receipt(rgb):
        data=rgb.detach().cpu().numpy().tobytes()
        return dict(shape=list(rgb.shape),dtype="uint8",bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest())

    def encode(self,received):
        from runtime.wan.vae import reencode_rgb24_readback
        return reencode_rgb24_readback(self.vae,received.float().div(255.0))

    read=staticmethod(read_detailed)

    def close(self):
        import gc,torch
        self.vae=None;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()

