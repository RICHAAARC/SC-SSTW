"""Thin replay adapter: unchanged Wan scaling/cache/codec, extra preclamp statistics only."""
from __future__ import annotations
import hashlib
from pathlib import Path
from runtime.wan import vae as shared_vae,io

def file_sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def decode_with_clamp_receipt(frozen_vae,normalized):
    import torch
    if tuple(normalized.shape)!=(1,16,46,40,64):raise ValueError('fixed full terminal required')
    device=next(frozen_vae.parameters()).device
    if next(frozen_vae.parameters()).dtype!=torch.float32:raise ValueError('fixed FP32 VAE required')
    normalized=normalized.to(device=device,dtype=torch.float32)
    mean,std=shared_vae._scale_tensors(frozen_vae,normalized)
    shared_vae._clear_cache(frozen_vae)
    try:
        with torch.inference_mode():decoded=frozen_vae.decode(normalized*std+mean,return_dict=False)[0]
    finally:shared_vae._clear_cache(frozen_vae)
    if tuple(decoded.shape)!=(1,3,181,320,512):raise ValueError('decoded full RGB geometry mismatch')
    # Exactly the shared adapter's FP32 mapping; reductions do not modify this tensor.
    rgb=decoded[0].permute(1,2,3,0)/2.+.5
    low=high=0;under=over=absolute=0.
    for frame in rgb:
        if not bool(torch.isfinite(frame).all()):raise ValueError('nonfinite preclamp RGB')
        low+=int((frame<0).sum());high+=int((frame>1).sum())
        under=max(under,max(0.,-float(frame.min())));over=max(over,max(0.,float(frame.max())-1.))
        absolute+=float((frame-frame.clamp(0,1)).abs().double().sum())
    receipt=dict(status='MEASURED',domain='decoded/2+.5 before clamp; not raw VAE [-1,1]',
        dtype=str(rgb.dtype),shape=list(rgb.shape),finite=True,fraction_below0=low/rgb.numel(),
        fraction_above1=high/rgb.numel(),maximum_underflow=under,maximum_overflow=over,
        mean_absolute_clamp_change=absolute/rgb.numel(),scalar_elements=rgb.numel())
    return rgb.clamp(0,1).detach().cpu().contiguous(),receipt

def validate_rgb(rgb,dtype):
    if tuple(rgb.shape)!=(181,320,512,3) or rgb.dtype!=dtype:raise ValueError('fixed RGB dtype/geometry required')
    if not bool(rgb.isfinite().all()):raise ValueError('nonfinite RGB')

def encode_raster(q8,path):
    """Keep the old io.encode_rgb byte raster exactly; no custom codec invocation."""
    import torch
    validate_rgb(q8,torch.uint8)
    restored=q8.float()/255.
    if not torch.equal(shared_vae.quantize_rgb8_no_codec(restored),q8):raise ValueError('Q8/255 roundtrip changed codec raster')
    io.encode_rgb(restored,Path(path),8,18)

def read_full_mp4(path):
    import torch
    rgb=io.read_mp4(Path(path));validate_rgb(rgb,torch.float32)
    return rgb

def encode_normalized(frozen_vae,rgb):
    import torch
    validate_rgb(rgb,torch.float32)
    z=shared_vae.reencode_rgb24_readback(frozen_vae,rgb)
    if tuple(z.shape)!=(1,16,46,40,64) or z.dtype!=torch.float32 or not bool(z.isfinite().all()):
        raise ValueError('fixed full normalized geometry/dtype required')
    return z.detach().cpu()
