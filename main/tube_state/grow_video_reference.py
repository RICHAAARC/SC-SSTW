"""Fixed FFT-real image-to-Wan reference: local mean-MSE, no tail gradient."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class PublicProtocol:
    latent_shape: tuple = (1, 16, 46, 40, 64)
    video_shape: tuple = (181, 320, 512, 3)
    channels: tuple = (0, 1, 2, 3)
    payload_bits: int = 32
    band_min: float = 0.2
    band_max: float = 0.5


PUBLIC = PublicProtocol()
ARMS = ("OFF", "MULTI", "LAST")
LAYERS = ("terminal", "float_rgb", "rgb8", "mp4")
REFERENCE_MASK_COUNT = 1600
REFERENCE_ETA = 200.0
ALPHA = 0.5


def message_bits(message):
    return [int(bit) for byte in message.encode("utf-8") for bit in f"{byte:08b}"]


def _validate(latent, public):
    import torch
    if public != PUBLIC or tuple(latent.shape) != public.latent_shape:
        raise ValueError("fixed public video/latent protocol required")
    if not latent.is_floating_point() or not bool(torch.isfinite(latent).all()):
        raise FloatingPointError("finite normalized floating latent required")


def coordinates(key, public=PUBLIC):
    import numpy as np
    height,width=public.latent_shape[-2:]
    coords=[(h,w) for h in range(int(height*public.band_min),int(height*public.band_max))
                  for w in range(int(width*public.band_min),int(width*public.band_max))]
    np.random.default_rng(sum(map(ord,key))).shuffle(coords)
    return coords


def layout_receipt(key, wrong_key, public=PUBLIC):
    def one(value):
        coords=coordinates(value,public)
        assignment=sorted((h,w,i%8) for i,(h,w) in enumerate(coords))
        digest=lambda x:hashlib.sha256(json.dumps(x,separators=(",",":")).encode()).hexdigest()
        return dict(seed=sum(map(ord,value)),coordinate_sha256=digest(coords),
                    assignment_sha256=digest(assignment),capacity=len(coords))
    right,wrong=one(key),one(wrong_key)
    if (right["seed"]==wrong["seed"] or right["assignment_sha256"]==wrong["assignment_sha256"]):
        raise ValueError("wrong-key seed/coordinate-to-bit assignment collision")
    return dict(correct=right,wrong=wrong,mask_support_same=True,assignment_different=True)


def build_target(latent,key,bits,public=PUBLIC):
    """Repeat the same eight bits/channel over all 46 latent times."""
    import torch
    _validate(latent,public)
    if len(bits)!=32 or any(type(x) is not int or x not in (0,1) for x in bits):
        raise ValueError("writer requires exactly 32 binary payload bits")
    target=torch.zeros_like(latent)
    mask=torch.zeros_like(latent,dtype=torch.bool)
    coords=coordinates(key,public)
    for i,ch in enumerate(public.channels):
        for j,(h,w) in enumerate(coords):
            target[0,ch,:,h,w]=(2*bits[i*8+j%8]-1)*ALPHA
            mask[0,ch,:,h,w]=True
    return target,mask


def local_delta(clean,target,mask,*,reference_eta=REFERENCE_ETA):
    """Differentiate through the entire real FFT; mean includes C,T and band.

    The local clean estimate is a fresh leaf. This deliberate video adapter
    boundary does not differentiate through denoisers or prior native steps.
    """
    import torch
    import torch.nn.functional as F
    if clean.shape!=target.shape or mask.shape!=clean.shape or mask.dtype!=torch.bool:
        raise ValueError("target/mask shape mismatch")
    count=int(mask.sum().item())
    if count!=44160 or not math.isfinite(reference_eta) or reference_eta<0:
        raise ValueError("fixed actual mask count and nonnegative reference step required")
    eta=reference_eta*count/REFERENCE_MASK_COUNT
    with torch.enable_grad():
        leaf=clean.detach().clone().requires_grad_(True)
        spectrum=torch.fft.fft2(leaf,dim=(-2,-1),norm="ortho").real
        loss=F.mse_loss(spectrum[mask],target[mask])
        gradient=torch.autograd.grad(loss,leaf,create_graph=False)[0]
        delta=(-eta*gradient).detach()
    if not bool(torch.isfinite(delta).all()) or not bool(torch.isfinite(loss)):
        raise FloatingPointError("nonfinite local loss/gradient")
    return delta,dict(mask_count=count,eta=eta,loss=float(loss.detach()),
        gradient_rms=float(gradient.detach().double().square().mean().sqrt()),
        conditional_clean_delta_rms=float(delta.double().square().mean().sqrt()),
        gradient_kind="full_local_FFT_real_mean_MSE_wrt_conditional_x0",
        denoiser_backward=False,tail_backward=False)


def guided_velocity(z,conditional,unconditional,sigma,target,mask,*,enabled,
                    guidance=5.0,reference_eta=REFERENCE_ETA):
    """Flow x0=z-sigma*v; conditional update precedes one FP32 CFG."""
    import torch
    if not math.isfinite(float(sigma)) or float(sigma)<=0:
        raise ValueError("positive native input sigma required")
    z,c,u=z.float(),conditional.float(),unconditional.float()
    receipt=dict(enabled=bool(enabled),branch_dtype="float32",cfg_dtype="float32")
    if enabled:
        delta,detail=local_delta(z-float(sigma)*c,target,mask,reference_eta=reference_eta)
        # Algebraic Flow mapping preserves exact zero-delta equivalence to OFF.
        c=c-delta/float(sigma)
        receipt.update(detail)
        receipt["cfg_clean_delta_rms"]=abs(guidance)*detail["conditional_clean_delta_rms"]
    velocity=u+guidance*(c-u)
    if not bool(torch.isfinite(velocity).all()):
        raise FloatingPointError("nonfinite FP32 CFG velocity")
    return velocity.detach(),receipt


def control_enabled(arm,index):
    if arm not in ARMS or not 0<=index<50:
        raise ValueError("fixed arm and 50-step index required")
    return (arm=="MULTI" and index>=25) or (arm=="LAST" and index==49)


def read_latent_bits(normalized,key,public=PUBLIC):
    """Truth-free diagnostic primitive; no writer target or message argument."""
    import torch
    _validate(normalized,public)
    with torch.no_grad():
        spectrum=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm="ortho").real
        coords=coordinates(key,public)
        h=[x[0] for x in coords];w=[x[1] for x in coords]
        decoded=[];vote_rows=[]
        for channel in public.channels:
            raw=(spectrum[0,channel,:,h,w]>0).to(torch.int64).cpu().numpy()
            # Time-major then coordinate repetition, matching first-vote tie rule.
            for bit in range(8):
                votes=raw[:,bit::8].reshape(-1).tolist()
                decoded.append(Counter(votes).most_common(1)[0][0])
                ones=sum(votes);vote_rows.append(dict(ones=ones,zeros=len(votes)-ones,
                    count=len(votes),tie=ones*2==len(votes)))
    return dict(status="READ",decoded_bits=decoded,votes=vote_rows,truth_used=False,
        payload_bits=32,temporal_interpretation="repeat voting over all46; no synchronization",
        normalized_shape=list(normalized.shape),transform="fft2_ortho_real")
