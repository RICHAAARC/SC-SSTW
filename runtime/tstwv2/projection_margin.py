"""Fixed public-protocol projection-margin write/read before the first Wan VAE decode.

This is an explicit adaptation of the old SyncTube support: all normalized
latent channels, temporal tubelets of four latent groups, and a tiled 4x4
spatial partition.  It intentionally has no sample/video identifier input.
"""
from __future__ import annotations
import hashlib,hmac
from typing import Any

PROTOCOL_NAMESPACE=b"SC-SSTW/WanProjectionMargin/v1"
TUBELET_LENGTH=4; PATCH=4; MARGIN=1.0; PAYLOAD_BITS=64

def _direction(key: bytes, reference_tubelet: int, length: int, device: Any) -> Any:
    import torch
    seed=int.from_bytes(hmac.new(key,PROTOCOL_NAMESPACE+b"/direction/"+reference_tubelet.to_bytes(4,"big"),hashlib.sha256).digest()[:8],"big")
    generator=torch.Generator(device="cpu").manual_seed(seed)
    v=torch.empty(length,dtype=torch.float32,device="cpu").uniform_(-1,1,generator=generator); return (v/v.norm()).to(device)

def payload_bits(message: bytes) -> list[int]:
    digest=hashlib.sha512(PROTOCOL_NAMESPACE+b"/payload/"+message).digest()
    return [(digest[i//8]>>(7-i%8))&1 for i in range(PAYLOAD_BITS)]

def _sync_sign(key:bytes,reference_tubelet:int)->int:
    return 1 if hmac.new(key,PROTOCOL_NAMESPACE+b"/sync/"+reference_tubelet.to_bytes(4,"big"),hashlib.sha256).digest()[0]&1 else -1

def supports(latent: Any):
    if latent.ndim!=5 or latent.shape[0]!=1: raise ValueError("requires [1,C,T,H,W] normalized Wan terminal")
    _,c,t,h,w=latent.shape
    if t<1+TUBELET_LENGTH or h%PATCH or w%PATCH: raise ValueError("incompatible Wan support geometry")
    for start in range(1,t-TUBELET_LENGTH+1,TUBELET_LENGTH):
        for y in range(0,h,PATCH):
            for x in range(0,w,PATCH): yield start,y,x,latent[0,:,start:start+TUBELET_LENGTH,y:y+PATCH,x:x+PATCH]

def write(normalized_latent: Any,key: bytes,message: bytes) -> tuple[Any,list[dict[str,float|int]]]:
    out=normalized_latent.detach().clone(); bits=payload_bits(message); records=[]
    for index,(start,y,x,block) in enumerate(supports(out)):
        reference=index; sign=(1 if bits[reference%PAYLOAD_BITS] else -1)*_sync_sign(key,reference); direction=_direction(key,reference,block.numel(),block.device).reshape_as(block); before=float((block.float()*direction).sum()); target=sign*MARGIN; delta=target-before if sign*before< MARGIN else 0.; block.copy_((block.float()+direction*delta).to(block.dtype)); after=float((block.float()*direction).sum()); records.append({"support":index,"reference_tubelet":reference,"tubelet_start":start,"y":y,"x":x,"projection_before":before,"projection_after":after})
    return out,records

def read(normalized_latent: Any,key: bytes) -> dict[str,Any]:
    votes=[[] for _ in range(PAYLOAD_BITS)]
    for index,(_,_,_,block) in enumerate(supports(normalized_latent)):
        votes[index%PAYLOAD_BITS].append(_sync_sign(key,index)*float((block.float()*_direction(key,index,block.numel(),block.device).reshape_as(block)).sum()))
    means=[sum(v)/len(v) if v else None for v in votes]
    return {"protocol_namespace":PROTOCOL_NAMESPACE.decode(),"projection_means":means,"bits":[None if x is None else int(x>=0) for x in means],"detector_write_support_weight":0.0}
