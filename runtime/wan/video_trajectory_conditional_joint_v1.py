"""Isolated conditional joint runtime; no historical experiment/state imports."""
from __future__ import annotations
import gc,hashlib
import numpy as np
from main.tube_state import grow_video_reference as layout,payload_reader as payload_method
from main.tube_state import video_trajectory_conditional_joint_v1 as method
from main.tube_state import video_trajectory_receiver_estimated_align_v1 as receiver
from runtime.wan import framewise_autoencoder_kl as framewise,rgb8_source
from runtime.wan.conditional_joint import generation,vae,trajectory,fixed_rgb_media as media,quality

def execution_device_dtype():
    import torch
    if not torch.cuda.is_available():return "cpu",torch.float32
    return "cuda",torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

def counted(count,kind,function):
    if count: count(kind,False)
    value=function()
    if count: count(kind,True)
    return value

def predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count):
    import torch
    outputs=[]
    with torch.no_grad():
        for name,embedding in (("transformer_conditional",prompt),("transformer_unconditional",negative)):
            value=counted(count,name,lambda:pipe.transformer(
                hidden_states=z.to(dtype),timestep=scheduler.timesteps[index].expand(z.shape[0]),
                encoder_hidden_states=embedding,attention_kwargs=None,return_dict=False)[0])
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError("nonfinite conditional/unconditional branch")
            outputs.append(value.float())
    return outputs[0],outputs[1]

def read_payload(normalized, key, received_frames):
    """Public length fixes support; original FFT coordinates and Counter ties stay intact."""
    expected_times = (int(received_frames)-1)//4+1
    if received_frames not in (181,177,89) or int(normalized.shape[2]) != expected_times:
        raise ValueError("received frame/latent length mismatch")
    support = min(44,expected_times-1)
    row = payload_method.payload_read(normalized,key,support)
    row["received_frames"] = int(received_frames)
    row["support_rule"] = "R_eff=min(44,T_latent-1); first latent excluded"
    row["bit_rows"] = [
        dict(bit_index=index,decoded=int(decoded),ones=int(vote["ones"]),zeros=int(vote["zeros"]),
             count=int(vote["count"]),margin=abs(int(vote["ones"])-int(vote["zeros"])),
             normalized_margin=abs(int(vote["ones"])-int(vote["zeros"]))/int(vote["count"]))
        for index,(decoded,vote) in enumerate(zip(row["decoded_bits"],row["votes"]))
    ]
    return row

def read_detailed(normalized,key,frames):
    import torch
    times=(frames-1)//4+1;R=receiver.support(frames)
    if tuple(normalized.shape)!=(1,16,times,40,64) or not bool(torch.isfinite(normalized).all()):
        raise ValueError("finite normalized latent with public length required")
    original=read_payload(normalized,key,frames)
    spectrum=torch.fft.fft2(normalized.float(),dim=(-2,-1),norm="ortho").real
    coords=layout.coordinates(key);h=[x[0] for x in coords];w=[x[1] for x in coords]
    selected=torch.stack([spectrum[0,ch,1:R+1,h,w] for ch in range(4)])
    detail=receiver.detailed_votes((2*(selected>0).to(torch.int8)-1).cpu().numpy(),coords,
                                (selected==0).to(torch.int8).cpu().numpy(),frames)
    if original["decoded_bits"]!=[x["decoded"] for x in detail["bit_rows"]] or original["votes"]!=[
        {k:x[k] for k in ("ones","zeros","count")} for x in detail["bit_rows"]]:
        raise RuntimeError("detailed evidence mismatches unchanged canonical reader")
    detail.update(original_readout=original,original_reader_match=True,key_id=hashlib.sha256(key.encode()).hexdigest())
    return detail


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,key,bits,count,record):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError("fresh full native history required")
    z=initial.detach().float().clone();target,mask=layout.build_target(z,key,bits);before=None
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        if index==25:before=dict(z=trajectory.fingerprint(z),history=trajectory.fingerprint(vars(scheduler)),conditional=trajectory.fingerprint(c),unconditional=trajectory.fingerprint(u))
        sigma=float(scheduler.sigmas[index]);enabled=index>=25
        if enabled:count("local_control",False)
        velocity,row=method.guided_velocity(z,c,u,sigma,target,mask,index,count)
        if enabled:count("local_control",True)
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind="native_step")
        record(dict(index=index,sigma=sigma,cursor_after=scheduler.step_index,**row))
    if scheduler.step_index!=50:raise RuntimeError("native trajectory incomplete")
    return z.detach().cpu(),dict(before_step25=before,final_history_sha256=trajectory.fingerprint(vars(scheduler)),terminal_sha256=trajectory.fingerprint(z),transformer_dtype=str(dtype),arm="PAYLOAD_MULTI",pilot_gradient=0)

def release():
    import torch
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()

class Inputs:
    @staticmethod
    def read(spec):return rgb8_source.read_rgb8_source(spec["path"],expected_sha256=spec["sha256"],shape=tuple(spec["shape"]))
    @staticmethod
    def construct(full,indices):
        import torch
        if full.dtype!=torch.uint8 or full.device.type!="cpu" or tuple(full.shape)!=(181,320,512,3) or len(indices) not in (181,177,89) or any(type(x) is not int or x not in range(181) for x in indices):raise ValueError("public source/map geometry")
        return full.index_select(0,torch.tensor(indices)).contiguous().clone()
    @staticmethod
    def receipt(rgb):
        raw=rgb.detach().cpu().contiguous().numpy().tobytes()
        return dict(sha256=hashlib.sha256(raw).hexdigest(),shape=list(rgb.shape),bytes=len(raw),dtype="uint8")

class FramewiseBackend:
    def __init__(self,cfg):
        self.device,_=execution_device_dtype();self.vae=framewise.load_frozen_framewise_vae(device=self.device)
    def encode(self,rgb):return framewise.encode_rgb_frames(self.vae,rgb.float().div(255),batch_frames=8).numpy()
    @staticmethod
    def score(z,key,protocol):
        if protocol=="GLOBAL":return receiver.sync.score_received_latent(z,key,receiver.PUBLIC)
        if protocol=="SINGLE_JUMP":return method.deletion.score(z,key)
        raise ValueError("public protocol")
    @staticmethod
    def write(z,key):return receiver.sync.apply_projection_margin(z,key,receiver.sync.PUBLIC,target_margin=.5)
    def decode(self,z):
        import torch
        return vae.quantize_rgb8_no_codec(framewise.decode_rgb_frames(self.vae,torch.from_numpy(z.copy()),batch_frames=8))
    def close(self):self.vae=None;release()

class WanBackend:
    def __init__(self,cfg):
        self.device,_=execution_device_dtype();self.vae=generation.load_frozen_vae(cfg,device=self.device)
    @staticmethod
    def operate(rgb,indices):
        import torch
        n=len(rgb)
        if n not in (181,177,89) or rgb.dtype!=torch.uint8 or len(indices)!=n or any(type(x) is not int or x not in range(n) for x in indices):raise ValueError("received-only full map")
        return rgb.index_select(0,torch.tensor(indices,device=rgb.device)).contiguous().clone()
    def encode(self,rgb):return vae.reencode_rgb24_readback(self.vae,rgb.float().div(255))
    read=staticmethod(read_detailed)
    receipt=staticmethod(Inputs.receipt)
    @staticmethod
    def cache(z):return z.detach().cpu().clone()
    def restore(self,z):return z.to(self.device)
    def close(self):self.vae=None;release()
