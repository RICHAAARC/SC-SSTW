"""Native Wan trajectories and public-only MP4 payload reception."""
from __future__ import annotations
import hashlib
from pathlib import Path

from main.tube_state import grow_video_reference as method
from runtime.wan import trajectory
from runtime.wan import io, vae as vae_adapter


def execution_device_dtype():
    import torch
    if not torch.cuda.is_available():return "cpu",torch.float32
    return "cuda",torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def counted(count,kind,function):
    if count: count(kind,False)
    value=function()
    if count: count(kind,True)
    return value


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


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


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,
                   record_step,*,reference_eta=method.REFERENCE_ETA,injected_control=None):
    """Caller supplies a separate complete pristine scheduler for each arm."""
    import torch
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:
        raise ValueError("arm must start with pristine complete native history")
    z=initial.detach().float().clone()
    if injected_control is None:
        target,mask=method.build_target(z,key,bits)
    elif not callable(injected_control):
        raise TypeError("injected control must be callable")
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        sigma=float(scheduler.sigmas[index])
        if injected_control is None:
            enabled=method.control_enabled(arm,index)
            if enabled: count("local_gradient",False)
            velocity,update=method.guided_velocity(z,c,u,sigma,target,mask,
                enabled=enabled,guidance=5.0,reference_eta=reference_eta)
            if enabled: count("local_gradient",True)
        else:
            count("injected_joint_control",False)
            velocity,update=injected_control(z=z,conditional=c,unconditional=u,
                sigma=sigma,index=index,total_steps=50)
            count("injected_joint_control",True)
        before=scheduler.step_index
        # No reset, deepcopy or speculative tail inside an arm.
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind="native_step")
        record_step(dict(index=index,sigma=sigma,cursor_before=before,
            cursor_after=scheduler.step_index,**update))
    if scheduler.step_index!=50:
        raise RuntimeError("native trajectory did not complete50 steps")
    receipt=dict(scheduler_class=type(scheduler).__name__,
        scheduler_config=dict(scheduler.config),final_cursor=scheduler.step_index,
        final_history_sha256=trajectory.fingerprint(vars(scheduler)),
        terminal_sha256=trajectory.fingerprint(z),transformer_dtype=str(dtype),
        state_dtype=str(z.dtype),control_dtype="torch.float32",cfg_dtype="torch.float32")
    if injected_control is not None:
        receipt["control_adapter"]="injected_local_joint_state_payload"
    return z.detach().cpu(),receipt


def read_diagnostic_latent(normalized,keys,public=method.PUBLIC,*,count=None):
    """Separate diagnostic API; never used as the primary MP4 receiver."""
    feature=trajectory.fingerprint(normalized)
    rows={}
    for name,key in keys.items():
        try:
            row=counted(count,"bit_read",lambda:method.read_latent_bits(normalized,key,public))
        except Exception as exc:
            row=dict(status="FAILED",decoded_bits=None,truth_used=False,error=f"{type(exc).__name__}: {exc}")
        rows[name]=dict(**row,feature_sha256=feature,feature_kind="diagnostic_normalized_latent")
    return rows


def reencode_diagnostic_rgb(rgb,keys,public,frozen_vae,*,count=None):
    if tuple(rgb.shape)!=public.video_shape:
        raise ValueError("fixed181x320x512 RGB diagnostic required")
    normalized=counted(count,"vae_encode",lambda:vae_adapter.reencode_rgb24_readback(frozen_vae,rgb))
    return read_diagnostic_latent(normalized,keys,public,count=count)


def read_mp4_payload(path,keys,public,frozen_vae,*,count=None):
    """Primary receiver: file + public keys/protocol + frozen VAE only.

    Two keys share one observed file/one VAE feature. There is no terminal,
    writer RGB, payload truth, target, confidence search or carrier argument.
    ``count`` only records attempts/completions and cannot supply model inputs.
    """
    path=Path(path)
    before=file_sha256(path)
    received=counted(count,"mp4_read",lambda:io.read_mp4(path))
    if tuple(received.shape)!=public.video_shape:
        raise ValueError(f"MP4 geometry mismatch: {tuple(received.shape)}")
    normalized=counted(count,"vae_encode",lambda:vae_adapter.reencode_rgb24_readback(frozen_vae,received))
    del received
    feature=trajectory.fingerprint(normalized)
    rows={}
    for name,key in keys.items():
        try:
            row=counted(count,"bit_read",lambda:method.read_latent_bits(normalized,key,public))
        except Exception as exc:
            row=dict(status="FAILED",decoded_bits=None,truth_used=False,error=f"{type(exc).__name__}: {exc}")
        rows[name]=dict(**row,feature_sha256=feature,
            input_kind="saved_MP4_RGB24_posterior_mode_Wan_normalized",
            mp4_sha256=before,public_feature_shared_between_keys=True)
    if file_sha256(path)!=before:
        raise RuntimeError("MP4 bytes changed during primary read")
    return rows
