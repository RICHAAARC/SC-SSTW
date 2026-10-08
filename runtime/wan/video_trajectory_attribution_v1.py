"""Runtime adapters for the frozen trajectory-attribution roster."""
from __future__ import annotations
import copy
import hashlib
from main.tube_state import grow_video_reference as layout
from main.tube_state import video_trajectory_attribution_v1 as method
from runtime.wan import video_trajectory_conditional_joint_v1 as joint

Inputs=joint.Inputs
FramewiseBackend=joint.FramewiseBackend
WanBackend=joint.WanBackend
generation=joint.generation
vae=joint.vae
media=joint.media
trajectory=joint.trajectory
release=joint.release
execution_device_dtype=joint.execution_device_dtype

def run_native_arm(pipe,initial,pristine,prompt,negative,dtype,arm,key,bits,count,record):
    """OFF is a true native trajectory; A/B preserve the adopted marked arithmetic."""
    if arm not in ("OFF","A","B"):raise ValueError("fixed OFF/A/B arm")
    scheduler=copy.deepcopy(pristine)
    if arm!="OFF":
        terminal,receipt=joint.run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,key,bits,count,record)
        receipt.update(attribution_arm=arm,message_sha256=hashlib.sha256(bytes(bits)).hexdigest())
        return terminal,receipt
    joint.trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError("fresh OFF history required")
    z=initial.detach().float().clone()
    for index in range(50):
        c,u=joint.predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        velocity=u+5.0*(c.float()-u.float())
        z=joint.trajectory.native_step(scheduler,z,velocity,index,count,kind="native_step")
        record(dict(index=index,sigma=float(scheduler.sigmas[index]),cursor_after=scheduler.step_index,
            enabled=False,arm="OFF",state_control_cfg="float32"))
    if scheduler.step_index!=50:raise RuntimeError("native OFF trajectory incomplete")
    return z.detach().cpu(),dict(attribution_arm="OFF",payload_enabled=False,
        final_history_sha256=joint.trajectory.fingerprint(vars(scheduler)),
        terminal_sha256=joint.trajectory.fingerprint(z),transformer_dtype=str(dtype))

def score_received(framewise_backend,latent,key,frames):
    if frames==177:return framewise_backend.score(latent,key,"SINGLE_JUMP")
    if frames in (181,89):return framewise_backend.score(latent,key,"GLOBAL")
    raise ValueError("public N must be 181/177/89")

def apply_action(rgb,action):
    if action is None:raise ValueError("unique action required before payload")
    return WanBackend.operate(rgb,action["received_index_map"])
