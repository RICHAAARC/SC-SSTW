"""Complete UniPC state snapshot/restore and the existing conditional-clean adapter."""
from __future__ import annotations
import copy
from runtime.wan import trajectory
from runtime.wan.local_joint_state_payload_v1 import apply_joint_control
from runtime.wan.local_joint_state_payload_provider_v1 import nominal_video_windows
from main.tube_state.local_joint_state_payload_v1 import JointControlRequest,JointControlResult,SamplingStepCoord
from main.tube_state import local_joint_paired_transfer_v1 as method

def tree(value,device="cpu"):
    import torch
    if torch.is_tensor(value):return value.detach().to(device).clone()
    if isinstance(value,dict):return {k:tree(v,device) for k,v in value.items()}
    if isinstance(value,list):return [tree(v,device) for v in value]
    if isinstance(value,tuple):return tuple(tree(v,device) for v in value)
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError("unsupported real scheduler-state value: "+type(value).__name__)

def snapshot(scheduler):
    from diffusers import UniPCMultistepScheduler
    if not isinstance(scheduler,UniPCMultistepScheduler):raise TypeError("actual UniPC multistep scheduler required")
    trajectory.validate_scheduler(scheduler)
    if scheduler.solver_p is not None:raise ValueError("external solver_p is not the native Wan interface")
    return dict(config=tree(dict(scheduler.config)),state=tree({k:v for k,v in vars(scheduler).items() if k!="_internal_dict"}))

def restore(bundle,device):
    from diffusers import UniPCMultistepScheduler
    scheduler=UniPCMultistepScheduler.from_config(bundle["config"])
    # No set_timesteps/reset: restore every saved runtime member, including model_outputs,
    # last_sample, this_order, lower_order_nums, timesteps/sigmas and cursor.
    for key,value in bundle["state"].items():setattr(scheduler,key,tree(value,device))
    trajectory.validate_scheduler(scheduler)
    return scheduler

def controlled_velocity(z,c,u,sigma,conditional_delta):
    request=JointControlRequest(SamplingStepCoord(48,50,sigma),nominal_video_windows(),
        {"kind":"unchanged_22_state"},{"message_hex":"8001a55a"},
        {"terminal_and_CFG_clean_amplitude":.10,"conditional_clean_amplitude":.02,
         "no_equivalence_to_old_cap":True})
    supplied=JointControlResult(conditional_delta,"saved_terminal_loss_direction_scaled_by_CFG",True)
    value,receipt=apply_joint_control(z,c,u,request,supplied,guidance=5.)
    zero=u.float()+5.*(c.float()-u.float())
    receipt.update(sigma=sigma,nominal_CFG_clean=method.measure(5.*conditional_delta),
                   realized_CFG_clean=method.measure(-sigma*(value-zero)),
                   coordinate_relation="delta_c=.10*d/5; delta_v_CFG=-5*delta_c/sigma; native effect is F(history,z,v+delta_v)-F(history,z,v), measured separately")
    return value,receipt
