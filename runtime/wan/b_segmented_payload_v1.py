"""Same native 50-step payload chain with an actual temporal target schedule."""
from main.tube_state import b_segmented_payload_v1 as layout
from main.tube_state import grow_video_reference as payload
import torch, math

from runtime.wan.conditional_joint import trajectory


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,key,bits,count,record):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh native scheduler history required')
    z=initial.detach().float().clone();target,mask=layout.build_target(z,key,bits)
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        sigma=float(scheduler.sigmas[index])
        velocity,detail=guided_velocity(z,c,u,sigma,target,mask,index,count)
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        record(dict(index=index,sigma=sigma,cursor_after=scheduler.step_index,**detail))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    return z.detach().cpu(),dict(arm='B_SEGMENTED_NATIVE',steps=50,active_indices=list(range(25,50)),
        segment_source_times=[[1,11],[12,22],[23,33],[34,44]],net_bits=32,mask_count=44160,
        alpha=0.5,reference_eta=200.0,actual_eta=5520.0,guard_times=[0,45],
        writer='actual distinct temporal target in generation; original local FFT/Flow/CFG/native scheduler')


def guided_velocity(z,conditional,unconditional,sigma,target,mask,index,count):
    """Exact PAYLOAD_MULTI active arithmetic; no pilot or old state dependency."""
    import torch
    if type(index) is not int or not 0<=index<50 or not math.isfinite(float(sigma)) or sigma<=0:raise ValueError("fixed native step/sigma")
    z,c,u=z.float(),conditional.float(),unconditional.float()
    enabled=index>=25;detail=dict(enabled=enabled,arm="PAYLOAD_MULTI",state_control_cfg="float32")
    if enabled:
        count("payload_gradient",False)
        dp,pinfo=payload.local_delta(z-float(sigma)*c,target,mask)
        count("payload_gradient",True)
        delta=dp+torch.zeros_like(dp) # retain successful payload-only branch arithmetic
        c=c-delta/float(sigma)
        detail.update(payload=pinfo,pilot=None,payload_delta_l2=float(dp.double().square().sum().sqrt()))
    velocity=u+5.0*(c-u)
    if not bool(torch.isfinite(velocity).all()):raise FloatingPointError("nonfinite CFG")
    return velocity.detach(),detail

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
