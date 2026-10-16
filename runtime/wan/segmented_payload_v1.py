"""Same native 50-step payload chain with an actual temporal target schedule."""
from main.tube_state import segmented_payload_v1 as layout
from main.tube_state import video_trajectory_conditional_joint_v1 as method
from runtime.wan.video_trajectory_conditional_joint_v1 import predict_branches
from runtime.wan.conditional_joint import trajectory


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,key,bits,count,record):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh native scheduler history required')
    z=initial.detach().float().clone();target,mask=layout.build_target(z,key,bits)
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        sigma=float(scheduler.sigmas[index])
        velocity,detail=method.guided_velocity(z,c,u,sigma,target,mask,index,count)
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        record(dict(index=index,sigma=sigma,cursor_after=scheduler.step_index,**detail))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    return z.detach().cpu(),dict(arm='SEGMENTED_NATIVE',steps=50,active_indices=list(range(25,50)),
        segment_source_times=[[1,11],[12,22],[23,33],[34,44]],net_bits=32,mask_count=44160,
        alpha=0.5,reference_eta=200.0,actual_eta=5520.0,guard_times=[0,45],
        writer='actual distinct temporal target in generation; original local FFT/Flow/CFG/native scheduler')
