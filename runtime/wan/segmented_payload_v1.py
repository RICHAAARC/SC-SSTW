"""Same native 50-step payload chain with an actual temporal target schedule."""
from main.tube_state import segmented_payload_v1 as layout
from runtime.wan.grow_video_reference import predict_branches
from runtime.wan.conditional_joint import trajectory


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,key,bits,count,record):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh native scheduler history required')
    z=initial.detach().float().clone();target,mask=layout.build_target(z,key,bits)
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        sigma=float(scheduler.sigmas[index])
        velocity,detail=layout.guided_velocity(z,c,u,sigma,target,mask,index,count)
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        record(dict(index=index,sigma=sigma,cursor_after=scheduler.step_index,**detail))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    return z.detach().cpu(),dict(arm='SEGMENTED_NATIVE',steps=50,active_indices=list(range(25,50)),
        segment_source_times=[[1,11],[12,22],[23,33],[34,44]],net_bits=32,mask_count=44160,
        alpha=0.5,reference_eta=200.0,actual_eta=5520.0,guard_times=[0,45],
        writer='actual distinct temporal target in generation; original local FFT/Flow/CFG/native scheduler')


def read_received_frames(vae, received_rgb8, key, received_to_source):
    """Apply the given time map to actual RGB before posterior-mode encoding.

    Synthetic copies provide encoder context only. The core reader excludes
    cells with any unrepresented source frame. This is conditional readout,
    not an estimator or an authentication/detection rule.
    """
    import torch
    from runtime.wan.vae import reencode_rgb24_readback
    plan=layout.segment_plan(received_to_source)
    if received_rgb8.dtype != torch.uint8 or tuple(received_rgb8.shape) != (len(plan['estimated_path']),320,512,3):
        raise ValueError('uint8 received RGB [len(map),320,512,3] required')
    operation=plan['operation']
    if operation['status'] != 'SUPPORTED':
        raise ValueError('supplied span has no complete Wan support')
    indices=torch.tensor(operation['received_index_map'],device=received_rgb8.device)
    mapped=received_rgb8.index_select(0,indices).float()/255.0
    normalized=reencode_rgb24_readback(vae,mapped)
    return layout.read_segments(normalized,key,plan['estimated_path'])
