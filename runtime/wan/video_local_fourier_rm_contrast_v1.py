"""Native four-arm masked union control; real fifty-step generation only."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from main.tube_state import video_local_fourier_rm_contrast_v1_control as method
from main.tube_state import video_local_fourier_rm_lowband_v1_state as state
from runtime.wan import trajectory,io,vae as vae_adapter
from runtime.wan.video_temporal_sync_bridge import file_sha256,counted,execution_device_dtype,predict_branches


DIAGNOSTIC_ARRAYS=('conditional_clean','pilot_delta','z_pre','z_post','cfg_clean')


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,record,*,diagnostic=None,direction=None):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh full native history required')
    z=initial.detach().float().clone();targets=method.build_targets(z,key,bits,arm);before=None
    last_projection=None;diagnostic_status=[]
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        if index==25:before=dict(z=trajectory.fingerprint(z),history=trajectory.fingerprint(vars(scheduler)),conditional=trajectory.fingerprint(c),unconditional=trajectory.fingerprint(u))
        sigma=float(scheduler.sigmas[index]);sigma_post=float(scheduler.sigmas[index+1])
        enabled=method.control_enabled(arm,index)
        capture={} if diagnostic is not None and index>=25 else None
        if enabled:count('local_control',False)
        velocity,row=method.guided_velocity(z,c,u,sigma,targets,arm,index,count,diagnostics=capture)
        if enabled:count('local_control',True)
        if capture is not None and 'error' not in capture:
            try:
                capture.update(conditional_clean=method.project_tensor(z-sigma*c.float(),targets['basis'],targets['blocks']),
                    z_pre=method.project_tensor(z,targets['basis'],targets['blocks']),
                    cfg_clean=method.project_tensor(z-sigma*velocity,targets['basis'],targets['blocks']))
            except Exception as exc:capture['error']=f'{type(exc).__name__}: {exc}'
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        if capture is not None:
            metadata=dict(index=index,sigma_pre=sigma,sigma_post=sigma_post,arm=arm,
                pilot_enabled=arm.startswith('STATE_'),method_version=method.VERSION,carrier_version=targets['carrier'],
                dtype='float32',shape=[45,4,4,8],role='writer diagnostic only; never blind input',
                cfg_clean_semantics='z_pre - sigma_pre * actual updated CFG velocity; clean estimate, not next state',
                derived_only='conditional_clean+pilot_delta: conditional projection after pilot; cfg_clean-5*pilot_delta: updated CFG projection without this pilot increment. Algebraic only, FP32 rounding; neither is a separately observed state')
            arrays=None
            try:
                if 'error' in capture:raise RuntimeError(capture['error'])
                capture['z_post']=method.project_tensor(z,targets['basis'],targets['blocks'])
                arrays={name:capture[name].detach().float().cpu().numpy() for name in DIAGNOSTIC_ARRAYS}
                if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite writer projection')
                metadata.update(status='COMPLETE',closure=capture['closure'])
                if index==49:last_projection=arrays['z_post'].copy()
            except Exception as exc:metadata.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
            # A save failure must not erase a successfully executed native step.
            try:diagnostic(index,arrays,metadata)
            except Exception as exc:metadata.update(status='FAILED',error=f'diagnostic persistence: {type(exc).__name__}: {exc}')
            diagnostic_status.append(metadata['status'])
            row['writer_diagnostic']=dict(status=metadata['status'],index=index,error=metadata.get('error'))
        if capture is not None and arm.startswith('STATE_') and direction is not None:
            try:direction(index,capture)
            except Exception as exc:row['direction_diagnostic']=dict(status='FAILED',error=f'{type(exc).__name__}: {exc}')
        record(dict(index=index,sigma=sigma,cursor_after=scheduler.step_index,**row))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    terminal_info=dict(status='NOT_REQUESTED',last_z_post_matches_terminal=None,terminal_projection_sha256=None)
    if diagnostic is not None:
        try:
            projection=method.project_tensor(z,targets['basis'],targets['blocks']).detach().float().cpu().numpy()
            if not np.isfinite(projection).all():raise ValueError('nonfinite terminal diagnostic projection')
            terminal_info.update(status='COMPLETE',last_z_post_matches_terminal=bool(np.array_equal(last_projection,projection)) if last_projection is not None else None,
                                 terminal_projection_sha256=file_array_sha256(projection))
        except Exception as exc:
            terminal_info.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    return z.detach().cpu(),dict(before_step25=before,final_history_sha256=trajectory.fingerprint(vars(scheduler)),
        terminal_sha256=trajectory.fingerprint(z),transformer_dtype=str(dtype),state_control_cfg='float32',
        method_version=method.VERSION,carrier_version=targets['carrier'],writer_diagnostics=dict(expected=25,completed=diagnostic_status.count('COMPLETE'),**terminal_info))


def file_array_sha256(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
