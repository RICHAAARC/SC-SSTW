"""Four native fifty-step trajectories with a writer-only OLD pilot norm schedule."""
from __future__ import annotations
import math
import numpy as np
from main.tube_state import video_local_fourier_rm_dwell4_v1_control as method
from main.tube_state import video_local_fourier_rm_dwell4_v1_state as state
from runtime.wan import trajectory
from runtime.wan.video_temporal_sync_bridge import execution_device_dtype,predict_branches
BASE_ARRAYS=('conditional_clean','pilot_delta','z_pre','z_post','cfg_clean')
DIAGNOSTIC_ARRAYS=BASE_ARRAYS+tuple(n+'_groups' for n in BASE_ARRAYS)

def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,record,*,diagnostic=None,matched_norm_schedule=None):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh full native history required')
    if arm=='STATE_DWELL4_MULTI':
        if matched_norm_schedule is None or set(matched_norm_schedule)!=set(range(25,50)) or any(not math.isfinite(float(v)) or float(v)<0 for v in matched_norm_schedule.values()):raise ValueError('complete same-run OLD 25-step physical pilot norm schedule required')
    z=initial.detach().float().clone();targets=method.build_targets(z,key,bits,arm);before=None;last_projection=None;last_groups=None;statuses=[]
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        if index==25:before=dict(z=trajectory.fingerprint(z),history=trajectory.fingerprint(vars(scheduler)),conditional=trajectory.fingerprint(c),unconditional=trajectory.fingerprint(u))
        sigma=float(scheduler.sigmas[index]);sigma_post=float(scheduler.sigmas[index+1]);enabled=method.control_enabled(arm,index)
        capture={} if diagnostic is not None and index>=25 else None
        if enabled:count('local_control',False)
        target=matched_norm_schedule[index] if arm=='STATE_DWELL4_MULTI' and enabled else None
        velocity,row=method.guided_velocity(z,c,u,sigma,targets,arm,index,count,diagnostics=capture,matched_norm=target)
        if enabled:count('local_control',True)
        if capture is not None and 'error' not in capture:
            try:
                for name,value in [('conditional_clean',z-sigma*c.float()),('z_pre',z),('cfg_clean',z-sigma*velocity)]:
                    groups=method.project_groups(value,targets['basis']);capture[name+'_groups']=groups;capture[name]=groups[0]
            except Exception as exc:capture['error']=f'{type(exc).__name__}: {exc}'
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        if capture is not None:
            metadata=dict(index=index,sigma_pre=sigma,sigma_post=sigma_post,arm=arm,pilot_enabled=arm.startswith('STATE_'),method_version=state.VERSION,carrier_version=targets['carrier'],dtype='float32',shape=[45,4,4,8],group_shape=[1,45,4,4,8],role='writer diagnostic only; never blind input',groups=[[list(b) for b in g] for g in state.GROUPS],cfg_clean_semantics='z_pre - sigma_pre * actual updated CFG velocity; clean estimate, not next state',derived_only='conditional_clean+pilot_delta: conditional projection after pilot; cfg_clean-5*pilot_delta: updated CFG projection without pilot increment. Algebraic only, not separately observed states',state_budget_target=target)
            arrays=None
            try:
                if 'error' in capture:raise RuntimeError(capture['error'])
                groups=method.project_groups(z,targets['basis']);capture['z_post_groups']=groups;capture['z_post']=groups[0]
                arrays={name:capture[name].detach().float().cpu().numpy() for name in DIAGNOSTIC_ARRAYS}
                if not all(np.isfinite(a).all() for a in arrays.values()):raise ValueError('nonfinite writer projection')
                metadata.update(status='COMPLETE',closure=capture['closure'])
                if index==49:last_projection=arrays['z_post'].copy();last_groups=arrays['z_post_groups'].copy()
            except Exception as exc:metadata.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
            try:diagnostic(index,arrays,metadata)
            except Exception as exc:metadata.update(status='FAILED',error=f'diagnostic persistence: {type(exc).__name__}: {exc}')
            statuses.append(metadata['status']);row['writer_diagnostic']=dict(status=metadata['status'],index=index,error=metadata.get('error'))
        record(dict(index=index,sigma=sigma,cursor_after=scheduler.step_index,**row))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    terminal_info=dict(status='NOT_REQUESTED',last_z_post_matches_terminal=None,terminal_projection_sha256=None,last_group_z_post_matches_terminal=None,terminal_group_projection_sha256=None)
    if diagnostic is not None:
        try:
            groups=method.project_groups(z,targets['basis']).detach().float().cpu().numpy();projection=groups[0]
            if not np.isfinite(groups).all():raise ValueError('nonfinite terminal projection')
            terminal_info.update(status='COMPLETE',last_z_post_matches_terminal=bool(np.array_equal(last_projection,projection)) if last_projection is not None else None,last_group_z_post_matches_terminal=bool(np.array_equal(last_groups,groups)) if last_groups is not None else None,terminal_projection_sha256=file_array_sha256(projection),terminal_group_projection_sha256=file_array_sha256(groups))
        except Exception as exc:terminal_info.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    return z.detach().cpu(),dict(before_step25=before,final_history_sha256=trajectory.fingerprint(vars(scheduler)),terminal_sha256=trajectory.fingerprint(z),transformer_dtype=str(dtype),state_control_cfg='float32',method_version=state.VERSION,carrier_version=targets['carrier'],state_budget_role='writer-only same-run OLD norm; never receiver input',writer_diagnostics=dict(expected=25,completed=statuses.count('COMPLETE'),**terminal_info))

def file_array_sha256(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
