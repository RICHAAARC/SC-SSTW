"""Native Wan frontend and four-phase blind overlap-state MP4 adapter."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from main.tube_state import video_overlap_zero_mean_control as method
from main.tube_state import video_overlap_zero_mean_state as state
from runtime.wan import trajectory,io,vae as vae_adapter
from runtime.wan.video_temporal_sync_bridge import file_sha256,counted,execution_device_dtype,predict_branches


DIAGNOSTIC_ARRAYS=('conditional_clean','pilot_delta','z_pre','z_post','cfg_clean')


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,record,*,diagnostic=None):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh full native history required')
    z=initial.detach().float().clone();targets=method.build_targets(z,key,bits);before=None
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
                capture.update(conditional_clean=method.project_tensor(z-sigma*c.float(),targets['basis']),
                    z_pre=method.project_tensor(z,targets['basis']),
                    cfg_clean=method.project_tensor(z-sigma*velocity,targets['basis']))
            except Exception as exc:capture['error']=f'{type(exc).__name__}: {exc}'
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        if capture is not None:
            metadata=dict(index=index,sigma_pre=sigma,sigma_post=sigma_post,arm=arm,
                pilot_enabled=arm=='OVERLAP_MULTI',method_version=state.PUBLIC.method_version,
                dtype='float32',shape=[45,4,4,2],role='writer diagnostic only; never blind input',
                cfg_clean_semantics='z_pre - sigma_pre * actual updated CFG velocity; clean estimate, not next state',
                derived_only='conditional_clean+pilot_delta: conditional projection after pilot; cfg_clean-5*pilot_delta: updated CFG projection without this pilot increment. Algebraic only, FP32 rounding; neither is a separately observed state')
            arrays=None
            try:
                if 'error' in capture:raise RuntimeError(capture['error'])
                capture['z_post']=method.project_tensor(z,targets['basis'])
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
        record(dict(index=index,sigma=sigma,cursor_after=scheduler.step_index,**row))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    terminal_info=dict(status='NOT_REQUESTED',last_z_post_matches_terminal=None,terminal_projection_sha256=None)
    if diagnostic is not None:
        try:
            projection=method.project_tensor(z,targets['basis']).detach().float().cpu().numpy()
            if not np.isfinite(projection).all():raise ValueError('nonfinite terminal diagnostic projection')
            terminal_info.update(status='COMPLETE',last_z_post_matches_terminal=bool(np.array_equal(last_projection,projection)) if last_projection is not None else None,
                                 terminal_projection_sha256=file_array_sha256(projection))
        except Exception as exc:
            terminal_info.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
    return z.detach().cpu(),dict(before_step25=before,final_history_sha256=trajectory.fingerprint(vars(scheduler)),
        terminal_sha256=trajectory.fingerprint(z),transformer_dtype=str(dtype),state_control_cfg='float32',
        method_version=state.PUBLIC.method_version,writer_diagnostics=dict(expected=25,completed=diagnostic_status.count('COMPLETE'),**terminal_info))


def file_array_sha256(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def read_mp4_search(path,keys,public,frozen_vae,*,count=None,persist=None):
    """File transport only: names, source origins, writer truth and attack labels unused."""
    if public!=state.PUBLIC:raise ValueError('frozen public overlap protocol required')
    before=file_sha256(path);rgb=counted(count,'primary_mp4_read',lambda:io.read_mp4(Path(path)))
    if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(320,512,3):raise ValueError('received RGB geometry mismatch')
    length=len(rgb);specs=[method.phase_spec(length,g) for g in range(4)];features={};results={}
    def emit(kind,slot,row):
        if persist:persist(kind,slot,row)
    for spec in specs:
        g=spec['g'];R=spec['R']
        try:
            normalized=counted(count,'vae_encode',lambda:vae_adapter.reencode_rgb24_readback(frozen_vae,rgb[spec['left']:spec['right']]))
            if tuple(normalized.shape)!=(1,16,spec['regular_count']+1,40,64):raise ValueError('phase VAE shape mismatch')
            if not bool(normalized.isfinite().all()):raise ValueError('nonfinite normalized phase')
            emit('normalized',g,normalized.detach().cpu())
            received=normalized[0,:,1:R+1].detach().float().cpu().numpy().transpose(1,0,2,3)
        except Exception as exc:
            for name in keys:
                row=dict(status='FAILED',g=g,error=f'{type(exc).__name__}: {exc}',inference=None,payload=None)
                features[g,name]=row;emit('phase',(g,name),row)
            continue
        for name,key in keys.items():
            try:
                inference=counted(count,'phase_path_search',lambda:state.infer(received,key,np.ones((R,4),bool)))
                row=dict(status=inference['summary']['status'],g=g,received_frames=length,phase=spec,
                    feature_sha256=trajectory.fingerprint(normalized),mp4_sha256=before,inference=inference,payload=None)
                try:row['payload']=counted(count,'phase_payload_read',lambda:method.payload_read(normalized,key,R))
                except Exception as exc:row['payload']=dict(status='FAILED',decoded_bits=None,error=str(exc))
            except Exception as exc:row=dict(status='FAILED',g=g,error=f'{type(exc).__name__}: {exc}',inference=None,payload=None)
            features[g,name]=row;emit('phase',(g,name),row)
        del normalized,received
    del rgb
    identity_ok=file_sha256(path)==before
    for name in keys:
        phases={g:features[g,name]['inference'] for g in range(4) if features[g,name]['inference'] is not None}
        if not identity_ok:phases={}
        decision=counted(count,'primary_search',lambda:method.combine_phases(phases))
        decision.update(method_version=state.PUBLIC.method_version,mp4_sha256=before,received_frames=length,canonical_phase_payload_diagnostic=None)
        if decision['canonical'] is not None:
            payload=features[decision['canonical']['g'],name]['payload']
            decision['canonical_phase_payload_diagnostic']=payload.get('decoded_bits') if payload else None
        results[name]=decision;emit('search',name,decision)
    return dict(searches=results,features=features)

