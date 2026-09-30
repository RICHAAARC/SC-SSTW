"""MULTI native trajectories and public-only window-exporting MP4 reception."""
from __future__ import annotations
from pathlib import Path
from main.tube_state import video_temporal_sync_multi as method
from runtime.wan import trajectory,io,vae as vae_adapter


from runtime.wan.video_temporal_sync_bridge import file_sha256,counted,execution_device_dtype,predict_branches


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,record):
    import torch
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh complete native scheduler required')
    z=initial.detach().float().clone();target,mask=method.build_target(z,key,bits,arm)
    before_first=None
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        if index==25:
            before_first=dict(z=trajectory.fingerprint(z),history=trajectory.fingerprint(vars(scheduler)),
                conditional=trajectory.fingerprint(c),unconditional=trajectory.fingerprint(u))
        enabled=method.control_enabled(arm,index)
        if enabled:count('local_gradient',False)
        velocity,row=method.guided_velocity(z,c,u,float(scheduler.sigmas[index]),target,mask,enabled)
        if enabled:count('local_gradient',True)
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        record(dict(index=index,sigma=float(scheduler.sigmas[index]),cursor_after=scheduler.step_index,**row))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    return z.detach().cpu(),dict(before_step25=before_first,final_history_sha256=trajectory.fingerprint(vars(scheduler)),
        terminal_sha256=trajectory.fingerprint(z),transformer_dtype=str(dtype),state_control_cfg='float32')


def read_mp4_search(path,keys,public,frozen_vae,*,count=None,persist=None):
    """Only file bytes and public protocol/keys enter; callback outputs are ignored.

    No path-name interpretation,arm,crop origin,message or writer evidence.
    Two keys share every actual phase VAE encoding. Raw phase features and every
    candidate are emitted before the pilot-only decision and later truth joins.
    """
    import torch
    if public!=method.PUBLIC:raise ValueError('fixed public protocol required')
    before=file_sha256(path)
    rgb=counted(count,'primary_mp4_read',lambda:io.read_mp4(Path(path)))
    if rgb.ndim!=4 or tuple(rgb.shape[1:])!=(public.height,public.width,3):raise ValueError('MP4 spatial geometry mismatch')
    length=len(rgb);roster=method.candidates(length);R=roster[0]['R'];features={};scores={};results={};window_results={}
    def emit(kind,slot,row):
        if persist:persist(kind,slot,row)
    for g in method.phases(length):
        left,right=method.phase_slice(length,g)
        try:
            normalized=counted(count,'vae_encode',lambda:vae_adapter.reencode_rgb24_readback(frozen_vae,rgb[left:right]))
            expected_t=(right-left-1)//4+1
            if tuple(normalized.shape)!=(1,16,expected_t,40,64):raise ValueError('phase VAE geometry mismatch')
            feature_hash=trajectory.fingerprint(normalized)
            emit('normalized',g,normalized.detach().cpu())
        except Exception as exc:
            for name in keys:
                row=dict(status='FAILED',g=g,decoded_bits=None,error=f'{type(exc).__name__}: {exc}',truth_used=False)
                features[(g,name)]=row;emit('phase',(g,name),row)
                window_results[(g,name)]=dict(status='FAILED',error=row['error'])
                emit('window_failure',(g,name),window_results[(g,name)])
            continue
        for name,key in keys.items():
            try:
                row,windows=counted(count,'phase_payload_read',lambda:method.phase_observations(normalized,key,length,g,public))
                row.update(g=g,feature_sha256=feature_hash,mp4_sha256=before,received_frames=length,phase_frames=right-left)
            except Exception as exc:
                row=dict(status='FAILED',g=g,decoded_bits=None,error=f'{type(exc).__name__}: {exc}',truth_used=False)
                features[(g,name)]=row;emit('phase',(g,name),row)
                window_results[(g,name)]=dict(status='FAILED',error=row['error'])
                emit('window_failure',(g,name),window_results[(g,name)])
                continue
            # Preserve valid primary observations even if sidecar export fails.
            features[(g,name)]=row;emit('phase',(g,name),row)
            try:
                if persist:
                    counted(count,'window_artifact_export',lambda:emit('windows',(g,name),windows))
                    window_results[(g,name)]=dict(status='EXPORTED',row_count=windows['window_count'])
                else:window_results[(g,name)]=windows
            except Exception as exc:
                failure=dict(status='FAILED',error=f'{type(exc).__name__}: {exc}')
                emit('window_failure',(g,name),failure);window_results[(g,name)]=failure
        del normalized
    del rgb
    identity_ok=file_sha256(path)==before
    for name,key in keys.items():
        rows=[]
        for candidate in roster:
            f=features[(candidate['g'],name)]
            if not identity_ok:row=dict(**candidate,status='FAILED',score=None,error='MP4 identity changed during read')
            elif f['status']!='READ':row=dict(**candidate,status='FAILED',score=None,error='required phase unavailable')
            else:row=counted(count,'pilot_candidate_score',lambda:method.score_candidate(f,key,candidate))
            rows.append(row);emit('candidate',(name,candidate['b']),row)
        decision=counted(count,'primary_search',lambda:method.decide(rows,length,public))
        decision.update(mp4_sha256=before,observed_frames=length,decoded_bits=None,selected_phase_feature_sha256=None)
        if decision['status']=='LOCATED':
            selected=features[(decision['selected_g'],name)]
            decision.update(decoded_bits=selected['decoded_bits'],selected_phase_feature_sha256=selected['feature_sha256'])
        emit('search',name,decision);scores[name]=rows;results[name]=decision
    return dict(searches=results,candidates=scores,features=features,window_exports=window_results)
