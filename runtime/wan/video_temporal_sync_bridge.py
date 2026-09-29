"""Native Wan writer and MP4-only public temporal-pilot reception."""
from __future__ import annotations
import hashlib
from pathlib import Path
from main.tube_state import video_temporal_sync_bridge as method
from runtime.wan import trajectory,io,vae as vae_adapter


def file_sha256(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def counted(count,kind,fn):
    if count:count(kind,False)
    value=fn()
    if count:count(kind,True)
    return value


def execution_device_dtype():
    import torch
    if not torch.cuda.is_available():return 'cpu',torch.float32
    return 'cuda',torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count):
    import torch
    values=[]
    with torch.no_grad():
        for kind,embedding in (('transformer_conditional',prompt),('transformer_unconditional',negative)):
            v=counted(count,kind,lambda:pipe.transformer(hidden_states=z.to(dtype),
                timestep=scheduler.timesteps[index].expand(z.shape[0]),encoder_hidden_states=embedding,
                attention_kwargs=None,return_dict=False)[0]).float()
            if not bool(torch.isfinite(v).all()):raise FloatingPointError('nonfinite transformer output')
            values.append(v)
    return values


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,record):
    import torch
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh complete native scheduler required')
    z=initial.detach().float().clone();target,mask=method.build_target(z,key,bits,arm)
    before_last=None
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        if index==49:
            before_last=dict(z=trajectory.fingerprint(z),history=trajectory.fingerprint(vars(scheduler)),
                conditional=trajectory.fingerprint(c),unconditional=trajectory.fingerprint(u))
        enabled=arm!='OFF' and index==49
        if enabled:count('local_gradient',False)
        velocity,row=method.guided_velocity(z,c,u,float(scheduler.sigmas[index]),target,mask,enabled)
        if enabled:count('local_gradient',True)
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        record(dict(index=index,cursor_after=scheduler.step_index,**row))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    return z.detach().cpu(),dict(before_step49=before_last,final_history_sha256=trajectory.fingerprint(vars(scheduler)),
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
    length=len(rgb);roster=method.candidates(length);R=roster[0]['R'];features={};scores={};results={}
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
            continue
        for name,key in keys.items():
            try:
                row=counted(count,'phase_payload_read',lambda:method.phase_features(normalized,key,R,public))
                row.update(g=g,feature_sha256=feature_hash,mp4_sha256=before,source_frames=length,phase_frames=right-left)
            except Exception as exc:row=dict(status='FAILED',g=g,decoded_bits=None,error=f'{type(exc).__name__}: {exc}',truth_used=False)
            features[(g,name)]=row;emit('phase',(g,name),row)
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
    return dict(searches=results,candidates=scores,features=features)
