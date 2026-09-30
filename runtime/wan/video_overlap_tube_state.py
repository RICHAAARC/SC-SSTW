"""Native Wan frontend and four-phase blind overlap-state MP4 adapter."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from main.tube_state import video_overlap_tube_control as method
from main.tube_state import video_overlap_tube_state as state
from runtime.wan import trajectory,io,vae as vae_adapter
from runtime.wan.video_temporal_sync_bridge import file_sha256,counted,execution_device_dtype,predict_branches


def run_trajectory(pipe,initial,scheduler,prompt,negative,dtype,arm,key,bits,count,record):
    trajectory.validate_scheduler(scheduler)
    if scheduler.step_index is not None:raise ValueError('fresh full native history required')
    z=initial.detach().float().clone();targets=method.build_targets(z,key,bits);before=None
    for index in range(50):
        c,u=predict_branches(pipe,z,scheduler,prompt,negative,dtype,index,count)
        if index==25:before=dict(z=trajectory.fingerprint(z),history=trajectory.fingerprint(vars(scheduler)),conditional=trajectory.fingerprint(c),unconditional=trajectory.fingerprint(u))
        enabled=method.control_enabled(arm,index)
        if enabled:count('local_control',False)
        velocity,row=method.guided_velocity(z,c,u,float(scheduler.sigmas[index]),targets,arm,index,count)
        if enabled:count('local_control',True)
        z=trajectory.native_step(scheduler,z,velocity,index,count,kind='native_step')
        record(dict(index=index,sigma=float(scheduler.sigmas[index]),cursor_after=scheduler.step_index,**row))
    if scheduler.step_index!=50:raise RuntimeError('native trajectory incomplete')
    return z.detach().cpu(),dict(before_step25=before,final_history_sha256=trajectory.fingerprint(vars(scheduler)),
        terminal_sha256=trajectory.fingerprint(z),transformer_dtype=str(dtype),state_control_cfg='float32')


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
        decision.update(mp4_sha256=before,received_frames=length,canonical_phase_payload_diagnostic=None)
        if decision['canonical'] is not None:
            payload=features[decision['canonical']['g'],name]['payload']
            decision['canonical_phase_payload_diagnostic']=payload.get('decoded_bits') if payload else None
        results[name]=decision;emit('search',name,decision)
    return dict(searches=results,features=features)


def prepare_reserved_edit(received,kind,off_donor=None):
    """C2 interface only; never invoked by fixed C1 runner. Actual RGB operations."""
    import torch
    if tuple(received.shape)!=(129,320,512,3):raise ValueError('reserved crop129 input required')
    if kind=='DELETE4':return torch.cat((received[:64],received[68:]))
    if kind=='REPEAT4':return torch.cat((received[:68],received[64:68],received[68:]))
    if kind=='OFF_DONOR_INSERT4':
        if off_donor is None or off_donor.shape!=received.shape:raise ValueError('actual OFF crop donor required')
        return torch.cat((received[:68],off_donor[64:68],received[68:]))
    raise ValueError('unknown insertion/interpolation is not a repeat shortcut')


def prepare_reserved_edit_mp4(crop_path,output_path,kind,keys,public,frozen_vae,*,off_donor_path=None,count=None,persist=None):
    """Explicit C2 entry, excluded from C1: actual RGB edit, codec, independent reader."""
    crop=counted(count,'c2_input_mp4_read',lambda:io.read_mp4(Path(crop_path)))
    donor=None
    if kind=='OFF_DONOR_INSERT4':
        if off_donor_path is None:raise ValueError('actual OFF donor MP4 required')
        donor=counted(count,'c2_donor_mp4_read',lambda:io.read_mp4(Path(off_donor_path)))
    edited=prepare_reserved_edit(crop,kind,donor)
    output_path=Path(output_path)
    if output_path.exists():raise ValueError('new derived output path required')
    output_path.parent.mkdir(parents=True,exist_ok=True)
    counted(count,'c2_derived_mp4_save',lambda:io.encode_rgb(edited,output_path,8,18))
    del crop,donor,edited
    return read_mp4_search(output_path,keys,public,frozen_vae,count=count,persist=persist)
