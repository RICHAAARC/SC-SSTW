"""Fixed full RGB8 -> H264 -> RGB24 persistence; no receiver/method imports."""
from __future__ import annotations
import hashlib,json,os,subprocess
from pathlib import Path
import numpy as np
SHAPE=(181,320,512,3)
RGB_BYTES=88965120


def file_sha256(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def observe_sha256(path):
    try:return file_sha256(path),None
    except OSError as exc:return None,f'{type(exc).__name__}: {exc}'

def identity_fields(prefix,path):
    actual,error=observe_sha256(path)
    return {prefix:actual,prefix+'_observation_error':error,prefix+'_blocking':False}

def _bytes(path,expected_sha):
    raw=Path(path).read_bytes()
    if len(raw)!=RGB_BYTES:raise ValueError('saved full RGB8 byte count mismatch')
    return raw

def reopen_raster(path,expected_sha):
    import torch
    return torch.from_numpy(np.frombuffer(_bytes(path,expected_sha),np.uint8).reshape(SHAPE).copy())

def save_raster(q8,path):
    import torch
    if q8.dtype!=torch.uint8 or tuple(q8.shape)!=SHAPE:raise ValueError('full uint8 RGB8 raster required')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp')
    raw=q8.detach().cpu().contiguous().numpy().tobytes()
    with temp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    return dict(status='SAVED',path=str(path),bytes=len(raw),shape=list(SHAPE),dtype='uint8',rounding='original np.rint(clamp(decoded_rgb,0,1)*255).astype(uint8)',**identity_fields('sha256',path))

def commands(mp4_path):
    t,h,w,_=SHAPE
    return dict(
        save=['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r','8','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-n',str(mp4_path)],
        probe=['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(mp4_path)],
        read=['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(mp4_path),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'])

def mp4_roundtrip(raster_path,raster_sha,mp4_path,rgb_path,*,count,event):
    """Exact old io command semantics; full bytes, command/stdout/stderr retained."""
    mp4_path,rgb_path=Path(mp4_path),Path(rgb_path);mp4_path.parent.mkdir(parents=True,exist_ok=True)
    partial=mp4_path.with_name(mp4_path.stem+'.partial.mp4')
    if mp4_path.exists() or partial.exists() or rgb_path.exists():raise FileExistsError('new media path already exists')
    source=_bytes(raster_path,raster_sha);actual_raster_sha=hashlib.sha256(source).hexdigest();row=dict(status='RUNNING',path=str(mp4_path),partial_path=str(partial),input_raster_path=str(raster_path),input_content_token=actual_raster_sha,declared_digest_is_admission_gate=False,command=commands(partial)['save'])
    event('mp4',dict(row));count('mp4_save',False)
    try:
        child=subprocess.run(row['command'],input=source,capture_output=True,check=False)
        out=mp4_path.with_suffix('.save.stdout.txt');err=mp4_path.with_suffix('.save.stderr.txt');out.write_bytes(child.stdout);err.write_bytes(child.stderr)
        row.update(returncode=child.returncode,stdout_path=str(out),stderr_path=str(err),**identity_fields('stdout_sha256',out),**identity_fields('stderr_sha256',err))
        if child.returncode:raise RuntimeError('original MP4 save exited '+str(child.returncode))
        os.replace(partial,mp4_path);row.update(status='SAVED',bytes=mp4_path.stat().st_size,**identity_fields('sha256',mp4_path));count('mp4_save',True);event('mp4',dict(row))
    except Exception as exc:row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');event('mp4',dict(row));raise
    del source
    cmds=commands(mp4_path);probe=dict(status='RUNNING',command=cmds['probe'],mp4_sha256=row['sha256']);event('probe',dict(probe));count('mp4_probe',False)
    try:
        child=subprocess.run(cmds['probe'],capture_output=True,check=False)
        out=mp4_path.with_suffix('.probe.json');err=mp4_path.with_suffix('.probe.stderr.txt');out.write_bytes(child.stdout);err.write_bytes(child.stderr)
        probe.update(returncode=child.returncode,stdout_path=str(out),stderr_path=str(err),**identity_fields('stdout_sha256',out),**identity_fields('stderr_sha256',err))
        if child.returncode:raise RuntimeError('original MP4 probe exited '+str(child.returncode))
        data=json.loads(child.stdout);stream=data['streams'][0]
        if (int(stream['height']),int(stream['width']))!=SHAPE[1:3]:raise ValueError('received MP4 frame geometry mismatch')
        probe.update(status='COMPLETE',metadata=data,color_metadata='not supplied by original probe; remains unknown');count('mp4_probe',True);event('probe',dict(probe))
    except Exception as exc:probe.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');event('probe',dict(probe));raise
    received=dict(status='RUNNING',path=str(rgb_path),command=cmds['read'],mp4_sha256=row['sha256'],input_raster_sha256=actual_raster_sha,input_raster_declared_sha256=raster_sha);event('rgb24',dict(received));count('mp4_readback',False)
    try:
        child=subprocess.run(cmds['read'],capture_output=True,check=False)
        err=rgb_path.with_suffix('.read.stderr.txt');err.write_bytes(child.stderr);received.update(returncode=child.returncode,stderr_path=str(err),**identity_fields('stderr_sha256',err))
        if child.returncode:raise RuntimeError('original MP4 RGB24 read exited '+str(child.returncode))
        raw=child.stdout
        if len(raw)!=RGB_BYTES:raise ValueError('received MP4 requires exactly181 full RGB24 frames; no truncation/padding')
        temp=rgb_path.with_suffix('.tmp');temp.write_bytes(raw);os.replace(temp,rgb_path)
        received.update(status='SAVED',bytes=len(raw),shape=list(SHAPE),**identity_fields('sha256',rgb_path));count('mp4_readback',True);event('rgb24',dict(received))
    except Exception as exc:received.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');event('rgb24',dict(received));raise
    final_mp4_sha,final_mp4_error=observe_sha256(mp4_path);final_raster_sha,final_raster_error=observe_sha256(raster_path)
    row.update(final_mp4_content_token=final_mp4_sha,final_raster_content_token=final_raster_sha,post_transport_observation_status='OBSERVATION_UNAVAILABLE' if final_mp4_error or final_raster_error else 'OBSERVED',post_transport_observation_error=final_mp4_error or final_raster_error);event('mp4',dict(row))
    return reopen_raster(rgb_path,received['sha256'])
