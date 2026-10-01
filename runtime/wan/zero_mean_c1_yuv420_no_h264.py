"""Materialized RGB24 -> raw420 -> RGB24; no H264, model or receiver changes."""
from __future__ import annotations
import hashlib,os,subprocess
from pathlib import Path
from runtime.wan.zero_mean_c1_layered_transport import file_sha256,encode_normalized
RGB_SHAPE=(181,320,512,3)
YUV_BYTES=44482560
RGB_BYTES=88965120

def conversion_commands(yuv_path,rgb_path):
    return [
        ['ffmpeg','-v','verbose','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','512x320','-r','8','-i','pipe:0',
         '-an','-c:v','rawvideo','-pix_fmt','yuv420p','-f','rawvideo','-n',str(yuv_path)],
        ['ffmpeg','-v','verbose','-threads','1','-noautorotate','-f','rawvideo','-pix_fmt','yuv420p','-s','512x320','-r','8',
         '-chroma_sample_location','left','-i',str(yuv_path),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-n',str(rgb_path)]]

def reopen_rgb(path,expected_sha):
    import numpy as np,torch
    if file_sha256(path)!=expected_sha:raise ValueError('roundtrip RGB identity mismatch')
    raw=Path(path).read_bytes()
    if len(raw)!=RGB_BYTES:raise ValueError('roundtrip RGB must contain exactly181 full frames')
    return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).reshape(RGB_SHAPE).copy())

def roundtrip(q8,yuv_path,rgb_path,*,count,event):
    """Persist each physical raw file and stage outcome; callbacks receive no truth."""
    import torch
    if tuple(q8.shape)!=RGB_SHAPE or q8.dtype!=torch.uint8:raise ValueError('fixed full uint8 input required')
    source=q8.detach().cpu().contiguous().numpy().tobytes()
    source_sha=hashlib.sha256(source).hexdigest()
    yuv_path,rgb_path=Path(yuv_path),Path(rgb_path)
    yuv_path.parent.mkdir(parents=True,exist_ok=True);rgb_path.parent.mkdir(parents=True,exist_ok=True)
    rows={}
    for stage,target,expected,kind in [('yuv420',yuv_path,YUV_BYTES,'rgb_to_raw420'),('rgb24',rgb_path,RGB_BYTES,'raw420_to_rgb24')]:
        partial=target.with_suffix(target.suffix+'.partial')
        if target.exists() or partial.exists():raise FileExistsError('new raw output already exists: '+str(target))
        commands=conversion_commands(partial if stage=='yuv420' else yuv_path,partial if stage=='rgb24' else rgb_path)
        command=commands[0 if stage=='yuv420' else 1]
        row=dict(status='RUNNING',command=command,path=str(target),partial_path=str(partial),expected_bytes=expected,
                 input_raster_sha256=source_sha if stage=='yuv420' else rows['yuv420']['sha256'],
                 chroma_location='AUTO_UNPROVEN_GRID' if stage=='yuv420' else 'left (known original SOURCE stream label, not proof of forward grid)',
                 color_space='UNKNOWN_AUTO',color_range='UNKNOWN_AUTO',effective_filtergraph='see verbatim verbose stderr; absent details remain unknown')
        event(stage,dict(row))
        count(kind,False)
        try:
            completed=subprocess.run(command,input=source if stage=='yuv420' else None,capture_output=True,check=False)
            stderr=target.with_suffix(target.suffix+'.stderr.txt');stderr.write_bytes(completed.stderr)
            row.update(returncode=completed.returncode,stderr_path=str(stderr),stderr_sha256=file_sha256(stderr))
            if completed.returncode:raise RuntimeError('conversion process returned '+str(completed.returncode))
            size=partial.stat().st_size
            if size!=expected:raise ValueError(f'raw byte count {size} != {expected}; no truncation/padding')
            row.update(sha256=file_sha256(partial),bytes=size,frames=181)
            os.replace(partial,target);row['status']='SAVED';rows[stage]=dict(row)
            count(kind,True);event(stage,dict(row))
        except Exception as exc:
            row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}')
            if partial.exists():row.update(partial_bytes=partial.stat().st_size,partial_sha256=file_sha256(partial))
            event(stage,dict(row));raise
        if stage=='yuv420':
            if file_sha256(yuv_path)!=rows['yuv420']['sha256']:raise ValueError('materialized raw420 changed before second process')
            source=None
    return reopen_rgb(rgb_path,rows['rgb24']['sha256'])
