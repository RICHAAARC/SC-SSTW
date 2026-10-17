"""P2 secondary publication of actual 17/35-frame RGB, fixed CRF18 commands."""
from pathlib import Path
import json
import os
import subprocess
import numpy as np
from runtime.wan.fixed_rgb_media import commands


def roundtrip(rgb, folder, *, count, event):
    if rgb.dtype != np.uint8 or rgb.shape not in ((17,320,512,3),(35,320,512,3)):
        raise ValueError('secondary publication requires actual uint8 17/35-frame RGB')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    mp4=folder/'candidate.mp4';partial=folder/'candidate.partial.mp4';received=folder/'candidate_POST.rgb8'
    if any(p.exists() for p in (mp4,partial,received)):
        raise FileExistsError('secondary media attempt output already exists')
    source=np.ascontiguousarray(rgb).tobytes()
    pre=folder/'candidate_PRE.rgb8';pre.write_bytes(source)
    for kind,args,data in [('mp4_save',commands(partial)['save'],source),
                           ('mp4_probe',commands(mp4)['probe'],None),
                           ('mp4_readback',commands(mp4)['read'],None)]:
        row=dict(status='STARTED',command=args,frames=len(rgb),crf=18,fps=8,pixel_format='yuv420p')
        event(kind,row);count(kind,False)
        try:
            child=subprocess.run(args,input=data,capture_output=True,check=False)
            (folder/(kind+'.stderr.txt')).write_bytes(child.stderr)
            if child.returncode:raise RuntimeError(kind+' failed: '+str(child.returncode))
            if kind=='mp4_save':
                os.replace(partial,mp4)
            elif kind=='mp4_probe':
                metadata=json.loads(child.stdout);stream=metadata['streams'][0]
                if (int(stream['height']),int(stream['width']))!=(320,512):
                    raise ValueError('secondary MP4 geometry differs')
                row['metadata']=metadata
            else:
                if len(child.stdout)!=rgb.size:
                    raise ValueError('secondary MP4 frame count differs; no truncation or padding')
                temp=received.with_suffix('.tmp');temp.write_bytes(child.stdout);os.replace(temp,received)
                value=np.frombuffer(child.stdout,np.uint8).reshape(rgb.shape).copy()
            count(kind,True);row['status']='COMPLETE';event(kind,row)
        except BaseException as exc:
            row.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');event(kind,row);raise
    return value
