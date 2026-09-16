"""Shared JSON and fixed RGB24/H.264 video I/O."""
from __future__ import annotations
import json
import subprocess
from pathlib import Path
from typing import Any

def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def read_mp4(path: Path) -> Any:
    import numpy as np, torch
    probe=subprocess.run(["ffprobe","-v","error","-select_streams","v:0","-show_entries","stream=width,height,nb_frames","-of","json",str(path)],capture_output=True,text=True,check=True)
    stream=json.loads(probe.stdout)["streams"][0]; w,h=int(stream["width"]),int(stream["height"])
    raw=subprocess.run(["ffmpeg","-v","error","-threads","1","-noautorotate","-i",str(path),"-map","0:v:0","-f","rawvideo","-pix_fmt","rgb24","-"],capture_output=True,check=True).stdout
    one=h*w*3
    if len(raw)%one: raise RuntimeError("received MP4 is not integral RGB24 frames")
    return torch.from_numpy(np.frombuffer(raw,dtype=np.uint8).reshape(len(raw)//one,h,w,3).copy()).float()/255


def encode_rgb(rgb: Any, path: Path, fps: int, crf: int) -> None:
    from runtime.wan.vae import quantize_rgb8_no_codec
    path.parent.mkdir(parents=True,exist_ok=True)
    t,h,w,_=map(int,rgb.shape); pixels=quantize_rgb8_no_codec(rgb).numpy()
    subprocess.run(["ffmpeg","-v","error","-threads","1","-f","rawvideo","-pix_fmt","rgb24","-s",f"{w}x{h}","-r",str(fps),"-i","pipe:0","-an","-c:v","libx264","-crf",str(crf),"-pix_fmt","yuv420p","-n",str(path)],input=pixels.tobytes(),check=True,capture_output=True)
