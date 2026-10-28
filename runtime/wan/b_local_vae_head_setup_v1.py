"""Shared pre-execution source/import preparation; never loads weights/models."""
from pathlib import Path
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

def prepare_vendor(scratch):
    scratch=Path(scratch).expanduser().resolve();scratch.mkdir(parents=True,exist_ok=True)
    vendor=scratch/'vendor/videoseal';vendor.parent.mkdir(parents=True,exist_ok=True)
    if (vendor/'videoseal/utils/cfg.py').is_file():return vendor
    if vendor.exists():raise FileNotFoundError('existing VideoSeal source lacks videoseal/utils/cfg.py: '+str(vendor))
    stage=Path(tempfile.mkdtemp(prefix='official-',dir=scratch));archive=stage/'source.zip'
    urllib.request.urlretrieve('https://codeload.github.com/facebookresearch/videoseal/zip/refs/heads/main',archive)
    destination=stage/'source';destination.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            normalized=member.filename.replace('\\','/');target=(destination/normalized).resolve()
            if normalized.startswith('/') or any(':' in p for p in normalized.split('/')) or not target.is_relative_to(destination.resolve()) or stat.S_ISLNK(member.external_attr>>16):raise ValueError('unsafe official source archive path')
        zipped.extractall(destination)
    entries=list(destination.glob('*/videoseal/utils/cfg.py'))
    if len(entries)!=1:raise FileNotFoundError('official source lacks usable VideoSeal cfg')
    shutil.move(str(entries[0].parents[2]),str(vendor));return vendor

def environment(root,vendor):
    result=os.environ.copy();result['PYTHONPATH']=os.pathsep.join((str(Path(root).resolve()),str(Path(vendor).resolve())))
    return result

def check_dependencies(root,vendor):
    """Actual import/tool failures before once-attempt receipt and scientific calls."""
    probe='import numpy,torch,torchvision,ftfy,sentencepiece,accelerate; from diffusers import WanPipeline,AutoencoderKLWan; import videoseal.utils.cfg; import runtime.wan.b_local_vae_head_backend_v1'
    p=subprocess.run([sys.executable,'-c',probe],cwd=vendor,env=environment(root,vendor),capture_output=True,text=True)
    if p.returncode:raise RuntimeError('pre-execution dependency import failed (no scientific calls started):\n'+p.stderr[-12000:])
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):raise FileNotFoundError('ffmpeg/ffprobe unavailable before scientific start')
    return dict(imports='AVAILABLE',ffmpeg=shutil.which('ffmpeg'),ffprobe=shutil.which('ffprobe'),model_calls=0)
