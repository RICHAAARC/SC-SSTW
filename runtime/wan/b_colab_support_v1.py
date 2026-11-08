"""Standalone Colab preparation helpers reused from the validated source paths.

Import repair follows actual failures; no version or source-identity admission gate.
"""
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import zipfile


def safe_extract(archive, destination):
    destination = Path(destination).resolve(); destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name = member.filename.replace('\\', '/')
            target = (destination/name).resolve()
            if name.startswith('/') or any(':' in part for part in name.split('/')) or not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('unsafe source archive path: '+name)
        zipped.extractall(destination)


def command(args, label, source_root, log, check=True):
    print(label, flush=True)
    with Path(log).open('a') as stream:
        stream.write('COMMAND '+repr(args)+'\n'); stream.flush()
        completed = subprocess.run(args, cwd=source_root, stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode and check:
        raise RuntimeError(label+' failed; see '+str(log))
    return completed.returncode


def ensure_imports(probe, packages, label, source_root, log):
    mapping = {p.replace('-', '_'):p for p in packages}
    repaired = set()
    while True:
        offset = log.stat().st_size if log.exists() else 0
        if not command([sys.executable, '-c', probe], 'Probe '+label, source_root, log, check=False):
            return
        with log.open('rb') as stream:
            stream.seek(offset); failure = stream.read().decode('utf-8', errors='replace')
        missing = re.search("ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]", failure)
        if "cannot import name 'FqnToConfig'" in failure and 'torchao.quantization' in failure:
            action = 'remove incompatible optional torchao'
            args = [sys.executable, '-m', 'pip', 'uninstall', '-y', 'torchao']
        elif missing and missing.group(1).split('.')[0] in mapping:
            package = mapping[missing.group(1).split('.')[0]]
            action = 'install missing '+package
            args = [sys.executable, '-m', 'pip', 'install', package]
        else:
            raise RuntimeError(label+' actual import failure; see '+str(log))
        if action in repaired:
            raise RuntimeError(label+' still fails after '+action+'; see '+str(log))
        repaired.add(action)
        command(args, action, source_root, log)


def install_dependencies(source, output):
    logs = output/'preparation'; logs.mkdir(parents=True, exist_ok=True)
    ensure_imports('import numpy,torch,psutil,accelerate,safetensors,huggingface_hub; '
        'from diffusers import WanPipeline,AutoencoderKLWan; '
        'import runtime.wan.b_joint_time_relative_physical_v1,runtime.wan.b_relative_time_payload_backend_v1',
        ['numpy','torch','psutil','accelerate','safetensors','huggingface-hub','diffusers','transformers','sentencepiece','ftfy'],
        'r02 actual generation/time/payload imports', source, logs/'setup.log')
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        command(['apt-get','-qq','update'], 'Prepare media dependency', source, logs/'setup.log')
        command(['apt-get','-qq','install','-y','ffmpeg'], 'Install media dependency', source, logs/'setup.log')
