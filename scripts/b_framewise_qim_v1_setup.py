"""Ordinary QIM dependency repair, copied from the working three-group setup logic.

Called only for a new usable input. No notebook builder or historical common helper
is imported; missing imports and the optional torchao failure drive each repair.
"""
from pathlib import Path
import re
import shutil
import subprocess
import sys


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


def setup_dependencies(source_root, log_directory):
    log = Path(log_directory)/'setup.log'
    ensure_imports('import numpy,torch', ['numpy','torch'], 'tensor dependencies', source_root, log)
    ensure_imports('import accelerate,safetensors,huggingface_hub,transformers,sentencepiece,ftfy; from diffusers import WanPipeline,AutoencoderKLWan',
        ['diffusers','accelerate','safetensors','huggingface-hub','transformers','sentencepiece','ftfy'],
        'Wan pipeline and VAE', source_root, log)
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        command(['apt-get','-qq','update'], 'Prepare media dependency', source_root, log)
        command(['apt-get','-qq','install','-y','ffmpeg'], 'Install FFmpeg', source_root, log)
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        raise RuntimeError('ffmpeg/ffprobe unavailable; see '+str(log))
    probe = ('import experiments.wan_state_clock.b_framewise_qim_v1_experiment; '
             'import runtime.wan.b_framewise_qim_v1_backend; '
             'import experiments.wan_state_clock.b_framewise_qim_v1_run')
    command([sys.executable, '-c', probe], 'Probe actual QIM experiment/backend imports', source_root, log)
