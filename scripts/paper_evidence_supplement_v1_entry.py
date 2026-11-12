"""Ordinary editable source bootstrap for the fixed user-run quality notebook."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request


def prepare_source(directory,ref,scratch,result_root,output):
    scratch=Path(scratch).resolve()
    for value in (result_root,output):
        retained=Path(value).resolve()
        if scratch==retained or scratch.is_relative_to(retained) or retained.is_relative_to(scratch):
            raise ValueError('source scratch overlaps retained results')
    if 'Video-WM_Result' in scratch.parts: raise ValueError('source scratch must be outside formal results')
    if directory is not None:
        root=Path(directory).resolve()
        if not (root/'runtime/wan/paper_evidence_supplement_v1.py').is_file():
            raise FileNotFoundError('source directory lacks the supplement CLI: '+str(root))
        return root
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='download-',dir=scratch) as temporary:
        archive=Path(temporary)/'source.tar.gz'
        urllib.request.urlretrieve('https://github.com/RICHAAARC/SC-SSTW/archive/refs/heads/'+ref+'.tar.gz',archive)
        target=Path(temporary)/'tree';target.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                destination=(target/member.name).resolve()
                if not destination.is_relative_to(target.resolve()) or member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise ValueError('unsafe source archive member: '+member.name)
            tar.extractall(target)
        roots=[p for p in target.iterdir() if p.is_dir()]
        if len(roots)!=1: raise ValueError('source archive needs one root')
        destination=scratch/('source-'+str(time.time_ns()));roots[0].rename(destination)
        return destination


def display(output):
    output=Path(output);path=output/'quality_summary.json'
    if path.is_file(): print(path.read_text(encoding='utf-8'))
    else: print('固定包已消耗；摘要尚缺，失败/中断保留，不重新加载模型或读取媒体。')
    print('完整记录：',output)
    return dict(status='READ_ONLY_CONSUMED',output=str(output))


def execute(config,source_directory=None,source_ref='main'):
    output=Path(config['output']).resolve()
    # Even a fresh runtime must take this branch before downloads or dependency imports.
    if (output/'attempt.json').exists(): return display(output)
    root=prepare_source(source_directory,source_ref,'/content/paper-evidence-supplement-source',config['result_root'],output)
    environment=os.environ.copy();environment.update(PYTHONPATH=str(root),PYTHONDONTWRITEBYTECODE='1')
    check='import torch,torchvision,decord; from pyiqa.archs.musiq_arch import MUSIQ'
    if subprocess.run([sys.executable,'-c',check],cwd=root,env=environment).returncode:
        subprocess.run([sys.executable,'-m','pip','install','torch','torchvision','pyiqa','decord'],check=True)
        subprocess.run([sys.executable,'-c',check],cwd=root,env=environment,check=True)
    command=[sys.executable,'-u','-B','-m','runtime.wan.paper_evidence_supplement_v1','quality']
    for key in ('result_root','output','cache'): command.extend(['--'+key.replace('_','-'),str(config[key])])
    # Stream progress directly to Colab; no timeout or hidden physical retry.
    subprocess.run(command,cwd=root,env=environment,check=True)
    return display(output)
