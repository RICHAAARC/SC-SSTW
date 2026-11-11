"""Thin Colab entry; ordinary source loading and a no-time-limit worker."""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

STEM='b_t12_saved_candidate_media_v1'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def prepare_source(directory,ref,scratch,inputs):
    scratch=Path(scratch).resolve()
    for source in inputs:
        source=Path(source).resolve()
        if scratch==source or scratch.is_relative_to(source) or source.is_relative_to(scratch):
            raise ValueError('source scratch overlaps retained input')
    if 'Video-WM_Result' in scratch.parts: raise ValueError('source scratch must be outside retained results')
    if directory is not None: return Path(directory).resolve()
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
        # A fresh ordinary directory per preparation; no overwrite or source identity test.
        destination=scratch/('source-'+str(time.time_ns()));roots[0].rename(destination)
        return destination


def execute(config,source_directory=None,source_ref='dev/local-joint-state-payload-v1'):
    output=Path(config['output']).resolve()
    if (output/'operations.json').is_file():
        rows=json.loads((output/'operations.json').read_text())
        if rows.get('post',{}).get('status')!='NOT_STARTED': return display(output)
    inputs=[config[k] for k in ('candidate','original','host')]
    for value in inputs:
        source=Path(value).resolve()
        if output==source or output.is_relative_to(source) or source.is_relative_to(output): raise ValueError('output overlaps read-only input')
    if 'Video-WM_Result' in output.parts: raise ValueError('new diagnostic must not write formal results')
    logs=output.with_name(output.name+'-logs');logs.mkdir(parents=True,exist_ok=True)
    root=prepare_source(source_directory,source_ref,'/content/video-wm-t12-saved-candidate-media-v1',inputs)
    records=load(root/'experiments/wan_state_clock'/f'{STEM}_records.py','t12_records')
    if not (output/'attempt.json').exists():
        check=records.preflight(config);records.write(output.with_name(output.name+'-preflight.json'),check)
        if check['status']!='INPUTS_READABLE':
            print('输入尚未就绪；未调用模型或媒体。',json.dumps(check,ensure_ascii=False));return check
    setup=load(root/'scripts/b_framewise_qim_v1_setup.py','t12_dependency_setup')
    setup.ensure_imports('import numpy,torch,psutil,accelerate,safetensors,huggingface_hub,transformers,sentencepiece,ftfy; from diffusers import AutoencoderKLWan',
        ['numpy','torch','psutil','diffusers','accelerate','safetensors','huggingface-hub','transformers','sentencepiece','ftfy'],
        'saved candidate runtime',root,logs/'setup.log')
    import shutil
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        setup.command(['apt-get','-qq','update'],'Prepare FFmpeg',root,logs/'setup.log')
        setup.command(['apt-get','-qq','install','-y','ffmpeg'],'Install FFmpeg',root,logs/'setup.log')
    command=[sys.executable,'-u','-B','-m','experiments.wan_state_clock.'+STEM+'_run']
    for key,value in config.items(): command.extend(['--'+key.replace('_','-'),str(value)])
    environment=os.environ.copy();environment['PYTHONPATH']=str(root);environment['PYTHONDONTWRITEBYTECODE']='1'
    with (logs/'runner.log').open('a') as stream:
        worker=subprocess.Popen(command,cwd=root,env=environment,stdout=stream,stderr=subprocess.STDOUT)
        try:
            while worker.poll() is None:
                time.sleep(5)
        except KeyboardInterrupt:
            worker.terminate();worker.wait();raise
    print('worker returncode',worker.returncode,'日志',logs/'runner.log')
    display(output)
    if worker.returncode: raise RuntimeError('worker failed; saved calls remain consumed; see log')


def display(output):
    output=Path(output)
    for name in ('result.json','evaluation_summary.json'):
        path=output/name
        if path.is_file():
            value=json.loads(path.read_text())
            if name.startswith('evaluation'): value={k:v for k,v in value.items() if k not in ('native','cpu_saved_result','startup_nodes','startup_edges','old_wrong_all')}
            print(name,json.dumps(value,ensure_ascii=False,indent=2))
    print('预算失败和数值未决保留；此处不表示科学PASS。完整结果：',output)
