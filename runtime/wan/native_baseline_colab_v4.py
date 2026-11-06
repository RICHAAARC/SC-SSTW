"""Ordinary source bootstrap for the thin Run-all delivery. Importing does no work."""
import json
import csv
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile

SOURCE_REF='dev/baselines-paper-v4'
OUTPUT=Path('/content/drive/MyDrive/Video-WM_Result/baselines/public-native-and-frameindex-fpr001-v4/attempt01')
CACHE=Path('/content/drive/MyDrive/Video-WM_Cache/baseline-v4')
WORK=Path('/content/video-wm-public-v4')


def source_download(repository,ref,destination):
    from urllib.parse import quote
    destination=Path(destination)
    if destination.is_dir():return destination
    WORK.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=WORK,prefix='source-') as tmp:
        temporary=Path(tmp);archive=temporary/'source.zip';unpack=temporary/'unpacked';unpack.mkdir()
        urllib.request.urlretrieve(f'https://codeload.github.com/{repository}/zip/refs/heads/{quote(ref,safe="/")}',archive)
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                name=member.filename.replace('\\','/');target=(unpack/name).resolve()
                if name.startswith('/') or any(':' in x for x in name.split('/')) or not target.is_relative_to(unpack.resolve()) or stat.S_ISLNK(member.external_attr>>16):
                    raise ValueError('unsafe source archive path: '+name)
            zipped.extractall(unpack)
        folders=list(unpack.iterdir())
        if len(folders)!=1 or not folders[0].is_dir():raise ValueError('unreadable source archive layout')
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(folders[0]),str(destination))
    return destination


def stop_group(worker):
    if worker is None:return
    try:os.killpg(worker.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:worker.wait(timeout=5)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(worker.pid,signal.SIGKILL)
    except ProcessLookupError:pass
    worker.wait(timeout=5)


def show_results(output=OUTPUT):
    output=Path(output);print('保留结果目录:',output)
    result={}
    for name in ('result.json','interruption.json','bootstrap_failure.json'):
        path=output/name
        if not path.is_file():continue
        try:value=json.loads(path.read_text());print(name,json.dumps(value,ensure_ascii=False,indent=2));result[name]=value
        except (OSError,ValueError) as error:print('已保存文件实际读取失败，保留原文件:',name,str(error))
    path=output/'tables/B_ownership.json'
    if path.is_file():
        try:
            for row in json.loads(path.read_text()):
                if row['split']=='TEST' and row['channel'] in ('FAMILY40','ALL40_ROBUST') and row['verification']=='query':
                    print(row['method'],row['arm'],row['channel'],{k:row[k] for k in ('planned','positive','negative','unknown','execution_incomplete','conservative_one_sided95_upper')})
        except (OSError,ValueError,KeyError) as error:print('结果表读取失败:',str(error))
    return result


def tables_ready(output):
    """Actual output files must still be readable before taking the display-only shortcut."""
    output=Path(output)
    try:
        result=json.loads((output/'result.json').read_text())
        if not isinstance(result,dict):return False
        for name in ('A_payload','B_ownership','C_time','D_quality'):
            with (output/'tables'/(name+'.json')).open(encoding='utf-8') as stream:
                if not isinstance(json.load(stream),list):return False
            with (output/'tables'/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
                rows=csv.reader(stream,strict=True);header=next(rows,None)
                if not header or any(len(row)!=len(header) for row in rows):return False
        with (output/'tables/D_resources.json').open(encoding='utf-8') as stream:
            return isinstance(json.load(stream),dict)
    except (OSError,ValueError,TypeError,csv.Error):return False


def run_colab(source_directory=None,source_ref=SOURCE_REF,previous_runtime_stopped=False):
    WORK.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    worker=None;status='SETUP'
    try:
        source=Path(source_directory) if source_directory else source_download('RICHAAARC/SC-SSTW',source_ref,WORK/'project')
        sys.path.insert(0,str(source))
        from runtime.wan.native_baselines_public_v4 import load_inputs,native_complete,quality_complete
        from runtime.wan.native_baseline_storage_v1 import write_json
        from main.baselines.public_fpr001_v4 import BACKENDS,METHODS,backend
        _,sources=load_inputs()
        if (OUTPUT/'sources.json').is_file():sources=json.loads((OUTPUT/'sources.json').read_text())
        completed={}
        pending=any(not native_complete(OUTPUT,s,b,'OFF',completed) for s in sources for b in BACKENDS)
        pending=pending or any(not native_complete(OUTPUT,s,backend(m),m,completed) or not quality_complete(OUTPUT,s,m) for s in sources if s['split']!='CAL' for m in METHODS)
        if not pending and tables_ready(OUTPUT):
            show_results();return dict(status='READ_ONLY_COMPLETED',output=str(OUTPUT))
        vendor=CACHE/'vendor'
        os.environ.update(HF_HOME=str(CACHE/'weights/huggingface'),TORCH_HOME=str(CACHE/'weights/torch'))
        environment=os.environ.copy();environment['PYTHONPATH']=str(source)
        if pending:
            source_download('facebookresearch/videoseal','main',vendor/'videoseal')
            source_download('facebookresearch/watermark-anything','main',vendor/'watermark-anything')
            from runtime.wan.native_baseline_temporal_colab_v2 import _install_dependencies
            environment=_install_dependencies(source,vendor)
        # Test actual imports needed by the execution path, without version equality gates.
        mapping=dict(numpy='numpy',scipy='scipy',torch='torch')
        if pending:mapping.update(skimage='scikit-image',lpips='lpips',PIL='pillow')
        for module,package in mapping.items():
            result=subprocess.run([sys.executable,'-c','import '+module],capture_output=True,env=environment)
            if result.returncode:subprocess.run([sys.executable,'-m','pip','install',package],check=True)
        weight=vendor/'watermark-anything/checkpoints/wam_mit.pth';weight.parent.mkdir(parents=True,exist_ok=True)
        if pending and not weight.is_file():
            temporary=weight.with_suffix('.download')
            urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/watermark_anything/wam_mit.pth',temporary);os.replace(temporary,weight)
        args=[sys.executable,'-u',str(source/'experiments/wan_state_clock/native_baselines_public_fpr001_v4_run.py'),
              '--output',str(OUTPUT),'--cache',str(CACHE),'--vendor',str(vendor)]
        if previous_runtime_stopped:args.append('--previous-runtime-stopped')
        with (OUTPUT/'runner.log').open('a') as log:
            worker=subprocess.Popen(args,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while worker.poll() is None:
                try:worker.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    path=OUTPUT/'progress.json'
                    print(path.read_text() if path.is_file() else '执行中；实际批次即时保存，详见 runner.log',flush=True)
        status='RUN_FINISHED_WITH_TABLES' if worker.returncode==0 else 'INTERRUPTED_RETAINED'
        write_json(OUTPUT/'bootstrap.json',dict(status=status,returncode=worker.returncode,at=time.time()))
    except BaseException as error:
        status='INTERRUPTED_RETAINED'
        value=dict(status=status,error=f'{type(error).__name__}: {error}',at=time.time())
        with (OUTPUT/'bootstrap_failure.json').open('w') as stream:
            json.dump(value,stream,ensure_ascii=False);stream.flush();os.fsync(stream.fileno())
        print(value,flush=True)
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:stop_group(worker)
    show_results();return dict(status=status,output=str(OUTPUT))
