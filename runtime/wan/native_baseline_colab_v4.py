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


def clean_stopped_sources(work=WORK,proc=Path('/proc')):
    """Only obsolete v4 source copies, after the old worker check; never result/cache data."""
    work=Path(work).resolve();removed=[];retained=[];errors=[]
    candidates=[work/'project',*work.glob('current-source-*')]
    for path in candidates:
        target=path.resolve()
        if path.is_symlink() or not path.is_dir() or target.parent!=work or 'Video-WM_Result' in target.parts:continue
        busy=False
        for process in Path(proc).iterdir():
            if not process.name.isdigit():continue
            try:
                cwd=(process/'cwd').resolve(strict=True)
                text=(process/'cmdline').read_bytes()+(process/'environ').read_bytes()
                if cwd==target or cwd.is_relative_to(target) or str(target).encode() in text:
                    busy=True;break
            except FileNotFoundError:continue  # process exited during inspection
            except PermissionError:busy=True;break  # cannot prove this source copy is unused
        if busy:retained.append(str(target));continue
        try:shutil.rmtree(target);removed.append(str(target))
        except OSError as error:errors.append(dict(path=str(target),error=str(error)))
    return dict(removed=removed,retained_in_use=retained,cleanup_errors=errors,scope='obsolete v4 source copies only; results, weights and checkpoints retained')


def current_progress(output,worker_pid,launched_at,last=None):
    """A heartbeat is liveness only; never relabel old progress as current work."""
    output=Path(output)
    def read(name):
        try:
            value=json.loads((output/name).read_text())
            return value if isinstance(value,dict) else {}
        except (OSError,ValueError):return {}
    owner=read('runtime_owner.json');row=read('progress.json')
    current=owner.get('pid')==worker_pid and owner.get('started_at',0)>=launched_at-1
    current=current and row.get('runtime_id')==owner.get('id') and row.get('at',0)>=launched_at-1
    if current:
        signature=(row.get('runtime_id'),row.get('at'))
        if signature==last:return dict(status='CURRENT_STAGE_UNCHANGED',stage=row.get('stage'),source=row.get('source'),
            seconds_since_progress=round(time.time()-row['at'],1),note='worker alive; no newly reported progress'),signature
        return dict(row,status='CURRENT_PROGRESS'),signature
    return dict(status='WAITING_FOR_CURRENT_PROGRESS',worker_pid=worker_pid,
        historical_progress=row or None,note='historical row is not current progress'),last


def check_previous_worker(output,previous_runtime_stopped):
    """Fail before setup if the existing worker may still own this package."""
    import socket
    path=Path(output)/'runtime_owner.json'
    if not path.is_file():return
    row=json.loads(path.read_text())
    if row.get('status')!='ACTIVE':return
    same_host=row.get('host')==socket.gethostname()
    alive=same_host and (Path('/proc')/str(row.get('pid'))).exists()
    if alive or (not same_host and not previous_runtime_stopped):
        raise RuntimeError('先停止旧 Colab worker；换到新 runtime 后将 PREVIOUS_RUNTIME_STOPPED=True。心跳不能证明旧任务已停止。')


def run_colab(source_directory=None,source_ref=SOURCE_REF,previous_runtime_stopped=False):
    WORK.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True);OUTPUT.mkdir(parents=True,exist_ok=True)
    worker=None;status='SETUP';source_stage=None
    try:
        check_previous_worker(OUTPUT,previous_runtime_stopped)
        # No preflight scan of all native arrays or 1,064 editor plans.
        saved={}
        try:
            value=json.loads((OUTPUT/'result.json').read_text())
            if isinstance(value,dict):saved=value
        except (OSError,ValueError):pass
        recorded_complete=(saved.get('status')=='ALL_NATIVE_OBSERVED' and
            saved.get('quality_planned_pairs',0)>0 and
            saved.get('quality_complete_stages',-1)*181==saved.get('quality_planned_pairs'))
        if recorded_complete and tables_ready(OUTPUT):
            show_results();return dict(status='READ_ONLY_COMPLETED',output=str(OUTPUT))
        pending=not recorded_complete
        if not source_directory:print('临时源码清理:',clean_stopped_sources(),flush=True)
        print('当前入口：准备新源码；复用同一 attempt01，从缺失部分继续',flush=True)
        if source_directory:source=Path(source_directory)
        else:
            source_stage=Path(tempfile.mkdtemp(prefix='current-source-',dir=WORK))
            source=source_download('RICHAAARC/SC-SSTW',source_ref,source_stage/'project')
        print('使用源码:',source,flush=True)
        vendor=CACHE/'vendor'
        os.environ.update(HF_HOME=str(CACHE/'weights/huggingface'),TORCH_HOME=str(CACHE/'weights/torch'))
        environment=os.environ.copy();environment['PYTHONPATH']=str(source)
        if pending:
            source_download('facebookresearch/videoseal','main',vendor/'videoseal')
            source_download('facebookresearch/watermark-anything','main',vendor/'watermark-anything')
            installer=source/'runtime/wan/native_baseline_temporal_colab_v2.py'
            namespace={'__name__':'baseline_dependency_installer','__file__':str(installer)}
            exec(compile(installer.read_text(),str(installer),'exec'),namespace)
            environment=namespace['_install_dependencies'](source,vendor)
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
        print('当前入口：启动续跑 worker；只根据轻量完成记录调度',flush=True)
        launched_at=time.time();last_progress=None
        with (OUTPUT/'runner.log').open('a') as log:
            worker=subprocess.Popen(args,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while worker.poll() is None:
                try:worker.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    message,last_progress=current_progress(OUTPUT,worker.pid,launched_at,last_progress)
                    print(json.dumps(message,ensure_ascii=False),flush=True)
        status='RUN_FINISHED_WITH_TABLES' if worker.returncode==0 else 'INTERRUPTED_RETAINED'
        (OUTPUT/'bootstrap.json').write_text(json.dumps(dict(status=status,returncode=worker.returncode,at=time.time())))
    except BaseException as error:
        status='INTERRUPTED_RETAINED'
        value=dict(status=status,error=f'{type(error).__name__}: {error}',at=time.time())
        with (OUTPUT/'bootstrap_failure.json').open('w') as stream:
            json.dump(value,stream,ensure_ascii=False);stream.flush();os.fsync(stream.fileno())
        print(value,flush=True)
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
    finally:
        stop_group(worker)
        if source_stage is not None:
            # This invocation's temporary source only, after its worker has stopped.
            target=source_stage.resolve()
            if target.parent==WORK.resolve() and target.name.startswith('current-source-') and 'Video-WM_Result' not in target.parts:
                try:
                    shutil.rmtree(target)
                    print('已清理本次临时源码:',target,flush=True)
                except OSError as error:print('临时源码清理未完成，结果保留:',target,str(error),flush=True)
    show_results();return dict(status=status,output=str(OUTPUT))
