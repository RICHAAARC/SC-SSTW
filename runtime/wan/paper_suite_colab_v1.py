"""Thin notebook bootstrap, isolated stage processes, preserved old result packages."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
import stat

WORK=Path('/content/video-wm-paper-suite-v1')
OUTPUT=Path('/content/drive/MyDrive/Video-WM_Result/paper_suite_v1')


def available_source(directory,ref):
    if directory:return Path(directory).resolve()
    from urllib.parse import quote
    WORK.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='source-',dir=WORK));archive=stage/'source.zip';dest=stage/'unpacked';dest.mkdir()
    urllib.request.urlretrieve(f'https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/{quote(ref,safe="/")}',archive)
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            name=member.filename.replace('\\','/');target=(dest/name).resolve()
            if name.startswith('/') or any(':' in x for x in name.split('/')) or not target.is_relative_to(dest.resolve()) or stat.S_ISLNK(member.external_attr>>16):
                raise ValueError('unsafe source archive path: '+name)
        zipped.extractall(dest)
    found=[p for p in dest.iterdir() if (p/'runtime/wan/paper_suite_worker_v1.py').is_file()]
    if len(found)!=1:raise ValueError('one readable paper suite source required')
    return found[0]


def stop(worker):
    if worker is None:return
    try:os.killpg(worker.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    try:worker.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:os.killpg(worker.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        worker.wait()


FULL_STAGES=('cohort','tables','wan','tables','queries','tables','public_compact','tables')
FPR_STAGES=('cohort','tables','queries','tables')


def run_colab(source_directory=None,source_ref='main',previous_runtime_stopped=False):
    return _run_sequence(FULL_STAGES,'FULL_PAPER_SUITE',source_directory,source_ref,previous_runtime_stopped)


def run_public_compact_colab(source_directory=None,source_ref='main',previous_runtime_stopped=False):
    return _run_sequence(('public_compact','tables'),'COMPACT_PUBLIC',source_directory,source_ref,previous_runtime_stopped)


def run_fpr_colab(source_directory=None,source_ref='main'):
    return _run_sequence(FPR_STAGES,'FINITE_FPR_VALIDATION',source_directory,source_ref,False)


def _run_sequence(stages,workflow,source_directory,source_ref,previous_runtime_stopped):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    invocation=OUTPUT/'invocations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');invocation.mkdir(parents=True)
    source=available_source(source_directory,source_ref)
    print('使用源码:',source,flush=True)
    environment=dict(os.environ,PYTHONPATH=str(source))
    # CPU dependencies for new statistics/tables only. Model preparations stay inside their own stages.
    for module,package in [('numpy','numpy'),('scipy','scipy'),('psutil','psutil')]:
        probe=subprocess.run([sys.executable,'-c','import '+module],capture_output=True,env=environment)
        if probe.returncode:subprocess.run([sys.executable,'-m','pip','install',package],check=True)
    rows=[]
    for ordinal,stage in enumerate(stages):
        worker=None;path=invocation/(str(ordinal)+'-'+stage+'.json');logpath=invocation/(str(ordinal)+'-'+stage+'.log');tick=time.monotonic()
        args=[sys.executable,'-u','-m','runtime.wan.paper_suite_worker_v1','--stage',stage,'--receipt',str(path)]
        if previous_runtime_stopped:args.append('--previous-runtime-stopped')
        print('开始:',stage,'日志:',logpath,flush=True)
        try:
            with logpath.open('w') as log:
                worker=subprocess.Popen(args,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while worker.poll() is None:
                    try:worker.wait(timeout=25)
                    except subprocess.TimeoutExpired:print(stage,'运行中；秒数',round(time.monotonic()-tick),flush=True)
            value=json.loads(path.read_text()) if path.is_file() else dict(status='WORKER_TERMINAL_MISSING')
            rows.append(dict(stage=stage,returncode=worker.returncode,result=value))
            print(stage,value.get('status'),flush=True)
            if worker.returncode==130:break
        finally:stop(worker)
    result=dict(status={'FULL_PAPER_SUITE':'RUN_ALL_RETURNED','FINITE_FPR_VALIDATION':'FINITE_FPR_RETURNED','COMPACT_PUBLIC':'COMPACT_PUBLIC_RETURNED'}[workflow] if len(rows)==len(stages) else 'INTERRUPTED_RETAINED',workflow=workflow,stages=rows,output=str(OUTPUT),invocation=str(invocation))
    (invocation/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=1))
    show_results();return result


def show_results():
    path=OUTPUT/'latest_report.json'
    if path.is_file():
        value=json.loads(path.read_text());print('论文表格目录:',value['report'])
        print('公开Baseline:',value.get('public_baseline_status'))
        print('查询统计:',(value.get('query_status') or {}).get('status'))
        print('各源执行状态:',value.get('components'))
    else:print('完整表格尚未返回；阶段日志和原结果保留:',OUTPUT)
