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


FULL_STAGES=('main0','main1','public0','public1','public2','public3','finalize')
FPR_STAGES=('main0','main1','finalize')


def run_colab(source_directory=None,source_ref='main',previous_runtime_stopped=False,max_items=None,max_hours=None):
    return _run_sequence(FULL_STAGES,'FULL_PAPER_SUITE',source_directory,source_ref,previous_runtime_stopped,max_items,max_hours)


def run_fpr_colab(source_directory=None,source_ref='main',previous_runtime_stopped=False,max_items=None,max_hours=None):
    return _run_sequence(FPR_STAGES,'FINITE_FPR_VALIDATION',source_directory,source_ref,previous_runtime_stopped,max_items,max_hours,cohort_only=True)


def run_public_compact_colab(source_directory=None,source_ref='main',previous_runtime_stopped=False,max_items=None,max_hours=None):
    return _run_sequence(('public0','public1','public2','public3'),'COMPACT_PUBLIC',source_directory,source_ref,previous_runtime_stopped,max_items,max_hours)


def run_parallel_colab(role='main',shard=0,source_directory=None,source_ref='main',previous_runtime_stopped=False,max_items=None,max_hours=None):
    if role not in ('main','public') or type(shard) is not int or not 0<=shard<(2 if role=='main' else 4):raise ValueError('fixed main0..1 or public0..3 worker required')
    return _run_sequence((role+str(shard),),'PARALLEL_WORKER',source_directory,source_ref,previous_runtime_stopped,max_items,max_hours)


def run_finalize_colab(source_directory=None,source_ref='main',previous_runtime_stopped=False):
    return _run_sequence(('finalize',),'CPU_FINALIZER',source_directory,source_ref,previous_runtime_stopped,None,None)


def _run_sequence(stages,workflow,source_directory,source_ref,previous_runtime_stopped,max_items=None,max_hours=None,cohort_only=False):
    if max_items is not None and (type(max_items) is not int or max_items<1):raise ValueError('items must be None or a positive integer')
    if max_hours is not None and (not isinstance(max_hours,(int,float)) or not 0<max_hours<float('inf')):raise ValueError('hours must be None or positive finite')
    import uuid
    OUTPUT.mkdir(parents=True,exist_ok=True)
    invocation=OUTPUT/'invocations'/uuid.uuid4().hex;invocation.mkdir(parents=True)
    source=available_source(source_directory,source_ref)
    print('使用源码:',source,flush=True)
    environment=dict(os.environ,PYTHONPATH=str(source))
    for module,package in [('numpy','numpy'),('scipy','scipy'),('psutil','psutil')]:
        probe=subprocess.run([sys.executable,'-c','import '+module],capture_output=True,env=environment)
        if probe.returncode:subprocess.run([sys.executable,'-m','pip','install',package],check=True)
    sys.path.insert(0,str(source))
    from contextlib import nullcontext
    from runtime.wan.paper_execution_v1 import check_full_owner,active_worker_check,OUTPUT as EXECUTION_OUTPUT
    from runtime.wan.native_baseline_storage_v4 import runtime_session
    if workflow=='FULL_PAPER_SUITE':
        owner=runtime_session(EXECUTION_OUTPUT/'owners/full-suite',previous_runtime_stopped=previous_runtime_stopped)
    else:
        check_full_owner(previous_runtime_stopped);owner=nullcontext()
    with owner:
        if workflow=='FULL_PAPER_SUITE':active_worker_check(previous_runtime_stopped)
        return _dispatch(stages,workflow,source,environment,invocation,previous_runtime_stopped,max_items,max_hours,cohort_only)


def _dispatch(stages,workflow,source,environment,invocation,previous_runtime_stopped,max_items,max_hours,cohort_only):
    import uuid
    # The orchestration cursor only rotates fixed owners; physical work uses their own durable checkpoints.
    cursor_path=OUTPUT/(workflow+'_scheduling_cursor.json')
    computation=[stage for stage in stages if stage!='finalize'];start_index=0
    if workflow!='PARALLEL_WORKER' and cursor_path.is_file() and computation:
        start_index=json.loads(cursor_path.read_text()).get('next_slot',0)%len(computation)
    ordered=computation[start_index:]+computation[:start_index]
    rows=[];deadline=None if max_hours is None else time.monotonic()+max_hours*3600
    for ordinal,stage in enumerate(ordered+(['finalize'] if 'finalize' in stages else [])):
        if stage!='finalize' and deadline is not None and time.monotonic()>=deadline:break
        worker=None;path=invocation/(str(ordinal)+'-'+stage+'.json');logpath=invocation/(str(ordinal)+'-'+stage+'.log');tick=time.monotonic()
        args=[sys.executable,'-u','-m','runtime.wan.paper_suite_worker_v1','--stage',stage,'--receipt',str(path),
              '--max-items',str(0 if max_items is None else max_items),'--max-hours',str(0 if deadline is None else max(.001,(deadline-time.monotonic())/3600))]
        if previous_runtime_stopped:args.append('--previous-runtime-stopped')
        if cohort_only:args.append('--cohort-only')
        print('开始:',stage,'日志:',logpath,flush=True)
        try:
            with logpath.open('w') as log:
                worker=subprocess.Popen(args,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                while worker.poll() is None:
                    try:worker.wait(timeout=25)
                    except subprocess.TimeoutExpired:print(stage,'运行中；秒数',round(time.monotonic()-tick),flush=True)
            value=json.loads(path.read_text()) if path.is_file() else dict(status='WORKER_TERMINAL_MISSING')
            rows.append(dict(stage=stage,returncode=worker.returncode,result=value));print(stage,value.get('status'),flush=True)
            if stage in computation and workflow!='PARALLEL_WORKER':
                # This index is not a computation checkpoint; no scientific work is repeated from it.
                temporary=cursor_path.with_name(cursor_path.name+'.'+uuid.uuid4().hex)
                temporary.write_text(json.dumps(dict(next_slot=(computation.index(stage)+1)%len(computation))));os.replace(temporary,cursor_path)
            if worker.returncode==130:break
        finally:stop(worker)
    result=dict(status='INTERRUPTED_RETAINED' if any(row['returncode']==130 for row in rows) else 'STAGES_FAILED_RETAINED' if any(row['returncode'] for row in rows) else 'INVOCATION_RETURNED',workflow=workflow,stages=rows,output=str(OUTPUT),invocation=str(invocation),
        scheduling_hours=max_hours,max_items_per_worker=max_items,not_a_scientific_pass=True)
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
