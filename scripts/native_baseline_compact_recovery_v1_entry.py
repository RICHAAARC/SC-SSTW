"""Ordinary source bootstrap for the finite compact12 repair notebook."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
import stat
import time


def import_cpu_package(output,parent=None,previous_runtime_stopped=True):
    """Reuse the closed local CPU result without overwriting Colab progress."""
    output=Path(output);archive=output/'local_cpu_recovery.zip'
    if not archive.is_file():return dict(status='NO_LOCAL_CPU_PACKAGE')
    from runtime.wan.native_baseline_colab_v4 import check_previous_worker
    from runtime.wan.native_baseline_storage_v4 import runtime_session
    parent=Path(parent) if parent else output.parent.parent
    # Check every possible active owner before any ledger/evidence modification.
    for root in [output,*[output/'shards'/f'part{i}' for i in range(4)],*[parent/'shards'/f'part{i}' for i in range(4)]]:
        check_previous_worker(root,previous_runtime_stopped)
    with runtime_session(output,previous_runtime_stopped=previous_runtime_stopped):
        return _import_cpu_package_contents(output,archive)


def _import_cpu_package_contents(output,archive):
    copied=[];skipped=[];costs=[];excluded=[]
    with zipfile.ZipFile(archive) as package:
        # Validate every member before touching the repair directory.
        members=[]
        for item in package.infolist():
            name=item.filename.replace('\\','/');relative=Path(name);target=(output/name).resolve()
            if name.startswith('/') or any(':' in x for x in name.split('/')) or not target.is_relative_to(output.resolve()) or stat.S_ISLNK(item.external_attr>>16):
                raise ValueError('unsafe CPU package member: '+name)
            if item.is_dir():continue
            if relative.parts[0]!='shards' and name not in ('cpu_recovery.json','cpu_recovery_history.jsonl'):
                excluded.append(name);continue
            # Runtime ownership and derived counters are local to the active run.
            # Ledger events below are merged exactly once into its own cost file.
            if any('lock' in part.lower() or 'owner' in part.lower() or 'runtime' in part.lower() for part in relative.parts) or relative.name=='physical_counts.json':
                excluded.append(name);continue
            members.append((item,name,target))
        for item,name,target in members:
            if Path(name).name=='physical_calls.jsonl':
                rows=[json.loads(line) for line in package.read(item).decode().splitlines() if line.strip()]
                if any(not isinstance(r,dict) or r.get('status')!='STARTED' for r in rows):
                    raise ValueError('local CPU cost ledger contains an unreadable/non-start event: '+name)
                existing=[json.loads(s) for s in target.read_text().splitlines() if s.strip()] if target.is_file() else []
                # Receipt identity is used only to avoid double-counting costs;
                # it is never a source/version/integrity admission check.
                def address(r):return tuple(r.get(k) for k in ('runtime_id','unit','operation','category','at','count'))
                known={address(r) for r in existing};target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('a') as stream:
                    for row in rows:
                        if address(row) in known:continue
                        stream.write(json.dumps(row,ensure_ascii=False)+'\n');known.add(address(row));costs.append(dict(file=name,**row))
                continue
            if target.exists():skipped.append(name);continue
            data=package.read(item)  # Finish decompression before creating a final path.
            target.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent,prefix='cpu-import-',suffix='.partial',delete=False) as stream:
                temporary=Path(stream.name);stream.write(data);stream.flush();os.fsync(stream.fileno())
            if target.exists():skipped.append(name);temporary.unlink();continue
            os.replace(temporary,target)
            copied.append(name)
    result=dict(status='LOCAL_CPU_PACKAGE_IMPORTED',at=time.time(),archive=str(archive),copied=copied,
        existing_paths_preserved=skipped,excluded=excluded,new_cost_events=costs,
        semantics='only missing evidence copied; original CPU costs appended once; runtime ownership never imported')
    with (output/'cpu_package_imports.jsonl').open('a') as stream:stream.write(json.dumps(result,ensure_ascii=False)+'\n')
    return result


def source_directory(directory=None,ref='main'):
    if directory:return Path(directory).resolve()
    from urllib.parse import quote
    stage=Path(tempfile.mkdtemp(prefix='baseline-compact-repair-',dir='/content'))
    archive=stage/'source.zip';dest=stage/'source';dest.mkdir()
    urllib.request.urlretrieve(f'https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/{quote(ref,safe="/")}',archive)
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            name=item.filename.replace('\\','/');target=(dest/name).resolve()
            if name.startswith('/') or any(':' in x for x in name.split('/')) or not target.is_relative_to(dest.resolve()) or stat.S_ISLNK(item.external_attr>>16):
                raise ValueError('unsafe source archive member: '+name)
        package.extractall(dest)
    roots=[d for d in dest.iterdir() if (d/'runtime/wan/native_baseline_compact_recovery_run_v1.py').is_file()]
    if len(roots)!=1:raise FileNotFoundError('one usable recovery source package required')
    return roots[0]


def execute(config,source=None,ref='main',previous_runtime_stopped=True):
    root=source_directory(source,ref);print('使用源码:',root,flush=True)
    output=Path(config['output'])
    sys.path.insert(0,str(root))
    try:imported=import_cpu_package(output,config['parent'],previous_runtime_stopped)
    finally:sys.path.pop(0)
    output.mkdir(parents=True,exist_ok=True)
    print('本地 CPU 恢复包:',{k:v for k,v in imported.items() if k not in ('copied','existing_paths_preserved','excluded','new_cost_events')},flush=True)
    environment=dict(os.environ,PYTHONPATH=str(root))
    for module,package in [('numpy','numpy'),('scipy','scipy'),('skimage','scikit-image'),('psutil','psutil')]:
        print('Probe compact repair',module,flush=True)
        probe=subprocess.run([sys.executable,'-c','import '+module],capture_output=True,env=environment)
        if probe.returncode:subprocess.run([sys.executable,'-m','pip','install',package],check=True)
    args=[sys.executable,'-u','-m','runtime.wan.native_baseline_compact_recovery_run_v1']
    for key,value in config.items():args.extend(['--'+key.replace('_','-'),str(value)])
    if previous_runtime_stopped:args.append('--previous-runtime-stopped')
    worker=None
    try:
        logpath=output/'runner.log'
        with logpath.open('a') as log:
            worker=subprocess.Popen(args,cwd=root,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while worker.poll() is None:
                try:worker.wait(timeout=25)
                except subprocess.TimeoutExpired:print('定向恢复运行中；日志:',logpath,flush=True)
        print('worker returncode',worker.returncode,'日志',logpath,flush=True)
        for name in ('result.json','cpu_recovery.json'):
            path=output/name
            if path.is_file():
                value=json.loads(path.read_text())
                print(name,{k:v for k,v in value.items() if k not in ('views','quality')},flush=True)
        return dict(returncode=worker.returncode,output=str(output))
    finally:
        if worker is not None and worker.poll() is None:
            import signal
            os.killpg(worker.pid,signal.SIGTERM)
            try:worker.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(worker.pid,signal.SIGKILL);worker.wait()
