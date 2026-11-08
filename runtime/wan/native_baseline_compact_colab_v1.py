"""User-run compact entry. It never resumes or stops the original v4 worker."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from main.baselines.public_compact12_v1 import OUTPUT,CACHE
from .native_baseline_colab_v4 import check_previous_worker,current_progress,stop_group
from .native_baseline_storage_v1 import read_json,write_json


def run_colab(source_directory=None,source_ref='main',previous_runtime_stopped=False):
    from .paper_suite_colab_v1 import available_source
    from .paper_suite_worker_v1 import prepare_vendor
    output=Path(OUTPUT);output.mkdir(parents=True,exist_ok=True)
    # Checks only ownership of the NEW package. Original v4 is never taken over.
    check_previous_worker(output,previous_runtime_stopped)
    checkpoint=read_json(output/'compute_checkpoint.json') if (output/'compute_checkpoint.json').is_file() else {}
    if checkpoint.get('status')=='DONE':
        return dict(status='READ_ONLY_COMPLETED',output=str(output),result=read_json(output/'result.json'))
    source=available_source(source_directory,source_ref);vendor=prepare_vendor();worker=None
    environment=dict(os.environ,PYTHONPATH=str(source))
    for module,package in [('numpy','numpy'),('scipy','scipy'),('torch','torch'),('skimage','scikit-image'),('lpips','lpips'),('PIL','pillow')]:
        probe=subprocess.run([sys.executable,'-c','import '+module],capture_output=True,env=environment)
        if probe.returncode:subprocess.run([sys.executable,'-m','pip','install',package],check=True)
    args=[sys.executable,'-u','-m','runtime.wan.native_baseline_compact_entry_v1','--vendor',str(vendor)]
    if previous_runtime_stopped:args.append('--previous-runtime-stopped')
    try:
        launched=time.time();last=None
        with (output/'runner.log').open('a') as log:
            worker=subprocess.Popen(args,cwd=source,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while worker.poll() is None:
                try:worker.wait(timeout=25)
                except subprocess.TimeoutExpired:
                    row,last=current_progress(output,worker.pid,launched,last);print(json.dumps(row,ensure_ascii=False),flush=True)
        status='RUN_FINISHED_WITH_TABLES' if worker.returncode==0 else 'INTERRUPTED' if worker.returncode==130 else 'INTERRUPTED_RETAINED'
        value=dict(status=status,returncode=worker.returncode,output=str(output),at=time.time())
        write_json(output/'bootstrap.json',value)
        if (output/'result.json').is_file():print(read_json(output/'result.json'),flush=True)
        return value
    finally:stop_group(worker)
