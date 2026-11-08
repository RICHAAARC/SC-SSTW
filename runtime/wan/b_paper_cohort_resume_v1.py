"""Durable source/stage continuation. Existing terminal attempts are immutable."""
import os
from pathlib import Path
import sys
import time
import uuid
from main.tube_state import b_paper_cohort_v1 as p
from . import b_paper_cohort_records_v1 as rec
from .paper_checkpoint_v1 import Boundary,BoundaryPause
from .native_baseline_storage_v4 import runtime_session

CACHE_ROOT=Path('/content/drive/MyDrive/Video-WM_Cache/paper-checkpoints-v1/cohort')
COORD_ROOT=Path('/content/drive/MyDrive/Video-WM_Result/paper_execution_v1/owners/cohort')
ROOT=Path(__file__).resolve().parents[2]


def stage_process(stage,output,scratch,seconds,limit,session):
    from .b_worker_process_v1 import stage_process as original,watch
    folder=output/'continuations'/session/stage;folder.mkdir(parents=True,exist_ok=True)
    args=[sys.executable,'-u','-m','runtime.wan.b_paper_cohort_worker_v1','--worker-stage',stage,
        '--output',str(output),'--scratch',str(scratch),'--stage-receipt',str(folder/'worker_result.json')]
    if seconds is not None:args+=['--boundary-seconds',str(max(.001,seconds))]
    if limit is not None:
        env=dict(os.environ,PYTHONPATH=str(ROOT))
        resources=watch(args,folder,cwd=ROOT,environment=env,limits=dict(worker_seconds=limit,
            process_tree_rss_bytes=8*1024**3,poll_seconds=.1,terminate_grace_seconds=2))
        code=resources['worker_returncode']
    else:resources=original(args,folder/'worker.log',cwd=ROOT);code=resources['returncode']
    rec.write(folder/'resources.json',resources)
    path=folder/'worker_result.json'
    result=rec.read(path) if path.is_file() else {}
    failure=resources.get('failure')
    if failure in ('WORKER_WALL_LIMIT','PROCESS_TREE_RSS_LIMIT'):
        return dict(status='STOPPED_FIXED_ATTEMPT',failure=failure,interrupted=False,resource_failure=True)
    if result.get('interrupted') or str(failure).startswith(('KeyboardInterrupt:','InterruptedError:')) or code is not None and code<0 or code==130:
        if failure not in ('WORKER_WALL_LIMIT','PROCESS_TREE_RSS_LIMIT'):
            return dict(status='PAUSED_BOUNDARY',interrupted=True,failure=failure or result.get('failure'),original_returncode=code)
    if code!=0 or failure or result.get('status') not in ('COMPLETED','PAUSED_BOUNDARY'):
        return dict(status='STOPPED_FIXED_ATTEMPT',failure=resources.get('failure') or result.get('failure') or 'worker returned without complete boundary',
                    interrupted=result.get('interrupted',False))
    return result


def close_failure(output,failure):
    from .b_paper_cohort_worker_v1 import finalize_raw
    from .b_paper_cohort_evaluation_v1 import post_once
    finalize_raw(output,rec.read(output/'state.json'));rec.close_blind(output,failure)
    if rec.read(output/'post_status.json')['status']=='NOT_STARTED':
        try:post_once(output)
        except Exception as error:rec.write(output/'post_failure.json',dict(error=str(error)))
    return rec.finish(output,failure)


def run_source(sid,output_root=p.OUTPUT_ROOT,cache_root=CACHE_ROOT,owner_root=COORD_ROOT,
               previous_runtime_stopped=False,seconds=None,max_stages=None,worker_runner=None):
    from .paper_stage_resume_v1 import run_source as stages
    return stages(sid,output_root,cache_root,owner_root,previous_runtime_stopped,seconds,max_stages,worker_runner)
