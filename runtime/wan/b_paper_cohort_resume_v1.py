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
    if code!=0 or resources.get('failure') or result.get('status') not in ('COMPLETED','PAUSED_BOUNDARY'):
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
    output=Path(output_root)/sid/'attempt01';scratch=Path(cache_root)/sid
    if sid not in {s['id'] for s in p.ROSTER}:raise ValueError('fixed source required')
    if output.exists() and any(output.iterdir()):
        attempt=rec.read(output/'attempt.json')
        if not (output/'checkpoint_control.json').is_file() or attempt['status'] in ('COMPLETED','STOPPED_FIXED_ATTEMPT'):
            return rec.recover(output)
    from .b_paper_cohort_runner_v1 import scratch_allowed
    scratch_allowed(scratch,output)
    with runtime_session(Path(owner_root)/sid,previous_runtime_stopped=previous_runtime_stopped) as owner:
        if not output.exists() or not any(output.iterdir()):
            if not rec.reserve(output,p.source_config(sid)):return rec.recover(output)
            rec.write(output/'checkpoint_control.json',dict(source=sid,cache=str(scratch),stages={},
                boundary_semantics='only complete durable returns can resume; consumed uncertain calls never repeat'))
        control=rec.read(output/'checkpoint_control.json');scratch=Path(control['cache']);scratch_allowed(scratch,output)
        rows=rec.read(output/'state.json')
        bad=[k for k,r in rows.items() if r['status'] not in ('NOT_STARTED','RETURNED') and not (output/'checkpoints/returns'/(k+'.json')).is_file()]
        if bad:return close_failure(output,'UNRECOVERABLE_CONSUMED_OPERATIONS: '+str(bad))
        attempt=rec.read(output/'attempt.json');attempt.update(status='RUNNING');rec.write(output/'attempt.json',attempt)
        boundary=Boundary(max_stages,seconds);start=time.monotonic();session=owner['id'];scratch.mkdir(parents=True,exist_ok=True)
        rec.write(output/'scratch.json',dict(path=str(scratch),root=str(cache_root),persistent_checkpoint_cache=True))
        failure=None
        try:
            for stage in p.STAGES:
                state=control['stages'].get(stage,{})
                if state.get('status')=='STARTED':
                    receipt=Path(state['receipt'])
                    if not receipt.is_file():raise RuntimeError('stage ended without a completed return/time checkpoint: '+stage)
                    saved=rec.read(receipt)
                    if saved.get('status') not in ('COMPLETED','PAUSED_BOUNDARY'):raise RuntimeError('saved stage failed: '+stage)
                    state=dict(state,status=saved['status'],active_seconds=state.get('active_seconds',0.)+saved['wall_seconds'])
                    control['stages'][stage]=state;rec.write(output/'checkpoint_control.json',control)
                if state.get('status')=='COMPLETED':continue
                boundary.before()
                elapsed=state.get('active_seconds',0.)
                cap=p.LIMITS['receive_seconds'] if stage=='receive' else p.LIMITS['identity_seconds'] if stage=='identity' else None
                if cap is not None and elapsed>=cap:raise RuntimeError('fixed cumulative stage time consumed: '+stage)
                remaining=None if boundary.deadline is None else max(.001,boundary.deadline-time.monotonic())
                tick=time.monotonic()
                control['stages'][stage]=dict(status='STARTED',active_seconds=elapsed,last_session=session,
                    receipt=str(output/'continuations'/session/stage/'worker_result.json'))
                rec.write(output/'checkpoint_control.json',control)
                result=(worker_runner or stage_process)(stage,output,scratch,remaining,None if cap is None else cap-elapsed,session)
                control['stages'][stage]=dict(status=result['status'],active_seconds=elapsed+time.monotonic()-tick,last_session=session)
                rec.write(output/'checkpoint_control.json',control)
                if result['status']=='PAUSED_BOUNDARY':raise BoundaryPause('worker paused at saved operation boundary')
                if result['status']!='COMPLETED':raise RuntimeError(result.get('failure') or 'stage failed')
                boundary.returned()
            if not (output/'blind_terminal.json').is_file():rec.close_blind(output)
            result=rec.finish(output)
        except BoundaryPause as error:
            attempt=rec.read(output/'attempt.json');attempt.update(status='PAUSED_BOUNDARY');rec.write(output/'attempt.json',attempt)
            result=dict(status='PAUSED_BOUNDARY',source=sid,reason=str(error),method_calls_reset=False)
            rec.write(output/'pause.json',result)
        except BaseException as error:
            failure=f'{type(error).__name__}: {error}'
            rows=rec.read(output/'state.json')
            uncertain=[k for k,r in rows.items() if r['status'] not in ('NOT_STARTED','RETURNED') and not (output/'checkpoints/returns'/(k+'.json')).is_file()]
            if isinstance(error,(KeyboardInterrupt,InterruptedError)) and not uncertain:
                attempt=rec.read(output/'attempt.json');attempt.update(status='PAUSED_BOUNDARY');rec.write(output/'attempt.json',attempt)
                result=dict(status='PAUSED_BOUNDARY',source=sid,reason=failure)
            else:result=close_failure(output,failure)
        rec.write(output/'continuations'/session/'result.json',dict(result,wall_seconds=time.monotonic()-start))
        rec.write(output/'chain_resources.json',dict(stage_active_seconds={k:v['active_seconds'] for k,v in control['stages'].items()},
            wall_seconds=sum(v['active_seconds'] for v in control['stages'].values()),
            restore_overhead=str(output/'restore_overhead.jsonl'),source_attempts=1,
            scratch_lifetime='persistent separate Drive cache; never deleted on interruption',whole_chain_limit_certified=False))
        if result.get('status')=='COMPLETED':
            from .paper_checkpoint_v1 import release_completed_cache
            try:release_completed_cache(output,scratch)
            except OSError as error:rec.write(output/'cache_cleanup_error.json',dict(error=str(error)))
        return result
