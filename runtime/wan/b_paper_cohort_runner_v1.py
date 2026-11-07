"""Sequential consumed sources with original worker boundaries and owned scratch."""
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from main.tube_state import b_paper_cohort_v1 as p
from . import b_paper_cohort_records_v1 as rec

ROOT=Path(__file__).resolve().parents[2]
CLI='experiments.wan_state_clock.b_paper_cohort_v1_run'


def subprocess_stage(stage,output,scratch):
    from .b_joint_time_payload_r02_v1 import stage_process
    from .b_relative_time_payload_runner_v1 import watch
    command=[sys.executable,'-u','-m',CLI,'--worker-stage',stage,'--output',str(output),'--scratch',str(scratch)]
    if stage=='identity':
        environment=os.environ.copy();environment['PYTHONPATH']=str(ROOT)
        resource=watch(command,output/'stages/identity',cwd=ROOT,environment=environment,
            limits=dict(worker_seconds=18900,process_tree_rss_bytes=8*1024**3,poll_seconds=.1,terminate_grace_seconds=2))
        code=resource['worker_returncode']
    else:
        resource=stage_process(command,output/(stage+'.log'),cwd=ROOT);code=resource['returncode']
    rec.write(output/'stages'/stage/'resources.json',resource)
    result=rec.read(output/'stages'/stage/'result.json') if (output/'stages'/stage/'result.json').is_file() else {}
    if code!=0 or resource.get('failure') or not result:
        saved=result if isinstance(result,dict) else {}
        return dict(status='STOPPED_FIXED_ATTEMPT',
            failure=resource.get('failure') or saved.get('failure') or f'worker exit {code}, result missing={not bool(result)}',
            interrupted=saved.get('interrupted') is True)
    return result


def directory_bytes(path):
    return sum(f.stat().st_size for f in Path(path).rglob('*') if f.is_file())


def scratch_allowed(scratch_root,output):
    scratch_root,output=Path(scratch_root).resolve(),Path(output).resolve()
    if (scratch_root==output or scratch_root in output.parents or output in scratch_root.parents
            or 'Video-WM_Result' in scratch_root.parts):
        raise ValueError('new scratch must be separate from retained result trees')
    return scratch_root


def execute_reserved(output,scratch_root,*,worker_runner=None):
    from .b_joint_time_payload_r02_v1 import termination_handler
    output=Path(output);attempt=rec.read(output/'attempt.json')
    if attempt['status']!='RESERVED': return rec.recover(output)
    try:
        with (output/'supervisor_started.json').open('x') as f: f.write('{"consumed":true}')
    except FileExistsError: return rec.recover(output)
    attempt['status']='RUNNING';rec.write(output/'attempt.json',attempt)
    failure=None;scratch=None;observations=[];started=time.monotonic()
    with termination_handler() as interruption:
        try:
            scratch_root=scratch_allowed(scratch_root,output);scratch_root.mkdir(parents=True,exist_ok=True)
            scratch=Path(tempfile.mkdtemp(prefix='paper-cohort-owned-',dir=scratch_root)).resolve()
            rec.write(output/'scratch.json',dict(created_by_this_invocation=True,path=str(scratch),root=str(scratch_root)))
            for stage in p.STAGES[:-1]:
                (output/'stages'/stage).mkdir(parents=True,exist_ok=True)
                result=(worker_runner or subprocess_stage)(stage,output,scratch)
                observations.append(dict(after_stage=stage,scratch_bytes=directory_bytes(scratch),result_bytes=directory_bytes(output)))
                if result.get('interrupted'): interruption['requested']=True
                if result['status']!='COMPLETED':
                    failure=result.get('failure') or f'{stage} did not complete';break
        except BaseException as error:
            failure=f'{type(error).__name__}: {error}'
            if isinstance(error,(KeyboardInterrupt,InterruptedError)): interruption['requested']=True
        finally:
            # Only this call's fresh mkdtemp is eligible; no retained or pre-existing path is removed.
            if scratch is not None:
                try:
                    target=scratch.resolve();base=Path(scratch_root).resolve()
                    if target.parent!=base or target==output.resolve() or 'Video-WM_Result' in target.parts:
                        raise ValueError('owned scratch cleanup path escaped its creation root')
                    size=directory_bytes(target);shutil.rmtree(target)
                    rec.write(output/'scratch_release.json',dict(status='RELEASED',path=str(target),bytes_at_release=size,
                        stage_boundary_observations=observations,peak_is_sampled_not_certified=True))
                except BaseException as error:
                    if isinstance(error,(KeyboardInterrupt,InterruptedError)): interruption['requested']=True
                    rec.write(output/'scratch_release.json',dict(status='FAILED',error=str(error),stage_boundary_observations=observations))
                    failure=failure or f'scratch release: {type(error).__name__}: {error}'
        if failure: rec.write(output/'dependency_failure.json',dict(failure=failure))
        if not interruption['requested']:
            try:
                (output/'stages/identity').mkdir(parents=True,exist_ok=True)
                result=(worker_runner or subprocess_stage)('identity',output,scratch or Path(scratch_root))
                if result.get('interrupted'): interruption['requested']=True
                if result['status']!='COMPLETED': failure=failure or result.get('failure') or 'identity stopped'
            except BaseException as error:
                failure=failure or f'{type(error).__name__}: {error}'
                if isinstance(error,(KeyboardInterrupt,InterruptedError)): interruption['requested']=True
        if not (output/'blind_terminal.json').is_file(): rec.close_blind(output,failure)
    rec.write(output/'chain_resources.json',dict(wall_seconds=time.monotonic()-started,
        whole_chain_limit_certified=False,external_interruption=interruption,
        scratch_lifetime='fresh per source, released after time stage termination before identity',
        source_attempts=1))
    return rec.finish(output,failure,interrupted=interruption['requested'])


def run_source(config,output,scratch_root,**kwargs):
    output=Path(output)
    if output.exists() and any(output.iterdir()): return rec.recover(output)
    if not rec.reserve(output,config): return rec.recover(output)
    return execute_reserved(output,scratch_root,**kwargs)


def run_batch(batch='TEST-A',output_root=p.OUTPUT_ROOT,scratch_root='/content/video-wm-paper-cohort-v1/work',**kwargs):
    from .b_paper_cohort_evaluation_v1 import cohort_summary
    results=[];root=Path(output_root);started=False
    for source in p.batch_sources(batch):
        output=root/source['id']/'attempt01'
        result=run_source(p.source_config(source['id']),output,scratch_root,**kwargs)
        started=started or result['status']!='READ_ONLY';results.append(dict(source=source['id'],result=result))
        if result.get('interrupted'): break
    summary=cohort_summary(root,batch);all_sources=cohort_summary(root,'ALL')
    if started:
        rec.write(root/(batch+'_summary.json'),summary);rec.write(root/'ALL_summary.json',all_sources)
    return dict(status='BATCH_INTERRUPTED' if results[-1]['result'].get('interrupted') else 'BATCH_RETURNED',
                batch=batch,results=results,summary=summary,all_sources=all_sources,output=str(root))
