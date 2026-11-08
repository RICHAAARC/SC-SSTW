"""Continuous stage commits. Interrupted stages restart from committed inputs.

Completed stages and every interrupted attempt remain separate. No internal
model, solver or VAE state checkpoint is required by this runner.
"""
from pathlib import Path
import shutil
import time
import uuid
from main.tube_state import b_paper_cohort_v1 as p
from . import b_paper_cohort_records_v1 as rec
from .paper_checkpoint_v1 import Boundary,BoundaryPause
from .native_baseline_storage_v4 import runtime_session

# Only completed-stage outputs are copied into a fresh next-stage workspace.
ARTIFACTS={
    'prepare':('private','generation_resources.json','resource_history','stages/prepare'),
    'solve':('cpu','private','stages/solve'),
    'replay':('writer_native','stages/replay'),
    'receive':('media','blind','internal_calls.json','stages/receive'),
}


def copy_entry(source,target):
    source,target=Path(source),Path(target)
    if not source.exists():return
    if source.is_dir():shutil.copytree(source,target,dirs_exist_ok=True)
    else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)


def readable_stage(stage,output,scratch):
    paths=[output/'stages'/stage/'result.json']
    if stage=='prepare':
        paths += [output/'private/head.pt',scratch/'OFF_terminal.pt',scratch/'P0_terminal.pt',
                  scratch/'OFF_RGB.npy',scratch/'host/BASE_preclip.npy']
        paths += [scratch/'host/features'/f'{i:03}.pt' for i in (0,*range(1,181,4))]
    elif stage=='solve':paths += [output/'cpu'/name for name in ('handoff.json','q0.npy','coefficients.pt','prediction.pt','schedule.npz')]
    elif stage=='replay':paths += [output/'writer_native/native.json',scratch/'new/NEW_preclip.npy']
    elif stage=='receive':
        paths += [output/'blind'/i['id']/'relative.json' for i in p.inputs()]
        paths += [output/'media'/i['id']/'received.mp4' for i in p.inputs()]
    else:paths += [output/'blind_terminal.json',output/'evaluation_summary.json']
    for path in paths:
        with path.open('rb') as stream:
            if not stream.read(1):raise ValueError('empty required stage output: '+str(path))


def process(stage,output,scratch,remaining,limit,session):
    from .b_paper_cohort_resume_v1 import stage_process
    # No scheduling stop inside a stage; its scientific resource limit remains.
    return stage_process(stage,output,scratch,None,limit,session)


def prepare_work(root,seed,completed,stage,work,sid):
    if not rec.reserve(work,p.source_config(sid)):raise RuntimeError('fresh stage attempt directory required')
    if seed is not None:
        for done in completed:
            for name in ARTIFACTS.get(done,()):copy_entry(seed/name,work/name)
        original=rec.read(seed/'state.json');rows=rec.read(work/'state.json')
        for key,row in original.items():
            if row['stage'] in completed:rows[key]=row
        rec.write(work/'state.json',rows)
    attempt=rec.read(work/'attempt.json');attempt['status']='RUNNING';rec.write(work/'attempt.json',attempt)


def publish(root,work,result):
    # Whole completed package is retained under stage_attempts. Publish ordinary
    # consumer paths last; interrupted publication can repeat only these copies.
    rec.write(root/'stage_terminal.json',dict(output=str(work),result=result))
    for source in work.rglob('*'):
        if not source.is_file():continue
        relative=source.relative_to(work)
        if str(relative) in ('result.json','attempt.json'):continue
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        temporary=target.with_name(target.name+'.publish-'+uuid.uuid4().hex)
        shutil.copy2(source,temporary);temporary.replace(target)
    rec.write(root/'attempt.json',rec.read(work/'attempt.json'))
    rec.write(root/'result.json',result)


def run_source(sid,output_root,cache_root,owner_root,previous_runtime_stopped=False,seconds=None,
               max_stages=None,worker_runner=None,validator=None):
    if sid not in {s['id'] for s in p.ROSTER}:raise ValueError('fixed source required')
    root=Path(output_root)/sid/'attempt01';cache=Path(cache_root)/sid
    from .b_paper_cohort_runner_v1 import scratch_allowed
    scratch_allowed(cache,root)
    if (root/'result.json').is_file() and rec.read(root/'result.json').get('status') in ('COMPLETED','STOPPED_FIXED_ATTEMPT'):
        return rec.recover(root)
    if not (root/'stage_terminal.json').is_file() and (root/'attempt.json').is_file() and rec.read(root/'attempt.json').get('status') in ('COMPLETED','STOPPED_FIXED_ATTEMPT'):
        return rec.recover(root)
    if root.exists() and any(root.iterdir()) and not (root/'checkpoint_control.json').is_file():return rec.recover(root)
    with runtime_session(Path(owner_root)/sid,previous_runtime_stopped=previous_runtime_stopped) as owner:
        if (root/'stage_terminal.json').is_file():
            saved=rec.read(root/'stage_terminal.json');publish(root,Path(saved['output']),saved['result']);return saved['result']
        if not root.exists() or not any(root.iterdir()):
            rec.reserve(root,p.source_config(sid))
            rec.write(root/'checkpoint_control.json',dict(cache=str(cache),stages={},granularity='STAGE'))
        control=rec.read(root/'checkpoint_control.json')
        recovery_budget=Path(control.get('recovery_budget') or Path(owner_root).parent/'recovery'/sid)
        control.update(recovery_budget=str(recovery_budget),granularity='STAGE');rec.write(root/'checkpoint_control.json',control)
        index_path=root/'stage_checkpoint.json'
        if index_path.is_file():index=rec.read(index_path)
        else:
            old=rec.read(root/'checkpoint_control.json');completed=[]
            for stage in p.STAGES:
                state=old.get('stages',{}).get(stage,{})
                saved=root/'stages'/stage/'result.json'
                if state.get('status')=='COMPLETED' or saved.is_file() and rec.read(saved).get('status')=='COMPLETED':completed.append(stage)
                else:break
            # Preserve the pre-migration package before any canonical publication.
            legacy=root/'before_stage_resume'
            legacy.mkdir(exist_ok=True)
            for child in list(root.iterdir()):
                if child.name in ('before_stage_resume','stage_attempts','stage_checkpoint.json'):continue
                copy_entry(child,legacy/child.name)
            index=dict(source=sid,completed=completed,committed_output=str(legacy) if completed else None,
                scratch=old.get('cache',str(cache)),attempts=[],extra_stage_attempts=0,granularity='STAGE',legacy=str(legacy))
            for done in completed:
                saved=legacy/'stages'/done/'result.json'
                stage_result=rec.read(saved) if saved.is_file() else {}
                elapsed=old.get('stages',{}).get(done,{}).get('active_seconds',stage_result.get('wall_seconds'))
                index['attempts'].append(dict(stage=done,status='COMPLETED',output=str(legacy),scratch=index['scratch'],
                    legacy=True,seconds=elapsed,unknown_elapsed=elapsed is None,original_state=str(legacy/'state.json')))
            # A legacy incomplete stage may already have physical calls.
            rows=rec.read(root/'state.json')
            if len(completed)<len(p.STAGES):
                stage=p.STAGES[len(completed)]
                saved=legacy/'stages'/stage/'result.json'
                failed=rec.read(saved) if saved.is_file() else {}
                if failed.get('status')=='STOPPED_FIXED_ATTEMPT' and not failed.get('interrupted'):
                    from .b_paper_cohort_resume_v1 import close_failure
                    work=root/'stage_attempts'/'legacy-terminal'/uuid.uuid4().hex/'output'
                    shutil.copytree(legacy,work)
                    result=close_failure(work,failed.get('failure') or 'saved non-infrastructure stage failure')
                    publish(root,work,result);return result
                if any(r.get('started') for r in rows.values() if r['stage']==stage):
                    index['attempts'].append(dict(stage=stage,status='INTERRUPTED',output=str(legacy),
                        scratch=index['scratch'],legacy=True,seconds=old.get('stages',{}).get(stage,{}).get('active_seconds'),unknown_elapsed=True,
                        original_state=str(legacy/'state.json')))
            rec.write(index_path,index)
        boundary=Boundary(max_stages,seconds)
        try:
            for stage in p.STAGES:
                if stage in index['completed']:continue
                boundary.before()
                previous=[a for a in index['attempts'] if a['stage']==stage]
                # A worker return may have been persisted immediately before the parent died.
                if previous and previous[-1]['status']=='STARTED':
                    attempt=previous[-1];receipt=Path(attempt['output'])/'stages'/stage/'result.json'
                    if receipt.is_file():
                        saved=rec.read(receipt)
                        if saved.get('status')=='COMPLETED':
                            (validator or readable_stage)(stage,Path(attempt['output']),Path(attempt['scratch']))
                            attempt.update(status='COMPLETED',seconds=saved.get('wall_seconds'))
                            index['completed'].append(stage);index.update(committed_output=attempt['output'],scratch=attempt['scratch'])
                            rec.write(index_path,index);continue
                        if not saved.get('interrupted'):
                            from .b_paper_cohort_resume_v1 import close_failure
                            result=close_failure(Path(attempt['output']),saved.get('failure') or 'saved stage failure')
                            publish(root,Path(attempt['output']),result);return result
                    attempt.update(status='INTERRUPTED',unknown_elapsed=True);rec.write(index_path,index)
                retry=bool(previous)
                if retry and index['extra_stage_attempts']>=2:
                    from .b_paper_cohort_resume_v1 import close_failure
                    work=Path(previous[-1]['output']);result=close_failure(work,'source infrastructure stage recoveries exhausted')
                    publish(root,work,result);return result
                uid=uuid.uuid4().hex;work=root/'stage_attempts'/stage/uid/'output'
                seed=Path(index['committed_output']) if index['committed_output'] else None
                scratch=Path(index['scratch'])
                if stage=='prepare':scratch=cache/('prepare-'+uid)
                elif stage=='replay' and retry:
                    # Replay owns only new/. Preserve the failed cache; copy its
                    # committed inputs to a fresh cache before the new attempt.
                    fresh=cache.parent/(sid+'-replay-'+uid)
                    shutil.copytree(scratch,fresh,ignore=shutil.ignore_patterns('new','operation_returns'))
                    scratch=fresh
                scratch.mkdir(parents=True,exist_ok=True)
                prepare_work(root,seed,index['completed'],stage,work,sid)
                recovery=None
                if retry:
                    from .paper_recovery_v1 import reserve
                    recovery=reserve(recovery_budget,root,stage,previous[-1])
                    index['extra_stage_attempts']+=1
                attempt=dict(stage=stage,status='STARTED',output=str(work),scratch=str(scratch),
                    recovery=retry,started_at=time.time(),seconds=None,original_state=str(work/'state.json'),
                    internal_calls=str(work/'stages'/stage/'internal_calls.json'))
                index['attempts'].append(attempt);rec.write(index_path,index)
                tick=time.monotonic();limit=p.LIMITS['receive_seconds'] if stage=='receive' else p.LIMITS['identity_seconds'] if stage=='identity' else None
                try:
                    result=(worker_runner or process)(stage,work,scratch,None,limit,uid)
                except (KeyboardInterrupt,InterruptedError) as error:
                    result=dict(status='PAUSED_BOUNDARY',interrupted=True,failure=str(error))
                attempt.update(seconds=time.monotonic()-tick,result=result)
                if recovery is not None:
                    path,receipt=recovery;receipt.update(status='RETURNED' if result.get('status')=='COMPLETED' else 'INTERRUPTED' if result.get('interrupted') else 'FAILED',
                        seconds=attempt['seconds'],stage_output=str(work));rec.write(path,receipt)
                if result.get('resource_failure') or result.get('failure') in ('WORKER_WALL_LIMIT','PROCESS_TREE_RSS_LIMIT'):
                    result=dict(result,status='STOPPED_FIXED_ATTEMPT',interrupted=False)
                if result.get('interrupted') or result.get('status')=='PAUSED_BOUNDARY':
                    attempt['status']='INTERRUPTED';rec.write(index_path,index)
                    raise BoundaryPause('interrupted stage retained; next Run all reexecutes only this stage')
                if result.get('status')!='COMPLETED':
                    attempt['status']='FAILED';rec.write(index_path,index)
                    from .b_paper_cohort_resume_v1 import close_failure
                    result=close_failure(work,result.get('failure') or 'stage failure')
                    publish(root,work,result);return result
                (validator or readable_stage)(stage,work,scratch)
                attempt['status']='COMPLETED';index['completed'].append(stage)
                index.update(committed_output=str(work),scratch=str(scratch));rec.write(index_path,index)
                boundary.returned()
            work=Path(index['committed_output']);result=rec.finish(work)
            rec.write(work/'chain_resources.json',dict(stage_attempts=index['attempts'],
                wall_seconds=sum(a.get('seconds') or 0 for a in index['attempts']),
                unknown_elapsed=any(a.get('unknown_elapsed') for a in index['attempts']),
                extra_stage_attempts=index['extra_stage_attempts'],
                scope='all original and infrastructure attempts; fixed logical denominators unchanged',whole_chain_limit_certified=False))
            publish(root,work,result);return result
        except BoundaryPause as error:
            rec.write(index_path,index)
            return dict(status='PAUSED_BOUNDARY',source=sid,reason=str(error),granularity='STAGE')
