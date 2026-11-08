"""Small artificial stage transactions; no model, media or scientific replay."""
from pathlib import Path
import pytest
from main.tube_state import b_paper_cohort_v1 as p
from runtime.wan import paper_stage_resume_v1 as stage
from runtime.wan import b_paper_cohort_records_v1 as rec


def runner(calls,interrupt=None):
    failed=set()
    def run(name,out,cache,*args):
        calls.append(name)
        rows=rec.read(out/'state.json')
        for row in rows.values():
            if row['stage']==name:row.update(status='RETURNED',started=1,returned=1)
        rec.write(out/'state.json',rows)
        directory={'prepare':'private','solve':'cpu','replay':'writer_native','receive':'blind','identity':'final'}[name]
        rec.write(out/directory/'fixture.json',dict(stage=name,value=p.STAGES.index(name)+1))
        if name=='prepare':(cache/'original.fixture').write_text('original input')
        if name==interrupt and name not in failed:
            failed.add(name);rec.write(out/'partial.json',dict(retained=True));raise KeyboardInterrupt('artificial stage lost')
        if name=='identity':
            rec.write(out/'post_status.json',dict(status='RETURNED',started=1,returned=1))
            rec.write(out/'evaluation_summary.json',dict(value=15))
        result=dict(stage=name,status='COMPLETED',wall_seconds=.01)
        rec.write(out/'stages'/name/'result.json',result);return result
    return run


def kwargs(tmp_path,call):
    return dict(output_root=tmp_path/'out',cache_root=tmp_path/'cache',owner_root=tmp_path/'owners/cohort',
        worker_runner=call,validator=lambda *args:None)


def test_completed_stages_skip_and_interrupted_attempt_is_retained(tmp_path):
    sid=p.ROSTER[0]['id'];calls=[];kw=kwargs(tmp_path,runner(calls,'solve'))
    first=stage.run_source(sid,**kw);assert first['status']=='PAUSED_BOUNDARY'
    root=tmp_path/'out'/sid/'attempt01';before=rec.read(root/'stage_checkpoint.json')
    assert before['completed']==['prepare'] and len(before['attempts'])==2
    failed=Path(before['attempts'][1]['output']);saved=(failed/'partial.json').read_bytes()
    result=stage.run_source(sid,**kw);assert result['status']=='COMPLETED'
    assert calls==['prepare','solve','solve','replay','receive','identity']
    assert (failed/'partial.json').read_bytes()==saved
    after=rec.read(root/'stage_checkpoint.json');assert after['completed']==list(p.STAGES)
    assert after['extra_stage_attempts']==1 and len(after['attempts'])==6
    assert all(r['started']==r['returned']==1 for r in rec.read(root/'state.json').values())
    assert stage.run_source(sid,**kw)['status']=='READ_ONLY'
    other=[];reference=stage.run_source(sid,**kwargs(tmp_path/'continuous',runner(other)))
    assert result['counts']==reference['counts'] and result['denominators']==reference['denominators']
    assert rec.read(root/'evaluation_summary.json')==dict(value=15)


def test_complete_worker_return_before_parent_commit_is_reused(tmp_path,monkeypatch):
    sid=p.ROSTER[0]['id'];calls=[];kw=kwargs(tmp_path,runner(calls));original=stage.rec.write;fail=[True]
    def lost(path,value):
        if str(path).endswith('stage_checkpoint.json') and value.get('completed')==['prepare'] and fail[0]:
            fail[0]=False;raise KeyboardInterrupt()
        original(path,value)
    with monkeypatch.context() as patch:
        patch.setattr(stage.rec,'write',lost)
        with pytest.raises(KeyboardInterrupt):stage.run_source(sid,**kw)
    result=stage.run_source(sid,**kw)
    assert result['status']=='COMPLETED' and calls==list(p.STAGES)


def test_unreadable_complete_stage_is_not_committed(tmp_path):
    sid=p.ROSTER[0]['id'];calls=[];kw=kwargs(tmp_path,runner(calls))
    def unreadable(*args):raise FileNotFoundError('missing actual output')
    kw['validator']=unreadable
    with pytest.raises(FileNotFoundError):stage.run_source(sid,**kw)
    saved=rec.read(tmp_path/'out'/sid/'attempt01/stage_checkpoint.json')
    assert saved['completed']==[] and saved['attempts'][0]['status']=='STARTED'


def test_stage_pause_default_and_fixed_shards():
    import inspect
    from runtime.wan.paper_suite_colab_v1 import run_parallel_colab
    from main.tube_state.paper_execution_v1 import main_sources
    assert inspect.signature(run_parallel_colab).parameters['max_items'].default is None
    assert inspect.signature(run_parallel_colab).parameters['max_hours'].default is None
    assert set(main_sources(0)).isdisjoint(main_sources(1))
    assert len(main_sources(0)+main_sources(1))==12


def test_legacy_real_failure_cannot_receive_infrastructure_retry(tmp_path):
    sid=p.ROSTER[0]['id'];root=tmp_path/'out'/sid/'attempt01'
    rec.reserve(root,p.source_config(sid))
    rec.write(root/'checkpoint_control.json',dict(cache=str(tmp_path/'cache'/sid),stages={}))
    rows=rec.read(root/'state.json');rows['generate_P0'].update(status='FAILED',started=1,error='nonfinite');rec.write(root/'state.json',rows)
    rec.write(root/'stages/prepare/result.json',dict(status='STOPPED_FIXED_ATTEMPT',interrupted=False,failure='nonfinite'))
    result=stage.run_source(sid,**kwargs(tmp_path,lambda *args:pytest.fail('scientific failure retried')))
    assert result['status']=='STOPPED_FIXED_ATTEMPT' and result['failure']=='nonfinite'
    assert rec.read(root/'before_stage_resume/state.json')['generate_P0']['error']=='nonfinite'


def test_hard_limit_wins_over_child_interrupt_flag(tmp_path,monkeypatch):
    from runtime.wan import b_worker_process_v1 as process,b_paper_cohort_resume_v1 as resume
    def watch(args,out,**kwargs):
        rec.write(out/'worker_result.json',dict(status='STOPPED_FIXED_ATTEMPT',interrupted=True,failure='KeyboardInterrupt'))
        return dict(worker_returncode=-15,failure='PROCESS_TREE_RSS_LIMIT')
    monkeypatch.setattr(process,'watch',watch)
    result=resume.stage_process('identity',tmp_path,tmp_path/'cache',None,10,'fixture')
    assert result['status']=='STOPPED_FIXED_ATTEMPT' and not result['interrupted'] and result['resource_failure']


def test_original_stage_worker_adapters_connect(tmp_path):
    import importlib.util
    from runtime.wan import b_paper_cohort_worker_v1 as worker
    path=Path(__file__).with_name('test_b_paper_cohort_v1.py')
    spec=importlib.util.spec_from_file_location('paper_stage_fixtures',path);fake=importlib.util.module_from_spec(spec);spec.loader.exec_module(fake)
    def actual(name,out,cache,*args):
        fake.FakeIdentity.output=out
        return worker.worker(name,out,cache,generation_factory=fake.FakeGeneration,time_factory=fake.FakeTime,identity_factory=fake.FakeIdentity)
    result=stage.run_source(p.ROSTER[0]['id'],**kwargs(tmp_path,actual))
    assert result['status']=='COMPLETED',result
    assert result['operations']=={'RETURNED':1114}


def test_same_wan_task_recovery_keeps_base_row_and_boundary_does_not_spend(tmp_path):
    from runtime.wan.paper_checkpoint_v1 import DurableJournal,Boundary,BoundaryPause
    from runtime.wan.paper_recovery_v1 import ACTIVE
    output=tmp_path/'out';state=output/'ops.json';cache=tmp_path/'cache';budget=tmp_path/'shared'
    rec.write(state,dict(x=dict(kind='reader',status='STARTED',started=1,returned=0)))
    j=DurableJournal(output,state,cache,Boundary(max_items=1));j.allow_infrastructure_retry=True;j.recovery_budget=budget
    j.boundary.completed=1
    with pytest.raises(BoundaryPause):j.call('x',lambda:pytest.fail('called beyond boundary'))
    assert not budget.exists() and ACTIVE.get() is None
    j.boundary=Boundary()
    assert j.call('x',lambda:dict(actual_extra=ACTIVE.get() is not None))==dict(actual_extra=True)
    assert rec.read(state)['x']['started']==1 and len(list(budget.glob('attempt-*.json')))==1
    assert ACTIVE.get() is None


def test_interrupted_final_publication_only_resumes_copy(tmp_path,monkeypatch):
    sid=p.ROSTER[0]['id'];root=tmp_path/'out'/sid/'attempt01';work=root/'stage_attempts/final/output'
    rec.write(root/'checkpoint_control.json',dict(cache=str(tmp_path/'cache'/sid)))
    rec.write(work/'attempt.json',dict(status='COMPLETED'))
    rec.write(work/'payload.json',dict(value=19))
    result=dict(status='COMPLETED');original=stage.shutil.copy2
    with monkeypatch.context() as patch:
        patch.setattr(stage.shutil,'copy2',lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):stage.publish(root,work,result)
    assert not (root/'result.json').exists()
    assert stage.run_source(sid,**kwargs(tmp_path,lambda *a:pytest.fail('publication restarted science')))==result
    assert rec.read(root/'payload.json')==dict(value=19)
