"""New artificial fixtures only; no real media, saved r01 results, weights or codec."""
import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
from main.tube_state import b_joint_time_payload_r02_v1 as protocol
from main.tube_state import b_joint_time_relative_validation_v1 as old_protocol
from runtime.wan import b_joint_time_payload_r02_v1 as runner
from runtime.wan import b_joint_time_payload_r02_records_v1 as rec
from runtime.wan import b_relative_time_payload_records_v1 as payload_rec


def fixture_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tests'/f'{name}.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


old_time = fixture_module('test_b_joint_time_relative_validation_v1')
old_payload = fixture_module('test_b_relative_time_payload_v1')


def config(): return rec.read(runner.DEFAULT_CONFIG)


class ArtificialTime(old_time.ArtificialBackend):
    def encode(self, pixels, path):
        super().encode(pixels, path)
        path.write_text('ARTIFICIAL FILE, NOT MEDIA')


class ArtificialPayload(old_payload.FakeBackend):
    def load(self):
        assert not (self.output.parent/'time_chain/post_truth_access.json').exists()
        assert not (self.output.parent/'joint_blind_terminal.json').exists()
        return super().load()
    def probe(self, path, n):
        assert 'joint-relative-r02' in path.parts and path.is_file()
        return super().probe(path,n)
    def bits(self, projection, uid, key):
        value = super().bits(projection, uid, key)
        value.update(view_id=uid, kind='E41', local_rows=list(range(1,11)))
        return value


def time_worker(stage, output, scratch):
    return runner.temporal.worker(stage, output, scratch, backend_factory=ArtificialTime)


def payload_inline(output):
    runner.payload_stage(output, backend_factory=lambda _: ArtificialPayload(output/'payload'))
    return dict(worker_returncode=0, failure=None, artificial=True)


def metadata_time(stage, output, scratch):
    """Artificial returned receipts for exercising failure branches without repeated method work."""
    cfg = rec.read(output/'protocol.json')
    rows = old_protocol.stage_operations(cfg, stage)
    folder = output/'stages'/stage; folder.mkdir(parents=True, exist_ok=True)
    (folder/'events.jsonl').write_text(''.join(json.dumps(dict(row,status='RETURNED'))+'\n' for row in rows))
    if stage == 'receive':
        for unit in old_protocol.units(cfg):
            n=unit['frames']; value=old_payload.relative(list(range(n)))
            # Use only legal observed source positions; keep uncertainty at received frame181.
            if n==182: value['conditional_nodes'][-1].update(conditional_position=None,conditional_candidates=[1,2],status='COMPARISON_TIE')
            value['original_f42']=dict(nodes=[dict(W=[]) for _ in range(n)],
                                      edges=[dict(exact_singleton_pair=False,pairs=[]) for _ in range(n-1)])
            value['conditional_edges']=value['original_f42']['edges']
            rec.write(output/'blind'/unit['id']/'relative.json',value)
            path=output/'media'/unit['id']/'received.mp4';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('ARTIFICIAL FILE, NOT MEDIA')
    return dict(status='COMPLETED', artificial=True)


def snapshot(output):
    return {str(p.relative_to(output)):(p.stat().st_size,p.stat().st_mtime_ns) for p in output.rglob('*') if p.is_file()}


def test_r02_fixed_protocol_and_reservation(tmp_path):
    c=config(); protocol.validate(c)
    with pytest.raises(ValueError): old_protocol.validate(c)
    wrong=deepcopy(c);wrong['writer']['gamma']=2
    with pytest.raises(ValueError):protocol.validate(wrong)
    output=tmp_path/'attempt';assert rec.reserve(output,c)
    a=rec.read(output/'attempt.json')
    assert (len(a['time_operations']),len(a['payload_operations']))==(78,551)
    assert sum(len(x['nodes']) for x in a['time_slots'])==1644
    assert sum(len(x['edges']) for x in a['time_slots'])==1634
    assert sum(len(bits) for keys in a['raw_slots'].values() for bits in keys.values())==28800
    assert sum(len(x['slots']) for x in a['payload_slots'] if x['kind']=='final')==800
    assert all('r01' not in row['id'] for row in a['time_operations'])
    assert 'writer' not in payload_rec.read(output/'payload/protocol.json')


def test_actual_time_worker_r02_and_full_629_joint_order(tmp_path,monkeypatch):
    output=tmp_path/'attempt'; observed=[]
    original=payload_rec.read
    def guarded(path):
        if Path(path) in (output/'time_chain/private/attack_truth.json',output/'time_chain/protocol.json'):
            assert rec.read(output/'joint_blind_terminal.json')['status']=='JOINT_BLIND_TERMINAL'
            observed.append(str(path))
        return original(path)
    monkeypatch.setattr(payload_rec,'read',guarded)
    from runtime.wan import b_joint_time_relative_evaluation_v1 as evaluation
    evaluate=evaluation.evaluate_saved
    def after_joint(path, **kwargs):
        assert rec.read(output/'joint_blind_terminal.json')['status']=='JOINT_BLIND_TERMINAL'
        return evaluate(path, **kwargs)
    monkeypatch.setattr(evaluation,'evaluate_saved',after_joint)
    ArtificialTime.fail=False
    result=runner.run(config(),output,tmp_path/'scratch',worker_runner=time_worker,payload_runner=payload_inline)
    assert result['status']=='COMPLETED',result
    assert result['operations']['time']['status_counts']=={'RETURNED':78}
    assert result['operations']['payload']['status_counts']=={'RETURNED':551}
    assert len(observed)==2
    posts=rec.read(output/'posts.json')
    assert posts['time']['started']==posts['payload']['started']==1
    assert rec.read(output/'payload/raw_terminal_summary.json')['confirmed_value_slots']==28800
    frozen=snapshot(output)
    assert runner.run({},output,worker_runner=lambda *a:pytest.fail('no retry'))['status']=='READ_ONLY'
    assert snapshot(output)==frozen and len(observed)==2


def test_time_dependency_failure_closes_then_posts_without_payload_calls(tmp_path):
    output=tmp_path/'attempt'
    def fail(*args):raise OSError('artificial missing upstream file')
    def payload(output):
        runner.payload_stage(output,backend_factory=lambda _:pytest.fail('dependency failure must not load'))
        return dict(worker_returncode=0,failure=None)
    result=runner.run(config(),output,tmp_path/'scratch',worker_runner=fail,payload_runner=payload)
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    assert result['operations']['payload']['status_counts']=={'NOT_ENTERED':551}
    assert all(row['returned']==1 for row in result['posts'].values())
    assert rec.read(output/'payload/evaluation_summary.json')['status']=='EVIDENCE_INCOMPLETE'
    nodes=rec.read(output/'time_chain/node_results.json')
    assert len(nodes)==1644 and all(row['conditional_correct'] is None for row in nodes)
    assert len(rec.read(output/'payload/blind_terminal.json')['artifacts'])==100


def test_payload_failure_and_supervisor_interrupt_keep_denominators(tmp_path):
    class Fail(ArtificialPayload):
        def load(self):raise OSError('artificial model load failure')
    def failed_payload(output):
        runner.payload_stage(output,backend_factory=lambda _:Fail(output/'payload'))
        return dict(worker_returncode=0,failure=None)
    output=tmp_path/'failed'
    result=runner.run(config(),output,tmp_path/'scratch',worker_runner=metadata_time,payload_runner=failed_payload)
    assert result['status']=='STOPPED_FIXED_ATTEMPT' and 'model load failure' in result['failure']
    assert result['posts']['payload']['returned']==1
    output=tmp_path/'interrupted'
    result=runner.run(config(),output,tmp_path/'scratch2',worker_runner=metadata_time,
        payload_runner=lambda _:dict(worker_returncode=-9,failure='ARTIFICIAL_INTERRUPTION'))
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    assert rec.read(output/'joint_blind_terminal.json')['denominators']['outer_operations']==629
    assert all(row['status']=='NOT_STARTED' for row in result['posts'].values())
    assert not (output/'time_chain/post_truth_access.json').exists()
    before=snapshot(output)
    assert runner.payload_stage(output,backend_factory=lambda _:pytest.fail('no internal reentry'))['status']=='READ_ONLY'
    assert snapshot(output)==before


def test_unconfirmed_time_artifact_is_retained_without_scoring(tmp_path):
    output=tmp_path/'attempt';c=config();rec.reserve(output,c)
    time_output=output/'time_chain'
    rec.write(time_output/'private/attack_truth.json',old_protocol.construction_maps())
    for stage in old_protocol.STAGES:metadata_time(stage,time_output,tmp_path/'scratch')
    events=time_output/'stages/receive/events.jsonl'
    uid=old_protocol.units(c)[0]['id']
    rows=[json.loads(line) for line in events.read_text().splitlines()]
    for row in rows:
        if row['id']==uid+'/relative_compare':row['status']='STARTED'
    events.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    rec.close_blind(output,'ARTIFICIAL_INTERRUPTION')
    runner.post_once(output)
    nodes=rec.read(time_output/'node_results.json')
    assert all(row['conditional_correct'] is None for row in nodes if row['unit']==uid)
    evaluation=rec.read(time_output/'evaluation.json')
    assert evaluation['summaries'][0]['status']=='INTERRUPTED'
    assert evaluation['summaries'][0]['execution']['unconfirmed_artifact']
    assert (time_output/'blind'/uid/'relative.json').is_file()


def test_bootstrap_preparation_failure_and_zero_network_reopen(tmp_path,monkeypatch):
    from runtime.wan import b_joint_time_payload_r02_colab_v1 as bootstrap
    output=tmp_path/'attempt';monkeypatch.setattr(bootstrap,'OUTPUT',output)
    monkeypatch.setattr(bootstrap,'WORK',tmp_path/'work')
    monkeypatch.setattr(bootstrap,'show_results',lambda *args:None)
    def fail(*args):raise ImportError('artificial dependency failure')
    monkeypatch.setattr(bootstrap,'install_dependencies',fail)
    result=bootstrap.run_colab(source_directory=ROOT)
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    assert rec.read(output/'payload/raw_terminal_summary.json')['actual_slots']==28800
    before=snapshot(output)
    monkeypatch.setattr(bootstrap,'source_text',lambda *args:pytest.fail('reopen must not read source/network'))
    assert bootstrap.run_colab(source_directory='/missing')['status']=='READ_ONLY'
    assert snapshot(output)==before


@pytest.mark.parametrize('local_source',[None,'/missing/source'])
def test_delivered_notebook_readonly_before_source_and_builder_sync(tmp_path,monkeypatch,local_source):
    import urllib.request
    from scripts.build_b_joint_time_payload_r02_v1_notebook import build
    expected=ROOT/'notebooks/b_joint_time_payload_r02_v1.ipynb'
    assert build(tmp_path).read_bytes()==expected.read_bytes()
    notebook=json.loads(expected.read_text());code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for text in code:ast.parse(text)
    output=tmp_path/'saved';output.mkdir();rec.write(output/'attempt.json',dict(consumed=True,status='STOPPED_FIXED_ATTEMPT'))
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:pytest.fail('no network on consumed output'))
    namespace=dict(SOURCE_DIRECTORY=local_source,SOURCE_REF='unused')
    exec(code[2].replace(protocol.OUTPUT,str(output)),namespace)
    exec(code[3],namespace)
    assert namespace['RUN']['status']=='READ_ONLY' and namespace['entry'] is None


@pytest.mark.parametrize('interrupted',[False,True])
def test_stage_process_terminates_inherited_child_group(tmp_path,monkeypatch,interrupted):
    import psutil
    child_path=tmp_path/'child.pid'
    child="import os,signal,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(%r).write_text(str(os.getpid()));time.sleep(60)"%str(child_path)
    parent="import subprocess,sys,time;from pathlib import Path;subprocess.Popen([sys.executable,'-c',%r]);\nwhile not Path(%r).exists():time.sleep(.01)\n"%(child,str(child_path))
    if interrupted:parent+='time.sleep(60)\n'
    sleep=runner.time.sleep
    def interrupt_after_child(seconds):
        if interrupted and child_path.exists():raise KeyboardInterrupt('artificial interrupt')
        sleep(seconds)
    monkeypatch.setattr(runner.time,'sleep',interrupt_after_child)
    if interrupted:
        with pytest.raises(KeyboardInterrupt):runner.stage_process([sys.executable,'-c',parent],tmp_path/'log')
    else:
        assert runner.stage_process([sys.executable,'-c',parent],tmp_path/'log')['returncode']==0
    pid=int(child_path.read_text())
    deadline=time.monotonic()+1
    while True:
        try:
            if not psutil.Process(pid).is_running() or psutil.Process(pid).status()==psutil.STATUS_ZOMBIE:break
        except psutil.NoSuchProcess:break
        assert time.monotonic()<deadline,'inherited child survived group termination'
        sleep(.01)


def test_outer_time_sigterm_closes_without_starting_payload_or_posts_v2(tmp_path):
    """Signal the real outer supervisor while its time-stage process tree is alive."""
    import psutil
    child_script=tmp_path/'artificial_stage.py'
    child_script.write_text('''import os,subprocess,sys,time
from pathlib import Path
grandchild="import os,signal,sys,time;from pathlib import Path;signal.signal(signal.SIGTERM,signal.SIG_IGN);Path(sys.argv[1]).write_text(str(os.getpid()));time.sleep(60)"
subprocess.Popen([sys.executable,'-c',grandchild,sys.argv[2]])
Path(sys.argv[1]).write_text(str(os.getpid()))
time.sleep(60)
''')
    supervisor_script=tmp_path/'artificial_supervisor.py'
    supervisor_script.write_text('''import json,sys
from pathlib import Path
from runtime.wan import b_joint_time_payload_r02_v1 as runner
from runtime.wan import b_joint_time_payload_r02_records_v1 as rec
from main.tube_state import b_joint_time_relative_validation_v1 as protocol
output,child,child_pid,grand_pid=map(Path,sys.argv[1:])
def time_stage(stage,time_output,scratch):
    rows=protocol.stage_operations(rec.read(time_output/'protocol.json'),stage)
    folder=time_output/'stages'/stage;folder.mkdir(parents=True)
    rec.write(folder/'plan.json',rows)
    (folder/'events.jsonl').write_text(json.dumps(dict(rows[0],status='STARTED'))+'\\n')
    runner.stage_process([sys.executable,str(child),str(child_pid),str(grand_pid)],folder/'worker.log')
    raise AssertionError('the interrupted artificial stage must not return')
def payload_forbidden(output):
    rec.write(output/'FORBIDDEN_PAYLOAD_START.json',{})
    raise AssertionError('external interruption must not launch payload')
result=runner.run(rec.read(runner.DEFAULT_CONFIG),output,output.parent/'scratch',
                  worker_runner=time_stage,payload_runner=payload_forbidden)
print(json.dumps(result),flush=True)
''')
    output=tmp_path/'attempt';child_pid=tmp_path/'child.pid';grand_pid=tmp_path/'grand.pid'
    environment=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONDONTWRITEBYTECODE='1')
    process=subprocess.Popen([sys.executable,str(supervisor_script),str(output),str(child_script),
        str(child_pid),str(grand_pid)],env=environment,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,start_new_session=True)
    pids=[]
    try:
        deadline=time.monotonic()+15
        while not (child_pid.is_file() and grand_pid.is_file()):
            assert process.poll() is None,process.communicate()[0]
            assert time.monotonic()<deadline,'artificial stage tree did not start'
            time.sleep(.02)
        pids=[int(path.read_text()) for path in (child_pid,grand_pid)]
        os.kill(process.pid,signal.SIGTERM)
        text,_=process.communicate(timeout=25)
        assert process.returncode==0,text
        result=rec.read(output/'result.json')
        assert result['status']=='STOPPED_FIXED_ATTEMPT' and result['denominators']['outer_operations']==629
        assert sum(result['operations']['time']['status_counts'].values())==78
        assert result['operations']['time']['status_counts']['INTERRUPTED_CONSUMED']==1
        assert result['operations']['payload']['status_counts']=={'NOT_ENTERED':551}
        assert rec.read(output/'payload/resources.json')['worker_started']==0
        assert rec.read(output/'external_interruption.json')['signal']==signal.SIGTERM
        assert all(row['status']=='NOT_STARTED' and row['started']==row['returned']==0
                   for row in result['posts'].values())
        for name in ('FORBIDDEN_PAYLOAD_START.json','payload_stage_started.json','payload/worker_started.json',
                     'post_started.json','time_chain/post_truth_access.json','payload/truth_access.json'):
            assert not (output/name).exists(),name
        terminal=rec.read(output/'time_terminal_slots.json')
        assert sum(len(unit['nodes']) for unit in terminal['units'])==1644
        assert sum(len(unit['edges']) for unit in terminal['units'])==1634
        assert all(slot['value'] is None for unit in terminal['units'] for slot in unit['nodes']+unit['edges'])
        assert rec.read(output/'payload/raw_terminal_summary.json')['actual_slots']==28800
        blind=rec.read(output/'payload/blind_terminal.json')
        assert sum(len(item['slots']) for item in blind['artifacts'] if item['kind']=='final')==800
        deadline=time.monotonic()+1
        for pid in pids:
            while psutil.pid_exists(pid) and psutil.Process(pid).status()!=psutil.STATUS_ZOMBIE:
                assert time.monotonic()<deadline,'descendant survived outer SIGTERM cleanup'
                time.sleep(.01)
        before=snapshot(output)
        assert runner.run({},output,worker_runner=lambda *a:pytest.fail('no time restart'),
                          payload_runner=lambda *a:pytest.fail('no payload restart'))['status']=='READ_ONLY'
        assert snapshot(output)==before
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.communicate()
        # Fixture-only cleanup if an assertion failed before production cleanup completed.
        for path in (child_pid,grand_pid):
            if path.is_file():
                try:os.kill(int(path.read_text()),signal.SIGKILL)
                except ProcessLookupError:pass


def test_payload_worker_sigterm_does_not_enter_post_v2(tmp_path):
    class Interrupted(ArtificialPayload):
        def load(self):
            os.kill(os.getpid(),signal.SIGTERM)
            pytest.fail('signal must interrupt load')
    def payload(output):
        runner.payload_stage(output,backend_factory=lambda _:Interrupted(output/'payload'),
                             time_evaluator=lambda *a:pytest.fail('no time post'),
                             payload_evaluator=lambda *a:pytest.fail('no payload post'))
        return dict(worker_returncode=0,failure=None)
    output=tmp_path/'attempt'
    result=runner.run(config(),output,tmp_path/'scratch',worker_runner=metadata_time,payload_runner=payload)
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    assert rec.read(output/'external_interruption.json')['phase']=='payload'
    assert all(row['status']=='NOT_STARTED' and row['started']==0 for row in result['posts'].values())
    assert not (output/'post_started.json').exists()


def test_sigterm_during_first_post_never_starts_second_post_v2(tmp_path):
    output=tmp_path/'attempt';rec.reserve(output,config());rec.close_blind(output,'ARTIFICIAL_METADATA_ONLY')
    def interrupted_post(*args):
        os.kill(os.getpid(),signal.SIGTERM)
        pytest.fail('signal must interrupt post')
    with runner.termination_handler() as interruption:
        posts=runner.post_once(output,time_evaluator=interrupted_post,
            payload_evaluator=lambda *a:pytest.fail('no continuation after external interrupt'),
            interruption=interruption)
    assert posts['time']['status']=='INTERRUPTED_CONSUMED' and posts['time']['returned']==0
    assert posts['payload']['status']=='NOT_STARTED' and posts['payload']['started']==0
    assert rec.read(output/'external_interruption.json')['phase']=='time_post'
