"""Notebook delivery tests: stdlib recovery and fake process only, no experiment cells."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import types
import nbformat
import pytest
from scripts import build_b_framewise_qim_v1_notebook as builder
from runtime.wan import b_framewise_qim_v1 as runner

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def notebook_codes():
    nb = nbformat.read(ROOT/'notebooks/b_framewise_qim_v1.ipynb',as_version=4)
    return [c.source for c in nb.cells if c.cell_type=='code']


def execute_settings(tmp_path, monkeypatch, *, inputs=True):
    source=tmp_path/'input';output=tmp_path/'output';scratch=tmp_path/'scratch'
    if inputs:
        terminal=source/'private/b2-m3-e01/P0_terminal.pt';terminal.parent.mkdir(parents=True,exist_ok=True)
        terminal.write_text('fake input; not deserialized')
        runner.dump(source/'protocol.json',dict(config=dict(writer=dict(bits=[0]*32))))
    # Only paths are substituted in the actual generated settings cell.
    tree=ast.parse(notebook_codes()[1])
    replacements={'SOURCE':str(source),'OUTPUT':str(output),'SCRATCH':str(scratch),'SOURCE_DIRECTORY':str(ROOT)}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name in replacements:
                node.value=ast.Constant(replacements[name]) if name=='SOURCE_DIRECTORY' else ast.Call(ast.Name('Path',ast.Load()),[ast.Constant(replacements[name])],[])
    env={};exec(compile(ast.fix_missing_locations(tree),'generated-settings','exec'),env)
    monkeypatch.setattr(env['urllib'].request,'urlretrieve',lambda *a:pytest.fail('network forbidden in fixture'))
    return env


def no_subprocess(*args,**kwargs):pytest.fail('existing/missing input path must not enter subprocess')


def test_structure_and_shared_recovery_exact(tmp_path):
    nb=nbformat.read(ROOT/'notebooks/b_framewise_qim_v1.ipynb',as_version=4);nbformat.validate(nb)
    codes=notebook_codes()
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert len(codes)==5
    for c in nb.cells:
        if c.cell_type=='code':
            ast.parse(c.source);assert c.execution_count is None and c.outputs==[]
    assert codes[1].startswith(builder.recovery_source(ROOT))
    assert 'import torch' not in builder.recovery_source(ROOT) and 'import numpy' not in builder.recovery_source(ROOT)
    assert "B-Terminal-Three-Group-V1/fixed-four-new-v1" in codes[1]
    assert "B-Framewise-QIM-V1/fixed-e01-181-v1" in codes[1]
    assert "SOURCE_REF = 'dev/local-joint-state-payload-v1'" in codes[1]
    # Deterministic builder output, no notebook execution.
    stage=tmp_path/'stage'
    for rel in builder.source_files(ROOT):
        dest=stage/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    built=builder.build(stage)
    assert Path(built['notebook']).read_bytes()==(ROOT/'notebooks/b_framewise_qim_v1.ipynb').read_bytes()
    assert built['source_zip_created'] is False


def test_first_launch_reaches_real_runner_once_then_reopen_is_read_only(tmp_path,monkeypatch):
    env=execute_settings(tmp_path,monkeypatch)
    assert env['CAN_START'] and not env['OUTPUT'].exists()
    exec(notebook_codes()[2],env)
    assert env['LOG_DIRECTORY'].parent==env['OUTPUT'].parent
    assert not env['OUTPUT'].exists()
    monkeypatch.setattr(shutil,'which',lambda name:'/fake/'+name)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0))
    launches=[]
    class FakeBackend:
        def __init__(self,cfg,output,event):self.event=event
        def capture(self,path):
            self.event('D181',False)
            raise RuntimeError('artificial failure; no model/latent read')
        def close(self):pass
    class FakeWorker:
        def __init__(self,args,**kwargs):
            assert args[args.index('-m')+1]=='experiments.wan_state_clock.b_framewise_qim_v1_run'
            assert Path(args[args.index('--source')+1])==env['SOURCE']
            assert not runner.has_existing_output(env['OUTPUT'])
            launches.append(args)
            result=runner.run(runner.load(env['CONFIG']),env['SOURCE'],env['OUTPUT'],FakeBackend)
            assert result['status']=='STOPPED_FIXED_ATTEMPT'
            self.returncode=0
        def poll(self):return self.returncode
    monkeypatch.setattr(subprocess,'Popen',FakeWorker)
    exec(notebook_codes()[3],env);exec(notebook_codes()[4],env)
    assert len(launches)==1
    assert (env['LOG_DIRECTORY']/'setup.log').exists() and not (env['OUTPUT']/'setup.log').exists()
    assert (env['LOG_DIRECTORY']/'runner.log').exists() and not (env['OUTPUT']/'runner.log').exists()
    monkeypatch.setattr(subprocess,'Popen',no_subprocess);monkeypatch.setattr(subprocess,'run',no_subprocess)
    env=execute_settings(tmp_path,monkeypatch)
    assert not env['CAN_START'] and env['EXISTING']
    for code in notebook_codes()[2:]:exec(code,env)
    assert env['summary']['physical_calls_this_invocation']==0
    assert env['summary']['calls']['D181']['started']==1 and env['summary']['calls']['D181']['returned']==0


@pytest.mark.parametrize('must_kill',[False,True])
def test_notebook_interrupt_stops_worker_and_reopen_preserves_unknown(tmp_path,monkeypatch,must_kill):
    env=execute_settings(tmp_path,monkeypatch);exec(notebook_codes()[2],env)
    monkeypatch.setattr(shutil,'which',lambda name:'/fake/'+name)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0))
    events=[]
    class InterruptedWorker:
        returncode=None
        def __init__(self,args,**kwargs):
            assert not runner.has_existing_output(env['OUTPUT'])
            runner.dump(env['OUTPUT']/'attempt.json',dict(status='STARTED'))
            journal=runner.Journal(env['OUTPUT']);journal.event('D181',False)
            journal.rows['BASE/preclip']['status']='STARTED';runner.dump(env['OUTPUT']/'state.json',journal.rows)
        def poll(self):return None
        def terminate(self):events.append('terminate')
        def wait(self,timeout=None):
            if must_kill and timeout is not None:raise subprocess.TimeoutExpired('fake',timeout)
            events.append('wait');return -15
        def kill(self):events.append('kill')
    monkeypatch.setattr(subprocess,'Popen',InterruptedWorker)
    monkeypatch.setattr(env['time'],'sleep',lambda n:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):exec(notebook_codes()[3],env)
    assert events==(['terminate','kill','wait'] if must_kill else ['terminate','wait'])
    monkeypatch.setattr(subprocess,'Popen',no_subprocess);monkeypatch.setattr(subprocess,'run',no_subprocess)
    env=execute_settings(tmp_path,monkeypatch)
    for code in notebook_codes()[2:]:exec(code,env)
    assert env['summary']['status']=='UNRESOLVED_RECORDS'
    assert env['summary']['state_counts']['UNRESOLVED_INTERRUPTED']==1
    assert env['summary']['denominators']['time_read_slots']==912240


def test_missing_input_does_not_start_setup_or_consume_attempt(tmp_path,monkeypatch,capsys):
    env=execute_settings(tmp_path,monkeypatch,inputs=False)
    monkeypatch.setattr(subprocess,'Popen',no_subprocess);monkeypatch.setattr(subprocess,'run',no_subprocess)
    for code in notebook_codes()[2:]:exec(code,env)
    assert not env['CAN_START'] and not env['OUTPUT'].exists()
    text=capsys.readouterr().out
    assert 'INPUT_MISSING' in text and 'private/b2-m3-e01/P0_terminal.pt' in text and 'protocol.json' in text


def test_corrupt_and_missing_saved_records_are_unknown_and_fixed(tmp_path):
    runner.dump(tmp_path/'state.json',{'BASE/preclip':dict(status='RETURNED',slots=0,kind='fake')})
    runner.dump(tmp_path/'result.json',[])
    snapshot=runner.recover(tmp_path);summary=runner.recovery_summary(snapshot)
    assert snapshot['rows']['BASE/preclip']['slots']==114030
    assert snapshot['rows']['BASE/preclip']['kind']=='time_read'
    assert 'result' in snapshot['read_errors']
    assert summary['status']=='UNRESOLVED_RECORDS'
    assert all(v['started'] is None and v['returned'] is None for v in summary['calls'].values())
    assert all(r['bit_errors'] is None for r in summary['time_layers'])
    assert all(r['observed_bits'] is None for r in summary['identity'])
    assert summary['identity_comparison'] is None
    states=runner.plan()
    for row in states.values():row['status']='RETURNED'
    runner.dump(tmp_path/'state.json',states)
    runner.dump(tmp_path/'counts.json',{k:dict(started=0,returned=0) for k in runner.BUDGETS})
    runner.dump(tmp_path/'result.json',dict(status='COMPLETED_FIXED_RECORDS',evaluation_status='POST_RECORDED'))
    runner.dump(tmp_path/'evaluation.json',dict(status='POST_UNRESOLVED',error='fake corruption'))
    assert runner.recovery_summary(runner.recover(tmp_path))['status']=='UNRESOLVED_POST_RECORD'


def test_embedded_recovery_equals_runner_and_has_no_scientific_imports(tmp_path):
    runner.dump(tmp_path/'attempt.json',dict(status='STARTED'))
    env={};exec(builder.recovery_source(ROOT),env)
    assert env['recover'](tmp_path)==runner.recover(tmp_path)
    assert env['recovery_summary'](env['recover'](tmp_path))==runner.recovery_summary(runner.recover(tmp_path))
    imported=[]
    for node in ast.walk(ast.parse(builder.recovery_source(ROOT))):
        if isinstance(node,ast.Import):imported.extend(n.name for n in node.names)
        elif isinstance(node,ast.ImportFrom):imported.append(node.module)
    assert set(imported)=={'pathlib','json'}


def completed_saved_snapshot():
    states=runner.plan()
    for row in states.values():row['status']='RETURNED'
    time_rows=[dict(key=f'{arm}/{layer}',status='RETURNED',fixed_slots=114030,active_slots=112770,
        observed_active_slots=112770,bit_errors=0,erased_active_slots=0,wrong_origin_candidate_count=0)
        for arm in runner.ARMS for layer in runner.LAYERS]
    identity=[]
    for arm in runner.ARMS:
        for start in (0,40):
            for phase in range(4):
                for region in ('R0','R1'):
                    for local_row in range(1,11):
                        for bit in range(8):
                            identity.append(dict(arm=arm,start=start,phase=phase,region=region,local_row=local_row,bit=bit,
                                status='AVAILABLE',value=0,common_complete_cell_eligible=phase==0,
                                group=0 if phase==0 else None,correct=True if phase==0 else None))
    return dict(rows=states,saved=dict(result=dict(status='COMPLETED_FIXED_RECORDS',evaluation_status='POST_RECORDED'),
        counts={k:dict(started=n,returned=n) for k,n in runner.BUDGETS.items()},
        evaluation=dict(time=time_rows,identity=identity,identity_comparison=dict(fixed_paired_slots=1280,
            common_complete_observed_slots=320,base_common_errors=0,qim_common_errors=0,
            base_common_ber=0.,qim_common_ber=0.,added_support=0,lost_support=0))))


@pytest.mark.parametrize('damage',['identity_missing_fields','identity_duplicate','negative_time_errors','malformed_time_status','comparison_missing'])
def test_nested_corrupt_post_records_do_not_fabricate_completion_or_zero_errors(damage):
    snapshot=completed_saved_snapshot()
    assert runner.recovery_summary(snapshot)['status']=='COMPLETED_FIXED_RECORDS'
    evaluation=snapshot['saved']['evaluation']
    if damage=='identity_missing_fields':evaluation['identity']=[dict(arm=a) for a in runner.ARMS for _ in range(1280)]
    elif damage=='identity_duplicate':evaluation['identity'][1]=dict(evaluation['identity'][0])
    elif damage=='negative_time_errors':evaluation['time'][0]['bit_errors']=-7
    elif damage=='malformed_time_status':evaluation['time'][0]['status']=[]
    else:evaluation['identity_comparison']={}
    env={};exec(builder.recovery_source(ROOT),env)
    summary=env['recovery_summary'](snapshot)
    assert summary==runner.recovery_summary(snapshot)
    assert summary['status']=='UNRESOLVED_POST_RECORD'
    if damage.startswith('identity'):
        assert summary['identity'][0]['full_cell_errors'] is None
        assert summary['identity'][0]['observed_bits'] is None
    elif damage.startswith('comparison'):assert summary['identity_comparison'] is None
    else:assert summary['time_layers'][0]['bit_errors'] is None
    assert summary['denominators']['identity_local_bit_slots']==2560


def test_source_closure_imports_from_editable_no_git_copy(tmp_path):
    files=builder.source_files(ROOT)
    for rel in files:
        dest=tmp_path/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/rel,dest)
    probe="import sys,json;sys.path.insert(0,sys.argv[1]);from runtime.wan.b_framewise_qim_v1_backend import Backend;from runtime.wan.b_framewise_qim_v1 import load;b=Backend(load(sys.argv[1]+'/experiments/wan_state_clock/configs/b_framewise_qim_v1.json'),sys.argv[1],lambda *a:None);assert b.loaded is False and b.base.vae is None;print('inert closure import')"
    result=subprocess.run([sys.executable,'-I','-c',probe,str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'inert closure import' in result.stdout
    help_result=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_framewise_qim_v1_run','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert help_result.returncode==0 and '--source' in help_result.stdout
