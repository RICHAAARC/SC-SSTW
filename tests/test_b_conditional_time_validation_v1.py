"""Artificial full-package wiring, inherited budgets and package-wide no-resume checks."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import subprocess
import sys
import pytest
from experiments.wan_state_clock import b_conditional_time_validation_v1_records as records
from experiments.wan_state_clock.b_conditional_time_validation_v1_experiment import run

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.quick
STEM='b_conditional_time_validation_v1'


def module(path):
    spec=importlib.util.spec_from_file_location('artificial_'+Path(path).stem,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def fixture(tmp_path):
    cfg=records.load(ROOT/'experiments/wan_state_clock/configs'/(STEM+'.json'))
    cfg.update(source=str(tmp_path/'source'),output=str(tmp_path/'out'))
    terminal=Path(cfg['source'])/cfg['terminal_relative'];terminal.parent.mkdir(parents=True);terminal.write_bytes(b'artificial P0 marker')
    old=module('tests/test_b_minimal_l_materialization_v1.py')
    receiver=module('tests/test_b_conditional_time_output_v1.py')
    return cfg,old,receiver


def test_explicit_c01_protocol_original_A_B_and_exact_43_media_wiring(tmp_path):
    cfg,old,cpu=fixture(tmp_path);observed=[]
    class Fake(old.Fake):
        def __init__(self,protocol,folder,event):
            observed.append(dict(protocol));super().__init__(protocol,folder,event)
        def capture(self,path):
            assert str(path).endswith('private/b2-m3-c01/P0_terminal.pt')
            return super().capture(path)
    output=Path(cfg['output'])
    result=run(cfg,backend_factories=(Fake,Fake),receiver_backend_factory=cpu.FakeBackend)
    assert [p['source_id'] for p in observed]==['b2-m3-c01']*2
    assert all(p['terminal_relative']==cfg['terminal_relative'] for p in observed)
    assert records.material.CONFIG['source_id']=='b2-m3-e02'  # No global mutation; old defaults stay old.
    for phase in ('A','B'):
        folder=output/'materialization'/phase
        for filename in ('attempt','result'):
            d=records.load(folder/(filename+'.json'))['denominators']
            assert d['source_id']=='b2-m3-c01' and d['receiver_codec_calls']==17 and d['chain_codec_calls']==23
        assert records.load(folder/'protocol.json')['source_id']=='b2-m3-c01'
    assert result['materialization_native_status']=='NUMERICALLY_UNRESOLVED'
    assert result['materialization_errors']['B']['native']
    assert result['scientific_pass'] is None
    assert result['phases']['receiver']['status']=='RETURNED'
    counts=records.load(output/'counts.json')
    assert sum(v['started'] for v in counts['receiver'].values())==43
    assert sum(v['started'] for k,v in counts['receiver'].items() if k.startswith('codec_'))==17
    assert sum(counts[p][k]['started'] for p in ('A','B') for k in ('codec_encode','codec_probe','codec_decode'))==6
    assert result['budgets']['materialization_A']['K']==4200 and result['budgets']['materialization_B']['K']==1
    assert result['budgets']['materialization_A']['KT']==4200
    before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    again=run(dict(output=str(output)),backend_factories=(lambda *a:pytest.fail('A rerun'),lambda *a:pytest.fail('B rerun')))
    assert again['status']=='READ_ONLY_EXISTING_PACKAGE' and again['physical_calls_this_invocation']==0
    assert before=={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mode,expected_B,expected_receiver',[
    ('interrupt_A',False,False),('A_codec',False,False),('linear',True,False),('interrupt_B',True,False)])
def test_failed_materialization_never_retries_or_grants_later_phase_on_Run_all(tmp_path,mode,expected_B,expected_receiver):
    cfg,old,cpu=fixture(tmp_path)
    class Fake(old.Fake):pass
    Fake.mode=mode
    result=run(cfg,backend_factories=(Fake,Fake),receiver_backend_factory=cpu.FakeBackend)
    output=Path(cfg['output'])
    assert (output/'materialization/B/attempt.json').exists()==expected_B
    assert (output/'receiver/attempt.json').exists()==expected_receiver
    unexecuted=records.load(output/'receiver_not_executed.json')
    assert len(unexecuted['rows'])==43 and all(row['status']=='DEPENDENCY_UNRESOLVED' for row in unexecuted['rows'].values())
    assert unexecuted['denominators']['received_frames']==822 and unexecuted['denominators']['relations']==817
    assert records.load(output/'counts.json')['receiver']['status']=='NOT_EXECUTED'
    if mode.startswith('interrupt'):
        assert result['status']=='PACKAGE_INTERRUPTED'
        assert result['phases']['A' if mode=='interrupt_A' else 'B']['status']=='INTERRUPTED'
    assert run(dict(output=str(output)))['physical_calls_this_invocation']==0
    assert (output/'receiver/attempt.json').exists()==expected_receiver


def test_completed_A_does_not_allow_new_B_on_reopened_package(tmp_path):
    cfg,old,cpu=fixture(tmp_path)
    from experiments.wan_state_clock.b_minimal_l_materialization_v1_experiment import run_stage
    def phase(source,output,stage,**kwargs):
        if stage=='B':raise KeyboardInterrupt('artificial interruption after completed A before B reservation')
        return run_stage(source,output,stage,**kwargs)
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_backend_factory=cpu.FakeBackend,phase_runner=phase)
    output=Path(cfg['output'])
    assert records.load(output/'materialization/A/result.json')['stage_complete'] is True
    assert records.material.ready_B(output/'materialization')=={}
    assert not (output/'materialization/B/attempt.json').exists()
    assert result['status']=='PACKAGE_INTERRUPTED'
    again=run(cfg,phase_runner=lambda *a,**k:pytest.fail('reopened package advanced B'))
    assert again['status']=='READ_ONLY_EXISTING_PACKAGE' and not (output/'materialization/B/attempt.json').exists()


def test_missing_A_cache_blocks_B_without_refill(tmp_path):
    cfg,old,cpu=fixture(tmp_path)
    from experiments.wan_state_clock.b_minimal_l_materialization_v1_experiment import run_stage
    def phase(source,output,stage,**kwargs):
        result=run_stage(source,output,stage,**kwargs)
        if stage=='A':(Path(output)/'A/coefficients.pt').unlink()
        return result
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_backend_factory=cpu.FakeBackend,phase_runner=phase)
    assert 'B_dependency' in result['errors']
    assert not (Path(cfg['output'])/'materialization/B/attempt.json').exists()
    assert run(cfg)['physical_calls_this_invocation']==0


@pytest.mark.parametrize('exit_kind',['preflight','interrupt'])
def test_receiver_exit_before_reservation_keeps_43_unexecuted_rows_and_reason(tmp_path,exit_kind):
    cfg,old,_=fixture(tmp_path)
    def receiver(*args,**kwargs):
        if exit_kind=='interrupt':raise KeyboardInterrupt('artificial receiver interruption before reservation')
        return dict(status='PREFLIGHT_UNRESOLVED',error='artificial unreadable media before reservation',physical_calls_this_invocation=0)
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_runner=receiver)
    output=Path(cfg['output'])
    assert not (output/'receiver/attempt.json').exists()
    saved=records.load(output/'receiver_not_executed.json')
    assert len(saved['rows'])==43 and all(row['status']=='DEPENDENCY_UNRESOLVED' for row in saved['rows'].values())
    assert saved['denominators']['received_frames']==822 and saved['denominators']['relations']==817
    assert 'artificial' in json.dumps(saved['exit_reason'])
    counts=records.load(output/'counts.json')['receiver']
    assert counts['status']=='NOT_EXECUTED' and all(v['started']==v['returned']==0 for v in counts['operations'].values())
    assert result['phases']['receiver']['status']==('INTERRUPTED' if exit_kind=='interrupt' else 'RETURNED')
    assert run(cfg,receiver_runner=lambda *a,**k:pytest.fail('receiver retried'))['physical_calls_this_invocation']==0


def test_reserved_receiver_interruption_propagates_without_erasing_actual_rows(tmp_path):
    cfg,old,cpu=fixture(tmp_path)
    class Interrupted(cpu.FakeBackend):
        def codebook(self):raise KeyboardInterrupt('artificial interruption after receiver reservation')
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_backend_factory=Interrupted)
    output=Path(cfg['output'])
    assert (output/'receiver/attempt.json').exists()
    assert result['status']=='PACKAGE_INTERRUPTED' and result['phases']['receiver']['status']=='INTERRUPTED'
    assert result['stage_results']['receiver']['status']=='INTERRUPTED'
    assert not (output/'receiver_not_executed.json').exists()
    actual=records.load(output/'receiver/counts.json')
    assert actual['codebook']['started']==1 and actual['codebook']['returned']==0
    assert records.load(output/'counts.json')['receiver']==actual
    assert records.load(output/'receiver/state.json')['codebook']['status']=='INTERRUPTED'
    assert run(cfg,receiver_backend_factory=lambda:pytest.fail('receiver retried'))['physical_calls_this_invocation']==0


def test_unfinished_package_receipt_is_read_only_without_science(tmp_path):
    output=tmp_path/'out';records.dump(output/'attempt.json',{'source_id':'b2-m3-c01'})
    records.dump(output/'state.json',{'A':{'status':'RETURNED'},'B':{'status':'STARTED'},'receiver':{'status':'NOT_STARTED'}})
    code="""
import sys
class Deny:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','psutil','runtime','scripts'):raise AssertionError(name)
sys.meta_path.insert(0,Deny())
from experiments.wan_state_clock.b_conditional_time_validation_v1_run import main
main()
"""
    p=subprocess.run([sys.executable,'-B','-c',code,'--output',str(output),'--config','/missing'],cwd=ROOT,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    value=json.loads(p.stdout)
    assert value['status']=='READ_ONLY_EXISTING_PACKAGE' and value['physical_calls_this_invocation']==0
    assert value['phases']['B']['status']=='UNRESOLVED_RECEIPT'


def test_P0_preflight_specific_missing_source_and_no_package_reservation(tmp_path):
    cfg,_,_=fixture(tmp_path);(Path(cfg['source'])/cfg['terminal_relative']).unlink()
    result=run(cfg)
    assert result['status']=='PREFLIGHT_UNRESOLVED' and 'b2-m3-c01' in result['error']
    assert not Path(cfg['output']).exists()
    cfg['terminal_relative']='../escape.pt'
    assert 'escapes' in run(cfg)['error']


def test_full_entry_cached_default_c01_and_setup_before_worker_only(tmp_path,monkeypatch):
    cfg,_,_=fixture(tmp_path)
    entry=module('scripts/'+STEM+'_entry.py');common=entry.cpu._common();seen=[]
    monkeypatch.setattr(entry.cpu,'_common',lambda:common)
    monkeypatch.setattr(common,'load_file',lambda *a:SimpleNamespace(setup_dependencies=lambda *args:seen.append('setup')))
    def worker(session,root):
        launch=records.load(session.config_path);seen.append(launch)
        assert launch['source_id']=='b2-m3-c01' and launch['terminal_relative']==cfg['terminal_relative']
        assert not session.output.exists()
        records.dump(session.output/'attempt.json',{'source_id':launch['source_id']})
        records.dump(session.output/'result.json',{'status':'ARTIFICIAL_FULL_PACKAGE'})
        assert STEM+'_run' in session._command(root)[4]
    monkeypatch.setattr(common.Session,'_worker',worker)
    result=entry.launch(cfg['source'],cfg['output'],source_directory=ROOT,scratch=tmp_path/'scratch')
    assert result['status']=='ARTIFICIAL_FULL_PACKAGE' and seen[0]=='setup'
    monkeypatch.setattr(common,'prepare_source',lambda *a,**k:pytest.fail('consumed preparation'))
    assert entry.launch('/missing',cfg['output'],source_directory='/missing')['physical_calls_this_invocation']==0


def test_full_setup_reuses_material_and_receiver_dependencies_only(tmp_path,monkeypatch):
    setup=module('scripts/'+STEM+'_setup.py');calls=[]
    material=SimpleNamespace(setup_dependencies=lambda *a:calls.append('material_setup'),command=lambda *a:calls.append(a[0]))
    cpu=SimpleNamespace(setup_dependencies=lambda *a:calls.append(('cpu_setup',a[-1])))
    monkeypatch.setattr(setup,'load',lambda path,name:material if 'material' in name else cpu)
    setup.setup_dependencies(ROOT,tmp_path)
    assert calls[:2]==['material_setup',('cpu_setup','media')]
    assert 'b_conditional_time_validation_v1_run' in calls[2][-1] and 'b_minimal_l_materialization_v1' in calls[2][-1]


def test_full_notebook_builder_default_and_remote_bootstrap_without_real_execution(tmp_path,monkeypatch):
    import nbformat,urllib.request,shutil
    builder=module('scripts/build_'+STEM+'_notebook.py');built=builder.build(tmp_path)
    notebook=nbformat.read(built['notebook'],as_version=4);nbformat.validate(notebook)
    assert notebook==nbformat.read(ROOT/'notebooks'/(STEM+'.ipynb'),as_version=4)
    assert notebook.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    config=records.load(ROOT/'experiments/wan_state_clock/configs'/(STEM+'.json'))
    assert config['source_id']=='b2-m3-c01' and config['receiver_core_calls']==43 and config['chain_codec_calls']==23
    urls=[]
    def retrieve(url,path):
        urls.append(url)
        if str(path).endswith(STEM+'_entry.py'):
            Path(path).write_text("def launch(source,output,source_directory,source_ref,model_id):\n return {'status':'ARTIFICIAL_FULL_LAUNCH','source':source,'source_directory':source_directory}\n")
        else:shutil.copyfile(ROOT/url.split('/fixture-ref/')[1],path)
    monkeypatch.setattr(urllib.request,'urlretrieve',retrieve)
    scope=dict(SOURCE='/artificial/identity',OUTPUT=str(tmp_path/'out'),SOURCE_DIRECTORY=None,SOURCE_REF='fixture-ref',MODEL_ID=None)
    exec(builder.RUN,scope)
    assert scope['result']['status']=='ARTIFICIAL_FULL_LAUNCH' and scope['result']['source_directory'] is None and len(urls)==10
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('consumed notebook download'))
    output=Path(scope['OUTPUT']);records.dump(output/'attempt.json',{})
    exec(builder.RUN,scope)
    assert scope['result']['status']=='READ_ONLY_EXISTING_PACKAGE' and scope['result']['physical_calls_this_invocation']==0
