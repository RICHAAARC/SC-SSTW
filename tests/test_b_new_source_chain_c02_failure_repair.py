"""Capacity/receipt diagnostics only; artificial CPU tensors and backends, no real data."""
from pathlib import Path
from types import SimpleNamespace
import errno
import importlib.util
import json
import subprocess
import sys
import pytest

from experiments.wan_state_clock import b_new_source_chain_confirmation_v1_records as rec
from experiments.wan_state_clock import b_new_source_chain_confirmation_v1_evaluation as evaluation
from experiments.wan_state_clock.b_new_source_chain_confirmation_v1_experiment import run
from runtime.wan import b_framewise_qim_v1_cache as capacity

ROOT=Path(__file__).resolve().parents[1]
STEM='b_new_source_chain_confirmation_v1'
pytestmark=pytest.mark.quick


def module(path):
    spec=importlib.util.spec_from_file_location('repair_'+Path(path).stem,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


@pytest.mark.parametrize('channels',[16,320])
def test_unchanged_capacity_formula_at_exact_required_boundary(tmp_path,monkeypatch,channels):
    shape_bytes=4*181*channels*320*512+4*181*320*512*3*5
    required=max(32*1024**3,shape_bytes)
    monkeypatch.setattr(capacity.shutil,'disk_usage',lambda path:SimpleNamespace(free=required))
    receipt=capacity.check_disk_cache_capacity(tmp_path,channels)
    assert receipt['status']=='CAPACITY_CHECK_PASSED'
    assert receipt['shape_estimate_bytes']==shape_bytes and receipt['required_bytes']==required
    assert receipt['actual_free_bytes']==required and receipt['minimum_bytes']==32*1024**3
    assert receipt['check_path']==str(tmp_path.resolve()) and receipt['channels']==channels
    assert receipt['feature_blocks']==46 and receipt['dtype']=='FP32'
    assert json.loads((tmp_path/'disk_cache_capacity.json').read_text())==receipt
    assert not (tmp_path/'features').exists()


@pytest.mark.parametrize('mode',['insufficient','write_error','receipt_error_then_cache_write'])
def test_actual_capture_capacity_branch_root_summary_and_consumed_recovery(tmp_path,monkeypatch,mode,capsys):
    import torch
    from runtime.wan import b_framewise_qim_v1_backend as backend
    from runtime.wan.b_minimal_l_materialization_v1 import CaptureBackend
    old=module('tests/test_b_new_source_chain_confirmation_v1.py');cfg,_,_,_=old.fixture(tmp_path)
    output=Path(cfg['output']);shape=(1,16,46,40,64)
    monkeypatch.setattr(backend.torch,'load',lambda *a,**k:torch.zeros(shape,dtype=torch.float32))
    free=1 if mode=='insufficient' else 128*1024**3
    monkeypatch.setattr(capacity.shutil,'disk_usage',lambda path:SimpleNamespace(free=free))
    def forbidden(*a,**k):pytest.fail('unentered real operation/post called')
    attempted=[]
    def write_error(*a,**k):
        attempted.append(a[0]);raise OSError(errno.EIO,'artificial actual cache write error')
    if mode=='receipt_error_then_cache_write':
        original=Path.write_text
        def optional_receipt_failure(path,*args,**kwargs):
            if path.name=='disk_cache_capacity.json':raise PermissionError(errno.EACCES,'artificial diagnostic receipt denied')
            return original(path,*args,**kwargs)
        monkeypatch.setattr(Path,'write_text',optional_receipt_failure)
    monkeypatch.setattr(backend.np.lib.format,'open_memmap',write_error if mode!='insufficient' else forbidden)
    monkeypatch.setattr(backend,'decode_normalized_latent',forbidden)
    class FakeCapture(CaptureBackend):
        def __init__(self,cfg,folder,event):
            self.output=Path(folder);self.event=event;self.loaded=False;self.chunks=[]
            head=torch.nn.Conv3d(16,3,3);head._padding=(1,1,1,1,2,0)
            class FakeVAE:
                decoder=SimpleNamespace(conv_out=head)
                def eval(self):return self
                def requires_grad_(self,value):return self
            self.base=SimpleNamespace(vae=FakeVAE(),device='cpu',load_vae=lambda:None,release_vae=lambda:None)
    result=run(cfg,backend_factories=(FakeCapture,forbidden),receiver_backend_factory=forbidden,
               payload_backend_factory=forbidden,time_evaluator=forbidden,payload_evaluator=forbidden)
    diag=result['execution_diagnostics'];root=diag['earliest_stage_error']
    assert root['stage']=='A' and root['result_status']=='STOPPED_FIXED_ATTEMPT'
    assert ('disk cache capacity insufficient' if mode=='insufficient' else 'artificial actual cache write error') in root['error']
    assert 'MemoryError' not in root['error']
    if mode=='receipt_error_then_cache_write':
        assert len(attempted)==1 and (output/'time_chain/materialization/A/features').is_dir()
        assert not (output/'time_chain/materialization/A/disk_cache_capacity.json').exists()
        assert 'Non-blocking disk cache capacity receipt write failure' in capsys.readouterr().err
    else:
        receipt=rec.load(output/'time_chain/materialization/A/disk_cache_capacity.json')
        assert receipt['actual_free_bytes']==free
        assert receipt['status']==('INSUFFICIENT_DISK_CACHE_CAPACITY' if mode=='insufficient' else 'CAPACITY_CHECK_PASSED')
    calls=diag['operation_receipts']['A']
    assert calls['started_nonzero']=={'model_load':{'started':1,'returned':1}}
    assert {'D181','baseline_head','solver','K','KT','codec_encode','time_reads'}<=set(calls['unstarted'])
    assert diag['receiver_execution']=='NOT_EXECUTED'
    assert diag['time_execution_counts']['nodes']['NOT_EXECUTED']==822
    assert diag['time_execution_counts']['relations']['NOT_EXECUTED']==817
    post=rec.load(output/'evaluation.json')
    assert post['time']['counts']['nodes']['NOT_EXECUTED']==822
    assert post['time']['counts']['relations']['NOT_EXECUTED']==817
    assert all(result['phases'][p]['started']==0 for p in ('B','time','payload','time_post','payload_post'))
    assert not (output/'time_chain/receiver/blind_terminal.json').exists()
    assert not (output/'payload/blind_terminal.json').exists()
    assert not (output/'private/post_fixture.json').exists()
    before={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    again=run(dict(output=str(output)),backend_factories=(forbidden,forbidden),time_evaluator=forbidden,payload_evaluator=forbidden)
    assert again['physical_calls_this_invocation']==again['post_calls_this_invocation']==0
    assert again['execution_diagnostics']['earliest_stage_error']['stage']=='A'
    assert before=={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('enough',[True,False])
def test_diagnostic_receipt_write_failure_never_replaces_capacity_decision(tmp_path,monkeypatch,capsys,enough):
    required=32*1024**3
    monkeypatch.setattr(capacity.shutil,'disk_usage',lambda path:SimpleNamespace(free=required if enough else 1))
    def denied(*args,**kwargs):raise PermissionError(errno.EACCES,'artificial diagnostic receipt denied')
    monkeypatch.setattr(Path,'write_text',denied)
    if enough:
        receipt=capacity.check_disk_cache_capacity(tmp_path,16)
        assert receipt['status']=='CAPACITY_CHECK_PASSED' and receipt['receipt_write_error']
        assert 'Non-blocking disk cache capacity receipt write failure' in capsys.readouterr().err
    else:
        with pytest.raises(OSError) as caught:capacity.check_disk_cache_capacity(tmp_path,16)
        assert caught.value.errno==errno.ENOSPC
        assert 'disk cache capacity insufficient' in str(caught.value) and 'capacity_receipt_write_error=' in str(caught.value)


@pytest.mark.parametrize('kind',['plan','missing','corrupt','started'])
def test_NOT_STARTED_is_distinct_from_missing_or_called_receipts(kind):
    rows=rec.chain.receiver.plan('media') if kind=='plan' else {}
    if kind=='corrupt':rows={v+'/submit':None for v in rec.payload.VIEWS}
    if kind=='started':rows={v+'/submit':{'status':'STARTED'} for v in rec.payload.VIEWS}
    report=evaluation.time_report(None,rows)
    expected='NOT_EXECUTED' if kind=='plan' else 'UNKNOWN'
    assert report['counts']['nodes'][expected]==822
    assert report['counts']['relations'][expected]==817
    assert sum(report['counts']['nodes'].values())==822
    # Missing whole-stage receipts cannot inherit NOT_STARTED from the reservation plan.
    diag=rec.execution_diagnostics({'phases':{}},{})
    assert diag['receiver_execution']=='UNKNOWN' and diag['time_execution_counts'] is None


def legacy_receipts(output):
    stages={k:dict(status='DEPENDENCY_UNRESOLVED',started=0,returned=0) for k in ('B','time','payload')}
    stages['A']=dict(status='RETURNED',started=1,returned=1,result_status='STOPPED_FIXED_ATTEMPT')
    counts={k:dict(planned=v,started=int(k=='model_load'),returned=int(k=='model_load')) for k,v in rec.chain.material.A_BUDGETS.items()}
    child=dict(status='STOPPED_FIXED_ATTEMPT',errors={'stage':'MemoryError: insufficient space for FP32 native feature cache'},counts=counts)
    result=dict(status='PACKAGE_FIXED_ATTEMPT_TERMINATED',phases=stages,evaluation_status='EVIDENCE_INCOMPLETE',
        stage_results=dict(time_chain=dict(stage_results={'A':child})),errors={'execution':'FileNotFoundError: receiver/blind_terminal.json'})
    rec.dump(output/'attempt.json',{'artificial':True,'recovery_layout':rec.recovery_layout()})
    rec.dump(output/'result.json',result);rec.dump(output/'counts.json',{'A':counts})
    rec.dump(output/'evaluation.json',{'status':'EVIDENCE_INCOMPLETE','time':{'counts':{'nodes':{'UNKNOWN':822},'relations':{'UNKNOWN':817}}}})
    return result


def test_legacy_failure_visible_in_CLI_and_sourceless_notebook_without_post_or_rewrite(tmp_path,monkeypatch,capsys):
    import urllib.request
    legacy_receipts(tmp_path);before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('consumed download'))
    builder=module('scripts/build_'+STEM+'_notebook.py')
    scope=dict(SOURCE='/missing',OUTPUT=str(tmp_path),SOURCE_DIRECTORY=None,SOURCE_REF='unused',MODEL_ID=None)
    exec(builder.RUN,scope);summary=scope['summary']
    assert summary['earliest_stage_error']['stage']=='A' and 'MemoryError' in summary['earliest_stage_error']['error']
    assert summary['earliest_stage_error']['failure_kind']=='DISK_CACHE_CAPACITY_CHECK'
    assert summary['operation_receipts']['A']['started_nonzero']=={'model_load':{'started':1,'returned':1}}
    assert summary['time_counts']['nodes']['NOT_EXECUTED']==822 and summary['time_counts']['relations']['NOT_EXECUTED']==817
    assert summary['physical_calls_this_invocation']==summary['post_calls_this_invocation']==0
    assert summary['disk_cache_capacity'] is None and 'cannot be reconstructed' in summary['missing_capacity_measurements']
    assert len(capsys.readouterr().out)<5000
    code="""
import sys
class Deny:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','runtime','main'):raise AssertionError(name)
sys.meta_path.insert(0,Deny())
from experiments.wan_state_clock.b_new_source_chain_confirmation_v1_run import main
main()
"""
    called=subprocess.run([sys.executable,'-B','-c',code,'--output',str(tmp_path),'--config','/missing'],cwd=ROOT,text=True,capture_output=True)
    assert called.returncode==0,called.stderr
    cli=json.loads(called.stdout)
    for key in ('earliest_stage_error','receiver_execution','time_counts','operation_receipts'):
        assert cli[key]==summary[key]
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_started_receiver_missing_state_remains_unknown_and_never_retries(tmp_path):
    old=module('tests/test_b_new_source_chain_confirmation_v1.py');cfg,material,_,_=old.fixture(tmp_path)
    def missing(*args,**kwargs):return dict(status='FIXED_ATTEMPT_TERMINATED',errors={'receipt':'artificial missing receiver records'})
    def forbidden(*a,**k):pytest.fail('unentered payload called')
    result=run(cfg,backend_factories=(material.Fake,material.Fake),receiver_runner=missing,payload_backend_factory=forbidden)
    assert result['phases']['time']['started']==1
    report=rec.load(Path(cfg['output'])/'evaluation.json')
    assert report['time']['counts']['nodes']['UNKNOWN']==822 and report['time']['counts']['relations']['UNKNOWN']==817
    assert run(cfg)['physical_calls_this_invocation']==0
