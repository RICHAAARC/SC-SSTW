"""Saved-candidate engineering: artificial headers/JSON, tiny heads, fake physical calls."""
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from main.tube_state import b_framewise_qim_v1 as p
from main.tube_state import b_framewise_qim_saved_candidate_v1 as checks
from runtime.wan.b_framewise_qim_saved_candidate_v1_backend import Backend
from runtime.wan.b_framewise_qim_v1_operator import HeadOperator, replay_head, feature_delta
from experiments.wan_state_clock import b_framewise_qim_saved_candidate_v1_records as records
from experiments.wan_state_clock import b_framewise_qim_saved_candidate_v1_experiment as experiment
from experiments.wan_state_clock import b_framewise_qim_saved_candidate_v1_evaluation as evaluation
from experiments.wan_state_clock import b_framewise_qim_saved_candidate_v1_report as report

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def identity_cache(status='AVAILABLE', value=0):
    return dict(regions={r:dict(local_cells=[dict(bits=[dict(status=status,value=value) for _ in range(8)]) for _ in range(10)]) for r in ('R0','R1')})


def assets(tmp_path):
    saved=tmp_path/'saved'; output=tmp_path/'new'; truth=tmp_path/'truth'
    cfg=records.load(ROOT/'experiments/wan_state_clock/configs/b_framewise_qim_v1.json')
    records.dump(saved/'protocol.json',cfg)
    records.dump(saved/'result.json',dict(status='STOPPED_FIXED_ATTEMPT'))
    records.dump(saved/'solver.json',dict(status='NUMERICALLY_UNRESOLVED',reason='iteration cap'))
    for relative in ['coefficients.pt']+[f'features/{s:03}.pt' for s in records.STARTS]:
        path=saved/relative;path.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(path,'w') as z:z.writestr('archive/data.pkl',b'fake header only, never torch-loaded')
    for relative,shape,dtype in [('fixed_targets.npy',(181,10,63),'<f8'),('BASE_preclip.npy',(181,320,512,3),'<f4')]:
        with (saved/relative).open('wb') as stream:
            np.lib.format.write_array_header_1_0(stream,dict(descr=dtype,fortran_order=False,shape=shape))
    state={}
    for start in (0,40):
        for phase in range(4):
            key=f'BASE/identity/{start}/{phase}';state[key]=dict(status='RETURNED')
            records.dump(saved/'identity'/(key+'.json'),identity_cache())
    records.dump(saved/'state.json',state)
    records.dump(truth/'protocol.json',dict(config=dict(writer=dict(bits=[0]*32))))
    return saved,truth,output


def snapshot(path): return {str(p.relative_to(path)):p.read_bytes() for p in path.rglob('*') if p.is_file()}


class FakeBackend:
    mode='ok'
    def __init__(self,cfg,output,event):self.output=Path(output);self.event=event
    def call(self,kind,fn=lambda:None):
        self.event(kind,False);value=fn();self.event(kind,True);return value
    def restore_saved(self,saved):self.call('model_load')
    def check_candidate(self):
        self.call('BASE_constraint_P')
        if self.mode=='interrupt':
            self.event('K',False);raise KeyboardInterrupt()
        self.call('K');self.call('Aa_constraint_P')
        return dict(status='DIRECT_CONSTRAINTS_FAILED' if self.mode=='linear_fail' else 'DIRECT_CONSTRAINTS_WITHIN_TOLERANCE')
    def apply_saved(self):
        for _ in range(46):self.call('head_replay')
        if self.mode=='nonfinite':raise FloatingPointError('artificial nonfinite native')
        return 'fake-preclip',dict(status='NUMERICALLY_UNRESOLVED' if self.mode=='native_fail' else 'NATIVE_RESIDUAL_WITHIN_TOLERANCE',finite_complete=True,frames=181)
    def read_layer(self,path,layer):
        q=np.zeros((181,5,2,63));return dict(q=q,d0=q,d1=q+16,soft=q-1,mask=np.ones(q.shape,dtype=bool))
    def media(self,path,arm):
        assert arm=='QIM'
        if self.mode=='codec_fail':
            self.event('codec_encode',False);raise RuntimeError('artificial codec failure')
        for name in ('codec_encode','codec_probe','codec_decode'):self.call(name)
        return 'fake-uint8-media'
    def identity(self,media,start,phase,key):return self.call('E41',identity_cache)
    def close(self):pass
    def capture(self,*a):raise AssertionError('no baseline capture')
    def solve(self,*a):raise AssertionError('no solver')


def test_preflight_missing_can_repair_and_reservation_is_one_shot(tmp_path):
    saved,truth,output=assets(tmp_path)
    missing=saved/'features/177.pt';content=missing.read_bytes();missing.unlink()
    def forbidden(*args):raise AssertionError('no backend before successful preflight')
    first=experiment.run(saved,truth,output,backend_factory=forbidden)
    assert first['status']=='PREFLIGHT_UNRESOLVED' and not output.exists()
    assert (output.with_name(output.name+'-preflight-v1')/'preflight.json').is_file()
    assert all(row['started']==0 for row in first['counts'].values())
    missing.write_bytes(content);before=snapshot(saved)
    result=experiment.run(saved,truth,output,backend_factory=FakeBackend)
    assert result['status']=='COMPLETED_FIXED_RECORDS'
    assert all(row['returned']==row['planned'] for row in result['counts'].values())
    assert result['denominators']['time_read_slots']==456120
    assert result['denominators']['complete_cell_eligible_slots']==320
    again=experiment.run(saved,truth,output,backend_factory=forbidden)
    assert again['physical_calls_this_invocation']==0 and snapshot(saved)==before
    assert len(list((output/'raw').rglob('*.npz')))==4
    assert len(list((output/'identity').rglob('*.json')))==8


@pytest.mark.parametrize('mode,time_calls,head_calls',[('linear_fail',0,0),('native_fail',4,46),('nonfinite',0,46),('codec_fail',3,46),('interrupt',0,0)])
def test_fixed_failure_semantics_and_no_reentry(tmp_path,mode,time_calls,head_calls):
    saved,truth,output=assets(tmp_path)
    class Fake(FakeBackend):pass
    Fake.mode=mode
    result=experiment.run(saved,truth,output,backend_factory=Fake)
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    assert result['counts']['time_reads']['returned']==time_calls
    assert result['counts']['head_replay']['returned']==head_calls
    for key in ('D181','baseline_head','solver','KT','DiT','training','BASE_new_physical'):
        assert result['counts'][key]['started']==result['counts'][key]['returned']==0
    if mode=='interrupt':assert result['counts']['K']['started']==1 and result['counts']['K']['returned']==0
    rows=records.load(output/'state.json')
    if mode=='codec_fail':
        assert all(rows['QIM/'+layer]['status']=='RETURNED' for layer in records.LAYERS[:3])
        assert rows['QIM/h264']['status']=='NOT_STARTED'
    if mode=='native_fail':assert rows['QIM/h264']['status']=='RETURNED'
    def forbidden(*args):raise AssertionError('spent attempt cannot start again')
    assert experiment.run(saved,truth,output,backend_factory=forbidden)['physical_calls_this_invocation']==0


def test_truth_missing_does_not_block_reads_and_access_is_last(tmp_path,monkeypatch):
    saved,truth,output=assets(tmp_path);(truth/'protocol.json').unlink()
    original=evaluation.load
    def checked(path):
        if Path(path)==truth/'protocol.json':
            rows=records.load(output/'state.json')
            assert all(row['status']=='RETURNED' for row in rows.values())
            assert len(list((output/'raw').rglob('*.npz')))==4
            assert len(list((output/'identity').rglob('*.json')))==8
        return original(path)
    monkeypatch.setattr(evaluation,'load',checked)
    result=experiment.run(saved,truth,output,backend_factory=FakeBackend)
    assert result['counts']['E41']['returned']==8
    post=records.load(output/'evaluation.json')
    assert post['truth_status']=='UNRESOLVED'
    assert post['support']['eligible_320']['common_ber']['QIM'] is None


def test_preflight_conflict_and_old_directory_protection(tmp_path):
    saved,truth,output=assets(tmp_path)
    with pytest.raises(ValueError):experiment.run(saved,truth,saved/'new',backend_factory=FakeBackend)
    records.dump(output/'unrelated.json',dict(value=1));before=snapshot(output)
    result=experiment.run(saved,truth,output,backend_factory=FakeBackend)
    assert result['status']=='PREFLIGHT_UNRESOLVED' and snapshot(output)==before
    assert 'new_output' in result['errors']


def test_support_distinguishes_unmeasured_erased_and_eligible(tmp_path):
    saved,truth,output=assets(tmp_path)
    state={}
    for start in (0,40):
        for phase in range(4):
            key=f'QIM/identity/{start}/{phase}';state[key]=dict(status='NOT_STARTED')
    base=evaluation.identity_rows(saved,'BASE',records.load(saved/'state.json'))
    unseen=evaluation.identity_rows(output,'QIM',state)
    result=evaluation.support_comparison(base,unseen,[0]*32)
    assert result['local_1280']['unmeasured_pairs']==1280 and result['local_1280']['lost_support'] is None
    assert result['eligible_320']['unmeasured_pairs']==320 and result['eligible_320']['common_ber']['QIM'] is None
    for key in state:
        state[key]['status']='RETURNED'
        records.dump(output/'identity'/(key+'.json'),identity_cache('ERASED',None))
    erased=evaluation.identity_rows(output,'QIM',state)
    result=evaluation.support_comparison(base,erased,[0]*32)
    assert result['local_1280']['lost_support']==1280 and result['eligible_320']['lost_support']==320
    assert result['eligible_320']['common_ber']['QIM'] is None
    result=evaluation.support_comparison(erased,base,[0]*32)
    assert result['local_1280']['added_support']==1280
    state['QIM/identity/0/1']['status']='FAILED'
    mixed=evaluation.identity_rows(output,'QIM',state)
    result=evaluation.support_comparison(base,mixed,[0]*32)
    assert result['local_1280']['lost_support']==1120 and result['local_1280']['unmeasured_pairs']==160
    assert result['eligible_320']['lost_support']==320


class TinyHead(torch.nn.Conv3d):
    def __init__(self):
        super().__init__(1,3,3,padding=0);self._padding=(1,1,1,1,2,0)
    def forward(self,value,tail=None):
        padding=list(self._padding)
        if tail is not None:value=torch.cat((tail,value),dim=2);padding[4]-=tail.shape[2]
        return super().forward(F.pad(value,padding))


def test_actual_cached_single_k_and_modified_singleton_tail(tmp_path):
    torch.manual_seed(21)
    head=TinyHead().eval().requires_grad_(False)
    features=torch.randn((1,1,5,320,512))*.01
    baseline=(head(features)[0].permute(1,2,3,0)/2+.5).detach()
    backend=Backend.__new__(Backend);backend.output=tmp_path
    events=[];backend.event=lambda kind,done:events.append((kind,done))
    backend.active=p.schedule(5)[0]>=0;backend.bits=p.schedule(5)[1]
    backend.operator=HeadOperator(head.weight,backend.active,backend.event)
    backend.coefficients=torch.randn(backend.operator.input_shape,dtype=torch.float64)*1e-4
    backend.baseline=baseline.numpy();backend.targets=p.nearest(255*p.project(p.pack(baseline.double())),backend.bits)
    class VAE:
        decoder=types.SimpleNamespace(conv_out=head)
        def to(self,device):return self
    backend.base=types.SimpleNamespace(vae=VAE(),device='cpu')
    backend._features=lambda:iter((features[:,:,:1],features[:,:,1:]))
    linear=backend.check_candidate();prediction=backend.prediction
    path,native=backend.apply_saved()
    expected=(head(features+feature_delta(backend.coefficients,backend.active,dtype=torch.float32,device='cpu'))[0].permute(1,2,3,0)/2+.5).detach().numpy()
    assert np.allclose(np.load(path),expected,atol=2e-7,rtol=0)
    assert backend.prediction is prediction and backend.operator.calls==dict(K=1,KT=0)
    assert events.count(('BASE_constraint_P',False))==events.count(('Aa_constraint_P',False))==1
    assert events.count(('head_replay',True))==2 and native['finite_complete']
    altered=backend.targets.clone();altered[0,0,0]+=32
    q0=255*p.project(p.pack(baseline.double()))
    with pytest.raises(ValueError):checks.validate_targets(q0,altered,backend.bits,backend.active)
    assert torch.equal(backend.targets,p.nearest(q0,backend.bits))


def test_all_46_native_chunks_keep_two_modified_frames(monkeypatch):
    import runtime.wan.b_framewise_qim_v1_operator as operator
    head=TinyHead().eval().requires_grad_(False)
    features=torch.arange(181,dtype=torch.float32).reshape(1,1,181,1,1)/181
    coefficients=torch.arange(181,dtype=torch.float64).reshape(181,1)/100
    def injection(a,active,*,dtype,device):return a[:,0].to(dtype=dtype,device=device).reshape(1,1,len(a),1,1)
    monkeypatch.setattr(operator,'feature_delta',injection)
    chunks=[features[:,:,start:start+(1 if start==0 else 4)] for start in records.STARTS]
    events=[]
    actual=torch.cat([value for _,value in replay_head(head,chunks,coefficients,torch.ones((181,1),dtype=torch.bool),lambda *event:events.append(event))])
    expected=head(features+injection(coefficients,None,dtype=torch.float32,device='cpu'))[0].permute(1,2,3,0)/2+.5
    assert torch.allclose(actual,expected,atol=1e-6,rtol=1e-6)
    assert len(chunks)==46 and events.count(('head_replay',True))==46


def test_h264_uint8_time_reader_divides_once(monkeypatch):
    import runtime.wan.b_framewise_qim_v1_backend as original
    class Pixels:
        dtype=np.dtype('uint8');shape=(181,320,512,3)
        def __getitem__(self,index):return np.full((len(range(*index.indices(181))),320,512,3),255,dtype=np.uint8)
    monkeypatch.setattr(original.np,'load',lambda *a,**kw:Pixels())
    ranges=[]
    def read(value):
        ranges.append((float(value.min()),float(value.max())))
        q=torch.zeros((len(value),5,2,63),dtype=torch.float64)
        return dict(q=q,d0=q,d1=q,soft=q,mask=torch.ones_like(q,dtype=torch.bool))
    monkeypatch.setattr(p,'read_rgb',read)
    result=Backend.__new__(Backend).read_layer('fake-h264','h264')
    assert result['q'].shape==(181,5,2,63) and set(ranges)=={(1.,1.)}
    Pixels.dtype=np.dtype('float32')
    with pytest.raises(ValueError):Backend.__new__(Backend).read_layer('fake-h264','h264')


def test_direct_actual_constraints_and_nearest_tie_are_not_optimization():
    active=p.schedule(1)[0]>=0
    prediction=torch.zeros((1,10,3,58,74),dtype=torch.float64)
    mask=p.support(active)
    q0=torch.zeros((1,10,63),dtype=torch.float64);projected=torch.zeros_like(q0)
    accepted=checks.direct_constraints(q0,projected,q0,prediction,mask,active)
    assert accepted['status']=='DIRECT_CONSTRAINTS_WITHIN_TOLERANCE' and accepted['optimality']=='UNPROVEN'
    projected[0,0,0]=.002/255
    assert checks.direct_constraints(q0,projected,q0,prediction,mask,active)['status']=='DIRECT_CONSTRAINTS_FAILED'
    projected.zero_();prediction[0,5,0,57,0]=1e-4
    assert checks.direct_constraints(q0,projected,q0,prediction,mask,active)['budget']['outside']==pytest.approx(1e-4)
    tied=torch.full_like(q0,16);bits=torch.zeros_like(q0);targets=torch.zeros_like(q0)
    checks.validate_targets(tied,targets,bits,active)
    with pytest.raises(ValueError):checks.validate_targets(tied,targets+32,bits,active)


def test_support_returned_unsupported_is_measured_but_failed_is_not(tmp_path):
    saved,truth,output=assets(tmp_path)
    state={}
    for start in (0,40):
        for phase in range(4):
            key=f'QIM/identity/{start}/{phase}';state[key]=dict(status='RETURNED')
            records.dump(output/'identity'/(key+'.json'),identity_cache('UNSUPPORTED',None))
    base=evaluation.identity_rows(saved,'BASE',records.load(saved/'state.json'))
    qim=evaluation.identity_rows(output,'QIM',state)
    summary=evaluation.support_comparison(base,qim,[0]*32)
    assert summary['eligible_320']['lost_support']==320
    state['QIM/identity/0/0']['status']='FAILED'
    summary=evaluation.support_comparison(base,evaluation.identity_rows(output,'QIM',state),[0]*32)
    assert summary['eligible_320']['lost_support']==160 and summary['eligible_320']['unmeasured_pairs']==160


def test_corrupt_post_report_keeps_unknown(tmp_path):
    saved,truth,output=assets(tmp_path);experiment.run(saved,truth,output,backend_factory=FakeBackend)
    snap=records.recover(output)
    assert report.recovery_summary(snap)['support']['local_1280']['comparable_measured_slots']==1280
    damaged=copy.deepcopy(snap);damaged['saved']['evaluation']['identity']['QIM'][1]=dict(damaged['saved']['evaluation']['identity']['QIM'][0])
    assert report.recovery_summary(damaged)['support'] is None
    damaged=copy.deepcopy(snap);damaged['saved']['evaluation']['time'][0]['bit_errors']=-1
    assert report.recovery_summary(damaged)['time'][0]['bit_errors'] is None
    assert report.recovery_summary(damaged)['status']=='UNRESOLVED_POST_RECORD'
    damaged=copy.deepcopy(snap);damaged['saved']['evaluation']['support']['local_1280']['lost_support']=-1
    assert report.recovery_summary(damaged)['support'] is None
    damaged=copy.deepcopy(snap);damaged['saved']['evaluation']['support']['eligible_320']['common_errors']={}
    assert report.recovery_summary(damaged)['support'] is None


def notebook_codes():
    import nbformat
    notebook=nbformat.read(ROOT/'notebooks/b_framewise_qim_saved_candidate_v1.ipynb',as_version=4)
    return [cell.source for cell in notebook.cells if cell.cell_type=='code']


def test_notebook_builder_structure_and_cli_without_outer_layers(tmp_path):
    import nbformat
    from scripts import build_b_framewise_qim_saved_candidate_v1_notebook as builder
    codes=notebook_codes()
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for code in codes:
        assert not any(isinstance(n,(ast.FunctionDef,ast.ClassDef)) for n in ast.walk(ast.parse(code)))
    result=builder.build(tmp_path/'built')
    actual=ROOT/'notebooks/b_framewise_qim_saved_candidate_v1.ipynb'
    assert Path(result['notebook']).read_bytes()==actual.read_bytes()
    nb=nbformat.read(actual,as_version=4)
    assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
    import shutil
    isolated=tmp_path/'source'
    for relative in ('experiments/__init__.py','experiments/wan_state_clock/__init__.py',
                     'experiments/wan_state_clock/b_framewise_qim_v1_records.py',
                     'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_records.py',
                     'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_experiment.py',
                     'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_run.py'):
        dest=isolated/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,dest)
    records.dump(tmp_path/'used/attempt.json',dict(status='STARTED'))
    probe=r'''
import runpy,sys
sys.path.insert(0,sys.argv[1])
class NoScience:
    def find_spec(self,name,*args):
        if name.split('.')[0] in ('torch','numpy','diffusers','transformers'):raise AssertionError(name)
sys.meta_path.insert(0,NoScience())
sys.argv=['qim','--saved-attempt','/missing-old','--truth-source','/missing-truth','--output',sys.argv[2]]
runpy.run_module('experiments.wan_state_clock.b_framewise_qim_saved_candidate_v1_run',run_name='__main__')
'''
    run=subprocess.run([sys.executable,'-I','-B','-c',probe,str(isolated),str(tmp_path/'used')],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)['physical_calls_this_invocation']==0
    help_run=subprocess.run([sys.executable,'-B','-m','experiments.wan_state_clock.b_framewise_qim_saved_candidate_v1_run','--help'],cwd=isolated,capture_output=True,text=True)
    assert help_run.returncode==0 and '--saved-attempt' in help_run.stdout


def test_notebook_fake_launch_and_missing_preflight_without_setup(tmp_path,monkeypatch):
    import shutil
    saved,truth,output=assets(tmp_path)
    env={};exec(notebook_codes()[1],env)
    env.update(SAVED_ATTEMPT=saved,TRUTH_SOURCE=truth,NEW_OUTPUT=output,SOURCE_DIRECTORY=ROOT)
    monkeypatch.setattr(shutil,'which',lambda name:'/fake/'+name)
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:types.SimpleNamespace(returncode=0))
    calls=[]
    class Worker:
        def __init__(self,args,**kwargs):
            assert args[args.index('-m')+1]=='experiments.wan_state_clock.b_framewise_qim_saved_candidate_v1_run'
            assert not output.exists() or not any(output.iterdir())
            assert (output.with_name(output.name+'-preflight-v1')/'preflight.json').is_file()
            calls.append(args);experiment.run(saved,truth,output,backend_factory=FakeBackend);self.returncode=0
        def poll(self):return 0
    monkeypatch.setattr(subprocess,'Popen',Worker)
    for code in notebook_codes()[2:]:exec(code,env)
    assert len(calls)==1
    assert (output.with_name(output.name+'-logs')/'runner.log').exists()
    assert (output.with_name(output.name+'-entry-cache')/'scripts/b_framewise_qim_saved_candidate_v1_entry.py').exists()
    def forbidden(*a,**kw):raise AssertionError('no setup or worker for missing preflight')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    new_output=tmp_path/'missing-new'
    session=env['entry'].Session(tmp_path/'missing-old',truth,new_output,source_directory=ROOT)
    session.execute()
    assert not new_output.exists()
    assert records.load(new_output.with_name(new_output.name+'-preflight-v1')/'preflight.json')['status']=='PREFLIGHT_UNRESOLVED'


@pytest.mark.parametrize('cache_mode',['complete','missing','report_missing'])
def test_notebook_consumed_reopen_no_network_science_or_writes(tmp_path,cache_mode):
    from scripts import b_framewise_qim_saved_candidate_v1_entry as entry
    output=tmp_path/'used';records.dump(output/'attempt.json',dict(status='STARTED'))
    cache=output.with_name(output.name+'-entry-cache')
    if cache_mode!='missing':
        entry.common.cache_entry(ROOT,cache,entry.CACHE_FILES)
        if cache_mode=='report_missing':(cache/'experiments/wan_state_clock/b_framewise_qim_saved_candidate_v1_report.py').unlink()
    before=snapshot(tmp_path)
    probe=r'''
import json,pathlib,sys,subprocess,urllib.request,socket
class NoScience:
    def find_spec(self,name,*args):
        if name.split('.')[0] in ('torch','numpy','diffusers','transformers'):raise AssertionError(name)
sys.meta_path.insert(0,NoScience())
def forbidden(*a,**kw):raise AssertionError('network/process forbidden')
urllib.request.urlretrieve=forbidden;urllib.request.urlopen=forbidden;socket.socket=forbidden
subprocess.run=forbidden;subprocess.Popen=forbidden
notebook=json.loads(pathlib.Path(sys.argv[1]).read_text())
codes=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
env=dict(Path=pathlib.Path,SAVED_ATTEMPT=pathlib.Path('/missing-old'),TRUTH_SOURCE=pathlib.Path('/missing-truth'),NEW_OUTPUT=pathlib.Path(sys.argv[2]),MODEL_ID=None,SOURCE_DIRECTORY=None,SOURCE_REF='unused')
for code in codes[2:]:exec(compile(code,'saved-notebook-fixture','exec'),env)
print('READ_ONLY_NO_SCIENCE_NO_NETWORK_OK')
'''
    result=subprocess.run([sys.executable,'-I','-c',probe,str(ROOT/'notebooks/b_framewise_qim_saved_candidate_v1.ipynb'),str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'READ_ONLY_NO_SCIENCE_NO_NETWORK_OK' in result.stdout
    assert before==snapshot(tmp_path)


@pytest.mark.parametrize('suffix',['-preflight-v1','-logs','-entry-cache'])
@pytest.mark.parametrize('protected_input',['saved','truth'])
def test_derived_write_directories_cannot_alias_read_only_inputs(tmp_path,monkeypatch,suffix,protected_input):
    from scripts import b_framewise_qim_saved_candidate_v1_entry as entry
    output=tmp_path/'run'
    protected=output.with_name(output.name+suffix)
    records.dump(protected/'old-marker.json',dict(old=True))
    saved=protected if protected_input=='saved' else tmp_path/'saved'
    truth=protected if protected_input=='truth' else tmp_path/'truth'
    before=snapshot(protected)
    with pytest.raises(ValueError):records.preflight(saved,output,truth)
    def forbidden(*a,**kw):raise AssertionError('no source/download/setup before path protection')
    monkeypatch.setattr(entry.common,'prepare_source',forbidden)
    session=entry.Session(saved,truth,output,source_directory=ROOT)
    session.execute()
    assert 'outside read-only inputs' in session.preparation_error
    env={};exec(notebook_codes()[1],env)
    env.update(SAVED_ATTEMPT=saved,TRUTH_SOURCE=truth,NEW_OUTPUT=output,SOURCE_DIRECTORY=None)
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlretrieve',forbidden)
    for code in notebook_codes()[2:]:exec(code,env)
    assert env['entry'] is None and snapshot(protected)==before and not output.exists()


def test_output_regular_file_is_read_only_conflict(tmp_path,monkeypatch,capsys):
    from scripts import b_framewise_qim_saved_candidate_v1_entry as entry
    output=tmp_path/'output';output.write_text('unrelated file')
    saved,truth=tmp_path/'saved',tmp_path/'truth'
    def forbidden(*a,**kw):raise AssertionError('output conflict must not prepare source')
    monkeypatch.setattr(entry.common,'prepare_source',forbidden)
    session=entry.Session(saved,truth,output,source_directory=ROOT)
    session.execute();session.display()
    env={};exec(notebook_codes()[1],env)
    env.update(SAVED_ATTEMPT=saved,TRUTH_SOURCE=truth,NEW_OUTPUT=output,SOURCE_DIRECTORY=None)
    for code in notebook_codes()[2:]:exec(code,env)
    assert env['conflict'] is True and env['entry'] is None
    assert 'OUTPUT_CONFLICT' in capsys.readouterr().out
    assert output.read_text()=='unrelated file'


def test_consumed_sibling_alias_recovers_before_new_write_checks(tmp_path,monkeypatch):
    output=tmp_path/'run';saved=tmp_path/'run-preflight-v1';truth=tmp_path/'truth'
    records.dump(output/'attempt.json',dict(status='STARTED'))
    records.dump(saved/'old-marker.json',dict(old=True))
    before=snapshot(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('consumed recovery cannot preflight or start backend')
    monkeypatch.setattr(experiment,'preflight',forbidden)
    recovered=experiment.run(saved,truth,output,backend_factory=forbidden)
    assert recovered['status']=='READ_ONLY_EXISTING_ATTEMPT' and recovered['physical_calls_this_invocation']==0
    assert snapshot(tmp_path)==before
