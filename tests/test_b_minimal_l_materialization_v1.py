"""Artificial CPU/stub checks for the GPU delivery; no real VAE/codec/media."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import b_framewise_qim_v1 as p
from experiments.wan_state_clock import b_minimal_l_materialization_v1_records as rec
from experiments.wan_state_clock import b_minimal_l_materialization_v1_experiment as exp
from runtime.wan import b_minimal_l_materialization_v1 as runtime

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.unit
torch.set_num_threads(2)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def test_q0_replica_major_once_and_no_hidden_read(tmp_path,monkeypatch):
    backend=runtime.CaptureBackend.__new__(runtime.CaptureBackend)
    backend.output=tmp_path;backend.active=torch.ones((181,10),dtype=torch.bool);backend.bits=torch.zeros((181,10,63),dtype=torch.int64)
    backend.base=SimpleNamespace(device='cpu',vae=SimpleNamespace(to=lambda device:None,decoder=SimpleNamespace(conv_out=SimpleNamespace(weight=None))))
    counts=[];backend.event=lambda kind,done:counts.append((kind,done))
    backend.read_layer=lambda *a:pytest.fail('no ninth physical layer read')
    monkeypatch.setattr(runtime,'HeadOperator',lambda *a:SimpleNamespace(input_shape=(1,),mask=None))
    q=np.arange(181*5*2*63,dtype=np.float64).reshape(181,5,2,63);seen=[]
    def nearest(q0,bits): seen.append(q0.clone());return q0+1
    monkeypatch.setattr(p,'nearest',nearest)
    monkeypatch.setattr(runtime,'solve',lambda *a,**k:dict(status='NUMERICALLY_UNRESOLVED',coefficients=torch.ones(1,dtype=torch.float64)))
    answer=backend.solve_cached('unused',q,lambda x:None)
    assert answer['finite_candidate_saved'] and len(seen)==1
    for replica in range(2):
        for lane in range(5): np.testing.assert_array_equal(seen[0][:,replica*5+lane],q[:,lane,replica])
    assert counts==[('targets',False),('targets',True)]
    np.testing.assert_array_equal(np.load(tmp_path/'fixed_targets.npy'),seen[0].numpy()+1)


def test_saved_target_membership_matches_original_ties_without_reconstruction():
    from main.tube_state.b_minimal_l_v1 import validate_cached_targets
    q=torch.tensor([-48.,-16.,0.,16.,48.,17.,-17.],dtype=torch.float64)
    for bit in (0,1):
        bits=torch.full_like(q,bit,dtype=torch.int64);target=p.nearest(q,bits)
        validate_cached_targets(q.numpy(),target.numpy(),bits.numpy())
        with pytest.raises(ValueError): validate_cached_targets(q.numpy(),(target+32).numpy(),bits.numpy())


def test_actual_new_B_uses_one_K_and_cached_prediction_in_native_replay(tmp_path,monkeypatch):
    # Reuse the exercised artificial tiny head fixture, with the new B adapter.
    previous=module(ROOT/'tests/test_b_framewise_qim_saved_candidate_v1.py','tiny_saved_fixture')
    monkeypatch.setattr(previous,'Backend',runtime.CandidateBackend)
    previous.test_actual_cached_single_k_and_modified_singleton_tail(tmp_path)


def test_B_check_does_not_reconstruct_targets_and_caches_one_Aa(monkeypatch):
    backend=runtime.CandidateBackend.__new__(runtime.CandidateBackend);events=[]
    backend.event=lambda kind,done:events.append((kind,done));backend.active=torch.ones((1,10),dtype=torch.bool)
    backend.baseline=np.zeros((1,1),dtype=np.float32);backend.targets=torch.zeros((1,10,63),dtype=torch.float64)
    backend.bits=torch.zeros_like(backend.targets,dtype=torch.int64)
    backend.coefficients=torch.ones(1,dtype=torch.float64);calls=[]
    prediction=torch.zeros((1,1),dtype=torch.float64)
    backend.operator=SimpleNamespace(output_shape=(1,1),mask=None,forward=lambda a:(calls.append(1) or prediction))
    monkeypatch.setattr(p,'pack',lambda x:x)
    monkeypatch.setattr(p,'project',lambda x:torch.zeros((1,10,63),dtype=torch.float64))
    monkeypatch.setattr(p,'nearest',lambda *a:pytest.fail('targets constructed only in A'))
    monkeypatch.setattr(runtime.checks,'direct_constraints',lambda *a:dict(status='DIRECT_CONSTRAINTS_WITHIN_TOLERANCE'))
    assert backend.check_candidate()['status']=='DIRECT_CONSTRAINTS_WITHIN_TOLERANCE'
    assert len(calls)==1 and backend.prediction is prediction
    assert events==[('BASE_constraint_P',False),('BASE_constraint_P',True),('Aa_constraint_P',False),('Aa_constraint_P',True)]


class Fake:
    mode='ok'
    def __init__(self,cfg,output,event): self.output=Path(output);self.event=event;self.q0=None;self.codec_times=[]
    def call(self,kind,fn=lambda:None): self.event(kind,False);value=fn();self.event(kind,True);return value
    def capture(self,path):
        self.call('model_load');self.call('D181')
        if self.mode=='interrupt_A': raise KeyboardInterrupt('fake interrupted capture')
        for _ in range(46): self.call('baseline_head')
        for name in rec.cache_paths():
            file=self.output/name;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'artificial cached tensor marker')
        return 'fake baseline'
    def read_layer(self,path,layer):
        q=np.zeros((181,5,2,63),dtype=np.float64)
        if layer=='preclip' and self.output.name=='A': self.q0=q
        return dict(q=q,d0=q,d1=q+16,soft=q-1,mask=np.ones(q.shape,bool))
    def media(self,path,arm):
        if self.mode=='A_codec' and arm=='BASE':
            self.event('codec_encode',False);raise RuntimeError('fake A codec failure')
        for kind in ('codec_encode','codec_probe','codec_decode'): self.call(kind)
        (self.output/(arm+'.mp4')).write_bytes(b'fake media')
        return 'fake h264'
    def solve_cached(self,path,q0,progress):
        assert q0 is self.q0
        self.call('targets');self.call('K');self.call('KT')
        return dict(status='NUMERICALLY_UNRESOLVED',finite_candidate_saved=True,optimality='UNPROVEN')
    def restore_saved(self,path): self.call('model_load')
    def check_candidate(self):
        self.call('BASE_constraint_P')
        if self.mode=='interrupt_B': self.event('K',False);raise KeyboardInterrupt('fake interrupted B')
        self.call('K');self.call('Aa_constraint_P')
        return dict(status='DIRECT_CONSTRAINTS_FAILED' if self.mode=='linear' else 'DIRECT_CONSTRAINTS_WITHIN_TOLERANCE')
    def apply_saved(self):
        for _ in range(46): self.call('head_replay')
        return 'fake changed',dict(status='NUMERICALLY_UNRESOLVED',finite_complete=True,frames=181)
    def close(self):
        if self.mode=='close': raise OSError('fake close failure')


def input_root(tmp_path):
    source=tmp_path/'source';path=source/rec.CONFIG['terminal_relative'];path.parent.mkdir(parents=True);path.write_bytes(b'fake tensor')
    return source,tmp_path/'new'


@pytest.mark.parametrize('mode',['ok','linear','interrupt_A','interrupt_B','close','A_codec'])
def test_independent_reservation_numeric_unresolved_budgets_and_consumed(tmp_path,monkeypatch,mode):
    source,output=input_root(tmp_path)
    class F(Fake): pass
    F.mode=mode
    def create(cfg,folder,event):
        assert (folder/'attempt.json').is_file()
        counts=rec.load(folder/'counts.json')
        assert all(v['started']==0 for v in counts.values())
        return F(cfg,folder,event)
    result=exp.run(source,output,backend_factories=(create,create))
    a=rec.load(output/'A/counts.json')
    assert a['model_load']['started']==1
    if mode in ('interrupt_A','close','A_codec'):
        assert not (output/'B/attempt.json').exists()
        if mode=='close': assert 'close' in result['A']['errors']
        if mode=='A_codec':
            assert result['A']['finite_candidate_saved'] and not result['A']['stage_complete']
            assert a['codec_encode']['started']==1 and a['time_reads']['returned']==3
    else:
        b=rec.load(output/'B/counts.json')
        assert a['time_reads']['returned']==4 and a['targets']['returned']==1
        assert b['targets']['started']==b['solver']['started']==b['KT']['started']==b['D181']['started']==0
        assert b['model_load']['started']==b['K']['started']==1
        if mode in ('linear','interrupt_B'):
            assert b['head_replay']['started']==b['codec_encode']['started']==b['time_reads']['started']==0
        else:
            assert b['time_reads']['returned']==4 and b['head_replay']['returned']==46
            assert sum(x[k]['returned'] for x in (a,b) for k in ('codec_encode','codec_probe','codec_decode'))==6
            assert result['B']['native_status']=='NUMERICALLY_UNRESOLVED'
    before={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    replay=exp.run('/missing',output,backend_factories=(lambda *a:pytest.fail('A retry'),lambda *a:pytest.fail('B retry')))
    assert replay['physical_calls_this_invocation']==0
    assert before=={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_A_completed_B_first_invocation_and_missing_cache(tmp_path):
    source,output=input_root(tmp_path)
    exp.run_stage(source,output,'A',backend_factory=Fake)
    before={str(p):p.read_bytes() for p in (output/'A').rglob('*') if p.is_file()}
    assert rec.needs_work(output)
    result=exp.run(source,output,backend_factories=(lambda *a:pytest.fail('A recapture'),Fake))
    assert result['B']['counts']['model_load']['started']==1
    assert before=={str(p):p.read_bytes() for p in (output/'A').rglob('*') if p.is_file()}
    other=tmp_path/'missing-cache'
    exp.run_stage(source,other,'A',backend_factory=Fake)
    (other/'A/coefficients.pt').unlink()
    assert not rec.needs_work(other)
    assert exp.run(source,other,backend_factories=(None,None))['physical_calls_this_invocation']==0


def test_fixed_K_KT_budgets_and_stage_preflight_outside_attempt(tmp_path):
    source,output=input_root(tmp_path)
    assert rec.preflight(source,output,'A')['status']=='PREFLIGHT_READY' and not output.exists()
    assert rec.A_BUDGETS['K']+rec.B_BUDGETS['K']==4201
    assert rec.A_BUDGETS['KT']+rec.B_BUDGETS['KT']==4200
    folder=tmp_path/'ledger';journal=rec.Journal(folder,'B')
    journal.call('K',lambda:None)
    with pytest.raises(RuntimeError,match='consumed'): journal.call('K',lambda:pytest.fail('second K'))
    with pytest.raises(RuntimeError): journal.call('E41',lambda:pytest.fail('identity forbidden'))


def test_codec_timeout_is_actual_bounded_command_and_gpu_routes_to_it(tmp_path,monkeypatch):
    from runtime.wan import b_framewise_qim_time_recovery_v1 as media
    ticks=iter((0,301));monkeypatch.setattr(media.time,'monotonic',lambda:next(ticks))
    actions=[]
    class Proc:
        def __init__(self,*a,**k): pass
        def terminate(self): actions.append('terminate')
        def communicate(self,**kw):
            actions.append(kw.get('timeout'))
            if kw.get('timeout')==30: raise subprocess.TimeoutExpired('fake',30)
            return b'',b''
        def kill(self): actions.append('kill')
    monkeypatch.setattr(media.subprocess,'Popen',Proc)
    with pytest.raises(media.ResourceExceeded,match='300'): media.command(['fake'],runtime.CodecGuard())
    assert actions==['terminate',30,'kill',None]
    monkeypatch.undo()
    calls=[]
    monkeypatch.setattr(media.Backend,'encode',lambda self,*a:calls.append('encode'))
    monkeypatch.setattr(media.Backend,'probe',lambda self,*a:calls.append('probe'))
    monkeypatch.setattr(media.Backend,'decode',lambda self,*a:(calls.append('decode') or np.zeros((1,1,1,3),np.uint8)))
    path=tmp_path/'fake.npy';np.save(path,np.zeros((1,1,1,3),np.float32))
    backend=runtime.CaptureBackend.__new__(runtime.CaptureBackend);backend.output=tmp_path;backend.event=lambda *a:None
    backend.media(path,'BASE')
    assert calls==['encode','probe','decode'] and len(backend.codec_times)==3


def test_actual_notebook_builder_and_consumed_reopen_without_network(tmp_path,monkeypatch):
    builder=module(ROOT/'scripts/build_b_minimal_l_materialization_v1_notebook.py','builder_minimal')
    notebook=json.loads((ROOT/'notebooks/b_minimal_l_materialization_v1.ipynb').read_text())
    code=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert code==["from google.colab import drive\ndrive.mount('/content/drive')\n",builder.SETTINGS,builder.BOOTSTRAP,builder.RUN,builder.DISPLAY]
    assert all(c['execution_count'] is None and not c['outputs'] for c in notebook['cells'] if c['cell_type']=='code')
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('no consumed download'))
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('no consumed worker'))
    out=tmp_path/'out';rec.dump(out/'A/attempt.json',{});rec.dump(out/'A/result.json',[])
    scope=dict(SOURCE=tmp_path/'source',OUTPUT=out,SOURCE_DIRECTORY=None,SOURCE_REF='unused',MODEL_ID=None)
    exec(builder.BOOTSTRAP,scope);exec(builder.RUN,scope);exec(builder.DISPLAY,scope)
    assert scope['entry'] is None and not (tmp_path/'out-entry-cache').exists()
    entry=module(ROOT/'scripts/b_minimal_l_materialization_v1_entry.py','entry_minimal')
    monkeypatch.setattr(entry.common,'prepare_source',lambda *a,**k:pytest.fail('no consumed setup'))
    session=entry.Session(scope['SOURCE'],out);session.execute();session.display()


def test_new_entry_prepares_only_first_B_after_completed_A(tmp_path,monkeypatch):
    source,out=input_root(tmp_path)
    exp.run_stage(source,out,'A',backend_factory=Fake)
    entry=module(ROOT/'scripts/b_minimal_l_materialization_v1_entry.py','entry_first_B')
    calls=[]
    monkeypatch.setattr(entry.common,'load_file',lambda *a:SimpleNamespace(setup_dependencies=lambda *a:calls.append('setup')))
    session=entry.Session(source,out,source_directory=ROOT)
    monkeypatch.setattr(session,'_worker',lambda root:calls.append(session._command(root)))
    session.execute()
    assert calls[0]=='setup' and 'experiments.wan_state_clock.b_minimal_l_materialization_v1_run' in calls[1]
    assert not (out/'B/attempt.json').exists()  # worker was stubbed, not a physical attempt
    assert (out.with_name(out.name+'-preflight-v1')/'B.json').is_file()


def test_consumed_gpu_cli_stdlib_only(tmp_path):
    out=tmp_path/'out';rec.dump(out/'B/attempt.json',{})
    code="""
import sys
class Deny:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','psutil','scripts'): raise AssertionError(name)
sys.meta_path.insert(0,Deny())
from experiments.wan_state_clock.b_minimal_l_materialization_v1_run import main
main()
"""
    result=subprocess.run([sys.executable,'-B','-c',code,'--source','/missing','--output',str(out)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['physical_calls_this_invocation']==0
