"""Single-row attempt and notebook checks; synthetic stubs only, no model/FFT/media."""
import ast
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest
from experiments.paper_results_v1 import single_read_v1 as m
from experiments.paper_results_v1.single_read_v1_evaluation import evaluate_saved
from experiments.paper_results_v1 import build_single_read_v1_colab as builder

pytestmark = pytest.mark.unit


class RGBStub:
    def __getitem__(self, indices):
        assert indices == [0,0,1,2,3,4,5,6,7,9,27,28,29,30,31,32,33]
        return '17 mapped synthetic frame tokens'


class Stub:
    def __init__(self, fail=None): self.calls=[]; self.fail=fail
    def call(self, name):
        self.calls.append(name)
        if name==self.fail: raise RuntimeError('injected '+name)
    def load(self): self.call('load'); return object()
    def encode(self, model, rgb):
        self.call('encode'); assert rgb=='17 mapped synthetic frame tokens'
        return np.zeros(m.NORMALIZED_SHAPE, dtype=np.float32)
    def read(self, z, public):
        self.call('read')
        assert not any(k in public for k in ['bits','true_path','truth','expected'])
        assert 'experiments.paper_results_v1.distinct_confirmation_p2' not in sys.modules
        return dict(plan=public['plan'], truth_inputs=False, decoded_bits=[1]*16+[None]*16,
                    decoded_bit_count=16, net_payload_bits=32, status='READ',
                    signed_votes=np.ones((4,4,240),dtype=np.int8),
                    zero_coefficient_mask=np.zeros((4,4,240),dtype=np.int8))
    def close(self, model): self.calls.append('close')


@pytest.fixture
def inputs(monkeypatch):
    monkeypatch.setattr(m,'choose_input',lambda:dict(path='synthetic-only',container='npy'))
    monkeypatch.setattr(m,'read_rgb',lambda chosen:RGBStub())


def test_completed_attempt_and_evaluation_retain_denominator(tmp_path, inputs):
    backend=Stub();state=m.run_blind(tmp_path,backend=backend)
    assert state['status']=='BLIND_COMPLETE'
    assert [state['stages'][s]['attempts'] for s in m.STAGES]==[1,1,1,1]
    assert backend.calls==['load','encode','read','close']
    assert (tmp_path/'read_result.json').exists() and not (tmp_path/'summary.json').exists()
    summary=evaluate_saved(tmp_path)
    assert summary['row']['planned_bits']==32
    assert summary['row']['scored_present_bits']==summary['row']['structural_missing_bits']==16
    assert summary['row']['absolute_path_error']==1
    assert summary['row']['wrong_present_bits']>0  # Never predeclare 16/16 correct.
    assert summary['RAW_optional_overlay']['complete_present_BER']==summary['row']['wrong_present_bits']/640
    again=Stub();assert 'READ_SAVED_ONLY' in m.run_blind(tmp_path,backend=again)['action']
    assert again.calls==[]


@pytest.mark.parametrize('failure', ['input','load','encode','read'])
def test_failed_stages_consumed_and_no_later_retry(tmp_path, inputs, monkeypatch, failure):
    backend=Stub(failure)
    if failure=='input':
        def bad(chosen): raise ValueError('invalid chosen container')
        monkeypatch.setattr(m,'read_rgb',bad)
    state=m.run_blind(tmp_path,backend=backend)
    assert state['status']=='FAILED' and state['stages'][failure]['attempts']==1
    later=Stub();m.run_blind(tmp_path,backend=later);assert later.calls==[]
    row=evaluate_saved(tmp_path)['row']
    assert row['missing_present_bits']==16 and row['structural_missing_bits']==16
    assert row['complete_present_BER'] is None


def test_interruption_after_encoded_artifact_never_starts_unstarted_read(tmp_path, inputs, monkeypatch):
    original=m.save_normalized
    def interrupted(path,value):
        original(path,value);raise KeyboardInterrupt('after persistence')
    monkeypatch.setattr(m,'save_normalized',interrupted)
    backend=Stub();state=m.run_blind(tmp_path,backend=backend)
    assert state['status']=='INTERRUPTED' and (tmp_path/'normalized.npy').exists()
    assert state['stages']['read']['attempts']==0
    later=Stub();m.run_blind(tmp_path,backend=later);assert later.calls==[]


def test_selected_invalid_npy_does_not_open_post(tmp_path, monkeypatch):
    root=tmp_path/'p2';received=root/'data/distinct_confirm_20261017_01/views/SHORT17_HOLD_MEDIA/received.npy'
    post=root/'media/distinct_confirm_20261017_01/SHORT17_HOLD_MEDIA/candidate_POST.rgb8'
    received.parent.mkdir(parents=True);post.parent.mkdir(parents=True)
    np.save(received,np.zeros((1,),dtype=np.uint8));post.write_bytes(b'not-to-open')
    choose=m.choose_input
    monkeypatch.setattr(m,'choose_input',lambda:choose(root))
    chosen=[];read=m.read_rgb
    def capture(value):chosen.append(value);return read(value)
    monkeypatch.setattr(m,'read_rgb',capture)
    backend=Stub();state=m.run_blind(tmp_path/'out',backend=backend)
    assert state['status']=='FAILED' and len(chosen)==1 and chosen[0]['path']==str(received)
    assert backend.calls==[]
    received.unlink();assert choose(root)['path']==str(post)


def test_existing_partial_or_corrupt_marker_never_creates_new_budget(tmp_path):
    (tmp_path/'attempt.json').write_text('{')
    backend=Stub();result=m.run_blind(tmp_path,backend=backend)
    assert result['status']=='UNREADABLE_SAVED_RECORD' and backend.calls==[]


def test_loader_uses_only_cached_fp32_vae_and_original_read_adapter(tmp_path, monkeypatch):
    cache=tmp_path/'models';vae=cache/'arbitrary-existing-directory'/'vae';vae.mkdir(parents=True)
    (vae/'config.json').write_text('{}');(vae/'model.safetensors').write_bytes(b'stub')
    monkeypatch.setattr(m,'MODEL_CACHE',cache)
    calls=[]
    class VAE:
        def eval(self):return self
        def parameters(self):return [types.SimpleNamespace(requires_grad_=lambda value: calls.append(('grad',value)))]
        def to(self,device):calls.append(('device',device));return self
    class Autoencoder:
        @classmethod
        def from_pretrained(cls,path,**kwargs):calls.append(('load',path,kwargs));return VAE()
    fake_torch=types.ModuleType('torch');fake_torch.float32='FP32';fake_torch.device=lambda x:x
    fake_diffusers=types.ModuleType('diffusers');fake_diffusers.AutoencoderKLWan=Autoencoder
    monkeypatch.setitem(sys.modules,'torch',fake_torch);monkeypatch.setitem(sys.modules,'diffusers',fake_diffusers)
    backend=m.Backend();backend.load()
    load=[v for v in calls if v[0]=='load'];assert len(load)==1
    assert load[0][1]==str(vae.parent) and load[0][2]==dict(torch_dtype='FP32',subfolder='vae')
    assert os.environ['HF_HUB_OFFLINE']=='1'
    from main.tube_state import segmented_payload_v1
    fake_torch.from_numpy=lambda x:('tensor',x)
    monkeypatch.setattr(segmented_payload_v1,'read_segments',lambda z,key,path:('original',z,key,path))
    public=m.read_plan();result=backend.read('normalized-token',public)
    assert result==('original',('tensor','normalized-token'),'watermark',public['plan']['estimated_path'])


def test_companion_cli_saved_results_need_no_site_packages(tmp_path):
    archive=builder.ROOT/'notebooks'/(builder.NAME+'_companion.zip')
    code=tmp_path/'source';code.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(code)
    out=tmp_path/'saved';out.mkdir();(out/'attempt.json').write_text('{}')
    (out/'summary.json').write_text(json.dumps(dict(status='READ',fixture='synthetic only')))
    result=subprocess.run([sys.executable,'-S','-m','experiments.paper_results_v1.single_read_v1_cli','--output',str(out)],cwd=code,
                          env={**os.environ,'PYTHONPATH':str(code)},capture_output=True,text=True)
    assert result.returncode==0 and 'READ_SAVED_ONLY' in result.stdout


def test_delivered_notebook_fresh_then_saved_run_all_with_stubs(tmp_path,monkeypatch):
    notebook=json.loads((builder.ROOT/'notebooks'/(builder.NAME+'_colab.ipynb')).read_text())
    assert ''.join(notebook['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    import urllib.request
    archive=(builder.ROOT/'notebooks'/(builder.NAME+'_companion.zip')).read_bytes()
    downloads=[];commands=[];mounts=[]
    monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**k:(downloads.append(a[0]) or io.BytesIO(archive)))
    colab=types.ModuleType('google.colab');colab.drive=types.SimpleNamespace(mount=lambda path:mounts.append(path))
    google=types.ModuleType('google');google.colab=colab
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.colab',colab)
    out=tmp_path/'out';content=tmp_path/'content';content.mkdir()
    class Child:
        def __init__(self,args,**kwargs):
            commands.append(args);self.pid=999
            if '-m' in args:
                assert args[args.index('-m')+1]=='experiments.paper_results_v1.single_read_v1_cli'
                (out/'attempt.json').write_text('{}');(out/'summary.json').write_text('{"status":"STUB_ONLY"}')
        def wait(self,**kwargs):return 0
    monkeypatch.setattr(subprocess,'Popen',Child)
    def execute_all():
        namespace={}
        for cell in notebook['cells']:
            if cell['cell_type']=='code':
                source=''.join(cell['source']).replace(m.OUTPUT,str(out)).replace("Path('/content')",repr(content).replace('PosixPath','Path'))
                exec(compile(source,'delivered-notebook','exec'),namespace)
    execute_all();assert len(downloads)==1 and len(commands)==4 and len(mounts)==1
    assert 'AutoencoderKLWan' in commands[2][-1] and 'WanPipeline' not in commands[2][-1]
    execute_all();assert len(downloads)==1 and len(commands)==4 and len(mounts)==2
