"""Targeted CPU/fake validation; never real models, GPU, media or Drive."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import urllib.error
import urllib.request
import zipfile

import nbformat
import numpy as np
import pytest
import torch

from experiments.paper_results_v1 import raw8_payload_v1 as task
from experiments.paper_results_v1 import build_raw8_payload_v1_colab as builder

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def fixture(tmp_path, *, operations=None, missing=False):
    source=tmp_path/'resumable-independent-four-v1'
    if not missing:
        rows=[dict(slot_id=f'{g.lower()}/{c}',group=g,case_id=c,media={
            'received_path':str(source/'unreadable'/g/c/'received.rgb8'),
            ('mp4_path' if g == 'CLEAN_MP4' else 'source_mp4'):str(source/'joint/run_state/media'/c/'published-attempt-1/published.mp4')}) for g in task.GROUPS for c in task.CASES]
        task.atomic_json(source/'joint/run_state/residual_joint_state.json',dict(key=task.KEY,rows=rows))
        task.atomic_json(source/'effective_independent_four_config.json',dict(payload_bits=[1]*32))
        task.atomic_json(source/'evaluation_report.json',dict(rows=[dict(slot_id='wm_k0_reused/'+r['slot_id'],arm='WM_K0_REUSED',status='EVALUATED',decoded_bits=[1]*32) for r in rows]))
    ops=tmp_path/'ops.json'
    task.atomic_json(ops,operations or task.read_json(task.OPERATIONS))
    return task.initialize(tmp_path/'output',source,ops,'/missing/model','cpu')


class Backend:
    def __init__(self, *, error_bits=False, encode_fail=None, read_interrupt=False):
        self.loads=0; self.inputs=[]; self.reads=[]; self.error_bits=error_bits
        self.encode_fail=encode_fail; self.read_interrupt=read_interrupt

    def load(self):self.loads+=1;return object()

    def rgb(self,row):
        # Distinguishable original source frame IDs; delete is a view of that
        # source, so an incorrect 181-frame identity substitution is observable.
        ids=list(range(181)) if row['group']=='CLEAN_MP4' else [i for i in range(181) if i not in task.DELETE_POSITIONS]
        return torch.tensor(ids,dtype=torch.uint8)[:,None,None,None].expand(-1,1,1,3).clone()

    def encode(self,vae,rgb,observer):
        self.inputs.append(rgb[:,0,0,0].tolist()); observer('STARTED',None)
        if self.encode_fail==len(self.inputs):raise RuntimeError('injected encode failure')
        observer('RETURNED',None)
        return torch.zeros((1,16,(len(rgb)-1)//4+1,40,64))

    def read(self,latent,key,frames,source_map):
        self.reads.append((frames,(frames-1)//4,len(source_map)))
        if self.read_interrupt and len(self.reads)==1:raise KeyboardInterrupt('injected after durable encode')
        R=min(44,(frames-1)//4)
        signed=np.ones((4,R,240),dtype=np.int8)
        if self.error_bits:signed[0,:,0::8]=-1
        votes=[];bits=[]
        for i in range(32):
            ch,rem=divmod(i,8);v=signed[ch,:,rem::8];ones=int((v>0).sum())
            votes.append(dict(ones=ones,zeros=v.size-ones,count=v.size));bits.append(int(ones>v.size//2))
        return dict(status='READ',R=R,output_frames=frames,decoded_bits=bits,votes=votes,
                    signed_votes=signed,zero_mask=(signed==0),truth_inputs=False)

    def args(self):return dict(load_vae=self.load,encode=self.encode,payload_read=self.read,load_rgb=self.rgb)


def test_fixed_eight_distinct_raw_inputs_variable_R_and_zero_gain(tmp_path):
    store=fixture(tmp_path); task.prepare(store)
    before=task.report(store)
    assert before['truth_join']=='NOT_JOINED' and before['raw_bit_errors'] is None
    backend=Backend();task.run_wan(store,**backend.args())
    result=task.read_json(store.output/'evaluation_report.json')
    assert backend.loads==1 and len(backend.inputs)==8 and len(backend.reads)==8
    for row,actual in zip(store.data['rows'],backend.inputs):
        received=backend.rgb(row)[:,0,0,0].tolist()
        assert actual==[received[i] for i in row['operation']['received_index_map']]
    assert [r[0] for r in backend.reads]==[181]*7+[149]
    assert [r[1] for r in backend.reads]==[45]*7+[37] # reader caps 45 at R44
    assert [r['R'] for r in result['rows']]==[44]*7+[37]
    assert result['summary']['paired_conclusion']=='BOTH_READERS_ZERO_ERROR_PATH_IMPROVEMENT_ONLY'
    assert result['summary']['planned_bits']==256 and result['summary']['paired_rows']==8
    assert result['summary']['centered_latent_reuses']==0
    assert 'payload_bits' not in store.path.read_text()
    # A rerun of readable completed rows causes zero load/encode/read.
    again=Backend();task.run_all(task.Store(store.output),**again.args())
    assert again.loads==0 and again.inputs==[] and again.reads==[]


def test_actual_raw_errors_have_bounded_paired_conclusion(tmp_path):
    store=fixture(tmp_path);backend=Backend(error_bits=True);task.run_all(store,**backend.args())
    s=task.report(store)
    assert s['raw_bit_errors']==8 and s['paired_error_reduction']==8
    assert s['paired_conclusion']=='CENTERED_REMOVES_RAW_ERRORS_THIS_BATCH'


def test_failure_retained_and_only_failed_encode_retried(tmp_path):
    store=fixture(tmp_path);backend=Backend(encode_fail=2);task.run_all(store,**backend.args())
    s=task.report(store)
    assert s['status_counts']=={'READ':7,'FAILED':1}
    assert s['missing_bits']==32 and s['paired_rows']==7 and s['paired_error_reduction'] is None
    retry=Backend();store=task.Store(store.output);task.run_all(store,**retry.args())
    assert len(retry.inputs)==1 and len(retry.reads)==1
    assert task.report(store)['cumulative_calls']['wan_encode_clips']=={'COMPLETE':8,'FAILED':1}
    assert store.data['rows'][1]['history'][0]['status']=='FAILED'


def test_interruption_after_raw_latent_does_not_reencode_or_use_centered(tmp_path):
    store=fixture(tmp_path);backend=Backend(read_interrupt=True)
    with pytest.raises(KeyboardInterrupt):task.run_all(store,**backend.args())
    assert task.report(store)['truth_join']=='NOT_JOINED'
    assert len(backend.inputs)==1 and task._paths(store,store.data['rows'][0])[2].is_file()
    retry=Backend();store=task.Store(store.output);task.run_all(store,**retry.args())
    assert len(retry.inputs)==7 and len(retry.reads)==8
    s=task.report(store)
    assert s['cumulative_calls']['wan_encode_clips']=={'COMPLETE':8}
    assert s['cumulative_calls']['payload_read_calls']=={'INTERRUPTED':1,'COMPLETE':8}


def test_missing_source_fixed_eight_no_model_attempt(tmp_path):
    store=fixture(tmp_path,missing=True);backend=Backend();task.run_all(store,**backend.args())
    s=task.report(store)
    assert s['status_counts']=={'MISSING':8} and s['missing_bits']==256
    assert s['paired_rows']==0 and s['planned_rows']==8 and backend.loads==0
    assert s['needs_wan_model'] is False


def test_own_latent_resumes_without_source_locator_json(tmp_path):
    store=fixture(tmp_path);backend=Backend(read_interrupt=True)
    with pytest.raises(KeyboardInterrupt):task.run_all(store,**backend.args())
    (Path(store.data['source_run'])/'joint/run_state/residual_joint_state.json').unlink()
    retry=Backend();store=task.Store(store.output);task.run_all(store,**retry.args())
    s=task.report(store)
    assert len(retry.reads)==1 and retry.loads==0 and retry.inputs==[]
    assert s['status_counts']=={'READ':1,'MISSING':7} and s['missing_bits']==224
    assert s['planned_rows']==8 and s['paired_rows']==1
    assert s['cumulative_calls']['wan_encode_clips']=={'COMPLETE':1}
    assert s['cumulative_calls']['payload_read_calls']=={'INTERRUPTED':1,'COMPLETE':1}


def test_real_source_schema_and_shared_mp4_fallback(tmp_path,monkeypatch):
    from runtime.wan import variable_rgb_media
    store=fixture(tmp_path);task.prepare(store)
    assert all(r['status']=='PREPARED' for r in store.data['rows'])
    clean,delete=store.data['rows'][0],store.data['rows'][4]
    assert clean['media']['mp4_path']==delete['media']['mp4_path']
    cached={};decodes=[]
    full=torch.arange(181,dtype=torch.uint8)[:,None,None,None].expand(-1,1,1,3).clone()
    def load(path,shape):
        if str(path) not in cached:raise FileNotFoundError(path)
        return cached[str(path)]
    def decode(source,target,expected_frames):
        assert str(source)==clean['media']['mp4_path'] and expected_frames==181
        decodes.append(str(source));cached[str(target)]=full
    monkeypatch.setattr(task,'_load_rgb_path',load)
    monkeypatch.setattr(variable_rgb_media,'decode_published',decode)
    got_clean=task._received_rgb(store,clean,1)
    got_delete=task._received_rgb(store,delete,1)
    assert torch.equal(got_clean,full)
    keep=[i for i in range(181) if i not in task.DELETE_POSITIONS]
    assert torch.equal(got_delete,full[keep]) and len(decodes)==1
    assert store.data['media_decodes'][0]['status']=='COMPLETE'


def test_unsupported_and_unresolved_retained_no_truth_selected_map(tmp_path):
    from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span
    operations=task.read_json(task.OPERATIONS)
    unsupported=operations['rows'][0];path=[10]*181
    unsupported['estimate'].update(status='ESTIMATED',path=path)
    unsupported['operation']=decode_visible_span(path)
    unresolved=operations['rows'][1]
    unresolved['estimate'].update(status='UNRESOLVED',path=None);unresolved['operation']=None
    store=fixture(tmp_path,operations=operations);backend=Backend();task.run_all(store,**backend.args())
    assert len(backend.inputs)==6
    assert task.report(store)['status_counts']=={'UNSUPPORTED':1,'UNRESOLVED':1,'READ':6}
    assert task.report(store)['missing_bits']==64
    saved=(store.output/'saved_raw_operations.json').read_bytes()
    # Changing only report truth cannot replace a saved path or cause work.
    task.atomic_json(Path(store.data['source_run'])/'effective_independent_four_config.json',dict(payload_bits=[0]*32))
    again=Backend();task.run_all(task.Store(store.output),**again.args())
    assert (store.output/'saved_raw_operations.json').read_bytes()==saved and not again.inputs


def test_missing_centered_reference_never_fills_or_changes_raw(tmp_path):
    store=fixture(tmp_path)
    task.atomic_json(Path(store.data['source_run'])/'evaluation_report.json',dict(rows=[]))
    backend=Backend();task.run_all(store,**backend.args())
    s=task.report(store)
    assert len(backend.inputs)==8 and s['raw_read_rows']==8 and s['paired_rows']==0
    assert s['paired_conclusion']=='INCOMPLETE_FIXED_EIGHT_PAIR_COMPARISON'


def test_missing_operations_retained_without_selector_fallback(tmp_path):
    store=fixture(tmp_path);Path(store.data['operations_input']).unlink()
    backend=Backend();task.run_all(store,**backend.args())
    s=task.report(store)
    assert s['status_counts']=={'MISSING':8} and backend.loads==0 and s['selector_calls']==0


def test_saved_source_key_error_does_not_trigger_model_setup(tmp_path):
    store=fixture(tmp_path);p=Path(store.data['source_run'])/'joint/run_state/residual_joint_state.json'
    d=task.read_json(p);d['key']='different';task.atomic_json(p,d)
    task.prepare(store);s=task.report(store)
    assert s['status_counts']=={'FAILED':8} and not s['needs_wan_model']


def test_portable_cli_and_notebook_missing_input_run_all(tmp_path,monkeypatch):
    monkeypatch.setattr(builder,'NOTEBOOK',tmp_path/'raw8.ipynb')
    monkeypatch.setattr(builder,'ZIP',tmp_path/'raw8.zip')
    builder.build()
    notebook=nbformat.read(builder.NOTEBOOK,as_version=4);nbformat.validate(notebook)
    code=[c.source for c in notebook.cells if c.cell_type=='code']
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for text in code:compile(text,'notebook','exec')
    with zipfile.ZipFile(builder.ZIP) as archive:
        assert 'experiments/paper_results_v1/raw8_saved_operations.json' in archive.namelist()
        assert not any('.git' in Path(n).parts or '..' in Path(n).parts or n.startswith('/') for n in archive.namelist())
        portable=tmp_path/'portable';archive.extractall(portable)
    env={**os.environ,'PYTHONPATH':str(portable)}
    result=subprocess.run([sys.executable,'-m','experiments.paper_results_v1.raw8_payload_v1_cli','--help'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    # Execute the actual delivered cells with a fake Colab mount/download and
    # missing source. Every real subprocess stays CPU/no-model and writes tmp.
    colab=ModuleType('google.colab')
    colab.drive=SimpleNamespace(mount=lambda _:None)
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    downloads=[]
    def download(url,timeout):
        downloads.append((url,timeout));return io.BytesIO(builder.ZIP.read_bytes())
    monkeypatch.setattr(urllib.request,'urlopen',download)
    workspace=tmp_path/'content';workspace.mkdir()
    namespace={}
    for text in code:
        exec(compile(text.replace('/content',str(workspace)),'delivered-notebook','exec'),namespace)
    summary=task.read_json(namespace['OUTPUT_ROOT']/'summary.json')
    assert summary['status_counts']=={'MISSING':8} and summary['planned_rows']==8
    assert summary['cumulative_calls']['vae_loads']=={}
    assert summary['media_encode_calls']==0
    assert downloads==[('https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/paper-results-v1/notebooks/paper_results_v1_raw8_payload_v1_companion.zip',60)]
    assert 'files.upload' not in '\n'.join(code)


@pytest.mark.parametrize('failure',['download','corrupt_zip','unsafe_path'])
def test_notebook_source_loading_reports_actual_failures(tmp_path,monkeypatch,failure):
    monkeypatch.setattr(builder,'NOTEBOOK',tmp_path/'raw8.ipynb')
    monkeypatch.setattr(builder,'ZIP',tmp_path/'raw8.zip')
    builder.build()
    code=next(c.source for c in nbformat.read(builder.NOTEBOOK,as_version=4).cells if c.id=='source-and-fixed-task')
    unsafe=io.BytesIO()
    with zipfile.ZipFile(unsafe,'w') as archive:
        archive.writestr('experiments/paper_results_v1/raw8_payload_v1_cli.py','')
        archive.writestr('../escape.py','raise RuntimeError("must not extract")')
    def download(url,timeout):
        if failure=='download':raise urllib.error.URLError('injected unavailable source')
        return io.BytesIO(b'invalid zip' if failure=='corrupt_zip' else unsafe.getvalue())
    monkeypatch.setattr(urllib.request,'urlopen',download)
    error={'download':RuntimeError,'corrupt_zip':zipfile.BadZipFile,'unsafe_path':ValueError}[failure]
    match={'download':'source download/save failed','corrupt_zip':'not a zip file','unsafe_path':'unsafe source archive path'}[failure]
    with pytest.raises(error,match=match):
        exec(compile(code.replace('/content',str(tmp_path/'content')),'delivered-source-cell','exec'),{})
    assert not (tmp_path/'content/escape.py').exists()


def test_real_saved_operations_have_adopted_geometry():
    rows=task.read_json(task.OPERATIONS)['rows']
    assert [r['slot_id'] for r in rows]==task.slots()
    for row in rows:assert task.validate_operation(row,row)=='SUPPORTED'
    assert [(r['operation']['output_frames'],r['operation']['wan_support']) for r in rows]==[(181,44)]*7+[(149,37)]
