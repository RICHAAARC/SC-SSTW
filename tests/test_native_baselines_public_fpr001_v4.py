"""Artificial CPU checks only: no real media, codecs, weights, model loads or GPU."""
import ast
from collections import Counter
import importlib
import json
from pathlib import Path
import shutil
import socket
import sys

import numpy as np
import pytest

from main.baselines import public_fpr001_v4 as p
from main.baselines.public_metrics_v4 import evaluate_view
from runtime.wan import native_baseline_io_v4 as io
from runtime.wan import native_baselines_public_v4 as runner
from runtime.wan.native_baseline_storage_v4 import (Budget,UnitStore,ArrayLog,ViewEpochs,
    Unavailable,RecoveryExhausted,runtime_session,append_json)
from runtime.wan.native_baseline_storage_v1 import write_json,read_json

pytestmark=pytest.mark.unit


def budget(path,runtime='one',**limits):
    default=p.physical_budget()
    for category,count in limits.items():default['base'][category]=count;default['recovery'][category]=count
    return Budget(path,runtime,default)


def source(sid,ordinal,split='DEV'):
    return dict(id=sid,global_ordinal=ordinal,split=split,action='Action'+str(ordinal),group=ordinal)


def raw(bits,observed=None):
    bits=np.asarray(bits,np.uint8);n=len(bits)
    return dict(bits=bits,valid=np.ones(n,bool),observed=np.ones(n,bool) if observed is None else np.array(observed,bool),
        logits=np.concatenate([np.ones((n,1),np.float32),bits.astype(np.float32)*2-1],axis=1))


def test_roster_and_independent_arithmetic():
    config,sources=runner.load_inputs()
    assert len(sources)==1064 and Counter(s['split'] for s in sources)=={'DEV':64,'CAL':500,'TEST':500}
    original=read_json(runner.ROOT/'experiments/wan_state_clock/configs/native_baselines_paper_fpr1_v3.sources.json')
    if isinstance(original,dict):original=original['sources']
    assert [{k:v for k,v in s.items() if k!='global_ordinal'} for s in sources[64:]]==original
    assert sum(v['frames'] for v in p.views())==6664
    assert sum(len(p.batches(v['frames'],8)) for v in p.views())==852
    assert sum(len(p.batches(v['frames'],32)) for v in p.views())==232
    assert sum(v['frames']-1 for v in p.views())==6624
    assert p.physical_budget()['detector_forwards']==1628*852+(2192+1628)*232
    assert all(0<x['upper']<.001 for x in p.nominal_bounds().values())


def test_query_full_bytes_collision_and_frameindex():
    calls=[]
    def random_bytes(n):calls.append(n);return bytes(n)
    queries=p.allocate_queries([source('a',1)],random_bytes)
    assert calls==[4,32,32,4,32,32,31]
    assert queries['frameindex/a']['query']==queries['pixelseal/a']['query'][:248]
    assert all(r['wrong_query_collision'] for r in queries.values())
    messages=p.frame_messages([1]*248,179,2)
    assert messages[-1][-8:]==[1,0,1,1,0,1,0,0]


def test_plan_semantics_and_donor_boundaries():
    a,b=source('a',1),source('b',2);plan=p.source_plan(a,[a,b])
    assert plan['donor']=='b'
    assert [len(plan['maps'][f'T{i:02}']) for i in range(16)]==list(p.LENGTHS)
    assert sum(r['kind']=='foreign' for r in plan['maps']['T12'])==18
    synthetic=[r for r in plan['maps']['T13'] if r['kind']=='synthetic']
    assert len(synthetic)==18 and all(r['index'] is None and r['weights']==[.5,.5] for r in synthetic)
    kept5={r['index'] for r in plan['maps']['T05']};kept6={r['index'] for r in plan['maps']['T06']}
    assert kept6<=kept5
    assert len({r['index'] for r in plan['maps']['T11']})==181
    assert plan==p.source_plan(a,[a,b])


@pytest.mark.parametrize('method,n,threshold',[('wam',32,30),('pixelseal',256,161),('videoseal',256,161),('frameindex',248,165)])
def test_query_threshold_and_partial(method,n,threshold):
    query=[1]*n;bits=np.zeros((2,p.METHODS[method]),np.uint8);bits[:,:threshold]=1
    value=raw(bits);assert p.verify(method,value,query,planned_frames=2)['decision'] is True
    value['observed'][1]=False;partial=p.verify(method,value,query,planned_frames=2)
    assert partial['complete'] is False
    assert partial['decision'] is (True if method in ('wam','frameindex') else None)
    if method in ('wam','frameindex'):assert partial['score'] is None and partial['score_lower_bound']==threshold
    value['bits'][:,:threshold]=0;value['logits'][:,1:]=-1
    assert p.verify(method,value,query,planned_frames=2)['decision'] is None


def test_empty_masks_and_family_unknown_disjoint():
    value=raw(np.zeros((2,32),np.uint8));value['valid'][:]=False
    assert p.verify('wam',value,[0]*32,planned_frames=2)['decision'] is False
    rows=[dict(decision=False,complete=True)]*39+[dict(decision=True,complete=False)]
    assert p.family_event(rows) is True and p.all_robust(rows) is False
    from main.baselines.native_paper_fpr1_v3 import rate_summary
    rate=rate_summary([True,None,False]);assert (rate['positive'],rate['unknown'],rate['negative'])==(1,1,1)
    assert rate_summary([False]*500)['conservative_one_sided95_upper']==pytest.approx(.0059735515)


def test_frameindex_synthetic_foreign_and_invalid_index():
    a,b=source('a',1),source('b',2);queries=p.allocate_queries([a,b],lambda n:bytes(n))
    mapping=[dict(kind='target',source_id='a',index=3),dict(kind='foreign',source_id='b',index=7),
        dict(kind='synthetic',source_id='a',index=None,parents=[8,9]),dict(kind='target',source_id='a',index=180)]
    bits=np.array([p.frame_messages([0]*248,3,1)[0],p.frame_messages([0]*248,7,1)[0],p.frame_messages([0]*248,8,1)[0],[0]*248+[1]*8],np.uint8)
    summary,arrays=evaluate_view('frameindex','frameindex',raw(bits),queries['frameindex/a'],mapping,queries)
    assert summary['payload']['undefined_bits']==8 and summary['payload']['defined_bits']==1016
    assert summary['payload']['error_bits']==4  # 255 versus 180; donor7 is not an error
    assert summary['time']['nodes']['foreign_false_target_attribution']==1
    assert summary['time']['nodes']['synthetic_false_precise_time']==1
    assert summary['time']['nodes']['target_owner_positive_invalid_index']==1
    assert summary['time']['query_accept_and_exact_index']==1
    assert arrays['index'].tolist()==[3,7,8,-1] and len(arrays['edge_truth'])==3


def test_actual_array_checkpoint_torn_tail_and_no_repeated_inference(tmp_path):
    b=budget(tmp_path,detect_wam=20);store=UnitStore(tmp_path,'u',b);calls=[]
    def call():calls.append(1);return dict(x=np.arange(3))
    store.perform('b','detect_wam',call,arrays=True)
    with store.segment.open('ab') as f:f.write(b'bad')
    restored=UnitStore(tmp_path,'u',budget(tmp_path,'two',detect_wam=20))
    assert restored.perform('b','detect_wam',call,arrays=True)['x'].tolist()==[0,1,2]
    assert len(calls)==1 and not restored.record_errors  # no historical tail scan
    assert store.segment.read_bytes().endswith(b'bad')
    restored.perform('c','detect_wam',call,arrays=True)
    assert len(UnitStore(tmp_path,'u',budget(tmp_path,'three',detect_wam=20)).arrays)==2


def test_bounded_infrastructure_calls_and_method_failure(tmp_path):
    def fail():raise OSError('synthetic unavailable storage')
    for i in range(3):
        store=UnitStore(tmp_path,'u',budget(tmp_path,str(i),detect_wam=20))
        with pytest.raises(Unavailable):store.perform('x','detect_wam',fail,arrays=True)
        with pytest.raises(Unavailable):store.perform('x','detect_wam',fail,arrays=True)
    fourth=UnitStore(tmp_path,'u',budget(tmp_path,'four',detect_wam=20))
    with pytest.raises(RecoveryExhausted):fourth.perform('x','detect_wam',fail,arrays=True)
    assert fourth.budget.base['detect_wam']==1 and fourth.budget.extra['detect_wam']==2
    with pytest.raises(Unavailable):fourth.perform('bad','detect_wam',lambda:dict(x=np.array([np.nan])),arrays=True,validate=lambda x:(_ for _ in ()).throw(ValueError('nonfinite')))
    restored=UnitStore(tmp_path,'u',budget(tmp_path,'five',detect_wam=20))
    with pytest.raises(Unavailable):restored.perform('bad','detect_wam',lambda:dict(x=np.ones(1)),arrays=True)


def test_whole_view_recovery_requires_every_batch_budget(tmp_path):
    b=budget(tmp_path,detect_wam=20);store=UnitStore(tmp_path,'u',b);ep=ViewEpochs(store,'G1/T00',tmp_path/'cache')
    first,_=ep.choose([(0,8),(8,1)],'detect_wam')
    store.perform('G1/T00/'+first['id']+'/batch0','detect_wam',lambda:dict(x=np.ones(1)),arrays=True,physical_key='G1/T00/batch0')
    next_store=UnitStore(tmp_path,'u',budget(tmp_path,'two',detect_wam=20))
    next_store.budget.limits['recovery']['detect_wam']=1
    with pytest.raises(RecoveryExhausted):ViewEpochs(next_store,'G1/T00',tmp_path/'cache').choose([(0,8),(8,1)],'detect_wam')
    assert next_store.budget.extra['detect_wam']==0


def test_partial_pixels_preserved_and_only_missing_batch_resumed(tmp_path,monkeypatch):
    def validate(value,n=181):assert value.shape==(n,1,256,3)
    monkeypatch.setattr(io,'validate_rgb',validate)
    output=tmp_path/'result';cache=tmp_path/'cache';source_row=source('a',1)
    view=dict(id='G1/T00',frames=9,operation='T00',generation='G1')
    class Interrupted(FakeAdapter):
        def detect(self,pixels):
            if len(pixels)==1:raise OSError('artificial interruption')
            return super().detect(pixels)
    ops=io.MediaOperations(output,cache,budget(output,detect_wam=20),FakeMedia())
    value,epoch=io.extract_view(ops,source_row,'wam','OFF',view,Interrupted('wam','unused'),lambda:np.zeros((9,1,256,3),np.uint8))
    assert value['observed'].sum()==8 and Path(epoch['input']).is_relative_to(output/'failures')
    first=FakeAdapter.calls['detect/wam']
    ops2=io.MediaOperations(output,cache,budget(output,'two',detect_wam=20),FakeMedia())
    value,again=io.extract_view(ops2,source_row,'wam','OFF',view,FakeAdapter('wam','unused'),lambda:(_ for _ in ()).throw(AssertionError('must reuse exact saved input')))
    assert value['observed'].all() and again['id']==epoch['id']
    assert FakeAdapter.calls['detect/wam']==first+1 and ops2.budget.extra['detect_wam']==1
    assert Path(epoch['input']).is_file()  # failed evidence never cleaned, even after successful continuation
    assert read_json(output/'units/native/a/wam/OFF/views/G1/T00/status.json')['complete']


def test_committed_receipt_is_trusted_until_array_is_used(tmp_path):
    store=UnitStore(tmp_path,'u',budget(tmp_path,detect_wam=20))
    store.event('batch0','COMMITTED',array_file='missing.bin')
    assert store.reusable('batch0')  # committed history is trusted by scheduling
    with pytest.raises(OSError):store.arrays['batch0']  # actual use reports the missing file


def test_extra_loader_cap_per_runtime(tmp_path):
    ops=io.MediaOperations(tmp_path,tmp_path/'cache',budget(tmp_path,model_load=30),FakeMedia())
    for i in range(4):append_json(tmp_path/'model_loads.jsonl',dict(key='pack'+str(i)+'/wam',category='model_load',runtime_id='old',status='STARTED'))
    loaders=runner.Loaders(ops,'unused',FakeAdapter,FakeLPIPS)
    for i in range(3):loaders.watermark('pack'+str(i),'wam').close()
    with pytest.raises(RecoveryExhausted):loaders.watermark('pack3','wam')
    assert ops.budget.extra['model_load']==3


@pytest.mark.parametrize('method,step,card',[('pixelseal',8,'pixelseal'),('videoseal',4,'videoseal_1.0')])
def test_seal_load_reads_official_blender_attributes(tmp_path,monkeypatch,method,step,card):
    import types
    import torch
    from runtime.wan.native_baseline_models_v4 import NativeAdapterV4
    (tmp_path/'videoseal').mkdir()
    class OfficialShape(torch.nn.Module):
        def __init__(self):
            super().__init__();self.chunk_size=32;self.step_size=step;self.img_size=256;self.video_mode='repeat';self.clamp=True
            self.blender=types.SimpleNamespace(scaling_w=.2,scaling_i=1.)
            self.attenuation=object();self.embedder=torch.nn.Identity();self.detector=torch.nn.Identity()
    model=OfficialShape();calls=[]
    monkeypatch.setitem(sys.modules,'videoseal',types.SimpleNamespace(load=lambda name:(calls.append(name),model)[1]))
    adapter=NativeAdapterV4(method,tmp_path,device='cpu')
    details=adapter.load();assert calls==[card] and details['parameters']['step_size']==step
    assert details['parameters']['scaling_w']==.2 and not hasattr(model,'scaling_w')
    adapter.close()


def test_execution_complete_subset_excludes_partial_positive():
    from runtime.wan.native_baseline_evaluation_v4 import rate_record
    result=rate_record([True]+[False]*499,[False]+[True]*499)
    assert (result['planned'],result['positive'],result['unknown'])==(500,1,0)
    assert result['known_decision_subset_denominator']==500
    assert (result['complete_subset_denominator'],result['complete_subset_positive'])==(499,0)
    partial=rate_record([True,None,False],[False,False,True])
    assert (partial['positive'],partial['unknown'],partial['negative'])==(1,1,1)
    assert partial['complete_subset_denominator']==1 and partial['execution_incomplete']==2
    complete=rate_record([True,False],[True,True])
    assert complete['complete_subset_cp95']==complete['known_decision_subset_cp95']
    # FAMILY40 with one partial witness is known TRUE but is not an execution-complete source.
    group=[dict(decision=False,complete=True)]*39+[dict(decision=True,complete=False)]
    family=rate_record([p.family_event(group)],[all(r['complete'] for r in group)])
    assert family['positive']==1 and family['unknown']==0 and family['complete_subset_denominator']==0


def test_initialize_only_after_runtime_ownership(tmp_path,monkeypatch):
    output=tmp_path/'result';cache=tmp_path/'cache';calls=[]
    monkeypatch.setattr(runner,'load_inputs',lambda config:({},[]))
    def initialize(*args):
        calls.append(1);assert read_json(output/'runtime_owner.json')['status']=='ACTIVE'
        raise RuntimeError('artificial stop after ownership check')
    monkeypatch.setattr(runner,'initialize',initialize)
    with pytest.raises(RuntimeError,match='artificial stop'):runner.run(output,cache,'unused')
    assert calls==[1]
    calls.clear()
    with runtime_session(output):
        with pytest.raises((BlockingIOError,Unavailable)):runner.run(output,cache,'unused')
    assert calls==[]  # denied concurrent startup did not allocate or replace queries


def test_status_receipt_without_actual_native_arrays_is_incomplete(tmp_path):
    source_row=source('a',1)
    for view in p.views():write_json(tmp_path/'units/native/a/wam/OFF/views'/view['id']/'status.json',dict(complete=True))
    assert runner.native_complete(tmp_path,source_row,'wam','OFF') is False


@pytest.mark.parametrize('name',['A_payload','C_time','D_quality','D_resources'])
def test_completed_tables_are_actually_parseable(tmp_path,name):
    from runtime.wan.native_baseline_colab_v4 import tables_ready
    write_json(tmp_path/'result.json',dict(status='fixture'))
    for table in ('A_payload','B_ownership','C_time','D_quality'):
        write_json(tmp_path/'tables'/(table+'.json'),[dict(value=1)])
        (tmp_path/'tables'/(table+'.csv')).write_text('value\n1\n')
    write_json(tmp_path/'tables/D_resources.json',dict(fixture=True))
    assert tables_ready(tmp_path)
    target=tmp_path/'tables'/(name+'.json');saved=target.read_text();target.write_text('{broken-json')
    assert not tables_ready(tmp_path)
    target.write_text(saved)
    (tmp_path/'tables/C_time.csv').write_text('value\n"unterminated\n')
    assert not tables_ready(tmp_path)


def test_owner_no_heartbeat_steal(tmp_path):
    write_json(tmp_path/'runtime_owner.json',dict(status='ACTIVE',host='different-host',pid=1))
    with pytest.raises(Unavailable):
        with runtime_session(tmp_path):pass
    with runtime_session(tmp_path,previous_runtime_stopped=True) as row:assert row['ordinal']==0
    assert read_json(tmp_path/'runtime_owner.json')['status']=='CLOSED'


def test_adapter_real_tail_api_without_loading_models():
    import torch
    from runtime.wan.native_baseline_models_v4 import NativeAdapterV4
    class Model:
        def __init__(self):self.embeds=[];self.detects=[]
        def embed(self,x,**kwargs):self.embeds.append((len(x),kwargs['is_video'],tuple(kwargs['msgs'].shape),x.dtype));return {'imgs_w':x}
        def detect(self,x,**kwargs):self.detects.append((len(x),kwargs['is_video'],x.dtype));return {'preds':torch.zeros((len(x),257),dtype=torch.float32)}
    a=NativeAdapterV4('videoseal','unused',device='cpu');a.model=Model()
    for start,n in p.writer_batches('videoseal'):a.embed_batch(np.zeros((n,1,1,3),np.uint8),[0]*256,method='videoseal',start=start)
    assert [x[0] for x in a.model.embeds]==[128,53]
    a.method='pixelseal';a.embed_batch(np.zeros((5,1,1,3),np.uint8),p.frame_messages([0]*248,176,5),method='frameindex',start=176)
    assert a.model.embeds[-1][1:]==(False,(5,256),torch.float32)
    result=a.detect(np.zeros((21,1,1,3),np.uint8));assert result['logits'].shape==(21,257)
    assert a.model.detects==[(21,False,torch.float32)]


def test_notebook_builder_and_passive_cli(tmp_path):
    from scripts.build_native_baselines_public_fpr001_v4_notebook import build
    notebook=read_json(build(tmp_path));actual=read_json(runner.ROOT/'notebooks/native_baselines_public_fpr001_v4.ipynb')
    assert notebook==actual
    assert ''.join(notebook['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']));assert not cell['outputs']
    module=importlib.import_module('experiments.wan_state_clock.native_baselines_public_fpr001_v4_run')
    with pytest.raises(SystemExit) as e:module.main(['--help'])
    assert e.value.code==0


class FakeMedia:
    def download(self,source,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'artificial');return dict(bytes=10)
    def source_probe(self,path):return dict(width=256,height=1,r_frame_rate='25/1',nb_frames='181')
    def normalize(self,path,metadata):return np.zeros((181,1,256,3),np.uint8)
    def encode(self,pixels,path,fps,**kw):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as f:np.save(f,pixels)
        return dict(path=str(path),bytes=path.stat().st_size)
    def probe(self,path,shape,fps,**kw):return dict(frames=shape[0])
    def decode(self,path,shape):
        with Path(path).open('rb') as f:return np.load(f)
    def jpeg_encode(self,frame):return frame.copy()
    def jpeg_decode(self,data):return data.copy()


class FakeAdapter:
    calls=Counter()
    def __init__(self,method,vendor):self.method=method
    def load(self):self.calls['load/'+self.method]+=1;return {'fixture':True}
    def close(self):pass
    def resources(self):return {'fixture':True}
    def embed_batch(self,pixels,message,*,method,start):
        self.calls['embed/'+method]+=1;out=pixels.copy();msg=np.asarray(message)
        if msg.ndim==1:msg=np.tile(msg,(len(pixels),1))
        out[:,0,:msg.shape[1],0]=msg*255;return out
    def detect(self,pixels):
        self.calls['detect/'+self.method]+=1;n=len(pixels);bits=(pixels[:,0,:,0]>127).astype(np.uint8)
        if self.method=='wam':
            return dict(bits=bits[:,:32],valid=np.ones(n,bool),selected_count=np.ones(n,np.uint32),
                center_count=np.ones(n,np.uint32),sums=np.ones((n,32),np.float64),means=np.ones((n,32),np.float64),mask_shape=np.array([256,256]))
        return dict(bits=bits,valid=np.ones(n,bool),logits=np.concatenate([np.ones((n,1),np.float32),bits.astype(np.float32)*2-1],axis=1))


class FakeLPIPS:
    def load(self):return {'fixture':True}
    def pair(self,a,b):return dict(lpips=np.zeros(len(a),np.float32))
    def close(self):pass


def test_artificial_full_scheduler_tables_shared_off_and_resume(tmp_path,monkeypatch):
    from runtime.wan import native_baseline_quality_v4 as quality
    def validate(value,n=181):assert value.shape==(n,1,256,3) and value.dtype==np.uint8
    monkeypatch.setattr(io,'validate_rgb',validate)
    monkeypatch.setattr(io,'spatial',lambda rgb,op,seed:rgb.copy())
    monkeypatch.setattr(quality,'cpu_quality',lambda a,b:dict(frames=[dict(frame=i,mse=0.,psnr_db=None,psnr_positive_infinity=True,ssim=1.) for i in range(181)],
        edges=[dict(edge=i,residual_difference_rms=0.) for i in range(180)],planned_frames=181,planned_edges=180))
    sources=[source('a',1),source('b',2),source('c',3,'CAL'),source('d',4,'TEST')]
    queries=p.allocate_queries(sources,__import__('os').urandom);plans={s['id']:p.source_plan(s,sources[:2]) for s in sources}
    output=tmp_path/'result';cache=tmp_path/'cache';b=budget(output,lpips_load=20)
    ops=io.MediaOperations(output,cache,b,FakeMedia());loaders=runner.Loaders(ops,'unused',FakeAdapter,FakeLPIPS)
    FakeAdapter.calls.clear();run=runner.Runner(ops,sources,queries,plans,loaders,quality.measure)
    result=run.run();loaders.close()
    assert result['logical_views']==4*(4+3)*40 and result['execution_incomplete_views']==0
    assert result['quality_complete_stages']==3*4*3
    assert b.base['detect_pixelseal']==(4+3*2)*232  # Pixel OFF is never read a second time for FrameIndex.
    assert b.base['detect_wam']==7*852 and b.base['detect_videoseal']==7*232
    assert b.base['video_encode']==(4+3*4)*25 and b.base['video_probe']==(4+3*4)*25
    assert b.base['normalize']==4 and b.base['jpeg_encode']==(4+3*4)*181
    assert not b.extra
    before=FakeAdapter.calls.copy();b2=budget(output,'two',lpips_load=20)
    # The new default trusts completed results and does not inspect/repair historical tables.
    shutil.rmtree(cache)
    (output/'tables/C_time.csv').unlink()
    from runtime.wan.native_baseline_colab_v4 import tables_ready
    assert not tables_ready(output)
    ops2=io.MediaOperations(output,cache,b2,FakeMedia());loaders2=runner.Loaders(ops2,'unused',FakeAdapter,FakeLPIPS)
    again=runner.Runner(ops2,sources,queries,plans,loaders2,quality.measure).run();loaders2.close()
    assert again['execution_incomplete_views']==0 and FakeAdapter.calls==before and not b2.extra
    table=read_json(output/'tables/B_ownership.json')
    assert all(r['planned']==1 for r in table if r['split']=='TEST')
    assert (output/'tables/A_payload.csv').is_file() and not (output/'tables/C_time.csv').is_file()
    assert not tables_ready(output)
    native_file=next((output/'units/native/a/wam/OFF').glob('arrays-*.bin'))
    held=native_file.with_suffix('.held');native_file.rename(held)
    assert runner.native_complete(output,sources[0],'wam','OFF') is False
    held.rename(native_file)
    assert runner.native_complete(output,sources[0],'wam','OFF') is True
