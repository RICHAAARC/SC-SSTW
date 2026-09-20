"""Necessary CPU/synthetic protocol and complete-path tests; no real scientific scoring."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch
from experiments.wan_state_clock import flow_tube_detection_protocol as p,flow_tube_detection_run as run
from main.tube_state import projection_margin as carrier
from test_flow_tube_state_guidance import native,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

def stat(a,b):return p.statistic({'rankings':{p.MODE:{'best_by_message':{'0':{'score':a},'1':{'score':b}}}}})
def cal():
    return {c:{'sources':{'OFF':{'views':{v:{'statistic':stat(i/100,-1)} for v in p.VIEWS},'sequence':stat(i/100,-1)}}} for i,c in enumerate(p.CALIBRATION)}

def test_nine_sources_strict_freeze_missing_and_ties():
    c=cal();f=p.freeze(c);assert f['threshold']==.08 and f['source_denominator']==9
    assert p.decide(stat(.08,0),f)['accepted'] is False
    tied=stat(.09,.09);assert p.decide(tied,f)['accepted'] is True and tied['selected_message'] is None
    del c['cal03']['sources']['OFF']['views']['speed125']
    f=p.freeze(c);assert f['calibration_status']=='UNCALIBRATED' and f['threshold'] is None
    assert p.decide(stat(100,0),f)['accepted'] is None
    assert 'cal03/speed125' in f['missing']

def test_sequence_same_message_average_and_positive_union_with_missing():
    views={v:{'statistic':s} for v,s in zip(p.SEQUENCE,[stat(.9,0),stat(0,.9),stat(0,.9)])}
    s=p.sequence(views);assert s['score']==pytest.approx(.6) and s['selected_message']==1
    del views['crop4'];assert p.sequence(views)['status']=='UNMEASURED'
    evaluation={'eval01':{'sources':{'OFF':{'views':{'speed125':{'decision':{'accepted':True}}}}}}}
    r=p.summary(evaluation)['off_source_any'];assert r['false_accept']==1 and r['complete']==0 and r['unresolved']==1

def test_geometry_transform_and_threshold_pre_model(tmp_path,monkeypatch):
    x=torch.arange(181)[:,None,None,None].expand(181,2,2,3)
    for v in p.VIEWS:
        assert run.transformed(x,v)[:,0,0,0].tolist()==run.frame_indices(v)
        assert len(run.frame_indices(v))==run.LENGTHS[v]
    assert run.frame_indices('delete90')[89:92]==[89,91,92]
    assert run.frame_indices('speed125')[-1]==180
    entered=[];monkeypatch.setattr(run,'prepare_generation',lambda *a,**kw:entered.append(True))
    with pytest.raises(ValueError,match='threshold'):run.generate('eval01',tmp_path/'eval')
    assert not entered and not (tmp_path/'eval').exists()

def test_fake_native_eval_complete_seven_views_and_blindness(tmp_path,monkeypatch,native):
    f=p.freeze(cal());threshold=tmp_path/'threshold.json';threshold.write_text(json.dumps(f));digest=run.sha(threshold)
    pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(run,'prepare_generation',lambda *a,**kw:(pipe,torch.zeros(carrier.SHAPE),torch.tensor(1.),torch.tensor(-1.),torch.float32))
    out=tmp_path/'eval01';g=run.generate('eval01',out,threshold,digest)
    assert g['status']=='GENERATION_COMPLETE',g['failures']
    assert g['actual_calls']['transformer_completed']==100 and g['actual_calls']['scheduler_step_completed']==52
    class VAE(torch.nn.Module):
        config=SimpleNamespace(_commit_hash='fake')
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.ones(1))
    monkeypatch.setattr(run,'load_frozen_vae',lambda *a:VAE());monkeypatch.setattr(run,'_clear_cache',lambda *a:None)
    monkeypatch.setattr(run,'SPATIAL',(2,2,3));monkeypatch.setattr(run,'release',lambda:None)
    monkeypatch.setattr(run,'decode_normalized_latent',lambda *a:(torch.arange(181).float()/255).reshape(181,1,1,1).expand(181,2,2,3).clone())
    cache={};saved=[]
    def encode(rgb,path,*a):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake');cache[str(path)]=rgb.clone();saved.append(str(path))
    monkeypatch.setattr(run,'encode_rgb',encode);monkeypatch.setattr(run,'read_mp4',lambda path:cache[str(path)].clone())
    inputs=[]
    def vae_encode(vae,rgb):
        inputs.append([round(float(v)*255) for v in rgb[:,0,0,0]])
        return torch.zeros((1,16,1+(len(rgb)-1)//4,40,64))
    monkeypatch.setattr(run,'reencode_rgb24_readback',vae_encode)
    real_save=torch.save
    def save(value,path):
        if Path(path).name.startswith('g'):Path(path).write_bytes(b'fake receiver')
        else:real_save(value,path)
    monkeypatch.setattr(run.torch,'save',save)
    blind=[]
    def receiver(obs,book):
        blind.append(sorted(obs));return {'rankings':{p.MODE:{'best_by_message':{'0':{'score':.2},'1':{'score':.1}},'best':None,'top_ties':[]}}}
    monkeypatch.setattr(run.state_clock,'read',receiver);monkeypatch.setattr(run,'sync_reporting',lambda d,v:{'reporting_only':True})
    r=run.media('eval01',out,threshold,digest)
    assert r['status']=='EXECUTION_COMPLETE',r['failures']
    expected=[]
    for arm in p.ARMS:
        for view in p.VIEWS:
            indices=run.frame_indices(view)
            for origin in range(4):
                groups=(len(indices)-origin-1)//4;expected.append(indices[origin:origin+1+4*groups])
    assert inputs==expected and len(blind)==21 and len(saved)==12
    assert all(r['actual_calls'][k+'_completed']==v for k,v in run.plan('eval01').items())
    assert r['sources']['OFF']['views']['full181']['attribution']['unique_correct'] is None
    assert r['sources']['LAST_B']['views']['full181']['decision']['accepted'] is True
    assert r['sources']['LAST_B']['views']['full181']['attribution']['unique_correct'] is False

def test_fixed_failure_workflow_freezes_before_eval_and_review_pending(tmp_path,monkeypatch):
    calls=[]
    def child(command,**kw):
        case=command[command.index('--case-id')+1];stage=command[command.index('--stage')+1]
        if case in p.EVALUATION:
            path=Path(command[command.index('--threshold')+1]);assert path.exists()
            assert run.sha(path)==command[command.index('--threshold-sha')+1]
            assert json.loads(path.read_text())['calibration_status']=='UNCALIBRATED'
        calls.append((case,stage));return SimpleNamespace(returncode=1)
    monkeypatch.setattr(run.subprocess,'run',child)
    r=run.run(tmp_path/'all')
    assert len(calls)==22 and calls[:18]==[(c,s) for c in p.CALIBRATION for s in ('generate','media')]
    assert r['source_denominator']==15 and r['view_denominator']==105 and r['receiver_encode_denominator']==420
    assert r['fixed_calls']==dict(transformer=1100,scheduler_step=554,vae_decode=15,mp4_save=60,crop_save=45,vae_encode=420,blind_read=105)
    assert sum(len(view['observations']) for split in ('calibration','evaluation') for c in r[split].values() for source in c['sources'].values() for view in source['views'].values())==420
    assert r['human_review']==dict(denominator=24,available=0,rating_status='PENDING_HUMAN_REVIEW')

def test_review_retains_valid_media_despite_receiver_failure(tmp_path):
    path=tmp_path/'eval01'/'videos'/'LAST_A'/'full181.mp4';path.parent.mkdir(parents=True);path.write_bytes(b'previously validated readback')
    evaluation={'eval01':{'sources':{'LAST_A':{'views':{'full181':dict(status='PARTIAL_OR_FAILED',media_readback_valid=True,received_path='videos/LAST_A/full181.mp4')}}}}}
    r=run.review_package(tmp_path,evaluation)
    assert r['available']==1 and r['denominator']==24
    mapping=json.loads((tmp_path/'reporting_only_review_mapping.json').read_text());name=next(v['anonymous_file'] for v in mapping if v['status']=='AVAILABLE')
    assert (tmp_path/'blind_review'/name).read_bytes()==path.read_bytes()
