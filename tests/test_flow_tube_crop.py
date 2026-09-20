"""CPU/synthetic crop geometry, blind-read contract and persisted RGB chain. No real video scoring."""
import copy,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch
from main.tube_state import state_clock,projection_margin as carrier
from experiments.wan_state_clock import flow_tube_crop_analysis as analysis,flow_tube_crop_run as run
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

@pytest.fixture(scope='module')
def book():return state_clock.codebook(b'crop-synthetic-key')

@pytest.fixture
def detection_factory(monkeypatch,book):
    original=state_clock.read
    def emissions(observations,book,n,g,num,den,b):
        z=observations.get(g);selected=[None]*4 if z is None else carrier.allocation(z.shape[2]-1,g,num,den,b)[4*n:4*n+4]
        valid=all(v is not None for v in selected)
        return dict(selected=selected,valid=valid,q=book['states'][0,n].tolist() if valid else [0.,0.],scores=[.6,.1] if valid else [0.,0.],agreement=[1.,0.] if valid else [None,None])
    monkeypatch.setattr(state_clock,'emissions',emissions);cache={}
    def build(origins=(0,1,2,3)):
        key=tuple(origins)
        if key not in cache:
            obs={g:np.zeros((1,16,33 if g==0 else 32,1,1),dtype=np.float32) for g in origins}
            cache[key]=original(obs,book)
        return copy.deepcopy(cache[key])
    return build


def test_fixed_geometry_positive_offsets_partial_and_missing():
    for start,g,windows in [(0,0,list(range(8))),(4,0,list(range(1,8))),(5,3,list(range(1,8)))]:
        ref=analysis.reference_geometry(start)
        assert ref['path']['g']==g and ref['path']['offset']==start
        assert ref['structural_complete_windows']==windows and ref['structural_supports']==160*len(windows)
        for n in windows:
            for k,j in enumerate(ref['selected'][n]):assert start+g+2.5+4*j==2.5+4*(4*n+k)
    assert run.PLAN['transformer']==run.PLAN['vae_decode']==run.PLAN['mp4_save']==0
    run.validate_manifest(run.load(run.MANIFEST))


def test_no_search_same_scores_and_missing_window_penalty(detection_factory):
    d=detection_factory();r=analysis.no_search_rankings(d);fixed=analysis.path(d,analysis.NO_SEARCH);detail=analysis.class_detail(d,fixed['class'])
    assert sum(detail['valid'])==8
    for name,field in analysis.NO_SEARCH_MODES.items():
        for score in detail['scores']:
            assert r[name]['best_by_message'][str(score['message'])]['score']==score[field]
    correct=detail['scores'][0];assert correct['matched_score']==pytest.approx(.6*8/11)
    assert correct['observer']['innovation_by_window'][-3:]==[1.,1.,1.]
    assert correct['state_score']==pytest.approx(correct['matched_score']-.05*correct['observer']['innovation_mean'])


def test_reference_ties_classes_and_no_missing_credit(detection_factory):
    d=detection_factory();ref=analysis.reference_geometry(5);p=analysis.path(d,ref['path']);detail=analysis.class_detail(d,p['class']);best=p|{'message':0,'score':1.,'matched_supports':1120}
    ranking={'best':best,'top_ties':[best,dict(best),dict(best)]}
    row=analysis.crop_alignment_reporting_only(d,ranking,5)
    assert row['matched_window_fraction']==1. and row['evaluable_window_denominator']==7
    assert row['top_tie_count']==3 and row['top_tie_unique_observation_classes']==1 and row['top_ties_contain_reference_class']
    missing=detection_factory(origins=());r=analysis.crop_alignment_reporting_only(missing,{'best':None,'top_ties':[]},5)
    assert r['evaluable_window_denominator']==0 and r['matched_window_fraction'] is None and r['matched_reference_windows']==[]


def test_persisted_uint8_crop_readback_and_hash(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'CROP_SHAPE',(129,2,2,3));raw=np.arange(181,dtype=np.uint8)[:,None,None,None]*np.ones((181,2,2,3),dtype=np.uint8)
    p=tmp_path/'clip.npy';np.save(p,raw[5:134]);rgb=run.load_received_crop(p,run.sha(p))
    assert rgb.shape==(129,2,2,3) and float(rgb[0,0,0,0])==pytest.approx(5/255) and float(rgb[-1,0,0,0])==pytest.approx(133/255)
    with pytest.raises(ValueError,match='hash'):run.load_received_crop(p,'wrong')
    np.save(p,raw[5:134].astype(np.float32))
    with pytest.raises(ValueError,match='uint8'):run.load_received_crop(p,run.sha(p))


def test_fake_full_case_blind_api_and_real_rgb_origins(tmp_path,monkeypatch,book,detection_factory):
    source=tmp_path/run.SOURCE_RUN;case=run.CASES[0];folder=source/case;folder.mkdir(parents=True);(folder/'received_videos').mkdir()
    config={'key_utf8':'crop-synthetic-key'};(folder/'config.json').write_text(json.dumps(config));np.savez(folder/'codebook.npz',**book)
    hashes={n:run.sha(folder/n) for n in ('config.json','codebook.npz')}
    for arm in run.ARMS:
        p=folder/'received_videos'/(arm+'.mp4');p.write_bytes(b'synthetic video');hashes['received_videos/'+arm+'.mp4']=run.sha(p)
    (source/'result.json').write_text(json.dumps({'cases':{case:{'source_commit':run.SOURCE_COMMIT,'file_sha256':hashes}}}))
    monkeypatch.setattr(run,'SOURCE_SHAPE',(181,2,2,3));monkeypatch.setattr(run,'CROP_SHAPE',(129,2,2,3));monkeypatch.setattr(run,'validate_manifest',lambda m:None)
    class VAE:config=SimpleNamespace(_commit_hash='synthetic')
    monkeypatch.setattr(run,'load_frozen_vae',lambda c:VAE());monkeypatch.setattr(run,'_clear_cache',lambda *a:None)
    monkeypatch.setattr(run,'read_mp4',lambda p:(torch.arange(181).float()/255).reshape(181,1,1,1).expand(run.SOURCE_SHAPE).clone())
    inputs=[]
    def encode(vae,rgb):
        inputs.append((len(rgb),round(float(rgb[0,0,0,0])*255)))
        return torch.zeros((1,16,1+(len(rgb)-1)//4,40,64))
    monkeypatch.setattr(run,'reencode_rgb24_readback',encode)
    monkeypatch.setattr(run.torch,'save',lambda z,p:Path(p).write_bytes(b'synthetic latent'))
    blind=[]
    def read(obs,public_book):
        blind.append(sorted(obs));assert set(public_book)==set(book)
        return detection_factory(tuple(sorted(obs)))
    monkeypatch.setattr(run.state_clock,'read',read)
    result=run.run_case(case,tmp_path/'out',source)
    assert result['status']=='EXECUTION_COMPLETE',result['failures']
    expected=[(129 if g==0 else 125,start+g) for arm in run.ARMS for start in run.STARTS for g in range(4)]
    assert inputs==expected and blind==[[0,1,2,3]]*9
    assert all(result['actual_calls'][k+'_completed']==v for k,v in run.PLAN.items())
    assert len(result['fragments'])==9 and all(len(v['reporting_only'])==8 for v in result['fragments'].values())
    assert all(v['reporting_only']['global_matched']['best_correct_minus_other_reporting_only'] is None for k,v in result['fragments'].items() if k.startswith('OFF'))


def test_phase_failure_and_fixed_root_denominators(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'CROP_SHAPE',(129,2,2,3));rgb=torch.zeros(run.CROP_SHAPE);calls=[];failures=[]
    def encode(*a):raise RuntimeError('synthetic VAE failure')
    monkeypatch.setattr(run,'reencode_rgb24_readback',encode);monkeypatch.setattr(run,'_clear_cache',lambda *a:None)
    obs,rows=run.encode_origins(None,rgb,lambda *a:calls.append(a),lambda *a:None,lambda *a:failures.append(a))
    assert not obs and len(rows)==len(failures)==4 and calls==[('vae_encode',False)]*4
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    result=run.run_all(tmp_path/'root',tmp_path/run.SOURCE_RUN)
    assert result['fragment_denominator']==36 and result['receiver_encode_denominator']==144
    assert sum(len(v['observations']) for c in result['cases'].values() for v in c['fragments'].values())==144
    assert len(result['compact_summary'])==24 and all(v['missing_or_failed']==8 for v in result['compact_summary'])
    with pytest.raises(ValueError,match='independent'):run.validate_output(tmp_path/'source'/'out',tmp_path/'source')
