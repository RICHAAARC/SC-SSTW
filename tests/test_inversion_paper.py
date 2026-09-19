"""Synthetic/local preparation checks only; no real data or pretrained models."""
import copy
from types import SimpleNamespace
import pytest
import torch
from main.tube_state import inversion_state as state
from experiments.wan_state_clock import inversion_paper_protocol as p,inversion_paper_run as run
from test_video_inversion import Model,scheduler

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def calibration(value=2.):
    result={c:run.empty(c) for c in p.CAL}
    for case in result.values():
        for row in case['sources']['OFF']['views'].values():
            row.update(status='COMPLETE',readout={'channels':{k:{'score':value} for k in p.CHANNELS}})
    return result


def test_frozen_roster_and_static_same_support():
    manifest=run.load(run.MANIFEST)
    old=run.load(run.MANIFEST.with_name('inversion_state.json'))['development']+run.load(run.MANIFEST.with_name('inversion_state_holdout.json'))['holdout']
    fresh=manifest['calibration']+manifest['evaluation']
    assert len({c['prompt'] for c in fresh})==4 and len({c['seed'] for c in fresh})==4
    assert not {c['prompt'] for c in fresh}&{c['prompt'] for c in old}
    assert not {c['seed'] for c in fresh}&{c['seed'] for c in old}
    books=p.books(b'paper-unit');a,b=books['STATE'],books['STATIC']
    assert torch.equal(a['order'],b['order']) and torch.equal(a['pads'],b['pads'])
    assert torch.equal(b['states'][0],a['states'][0,0].repeat(11,1))
    assert torch.equal(b['states'][1],-b['states'][0]) and not b['steps'].any()
    z=torch.randn(state.SHAPE,generator=torch.Generator().manual_seed(39))
    for book in books.values():
        for m in (0,1):
            marked=state.write(z,book,m)
            assert torch.equal(z.abs(),marked.abs()) and torch.equal(z[:,1:],marked[:,1:])
            assert torch.equal(z[:,:,0],marked[:,:,0]) and torch.equal(z[:,:,45],marked[:,:,45])


def test_cluster_max_strict_threshold_missing_and_partial_fp():
    cal=calibration();cal[p.CAL[1]]['sources']['OFF']['views']['crop17']['readout']['channels']['STATE/search14']['score']=3.
    frozen=p.freeze(cal);assert frozen['channels']['STATE/search14']['threshold']==3
    scored={'channels':{k:dict(score=3.,selected_message=0) for k in p.CHANNELS}}
    assert not p.decide(scored,frozen)['STATE/search14']['accepted']
    cal[p.CAL[0]]['sources']['OFF']['views']['full181']['status']='FAILED'
    assert all(r['status']=='UNCALIBRATED' for r in p.freeze(cal)['channels'].values())
    evaluation={c:run.empty(c) for c in p.EVAL}
    row=evaluation[p.EVAL[0]]['sources']['OFF']['views']['full181']
    row['decisions']={k:dict(status='DECIDED',accepted=True,selected_message=0) for k in p.CHANNELS}
    s=p.summary(evaluation)['channels']['STATE/search14']
    assert s['off_complete']==0 and s['off_any_observed_false_accept']==1
    assert s['off_partially_observed_sources']==1 and s['off_no_decided_views']==1


def test_freeze_before_eval_launch_even_when_cal_missing(tmp_path,monkeypatch):
    calls=[]
    def child(command,**kwargs):
        case=command[command.index('--case-id')+1];calls.append(case)
        if case in p.EVAL:
            path=command[command.index('--threshold')+1];digest=command[command.index('--threshold-sha')+1]
            assert run.sha(path)==digest and run.load(path)['status']=='FROZEN'
            assert all(v['status']=='UNCALIBRATED' for v in run.load(path)['channels'].values())
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(run.subprocess,'run',child)
    result=run.run(tmp_path/'out')
    assert calls==list(p.CAL+p.EVAL) and result['source_denominator']==12 and result['view_denominator']==36
    assert sum(len(s['views']) for split in ('calibration','evaluation') for c in result[split].values() for s in c['sources'].values())==36
    assert result['fixed_calls']==run.plan(12)
    with pytest.raises(ValueError,match='threshold hash'):run.run_case(p.EVAL[0],tmp_path/'forbidden')


def test_synthetic_eval_case_full_crop_chain_and_budget(tmp_path,monkeypatch):
    frozen=p.freeze(calibration());threshold=tmp_path/'threshold.json';run.dump(threshold,frozen);digest=run.sha(threshold)
    initial=torch.randn(state.SHAPE,generator=torch.Generator().manual_seed(79));stored={};config_calls=[]
    def prepare(config,**kwargs):
        config_calls.append((config['generation']['frames'],config['generation']['seed']))
        return SimpleNamespace(transformer=Model(),scheduler=scheduler()),initial,torch.tensor(1.),torch.tensor(-1.),torch.float32
    def forward(*args,**kwargs):
        for i in range(50):
            for _ in range(2):args[9]('transformer',False);args[9]('transformer',True)
            args[9]('scheduler_step',False);args[9]('scheduler_step',True)
        return args[2]
    def save(rgb,path,*args):path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake media');stored[path]=rgb.clone()
    def read(path):return stored[path].clone()
    def encode(vae,rgb):return torch.zeros((1,16,46 if len(rgb)==181 else 33,40,64))
    def invert(transformer,z,prompt,negative,dtype,cfg,sigmas,times,count,record):
        for i in range(50):
            for _ in range(2):count('transformer',False);count('transformer',True)
            count('inverse_update',False);count('inverse_update',True);record({'index':i})
        return z
    monkeypatch.setattr(run,'RGB',(2,2,3));monkeypatch.setattr(run,'prepare_generation',prepare)
    monkeypatch.setattr(run,'continue_steps',forward);monkeypatch.setattr(run,'load_frozen_vae',lambda *a:Model())
    monkeypatch.setattr(run,'decode_normalized_latent',lambda *a:torch.zeros((181,2,2,3)))
    monkeypatch.setattr(run,'encode_rgb',save);monkeypatch.setattr(run,'encode_lossless',save);monkeypatch.setattr(run,'read_mp4',read)
    monkeypatch.setattr(run,'reencode_rgb24_readback',encode);monkeypatch.setattr(run,'invert_received_latent',invert)
    monkeypatch.setattr(run,'_clear_cache',lambda *a:None)
    result=run.run_case(p.EVAL[0],tmp_path/'case',threshold,digest)
    assert result['status']=='EXECUTION_COMPLETE' and config_calls==[(181,20261201),(181,0),(129,0)]
    assert run.sha(threshold)==digest
    for key,value in run.plan(5).items():assert result['actual_calls'][key+'_completed']==value
    assert sum(len(s['views']) for s in result['sources'].values())==15
    assert all(v['decisions']['STATE/search14']['status']=='DECIDED' for s in result['sources'].values() for v in s['views'].values())


def test_nonfinite_decoded_rgb_never_quantized_to_video(tmp_path,monkeypatch):
    output=tmp_path/'out';(output/'writer').mkdir(parents=True)
    torch.save(torch.zeros(state.SHAPE),output/'writer/OFF_terminal.pt')
    result=run.empty(p.CAL[0]);result['sources']['OFF']['generation_status']='COMPLETE'
    failures=[];saved=[]
    monkeypatch.setattr(run,'RGB',(2,2,3));monkeypatch.setattr(run,'load_frozen_vae',lambda *a:Model())
    monkeypatch.setattr(run,'decode_normalized_latent',lambda *a:torch.full((181,2,2,3),float('nan')))
    monkeypatch.setattr(run,'encode_rgb',lambda *a:saved.append(a));monkeypatch.setattr(run,'_clear_cache',lambda *a:None)
    run.media_case({},output,result,lambda *a:None,lambda:None,lambda stage,exc:failures.append(str(exc)))
    assert not saved and any('nonfinite' in error for error in failures)
    assert all(v['media_status']!='COMPLETE' for v in result['sources']['OFF']['views'].values())
