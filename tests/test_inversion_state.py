"""CPU state mapping/observer and fake actual-MP4 dependency checks."""
import copy
import inspect
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import inversion_state as method,state_clock
from experiments.wan_state_clock import inversion_state_run as runner
from experiments.wan_state_clock import video_inversion_run as base
from test_video_inversion import Model,scheduler

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def test_state_mapping_roundtrip_boundary_and_channels():
    key=b'state-test';book=method.codebook(key)
    z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(13))
    for message in (0,1):
        marked=method.write(z,book,message)
        assert torch.equal(marked[:,1:],z[:,1:])
        assert torch.equal(marked[:,:,0],z[:,:,0]) and torch.equal(marked[:,:,45],z[:,:,45])
        assert torch.equal(marked.abs(),z.abs())
        assert np.array_equal(book['states'][message],state_clock.trajectory(key,message)[0])
        decoded=method.read(marked,book)
        report=method.report(decoded,book,message)
        assert report['raw_state']['exact_windows']==11 and report['raw_state']['exact_trajectory']
        assert all(m['unique_correct'] for m in report['modes'].values())
        assert len(decoded['per_time'])==46 and len(decoded['boundaries'])==2
        before=copy.deepcopy(decoded);method.report(decoded,book,1-message)
        assert decoded==before
    assert 'truth' not in inspect.signature(method.read).parameters


def test_observer_exact_pure_reuse_ties_and_missing():
    book=method.codebook(b'observer-test');q=book['states'][0].numpy()
    ranking=method.rank(q,[True]*11,book)
    for name,update in [('without_update',False),('with_update',True)]:
        for m in (0,1):
            expected=state_clock.observe(q,[True]*11,book['states'][m].numpy(),book['steps'][m].numpy(),update=update)
            assert ranking[name]['candidates'][m]['observer']==expected
            assert ranking[name]['candidates'][m]['score']==-expected['innovation_mean']
    zero=method.rank(np.zeros((11,2)),[True]*11,book)
    assert all(r['selected_message'] is None and r['status']=='NO_VALID_OBSERVATIONS' for r in zero.values())
    assert zero['with_update']['candidates'][0]['observer']['innovation_by_window']==[1.]*11
    same=copy.deepcopy(book);same['states'][1]=same['states'][0];same['steps'][1]=same['steps'][0]
    ties=method.rank(q,[True]*11,same)
    assert all(r['status']=='TIE' and r['selected_message'] is None for r in ties.values())


def test_invalid_tiny_phase_not_counted_as_recovered():
    book=method.codebook(b'tiny-test')
    q=book['states'][0].numpy().copy();q[3]*=1e-13
    core=[method.observation(v) for v in q]
    decoded=dict(core=core,rankings=method.rank(q,[r['valid'] for r in core],book))
    report=method.report(decoded,book,0)
    assert core[3]['signs']==book['states'][0,3].int().tolist() and not core[3]['valid']
    assert report['raw_state']['exact_windows']==10 and report['raw_state']['component_errors']==2
    assert report['raw_state']['invalid_windows']==1 and not report['raw_state']['exact_trajectory']
    assert all(r['valid_windows']==10 for r in decoded['rankings'].values())


@pytest.mark.parametrize('use_holdout',[False,True])
def test_fake_saved_mp4_receiver_and_fixed_budget(tmp_path,monkeypatch,use_holdout):
    from experiments.wan_state_clock import inversion_state_holdout_run
    active=inversion_state_holdout_run if use_holdout else runner
    initial=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(5))
    stored={};decoded_latents=[];loaded=[];inverse_inputs=[];readback=[]
    def prepare(config,**kwargs):
        loaded.append(config['generation']['seed'])
        return SimpleNamespace(transformer=Model(),scheduler=scheduler()),initial,torch.tensor(1.),torch.tensor(-1.),torch.float32
    def forward(*args,**kwargs):
        count=args[9]
        for _ in range(100):count('transformer',False);count('transformer',True)
        for _ in range(50):count('scheduler_step',False);count('scheduler_step',True)
        return args[2]
    def decode(vae,z):
        decoded_latents.append(z.clone());return torch.full((1,1,1,3),float(len(decoded_latents)))
    def encode(rgb,path,fps,crf):
        path.parent.mkdir(exist_ok=True);path.write_bytes(b'fake persisted media')
        stored[path]=rgb+10 # readback observably distinct from decoded pixels
    def read(path):assert path.exists();return stored[path]
    def reencode(vae,rgb):
        readback.append(float(rgb.mean()));i=int(float(rgb.mean()))-11
        return decoded_latents[i]+.01
    def inverse(transformer,received,prompt,negative,dtype,cfg,sigmas,timesteps,count,record):
        inverse_inputs.append(received.clone())
        for _ in range(100):count('transformer',False);count('transformer',True)
        for i in range(50):count('inverse_update',False);count('inverse_update',True);record({'index':i})
        return received
    monkeypatch.setattr(base,'prepare_generation',prepare);monkeypatch.setattr(base,'continue_steps',forward)
    monkeypatch.setattr(base,'load_frozen_vae',lambda *a:Model());monkeypatch.setattr(base,'RGB_SHAPE',(1,1,1,3))
    monkeypatch.setattr(base,'decode_normalized_latent',decode);monkeypatch.setattr(base,'encode_rgb',encode)
    monkeypatch.setattr(base,'read_mp4',read);monkeypatch.setattr(base,'reencode_rgb24_readback',reencode)
    monkeypatch.setattr(base,'invert_received_latent',inverse);monkeypatch.setattr(base,'quality',lambda *a:{})
    monkeypatch.setattr(base,'_clear_cache',lambda *a:None)
    result=active.run_case(active.CASES[0],tmp_path/'case')
    assert result['status']=='EXECUTION_COMPLETE' and loaded==[20261001 if use_holdout else 20260916,0]
    assert readback==[11.,12.,13.] and len(inverse_inputs)==3
    for inp,original in zip(inverse_inputs,decoded_latents):torch.testing.assert_close(inp,original+.01)
    for stage,counts in runner.PLAN.items():
        for name,total in counts.items():assert result['actual_calls'][stage][name+'_completed']==total
    assert all(len(v['decoded']['core'])==11 for v in result['videos'].values())
    assert result['videos']['OFF']['posthoc']['truth'] is None
    assert 'state observations' in result['claim']


def test_missing_fixed_roster(tmp_path,monkeypatch):
    monkeypatch.setattr(runner.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    result=runner.run_all(tmp_path/'run')
    assert result['video_denominator']==12 and result['state_summary']['core_window_denominator']==88
    assert result['state_summary']['unmeasured_marked_videos']==8
    assert all(len(v['decoded']['core'])==11 and len(v['decoded']['boundaries'])==2 and len(v['decoded']['per_time'])==46
        for c in result['cases'].values() for v in c['videos'].values())
