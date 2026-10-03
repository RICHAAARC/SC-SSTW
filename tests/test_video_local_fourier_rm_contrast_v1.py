"""Frozen-space mechanism and fake runtime verification; zero real model calls."""
import ast,copy,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_local_fourier_rm_lowband_v1_state as state
from main.tube_state import video_local_fourier_rm_lowband_v1_control as original
from main.tube_state import video_local_fourier_rm_contrast_v1_space as space
from main.tube_state import video_local_fourier_rm_contrast_v1_control as control
from experiments.wan_state_clock import video_local_fourier_rm_contrast_v1_run as run
pytestmark=pytest.mark.unit
SMALL=(2,2,4,3)
@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)

class FakeScheduler:
    """History-bearing CPU stub exercising only the native protocol interfaces."""
    def __init__(self):
        self.config=SimpleNamespace(prediction_type='flow_prediction',thresholding=False,lower_order_final=True);self.predict_x0=True;self.timesteps=torch.arange(50,0,-1);self.sigmas=torch.linspace(1,0,51);self.step_index=None;self.model_outputs=[None,None]
    def step(self,velocity,timestep,z,return_dict=False):
        index=0 if self.step_index is None else self.step_index;self.model_outputs=[self.model_outputs[-1],velocity.clone()];self.step_index=index+1
        return (z+(self.sigmas[index+1]-self.sigmas[index])*velocity,)

class FakeTransformer:
    def __call__(self,hidden_states,encoder_hidden_states,**kwargs):return (hidden_states*.01+encoder_hidden_states.mean(),)

def fake_media_runtime(monkeypatch,*,fail_encode=None,fail_load=False):
    from runtime.wan import generation
    observed=SimpleNamespace(decode=0,encodes=[],input_hashes=[])
    monkeypatch.setattr(run.media,'SHAPE',SMALL);monkeypatch.setattr(run.media,'RGB_BYTES',int(np.prod(SMALL)));monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    class FakeVAE(torch.nn.Module):
        def __init__(self):super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False);self.config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def decode(self,value,return_dict=False):observed.decode+=1;return (torch.linspace(-1.2,1.2,int(np.prod(SMALL))).reshape(1,3,*SMALL[:3])+float(value.mean())/100,)
        def encode(self,video):
            observed.encodes.append(video.clone())
            if len(observed.encodes)==fail_encode:raise RuntimeError('fake isolated encode failure')
            # Hand-constructed received fixture, independent of arm names or truth labels.
            value=torch.tensor(state.synthesize('watermark',dtype=np.float32))+float(video.mean())/1000
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))
    fake=FakeVAE()
    def load(cfg,device):
        if fail_load:raise RuntimeError('fake VAE unavailable')
        return fake
    monkeypatch.setattr(generation,'load_frozen_vae',load)
    def save_received(path,source,flip):
        output=source.clone();output.reshape(-1)[0]^=flip;Path(path).write_bytes(output.numpy().tobytes());return output,dict(status='SAVED',path=str(path),sha256=run.sha(path),bytes=int(np.prod(SMALL)))
    def raw(source,yuv_path,rgb_path,*,count,event):
        source_hash=hashlib.sha256(source.numpy().tobytes()).hexdigest();observed.input_hashes.append(('RAW420',source_hash));count('rgb_to_raw420',False);Path(yuv_path).write_bytes(b'fake YUV');count('rgb_to_raw420',True);event('yuv420',dict(status='SAVED',sha256=run.sha(yuv_path),path=str(yuv_path),input_raster_sha256=source_hash))
        count('raw420_to_rgb24',False);output,row=save_received(rgb_path,source,1);count('raw420_to_rgb24',True);event('rgb24',row);return output
    def mp4(raster_path,raster_sha,mp4_path,rgb_path,*,count,event):
        source=run.media.reopen_raster(raster_path,raster_sha);observed.input_hashes.append(('MP4',hashlib.sha256(source.numpy().tobytes()).hexdigest()));count('mp4_save',False);Path(mp4_path).write_bytes(b'fake MP4');count('mp4_save',True);event('mp4',dict(status='SAVED',path=str(mp4_path),sha256=run.sha(mp4_path),input_raster_sha256=raster_sha));count('mp4_probe',False);count('mp4_probe',True);event('probe',dict(status='COMPLETE'));count('mp4_readback',False);output,row=save_received(rgb_path,source,2);count('mp4_readback',True);event('rgb24',row);return output
    monkeypatch.setattr(run.raw420,'roundtrip',raw);monkeypatch.setattr(run.media,'mp4_roundtrip',mp4);return observed

def fake_generation_inputs(store):
    for arm,row in store.data['generation'].items():
        path=Path(row['terminal_path']);path.parent.mkdir(parents=True,exist_ok=True);torch.save(torch.zeros((1,16,46,40,64)),path);row.update(status='COMPLETE',sha256=run.sha(path),receipt=dict(writer_diagnostics=dict(last_z_post_matches_terminal=True)))
    store.save()

def test_public_masked_union_space_support_adjoint_span_and_relative_scores():
    for key in ('watermark','watermark-wrong'):
        f=space.build_space(key);U=f['basis'];mask=f['mask'];assert U.shape==(5632,214) and mask.shape==(44,4,4,8) and int(mask.sum())==5440
        assert np.count_nonzero(U[~mask.reshape(-1)])==0 and f['receipt']['orthogonality_max_abs']<1e-13
        np.testing.assert_allclose(U.T@U,np.eye(214),atol=1e-13)
        rng=np.random.default_rng(13);x=rng.normal(size=space.SHAPE)*mask;y=rng.normal(size=space.SHAPE)*mask;px=space.project(x,key);py=space.project(y,key)
        np.testing.assert_allclose(space.project(px,key),px,atol=5e-14);assert np.vdot(px,y)==pytest.approx(np.vdot(x,py),abs=2e-12)
        mu=f['templates'];difference=mu-mu[0];generators=np.concatenate([(difference*mask).reshape(174,-1),(np.asarray([space.laplacian(v) for v in difference])*mask).reshape(174,-1)],axis=0).T
        np.testing.assert_allclose(U@(U.T@generators),generators,atol=2e-14)
        for mode in run.MODES:np.testing.assert_allclose(space.relative_effects(x,key)[mode],space.relative_effects(px,key)[mode],atol=2e-16)
        for q,expected in ((x,px),(y,py)):
            for transformed,n in ((lambda z:z,5632),(lambda z:np.diff(z,axis=0),5504)):
                costs=np.asarray([np.square(transformed(q-v)).sum()/n for v in mu]);projected=np.asarray([np.square(transformed(expected-v)).sum()/n for v in mu]);np.testing.assert_allclose(costs-costs[0],projected-projected[0],atol=2e-15)
        assert f['receipt']['recipe']['anchor_rule'].startswith('first public') and f['receipt']['generator_span_residual_l2']<1e-12
        f['basis'][:]=999;assert np.max(np.abs(space.build_space(key)['basis']))<1


def test_contrast_raw_row45_cap_support_and_actual_direction_records():
    latent=torch.zeros((1,16,46,40,64));t=control.build_targets(latent,'watermark',control.message_bits('OKOK'),'STATE_CONTRAST_MULTI')
    clean=torch.randn(latent.shape,generator=torch.Generator().manual_seed(15))/10;leaf=clean.clone().requires_grad_(True);features=[]
    for b,(h0,h1,w0,w1) in enumerate(t['blocks']):features.append((leaf[0,4,1:,h0:h1,w0:w1].reshape(45,256)@t['basis'][b]).reshape(45,4,8))
    q=torch.stack(features,dim=1);loss=(q[t['pilot_active']]-t['pilot_target'][t['pilot_active']]).square().mean();raw=(-696.*torch.autograd.grad(loss,leaf)[0]).detach();restricted,rq,pq=control.restrict_raw(raw,t)
    assert torch.equal(raw[0,4,45],restricted[0,4,45]);assert torch.count_nonzero(pq[~t['contrast_mask']])==0
    outside=restricted.clone()
    for h0,h1,w0,w1 in t['blocks']:outside[0,4,1:,h0:h1,w0:w1]=0
    assert torch.count_nonzero(outside)==0
    for clean in (latent,clean*100):
        capture={};delta,info=control.pilot_delta(clean,t,diagnostics=capture);assert info['cap_scale']<=1 and info['actual_delta_l2']<=1+2e-7 and info['row45_precap_exact']
        assert info['row45_actual_spatial_l2']==pytest.approx(info['row45_raw_spatial_l2']*info['cap_scale'],abs=2e-7)
        if clean is latent:assert info['restricted_raw_delta_l2']<1 and info['cap_scale']==1 and info['actual_delta_l2']==pytest.approx(info['restricted_raw_delta_l2'],abs=1e-12)
        actual=control.project_tensor(delta,t['basis'],t['blocks'])[:44];assert torch.equal(capture['direction_actual'],actual)
        record=control.direction_diagnostic(capture,'watermark');assert record['candidate_effect_values']==1044 and record['truth_inputs'] is False and len(record['valid_catalog_indices'])==174
        for direction in record['effects'].values():assert set(direction)==set(run.MODES) and all(len(v)==174 for v in direction.values())


def test_original_three_arm_actual_velocity_exact():
    z=torch.randn((1,16,46,40,64),generator=torch.Generator().manual_seed(71))/10;c=z/100+.03;u=z/100-.02
    for arm in ('OFF','PAYLOAD_MULTI','STATE_LOWBAND_MULTI'):
        t=control.build_targets(z,'watermark',control.message_bits('OKOK'),arm);ot=original.build_targets(z,'watermark',original.message_bits('OKOK'),arm)
        for index in (24,25,49):
            actual,detail=control.guided_velocity(z,c,u,.8,t,arm,index);expected,oldrow=original.guided_velocity(z,c,u,.8,ot,arm,index)
            assert torch.equal(actual,expected) and detail==oldrow


def test_complete_native_four_arm_main_and_independent_diagnostics(tmp_path,monkeypatch):
    from runtime.wan import generation
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True);pipe=SimpleNamespace(transformer=FakeTransformer(),scheduler=FakeScheduler());initial=torch.zeros((1,16,46,40,64));before=run.trajectory.fingerprint(vars(pipe.scheduler))
    monkeypatch.setattr(run.backend,'execution_device_dtype',lambda:('cpu',torch.float32))
    def prepare(*args,**kwargs):assert store.data['space_recipe']['status']=='SAVED' and kwargs['load_vae'] is False;return pipe,initial,torch.tensor([.03]),torch.tensor([-.02]),torch.float32
    monkeypatch.setattr(generation,'prepare_generation',prepare);run.generation_worker(store,cfg)
    assert store.data['before_step25_identity']['status']=='MATCH' and run.trajectory.fingerprint(vars(pipe.scheduler))==before and torch.count_nonzero(initial)==0
    assert store.data['counts']['generation_steps']==200 and store.data['counts']['controlled_steps']==75 and store.data['counts']['writer_sidecars']==100 and store.data['direction_counts']==dict(records=50,candidate_effect_values=52200)
    assert store.data['direction_calls']==dict(direction_diagnostic_compute=dict(attempted=50,completed=50))
    assert all(row['receipt']['writer_diagnostics']['last_z_post_matches_terminal'] for row in store.data['generation'].values())
    assert all(row['source'].startswith('LOW control unchanged') for name,row in store.data['direction_diagnostics'].items() if name.startswith('STATE_LOWBAND_MULTI/'))
    observed=fake_media_runtime(monkeypatch);run.media_worker(store,cfg);assert observed.decode==4 and len(observed.encodes)==12
    primary_paths=[store.output/'receiver_readouts.json',store.output/'payload_readouts.json'];sealed={str(p):p.read_bytes() for p in primary_paths}
    run.terminal_diagnostics(store,cfg);t=store.data['terminal_diagnostics'];assert t['counts']['keyed_projections']==16 and t['counts']['path_score_records']==32 and t['counts']['path_costs']==5568
    assert t['calls']=={k:dict(attempted=n,completed=n) for k,n in cfg['planned_terminal_diagnostic_calls'].items()}
    terminal_path=store.output/'terminal_diagnostic_readouts.json';terminal_bytes=terminal_path.read_bytes()
    assert sealed=={str(p):p.read_bytes() for p in primary_paths}
    for p in [*primary_paths,terminal_path]:
        text=p.read_text();assert 'registered_tau' not in text and 'true_path_rank' not in text and 'bit_errors' not in text
    for p in primary_paths:
        text=p.read_text();assert 'terminal.pt' not in text and 'direction_diagnostics' not in text and 'space_recipe' not in text and 'before_step25' not in text
    store.data['workers']={p:dict(status='COMPLETE') for p in ('generation','media')};run.evaluate(store,cfg);run.quality_diagnostics(store);assert sealed=={str(p):p.read_bytes() for p in primary_paths} and terminal_path.read_bytes()==terminal_bytes and run.finish(store,cfg)
    assert store.data['counts']==dict(generated=4,generation_steps=200,controlled_steps=75,writer_sidecars=100,rasters=4,transport=12,normalized=12,keyed_projections=48,path_score_records=96,path_costs=16704,difference_edge_costs=272448,absolute_local_state_costs=95040,payload_reads=24,path_posthoc=96,difference_edge_posthoc=48,payload_evaluations=48,quality_pairs=18)
    assert t['counts']=={k:v for k,v in run.TERMINAL_FIXED.items() if k not in ('catalog_slots','structural_exclusions')}
    assert all(not row['accepted_payload'] and not row.get('state_path_accepted',False) for group in ('mode_reads','path_posthoc','payload_evaluations') for row in store.data[group].values())


def test_missing_terminal_retains_terminal_matrix_and_main_blind_isolation(tmp_path,monkeypatch):
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True);fake_generation_inputs(store);Path(store.data['generation']['OFF']['terminal_path']).unlink();observed=fake_media_runtime(monkeypatch);run.media_worker(store,cfg);run.terminal_diagnostics(store,cfg);snapshots={p:p.read_bytes() for p in (store.output/'receiver_readouts.json',store.output/'payload_readouts.json',store.output/'terminal_diagnostic_readouts.json')};run.evaluate(store,cfg);run.quality_diagnostics(store)
    assert observed.decode==3 and len(observed.encodes)==9 and not run.finish(store,cfg)
    t=store.data['terminal_diagnostics'];assert t['calls']['terminal_tensor_load']==dict(attempted=4,completed=3) and t['counts']['path_score_records']==24
    assert all(len(t[k])==n for k,n in run.TERMINAL_SIZES.items()) and all(r['status']=='MISSING_READ' for name,r in t['path_posthoc'].items() if name.startswith('OFF/'))
    assert all(p.read_bytes()==b for p,b in snapshots.items()) and len(store.data['direction_diagnostics'])==50 and len(store.data['path_posthoc'])==96
    assert store.data['normalized']['STATE_CONTRAST_MULTI/MP4']['status']=='SAVED'


def test_worker_failure_preserves_main_direction_and_terminal_denominators(tmp_path,monkeypatch):
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True)
    def fail(*args,**kwargs):raise OSError('fake child unavailable')
    monkeypatch.setattr(run.subprocess,'Popen',fail);store,halt=run.run_worker_phase(store,'generation');run.settle(store,'media','no model');run.terminal_diagnostics(store,cfg);run.evaluate(store,cfg);run.quality_diagnostics(store)
    assert not halt and not run.finish(store,cfg) and len(store.data['direction_diagnostics'])==50 and all(r['status']=='NOT_COMPLETED' for r in store.data['direction_diagnostics'].values())
    assert all(len(store.data[k])==n for k,n in run.SIZES.items()) and all(len(store.data['terminal_diagnostics'][k])==n for k,n in run.TERMINAL_SIZES.items())
    assert all(r['status']=='MISSING_READ' for r in store.data['terminal_diagnostics']['path_posthoc'].values()) and store.data['calls']=={} and store.data['terminal_diagnostics']['calls']['terminal_tensor_load']==dict(attempted=4,completed=0)


def test_architecture_and_public_space_signature():
    assert list(__import__('inspect').signature(space.build_space).parameters)==['key'] and run.load_config()['direction_control']['truth_selects_direction'] is False
    for relative in ('main/tube_state/video_local_fourier_rm_contrast_v1_space.py','main/tube_state/video_local_fourier_rm_contrast_v1_control.py','runtime/wan/video_local_fourier_rm_contrast_v1.py'):
        tree=ast.parse((run.ROOT/relative).read_text());imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)];assert not any(n and n.startswith('experiments') for n in imports)
        if relative.startswith('main/'):assert not any(n and n.startswith('runtime') for n in imports)
