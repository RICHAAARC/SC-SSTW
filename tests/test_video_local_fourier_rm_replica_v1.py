"""Independent replica mathematics and fake native pipeline; zero real model/media calls."""
import ast,copy,gzip,hashlib,inspect,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_local_fourier_rm_state as original_state
from main.tube_state import video_local_fourier_rm_control as original_control
from main.tube_state import video_local_fourier_rm_replica_v1_state as state
from main.tube_state import video_local_fourier_rm_replica_v1_control as control
from experiments.wan_state_clock import video_local_fourier_rm_difference_receiver as scorer
from experiments.wan_state_clock import video_local_fourier_rm_replica_v1_run as run
from scripts import build_video_local_fourier_rm_replica_v1_notebook as builder

pytestmark=pytest.mark.unit
SHAPE=(1,16,46,40,64)
SMALL=(2,2,4,3)
A=((8,16,12,20),(8,16,44,52),(28,36,12,20),(28,36,44,52))
B=((8,16,20,28),(8,16,52,60),(28,36,20,28),(28,36,52,60))

@pytest.fixture(autouse=True)
def cpu_threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)

def independent_basis(key):
    """Public keyed slots from frozen OLD8, numerical columns independently defined."""
    y,x=np.mgrid[:8,:8];out=[]
    for block in original_state.basis_layout(key)['blocks']:
        out.append(np.stack([s*f*np.cos(2*np.pi*(h*y+w*x)/8)/8
          for (h,w),s,f in zip(block['modes'],block['signs'],block['factors'])],axis=-1).reshape(64,32))
    return np.array(out)

def independent_group_fft(received,key):
    rows=original_state.basis_layout(key)['blocks'];q=[]
    for group in (A,B):
        group_rows=[]
        for (h0,h1,w0,w1),row in zip(group,rows):
            spectrum=np.fft.fft2(received[:,4,h0:h1,w0:w1],axes=(-2,-1),norm='ortho').real
            group_rows.append(np.stack([s*f*spectrum[:,h,w]
               for (h,w),s,f in zip(row['modes'],row['signs'],row['factors'])],axis=-1).reshape(len(received),4,8))
        q.append(np.stack(group_rows,axis=1))
    return np.stack(q)

def independently_synthesized_target(key):
    basis=independent_basis(key)
    q=original_state.composite_signs(key).astype(float)*original_state.PUBLIC.alpha/np.sqrt(2)
    out=np.zeros(SHAPE)
    for group in (A,B):
        for b,(h0,h1,w0,w1) in enumerate(group):
            out[0,4,1:,h0:h1,w0:w1]=(q[:,b].reshape(45,32)@basis[b].T).reshape(45,8,8)
    return out

def test_geometry_copied_basis_full_target_energy_and_independent_FFT():
    assert state.GROUPS==(A,B) and state.PUBLIC.alpha==original_state.PUBLIC.alpha
    support=np.zeros((40,64),int)
    for h0,h1,w0,w1 in A+B:
        assert 0<=h0<h1<=40 and 0<=w0<w1<=64 and (h1-h0,w1-w0)==(8,8)
        support[h0:h1,w0:w1]+=1
    assert support.max()==1 and np.count_nonzero(support)==512
    for key in ('watermark','watermark-wrong'):
        basis=independent_basis(key)
        np.testing.assert_allclose(state.bases(key),basis,atol=2e-15)
        np.testing.assert_allclose(basis.transpose(0,2,1)@basis,np.broadcast_to(np.eye(32),(4,32,32)),atol=3e-15)
        np.testing.assert_allclose(basis.sum(axis=1),0,atol=2e-14)
        expected=independently_synthesized_target(key);actual=state.synthesize(key)
        np.testing.assert_allclose(actual,expected,atol=2e-16)
        assert np.linalg.norm(actual)==pytest.approx(np.linalg.norm(original_state.synthesize(key)),abs=2e-14)
        assert np.count_nonzero(actual[0,:4])==0 and np.count_nonzero(actual[0,4,0])==0
        for group in (A,B):
            assert sum(np.square(actual[0,4,1:,h0:h1,w0:w1]).sum() for h0,h1,w0,w1 in group)==pytest.approx(.5,abs=2e-14)
        received=actual[0,:,1:].transpose(1,0,2,3)
        read=state.extract_groups(received,key,np.ones((2,45,4),bool))
        oracle=independent_group_fft(received,key)
        np.testing.assert_allclose(read['raw_groups'],oracle,atol=4e-16)
        target=original_state.composite_signs(key)*original_state.PUBLIC.alpha
        np.testing.assert_allclose(read['scaled_groups'],np.broadcast_to(target,(2,45,4,4,8)),atol=4e-16)
        np.testing.assert_allclose(read['q'],target,atol=4e-16)
        assert read['q'].shape==(45,4,4,8) and read['replicas_are_independent_votes'] is False

def test_same_window_equal_soft_fusion_coordinates_noise_free_paths_and_missing_support():
    target=original_state.composite_signs('watermark')[:44]*original_state.PUBLIC.alpha
    noise=np.random.default_rng(88).normal(scale=.03,size=target.shape)
    raw=np.stack([(target+noise)/np.sqrt(2),(target-noise)/np.sqrt(2)])
    masks=np.ones((2,44,4),bool)
    fused=state.fuse_groups(raw,masks)
    np.testing.assert_allclose(fused['q'],target,atol=2e-17)
    assert fused['q'].shape==(44,4,4,8) and fused['replicas_are_independent_votes'] is False
    assert fused['received_regular_indices'].tolist()==list(range(1,45))
    assert np.array_equal(fused['group_availability'],masks) and np.array_equal(fused['availability'],masks[0]&masks[1])
    for infer in (scorer.absolute_control,scorer.infer_difference):
        scored=infer(fused['q'],'watermark',fused['availability'])
        assert scored['summary']['canonical_catalog_index']==0 and scored['summary']['top_catalog_indices']==[0]
        assert scored['counts']['catalog']==3915 and scored['counts']['scored']==174
        assert not scored['summary']['accepted_payload'] and not scored['summary']['state_path_accepted']
    masks[0,9,2]=False;raw[0,9,2]=np.nan
    missing=state.fuse_groups(raw,masks)
    assert not missing['availability'][9,2] and np.count_nonzero(missing['q'][9,2])==0
    assert np.count_nonzero(missing['raw_groups'][0,9,2])==0 and np.isfinite(missing['scaled_groups']).all()
    assert np.count_nonzero(missing['raw_groups'][1,9,2])>0  # Kept, never used as fallback.
    absolute=scorer.absolute_control(missing['q'],'watermark',missing['availability'])
    difference=scorer.infer_difference(missing['q'],'watermark',missing['availability'])
    assert absolute['available_dimensions']==5632-32 and difference['available_dimensions']==5504-64
    absent=state.fuse_groups(np.full_like(raw,np.nan),np.zeros_like(masks))
    for infer in (scorer.absolute_control,scorer.infer_difference):
        scored=infer(absent['q'],'watermark',absent['availability'])
        assert scored['summary']['reason']=='NO_OBSERVATIONS' and scored['counts']['scored']==0

def test_physical_missing_support_and_wrong_key_do_not_supply_truth_to_receiver():
    target=independently_synthesized_target('watermark')
    received=target[0,:,1:45].transpose(1,0,2,3).copy()
    mask=np.ones((2,44,4),bool)
    good=state.extract_groups(received,'watermark',mask)
    bad=state.extract_groups(received,'watermark-wrong',mask)
    assert list(inspect.signature(state.fuse_groups).parameters)==['raw_groups','availability']
    assert list(inspect.signature(state.extract_groups).parameters)==['received_tensor','key','availability','public']
    assert not np.allclose(bad['q'],good['q'])
    for infer in (scorer.absolute_control,scorer.infer_difference):
        assert infer(good['q'],'watermark',good['availability'])['summary']['min_cost']<1e-25
        wrong=infer(bad['q'],'watermark-wrong',bad['availability'])
        assert wrong['summary']['min_cost']>1e-7 and wrong['summary']['canonical_catalog_index']!=0
    mask[1,7,1]=False
    h0,h1,w0,w1=B[1];received[7,4,h0:h1,w0:w1]=np.nan
    read=state.extract_groups(received,'watermark',mask)
    assert np.isfinite(read['q']).all() and np.count_nonzero(read['q'][7,1])==0
    mask[1,7,1]=True
    with pytest.raises(ValueError,match='nonfinite'):state.extract_groups(received,'watermark',mask)

def test_independent_joint_FFT_gradient_actual_norm_match_amplification_and_zero_cases():
    zero=torch.zeros(SHAPE)
    targets=control.build_targets(zero,'watermark',control.message_bits('OKOK'),'STATE_REPLICA_MULTI')
    for clean,requested in ((zero,.7),(torch.randn(SHAPE,generator=torch.Generator().manual_seed(45))*.2,.3)):
        leaf=clean.clone().requires_grad_(True);features=[]
        for group in (A,B):
            patches=[]
            for (h0,h1,w0,w1),row in zip(group,original_state.basis_layout('watermark')['blocks']):
                spectrum=torch.fft.fft2(leaf[0,4,1:,h0:h1,w0:w1],norm='ortho').real
                patches.append(torch.stack([s*f*spectrum[:,h,w] for (h,w),s,f in zip(row['modes'],row['signs'],row['factors'])],dim=-1).reshape(45,4,8))
            features.append(torch.stack(patches,dim=1))
        q=torch.stack(features)
        goal=torch.from_numpy(original_state.composite_signs('watermark').astype(np.float32))*np.float32(original_state.PUBLIC.alpha/np.sqrt(2))
        active=torch.from_numpy((original_state.composite_signs('watermark')!=0).copy()).unsqueeze(0).expand_as(q)
        residual=q-goal.unsqueeze(0)
        assert int(active.sum())==11136 and int((~active).sum())==384
        loss=residual[active].square().mean()
        raw=(-1392.*torch.autograd.grad(loss,leaf)[0]).detach();norm=float(raw.double().norm())
        expected=raw*(requested/norm)
        delta,receipt=control.pilot_delta(clean,targets,'STATE_REPLICA_MULTI',matched_norm=requested)
        torch.testing.assert_close(delta,expected,atol=3e-8,rtol=4e-6)
        assert receipt['eta']==1392 and receipt['active_coefficients']==11136 and receipt['boundary_coefficients']==384
        assert float(delta.double().norm())==pytest.approx(requested,abs=2e-7)
        assert receipt['actual_delta_l2']==pytest.approx(requested,abs=2e-7)
        assert receipt['norm_match_scale']==pytest.approx(requested/receipt['raw_delta_l2'])
        assert receipt['scale_is_shrink_only'] is False and receipt['cap'] is None
        if clean is zero:assert receipt['norm_match_scale']>1 and receipt['raw_delta_l2']==pytest.approx(.25,abs=1e-7)
        groups=control.project_groups(delta,targets['basis']);closure=control.delta_closure(delta,groups,targets)
        assert abs(closure['parseval_error'])<2e-7 and abs(closure['outside_ROI_energy'])<1e-12
        assert np.square(receipt['actual_group_l2']).sum()==pytest.approx(requested**2,abs=3e-7)
    # Unequal host loads must not be forcibly assigned half of the actual update.
    asymmetric=zero.clone();asymmetric[0,4,1:,8:16,52:60]=torch.from_numpy(independent_basis('watermark')[1,:,0].reshape(8,8).astype(np.float32))*4
    _,info=control.pilot_delta(asymmetric,targets,'STATE_REPLICA_MULTI',matched_norm=.5)
    assert info['actual_group_l2'][0]!=pytest.approx(info['actual_group_l2'][1],abs=1e-5)
    delta,info=control.pilot_delta(zero,targets,'STATE_REPLICA_MULTI',matched_norm=0)
    assert torch.count_nonzero(delta)==0 and info['norm_match_scale']==0
    empty=dict(targets);empty['pilot_target']=torch.zeros_like(targets['pilot_target'])
    empty_delta,empty_info=control.pilot_delta(zero,empty,'STATE_REPLICA_MULTI',matched_norm=0)
    assert torch.count_nonzero(empty_delta)==0 and empty_info['raw_delta_l2']==0 and empty_info['norm_match_scale']==0
    with pytest.raises(ValueError,match='zero replica raw direction'):control.pilot_delta(zero,empty,'STATE_REPLICA_MULTI',matched_norm=.5)
    for value in (None,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):control.pilot_delta(zero,targets,'STATE_REPLICA_MULTI',matched_norm=value)

def test_original_three_arm_actual_velocity_exact_and_schedule_writer_only():
    z=torch.randn(SHAPE,generator=torch.Generator().manual_seed(71))/10;c=z/100+.03;u=z/100-.02
    for arm,oldarm in (('OFF','OFF'),('PAYLOAD_MULTI','PAYLOAD_MULTI'),('STATE_OLD_MULTI','STATE_MULTI')):
        t=control.build_targets(z,'watermark',control.message_bits('OKOK'),arm)
        ot=original_control.build_targets(z,'watermark',original_control.message_bits('OKOK'))
        for index in (24,25,49):
            actual,row=control.guided_velocity(z,c,u,.8,t,arm,index)
            old,oldrow=original_control.guided_velocity(z,c,u,.8,ot,oldarm,index)
            assert torch.equal(actual,old)
            assert row.get('payload')==oldrow.get('payload') and row.get('pilot')==oldrow.get('pilot')
    assert 'matched_norm' not in inspect.signature(state.extract_groups).parameters
    assert list(inspect.signature(scorer.absolute_control).parameters)==['q','key','availability']
    assert list(inspect.signature(scorer.infer_difference).parameters)==['q','key','availability']

class FakeScheduler:
    """History-bearing CPU stub: native API only, no real scheduler/model."""
    def __init__(self):
        self.config=SimpleNamespace(prediction_type='flow_prediction',thresholding=False,lower_order_final=True)
        self.predict_x0=True;self.timesteps=torch.arange(50,0,-1);self.sigmas=torch.linspace(1,0,51)
        self.step_index=None;self.model_outputs=[None,None]
    def step(self,velocity,timestep,z,return_dict=False):
        index=0 if self.step_index is None else self.step_index
        self.model_outputs=[self.model_outputs[-1],velocity.clone()];self.step_index=index+1
        return (z+(self.sigmas[index+1]-self.sigmas[index])*velocity,)

class FakeTransformer:
    def __call__(self,hidden_states,encoder_hidden_states,**kwargs):
        return (hidden_states*.01+encoder_hidden_states.mean(),)

def fake_media_runtime(monkeypatch,*,fail_encode=None):
    from runtime.wan import generation
    observed=SimpleNamespace(decode=0,encodes=[],input_hashes=[])
    monkeypatch.setattr(run.media,'SHAPE',SMALL)
    monkeypatch.setattr(run.media,'RGB_BYTES',int(np.prod(SMALL)))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
            self.config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def decode(self,value,return_dict=False):
            observed.decode+=1
            return (torch.linspace(-1.2,1.2,int(np.prod(SMALL))).reshape(1,3,*SMALL[:3])+float(value.mean())/100,)
        def encode(self,video):
            observed.encodes.append(video.clone())
            if len(observed.encodes)==fail_encode:raise RuntimeError('fake isolated encode failure')
            # Fixture has no arm label or truth argument; media adapter alone feeds it.
            value=torch.tensor(independently_synthesized_target('watermark'),dtype=torch.float32)+float(video.mean())/1000
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:value))
    fake=FakeVAE()
    monkeypatch.setattr(generation,'load_frozen_vae',lambda cfg,device:fake)
    def save_received(path,source,flip):
        output=source.clone();output.reshape(-1)[0]^=flip
        Path(path).write_bytes(output.numpy().tobytes())
        return output,dict(status='SAVED',path=str(path),sha256=run.sha(path),bytes=int(np.prod(SMALL)))
    def raw(source,yuv_path,rgb_path,*,count,event):
        sh=hashlib.sha256(source.numpy().tobytes()).hexdigest();observed.input_hashes.append(('RAW420',sh))
        count('rgb_to_raw420',False);Path(yuv_path).write_bytes(b'fake YUV');count('rgb_to_raw420',True)
        event('yuv420',dict(status='SAVED',sha256=run.sha(yuv_path),path=str(yuv_path),input_raster_sha256=sh))
        count('raw420_to_rgb24',False);out,row=save_received(rgb_path,source,1);count('raw420_to_rgb24',True);event('rgb24',row);return out
    def mp4(raster_path,raster_sha,mp4_path,rgb_path,*,count,event):
        source=run.media.reopen_raster(raster_path,raster_sha)
        observed.input_hashes.append(('MP4',hashlib.sha256(source.numpy().tobytes()).hexdigest()))
        count('mp4_save',False);Path(mp4_path).write_bytes(b'fake MP4');count('mp4_save',True)
        event('mp4',dict(status='SAVED',path=str(mp4_path),sha256=run.sha(mp4_path),input_raster_sha256=raster_sha))
        count('mp4_probe',False);count('mp4_probe',True);event('probe',dict(status='COMPLETE'))
        count('mp4_readback',False);out,row=save_received(rgb_path,source,2);count('mp4_readback',True);event('rgb24',row);return out
    monkeypatch.setattr(run.raw420,'roundtrip',raw);monkeypatch.setattr(run.media,'mp4_roundtrip',mp4)
    return observed

def fake_generation_inputs(store):
    for arm,row in store.data['generation'].items():
        path=Path(row['terminal_path']);path.parent.mkdir(parents=True,exist_ok=True)
        torch.save(torch.zeros(SHAPE),path)
        row.update(status='COMPLETE',sha256=run.sha(path),receipt=dict(writer_diagnostics=dict(last_z_post_matches_terminal=True)))
    store.save()

def assert_fixed_rosters(store):
    assert all(len(store.data[k])==n for k,n in run.SIZES.items())
    terminal=store.data['terminal_diagnostics']
    assert all(len(terminal[k])==n for k,n in run.TERMINAL_SIZES.items())

def test_complete_native_replica_pipeline_all_public_readers_budget_and_sealed_snapshots(tmp_path,monkeypatch):
    from runtime.wan import generation
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True)
    pipe=SimpleNamespace(transformer=FakeTransformer(),scheduler=FakeScheduler());initial=torch.zeros(SHAPE)
    pristine=run.trajectory.fingerprint(vars(pipe.scheduler))
    monkeypatch.setattr(run.backend,'execution_device_dtype',lambda:('cpu',torch.float32))
    def prepare(*args,**kwargs):
        assert kwargs['load_vae'] is False
        return pipe,initial,torch.tensor([.03]),torch.tensor([-.02]),torch.float32
    monkeypatch.setattr(generation,'prepare_generation',prepare)
    run.generation_worker(store,cfg)
    assert store.data['before_step25_identity']['status']=='MATCH'
    assert run.trajectory.fingerprint(vars(pipe.scheduler))==pristine and torch.count_nonzero(initial)==0
    assert store.data['counts']['generation_steps']==200 and store.data['counts']['controlled_steps']==75 and store.data['counts']['writer_sidecars']==100
    oldsteps=store.data['generation']['STATE_OLD_MULTI']['steps']
    newsteps=store.data['generation']['STATE_REPLICA_MULTI']['steps']
    for index in range(25,50):
        old=oldsteps[index]['pilot'];new=newsteps[index]['pilot']
        assert new['matched_actual_norm_l2']==old['actual_delta_l2']
        assert new['actual_delta_l2']==pytest.approx(old['actual_delta_l2'],abs=2e-7)
    for arm,row in store.data['generation'].items():
        assert [x['index'] for x in row['steps'] if x['enabled']]==([] if arm=='OFF' else list(range(25,50)))
        wd=row['receipt']['writer_diagnostics']
        assert wd['last_z_post_matches_terminal'] and wd['last_group_z_post_matches_terminal']
        last=store.data['writer_diagnostics'][arm+'/49']
        with np.load(last['path']) as archive:arrays={k:archive[k] for k in archive.files}
        assert set(arrays)==set(run.backend.DIAGNOSTIC_ARRAYS) and len(arrays)==10
        terminal=torch.load(row['terminal_path'],map_location='cpu',weights_only=True)
        t=control.build_targets(terminal,cfg['key'],control.message_bits(cfg['message']),arm)
        group_q=control.project_groups(terminal,t['basis']).numpy()
        assert np.array_equal(arrays['z_post_groups'],group_q) and np.array_equal(arrays['z_post'],group_q[0])
    observed=fake_media_runtime(monkeypatch)
    run.media_worker(store,cfg)
    assert observed.decode==4 and len(observed.encodes)==12
    assert len(store.data['group_observations'])==24 and len(store.data['projections'])==48 and len(store.data['mode_reads'])==96
    for i,arm in enumerate(run.ARMS):
        rr=store.data['rasters'][arm]
        assert observed.input_hashes[2*i:2*i+2]==[('RAW420',rr['sha256']),('MP4',rr['sha256'])]
        for channel in run.CHANNELS:
            for family in run.FAMILIES:
                for key in run.KEYS:
                    assert store.data['projections'][arm+'/'+channel+'/'+family+'/'+key]['status']=='SAVED'
    assert store.data['calls']['payload_read']==dict(attempted=24,completed=24)
    assert store.data['calls']['absolute_control']==store.data['calls']['difference_infer']==dict(attempted=48,completed=48)
    primary=[store.output/'receiver_readouts.json',store.output/'payload_readouts.json']
    sealed={p:p.read_bytes() for p in primary}
    run.terminal_diagnostics(store,cfg)
    terminal_path=store.output/'terminal_diagnostic_readouts.json'
    terminal_sealed=terminal_path.read_bytes();terminal=store.data['terminal_diagnostics']
    assert len(terminal['group_observations'])==8 and terminal['counts']['keyed_projections']==16 and terminal['counts']['path_score_records']==32
    assert sealed=={p:p.read_bytes() for p in primary}
    for path in [*primary,terminal_path]:
        text=path.read_text()
        for forbidden in ('registered_tau','true_path_rank','bit_errors'):assert forbidden not in text
    for path in primary:
        text=path.read_text()
        for forbidden in ('state_budget_schedule','matched_actual_norm_l2','writer_diagnostics','before_step25','terminal.pt'):assert forbidden not in text
    store.data['workers']={name:dict(status='COMPLETE') for name in ('generation','media')}
    run.evaluate(store,cfg);run.quality_diagnostics(store)
    assert run.finish(store,cfg) and sealed=={p:p.read_bytes() for p in primary} and terminal_path.read_bytes()==terminal_sealed
    counts=store.data['counts']
    assert counts['path_costs']==16704 and counts['difference_edge_costs']==272448 and counts['absolute_local_state_costs']==95040
    assert counts['payload_evaluations']==48 and counts['quality_pairs']==18
    assert_fixed_rosters(store)
    assert all(not x['accepted_payload'] and not x.get('state_path_accepted',False)
       for group in ('mode_reads','path_posthoc','payload_evaluations') for x in store.data[group].values())

def test_isolated_channel_and_worker_failures_preserve_fixed_group_and_logical_rows(tmp_path,monkeypatch):
    cfg=run.load_config();store=run.Store(tmp_path/'channel_failure',create=True);fake_generation_inputs(store)
    observed=fake_media_runtime(monkeypatch,fail_encode=2)
    run.media_worker(store,cfg);run.terminal_diagnostics(store,cfg);run.evaluate(store,cfg);run.quality_diagnostics(store)
    assert observed.decode==4 and len(observed.encodes)==12 and not run.finish(store,cfg)
    assert store.data['normalized']['OFF/RAW420']['status']=='FAILED'
    assert store.data['normalized']['OFF/MP4']['status']=='SAVED'
    assert store.data['counts']['normalized']==11 and store.data['counts']['keyed_projections']==44 and store.data['counts']['path_score_records']==88
    assert_fixed_rosters(store)
    assert all(x['status']=='MISSING_READ' for name,x in store.data['path_posthoc'].items() if name.startswith('OFF/RAW420/'))
    failed=run.Store(tmp_path/'worker_failure',create=True)
    def cannot_spawn(*args,**kwargs):raise OSError('fake child unavailable')
    monkeypatch.setattr(run.subprocess,'Popen',cannot_spawn)
    failed,halt=run.run_worker_phase(failed,'generation')
    run.settle(failed,'media','fake worker unavailable')
    run.terminal_diagnostics(failed,cfg);run.evaluate(failed,cfg);run.quality_diagnostics(failed)
    assert not halt and not run.finish(failed,cfg) and failed.data['counts']['generated']==0
    assert_fixed_rosters(failed)
    assert all(x['status'] not in ('PENDING','RUNNING') for x in failed.data['mode_reads'].values())

def test_notebook_AST_appended_group_and_terminal_setup_and_pure_layer_architecture(tmp_path):
    notebook=json.loads(builder.build('a'*40,tmp_path/'bound.ipynb').read_text())
    code=[''.join(c['source']) for c in notebook['cells']]
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert notebook['metadata']['candidate_binding']['source_sha']=='a'*40
    for cell,text in zip(notebook['cells'],code):
        if cell['cell_type']=='code':
            ast.parse(text)
            assert cell['outputs']==[] and cell['execution_count'] is None
    assert '--mode' not in builder.RUN and '--baseline' not in builder.RUN and '--worker' not in builder.RUN
    prefix=ast.parse(builder.SETUP.split('def write_json',1)[0])
    def is_setup_assignment(node):
        if not isinstance(node,ast.Assign):return False
        return any(isinstance(t,ast.Name) and t.id=='setup' or isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='setup' for t in node.targets)
    selected=[node for node in prefix.body if is_setup_assignment(node)]
    scope=dict(SOURCE_SHA='a'*40,FIXED=run.FIXED,TERMINAL_FIXED=run.TERMINAL_FIXED,ARMS=run.ARMS,
       CHANNELS=run.CHANNELS,FAMILIES=run.FAMILIES,KEYS=run.KEYS,MODES=run.MODES)
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<setup-dictionaries-only>','exec'),scope)
    setup=scope['setup']
    assert all(len(setup[k])==n for k,n in run.SIZES.items())
    assert len(setup['group_observations'])==24 and len(setup['projections'])==48
    assert len(setup['terminal_diagnostics']['group_observations'])==8 and len(setup['terminal_diagnostics']['projections'])==16
    assert all(len(setup['terminal_diagnostics'][k])==n for k,n in run.TERMINAL_SIZES.items())
    assert setup['state_budget_schedule']['role']=='writer only; never receiver input'
    for relative in ('main/tube_state/video_local_fourier_rm_replica_v1_state.py',
                     'main/tube_state/video_local_fourier_rm_replica_v1_control.py',
                     'runtime/wan/video_local_fourier_rm_replica_v1.py'):
        tree=ast.parse((run.ROOT/relative).read_text())
        imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        assert not any(name and name.startswith('experiments') for name in imports)
        if relative.startswith('main/'):assert not any(name and name.startswith('runtime') for name in imports)
    with pytest.raises(ValueError):builder.build('not-an-immutable-sha',tmp_path/'bad.ipynb')


def test_budget_write_failure_keeps_completed_old_and_unexecuted_candidate(tmp_path,monkeypatch):
    """A failed writer-only schedule never rewrites a completed OLD trajectory."""
    from runtime.wan import generation
    cfg=run.load_config();store=run.Store(tmp_path/'budget_failure',create=True)
    pipe=SimpleNamespace(transformer=FakeTransformer(),scheduler=FakeScheduler());initial=torch.zeros(SHAPE)
    monkeypatch.setattr(run.backend,'execution_device_dtype',lambda:('cpu',torch.float32))
    monkeypatch.setattr(generation,'prepare_generation',
        lambda *args,**kwargs:(pipe,initial,torch.tensor([.03]),torch.tensor([-.02]),torch.float32))
    executed=[]
    def completed_fake_trajectory(pipe,initial,scheduler,prompt,negative,input_dtype,arm,key,bits,count,record,**kwargs):
        executed.append(arm)
        assert kwargs['matched_norm_schedule'] is None
        for index in range(50):
            record(dict(index=index,enabled=arm!='OFF' and index>=25,
                pilot_delta_l2=(index+1)/100 if arm=='STATE_OLD_MULTI' and index>=25 else 0.,
                payload_delta_l2=0.,merged_delta_l2=0.))
        return initial.clone(),dict(before_step25={'fake_saved_prefix':True},
            writer_diagnostics=dict(last_z_post_matches_terminal=True,last_group_z_post_matches_terminal=True))
    monkeypatch.setattr(run.backend,'run_trajectory',completed_fake_trajectory)
    original_dump=run.dump
    def fail_budget(path,value):
        if Path(path).name=='state_budget_schedule.json':raise OSError('fake budget atomic write unavailable')
        return original_dump(path,value)
    monkeypatch.setattr(run,'dump',fail_budget)
    run.generation_worker(store,cfg)
    assert executed==list(run.ARMS[:3]) and len(store.data['generation'])==4
    old=store.data['generation']['STATE_OLD_MULTI'];candidate=store.data['generation']['STATE_REPLICA_MULTI']
    assert old['status']=='COMPLETE' and len(old['steps'])==50 and Path(old['terminal_path']).is_file()
    assert candidate['status']=='FAILED' and candidate['steps']==[]
    assert 'schedule unavailable' in candidate['error']
    budget=store.data['state_budget_schedule']
    assert budget['status']=='FAILED' and len(budget['steps'])==25
    assert all(r['status']=='RECORDED' for r in budget['steps'].values())
    assert store.data['calls']['state_budget_schedule_save']==dict(attempted=1,completed=0)
    assert 'state_budget_schedule_load' not in store.data['calls']
    assert any(r['stage']=='STATE_BUDGET_SCHEDULE' for r in store.data['failures'])
    assert_fixed_rosters(store)
    run.settle(store,'media','no fake media execution in budget failure test')
    assert not run.finish(store,cfg) and store.data['status']=='INCOMPLETE'
