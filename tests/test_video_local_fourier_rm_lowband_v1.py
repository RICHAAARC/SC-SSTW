"""One-candidate CPU mechanisms and fake model pipeline; no real GPU/VAE run."""
import ast,copy,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_local_fourier_rm_state as oldstate
from main.tube_state import video_local_fourier_rm_control as oldcontrol
from main.tube_state import video_local_fourier_rm_lowband_v1_state as state
from main.tube_state import video_local_fourier_rm_lowband_v1_control as control
from experiments.wan_state_clock import video_local_fourier_rm_lowband_v1_run as run
from scripts import build_video_local_fourier_rm_lowband_v1_notebook as builder
pytestmark=pytest.mark.unit
SMALL=(2,2,4,3)

@pytest.fixture(autouse=True)
def threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_public_lower_bandwidth_selection_orthogonality_and_scatter_adjoint():
    modes=state.fourier_modes();reps=[(h,w) for h in range(16) for w in range(16) if (h,w)<=((-h)%16,(-w)%16) and (h,w)!=(0,0)]
    assert len(reps)==129 and modes==sorted(reps,key=lambda x:(min(x[0],16-x[0])**2+min(x[1],16-x[1])**2,*x))[:32]
    assert modes[-2:]==[(2,4),(2,12)] and (4,2) not in modes and all((h,w)!=((-h)%16,(-w)%16) for h,w in modes)
    assert state.PUBLIC.blocks==((4,20,8,24),(4,20,40,56),(24,40,8,24),(24,40,40,56))
    for key in ('watermark','watermark-wrong'):
        assert np.array_equal(state.state_code(key),oldstate.state_code(key)) and np.array_equal(state.composite_signs(key),oldstate.composite_signs(key))
        U=state.bases(key);assert U.shape==(4,256,32)
        np.testing.assert_allclose(U.sum(axis=1),0,atol=4e-14);np.testing.assert_allclose(U.transpose(0,2,1)@U,np.broadcast_to(np.eye(32),(4,32,32)),atol=3e-15)
        host=np.random.default_rng(14).normal(size=(16,16));F=np.fft.fft2(host,norm='ortho').real
        for b,row in enumerate(state.basis_layout(key)['blocks']):
            np.testing.assert_allclose(host.reshape(-1)@U[b],[F[h,w]*factor*sign for (h,w),factor,sign in zip(row['modes'],row['factors'],row['signs'])],atol=2e-14)
            assert row['signs']==oldstate.basis_layout(key)['blocks'][b]['signs']
        target=state.synthesize(key);assert np.linalg.norm(target)==pytest.approx(1,abs=2e-14)
        q=state.extract(target[0,:,1:].transpose(1,0,2,3),key,np.ones((45,4),bool));np.testing.assert_allclose(q,state.composite_signs(key)*state.PUBLIC.alpha,atol=2e-16)
        assert np.count_nonzero(state.composite_signs(key))==5568 and np.count_nonzero(target[0,4,0])==0 and np.count_nonzero(target[0,:4])==0
        mask=np.ones((44,4),bool);mask[9,2]=False;received=target[0,:,1:45].transpose(1,0,2,3).copy();received[9,4,24:40,8:24]=np.nan
        masked=state.extract(received,key,mask);assert np.isfinite(masked).all() and np.count_nonzero(masked[9,2])==0
        U[:]=999;assert np.max(np.abs(state.bases(key)))<1


def test_shrink_only_new_gradient_and_exact_original_arm_velocity():
    z=torch.zeros((1,16,46,40,64));targets=control.build_targets(z,'watermark',control.message_bits('OKOK'),'STATE_LOWBAND_MULTI')
    for scale in (0.,10.):
        clean=torch.tensor(state.synthesize('watermark',dtype=np.float32))*scale;delta,receipt=control.pilot_delta(clean,targets,'STATE_LOWBAND_MULTI')
        leaf=clean.clone().requires_grad_(True);features=[]
        for block,((h0,h1,w0,w1),layout) in enumerate(zip(state.PUBLIC.blocks,state.basis_layout('watermark')['blocks'])):
            F=torch.fft.fft2(leaf[0,4,1:,h0:h1,w0:w1],norm='ortho').real
            features.append(torch.stack([F[:,h,w]*factor*sign for (h,w),factor,sign in zip(layout['modes'],layout['factors'],layout['signs'])],-1).reshape(45,4,8))
        q=torch.stack(features,1);loss=(q[targets['pilot_active']]-targets['pilot_target'][targets['pilot_active']]).square().mean();raw=-696.*torch.autograd.grad(loss,leaf)[0];norm=float(raw.double().norm());expected=raw*min(1.,1/norm)
        torch.testing.assert_close(delta,expected,atol=2e-8,rtol=3e-6);assert receipt['actual_delta_l2']==pytest.approx(min(norm,1.),abs=2e-7)
        if scale==0:assert receipt['actual_delta_l2']==pytest.approx(.25,abs=1e-7) and receipt['cap_scale']==1
        else:assert receipt['cap_scale']<1 and receipt['actual_delta_l2']==pytest.approx(1,abs=1e-7)
        closure=control.delta_closure(delta,control.project_tensor(delta,targets['basis'],targets['blocks']),targets);assert abs(closure['parseval_error'])<2e-7 and abs(closure['outside_ROI_energy'])<1e-12
    z=torch.randn(z.shape,generator=torch.Generator().manual_seed(71))/10;c=z/100+.03;u=z/100-.02
    for arm,original in [('OFF','OFF'),('PAYLOAD_MULTI','PAYLOAD_MULTI'),('STATE_OLD_MULTI','STATE_MULTI')]:
        newtargets=control.build_targets(z,'watermark',control.message_bits('OKOK'),arm);oldtargets=oldcontrol.build_targets(z,'watermark',oldcontrol.message_bits('OKOK'))
        for index in (24,25,49):
            actual,detail=control.guided_velocity(z,c,u,.8,newtargets,arm,index);expected,oldrow=oldcontrol.guided_velocity(z,c,u,.8,oldtargets,original,index)
            assert torch.equal(actual,expected) and detail.get('payload')==oldrow.get('payload') and detail.get('pilot')==oldrow.get('pilot')

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


def test_complete_four_arm_native_writer_and_two_reader_family_pipeline(tmp_path,monkeypatch):
    from runtime.wan import generation
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True);pipe=SimpleNamespace(transformer=FakeTransformer(),scheduler=FakeScheduler());initial=torch.zeros((1,16,46,40,64));before=run.trajectory.fingerprint(vars(pipe.scheduler))
    monkeypatch.setattr(run.backend,'execution_device_dtype',lambda:('cpu',torch.float32))
    def prepare(*args,**kwargs):assert kwargs['load_vae'] is False;return pipe,initial,torch.tensor([.03]),torch.tensor([-.02]),torch.float32
    monkeypatch.setattr(generation,'prepare_generation',prepare);run.generation_worker(store,cfg)
    assert run.trajectory.fingerprint(vars(pipe.scheduler))==before and torch.count_nonzero(initial)==0 and store.data['before_step25_identity']['status']=='MATCH'
    assert len({json.dumps(r['receipt']['before_step25'],sort_keys=True) for r in store.data['generation'].values()})==1
    assert store.data['counts']['generation_steps']==200 and store.data['counts']['controlled_steps']==75 and store.data['counts']['writer_sidecars']==100
    for arm,row in store.data['generation'].items():
        assert [s['index'] for s in row['steps'] if s['enabled']]==([] if arm=='OFF' else list(range(25,50))) and row['receipt']['writer_diagnostics']['last_z_post_matches_terminal'] is True
        last=store.data['writer_diagnostics'][arm+'/49'];arrays=np.load(last['path']);terminal=torch.load(row['terminal_path'],map_location='cpu',weights_only=True);targets=control.build_targets(terminal,cfg['key'],control.message_bits(cfg['message']),arm)
        assert np.array_equal(arrays['z_post'],control.project_tensor(terminal,targets['basis'],targets['blocks']).numpy())
    observed=fake_media_runtime(monkeypatch);run.media_worker(store,cfg)
    assert observed.decode==4 and len(observed.encodes)==12 and len(store.data['projections'])==48 and len(store.data['mode_reads'])==96
    for i,arm in enumerate(run.ARMS):
        rr=store.data['rasters'][arm];assert observed.input_hashes[2*i:2*i+2]==[('RAW420',rr['sha256']),('MP4',rr['sha256'])]
        for c,channel in enumerate(run.CHANNELS):
            aid=arm+'/'+channel;tr=store.data['transport'][aid];assert tr['input_raster_sha256']==rr['sha256'];rgb=torch.from_numpy(np.frombuffer(Path(tr['events']['rgb24']['path']).read_bytes(),np.uint8).reshape(SMALL).copy())
            assert torch.equal(observed.encodes[3*i+c],(rgb.float()/255).permute(3,0,1,2).unsqueeze(0)*2-1)
            for family in run.FAMILIES:
                for key in run.KEYS:assert store.data['projections'][aid+'/'+family+'/'+key]['status']=='SAVED'
    assert store.data['calls']['payload_read']==dict(attempted=24,completed=24) and store.data['calls']['absolute_control']==store.data['calls']['difference_infer']==dict(attempted=48,completed=48)
    snapshots=[store.output/'receiver_readouts.json',store.output/'payload_readouts.json'];sealed={str(p):p.read_bytes() for p in snapshots}
    for p in snapshots:
        text=p.read_text();assert 'writer_diagnostics' not in text and 'before_step25' not in text and 'terminal.pt' not in text and 'registered_tau' not in text and 'bit_errors' not in text
    store.data['workers']={p:dict(status='COMPLETE') for p in ('generation','media')};run.evaluate(store,cfg);run.quality_diagnostics(store);assert sealed=={str(p):p.read_bytes() for p in snapshots} and run.finish(store,cfg)
    assert store.data['counts']['path_costs']==16704 and store.data['counts']['difference_edge_costs']==272448 and store.data['counts']['absolute_local_state_costs']==95040 and store.data['counts']['payload_evaluations']==48 and store.data['counts']['quality_pairs']==18
    assert all(not r['accepted_payload'] and not r.get('state_path_accepted',False) for group in ('mode_reads','path_posthoc','payload_evaluations') for r in store.data[group].values())
    assert all(v['match'] for v in store.data['call_integrity']['rows'].values())


def test_single_encode_failure_retains_both_family_rows_and_sibling_channels(tmp_path,monkeypatch):
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True);fake_generation_inputs(store);observed=fake_media_runtime(monkeypatch,fail_encode=2);run.media_worker(store,cfg);run.evaluate(store,cfg);run.quality_diagnostics(store)
    assert observed.decode==4 and len(observed.encodes)==12 and not run.finish(store,cfg)
    assert store.data['normalized']['OFF/RAW420']['status']=='FAILED' and store.data['normalized']['OFF/MP4']['status']=='SAVED'
    assert store.data['counts']['normalized']==11 and store.data['counts']['keyed_projections']==44 and store.data['counts']['path_score_records']==88 and store.data['counts']['payload_reads']==22
    assert all(len(store.data[k])==n for k,n in run.SIZES.items()) and all(r['status'] not in ('PENDING','RUNNING') for group in ('projections','mode_reads','payload_reads') for r in store.data[group].values())
    assert all(r['status']=='MISSING_READ' for name,r in store.data['path_posthoc'].items() if name.startswith('OFF/RAW420/'))


def test_missing_terminal_keeps_other_arms_and_full_public_receiver_roster(tmp_path,monkeypatch):
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True);fake_generation_inputs(store);Path(store.data['generation']['OFF']['terminal_path']).unlink();observed=fake_media_runtime(monkeypatch);run.media_worker(store,cfg);run.evaluate(store,cfg);run.quality_diagnostics(store)
    assert observed.decode==3 and len(observed.encodes)==9 and store.data['counts']['path_score_records']==72 and not run.finish(store,cfg)
    assert store.data['normalized']['STATE_LOWBAND_MULTI/MP4']['status']=='SAVED' and store.data['terminal_inputs']['OFF']['status']=='FAILED'
    assert len(store.data['path_posthoc'])==96 and len(store.data['edge_posthoc'])==48 and len(store.data['payload_evaluations'])==48
    assert all(r['status']=='MISSING_READ' for name,r in store.data['path_posthoc'].items() if name.startswith('OFF/'))


def test_worker_spawn_failure_retains_all_pending_denominators(tmp_path,monkeypatch):
    cfg=run.load_config();store=run.Store(tmp_path/'output',create=True)
    def fail(*args,**kwargs):raise OSError('fake generation worker unavailable')
    monkeypatch.setattr(run.subprocess,'Popen',fail);store,halt=run.run_worker_phase(store,'generation');run.settle(store,'media','no model worker');run.evaluate(store,cfg);run.quality_diagnostics(store)
    assert not halt and not run.finish(store,cfg) and store.data['counts']['generated']==0 and store.data['counts']['path_score_records']==0
    assert all(len(store.data[k])==n for k,n in run.SIZES.items()) and all(r['status']=='MISSING_READ' for r in store.data['path_posthoc'].values())


def test_notebook_binding_actual_generation_probe_and_fixed_setup_roster(tmp_path):
    nb=json.loads(builder.build('a'*40,tmp_path/'bound.ipynb').read_text());sources=[''.join(c['source']) for c in nb['cells']]
    assert sources[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n" and nb['metadata']['candidate_binding']['source_sha']=='a'*40
    for cell,code in zip(nb['cells'],sources):
        if cell['cell_type']=='code':ast.parse(code);assert cell['outputs']==[] and cell['execution_count'] is None
    assert 'AutoencoderKLWan,WanPipeline' in builder.ENVIRONMENT and 'shutil.which' in builder.ENVIRONMENT and "'MEDIA_TOOL_REPAIR',check=False" in builder.ENVIRONMENT
    assert 'venv' not in '\n'.join(sources[2:]) and '--mode' not in builder.RUN and '--baseline' not in builder.RUN and '--worker' not in builder.RUN
    nodes=ast.parse(builder.SETUP.split('def write_json')[0]);assignment=next(n for n in nodes.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='setup' for t in n.targets));scope=dict(SOURCE_SHA='a'*40,FIXED=run.FIXED,ARMS=run.ARMS,CHANNELS=run.CHANNELS,FAMILIES=run.FAMILIES,KEYS=run.KEYS,MODES=run.MODES)
    setup=eval(compile(ast.Expression(assignment.value),'<setup>','eval'),scope);assert all(len(setup[k])==n for k,n in run.SIZES.items())
    for relative in ('main/tube_state/video_local_fourier_rm_lowband_v1_state.py','main/tube_state/video_local_fourier_rm_lowband_v1_control.py','runtime/wan/video_local_fourier_rm_lowband_v1.py'):
        text=(run.ROOT/relative).read_text();tree=ast.parse(text);imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)];assert not any(name and name.startswith('experiments') for name in imports)
        if relative.startswith('main/'):assert not any(name and name.startswith('runtime') for name in imports)
    with pytest.raises(ValueError):builder.build('not-an-immutable-sha',tmp_path/'bad.ipynb')
