"""Independent replica mathematics and fake native pipeline; zero real model/media calls."""
import ast,copy,gzip,hashlib,inspect,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_local_fourier_rm_state as original_state
from main.tube_state import video_local_fourier_rm_control as original_control
from main.tube_state import video_local_fourier_rm_dwell4_v1_state as state
from main.tube_state import video_local_fourier_rm_dwell4_v1_control as control
from main.tube_state import video_local_fourier_rm_dwell4_v1_receiver as scorer
from experiments.wan_state_clock import video_local_fourier_rm_dwell4_v1_run as run
from scripts import build_video_local_fourier_rm_dwell4_v1_notebook as builder

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
    basis=independent_basis(key);words=original_state.state_code(key);out=np.zeros(SHAPE)
    for start in range(1,46):
        for age in range(min(4,46-start)):
            for block,(h0,h1,w0,w1) in enumerate(A):
                coeff=words[(start-1)//4,8*block:8*block+8]
                out[0,4,start+age,h0:h1,w0:w1]+=(basis[block,:,8*age:8*age+8]@coeff).reshape(8,8)*original_state.PUBLIC.alpha
    return out


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
    assert list(inspect.signature(scorer.absolute_control).parameters)==['q','key','availability','protocol']
    assert list(inspect.signature(scorer.infer_difference).parameters)==['q','key','availability','protocol']

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
    newsteps=store.data['generation']['STATE_DWELL4_MULTI']['steps']
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
    for channel in run.CHANNELS:
        for mode in run.MODES:
            mid='STATE_DWELL4_MULTI/'+channel+'/DWELL4/CORRECT/'+mode
            assert store.data['path_posthoc'][mid]['unique_exact_true_path']
            assert store.data['path_posthoc'][mid]['top_equals_true_exact_class']
            assert not store.data['path_posthoc'][mid]['near_paths_count_as_correct']
            oldmid='STATE_DWELL4_MULTI/'+channel+'/OLD8/CORRECT/'+mode
            assert store.data['mode_reads'][mid]['summary']['min_cost'] < store.data['mode_reads'][oldmid]['summary']['min_cost']
    assert len(store.data['shared'])==4
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
    for relative in ('main/tube_state/video_local_fourier_rm_dwell4_v1_state.py',
                     'main/tube_state/video_local_fourier_rm_dwell4_v1_control.py',
                     'runtime/wan/video_local_fourier_rm_dwell4_v1.py'):
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
    old=store.data['generation']['STATE_OLD_MULTI'];candidate=store.data['generation']['STATE_DWELL4_MULTI']
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

@pytest.mark.parametrize('key',['watermark','watermark-wrong'])
def test_dwell_math_and_every_ideal_path_exact_equivalence(key):
    expected=independently_synthesized_target(key)
    np.testing.assert_allclose(state.synthesize(key),expected,atol=1e-16)
    assert np.square(expected).sum()==pytest.approx(1.,abs=1e-14)
    assert np.count_nonzero(state.composite_signs(key))==5568
    assert np.array_equal(state.bases(key),original_state.bases(key))
    words=original_state.state_code(key)
    for u in range(1,46):
        for age in range(4):
            actual=state.composite_signs(key)[u-1,:,age]
            expected_code=words[(u-age-1)//4].reshape(4,8) if u>age else np.zeros((4,8))
            np.testing.assert_array_equal(actual,expected_code)
    mask=np.ones((44,4),bool)
    got=state.extract(expected[0,:,1:45].transpose(1,0,2,3),key,mask)
    np.testing.assert_allclose(got,state.composite_signs(key)[:44]*state.PUBLIC.alpha,atol=1e-15)
    # Exhaustive ideal paths via independently computed direct class distance matrix.
    for kind in ('absolute','difference'):
        family=state.family(key,44,mask) if kind=='absolute' else scorer.difference_family(key,mask,protocol=state)
        means=family['means']
        assert len(means)==174 and len(family['classes'])==174
        distance=np.sum(means*means,axis=1)[:,None]+np.sum(means*means,axis=1)[None,:]-2*means@means.T
        assert np.all(np.argmin(distance,axis=1)==np.arange(174))
        assert np.all(np.sum(distance<1e-12,axis=1)==1)
    catalog=state.catalog(44);truth=next(i for i,r in enumerate(catalog) if r['structurally_valid'] and r['taus']==list(range(1,45)))
    for fn in (scorer.absolute_control,scorer.infer_difference):
        inf=fn(got,key,mask,protocol=state)
        assert inf['summary']['top_catalog_indices']==[truth]
        assert inf['summary']['state_path_accepted'] is False
        none=fn(got,key,np.zeros_like(mask),protocol=state)
        assert none['summary']['reason']=='NO_OBSERVATIONS'
        partial=mask.copy();partial[1:]=False
        inf=fn(got,key,partial,protocol=state)
        assert not inf['summary']['unique_model_hypothesis']
        zero=fn(np.zeros_like(got),key,mask,protocol=state)
        assert zero['summary']['status']=='NO_ENERGY'
        invalid=got.copy();invalid[0,0,0,0]=np.nan
        assert fn(invalid,key,mask,protocol=state)['summary']['reason']=='INVALID_OBSERVATION'

def test_dwell_applied_budget_gradient_zero_and_failure():
    zero=torch.zeros(SHAPE)
    targets=control.build_targets(zero,'watermark',control.message_bits('OKOK'),'STATE_DWELL4_MULTI')
    expected=torch.tensor(independently_synthesized_target('watermark'),dtype=torch.float32)
    # Orthonormal active MSE derivative: raw update = 2*696/5568 * target = target/4.
    delta,receipt=control.pilot_delta(zero,targets,'STATE_DWELL4_MULTI',matched_norm=.7)
    torch.testing.assert_close(delta,expected*.7,atol=1e-8,rtol=3e-6)
    assert receipt['raw_delta_l2']==pytest.approx(.25,abs=1e-7)
    assert receipt['norm_match_scale']>1 and receipt['actual_delta_l2']==pytest.approx(.7,abs=1e-7)
    for clean in (zero,torch.randn(SHAPE,generator=torch.Generator().manual_seed(823))*.1):
        U=independent_basis('watermark')
        raw_expected=torch.zeros_like(clean)
        signs=state.composite_signs('watermark')
        for b,(h0,h1,w0,w1) in enumerate(A):
            patch=clean[0,4,1:,h0:h1,w0:w1].numpy().reshape(45,64).astype(np.float64)
            q=patch@U[b]
            residual=np.where(signs[:,b].reshape(45,32)!=0,
                signs[:,b].reshape(45,32)*state.PUBLIC.alpha-q,0.)
            raw_expected[0,4,1:,h0:h1,w0:w1]=torch.tensor((.25*residual@U[b].T).reshape(45,8,8),dtype=torch.float32)
        direction=raw_expected*(.37/float(raw_expected.double().norm()))
        delta,receipt=control.pilot_delta(clean,targets,'STATE_DWELL4_MULTI',matched_norm=.37)
        torch.testing.assert_close(delta,direction,atol=2e-8,rtol=5e-5)
        assert float(delta.double().norm())==pytest.approx(.37,abs=1e-7)
        closure=control.delta_closure(delta,control.project_groups(delta,targets['basis']),targets)
        assert abs(closure['parseval_error'])<1e-7 and abs(closure['outside_ROI_energy'])<1e-12
        assert closure['boundary_projection_energy']<1e-14
    delta,_=control.pilot_delta(zero,targets,'STATE_DWELL4_MULTI',matched_norm=0)
    assert torch.count_nonzero(delta)==0
    empty=dict(targets,pilot_target=torch.zeros_like(targets['pilot_target']))
    with pytest.raises(ValueError,match='zero dwell raw direction'):
        control.pilot_delta(zero,empty,'STATE_DWELL4_MULTI',matched_norm=.1)
    for bad in (None,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError):
            control.pilot_delta(zero,targets,'STATE_DWELL4_MULTI',matched_norm=bad)
