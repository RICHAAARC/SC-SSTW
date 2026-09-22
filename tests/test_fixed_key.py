"""Directed CPU/stub checks for the fixed-key candidate, not real-video validation."""
import ast
import copy
import importlib.metadata
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import fixed_key as fk, projection_margin as carrier, state_clock
from runtime.wan import fixed_key_cli, fixed_key_core as core, fixed_key_control as control, trajectory, window_state_mse
from runtime.wan.io import dump
from experiments.wan_state_clock import flow_fixed_key_run as run

pytestmark = pytest.mark.unit
torch.set_num_threads(1)
ROOT=Path(__file__).parents[1]
KEY=b'WanProjection-first-validation-key-v1'
@pytest.fixture(scope='module')
def book(): return fk.codebook(KEY)
def marked(book):
    z=np.zeros(carrier.SHAPE,dtype=np.float32)
    carrier.put_blocks(z,(2*book['directions']*book['code'][:,None]).reshape(11,160,1024).astype(np.float32))
    return z

def test_exact_legacy_template_and_all_window_gradient(book):
    legacy=state_clock.codebook(KEY)
    for k in ('directions','sync','polarity'):np.testing.assert_array_equal(book[k],legacy[k])
    np.testing.assert_array_equal(book['code'],legacy['codes'][0])
    np.testing.assert_array_equal(book['states'],legacy['states'][0])
    z=torch.zeros(carrier.SHAPE,dtype=torch.float64,requires_grad=True)
    loss=fk.loss(z,book);g,=torch.autograd.grad(loss,z)
    assert (fk.torch_projections(g,torch.as_tensor(book['directions'])).reshape(11,160).abs().sum(1)>0).all()
    assert torch.count_nonzero(g[:,:,(0,45)])==0
    p=fk.torch_projections(z,torch.as_tensor(book['directions']))
    torch.testing.assert_close(loss,-(p.tanh()*torch.as_tensor(book['code'])).mean(),rtol=0,atol=0)

def test_window_state_mse_formula_and_projection_finite_difference(book):
    p=torch.linspace(-.7,.7,1760,dtype=torch.float64,requires_grad=True)
    loss,means=window_state_mse.loss_from_projections(p,book)
    signed=(p.tanh()*torch.as_tensor(book['sync']*book['polarity'],dtype=torch.float64)).reshape(11,160)
    expected=torch.stack((signed[:,0::2].mean(1),signed[:,1::2].mean(1)),1)
    target=torch.as_tensor(book['states'],dtype=torch.float64)
    torch.testing.assert_close(means,expected,rtol=0,atol=1e-15)
    torch.testing.assert_close(loss,(expected-target).square().mean(),rtol=0,atol=1e-15)
    gradient,=torch.autograd.grad(loss,p)
    expanded=(means-target)[:,None,:].expand(11,80,2).reshape(-1)
    analytic=(2/1760)*expanded*torch.as_tensor(book['sync']*book['polarity'],dtype=torch.float64)*(1-p.tanh().square())
    torch.testing.assert_close(gradient,analytic,rtol=1e-12,atol=1e-14)
    index=317;epsilon=1e-6
    with torch.no_grad():
        plus=p.detach().clone();minus=p.detach().clone();plus[index]+=epsilon;minus[index]-=epsilon
        numerical=(window_state_mse.loss_from_projections(plus,book)[0]-window_state_mse.loss_from_projections(minus,book)[0])/(2*epsilon)
    assert gradient[index]==pytest.approx(float(numerical),rel=1e-7,abs=1e-8)

def test_window_state_mse_reweights_unbalanced_axes_and_matches_legacy_for_equal_gap(book):
    sync=torch.as_tensor(book['sync'],dtype=torch.float64);polarity=torch.as_tensor(book['polarity'],dtype=torch.float64)
    states=torch.as_tensor(book['states'],dtype=torch.float64).reshape(11,2)
    state_per_block=states[:,None,:].expand(11,80,2).reshape(-1)
    torch.testing.assert_close(torch.as_tensor(book['code'],dtype=torch.float64),state_per_block*sync*polarity,rtol=0,atol=0)
    aligned=torch.atanh(.8*state_per_block*sync*polarity)
    unbalanced=aligned.clone();unbalanced[:160:2]=0.
    unbalanced.requires_grad_()
    means=window_state_mse.state_means_from_projections(unbalanced,book)
    torch.testing.assert_close(means[0,0],torch.zeros((),dtype=torch.float64),rtol=0,atol=1e-15)
    torch.testing.assert_close(means[0,1],.8*states[0,1],rtol=0,atol=1e-14)
    torch.testing.assert_close(means[1:],.8*states[1:],rtol=0,atol=1e-14)
    old=-(unbalanced.tanh()*torch.as_tensor(book['code'],dtype=torch.float64)).mean()
    new,_=window_state_mse.loss_from_projections(unbalanced,book)
    old_gradient,=torch.autograd.grad(old,unbalanced,retain_graph=True)
    new_gradient,=torch.autograd.grad(new,unbalanced)
    old_groups=old_gradient.abs().reshape(11,160)
    new_groups=new_gradient.abs().reshape(11,160)
    old_ratio=old_groups[0,0::2].mean()/old_groups[1,0::2].mean()
    new_ratio=new_groups[0,0::2].mean()/new_groups[1,0::2].mean()
    assert new_ratio > old_ratio*4
    common_gap=torch.atanh(.4*state_per_block*sync*polarity).detach().requires_grad_(True)
    legacy=-(common_gap.tanh()*torch.as_tensor(book['code'],dtype=torch.float64)).mean()
    state_loss,common_means=window_state_mse.loss_from_projections(common_gap,book)
    torch.testing.assert_close(common_means,.4*states,rtol=0,atol=1e-14)
    legacy_gradient,=torch.autograd.grad(legacy,common_gap,retain_graph=True)
    state_gradient,=torch.autograd.grad(state_loss,common_gap)
    torch.testing.assert_close(state_gradient/state_gradient.norm(),legacy_gradient/legacy_gradient.norm(),rtol=1e-12,atol=1e-12)

def test_window_state_mse_clean_leaf_support_and_descent(book):
    clean=torch.zeros(carrier.SHAPE,dtype=torch.float32)
    raw,record=control.clean_direction(clean,book,objective=control.WINDOW_STATE_MSE_OBJECTIVE)
    assert record['objective']==control.WINDOW_STATE_MSE_OBJECTIVE
    assert record['objective_identity']['states']=='existing +/-1 fixed codebook states'
    assert torch.isfinite(raw).all() and torch.count_nonzero(raw)>0
    assert torch.count_nonzero(raw[:,:,(0,45)])==0
    with torch.no_grad():
        before=window_state_mse.loss(clean.double(),book)[0]
        step=1e-3/raw.abs().max()
        after=window_state_mse.loss(clean.double()+step*raw.double(),book)[0]
    assert after < before

def test_full_projection_and_native_identity_score_share_definition(book):
    z=marked(book);row=fk.score_path({0:z},book,fk.REFERENCE_PATHS['IDENTITY'])
    direct=np.einsum('ij,ij->i',carrier.blocks(z).reshape(1760,1024).astype(float),book['directions'].astype(float))
    np.testing.assert_allclose([w['signed_evidence'] for w in row['windows']],(np.clip(direct,-1,1)*book['code']).reshape(11,160).mean(1),rtol=0,atol=1e-15)
    assert row['matched_components']==7040 and row['full_blocks']==1760
    assert row['score']==pytest.approx(1.)
    record=fk.nominal_record(torch.from_numpy(z),book,include_state=True)
    assert record['identity_state']['score']==row['score']

def test_partial3_physical_components_clip_once_and_missing_penalty(book):
    z=np.zeros((1,16,37,40,64),dtype=np.float32)
    selected=carrier.allocation(36,0,5,4,0)[:4];assert sum(x is not None for x in selected)==3
    directions=book['directions'][:160].reshape(160,4,256)
    expected=np.zeros(160)
    for slot,j in enumerate(selected):
        if j is None:continue
        component=(slot+1)*directions[:,slot]*book['code'][:160,None]
        expected+=np.einsum('ij,ij->i',component.astype(float),directions[:,slot].astype(float))
        z[0,:,j+1]=component.reshape(10,16,16,4,4).transpose(2,0,3,1,4).reshape(16,40,64)
    e=fk.emission({0:z},book,0,0,5,4,0)
    assert e['observed_components']==480 and e['support_kind']=='PARTIAL3'
    assert e['signed_evidence']==pytest.approx(float(np.mean(np.clip(expected,-1,1)*book['code'][:160])),abs=1e-14)
    short=marked(book)[:,:,:5]
    row=fk.score_path({0:short},book,fk.REFERENCE_PATHS['IDENTITY'])
    assert row['matched_components']==640 and row['matched_windows']==[0]
    assert row['innovation_by_window'][1:]==[1.]*10
    assert fk.score_path({},book,fk.REFERENCE_PATHS['IDENTITY'])['status']=='INVALID'

def test_full_fixed_search_has_one_template_and_diagnostics_do_not_select(book):
    z=marked(book);result=fk.read({0:z},book)
    assert result['attempted_path_count']==4284 and result['template_count']==1
    assert result['best']['path']==fk.REFERENCE_PATHS['IDENTITY']
    assert 'payload' not in result and 'best_by_payload' not in result
    assert set(result['fixed_path_diagnostics'])==set(fk.REFERENCE_PATHS)
    z.flat[0]=np.nan;assert fk.read({0:z},book)['status']=='INVALID'

def fake_detection(score):
    return dict(status='SCORED',existence_statistic=score,receiver_protocol_id=fk.PROTOCOL_ID,key_id=fk.key_identifier(KEY),best=None)
def complete(score):
    return dict(status='SCORED',detection=fake_detection(score),observations={str(i):dict(status='COMPLETE') for i in range(4)})

def test_calibration_same_max_family_views_independent_and_missing_retained():
    item={'views':{v:complete(x) for v,x in zip(fk.FIXED_VIEWS,[.1,.2,.3])}}
    source=run._arm_source_record(item);assert source['statistic']==.3
    cal=fk.freeze_calibration([source,dict(source,statistic=.4)],key_id=fk.key_identifier(KEY))
    assert cal['threshold']==.400001
    assert fk.decide(fake_detection(.4),cal)['status']=='REJECTED'
    assert fk.decide(fake_detection(.5),cal)['status']=='DETECTED'
    assert fk.decide(fake_detection(.5),dict(cal,key_id='wrong'))['status']=='UNCALIBRATED'
    item['views']['FULL']['observations']['2']['status']='FAILED'
    assert run._arm_source_record(item)['status']=='INVALID'
    run._attach_decisions(item,cal,marked=True,calibration_sample=False)
    assert item['decision']['status']=='INVALID' and item['views']['FULL']['decision']['status']=='INVALID'
    assert item['views']['DELETE90']['decision']['status']=='REJECTED'

def test_alignment_retains_reference_denominator_when_best_missing(book):
    ref=fk.score_path({0:marked(book)},book,fk.REFERENCE_PATHS['IDENTITY'])
    best=copy.deepcopy(ref);best['windows'][2].update(valid=False,group_count=0,selected=[None]*4)
    d={'best':best,'fixed_path_diagnostics':{'IDENTITY':ref,'DELETE90_REFERENCE':ref}}
    report=fk.alignment_report(d,'FULL')
    assert report['reference_eligible_windows']==11 and report['comparable_windows']==10
    assert report['best_missing_reference_windows']==[2]
    delete=fk.alignment_report(d,'DELETE90');assert delete['reference_eligible_windows']==10
    assert delete['excluded_from_alignment_only']==[5]

class FakeScheduler:
    def __init__(self):
        self.config=SimpleNamespace(prediction_type='flow_prediction',thresholding=False,lower_order_final=True)
        self.predict_x0=True;self.timesteps=torch.arange(50,0,-1);self.sigmas=torch.linspace(1,0,51);self.step_index=None
        self.history=[]
    def step(self,v,t,z,return_dict=False):
        if self.step_index is None:self.step_index=0
        self.history.append(float(z.mean()));self.step_index+=1
        return (z-.01*v,)
class FakeTransformer(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
    def forward(self,hidden_states,**kwargs):return (hidden_states*.1+self.weight*.01,)

def test_explicit_window_state_objective_routes_on_comparable_fake_histories(monkeypatch):
    def make_pipe():
        return SimpleNamespace(scheduler=FakeScheduler(),transformer=FakeTransformer())
    monkeypatch.setattr(core,'prepare_generation',lambda *a,**k:(make_pipe(),torch.zeros(carrier.SHAPE),torch.tensor(1.),torch.tensor(-1.),torch.float32))
    snapshots=[];pending=[];captured=[]
    original_prefix=trajectory.prefix44
    def prefix(*args,**kwargs):
        z44,snapshot=original_prefix(*args,**kwargs)
        snapshots.append((snapshot,trajectory.fingerprint(vars(snapshot))))
        return z44,snapshot
    monkeypatch.setattr(trajectory,'prefix44',prefix)
    original_velocity=trajectory.velocity
    def velocity(pipe,z,scheduler,prompt,negative,dtype,guidance,index,count):
        value=original_velocity(pipe,z,scheduler,prompt,negative,dtype,guidance,index,count)
        if index in (44,46):pending.append((index,z.detach().clone(),value.detach().clone(),float(scheduler.sigmas[index])))
        return value
    monkeypatch.setattr(trajectory,'velocity',velocity)
    original_direction=control.clean_direction
    def direction(clean,book,count=None,*,objective=control.DEFAULT_OBJECTIVE):
        index,z,velocity_value,sigma=pending.pop()
        expected=z-sigma*velocity_value
        torch.testing.assert_close(clean,expected,rtol=0,atol=0)
        if objective==control.WINDOW_STATE_MSE_OBJECTIVE:
            captured.append((index,clean.detach().clone()))
        return original_direction(clean,book,count,objective=objective)
    monkeypatch.setattr(control,'clean_direction',direction)
    config=core.generation_config(core.load_protocol(),'fixed prompt',17)
    implicit=core.generate_key_terminals(config,KEY,('SINGLE46','MULTI44_46'))
    legacy=core.generate_key_terminals(config,KEY,('SINGLE46','MULTI44_46'),objective=control.DEFAULT_OBJECTIVE)
    state=core.generate_key_terminals(config,KEY,('SINGLE46','MULTI44_46'),objective=control.WINDOW_STATE_MSE_OBJECTIVE)
    assert implicit['initial_noise_fingerprint']==legacy['initial_noise_fingerprint']
    assert implicit['state44_fingerprint']==legacy['state44_fingerprint']
    for arm in implicit['arms']:
        torch.testing.assert_close(implicit['arms'][arm]['terminal'],legacy['arms'][arm]['terminal'],rtol=0,atol=0)
    assert legacy['initial_noise_fingerprint']==state['initial_noise_fingerprint']
    assert legacy['state44_fingerprint']==state['state44_fingerprint']
    assert legacy['writer_objective']['id']==control.DEFAULT_OBJECTIVE
    assert state['writer_objective']['id']==control.WINDOW_STATE_MSE_OBJECTIVE
    for result,objective in ((legacy,control.DEFAULT_OBJECTIVE),(state,control.WINDOW_STATE_MSE_OBJECTIVE)):
        for arm in result['arms'].values():
            assert arm['status']=='GENERATED'
            assert all(row['objective']==objective for row in arm['clean_gradients'])
            assert arm['cumulative_native_response']['actual_D']['support']['sum_rms']==pytest.approx(control.R_STAR,rel=2e-5)
        single=result['arms']['SINGLE46']['control_steps'];multi=result['arms']['MULTI44_46']['control_steps']
        assert [row['target_D_support_rms'] for row in single]==[control.R_STAR]
        assert [row['target_D_support_rms'] for row in multi]==[control.R_STAR/2]*2
        for row in single+multi:
            assert row['actual_D']['support_rms']==pytest.approx(row['target_D_support_rms'],rel=2e-5)
    assert [row['index'] for row in state['arms']['MULTI44_46']['control_steps']]==[44,46]
    assert [index for index,_ in captured]==[46,44,46]
    assert not torch.equal(captured[1][1],captured[2][1])
    assert all(trajectory.fingerprint(vars(snapshot))==before for snapshot,before in snapshots)

def test_cli_exposes_explicit_writer_objective(monkeypatch,tmp_path):
    seen={}
    monkeypatch.setattr(fixed_key_cli,'load_protocol',lambda path:core.load_protocol())
    def generate(prompt,seed,key,output,protocol,arm,*,objective):
        seen.update(prompt=prompt,seed=seed,key=key,output=output,protocol=protocol,arm=arm,objective=objective)
        return {'status':'VIDEO_PERSISTED'}
    monkeypatch.setattr(fixed_key_cli,'generate_video',generate)
    result=fixed_key_cli.main(['generate','--prompt','p','--seed','7','--key','k','--protocol','protocol.json',
                               '--output',str(tmp_path/'out.mp4'),'--objective',control.WINDOW_STATE_MSE_OBJECTIVE])
    assert result['status']=='VIDEO_PERSISTED'
    assert seen['objective']==control.WINDOW_STATE_MSE_OBJECTIVE

def test_public_writer_routes_window_state_objective_and_persists_identity(monkeypatch,tmp_path):
    def make_pipe():
        return SimpleNamespace(scheduler=FakeScheduler(),transformer=FakeTransformer())
    class FakeVAE(torch.nn.Module):
        def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.ones(1))
    saved_media=[]
    monkeypatch.setattr(core,'prepare_generation',lambda *a,**k:(make_pipe(),torch.zeros(carrier.SHAPE),torch.tensor(1.),torch.tensor(-1.),torch.float32))
    monkeypatch.setattr(core,'load_frozen_vae',lambda config:FakeVAE())
    monkeypatch.setattr(core,'decode_normalized_latent',lambda vae,terminal:torch.zeros(181,1,1,3))
    monkeypatch.setattr(core,'encode_rgb',lambda rgb,path,fps,crf:saved_media.append((rgb.shape,path,fps,crf)))
    output=tmp_path/'writer.mp4'
    result=core.generate_video('fixed prompt',17,KEY,output,core.load_protocol(),'SINGLE46',objective=control.WINDOW_STATE_MSE_OBJECTIVE)
    assert result['status']=='VIDEO_PERSISTED'
    assert result['writer']['objective']['id']==control.WINDOW_STATE_MSE_OBJECTIVE
    assert all(row['objective']==control.WINDOW_STATE_MSE_OBJECTIVE for row in result['writer']['clean_gradients'])
    assert saved_media==[((181,1,1,3),output,8,18)]
    record=tmp_path/'writer.json';dump(record,result)
    assert json.loads(record.read_text())['writer']['objective']['id']==control.WINDOW_STATE_MSE_OBJECTIVE

def test_fresh_shared_prefix_live_second_control_counts_and_budget(monkeypatch):
    pipe=SimpleNamespace(scheduler=FakeScheduler(),transformer=FakeTransformer())
    monkeypatch.setattr(core,'prepare_generation',lambda *a,**k:(pipe,torch.zeros(carrier.SHAPE),torch.tensor(1.),torch.tensor(-1.),torch.float32))
    events=[];calls=[];original=control.clean_direction
    def direction(clean,book,count):
        calls.append(clean.clone());return original(clean,book,count)
    monkeypatch.setattr(control,'clean_direction',direction)
    config=core.generation_config(core.load_protocol(),'fixed prompt',17)
    got=core.generate_key_terminals(config,KEY,('OFF','SINGLE46','MULTI44_46'),lambda k,c:events.append((k,c)))
    assert all(x['status']=='GENERATED' for x in got['arms'].values())
    assert events.count(('transformer',True))==124 and events.count(('scheduler_step',True))==62
    for kind in ('zero_shadow_step','unit_response_probe_step','clean_leaf_backward'):assert events.count((kind,True))==3
    assert len(calls)==3 and not torch.equal(calls[0],calls[2])
    multi=got['arms']['MULTI44_46'];single=got['arms']['SINGLE46']
    assert [x['index'] for x in multi['control_steps']]==[44,46]
    assert multi['control_steps'][1]['history_fingerprint']!=single['control_steps'][0]['history_fingerprint']
    for arm in (single,multi):assert arm['cumulative_native_response']['actual_D']['support']['sum_rms']==pytest.approx(control.R_STAR,rel=2e-5)
    assert len(multi['nominal_terminal']['identity_state']['windows'])==11

def test_public_receiver_missing_phase_and_wrong_protocol_fail(monkeypatch,tmp_path):
    protocol=core.load_protocol();seen=[]
    changed=copy.deepcopy(protocol);changed['receiver']['path_count']=1
    monkeypatch.setattr(core,'load_frozen_vae',lambda c:seen.append(c))
    with pytest.raises(ValueError):core.receive_pixels(None,KEY,changed,None)
    assert not seen
    monkeypatch.setattr(core,'read_mp4',lambda p:torch.zeros(181,1,1,3))
    monkeypatch.setattr(core,'encode_four_phases',lambda *a,**k:({}, {'0':{'status':'FAILED'},**{str(i):{'status':'COMPLETE'} for i in (1,2,3)}}))
    monkeypatch.setattr(fk,'read',lambda *a:fake_detection(1.))
    got=core.receive_mp4(tmp_path/'received.mp4',KEY,protocol,None,vae=object())
    assert got['decision']['status']=='INVALID'
    assert tuple(inspect.signature(core.receive_mp4).parameters)[:4]==('mp4_path','key','protocol','calibration')

def test_runner_failure_slots_and_eval_after_failed_calibration(monkeypatch,tmp_path):
    attempted=[]
    def fail(command,log):attempted.append(command);raise OSError('planned failure')
    monkeypatch.setattr(run,'_run_child',fail)
    got=run.run_all(run.MANIFEST,tmp_path/'fail')
    assert len(attempted)==8 and len(got['failures'])==8
    assert got['calibration']['status']=='UNCALIBRATED'
    arms=[v for c in got['cases'].values() for v in c['videos'].values()]
    assert len(arms)==8 and sum(len(a['views']) for a in arms)==24
    assert sum(len(v['observations']) for a in arms for v in a['views'].values())==96
    assert got['fixed_calls']['mp4_read']==24
    assert (tmp_path/'fail/environment.json').exists()

def test_runner_success_calibrates_before_eval_and_independent_views(monkeypatch,tmp_path):
    cfg=run.load(run.MANIFEST);case_map={c['id']:c for c in cfg['cases']};order=[]
    def child(command,log):
        cid=command[command.index('--case-id')+1];stage=command[command.index('--stage')+1];p=Path(command[command.index('--output')+1]);p.mkdir(parents=True,exist_ok=True);case=case_map[cid];record=run.empty_case(case);order.append((cid,stage))
        if stage=='generate':record['status']='GENERATION_COMPLETE';name='generation.json'
        else:
            if case['role']=='evaluation':assert run.load(p.parent/'calibration.json')['status']=='FROZEN'
            record['status']='EXECUTION_COMPLETE';name='result.json'
            for arm,item in record['videos'].items():item['views']={v:complete(.1 if arm=='OFF' else (.5 if v=='FULL' else .05)) for v in run.VIEWS}
        run.dump(p/name,record);return 0
    monkeypatch.setattr(run,'_run_child',child)
    got=run.run_all(run.MANIFEST,tmp_path/'ok');assert got['status']=='EXECUTION_COMPLETE'
    item=got['cases']['eval_p2_s2']['videos']['SINGLE46']
    assert item['decision']['status']=='DETECTED'
    assert item['views']['FULL']['decision']['status']=='DETECTED'
    assert item['views']['DELETE90']['decision']['status']=='REJECTED'
    assert got['cases']['cal_off_p0_s1']['videos']['OFF']['decision_role']=='THRESHOLD_CONSTRUCTION_SAMPLE_NOT_HELDOUT_FPR'
    with pytest.raises(FileExistsError):run.run_all(run.MANIFEST,tmp_path/'ok')

def test_notebook_schema_ast_reproducible_and_n5_install_args(monkeypatch,tmp_path):
    from scripts import build_flow_fixed_key_notebook as build
    p=build.OUTPUT;before=p.read_bytes();build.build();assert p.read_bytes()==before
    nb=json.loads(before)
    try:
        import nbformat
        nbformat.validate(nb)
    except ImportError:pass
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')"
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['execution_count'] is None and not c['outputs']
    commands=[]
    class Child:
        def __init__(self,command,**kwargs):commands.append(command);self.stdout=iter(['stub output\n'])
        def wait(self):return 0
    monkeypatch.setattr(subprocess,'Popen',Child)
    monkeypatch.setattr(importlib.metadata,'version',lambda name:'old')
    env={'OUTPUT':tmp_path};exec(''.join(nb['cells'][3]['source']),env)
    pip=[x for x in commands if 'install' in x and 'pip' in x]
    assert pip[0][4:]==['torch==2.11.0','torchvision','--index-url','https://download.pytorch.org/whl/cu128']
    assert pip[1][4:]==['diffusers==0.40.0','transformers','accelerate','ftfy','sentencepiece','safetensors','huggingface_hub','numpy','Pillow']
    assert '-q' not in pip[1];ast.parse(commands[-1][-1]);assert 'environment_setup.json' in commands[-1][-1]
    assert (tmp_path/'setup.log').read_text().count('stub output')==len(commands)
