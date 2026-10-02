import ast,json,gzip
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import zero_mean_raw420_objective_ablation_run as runner
from main.tube_state import video_overlap_zero_mean_state as state,zero_mean_channel_margin as method
pytestmark=pytest.mark.unit

@pytest.mark.parametrize('seed',[9,17,31])
def test_worst_only_matches_full_family_minimum_and_finite_difference(seed):
    key='watermark';spec=method.objective_spec(key)
    q=torch.tensor(np.random.default_rng(seed).normal(0,.1,1408),requires_grad=True,dtype=torch.float64)
    loss,metrics=method.margin_loss(q,key,objective='worst_only')
    means=spec['means'];cost=((means-q.detach().numpy())**2).mean(1)
    margins=cost[spec['wrong_classes']]-cost[spec['true_class']]
    assert len(margins)==173 and loss.item()==pytest.approx(max(0,spec['global_margin']-margins.min()),abs=1e-15)
    grad,=torch.autograd.grad(loss,q);worst=spec['wrong_classes'][int(np.argmin(margins))]
    np.testing.assert_allclose(grad.numpy(),2*(means[worst]-means[spec['true_class']])/1408,atol=1e-15)
    direction=torch.tensor(np.random.default_rng(seed+1).normal(size=1408));eps=1e-6
    plus=method.margin_loss(q.detach()+eps*direction,key,objective='worst_only')[0]
    minus=method.margin_loss(q.detach()-eps*direction,key,objective='worst_only')[0]
    assert float((plus-minus)/(2*eps))==pytest.approx(float(grad@direction),rel=1e-7,abs=1e-12)
    composite,full=method.margin_loss(q,key)
    assert composite.item()==pytest.approx(loss.item()+metrics['local_mean'])
    assert metrics['margins']==full['margins']
    with pytest.raises(ValueError):method.margin_loss(q,key,objective='candidate88')

@pytest.mark.parametrize('objective',['composite','worst_only'])
def test_native_encoder_routes_selected_objective_and_gradient(monkeypatch,objective):
    from diffusers import AutoencoderKLWan
    from runtime.wan import zero_mean_channel_margin as rt
    torch.manual_seed(33)
    vae=AutoencoderKLWan(base_dim=4,z_dim=16,dim_mult=[1,2,2,2],num_res_blocks=1,latents_mean=[0.]*16,latents_std=[1.]*16).eval()
    for p in vae.parameters():p.requires_grad_(False)
    def project(z,key):return z.flatten().repeat(5)[:1408]
    monkeypatch.setattr(method,'project',project)
    rgb=torch.rand(17,16,16,3);leaf=rgb.clone().requires_grad_(True)
    raw=vae.encode(leaf.permute(3,0,1,2).unsqueeze(0)*2-1).latent_dist.mode()
    loss,_=method.margin_loss(project(raw,'watermark'),'watermark',objective=objective)
    expected,=torch.autograd.grad(loss,leaf)
    z,actual,metrics=rt.encoder_cotangent(vae,rgb,'watermark',lambda *a:None,{},objective=objective)
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-8)
    assert metrics['loss']==pytest.approx(float(loss))

@pytest.fixture
def simulated(tmp_path,monkeypatch):
    cfg=json.loads(runner.CONFIG.read_text());run=runner.Run(tmp_path/'out',cfg)
    z=torch.tensor(state.synthesize(cfg['key']),dtype=torch.float32);old=tmp_path/'parent.pt';torch.save(z,old)
    def inputs(*args):
        run.data['parent_references']={ch:dict(normalized=dict(path=str(old)),reads={'CORRECT':dict(posthoc={})}) for ch in ('RAW420','RAW444')}
        run.data['prior_raw420_step_l2']=.2006495665328446;run.data['writer_input']=dict(sha256='simulated-shared-source')
        return z.clone()
    monkeypatch.setattr(runner,'inputs',inputs)
    from runtime.wan import generation
    class VAE:
        use_tiling=False
        config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def parameters(self):return iter([torch.zeros(1)])
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:VAE())
    decoded=[]
    def decode(vae,terminal):decoded.append(terminal.clone());return torch.ones(2,2,2,3)*.5,{}
    monkeypatch.setattr(runner.media,'decode_with_clamp_receipt',decode)
    channels=[]
    def transport(ch):
        def f(q,yuv,rgb,*,count,event):
            channels.append((ch,q.clone()))
            for kind in ('rgb_to_raw'+ch,'raw'+ch+'_to_rgb24'):count(kind,False);count(kind,True)
            event('rgb24',dict(status='SIMULATED'));return q.clone()
        return f
    monkeypatch.setattr(runner.single.raw420,'roundtrip',transport('420'));monkeypatch.setattr(runner.single.raw444,'roundtrip',transport('444'))
    enc=[];dec=[]
    def encoder(v,rgb,key,count,receipt,*,objective):
        enc.append((objective,rgb.clone()))
        for kind in ('vae_encode_gradient','encoder_vjp'):count(kind,False);count(kind,True)
        return z.clone(),torch.ones_like(rgb),{}
    def decoder(v,terminal,cotangent,rgb,count,receipt):
        dec.append(terminal.clone())
        for kind in ('vae_decode_gradient','decoder_vjp'):count(kind,False);count(kind,True)
        torch.manual_seed(35+len(dec));return torch.randn_like(terminal)*1e-3
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',encoder);monkeypatch.setattr(runner.gradient,'decoder_vjp',decoder)
    monkeypatch.setattr(runner.media,'encode_normalized',lambda *a:z.clone())
    return run,decoded,enc,dec,channels

def test_shared_baseline_separate_arms_full_denominator(simulated,tmp_path):
    run,decoded,enc,dec,channels=simulated;runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE'
    assert run.data['counts']==dict(observations=6,updates=2,path_reads=12,payload_reads=12,message_evaluations=24,valid_costs=2088)
    assert [x[0] for x in enc]==['composite','worst_only'] and torch.equal(enc[0][1],enc[1][1])
    assert torch.equal(dec[0],decoded[0]) and torch.equal(dec[1],decoded[0])
    for i,arm in enumerate(run.cfg['arms']):
        row=run.data['updates'][arm];step=torch.load(row['step']['path'],weights_only=True)
        assert torch.equal(decoded[i+1],decoded[0]+step)
        assert row['shared_input_sha256']=='simulated-shared-source'
        assert not run.data['fixed_endpoints'][arm]['selected_by_score']
    for k,n in run.cfg['planned_calls'].items():
        if '_chunk_' not in k:assert run.data['calls'][k]==dict(attempted=n,completed=n)
    assert len(run.data['boundary_diagnostics'])==6
    for row in run.data['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert len(raw['inference']['path_costs'])==174 and not raw['truth_used'] and not raw['writer_inputs']

def test_one_failed_arm_does_not_erase_other_arm(simulated,tmp_path,monkeypatch):
    run,*_=simulated;original=runner.gradient.encoder_cotangent
    def encoder(*a,objective,**kw):
        if objective=='composite':raise RuntimeError('simulated composite failure')
        return original(*a,objective=objective,**kw)
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',encoder)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_PARTIAL' and run.data['counts']['updates']==1
    assert run.data['counts']['observations']==4 and run.data['counts']['valid_costs']==1392
    assert run.data['updates']['COMPOSITE']['status']=='FAILED'
    assert run.data['reads']['COMPOSITE_RAW420/CORRECT']['status']=='NOT_COMPLETED'
    assert run.data['fixed_endpoints']['WORST_ONLY']['status']=='SAVED'
    assert len(run.data['message_evaluations'])==24

def test_notebook_binding(tmp_path):
    from scripts.build_zero_mean_raw420_objective_ablation_notebook import build
    cfg=json.loads(runner.CONFIG.read_text());p=build('a'*40,tmp_path/'nb.ipynb');nb=json.loads(p.read_text())
    assert cfg['input']['terminal_relative_path']=='POINT3/terminal.pt' and cfg['update']['steps']==1
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    text=''.join(''.join(c['source']) for c in nb['cells'])
    assert cfg['input']['root'] in text and 'zero_mean_raw420_objective_ablation_run' in text
    assert 'Zero-Mean-Raw420-Objective-Ablation-V1' in text and 'ensurepip' not in text
    assert "result['fixed_endpoint']" not in text
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert nb['metadata']['candidate_binding']['source_sha']=='a'*40
