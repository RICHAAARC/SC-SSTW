import ast,json
from pathlib import Path
from collections import Counter
import numpy as np
import pytest
import torch
from main.tube_state import zero_mean_channel_margin as m,video_overlap_zero_mean_state as state
from runtime.wan.zero_mean_channel_margin import checkpoint_native

pytestmark=pytest.mark.unit
KEY='watermark'

def test_full_family_margins_and_gradient():
    spec=m.objective_spec(KEY)
    assert len(spec['wrong_classes'])==173 and len(spec['valid_catalog_indices'])==174
    rows=state.catalog(44)
    assert {rows[i]['event_type'] for i in spec['valid_catalog_indices']}=={'ZERO_EDIT','REPEAT','SKIP'}
    q=torch.tensor(np.random.default_rng(2101).normal(0,.1,1408),requires_grad=True,dtype=torch.float64)
    loss,info=m.margin_loss(q,KEY)
    means=spec['means'];costs=((means-q.detach().numpy())**2).mean(1)
    expected=costs[spec['wrong_classes']]-costs[spec['true_class']]
    np.testing.assert_allclose(info['margins'],expected,rtol=1e-12,atol=1e-16)
    grad,=torch.autograd.grad(loss,q)
    direction=torch.tensor(np.random.default_rng(2102).normal(size=1408))
    epsilon=1e-6
    plus=m.margin_loss(q.detach()+epsilon*direction,KEY)[0]
    minus=m.margin_loss(q.detach()-epsilon*direction,KEY)[0]
    assert float((plus-minus)/(2*epsilon))==pytest.approx(float(grad@direction),rel=1e-7,abs=1e-12)
    perfect=torch.tensor(means[spec['true_class']]);assert m.margin_loss(perfect,KEY)[0].item()==0


def test_step_is_same_support_cap_and_descent():
    g=torch.tensor(np.random.default_rng(19).normal(size=(1,16,46,40,64)),dtype=torch.float32)
    step,receipt=m.constrained_step(g,KEY)
    assert receipt['actual_l2']<=1.0000001 and receipt['gradient_dot_step']<0
    allowed=torch.zeros_like(step,dtype=torch.bool)
    for h0,h1,w0,w1 in state.PUBLIC.blocks:allowed[0,4,1:,h0:h1,w0:w1]=True
    assert torch.count_nonzero(step[~allowed])==0
    for i,(h0,h1,w0,w1) in enumerate(state.PUBLIC.blocks):
        patch=step[0,4,1:,h0:h1,w0:w1].double().reshape(45,16)
        U=torch.tensor(state.bases(KEY)[i]);q=patch@U
        assert float((patch-q@U.T).abs().max())<1e-8
        active=torch.tensor(state.composite_signs(KEY)[:,i].reshape(45,8)!=0)
        assert float(q[~active].abs().max())<1e-8
    with pytest.raises(ValueError,match='zero supported'):m.constrained_step(torch.zeros_like(g),KEY)


def tiny_vae():
    from diffusers import AutoencoderKLWan
    torch.manual_seed(33)
    vae=AutoencoderKLWan(base_dim=4,z_dim=16,dim_mult=[1,2,2,2],num_res_blocks=1,
                         latents_mean=[0.]*16,latents_std=[1.]*16).eval()
    for p in vae.parameters():p.requires_grad_(False)
    return vae

@pytest.mark.parametrize('component',['encoder','decoder'])
def test_native_wan_checkpoint_value_gradient_and_restore(component):
    vae=tiny_vae();shape=(1,3,17,16,16) if component=='encoder' else (1,16,5,2,2)
    torch.manual_seed(91);x=(torch.randn(shape)*.1).requires_grad_(True)
    def call(v):return vae.encode(v).latent_dist.mode() if component=='encoder' else vae.decode(v).sample
    y=call(x);probe=torch.linspace(-.2,.3,y.numel()).reshape(y.shape)
    normal_grad,=torch.autograd.grad((y*probe).sum(),x)
    original=getattr(vae,component).forward;counts=Counter();receipt={}
    def count(k,done):counts[k,done]+=1
    leaf=x.detach().clone().requires_grad_(True)
    with checkpoint_native(vae,component,count,receipt):
        replay=call(leaf);actual,=torch.autograd.grad((replay*probe).sum(),leaf)
    assert getattr(vae,component).forward==original
    torch.testing.assert_close(replay,y,rtol=0,atol=0)
    torch.testing.assert_close(actual,normal_grad,rtol=1e-5,atol=1e-7)
    assert receipt['chunks']==5
    assert counts[component+'_chunk_forward',True]==5
    assert counts[component+'_chunk_recompute',True]==5
    assert receipt['storage_cleanup_complete']


def test_split_native_vjp_matches_joined_chain():
    vae=tiny_vae();torch.manual_seed(71);z=(torch.randn(1,16,5,2,2)*.1).requires_grad_(True)
    def decode(v):return (vae.decode(v).sample/2+.5).clamp(0,1)
    def encode(v):return vae.encode(v*2-1).latent_dist.mode()
    rgb=decode(z);out=encode(rgb);joined,=torch.autograd.grad(out.square().mean(),z)
    rgb_leaf=rgb.detach().requires_grad_(True)
    with checkpoint_native(vae,'encoder',lambda *x:None,{}):
        enc=encode(rgb_leaf);cotangent,=torch.autograd.grad(enc.square().mean(),rgb_leaf)
    zz=z.detach().requires_grad_(True)
    with checkpoint_native(vae,'decoder',lambda *x:None,{}):
        dec=decode(zz);split,=torch.autograd.grad(dec,zz,grad_outputs=cotangent)
    torch.testing.assert_close(split,joined,rtol=2e-5,atol=1e-8)


def test_checkpoint_exception_restores_instance():
    vae=tiny_vae();original=vae.encoder.forward
    with pytest.raises(RuntimeError,match='fixture'):
        with checkpoint_native(vae,'encoder',lambda *x:None,{}):raise RuntimeError('fixture')
    assert vae.encoder.forward==original


def test_raw_persists_before_reporting_failure(tmp_path,monkeypatch):
    from experiments.wan_state_clock import zero_mean_channel_margin_run as runner
    cfg=json.loads(runner.CONFIG.read_text());run=runner.Run(tmp_path/'out',cfg)
    z=torch.zeros(1,16,46,40,64)
    def fail(_):raise RuntimeError('posthoc fixture failure')
    monkeypatch.setattr(runner,'posthoc',fail)
    with pytest.raises(RuntimeError,match='posthoc fixture'):
        run.read('BEFORE',z)
    saved=json.loads((tmp_path/'out/result.json').read_text())
    row=saved['reads']['BEFORE/CORRECT']
    assert row['status']=='SAVED' and row['scored']==174 and Path(row['path']).is_file()
    assert len(saved['reads'])==4 and len(saved['message_evaluations'])==8
    assert saved['reads']['AFTER/WRONG']['status']=='PENDING'


def test_draft_notebook_is_structurally_valid_and_fixed(tmp_path):
    from scripts.build_zero_mean_channel_margin_notebook import build
    p=build(output=tmp_path/'draft.ipynb');nb=json.loads(p.read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert nb['metadata']['candidate_binding']['source_sha'] is None
    assert nb['metadata']['candidate_binding']['status']=='UNPUBLISHED_DRAFT'
    text=''.join(''.join(c['source']) for c in nb['cells'])
    assert 'zero_mean_channel_margin_run' in text
    assert '--reference-root' in text and '--reference420-root' not in text
    assert 'ensurepip' not in text and 'venv' not in text


def test_receiver_roster_and_truth_join_counts(tmp_path):
    import gzip
    from experiments.wan_state_clock import zero_mean_channel_margin_run as runner
    cfg=json.loads(runner.CONFIG.read_text());run=runner.Run(tmp_path/'out',cfg)
    z=torch.tensor(state.synthesize(KEY),dtype=torch.float32)
    for obs in runner.OBS:run.read(obs,z)
    result=json.loads((tmp_path/'out/result.json').read_text())
    assert sum(v['scored'] for v in result['reads'].values())==696
    assert all(v['status']=='EVALUATED' for v in result['message_evaluations'].values())
    assert result['posthoc']['BEFORE/CORRECT']['true_rank']==1
    assert result['calls']['path_read']==dict(attempted=4,completed=4)
    assert result['calls']['payload_read']==dict(attempted=4,completed=4)
    for row in result['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert not raw['truth_used'] and not raw['writer_inputs']
        assert 'posthoc' not in raw and 'writer_loss' not in raw
