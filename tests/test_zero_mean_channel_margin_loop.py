import ast,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import zero_mean_channel_margin_loop_run as loop
from main.tube_state import video_overlap_zero_mean_state as state
pytestmark=pytest.mark.unit


def install_fake_channel(monkeypatch,run,*,zero=False,fail_decoder=None):
    target=torch.tensor(state.synthesize('watermark'),dtype=torch.float32)
    original=torch.zeros_like(target);start=target*.2;tracking=dict(current=start.clone(),sources=[])
    def rgb_of(z):return torch.full((1,2,2,3),.1+float(z.double().norm())/2)
    def qrgb(z):return torch.from_numpy(np.rint(rgb_of(z).numpy()*255).astype(np.uint8)).float()/255
    def inputs(*args):
        run.data['parent_reference']=dict(baseline_step_l2=.2,step1_rgb_sha256=hashlib.sha256((qrgb(start)*255).to(torch.uint8).numpy().tobytes()).hexdigest())
        return original,start.clone(),start.clone(),dict(BEFORE=qrgb(original),AFTER=qrgb(start)),{}
    monkeypatch.setattr(loop,'load_inputs',inputs)
    import runtime.wan.generation as generation
    dummy=SimpleNamespace(parameters=lambda:iter([torch.zeros(1)]),use_tiling=False,config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16))
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:dummy)
    def decode(vae,z):tracking['current']=z.clone();return rgb_of(z),dict(fixture=True)
    monkeypatch.setattr(loop.media,'decode_with_clamp_receipt',decode)
    def roundtrip(q8,yuv,rgb,*,count,event):
        for name,path,kind in [('yuv444',yuv,'rgb_to_raw444'),('rgb24',rgb,'raw444_to_rgb24')]:
            count(kind,False);path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(q8.numpy().tobytes())
            event(name,dict(status='SAVED',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),path=str(path)))
            count(kind,True)
        return q8.clone()
    monkeypatch.setattr(loop.color,'roundtrip',roundtrip)
    def account(count,component):
        for phase in ['forward','recompute']:
            for _ in range(46):count(component+'_chunk_'+phase,False);count(component+'_chunk_'+phase,True)
    def encoder(vae,rgb,key,count,receipt):
        count('vae_encode_gradient',False);z=tracking['current'].clone();count('vae_encode_gradient',True)
        count('encoder_vjp',False);account(count,'encoder');count('encoder_vjp',True)
        receipt.update(fixture=True);_,loss=loop.method.margin_loss(loop.method.project(z,key),key)
        return z,torch.zeros_like(rgb),loss
    monkeypatch.setattr(loop.gradient,'encoder_cotangent',encoder)
    def decoder(vae,z,cot,rgb,count,receipt):
        tracking['sources'].append(z.clone())
        if fail_decoder==len(tracking['sources']):raise RuntimeError('injected decoder failure')
        count('vae_decode_gradient',False);count('vae_decode_gradient',True)
        count('decoder_vjp',False);account(count,'decoder');count('decoder_vjp',True)
        receipt.update(fixture=True)
        return torch.zeros_like(z) if zero else -(target-z)*.005
    monkeypatch.setattr(loop.gradient,'decoder_vjp',decoder)
    monkeypatch.setattr(loop.media,'encode_normalized',lambda *a:tracking['current'].clone())
    return tracking,start,target


def make_run(tmp_path):return loop.Run(tmp_path/'out',json.loads(loop.CONFIG.read_text()))


def test_fixed_four_points_fresh_gradients_and_final_endpoint(tmp_path,monkeypatch):
    run=make_run(tmp_path);tracking,start,target=install_fake_channel(monkeypatch,run)
    loop.execute(run,tmp_path,tmp_path,tmp_path);run.finalize();r=run.data
    assert r['status']=='EXECUTION_COMPLETE'
    assert r['counts']==dict(observations=4,updates=3,path_reads=8,message_evaluations=16,valid_costs=1392)
    assert len(tracking['sources'])==3
    assert torch.equal(tracking['sources'][0],start)
    assert not torch.equal(tracking['sources'][0],tracking['sources'][1])
    assert not torch.equal(tracking['sources'][1],tracking['sources'][2])
    assert r['posthoc']['POINT2/CORRECT']['true_rank']==1
    assert r['fixed_endpoint']['point']=='POINT4' and not r['fixed_endpoint']['best_iterate_selected']
    for key,n in run.cfg['planned_calls'].items():assert r['calls'][key]==dict(attempted=n,completed=n)
    assert r['baseline_replay']==dict(normalized_max_error=0.,rgb_sha_matches_parent=True)
    for i,name in enumerate(loop.TRANSITIONS):
        u=r['updates'][name]
        before=torch.load(r['observations'][loop.POINTS[i]]['terminal']['path'],weights_only=True)
        after=torch.load(u['terminal']['path'],weights_only=True)
        step=torch.load(u['step']['path'],weights_only=True)
        assert torch.equal(after,before+step)
        assert loop.media.file_sha256(u['terminal']['path'])==u['terminal']['sha256']
        assert u['original_terminal_displacement_l2']==pytest.approx(float(after.double().norm()))
        assert 'sum_new_step_l2_squared' in u and u['sum_new_step_l2']<=3.000001
    for row in r['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert not raw['truth_used'] and not raw['writer_inputs']
        assert raw['inference']['counts']['scored']==174
    for p in loop.POINTS:
        assert all(k in r['observations'][p] for k in ['quality_vs_original_marked','quality_vs_step1','quality_vs_previous_point'])


def test_zero_gradient_retains_all_fixed_iterations(tmp_path,monkeypatch):
    run=make_run(tmp_path);tracking,start,_=install_fake_channel(monkeypatch,run,zero=True)
    loop.execute(run,tmp_path,tmp_path,tmp_path);run.finalize()
    assert run.data['counts']['updates']==3 and len(tracking['sources'])==3
    assert all(torch.equal(v,start) for v in tracking['sources'])
    for u in run.data['updates'].values():
        assert u['receipt']['reason']=='ZERO_SUPPORTED_GRADIENT_NOOP' and u['actual_change_l2']==0
    assert run.data['fixed_endpoint']['point']=='POINT4'


def test_failed_second_gradient_retains_prior_and_current_readouts(tmp_path,monkeypatch):
    run=make_run(tmp_path);install_fake_channel(monkeypatch,run,fail_decoder=2)
    try:loop.execute(run,tmp_path,tmp_path,tmp_path)
    except RuntimeError as exc:run.fail(exc)
    else:pytest.fail('expected fixture failure')
    run.finalize();r=run.data
    assert r['status']=='EXECUTION_FAILED' and r['counts']['updates']==1
    assert r['counts']['observations']==2 and r['counts']['path_reads']==4 and r['counts']['valid_costs']==696
    assert r['updates']['POINT2_TO_POINT3']['status']=='FAILED'
    assert r['updates']['POINT3_TO_POINT4']['status']=='NOT_COMPLETED'
    assert r['observations']['POINT3']['status']=='NOT_COMPLETED'
    assert r['message_evaluations']['POINT4/WRONG/WRONG_MESSAGE']['status']=='NOT_COMPLETED'
    assert len(r['reads'])==8 and len(r['message_evaluations'])==16
    assert 'fixed_endpoint' not in r


def test_noop_does_not_hide_nonfinite_gradients():
    z=torch.zeros(1,16,46,40,64);z[0,0,0,0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):loop.apply_update(z,'watermark',174.,1.)


def test_notebook_parent_binding_and_fixed_roster(tmp_path):
    from scripts.build_zero_mean_channel_margin_loop_notebook import build
    for sha in [None,'a'*40]:
        nb=json.loads(build(sha,tmp_path/'loop.ipynb').read_text())
        assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
        assert nb['metadata']['candidate_binding']['source_sha']==sha
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[]
        setup=''.join(nb['cells'][2]['source']);run=''.join(nb['cells'][4]['source'])
        assert "INPUT_ROOT=Path('/content/drive/MyDrive/Video-WM/Zero-Mean-Channel-Margin-V1/20261001T133512881485Z/fixed_reference')" in setup
        assert "OUTPUT=Path('/content/drive/MyDrive/Video-WM/Zero-Mean-Channel-Margin-Loop-V1')" in setup
        assert 'zero_mean_channel_margin_loop_run' in run and '--reference420-root' not in run
        assert 'new_updates' in setup and 'POINT4' in setup
