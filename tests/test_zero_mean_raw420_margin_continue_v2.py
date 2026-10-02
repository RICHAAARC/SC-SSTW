import ast,json,gzip
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.wan_state_clock import zero_mean_raw420_margin_continue_v2_run as runner
from main.tube_state import video_overlap_zero_mean_state as state
pytestmark=pytest.mark.unit

@pytest.fixture
def simulated(tmp_path,monkeypatch):
    cfg=json.loads(runner.CONFIG.read_text());run=runner.Run(tmp_path/'out',cfg)
    z=torch.tensor(state.synthesize(cfg['key']),dtype=torch.float32)
    old=tmp_path/'parent.pt';torch.save(z,old)
    def inputs(*args):
        run.data['parent_references']={ch:dict(normalized=dict(path=str(old)),reads={'CORRECT':dict(posthoc={})}) for ch in ('RAW420','RAW444')}
        run.data['prior_raw420_step_l2']=.14719737679731978
        return z.clone()
    monkeypatch.setattr(runner,'inputs',inputs)
    from runtime.wan import generation
    class VAE:
        use_tiling=False
        config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def parameters(self):return iter([torch.zeros(1)])
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:VAE())
    decode_inputs=[]
    def decode(vae,terminal):
        decode_inputs.append(terminal.clone());return torch.ones(2,2,2,3)*.5,{}
    monkeypatch.setattr(runner.media,'decode_with_clamp_receipt',decode)
    channels=[]
    def make_transport(channel):
        def transport(q,yuv,rgb,*,count,event):
            channels.append((channel,q.clone()))
            for kind in ('rgb_to_raw'+channel,'raw'+channel+'_to_rgb24'):
                count(kind,False);count(kind,True)
            event('rgb24',dict(status='SIMULATED'))
            return q.clone() if channel=='444' else torch.clamp(q.to(torch.int16)-1,0,255).to(torch.uint8)
        return transport
    monkeypatch.setattr(runner.single.raw420,'roundtrip',make_transport('420'))
    monkeypatch.setattr(runner.single.raw444,'roundtrip',make_transport('444'))
    enc_inputs=[];dec_inputs=[]
    def encoder(v,rgb,key,count,receipt):
        enc_inputs.append(rgb.clone())
        for kind in ('vae_encode_gradient','encoder_vjp'):
            count(kind,False);count(kind,True)
        return z.clone(),torch.ones_like(rgb),{}
    def decoder(v,terminal,cotangent,rgb,count,receipt):
        dec_inputs.append(terminal.clone())
        for kind in ('vae_decode_gradient','decoder_vjp'):
            count(kind,False);count(kind,True)
        torch.manual_seed(31+len(dec_inputs));return torch.randn_like(terminal)*1e-3
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',encoder)
    monkeypatch.setattr(runner.gradient,'decoder_vjp',decoder)
    monkeypatch.setattr(runner.media,'encode_normalized',lambda *a:z.clone())
    return run,channels,decode_inputs,enc_inputs,dec_inputs

def test_three_fresh_updates_even_when_first_point_already_rank1(simulated,tmp_path):
    run,channels,decode_inputs,enc_inputs,dec_inputs=simulated
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE'
    assert run.data['counts']==dict(observations=8,updates=3,path_reads=16,payload_reads=16,message_evaluations=32,valid_costs=2784)
    assert len(enc_inputs)==len(dec_inputs)==3 and len(decode_inputs)==4
    assert [c for c,q in channels]==['420','444']*4
    for i in range(4):assert torch.equal(channels[2*i][1],channels[2*i+1][1])
    for i in range(3):
        torch.testing.assert_close(enc_inputs[i],(channels[2*i][1].to(torch.int16)-1).float()/255)
        assert torch.equal(dec_inputs[i],decode_inputs[i])
        row=run.data['updates'][f'POINT{i}_TO_POINT{i+1}']
        step=torch.load(row['step']['path'],weights_only=True)
        assert torch.equal(decode_inputs[i]+step,decode_inputs[i+1])
        assert runner.media.file_sha256(row['terminal']['path'])==row['terminal']['sha256']
        assert row['raw420_update_index']==5+i
        assert row['sum_raw420_actual_step_l2']==row['sum_new_actual_step_l2']+.14719737679731978
    for k,n in run.cfg['planned_calls'].items():
        if '_chunk_' not in k:assert run.data['calls'][k]==dict(attempted=n,completed=n)
    assert run.data['fixed_endpoint']['point']=='POINT3' and not run.data['fixed_endpoint']['selected_by_score']
    assert run.data['posthoc']['POINT0_RAW420/CORRECT']['truth_unique_top']
    assert len(run.data['boundary_diagnostics'])==8
    for obs,bd in run.data['boundary_diagnostics'].items():
        raw=json.loads(gzip.decompress(Path(run.data['reads'][obs+'/CORRECT']['path']).read_bytes()))['inference']
        for cid in (86,88):assert bd[str(cid)]['delta']==raw['path_costs'][raw['valid_catalog_indices'].index(cid)]-raw['path_costs'][0]
    for row in run.data['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert len(raw['inference']['path_costs'])==174 and not raw['truth_used'] and not raw['writer_inputs']

def test_failed_anchor_retained_without_stopping_primary(simulated,tmp_path,monkeypatch):
    run,*_=simulated;original=run.transport
    def transport(obs,q):
        if obs=='POINT1_RAW444':raise RuntimeError('simulated anchor failure')
        return original(obs,q)
    monkeypatch.setattr(run,'transport',transport)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_PARTIAL' and run.data['counts']['updates']==3
    assert run.data['counts']['observations']==7 and run.data['counts']['valid_costs']==2436
    assert run.data['reads']['POINT1_RAW444/CORRECT']['status']=='NOT_COMPLETED'
    assert run.data['posthoc']['POINT3_RAW420/CORRECT']['status']=='EVALUATED'
    assert run.data['observations']['POINT2_RAW444']['quality_previous_point']=='POINT0'

def test_failed_update_dependency_retains_future_rows(simulated,tmp_path,monkeypatch):
    run,*_=simulated;original=runner.gradient.encoder_cotangent;calls=0
    def encoder(*a,**kw):
        nonlocal calls
        calls+=1
        if calls==2:raise RuntimeError('simulated next encoder failure')
        return original(*a,**kw)
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',encoder)
    with pytest.raises(RuntimeError,match='next encoder'):runner.execute(run,tmp_path,tmp_path)
    runner.finalize(run)
    assert len(run.data['updates'])==3 and run.data['counts']['updates']==1
    assert run.data['updates']['POINT1_TO_POINT2']['status']=='NOT_COMPLETED'
    assert len(run.data['message_evaluations'])==32
    assert run.data['fixed_endpoint']['status']=='NOT_COMPLETED'
    assert run.data['reads']['POINT3_RAW420/CORRECT']['status']=='NOT_COMPLETED'

def test_notebook_input_binding_and_fixed_roster(tmp_path):
    from scripts.build_zero_mean_raw420_margin_continue_v2_notebook import build
    cfg=json.loads(runner.CONFIG.read_text());p=build('a'*40,tmp_path/'nb.ipynb');nb=json.loads(p.read_text())
    assert cfg['input']['terminal_relative_path']=='POINT3/terminal.pt' and cfg['update']['steps']==3
    assert cfg['input']['result_sha256']=='caf59e634f266b043609d5eb40ec4e707ff655435ec10604cea3043a133bacda'
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    text=''.join(''.join(c['source']) for c in nb['cells'])
    assert cfg['input']['root'] in text and 'zero_mean_raw420_margin_continue_v2_run' in text
    assert 'Zero-Mean-Raw420-Margin-Continue-V2' in text and 'ensurepip' not in text
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert nb['metadata']['candidate_binding']['source_sha']=='a'*40
