import ast,json,gzip
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import zero_mean_raw420_forward_accept_run as runner
from main.tube_state import video_overlap_zero_mean_state as state,zero_mean_channel_margin as method
pytestmark=pytest.mark.unit

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

@pytest.mark.parametrize('before,after,status',[(1e-6,1e-6,'REJECTED'),(1e-6,0.,'REJECTED'),(-2e-6,-1e-6,'ACCEPTED'),(1e-6,float('nan'),'UNDECIDED')])
def test_predeclared_strict_rule(tmp_path,before,after,status):
    run=runner.Run(tmp_path/'out',json.loads(runner.CONFIG.read_text()))
    run.data['baseline_terminal']={'path':'baseline'}
    run.data['updates']['PROPOSAL'].update(status='SAVED',terminal={'path':'proposal'})
    for point,value in [('BASELINE',before),('PROPOSAL',after)]:
        run.data['posthoc'][point+'_RAW420/CORRECT']=dict(status='EVALUATED',delta=value,true_rank=2)
        run.data['reads'][point+'_RAW420/CORRECT'].update(status='SAVED',scored=174)
    runner.decide(run)
    assert run.data['acceptance']['status']==status
    assert run.data['retained_endpoint']['point']==('PROPOSAL' if status=='ACCEPTED' else 'BASELINE')
    assert run.data['acceptance']['scientific_pass'] is False
    assert run.data['retained_endpoint']['posthoc']['true_rank']==2

@pytest.mark.parametrize('gain,expected',[(1.,'REJECTED'),(1.01,'ACCEPTED'),(.99,'REJECTED')])
def test_full_proposal_and_retained_artifacts(simulated,tmp_path,monkeypatch,gain,expected):
    run,decoded,enc,dec,channels=simulated;original=runner.media.encode_normalized;calls=[]
    def encode(*a):
        calls.append(1);return original(*a)*(gain if len(calls)>=3 else 1.)
    monkeypatch.setattr(runner.media,'encode_normalized',encode)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE' and run.data['acceptance']['status']==expected
    assert run.data['counts']==dict(observations=4,proposals=1,accepted_updates=int(expected=='ACCEPTED'),path_reads=8,payload_reads=8,message_evaluations=16,valid_costs=1392)
    assert len(decoded)==2 and len(enc)==len(dec)==1 and len(channels)==4
    for k,n in run.cfg['planned_calls'].items():
        if '_chunk_' not in k:assert run.data['calls'][k]==dict(attempted=n,completed=n)
    proposal=run.data['updates']['PROPOSAL'];assert Path(proposal['terminal']['path']).is_file()
    assert Path(proposal['gradient']['path']).is_file() and Path(proposal['step']['path']).is_file()
    assert torch.equal(decoded[0]+torch.load(proposal['step']['path'],weights_only=True),decoded[1])
    expected_terminal=proposal['terminal'] if expected=='ACCEPTED' else run.data['baseline_terminal']
    assert run.data['retained_endpoint']['terminal']==expected_terminal
    for row in run.data['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert not raw['truth_used'] and not raw['writer_inputs'] and len(raw['inference']['path_costs'])==174

def test_failed_primary_is_undecided_with_baseline_retained(simulated,tmp_path,monkeypatch):
    run,*_=simulated;original=run.transport
    def transport(obs,q):
        if obs=='PROPOSAL_RAW420':raise RuntimeError('simulated primary failure')
        return original(obs,q)
    monkeypatch.setattr(run,'transport',transport)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_PARTIAL' and run.data['acceptance']['status']=='UNDECIDED'
    assert run.data['retained_endpoint']['point']=='BASELINE'
    assert run.data['observations']['PROPOSAL_RAW444']['status']=='SAVED'
    assert run.data['counts']['proposals']==1 and run.data['counts']['accepted_updates']==0
    assert len(run.data['message_evaluations'])==16

def test_failed_contrast_keeps_primary_decision(simulated,tmp_path,monkeypatch):
    run,*_=simulated;original=run.transport
    def transport(obs,q):
        if obs=='BASELINE_RAW444':raise RuntimeError('simulated contrast failure')
        return original(obs,q)
    monkeypatch.setattr(run,'transport',transport)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_PARTIAL'
    assert run.data['acceptance']['status']=='REJECTED'  # equal raw420 scores still decide
    assert run.data['reads']['PROPOSAL_RAW420/CORRECT']['status']=='SAVED'
    assert run.data['counts']['proposals']==1

def test_incomplete_search_cannot_accept(tmp_path):
    run=runner.Run(tmp_path/'out',json.loads(runner.CONFIG.read_text()));runner.finalize(run)
    assert run.data['acceptance']['status']=='UNDECIDED' and run.data['retained_endpoint']['status']=='NOT_COMPLETED'

def test_notebook_is_pinned_and_reports_acceptance(tmp_path):
    from scripts.build_zero_mean_raw420_forward_accept_notebook import build
    cfg=json.loads(runner.CONFIG.read_text());nb=json.loads(build('a'*40,tmp_path/'nb.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    text=''.join(''.join(c['source']) for c in nb['cells'])
    assert cfg['input']['root'] in text and 'zero_mean_raw420_forward_accept_run' in text
    assert "result['acceptance']" in text and "result['retained_endpoint']" in text
    assert nb['metadata']['candidate_binding']['source_sha']=='a'*40
    assert cfg['arms']=={'PROPOSAL':'composite'} and cfg['input']['terminal_relative_path']=='COMPOSITE/terminal.pt'
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
