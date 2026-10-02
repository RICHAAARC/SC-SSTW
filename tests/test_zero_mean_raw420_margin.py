import ast,json,gzip
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.wan_state_clock import zero_mean_raw420_margin_run as runner
from main.tube_state import video_overlap_zero_mean_state as state
pytestmark=pytest.mark.unit

@pytest.fixture
def simulated(tmp_path,monkeypatch):
    cfg=json.loads(runner.CONFIG.read_text());run=runner.Run(tmp_path/'out',cfg)
    z=torch.tensor(state.synthesize(cfg['key']),dtype=torch.float32)
    monkeypatch.setattr(runner.base,'inputs',lambda *a:(z.clone(),{}))
    from runtime.wan import generation
    class VAE:
        use_tiling=False
        config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def parameters(self):return iter([torch.zeros(1)])
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:VAE())
    monkeypatch.setattr(runner.media,'decode_with_clamp_receipt',lambda *a:(torch.ones(2,2,2,3)*.5,{}))
    channels=[]
    def make_transport(channel):
        def transport(q,yuv,rgb,*,count,event):
            channels.append((channel,q.clone()))
            for kind in ('rgb_to_raw'+channel,'raw'+channel+'_to_rgb24'):
                count(kind,False);count(kind,True)
            event('rgb24',dict(status='SIMULATED'))
            return q.clone() if channel=='444' else torch.clamp(q.to(torch.int16)-1,0,255).to(torch.uint8)
        return transport
    monkeypatch.setattr(runner.raw420,'roundtrip',make_transport('420'))
    monkeypatch.setattr(runner.raw444,'roundtrip',make_transport('444'))
    encoder_inputs=[]
    def encoder(v,rgb,key,count,receipt):
        encoder_inputs.append(rgb.clone())
        for kind in ('vae_encode_gradient','encoder_vjp'):
            count(kind,False);count(kind,True)
        return z.clone(),torch.ones_like(rgb),{}
    def decoder(v,terminal,cotangent,rgb,count,receipt):
        for kind in ('vae_decode_gradient','decoder_vjp'):
            count(kind,False);count(kind,True)
        torch.manual_seed(31);return torch.randn_like(terminal)*1e-3
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',encoder)
    monkeypatch.setattr(runner.gradient,'decoder_vjp',decoder)
    monkeypatch.setattr(runner.media,'encode_normalized',lambda *a:z.clone())
    return run,channels,encoder_inputs

def test_one_step_uses_actual420_and_preserves_all_candidates(simulated,tmp_path):
    run,channels,encoder_inputs=simulated
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE'
    assert [c for c,q in channels]==['420','444','420','444']
    assert torch.equal(channels[0][1],channels[1][1]) and torch.equal(channels[2][1],channels[3][1])
    torch.testing.assert_close(encoder_inputs[0],(channels[0][1].to(torch.int16)-1).float()/255)
    assert run.data['counts']==dict(observations=4,updates=1,path_reads=8,payload_reads=8,message_evaluations=16,valid_costs=1392)
    for k,n in run.cfg['planned_calls'].items():
        if '_chunk_' not in k:assert run.data['calls'][k]==dict(attempted=n,completed=n)
    assert set(run.data['comparisons'])=={'RAW420','RAW444'}
    for row in run.data['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert len(raw['inference']['path_costs'])==174
        assert not raw['truth_used'] and not raw['writer_inputs']
    assert run.data['update']['receipt']['actual_l2']<=1.0000001

def test_failed_after420_keeps_after444_and_fixed_denominator(simulated,tmp_path,monkeypatch):
    run,_,_=simulated;original=run.transport
    def transport(obs,q):
        if obs=='AFTER_RAW420':raise RuntimeError('simulated physical channel failure')
        return original(obs,q)
    monkeypatch.setattr(run,'transport',transport)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_PARTIAL'
    assert run.data['observations']['AFTER_RAW420']['status']=='FAILED'
    assert run.data['observations']['AFTER_RAW444']['status']=='SAVED'
    assert len(run.data['reads'])==8 and len(run.data['message_evaluations'])==16
    assert run.data['reads']['AFTER_RAW420/CORRECT']['status']=='NOT_COMPLETED'
    assert run.data['message_evaluations']['AFTER_RAW420/CORRECT/REGISTERED']['status']=='NOT_COMPLETED'
    assert run.data['counts']['valid_costs']==1044

def test_fixed_input_and_notebook_binding(tmp_path):
    from scripts.build_zero_mean_raw420_margin_notebook import build
    cfg=json.loads(runner.CONFIG.read_text())
    assert cfg['input']['terminal_relative_path']=='POINT4/terminal.pt'
    assert cfg['update']['steps']==1 and cfg['update']['eta']==174 and cfg['update']['cap_l2']==1
    p=build('a'*40,tmp_path/'nb.ipynb');nb=json.loads(p.read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    text=''.join(''.join(c['source']) for c in nb['cells'])
    assert cfg['input']['root'] in text and 'zero_mean_raw420_margin_run' in text
    assert 'ensurepip' not in text and 'venv' not in text
    assert 'Zero-Mean-Raw420-Margin-V1' in text
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    assert nb['metadata']['candidate_binding']['source_sha']=='a'*40

def test_finalization_keeps_unattempted_rows(tmp_path):
    cfg=json.loads(runner.CONFIG.read_text());run=runner.Run(tmp_path/'out',cfg)
    run.data['status']='EXECUTION_FAILED';runner.finalize(run)
    assert run.data['counts']['valid_costs']==0 and len(run.data['reads'])==8
    for group in ('observations','reads','posthoc','message_evaluations'):
        assert all(row['status']=='NOT_COMPLETED' for row in run.data[group].values())
