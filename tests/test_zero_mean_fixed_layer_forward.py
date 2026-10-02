import ast,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.wan_state_clock import zero_mean_fixed_layer_forward_run as runner
from main.tube_state import video_overlap_zero_mean_state as state
pytestmark=pytest.mark.unit

@pytest.fixture
def simulated(tmp_path,monkeypatch):
    cfg=json.loads(runner.CONFIG.read_text());run=runner.Run(tmp_path/'out',cfg)
    z=torch.tensor(state.synthesize(cfg['key']),dtype=torch.float32)
    terminals={p:z+i*.001 for i,p in enumerate(cfg['points'])}
    images={p:torch.full((2,2,2,3),.501+i*.01) for i,p in enumerate(cfg['points'])}
    refs={}
    for point in cfg['points']:
        q8=runner.shared.quantize_rgb8_no_codec(images[point]);sha=hashlib.sha256(q8.numpy().tobytes()).hexdigest()
        for layer in ('RAW420','RAW444'):
            refs[point+'_'+layer]=dict(input_raster_sha256=sha,reads={'CORRECT':{},'WRONG':{}},posthoc={'CORRECT':dict(status='EVALUATED',delta=-1e-6,true_rank=2,truth_unique_top=False)},writer_loss=dict(loss=5e-6,global_term=3e-6,local_mean=2e-6))
    def inputs(*args):run.data['reference_observations']=refs;return terminals
    monkeypatch.setattr(runner,'inputs',inputs)
    from runtime.wan import generation
    class VAE:
        use_tiling=False
        config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def parameters(self):return iter([torch.zeros(1)])
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:VAE())
    decoded=[];encoded=[]
    def decode(v,t):
        point=next(k for k,val in terminals.items() if torch.equal(val,t));decoded.append(point)
        return images[point].clone(),{}
    def encode(v,rgb):encoded.append(rgb.clone());return z.clone()
    monkeypatch.setattr(runner.media,'decode_with_clamp_receipt',decode)
    monkeypatch.setattr(runner.media,'encode_normalized',encode)
    return run,decoded,encoded,images

def test_fixed_points_layers_and_reused_references(simulated,tmp_path):
    run,decoded,encoded,images=simulated;runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE'
    assert decoded==run.cfg['points'] and len(encoded)==6
    for i,point in enumerate(run.cfg['points']):
        assert torch.equal(encoded[2*i],images[point])
        assert torch.equal(encoded[2*i+1],runner.shared.quantize_rgb8_no_codec(images[point]).float()/255.)
        assert not torch.equal(encoded[2*i],encoded[2*i+1])
    assert run.data['counts']==dict(observations=6,updates=0,path_reads=12,payload_reads=12,message_evaluations=24,valid_costs=2088,reference_observations=6,reference_raw=12)
    for k,n in run.cfg['planned_calls'].items():assert run.data['calls'][k]==dict(attempted=n,completed=n)
    assert run.data['reference_comparison_alignment']=='EXACT_RGB8_REPLAY'
    assert len(run.data['comparisons'])==8 and len(run.data['layer_increments'])==6
    for row in run.data['reads'].values():
        raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
        assert not raw['truth_used'] and not raw['writer_inputs'] and len(raw['inference']['path_costs'])==174
    c=run.data['comparisons'];v=run.data['layer_increments']['COMPOSITE/RGB8->RAW420']
    assert v['delta_response_difference']==c['COMPOSITE_RAW420']['delta_change']-c['COMPOSITE_RGB8']['delta_change']

def test_layer_failure_keeps_other_layer_and_points(simulated,tmp_path,monkeypatch):
    run,*_=simulated;original=runner.media.encode_normalized;calls=[]
    def encode(*args):
        calls.append(1)
        if len(calls)==3:raise RuntimeError('fixed FLOAT layer failure')
        return original(*args)
    monkeypatch.setattr(runner.media,'encode_normalized',encode)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_PARTIAL' and run.data['counts']['observations']==5
    assert run.data['observations']['COMPOSITE_FLOAT']['status']=='FAILED'
    assert run.data['observations']['COMPOSITE_RGB8']['status']=='SAVED'
    assert run.data['observations']['WORST_ONLY_FLOAT']['status']=='SAVED'
    assert run.data['reads']['COMPOSITE_FLOAT/CORRECT']['status']=='NOT_COMPLETED'
    assert len(run.data['message_evaluations'])==24

def test_replay_mismatch_is_reported_without_discarding(simulated,tmp_path,monkeypatch):
    run,*_=simulated;original=runner.inputs
    def inputs(*a):
        z=original(*a);run.data['reference_observations']['COMPOSITE_RAW420']['input_raster_sha256']='different';return z
    monkeypatch.setattr(runner,'inputs',inputs)
    runner.execute(run,tmp_path,tmp_path);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE' and run.data['counts']['observations']==6
    assert run.data['reference_comparison_alignment']=='REPLAY_DIFFERS_OR_INCOMPLETE'

def test_notebook_is_fixed_and_pinned(tmp_path):
    from scripts.build_zero_mean_fixed_layer_forward_notebook import build
    p=build('a'*40,tmp_path/'nb.ipynb');nb=json.loads(p.read_text());cfg=json.loads(runner.CONFIG.read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    text=''.join(''.join(c['source']) for c in nb['cells'])
    for key in ('input','reference'):assert cfg[key]['root'] in text
    assert 'zero_mean_fixed_layer_forward_run' in text and 'Zero-Mean-Fixed-Layer-Forward-V1' in text
    code=''.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert 'FFMPEG_PROBE' not in code and 'ensurepip' not in code and "result['updates']" not in code
    assert nb['metadata']['candidate_binding']['source_sha']=='a'*40
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
