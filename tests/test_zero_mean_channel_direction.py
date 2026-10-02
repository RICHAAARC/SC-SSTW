import ast,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import zero_mean_channel_direction_run as runner
from main.tube_state import video_overlap_zero_mean_state as state,zero_mean_channel_margin as method
pytestmark=pytest.mark.unit

@pytest.fixture
def fixture(tmp_path,monkeypatch):
    cfg=json.loads(runner.CONFIG.read_text());cfg['cotangent_chunk_frames']=2
    roots=[tmp_path/k for k in ('layers','ablation','terminal')]
    for p in roots:p.mkdir()
    layers,ablation,terminal=roots;shape=(3,2,2,3);monkeypatch.setattr(runner,'RGB_SHAPE',shape)
    z=torch.tensor(state.synthesize(cfg['key']),dtype=torch.float32)
    def saved(path,x):
        path.parent.mkdir(parents=True,exist_ok=True);torch.save(x,path);return dict(path=str(path),sha256=runner.media.file_sha256(path))
    base=saved(terminal/'POINT3/terminal.pt',z);cfg['terminal']['terminal_sha256']=base['sha256']
    fl=dict(replay={},config=dict(terminals={}));old=dict(observations={},updates={},reads={},posthoc={},message_evaluations={})
    for i,point in enumerate(['BASELINE',*cfg['arms']]):
        f=torch.full(shape,.501+i*.003);q=runner.shared.quantize_rgb8_no_codec(f)
        fl['replay'][point]=dict(float_rgb=saved(layers/point/'float_rgb.pt',f),rgb8=saved(layers/point/'rgb8.pt',q))
        sha=hashlib.sha256(q.numpy().tobytes()).hexdigest();obs=point+'_RAW420';d=ablation/obs;d.mkdir()
        raw=d/'roundtrip.rgb';raw.write_bytes(q.numpy().tobytes())
        _,metrics=method.margin_loss(method.project(z+i*.0001,cfg['key']),cfg['key'])
        old['observations'][obs]=dict(normalized=saved(d/'normalized.pt',z+i*.0001),writer_loss=metrics,conversion={'rgb24':dict(sha256=runner.media.file_sha256(raw)),'yuv420':dict(input_raster_sha256=sha)})
        fl['config']['terminals'][point]=base['sha256']
        if point!='BASELINE':
            step=torch.full_like(z,i*.0001);g=torch.full_like(z,.0002)
            gr=saved(ablation/point/'terminal_gradient.pt',g);sr=saved(ablation/point/'step.pt',step);tr=saved(ablation/point/'terminal.pt',z+step)
            fl['config']['terminals'][point]=tr['sha256']
            old['updates'][point]=dict(shared_input_sha256=base['sha256'],gradient=gr,step=sr,receipt=dict(gradient_dot_step=float((g.double()*step.double()).sum())))
        for kid in cfg['keys']:
            raw=d/(kid+'.raw.json.gz');raw.write_bytes(gzip.compress(json.dumps(dict(truth_used=False,writer_inputs=False,inference=dict(path_costs=[0]*174))).encode()))
            sid=obs+'/'+kid;old['reads'][sid]=dict(sha256=runner.media.file_sha256(raw));old['posthoc'][sid]=dict(status='EVALUATED')
            for msg in cfg['messages']:old['message_evaluations'][sid+'/'+msg]=dict(status='EVALUATED',exact=False)
    for key,path,data in [('input',layers,fl),('reference',ablation,old),('terminal',terminal,{})]:
        (path/'result.json').write_text(json.dumps(data));cfg[key]['result_sha256']=runner.media.file_sha256(path/'result.json')
    run=runner.Run(tmp_path/'output',cfg)
    from runtime.wan import generation
    class VAE:
        use_tiling=False
        config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def parameters(self):return iter([torch.zeros(1)])
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**kw:VAE())
    calls=[]
    def encode(v,rgb,key,count,receipt,*,objective):
        calls.append((objective,rgb.clone()))
        for k in ('vae_encode_gradient','encoder_vjp'):count(k,False);count(k,True)
        _,metrics=method.margin_loss(method.project(z,key),key,objective=objective)
        return z.clone(),torch.full_like(rgb,2 if objective=='composite' else 3),metrics
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',encode)
    monkeypatch.setattr(runner.gradient,'decoder_vjp',lambda *a,**kw:pytest.fail('decoder VJP forbidden'))
    monkeypatch.setattr(runner.media,'decode_with_clamp_receipt',lambda *a,**kw:pytest.fail('decode forbidden'))
    return run,roots,calls

def test_saved_pipeline_and_telescoping(fixture):
    run,roots,calls=fixture;runner.execute(run,*roots);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE' and run.data['raster_alignment']=='EXACT'
    assert len(calls)==2 and torch.equal(calls[0][1],calls[1][1])
    assert run.data['counts']==dict(encoder_vjps=2,replay_normalized=2,rgb_cotangents=2,attributions=2,updates=0,new_path_reads=0,new_payload_reads=0,reference_observations=3,reference_raw=6)
    for i,arm in enumerate(run.cfg['arms']):
        row=run.data['attributions'][arm];rep=run.data['encoder_replays'][arm]
        assert row['status']=='EVALUATED' and abs(row['telescoping_residual'])<1e-12
        assert rep['normalized_max_error']==rep['normalized_rmse']==0
        parts=rep['cotangent_parts'];assert len(parts)==2
        assert parts[0]['frame_start']==0 and parts[-1]['frame_end_exclusive']==3
        cot=torch.cat([runner.load(p['path']) for p in parts]);assert torch.equal(cot,torch.full_like(calls[i][1],i+2))
        assert max(Path(p['path']).stat().st_size for p in parts)<10000
        for layer,key in [('FLOAT','float_rgb'),('RGB8','rgb8')]:
            before=runner.rgb(roots[0]/'BASELINE'/(key+'.pt'),layer);after=runner.rgb(roots[0]/arm/(key+'.pt'),layer)
            expected=float((cot.double()*(after.double()-before.double())).sum())
            assert row[layer]['total']==pytest.approx(expected,abs=1e-14)
        assert row['stored_step_prediction_recomputed']==row['stored_gradient_dot_step']
    for k,n in run.cfg['planned_calls'].items():
        if '_chunk_' not in k:assert run.data['calls'][k]==dict(attempted=n,completed=n)

def test_failed_first_objective_keeps_second(fixture,monkeypatch):
    run,roots,_=fixture;original=runner.gradient.encoder_cotangent
    def fail(*a,objective,**kw):
        if objective=='composite':raise RuntimeError('simulated failure')
        return original(*a,objective=objective,**kw)
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',fail)
    runner.execute(run,*roots);runner.finalize(run)
    assert run.data['status']=='EXECUTION_PARTIAL' and run.data['counts']['attributions']==1
    assert run.data['attributions']['COMPOSITE']['status']=='FAILED'
    assert run.data['encoder_replays']['COMPOSITE']['status']=='NOT_COMPLETED'
    assert run.data['attributions']['WORST_ONLY']['status']=='EVALUATED'

def test_replay_difference_is_reported(fixture,monkeypatch):
    run,roots,_=fixture;original=runner.gradient.encoder_cotangent
    def replay(*a,**kw):
        z,c,m=original(*a,**kw);return z+.01,c,m
    monkeypatch.setattr(runner.gradient,'encoder_cotangent',replay)
    runner.execute(run,*roots);runner.finalize(run)
    assert run.data['status']=='EXECUTION_COMPLETE'
    assert all(v['normalized_max_error']>.009 for v in run.data['encoder_replays'].values())

def test_changed_saved_input_rejected(fixture):
    run,roots,_=fixture;(roots[0]/'COMPOSITE/float_rgb.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='identity mismatch'):runner.inputs(run,*roots)

def test_dot_direction_and_finite_remainder():
    x=torch.tensor([[[[.2,.4,.7]]],[[[.3,.8,.5]]]],dtype=torch.float32);d=torch.ones_like(x)*.01;y=x+d
    # L(x)=sum(x^2)/2; first-order dot plus quadratic remainder is exact.
    v=runner.direction_dot(x,x,y);actual=float((y.double().square()-x.double().square()).sum()/2)
    remainder=float((y.double()-x.double()).square().sum()/2)
    assert actual-v['total']==pytest.approx(remainder,abs=1e-15)
    assert len(v['per_frame'])==2

def test_pinned_notebook_paths_and_no_new_search(tmp_path):
    from scripts.build_zero_mean_channel_direction_notebook import build
    cfg=json.loads(runner.CONFIG.read_text());nb=json.loads(build('a'*40,tmp_path/'nb.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    text=''.join(''.join(c['source']) for c in nb['cells'])
    for k in ('input','reference','terminal'):assert cfg[k]['root'] in text
    assert "'--terminal-root',str(TERMINAL_ROOT)" in text and 'zero_mean_channel_direction_run' in text
    assert nb['metadata']['candidate_binding']['source_sha']=='a'*40
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
