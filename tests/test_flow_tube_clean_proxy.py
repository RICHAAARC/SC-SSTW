"""CPU/fake saved-history, budget, layer and retained-failure checks."""
import copy,json
from types import SimpleNamespace
import numpy as np
import pytest,torch
from main.tube_state import projection_margin,state_clock
from runtime.wan import tube_retention
from experiments.wan_state_clock import flow_tube_clean_proxy_run as run
from test_flow_tube_state_guidance import native,book,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

def source_fixture(tmp_path,native,book):
    source=tmp_path/'source';case=source/run.CASES[0];case.mkdir(parents=True)
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());prompt=torch.tensor(1.);negative=torch.tensor(-1.)
    nodes,snapshots,_=tube_retention.reference(pipe,torch.zeros(projection_margin.SHAPE),prompt,negative,torch.float32,5.,lambda *a:None)
    config=dict(model=dict(id='fake',revision=None),key_utf8='WanProjection-first-validation-key-v1',generation=dict(guidance_scale=5.))
    run.dump(case/'config.json',config);np.savez(case/'codebook.npz',**book)
    for filename,obj in [('OFF_nodes.pt',nodes),('OFF_snapshots.pt',snapshots),('prompt.pt',prompt),('negative.pt',negative),('OFF_terminal.pt',nodes[50]['z'])]:torch.save(obj,case/filename)
    generation=dict(source_commit=run.SOURCE_COMMIT,file_sha256={p.name:run.sha(p) for p in case.iterdir()},
        source_sha256={str(m.__file__):run.sha(m.__file__) for m in (projection_margin,state_clock)},
        reference_fingerprints={'46':dict(input=run.runtime.fingerprint(nodes[46]['z']),history=run.runtime.fingerprint(vars(snapshots[46])))},
        videos={f'LOCAL_{s}':{'control':{'actual_D':{'support_rms':.01}}} for s in ('A','B')})
    run.dump(case/'generation.json',generation)
    return source

def patch_model(monkeypatch):
    model=Model();monkeypatch.setattr(run.runtime,'load_transformer',lambda config:(SimpleNamespace(transformer=model),torch.float32,{'resolved_revision':'fake-new','original_weight_identity':'UNVERIFIED'}))
    return model

def patch_media(monkeypatch,fail_off=False):
    class VAE(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.ones(1));self.config=SimpleNamespace(_commit_hash='fake-new')
    monkeypatch.setattr(run,'RGB_SHAPE',(181,4,4,3));monkeypatch.setattr(run,'load_frozen_vae',lambda *a:VAE());monkeypatch.setattr(run,'_clear_cache',lambda *a:None)
    base=(torch.arange(181).float()/181+.0004).reshape(181,1,1,1).expand(run.RGB_SHAPE).clone()
    monkeypatch.setattr(run,'decode_normalized_latent',lambda *a:base.clone());cache={};reads=[];inputs=[];blind=[]
    def save(rgb,path,*a):
        path.parent.mkdir(exist_ok=True);path.write_bytes(b'fake_codec');cache[str(path)]=run.quantize_rgb8_no_codec(rgb).float()/255+.0001
        if fail_off and path.stem=='OFF':raise RuntimeError('partial OFF save')
    def read(path):reads.append(path.stem);return cache[str(path)]
    def encode(vae,rgb):inputs.append(rgb.clone());return torch.zeros((1,16,1+(len(rgb)-1)//4,40,64))
    def receiver(obs,book):
        blind.append(sorted(obs));by={str(m):{'message':m,'score':.1-m*.05} for m in (0,1)}
        return {'rankings':{mode:dict(best=by['0'],best_by_message=by,message_unique=True) for mode in run.MODES}}
    monkeypatch.setattr(run,'encode_rgb',save);monkeypatch.setattr(run,'read_mp4',read);monkeypatch.setattr(run,'reencode_rgb24_readback',encode)
    monkeypatch.setattr(run.state_clock,'read',receiver);monkeypatch.setattr(run.state_clock,'report',lambda *a:{'posthoc':True})
    return base,inputs,blind,reads

def test_saved_history_clean_proxy_budget_and_all_real_layers(monkeypatch,tmp_path,native,book):
    source=source_fixture(tmp_path,native,book);model=patch_model(monkeypatch);out=tmp_path/'new'
    r=run.generate_case(run.CASES[0],source,out)
    assert r['status']=='GENERATION_COMPLETE',r['failures']
    assert r['historical_OFF_compatibility']['status']=='MATCH'
    for k in ('transformer','scheduler_step','unit_response_probe_step','clean_leaf_backward'):assert r['actual_calls'][k+'_completed']==run.PLAN[k]
    for arm in run.ARMS[1:]:
        c=r['videos'][arm]['control'];assert c['actual_D']['support_rms']==pytest.approx(.01,rel=1e-5)
        assert c['history_fingerprint']==r['original46_fingerprints']['history']
        assert r['videos'][arm]['clean_gradient']['objective']==arm.split('_')[0].lower()
    assert not any(model.flags)
    base,inputs,blind,reads=patch_media(monkeypatch);r=run.media_case(run.CASES[0],out)
    assert r['status']=='EXECUTION_COMPLETE',r['failures']
    assert all(r['actual_calls'][k+'_completed']==n for k,n in run.PLAN.items())
    assert len(inputs)==60 and len(blind)==15 and all(keys==[0,1,2,3] for keys in blind)
    assert [len(v) for v in inputs]==[181,177,177,177]*15
    torch.testing.assert_close(inputs[0],base);torch.testing.assert_close(inputs[4],run.quantize_rgb8_no_codec(base).float()/255)
    assert not torch.equal(inputs[0],inputs[4]) and not torch.equal(inputs[4],inputs[8])
    for g in range(4):assert inputs[g][0,0,0,0]==base[g,0,0,0]

def test_historical_nonmatch_not_gate_and_partial_off_not_reused(monkeypatch,tmp_path,native,book):
    source=source_fixture(tmp_path,native,book);root=source/run.CASES[0]
    old=torch.load(root/'OFF_terminal.pt',weights_only=True);torch.save(old+.2,root/'OFF_terminal.pt')
    gen=run.load(root/'generation.json');gen['file_sha256']['OFF_terminal.pt']=run.sha(root/'OFF_terminal.pt');run.dump(root/'generation.json',gen)
    patch_model(monkeypatch);out=tmp_path/'new';r=run.generate_case(run.CASES[0],source,out)
    assert r['status']=='GENERATION_COMPLETE' and r['historical_OFF_compatibility']['status']=='NONMATCH'
    _,inputs,blind,reads=patch_media(monkeypatch,fail_off=True);r=run.media_case(run.CASES[0],out)
    assert r['status']=='WITH_RETAINED_FAILURES' and 'OFF' not in reads
    assert all(r['videos'][a]['status']=='COMPLETE' for a in run.ARMS[1:])
    assert r['videos']['OFF']['layers']['floatRGB']['status']=='COMPLETE'
    assert r['videos']['HINGE_A']['layers']['MP4']['quality_vs_new_OFF']['status']=='MISSING_NEW_OFF_REFERENCE'
    assert len(inputs)==56

def test_bad_source_rejected_before_model_and_fixed_missing_roster(monkeypatch,tmp_path,native,book):
    source=source_fixture(tmp_path,native,book);(source/run.CASES[0]/'prompt.pt').write_bytes(b'bad')
    monkeypatch.setattr(run.runtime,'load_transformer',lambda *a:pytest.fail('model must not load unverified source'))
    r=run.generate_case(run.CASES[0],source,tmp_path/'bad');assert r['status']=='WITH_RETAINED_FAILURES'
    assert r['actual_calls']['transformer_attempted']==0
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    root=run.run_all(source,tmp_path/'all');assert root['video_denominator']==10 and root['receiver_encode_denominator']==120
    assert sum(len(l['observations']) for c in root['cases'].values() for v in c['videos'].values() for l in v['layers'].values())==120
    assert len(root['paired_summary']['rows'])==4
