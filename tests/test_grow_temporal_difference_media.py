"""Fake media tests use stored tensors only; no models/GPU/real codec."""
import copy,hashlib,json
from types import SimpleNamespace
import torch
import pytest
from main.tube_state import grow_temporal_difference as method
from experiments.wan_state_clock import grow_temporal_difference_media as run
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

@pytest.fixture
def setup(tmp_path,monkeypatch):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12));monkeypatch.setattr(run,'RGB_SHAPE',(3,4,4,3))
    root=tmp_path/run.SOURCE_RUN;case=run.CASES[0];d=root/case;(d/'latents').mkdir(parents=True)
    manifest=run.load(run.MANIFEST);cfg=copy.deepcopy(manifest['base_config']);cfg['generation'].update(prompt=manifest['development'][0]['prompt'],seed=manifest['development'][0]['seed'])
    book=method.codebook(cfg['key_utf8'].encode());s={'config':cfg,'source_commit':run.SOURCE_COMMIT,'file_sha256':{},'videos':{}}
    tensors={}
    for n,arm in enumerate(run.ARMS):
        z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(n))*.01;tensors[arm]=z
        path=d/'latents'/f'{arm}_terminal.pt';torch.save(z,path);s['file_sha256']['latents/'+path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        rd=method.read(z,book);rd['payload_comparisons_reporting_only']=method.compare_payloads(rd,book);s['videos'][arm]={'terminal':rd}
    for path,obj in [(root/'result.json',{'cases':{case:s}}),(d/'result.json',s),(d/'config.json',cfg)]:path.write_text(json.dumps(obj))
    vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(_commit_hash='fake');calls={'decode':0,'encode':0,'load':0};images={}
    def load(*a):calls['load']+=1;return vae
    def decode(v,z):calls['decode']+=1;return torch.full(run.RGB_SHAPE,.5+calls['decode']*.001)
    def encode(v,rgb):calls['encode']+=1;return torch.zeros(method.SHAPE)+rgb.mean()
    def save(rgb,path,*a):path.parent.mkdir(exist_ok=True);path.write_bytes(b'fake-mp4');images[str(path)]=rgb.clone()
    monkeypatch.setattr(run,'load_frozen_vae',load);monkeypatch.setattr(run,'decode_normalized_latent',decode);monkeypatch.setattr(run,'reencode_rgb24_readback',encode);monkeypatch.setattr(run,'encode_rgb',save);monkeypatch.setattr(run,'read_mp4',lambda p:images[str(p)])
    return root,case,calls

def test_fixed_saved_case_four_layers(setup,tmp_path):
    root,case,calls=setup;out=tmp_path/'media';r=run.run_case(case,out,root)
    assert r['status']=='EXECUTION_COMPLETE' and calls=={'decode':3,'encode':9,'load':1}
    assert r['actual_calls']['transformer_completed']==0 and r['actual_calls']['mp4_read_completed']==3
    assert all(v['layers'][l]['status']=='COMPLETE' for v in r['videos'].values() for l in run.LAYERS)
    assert all(len(v['layers']['mp4']['per_pair'])==23 for v in r['videos'].values())
    assert (out/'rgb/A_float_rgb.pt').exists() and (out/'rgb/A_mp4_rgb8.pt').exists()
    assert r['quality_tolerance'] is None

def test_corrupt_source_not_decoded_and_failures_retained(setup,tmp_path):
    root,case,calls=setup;(root/case/'latents/A_terminal.pt').write_bytes(b'corrupt')
    r=run.run_case(case,tmp_path/'bad',root)
    assert r['status']=='WITH_RETAINED_FAILURES' and calls['decode']==2 and calls['encode']==6
    assert r['videos']['A']['generation_status']=='FAILED_SOURCE'
    assert len(r['videos']['A']['layers']['mp4']['per_pair'])==23

def test_mp4_failed_save_not_read_or_encoded(setup,tmp_path,monkeypatch):
    root,case,calls=setup
    def failed(rgb,path,*a):path.parent.mkdir(exist_ok=True);path.write_bytes(b'residual');raise RuntimeError('save failure')
    monkeypatch.setattr(run,'encode_rgb',failed);r=run.run_case(case,tmp_path/'savefail',root)
    assert r['actual_calls']['mp4_read_attempted']==0 and calls['encode']==6
    assert all(v['layers']['float_rgb']['status']=='COMPLETE' and v['layers']['mp4']['status']=='FAILED_SAVE_OR_READBACK' for v in r['videos'].values())

def test_missing_case_denominators_and_source_safety(tmp_path,monkeypatch):
    root=tmp_path/'source';root.mkdir()
    with pytest.raises(ValueError):run.run_all(root/'inside',root)
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    r=run.run_all(tmp_path/'out',root)
    assert r['video_denominator']==12 and r['layer_denominator']==48
    assert all(v['fixed_bit_denominator']==128 and v['missing_or_failed_marked']==8 and v['OFF_missing_or_failed']==4 for v in r['recovery_summary'].values())
    assert all(len(c['videos'])==3 for c in r['cases'].values())
