"""CPU/fake diagnostics verify transport separation, retained failures and fixed calls."""
from pathlib import Path
from types import SimpleNamespace
import ast,gzip,json
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import video_local_fourier_rm_channel_run as run
from main.tube_state import video_local_fourier_rm_state as state
from runtime.wan.video_temporal_sync_bridge import file_sha256
pytestmark=pytest.mark.unit


def baseline(tmp_path):
    cfg=run.load_config();root=tmp_path/'baseline';root.mkdir()
    meta=dict(source_sha=cfg['baseline']['source_sha'],method_version=state.PUBLIC.method_version,generation={},normalized={},phase_reads={},environment={'packages':{'torch':'previous-fake'}})
    for arm in run.ARMS:
        folder=root/arm;folder.mkdir();value=torch.from_numpy(state.synthesize(cfg['key'],dtype=np.float32)) if arm=='STATE_MULTI' else torch.zeros((1,16,46,40,64))
        terminal=folder/'terminal.pt';mp4=folder/'FULL_SOURCE181.phase0.pt';torch.save(value,terminal);torch.save(value,mp4)
        meta['generation'][arm]=dict(status='COMPLETE',sha256=file_sha256(terminal),terminal_path='/not-the-local-baseline/terminal.pt')
        meta['normalized'][arm+'/FULL_SOURCE181/0']=dict(status='SAVED',sha256=file_sha256(mp4),path='/not-the-local-baseline/phase0.pt')
        for key_id,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
            inf=state.infer(value[0,:,1:45].numpy().transpose(1,0,2,3),key,np.ones((44,4),bool));payload=run.method.payload_read(value,key,44)
            raw=folder/('FULL_SOURCE181.phase0.'+key_id+'.raw.json.gz');run.dump_gzip(raw,dict(inference=inf,payload=payload))
            meta['phase_reads'][arm+'/FULL_SOURCE181/0/'+key_id]=dict(status=inf['summary']['status'],sha256=file_sha256(raw),path='/not-local/raw.gz')
    run.dump(root/'result.json',meta);return root


def setup(tmp_path):
    root=baseline(tmp_path);cfg=run.load_config();store=run.Store(tmp_path/'output',create=True,baseline=root)
    meta=run.baseline_metadata(store,cfg,root);run.offline_references(store,cfg,root,meta);return root,cfg,store


def fake_channels(monkeypatch,store,cfg,root,*,fail_float=False):
    import runtime.wan.generation as generation
    frozen=SimpleNamespace(parameters=lambda:iter([torch.nn.Parameter(torch.zeros(1))]))
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:frozen)
    rgb=torch.tensor([0.12345,0.501,0.87654,0.,1.,0.5],dtype=torch.float32).reshape(1,1,2,3)
    decodes=[];encodes=[];fail=[fail_float]
    def decode(vae,value):decodes.append(value.clone());return rgb.clone()
    def encode(vae,value):
        encodes.append(value.clone())
        if fail[0]:fail[0]=False;raise RuntimeError('one fixed FLOAT encode failure')
        return decodes[-1].clone()
    monkeypatch.setattr(run.vae_adapter,'decode_normalized_latent',decode)
    monkeypatch.setattr(run.vae_adapter,'reencode_rgb24_readback',encode)
    monkeypatch.setattr(run,'rgb_receipt',lambda value:dict(shape=list(value.shape),fake=True))
    run.channel_worker(store,cfg,root);store.data['workers']['channel']=dict(status='COMPLETE');store.save()
    return rgb,decodes,encodes


def test_same_decode_distinct_float_rgb8_fixed_calls_and_posthoc(monkeypatch,tmp_path):
    root,cfg,store=setup(tmp_path);rgb,decodes,encodes=fake_channels(monkeypatch,store,cfg,root)
    assert len(decodes)==3 and len(encodes)==6
    expected=torch.from_numpy(np.rint(rgb.numpy()*255).astype(np.uint8)).float()/255
    for i in range(3):
        assert torch.equal(encodes[2*i],rgb)
        assert torch.equal(encodes[2*i+1],expected)
        assert not torch.equal(encodes[2*i],encodes[2*i+1])
    store.receiver_snapshot();snapshot=(store.output/'blind_channel_readouts.json').read_bytes();refs=(store.output/'reference_readouts.json').read_bytes()
    assert run.finish(store,cfg)
    assert store.data['counts']==dict(new_normalized=6,reference_inputs=6,new_reads=12,reference_reads=12,path_reads=24,payload_reads=24,valid_costs=4176,local_state_costs=47520,evaluated=48,local_posthoc=24)
    assert store.data['call_integrity']['status']=='MATCH'
    assert snapshot==(store.output/'blind_channel_readouts.json').read_bytes() and refs==(store.output/'reference_readouts.json').read_bytes()
    blind=json.loads(snapshot);reference=json.loads(refs)
    assert len(blind['reads'])==12 and len(reference['reads'])==12
    assert 'TERMINAL44_REFERENCE' not in snapshot.decode() and 'MP4_G0_REFERENCE' not in snapshot.decode()
    assert not any('true_state' in str(v) or 'true_path' in str(v) for v in blind.values())
    assert all(r['state_path_accepted'] is False and r['accepted_payload'] is False for r in store.data['reads'].values())
    assert all(r['old_raw_comparison']['status']=='MATCH' for sid,r in store.data['reads'].items() if '/MP4_G0_REFERENCE/' in sid)
    assert store.data['local_posthoc']['STATE_MULTI/TERMINAL44_REFERENCE/CORRECT']['true_path_rank']==1
    # Exact load/adapter sematics are frozen in the original config, not tuned to outputs.
    assert cfg['control']['steps']==list(range(25,50)) and cfg['control']['pilot_eta']==696


def test_missing_one_terminal_keeps_other_arms_and_mp4_reference(monkeypatch,tmp_path):
    root=baseline(tmp_path);(root/'OFF'/'terminal.pt').unlink();cfg=run.load_config();store=run.Store(tmp_path/'output',create=True,baseline=root)
    meta=run.baseline_metadata(store,cfg,root);run.offline_references(store,cfg,root,meta);rgb,decodes,encodes=fake_channels(monkeypatch,store,cfg,root)
    assert len(decodes)==2 and len(encodes)==4
    assert not run.finish(store,cfg)
    assert len(store.data['reads'])==24 and len(store.data['evaluations'])==48 and len(store.data['local_posthoc'])==24
    assert store.data['reference_inputs']['OFF/MP4_G0_REFERENCE']['status']=='LOADED'
    assert store.data['reads']['OFF/FLOAT_VAE_ROUNDTRIP/CORRECT']['status']=='NOT_COMPLETED'
    assert store.data['reads']['STATE_MULTI/RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC/CORRECT']['status']=='COMPLETE'
    assert store.data['counts']['reference_reads']==10 and store.data['counts']['new_reads']==8


def test_single_channel_encode_failure_preserves_other_channel(monkeypatch,tmp_path):
    root,cfg,store=setup(tmp_path);rgb,decodes,encodes=fake_channels(monkeypatch,store,cfg,root,fail_float=True)
    assert len(decodes)==3 and len(encodes)==6 and store.data['calls']['vae_encode']==dict(attempted=6,completed=5)
    assert store.data['normalized']['OFF/FLOAT_VAE_ROUNDTRIP']['status']=='FAILED'
    assert store.data['normalized']['OFF/RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC']['status']=='SAVED'
    assert not run.finish(store,cfg) and store.data['counts']['path_reads']==22
    assert len(store.data['evaluations'])==48 and sum(r['status']=='MISSING_READ' for r in store.data['evaluations'].values())==4


def test_vae_load_failure_retains_references(monkeypatch,tmp_path):
    import runtime.wan.generation as generation
    root,cfg,store=setup(tmp_path)
    def failed(*a,**k):raise RuntimeError('actual VAE load unavailable')
    monkeypatch.setattr(generation,'load_frozen_vae',failed)
    with pytest.raises(RuntimeError):run.channel_worker(store,cfg,root)
    store.data['workers']['channel']=dict(status='FAILED');store.save();assert not run.finish(store,cfg)
    assert store.data['calls']['vae_load']==dict(attempted=1,completed=0)
    assert store.data['counts']['reference_reads']==12 and store.data['counts']['new_reads']==0
    assert len(store.data['evaluations'])==48 and store.data['counts']['evaluated']==24


def test_child_spawn_failure_keeps_fixed_roster(monkeypatch,tmp_path):
    root,cfg,store=setup(tmp_path)
    def failed(*a,**k):raise OSError('fixed spawn failure')
    monkeypatch.setattr(run.subprocess,'Popen',failed)
    store,interrupted=run.run_channel_child(store,root)
    assert interrupted is False and store.data['workers']['channel']['child_started'] is False
    assert not run.finish(store,cfg)
    assert store.data['counts']['reference_reads']==12 and all(r['status']!='PENDING' for r in store.data['reads'].values())
    assert len(store.data['evaluations'])==48


def test_notebook_direct_fixed_VAE_only(tmp_path):
    from scripts import build_video_local_fourier_rm_channel_notebook as builder
    path=builder.build('a'*40,tmp_path/'bound.ipynb');nb=json.loads(path.read_text());sources=[''.join(c['source']) for c in nb['cells']]
    assert sources[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell,text in zip(nb['cells'],sources):
        if cell['cell_type']=='code':ast.parse(text);assert cell['outputs']==[] and cell['execution_count'] is None
    joined='\n'.join(sources)
    assert 'venv' not in '\n'.join(sources[2:]) and 'ffmpeg' not in '\n'.join(sources[2:]).lower() and 'ffprobe' not in joined.lower()
    assert 'WanPipeline' not in joined and 'UniPCMultistepScheduler' not in joined and 'AutoencoderKLWan' in joined
    assert 'current_python_fresh_child_VAE_only' in joined and '--baseline' not in builder.RUN
    assert nb['metadata']['candidate_binding']['source_sha']=='a'*40
