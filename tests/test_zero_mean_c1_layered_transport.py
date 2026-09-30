"""CPU synthetic/mock replay only: never load a real VAE/model or invoke a codec."""
import ast,copy,hashlib,importlib.util,json,re,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_overlap_zero_mean_state as state
from runtime.wan import zero_mean_c1_layered_transport as b,generation
from experiments.wan_state_clock import zero_mean_c1_layered_transport_run as run
pytestmark=pytest.mark.unit

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)

def rgb(value=.5,dtype=torch.float32):
    return torch.tensor(value,dtype=dtype).reshape(1,1,1,1).expand(181,320,512,3)

def latent():
    return torch.from_numpy(state.synthesize('watermark',dtype=np.float32))

def fixture_store(tmp_path):
    cfg=copy.deepcopy(run.load_config());source=tmp_path/'inputs';source.mkdir()
    for entry in cfg['fixed_input']['required_terminal']:
        p=source/entry['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);torch.save(latent(),p)
        entry['sha256']=b.file_sha256(p)
    rp=source/'result.json';run.dump(rp,dict(source_sha=cfg['fixed_input']['source_sha'],environment=dict(mock=True)))
    cfg['fixed_input']['result_sha256']=b.file_sha256(rp)
    store=run.Store(tmp_path/'output',create=True,cfg=cfg,input_root=source)
    run.preflight(store,cfg)
    return store,cfg

class FakeVAE:
    def __init__(self):
        self.p=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float32),requires_grad=False)
        self.config=SimpleNamespace(latents_mean=[1.]*16,latents_std=[2.]*16)
        self.clears=0;self.decode_inputs=[];self.encode_inputs=[]
    def parameters(self):return iter([self.p])
    def clear_cache(self):self.clears+=1
    def decode(self,z,return_dict):
        self.decode_inputs.append((tuple(z.shape),float(z[0,0,0,0,0])))
        return (torch.tensor([-2.,0.,2.]).reshape(1,3,1,1,1).expand(1,3,181,320,512),)
    def encode(self,x):
        self.encode_inputs.append((tuple(x.shape),x[0,:,0,0,0].tolist()))
        return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.ones(1,16,46,40,64)*5.))

def test_thin_adapter_original_mapping_cache_preclamp_and_mode():
    vae=FakeVAE();z=torch.zeros(1,16,46,40,64)
    image,receipt=b.decode_with_clamp_receipt(vae,z)
    assert image.shape==(181,320,512,3) and image.dtype==torch.float32
    assert image[0,0,0].tolist()==[0.,.5,1.]
    assert receipt['fraction_below0']==pytest.approx(1/3)
    assert receipt['fraction_above1']==pytest.approx(1/3)
    assert receipt['maximum_underflow']==receipt['maximum_overflow']==.5
    assert receipt['mean_absolute_clamp_change']==pytest.approx(1/3)
    assert vae.decode_inputs==[((1,16,46,40,64),1.)]
    encoded=b.encode_normalized(vae,image)
    assert torch.equal(encoded,torch.full_like(encoded,2.))
    assert vae.encode_inputs==[((1,3,181,320,512),[-1.,0.,1.])]
    assert vae.clears==4

def test_q8_original_quantizer_preserves_all_values_and_codec_settings(monkeypatch,tmp_path):
    q=torch.arange(256,dtype=torch.uint8).repeat(2).reshape(1,1,512,1).expand(181,320,512,3)
    seen={}
    def fake_encode(value,path,fps,crf):
        assert torch.equal(b.shared_vae.quantize_rgb8_no_codec(value),q)
        seen.update(path=path,fps=fps,crf=crf)
    monkeypatch.setattr(b.io,'encode_rgb',fake_encode)
    b.encode_raster(q,tmp_path/'mock.mp4')
    assert seen==dict(path=tmp_path/'mock.mp4',fps=8,crf=18)

def test_original_infer_full_family_blind_and_extra_row_exclusion():
    calls=[]
    def call(k,fn):calls.append(k);return fn()
    z=latent();before=run.read_normalized(z,'watermark',call)
    h0,h1,w0,w1=state.PUBLIC.blocks[0]
    z[0,4,45,h0:h1,w0:w1]+=torch.from_numpy(state.bases('watermark')[0,:,0].reshape(4,4)).float()*123.
    after=run.read_normalized(z,'watermark',call)
    assert before['inference']==after['inference']
    assert before['payload']==after['payload']
    assert before['extra_regular45_projection']!=after['extra_regular45_projection']
    result=before['inference']
    assert result['counts']['catalog']==3915 and result['counts']['scored']==174
    assert result['counts']['structurally_excluded']==3741
    assert result['summary']['top_catalog_indices']==[0]
    assert all(v['count']==1320 for v in before['payload']['votes'])
    assert calls==['new_path_infer','new_payload_read']*2
    assert list(__import__('inspect').signature(run.read_normalized).parameters)==['z','key','call']

def install_replay_fixture(monkeypatch,*,fail_float=False):
    trace=[];fake=FakeVAE()
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:fake)
    monkeypatch.setattr(b,'decode_with_clamp_receipt',lambda *a:(rgb(),dict(status='MEASURED',mock=True)))
    monkeypatch.setattr(b.shared_vae,'quantize_rgb8_no_codec',lambda value:rgb(128,torch.uint8))
    def encode(vae,value):
        assert tuple(value.shape)==(181,320,512,3)
        first=float(value[0,0,0,0]);trace.append(('encode',first))
        if fail_float and first==.5:raise RuntimeError('injected float encode failure')
        return latent()
    monkeypatch.setattr(b,'encode_normalized',encode)
    def source(q,path):
        assert q.dtype==torch.uint8 and int(q[0,0,0,0])==128
        Path(path).write_bytes(b'mock source');trace.append(('source',128))
    monkeypatch.setattr(b,'encode_raster',source)
    def full(value,path,fps,crf):
        assert float(value[0,0,0,0])==.25 and (fps,crf)==(8,18)
        Path(path).write_bytes(b'mock full');trace.append(('full',.25))
    monkeypatch.setattr(b.io,'encode_rgb',full)
    monkeypatch.setattr(b,'read_full_mp4',lambda p:rgb(.25 if Path(p).name.startswith('MATCHED_SOURCE') else .3125))
    return trace

def test_matched_full_chain_original_reader_fixed_counts_and_posthoc_isolation(monkeypatch,tmp_path):
    store,cfg=fixture_store(tmp_path)
    trace=install_replay_fixture(monkeypatch)
    run.terminal_layers(store,cfg);run.replay_worker(store,cfg)
    # Supply one hash-bound historical phase0 winner. This is posthoc only.
    rawrow=store.data['reads']['OFF/MATCHED_FULL_RESAVED/CORRECT']
    old=next(r for r in store.data['historical'].values() if r['arm']=='OFF' and r.get('key_id')=='CORRECT')
    old.update(status='VERIFIED',path=rawrow['path'],sha256=rawrow['sha256'])
    store.checkpoint();before=b.file_sha256(store.output/'blind_readouts.json')
    assert run.finalize(store,cfg)
    assert store.data['primary_status']=='EXECUTION_COMPLETE'
    assert store.data['historical_reference_status']==store.data['package_status']=='INCOMPLETE'
    assert store.data['counts']==dict(normalized=15,path_reads=30,payload_reads=30,valid_costs=5220,
        catalog_slots=117450,structural_exclusions=112230,path_posthoc=30,message_evaluations=60,
        rasters=6,mp4_saved=6,mp4_read=6,decode_diagnostics=3)
    for kind,n in cfg['planned_calls'].items():assert store.data['calls'][kind]['completed']==n
    assert [v for k,v in trace if k=='encode']==pytest.approx([.5,128/255.,.25,.3125]*3)
    assert b.file_sha256(store.output/'blind_readouts.json')==before
    post=store.data['path_posthoc']['OFF/TERMINAL_PREFIX44/CORRECT']
    assert post['true_rank']==1 and post['truth_unique_top']
    assert post['canonical_start_correct'] and post['invented_edit'] is False
    assert post['fixed_historical_phase0_opponent']['status']=='COMPARED'
    assert post['fixed_historical_phase0_opponent']['catalog_index']==0
    assert all(not r['accepted_payload'] for r in store.data['path_posthoc'].values())
    text=(store.output/'blind_readouts.json').read_text()
    assert all(x not in text for x in ('registered_tau','true_rank','REGISTERED','OKOK','NOPE','invented_edit'))
    assert len(store.data['historical'])==15
    assert not list(store.output.rglob('*.tmp'))

def test_float_encode_failure_does_not_block_same_raster_media(monkeypatch,tmp_path):
    store,cfg=fixture_store(tmp_path);install_replay_fixture(monkeypatch,fail_float=True)
    run.replay_worker(store,cfg);run.recover_unfinished(store,'fixture end');run.evaluate_saved(store,cfg)
    assert sum(r['status']=='FAILED' for r in store.data['normalized'].values())==3
    assert all(store.data['normalized'][a+'/'+l]['status']=='SAVED' for a in run.ARMS for l in run.LAYERS[2:])
    assert store.data['counts']['mp4_saved']==6 and store.data['counts']['mp4_read']==6
    assert store.data['counts']['valid_costs']==18*174
    assert store.data['counts']['path_reads']==18
    assert all(r['status']!='PENDING' for group in ('reads','path_posthoc','message_evaluations') for r in store.data[group].values())

def test_missing_input_or_hash_keeps_fixed_roster_and_terminal_independence(tmp_path,monkeypatch):
    store,cfg=fixture_store(tmp_path)
    source=Path(store.data['input_root'])/'OFF/terminal.pt';source.write_bytes(b'corrupt fixture')
    run.preflight(store,cfg)
    assert store.data['inputs']['OFF']['status']=='FAILED'
    run.terminal_layers(store,cfg)
    assert store.data['counts']['path_reads']==4
    def no_vae(*a,**k):raise RuntimeError('fixture VAE load failed')
    monkeypatch.setattr(generation,'load_frozen_vae',no_vae)
    with pytest.raises(RuntimeError):run.replay_worker(store,cfg)
    assert not run.finalize(store,cfg)
    assert store.data['counts']['path_reads']==4 and store.data['counts']['catalog_slots']==117450
    assert len(store.data['reads'])==30 and len(store.data['message_evaluations'])==60
    assert store.data['calls']['vae_decode']['attempted']==0

@pytest.mark.parametrize('after_canonical',[False,True])
def test_canonical_window_recovery_retains_only_committed_rows(tmp_path,monkeypatch,after_canonical):
    store,cfg=fixture_store(tmp_path);sid='OFF/FLOAT_CLAMPED'
    store.data['normalized'][sid].update(status='SAVED',sha256='fixture-hash')
    def stop():raise SystemExit('fixture hard exit')
    monkeypatch.setattr(store,'blind' if after_canonical else 'save',stop)
    with pytest.raises(SystemExit):store.checkpoint()
    restored=run.Store(store.output);run.recover_unfinished(restored,'hard exit')
    assert restored.data['normalized'][sid]['status']==('SAVED' if after_canonical else 'NOT_COMPLETED')
    blind=json.loads((store.output/'blind_readouts.json').read_text())
    assert blind['normalized'][sid]==restored.data['normalized'][sid]
    assert len(restored.data['reads'])==30

def test_worker_monitor_failure_terminates_before_reload(tmp_path,monkeypatch):
    store,cfg=fixture_store(tmp_path);events=[]
    class Child:
        stdout=None
        def poll(self):return None
        def terminate(self):events.append('terminate')
        def wait(self,timeout=None):events.append('wait');return -15
    class BrokenStream:
        def __iter__(self):
            disk=run.Store(store.output)
            disk.data['decode_diagnostics']['OFF'].update(status='MEASURED',mock=True);disk.save()
            raise OSError('monitor fixture')
        def close(self):events.append('close')
    child=Child();child.stdout=BrokenStream()
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**k:child)
    restored,halt=run.worker_phase(store.output)
    assert not halt and events[:2]==['terminate','wait']
    assert restored.data['decode_diagnostics']['OFF']['status']=='MEASURED'
    assert restored.data['workers']['replay']['status']=='FAILED'

def test_output_guard_prevents_writes_inside_fixed_input(tmp_path):
    cfg=run.load_config();source=tmp_path/'input';source.mkdir()
    for output in (source,source/'new'/'nested'):
        with pytest.raises(ValueError,match='read-only'):run.Store(output,create=True,cfg=cfg,input_root=source)
    assert list(source.iterdir())==[]

def test_notebook_draft_fixed_input_dependencies_and_fresh_child(tmp_path):
    builder_path=run.ROOT/'scripts/build_zero_mean_c1_layered_transport_notebook.py'
    spec=importlib.util.spec_from_file_location('layered_builder',builder_path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    candidate=builder.build(output=tmp_path/'n.ipynb')
    assert candidate.read_bytes()==(run.ROOT/'notebooks/zero_mean_c1_layered_transport_v1_colab.ipynb').read_bytes()
    nb=json.loads(candidate.read_text());codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']));assert cell['outputs']==[] and cell['execution_count'] is None
    text='\n'.join(codes)
    assert 'SOURCE_SHA = None' in text and '20260930T113335113338Z/fixed_reference' in text
    assert 'PYTHON=sys.executable' in text and 'torch,torchvision' in text
    for path in re.findall(r"REPO/'([^']+)'",text):
        if 'requirements' in path:assert (run.ROOT/path).is_file()
    assert 'Video(str(' in text and 'pip' in text and 'venv' not in text and 'ensurepip' not in text
    assert "'--input-root',str(INPUT_ROOT)" in text
    assert 'sys.executable' in __import__('inspect').getsource(run.worker_phase)
    assert 'WanPipeline' not in text
