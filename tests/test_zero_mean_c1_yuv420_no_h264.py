"""Necessary mock/static tests only; no real FFmpeg conversion or VAE/model."""
import ast,copy,importlib.util,json,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_overlap_zero_mean_state as state
from runtime.wan import zero_mean_c1_yuv420_no_h264 as b,generation
from experiments.wan_state_clock import zero_mean_c1_yuv420_no_h264_run as run
pytestmark=pytest.mark.unit

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)

def latent():return torch.from_numpy(state.synthesize('watermark',dtype=np.float32))
def q8():return torch.tensor(128,dtype=torch.uint8).reshape(1,1,1,1).expand(*b.RGB_SHAPE)

def fixture_store(tmp_path):
    cfg=copy.deepcopy(run.load_config());source=tmp_path/'input';source.mkdir()
    raw_by_key={k:dict(inference=state.infer(latent()[0,:,1:45].numpy().transpose(1,0,2,3),v,np.ones((44,4),bool)),
                      payload=dict(status='READ',decoded_bits=[0]*32,R=44)) for k,v in cfg['keys'].items()}
    for r in cfg['fixed_input']['required_q8']:
        p=source/r['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);torch.save(q8(),p)
        r.update(sha256=b.file_sha256(p),bytes=p.stat().st_size)
    for r in cfg['fixed_input']['reference_artifacts']:
        p=source/r['relative_path']
        if r['kind']=='normalized':torch.save(latent(),p)
        else:run.dump_gzip(p,raw_by_key[r['key_id']])
        r['sha256']=b.file_sha256(p)
    parent=dict(source_sha=cfg['fixed_input']['source_sha'],path_posthoc={
        a+'/RGB8_NO_CODEC/'+k:dict(fixed_historical_phase0_opponent=dict(status='COMPARED',catalog_index=1,path=state.catalog(44)[1],raw_sha256='fixture-old-phase0-sha'))
        for a in run.ARMS for k in run.KEYS})
    rp=source/'result.json';run.dump(rp,parent);cfg['fixed_input']['result_sha256']=b.file_sha256(rp)
    store=run.Store(tmp_path/'output',create=True,cfg=cfg,input_root=source);run.preflight(store,cfg)
    return store,cfg

class FakeVAE:
    config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
    def parameters(self):return iter([torch.nn.Parameter(torch.tensor(0.),requires_grad=False)])

def install_fake_commands(monkeypatch,*,fail_arm=None,short_stage=None):
    calls=[]
    def fake(command,**kwargs):
        if command[-1]=='-version':
            return SimpleNamespace(returncode=0,stdout='mock ffmpeg version; no codec executed\n',stderr='')
        assert command[0]=='ffmpeg' and '-v' in command and 'verbose' in command
        assert 'libx264' not in command and '-crf' not in command and '-vf' not in command
        first=command[command.index('-i')+1]=='pipe:0'
        path=Path(command[-1]);size=b.YUV_BYTES if first else b.RGB_BYTES
        if first:
            assert len(kwargs['input'])==b.RGB_BYTES and kwargs['input'][0]==128
            assert '-chroma_sample_location' not in command
        else:
            source=Path(command[command.index('-i')+1])
            assert source.is_file() and source.stat().st_size==b.YUV_BYTES
            assert command[command.index('-chroma_sample_location')+1]=='left'
            assert kwargs['input'] is None
        path.parent.mkdir(parents=True,exist_ok=True)
        fail=fail_arm is not None and fail_arm in path.parts and not first
        with path.open('wb') as f:f.truncate(19 if fail else size-1 if short_stage==('yuv420' if first else 'rgb24') else size)
        calls.append(dict(first=first,path=str(path)))
        return SimpleNamespace(returncode=1 if fail else 0,stdout=b'',stderr=b'mock auto_scale; intentionally not a real conversion\n')
    monkeypatch.setattr(b.subprocess,'run',fake)
    monkeypatch.setattr(generation,'load_frozen_vae',lambda *a,**k:FakeVAE())
    def encode(frozen,rgb):
        assert tuple(rgb.shape)==b.RGB_SHAPE and rgb.dtype==torch.float32
        assert torch.all(rgb==0) # actual saved sparse mock RGB bytes were reopened
        return latent()
    monkeypatch.setattr(b,'encode_normalized',encode)
    return calls

def test_frozen_command_pair_and_denominators():
    cfg=run.load_config();a,c=b.conversion_commands('{yuv420_output}','{rgb24_output}')
    assert a==cfg['conversion']['segment1_argv']
    assert c==[s.replace('{yuv420_input}','{yuv420_output}') for s in cfg['conversion']['segment2_argv']]
    assert b.YUV_BYTES==181*320*512*3//2 and b.RGB_BYTES==181*320*512*3
    assert sum((b.YUV_BYTES,b.RGB_BYTES))*3==400343040
    assert cfg['fixed_denominator']['valid_costs']==1044
    assert len(cfg['fixed_input']['reference_artifacts'])==18

def test_full_mock_chain_reference_reuse_and_raw_posthoc_isolation(tmp_path,monkeypatch):
    store,cfg=fixture_store(tmp_path);calls=install_fake_commands(monkeypatch)
    run.conversion_worker(store,cfg)
    store.checkpoint();before=b.file_sha256(store.output/'blind_readouts.json')
    assert run.finalize(store,cfg)
    assert store.data['primary_execution']==store.data['status']=='EXECUTION_COMPLETE'
    assert store.data['reference_completeness']==store.data['package_completeness']=='COMPLETE'
    assert store.data['counts']==dict(yuv420_files=3,rgb24_files=3,normalized=3,path_reads=6,payload_reads=6,valid_costs=1044,
        catalog_slots=23490,structural_exclusions=22446,path_posthoc=6,message_evaluations=12)
    assert [c['first'] for c in calls]==[True,False]*3
    for k,n in cfg['planned_calls'].items():assert store.data['calls'][k]['completed']==n
    assert len(store.data['reference_posthoc'])==12
    assert all(r['new_infer_calls']==0 for r in store.data['reference_posthoc'].values())
    assert b.file_sha256(store.output/'blind_readouts.json')==before
    for a in run.ARMS:
        r=store.data['path_posthoc'][a+'/CORRECT']
        assert r['true_rank']==1 and r['truth_unique_top'] and r['canonical_start_correct']
        assert r['invented_edit'] is False
        assert r['fixed_historical_phase0_opponent']['catalog_index']==1
        assert r['projection']['sign_denominator']==1360 and r['projection']['dimensions']==1408
        assert r['projection']['target_energy']==pytest.approx(1360/1392)
    text=(store.output/'blind_readouts.json').read_text()
    assert all(w not in text for w in ('registered_tau','true_rank','OKOK','NOPE','fixed_historical_phase0_opponent'))
    assert not list(store.output.rglob('*.partial'))
    # Reference-only loss does not remove new primary evidence.
    first=next(iter(store.data['references'].values()));first['status']='MISSING_OR_INVALID'
    assert run.finalize(store,cfg)
    assert store.data['primary_execution']=='EXECUTION_COMPLETE' and store.data['package_completeness']=='INCOMPLETE'

def test_second_conversion_failure_keeps_first_and_other_arms(tmp_path,monkeypatch):
    store,cfg=fixture_store(tmp_path);install_fake_commands(monkeypatch,fail_arm='PAYLOAD_MULTI')
    run.conversion_worker(store,cfg)
    assert not run.finalize(store,cfg)
    assert store.data['conversions']['PAYLOAD_MULTI/yuv420']['status']=='SAVED'
    failed=store.data['conversions']['PAYLOAD_MULTI/rgb24']
    assert failed['status']=='FAILED' and failed['partial_bytes']==19 and Path(failed['partial_path']).is_file()
    assert store.data['counts']['path_reads']==4 and store.data['counts']['valid_costs']==696
    assert store.data['reference_completeness']=='COMPLETE'
    assert len(store.data['reads'])==6 and len(store.data['message_evaluations'])==12
    assert all(r['status']!='PENDING' for g in ('conversions','normalized','reads','path_posthoc','message_evaluations') for r in store.data[g].values())

def test_short_raw420_fails_before_second_process(tmp_path,monkeypatch):
    calls=install_fake_commands(monkeypatch,short_stage='yuv420');events=[];counts=[]
    with pytest.raises(ValueError,match='no truncation/padding'):
        b.roundtrip(q8(),tmp_path/'raw.yuv',tmp_path/'out.rgb',count=lambda k,v:counts.append((k,v)),event=lambda k,r:events.append((k,r)))
    assert len(calls)==1 and events[-1][1]['status']=='FAILED'
    assert events[-1][1]['partial_bytes']==b.YUV_BYTES-1 and not (tmp_path/'raw.yuv').exists()
    assert counts==[('rgb_to_raw420',False)]

def test_projection_metrics_full_boundary_and_zero_sign():
    mu=state.composite_signs('watermark')[:44].astype(np.float64)*state.PUBLIC.alpha
    q=.75*mu;q[0,0,1,0]=2. # boundary coordinate
    stats=run.projection_metrics(q,mu)
    assert stats['signal_gain']==pytest.approx(.75)
    assert stats['orthogonal_residual_full1408_l2']==pytest.approx(2.)
    assert stats['active_dimensions']==1360 and stats['boundary_dimensions']==48
    assert stats['target_energy']==pytest.approx(1360/1392)
    q[0,0,0,0]=0.
    zero=run.projection_metrics(q,mu)
    assert zero['sign_agreements']==1359 and zero['zero_observation_active']==1
    assert zero['orthogonal_residual_full1408_l2']==pytest.approx(np.linalg.norm(q-(np.sum(q*mu)/np.sum(mu*mu))*mu))

@pytest.mark.parametrize('after_canonical',[False,True])
def test_canonical_failure_window_preserves_committed_conversion(tmp_path,monkeypatch,after_canonical):
    store,cfg=fixture_store(tmp_path)
    store.data['conversions']['OFF/yuv420'].update(status='SAVED',sha256='fixture')
    def stop():raise SystemExit('fixture hard exit')
    monkeypatch.setattr(store,'blind' if after_canonical else 'save',stop)
    with pytest.raises(SystemExit):store.checkpoint()
    restored=run.Store(store.output);run.recover_unfinished(restored,'fixture')
    assert restored.data['conversions']['OFF/yuv420']['status']==('SAVED' if after_canonical else 'NOT_COMPLETED')
    assert len(restored.data['reads'])==6

def test_monitor_cleanup_owns_group_and_precedes_latest_reload(tmp_path,monkeypatch):
    store,cfg=fixture_store(tmp_path);events=[]
    class Stream:
        def __iter__(self):
            latest=run.Store(store.output);latest.data['conversions']['OFF/yuv420']['status']='SAVED';latest.save()
            raise OSError('monitor failure')
        def close(self):events.append('close')
    class Child:
        pid=12345678;stdout=Stream()
        def wait(self,timeout=None):events.append('wait');return -15
    def spawn(*a,**k):
        assert k['start_new_session'] is True
        return Child()
    monkeypatch.setattr(run.subprocess,'Popen',spawn)
    monkeypatch.setattr(run.os,'killpg',lambda pid,sig:events.append(('killpg',pid,sig)))
    restored,halt=run.worker_phase(store.output)
    assert not halt and restored.data['conversions']['OFF/yuv420']['status']=='SAVED'
    assert events[:3]==[('killpg',12345678,run.signal.SIGTERM),'wait',('killpg',12345678,run.signal.SIGKILL)]

def test_notebook_and_source_scope_static(tmp_path):
    path=run.ROOT/'scripts/build_zero_mean_c1_yuv420_no_h264_notebook.py'
    spec=importlib.util.spec_from_file_location('builder',path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    built=builder.build(output=tmp_path/'new.ipynb')
    assert built.read_bytes()==(run.ROOT/'notebooks/zero_mean_c1_yuv420_no_h264_v1_colab.ipynb').read_bytes()
    nb=json.loads(built.read_text());code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert 'SOURCE_SHA = None' in code and '20260930T165530863126Z/fixed_reference' in code
    assert 'PYTHON=sys.executable' in code and 'torch,torchvision' in code
    assert 'true_rank' in code and 'orthogonal_residual_full1408_l2' in code
    assert all(x not in code for x in ('ensurepip','venv','prepare_generation','WanPipeline'))
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[] and c['execution_count'] is None
    for rel in re.findall(r"REPO/'([^']+)'",code):
        if 'requirements' in rel:assert (run.ROOT/rel).is_file()
    assert 'experiments.' not in Path(b.__file__).read_text()
    with pytest.raises(ValueError,match='read-only'):
        run.Store(tmp_path/'input'/'out',create=True,cfg=run.load_config(),input_root=tmp_path/'input')
