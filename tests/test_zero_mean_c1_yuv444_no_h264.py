"""Necessary mock/static tests only; no real FFmpeg conversion or VAE/model."""
import ast,copy,importlib.util,json,re
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_overlap_zero_mean_state as state
from runtime.wan import zero_mean_c1_yuv444_no_h264 as b,generation
from experiments.wan_state_clock import zero_mean_c1_yuv444_no_h264_run as run
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
    reference420=tmp_path/'reference420';reference420.mkdir()
    raw_by_key={k:dict(inference=state.infer(latent()[0,:,1:45].numpy().transpose(1,0,2,3),v,np.ones((44,4),bool)),
                      payload=dict(status='READ',decoded_bits=[0]*32,R=44)) for k,v in cfg['keys'].items()}
    for r in cfg['fixed_input']['required_q8']:
        p=source/r['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);torch.save(q8(),p)
        r.update(sha256=b.file_sha256(p),bytes=p.stat().st_size)
    for r in cfg['fixed_input']['reference_artifacts']:
        p=(source if r['root_id']=='layered' else reference420)/r['relative_path']
        p.parent.mkdir(parents=True,exist_ok=True)
        if r['kind']=='normalized':torch.save(latent(),p)
        else:run.dump_gzip(p,raw_by_key[r['key_id']])
        r['sha256']=b.file_sha256(p)
    parent=dict(source_sha=cfg['fixed_input']['source_sha'],path_posthoc={
        a+'/RGB8_NO_CODEC/'+k:dict(fixed_historical_phase0_opponent=dict(status='COMPARED',catalog_index=34,path=state.catalog(44)[34],raw_sha256='fixture-old-phase0-sha'))
        for a in run.ARMS for k in run.KEYS})
    rp=source/'result.json';run.dump(rp,parent);cfg['fixed_input']['result_sha256']=b.file_sha256(rp)
    rp420=reference420/'result.json'
    run.dump(rp420,dict(source_sha=cfg['reference420_input']['source_sha'],fixed_opponents={'OVERLAP_MULTI/CORRECT':{'catalog_index':86}}))
    cfg['reference420_input']['result_sha256']=b.file_sha256(rp420)
    store=run.Store(tmp_path/'output',create=True,cfg=cfg,input_root=source,reference420_root=reference420);run.preflight(store,cfg)
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
        with path.open('wb') as f:f.truncate(19 if fail else size-1 if short_stage==('yuv444' if first else 'rgb24') else size)
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
    cfg=run.load_config();a,c=b.conversion_commands('{yuv444_output}','{rgb24_output}')
    assert a==cfg['conversion']['segment1_argv']
    assert c==[s.replace('{yuv444_input}','{yuv444_output}') for s in cfg['conversion']['segment2_argv']]
    assert b.YUV_BYTES==181*320*512*3 and b.RGB_BYTES==181*320*512*3
    assert sum((b.YUV_BYTES,b.RGB_BYTES))*3==533790720
    assert cfg['fixed_denominator']['valid_costs']==1044
    assert len(cfg['fixed_input']['reference_artifacts'])==27

def test_full_mock_chain_reference_reuse_and_raw_posthoc_isolation(tmp_path,monkeypatch):
    store,cfg=fixture_store(tmp_path);calls=install_fake_commands(monkeypatch)
    real_infer=state.infer;infer_calls=[]
    def tracked_infer(*args,**kwargs):
        infer_calls.append(1);return real_infer(*args,**kwargs)
    monkeypatch.setattr(state,'infer',tracked_infer)
    run.conversion_worker(store,cfg)
    store.checkpoint();before=b.file_sha256(store.output/'blind_readouts.json')
    assert run.finalize(store,cfg)
    assert store.data['primary_execution']==store.data['status']=='EXECUTION_COMPLETE'
    assert store.data['reference_completeness']==store.data['package_completeness']=='COMPLETE'
    assert store.data['counts']==dict(yuv444_files=3,rgb24_files=3,normalized=3,path_reads=6,payload_reads=6,valid_costs=1044,
        catalog_slots=23490,structural_exclusions=22446,path_posthoc=6,message_evaluations=12)
    assert [c['first'] for c in calls]==[True,False]*3
    for k,n in cfg['planned_calls'].items():assert store.data['calls'][k]['completed']==n
    assert len(store.data['reference_posthoc'])==18
    assert len(infer_calls)==6
    assert all(r['status']=='VERIFIED' for r in store.data['references'].values())
    for counts in store.data['reference_counts'].values():
        assert counts==dict(normalized=dict(expected=3,verified=3),raw=dict(expected=6,verified=6))
    assert all(r['new_infer_calls']==0 for r in store.data['reference_posthoc'].values())
    assert b.file_sha256(store.output/'blind_readouts.json')==before
    for a in run.ARMS:
        r=store.data['path_posthoc'][a+'/CORRECT']
        assert r['true_rank']==1 and r['truth_unique_top'] and r['canonical_start_correct']
        assert r['invented_edit'] is False
        assert r['fixed_historical_phase0_opponent']['catalog_index']==34
        assert r['projection']['sign_denominator']==1360 and r['projection']['dimensions']==1408
        assert r['projection']['target_energy']==pytest.approx(1360/1392)
    text=(store.output/'blind_readouts.json').read_text()
    assert all(w not in text for w in ('registered_tau','true_rank','OKOK','NOPE','fixed_historical_phase0_opponent'))
    assert not list(store.output.rglob('*.partial'))
    # Reference-only loss does not remove new primary evidence.
    (Path(store.data['reference420_root'])/'result.json').unlink()
    run.preflight(store,cfg)
    assert store.data['reference420_identity']['status']=='FAILED'
    assert sum(r['status']=='VERIFIED' for r in store.data['references'].values())==18
    assert all(r['status']=='VERIFIED' for r in store.data['inputs'].values())
    assert run.finalize(store,cfg)
    assert store.data['primary_execution']=='EXECUTION_COMPLETE' and store.data['package_completeness']=='INCOMPLETE'
    assert store.data['path_posthoc']['OVERLAP_MULTI/CORRECT']['fixed_historical_phase0_opponent']['catalog_index']==34
    assert len(infer_calls)==6

def test_second_conversion_failure_keeps_first_and_other_arms(tmp_path,monkeypatch):
    store,cfg=fixture_store(tmp_path);install_fake_commands(monkeypatch,fail_arm='PAYLOAD_MULTI')
    run.conversion_worker(store,cfg)
    assert not run.finalize(store,cfg)
    assert store.data['conversions']['PAYLOAD_MULTI/yuv444']['status']=='SAVED'
    failed=store.data['conversions']['PAYLOAD_MULTI/rgb24']
    assert failed['status']=='FAILED' and failed['partial_bytes']==19 and Path(failed['partial_path']).is_file()
    assert store.data['counts']['path_reads']==4 and store.data['counts']['valid_costs']==696
    assert store.data['reference_completeness']=='COMPLETE'
    assert len(store.data['reads'])==6 and len(store.data['message_evaluations'])==12
    assert all(r['status']!='PENDING' for g in ('conversions','normalized','reads','path_posthoc','message_evaluations') for r in store.data[g].values())

def test_short_raw444_fails_before_second_process(tmp_path,monkeypatch):
    calls=install_fake_commands(monkeypatch,short_stage='yuv444');events=[];counts=[]
    with pytest.raises(ValueError,match='no truncation/padding'):
        b.roundtrip(q8(),tmp_path/'raw.yuv',tmp_path/'out.rgb',count=lambda k,v:counts.append((k,v)),event=lambda k,r:events.append((k,r)))
    assert len(calls)==1 and events[-1][1]['status']=='FAILED'
    assert events[-1][1]['partial_bytes']==b.YUV_BYTES-1 and not (tmp_path/'raw.yuv').exists()
    assert counts==[('rgb_to_raw444',False)]

def test_notebook_and_source_scope_static(tmp_path):
    path=run.ROOT/'scripts/build_zero_mean_c1_yuv444_no_h264_notebook.py'
    spec=importlib.util.spec_from_file_location('builder',path);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    draft=json.loads(builder.build(output=tmp_path/'draft.ipynb').read_text())
    assert draft['metadata']['candidate_binding']['source_sha'] is None
    assert draft['metadata']['candidate_binding']['status']=='UNPUBLISHED_DRAFT'
    assert 'SOURCE_SHA = None' in ''.join(draft['cells'][2]['source'])
    current=run.ROOT/'notebooks/zero_mean_c1_yuv444_no_h264_v1_colab.ipynb'
    binding=json.loads(current.read_text())['metadata']['candidate_binding'];source_sha=binding['source_sha']
    assert source_sha is None or re.fullmatch('[0-9a-f]{40}',source_sha)
    assert binding['status']==('PUBLISHED_SHA_BOUND' if source_sha else 'UNPUBLISHED_DRAFT')
    built=builder.build(source_sha,output=tmp_path/'new.ipynb')
    assert built.read_bytes()==current.read_bytes()
    nb=json.loads(built.read_text());code='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert f'SOURCE_SHA = {source_sha!r}' in code and '20260930T165530863126Z/fixed_reference' in code
    assert 'PYTHON=sys.executable' in code and 'torch,torchvision' in code
    assert 'true_rank' in code and 'orthogonal_residual_full1408_l2' in code
    assert '20261001T092411269871Z/fixed_reference' in code and '--reference420-root' in code
    assert all(x in code for x in ('canonical_catalog_index','top_catalog_indices','true_minus_winner','fixed_catalog_index'))
    setup_tree=ast.parse(''.join(nb['cells'][2]['source']))
    fixed=next(ast.literal_eval(n.value) for n in setup_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIXED' for t in n.targets))
    assert fixed==run.load_config()['fixed_denominator']
    assert all(x not in code for x in ('ensurepip','venv','prepare_generation','WanPipeline'))
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[] and c['execution_count'] is None
    for rel in re.findall(r"REPO/'([^']+)'",code):
        if 'requirements' in rel:assert (run.ROOT/rel).is_file()
    assert 'experiments.' not in Path(b.__file__).read_text()
    with pytest.raises(ValueError,match='read-only'):
        run.Store(tmp_path/'input'/'out',create=True,cfg=run.load_config(),input_root=tmp_path/'input')

    with pytest.raises(ValueError,match="read-only"):
        run.Store(tmp_path/"ref420"/"out",create=True,cfg=run.load_config(),input_root=tmp_path/"input",reference420_root=tmp_path/"ref420")
