"""CPU engineering tests for fixed same-raster flow; never load a real VAE."""
import ast,gzip,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import video_local_fourier_rm_same_raster_run as run
from scripts import build_video_local_fourier_rm_same_raster_notebook as builder
pytestmark=pytest.mark.unit
SMALL=(2,2,4,3)

@pytest.fixture(autouse=True)
def cpu_threads():
    before=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def synthetic_baseline(root):
    root.mkdir();cfg=run.load_config();meta=dict(source_sha=cfg['baseline']['source_sha'],method_version=run.receiver.PUBLIC.method_version,generation={})
    for i,arm in enumerate(run.ARMS):
        folder=root/arm;folder.mkdir();path=folder/'terminal.pt';torch.save(torch.full((1,16,46,40,64),float(i)),path)
        meta['generation'][arm]=dict(status='COMPLETE',sha256=run.sha(path),path='/wrong/old/Colab/path/terminal.pt')
    run.dump(root/'result.json',meta);return root


def fake_runtime(monkeypatch,*,fail_encode=None,fail_load=False,fail_media=False):
    from runtime.wan import generation
    state=SimpleNamespace(decodes=[],encodes=[],quantized=[],media_sources=[])
    monkeypatch.setattr(run.media,'SHAPE',SMALL);monkeypatch.setattr(run.media,'RGB_BYTES',int(np.prod(SMALL)))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
            self.config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
            self.template=torch.zeros((1,16,46,40,64))
            basis=run.receiver.frozen.bases(run.load_config()['key']);signs=run.receiver.frozen.composite_signs(run.load_config()['key'])*run.receiver.PUBLIC.alpha
            for b,(h0,h1,w0,w1) in enumerate(run.receiver.PUBLIC.blocks):
                self.template[0,4,1:,h0:h1,w0:w1]=torch.tensor(signs[:,b].reshape(45,32)@basis[b].T,dtype=torch.float32).reshape(45,8,8)
        def decode(self,value,return_dict=False):
            assert tuple(value.shape)==(1,16,46,40,64) and not return_dict
            offset=float(value[0,0,0,0,0]);rgb=torch.linspace(-1.2,1.2,int(np.prod(SMALL))).reshape(1,3,*SMALL[:3])+offset/100
            state.decodes.append(rgb.clone());return (rgb,)
        def encode(self,video):
            state.encodes.append(video.clone())
            if fail_encode==len(state.encodes):raise RuntimeError('fake single-channel encode failure')
            assert video.dtype==torch.float32 and tuple(video.shape)==(1,3,*SMALL[:3]) and float(video.min())>=-1 and float(video.max())<=1
            normalized=self.template+float(video.mean())/1000
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:normalized))
    fake=FakeVAE()
    def load(cfg,device):
        assert device=='cpu'
        if fail_load:raise RuntimeError('fake VAE load failure')
        return fake
    monkeypatch.setattr(generation,'load_frozen_vae',load)
    original=run.vae_adapter.quantize_rgb8_no_codec
    def quantize(rgb):
        q8=original(rgb);state.quantized.append(q8.clone());return q8
    monkeypatch.setattr(run.vae_adapter,'quantize_rgb8_no_codec',quantize)
    def write_received(path,source,flip):
        output=source.clone();output.reshape(-1)[0]^=flip
        Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(output.numpy().tobytes())
        return output,dict(status='SAVED',path=str(path),sha256=run.sha(path),bytes=int(np.prod(SMALL)))
    def raw(source,yuv_path,rgb_path,*,count,event):
        source_sha=hashlib.sha256(source.numpy().tobytes()).hexdigest();state.media_sources.append(('RAW420',source_sha))
        count('rgb_to_raw420',False);Path(yuv_path).write_bytes(b'fake engineering YUV');count('rgb_to_raw420',True)
        event('yuv420',dict(status='SAVED',path=str(yuv_path),sha256=run.sha(yuv_path),input_raster_sha256=source_sha))
        count('raw420_to_rgb24',False)
        if fail_media:event('rgb24',dict(status='FAILED',error='fake RAW420 failure'));raise RuntimeError('fake RAW420 failure')
        output,row=write_received(rgb_path,source,1);count('raw420_to_rgb24',True);event('rgb24',row);return output
    def mp4(raster_path,raster_sha,mp4_path,rgb_path,*,count,event):
        source=run.media.reopen_raster(raster_path,raster_sha);state.media_sources.append(('MP4',hashlib.sha256(source.numpy().tobytes()).hexdigest()))
        count('mp4_save',False);Path(mp4_path).write_bytes(b'fake engineering MP4');count('mp4_save',True);event('mp4',dict(status='SAVED',path=str(mp4_path),sha256=run.sha(mp4_path),input_raster_sha256=raster_sha))
        count('mp4_probe',False);count('mp4_probe',True);event('probe',dict(status='COMPLETE'))
        count('mp4_readback',False);output,row=write_received(rgb_path,source,2);count('mp4_readback',True);event('rgb24',row);return output
    monkeypatch.setattr(run.raw420,'roundtrip',raw);monkeypatch.setattr(run.media,'mp4_roundtrip',mp4)
    return state


def execute_fake(root,monkeypatch,**kwargs):
    baseline=synthetic_baseline(root/'input');state=fake_runtime(monkeypatch,**kwargs);cfg=run.load_config();store=run.Store(root/'output',create=True,baseline=baseline)
    run.vae_worker(store,cfg,baseline);store.data['workers']['vae_media']=dict(status='COMPLETE');store.save()
    return store,cfg,state,baseline


def test_same_saved_bytes_three_decode_nine_encode_and_frozen_receivers(tmp_path,monkeypatch):
    store,cfg,state,_=execute_fake(tmp_path,monkeypatch)
    assert len(state.decodes)==3 and len(state.encodes)==9 and len(state.quantized)==3
    for i,arm in enumerate(run.ARMS):
        decoded=(state.decodes[i][0].permute(1,2,3,0)/2+.5).clamp(0,1)
        expected=np.rint(decoded.numpy()*255).astype(np.uint8)
        raster=store.data['rasters'][arm];assert Path(raster['path']).read_bytes()==expected.tobytes()
        assert state.media_sources[2*i:2*i+2]==[('RAW420',raster['sha256']),('MP4',raster['sha256'])]
        for c,channel in enumerate(run.CHANNELS):
            aid=arm+'/'+channel;transport=store.data['transport'][aid]
            assert transport['input_raster_sha256']==raster['sha256'] and transport['input_raster_path']==raster['path']
            observed=torch.from_numpy(np.frombuffer(Path(transport['events']['rgb24']['path']).read_bytes(),np.uint8).reshape(SMALL).copy())
            assert torch.equal(state.encodes[3*i+c],(observed.float()/255).permute(3,0,1,2).unsqueeze(0)*2-1)
    for sid,qr in store.data['projections'].items():
        qdata=json.loads(gzip.decompress(Path(qr['path']).read_bytes()));q=np.asarray(qdata['q']);mask=np.asarray(qdata['availability']);key=cfg['key'] if sid.endswith('/CORRECT') else cfg['wrong_key']
        for mode,fn in zip(run.MODES,(run.receiver.absolute_control,run.receiver.infer_difference)):
            row=store.data['mode_reads'][sid+'/'+mode];raw=json.loads(gzip.decompress(Path(row['path']).read_bytes()))
            assert raw['inference']==fn(q,key,mask) and raw['q_sha256']==qr['sha256']
            assert not raw['truth_inputs'] and not raw['inference']['summary']['state_path_accepted'] and not raw['inference']['summary']['accepted_payload']
    files=[store.output/'receiver_readouts.json',store.output/'payload_readouts.json'];before={str(p):p.read_bytes() for p in files}
    for p in files:
        text=p.read_text();assert 'registered_tau' not in text and 'true_path_rank' not in text and 'bit_errors' not in text and 'terminal.pt' not in text
    run.evaluate(store,cfg);assert before=={str(p):p.read_bytes() for p in files} and run.finish(store,cfg)
    assert store.data['counts']==dict(rasters=3,transport=9,normalized=9,keyed_projections=18,path_score_records=36,path_costs=6264,difference_edge_costs=102168,absolute_local_state_costs=35640,payload_reads=18,path_posthoc=36,difference_edge_posthoc=18,payload_evaluations=36)
    assert store.data['call_integrity']['status']=='MATCH' and not store.data['actual_generation_calls'] and not store.data['actual_writer_update_calls']
    assert all(row['source']=='NEW_SAME_RASTER_CHANNEL_READ' for row in store.data['payload_reads'].values())
    # An auxiliary failure cannot silently be marked complete even with complete counts.
    store.failure('FAKE_AUXILIARY',RuntimeError('retained engineering failure'));assert not run.finish(store,cfg)


def test_missing_arm_independently_retains_full_failure_matrix(tmp_path,monkeypatch):
    baseline=synthetic_baseline(tmp_path/'input');(baseline/'OFF'/'terminal.pt').unlink();state=fake_runtime(monkeypatch);cfg=run.load_config();store=run.Store(tmp_path/'output',create=True,baseline=baseline)
    run.vae_worker(store,cfg,baseline);store.data['workers']['vae_media']=dict(status='COMPLETE');run.evaluate(store,cfg)
    assert not run.finish(store,cfg) and store.data['terminal_inputs']['OFF']['status']=='FAILED'
    assert len(state.decodes)==2 and len(state.encodes)==6 and store.data['counts']['path_score_records']==24
    assert store.data['normalized']['STATE_MULTI/MP4']['status']=='SAVED'
    assert all(len(store.data[k])==n for k,n in run.SIZES.items())
    assert all(row['status']=='MISSING_READ' for name,row in store.data['path_posthoc'].items() if name.startswith('OFF/'))
    assert all(row['status'] not in ('PENDING','RUNNING') for group in ('normalized','projections','mode_reads','payload_reads') for row in store.data[group].values())


def test_single_encode_failure_leaves_other_channels_and_arm_readable(tmp_path,monkeypatch):
    store,cfg,state,_=execute_fake(tmp_path,monkeypatch,fail_encode=2);run.evaluate(store,cfg)
    assert not run.finish(store,cfg) and len(state.decodes)==3 and len(state.encodes)==9
    assert store.data['normalized']['OFF/RAW420']['status']=='FAILED' and store.data['transport']['OFF/RAW420']['status']=='COMPLETE'
    assert store.data['normalized']['OFF/MP4']['status']=='SAVED' and store.data['normalized']['STATE_MULTI/RAW420']['status']=='SAVED'
    assert store.data['counts']['normalized']==8 and store.data['counts']['path_score_records']==32 and store.data['counts']['payload_reads']==16
    assert store.data['path_posthoc']['OFF/RAW420/CORRECT/ADJACENT_DIFFERENCE']['status']=='MISSING_READ'


def test_vae_load_failure_preserves_all_rows_and_zero_decode(tmp_path,monkeypatch):
    baseline=synthetic_baseline(tmp_path/'input');state=fake_runtime(monkeypatch,fail_load=True);cfg=run.load_config();store=run.Store(tmp_path/'output',create=True,baseline=baseline)
    with pytest.raises(RuntimeError,match='fake VAE load'):run.vae_worker(store,cfg,baseline)
    run.settle(store,'fake worker load failure');run.evaluate(store,cfg)
    assert not run.finish(store,cfg) and not state.decodes and not state.encodes
    assert store.data['calls']['vae_load']==dict(attempted=1,completed=0) and store.data['counts']['path_score_records']==0
    assert all(len(store.data[k])==n for k,n in run.SIZES.items()) and all(r['status']=='MISSING_READ' for r in store.data['path_posthoc'].values())


def test_child_spawn_failure_retains_posthoc_denominator(tmp_path,monkeypatch):
    baseline=synthetic_baseline(tmp_path/'input');cfg=run.load_config();store=run.Store(tmp_path/'output',create=True,baseline=baseline)
    def fail(*args,**kwargs):raise OSError('fake child could not start')
    monkeypatch.setattr(run.subprocess,'Popen',fail)
    store,interrupted=run.run_child(store,baseline);run.evaluate(store,cfg)
    assert not interrupted and not run.finish(store,cfg) and store.data['workers']['vae_media']['status']=='FAILED'
    assert len(store.data['path_posthoc'])==36 and len(store.data['edge_posthoc'])==18 and len(store.data['payload_evaluations'])==36
    assert store.data['counts']['normalized']==0 and store.data['failures']


def test_saved_raster_reopen_requires_full_identity(tmp_path,monkeypatch):
    monkeypatch.setattr(run.media,'SHAPE',SMALL);monkeypatch.setattr(run.media,'RGB_BYTES',int(np.prod(SMALL)))
    source=torch.arange(int(np.prod(SMALL)),dtype=torch.uint8).reshape(SMALL);row=run.media.save_raster(source,tmp_path/'source.rgb8')
    assert torch.equal(source,run.media.reopen_raster(row['path'],row['sha256']))
    Path(row['path']).write_bytes(Path(row['path']).read_bytes()[:-1])
    with pytest.raises(ValueError,match='identity/byte'):run.media.reopen_raster(row['path'],row['sha256'])


def test_notebook_fixed_binding_setup_roster_and_minimal_repairs(tmp_path):
    nb=json.loads(builder.build('a'*40,tmp_path/'bound.ipynb').read_text());sources=[''.join(c['source']) for c in nb['cells']]
    assert sources[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell,text in zip(nb['cells'],sources):
        if cell['cell_type']=='code':ast.parse(text);assert cell['outputs']==[] and cell['execution_count'] is None
    code='\n'.join(sources);assert 'venv' not in '\n'.join(sources[2:]) and '--baseline' not in builder.RUN and '--mode' not in builder.RUN and 'WanPipeline' not in code
    assert 'AutoencoderKLWan' in builder.ENVIRONMENT and "shutil.which(name)" in builder.ENVIRONMENT and "['apt-get','install','-y','ffmpeg']" in builder.ENVIRONMENT
    assert "'MEDIA_TOOL_REPAIR',check=False" in builder.ENVIRONMENT and nb['metadata']['candidate_binding']['source_sha']=='a'*40
    prefix=builder.SETUP.split('def write_json')[0];scope=dict(SOURCE_SHA='a'*40,FIXED=run.FIXED)
    # Inspect literal comprehensions without executing Colab filesystem setup.
    setup_node=next(node for node in ast.parse(prefix).body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='setup' for t in node.targets))
    scope.update(ARMS=run.ARMS,KEYS=run.KEYS,CHANNELS=run.CHANNELS,MODES=run.MODES)
    setup=eval(compile(ast.Expression(setup_node.value),'<setup-roster>','eval'),scope)
    assert [len(setup[k]) for k in ('terminal_inputs','rasters','transport','normalized','projections','mode_reads','path_posthoc','edge_posthoc','payload_reads','payload_evaluations')]==[3,3,9,9,18,36,36,18,18,36]
    with pytest.raises(ValueError):builder.build('branch-name',tmp_path/'bad.ipynb')
    draft=json.loads(builder.build(None,tmp_path/'draft.ipynb').read_text());assert draft['metadata']['candidate_binding']['status']=='UNPUBLISHED_DRAFT'
