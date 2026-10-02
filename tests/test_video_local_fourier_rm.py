"""CPU mechanism/adapter checks; no real Wan/VAE/MP4 or scientific evidence."""
import ast,copy,gzip,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_local_fourier_rm_state as s,video_local_fourier_rm_control as m
from runtime.wan import video_local_fourier_rm as b,trajectory,io,vae
from experiments.wan_state_clock import video_local_fourier_rm_run as run
pytestmark=pytest.mark.unit

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)

def hist():
    from diffusers import UniPCMultistepScheduler
    x=UniPCMultistepScheduler(prediction_type='flow_prediction',thresholding=False,predict_x0=True,
        lower_order_final=True,use_flow_sigmas=True,flow_shift=3.,final_sigmas_type='zero')
    x.set_timesteps(50,device='cpu');x.set_begin_index(0);return x

class Transformer:
    def __call__(self,hidden_states,encoder_hidden_states,**kw):return (hidden_states*.01+encoder_hidden_states.mean(),)


def test_code_distance_local_fft_realizability_and_target_budget():
    import itertools
    for key in ('watermark','watermark-wrong'):
        code=s.state_code(key);assert code.shape==(45,32) and np.all(code.sum(axis=1)==0)
        assert min(np.count_nonzero(a!=c) for a,c in itertools.combinations(code,2))==16
        U=s.bases(key);assert U.shape==(4,64,32)
        np.testing.assert_allclose(U.sum(axis=1),0,atol=2e-14)
        np.testing.assert_allclose(U.transpose(0,2,1)@U,np.broadcast_to(np.eye(32),(4,32,32)),atol=4e-15)
        host=np.random.default_rng(17).normal(size=(8,8));F=np.fft.fft2(host,norm='ortho').real
        for i,row in enumerate(s.basis_layout(key)['blocks']):
            fft=[F[h,w]*factor*sign for (h,w),factor,sign in zip(row['modes'],row['factors'],row['signs'])]
            np.testing.assert_allclose(host.reshape(64)@U[i],fft,atol=4e-15)
        target=s.synthesize(key);assert np.linalg.norm(target)==pytest.approx(1,abs=1e-14)
        received=target[0,:,1:].transpose(1,0,2,3);q=s.extract(received,key,np.ones((45,4),bool))
        np.testing.assert_allclose(q,s.composite_signs(key)*s.PUBLIC.alpha,atol=5e-17)
        assert np.count_nonzero(s.composite_signs(key))==5568
        assert np.count_nonzero(target[0,4,0])==0 and np.count_nonzero(target[0,:4])==0
        U[:]=999;assert np.max(np.abs(s.bases(key)))<1
    assert not np.array_equal(s.state_code('watermark'),s.state_code('watermark-wrong'))


def test_soft_state_and_finite_path_ideal_cases_missing_support():
    source=s.synthesize('watermark')[0,:,1:].transpose(1,0,2,3)
    paths=[list(range(1,45)),list(range(2,46)),list(range(1,17))+list(range(16,44)),
           list(range(1,17))+list(range(18,33))]
    for taus in paths:
        out=s.infer(source[np.asarray(taus)-1],'watermark',np.ones((len(taus),4),bool))
        assert out['summary']['unique_model_hypothesis'] and not out['summary']['state_path_accepted']
        assert s.catalog(len(taus))[out['summary']['canonical_catalog_index']]['taus']==taus
        assert [v['top'] for v in out['local_state']['rows']]==[[t] for t in taus]
        assert out['local_state']['scored']==len(taus)*45
    missing=np.zeros((44,4),bool)
    assert s.infer(source[:44],'watermark',missing)['summary']['reason']=='NO_OBSERVATIONS'
    mask=np.ones((44,4),bool);mask[12]=False
    out=s.infer(source[:44],'watermark',mask)
    assert out['local_state']['rows'][12]['status']=='NO_SUPPORT'
    assert out['local_state']['scored']==43*45
    assert len(s.catalog(44))==3915 and out['counts']['scored']==174


def test_actual_gradient_matches_independent_fft_and_shrink_only_cap():
    for amplitude in (0.,20.):
        z=torch.randn((1,16,46,40,64),generator=torch.Generator().manual_seed(5))*amplitude
        targets=m.build_targets(z,'watermark',m.message_bits('OKOK'));delta,receipt=m.pilot_delta(z,targets)
        leaf=z.clone().requires_grad_(True);features=[]
        for i,((h0,h1,w0,w1),row) in enumerate(zip(s.PUBLIC.blocks,s.basis_layout('watermark')['blocks'])):
            F=torch.fft.fft2(leaf[0,4,1:,h0:h1,w0:w1],norm='ortho').real
            q=torch.stack([F[:,h,w]*factor*sign for (h,w),factor,sign in zip(row['modes'],row['factors'],row['signs'])],-1)
            features.append(q.reshape(45,4,8))
        q=torch.stack(features,1);loss=(q[targets['pilot_active']]-targets['pilot_target'][targets['pilot_active']]).square().mean()
        raw=-696.*torch.autograd.grad(loss,leaf)[0];norm=float(raw.double().norm());expected=raw*min(1.,1/norm)
        torch.testing.assert_close(delta,expected,atol=2e-8,rtol=3e-6)
        assert receipt['actual_delta_l2']==pytest.approx(min(norm,1.),abs=2e-7)
        if amplitude==0:assert receipt['actual_delta_l2']==pytest.approx(.25,abs=1e-7)
        else:assert receipt['cap_scale']<1
        closure=m.delta_closure(delta,m.project_tensor(delta,targets['basis']),targets)
        assert abs(closure['parseval_error'])<2e-7 and abs(closure['outside_ROI_energy'])<1e-12


def test_true_native_multi_frontend_no_last_replacement():
    receipts=[]
    for arm in m.ARMS:
        scheduler=hist();rows=[];ledger={};captured={};initial=torch.zeros(1,16,46,40,64)
        def count(kind,done):ledger.setdefault(kind,[0,0])[int(done)]+=1
        terminal,receipt=b.run_trajectory(SimpleNamespace(transformer=Transformer()),initial,scheduler,
            torch.tensor([.03]),torch.tensor([-.02]),torch.bfloat16,arm,'watermark',m.message_bits('OKOK'),count,rows.append,
            diagnostic=lambda i,a,d:captured.update({i:(a,d)}))
        assert len(rows)==50 and [r['index'] for r in rows if r['enabled']]==([] if arm=='OFF' else list(range(25,50)))
        assert ledger['native_step']==ledger['transformer_conditional']==ledger['transformer_unconditional']==[50,50]
        assert len(captured)==25 and all(v[0]['z_post'].shape==(45,4,4,8) for v in captured.values())
        assert receipt['writer_diagnostics']['last_z_post_matches_terminal'] is True
        if arm=='STATE_MULTI':assert ledger['pilot_gradient']==[25,25]
        else:assert 'pilot_gradient' not in ledger
        assert torch.count_nonzero(initial)==0;receipts.append(receipt)
    assert receipts[0]['before_step25']==receipts[1]['before_step25']==receipts[2]['before_step25']


def test_receiver_rename_and_missing_phase_are_not_truth_oracles(tmp_path,monkeypatch):
    paths=[tmp_path/'STATE_TRUE_PHASE0.mp4',tmp_path/'arbitrary.mp4']
    for p in paths:p.write_bytes(b'same saved bytes')
    monkeypatch.setattr(io,'read_mp4',lambda p:torch.zeros(181,320,512,3))
    full=torch.from_numpy(s.synthesize('watermark',dtype=np.float32));lengths=[]
    def encode(_,rgb):lengths.append(len(rgb));return full[:,:,:((len(rgb)-1)//4+1)]
    monkeypatch.setattr(vae,'reencode_rgb24_readback',encode)
    keys={'CORRECT':'watermark','WRONG':'watermark-wrong'}
    one=b.read_mp4_search(paths[0],keys,s.PUBLIC,object());two=b.read_mp4_search(paths[1],keys,s.PUBLIC,object())
    assert one==two and lengths==[181,177,177,177]*2
    for (g,k),v in one['features'].items():assert len(v['inference']['local_state']['rows'])==44
    n=0
    def fail(_,rgb):
        nonlocal n
        n+=1
        if n==2:raise RuntimeError('missing legal phase')
        return full[:,:,:((len(rgb)-1)//4+1)]
    monkeypatch.setattr(vae,'reencode_rgb24_readback',fail)
    partial=b.read_mp4_search(paths[0],keys,s.PUBLIC,object())
    assert len(partial['features'])==8 and all(v['status']=='SEARCH_INCOMPLETE' for v in partial['searches'].values())


def test_fixed_saved_media_pipeline_and_blind_posthoc_separation(tmp_path,monkeypatch):
    store=run.Store(tmp_path/'fixed',create=True);cfg=run.load_config()
    monkeypatch.setattr(b,'execution_device_dtype',lambda:('cpu',torch.float32))
    import runtime.wan.generation as gen
    pipe=SimpleNamespace(transformer=Transformer(),scheduler=hist())
    monkeypatch.setattr(gen,'prepare_generation',lambda *a,**kw:(pipe,torch.zeros(1,16,46,40,64),torch.tensor([.03]),torch.tensor([-.02]),torch.bfloat16))
    run.generation_worker(store,cfg)
    monkeypatch.setattr(gen,'load_frozen_vae',lambda *a,**kw:SimpleNamespace(parameters=lambda:iter([torch.zeros(1)])))
    monkeypatch.setattr(vae,'decode_normalized_latent',lambda *a:torch.zeros(181,320,512,3))
    monkeypatch.setattr(io,'encode_rgb',lambda rgb,path,*a:Path(path).write_bytes(b'fixed RGB fixture'))
    monkeypatch.setattr(io,'read_mp4',lambda p:torch.zeros(181,320,512,3))
    full=torch.from_numpy(s.synthesize('watermark',dtype=np.float32))
    monkeypatch.setattr(vae,'reencode_rgb24_readback',lambda _,rgb:full[:,:,:((len(rgb)-1)//4+1)])
    run.media_worker(store,cfg);raw=(store.output/'blind_readouts.json').read_bytes()
    run.evaluate_saved_reads(store,cfg);run.quality_diagnostics(store)
    assert (store.output/'blind_readouts.json').read_bytes()==raw and 'writer_diagnostics' not in json.loads(raw)
    assert store.data['counts']==dict(generated=3,source_mp4=3,derived_mp4=0,normalized=12,phase_searches=24,phase_payload=24,
        path_costs=4176,local_state_costs=47520,searches=6,evaluated=12)
    assert len(store.data['local_posthoc'])==6 and store.data['local_posthoc']['STATE_MULTI/CORRECT']['unique_truth_rows']==44
    assert store.data['writer_diagnostic_counts']['saved']==75
    assert not any(v['accepted_payload'] for v in store.data['evaluations'].values())
    calls={}
    for stages in store.data['calls'].values():
        for kind,row in stages.items():
            for k,v in row.items():calls.setdefault(kind,dict(attempted=0,completed=0))[k]+=v
    for kind,n in cfg['planned_calls'].items():assert calls.get(kind,dict(attempted=0,completed=0))==dict(attempted=n,completed=n)
    for arm in m.ARMS:assert store.data['sources'][arm]['path']==store.data['views'][arm+'/FULL_SOURCE181']['path']


def test_parent_spawn_failure_keeps_all_fixed_slots(tmp_path,monkeypatch):
    output=tmp_path/'failed';monkeypatch.setattr(sys,'argv',['test','--output',str(output)])
    monkeypatch.setattr(run.subprocess,'check_output',lambda *a,**kw:'0'*40+'\n')
    monkeypatch.setattr(run.subprocess,'Popen',lambda *a,**kw:(_ for _ in ()).throw(OSError('spawn fixture')))
    with pytest.raises(SystemExit):run.main()
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='INCOMPLETE' and len(result['local_posthoc'])==6
    assert all(v['status']!='PENDING' for group in run.TABLE_SIZES for v in result[group].values())


def test_notebook_and_pure_method_architecture(tmp_path):
    from scripts import build_video_local_fourier_rm_notebook as builder
    sha='0'*40;path=builder.build(sha,tmp_path/'fixed.ipynb');nb=json.loads(path.read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert nb['metadata']['candidate_binding']['source_sha']==sha
    text='\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code')
    assert 'video_local_fourier_rm_run' in text and 'Video-Local-Fourier-RM-V1' in text and 'venv' not in text
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['execution_count'] is None and c['outputs']==[]
    for mod in (s,m):
        tree=ast.parse(Path(mod.__file__).read_text())
        assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith(('runtime','experiments','governance')) for n in ast.walk(tree))
