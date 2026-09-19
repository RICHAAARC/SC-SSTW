"""Synthetic engineering only; no pretrained models, video experiment, or historical rescoring."""
import copy,hashlib,importlib.util,json,os,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest,torch
from main.tube_state import projection_margin as carrier,state_clock,terminal_guidance as method
from runtime.wan import tube_terminal_guidance as rt
from experiments.wan_state_clock import flow_tube_state_guidance_run as run
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

@pytest.fixture
def native(monkeypatch):
    import diffusers
    source=os.environ.get('DIFFUSERS_UNIPC_SOURCE')
    if source:
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()=='5bfe1dcf55ebea6dbbf624d3af676b2529b81fbcaf493150d562ec9e1aba3872'
        name='diffusers.schedulers._tube_native040';spec=importlib.util.spec_from_file_location(name,source);m=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,name,m);spec.loader.exec_module(m);cls=m.UniPCMultistepScheduler
    else:cls=diffusers.UniPCMultistepScheduler
    def make():
        s=cls(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3.);s.set_timesteps(50);s.set_begin_index(0);return s
    return make

@pytest.fixture(scope='module')
def book():return state_clock.codebook(b'WanProjection-first-validation-key-v1')

class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1));self.config=SimpleNamespace(_commit_hash='fake');self.flags=[]
    def forward(self,hidden_states,timestep,encoder_hidden_states,**kw):
        self.flags.append(torch.is_grad_enabled());return (hidden_states*self.weight+encoder_hidden_states*.001,)


def test_original_projection_preconditioned_margin_and_support(book):
    z=np.random.default_rng(16).normal(size=carrier.SHAPE).astype(np.float32)*.1
    u,p,e=method.correction(z,book,0);original,oe=carrier.write(z,book,0)
    np.testing.assert_array_equal(p,original)
    changed=copy.deepcopy(book);changed['payload']=-changed['payload'];np.testing.assert_array_equal(carrier.write(z,changed,0)[0],original)
    x=carrier.blocks(z).reshape(1760,1024).astype(float);d=book['directions'].astype(float);c=book['codes'][0]
    deficit=np.maximum(0,1-c*np.einsum('ij,ij->i',x,d));gradient=-2/1760*(deficit*c)[:,None]*d
    expected=-gradient*1760/(2*(d*d).sum(1)[:,None])
    np.testing.assert_allclose(carrier.blocks(u).reshape(1760,1024),expected,rtol=3e-5,atol=6e-8)
    assert np.count_nonzero(u[:,:,0])==np.count_nonzero(u[:,:,45])==0
    assert e['minimum_signed_projection_after']>1-1e-5
    expected_code=np.stack([s[np.arange(1760)//160,np.arange(1760)%2]*book['polarity']*book['sync'] for s in book['states']])
    np.testing.assert_array_equal(book['codes'],expected_code)


def test_native_last_equals_direct_with_shared_history(native,book):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(1))*.1;calls=[]
    count=lambda k,d:calls.append((k,d))
    z,v,s,_=rt.prepare_last(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.bfloat16,5.,count);before=rt.fingerprint(vars(s))
    off,info=rt.final_step(s,z,v,count);clean=z-float(s.sigmas[49])*v
    assert method.numerical_equivalence(off,clean)['pass']
    for m in (0,1):
        u,p,_=method.correction(clean.numpy(),book,m);last,_=rt.final_step(s,z,v,count,torch.from_numpy(u));direct,_=carrier.write(off.numpy(),book,m)
        assert method.numerical_equivalence(last,p)['pass'] and method.numerical_equivalence(last,direct)['pass']
    assert before==rt.fingerprint(vars(s)) and info['native_order']==1 and info['next_sigma']==0
    assert calls.count(('transformer',True))==100 and calls.count(('scheduler_step',True))==52 and not any(pipe.transformer.flags)


def generate_fixture(monkeypatch,native,path):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());z=torch.zeros(carrier.SHAPE)
    monkeypatch.setattr(run,'prepare_generation',lambda *a,**kw:(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    r=run.generate_case(run.CASES[0],path);assert r['status']=='GENERATION_COMPLETE',r['failures']
    return r


def media_fixture(monkeypatch,path,fail=None):
    class VAE(torch.nn.Module):
        def __init__(self):super().__init__();self.w=torch.nn.Parameter(torch.ones(1))
        def to(self,*a,**kw):return self
    monkeypatch.setattr(run,'RGB_SHAPE',(181,4,4,3));monkeypatch.setattr(run,'load_frozen_vae',lambda *a:VAE());monkeypatch.setattr(run,'_clear_cache',lambda *a:None)
    def decode(*a):return torch.full(run.RGB_SHAPE,float('nan')) if fail=='decode_nan' else (torch.arange(181).float()/181).reshape(181,1,1,1).expand(run.RGB_SHAPE).clone()
    monkeypatch.setattr(run,'decode_normalized_latent',decode);cache={};reads=[];lengths=[];blind=[]
    def save(rgb,p,*a):
        p.parent.mkdir(exist_ok=True);p.write_bytes(b'mock');cache[str(p)]=rgb
        if fail=='partial_off' and p.stem=='OFF':raise RuntimeError('partial saved file')
    def read(p):reads.append(p.stem);return cache[str(p)]
    def encode(vae,rgb):
        assert float(rgb[0,0,0,0])==pytest.approx((len(lengths)%4)/181)
        lengths.append(len(rgb));return torch.full((1,16,1+(len(rgb)-1)//4,40,64),float('nan') if fail=='encode_nan' else 0.)
    def receiver(obs,public_book):
        blind.append((sorted(obs),set(public_book)))
        return {'rankings':{mode:{'best':{'message':0},'message_unique':True} for mode in run.MODES}}
    monkeypatch.setattr(run,'encode_rgb',save);monkeypatch.setattr(run,'read_mp4',read);monkeypatch.setattr(run,'reencode_rgb24_readback',encode)
    monkeypatch.setattr(run.state_clock,'read',receiver);monkeypatch.setattr(run.state_clock,'report',lambda *a:{'reporting_only':True})
    return reads,lengths,blind


def test_five_arm_media_uses_real_rgb_origins_and_blind_api(monkeypatch,native,tmp_path):
    path=tmp_path/'case';generate_fixture(monkeypatch,native,path);reads,lengths,blind=media_fixture(monkeypatch,path)
    r=run.media_case(run.CASES[0],path)
    assert r['status']=='EXECUTION_COMPLETE',r['failures']
    assert lengths==[181,177,177,177]*5 and len(blind)==5 and all(v[0]==[0,1,2,3] for v in blind)
    assert all(r['actual_calls'][k+'_completed']==v for k,v in run.PLAN.items())
    assert all(v['status']=='COMPLETE' for v in r['videos'].values())

@pytest.mark.parametrize('failure',['partial_off','decode_nan','encode_nan'])
def test_media_invalid_data_or_partial_save_not_promoted(monkeypatch,native,tmp_path,failure):
    path=tmp_path/'case';generate_fixture(monkeypatch,native,path);reads,lengths,blind=media_fixture(monkeypatch,path,failure)
    r=run.media_case(run.CASES[0],path);assert r['status']=='WITH_RETAINED_FAILURES'
    if failure=='partial_off':
        assert 'OFF' not in reads and all(v['status']=='COMPLETE' for k,v in r['videos'].items() if k!='OFF')
        assert all(v['saved_quality_vs_off']['status']=='MISSING_OFF_REFERENCE' for k,v in r['videos'].items() if k!='OFF')
    elif failure=='decode_nan':assert reads==[] and lengths==[]
    else:assert all(v['status']!='COMPLETE' for v in r['videos'].values())
    assert len(r['videos'])==5 and all(len(v['observations'])==4 for v in r['videos'].values())


def test_full_failure_fixed_roster(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:(calls.append(a) or SimpleNamespace(returncode=1)))
    r=run.run_all(tmp_path/'full');assert len(calls)==8 and r['video_denominator']==20 and r['receiver_encode_denominator']==80
    assert sum(len(v['observations']) for c in r['cases'].values() for v in c['videos'].values())==80
    assert all(v['missing_or_failed']==8 for v in r['recovery_summary']['LAST'].values())
