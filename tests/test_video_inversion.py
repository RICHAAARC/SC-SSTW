"""CPU mathematical, real scheduler, and fake-media receiver isolation checks."""
import inspect
import json
import weakref
from types import SimpleNamespace
import pytest
import torch
from main.tube_state import initial_noise as method
from runtime.wan.flow_inversion import public_schedule,reverse_grid,invert_received_latent
from runtime.wan.flow_generation import continue_steps
from experiments.wan_state_clock import video_inversion_run as runner
from test_velocity_direction import scheduler

pytestmark=pytest.mark.unit
torch.set_num_threads(1)
PAYLOADS=runner.load(runner.MANIFEST)['payloads']


def test_sign_pad_magnitude_payload_free_reader_and_erasures():
    book=method.codebook(b'fixed-test')
    assert set(book)=={'order','pads'}
    base=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(7))
    a,b=[method.write(base,book,p) for p in PAYLOADS]
    torch.testing.assert_close(a[:,0].abs(),base[:,0].abs(),rtol=0,atol=0)
    torch.testing.assert_close(a[:,1:],base[:,1:],rtol=0,atol=0)
    torch.testing.assert_close(a[:,0],-b[:,0],rtol=0,atol=0)
    for marked,payload in zip((a,b),PAYLOADS):
        decoded=method.read(marked,book)
        assert decoded['aggregate']['signs']==payload and decoded['aggregate']['votes_per_bit']==7360
        assert all(v['signs']==payload and v['votes_per_bit']==160 for v in decoded['per_time'])
    tied=a.clone();tied[0,0].reshape(46,2560)[:,book['order'][:1280]]*=-1
    assert method.read(tied,book)['aggregate']['bit_erasures']==16
    assert method.read(torch.zeros_like(base),book)['aggregate']['bit_erasures']==16


class Model(torch.nn.Module):
    def __init__(self,value=2.):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(value));self.times=[]
        self.config=SimpleNamespace(_commit_hash='cpu-fake')
    def forward(self,hidden_states,timestep,encoder_hidden_states,**kw):
        self.times.append((float(timestep[0]),torch.is_grad_enabled()))
        return (torch.ones_like(hidden_states)*self.weight,)


def test_reverse_left_nodes_constant_flow_direction_and_endpoint():
    s=scheduler();grid=public_schedule(s)
    r,tau=reverse_grid(grid['sigmas'],grid['timesteps'])
    assert len(r)==51 and len(tau)==50 and r[0]==0 and tau[0]==0
    assert tau[1:].tolist()==list(reversed(grid['timesteps']))[:-1]
    assert not torch.equal(tau[1:],r[1:-1]*1000) # native integer timestep truncation retained
    model=Model();calls=[];rows=[]
    noise=torch.ones(1,1,1,2,2)*3
    terminal=noise-float(r[-1])*2 # exact constant-flow forward solution
    recovered=invert_received_latent(model,terminal,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,
        grid['sigmas'],grid['timesteps'],lambda k,d:calls.append((k,d)),rows.append)
    torch.testing.assert_close(recovered,noise,atol=2e-6,rtol=0)
    assert [x[0] for x in model.times]==[float(t) for t in tau for _ in range(2)]
    assert not any(x[1] for x in model.times) and model.weight.grad is None
    assert calls.count(('transformer',True))==100 and calls.count(('inverse_update',True))==50
    assert all(v['delta_sigma']>0 for v in rows)
    assert rows[-1]['sigma_right']==grid['sigmas'][0] and rows[-1]['timestep_left']==grid['timesteps'][1]
    assert 'truth' not in inspect.signature(invert_received_latent).parameters
    assert not any(k in inspect.signature(invert_received_latent).parameters for k in ('noise','history','terminal','seed','message'))


def test_real_unipc_forward_counts_and_public_json_roundtrip():
    s=scheduler();schedule=public_schedule(s)
    assert json.loads(json.dumps(schedule))==public_schedule(scheduler())
    model=Model(.1);pipe=SimpleNamespace(transformer=model);calls=[]
    value=continue_steps(pipe,s,torch.ones(1,1,1,2,2),torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,0,50,lambda k,d:calls.append((k,d)))
    assert s.step_index==50 and s.model_outputs[-1] is not None and torch.isfinite(value).all()
    assert calls.count(('transformer',True))==100 and calls.count(('scheduler_step',True))==50
    assert not any(v[1] for v in model.times)
    broken=schedule['sigmas'][:-1]
    with pytest.raises(ValueError):reverse_grid(broken,schedule['timesteps'])
    wrong=list(schedule['sigmas']);wrong[-1]=.01
    with pytest.raises(ValueError):reverse_grid(wrong,schedule['timesteps'])


def test_staged_mp4_receiver_seed0_no_writer_reads_and_partial_save(tmp_path,monkeypatch):
    loaded=[];model_refs=[];vae_refs=[];read_pixels=[];inverse_inputs=[];saved={}
    initial=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(8))
    def prepare(config,*,load_vae):
        assert load_vae is False
        if loaded:assert config['generation']['seed']==0 and vae_refs[0]() is None
        loaded.append(config['generation']['seed'])
        model=Model();model_refs.append(weakref.ref(model))
        return SimpleNamespace(transformer=model,scheduler=scheduler()),(initial if len(loaded)==1 else torch.full_like(initial,999)),torch.tensor(1.),torch.tensor(-1.),torch.float32
    def load_vae(config):
        assert model_refs[0]() is None
        vae=Model();vae_refs.append(weakref.ref(vae));return vae
    def forward(*args,**kw):
        count=args[9]
        for _ in range(100):count('transformer',False);count('transformer',True)
        for _ in range(50):count('scheduler_step',False);count('scheduler_step',True)
        return args[2]*.1
    def encode(rgb,path,fps,crf):
        saved[path]=torch.full_like(rgb,.234 if path.stem=='OFF' else .345)
        if path.stem=='B':raise RuntimeError('intentional failed save leaves residue')
    def vae_encode(vae,rgb):read_pixels.append(rgb.clone());return torch.full_like(initial,float(rgb.mean()))
    def invert(transformer,received,prompt,negative,dtype,cfg,sigmas,timesteps,count,record):
        inverse_inputs.append(received.clone())
        assert float(received.mean())<1 # never loader seed0 dummy999
        for j in range(50):
            for _ in range(2):count('transformer',False);count('transformer',True)
            count('inverse_update',False);count('inverse_update',True);record({'index':j})
        return received+1
    monkeypatch.setattr(runner,'prepare_generation',prepare);monkeypatch.setattr(runner,'load_frozen_vae',load_vae)
    monkeypatch.setattr(runner,'continue_steps',forward);monkeypatch.setattr(runner,'invert_received_latent',invert)
    monkeypatch.setattr(runner,'RGB_SHAPE',(5,4,4,3))
    monkeypatch.setattr(runner,'decode_normalized_latent',lambda *a:torch.full((5,4,4,3),.123))
    monkeypatch.setattr(runner,'encode_rgb',encode);monkeypatch.setattr(runner,'read_mp4',lambda p:saved[p])
    monkeypatch.setattr(runner,'reencode_rgb24_readback',vae_encode)
    real_load=torch.load
    def guarded_load(path,*a,**kw):
        # Writer artifacts are allowed for generation/media and posthoc, never while receiver model exists.
        if len(model_refs)==2 and model_refs[1]() is not None:assert '/writer/' not in str(path)
        return real_load(path,*a,**kw)
    monkeypatch.setattr(runner.torch,'load',guarded_load)
    result=runner.run_case(runner.CASES[0],tmp_path/'case')
    assert loaded==[20260916,0] and len(inverse_inputs)==2
    assert all(float(v.mean())!=pytest.approx(.123) for v in read_pixels)
    assert result['receiver_contract']['writer_state_used'] is False
    assert result['videos']['B']['media_status']=='FAILED' and result['videos']['B']['inversion_status']=='MISSING_MP4_LATENT'
    assert all(len(v['decoded']['per_time'])==46 for v in result['videos'].values())
    assert result['actual_calls']['generation']['transformer_completed']==300
    assert result['actual_calls']['inversion']['transformer_completed']==200
    assert result['actual_calls']['inversion']['inverse_update_completed']==100
    assert result['actual_calls']['media']['vae_encode_completed']==2
    assert result['videos']['A']['posthoc']['status']=='MEASURED'
    assert model_refs[-1]() is None and vae_refs[-1]() is None
    assert result['status']=='WITH_RETAINED_FAILURES' and result['scientific_pass'] is None


def test_four_cases_retained_after_missing_child(tmp_path,monkeypatch):
    calls=[]
    def child(command,**kw):calls.append(command);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.subprocess,'run',child)
    result=runner.run_all(tmp_path/'run')
    assert len(calls)==4 and result['video_denominator']==12
    assert all(len(v['videos'])==3 for v in result['cases'].values())
    assert all(len(a['decoded']['per_time'])==46 for c in result['cases'].values() for a in c['videos'].values())
    assert result['call_count_case_coverage']==0 and result['status']=='WITH_RETAINED_FAILURES'
