"""CPU-only difference carrier and native fake-model integration."""
from types import SimpleNamespace
import math
import pytest
import torch
from test_grow_control_transfer import native,Model
from main.tube_state import grow_frequency as old,grow_temporal_difference as method
from runtime.wan import grow_temporal_difference as rt,grow_late_control as prior
from experiments.wan_state_clock import grow_temporal_difference_run as runner
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12));monkeypatch.setattr(old,'SHAPE',method.SHAPE)
    return torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(47))*.2

def test_gradient_projection_common_mode_and_energy(small):
    book=method.codebook(b'fixed');z=small.clone().requires_grad_();loss=method.loss(z,book,0);g=torch.autograd.grad(loss,z)[0]
    e=method.selected(z,book)-old.target(book,0,z)
    gc=old.selected(g,book)
    torch.testing.assert_close(gc[:,:,::2],e/math.sqrt(2),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(gc[:,:,1::2],-e/math.sqrt(2),atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(g.square().sum(),2*loss,atol=1e-4,rtol=2e-5)
    after=z-.1*g
    torch.testing.assert_close(after[:,:,::2]+after[:,:,1::2],z[:,:,::2]+z[:,:,1::2],atol=1e-7,rtol=1e-6)
    assert torch.count_nonzero(g[:,1:])==0
    spectrum=old.dct2(g[:,0]);spectrum[:,:,2:10,2:10]=0;assert float(spectrum.abs().max())<2e-6
    torch.testing.assert_close(method.loss(after,book,0)/loss,torch.tensor(.81),atol=2e-6,rtol=1e-5)

def test_payload_roundtrip_zero_and_tie(small):
    book=method.codebook(b'fixed');c=torch.zeros_like(small);uv=torch.tensor(book['coordinates']);p=torch.tensor(book['payloads'][0]).repeat(4)*.5
    spatial=torch.zeros_like(c[:,0:1]);spatial[:,:,::2,uv[:,0],uv[:,1]]=p/math.sqrt(2);spatial[:,:,1::2,uv[:,0],uv[:,1]]=-p/math.sqrt(2)
    c[:,0:1]=old.idct2(spatial);rd=method.read(c,book)
    assert rd['aggregate']['votes_per_bit']==92 and len(rd['per_pair'])==23
    assert method.compare_payloads(rd,book)['aggregate'][0]['exact_payload_match']
    zeros=method.read(torch.zeros_like(c),book);assert zeros['aggregate']['bit_erasures']==16
    # Two positive and two negative repetitions for every pair create exact voting ties.
    spatial[:,:,::2,uv[:,0],uv[:,1]]=torch.tensor([1.]*32+[-1.]*32)/math.sqrt(2)
    spatial[:,:,1::2]=-spatial[:,:,::2];c[:,0:1]=old.idct2(spatial)
    assert method.read(c,book)['aggregate']['bit_erasures']==16

def test_native_off_unchanged_and_marked_pulse(small,native):
    book=method.codebook(b'fixed');args=(small,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,book,None,lambda *x:None,lambda x:None,lambda *x:None)
    new=rt.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*args,control_indices=rt.CONTROL_INDICES)
    before=prior.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*args,control_indices=prior.CONTROL_INDICES)
    torch.testing.assert_close(new,before,rtol=0,atol=0)
    rows=[];calls=[];model=Model()
    rt.generate(SimpleNamespace(transformer=model,scheduler=native()),small,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,book,0,lambda k,d:calls.append((k,d)),rows.append,lambda *x:None,control_indices=rt.CONTROL_INDICES)
    assert calls.count(('local_gradient',True))==20 and calls.count(('response_probe_step',True))==20
    assert all(abs(r['loss_ratio']-.81)<1e-5 for r in rows if r['controlled'])
    assert rows[-1]['terminal_vs_last_controlled_clean_rms']<1e-6
    assert old.CONTROL_INDICES==tuple(range(10,30)) and not any(model.grad_enabled)

def test_manifest_failure_roster_gate():
    runner.validate(runner.load(runner.MANIFEST));row=runner.empty('FAILED');assert len(row['videos'])==3
    assert all(len(v['terminal']['per_pair'])==23 for v in row['videos'].values())
    assert not runner.media_gate({})['media_eligible']
    cases={c:runner.empty('EXECUTION_COMPLETE')|{'exit_code':0} for c in runner.CASES}
    for case in cases.values():
        for arm in ('A','B'):
            case['videos'][arm].update(status='COMPLETE',terminal={'status':'COMPLETE','aggregate':{'bit_erasures':0},'payload_comparisons_reporting_only':{'aggregate':[{'exact_payload_match':arm=='A'},{'exact_payload_match':arm=='B'}]}})
    assert runner.media_gate(cases)['media_eligible']
    cases[runner.CASES[0]]['exit_code']=1
    assert runner.media_gate(cases)['exact_no_erasure']==6
    bad=runner.load(runner.MANIFEST);bad['frequency']['temporal_carrier']['votes_per_bit']=184
    with pytest.raises(ValueError):runner.validate(bad)

def test_fake_case_new_all_arms(small,native,tmp_path,monkeypatch):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(runner,'prepare_generation',lambda *a,**kw:(pipe,small,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    result=runner.run_case(runner.CASES[0],tmp_path/'case')
    assert result['status']=='EXECUTION_COMPLETE'
    assert result['actual_calls']['transformer_completed']==300
    assert result['actual_calls']['response_probe_step_completed']==40
    assert all(v['terminal']['aggregate']['votes_per_bit']==92 for v in result['videos'].values())
    assert result['videos']['OFF']['difference_domain_rms']>=0
