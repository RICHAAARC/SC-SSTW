"""CPU objective, explicit injection and unconditional orchestration checks."""
import copy
from types import SimpleNamespace
import pytest,torch
from main.tube_state import grow_signed_margin as margin,grow_temporal_difference as carrier,grow_frequency as spatial
from runtime.wan import grow_temporal_difference as rt
from experiments.wan_state_clock import grow_temporal_margin_full as full,grow_temporal_margin_run as gen,grow_temporal_difference_media as media
from test_grow_control_transfer import native,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

def test_one_sided_loss_gradient_and_inactive_margins(monkeypatch):
    monkeypatch.setattr(carrier,'SHAPE',(1,2,46,12,12));z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(8)).requires_grad_();book=carrier.codebook(b'test')
    d=carrier.selected(z,book);bits=torch.tensor(book['payloads'][0],dtype=d.dtype).repeat(4).reshape(1,1,1,64)
    value=margin.loss(z,book,0);g=torch.autograd.grad(value,z)[0]
    expect=-bits*torch.relu(.5-bits*d);gc=spatial.selected(g,book)
    torch.testing.assert_close((gc[:,:,::2]-gc[:,:,1::2])/2**.5,expect,atol=3e-6,rtol=3e-5)
    torch.testing.assert_close(gc[:,:,::2]+gc[:,:,1::2],torch.zeros_like(expect),atol=3e-6,rtol=3e-5)
    assert torch.count_nonzero(g[:,1:])==0
    torch.testing.assert_close(margin.loss(z-.1*g,book,0),value*.81,atol=1e-4,rtol=2e-5)
    assert float(value)<=float(carrier.loss(z,book,0))

def test_default_unchanged_and_explicit_margin_pulse(monkeypatch,native):
    monkeypatch.setattr(carrier,'SHAPE',(1,2,46,12,12));z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(9))*.2;book=carrier.codebook(b'test')
    args=(z,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,book,0,lambda *x:None,lambda x:None,lambda *x:None)
    a=rt.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*args,control_indices=rt.CONTROL_INDICES)
    b=rt.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*args,control_indices=rt.CONTROL_INDICES,loss_fn=carrier.loss)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    u,v,after,info=rt.pulse(z,z*.1,.5,book,0,lambda *x:None,loss_fn=margin.loss)
    assert info['loss_before']==pytest.approx(float(margin.loss(z-.5*z*.1,book,0)))
    assert info['loss_after']==pytest.approx(float(margin.loss(after,book,0)))
    gen.validate(gen.load(gen.MANIFEST))

def test_terminal_failure_does_not_skip_media(tmp_path,monkeypatch):
    monkeypatch.setattr(gen,'run_all',lambda output:{'status':'WITH_RETAINED_FAILURES','media_gate':{'media_eligible':False},'actual_calls_observed':{'transformer_completed':1200}})
    calls=[]
    def child(command,**kw):calls.append(command);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(full.subprocess,'run',child)
    r=full.run(tmp_path/'out')
    assert len(calls)==4 and r['video_denominator']==12 and r['layer_denominator']==48
    assert len(r['stages']['media']['cases'])==4 and all(v['missing_or_failed_marked']==8 for v in r['recovery_summary'].values())
    assert r['fixed_calls']['transformer']==1200 and r['fixed_calls']['vae_encode']==36
    assert r['actual_calls_observed']['transformer_completed']==1200

def test_old_media_source_defaults_unchanged():
    import inspect
    sig=inspect.signature(media.run_case)
    assert sig.parameters['expected_source_run'].default==media.SOURCE_RUN
    assert sig.parameters['expected_source_commit'].default==media.SOURCE_COMMIT
    assert sig.parameters['expected_manifest'].default is None


def test_generation_exception_preserves_fixed_slots(tmp_path,monkeypatch):
    def fail(output):raise RuntimeError('generation failed before result')
    monkeypatch.setattr(gen,'run_all',fail)
    calls=[]
    monkeypatch.setattr(full.subprocess,'run',lambda *a,**kw:(calls.append(a) or SimpleNamespace(returncode=1)))
    r=full.run(tmp_path/'out')
    assert len(r['stages']['generation']['cases'])==4
    assert all(len(c['videos'])==3 for c in r['stages']['generation']['cases'].values())
    assert r['fixed_calls']['transformer']==1200 and len(calls)==4


def test_margin_case_then_same_saved_media(tmp_path,monkeypatch,native):
    import json
    monkeypatch.setattr(carrier,'SHAPE',(1,2,46,12,12));monkeypatch.setattr(media,'RGB_SHAPE',(3,4,4,3))
    z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(43))*.1
    pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(gen,'prepare_generation',lambda *a,**kw:(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    source=tmp_path/'generation';case=gen.CASES[0];result=gen.run_case(case,source/case)
    assert result['status']=='EXECUTION_COMPLETE' and 'terminal_writer_objective_losses' in result['videos']['A']
    (source/'result.json').write_text(json.dumps({'cases':{case:result}}))
    vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(_commit_hash='fake');cache={}
    monkeypatch.setattr(media,'load_frozen_vae',lambda *a:vae)
    monkeypatch.setattr(media,'decode_normalized_latent',lambda *a:torch.zeros(media.RGB_SHAPE)+.5)
    monkeypatch.setattr(media,'reencode_rgb24_readback',lambda *a:torch.zeros(carrier.SHAPE))
    def save(rgb,path,*args):path.parent.mkdir(exist_ok=True);path.write_bytes(b'fake');cache[str(path)]=rgb
    monkeypatch.setattr(media,'encode_rgb',save);monkeypatch.setattr(media,'read_mp4',lambda p:cache[str(p)])
    measured=media.run_case(case,tmp_path/'media'/case,source,expected_source_run='generation',expected_source_commit=result['source_commit'],expected_manifest=gen.load(gen.MANIFEST))
    assert measured['status']=='EXECUTION_COMPLETE'
    assert measured['effective_manifest']['writer_objective']['margin']==.5
    assert measured['actual_calls']['vae_encode_completed']==9 and measured['actual_calls']['transformer_completed']==0
