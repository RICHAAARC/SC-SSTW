"""Synthetic-only single-step Jacobian direction and native history tests."""
import copy,json
from types import SimpleNamespace
import pytest,torch
from diffusers.models.modeling_utils import ModelMixin
from main.tube_state import grow_temporal_difference as method
from runtime.wan import grow_single_step_jacobian as rt
from runtime.wan.grow_control_transfer import cfg,fingerprint
from experiments.wan_state_clock import grow_single_step_jacobian_run as gen,grow_single_step_jacobian_full as full,grow_temporal_difference_media as media
from test_grow_control_transfer import native
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

class Model(ModelMixin):
    _supports_gradient_checkpointing=True
    def __init__(self,k=.1):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(k));self.gradient_checkpointing=False
        self.config=SimpleNamespace(_commit_hash='synthetic');self.calls=[]
    def block(self,x):return self.weight*x.square()
    def forward(self,hidden_states,timestep,encoder_hidden_states,**kwargs):
        self.calls.append(torch.is_grad_enabled())
        x=self._gradient_checkpointing_func(self.block,hidden_states) if torch.is_grad_enabled() and self.gradient_checkpointing else self.block(hidden_states)
        return (x*(1+encoder_hidden_states*.2)+encoder_hidden_states*.001,)

def inputs(monkeypatch,native,dtype=torch.float32,k=.1):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12))
    z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(317))*.1
    pipe=SimpleNamespace(transformer=Model(k),scheduler=native());stats={};rt.enable_input_checkpointing(pipe.transformer,stats)
    prompt,negative=torch.tensor(1.,dtype=dtype),torch.tensor(-1.,dtype=dtype)
    z=rt.prefix(pipe,z,prompt,negative,dtype,5.,lambda *a:None,lambda *a:None)
    v=cfg(pipe,z,prompt,negative,dtype,5.,30,lambda *a:None)
    return pipe,z,v,prompt,negative,method.codebook(b'jacobian'),stats

@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_full_cfg_gradient_checkpoint_equivalence_and_local_difference(monkeypatch,native,dtype):
    pipe,z,v,p,n,book,stats=inputs(monkeypatch,native,dtype);count=lambda *a:None
    local,_=rt.direction(pipe,z,v,p,n,dtype,5.,book,0,'LOCAL',count,stats)
    jac,info=rt.direction(pipe,z,v,p,n,dtype,5.,book,0,'JAC',count,stats)
    assert stats['last_direction_resources']['checkpoint_delta']['checkpoint_recompute_contexts']>0
    pipe.transformer.disable_gradient_checkpointing();leaf=z.detach().requires_grad_(True)
    vv=rt.differentiable_cfg(pipe,leaf,p,n,dtype,5.,count)
    expected=torch.autograd.grad(method.loss(leaf-float(pipe.scheduler.sigmas[30])*vv,book,0),leaf)[0]
    torch.testing.assert_close(jac,expected,rtol=0,atol=0)
    assert not torch.equal(jac,local) and not jac.requires_grad and info['forward_velocity_maxabs']==0
    assert not pipe.transformer.training and all(not p.requires_grad and p.grad is None for p in pipe.transformer.parameters())


def test_zero_jacobian_reduces_to_local(monkeypatch,native):
    pipe,z,v,p,n,b,stats=inputs(monkeypatch,native,k=0.)
    local,_=rt.direction(pipe,z,v,p,n,torch.float32,5.,b,0,'LOCAL',lambda *a:None,stats)
    jac,_=rt.direction(pipe,z,v,p,n,torch.float32,5.,b,0,'JAC',lambda *a:None,stats)
    torch.testing.assert_close(jac,local,rtol=0,atol=0)


def test_native_shared_history_energy_and_no_grad_tail(monkeypatch,native):
    pipe,z,v,p,n,b,stats=inputs(monkeypatch,native);origin=copy.deepcopy(pipe.scheduler);before=fingerprint(vars(origin));calls=[]
    count=lambda k,d:calls.append((k,d))
    off,_=rt.one_step(pipe,z,v,b,None,None,count)
    pipe.scheduler=copy.deepcopy(origin);local,_=rt.direction(pipe,z,v,p,n,torch.float32,5.,b,0,'LOCAL',count,stats)
    _,lr=rt.one_step(pipe,z,v,b,0,local,count,off_next=off)
    pipe.scheduler=copy.deepcopy(origin);jac,_=rt.direction(pipe,z,v,p,n,torch.float32,5.,b,0,'JAC',count,stats)
    after,jr=rt.one_step(pipe,z,v,b,0,jac,count,off_next=off,target_energy=lr['actual_response_energy'])
    assert fingerprint(vars(origin))==before and pipe.scheduler.step_index==31
    assert jr['actual_response_energy']==pytest.approx(lr['actual_response_energy'],rel=2e-4,abs=1e-10)
    pipe.transformer.calls.clear();terminal=rt.tail(pipe,after,p,n,torch.float32,5.,count,lambda *a:None)
    assert not any(pipe.transformer.calls) and not terminal.requires_grad and calls.count(('unit_response_probe_step',True))==1


def test_fake_case_prefix_reuse_complete_and_media(monkeypatch,native,tmp_path):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12));z=torch.randn(method.SHAPE)*.1;pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(gen,'prepare_generation',lambda *a,**kw:(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    root=tmp_path/'generation';case=gen.CASES[0];g=gen.run_case(case,root/case)
    assert g['status']=='EXECUTION_COMPLETE',g['failures']
    assert all(g['actual_calls'][k+'_completed']==v for k,v in gen.PLAN.items())
    assert len(g['prefix_steps'])==30 and all(len(v['steps'])==20 for v in g['videos'].values())
    assert len({v['shared_input_fingerprint'] for v in g['videos'].values()})==1
    assert len({v['shared_history_fingerprint'] for v in g['videos'].values()})==1
    (root/'result.json').write_text(json.dumps({'cases':{case:g}}));monkeypatch.setattr(media,'RGB_SHAPE',(3,4,4,3))
    vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(_commit_hash='fake');cache={}
    monkeypatch.setattr(media,'load_frozen_vae',lambda *a:vae);monkeypatch.setattr(media,'decode_normalized_latent',lambda *a:torch.zeros(media.RGB_SHAPE)+.5)
    monkeypatch.setattr(media,'reencode_rgb24_readback',lambda *a:torch.zeros(method.SHAPE))
    def save(rgb,path,*a):path.parent.mkdir(exist_ok=True);path.write_bytes(b'fake');cache[str(path)]=rgb
    monkeypatch.setattr(media,'encode_rgb',save);monkeypatch.setattr(media,'read_mp4',lambda p:cache[str(p)])
    m=media.run_case(case,tmp_path/'media'/case,root,expected_source_run='generation',expected_source_commit=g['source_commit'],expected_manifest=gen.load(gen.MANIFEST),arms=gen.ARMS,messages=gen.MESSAGES)
    assert m['status']=='EXECUTION_COMPLETE' and m['actual_calls']['vae_encode_completed']==15


def test_full_failure_roster(tmp_path,monkeypatch):
    monkeypatch.setattr(gen,'run_all',lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic failure')));calls=[]
    monkeypatch.setattr(full.subprocess,'run',lambda *a,**kw:(calls.append(a) or SimpleNamespace(returncode=1)))
    r=full.run(tmp_path/'full');assert len(calls)==4 and r['video_denominator']==20 and r['layer_denominator']==80
    assert r['fixed_calls']['transformer']==1024 and r['fixed_calls']['input_vjp']==8
    assert all(len(c['videos'])==5 for c in r['stages']['generation']['cases'].values())
    assert all(v['missing_or_failed_marked']==8 for v in r['recovery_summary']['JAC'].values())


def test_same_field_mismatch_is_retained_not_substituted(monkeypatch,native):
    pipe,z,v,p,n,b,stats=inputs(monkeypatch,native);original=rt.differentiable_cfg
    monkeypatch.setattr(rt,'differentiable_cfg',lambda *a,**kw:original(*a,**kw)+1)
    with pytest.raises(RuntimeError,match='SAME_FIELD_FORWARD_MISMATCH'):
        rt.direction(pipe,z,v,p,n,torch.float32,5.,b,0,'JAC',lambda *a:None,stats)
    assert not stats['last_direction_diagnostics']['forward_consistency']['pass']
    assert stats['last_direction_resources']['seconds']>0


def test_jacobian_exception_retains_other_arms_without_fallback(monkeypatch,native,tmp_path):
    monkeypatch.setattr(method,'SHAPE',(1,2,46,12,12));z=torch.randn(method.SHAPE)*.1;pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(gen,'prepare_generation',lambda *a,**kw:(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    original=torch.autograd.grad;attempts=[]
    def broken(*args,**kwargs):
        attempts.append(1)
        if len(attempts)==3:raise RuntimeError('synthetic backward OOM no local fallback')
        return original(*args,**kwargs)
    monkeypatch.setattr(torch.autograd,'grad',broken)
    g=gen.run_case(gen.CASES[0],tmp_path/'case')
    assert len(g['videos'])==5 and g['videos']['JAC_A']['status']=='FAILED'
    assert g['videos']['JAC_A']['terminal']['status']=='NOT_RUN'
    assert g['actual_calls']['input_vjp_attempted']==2 and g['actual_calls']['input_vjp_completed']==1
    assert g['videos']['JAC_A']['last_direction_resources']['seconds']>0
    assert all(g['videos'][a]['status']=='COMPLETE' for a in gen.ARMS if a!='JAC_A')


def test_independent_two_cfg_jacobian_analytic_formula(monkeypatch,native):
    pipe,z,v,p,n,b,stats=inputs(monkeypatch,native);sigma=float(pipe.scheduler.sigmas[30]);scale=5.
    # Independent formula, not the runtime CFG helper; each branch has a distinct Jacobian.
    k=float(pipe.transformer.weight);ac=k*(1+float(p)*.2);au=k*(1+float(n)*.2)
    manual_v=(au*z.square()+float(n)*.001)+scale*((ac*z.square()+float(p)*.001)-(au*z.square()+float(n)*.001))
    clean=(z-sigma*manual_v).detach().requires_grad_(True)
    gclean=torch.autograd.grad(method.loss(clean,b,0),clean)[0]
    expected=(1-sigma*2*z*(au+scale*(ac-au)))*gclean
    jac,_=rt.direction(pipe,z,v,p,n,torch.float32,scale,b,0,'JAC',lambda *a:None,stats)
    torch.testing.assert_close(jac,expected,rtol=2e-5,atol=2e-7)
    cosine=float((jac.double()*gclean.double()).sum()/(jac.double().norm()*gclean.double().norm()))
    assert cosine<1-1e-6
