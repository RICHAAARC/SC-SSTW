"""Fixed-state GROW transmission measurements using native UniPC steps."""
import copy
import hashlib
import torch
from main.tube_state import grow_frequency as method

ATOL=2e-5
RTOL=2e-4


def tree_cpu(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return type(value)({k:tree_cpu(v) for k,v in value.items()})
    if isinstance(value,list):return [tree_cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(tree_cpu(v) for v in value)
    return copy.deepcopy(value)


def fingerprint(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            a=x.detach().cpu().contiguous();h.update(str((a.dtype,tuple(a.shape))).encode());h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(str(type(x)).encode())
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def finite(value):
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
    return True


def maxabs(x):return float(x.detach().double().abs().max())


def rms(x):return float(x.detach().double().square().mean().sqrt())


def check_delta(actual,expected,*references):
    scale=max([1.]+[maxabs(x) for x in references])
    tolerance=ATOL*scale+RTOL*maxabs(expected)
    error=maxabs(actual-expected)
    return {'pass':finite((actual,expected,references)) and error<=tolerance,
            'max_abs_error':error,'max_abs_tolerance':tolerance,'reference_scale':scale,
            'expected_delta_maxabs':maxabs(expected),'expected_delta_rms':rms(expected),
            'tolerance_over_expected_maxabs':tolerance/maxabs(expected) if maxabs(expected) else None,
            'interpretation':'numerical compatibility only; no minimum scientific response'}


def require_scheduler(scheduler):
    from diffusers import UniPCMultistepScheduler
    if not isinstance(scheduler,UniPCMultistepScheduler) or scheduler.config.prediction_type!='flow_prediction' or not scheduler.predict_x0 or scheduler.config.thresholding or scheduler.solver_p is not None:
        raise ValueError('requires native UniPC flow_prediction predict_x0 without thresholding/solver_p')
    if len(scheduler.timesteps)!=50:raise ValueError('requires 50 original steps')


@torch.no_grad()
def cfg(pipe,z,prompt,negative,dtype,guidance,index,count):
    hidden=z.to(dtype);t=pipe.scheduler.timesteps[index].expand(z.shape[0])
    values=[]
    for embedding in (prompt,negative):
        count('transformer',False)
        values.append(pipe.transformer(hidden_states=hidden,timestep=t,encoder_hidden_states=embedding,attention_kwargs=None,return_dict=False)[0])
        count('transformer',True)
    v=(values[1]+guidance*(values[0]-values[1])).float()
    if not finite(v):raise FloatingPointError('nonfinite completed CFG')
    return v


@torch.no_grad()
def pulse(z,v,sigma,book,message,count):
    clean=z-sigma*v
    if message is None:return torch.zeros_like(z),v,clean,{'controlled':False}
    with torch.enable_grad():
        leaf=clean.detach().requires_grad_(True);loss=method.loss(leaf,book,message)
        count('local_gradient',False);gradient=torch.autograd.grad(loss,leaf)[0];count('local_gradient',True)
    u=-method.ETA*gradient.detach();controlled=v-u/sigma;after=z-sigma*controlled
    if not finite((u,controlled,after,loss)):raise FloatingPointError('nonfinite local control')
    return u,controlled,after,{'controlled':True,'loss_before':float(loss.detach()),
        'loss_after':float(method.loss(after,book,message)),
        'loss_ratio':float(method.loss(after,book,message)/loss.detach()) if float(loss.detach()) else None,
        'algebra':check_delta(after-clean,u,z,clean,after),'u_rms':rms(u)}


@torch.no_grad()
def scalar_response(snapshot,count):
    """Two real CPU native steps, preserving order/history presence, not Euler."""
    require_scheduler(snapshot)
    base=copy.deepcopy(snapshot);base.__dict__=tree_cpu(base.__dict__)
    zero=torch.zeros((1,1),dtype=torch.float64)
    base.model_outputs=[None if x is None else zero.clone() for x in base.model_outputs]
    base.last_sample=None if base.last_sample is None else zero.clone()
    results=[]
    for velocity in (zero,torch.ones_like(zero)):
        s=copy.deepcopy(base);count('response_probe_step',False)
        out=s.step(velocity,s.timesteps[s.step_index],zero,return_dict=False)[0]
        count('response_probe_step',True);results.append((out,s.last_sample))
    sigma=float(snapshot.sigmas[snapshot.step_index])
    if sigma<=0:raise ValueError('positive intervention sigma required')
    h=float((results[1][0]-results[0][0]).item())
    hc=float((results[1][1]-results[0][1]).item())
    if not finite((results[0],results[1])):raise FloatingPointError('nonfinite native response')
    return {'index':snapshot.step_index,'sigma':sigma,'h_velocity':h,'K_clean':-h/sigma,
            'K_corrected_sample':-hc/sigma,'probe_dtype':'torch.float64','probe_device':'cpu',
            'meaning':'fixed history current control response; two native solver calls, no model'}


def stats(delta,book):
    c=method.selected(delta.double(),book).reshape(46,64)
    s=torch.tensor(book['payloads'][0],device=c.device,dtype=c.dtype).repeat(4)
    return {'rms':rms(delta),'nonzero_fraction':float((delta!=0).double().mean()),
            'selected_coefficients':c.cpu().tolist(),
            'payload_A_signed_gain_by_time':(c*s).mean(-1).cpu().tolist(),
            'payload_A_signed_gain':float((c*s).mean()),'payload_B_signed_gain':float((-c*s).mean())}


def transmission(z,zoff,u,book,dtype):
    d=z-zoff;den=float(u.double().square().sum())
    khat=float((d.double()*u.double()).sum())/den if den else None
    return {'delta_z':stats(d,book),'delta_next_model_input':stats(z.to(dtype).float()-zoff.to(dtype).float(),book),
            'Khat':khat,'fit_residual_rms':rms(d-khat*u) if khat is not None else None,
            'fit_is_not_independent_validation':True}
