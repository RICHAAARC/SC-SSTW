"""Native UniPC last-step clean projection; no model or VAE differentiation."""
import copy,hashlib
import torch
from main.tube_state import terminal_guidance as method
from .flow_generation import continue_steps

def fingerprint(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            x=x.detach().cpu().contiguous();h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(tuple,list)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

@torch.no_grad()
def prepare_last(pipe,initial,prompt,negative,dtype,guidance,count):
    s=pipe.scheduler
    if type(s).__name__!='UniPCMultistepScheduler' or s.config.prediction_type!='flow_prediction' or not s.predict_x0 or s.config.thresholding or s.solver_p is not None:
        raise ValueError('native nonthresholded UniPC flow predict_x0 required')
    if len(s.timesteps)!=50 or not s.config.lower_order_final or float(s.sigmas[-1])!=0.:raise ValueError('fixed 50 steps lower-order zero-sigma final required')
    precision={}
    z=continue_steps(pipe,s,initial,prompt,negative,dtype,guidance,0,49,count,precision=precision)
    t=s.timesteps[49].expand(z.shape[0]);values=[]
    for embedding in (prompt,negative):
        count('transformer',False);values.append(pipe.transformer(hidden_states=z.to(dtype),timestep=t,encoder_hidden_states=embedding,attention_kwargs=None,return_dict=False)[0]);count('transformer',True)
    v=(values[1]+guidance*(values[0]-values[1])).float()
    if not torch.isfinite(v).all() or not float(s.sigmas[49])>0:raise FloatingPointError('invalid last velocity/sigma')
    return z.detach(),v.detach(),copy.deepcopy(s),precision

@torch.no_grad()
def final_step(snapshot,z,v,count,u=None):
    s=copy.deepcopy(snapshot);sigma=float(s.sigmas[49]);controlled=v if u is None else v-u/sigma
    if not torch.isfinite(controlled).all():raise FloatingPointError('nonfinite controlled velocity')
    count('scheduler_step',False);terminal=s.step(controlled,s.timesteps[49],z.clone(),return_dict=False)[0];count('scheduler_step',True)
    if not torch.isfinite(terminal).all() or s.step_index!=50 or s.this_order!=1:raise RuntimeError('invalid native final step')
    return terminal.detach(),{'sigma':sigma,'next_sigma':float(s.sigmas[50]),'native_order':s.this_order,
        'controlled_velocity_fingerprint':fingerprint(controlled)}
