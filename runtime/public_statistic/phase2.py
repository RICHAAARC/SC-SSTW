"""Differentiable Wan1-style remaining-rollout adapter. No weights/loading/GPU launcher."""
from dataclasses import dataclass,asdict
from contextlib import nullcontext
import math
from main.sc_sstw.public_luma_statistic import read_rgb_differentiable
from main.sc_sstw.terminal_feedback import clone_graph_state,terminal_feedback

@dataclass(frozen=True)
class FeedbackConfig:
    feedback_after_indices:tuple
    learning_rate:float
    per_update_rms:float
    cumulative_rms:float
    terminal_quality_rms:float
    target_amplitude:float
    target_axis:int
    target_sign:int
    guidance_scale:float=5.
    status:str='CPU_ENGINEERING_CONFIGURATION_NOT_GPU_APPROVAL'
    def validate(self,total_steps):
        if tuple(sorted(set(self.feedback_after_indices)))!=self.feedback_after_indices:raise ValueError('feedback indices')
        if any(type(i) is not int or i<0 or i>=total_steps-1 for i in self.feedback_after_indices):raise ValueError('feedback after index out of range')
        if any(not math.isfinite(x) or x<=0 for x in (self.learning_rate,self.per_update_rms,self.cumulative_rms,self.terminal_quality_rms,self.target_amplitude)):raise ValueError('positive budgets')
        if not math.isfinite(self.guidance_scale) or self.guidance_scale<1:raise ValueError('finite guidance >=1 required')
        if self.target_axis not in (0,1) or self.target_sign not in (-1,1):raise ValueError('target axis/sign')

@dataclass
class SolverState:
    scheduler:object
    next_index:int

class WanTerminalAdapter:
    """Real Wan transformer and VAE call signatures; one transformer, standard timestep.
    expand_timesteps/dual-transformer Wan2.2 unsupported. FP32 floating VAE only.
    Runtime offload/checkpoint/backward with real Wan weights remains unverified.
    """
    def __init__(self,transformer,vae,prompt_embeds,negative_prompt_embeds,*,guidance_scale,torch,transformer_2=None,boundary_ratio=None,expand_timesteps=False,offload_enabled=False,persistent_transformer_cache=False,resource_guard=None):
        if transformer_2 is not None or boundary_ratio is not None or expand_timesteps or offload_enabled or persistent_transformer_cache:raise ValueError('unsupported Wan2.2/expanded/offload/persistent-cache path')
        self.resource_guard=resource_guard
        self.torch=torch;self.transformer=transformer;self.vae=vae
        self.prompt=prompt_embeds.detach();self.negative=negative_prompt_embeds.detach() if negative_prompt_embeds is not None else None;self.guidance_scale=guidance_scale
        for model in (transformer,vae):
            model.eval()
            for p in model.parameters():p.requires_grad_(False)
        if any(p.dtype!=torch.float32 for p in vae.parameters()):raise ValueError('VAE must be independently loaded FP32 weights')
        self.counts=dict(transformer_calls=0,vae_calls=0,solver_steps=0)
    def _context(self,name):return self.transformer.cache_context(name) if hasattr(self.transformer,'cache_context') else nullcontext()
    def velocity(self,z,t):
        params=list(self.transformer.parameters());dtype=params[0].dtype if params else z.dtype
        hidden=z.to(dtype);timestep=t.expand(z.shape[0])
        if self.resource_guard:self.resource_guard.consume('transformer_calls')
        with self._context('cond'):
            cond=self.transformer(hidden_states=hidden,timestep=timestep,encoder_hidden_states=self.prompt,attention_kwargs=None,return_dict=False)[0];self.counts['transformer_calls']+=1
        if self.guidance_scale>1:
            if self.negative is None:raise ValueError('CFG negative embeddings missing')
            if self.resource_guard:self.resource_guard.consume('transformer_calls')
            with self._context('uncond'):
                uncond=self.transformer(hidden_states=hidden,timestep=timestep,encoder_hidden_states=self.negative,attention_kwargs=None,return_dict=False)[0];self.counts['transformer_calls']+=1
            return uncond+self.guidance_scale*(cond-uncond)
        return cond
    def advance(self,z,state):
        i=state.next_index
        if i>=len(state.scheduler.timesteps):raise ValueError('solver already complete')
        internal=state.scheduler.step_index
        expected=internal if internal is not None else (state.scheduler.begin_index if state.scheduler.begin_index is not None else 0)
        if i!=expected:raise ValueError('external index and scheduler internal step index mismatch')
        t=state.scheduler.timesteps[i];prediction=self.velocity(z,t)
        result=state.scheduler.step(prediction,t,z,return_dict=False)[0]
        state.next_index=i+1;self.counts['solver_steps']+=1
        return result,state
    def decode_float(self,z):
        torch=self.torch;z=z.to(torch.float32);config=self.vae.config
        mean=z.new_tensor(config.latents_mean).reshape(1,config.z_dim,1,1,1);std=z.new_tensor(config.latents_std).reshape(1,config.z_dim,1,1,1)
        if mean.shape[1]!=z.shape[1] or not bool(torch.isfinite(std).all()) or bool((std<=0).any()):raise ValueError('VAE scaling')
        if self.resource_guard:self.resource_guard.consume('vae_calls')
        self.counts['vae_calls']+=1
        try:decoded=self.vae.decode(z*std+mean,return_dict=False)[0]
        finally:
            clear=getattr(self.vae,'clear_cache',None) or getattr(self.vae,'_clear_cache',None)
            if clear is not None:clear()
        if decoded.ndim!=5 or decoded.shape[0]!=1 or decoded.shape[1]!=3:raise ValueError('single B,C,T,H,W RGB output required')
        return (decoded[0].permute(1,2,3,0)/2+.5).clamp(0,1)
    def rollout(self,z,state):
        while state.next_index<len(state.scheduler.timesteps):z,state=self.advance(z,state)
        return self.decode_float(z)
    def readout(self,rgb):return read_rgb_differentiable(rgb,self.torch)

def controlled_run(adapter,latent,state,config,*,off1_terminal_rgb,enabled,strict=False,event_callback=None):
    """No retries. Fixed OFF terminal reference and additive target at every event.
    Caller must supply OFF1 from the same normal full endpoint path.
    """
    torch=adapter.torch;config.validate(len(state.scheduler.timesteps))
    if config.guidance_scale!=adapter.guidance_scale:raise ValueError('configuration guidance differs from actual adapter')
    state=clone_graph_state(state,torch);z=latent.detach().clone()
    reference=off1_terminal_rgb.detach().clone();target=adapter.readout(reference).detach().clone();target[:,config.target_axis]+=config.target_sign*config.target_amplitude
    used=0.;records=[]
    while state.next_index<len(state.scheduler.timesteps):
        index=state.next_index
        # Executed normal solver steps never reset or translate UniPC history.
        with torch.no_grad():z,state=adapter.advance(z,state)
        if enabled and index in config.feedback_after_indices:
            remaining=config.cumulative_rms-used
            if remaining<=0:record=dict(accepted=False,reason='CUMULATIVE_BUDGET_EXHAUSTED',attempts=0)
            else:
                z,record=terminal_feedback(z,state,adapter.rollout,adapter.readout,target,reference,learning_rate=config.learning_rate,step_rms=min(config.per_update_rms,remaining),quality_rms_budget=config.terminal_quality_rms,torch=torch,strict=strict,before_backward=(lambda:adapter.resource_guard.consume('backward_calls')) if adapter.resource_guard else None)
                if record['accepted']:used+=record['actual_update_rms']
            record.update(after_index=index,cumulative_accepted_rms=used,next_normal_index=state.next_index);records.append(record)
            if event_callback:event_callback(record)
    with torch.no_grad():rgb=adapter.decode_float(z);q=adapter.readout(rgb)
    return dict(latent=z,state=state,rgb=rgb,q=q,target=target,reference_rgb=reference,feedback=records,configuration=asdict(config),cumulative_accepted_rms=used)
