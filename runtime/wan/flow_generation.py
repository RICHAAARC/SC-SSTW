"""Shared no-gradient continuation; orchestration supplies an optional controller."""
from __future__ import annotations
import torch


@torch.no_grad()
def continue_steps(pipe, scheduler, latent, prompt, negative, input_dtype,
                   guidance_scale, start, stop, count, control=None, precision=None):
    for index in range(start, stop):
        timestep = scheduler.timesteps[index]
        hidden = latent.to(input_dtype)
        time = timestep.expand(latent.shape[0])
        count('transformer', False)
        conditional = pipe.transformer(hidden_states=hidden, timestep=time,
            encoder_hidden_states=prompt, attention_kwargs=None, return_dict=False)[0]
        count('transformer', True)
        count('transformer', False)
        unconditional = pipe.transformer(hidden_states=hidden, timestep=time,
            encoder_hidden_states=negative, attention_kwargs=None, return_dict=False)[0]
        count('transformer', True)
        velocity = unconditional + guidance_scale*(conditional-unconditional)
        if precision is not None:
            precision.update(transformer_output_dtype=str(conditional.dtype), cfg_output_dtype=str(velocity.dtype),
                             scheduler_input_dtype='torch.float32',
                             off_reference='same-run matched FP32 scheduler path; no historical BF16 equivalence claim')
        # Preserve the original CFG arithmetic; normalize only its completed output.
        # Otherwise BF16 sigma multiplication in the baseline changes precision
        # when an FP32 delta is added, violating the fixed-history affine premise.
        velocity = velocity.float()
        if not torch.isfinite(velocity).all():
            raise FloatingPointError('nonfinite CFG velocity')
        if control is not None and index in (44, 45, 46):
            latent = control(index, scheduler, latent, velocity, timestep)
        else:
            count('scheduler_step', False)
            latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]
            count('scheduler_step', True)
        if not torch.isfinite(latent).all():
            raise FloatingPointError('nonfinite continuation latent')
        if scheduler.step_index != index+1:
            raise RuntimeError('UniPC continuation cursor diverged')
    return latent.detach()
