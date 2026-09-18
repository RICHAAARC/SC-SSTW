"""Post-CFG local predicted-clean control; no Transformer gradient or solver replay."""
import math
import torch
from main.tube_state import grow_frequency as method


@torch.no_grad()
def generate(pipe,initial,prompt,negative,input_dtype,guidance_scale,book,message,count,record):
    from diffusers import UniPCMultistepScheduler
    scheduler=pipe.scheduler
    if not isinstance(scheduler,UniPCMultistepScheduler) or scheduler.config.prediction_type!='flow_prediction' or not scheduler.predict_x0 or scheduler.config.thresholding or scheduler.solver_p is not None:
        raise ValueError('requires native UniPC flow_prediction, predict_x0, no thresholding or solver_p')
    if tuple(initial.shape)!=method.SHAPE:raise ValueError('fixed full latent geometry required')
    if len(scheduler.timesteps)!=50:raise ValueError('fixed 50-step schedule required')
    latent=initial.detach().float()
    for index,timestep in enumerate(scheduler.timesteps):
        hidden=latent.to(input_dtype);time=timestep.expand(latent.shape[0])
        count('transformer',False)
        conditional=pipe.transformer(hidden_states=hidden,timestep=time,encoder_hidden_states=prompt,attention_kwargs=None,return_dict=False)[0]
        count('transformer',True);count('transformer',False)
        unconditional=pipe.transformer(hidden_states=hidden,timestep=time,encoder_hidden_states=negative,attention_kwargs=None,return_dict=False)[0]
        count('transformer',True)
        velocity=(unconditional+guidance_scale*(conditional-unconditional)).float()
        if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('nonfinite CFG velocity')
        sigma=float(scheduler.sigmas[index])
        if not math.isfinite(sigma):raise FloatingPointError('nonfinite scheduler sigma')
        row={'index':index,'sigma':sigma,'controlled':False,'cfg':'completed CFG then FP32',
            'control_rms':0.,'delta_velocity_rms':0.}
        if message is not None and index in method.CONTROL_INDICES and sigma>0:
            clean=(latent-sigma*velocity).detach()
            with torch.enable_grad():
                leaf=clean.requires_grad_(True)
                loss=method.loss(leaf,book,message)
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite masked DCT loss')
                count('local_gradient',False)
                gradient=torch.autograd.grad(loss,leaf)[0]
                count('local_gradient',True)
            if not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite local DCT gradient')
            control=-method.ETA*gradient.detach()
            controlled_velocity=velocity-control/sigma
            if not bool(torch.isfinite(controlled_velocity).all()):raise FloatingPointError('nonfinite controlled velocity')
            effective=controlled_velocity-velocity
            clean_after=latent-sigma*controlled_velocity
            row.update(controlled=True,loss_before=float(loss.detach()),loss_after_local=float(method.loss(clean_after,book,message)),
                masked_mse_before=float(2*loss.detach()/(46*64)),gradient_l2=float(gradient.double().norm()),
                control_rms=float(control.double().square().mean().sqrt()),control_l2=float(control.double().norm()),
                delta_velocity_rms=float(effective.double().square().mean().sqrt()),
                effective_clean_control_rms=float((sigma*effective).double().square().mean().sqrt()),
                clean_update_error_rms=float((clean_after-clean-control).double().square().mean().sqrt()))
            velocity=controlled_velocity
            del leaf,loss,gradient,control,effective,clean,clean_after,controlled_velocity
        elif message is not None and index in method.CONTROL_INDICES:
            row['skip_reason']='NONPOSITIVE_SIGMA'
        count('scheduler_step',False)
        previous=latent
        latent=scheduler.step(velocity,timestep,latent,return_dict=False)[0]
        row['actual_step_delta_rms']=float((latent-previous).double().square().mean().sqrt())
        del previous
        count('scheduler_step',True)
        if scheduler.step_index!=index+1:raise RuntimeError('scheduler history cursor diverged')
        if not bool(torch.isfinite(latent).all()):raise FloatingPointError('nonfinite generation latent')
        record(row)
    return latent.detach()
