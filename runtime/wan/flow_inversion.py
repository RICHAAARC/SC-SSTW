"""Known-conditioning flow Euler approximation; never a UniPC discrete inverse."""
import torch


def public_schedule(scheduler):
    from diffusers import UniPCMultistepScheduler
    if not isinstance(scheduler,UniPCMultistepScheduler) or scheduler.config.prediction_type!='flow_prediction' or not scheduler.predict_x0 or scheduler.config.thresholding or scheduler.solver_p is not None:
        raise ValueError('native flow_prediction UniPC predict_x0 without thresholding/solver_p required')
    value={'sigmas':scheduler.sigmas.detach().cpu().tolist(),'timesteps':scheduler.timesteps.detach().cpu().tolist(),
        'config':dict(scheduler.config),'class':type(scheduler).__name__}
    reverse_grid(value['sigmas'],value['timesteps'])
    return value


def reverse_grid(sigmas,timesteps):
    sigma=torch.as_tensor(sigmas,dtype=torch.float64);time=torch.as_tensor(timesteps)
    if sigma.shape!=(51,) or time.shape!=(50,) or not torch.isfinite(sigma).all() or not torch.isfinite(time).all():
        raise ValueError('finite 51 sigma/50 timestep grid required')
    if sigma[-1]!=0 or not bool((sigma[:-1]>sigma[1:]).all()) or not bool((time>=0).all()):
        raise ValueError('strictly descending positive sigmas ending at zero required')
    # The original timesteps may be integer-rounded; do not reconstruct sigma*T.
    return sigma.flip(0),torch.cat((time.new_zeros(1),time.flip(0)[:-1]))


@torch.no_grad()
def invert_received_latent(transformer,received_latent,prompt,negative,input_dtype,guidance_scale,
                           sigmas,timesteps,count,record):
    grid,times=reverse_grid(sigmas,timesteps)
    value=received_latent.detach().float()
    if not bool(torch.isfinite(value).all()):raise FloatingPointError('nonfinite received latent')
    for index in range(50):
        hidden=value.to(input_dtype)
        timestep=times[index].to(device=value.device).expand(value.shape[0])
        count('transformer',False)
        conditional=transformer(hidden_states=hidden,timestep=timestep,encoder_hidden_states=prompt,attention_kwargs=None,return_dict=False)[0]
        count('transformer',True);count('transformer',False)
        unconditional=transformer(hidden_states=hidden,timestep=timestep,encoder_hidden_states=negative,attention_kwargs=None,return_dict=False)[0]
        count('transformer',True)
        velocity=(unconditional+guidance_scale*(conditional-unconditional)).float()
        if not bool(torch.isfinite(velocity).all()):raise FloatingPointError('nonfinite inverse CFG velocity')
        delta=float(grid[index+1]-grid[index])
        count('inverse_update',False)
        value=value+delta*velocity
        if not bool(torch.isfinite(value).all()):raise FloatingPointError('nonfinite inverse update')
        count('inverse_update',True)
        record({'index':index,'sigma_left':float(grid[index]),'sigma_right':float(grid[index+1]),
            'timestep_left':float(times[index]),'delta_sigma':delta,
            'update_rms':float((delta*velocity).double().square().mean().sqrt())})
    return value.detach()
