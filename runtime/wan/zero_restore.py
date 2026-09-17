"""Restore a trusted saved Wan/UniPC prefix without replaying its prefix."""
from __future__ import annotations
import copy
import json
import torch
from .velocity_direction import require_affine_flow


def public_config(scheduler):
    return {key:value for key,value in dict(scheduler.config).items() if not key.startswith('_')}


def validate_snapshot(payload,record,shape):
    if not isinstance(payload,dict) or set(payload)!= {'latent','scheduler'}:
        raise ValueError('saved prefix requires latent and complete scheduler object; no automatic prefix regeneration')
    latent,scheduler=payload['latent'],payload['scheduler']
    require_affine_flow(scheduler)
    if tuple(latent.shape)!=tuple(shape) or latent.dtype!=torch.float32 or not torch.isfinite(latent).all():
        raise ValueError('saved normalized latent shape/dtype/values invalid')
    if scheduler.step_index!=44 or scheduler.num_inference_steps!=50 or len(scheduler.timesteps)!=50:
        raise ValueError('saved prefix must be immediately before step44 of the original50')
    if record['class']!=type(scheduler).__name__ or json.dumps(record['config'],sort_keys=True)!=json.dumps(dict(scheduler.config),sort_keys=True):
        raise ValueError('saved scheduler config differs from original run record')
    if scheduler.sigmas.tolist()!=record['sigmas']:raise ValueError('saved sigmas differ from original record')
    order=scheduler.config.solver_order
    if len(scheduler.model_outputs)!=order or len(scheduler.timestep_list)!=order:
        raise ValueError('incomplete multistep history')
    for value in [scheduler.last_sample,*scheduler.model_outputs]:
        if not torch.is_tensor(value) or tuple(value.shape)!=tuple(shape) or value.dtype!=torch.float32 or not torch.isfinite(value).all():
            raise ValueError('missing or invalid live-history sample/model outputs; cannot replace with zeros')
    if any(not torch.is_tensor(t) or t.numel()!=1 for t in scheduler.timestep_list):
        raise ValueError('saved timestep history incomplete')
    if scheduler.begin_index!=0 or any(not torch.equal(t.cpu().reshape(()),scheduler.timesteps[44-order+i].cpu().reshape(())) for i,t in enumerate(scheduler.timestep_list)):
        raise ValueError('saved begin index or timestep history differs from completed prefix')
    if scheduler.lower_order_nums!=order or not 1<=scheduler.this_order<=order:
        raise ValueError('saved solver order state incomplete')
    return latent,scheduler


def _move(value,device):
    if torch.is_tensor(value):return value.detach().to(device=device,copy=True)
    if isinstance(value,list):return [_move(item,device) for item in value]
    if isinstance(value,tuple):return tuple(_move(item,device) for item in value)
    if isinstance(value,dict):return type(value)({key:_move(item,device) for key,item in value.items()})
    return copy.deepcopy(value)


def _same_values(before,after):
    if torch.is_tensor(before):
        if before.dtype!=after.dtype:raise ValueError('restoration changed tensor dtype')
        torch.testing.assert_close(before.cpu(),after.cpu(),rtol=0,atol=0,equal_nan=True)
    elif isinstance(before,(list,tuple)):
        if len(before)!=len(after):raise ValueError('restoration changed history length')
        for a,b in zip(before,after):_same_values(a,b)
    elif isinstance(before,dict):
        if before.keys()!=after.keys():raise ValueError('restoration lost fields')
        for key in before:_same_values(before[key],after[key])
    elif before!=after:raise ValueError('restoration changed a scalar field')


def restore_snapshot(payload,fresh_scheduler,device,record,shape):
    """All history values/order/cursors preserved; device placement is field-aware.

    Native UniPC keeps sigmas and training schedule tables on CPU, timesteps on
    the requested scheduler device, and history samples on the model device.
    map_location='cpu' discards storage device labels, so explicitly rebuild only
    these placements from the same-version fresh scheduler, never reset history.
    """
    latent,saved=validate_snapshot(payload,record,shape)
    if type(saved) is not type(fresh_scheduler) or public_config(saved)!=public_config(fresh_scheduler):
        raise ValueError('loaded model scheduler differs from saved scheduler settings')
    if not torch.equal(saved.timesteps.cpu(),fresh_scheduler.timesteps.cpu()) or not torch.equal(saved.sigmas.cpu(),fresh_scheduler.sigmas.cpu()):
        raise ValueError('fresh model scheduler schedule differs from saved schedule')
    restored=copy.copy(saved);restored.__dict__={}
    layout={}
    for key,value in saved.__dict__.items():
        if key in ('model_outputs','last_sample'):target=device
        elif key in ('timesteps','timestep_list'):target=fresh_scheduler.timesteps.device
        elif torch.is_tensor(value):
            template=getattr(fresh_scheduler,key,None)
            if not torch.is_tensor(template):raise ValueError('unknown tensor placement for scheduler field '+key)
            target=template.device
        else:target='cpu'
        restored.__dict__[key]=_move(value,target)
        if torch.is_tensor(value) or key in ('model_outputs','timestep_list'):
            layout[key]=str(target)
    _same_values(saved.__dict__,restored.__dict__)
    recovered=latent.detach().to(device=device,copy=True)
    _same_values(latent,recovered)
    return recovered,restored,{'status':'RESTORED_WITH_REENCODED_CONDITIONING',
        'all_saved_scheduler_fields_preserved':True,'values_and_dtypes_after_device_transfer_equal':True,
        'step_index':restored.step_index,'this_order':restored.this_order,'lower_order_nums':restored.lower_order_nums,
        'device_layout':layout,'latent_device':str(recovered.device),
        'conditioning_limit':'prompt embeddings regenerated from recorded prompt/negative; original embeddings and resolved model revision/hash were not persisted, so not a bitwise model-condition proof'}
