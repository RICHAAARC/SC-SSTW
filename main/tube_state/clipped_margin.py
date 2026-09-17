"""Exact negative nominal clipped matched margin; no surrogate gradient."""
import torch
from .velocity_coefficients import projections


def projection_loss(p, codes, message):
    return ((codes[1-message]*p).clamp(-1,1)-(codes[message]*p).clamp(-1,1)).mean()


def terminal_loss(z, directions, codes, message):
    return projection_loss(projections(z,directions),codes,message)


def gradient_support(p, codes):
    different=codes[0]!=codes[1]
    return {'support_count':p.numel(),'different_code_supports':int(different.sum()),
        'different_code_interior':int((different & (p.abs()<1)).sum()),
        'different_code_saturated':int((different & (p.abs()>1)).sum()),
        'different_code_boundary':int((different & (p.abs()==1)).sum()),
        'boundary_rule':'native torch.clamp uses derivative 1 at +/-1; kink has no unique classical derivative',
        'claim':'projection-level support only; coefficient gradient can still cancel or vanish'}
