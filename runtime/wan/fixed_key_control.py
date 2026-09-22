"""Fixed-marker clean-leaf objective with unchanged native-response controls."""
import torch
from main.tube_state import fixed_key
from . import trajectory
from .payload_control import R_STAR, prepare_direction, controlled_step, cumulative


def clean_direction(clean, book, count=None):
    with torch.enable_grad():
        leaf = clean.detach().cpu().double().clone().requires_grad_(True)
        value = fixed_key.loss(leaf, book)
        if count: count('clean_leaf_backward', False)
        gradient, = torch.autograd.grad(value, leaf)
        if count: count('clean_leaf_backward', True)
    norm = trajectory.measures(gradient)['support_rms']
    if not torch.isfinite(gradient).all() or not 0 < norm < float('inf'):
        raise ValueError('zero/nonfinite fixed-marker gradient')
    return -gradient.detach().float(), dict(objective='fixed-marker-11-window-tanh-mean', temperature=1,
        loss=float(value.detach()), gradient_support_rms=norm,
        nominal_clean=fixed_key.nominal_record(clean, book),
        meaning='detached CPU current clean leaf; no terminal/model/VAE gradient transport')
