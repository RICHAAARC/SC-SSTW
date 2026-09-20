"""Differentiable fixed tube hinge objective; detached terminal leaf only."""
import torch
from . import projection_margin as carrier

def loss(z,book,message):
    if tuple(z.shape)!=carrier.SHAPE or message not in (0,1):raise ValueError('fixed tube shape/message required')
    x=z[0,:,1:45].permute(1,0,2,3).reshape(11,4,16,10,4,16,4).permute(0,3,5,1,2,4,6).reshape(1760,1024)
    d=torch.as_tensor(book['directions'],device=z.device,dtype=z.dtype)
    c=torch.as_tensor(book['codes'][message],device=z.device,dtype=z.dtype)
    projection=(x*d).sum(1)
    return .5*torch.relu(1-c*projection).square().mean()

def terminal_gradient(terminal,book,message):
    """Explicit graph break: no solver, Transformer, VAE or reader backpropagation."""
    with torch.enable_grad():
        leaf=terminal.detach().to(dtype=torch.float64).clone().requires_grad_(True)
        value=loss(leaf,book,message);gradient,=torch.autograd.grad(value,leaf)
    if not torch.isfinite(value) or not torch.isfinite(gradient).all():raise FloatingPointError('nonfinite terminal objective/gradient')
    return gradient.detach().float(),dict(loss=float(value.detach()),terminal_leaf_dtype='torch.float64',gradient_only_at_detached_terminal=True)

def value(z,book,message):
    with torch.no_grad():return float(loss(z.detach().double(),book,message))
