"""Fixed one-sided writer objective; receiver remains temporal-difference hard votes."""
import torch
from . import grow_temporal_difference as carrier
MARGIN=.5

def loss(z,book,message):
    d=carrier.selected(z,book)
    bits=d.new_tensor(book['payloads'][message])[torch.tensor(book['bit_index'],device=d.device)].reshape(1,1,1,64)
    return .5*torch.relu(MARGIN-bits*d).square().sum()
