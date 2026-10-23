"""Normalized full181 tensor adapter for one terminal LLL construction."""
import numpy as np
import torch
from main.tube_state import b_lfg_posthoc_reference_v1 as core
from main.tube_state.b_lfg_posthoc_plan_v1 import SHAPE

def validate(z):
    if not isinstance(z,torch.Tensor) or tuple(z.shape)!=SHAPE or z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):raise ValueError('finite normalized FP32[1,16,46,40,64] required')
    return z.detach().cpu()

def construct(z,bits,key):
    source=validate(z);before=source.clone();marked,receipt=core.construct(source[0].numpy(),bits,key)
    out=torch.from_numpy(marked[None].copy());validate(out)
    unchanged=bool(torch.equal(source,before))
    if not unchanged:raise ValueError('LFG writer modified original P0 in place')
    receipt.update(source_unchanged=unchanged,channels=list(range(16)),times=list(range(46)),new_user_payload=False)
    return out,receipt

def received_lll(z):return core.lll(validate(z)[0].numpy())
