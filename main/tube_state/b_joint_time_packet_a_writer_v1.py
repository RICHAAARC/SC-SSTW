"""Joint packet A writer targets; old geometry, nearest rule and budgets unchanged."""
import numpy as np
import torch
from . import b_framewise_qim_v1 as legacy
from . import b_joint_time_packet_a_v1 as packet
from .b_minimal_l_v1 import replica_major_q0


def schedule(frames=181, start=0):
    if type(frames) is not int or type(start) is not int or frames <= 0 or start < 0 or start+frames > packet.N:
        raise ValueError('source interval outside the public domain')
    fields = packet.state_fields(np.arange(start, start+frames, dtype=np.int64))
    active = torch.from_numpy(np.tile(fields['active'], (1, 2)))
    js = torch.from_numpy(np.tile(fields['j'], (1, 2)))
    ages = torch.from_numpy(np.tile(fields['age'], (1, 2)))
    messages = torch.where(active, 8*js+ages, -1)
    # -1 is an unwritten slot, never a zero-code filler.
    bits = torch.full((frames, 10, 63), -1., dtype=torch.float64)
    for frame, tile in active.nonzero().tolist():
        bits[frame, tile] = torch.tensor(packet.encode_raw24(int(messages[frame, tile])), dtype=torch.float64)
    return dict(j=js, age=ages, active=active, messages=messages, bits=bits)


def validate_raw_q0(raw):
    shape = (181, 5, 2, 63)
    for key in ('q', 'd0', 'd1', 'soft'):
        value = np.asarray(raw[key])
        if value.shape != shape or value.dtype != np.float64 or not np.isfinite(value).all():
            raise ValueError('BASE/preclip '+key+' must be finite FP64 [181,5,2,63]')
    mask = np.asarray(raw['mask'])
    if mask.shape != shape or mask.dtype != np.bool_ or not mask.all():
        raise ValueError('complete native BASE/preclip q0 requires all finite physical slots')
    q = torch.from_numpy(raw['q'])
    # These are the original 32/16 class distances in 255*RGB projection units.
    # A normalized/Y/codec observation is not substituted for this input role.
    d0, d1, soft = legacy.distances(q)
    for key, expected in (('d0', d0), ('d1', d1), ('soft', soft)):
        if not torch.allclose(torch.from_numpy(raw[key]), expected, rtol=1e-12, atol=1e-10):
            raise ValueError('q0 '+key+' inconsistent with original RGB QIM units')
    return torch.from_numpy(replica_major_q0(raw['q']))


def targets(q0, fields):
    if q0.device.type != 'cpu' or q0.dtype != torch.float64 or q0.shape != fields['bits'].shape or not bool(torch.isfinite(q0).all()):
        raise ValueError('finite CPU FP64 replica-major q0 required')
    active = fields['active']
    result = q0.clone()
    # Exactly one nearest construction. Inactive slots remain q0 with no constraint.
    result[active] = legacy.nearest(q0[active], fields['bits'][active])
    return result


def validate_targets(q0, target, fields):
    if target.dtype != torch.float64 or target.shape != q0.shape or not bool(torch.isfinite(target).all()):
        raise ValueError('finite FP64 target required')
    active = fields['active']
    values, bits = target[active], fields['bits'][active]
    cells = (values-16*bits)/32
    delta = q0[active]-values
    if not bool((cells == cells.floor()).all() and ((delta > -16) & (delta <= 16)).all()):
        raise ValueError('active target violates original class/nearest/lower-tie rule')
    if not torch.equal(target[~active], q0[~active]):
        raise ValueError('startup unwritten slots must have no target displacement')
