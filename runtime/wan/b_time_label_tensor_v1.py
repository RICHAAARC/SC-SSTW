"""Actual FP32 tensor mapping only. Tests use synthetic tensors; no models/I/O."""
import torch
from main.tube_state import b_detectable_time_label_v1 as core
ROIS=((8,16,12,20),(8,16,44,52),(28,36,12,20),(28,36,44,52))

def validate(z,times):
    if not isinstance(z,torch.Tensor) or tuple(z.shape)!=(1,16,times,40,64) or z.dtype!=torch.float32 or not bool(torch.isfinite(z).all()):
        raise ValueError(f'finite normalized FP32[1,16,{times},40,64] latent required')
    return z.detach().cpu()

def blocks(z,t):
    result=[]
    for y0,y1,x0,x1 in ROIS:result.extend(z[0,4:16,t,y0:y1,x0:x1].reshape(12,64).tolist())
    return result

def construct(source,key):
    source=validate(source,46);out=source.clone();rows=[]
    for j in range(1,45):
        changed,receipt=core.encode_label(blocks(source,j),j,key)
        for roi,(y0,y1,x0,x1) in enumerate(ROIS):
            out[0,4:16,j,y0:y1,x0:x1]=torch.tensor(changed[12*roi:12*roi+12],dtype=torch.float32).reshape(12,8,8)
        rows.append(receipt)
    validate(out,46)
    payload_equal=bool(torch.equal(source[:,:4],out[:,:4]))
    guards_equal=bool(torch.equal(source[:,:,0],out[:,:,0]) and torch.equal(source[:,:,45],out[:,:,45]))
    if not payload_equal or not guards_equal:raise ValueError('metadata writer modified payload channels or guards')
    delta=out.double()-source.double()
    return out,dict(status='CONSTRUCTED_TIME_METADATA',rows=rows,cell_denominator=44,chip_denominator=2112,
        ideal_energy=sum(r['ideal_energy'] for r in rows),actual_FP32_energy=float(delta.square().sum()),
        actual_Linf=float(delta.abs().max()),whole_terminal_coordinates=source.numel(),
        whole_terminal_RMS=float(delta.square().mean().sqrt()),ideal_energy_bound=176.,
        payload_channels_bitwise_equal=payload_equal,guard_layers_bitwise_equal=guards_equal,
        payload_rewritten=False,FP32_correction_attempts=0,delta=core.DELTA,alpha=1)

def project_rows(z,key,native=False):
    z=validate(z,46 if native else 11);b=core.book(key);rows=[]
    for t in range(1,45 if native else 11):
        try:rows.append(dict(projections=list(core.projections(blocks(z,t),b))))
        except Exception as error:rows.append(dict(projections=None,error=f'{type(error).__name__}: {error}'))
    return rows
