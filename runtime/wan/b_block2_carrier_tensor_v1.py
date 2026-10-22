"""Explicit family Book wiring. Actual tensor work; synthetic CPU validation only."""
import math
import torch
from main.tube_state import b_block2_carrier_v1 as method
from runtime.wan import b_time_label_tensor_v1 as legacy

def construct(source,family,key):
    source=legacy.validate(source,46);out=source.clone();b=method.book(family,key);rows=[]
    for j in range(1,45):
        changed,receipt=method.core.embed_packet(legacy.blocks(source,j),method.core.packet(j-1,b),b)
        for r,(y0,y1,x0,x1) in enumerate(legacy.ROIS):
            out[0,4:16,j,y0:y1,x0:x1]=torch.tensor(changed[12*r:12*r+12],dtype=torch.float32).reshape(12,8,8)
        for record in receipt['records']:
            record['actual_L2']=math.sqrt(record['actual_FP32_energy']);record['actual_RMS']=math.sqrt(record['actual_FP32_energy']/64)
        receipt['actual_L2']=math.sqrt(receipt['actual_FP32_energy']);receipt['actual_RMS']=math.sqrt(receipt['actual_FP32_energy']/(48*64))
        receipt['writer_label']=j
        receipt['ROI_totals']=[dict(ROI=r,actual_FP32_energy=math.fsum(x['actual_FP32_energy'] for x in receipt['records'][12*r:12*r+12]),actual_Linf=max(x['actual_Linf'] for x in receipt['records'][12*r:12*r+12])) for r in range(4)]
        rows.append(receipt)
    legacy.validate(out,46)
    payload=bool(torch.equal(source[:,:4],out[:,:4]));guards=bool(torch.equal(source[:,:,0],out[:,:,0]) and torch.equal(source[:,:,45],out[:,:,45]))
    delta=out.double()-source.double();mask=torch.zeros_like(source,dtype=torch.bool)
    for y0,y1,x0,x1 in legacy.ROIS:mask[0,4:16,1:45,y0:y1,x0:x1]=True
    outside=bool(torch.equal(source[~mask],out[~mask]))
    validate_protected(source,out,mask)
    energy=float(delta.square().sum())
    return out,dict(status='CONSTRUCTED_TIME_METADATA',family=family,rows=rows,cell_denominator=44,chip_denominator=2112,
        ideal_energy=sum(r['ideal_energy'] for r in rows),actual_FP32_energy=energy,actual_L2=math.sqrt(energy),
        actual_Linf=float(delta.abs().max()),whole_terminal_coordinates=source.numel(),whole_terminal_RMS=math.sqrt(energy/source.numel()),
        support_coordinates=int(mask.sum()),support_RMS=math.sqrt(energy/int(mask.sum())),ideal_energy_bound=176.,
        payload_channels_bitwise_equal=payload,guard_layers_bitwise_equal=guards,outside_support_bitwise_equal=outside,
        payload_rewritten=False,FP32_materializations_per_coordinate=1,FP32_correction_attempts=0,alpha=1,delta=method.core.DELTA)

def project_rows(z,family,key):
    z=legacy.validate(z,46);b=method.book(family,key);rows=[]
    for j in range(1,45):
        try:rows.append(dict(projections=list(method.core.projections(legacy.blocks(z,j),b))))
        except Exception as error:rows.append(dict(error=f'{type(error).__name__}: {error}',projections=None))
    return rows


def validate_protected(source,out,mask):
    if not (torch.equal(source[:,:4],out[:,:4]) and torch.equal(source[:,:,0],out[:,:,0]) and torch.equal(source[:,:,45],out[:,:,45]) and torch.equal(source[~mask],out[~mask])):
        raise ValueError('metadata writer changed protected coordinates')
