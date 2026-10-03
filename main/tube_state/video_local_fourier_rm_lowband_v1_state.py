"""One fixed 16x16 lower-bandwidth local RM carrier; temporal code unchanged."""
from __future__ import annotations
from dataclasses import replace
from functools import lru_cache
import numpy as np
from main.tube_state import video_local_fourier_rm_state as old
PUBLIC=replace(old.PUBLIC,method_version='video-local-fourier-rm-lowband-v1',blocks=((4,20,8,24),(4,20,40,56),(24,40,8,24),(24,40,40,56)))
keyed=old.keyed
state_code=old.state_code
composite_signs=old.composite_signs
catalog=old.catalog
family_receipt=old.family_receipt


def frequency_squared(mode):
    h,w=mode;return min(h,16-h)**2+min(w,16-w)**2

def fourier_modes():
    """32 lowest signed physical frequency pairs; conjugate representative lex tie."""
    representatives=[(h,w) for h in range(16) for w in range(16) if (h,w)<=((-h)%16,(-w)%16) and (h,w)!=(0,0)]
    return sorted(representatives,key=lambda hw:(frequency_squared(hw),hw[0],hw[1]))[:32]

def basis_layout(key):
    modes=fourier_modes();rows=[]
    for block in range(4):
        order=sorted(range(32),key=lambda j:(keyed('VLFRM1/basis/order',key,block,j),j))
        signs=[2*(keyed('VLFRM1/basis/sign',key,block,q)[0]&1)-1 for q in range(32)]
        rows.append(dict(block=block,modes=[modes[j] for j in order],signs=signs,factors=[float(np.sqrt(2))]*32))
    return dict(method_version=PUBLIC.method_version,blocks=rows,public_modes=fourier_modes(),
        frequency_rule='min(h,16-h)^2+min(w,16-w)^2, then original(h,w) lex; physical squared frequency is /256',
        transform='16x16 orthonormal real cosine; non-DC conjugate representative32 public lowest-frequency columns',
        allocation='unchanged slot=age*8+chip_in_block; no independent-vote interpretation')

@lru_cache(maxsize=8)
def _bases(key):
    y,x=np.meshgrid(np.arange(16),np.arange(16),indexing='ij');rows=[]
    for row in basis_layout(key)['blocks']:
        columns=[sign*factor*np.cos(2*np.pi*(h*y+w*x)/16)/16 for (h,w),sign,factor in zip(row['modes'],row['signs'],row['factors'])]
        rows.append(np.stack(columns,axis=-1).reshape(256,32))
    result=np.asarray(rows);result.flags.writeable=False;return result

def bases(key):return _bases(key).copy()


def extract(received_tensor,key,availability,public=PUBLIC):
    if public!=PUBLIC:raise ValueError('frozen lowband public protocol required')
    Y=np.asarray(received_tensor);mask=np.asarray(availability)
    if Y.ndim!=4 or Y.shape[1:]!=(16,40,64) or Y.shape[0] not in PUBLIC.observed_lengths:raise ValueError('frozen received geometry required')
    if mask.dtype!=np.bool_ or mask.shape!=(len(Y),4):raise ValueError('boolean received-coordinate mask required')
    values=np.zeros((len(Y),4,4,8),dtype=np.float64);U=bases(key)
    for block,(h0,h1,w0,w1) in enumerate(PUBLIC.blocks):
        selected=Y[mask[:,block],4,h0:h1,w0:w1].reshape(-1,256)
        if not np.isfinite(selected).all():raise ValueError('nonfinite available physical ROI')
        with np.errstate(over='raise',invalid='raise'):projected=selected.astype(np.float64)@U[block]
        if not np.isfinite(projected).all():raise ValueError('nonfinite projection')
        values[mask[:,block],block]=projected.reshape(-1,4,8)
    return values


def synthesize(key,public=PUBLIC,dtype=np.float64):
    if public!=PUBLIC:raise ValueError('frozen lowband public protocol required')
    target=np.zeros((1,16,46,40,64),dtype=dtype);S=state_code(key);U=bases(key).astype(dtype)
    for start in range(1,46):
        for age in range(4):
            u=start+age
            if u>45:continue
            for block,(h0,h1,w0,w1) in enumerate(PUBLIC.blocks):
                contribution=U[block,:,8*age:8*age+8]@S[start-1,8*block:8*block+8].astype(dtype)
                target[0,4,u,h0:h1,w0:w1]+=dtype(PUBLIC.alpha)*contribution.reshape(16,16)
    return target
