"""Two fixed spatial copies of the unchanged OLD8 keyed RM carrier."""
from __future__ import annotations
from dataclasses import replace
import numpy as np
from main.tube_state import video_local_fourier_rm_state as old
VERSION='video-local-fourier-rm-replica-v1'
GROUP_A=old.PUBLIC.blocks
GROUP_B=((8,16,20,28),(8,16,52,60),(28,36,20,28),(28,36,52,60))
GROUPS=(GROUP_A,GROUP_B)
PUBLIC=replace(old.PUBLIC,method_version=VERSION)
GROUP_ALPHA=PUBLIC.alpha/np.sqrt(2.)
state_code=old.state_code
composite_signs=old.composite_signs
bases=old.bases
catalog=old.catalog

def basis_layout(key):
    return dict(method_version=VERSION,logical_basis=old.basis_layout(key),groups=[[list(b) for b in g] for g in GROUPS],group_alpha=GROUP_ALPHA,read_scale=float(np.sqrt(2.)),fusion='equal mean of two scaled groups on public support intersection',replicas_are_independent_votes=False)

def synthesize(key,public=PUBLIC,dtype=np.float64):
    if public!=PUBLIC:raise ValueError('frozen public protocol required')
    out=np.zeros((1,16,46,40,64),dtype=dtype);q=composite_signs(key).astype(dtype)*dtype(GROUP_ALPHA);U=bases(key).astype(dtype)
    for group in GROUPS:
        for i,(h0,h1,w0,w1) in enumerate(group):out[0,4,1:,h0:h1,w0:w1]=(q[:,i].reshape(45,32)@U[i].T).reshape(45,8,8)
    return out

def fuse_groups(raw_groups,availability):
    raw=np.asarray(raw_groups);mask=np.asarray(availability)
    if raw.ndim!=5 or raw.shape[0]!=2 or raw.shape[2:]!=(4,4,8) or raw.shape[1] not in PUBLIC.observed_lengths:raise ValueError('two public R x4x4x8 projection groups required')
    R=raw.shape[1]
    if mask.dtype!=np.bool_ or mask.shape!=(2,R,4):raise ValueError('boolean two-group public availability required')
    observed=np.broadcast_to(mask[:,:,:,None,None],raw.shape)
    if not np.isfinite(raw[observed]).all():raise ValueError('nonfinite available group projection')
    with np.errstate(over='raise',invalid='raise'):
        clean=np.where(observed,raw.astype(np.float64),0.)
        scaled=clean*np.sqrt(2.)
        common=mask[0]&mask[1]
        fused=np.where(common[:,:,None,None],(scaled[0]+scaled[1])/2.,0.)
    if not np.isfinite(scaled).all() or not np.isfinite(fused).all():raise ValueError('nonfinite replica fusion')
    return dict(raw_groups=clean,scaled_groups=scaled,fused=fused,q=fused,group_availability=mask.copy(),availability=common,received_regular_indices=np.arange(1,R+1,dtype=np.int64),fusion='fixed sqrt2 per group followed by equal mean; intersection support; no single-copy fallback',replicas_are_independent_votes=False)

def extract_groups(received_tensor,key,availability,public=PUBLIC):
    if public!=PUBLIC:raise ValueError('frozen public protocol required')
    Y=np.asarray(received_tensor);mask=np.asarray(availability)
    if Y.ndim!=4 or Y.shape[1:]!=(16,40,64) or Y.shape[0] not in public.observed_lengths:raise ValueError('received whole-frame latent geometry required')
    if mask.dtype!=np.bool_ or mask.shape!=(2,len(Y),4):raise ValueError('boolean two-group received mask required')
    raw=np.zeros((2,len(Y),4,4,8),dtype=np.float64);U=bases(key)
    for g,group in enumerate(GROUPS):
        for b,(h0,h1,w0,w1) in enumerate(group):
            selected=Y[mask[g,:,b],4,h0:h1,w0:w1].reshape(-1,64)
            if not np.isfinite(selected).all():raise ValueError('nonfinite available physical ROI')
            with np.errstate(over='raise',invalid='raise'):q=selected.astype(np.float64)@U[b]
            if not np.isfinite(q).all():raise ValueError('nonfinite group projection')
            raw[g,mask[g,:,b],b]=q.reshape(-1,4,8)
    return fuse_groups(raw,mask)

def extract(received_tensor,key,availability,public=PUBLIC):
    return extract_groups(received_tensor,key,availability,public)['fused']
