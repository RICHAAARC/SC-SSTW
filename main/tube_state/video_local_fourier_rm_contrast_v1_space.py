"""Public masked ABS/DIFF union subspace; no observations or truth enter its recipe."""
from __future__ import annotations
import hashlib,json
from copy import deepcopy
from functools import lru_cache
import numpy as np
from main.tube_state import video_local_fourier_rm_state as state
VERSION='video-local-fourier-rm-contrast-v1'
SHAPE=(44,4,4,8)
RANK_TOLERANCE='eps64 * max(generator.shape) * largest_singular_value'


def laplacian(value):
    """D transpose D, with 43 adjacent edges on the 44 received positions."""
    out=np.zeros_like(value);d=np.diff(value,axis=0);out[:-1]-=d;out[1:]+=d;return out


def templates(key):
    rows=state.catalog(44);ids=[i for i,row in enumerate(rows) if row['structurally_valid']]
    paths=np.asarray([rows[i]['taus'] for i in ids])-1
    return state.composite_signs(key)[paths].astype(np.float64)*state.PUBLIC.alpha,ids


@lru_cache(maxsize=4)
def _build(key):
    means,ids=templates(key);mask=state.composite_signs(key)[:44]!=0;diff=means-means[0]
    absolute=diff*mask;pulled=np.asarray([laplacian(v) for v in diff])*mask
    generator=np.concatenate([absolute.reshape(174,-1),pulled.reshape(174,-1)],axis=0).T
    u,s,_=np.linalg.svd(generator,full_matrices=False);tol=np.finfo(np.float64).eps*max(generator.shape)*float(s[0]);rank=int(np.sum(s>tol));basis=u[:,:rank].copy()
    basis[~mask.reshape(-1)]=0.
    for j in range(rank):
        if basis[np.argmax(np.abs(basis[:,j])),j]<0:basis[:,j]*=-1
    recipe=dict(method_version=VERSION,key_sha256=hashlib.sha256(key.encode()).hexdigest(),R=44,shape=list(SHAPE),alpha=state.PUBLIC.alpha,valid_catalog_indices=ids,anchor_catalog_index=ids[0],anchor_rule='first public structurally valid catalog row, independent of measurements',mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),generator_sha256=hashlib.sha256(generator.tobytes()).hexdigest(),construction='columns M(mu_i-mu_anchor), M D^T D(mu_i-mu_anchor), all 174 valid paths',rank_tolerance=RANK_TOLERANCE,sign_rule='largest-absolute-entry nonnegative; basis orientation is not method semantics')
    receipt=dict(recipe=recipe,recipe_sha256=hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest(),rank=rank,rank_tolerance_value=tol,singular_values=s.tolist(),smallest_retained_singular_value=float(s[rank-1]) if rank else None,largest_discarded_singular_value=float(s[rank]) if rank<len(s) else None,orthogonality_max_abs=float(np.max(np.abs(basis.T@basis-np.eye(rank)))) if rank else 0.,support_max_abs=float(np.max(np.abs(basis[~mask.reshape(-1)]))),generator_span_residual_l2=float(np.linalg.norm(generator-basis@(basis.T@generator))),basis_sha256=hashlib.sha256(basis.tobytes()).hexdigest(),active_R44=int(mask.sum()),boundary_R44=int((~mask).sum()),basis_shape=list(basis.shape),note='FP64 linear-algebra tolerance only; no detection threshold or calibration')
    for a in (basis,mask,means):a.flags.writeable=False
    return dict(basis=basis,mask=mask,templates=means,valid_catalog_indices=ids,receipt=receipt)


def build_space(key):
    """Public key-only recipe; detached arrays prevent callers mutating cached protocol."""
    value=_build(key);return {k:(v.copy() if isinstance(v,np.ndarray) else deepcopy(v)) for k,v in value.items()}


def project(value,key):
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=SHAPE or not np.isfinite(value).all():raise ValueError('finite R44 coefficient direction required')
    f=_build(key);x=(value*f['mask']).reshape(-1);out=(f['basis']@(f['basis'].T@x)).reshape(SHAPE);out[~f['mask']]=0.;return out


def relative_effects(value,key):
    """All public relative cost changes under q -> q+value; no q or winner needed."""
    value=np.asarray(value,dtype=np.float64)
    if value.shape!=SHAPE or not np.isfinite(value).all():raise ValueError('finite R44 direction required')
    f=_build(key);dif=f['templates']-f['templates'][0]
    absolute=-2*np.einsum('ijkl,nijkl->n',value,dif)/5632
    difference=-2*np.einsum('ijkl,nijkl->n',np.diff(value,axis=0),np.diff(dif,axis=1))/5504
    return dict(ABSOLUTE_CONTROL=absolute.tolist(),ADJACENT_DIFFERENCE=difference.tolist())
