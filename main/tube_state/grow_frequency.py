"""Whole-slice spatial DCT carrier for an explicitly defined video adaptation."""
from __future__ import annotations
import hashlib
import math
import torch

SHAPE=(1,16,46,40,64)
BITS=16
AMPLITUDE=.5
ETA=.1
CONTROL_INDICES=tuple(range(10,30))


def basis(n,like):
    k=torch.arange(n,device=like.device,dtype=like.dtype)[:,None]
    j=torch.arange(n,device=like.device,dtype=like.dtype)[None,:]
    q=torch.cos(math.pi*(j+.5)*k/n)*math.sqrt(2/n)
    q[0]*=1/math.sqrt(2)
    return q


def dct2(x):
    h,w=basis(x.shape[-2],x),basis(x.shape[-1],x)
    return h@x@w.T


def idct2(x):
    h,w=basis(x.shape[-2],x),basis(x.shape[-1],x)
    return h.T@x@w


def codebook(key):
    def digest(role):return hashlib.sha256(key+b'/grow-video-v1/'+role.encode()).digest()
    coords=sorted(((u,v) for u in range(2,10) for v in range(2,10)),key=lambda uv:digest(f'frequency/{uv[0]}/{uv[1]}'))
    payload=[1 if digest('payload')[i//8] & (1<<(i%8)) else -1 for i in range(BITS)]
    return {'schema':'grow_video_frequency_v1','shape':list(SHAPE),'bits':BITS,
        'coordinates':[list(p) for p in coords],'bit_index':[i%BITS for i in range(64)],'channel':0,
        'payloads':[payload,[-v for v in payload]],'amplitude':AMPLITUDE,
        'normalization':'orthonormal DCT-II; no empirical spectral normalization',
        'frequency_mask':'2<=u<=9 and 2<=v<=9, zero-based spatial frequencies'}


def selected(z,book):
    uv=torch.tensor(book['coordinates'],device=z.device)
    return dct2(z[:,0:1])[...,uv[:,0],uv[:,1]]


def target(book,message,like):
    payload=like.new_tensor(book['payloads'][message])[torch.tensor(book['bit_index'],device=like.device)]
    return (book['amplitude']*payload).reshape(1,1,1,64)


def loss(z,book,message):
    return .5*(selected(z,book)-target(book,message,z)).square().sum()


def analytic_gradient(z,book,message):
    coefficients=selected(z,book)-target(book,message,z)
    uv=torch.tensor(book['coordinates'],device=z.device)
    masked=torch.zeros_like(z[:,0:1]);masked[...,uv[:,0],uv[:,1]]=coefficients
    gradient=torch.zeros_like(z);gradient[:,0:1]=idct2(masked)
    return gradient


def read(z,book):
    if tuple(z.shape)!=SHAPE or not bool(torch.isfinite(z).all()):raise ValueError('fixed finite full latent required')
    with torch.no_grad():
        coefficients=selected(z.double(),book).reshape(46,4,BITS)
        votes=torch.sign(coefficients)
        def summary(sums,soft,denominator,zero_votes):
            signs=torch.sign(sums)
            return {'vote_sums':sums.int().tolist(),'signs':signs.int().tolist(),
                'bit_erasures':int((signs==0).sum()),'zero_coefficient_votes':int(zero_votes),
                'votes_per_bit':denominator,'coefficient_mean_diagnostic':soft.tolist()}
        return {'status':'COMPLETE','latent_time_denominator':46,'bits':BITS,
            'per_time':[dict(time=t,**summary(votes[t].sum(0),coefficients[t].mean(0),4,(votes[t]==0).sum())) for t in range(46)],
            'aggregate':summary(votes.sum((0,1)),coefficients.mean((0,1)),184,(votes==0).sum()),
            'claim':'coefficient sign voting, zero/tie erasures; no candidate ranking or calibrated detection threshold'}


def compare_payloads(readout,book):
    def compare(row):
        signs=row['signs']
        return [{'message':m,'bit_errors_including_erasures':sum(a!=b for a,b in zip(signs,p)),
                 'ber_including_erasures':sum(a!=b for a,b in zip(signs,p))/BITS,
                 'exact_payload_match':signs==p} for m,p in enumerate(book['payloads'])]
    return {'aggregate':compare(readout['aggregate']),'per_time':[compare(v) for v in readout['per_time']]}
