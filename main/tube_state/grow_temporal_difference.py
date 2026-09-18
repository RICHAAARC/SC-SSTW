"""Fixed adjacent-time orthonormal difference carrier; no content-dependent selection."""
import math
import torch
from . import grow_frequency as spatial
SHAPE=spatial.SHAPE
BITS=16
AMPLITUDE=.5
ETA=.1
CONTROL_INDICES=tuple(range(30,50))

def codebook(key):
    book=spatial.codebook(key)
    book.update(schema='grow_temporal_difference_v1',temporal_pairs=[[2*k,2*k+1] for k in range(23)],temporal_transform='(even-odd)/sqrt(2)',votes_per_bit=92,normalization='orthonormal spatial DCT-II plus adjacent-time orthonormal difference (even-odd)/sqrt(2)')
    return book

def selected(z,book):
    c=spatial.selected(z,book)
    return (c[:,:,::2]-c[:,:,1::2])/math.sqrt(2)

def loss(z,book,message):
    return .5*(selected(z,book)-spatial.target(book,message,z)).square().sum()

def read(z,book):
    if tuple(z.shape)!=SHAPE or not bool(torch.isfinite(z).all()):raise ValueError('fixed finite full latent required')
    with torch.no_grad():
        coefficients=selected(z.double(),book).reshape(23,4,BITS);votes=coefficients.sign()
        def summary(sums,soft,n,zeros):
            signs=sums.sign()
            return dict(vote_sums=sums.int().tolist(),signs=signs.int().tolist(),bit_erasures=int((signs==0).sum()),zero_coefficient_votes=int(zeros),votes_per_bit=n,coefficient_mean_diagnostic=soft.tolist())
        return dict(status='COMPLETE',latent_time_denominator=46,pair_denominator=23,bits=BITS,
            per_pair=[dict(pair=k,times=[2*k,2*k+1],**summary(votes[k].sum(0),coefficients[k].mean(0),4,(votes[k]==0).sum())) for k in range(23)],
            aggregate=summary(votes.sum((0,1)),coefficients.mean((0,1)),92,(votes==0).sum()),
            claim='fixed adjacent-difference sign votes; no content-selected pairs or receiver OFF reference')

def compare_payloads(readout,book):
    def compare(row):
        return [dict(message=m,bit_errors_including_erasures=sum(a!=b for a,b in zip(row['signs'],p)),ber_including_erasures=sum(a!=b for a,b in zip(row['signs'],p))/BITS,exact_payload_match=row['signs']==p) for m,p in enumerate(book['payloads'])]
    return dict(aggregate=compare(readout['aggregate']),per_pair=[compare(v) for v in readout['per_pair']])
