"""Keyed sign/pad/repetition initial-noise mechanism; no security claim."""
import hashlib
import torch

SHAPE=(1,16,46,40,64)
BITS=16


def codebook(key):
    def digest(role):return hashlib.sha256(key+b'/video-inversion-v1/'+role.encode()).digest()
    order=sorted(range(2560),key=lambda j:digest(f'coordinate/{j}'))
    pads=[[1 if digest(f'pad/{t}/{j}')[0]&1 else -1 for j in order] for t in range(46)]
    return {'order':torch.tensor(order),'pads':torch.tensor(pads,dtype=torch.int8)}


def check(z):
    if tuple(z.shape)!=SHAPE or not bool(torch.isfinite(z).all()):raise ValueError('fixed finite full latent required')


def write(base,book,payload):
    check(base)
    if len(payload)!=BITS or any(b not in (-1,1) for b in payload):raise ValueError('fixed 16-bit sign payload required')
    out=base.detach().clone();flat=out[0,0].reshape(46,2560)
    order=book['order'].to(base.device);pads=book['pads'].to(device=base.device,dtype=base.dtype)
    bits=base.new_tensor(payload).repeat(160)
    flat[:,order]=flat[:,order].abs()*pads*bits
    return out


def read(recovered,book):
    check(recovered)
    with torch.no_grad():
        order=book['order'].to(recovered.device)
        values=recovered[0,0].reshape(46,2560)[:,order]
        votes=torch.sign(values)*book['pads'].to(recovered.device)
        votes=votes.reshape(46,160,BITS)
        def summary(sums,total,zero):
            signs=torch.sign(sums)
            return {'signs':signs.int().tolist(),'vote_sums':sums.int().tolist(),
                'votes_per_bit':total,'bit_erasures':int((signs==0).sum()),'zero_coordinate_votes':int(zero)}
        return {'status':'COMPLETE','bits':16,'latent_time_denominator':46,
            'per_time':[dict(time=t,**summary(votes[t].sum(0),160,(votes[t]==0).sum())) for t in range(46)],
            'aggregate':summary(votes.sum((0,1)),7360,(votes==0).sum())}


def compare_payloads(decoded,payloads):
    def compare(row):
        return [{'message':m,'bit_errors_including_erasures':sum(a!=b for a,b in zip(row['signs'],p)),
            'ber_including_erasures':sum(a!=b for a,b in zip(row['signs'],p))/BITS,
            'exact_payload_match':row['signs']==p} for m,p in enumerate(payloads)]
    return {'aggregate':compare(decoded['aggregate']),'per_time':[compare(row) for row in decoded['per_time']]}
