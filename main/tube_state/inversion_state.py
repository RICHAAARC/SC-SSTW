"""Keyed initial-noise state sequence and truth-free local observer readout."""
import numpy as np
import torch
from . import initial_noise,state_clock

SHAPE=initial_noise.SHAPE
WINDOWS=tuple((t,t+4) for t in range(1,45,4))
EPS=1e-12
check=initial_noise.check


def codebook(key):
    book=initial_noise.codebook(key)
    trajectories=[state_clock.trajectory(key,m) for m in (0,1)]
    book.update(states=torch.tensor(np.stack([t[0] for t in trajectories])),
        steps=torch.tensor([t[1] for t in trajectories]),drives=torch.tensor([t[2] for t in trajectories]))
    return book


def write(base,book,message):
    check(base)
    if message not in (0,1):raise ValueError('fixed candidate message 0/1 required')
    out=base.detach().clone();flat=out[0,0].reshape(46,2560)
    order=book['order'].to(base.device);pads=book['pads'].to(device=base.device,dtype=base.dtype)
    states=book['states'][message].to(device=base.device,dtype=base.dtype)
    axes=torch.arange(2560,device=base.device)%2
    for n,(a,b) in enumerate(WINDOWS):
        flat[a:b,order]=flat[a:b,order].abs()*pads[a:b]*states[n,axes]
    return out


def observation(q,**extra):
    q=np.asarray(q,dtype=np.float64)
    norm=float(np.linalg.norm(q));valid=bool(np.isfinite(q).all() and norm>EPS)
    return dict(extra,q=q.tolist(),norm=norm,signs=np.sign(q).astype(int).tolist(),
        zeros=int(np.count_nonzero(q==0)),valid=valid,status='COMPLETE')


def rank(qs,valid,book):
    qs=np.asarray(qs,dtype=np.float64)
    if qs.shape!=(11,2) or len(valid)!=11 or not np.isfinite(qs).all():raise ValueError('11 finite two-axis observations required')
    valid=[bool(v and np.linalg.norm(q)>EPS) for v,q in zip(valid,qs)]
    result={}
    for mode,update in [('without_update',False),('with_update',True)]:
        candidates=[]
        for m in (0,1):
            obs=state_clock.observe(qs,valid,np.asarray(book['states'][m]),np.asarray(book['steps'][m]),update=update)
            candidates.append(dict(message=m,score=-obs['innovation_mean'],observer=obs))
        delta=candidates[0]['score']-candidates[1]['score']
        unique=any(valid) and abs(delta)>EPS
        result[mode]=dict(candidates=candidates,valid_windows=sum(valid),message_unique=unique,
            selected_message=(0 if delta>0 else 1) if unique else None,
            status='NO_VALID_OBSERVATIONS' if not any(valid) else ('UNIQUE' if unique else 'TIE'))
    return result


@torch.no_grad()
def read(recovered,book):
    check(recovered)
    order=book['order'].to(recovered.device);pads=book['pads'].to(recovered.device)
    demodulated=recovered[0,0].reshape(46,2560)[:,order].double()*pads
    q=demodulated.reshape(46,1280,2).mean(1).cpu().numpy()
    slices=[observation(q[t],time=t,coordinates_per_axis=1280) for t in range(46)]
    core=[observation(q[a:b].mean(0),window=n,start=a,stop=b,coordinates_per_axis=5120) for n,(a,b) in enumerate(WINDOWS)]
    return dict(status='COMPLETE',latent_time_denominator=46,core_window_denominator=11,
        per_time=slices,core=core,boundaries=[slices[0],slices[45]],
        rankings=rank([r['q'] for r in core],[r['valid'] for r in core],book),
        coordinate_assumption='known original latent-time index; no crop synchronization')


def report(decoded,book,truth):
    """Reporting only; raw signs and candidate rankings already fixed."""
    if truth is None:return dict(truth=None,modes=None,raw_state=None,observer_margin_change=None)
    target=np.asarray(book['states'][truth]);actual=np.asarray([r['signs'] for r in decoded['core']])
    valid=np.asarray([r['valid'] for r in decoded['core']],dtype=bool)
    matches=(target==actual)&valid[:,None]
    modes={}
    for mode,row in decoded['rankings'].items():
        candidates=row['candidates']
        modes[mode]=dict(unique_correct=row['message_unique'] and row['selected_message']==truth,
            true_wrong_margin=candidates[truth]['score']-candidates[1-truth]['score'],selected_message=row['selected_message'])
    return dict(truth=truth,raw_state=dict(window_matches=matches.all(1).tolist(),exact_windows=int(matches.all(1).sum()),
        window_denominator=11,component_matches=matches.tolist(),component_errors=int((~matches).sum()),component_denominator=22,
        exact_trajectory=bool(matches.all()),erasures=sum(r['zeros'] for r in decoded['core']),
        invalid_windows=int((~valid).sum()),invalid_components=2*int((~valid).sum())),modes=modes,
        observer_margin_change=modes['with_update']['true_wrong_margin']-modes['without_update']['true_wrong_margin'],
        observer_decision_changed=modes['with_update']['selected_message']!=modes['without_update']['selected_message'])
