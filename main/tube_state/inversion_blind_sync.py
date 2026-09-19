"""Fixed raw-amplitude nominal-shift search; no timing or truth input."""
import torch
from . import inversion_state as state

SHAPE=(1,16,33,40,64)
SHIFTS=tuple(range(14))
LOCAL=tuple(range(1,32))
EPS=1e-12


def rankings(candidates):
    ordered=sorted(candidates,key=lambda r:(-r['score'],r['shift'],r['message']))
    best=ordered[0]['score'];ties=[dict(r) for r in ordered if best-r['score']<=EPS]
    messages={r['message'] for r in ties};shifts={r['shift'] for r in ties}
    profiles=[]
    for m in (0,1):
        rows=[r for r in ordered if r['message']==m];top=rows[0]['score']
        profiles.append(dict(message=m,max_score=top,top_shifts=[r['shift'] for r in rows if top-r['score']<=EPS]))
    offsets=[dict(shift=s,max_score=max(r['score'] for r in ordered if r['shift']==s)) for s in SHIFTS]
    offset_order=sorted(offsets,key=lambda r:(-r['max_score'],r['shift']))
    offset_ties=[r['shift'] for r in offset_order if offset_order[0]['max_score']-r['max_score']<=EPS]
    return dict(ordered=ordered,top_ties=ties,top_runner_gap=best-ordered[1]['score'],
        selected_message=next(iter(messages)) if len(messages)==1 else None,
        selected_shift=next(iter(shifts)) if len(shifts)==1 else None,
        selected_pair={'shift':ties[0]['shift'],'message':ties[0]['message']} if len(ties)==1 else None,
        message_profiles=profiles,message0_minus_message1=profiles[0]['max_score']-profiles[1]['max_score'],
        offset_profiles=offsets,offset_top_ties=offset_ties,
        different_offset_top_runner_gap=offset_order[0]['max_score']-offset_order[1]['max_score'],
        message_margin=abs(profiles[0]['max_score']-profiles[1]['max_score']),
        phase_claim='within-latent RGB start phase unresolved; no detection threshold')


@torch.no_grad()
def search(recovered,book):
    if tuple(recovered.shape)!=SHAPE or not bool(torch.isfinite(recovered).all()):raise ValueError('fixed finite recovered clip tensor required')
    if recovered.device.type!='cpu':raise ValueError('offline search requires CPU tensor')
    values=recovered[0,0].reshape(33,2560).double()[:,book['order']]
    details=[];candidates=[]
    for shift in SHIFTS:
        indices=torch.tensor(LOCAL)+shift
        q=(values[1:32]*book['pads'][indices]).reshape(31,1280,2).mean(1)
        windows=((indices-1)//4).long()
        scores=[]
        for message in (0,1):
            target=book['states'][message,windows].double()
            score=float((q*target).mean());scores.append(score)
            candidates.append(dict(shift=shift,message=message,score=score,slice_denominator=31,axis_denominator=62,status='SCORED'))
        core=[]
        for n,(a,b) in enumerate(state.WINDOWS):
            mask=(indices>=a)&(indices<b);coverage=int(mask.sum())
            mean=q[mask].mean(0).tolist() if coverage else [0.,0.]
            row=state.observation(mean,window=n,source_start=a,source_stop=b,coverage=coverage,
                local_indices=[LOCAL[i] for i in range(31) if bool(mask[i])])
            row['valid']=row['valid'] and coverage==4
            row['coverage_status']='COMPLETE' if coverage==4 else ('PARTIAL' if coverage else 'MISSING')
            core.append(row)
        auxiliary=state.rank([r['q'] for r in core],[r['valid'] for r in core],book)
        details.append(dict(shift=shift,q=q.tolist(),nominal_source_indices=indices.tolist(),local_indices=list(LOCAL),
            primary_scores=scores,core=core,auxiliary_observer=auxiliary,
            complete_windows=sum(r['coverage']==4 for r in core),partial_windows=sum(0<r['coverage']<4 for r in core)))
    for row in candidates:row['rank']=1+sum(v['score']>row['score']+EPS for v in candidates)
    return dict(status='COMPLETE',candidate_denominator=28,shift_denominator=14,scored_local_denominator=31,
        candidates=candidates,shifts=details,ranking=rankings(candidates),
        excluded_boundaries=[dict(local_index=j,raw_mean=float(values[j].mean()),raw_rms=float(values[j].square().mean().sqrt())) for j in (0,32)],
        claim='blind nominal-shift/message ranking from saved tensor only; phase unresolved and OFF is not detection')
