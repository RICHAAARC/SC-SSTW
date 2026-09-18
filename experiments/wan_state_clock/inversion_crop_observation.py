"""Start-free raw observations and explicitly separate oracle correspondence."""
import torch
from main.tube_state import inversion_state as state

SHAPE=(1,16,33,40,64)
MAPS={0:(0,),16:(4,),17:(4,5)}


def check(z):
    if tuple(z.shape)!=SHAPE or not bool(torch.isfinite(z).all()):raise ValueError('finite actual 129-frame clip latent required; no padding')


def raw_observations(z):
    """No key, pad, start, message or source evidence used here."""
    check(z);flat=z[0,0].reshape(33,2560).double()
    return dict(status='COMPLETE',local_time_denominator=33,
        per_time=[dict(local_time=j,raw_mean=float(flat[j].mean()),raw_rms=float(flat[j].square().mean().sqrt()),
            local_boundary='causal_first' if j==0 else ('last_group' if j==32 else None)) for j in range(33)],
        claim='raw recovered channel statistics, not time-demodulated state observations')


def oracle(z,book,start,truth):
    """True start/pad mapping used only after receiver completion; never blind."""
    check(z)
    if start not in MAPS:raise ValueError('fixed crop positions only')
    values=z[0,0].reshape(33,2560).double()[:,book['order']]
    outputs=[]
    for shift in MAPS[start]:
        rows=[]
        for j in range(33):
            source=j+shift
            q=(values[j]*book['pads'][source]).reshape(1280,2).mean(0).tolist()
            frame_range=[start,start] if j==0 else [start+4*j-3,start+4*j]
            rows.append(state.observation(q,local_time=j,nominal_source_index=source,
                source_frame_range=frame_range,phase_ambiguous=start==17 and j>0,
                core_eligible=j>0,local_boundary='causal_first' if j==0 else ('last_group' if j==32 else None)))
        core=[]
        for n,(a,b) in enumerate(state.WINDOWS):
            selected=[r for r in rows if r['core_eligible'] and a<=r['nominal_source_index']<b]
            complete=len(selected)==4
            q=torch.tensor([r['q'] for r in selected],dtype=torch.float64).mean(0).tolist() if selected else [0.,0.]
            row=state.observation(q,window=n,start=a,stop=b,coverage=len(selected),expected_slices=4,
                local_indices=[r['local_time'] for r in selected],coverage_status='COMPLETE' if complete else ('PARTIAL' if selected else 'MISSING'))
            row['valid']=row['valid'] and complete
            core.append(row)
        decoded=dict(core=core,rankings=state.rank([r['q'] for r in core],[r['valid'] for r in core],book))
        report=state.report(decoded,book,truth)
        full=[n for n,r in enumerate(core) if r['coverage_status']=='COMPLETE']
        complete_report=dict(complete_window_denominator=len(full),complete_component_denominator=2*len(full),
            invalid_complete_windows=sum(not core[n]['valid'] for n in full),
            erasures_complete_components=sum(core[n]['zeros'] for n in full),
            interpretation='only predeclared geometrically complete windows; OFF has no truth')
        if truth is not None:
            raw=report['raw_state']
            complete_report.update(exact_windows=sum(raw['window_matches'][n] for n in full),
                component_errors=sum(not v for n in full for v in raw['component_matches'][n]),
                all_observed_complete_windows_correct=bool(full) and all(raw['window_matches'][n] for n in full))
        outputs.append(dict(status='MEASURED',oracle=True,source_start_reporting_only=start,
            nominal_shift=shift,mapping='exact_grid_nominal' if start!=17 else ('floor_nominal' if shift==4 else 'ceil_nominal'),
            mapping_is_exact_latent_correspondence=False,per_time=rows,core=core,rankings=decoded['rankings'],reporting_only=report,
            reporting_scope='full-grid raw errors include unobservable partial/missing windows; use complete_window_report separately',
            complete_window_report=complete_report,
            complete_windows=sum(r['coverage_status']=='COMPLETE' for r in core),
            partial_windows=sum(r['coverage_status']=='PARTIAL' for r in core),missing_windows=sum(r['coverage_status']=='MISSING' for r in core)))
    return outputs
