"""Frozen preparation-only protocol; threshold fitting uses negative calibration only."""
import copy
import math
import torch
from main.tube_state import inversion_state as state,inversion_blind_sync as scorer

CAL=('cal_p0','cal_p1')
EVAL=('eval_p0','eval_p1')
ARMS=('OFF','STATE_A','STATE_B','STATIC_A','STATIC_B')
VIEWS=('full181','crop16','crop17')
CHANNELS=tuple(f'{d}/{s}' for d in ('STATE','STATIC') for s in ('search14','shift0'))


def books(key):
    dynamic=state.codebook(key);static=copy.deepcopy(dynamic)
    direction=dynamic['states'][0,0].clone()
    static['states']=torch.stack((direction.repeat(11,1),-direction.repeat(11,1)))
    static['steps']=torch.zeros_like(static['steps']);static['drives']=torch.zeros_like(static['drives'])
    return {'STATE':dynamic,'STATIC':static}


def score(recovered,public_books):
    if tuple(recovered.shape) not in ((1,16,46,40,64),(1,16,33,40,64)):raise ValueError('observed inversion geometry mismatch; no padding')
    # Full view was actually inverted at full length; only readout uses this fixed prefix.
    prefix=recovered[:,:,:33].cpu()
    result={'status':'COMPLETE','channels':{},'decoder_readouts':{}}
    for decoder,book in public_books.items():
        read=scorer.search(prefix,book);result['decoder_readouts'][decoder]=read
        for search in ('search14','shift0'):
            candidates=read['candidates'] if search=='search14' else [r for r in read['candidates'] if r['shift']==0]
            high=max(r['score'] for r in candidates);ties=[r for r in candidates if high-r['score']<=scorer.EPS]
            messages={r['message'] for r in ties}
            result['channels'][decoder+'/'+search]=dict(score=high,selected_message=next(iter(messages)) if len(messages)==1 else None,
                top_ties=ties,candidate_denominator=len(candidates))
    return result


def freeze(calibration):
    output=dict(status='FROZEN',calibration_source_denominator=2,views_per_source=3,channels={},rule='strict score > max(per-source max across views)')
    for channel in CHANNELS:
        values={};missing=[]
        for case in CAL:
            per_view=[]
            for view in VIEWS:
                row=calibration.get(case,{}).get('sources',{}).get('OFF',{}).get('views',{}).get(view,{})
                entry=row.get('readout',{}).get('channels',{}).get(channel,{})
                value=entry.get('score')
                if row.get('status')!='COMPLETE' or not isinstance(value,(int,float)) or not math.isfinite(value):missing.append(case+'/'+view)
                else:per_view.append(value)
            if len(per_view)==3:values[case]=max(per_view)
        output['channels'][channel]=dict(status='UNCALIBRATED' if missing else 'CALIBRATED',
            threshold=None if missing else max(values.values()),source_maxima=values,missing_views=missing)
    return output


def decide(readout,frozen):
    output={}
    for channel in CHANNELS:
        calibration=frozen['channels'][channel];row=readout.get('channels',{}).get(channel)
        if row is None:output[channel]={'status':'MISSING_SCORE','accepted':None};continue
        if calibration['status']!='CALIBRATED':output[channel]={'status':'UNCALIBRATED','accepted':None};continue
        output[channel]=dict(status='DECIDED',accepted=row['score']>calibration['threshold'],threshold=calibration['threshold'],
            score=row['score'],selected_message=row['selected_message'])
    return output


def report(readout,arm,view,public_books):
    """Labels/time are joined only after the blind score record has persisted."""
    if arm=='OFF':return {'truth':None,'existence_claim':'OFF decisions are evaluated separately, not forced absent'}
    decoder,message=arm.split('_');truth=0 if message=='A' else 1
    if readout.get('status')!='COMPLETE':return {'status':'MISSING_READOUT','truth':truth}
    book=public_books[decoder];result={'truth':truth,'matched_decoder':decoder,'nominal_diagnostics':[]}
    for shift in ((0,) if view=='full181' else ((4,) if view=='crop16' else (4,5))):
        detail=readout['decoder_readouts'][decoder]['shifts'][shift]
        complete=[(n,r) for n,r in enumerate(detail['core']) if r['coverage']==4]
        matches=[[bool(r['valid'] and sign==int(book['states'][truth,n,a])) for a,sign in enumerate(r['signs'])] for n,r in complete]
        result['nominal_diagnostics'].append(dict(shift=shift,complete_window_denominator=len(complete),component_denominator=2*len(complete),
            exact_windows=sum(all(r) for r in matches),component_errors=sum(not v for row in matches for v in row),
            all_observed_complete_windows_correct=bool(complete) and all(all(r) for r in matches),
            auxiliary=state.report({'core':detail['core'],'rankings':detail['auxiliary_observer']},book,truth)))
    result['claim']='nominal mapping diagnostics only; crop17 floor/ceil not selected for success; seven observed windows are not a full eleven-window trajectory'
    return result


def summary(evaluation):
    channels={}
    for channel in CHANNELS:
        decoder=channel.split('/')[0]
        rows={};off_complete=off_any=off_observed_any=off_no_observations=off_partial=marked_complete=marked_all=marked_correct=0
        for case in EVAL:
            for arm in ('OFF',decoder+'_A',decoder+'_B'):
                source=evaluation.get(case,{}).get('sources',{}).get(arm,{})
                decisions=[source.get('views',{}).get(view,{}).get('decisions',{}).get(channel,{'status':'MISSING','accepted':None}) for view in VIEWS]
                complete=all(r.get('status')=='DECIDED' for r in decisions)
                entry=dict(complete=complete,views=decisions,source_cluster=True)
                if arm=='OFF':
                    off_complete+=complete
                    observed=[r for r in decisions if r.get('status')=='DECIDED']
                    entry['any_observed_false_accept']=any(r['accepted'] for r in observed)
                    off_observed_any+=entry['any_observed_false_accept']
                    off_no_observations+=len(observed)==0;off_partial+=0<len(observed)<3
                    if complete:entry['any_view_false_accept']=any(r['accepted'] for r in decisions);off_any+=entry['any_view_false_accept']
                else:
                    truth=0 if arm.endswith('_A') else 1
                    marked_complete+=complete
                    if complete:
                        entry['all_view_accept']=all(r['accepted'] for r in decisions)
                        entry['all_view_accept_and_correct']=all(r['accepted'] and r.get('selected_message')==truth for r in decisions)
                        marked_all+=entry['all_view_accept'];marked_correct+=entry['all_view_accept_and_correct']
                rows[case+'/'+arm]=entry
        channels[channel]=dict(off_source_denominator=2,off_complete=off_complete,off_any_view_false_accept=off_any,
            off_any_observed_false_accept=off_observed_any,off_no_decided_views=off_no_observations,off_partially_observed_sources=off_partial,
            off_missing_or_uncalibrated=2-off_complete,matched_marked_source_denominator=4,marked_complete=marked_complete,
            marked_all_view_accept=marked_all,marked_all_view_accept_and_correct=marked_correct,marked_missing_or_uncalibrated=4-marked_complete,sources=rows)
    return dict(evaluation_source_denominator=10,evaluation_view_denominator=30,channels=channels,
        claim='two held-out OFF sources; no low-FPR inference; crops clustered by generated source; temporal-pattern ablation only')
