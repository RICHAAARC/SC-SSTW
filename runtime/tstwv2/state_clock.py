"""Finite circular state + fixed-gain observer + bounded clock hypotheses.

This is a terminal-carrier adaptation, not a calibrated Kalman filter, LLR,
AISB, or generative Flow control. Receiver inputs exclude writer evidence.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np
from runtime.tstwv2 import projection_margin as carrier

CONTEXT = {"schema": "wan_projection_state_clock_v1", "nonce": "2d9800a4d5dc40669ba089a042335c66", "windows": 11, "source_frames": 181}
CONTEXT_DIGEST = hashlib.sha256(json.dumps(CONTEXT,sort_keys=True,separators=(',',':')).encode()).hexdigest()
PHASES = np.array([[1.,1.],[-1.,1.],[-1.,-1.],[1.,-1.]])
GAIN, INNOVATION_WEIGHT, EDIT_COST = .5, .05, .002

def trajectory(key: bytes, message: int) -> tuple:
    """s[n+1]=R(pi/2*(base+2*u[n])) s[n]; ||s||=sqrt(2)."""
    role = 'state_clock/' + CONTEXT_DIGEST + '/'
    initial = carrier.digest(key,role+'initial',0)[0] % 4
    base = 1 if carrier.sign(key,role+'transition',0)>0 else 3
    drive = [int(carrier.sign(key,role+f'message/{message}',n)>0) for n in range(10)]
    phases=[initial]
    steps=[]
    for u in drive:
        step=(base+2*u)%4
        steps.append(step)
        phases.append((phases[-1]+step)%4)
    return PHASES[phases].copy(), steps, drive

def codebook(key: bytes) -> dict:
    book=carrier.codebook(key)
    polarity=np.array([carrier.sign(key,'state_clock/polarity/'+CONTEXT_DIGEST,i) for i in range(1760)])
    states,steps,drives=zip(*(trajectory(key,m) for m in (0,1)))
    axes=np.arange(1760)%2
    book['states']=np.stack(states)
    book['steps']=np.asarray(steps)
    book['drives']=np.asarray(drives)
    book['polarity']=polarity
    book['codes']=np.stack([s[np.arange(1760)//160,axes]*polarity*book['sync'] for s in states])
    return book

def rotate(state: np.ndarray, step: int) -> np.ndarray:
    # Integer rotations avoid floating phase drift.
    x,y=state
    return np.array(((x,y),(-y,x),(-x,-y),(y,-x))[step%4])

def observe(qs: np.ndarray, valid: list, states: np.ndarray, steps: np.ndarray, *, update=True) -> dict:
    """Prediction -> directional innovation -> fixed gain correction.

    No affine channel is fitted. Missing windows get innovation cost 1 and
    prediction-only propagation. Weights are protocol constants, not noise
    probabilities estimated from the test video.
    """
    estimate=states[0]/np.sqrt(2)
    errors=[]; estimates=[]; predictions=[]
    for n,q in enumerate(qs):
        if n:
            estimate=rotate(estimate,int(steps[n-1]))
        predicted=estimate.copy(); predictions.append(predicted.tolist())
        norm=float(np.linalg.norm(q))
        if valid[n] and norm>1e-12:
            measured=q/norm
            errors.append(float(np.sum((measured-predicted)**2)/4))
            if update:
                merged=(1-GAIN)*predicted+GAIN*measured
                length=np.linalg.norm(merged)
                estimate=merged/length if length>1e-12 else predicted
        else:
            errors.append(1.)
        estimates.append(estimate.tolist())
    return {'innovation_mean':float(np.mean(errors)), 'innovation_by_window':errors,
            'predictions':predictions,'updated_states':estimates}

def emissions(observations: dict, book: dict, n:int,g:int,num:int,den:int,b:int) -> dict:
    z=observations.get(g)
    selected=[None]*4 if z is None else carrier.allocation(z.shape[2]-1,g,num,den,b)[4*n:4*n+4]
    if any(j is None for j in selected):
        return {'selected':selected,'valid':False,'q':[0.,0.],'scores':[0.,0.],'agreement':[None,None]}
    data=np.take(z[0],[j+1 for j in selected],axis=1).transpose(1,0,2,3)
    data=data.reshape(4,16,10,4,16,4).transpose(2,4,0,1,3,5).reshape(160,1024)
    sl=slice(160*n,160*(n+1))
    values=np.sum(data.astype(float)*book['directions'][sl].astype(float),axis=1)
    clipped=np.clip(values,-1,1)
    decoded=clipped*book['sync'][sl]*book['polarity'][sl]
    return {'selected':selected,'valid':True,'q':[float(decoded[::2].mean()),float(decoded[1::2].mean())],
            'scores':[float((clipped*book['codes'][m,sl]).mean()) for m in (0,1)],
            'agreement':[float((values*book['codes'][m,sl]>0).mean()) for m in (0,1)]}

def clock_paths():
    """204 global hypotheses, each with no change or one +/-1 source-frame step.

    Change at a reference-window boundary n=1..10. The receiver origin changes
    with the clock phase: g_n=(g-delta)%4, b_n=b+delta. This selects only the
    already persisted four VAE encodes, never a new re-encode or repaired RGB.
    """
    for g in range(4):
        for num,den in carrier.SCALES:
            for b in carrier.OFFSETS:
                for boundary,delta in [(11,0)]+[(n,d) for n in range(1,11) for d in (-1,1)]:
                    yield {'g':g,'scale':[num,den],'offset':b,'boundary':boundary,'delta':delta}

def read(observations:dict,book:dict) -> dict:
    cache={}; classes={}; details={}; candidates=[]
    for path in clock_paths():
        windows=[]; origins=[]
        for n in range(11):
            delta=path['delta'] if n>=path['boundary'] else 0
            g=(path['g']-delta)%4; b=path['offset']+delta
            key=(n,g,*path['scale'],b)
            if key not in cache:
                cache[key]=emissions(observations,book,n,g,*path['scale'],b)
            windows.append(cache[key]);origins.append(g)
        # Observation equivalence; event cost stays explicit on each candidate.
        signature=tuple((g,tuple(w['selected'])) for g,w in zip(origins,windows))
        if signature not in classes:
            cid=len(classes);classes[signature]=cid
            qs=np.array([w['q'] for w in windows]);valid=[w['valid'] for w in windows]
            scores=[]
            for m in (0,1):
                obs=observe(qs,valid,book['states'][m],book['steps'][m])
                frozen=observe(qs,valid,book['states'][m],book['steps'][m],update=False)
                matched=float(np.mean([w['scores'][m] for w in windows]))
                scores.append({'message':m,'matched_score':matched,'state_score':matched-INNOVATION_WEIGHT*obs['innovation_mean'],
                               'without_update_score':matched-INNOVATION_WEIGHT*frozen['innovation_mean'],
                               'observer':obs,'matched_supports':160*sum(valid),
                               'aligned_payload_agreement':float(np.mean([w['agreement'][m] for w in windows if w['valid']])) if any(valid) else None})
            details[cid]={'origins':origins,'selected':[w['selected'] for w in windows],'q':qs.tolist(),'valid':valid,'scores':scores}
        cid=classes[signature]
        candidates.append(path|{'class':cid,'event_cost':EDIT_COST if path['delta'] else 0.})
    rankings={}
    for mode in ('global_matched','global_state','local_matched','local_without_update','local_state'):
        pairs=[]
        for row in candidates:
            if mode.startswith('global') and row['delta']:
                continue
            for score in details[row['class']]['scores']:
                if not score['matched_supports']:
                    continue
                field = 'matched_score' if mode in ('global_matched','local_matched') else ('without_update_score' if mode=='local_without_update' else 'state_score')
                value=score[field]-row['event_cost']
                pairs.append((value,row,score))
        pairs.sort(key=lambda p:(-p[0],abs(p[1]['delta']),abs(p[1]['offset']),abs(p[1]['scale'][0]/p[1]['scale'][1]-1),p[1]['g'],p[1]['boundary'],p[1]['delta'],p[2]['message']))
        def compact(p):
            value,row,s=p
            return row|{k:v for k,v in s.items() if k!='observer'}|{'score':value}
        best=compact(pairs[0]) if pairs else None
        ties=[compact(p) for p in pairs if abs(p[0]-pairs[0][0])<=1e-12]
        rankings[mode]={'best':best,'best_by_message':{str(m):next((compact(p) for p in pairs if p[2]['message']==m),None) for m in (0,1)},
                        'top_ties':ties,'message_unique':len({r['message'] for r in ties})==1,
                        'best_observer':None if not pairs else pairs[0][2]['observer']}
    return {'candidate_count':len(candidates),'global_candidate_count':204,'effective_observation_classes':len(classes),
            'rankings':rankings,'candidates':candidates,'classes':details,
            'claim':'fixed-gain directional observer and bounded clock comparison; no calibrated likelihood or AISB'}

def report(detection:dict,truth:int,delta:int,edit_frame:int=138) -> dict:
    """Post-ranking truth join. The edited window is retained in every score."""
    event_window=(edit_frame-1)//16
    reference=next(r for r in detection['candidates'] if r['g']==0 and r['scale']==[1,1] and r['offset']==0 and r['delta']==delta and r['boundary']==(event_window+1 if delta else 11))
    target=detection['classes'][reference['class']]
    checked=[n for n in range(11) if not (delta and n==event_window)]
    result={'true_message':truth,'nominal_clock_reference':reference,'time_checked_windows':checked,
            'edited_source_frame_zero_based':edit_frame if delta else None,
            'event_window_excluded_from_time_metric_only':event_window if delta else None,
            'time_claim':'window correspondence only, not unique frame-level edit localization','modes':{}}
    for mode,ranking in detection['rankings'].items():
        by=ranking['best_by_message'];true,wrong=by[str(truth)],by[str(1-truth)]
        best=ranking['best']
        current=None if best is None else detection['classes'][best['class']]
        matched=[] if current is None else [n for n in checked if current['origins'][n]==target['origins'][n] and current['selected'][n]==target['selected'][n]]
        result['modes'][mode]={'true_minus_wrong_best_score':None if true is None or wrong is None else true['score']-wrong['score'],
                               'nominal_time_matching_windows':matched,'nominal_time_denominator':len(checked),
                               'all_checked_windows_match':len(matched)==len(checked)}
    return result
