"""Frozen source-cluster existence calibration, distinct from message ranking."""
import math

CALIBRATION=tuple('cal%02d'%i for i in range(1,10))
EVALUATION=('eval01','eval02')
VIEWS=('full181','crop0','crop4','crop5','delete90','speed125','resaved')
SEQUENCE=('crop0','crop4','crop5')
MODE='local_state'
ARMS=('OFF','LAST_A','LAST_B')
TIE=1e-12

def statistic(detection,complete=True):
    """Only complete blind receiver output. Truth/attack labels are not inputs."""
    r=detection['rankings'][MODE]
    by=r.get('best_by_message',{})
    scores={str(m):by.get(str(m),{}).get('score') if by.get(str(m)) else None for m in (0,1)}
    if not complete or any(v is None or not math.isfinite(v) for v in scores.values()):
        return dict(status='UNMEASURED',score=None,message_scores=scores,selected_message=None,message_unique=False)
    best=max(scores.values());ties=[int(m) for m,v in scores.items() if abs(v-best)<=TIE]
    return dict(status='MEASURED',score=best,message_scores=scores,selected_message=ties[0] if len(ties)==1 else None,message_unique=len(ties)==1,message_ties=ties)

def sequence(views):
    rows=[views.get(v,{}).get('statistic',{}) for v in SEQUENCE]
    if any(r.get('status')!='MEASURED' for r in rows):
        return dict(status='UNMEASURED',score=None,selected_message=None,message_unique=False,required_views=list(SEQUENCE))
    scores={str(m):sum(r['message_scores'][str(m)] for r in rows)/3 for m in (0,1)}
    best=max(scores.values());ties=[int(m) for m,v in scores.items() if abs(v-best)<=TIE]
    return dict(status='MEASURED',score=best,message_scores=scores,selected_message=ties[0] if len(ties)==1 else None,message_unique=len(ties)==1,message_ties=ties,
                required_views=list(SEQUENCE),meaning='one source; mean of three fixed overlapping fragment message scores, not independent samples')

def freeze(calibration):
    missing=[];maxima={}
    for case in CALIBRATION:
        source=calibration.get(case,{}).get('sources',{}).get('OFF',{})
        rows=[source.get('views',{}).get(v,{}).get('statistic',{}) for v in VIEWS]+[source.get('sequence',{})]
        for name,row in zip((*VIEWS,'sequence'),rows):
            if row.get('status')!='MEASURED' or row.get('score') is None or not math.isfinite(row['score']):missing.append(case+'/'+name)
        if all(r.get('status')=='MEASURED' and r.get('score') is not None and math.isfinite(r['score']) for r in rows):maxima[case]=max(r['score'] for r in rows)
    return dict(status='FROZEN',calibration_status='UNCALIBRATED' if missing else 'CALIBRATED',source_denominator=9,
                source_maxima=maxima,missing=missing,threshold=None if missing else max(maxima.values()),mode=MODE,
                rule='strict score > maximum of nine source maxima',rank_resolution=.1,
                claim='conditional exchangeability rank rule only; nine hand-selected sources do not establish population FPR')

def decide(statistic,frozen):
    if frozen.get('calibration_status')!='CALIBRATED' or statistic.get('status')!='MEASURED':
        return dict(status='UNDECIDED',accepted=None,score=statistic.get('score'),threshold=frozen.get('threshold'))
    return dict(status='DECIDED',accepted=statistic['score']>frozen['threshold'],score=statistic['score'],threshold=frozen['threshold'])

def attribution(statistic,truth):
    scores=statistic.get('message_scores',{})
    return dict(truth_reporting_only=truth,message_unique=statistic.get('message_unique'),selected_message=statistic.get('selected_message'),
                unique_correct=None if truth is None or statistic.get('status')!='MEASURED' else bool(statistic.get('message_unique') and statistic.get('selected_message')==truth),
                correct_minus_other=None if truth is None or any(scores.get(str(m)) is None for m in (0,1)) else scores[str(truth)]-scores[str(1-truth)])

def summary(evaluation):
    result={}
    for view in (*VIEWS,'sequence'):
        rows=[]
        for case in EVALUATION:
            for arm in ARMS:
                source=evaluation.get(case,{}).get('sources',{}).get(arm,{})
                row=source.get('sequence',{}) if view=='sequence' else source.get('views',{}).get(view,{})
                rows.append(dict(case=case,arm=arm,decision=row.get('decision',{}),attribution=row.get('attribution',{})))
        off=[r for r in rows if r['arm']=='OFF'];marked=[r for r in rows if r['arm']!='OFF']
        result[view]=dict(off_denominator=2,off_decided=sum(r['decision'].get('accepted') is not None for r in off),off_false_accept=sum(r['decision'].get('accepted') is True for r in off),
                         marked_denominator=4,marked_decided=sum(r['decision'].get('accepted') is not None for r in marked),marked_accept=sum(r['decision'].get('accepted') is True for r in marked),
                         marked_accept_and_correct=sum(r['decision'].get('accepted') is True and r['attribution'].get('unique_correct') is True for r in marked),rows=rows)
    source_rows=[]
    for case in EVALUATION:
        ds=[result[v]['rows'] for v in (*VIEWS,'sequence')]
        off=[r['decision'].get('accepted') for group in ds for r in group if r['case']==case and r['arm']=='OFF']
        source_rows.append(dict(case=case,complete=all(v is not None for v in off),any_observed_accept=any(v is True for v in off),any_accept=True if any(v is True for v in off) else (None if any(v is None for v in off) else False)))
    return dict(by_view=result,off_source_any=dict(denominator=2,complete=sum(r['complete'] for r in source_rows),false_accept=sum(r['any_accept'] is True for r in source_rows),unresolved=sum(r['any_accept'] is None for r in source_rows),sources=source_rows),
                claim='two evaluation OFF source clusters; report attack-specific outcomes, not low FPR or independent crop counts')
