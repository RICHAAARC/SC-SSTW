"""Experimental readout comparisons and crop-only posthoc geometry; no writer inputs to scores."""
from main.tube_state import projection_margin as carrier

NO_SEARCH={'g':0,'scale':[1,1],'offset':0,'delta':0,'boundary':11}
NO_SEARCH_MODES={'no_search_matched':'matched_score','no_search_state':'state_score','no_search_without_update':'without_update_score'}

def class_detail(detection,cid):
    return detection['classes'].get(cid,detection['classes'].get(str(cid)))

def path(detection,fields):
    return next(v for v in detection['candidates'] if all(v[k]==value for k,value in fields.items()))

def no_search_rankings(detection):
    """Same already-blind candidate/class and existing score fields; no new search or normalization."""
    fixed=path(detection,NO_SEARCH);detail=class_detail(detection,fixed['class']);result={}
    for mode,field in NO_SEARCH_MODES.items():
        pairs=[fixed|{k:v for k,v in score.items() if k!='observer'}|{'score':score[field]} for score in detail['scores'] if score['matched_supports']]
        pairs.sort(key=lambda v:(-v['score'],v['message']))
        ties=[v for v in pairs if abs(v['score']-pairs[0]['score'])<=1e-12] if pairs else []
        result[mode]=dict(best=pairs[0] if pairs else None,best_by_message={str(m):next((v for v in pairs if v['message']==m),None) for m in (0,1)},top_ties=ties,message_unique=len({v['message'] for v in ties})==1)
    return result


def reference_geometry(start,frames=129):
    """Only posthoc: source center=start+g+2.5+4*j, hence positive offset=start."""
    g=(-start)%4;groups=(frames-g-1)//4;selected=carrier.allocation(groups,g,1,1,start)
    windows=[selected[4*n:4*n+4] for n in range(11)];valid=[all(v is not None for v in row) for row in windows]
    return dict(path=dict(g=g,scale=[1,1],offset=start,delta=0,boundary=11),selected=windows,origins=[g]*11,valid=valid,
        structural_complete_windows=[n for n,v in enumerate(valid) if v],structural_supports=160*sum(valid),fixed_window_denominator=11)


def crop_alignment_reporting_only(detection,ranking,start):
    reference=reference_geometry(start);candidate=path(detection,reference['path']);detail=class_detail(detection,candidate['class'])
    available=[n for n in reference['structural_complete_windows'] if detail['valid'][n]]
    best=ranking.get('best');selected=class_detail(detection,best['class']) if best else None
    matched=[n for n in available if selected and selected['valid'][n] and selected['origins'][n]==reference['origins'][n] and selected['selected'][n]==reference['selected'][n]]
    ties=ranking.get('top_ties',[]);classes={v['class'] for v in ties}
    return dict(reference=reference,reference_class=candidate['class'],reference_available_windows=available,
        evaluable_window_denominator=len(available),matched_reference_windows=matched,matched_window_fraction=len(matched)/len(available) if available and selected else None,
        selected_class_equals_reference=(best['class']==candidate['class']) if best and available else None,
        top_tie_count=len(ties),top_tie_unique_observation_classes=len(classes),
        top_ties_contain_reference_class=(candidate['class'] in classes) if available else None,
        meaning='observable complete-window mapping only; no missing==missing credit, no frame-level localization or inverse-VAE claim')


def ranked_report(detection,rankings,truth,start):
    """Truth and crop start join only after all blind scoring has completed."""
    rows={}
    for mode,ranking in rankings.items():
        by=ranking.get('best_by_message',{});a=by.get('0');b=by.get('1');best=ranking.get('best')
        gap=(a['score']-b['score']) if a is not None and b is not None else None
        rows[mode]=dict(best=best,best_score_by_message={str(m):by.get(str(m),{}).get('score') if by.get(str(m)) else None for m in (0,1)},
            message_unique=ranking.get('message_unique'),unique_correct_reporting_only=(bool(ranking.get('message_unique')) and best['message']==truth) if best and truth is not None else None,
            best_correct_minus_other_reporting_only=(gap*(1 if truth==0 else -1)) if gap is not None and truth is not None else None,
            reference_score_0_minus_1=gap,nominal_support_denominator=1760,best_matched_supports=best.get('matched_supports') if best else None,
            selected_clock={k:best[k] for k in ('g','scale','offset','delta','boundary','class')} if best else None,
            crop_alignment_reporting_only=crop_alignment_reporting_only(detection,ranking,start))
    return rows
