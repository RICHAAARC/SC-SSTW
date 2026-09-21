"""Post-read nominal support diagnostics. Never searches, reranks or changes detection."""
from collections import Counter
import math
import numpy as np
from main.tube_state import state_clock

WINDOWS=11
PATH_FIELDS=('g','scale','offset','boundary','delta')
MODES=('global_matched','global_state','local_matched','local_without_update','local_state')

def frame_map(view):
    if view.startswith('crop'):return list(range(int(view[4:]),int(view[4:])+129))
    if view=='delete90':return list(range(90))+list(range(91,181))
    if view=='speed125':return [(5*i+2)//4 for i in range(145)]
    if view in ('full181','resaved'):return list(range(181))
    raise ValueError('unknown fixed view')

def geometry(detail,mapping):
    """Equality is literal RGB support equality, not synchronization success."""
    rows=[];reuse=[];available=set(mapping)
    for n in range(WINDOWS):
        origin=None if detail is None else detail['origins'][n]
        selected=[None]*4 if detail is None else detail['selected'][n]
        valid=False if detail is None else bool(detail['valid'][n])
        groups=[]
        for k,j in enumerate(selected):
            expected=list(range(1+16*n+4*k,5+16*n+4*k))
            received=[] if j is None or origin is None else list(range(origin+1+4*j,origin+5+4*j))
            actual=[mapping[r] for r in received if 0<=r<len(mapping)]
            common=set(expected)&set(actual)
            if j is not None:reuse.append((origin,j))
            groups.append(dict(group=k,selected_group=j,received_rgb_indices=received,source_rgb_indices=actual,nominal_source_rgb_indices=expected,
                               exact_source_list=actual==expected if len(actual)==4 else False,overlap_count=len(common),
                               missing_source=sorted(set(expected)-set(actual)),extra_source=sorted(set(actual)-set(expected)),
                               source_center_error=None if not actual else sum(actual)/len(actual)-sum(expected)/4))
        actual=[v for g in groups for v in g['source_rgb_indices']];nominal=list(range(1+16*n,17+16*n))
        equality=valid and all(g['exact_source_list'] for g in groups)
        rows.append(dict(reference_window=n,valid=valid,origin=origin,selected_groups=selected,groups=groups,
                         nominal_all_frames_present_in_received=all(v in available for v in nominal),
                         observed_group_count=sum(len(g['source_rgb_indices'])==4 for g in groups),
                         support_status='UNOBSERVED_OR_INCOMPLETE' if not valid else ('EQUAL_NOMINAL_SUPPORT' if equality else 'PARTIAL_OR_DIFFERENT_NOMINAL_SUPPORT'),
                         exact_source_support_equality=equality,overlap_count=len(set(actual)&set(nominal)),
                         missing_source=sorted(set(nominal)-set(actual)),extra_source=sorted(set(actual)-set(nominal)),
                         repeated_source_indices=[v for v,c in Counter(actual).items() if c>1],
                         synchronization_success=None,VAE_equivalence=None))
    return dict(window_denominator=11,valid_windows=sum(r['valid'] for r in rows),missing_or_incomplete_windows=sum(not r['valid'] for r in rows),
                equal_nominal_support_windows=sum(r['exact_source_support_equality'] for r in rows),
                nominal_fully_available_windows=sum(r['nominal_all_frames_present_in_received'] for r in rows),
                repeated_observation_groups=[dict(origin=g,group=j,count=c) for (g,j),c in Counter(reuse).items() if c>1],windows=rows,
                exact_synchronization_success=None,exact_frame_localization=None,
                meaning='nominal four-RGB group support only, not VAE receptive fields; equality is not synchronization accuracy. Resampling necessarily omits source frames; missing/equality counts are geometric coverage, not algorithm failure rates')

def existing_geometry(view):
    saved=view.get('sync_reporting_only',{}).get('selected_window_geometry',[])
    if len(saved)!=11:return None
    return dict(origins=[r['origin'] for r in saved],selected=[r['selected_groups'] for r in saved],valid=[r['valid'] for r in saved])

def signature(detail):return tuple((g,tuple(s)) for g,s in zip(detail['origins'],detail['selected']))

def path_windows(path,detail,observations):
    """Check persisted selected groups against the unchanged allocation, without observations/tensors."""
    rows=[];cache={}
    for n in range(11):
        delta=path['delta'] if n>=path['boundary'] else 0
        origin=(path['g']-delta)%4;offset=path['offset']+delta
        metadata=observations.get(str(origin),{})
        count=(metadata.get('frames_used',1)-1)//4 if metadata.get('status')=='COMPLETE' else None
        key=(origin,offset,count)
        if key not in cache:cache[key]=[None]*44 if count is None else state_clock.carrier.allocation(count,origin,*path['scale'],offset)
        expected=cache[key][4*n:4*n+4]
        rows.append(dict(reference_window=n,origin=origin,offset=offset,scale=path['scale'],selected_groups=detail['selected'][n],
                         allocation_matches_record=origin==detail['origins'][n] and expected==detail['selected'][n],
                         complete_four_groups=all(v is not None for v in expected),
                         interpretation='scale/offset on received RGB centers; boundary step +/-1 source RGB unit, not arbitrary deletion/repetition DP'))
    return rows

def observer_check(detail,key):
    qs=np.asarray(detail['q'],dtype=float);valid=detail['valid'];rows=[]
    if qs.shape!=(11,2) or len(valid)!=11 or not np.isfinite(qs).all():raise ValueError('invalid saved observer observations')
    for message in (0,1):
        states,steps,_=state_clock.trajectory(key,message)
        current=state_clock.observe(qs,valid,states,np.asarray(steps))
        frozen=state_clock.observe(qs,valid,states,np.asarray(steps),update=False)
        old=next(r for r in detail['scores'] if r['message']==message);stored=old['observer']
        residual={field:float(np.max(np.abs(np.asarray(current[field])-np.asarray(stored[field])))) for field in ('innovation_by_window','predictions','updated_states')}
        residual['innovation_mean']=abs(current['innovation_mean']-stored['innovation_mean'])
        matched=float(np.mean(np.sum(qs*states,axis=1)/2))
        residual['matched_score']=abs(matched-old['matched_score'])
        residual['state_score']=abs(matched-state_clock.INNOVATION_WEIGHT*current['innovation_mean']-old['state_score'])
        residual['without_update_score']=abs(matched-state_clock.INNOVATION_WEIGHT*frozen['innovation_mean']-old['without_update_score'])
        rows.append(dict(message=message,max_abs_residual=residual,consistent_with_saved=max(residual.values())<=1e-12,
                         reference_steps=list(steps),predictions=current['predictions'],updated_states=current['updated_states'],
                         innovation_by_window=current['innovation_by_window'],
                         measurement_update_used=[bool(v and np.linalg.norm(q)>1e-12) for v,q in zip(valid,qs)]))
    return dict(status='CHECKED',messages=rows,meaning='original reference-window n propagation; invalid/zero-q windows propagate prediction only with innovation1; no receiver or ranking update')

def analyze(view_name,view,blind=None,key=None):
    mapping=frame_map(view_name);savedmap=view.get('sync_reporting_only',{}).get('received_to_source_frame_indices')
    result=dict(original_status=view.get('status','MISSING'),original_statistic=view.get('statistic'),original_decision=view.get('decision'),
                original_attribution=view.get('attribution'),original_sync_reporting=view.get('sync_reporting_only'),
                mapping_matches_saved=None if savedmap is None else mapping==savedmap,
                selected_geometry=geometry(existing_geometry(view),mapping),full_receiver_status='MISSING_FULL_RECEIVER',
                candidate_diagnostics=None,observer_diagnostics=None,exact_sync=None)
    if blind is None:return result
    candidates=blind['candidates'];classes=blind['classes'];rankings=blind['rankings']
    def cls(cid):return classes[str(cid)] if str(cid) in classes else classes[cid]
    expected=list(state_clock.clock_paths())
    protocol_match=len(candidates)==4284 and len(expected)==4284 and all(all(a[k]==b[k] for k in PATH_FIELDS) for a,b in zip(candidates,expected))
    signatures=[signature(d) for d in classes.values()]
    best=rankings['local_state'].get('best');detail=cls(best['class']) if best is not None else None
    mode_rows={}
    for mode in MODES:
        ranking=rankings[mode];winner=ranking.get('best');ties=ranking.get('top_ties',[])
        tie_support=[]
        for cid in sorted({t['class'] for t in ties}):
            g=geometry(cls(cid),mapping)
            tie_support.append(dict(observation_class=cid,valid_windows=g['valid_windows'],equal_nominal_support_windows=g['equal_nominal_support_windows'],
                                    nominal_fully_available_windows=g['nominal_fully_available_windows'],
                                    support_by_window=[dict(n=w['reference_window'],valid=w['valid'],status=w['support_status'],overlap_count=w['overlap_count']) for w in g['windows']],
                                    exact_reference_compatibility=None,meaning='posthoc support summary, no truth reranking; equality/partial coverage is not exact clock compatibility'))
        observer_matches=None;score_residual=None
        if winner is not None:
            score=next(r for r in cls(winner['class'])['scores'] if r['message']==winner['message'])
            observer_matches=ranking.get('best_observer')==score['observer']
            field='matched_score' if mode in ('global_matched','local_matched') else ('without_update_score' if mode=='local_without_update' else 'state_score')
            score_residual=abs(winner['score']-(score[field]-(state_clock.EDIT_COST if winner['delta'] else 0.)))
        mode_rows[mode]=dict(original_ranking=ranking,top_tie_paths=len(ties),top_tie_classes=len({t['class'] for t in ties}),
                             top_tie_support_diagnostics=tie_support,best_observer_matches_class=observer_matches,best_score_class_residual=score_residual,
                             best_class_path_members=[] if winner is None else [r for r in candidates if r['class']==winner['class']],
                             selected_path_windows=[] if winner is None else path_windows(winner,cls(winner['class']),view.get('observations',{})))
    result.update(full_receiver_status='CHECKED',candidate_diagnostics=dict(recorded_count=blind.get('candidate_count'),actual_count=len(candidates),
        original_protocol_paths_match=protocol_match,recorded_classes=blind.get('effective_observation_classes'),actual_classes=len(classes),
        recorded_counts_consistent=blind.get('candidate_count')==len(candidates) and blind.get('effective_observation_classes')==len(classes) and len(set(signatures))==len(classes),
        unique_observation_signatures=len(set(signatures)),scale_counts=dict(Counter(str(c['scale']) for c in candidates)),
        local_edit_counts=dict(Counter(str(c['delta']) for c in candidates)),modes=mode_rows,
        meaning='persisted candidates only; observation-equivalent paths need not identify unique RGB time or editing event'),
        selected_geometry=geometry(detail,mapping),observer_diagnostics=None if detail is None or key is None else observer_check(detail,key))
    saved=existing_geometry(view)
    result['best_geometry_matches_saved']=None if saved is None or detail is None else signature(saved)==signature(detail) and saved['valid']==detail['valid']
    by=rankings['local_state'].get('best_by_message',{})
    result['message_scores_match_saved']=all(by.get(str(m)) is not None and by[str(m)]['score']==view.get('statistic',{}).get('message_scores',{}).get(str(m)) for m in (0,1))
    stat=view.get('statistic',{});unique=rankings['local_state'].get('message_unique')
    result['best_statistic_matches_saved']=best is not None and best['score']==stat.get('score') and unique==stat.get('message_unique') and (best['message'] if unique else None)==stat.get('selected_message')
    return result
