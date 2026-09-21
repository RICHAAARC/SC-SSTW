"""Small synthetic/JSON CPU checks; no model, media or new research sample."""
import copy,hashlib,json
import numpy as np
import pytest
from main.tube_state import state_clock
from experiments.wan_state_clock import flow_tube_sync_diagnostics as core,flow_tube_sync_diagnostics_run as run
pytestmark=pytest.mark.unit

def detail():return dict(origins=[0]*11,selected=[list(range(4*n,4*n+4)) for n in range(11)],valid=[True]*11)

def test_deleted_boundary_and_speed_supports_are_not_sync_failures():
    d=detail()
    for n in range(6,11):d['origins'][n]=3;d['selected'][n]=list(range(4*n-1,4*n+3))
    g=core.geometry(d,core.frame_map('delete90'))
    assert g['valid_windows']==11 and g['equal_nominal_support_windows']==10
    assert g['windows'][5]['overlap_count']==15
    assert g['windows'][5]['support_status']=='PARTIAL_OR_DIFFERENT_NOMINAL_SUPPORT'
    assert g['nominal_fully_available_windows']==10 and g['exact_synchronization_success'] is None
    # Real frozen speed winner geometry, with reference boundary9 and -1 RGB local change.
    path=dict(g=1,scale=[5,4],offset=0,boundary=9,delta=-1);d=detail()
    for n in range(11):
        delta=path['delta'] if n>=path['boundary'] else 0;origin=(1-delta)%4
        count=(145-origin-1)//4;d['origins'][n]=origin
        d['selected'][n]=state_clock.carrier.allocation(count,origin,5,4,delta)[4*n:4*n+4]
        d['valid'][n]=all(v is not None for v in d['selected'][n])
    g=core.geometry(d,core.frame_map('speed125'))
    assert g['valid_windows']==3 and g['equal_nominal_support_windows']==0
    assert all(r['overlap_count']==13 for r in g['windows'] if r['valid'])
    assert all(r['synchronization_success'] is None for r in g['windows'])
    assert len(g['windows'])==11 and g['missing_or_incomplete_windows']==8

def test_full_saved_candidates_observer_and_ties_never_reranked():
    key=b'test';d=detail();states,_,_=state_clock.trajectory(key,0);d['q']=(states*.25).tolist();d['scores']=[]
    for m in (0,1):
        s,steps,_=state_clock.trajectory(key,m);o=state_clock.observe(np.array(d['q']),d['valid'],s,np.array(steps));f=state_clock.observe(np.array(d['q']),d['valid'],s,np.array(steps),update=False)
        matched=float(np.mean(np.sum(np.asarray(d['q'])*s,axis=1)/2))
        d['scores'].append(dict(message=m,matched_score=matched,state_score=matched-.05*o['innovation_mean'],without_update_score=matched-.05*f['innovation_mean'],observer=o))
    candidates=[p|{'class':0,'event_cost':.002 if p['delta'] else 0.} for p in state_clock.clock_paths()]
    base=next(c for c in candidates if c['g']==0 and c['scale']==[1,1] and c['offset']==0 and c['delta']==0)
    rankings={}
    for mode in core.MODES:
        field='matched_score' if mode in ('global_matched','local_matched') else ('without_update_score' if mode=='local_without_update' else 'state_score')
        by={str(m):base|{'message':m,'score':d['scores'][m][field]} for m in (0,1)}
        rankings[mode]=dict(best=by['0'],best_by_message=by,top_ties=[by['0']],message_unique=True,best_observer=d['scores'][0]['observer'])
    blind=dict(candidates=candidates,classes={'0':d},rankings=rankings,candidate_count=4284,effective_observation_classes=1)
    stat=dict(score=rankings['local_state']['best']['score'],message_scores={str(m):rankings['local_state']['best_by_message'][str(m)]['score'] for m in (0,1)},message_unique=True,selected_message=0)
    view=dict(statistic=stat,decision={'accepted':True,'threshold':.01},attribution={'unique_correct':True},observations={str(g):dict(status='COMPLETE',frames_used=181 if g==0 else 177) for g in range(4)},sync_reporting_only=dict(received_to_source_frame_indices=core.frame_map('full181'),selected_window_geometry=[dict(origin=0,selected_groups=s,valid=True) for s in d['selected']]))
    before=copy.deepcopy((view,blind));r=core.analyze('full181',view,blind,key)
    assert (view,blind)==before
    assert r['best_statistic_matches_saved'] and r['message_scores_match_saved']
    assert r['candidate_diagnostics']['original_protocol_paths_match'] and r['candidate_diagnostics']['recorded_counts_consistent']
    assert all(x['consistent_with_saved'] for x in r['observer_diagnostics']['messages'])
    assert r['candidate_diagnostics']['modes']['local_state']['top_tie_support_diagnostics'][0]['exact_reference_compatibility'] is None
    assert r['original_decision']==view['decision'] and r['exact_sync'] is None

def test_missing_details_hash_and_fixed_denominators(tmp_path):
    r=core.analyze('delete90',{})
    assert r['full_receiver_status']=='MISSING_FULL_RECEIVER' and r['candidate_diagnostics'] is None
    assert r['selected_geometry']['equal_nominal_support_windows']==0 and len(r['selected_geometry']['windows'])==11
    path=tmp_path/'x.json';path.write_text('{}');inputs=[]
    with pytest.raises(ValueError,match='hash mismatch'):run.read_json(path,inputs,'0'*64)
    r=run.run(tmp_path/'missing-source',tmp_path/'audit')
    assert r['view_denominator']==42 and len(r['rows'])==42 and r['window_denominator']==462
    assert len(r['original_sequences'])==6 and r['scientific_new_samples']==0
    assert all(v['status']=='MISSING_OR_INVALID' for v in r['rows'])
    with pytest.raises(ValueError,match='nonnested'):run.run(tmp_path/'source',tmp_path/'source'/'audit')
