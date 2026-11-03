import numpy as np
import pytest
from main.tube_state import b_joint_time_packet_a_receiver_v1 as r
from main.tube_state import b_joint_time_packet_a_v1 as p

pytestmark=pytest.mark.unit


def observed(states,coding='NEW'):
    bits=r.template(np.asarray(states),coding)['bits']
    return np.repeat((2*bits.astype(float)-1)[:,:,None,:],2,axis=2),np.ones((len(states),5,2,63),bool)


def test_full_hinge_copy_average_zero_support_and_startup():
    soft,mask=observed([14]);soft[0,0,0,0]=0.;soft[0,1,1,1]=np.nan;mask[0,2,0,2]=False
    terms=r.hinge_terms(soft,mask);states=np.array([0,1,2,3,13,14,15])
    got=r.score_states(terms,states,'NEW');t=r.template(states,'NEW')
    for k in range(len(states)):
        total=0.;count=0
        for lane in range(5):
            if not t['active'][k,lane]:continue
            for b in range(63):
                valid=mask[0,lane,:,b]&np.isfinite(soft[0,lane,:,b])
                if not valid.any():continue
                count+=1;v=soft[0,lane,valid,b]
                total+=np.maximum(0,-v if t['bits'][k,lane,b] else v).mean()
        assert got['denominator'][0,k]==count
        assert got['numerator'][0,k]==pytest.approx(total,abs=1e-12)
    assert terms['support'][0,0,0]  # hard erasure, soft support
    startup=r.startup_scores(soft,mask,'NEW')
    assert startup['active'].sum(axis=1).tolist()==[1,2,3,4]


def test_conflicting_copies_and_derived_f_retained_without_truth():
    soft,mask=observed([20])
    alternate,_=observed([21]);soft[:,:,1]=alternate[:,:,1]
    out=r.diagnostics(soft,mask,'NEW');node=out['nodes'][0]
    assert node['compatible_n']==[20,21]
    assert all(l['copy_conflict'] for l in node['lanes'])
    assert all(x['status']=='PARTIAL_ERASURE_UNRESOLVED' for x in node['observations'] if x['role']=='F')
    assert out['formal_output'] is None
    # Scores can evaluate a state absent from every hard candidate.
    assert r.score_states(r.hinge_terms(soft,mask),np.array([123]),'NEW')['scores'].shape==(1,1)


def test_exact_edges_preserve_inversion_and_old_semantics():
    soft,mask=observed([10,10,11,14,13])
    out=r.diagnostics(soft,mask,'NEW')
    assert [x['status'] for x in out['edges']]==['UNCALIBRATED_EXACT_REPEAT','UNCALIBRATED_EXACT_NORMAL','UNCALIBRATED_EXACT_GAP','INVERSION']
    old,mask=observed([20],'OLD');out=r.diagnostics(old,mask,'OLD')
    assert out['edges']==[] and out['nodes'][0]['compatible_n'] is None
    assert all(x.get('candidate_j') is not None and x['candidate_n'] is None for x in out['nodes'][0]['observations'])


def test_t7_illegal_and_erasure_no_fill():
    soft,mask=observed([11]);soft[0,1,0,:7]*=-1
    out=r.diagnostics(soft,mask,'NEW')
    row=out['nodes'][0]['observations'][3]
    assert row['candidate_n']==11 and len(row['corrected_positions'])==7
    illegal=np.array(p.encode_raw24(8*10+7));soft[0,0,0]=2*illegal-1
    soft[0,0,1,0]=0
    out=r.diagnostics(soft,mask,'NEW');rows=out['nodes'][0]['observations']
    assert rows[0]['status']=='INVALID_PACKET' and 'INVALID_AGE' in rows[0]['violations']
    assert rows[1]['status']=='PARTIAL_ERASURE_UNRESOLVED'
