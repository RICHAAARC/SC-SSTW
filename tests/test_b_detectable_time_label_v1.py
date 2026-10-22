"""Card105 fixed synthetic/algebraic checks only. No real observations or model IO."""
import inspect,math
import pytest
from main.tube_state import b_detectable_time_label_v1 as m
pytestmark=pytest.mark.unit
def zero():return tuple((0.,)*64 for _ in range(48))
def qword(word,b,pilot=None):
    bits=[0]*48
    for i,l in enumerate(m.PILOT_CHIPS):bits[l]=b.pilot[i] if pilot is None else pilot[i]
    for i,l in enumerate(m.DATA_CHIPS):bits[l]=word[i]^b.mask[i]
    return [d+m.DELTA*bit for d,bit in zip(b.dithers,bits)]

def test_rm_definition_distance_and_radius3_disjoint():
    assert len(set(m.CODEWORDS))==64
    assert m.CODEWORDS[0]==(0,)*32 and m.CODEWORDS[1]==(1,)*32
    distances=[sum(x!=y for x,y in zip(a,b)) for i,a in enumerate(m.CODEWORDS) for b in m.CODEWORDS[i+1:]]
    assert len(distances)==2016 and min(distances)==16 and set(distances)=={16,32}
    assert min(distances)>2*m.RADIUS
    assert len(m.PILOT_CHIPS)==16 and len(m.DATA_CHIPS)==32
    assert all(sum(l//12==r for l in m.PILOT_CHIPS)==4 for r in range(4))

def test_balanced_keyed_blocks_and_host_aware_bounds():
    b=m.book('fixture-key')
    for u in b.vectors:
        assert sum(u)==0 and sum(v*v for v in u)==1 and u.count(1/8)==32
    assert b==m.book('fixture-key') and b!=m.book('other-fixture-key')
    source=tuple(tuple(m.fp32((l-24)*.03125+(i-32)*.015625) for i in range(64)) for l in range(48))
    changed,r=m.encode_label(source,17,'fixture-key')
    assert source!=changed and r['ideal_energy']<=4+1e-12
    for roi in range(4):
        assert sum(x['ideal_energy'] for x in r['records'] if x['ROI']==roi)<=1+1e-12
    assert r['actual_FP32_energy']==pytest.approx(sum((x-y)**2 for a,b0 in zip(source,changed) for x,y in zip(a,b0)))
    assert r['FP32_materializations']==1 and r['FP32_correction_attempts']==0 and not r['payload_rewritten']
    for rr in r['records']:
        assert abs(rr['coefficient'])<=m.DELTA+1e-12
        for k in (rr['k']-1,rr['k']+1):
            other=b.dithers[rr['chip']]+m.DELTA*(2*k+rr['bit'])
            assert abs(rr['target_projection']-rr['q_before'])<=abs(other-rr['q_before'])+1e-12

def test_all44_native_fp32_synthetic_roundtrip_not_media_evidence():
    source=zero()
    for label in range(1,45):
        changed,r=m.encode_label(source,label,'fixture-key')
        d=m.decode_blocks(changed,'fixture-key',.5)
        assert d['tag_label']==label and d['code_status']=='LEGAL_ID' and d['nearest_distance']==0
        assert d['pilot_H']==0 and d['pilot_P']>1-1e-10 and r['ideal_energy']<=4
        assert not d['authentication_claim'] and not d['exact_geometry_certified']
    assert 'label' not in inspect.signature(m.decode_projections).parameters
    assert 'phase' not in inspect.signature(m.decode_projections).parameters
    assert m.decode_blocks(changed,'fixture-key',None)['status']=='CALIBRATION_UNRESOLVED'

def test_embed_tie_and_fixed_float_half_integer_symbol_erasure():
    target,k=m.target_projection(m.DELTA,0,0.)
    assert k==0 and target==0
    target,k=m.target_projection(-m.DELTA,0,0.)
    assert k==-1 and target==-2*m.DELTA
    b=m.book('fixture-key');q=qword(m.CODEWORDS[4],b)
    # One data midpoint erases the entire ID; never puncture/fill to decode.
    l=m.DATA_CHIPS[0];q[l]=b.dithers[l]+m.DELTA*.5
    d=m.decode_projections(q,b,.5)
    assert d['code_status']=='SYMBOL_ERASURE' and d['code_distances']==[None]*64
    assert d['nearest'] is None and d['nearest_ids'] is None and d['nearest_distance'] is None
    assert d['pilot_detected'] and d['status']=='DETECTED_LABEL_ERASURE' and d['tag_label'] is None
    q=qword(m.CODEWORDS[4],b);l=m.PILOT_CHIPS[0];q[l]=b.dithers[l]+m.DELTA*.5
    d=m.decode_projections(q,b,.5)
    assert d['pilot_H'] is None and not d['pilot_detected']
    assert math.isfinite(d['pilot_P']) and d['pilot_status']=='PILOT_SYMBOL_ERASURE' and d['tag_label'] is None

def test_reserved_radius_ties_and_weak_pilot_are_separate_outcomes():
    b=m.book('fixture-key')
    d=m.decode_projections(qword(m.CODEWORDS[50],b),b,.5)
    assert d['code_status']=='RESERVED_ID' and d['tag_label'] is None and d['pilot_detected']
    word=list(m.CODEWORDS[0]);word[:3]=[1]*3
    d=m.decode_projections(qword(word,b),b,.5)
    assert d['tag_label']==1 and d['nearest_distance']==3
    word[3]=1;d=m.decode_projections(qword(word,b),b,.5)
    assert d['code_status']=='OUTSIDE_RADIUS' and d['nearest_distance']==4 and d['tag_label'] is None
    word=[0]*32
    for i in [i for i,v in enumerate(m.CODEWORDS[2]) if v][:8]:word[i]=1
    d=m.decode_projections(qword(word,b),b,.5)
    assert d['code_status']=='CODE_TIE' and len(d['nearest_ids'])>1 and d['tag_label'] is None
    pilot=list(b.pilot);pilot[:3]=[1-x for x in pilot[:3]]
    d=m.decode_projections(qword(m.CODEWORDS[2],b,pilot),b,.1)
    assert d['code_status']=='LEGAL_ID' and d['pilot_H']==3 and not d['pilot_detected']
    assert d['pilot_status']=='PILOT_MISMATCH' and d['status']=='CODE_ONLY_REJECT'

def records(P=.1):
    return [dict(**{'class':c},view=v,phase=p,row=r,P=P) for c,v,p,r in sorted(m.null_domain())]
def test_complete_null_max_does_not_filter_Hp_or_invalid_code():
    a,b=records(),records()
    a[0].update(P=.8,pilot_H=16,code_status='RESERVED_ID',geometry='unknown')
    cal=m.calibrate_complete_nulls({'cal-a':a,'cal-b':b})
    assert cal['status']=='FROZEN' and cal['tau']==.8 and cal['hosts']['cal-a']['null_rows']==1200
    assert not cal['ID_screening_used'] and not cal['Hp_screening_used']
    assert m.calibrate_complete_nulls({'a':a[:-1],'b':b})['status']=='CALIBRATION_UNRESOLVED'
    b[3]['technical_failure']=True
    assert m.calibrate_complete_nulls({'a':a,'b':b})['status']=='CALIBRATION_UNRESOLVED'

def test_synthetic_mixture_can_be_legal_so_not_geometry_or_authentication():
    a,_=m.encode_label(zero(),1,'fixture-key')
    b,_=m.encode_label(zero(),2,'fixture-key')
    mixture=tuple(tuple(m.fp32(.9*x+.1*y) for x,y in zip(aa,bb)) for aa,bb in zip(a,b))
    d=m.decode_blocks(mixture,'fixture-key',.5)
    # Deliberate limitation witness, not a prediction of actual VAE mixtures.
    assert d['status']=='ACCEPTED_TAG_CANDIDATE' and d['tag_label']==1
    assert not d['exact_geometry_certified'] and not d['authentication_claim']
