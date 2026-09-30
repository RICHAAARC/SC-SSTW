import copy
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
import pytest
from main.tube_state import video_temporal_sync_path as m
from main.tube_state import video_temporal_sync_multi as w
from experiments.wan_state_clock import video_temporal_sync_path_run as run

pytestmark = pytest.mark.unit

KEY='watermark'


def independent_paths(length):
    R=45 if length==181 else 31
    for b in ([0] if length==181 else range(53)):
        g=(-b)%4; first=(b+g)//4+1
        for kind,edge,delta in [('ZERO_EDIT',None,0)]+[(k,i,d) for k,d in [('REPEAT',-1),('SKIP',1)] for i in range(2,R+1)]:
            ts=[first+j+(delta if edge is not None and j+1>=edge else 0) for j in range(R)]
            yield dict(id=f'b{b:02d}:{kind}:{edge or 0:02d}',g=g,b=b,ts=ts,valid=all(1<=t<=45 for t in ts))


def brute_scores(X,length,key=KEY):
    pn=np.asarray(m.carrier.pilot_layout(key)[1]);bits=[1,1,1,1,0,0,0,1,0,0,1,1,0,1,0]
    scores={}
    for p in independent_paths(length):
        if not p['valid']:continue
        A=X[p['g']];sign=np.asarray([2*bits[(t-1)//3]-1 for t in p['ts']])
        energy=np.sum(A*A,dtype=np.float64)
        score=0.0 if energy==0 else float(np.sum(A*pn[None,:]*sign[:,None],dtype=np.float64)/np.sqrt(A.size*energy))
        scores[p['id']]=score
    return scores


@pytest.mark.parametrize('length,counts',[(181,(89,45,8)),(129,(3233,3233,499))])
def test_catalog_classes_boundaries_and_exact_equivalence(length,counts):
    rows=m.catalog(length);classes=m.equivalence_classes(rows)
    assert (len(rows),sum(r['structurally_valid'] for r in rows),len(classes))==counts
    independent={p['id']:p for p in independent_paths(length)}
    for row in rows:
        assert list(m.path_taus(row))==independent[row['id']]['ts']
        assert row['structurally_valid']==independent[row['id']]['valid']
    byid={r['id']:r for r in rows}
    assert min(len(c['member_ids']) for c in classes)>1
    assert max(len(c['member_ids']) for c in classes)==(12 if length==181 else 25)
    for c in classes:
        ts=[m.path_taus(byid[pid]) for pid in c['member_ids']]
        assert c['tau_feasible_sets']==[sorted({t[i] for t in ts}) for i in range(len(ts[0]))]
        for t in ts:assert [m.carrier.temporal_sign(x) for x in t]==c['emitted_signs']
    if length==181:
        assert all(r['event_type']=='SKIP' for r in rows if not r['structurally_valid'])
        assert len(next(c for c in classes if c['contains_zero_edit'])['member_ids'])==3
    else:
        assert m.path_taus(byid['b52:SKIP:31'])[-1]==45
        assert m.path_taus(byid['b00:REPEAT:02'])[0:2]==(1,1)
    assert 6*89+24*3233==78126 and 6*44==264 and 6*45+24*3233==77862


@pytest.mark.parametrize('length',[181,129])
def test_dp_independent_enumeration_and_old_zero_baseline(length):
    rng=np.random.default_rng(20260930);R=45 if length==181 else 31
    X={g:rng.normal(size=(R,64)) for g in m.carrier.phases(length)}
    actual=m.analyze(X,length,KEY);scores=brute_scores(X,length)
    assert actual['summary']['status']=='COMPLETE'
    for r in actual['catalog']:
        if r['structurally_valid']:assert abs(r['score']-scores[r['id']])<2e-16
    best=max(scores.values());ties={pid for pid,s in scores.items() if best-s<=1e-12}
    assert set(actual['summary']['top_path_ids'])==ties==set(actual['dp']['final_numerical_top_path_ids'])
    assert abs(best-actual['dp']['max_score'])<2e-16
    byid={r['id']:r for r in actual['catalog']}
    for start in m.carrier.candidates(length):
        old=m.carrier.score_candidate({'pilot_fft':X[start['g']].tolist()},KEY,start)
        assert abs(old['score']-byid[m.path_id(start['b'],'ZERO_EDIT',None)]['score'])<2e-16


def test_near_numerical_ties_final_catalog_filter():
    rng=np.random.default_rng(173);pn=np.asarray(m.carrier.pilot_layout(KEY)[1])
    carrier=np.asarray([1.0]*32+[-1.0]*32)
    X={g:pn[None,:]*(carrier[None,:]+rng.normal(size=(31,1))*2e-11) for g in range(4)}
    a=m.analyze(X,129,KEY);scores=brute_scores(X,129);best=max(scores.values())
    expected={pid for pid,s in scores.items() if best-s<=1e-12}
    assert 1<len(expected)<3233
    assert a['summary']['status']=='COMPLETE'
    assert set(a['summary']['top_path_ids'])==expected==set(a['dp']['final_numerical_top_path_ids'])
    assert len(a['dp']['locally_tied_backtrace_path_ids'])>=len(expected)
    assert a['summary']['accepted_payload'] is False


def test_constant_code_and_real_same_sign_ambiguity_no_zero_preference(monkeypatch):
    pn=np.asarray(m.carrier.pilot_layout(KEY)[1]);signs=np.asarray([m.carrier.temporal_sign(t) for t in range(1,46)])
    a=m.analyze({0:signs[:,None]*pn[None,:]},181,KEY)
    assert a['summary']['max_score']==pytest.approx(1.0)
    assert a['summary']['any_top_class_contains_zero_edit'] is True
    assert a['summary']['all_top_paths_require_event'] is False
    assert a['summary']['canonical_event_type']=='REPEAT'
    assert len(a['summary']['top_path_ids'])==3 and not a['summary']['unique_sync_claim']
    monkeypatch.setattr(m.carrier,'temporal_sign',lambda t:1)
    a=m.analyze({g:np.tile(pn,(31,1)) for g in range(4)},129,KEY)
    assert len(a['equivalence_classes'])==4 and len(a['summary']['top_path_ids'])==3233
    assert len(a['dp']['final_numerical_top_path_ids'])==3233
    assert a['summary']['canonical_event_type']=='REPEAT' and not a['summary']['unique_sync_claim']


@pytest.mark.parametrize('event,i',[('REPEAT',2),('REPEAT',31),('SKIP',2),('SKIP',31)])
def test_edit_boundary_backtrace_contains_generating_path(event,i):
    row=next(r for r in m.catalog(129) if r['id']==m.path_id(52,event,i))
    pn=np.asarray(m.carrier.pilot_layout(KEY)[1]);signs=np.asarray([m.carrier.temporal_sign(t) for t in m.path_taus(row)])
    X={g:np.zeros((31,64)) for g in range(4)};X[0]=signs[:,None]*pn[None,:]
    a=m.analyze(X,129,KEY)
    assert row['id'] in a['dp']['final_numerical_top_path_ids']
    assert a['summary']['max_score']==pytest.approx(1.0)


def test_zero_missing_nonfinite_and_api_whitelist():
    X={g:np.zeros((31,64)) for g in range(4)}
    a=m.analyze(X,129,KEY)
    assert a['summary']['status']=='COMPLETE' and a['summary']['synchronization_status']=='NO_ENERGY'
    assert len(a['summary']['top_path_ids'])==3233
    del X[2];X[3][0,0]=np.nan
    a=m.analyze(X,129,KEY)
    assert len(a['catalog'])==3233 and a['summary']['status']=='INCOMPLETE'
    assert a['summary']['canonical_path_id'] is None and not a['summary']['top_path_ids']
    assert a['phases'][2]['status']=='MISSING_PHASE' and a['phases'][3]['status']=='INVALID_PHASE'
    assert set(inspect.signature(m.analyze).parameters)=={'pilot_by_phase','received_frame_count','key','public'}
    with pytest.raises(ValueError):m.analyze({'arm':'PILOT_MULTI'},129,KEY)
    with pytest.raises(TypeError):m.analyze({},129,KEY,truth='OKOK')


def artifact(length,g,key):
    n=w.window_row_count(length,g);coords,pn=m.carrier.pilot_layout(key)
    return dict(schema=w.WINDOW_SCHEMA,phase_g=g,received_frame_count=length,
      used_received_frame_interval=list(m.carrier.phase_slice(length,g)),public_key_sha256=hashlib.sha256(key.encode()).hexdigest(),
      public_layout_sha256=m.carrier.digest(dict(payload_coordinates=m.carrier.payload_coordinates(key),pilot_coordinates=coords,pilot_pn=pn)),
      nominal_stride4=4,grid_meaning='nominal newly supplied RGB frames, not VAE receptive field',first_latent_excluded=True,
      primary_regular_count=45 if length==181 else 31,window_count=n,payload_bit_count=32,pilot_coefficient_count=64,
      transform='fft2_ortho_real_float32',statistic_reduction='float64',time_dependent_payload='disabled_unverified',
      windows=[dict(phase_g=g,observed_regular_j=j,nominal_stride4=4,nominal_received_new_frame_interval=[g+4*j-3,g+4*j+1],
        pilot_fft=[0.0]*64,payload_stats=[dict(sum=0.0,sumsq=0.0,positive_count=0,negative_count=0,zero_count=30,support_count=30,first_bit=0) for _ in range(32)]) for j in range(1,n+1)])


def test_extra_row_pathname_and_metadata_blindness(tmp_path):
    data=artifact(129,0,KEY);other=copy.deepcopy(data);other['windows'][31]['pilot_fft']=[999.0]*64
    p=tmp_path/'arbitrary-truth-label.json';q=tmp_path/'renamed-no-truth.json'
    p.write_text(json.dumps(data));q.write_text(json.dumps(other))
    a=run.validate_artifact(json.loads(p.read_text()),129,0,KEY)
    b=run.validate_artifact(json.loads(q.read_text()),129,0,KEY)
    assert a==b and len(a)==31
    data['source_b']=5
    with pytest.raises(ValueError):run.validate_artifact(data,129,0,KEY)


def test_runner_fixed_failures_raw_then_posthoc_and_normal_rejection(tmp_path,monkeypatch):
    cfg=run.load_config();source=tmp_path/'source';source.mkdir();saved={'windows':{}}
    for item in run.roster(cfg):
        for g in m.carrier.phases(item['length']):
            slot=f"{item['arm']}/{item['view']}/{g}/{item['key_id']}"
            path=source/item['arm']/f"{item['view']}.phase{g}.{item['key_id']}.windows.json"
            data=artifact(item['length'],g,cfg['keys'][item['key_id']]);run.dump(path,data)
            saved['windows'][slot]=dict(status='SAVED',schema=w.WINDOW_SCHEMA,row_count=len(data['windows']),sha256=run.sha(path))
    # One missing phase file; a second file has nonfinite data but matching source hash.
    (source/'OFF'/'CROP4_129.phase2.CORRECT.windows.json').unlink()
    bad=source/'PILOT_MULTI'/'CROP7_129.phase1.WRONG.windows.json'
    value=json.loads(bad.read_text());value['windows'][0]['pilot_fft'][0]=float('nan');bad.write_text(json.dumps(value))
    saved['windows']['PILOT_MULTI/CROP7_129/1/WRONG']['sha256']=run.sha(bad)
    run.dump(source/'result.json',saved);cfg=copy.deepcopy(cfg);cfg['source_result_sha256']=run.sha(source/'result.json')
    original_evaluate=run.evaluate
    def guarded_evaluate(output,result,cfg):
        assert result['stage']=='RAW_COMPLETE_BEFORE_TRUTH'
        assert len(result['raw_sha256_before_truth'])==30 and len(list((output/'raw').glob('*.json.gz')))==30
        with monkeypatch.context() as mp:
            mp.setattr(m,'analyze',lambda *a,**k:pytest.fail('posthoc cannot rerun path search'))
            return original_evaluate(output,result,cfg)
    monkeypatch.setattr(run,'evaluate',guarded_evaluate)
    out=tmp_path/'output';result=run.execute(source,out,cfg)
    assert result['status']=='INCOMPLETE' and result['stage']=='FINISHED'
    assert len(result['inputs'])==102 and len(result['conditions'])==30 and len(result['evaluations'])==60
    assert result['counts']==dict(inputs_verified=100,conditions_complete=28,evaluations_complete=56,catalog_paths=78126,
        scored_paths=77862-2*13*61,scorable_paths=77862,zero_edit_paths=1278,structurally_excluded=264,equivalence_classes=12024)
    assert result['raw_sha256_before_truth']==result['raw_sha256_after_truth']
    assert all(r['status']!='PENDING' for table in ('inputs','conditions','evaluations') for r in result[table].values())
    assert all(not r['accepted_payload'] and r['bit_errors'] is None for r in result['evaluations'].values())
    assert all(v==0 for v in result['calls'].values())
    for ref in result['conditions'].values():
        raw=run.read_json(out/ref['raw_path'])
        if ref['status']=='COMPLETE':assert raw['summary']['synchronization_status']=='NO_ENERGY'
        else:assert raw['summary']['canonical_path_id'] is None


def test_entire_source_unavailable_keeps_fixed_catalogs(tmp_path):
    result=run.execute(tmp_path/'absent-source',tmp_path/'out',run.load_config())
    assert result['status']=='INCOMPLETE' and result['stage']=='FINISHED'
    assert len(result['inputs'])==102 and len(result['conditions'])==30 and len(result['evaluations'])==60
    assert result['counts']==dict(inputs_verified=0,conditions_complete=0,evaluations_complete=0,catalog_paths=78126,
        scored_paths=0,scorable_paths=77862,zero_edit_paths=1278,structurally_excluded=264,equivalence_classes=12024)
    assert result['failures'][0]['stage']=='SOURCE_RESULT'
    assert all(r['status']=='FAILED' for r in result['inputs'].values())
    assert all(r['status']=='MISSING' for r in result['evaluations'].values())


def test_dp_enumeration_disagreement_fails_closed(monkeypatch):
    original=m.dynamic_program
    def faulty(*args,**kwargs):
        dp=original(*args,**kwargs);dp['locally_tied_backtrace_path_ids']=[];return dp
    monkeypatch.setattr(m,'dynamic_program',faulty)
    a=m.analyze({0:np.ones((45,64))},181,KEY)
    assert a['summary']['status']=='INCOMPLETE'
    assert a['summary']['synchronization_status']=='DP_ENUMERATION_DISAGREEMENT'
    assert a['summary']['canonical_path_id'] is None and not a['summary']['accepted_payload']
