"""Fixed holdout orchestration checks using synthetic generation/media only."""
import copy,json
from types import SimpleNamespace
import pytest
from experiments.wan_state_clock import flow_tube_state_holdout_run as run
import test_flow_tube_state_guidance as fixture
from test_flow_tube_state_guidance import native
pytestmark=pytest.mark.unit


def test_fixed_holdout_no_overlap_and_original_method():
    m=run.load(run.MANIFEST);run.validate(m);old=run.load(run.MANIFEST.with_name('velocity_calibration.json'))
    assert m['holdout']==old['holdout']
    assert not {c['prompt'] for c in m['holdout']}&{c['prompt'] for c in old['development']}
    assert not {c['seed'] for c in m['holdout']}&{c['seed'] for c in old['development']}
    assert run.PLAN==dict(transformer=100,scheduler_step=52,vae_decode=3,mp4_save=3,vae_encode=12)


def test_three_arm_media_and_cpu_direct_only(monkeypatch,native,tmp_path):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case';g=fixture.generate_fixture(monkeypatch,native,path)
    assert set(g['videos'])==set(run.ARMS) and all(v['pass'] for v in g['equivalence'].values())
    assert not list(path.glob('TERMINAL*'))
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path);r=run.media_case(run.CASES[0],path)
    assert r['status']=='EXECUTION_COMPLETE',r['failures']
    assert lengths==[181,177,177,177]*3 and len(blind)==3
    assert all(r['actual_calls'][k+'_completed']==v for k,v in run.PLAN.items())


def test_message_margin_null_off_and_coverage():
    rank={'best_by_message':{'0':{'score':.7},'1':{'score':.2}},'best':{'message':0,'matched_supports':1760},'message_unique':True,'top_ties':[{'message':0}]}
    case={'videos':{a:{'status':'COMPLETE','observations':{str(g):{'status':'COMPLETE'} for g in range(4)},'rankings':{mode:copy.deepcopy(rank) for mode in run.MODES}} for a in run.ARMS}}
    rows=run.message_summary({run.CASES[0]:case});a=next(r for r in rows if r['arm']=='LAST_A');b=next(r for r in rows if r['arm']=='LAST_B');off=rows[0]
    assert a['best_correct_minus_other_reporting_only']==pytest.approx(.5) and b['best_correct_minus_other_reporting_only']==pytest.approx(-.5)
    assert off['truth_reporting_only'] is None and off['best_correct_minus_other_reporting_only'] is None and off['reference_score_0_minus_1']==pytest.approx(.5)
    assert a['completed_origins']==4 and a['best_matched_supports']==1760
    case['videos']['LAST_A']['rankings'][run.MODES[0]]['best_by_message']['1']=None
    assert next(r for r in run.message_summary({run.CASES[0]:case}) if r['arm']=='LAST_A')['best_correct_minus_other_reporting_only'] is None
    assert len(rows)==2*3*5


def test_full_missing_fixed_six_twentyfour(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:(calls.append(a) or SimpleNamespace(returncode=1)))
    r=run.run_all(tmp_path/'full');assert len(calls)==4 and r['video_denominator']==6 and r['receiver_encode_denominator']==24
    assert sum(len(v['observations']) for c in r['cases'].values() for v in c['videos'].values())==24
    assert all(v['missing_or_failed']==4 for v in r['recovery_summary']['LAST'].values())
    assert r['OFF_summary']['denominator']==2 and len(r['message_and_coverage_summary'])==30


def test_holdout_partial_off_media_not_reused(monkeypatch,native,tmp_path):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case';fixture.generate_fixture(monkeypatch,native,path)
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path,fail='partial_off');r=run.media_case(run.CASES[0],path)
    assert r['status']=='WITH_RETAINED_FAILURES' and 'OFF' not in reads
    assert all(r['videos'][a]['status']=='COMPLETE' for a in ('LAST_A','LAST_B'))
    assert all(r['videos'][a]['saved_quality_vs_off']['status']=='MISSING_OFF_REFERENCE' for a in ('LAST_A','LAST_B'))
    assert sum(len(v['observations']) for v in r['videos'].values())==12
