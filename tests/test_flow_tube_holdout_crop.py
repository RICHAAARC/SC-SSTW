"""Synthetic orchestration delta only; unchanged crop math tested by original suite."""
from types import SimpleNamespace
import subprocess
import pytest
from experiments.wan_state_clock import flow_tube_holdout_crop_run as run
from experiments.wan_state_clock import flow_tube_crop_run as original
import test_flow_tube_crop as fixture
from test_flow_tube_crop import book,detection_factory
pytestmark=pytest.mark.unit


def test_frozen_source_and_protocol():
    run.validate_manifest(run.load(run.MANIFEST))
    assert run.SOURCE_COMMIT=='0797064cc5f1166170ed5c229e18defd681d8185'
    assert run.SOURCE_ROOT.name=='flow_tube_state_holdout_20260920T033109891109Z'
    assert run.CASES==('holdout_p0_s0','holdout_p1_s0')
    assert not set(run.CASES)&set(original.CASES)
    assert run.analysis is original.analysis and run.MODES==original.MODES and run.PLAN==original.PLAN
    assert run.STARTS==(0,4,5) and run.FRAMES==129


def test_complete_case_saved_crop_and_blind_receiver(monkeypatch,tmp_path,book,detection_factory):
    monkeypatch.setattr(fixture,'run',run)
    fixture.test_fake_full_case_blind_api_and_real_rgb_origins(tmp_path,monkeypatch,book,detection_factory)


def test_fixed_eighteen_seventytwo_missing_sources(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:(calls.append(a) or SimpleNamespace(returncode=1)))
    result=run.run_all(tmp_path/'out',tmp_path/run.SOURCE_RUN)
    assert len(calls)==2 and result['source_video_denominator']==6
    assert result['fragment_denominator']==18 and result['marked_denominator']==12 and result['receiver_encode_denominator']==72
    assert result['fixed_calls']['source_mp4_read']==6 and result['fixed_calls']['vae_encode']==72
    assert result['fixed_calls']['transformer']==0
    assert sum(len(v['observations']) for c in result['cases'].values() for v in c['fragments'].values())==72
    assert len(result['compact_summary'])==24
    assert all(v['denominator']==4 and v['missing_or_failed']==4 and v['OFF_denominator']==2 for v in result['compact_summary'])
    missing=run.run_case(run.CASES[0],tmp_path/'missing',tmp_path/run.SOURCE_RUN)
    assert missing['status']=='WITH_RETAINED_FAILURES' and len(missing['fragments'])==9
    assert all(v==0 for v in missing['actual_calls'].values())
