"""Frozen roster, unchanged candidate and fixed six-video failure accounting."""
import copy
import json
from types import SimpleNamespace
import pytest
from experiments.wan_state_clock import inversion_state_holdout_run as runner
from experiments.wan_state_clock import inversion_state_run as state

pytestmark=pytest.mark.unit


def test_prelocked_roster_and_frozen_method(tmp_path,monkeypatch):
    manifest=runner.validate()
    assert [c['seed'] for c in manifest['holdout']]==[20261001,20261002]
    development=state.base.load(state.MANIFEST)
    for field in ('base_config','payloads','carrier','receiver'):
        assert manifest[field]==development[field]
    modified=copy.deepcopy(manifest);modified['carrier']['observer_gain']=.7
    path=tmp_path/'bad.json';path.write_text(json.dumps(modified));monkeypatch.setattr(runner,'MANIFEST',path)
    with pytest.raises(ValueError,match='may not change'):runner.validate()


def test_holdout_failure_denominators_and_no_extra_cases(tmp_path,monkeypatch):
    launched=[]
    def fail(command,**kwargs):launched.append(command);return SimpleNamespace(returncode=1)
    monkeypatch.setattr(runner.state.subprocess,'run',fail)
    result=runner.run_all(tmp_path/'out')
    assert len(launched)==2 and all(runner.MODULE in args for args in launched)
    assert result['case_denominator']==2 and result['video_denominator']==6 and result['core_window_denominator']==66
    assert result['fixed_calls']=={
        'generation':{'transformer':600,'scheduler_step':300},
        'media':{'vae_decode':6,'vae_encode':6,'mp4_save':6},
        'inversion':{'transformer':600,'inverse_update':300}}
    summary=result['state_summary']
    assert summary['marked_video_denominator']==4 and summary['off_video_denominator']==2
    assert summary['core_window_denominator']==44 and summary['component_denominator']==88
    assert summary['unmeasured_core_windows']==44 and summary['unmeasured_components']==88
    assert len(summary['per_video'])==6
    for case in result['cases'].values():
        for arm in case['videos'].values():
            assert len(arm['decoded']['core'])==11 and len(arm['decoded']['boundaries'])==2
            assert len(arm['decoded']['per_time'])==46
