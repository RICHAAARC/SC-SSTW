"""CPU fake model plus native UniPC; no real model or historical scoring."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest,torch
from main.tube_state import projection_margin as carrier,terminal_guidance as method
from runtime.wan import tube_retention as rt,tube_terminal_guidance as prior
from experiments.wan_state_clock import flow_tube_retention_run as run
import test_flow_tube_state_guidance as fixture
from test_flow_tube_state_guidance import native,book,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def test_matched_native_response_not_request_and_shared_history(native,book,monkeypatch):
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());z=torch.zeros(carrier.SHAPE);count=lambda *a:None
    nodes,snapshots,_=rt.reference(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,count)
    sourcefinger=rt.fingerprint(vars(snapshots[44]));sigma=float(snapshots[44].sigmas[44])
    last,_,anchor,_=rt.matched_update(snapshots[49],nodes[49]['z'],nodes[49]['v'],nodes[50]['z'],book,0,49,count)
    R=anchor['actual_D']['support_rms']
    actual,s,row,arrays=rt.matched_update(snapshots[44],nodes[44]['z'],nodes[44]['v'],nodes[45]['z'],book,0,44,count,R)
    expected=copy.deepcopy(snapshots[44]).step(nodes[44]['v']-arrays['applied_u']/sigma,snapshots[44].timesteps[44],nodes[44]['z'],return_dict=False)[0]
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert row['matching_relative_error']<2e-4 and not torch.allclose(arrays['actual_D'],arrays['applied_u'],rtol=1e-4,atol=1e-7)
    assert rt.fingerprint(vars(snapshots[44]))==sourcefinger and not actual.requires_grad
    terminal,stages=rt.continue_single(pipe,actual,s,nodes,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,44,book,count)
    assert stages['immediate_next']['node']==45 and stages['one_free_step']['node']==46
    assert not any(pipe.transformer.flags)
    monkeypatch.setattr(rt.method,'correction',lambda z,b,m:(np.zeros_like(z),z,{}))
    with pytest.raises(ValueError,match='unit native response'):
        rt.matched_update(snapshots[44],nodes[44]['z'],nodes[44]['v'],nodes[45]['z'],book,0,44,count,R)


def test_fixed_tree_last_equivalence_cost_and_media(monkeypatch,native,tmp_path,book):
    monkeypatch.setattr(fixture,'run',run);path=tmp_path/'case';g=fixture.generate_fixture(monkeypatch,native,path)
    assert len(g['videos'])==7 and all(g['actual_calls'][k+'_completed']==run.PLAN[k] for k in ('transformer','scheduler_step','unit_response_probe_step'))
    assert g['calls_by_path']['OFF_reference']['transformer_completed']==100 and g['calls_by_path']['OFF_reference']['scheduler_step_completed']==50
    pipe=SimpleNamespace(transformer=Model(),scheduler=native());initial=torch.zeros(carrier.SHAPE)
    z,v,s,_=prior.prepare_last(pipe,initial,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,lambda *a:None)
    off,_=prior.final_step(s,z,v,lambda *a:None)
    torch.testing.assert_close(torch.load(path/'OFF_terminal.pt',weights_only=True),off,rtol=0,atol=0)
    for m,suffix in enumerate(('A','B')):
        u,_,_=method.correction((z-float(s.sigmas[49])*v).numpy(),book,m)
        expected,_=prior.final_step(s,z,v,lambda *a:None,u=torch.from_numpy(u))
        torch.testing.assert_close(torch.load(path/('T49_'+suffix+'_terminal.pt'),weights_only=True),expected,rtol=0,atol=0)
        R=g['videos']['T49_'+suffix]['control']['actual_D']['support_rms']
        for t in (44,46):
            item=g['videos'][f'T{t}_{suffix}'];assert item['control']['target_D_support_rms']==R
            assert item['control']['matching_relative_error']<2e-4
            assert item['short_stages']['one_free_step']['node']==t+2
    reads,lengths,blind=fixture.media_fixture(monkeypatch,path);r=run.media_case(run.CASES[0],path)
    assert r['status']=='EXECUTION_COMPLETE',r['failures']
    assert lengths==[181,177,177,177]*7 and len(blind)==7
    assert all(r['actual_calls'][k+'_completed']==v for k,v in run.PLAN.items())


def test_fixed_failure_roster_and_no_truth_selected_timing(monkeypatch,tmp_path):
    manifest=run.load(run.MANIFEST);run.validate(manifest)
    assert [v['id'] for v in manifest['development']]==list(run.CASES)
    assert manifest['write_indices']==[44,46,49] and manifest['future_holdout']['selection_rule'] is None
    monkeypatch.setattr(run.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=1))
    r=run.run_all(tmp_path/'full')
    assert r['video_denominator']==14 and r['receiver_encode_denominator']==56
    assert sum(len(v['observations']) for c in r['cases'].values() for v in c['videos'].values())==56
    assert len(r['mechanism_summary'])==14 and all(x['missing_or_failed']==4 for g in r['recovery_summary'].values() for x in g.values())
    assert all(x['current_clean_gain'] is None for x in r['mechanism_summary'])


def test_prespecified_sign_diagnostics_retain_missing_zeros_and_exclude_49():
    rows=[dict(case=run.CASES[0],arm='T44_A',current_clean_gain=1.,short_clean_gain=0.,terminal_projection_gain=-1.,receiver={'local_state':{'gain_vs_OFF':0.}}),
          dict(case=run.CASES[0],arm='T49_A',current_clean_gain=1.,short_clean_gain=1.,terminal_projection_gain=1.)]
    r=run.predictive_diagnostics(rows)['comparisons']
    assert r['current_clean_gain__terminal_projection_gain']['opposite_nonzero_sign']==1
    assert r['short_clean_gain__MP4_local_state_gain']['both_zero']==1
    assert all(v['fixed_pair_denominator']==8 and v['valid']==1 and v['missing_or_nonfinite']==7 for v in r.values())


@pytest.mark.parametrize('partial_arm',['OFF','T44_A'])
def test_partial_media_marked_or_off_is_missing_for_prediction(partial_arm):
    ranks={'local_state':{'best_by_message':{'0':{'score':.7},'1':{'score':.2}}}}
    videos={'OFF':{'status':'COMPLETE','rankings':ranks},'T44_A':{'status':'COMPLETE','rankings':ranks,
        'control':{'before_clean':{'nominal_score_A_minus_B':0.},'controlled_clean':{'nominal_score_A_minus_B':.2}},
        'terminal_projection':{'nominal_score_A_minus_B':.3},'OFF_terminal_projection':{'nominal_score_A_minus_B':0.}}}
    videos[partial_arm]['status']='PARTIAL_OR_FAILED'
    rows=run.mechanism_summary({run.CASES[0]:{'videos':videos}})
    row=next(r for r in rows if r['case']==run.CASES[0] and r['arm']=='T44_A')
    assert row['receiver']['local_state']['gain_vs_OFF'] is None
    assert row['terminal_projection_gain']==.3
    comparisons=run.predictive_diagnostics(rows)['comparisons']
    assert comparisons['current_clean_gain__MP4_local_state_gain']['valid']==0
    assert comparisons['current_clean_gain__MP4_local_state_gain']['missing_or_nonfinite']==8
    assert comparisons['current_clean_gain__terminal_projection_gain']['valid']==1
