"""CPU objective/restore-path checks; no pretrained execution."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import clipped_margin as objective,velocity_coefficients as method,state_clock
from experiments.wan_state_clock import clipped_margin_run as runner,velocity_direction_run as legacy
from test_velocity_direction import setup,scheduler
from test_zero_path import FrozenFake,metadata
from runtime.wan.velocity_direction import detached_scheduler,response_coefficients

pytestmark=pytest.mark.unit


def test_gradient_saturation_cancellation_and_finite_difference():
    p=torch.tensor([-2.,-1.,-.3,0.,.4,1.,2.,-1.,1.],dtype=torch.float64,requires_grad=True)
    codes=torch.tensor([[1.]*9,[-1.]*7+[1.,1.]],dtype=torch.float64)
    loss=objective.projection_loss(p,codes,0)
    grad=torch.autograd.grad(loss,p)[0]
    torch.testing.assert_close(grad,torch.tensor([0.,-2/9,-2/9,-2/9,-2/9,-2/9,0.,0.,0.],dtype=torch.float64))
    eps=1e-6
    for i in (0,2,3,4,6):
        e=torch.zeros_like(p);e[i]=eps
        fd=(objective.projection_loss(p+e,codes,0)-objective.projection_loss(p-e,codes,0))/(2*eps)
        assert float(fd)==pytest.approx(float(grad[i]),abs=1e-9)
    for i,inward in ((1,1),(5,-1)):
        e=torch.zeros_like(p);e[i]=inward*eps
        inner=(objective.projection_loss(p+e,codes,0)-loss)/(inward*eps)
        outer=(loss-objective.projection_loss(p-e,codes,0))/(inward*eps)
        assert float(inner)==pytest.approx(float(grad[i]),abs=1e-9)
        assert float(outer)==pytest.approx(0.,abs=1e-9)
    assert objective.projection_loss(p,codes,1)==-loss
    with pytest.raises(ValueError,match='zero coefficient gradient'):
        method.direction_and_amplitude(torch.zeros(3,1760),torch.ones(1760,1),[1]*3,[1]*3,.1)


def test_exact_receiver_matched_equivalence(setup):
    book,directions,codes,s,z=setup
    for message in (0,1):
        score=[state_clock.emissions({0:z.numpy()},book,n,0,1,1,0)['scores'] for n in range(11)]
        margin=np.mean([v[message]-v[1-message] for v in score])
        assert float(objective.terminal_loss(z,directions,codes,message))==pytest.approx(-margin,abs=1e-12)
    same=torch.stack([codes[0],codes[0]])
    p=method.projections(z,directions).requires_grad_()
    assert torch.count_nonzero(torch.autograd.grad(objective.projection_loss(p,same,0),p)[0])==0


@pytest.mark.parametrize('zero_gradient',[False,True])
def test_restored_path_calls_budget_and_no_prefix(setup,tmp_path,monkeypatch,zero_gradient):
    book,d,c,s,z=setup
    if zero_gradient:book['codes'][1]=book['codes'][0]
    manifest=json.loads(Path('experiments/wan_state_clock/configs/velocity_calibration.json').read_text())
    config=copy.deepcopy(manifest['base_config'])
    pipe=SimpleNamespace(transformer=FrozenFake(),scheduler=scheduler(),vae=None)
    old={'R':.02,'responses':response_coefficients(s,lambda *args:None),
        'scheduler':metadata(s),'precision':{'transformer_input':'torch.bfloat16'}}
    monkeypatch.setattr(legacy,'prepare_generation',lambda *a,**kw:(pipe,torch.full_like(z,999),
        torch.tensor(1.).bfloat16(),torch.tensor(-1.).bfloat16(),torch.bfloat16))
    def forbidden(*a,**kw):raise AssertionError('prefix/probe rerun forbidden')
    monkeypatch.setattr(legacy,'continue_steps',forbidden)
    monkeypatch.setattr(legacy,'response_coefficients',forbidden)
    out=tmp_path/'new'
    result=legacy.run(config,out,strengths=(.1,.3,1.),_restored={'record':old,'book':book,
        'payload':{'latent':z,'scheduler':detached_scheduler(s)},'original_run':'read-only original'})
    if zero_gradient:
        assert result['status']=='EXECUTED_WITH_RETAINED_FAILURES'
        assert all(result['conditions'][n]['status']=='MISSING_DIRECTION' for n in runner.saved.FLOW_NAMES)
        assert result['actual_calls']['transformer_completed']==24
        assert result['actual_calls']['backward_completed']==2
        assert all(result['conditions']['ZERO_'+m]['coefficient_gradient_l2']==0 for m in ('A','B'))
        assert len(result['conditions'])==8 and len(result['failures'])==2
        return
    assert result['status']=='EXECUTED_REQUIRES_DIRECTION_REVIEW',result['failures']
    assert len(result['conditions'])==8 and result['R']==old['R']
    for key,n in [('transformer',96),('scheduler_step',48),('shadow_step',48),('backward',2),('response_probe_step',0)]:
        assert result['actual_calls'][key+'_completed']==n
    for name,row in result['conditions'].items():
        m=0 if name.endswith('A') else 1
        scores=row['endpoint']['clipped_matched_score_by_message']
        assert row['loss']==pytest.approx(scores[1-m]-scores[m],abs=1e-12)
        assert row['projection_gradient_support']['support_count']==1760
        assert [v['index'] for v in row['steps']]==[44,45,46]
    assert not (out/'pre_intervention_state.pt').exists()
    assert not (out/'terminal_reference_0.pt').exists()


def test_report_retains_missing_roster_and_no_holdout(tmp_path):
    source=tmp_path/runner.saved.ORIGINAL_RUN;source.mkdir()
    zero=tmp_path/runner.ZERO_RUN;zero.mkdir()
    output=tmp_path/'new';output.mkdir()
    report=runner.report(source,zero,output)
    assert report['flow_denominator_per_method']==24
    assert len(report['cases'])==4
    assert all(len(c['flows'])==6 for c in report['cases'].values())
    assert report['holdout']=='NOT_RUN_NOT_AUTHORIZED'
    with pytest.raises(ValueError):runner.inputs(source,zero,source/'overwrite')
    with pytest.raises(ValueError):runner.inputs(source,zero,zero/'overwrite')


def test_effective_metadata_preserves_method_inputs(tmp_path):
    original={'generation':{'seed':123,'role':'original'},'model':{'id':'fixed'},'key_utf8':'key',
        'source_snapshot':'old','output_drive_parent':'old','artifact_paths':{'source_archive':'old.zip'}}
    before=copy.deepcopy(original)
    updated=runner.effective_config(original,tmp_path/'old',tmp_path/'new','dev_p0_s0','terminal')
    assert original==before
    assert all(updated[k]==original[k] for k in ('generation','model','key_utf8'))
    assert updated['source_snapshot']!='old' and 'source_archive' not in updated['artifact_paths']
    assert updated['artifact_paths']['launcher_log']==str(tmp_path/'new/terminal_dev_p0_s0.log')


def test_historical_control_selection_is_explicitly_provisional(tmp_path,monkeypatch):
    source=tmp_path/runner.saved.ORIGINAL_RUN;source.mkdir()
    zero=tmp_path/runner.ZERO_RUN;zero.mkdir()
    output=tmp_path/'new';output.mkdir()
    manifest={'development':[{'id':c} for c in runner.saved.CASES]}
    (output/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(runner,'read_cases',lambda *a,**k:{c:None for c in runner.saved.CASES})
    monkeypatch.setattr(runner,'select_terminal',lambda *a:{'selected_indices':[]})
    monkeypatch.setattr(runner,'select_media',lambda *a:{'status':'SELECTED','rho':.3})
    monkeypatch.setattr(runner,'report',lambda *a:{'historical_control_comparability':{'status':'NONMATCHING'},
                                               'paired_method_comparison_valid':False})
    result=runner.run_stage(source,zero,output,'media')
    assert result['status']=='PROVISIONAL_SELECTED_HISTORICAL_CONTROLS'
    assert result['raw_selector_result']=={'status':'SELECTED','rho':.3}
    assert result['paired_method_comparison_valid'] is False
