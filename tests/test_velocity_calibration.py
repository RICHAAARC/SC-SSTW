import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import calibration_run as stages, calibration_selection as selection
from test_velocity_direction import setup,Model,scheduler

pytestmark=pytest.mark.unit


def manifest():return json.loads(Path('experiments/wan_state_clock/configs/velocity_calibration.json').read_text())


def terminal_results(m):
    return {case['id']:{'conditions':dict(
        {f'ZERO_{s}':{'status':'COMPLETE','loss':1.} for s in ('A','B')},
        **{selection.flow_name(i,s):{'status':'COMPLETE','loss':1-.01*i,
        'correct_minus_competitor_margin':-.2+.1*i,'correct_code_signed_projection_gain':.01*i,
        'steps':[{'step_within_budget':True,'sum_within_budget':True,'response_valid':True} for _ in range(3)]}
        for i in range(3) for s in ('A','B')})} for case in m['development']}


def media_results(m):
    return {case['id']:{'videos':{name:{'status':'COMPLETE','five_methods_unique_correct':True,
        'saved_quality_vs_off':{'rgb_mse':1.,'residual_temporal_mse':1.}}
        for name in ['OFF','TERMINAL_A','TERMINAL_B']+[selection.flow_name(i,s) for i in range(3) for s in ('A','B')]}}
        for case in m['development']}


def test_fixed_roster_selection_denominators_and_no_holdout_feedback():
    m=manifest();stages.validate(m)
    assert stages.call_plan(m)['development']['transformer']==736
    assert stages.call_plan(m)['holdout_if_selected']['transformer']==272
    with pytest.raises(ValueError):stages.validate(dict(m,development=[]))
    values=terminal_results(m);result=selection.select_terminal(m,values)
    assert result['selected_indices']==[2,1]  # Negative margin allowed for r1.
    assert all(row['denominator']==8 and len(row['rows'])==8 for row in result['candidates'])
    assert len(result['strength_curves'])==8
    values[m['development'][0]['id']]['conditions'][selection.flow_name(2,'A')]['status']='FAILED_AFTER_FORWARD'
    result=selection.select_terminal(m,values)
    assert result['selected_indices']==[1,0] and len(result['candidates'][2]['rows'])==8
    media=media_results(m)
    choice=selection.select_media(m,result,media)
    assert choice['status']=='SELECTED' and choice['rho']==.1
    media['holdout_p0_s0']={'videos':{'bad':'not allowed to affect selection'}}
    assert selection.select_media(m,result,media)==choice
    for case in m['development']:
        for s in ('A','B'):
            media[case['id']]['videos'][selection.flow_name(0,s)]['saved_quality_vs_off']['rgb_mse']=1.50001
    assert selection.select_media(m,result,media)['rho']==.3
    media[m['development'][0]['id']]['videos'][selection.flow_name(1,'A')]['status']='MISSING_TERMINAL'
    assert selection.select_media(m,result,media)['status']=='NO_SELECTION'
    assert selection.quality_ratio(0,0)==0 and selection.quality_ratio(1,0) is None


def test_stages_no_selection_and_holdout_single_rho_mapping(monkeypatch,tmp_path):
    from experiments.wan_state_clock import velocity_direction_run,calibration_media
    m=manifest();root=tmp_path
    (root/'selection.json').write_text(json.dumps({'status':'NO_SELECTION','rho':None}))
    def forbidden(*args,**kwargs):raise AssertionError('no holdout execution allowed')
    monkeypatch.setattr(stages,'run_children',forbidden)
    result=stages.run_stage(m,root/'manifest.json',root,'holdout')
    assert result['status']=='NOT_RUN_NO_SELECTION' and len(result['cases'])==2
    assert all(row['status']=='NOT_RUN_NO_SELECTION' for row in result['cases'].values())
    (root/'selection.json').write_text(json.dumps({'status':'SELECTED','rho':.3}))
    seen={}
    def terminal(config,output,*,strengths):seen.update(config=config,output=output,strengths=strengths);return {}
    monkeypatch.setattr(velocity_direction_run,'run',terminal)
    case=m['holdout'][0]
    stages.child(m,root,'holdout',case['id'],False)
    assert seen['strengths']==[.3]
    assert seen['config']['generation']['prompt']==case['prompt']
    assert seen['config']['output_drive_parent']==str(root/'holdout')
    assert seen['config']['generation']['role']=='calibration_holdout_zero_gradient_fixed_strengths'
    def media(config,terminals,output):seen['terminals']=terminals;return {}
    monkeypatch.setattr(calibration_media,'run',media)
    stages.child(m,root,'holdout',case['id'],True)
    assert tuple(seen['terminals'])==('OFF','TERMINAL_A','TERMINAL_B','FLOW_r00_A','FLOW_r00_B')
    (root/'terminal_selection.json').write_text(json.dumps({'selected_indices':[2,1]}))
    stages.child(m,root,'dev',m['development'][0]['id'],True)
    assert tuple(seen['terminals'])==('OFF','TERMINAL_A','TERMINAL_B','FLOW_r02_A','FLOW_r02_B','FLOW_r01_A','FLOW_r01_B')


def test_calibration_q_computed_once_and_all_strengths_retained(monkeypatch,tmp_path,setup):
    from experiments.wan_state_clock import velocity_direction_run as runner
    def prepare(config,*,load_vae):
        assert load_vae is False
        return SimpleNamespace(transformer=Model(),scheduler=scheduler(),vae=None),torch.zeros((1,16,46,40,64)),torch.tensor(1.),torch.tensor(-1.),torch.float32
    monkeypatch.setattr(runner,'prepare_generation',prepare)
    original=runner.method.direction_and_amplitude;calls=[]
    def measured(*args):calls.append(True);return original(*args)
    monkeypatch.setattr(runner.method,'direction_and_amplitude',measured)
    result=runner.run(manifest()['base_config'],tmp_path/'case',strengths=[.1,.3,1.])
    assert len(calls)==2 and not result['failures']
    assert result['diagnostic_denominator']==8 and len(result['conditions'])==8
    assert result['actual_calls']['backward_completed']==2
    assert result['actual_calls']['transformer_completed']==184
    assert result['actual_calls']['scheduler_step_completed']==92
    assert result['actual_calls']['shadow_step_completed']==48
    for suffix in ('A','B'):
        q=torch.load(tmp_path/'case'/f'{suffix}_q.pt',weights_only=True)
        for i,rho in enumerate((.1,.3,1.)):
            row=result['conditions'][selection.flow_name(i,suffix)]
            assert row['status']=='COMPLETE' and row['rho']==rho
            epsilon=rho*.999*result['directions'][suffix]['epsilon_cap']
            a=torch.load(tmp_path/'case'/(selection.flow_name(i,suffix)+'_coefficients.pt'),weights_only=True)
            assert row['requested_epsilon']==epsilon
            torch.testing.assert_close(a,epsilon*q,rtol=0,atol=0)


def test_media_failed_save_readable_residue_never_upgrades(monkeypatch,tmp_path):
    from experiments.wan_state_clock import calibration_media as media
    vae=torch.nn.Linear(1,1);vae.to=lambda *args:vae
    monkeypatch.setattr(media,'load_frozen_vae',lambda config:vae)
    monkeypatch.setattr(media,'_clear_cache',lambda vae:None)
    monkeypatch.setattr(media,'decode_normalized_latent',lambda *args:torch.zeros(181,2,2,3))
    def bad_save(*args):raise RuntimeError('partial MP4 remains readable')
    monkeypatch.setattr(media,'encode_rgb',bad_save)
    monkeypatch.setattr(media,'read_mp4',lambda path:torch.zeros(181,2,2,3))
    monkeypatch.setattr(media,'reencode_rgb24_readback',lambda vae,rgb:torch.zeros(1,16,1+(len(rgb)-1)//4,40,64))
    reader_calls=[]
    def read(obs,book):
        reader_calls.append((len(obs),set(book)))
        return {'rankings':{str(i):{'message_unique':True,'best':{'message':0}} for i in range(5)}}
    monkeypatch.setattr(media.state_clock,'read',read)
    monkeypatch.setattr(media.state_clock,'report',lambda *args:{})
    path=tmp_path/'terminal.pt';torch.save(torch.zeros(1),path)
    original_save=torch.save
    monkeypatch.setattr(torch,'save',lambda *args,**kwargs:None)
    result=media.run(manifest()['base_config'],{'FLOW_r00_A':path},tmp_path/'media')
    row=result['videos']['FLOW_r00_A']
    assert row['status']=='PARTIAL_OR_FAILED' and row['five_methods_unique_correct'] is True
    assert reader_calls[0][0]==4
    assert result['actual_calls']['mp4_save_attempted']==1 and result['actual_calls']['mp4_save_completed']==0
    assert result['actual_calls']['vae_encode_completed']==4 and result['failures']
    assert 'partial MP4' in result['failures'][0]['traceback']


def test_notebook_binds_published_calibration_source():
    nb=json.loads(Path('notebooks/velocity_calibration_colab.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    source=''.join(nb['cells'][2]['source']);launch=''.join(nb['cells'][3]['source'])
    assert "SOURCE_COMMIT = '5db00fa60e1a3cad88b6736e1a402381afad3623'" in source
    assert 'if ACTUAL_SOURCE != SOURCE_COMMIT:' in source
    assert "'fetch', '--depth', '1', 'origin', SOURCE_COMMIT" in source
    assert "'checkout', '--detach', SOURCE_COMMIT" in source
    assert 'PAYLOAD' not in json.dumps(nb) and 'base64' not in json.dumps(nb)
    assert "'--stage', 'all'" in launch and 'os.killpg' in launch


def test_holdout_reports_same_criteria_and_terminal_failures_without_selecting():
    m=manifest();terminal=next(iter(terminal_results(m).values()));media=next(iter(media_results(m).values()))
    rows=selection.evaluate_holdout_case(terminal,media,'holdout_case')
    assert len(rows)==2 and all(row['fixed_criteria_met'] for row in rows)
    assert rows[0]['quality_ratios']=={'rgb_mse':1.,'residual_temporal_mse':1.}
    terminal['conditions']['FLOW_r00_A']['steps'][0]['step_within_budget']=False
    rows=selection.evaluate_holdout_case(terminal,media,'holdout_case')
    assert rows[0]['eligible'] is True and rows[0]['fixed_criteria_met'] is False
    assert rows[0]['control_steps'][0]['step_within_budget'] is False
    assert rows[1]['fixed_criteria_met'] is True
    missing=selection.evaluate_holdout_case(None,None,'missing_case')
    assert len(missing)==2 and all(row['terminal_status']=='MISSING' and not row['fixed_criteria_met'] for row in missing)
