"""Bounded CPU restore/report checks. No pretrained model or GPU execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import projection_margin as carrier,velocity_coefficients as method
from runtime.wan.velocity_direction import tail,detached_scheduler,response_coefficients
from runtime.wan.zero_restore import restore_snapshot,validate_snapshot
from experiments.wan_state_clock import zero_path_run as runner
from test_velocity_direction import setup,scheduler
from test_velocity_checkpointing import tiny_wan

pytestmark=pytest.mark.unit


def metadata(s):return {'class':type(s).__name__,'config':dict(s.config),'sigmas':s.sigmas.tolist()}


def test_real_wan_saved_prefix_roundtrip_full_tail(setup,tmp_path):
    book,directions,codes,s,z=setup
    model=tiny_wan(torch.bfloat16,patch=(2,8,8),channels=16,layers=2)
    prompt=torch.randn(1,3,8).bfloat16();negative=-prompt
    file=tmp_path/'prefix.pt';torch.save({'latent':z,'scheduler':detached_scheduler(s)},file)
    payload=torch.load(file,map_location='cpu',weights_only=False)
    original_history=copy.deepcopy(payload['scheduler'].__dict__)
    recovered,restored,info=restore_snapshot(payload,scheduler(),'cpu',metadata(s),carrier.SHAPE)
    assert info['status']=='RESTORED_WITH_REENCODED_CONDITIONING'
    assert restored.step_index==44 and restored.this_order==s.this_order and restored.begin_index==0
    assert restored.config.solver_order==s.config.solver_order  # FrozenDict preserved.
    outputs=[]
    for latent,snapshot in ((z,s),(recovered,restored)):
        calls=[]
        with torch.no_grad():
            output=tail(SimpleNamespace(transformer=model),snapshot,latent,prompt,negative,torch.bfloat16,5.,
                torch.zeros(method.COEFFICIENT_SHAPE),directions,.02,lambda *args:calls.append(args),lambda *args:None,
                use_checkpoint=True,responses=response_coefficients(s,lambda *args:None))
        assert calls.count(('transformer',True))==12 and calls.count(('scheduler_step',True))==6
        assert calls.count(('shadow_step',True))==6 and not any('replay' in kind for kind,done in calls)
        assert output.grad_fn is None and snapshot.step_index==44
        outputs.append(output)
    torch.testing.assert_close(*outputs,rtol=0,atol=0)
    for key in ('last_sample','timesteps','sigmas'):torch.testing.assert_close(payload['scheduler'].__dict__[key],original_history[key],rtol=0,atol=0)
    broken=copy.deepcopy(payload);broken['scheduler'].model_outputs[0]=None
    with pytest.raises(ValueError,match='history'):validate_snapshot(broken,metadata(s),carrier.SHAPE)
    broken=copy.deepcopy(payload);broken['scheduler'].timestep_list[0]+=1
    with pytest.raises(ValueError,match='timestep history'):validate_snapshot(broken,metadata(s),carrier.SHAPE)


class FrozenFake(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.ones(1,dtype=torch.bfloat16),requires_grad=False)
        self.config=SimpleNamespace(_commit_hash='cpu-fixture');self.eval()
    def forward(self,hidden_states,timestep,encoder_hidden_states,**kwargs):
        return (torch.sin(hidden_states*.1)+.001*encoder_hidden_states,)


def test_case_readonly_single_zero_and_preserved_originals(setup,tmp_path,monkeypatch):
    book,d,c,s,z=setup
    source=tmp_path/runner.ORIGINAL_RUN;directory=source/'dev'/runner.CASES[0]/'terminal';directory.mkdir(parents=True)
    manifest=json.loads(Path('experiments/wan_state_clock/configs/velocity_calibration.json').read_text())
    (source/'manifest.json').write_text(json.dumps(manifest))
    config=copy.deepcopy(manifest['base_config']);case=manifest['development'][0]
    config['generation'].update(prompt=case['prompt'],seed=case['seed'])
    (directory/'config.json').write_text(json.dumps(config))
    np.savez(directory/'codebook.npz',**book)
    torch.save({'latent':z,'scheduler':detached_scheduler(s)},directory/'pre_intervention_state.pt')
    model=FrozenFake();pipe=SimpleNamespace(transformer=model,scheduler=scheduler(),vae=None)
    record={'source_commit':runner.ORIGINAL_SOURCE,'source_dirty':False,'environment':runner.environment(),
        'scheduler':metadata(s),'precision':{'transformer_input':'torch.bfloat16'},'R':.02,
        'responses':response_coefficients(s,lambda *args:None),'conditions':{}}
    with torch.no_grad():
        expected=tail(pipe,s,z,torch.tensor(1.).bfloat16(),torch.tensor(-1.).bfloat16(),torch.bfloat16,5.,
            torch.zeros(method.COEFFICIENT_SHAPE),d,.02,lambda *args:None,lambda *args:None,responses=record['responses'])
    for i,name in enumerate(('ZERO_A','ZERO_B',*runner.FLOW_NAMES)):
        torch.save(expected+.001*i,directory/(name+'_terminal.pt'))
        record['conditions'][name]={'status':'COMPLETE','correct_minus_competitor_margin':123.45,'loss':9.}
    (directory/'result.json').write_text(json.dumps(record))
    media=directory.parent/'media';media.mkdir();(media/'result.json').write_text(json.dumps({'status':'COMPLETE','readout':'original-negative-evidence'}))
    (source/'selection.json').write_text(json.dumps({'status':'NO_SELECTION','rho':None}))
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    prepared=[]
    def prepare(config,*,load_vae):
        prepared.append(load_vae)
        return pipe,torch.full_like(z,999),torch.tensor(1.).bfloat16(),torch.tensor(-1.).bfloat16(),torch.bfloat16
    monkeypatch.setattr(runner,'prepare_generation',prepare)
    output=tmp_path/'new'/runner.CASES[0]
    result=runner.run_case(source,output,runner.CASES[0])
    assert result['status']=='MEASURED_REQUIRES_REVIEW' and prepared==[False]
    torch.testing.assert_close(torch.load(output/'Z0_terminal.pt',weights_only=True),expected,rtol=0,atol=0)
    assert result['actual_calls']['transformer_completed']==12 and result['actual_calls']['backward_attempted']==0
    assert result['actual_calls']['prefix_transformer_attempted']==0 and result['actual_calls']['response_probe_step_attempted']==0
    assert result['original_media']['value']['readout']=='original-negative-evidence'
    row=result['comparison']['flow']['FLOW_r00_A']
    assert row['original_record']['correct_minus_competitor_margin']==123.45
    value=torch.load(directory/'FLOW_r00_A_terminal.pt',weights_only=True)
    assert row['tensor_difference_rms']['global_rms']==pytest.approx(float((value-expected).double().square().mean().sqrt()))
    assert before=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    # Required history missing: report case failure, never invoke model/prefix fallback.
    (directory/'pre_intervention_state.pt').unlink();prepared.clear()
    failed=runner.run_case(source,tmp_path/'failed',runner.CASES[0])
    assert failed['status']=='FAILED_RESTORATION_OR_CONTINUATION' and prepared==[]
    assert failed['actual_calls']['transformer_attempted']==0 and failed['failures']


def test_rms_is_tensor_difference_not_difference_of_norms(setup,tmp_path):
    book,d,c,s,z=setup
    zero=torch.ones(carrier.SHAPE)
    torch.save(-zero,tmp_path/'FLOW_r00_A_terminal.pt')
    report=runner.compare_terminals(zero,tmp_path,book,{'conditions':{}})
    row=report['flow']['FLOW_r00_A']
    assert row['tensor_difference_rms']['global_rms']==2.
    assert len(report['gradient_zeros'])==2 and len(report['flow'])==6
    assert report['flow']['FLOW_r01_A']['status']=='MISSING_OR_FAILED'
    assert row['competition_margin_gain_vs_Z0']==pytest.approx(row['recomputed_absolute_competition_margin']-row['Z0_competition_margin'])


def test_four_cases_failures_fixed_budget_no_selection_update(tmp_path,monkeypatch):
    source=tmp_path/runner.ORIGINAL_RUN;source.mkdir()
    (source/'selection.json').write_text(json.dumps({'status':'NO_SELECTION','rho':None}))
    commands=[]
    def launch(command,**kwargs):
        commands.append(command);dest=Path(command[command.index('--output')+1]);dest.mkdir()
        if len(commands)==1:raise RuntimeError('first child fails')
        runner.dump(dest/'result.json',{'status':'MEASURED_REQUIRES_REVIEW','actual_calls':{'transformer_completed':12,'scheduler_step_completed':6,'shadow_step_completed':6}})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess,'run',launch)
    result=runner.run_all(source,tmp_path/'new')
    assert len(commands)==4 and tuple(result['cases'])==runner.CASES
    assert result['fixed_total_calls']['transformer']==48 and result['actual_calls']['transformer_completed']==36
    assert result['original_selection']['value']['status']=='NO_SELECTION' and result['selection_recomputed'] is False
    with pytest.raises(ValueError):runner.check_paths(source,source/'new')


def test_zero_notebook_static():
    nb=json.loads(Path('notebooks/zero_path_colab.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    text=json.dumps(nb)
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    assert runner.ORIGINAL_RUN in text and 'diffusers==0.40.0' in text
    assert 'PAYLOAD' not in text and 'base64' not in text
    assert 'package in sys.modules' not in text and 'Restart the runtime' not in text
    assert 'import torch,diffusers; assert str(torch.__version__)' in text
    assert "'checkout', '--detach', SOURCE_COMMIT" in ''.join(nb['cells'][2]['source'])
