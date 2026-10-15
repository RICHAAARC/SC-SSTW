from __future__ import annotations
import ast,copy,json,shutil,subprocess,sys,types,zipfile
from pathlib import Path
import pytest,torch
from runtime.wan import local_joint_fixed_paired_tail_v1 as runtime
from main.tube_state import local_joint_fixed_paired_tail_v1 as method
from runtime.wan import local_joint_all_bit_resume_v1 as resume
from tests.test_local_joint_paired_transfer_v1 import Transformer,invoke as source_invoke,config as source_config
from tests.test_local_joint_readout_m0_v1 import TinyWan,PROTOCOL,SHAPE,SUPPORT

pytestmark=pytest.mark.quick
ROOT=Path(__file__).resolve().parents[1]
@pytest.fixture(scope="module")
def source(tmp_path_factory):
    old=torch.get_num_threads();torch.set_num_threads(1)
    p=tmp_path_factory.mktemp("saved-paired-source")/"run"
    r=source_invoke(source_config(),p)
    assert r['status']=='COMPLETE',r['failures']
    yield p
    torch.set_num_threads(old)

def config(source):
    c=json.loads((ROOT/'experiments/wan_state_clock/configs/local_joint_fixed_paired_tail_v1.json').read_text())
    c.update(device='cpu',transformer_dtype='float32',source_run=str(source));return c

def invoke(c,out,models=None,transformers=None):
    models=[] if models is None else models;transformers=[] if transformers is None else transformers
    def prepare(*_a,**_k):
        model=Transformer();transformers.append(model);return types.SimpleNamespace(transformer=model)
    def loader(*_a,**_k):
        model=TinyWan();models.append(model);return model
    return runtime.run(c,out,preparer=prepare,loader=loader,protocol=PROTOCOL,latent_shape=SHAPE,support=SUPPORT)

def complete(r):
    assert r['status']=='COMPLETE',r['failures']
    assert all(v['status']=='SCORED' for v in r['views'].values())

def test_fixed_budget_history_isolation_exact_saved_direction_and_report_only(source,tmp_path,monkeypatch):
    c=config(source);out=tmp_path/'out';models=[];transformers=[];seen=[]
    native_step=runtime.trajectory.native_step
    def step(s,z,v,i,count,**kwargs):
        if i==48:seen.append((s,s.model_outputs[-1].clone()))
        return native_step(s,z,v,i,count,**kwargs)
    monkeypatch.setattr(runtime.trajectory,'native_step',step)
    monkeypatch.setattr(torch.autograd,'grad',lambda *_a,**_k:pytest.fail('new VJP forbidden'))
    monkeypatch.setattr(runtime.generation,'prepare_generation',lambda *_a,**_k:pytest.fail('new baseline/noise forbidden'))
    r=invoke(c,out,models,transformers);complete(r)
    assert len(seen)==2 and seen[0][0] is not seen[1][0] and torch.equal(seen[0][1],seen[1][1])
    assert len(transformers)==len(models)==1 and transformers[0].calls==4 and models[0].calls==2
    assert r['counts']['generation_prepare_completed']==r['counts']['vae_load_completed']==1
    assert r['counts']['dit_completed']==r['counts']['native_step_completed']==4
    assert r['counts']['decode_completed']==2 and r['counts']['decoder_vjp_attempted']==0
    assert r['ordinary_chunks']=={'attempted':92,'completed':92}
    assert r['observation_accounting']['endpoints']['expected_metrics']==110
    assert r['observation_accounting']['endpoints']['observed_windows']==176
    assert torch.equal(runtime.tensor(out/'ZERO_TAIL/terminal_latent.pt'),runtime.tensor(source/'ZERO/terminal_latent.pt'))
    assert r['transport']['ZERO_TAIL_vs_SAVED_ZERO']['l2']==0
    assert r['transport']['CONTROL_TAIL']['conditional_clean_operand']['l2']==pytest.approx(.02,abs=1e-7)
    assert r['transport']['ZERO_TAIL']['control']['nominal_CFG_clean']['l2']==0
    zero_receipt=resume.read_json(out/'native/ZERO_TAIL_transport.json')['control']
    assert zero_receipt['provider_composition']=='zero_conditional_clean'
    assert zero_receipt['coordinate_relation']=='delta_c=0; delta_v_CFG=0; native update uses unchanged saved c/u'
    assert zero_receipt['requested_budget']=={'nominal_CFG_clean':0.,'conditional_clean':0.}
    assert r['transport']['paired_step48_effect']['l2']>0
    assert r['pair_execution_context']['same_preparation']
    for name in method.VIEWS:assert (out/'native'/(name+'_step49_predictions.pt')).exists()
    for pair in method.PAIRS:
        q=resume.read_json(out/'quality'/pair/'quality.json')
        assert len(q['framewise'])==181 and len(q['temporal_framewise'])==180
        assert q['exact_match'] or q['peak_events']
    report=resume.read_json(out/'comparison.json')
    assert len(report['views'])==2 and all(len(x['payload_rows'])==32 and len(x['state_rows'])==22 for x in report['views'].values())
    assert not report['scientific_pass'] and not runtime.model_work_remaining(out)
    image=next((out/'quality'/method.PAIRS[0]/'frames').glob('*.png'));image.unlink()
    monkeypatch.setattr(runtime.observer,'evaluate_saved',lambda *_a,**_k:pytest.fail('rescore'))
    monkeypatch.setattr(runtime.previous.visual,'quality',lambda *_a,**_k:pytest.fail('quality rescore'))
    models=[];transformers=[];r=invoke(c,out,models,transformers);complete(r)
    assert not models and not transformers and not any(r['current_attempt_counts'].values()) and image.exists()
    predictions=out/'native/CONTROL_TAIL_step49_predictions.pt';saved=runtime.tensor(predictions)
    assert runtime.committed(out,'CONTROL_TAIL_step49_predictions')
    predictions.unlink()
    r=invoke(c,out,models,transformers)
    assert r['status']=='ENGINEERING_FAILURE' and not models and not transformers
    assert not any(r['current_attempt_counts'].values()) and not predictions.exists()
    assert any('step49_predictions missing' in x['reason'] for x in r['failures'])
    invalid=copy.deepcopy(saved);invalid['conditional'].flatten()[0]=float('nan');resume.atomic_tensor(predictions,invalid)
    r=invoke(c,out,models,transformers)
    assert r['status']=='ENGINEERING_FAILURE' and not any(r['current_attempt_counts'].values())
    assert not models and not transformers
    resume.atomic_tensor(predictions,saved)
    qfile=out/'quality'/method.PAIRS[0]/'quality.json';qfile.unlink()
    r=invoke(c,out,models,transformers)
    assert r['status']=='ENGINEERING_FAILURE' and not models and not transformers
    assert any('committed quality missing' in x['reason'] for x in r['failures'])

@pytest.mark.parametrize('arm,next_index',[('ZERO_TAIL',49),('ZERO_TAIL',50),('CONTROL_TAIL',49),('CONTROL_TAIL',50)])
def test_durable_step_resume_no_baseline_or_vjp(source,tmp_path,monkeypatch,arm,next_index):
    out=tmp_path/'out';atomic=resume.atomic_tensor
    def stop(path,value):
        atomic(path,value)
        if str(path).endswith('/native/'+arm+'.pt') and value['next_index']==next_index:raise KeyboardInterrupt('expiry after saved history')
    monkeypatch.setattr(resume,'atomic_tensor',stop)
    with pytest.raises(KeyboardInterrupt):invoke(config(source),out)
    monkeypatch.setattr(resume,'atomic_tensor',atomic)
    r=invoke(config(source),out);complete(r)
    assert r['counts']['dit_completed']==r['counts']['native_step_completed']==4
    assert r['counts']['decoder_vjp_attempted']==0 and r['counts']['decode_completed']==2
    assert r['failure_history']
    assert r['counts']['generation_prepare_completed']==(1 if (arm,next_index)==('CONTROL_TAIL',50) else 2)
    paired=r['pair_execution_context'];cross=(arm,next_index)!=('CONTROL_TAIL',50)
    assert paired['same_preparation'] is not cross and paired['spans_attempts'] is cross
    assert len(paired['model_preparation_call_ids'])==(2 if cross else 1)
    if next_index==49:
        contexts=r['transport'][arm]['execution_context']
        assert contexts['update48']['attempt_id']!=contexts['prediction49']['attempt_id']==contexts['update49']['attempt_id']
        assert contexts['update48']['model_preparation_call_id']!=contexts['prediction49']['model_preparation_call_id']==contexts['update49']['model_preparation_call_id']

@pytest.mark.parametrize('arm',['ZERO_TAIL','CONTROL_TAIL'])
def test_saved_predictions_retain_original_context_when_update_resumes(source,tmp_path,monkeypatch,arm):
    out=tmp_path/'out';atomic=resume.atomic_tensor
    def stop(path,value):
        atomic(path,value)
        if str(path).endswith('/native/'+arm+'_step49_predictions.pt'):raise KeyboardInterrupt('expiry after saved predictions')
    monkeypatch.setattr(resume,'atomic_tensor',stop)
    with pytest.raises(KeyboardInterrupt):invoke(config(source),out)
    original=runtime.tensor(out/'native'/(arm+'_step49_predictions.pt'))['execution_context']
    monkeypatch.setattr(resume,'atomic_tensor',atomic)
    r=invoke(config(source),out);complete(r)
    contexts=r['transport'][arm]['execution_context']
    assert contexts['prediction49']==original
    assert contexts['update49']['attempt_id']!=original['attempt_id']
    assert contexts['update49']['model_preparation_call_id']!=original['model_preparation_call_id']
    assert r['transport'][arm]['step49_predictions_reused']
    assert not r['pair_execution_context']['same_preparation'] and r['pair_execution_context']['spans_attempts']
    assert r['counts']['dit_completed']==r['counts']['native_step_completed']==4
    assert r['counts']['generation_prepare_completed']==2 and r['counts']['decoder_vjp_attempted']==0
    assert r['current_attempt_counts']['dit_completed']==(2 if arm=='ZERO_TAIL' else 0)
    assert runtime.committed(out,arm+'_step49_predictions')

def test_failed_control_preserves_zero_rows_and_unknown_interrupt_accounting(source,tmp_path,monkeypatch):
    original=Transformer.forward
    def fail(self,*a,**k):
        if self.calls==2:raise RuntimeError('fixed control final conditional failure')
        return original(self,*a,**k)
    monkeypatch.setattr(Transformer,'forward',fail)
    out=tmp_path/'failure';r=invoke(config(source),out)
    assert r['status']=='ENGINEERING_FAILURE' and r['views']['ZERO_TAIL']['status']=='SCORED'
    assert r['views']['CONTROL_TAIL']['status']=='FAILED' and r['counts']['dit_failed']==1
    assert r['counts']['dit_attempted']==3 and r['counts']['decode_completed']==1
    report=resume.read_json(out/'comparison.json')
    assert len(report['views']['CONTROL_TAIL']['payload_rows'])==32
    assert {x['status'] for x in report['views']['CONTROL_TAIL']['state_rows']}=={'FAILED'}
    monkeypatch.setattr(Transformer,'forward',original)
    r=invoke(config(source),out);complete(r)
    assert r['counts']['dit_failed']==1 and r['counts']['dit_completed']==4
    assert r['current_attempt_counts']['dit_completed']==2
    # Interrupt inside a physical call, not after its durable return.
    def stop(self,*a,**k):raise KeyboardInterrupt('expiry inside conditional')
    monkeypatch.setattr(Transformer,'forward',stop)
    out2=tmp_path/'interrupted'
    with pytest.raises(KeyboardInterrupt):invoke(config(source),out2)
    before=runtime.read_state(out2)
    assert before['counts']['dit_interrupted_completion_unknown']==1 and before['counts']['dit_failed']==0
    monkeypatch.setattr(Transformer,'forward',original)
    r=invoke(config(source),out2);complete(r)
    assert r['counts']['dit_attempted']==5 and r['counts']['dit_completed']==4
    assert r['counts']['dit_interrupted_completion_unknown']==1

def test_unreadable_direction_errors_before_models_no_substitute(source,tmp_path):
    copied=tmp_path/'source';shutil.copytree(source,copied)
    unit=runtime.tensor(copied/'units/gradient.pt');unit.pop('direction');resume.atomic_tensor(copied/'units/gradient.pt',unit)
    models=[];transformers=[];r=invoke(config(copied),tmp_path/'out',models,transformers)
    assert r['status']=='ENGINEERING_FAILURE' and not models and not transformers
    assert not any(r['counts'].values()) and r['failure_history']

def test_prepare_loads_only_transformer_and_overlap_never_writes_source(source,tmp_path,monkeypatch):
    import diffusers
    seen=[]
    def loaded(identifier,**kwargs):
        seen.append((identifier,kwargs));return Transformer()
    monkeypatch.setattr(diffusers.WanTransformer3DModel,'from_pretrained',loaded)
    p=runtime.prepare_saved_transformer(config(source),device='cpu',model_dtype=torch.float32)
    assert len(seen)==1 and seen[0][1]['subfolder']=='transformer' and seen[0][1]['torch_dtype']==torch.float32
    assert not p.transformer.training and not any(x.requires_grad for x in p.transformer.parameters())
    nested=source/'forbidden_new_output'
    with pytest.raises(ValueError,match='separate'):invoke(config(source),nested)
    assert not nested.exists()
    c=config(source);c['generation']['seed']=123;c['generation']['prompt']='optional stale location note'
    c['model']['revision']='optional revision note'
    runtime.validate_config(c)

def test_notebook_direct_mount_fixed_source_output_zip_and_no_git_cli(tmp_path):
    import nbformat
    from scripts import build_local_joint_fixed_paired_tail_v1_notebook as b
    nb=b.build_notebook();nbformat.validate(nbformat.from_dict(nb))
    assert json.loads((ROOT/'notebooks'/ (b.NAME+'_colab.ipynb')).read_text())==nb
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']));assert c['outputs']==[]
    assert ''.join(nb['cells'][0]['source']).strip()=="from google.colab import drive\ndrive.mount('/content/drive')"
    cell=''.join(nb['cells'][2]['source']);tree=ast.parse(cell)
    assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FIXED_CONFIG' for x in n.targets))
    fixed=ast.literal_eval(assignment.value)
    assert fixed==json.loads((ROOT/b.CONFIG_PATH).read_text())
    assert '/Local-Joint-State-Payload-V1-Paired-Transfer/resumable-paired-transfer-v1/run' in fixed['source_run']
    assert "OUTPUT_PARENT / 'resumable-fixed-paired-tail-v1'" in cell
    assert 'Fixed-Paired-Tail' in cell and 'force_remount' not in cell
    assert 'local_joint_fixed_paired_tail_v1_portable_source.zip' in ''.join(nb['cells'][3]['source'])
    archive=ROOT/'notebooks'/(b.NAME+'_portable_source.zip')
    with zipfile.ZipFile(archive) as z:
        for n in dict.fromkeys(b.PORTABLE_FILES):assert z.read(n)==(ROOT/n).read_bytes()
        z.extractall(tmp_path/'editable')
    cp=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.local_joint_fixed_paired_tail_v1_run','--help'],cwd=tmp_path/'editable',capture_output=True,text=True)
    assert cp.returncode==0,cp.stderr
    assert not (tmp_path/'editable/.git').exists()
