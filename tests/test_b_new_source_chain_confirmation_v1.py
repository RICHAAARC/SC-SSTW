"""Artificial CPU checks for new orchestration/post boundaries; no real assets."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import pytest
from experiments.wan_state_clock import b_new_source_chain_confirmation_v1_records as rec
from experiments.wan_state_clock import b_new_source_chain_confirmation_v1_evaluation as post
from experiments.wan_state_clock.b_new_source_chain_confirmation_v1_experiment import run

ROOT=Path(__file__).resolve().parents[1]
STEM='b_new_source_chain_confirmation_v1'
pytestmark=pytest.mark.quick


def module(path):
    spec=importlib.util.spec_from_file_location('artificial_'+Path(path).stem,ROOT/path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def fixture(tmp):
    cfg=rec.load(ROOT/'experiments/wan_state_clock/configs'/f'{STEM}.json')
    cfg.update(source=str(tmp/'source'),output=str(tmp/'output'))
    terminal=Path(cfg['source'])/cfg['terminal_relative'];terminal.parent.mkdir(parents=True);terminal.write_bytes(b'artificial marker')
    protocol=dict(config=dict(writer=dict(key='watermark',bits=[0,1]*16),
        generation=dict(height=320,width=512,frames=181,fps=8),sources=[dict(id='b2-m3-c02',a=11,seed=123,role='cal',prompt='artificial')]))
    rec.dump(Path(cfg['source'])/'protocol.json',protocol)
    material=module('tests/test_b_minimal_l_materialization_v1.py')
    timing=module('tests/test_b_conditional_time_output_v1.py')
    payload=module('tests/test_b_time_payload_cdc32_v1.py')
    class P(payload.Fake):output=Path(cfg['output'])/'payload'
    return cfg,material,timing,P


def assert_fixed(recovered):
    artifacts=recovered['recovered_artifacts']
    assert sum(len(r['value']['slots']) for r in artifacts.values() if r['kind']=='local')==1440
    assert sum(len(r['value']['slots']) for r in artifacts.values() if r['kind']=='final')==480
    assert sum(len(r['value']['final_nodes']) for r in artifacts.values() if r['kind']=='time')==822
    assert sum(len(r['value']['final_relations']) for r in artifacts.values() if r['kind']=='time')==817
    raw=next(r['value'] for r in artifacts.values() if r['kind']=='raw_bank')
    assert len(raw)==60 and sum(r['raw_bit_slots'] for r in raw.values())==9600
    assert sum(r['vote_slots'] for r in raw.values())==76800


def test_full_original_chain_deferred_posts_private_message_and_read_only(tmp_path,monkeypatch):
    cfg,old,timing,P=fixture(tmp_path);output=Path(cfg['output']);seen=[]
    original=Path.read_text
    def guarded(path,*args,**kwargs):
        if path==Path(cfg['source'])/'protocol.json' or path.name=='attack_truth.json':
            assert (output/'blind_terminal.json').is_file()
            seen.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    class M(old.Fake):
        def __init__(self,protocol,folder,event):
            assert protocol['source_id']=='b2-m3-c02' and protocol['global_start']==0
            assert not set(protocol)&{'a','seed','bits','prompt','writer'}
            reservation=rec.load(output/'attempt.json')
            assert reservation['budgets']['heterogeneous_arithmetic_upper_bound']==8813
            assert len(reservation['recovery_layout'])==66
            super().__init__(protocol,folder,event)
    result=run(cfg,backend_factories=(M,M),receiver_backend_factory=timing.FakeBackend,payload_backend_factory=P)
    assert result['status']=='PACKAGE_FIXED_ATTEMPT_TERMINATED',result['errors']
    assert not result['errors'],result['errors']
    assert not (output/'time_chain/receiver/evaluation.json').exists()
    assert not (output/'payload/evaluation.json').exists()
    assert result['stage_results']['time_chain']['receiver_evaluation_status']=='DEFERRED_POST'
    assert result['stage_results']['payload']['evaluation_status']=='DEFERRED_POST'
    assert result['stage_results']['time_chain']['materialization_native_status']=='NUMERICALLY_UNRESOLVED'
    assert len(seen)==3  # time truth, original protocol, payload truth, all after whole blind save
    assert rec.load(output/'private/post_fixture.json')['message_bits']==[0,1]*16
    e=rec.load(output/'payload_evaluation.json')
    for v in e['views'].values():
        for row in v['arms']['B']['rows']:
            if row['value'] is not None:assert row['literal_correct']==(row['value']==row['slot']%2)
    budgets=result['budgets'];assert sum(budgets['materialization_A'].values())==8457
    assert sum(budgets['materialization_B'].values())==57
    assert sum(budgets['time'].values())==43 and sum(budgets['payload'].values())==256
    counts=rec.load(output/'counts.json')
    assert sum(v['started'] for v in counts['time'].values())==43
    assert sum(v['started'] for v in counts['payload'].values())==256
    assert result['phases']['time_post']['started']==result['phases']['payload_post']['started']==result['phases']['summary']['started']==1
    before={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()};seen.clear()
    recovered=run(dict(output=str(output)));assert recovered['physical_calls_this_invocation']==recovered['post_calls_this_invocation']==0
    assert not seen;assert_fixed(recovered)
    assert before=={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mode',['interrupt_A','A_codec','interrupt_B','linear','time_interrupt','payload_interrupt'])
def test_failure_consumes_unstarted_stages_and_fixed_slots(tmp_path,mode):
    cfg,old,timing,P=fixture(tmp_path)
    class M(old.Fake):pass
    class T(timing.FakeBackend):
        def codebook(self):
            if mode=='time_interrupt':raise KeyboardInterrupt('artificial time interruption')
            return super().codebook()
    if mode in ('interrupt_A','A_codec','interrupt_B','linear'):M.mode=mode
    if mode=='payload_interrupt':P.mode='interrupt'
    result=run(cfg,backend_factories=(M,M),receiver_backend_factory=T,payload_backend_factory=P)
    output=Path(cfg['output']);recovered=run(dict(output=str(output)))
    assert_fixed(recovered)
    assert recovered['physical_calls_this_invocation']==recovered['post_calls_this_invocation']==0
    if mode!='payload_interrupt':assert not (output/'payload/attempt.json').exists()
    assert all(r['status'] not in ('STARTED','NOT_STARTED') for r in result['phases'].values())
    assert rec.load(output/'evaluation.json')['status'] in ('COUNTEREXAMPLE','EVIDENCE_INCOMPLETE')


@pytest.mark.parametrize('damage',['missing','bool','short','source','geometry','key'])
def test_private_message_failure_never_falls_back(tmp_path,damage):
    cfg,_,_,_=fixture(tmp_path);output=Path(cfg['output']);path=Path(cfg['source'])/'protocol.json'
    rec.dump(output/'blind_terminal.json',dict(status='ALL_CHAIN_BLIND_SLOTS_TERMINAL'))
    value=rec.load(path)
    if damage=='missing':path.unlink()
    else:
        if damage=='bool':value['config']['writer']['bits'][0]=True
        if damage=='short':value['config']['writer']['bits'].pop()
        if damage=='source':value['config']['sources'][0]['id']='b2-m3-c01'
        if damage=='geometry':value['config']['generation']['frames']=180
        if damage=='key':value['config']['writer']['key']='other'
        rec.dump(path,value)
    with pytest.raises((ValueError,OSError)):post.private_message(cfg['source'],output)
    assert not (output/'private/post_fixture.json').exists()


def evidence(full=True):
    from experiments.wan_state_clock.b_time_payload_cdc32_v1_evaluation import compare
    time=dict(evidence_complete=True,views={})
    payload=dict(evidence_complete=True,B_counterexamples=[],views={})
    rows=rec.chain.receiver.plan('media')
    for r in rows.values():r.update(started=1,returned=1,status='RETURNED')
    for v,n in zip(rec.payload.VIEWS,rec.payload.LENGTHS):
        time['views'][v]=dict(preservation=True,structural_errors=[],nodes=[dict(old_status='CORRECT',new_status='NOT_ADDED') for _ in range(n)],
            relations=[dict(old_status='CORRECT',new_status='NOT_ADDED') for _ in range(n-1)])
        arm=dict(full32_supported=full,correct_supported=32 if full else 0,group_support=[full]*4,output_count=32 if full else 0,
            rows=[dict(value=1 if full else None,literal_correct=True if full else None,correct_supported=full,wrong_attribution=False) for _ in range(32)])
        payload['views'][v]=dict(arms={a:deepcopy(arm) for a in rec.payload.ARMS},B_vs_U=compare(arm['rows'],arm['rows']))
    return time,payload,rows


def test_old_H_error_survives_new_L_clean_and_incomplete_payload():
    t,p,r=evidence();t['views']['FULL']['nodes'][7]['old_status']='WRONG'
    t['views']['CROP']['relations'][2]['old_status']='WRONG';p['evidence_complete']=False
    e=post.summarize(t,p,r,{},dict(technical='artificial'))
    assert e['status']=='COUNTEREXAMPLE' and not e['evidence_complete']
    assert e['time']['counts']['nodes']['WRONG']==1 and e['time']['counts']['relations']['WRONG']==1
    assert len(e['counterexamples'])==2


def test_transfer_and_effect_are_separate_and_attribution_is_not_decoding():
    t,p,r=evidence();e=post.summarize(t,p,r,{}, {})
    assert e['status']=='FINITE_FIVE_VIEW_PAYLOAD_TRANSFER_CONFIRMED'
    assert e['synchronization_effect']['status']=='NO_DISCRIMINATING_EFFECT_OBSERVED'
    from experiments.wan_state_clock.b_time_payload_cdc32_v1_evaluation import compare
    view=p['views']['CROP'];u=view['arms']['U']['rows'];u[0].update(correct_supported=False,wrong_attribution=True)
    view['B_vs_U']=compare(view['arms']['B']['rows'],u)
    effect=post.effect_report(p)
    assert effect['status']=='TIME_ATTRIBUTION_OR_SUPPORT_IMPROVEMENT'
    assert effect['edited_view_attribution_only']==[('CROP',0)]
    assert not effect['edited_view_common_corrections'] and not effect['edited_view_new_output_support']
    assert effect['views']['CROP']['B_common_BER']==effect['views']['CROP']['control_common_BER']==0


def test_partial_all_rejected_technical_and_coverage_remain_distinct():
    t,p,r=evidence(False);assert post.summarize(t,p,r,{}, {})['status']=='NO_SUPPORTED_READBACK'
    p['views']['FULL']['arms']['B']['correct_supported']=1
    assert post.summarize(t,p,r,{}, {})['status']=='PARTIAL_SUPPORTED_READBACK'
    p['evidence_complete']=False
    assert post.summarize(t,p,r,{}, {})['status']=='EVIDENCE_INCOMPLETE'
    t,p,r=evidence();t['views']['FULL']['nodes'][0]['old_status']='UNRESOLVED'
    e=post.summarize(t,p,r,{},{});assert e['status']=='FINITE_FIVE_VIEW_PAYLOAD_TRANSFER_CONFIRMED'
    assert e['time']['counts']['nodes']['UNRESOLVED']==1


def test_terminal_save_failure_never_opens_post(tmp_path,monkeypatch):
    cfg,old,timing,P=fixture(tmp_path);output=Path(cfg['output']);original=rec.dump
    def fail(path,value):
        if Path(path)==output/'blind_terminal.json':raise OSError('artificial whole terminal save failure')
        return original(path,value)
    monkeypatch.setattr(rec,'dump',fail)
    def forbidden(*a,**k):pytest.fail('post opened without whole blind terminal')
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_backend_factory=timing.FakeBackend,payload_backend_factory=P,
               time_evaluator=forbidden,payload_evaluator=forbidden)
    assert not result['blind_terminal_saved'] and result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    assert not (output/'private/post_fixture.json').exists()
    assert_fixed(run(dict(output=str(output))))


def test_builder_sync_remote_bootstrap_and_no_source_recovery(tmp_path,monkeypatch,capsys):
    import nbformat,urllib.request,shutil
    builder=module('scripts/build_'+STEM+'_notebook.py');built=builder.build(tmp_path)
    notebook=nbformat.read(built['notebook'],as_version=4);nbformat.validate(notebook)
    assert notebook==nbformat.read(ROOT/'notebooks'/f'{STEM}.ipynb',as_version=4)
    assert notebook.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    urls=[]
    def retrieve(url,path):
        urls.append(url)
        if str(path).endswith(STEM+'_entry.py'):Path(path).write_text("def launch(*args): return {'status':'ARTIFICIAL_BOOTSTRAP'}\n")
        else:shutil.copyfile(ROOT/url.split('/fixture-ref/')[1],path)
    monkeypatch.setattr(urllib.request,'urlretrieve',retrieve)
    scope=dict(SOURCE='/artificial',OUTPUT=str(tmp_path/'out'),SOURCE_DIRECTORY=None,SOURCE_REF='fixture-ref',MODEL_ID=None)
    exec(builder.RUN,scope);assert scope['result']['status']=='ARTIFICIAL_BOOTSTRAP' and len(urls)==12
    rec.dump(Path(scope['OUTPUT'])/'attempt.json',dict(recovery_layout=rec.recovery_layout()))
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('recovery download'))
    exec(builder.RUN,scope);assert_fixed(scope['result']);assert len(capsys.readouterr().out)<4000
    code="""
import sys
class Deny:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','psutil','runtime'):raise AssertionError(name)
sys.meta_path.insert(0,Deny())
from experiments.wan_state_clock.b_new_source_chain_confirmation_v1_run import main
main()
"""
    called=subprocess.run([sys.executable,'-B','-c',code,'--output',scope['OUTPUT'],'--config','/missing'],cwd=ROOT,text=True,capture_output=True)
    assert called.returncode==0,called.stderr
    assert json.loads(called.stdout)['physical_calls_this_invocation']==0


def test_entry_actual_preflight_setup_then_new_worker_and_saved_recovery(tmp_path,monkeypatch):
    from types import SimpleNamespace
    cfg,_,_,_=fixture(tmp_path);entry=module('scripts/'+STEM+'_entry.py');common=entry.cpu._common();seen=[]
    monkeypatch.setattr(entry.cpu,'_common',lambda:common)
    monkeypatch.setattr(common,'load_file',lambda *a:SimpleNamespace(setup_dependencies=lambda *a:seen.append('setup')))
    def worker(session,root):
        actual=rec.load(session.config_path);seen.append(actual)
        assert actual['source_id']=='b2-m3-c02' and actual['terminal_relative']==cfg['terminal_relative']
        assert STEM+'_run' in session._command(root)[4]
        assert not session.output.exists()
        rec.dump(session.output/'attempt.json',dict(recovery_layout=rec.recovery_layout()))
        rec.dump(session.output/'result.json',dict(status='ARTIFICIAL_NEW_CHAIN'))
    monkeypatch.setattr(common.Session,'_worker',worker)
    result=entry.launch(cfg['source'],cfg['output'],source_directory=ROOT,scratch=tmp_path/'scratch')
    assert result['status']=='ARTIFICIAL_NEW_CHAIN' and seen[0]=='setup',result
    monkeypatch.setattr(common,'prepare_source',lambda *a,**k:pytest.fail('consumed preparation'))
    assert_fixed(entry.launch('/missing',cfg['output'],source_directory='/missing'))


def test_bad_original_message_keeps_time_counterexample_and_no_post_retry(tmp_path):
    cfg,old,timing,P=fixture(tmp_path);source=Path(cfg['source']);output=Path(cfg['output'])
    protocol=rec.load(source/'protocol.json');protocol['config']['writer']['bits']=[True]*32;rec.dump(source/'protocol.json',protocol)
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_backend_factory=timing.FakeBackend,payload_backend_factory=P)
    assert result['phases']['payload_post']['started']==1 and result['phases']['payload_post']['returned']==0
    assert '32 binary integers' in result['errors']['payload_post']
    e=rec.load(output/'evaluation.json');assert not e['evidence_complete'] and e['status']=='COUNTEREXAMPLE'
    assert not (output/'payload_evaluation.json').exists() and not (output/'private/post_fixture.json').exists()
    protocol['config']['writer']['bits']=[1]*32;rec.dump(source/'protocol.json',protocol)
    assert run(cfg)['post_calls_this_invocation']==0
    assert not (output/'payload_evaluation.json').exists()


def test_summary_interrupt_marks_whole_package_and_never_repeats(tmp_path,monkeypatch):
    cfg,old,timing,P=fixture(tmp_path)
    def stop(*args):raise KeyboardInterrupt('artificial summary interruption')
    monkeypatch.setattr(post,'summarize',stop)
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_backend_factory=timing.FakeBackend,payload_backend_factory=P)
    assert result['phases']['summary']['status']=='INTERRUPTED'
    assert result['status']=='PACKAGE_INTERRUPTED'
    assert run(cfg)['post_calls_this_invocation']==0


def test_corrupt_time_operation_receipt_keeps_unknown_slots_and_unconfirmed_artifact(tmp_path,monkeypatch):
    output=tmp_path/'out';rec.dump(output/'attempt.json',dict(recovery_layout=rec.recovery_layout()))
    actual=dict(final_nodes=[dict(received_frame=r,output_index=r) for r in range(181)],final_relations=[])
    path='time_chain/receiver/received/FULL/submission.json'
    rec.dump(output/path,actual);rec.dump(output/'time_chain/receiver/state.json',{'FULL/submit':None})
    recovered=run(dict(output=str(output)));assert_fixed(recovered)
    item=recovered['recovered_artifacts'][path]
    assert item['unconfirmed_artifact']==actual and item['error']
    assert all(r['output_index'] is None for r in item['value']['final_nodes'])
    # The source-free notebook and CLI share the same fixed fallback interpretation.
    builder=module('scripts/build_'+STEM+'_notebook.py')
    scope=dict(SOURCE='/missing',OUTPUT=str(output),SOURCE_DIRECTORY=None,SOURCE_REF='unused',MODEL_ID=None)
    exec(builder.RUN,scope);assert scope['result']['recovered_artifacts'][path]['value']==item['value']
    called=subprocess.run([sys.executable,'-B','-m','experiments.wan_state_clock.'+STEM+'_run','--output',str(output),
                           '--config','/missing'],cwd=ROOT,text=True,capture_output=True)
    assert called.returncode==0,called.stderr


@pytest.mark.parametrize('stage',['time','payload','summary'])
def test_post_receipt_write_failure_retains_observed_counterexample(tmp_path,monkeypatch,stage):
    cfg,old,timing,P=fixture(tmp_path);output=Path(cfg['output']);original=rec.dump
    t,p,_=evidence()
    t['views']['FULL']['nodes'][2]['old_status']='WRONG'
    p['B_counterexamples']=[dict(view='FULL',slot=3,literal_error=True,wrong_attribution=False,unsupported_output=False)]
    filename='evaluation.json' if stage=='summary' else stage+'_evaluation.json'
    phase='summary' if stage=='summary' else stage+'_post'
    def fail(path,value):
        if Path(path)==output/filename:raise OSError('artificial post receipt write failure')
        return original(path,value)
    monkeypatch.setattr(rec,'dump',fail)
    result=run(cfg,backend_factories=(old.Fake,old.Fake),receiver_backend_factory=timing.FakeBackend,payload_backend_factory=P,
               time_evaluator=lambda *a:t if stage in ('time','summary') else evidence()[0],
               payload_evaluator=lambda *a,**k:p if stage=='payload' else evidence()[1])
    assert result['phases'][phase]['status']=='FAILED'
    assert result['evaluation_status']=='COUNTEREXAMPLE' and result['known_counterexamples']
    assert phase in result['errors']
    if stage!='summary':
        e=rec.load(output/'evaluation.json')
        assert e['status']=='COUNTEREXAMPLE' and not e['evidence_complete'] and e['counterexamples']
        assert phase in e['errors']
    assert run(cfg)['post_calls_this_invocation']==0
