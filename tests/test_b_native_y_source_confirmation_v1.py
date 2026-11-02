"""Artificial CPU engineering only; no real P0, model, codec or source data."""
from pathlib import Path
from copy import deepcopy
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest
from experiments.wan_state_clock import b_native_y_source_confirmation_v1_records as rec
from experiments.wan_state_clock import b_native_y_source_confirmation_v1_evaluation as post
from experiments.wan_state_clock.b_native_y_source_confirmation_v1_experiment import run

ROOT = Path(__file__).resolve().parents[1]
STEM = 'b_native_y_source_confirmation_v1'
pytestmark = pytest.mark.quick


def module(relative):
    spec = importlib.util.spec_from_file_location('artificial_'+Path(relative).stem, ROOT/relative)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


old_time = module('tests/test_b_conditional_time_output_v1.py')


class TimeFake(old_time.FakeBackend):
    calls = []
    mode = 'ok'
    def check(self): pass
    def begin_view(self): pass
    def end_view(self): pass
    def source_probe(self, path): return super().probe(path, 181)
    def source_decode(self, path): return super().decode(path, 181)
    def encode_view(self, pixels, path): return super().encode(pixels, path)
    def make_codebook(self, path):
        self.calls.append('one_codebook')
        if self.mode == 'interrupt': raise KeyboardInterrupt('artificial time interrupt')
        np.savez(path, artificial=np.zeros(1))
        return 'artificial buckets', 'artificial flat'
    def probe(self, path, n):
        return dict(frames=n, profile=dict(status='MODEL_UNCONFIRMED'))
    def decode(self, path, n, probe): return super().decode(path, n)
    def compatible(self, raw, table): return dict(n=len(raw['soft']), labels=[[] for _ in raw['soft']])
    def H(self, compatible, n):
        hard = old_time.fixture([[10,12]]*n, [11]*n)['hard']
        hard['scope'] = 'Y_H_SHADOW_ORIGINAL_RULE'
        return hard, hard
    def L(self, graph, cost):
        local = super().L(cost)
        return local, super().candidates(graph, cost, local)
    def submit(self, hard, candidates): return super().semantic(hard, candidates)


def fixture(tmp_path):
    cfg = rec.load(ROOT/'experiments/wan_state_clock/configs'/f'{STEM}.json')
    cfg.update(source=str(tmp_path/'source'), output=str(tmp_path/'output'))
    terminal = Path(cfg['source'])/rec.TERMINAL
    terminal.parent.mkdir(parents=True); terminal.write_bytes(b'artificial marker')
    rec.dump(Path(cfg['source'])/'protocol.json', dict(config=dict(writer=dict(key='watermark',bits=[0,1]*16),
        generation=dict(height=320,width=512,frames=181,fps=8), sources=[dict(id=rec.SOURCE_ID, seed=2026102501)])))
    material = module('tests/test_b_minimal_l_materialization_v1.py')
    payload = module('tests/test_b_time_payload_cdc32_v1.py')
    class P(payload.Fake): output = Path(cfg['output'])/'payload'
    class T(TimeFake): calls = []
    return cfg, material.Fake, T, P


def test_actual_full_orchestration_blind_order_fixed_counts_and_metadata_recovery(tmp_path, monkeypatch):
    cfg, M, T, P = fixture(tmp_path)
    output = Path(cfg['output'])
    seen = []
    original = Path.read_text
    def guarded(path, *args, **kwargs):
        if path.name in ('protocol.json','attack_truth.json') and ('materialization' not in path.parts):
            assert (output/'blind_terminal.json').is_file()
            seen.append(path.name)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    result = run(cfg, backend_factories=(M,M), receiver_backend_factory=T, payload_backend_factory=P)
    assert not result['errors'], result
    assert result['stage_results']['B']['native_status'] == 'NUMERICALLY_UNRESOLVED'
    assert T.calls == ['one_codebook']
    counts = rec.load(output/'counts.json')
    assert sum(row['started'] for row in counts['time'].values()) == 96
    assert sum(row['started'] for row in counts['payload'].values()) == 256
    assert result['budgets']['heterogeneous_arithmetic_upper_bound'] == 8866
    assert result['budgets']['codec'] == dict(encode=12,probe=19,decode=19,total=50)
    assert seen == ['attack_truth.json','protocol.json','attack_truth.json']
    assert all(result['phases'][p]['started'] == 1 for p in ('time_post','payload_post','summary'))
    time_result = rec.load(output/'time_evaluation.json')
    assert sum(time_result['counts'].get(a+'/final/nodes/'+s,0) for a in ('BASE','QIM') for s in ('CORRECT','WRONG','UNRESOLVED','FALSE_ACCEPT')) == 1644
    assert sum(time_result['counts'].get(a+'/final/edges/'+s,0) for a in ('BASE','QIM') for s in ('CORRECT','WRONG','UNRESOLVED','FALSE_ACCEPT')) == 1634
    assert time_result['status'] == 'COUNTEREXAMPLE'
    assert any(b['kind']=='BASE_SUPPORTED_RELATION_SET' for b in time_result['counterexamples'])
    assert rec.load(output/'private/post_fixture.json')['message_bits'] == [0,1]*16
    before = {str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
    forbidden = {'blind_terminal.json','plans.json','submission.json','readout.npz','raw_bank.json','attack_truth.json','protocol.json','evaluation.json','time_evaluation.json','payload_evaluation.json'}
    def metadata_only(path,*args,**kwargs):
        assert path.name not in forbidden, path
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',metadata_only)
    recovered = run(dict(output=str(output)))
    assert recovered['physical_calls_this_invocation'] == recovered['post_calls_this_invocation'] == 0
    assert before == {str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mode', ['interrupt_A','linear','time_interrupt','payload_interrupt'])
def test_consumed_failures_keep_all_fixed_rows_and_no_resume(tmp_path, mode):
    cfg, Original, T, P = fixture(tmp_path)
    class M(Original): pass
    if mode in ('interrupt_A','linear'): M.mode = mode
    if mode == 'time_interrupt': T.mode = 'interrupt'
    if mode == 'payload_interrupt': P.mode = 'interrupt'
    result = run(cfg,backend_factories=(M,M),receiver_backend_factory=T,payload_backend_factory=P)
    assert all(r['status'] not in ('STARTED','NOT_STARTED') for r in result['phases'].values())
    slots = rec.load(Path(cfg['output'])/'fixed_slots.json')
    assert sum(v['nodes'] for v in slots['time'].values()) == 1644
    assert len(slots['payload']['rows']) == 256 and slots['payload']['denominators']['final_slots'] == 480
    assert result['evaluation_status'] in ('COUNTEREXAMPLE','EVIDENCE_INCOMPLETE')
    assert run(cfg)['post_calls_this_invocation'] == 0


def test_whole_terminal_write_failure_prevents_truth_and_all_posts(tmp_path, monkeypatch):
    cfg, M, T, P = fixture(tmp_path); output = Path(cfg['output']); original=rec.dump
    def dump(path,value):
        if Path(path)==output/'blind_terminal.json': raise OSError('artificial terminal write failure')
        return original(path,value)
    monkeypatch.setattr(rec,'dump',dump)
    def forbidden(*a,**k): pytest.fail('post after failed whole blind save')
    result=run(cfg,backend_factories=(M,M),receiver_backend_factory=T,payload_backend_factory=P,
               time_evaluator=forbidden,payload_evaluator=forbidden)
    assert not result['blind_terminal_saved'] and result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    assert not (output/'private/post_fixture.json').exists()


def test_BASE_supported_with_zero_exact_is_still_false_accept(tmp_path):
    from experiments.wan_state_clock.b_native_y_source_confirmation_v1_time import run as time_run
    output=tmp_path/'out';time_output=output/'time_chain/receiver'
    class T(TimeFake):
        def L(self, graph, cost):
            local,candidate=super().L(graph,cost)
            for row in candidate['nodes']:
                row.update(candidate_index=None,model_index=None,best_count=2,L_status='MODEL_TIE')
            return local,candidate
    time_run(dict(media=dict(QIM='artificial',BASE='artificial')),time_output,backend_factory=T)
    rec.dump(output/'blind_terminal.json',dict(status='ALL_CHAIN_BLIND_SLOTS_TERMINAL',
        phases={k:dict(status='RETURNED') for k in ('A','B','time','payload')}))
    result=post.evaluate_time(time_output,rec.load(time_output/'state.json'),time_output/'private/attack_truth.json',whole_output=output)
    base=[v for v in result['views'].values() if v['arm']=='BASE']
    assert all(v['exact_nodes']==v['exact_edges']==0 for v in base)
    assert all(v['BASE_qualification_false_accept'] for v in base)
    assert result['status']=='COUNTEREXAMPLE'


@pytest.mark.parametrize('damage', ['missing','bool','source','seed'])
def test_original_message_has_no_fallback(tmp_path, damage):
    cfg,_,_,_=fixture(tmp_path); output=Path(cfg['output']); path=Path(cfg['source'])/'protocol.json'
    rec.dump(output/'blind_terminal.json',dict(status='ALL_CHAIN_BLIND_SLOTS_TERMINAL',
        phases={k:dict(status='RETURNED') for k in ('A','B','time','payload')}))
    value=rec.load(path)
    if damage=='missing': path.unlink()
    else:
        if damage=='bool':value['config']['writer']['bits'][0]=True
        if damage=='source':value['config']['sources'][0]['id']='b-rank3-n02'
        if damage=='seed':value['config']['sources'][0]['seed']=1
        rec.dump(path,value)
    with pytest.raises((OSError,ValueError)):post.private_message(cfg['source'],output)


def test_one_generated_codebook_saved_and_flattened_without_second_generation(tmp_path, monkeypatch):
    from runtime.wan.b_native_y_source_confirmation_v1 import Backend
    from runtime.wan.b_native_y_diagnostic_v1 import Backend as Native
    tables=module('tests/test_b_native_y_conditional_v1.py').table_dict()
    calls=[]
    def generate(self):
        calls.append(1);return [(tables[f'j_{i}'],tables[f'word_{i}']) for i in range(5)]
    monkeypatch.setattr(Native,'codebook',generate)
    backend=Backend.__new__(Backend);backend.check=lambda:None
    grouped,flat=backend.make_codebook(tmp_path/'codebook.npz')
    assert len(calls)==1 and len(grouped)==5 and len(flat)==92672
    assert int(flat[11])==(1<<63)-1
    with np.load(tmp_path/'codebook.npz') as saved: assert set(saved)==set(tables)


def test_notebook_builder_sync_and_consumed_without_source_or_numpy(tmp_path, monkeypatch, capsys):
    import nbformat, urllib.request
    builder=module('scripts/build_'+STEM+'_notebook.py')
    notebook=nbformat.read(builder.build(tmp_path)['notebook'],as_version=4)
    assert notebook==nbformat.read(ROOT/'notebooks'/f'{STEM}.ipynb',as_version=4)
    assert notebook.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    output=tmp_path/'consumed';rec.dump(output/'attempt.json',{})
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('consumed network call'))
    scope=dict(SOURCE='/missing',OUTPUT=str(output),SOURCE_DIRECTORY=None,SOURCE_REF='unused',MODEL_ID=None)
    exec(builder.RUN,scope)
    assert scope['result']['physical_calls_this_invocation']==scope['result']['post_calls_this_invocation']==0
    assert len(capsys.readouterr().out)<3000
    called=subprocess.run(['/usr/bin/python3','-S','-B',str(ROOT/'experiments/wan_state_clock'/f'{STEM}_run.py'),
        '--output',str(output),'--config','/missing'],cwd='/tmp',text=True,capture_output=True)
    assert called.returncode==0,called.stderr
    assert json.loads(called.stdout)['physical_calls_this_invocation']==0


def test_real_entry_setup_worker_wiring_with_fake_worker(tmp_path,monkeypatch):
    from types import SimpleNamespace
    cfg,_,_,_=fixture(tmp_path);entry=module('scripts/'+STEM+'_entry.py');common=entry.cpu._common();seen=[]
    monkeypatch.setattr(entry.cpu,'_common',lambda:common)
    monkeypatch.setattr(common,'load_file',lambda *a:SimpleNamespace(setup_dependencies=lambda *a:seen.append('setup')))
    def worker(session,root):
        actual=rec.load(session.config_path);seen.append(actual)
        assert actual['source_id']==rec.SOURCE_ID and actual['terminal_relative']==rec.TERMINAL
        assert STEM+'_run' in session._command(root)[4]
        rec.dump(session.output/'attempt.json',{})
        rec.dump(session.output/'result.json',dict(status='ARTIFICIAL_N01_WORKER'))
    monkeypatch.setattr(common.Session,'_worker',worker)
    result=entry.launch(cfg['source'],cfg['output'],source_directory=ROOT,scratch=tmp_path/'scratch')
    assert result['status']=='ARTIFICIAL_N01_WORKER',result
    assert seen[0]=='setup'
    monkeypatch.setattr(common,'prepare_source',lambda *a,**k:pytest.fail('consumed source preparation'))
    assert entry.launch('/missing',cfg['output'],source_directory='/missing')['physical_calls_this_invocation']==0


@pytest.mark.parametrize('close_result', [None, {'error':'artificial late watcher error'}, {'peak_resident_bytes':9*1024**3}])
def test_time_close_failure_is_retained_and_closed_once(tmp_path,close_result):
    from experiments.wan_state_clock.b_native_y_source_confirmation_v1_time import run as time_run
    class T(TimeFake):
        closed=0
        def close(self):
            type(self).closed+=1
            return close_result
    result=time_run(dict(media=dict(QIM='artificial',BASE='artificial')),tmp_path/'time',backend_factory=T)
    assert T.closed==1 and result['errors']
    assert rec.load(tmp_path/'time/resources.json')['completion_errors']


def test_observed_time_counterexample_survives_post_write_failure(tmp_path,monkeypatch):
    cfg,M,T,P=fixture(tmp_path);output=Path(cfg['output']);original=rec.dump
    def dump(path,value):
        if Path(path)==output/'time_evaluation.json':raise OSError('artificial late post write failure')
        return original(path,value)
    monkeypatch.setattr(rec,'dump',dump)
    result=run(cfg,backend_factories=(M,M),receiver_backend_factory=T,payload_backend_factory=P)
    assert result['evaluation_status']=='COUNTEREXAMPLE' and result['known_counterexamples']
    assert result['phases']['time_post']['status']=='FAILED'
    assert run(cfg)['post_calls_this_invocation']==0


def test_fresh_notebook_remote_bootstrap_is_ordinary_and_single(tmp_path,monkeypatch,capsys):
    import urllib.request
    builder=module('scripts/build_'+STEM+'_notebook.py');urls=[]
    def retrieve(url,path):
        urls.append(url)
        Path(path).write_text("def launch(*args): return {'status':'ARTIFICIAL_REMOTE_BOOTSTRAP'}\n")
    monkeypatch.setattr(urllib.request,'urlretrieve',retrieve)
    scope=dict(SOURCE='/artificial',OUTPUT=str(tmp_path/'fresh'),SOURCE_DIRECTORY=None,SOURCE_REF='editable-ref',MODEL_ID=None)
    exec(builder.RUN,scope)
    assert scope['result']['status']=='ARTIFICIAL_REMOTE_BOOTSTRAP'
    assert len(urls)==1 and '/editable-ref/scripts/'+STEM+'_bootstrap.py' in urls[0]


def saved_post_fixture(tmp_path, base_exact=False, corrupted_final=False):
    output=tmp_path/'whole'; timing=output/'time_chain/receiver'
    rows=rec.time_plan()
    for row in rows.values():row.update(status='RETURNED',started=1,returned=1)
    rec.dump(timing/'state.json',rows)
    rec.dump(timing/'blind_terminal.json',dict(status='ALL_BLIND_ROWS_TERMINAL',rows={k:v for k,v in rows.items() if k!='blind_save'}))
    rec.dump(timing/'resources.json',dict(completion_errors={}))
    rec.dump(timing/'private/attack_truth.json',dict(views=rec.mappings()))
    rec.dump(output/'blind_terminal.json',dict(status='ALL_CHAIN_BLIND_SLOTS_TERMINAL',
        phases={k:dict(status='RETURNED') for k in ('A','B','time','payload')}))
    for unit in rec.units():
        path=timing/unit['folder']; mapping=rec.mappings()[unit['view']]
        exact=unit['arm']=='QIM' or base_exact
        bounds=[[t,t] for t in mapping] if exact else [[0,92671] for t in mapping]
        bundle=old_time.fixture(bounds, mapping if exact else [None]*len(mapping),
                                {i:t for i,t in enumerate(mapping)} if exact else {})
        h=bundle['hard']
        if not exact:
            h['qualification']=bundle['L']['qualification']='INSUFFICIENT_MULTILANE_EVIDENCE'
        final=dict(status='CONDITIONAL_OUTPUTS_SAVED' if exact else 'OLD_GRAPH_UNSUPPORTED',
            old_hard=deepcopy(h),original_L=deepcopy(bundle['L']),new_nodes=[],new_relations=[],
            final_nodes=[dict(n,origin='OLD' if exact else 'UNRESOLVED') for n in h['nodes']],final_relations=deepcopy(h['relations']))
        if corrupted_final and unit['id']=='v00':final['final_nodes'][10]['output_index']=9
        for name,value in [('Y_H.json',h),('submission.json',final),('candidates.json',bundle['L']),
            ('probe.json',dict(profile=dict(status='MODEL_UNCONFIRMED'))),('compatibility_F.json',dict(labels=[[] for _ in mapping])),
            ('C.json',dict(defined_rows=len(mapping) if exact else 0,comparisons=0,candidate_slots=0,state_counts={}))]:rec.dump(path/name,value)
    return output,timing,rows


def test_post_accepts_complete_BASE_rejection_without_technical_failure(tmp_path):
    whole,timing,rows=saved_post_fixture(tmp_path)
    result=post.evaluate_time(timing,rows,timing/'private/attack_truth.json',whole_output=whole)
    assert result['evidence_complete'] and not result['counterexamples'] and not result['structural_errors']
    assert result['counts']['BASE/final/nodes/UNRESOLVED']==822
    assert result['QIM_true_events_exact']


def test_post_BASE_geometric_match_is_false_accept_never_positive_recovery(tmp_path):
    whole,timing,rows=saved_post_fixture(tmp_path,base_exact=True)
    result=post.evaluate_time(timing,rows,timing/'private/attack_truth.json',whole_output=whole)
    assert result['evidence_complete'] and result['status']=='COUNTEREXAMPLE'
    assert result['counts']['BASE/final/nodes/FALSE_ACCEPT']==822
    assert result['counts']['BASE/final/edges/FALSE_ACCEPT']==817
    assert result['counts'].get('BASE/final/nodes/CORRECT',0)==0
    lines=[json.loads(s) for s in (timing/'post/nodes.jsonl').read_text().splitlines()]
    assert all(r['geometric_match_status']=='CORRECT' and r['status']=='FALSE_ACCEPT' for r in lines if r['arm']=='BASE')


def test_final_structural_rewrite_is_engineering_but_independent_H_error_retained(tmp_path):
    whole,timing,rows=saved_post_fixture(tmp_path,corrupted_final=True)
    result=post.evaluate_time(timing,rows,timing/'private/attack_truth.json',whole_output=whole)
    assert not result['evidence_complete'] and result['structural_errors']
    assert not result['counterexamples'] and result['status']=='EVIDENCE_INCOMPLETE'
    assert result['counts']['QIM/final/nodes/WRONG']==1
    h=rec.load(timing/'received/CROP/Y_H.json');h['nodes'][10]['output_index']=47
    rec.dump(timing/'received/CROP/Y_H.json',h)
    result=post.evaluate_time(timing,rows,timing/'private/attack_truth.json',whole_output=whole)
    assert any(b['unit']=='v01' and b['layer']=='H' and b['kind']=='TIME_POSITION' for b in result['counterexamples'])
    assert result['status']=='COUNTEREXAMPLE'


def test_material_actual_close_errors_incomplete_but_native_tolerance_not_new_gate():
    _,payload,_=module('tests/test_b_new_source_chain_confirmation_v1.py').evidence()
    time=dict(evidence_complete=True,QIM_true_events_exact=True,QIM_coverage=dict(nonstartup_correct=806,nonstartup_denominator=806))
    expected=post.summarize(time,payload,dict(B=dict(errors=dict(native='finite native target/budget failure retained'))),{})
    assert expected['evidence_complete'] and expected['status']=='FINITE_SINGLE_SOURCE_CONDITIONAL_READBACK'
    failed=post.summarize(time,payload,dict(B=dict(errors=dict(close='actual resource close failure'))),{})
    assert not failed['evidence_complete'] and failed['status']=='EVIDENCE_INCOMPLETE'
    assert failed['materialization']['technical_errors']['B']['close']


@pytest.mark.parametrize('when', ['physical','final'])
def test_root_terminal_state_failure_keeps_other_receipts_and_consumes(tmp_path,monkeypatch,when):
    cfg,_,_,_=fixture(tmp_path);output=Path(cfg['output']);original=rec.Journal.terminalize
    def terminalize(self,names=rec.PHASES):
        if (when=='physical' and tuple(names)==('A','B','time','payload')) or (when=='final' and tuple(names)==rec.PHASES):
            raise OSError('artificial root terminal state save failure')
        return original(self,names)
    monkeypatch.setattr(rec.Journal,'terminalize',terminalize)
    def phase(*a,**k):raise ValueError('artificial early A failure')
    result=run(cfg,phase_runner=phase)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    assert (output/'resources.json').is_file() and (output/'result.json').is_file() and (output/'fixed_slots.json').is_file()
    assert all(r['status'] not in ('NOT_STARTED','STARTED') for r in result['phases'].values())
    if when=='physical':
        assert not result['blind_terminal_saved'] and all(result['phases'][p]['started']==0 for p in ('time_post','payload_post','summary'))
    assert run(cfg)['post_calls_this_invocation']==0


@pytest.mark.parametrize('fault', ['B_close','final_state'])
def test_actual_chain_late_fault_retains_observed_counterexamples(tmp_path,monkeypatch,fault):
    cfg,Original,T,P=fixture(tmp_path)
    class M(Original):
        def close(self):
            if fault=='B_close' and self.output.name=='B':raise RuntimeError('artificial B close failure')
            return super().close()
    original=rec.Journal.terminalize
    def terminalize(self,names=rec.PHASES):
        if fault=='final_state' and tuple(names)==rec.PHASES:raise OSError('artificial late root state failure')
        return original(self,names)
    monkeypatch.setattr(rec.Journal,'terminalize',terminalize)
    result=run(cfg,backend_factories=(M,M),receiver_backend_factory=T,payload_backend_factory=P)
    assert result['evaluation_status']=='COUNTEREXAMPLE' and result['known_counterexamples']
    assert not result['evidence_complete']
    assert ('B/resource' if fault=='B_close' else 'final_terminal_state_save') in result['errors']
    assert Path(cfg['output'],'resources.json').is_file() and Path(cfg['output'],'result.json').is_file()
