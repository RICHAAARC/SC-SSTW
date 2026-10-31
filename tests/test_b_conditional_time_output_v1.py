"""Artificial inputs only: method boundaries, actual entry wiring and receipts."""
from copy import deepcopy
from contextlib import contextmanager
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest
from main.tube_state import b_conditional_time_output_v1 as method
from experiments.wan_state_clock import b_conditional_time_output_v1_records as rec
from experiments.wan_state_clock.b_conditional_time_output_v1_experiment import run, attack_mappings
from experiments.wan_state_clock.b_conditional_time_output_v1_evaluation import evaluate_view, evaluate

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.quick


def fixture(bounds, choices, H=None):
    H = H or {}
    hard = dict(path_status='COMPATIBLE_PATH', qualification='SUPPORTED_RELATION_SET', nodes=[], relations=[])
    local = dict(qualification=hard['qualification'], epsilon=1e-12, nodes=[])
    for r, (interval, n) in enumerate(zip(bounds, choices)):
        hard['nodes'].append(dict(received_frame=r, interval=list(interval), output_index=H.get(r), status='LOCAL_LABEL'))
        local['nodes'].append(dict(received_frame=r, model_status='DEFINED', L_status='SOFT_MODEL_BEST' if n is not None else 'MODEL_TIE',
                                  best_count=1 if n is not None else 2, model_index=n, candidate_index=n))
    for r in range(len(bounds)-1):
        delta=[max(0,bounds[r+1][0]-bounds[r][1]),bounds[r+1][1]-bounds[r][0]]
        kind='REPEAT' if delta==[0,0] else 'NORMAL' if delta==[1,1] else 'GAP' if delta[0]>1 else 'AMBIGUOUS'
        row = dict(received_left=r, output_endpoints=None, kind=kind, delta=delta,
                   output_missing_indices=None, output_repeated_index=None, status='AMBIGUOUS')
        if r in H and r+1 in H:
            row.update(method.event(H[r], H[r+1]));row['delta']=[row['delta']]*2
            row['status']='EXACT';row['extra_old_field']={'preserve': True}
        hard['relations'].append(row)
    return dict(hard=hard, L=local)


def submit(bundle):
    return method.submit(bundle['hard'], bundle['L'])


def test_empty_H_and_preserve_old_records():
    b = fixture([[10, 12], [11, 13]], [11, 12]); original=deepcopy(b)
    s=submit(b)
    assert len(s['new_nodes']) == 2 and len(s['new_relations']) == 1
    assert b == original and s['old_hard'] == b['hard'] and s['original_L'] == b['L']
    b=fixture([[10,10],[11,11],[12,14]], [10,11,13], {0:10,1:11})
    s=submit(b)
    assert s['final_relations'][0] == b['hard']['relations'][0]
    assert [r['output_index'] for r in s['final_nodes'][:2]] == [10,11]


def test_H_removals_do_not_knock_out_T():
    SH,T,conflicts,proposed=method.prune_candidates({1:10}, {0:11,2:10})
    assert SH == {0} and T == {2:10} and not conflicts and proposed == {2:10}


def test_all_inversion_participants_once_no_rescue():
    s=submit(fixture([[10,14]]*3, [13,11,12]))
    assert s['inversion_participants'] == [0,1,2] and s['new_nodes'] == []
    assert s['T'] == [0,1,2]


def test_propagation_cannot_create_outputs_or_cross_unresolved():
    s=submit(fixture([[10,12]]*3, [11,None,11]))
    assert s['extension_check']['bounds'] == [[11,11]]*3
    assert [r['received_frame'] for r in s['new_nodes']] == [0,2]
    assert s['new_relations'] == []


def test_extension_failure_has_no_additions(monkeypatch):
    b=fixture([[10,12],[11,13]], [11,12])
    monkeypatch.setattr(method,'extend',lambda bounds,fixed:(False,bounds))
    s=submit(b)
    assert s['status']=='NOT_EXTENDABLE' and not s['new_nodes'] and not s['new_relations']
    assert s['old_hard']==b['hard']
    assert method.event(10,10)['output_repeated_index']==10


def test_extend_actual_infeasible():
    assert method.extend([[1,2],[3,4]], {0:3})[0] is False
    assert method.extend([[1,5],[1,5]], {0:4,1:2})[0] is False


@pytest.mark.parametrize('damage', ['relations','coordinate','epsilon','interval','candidate','old_edge','old_node','missing_status'])
def test_invalid_full_semantics_blocks_view(damage):
    b=fixture([[10,10],[11,11],[12,14]], [10,11,13], {0:10,1:11})
    if damage=='relations': del b['hard']['relations']
    elif damage=='coordinate': b['L']['nodes'][2]['received_frame']=7
    elif damage=='epsilon': b['L']['epsilon']=1e-6
    elif damage=='interval': b['hard']['nodes'][2]['interval']=[14,12]
    elif damage=='candidate': b['L']['nodes'][2]['candidate_index']=14
    elif damage=='old_edge': b['hard']['relations'][0]['output_endpoints']=[10,12]
    elif damage=='old_node': b['hard']['nodes'][0]['output_index']=11
    else: del b['L']['nodes'][2]['model_status']
    s=submit(b)
    assert s['status']=='SEMANTIC_UNRESOLVED' and not s['new_nodes'] and not s['new_relations']
    assert s['old_hard']==b['hard']


def test_partial_local_conflict_is_unresolved_not_a_new_edge():
    b=fixture([[10,14]]*4, [11,None,12,13])
    b['hard']['path_status']='PARTIAL_PATH_WITH_LOCAL_CONFLICT'
    b['hard']['nodes'][1]['status']='LOCAL_CONFLICT'
    b['L']['nodes'][1].update(model_status='LOCAL_CONFLICT', L_status='LOCAL_CONFLICT')
    for r in (0,1): b['hard']['relations'][r].update(status='LOCAL_CONFLICT_UNRESOLVED',kind=None)
    s=submit(b)
    assert s['status']=='CONDITIONAL_OUTPUTS_SAVED'
    assert [r['received_left'] for r in s['new_relations']] == [2]


def test_nonempty_old_kind_does_not_hide_new_exact_relation():
    b=fixture([[10,12],[15,17]], [11,16]);assert b['hard']['relations'][0]['kind']=='GAP'
    s=submit(b)
    assert s['new_relations'][0]['output_missing_indices']==[12,13,14,15]


@pytest.mark.parametrize('changes', [dict(delta=[-1,-1],status='LOCAL_CONFLICT_UNRESOLVED'),
    dict(status='LOCAL_CONFLICT_UNRESOLVED'),dict(delta=[10,20]),dict(kind='GAP'),dict(output_missing_indices=[11])])
def test_nonexact_relation_visible_contradictions_block_view(changes):
    b=fixture([[10,12],[11,13]],[11,12]);b['hard']['relations'][0].update(changes)
    s=submit(b)
    assert s['status']=='SEMANTIC_UNRESOLVED' and not s['new_nodes'] and not s['new_relations']
    assert s['old_hard']==b['hard']


@pytest.mark.parametrize('damage', ['missing_L_status', 'bad_L_epsilon', 'bad_nonexact_relation'])
def test_semantic_failure_preserves_valid_old_H_and_relations(damage):
    b=fixture([[10,10],[11,11],[12,14]],[10,11,13],{0:10,1:11})
    if damage=='missing_L_status': del b['L']['nodes'][2]['model_status']
    elif damage=='bad_L_epsilon': b['L']['epsilon']=1e-6
    else: b['hard']['relations'][1]['delta']=[-1,-1]
    s=submit(b)
    assert s['status']=='SEMANTIC_UNRESOLVED' and s['error']
    assert not s['new_nodes'] and not s['new_relations']
    assert s['old_hard']==b['hard'] and s['original_L']==b['L']
    assert s['final_relations']==b['hard']['relations']
    assert len(s['final_nodes'])==3
    assert [(n['output_index'],n['origin']) for n in s['final_nodes']]==[(10,'OLD'),(11,'OLD'),(None,'UNRESOLVED')]
    evaluation=evaluate_view(b,s,[10,11,13],3)
    assert evaluation['preservation'] is True and not evaluation['structural_errors']
    assert evaluation['counts']['old_nodes']=={'CORRECT':2,'UNRESOLVED':1}
    assert evaluation['counts']['old_relations']=={'CORRECT':1,'UNRESOLVED':1}


@pytest.mark.parametrize('damage', ['coordinate','interval','output','missing_status','not_object'])
def test_invalid_old_node_is_not_fabricated_and_valid_neighbor_is_kept(damage):
    b=fixture([[10,10],[11,11],[12,14]],[10,11,13],{0:10,1:11})
    node=b['hard']['nodes'][1]
    if damage=='coordinate': node['received_frame']=8
    elif damage=='interval': node['interval']=[11,10]
    elif damage=='output': node['output_index']='11'
    elif damage=='missing_status': del node['status']
    else: b['hard']['nodes'][1]=None
    s=submit(b)
    assert s['status']=='SEMANTIC_UNRESOLVED' and s['old_node_errors']
    assert not s['new_nodes'] and not s['new_relations']
    assert s['final_nodes'][0]['output_index']==10 and s['final_nodes'][0]['origin']=='OLD'
    invalid=s['final_nodes'][1]
    assert invalid['origin']=='INVALID_OLD_NODE' and invalid['received_frame'] is None and invalid['output_index'] is None
    assert s['old_hard']==b['hard'] and s['final_relations']==b['hard']['relations']


def test_unreadable_old_node_container_is_not_inferred():
    b=fixture([[10,10],[11,11]],[10,11],{0:10,1:11})
    b['hard']['nodes']=None
    s=submit(b)
    assert s['status']=='SEMANTIC_UNRESOLVED' and s['final_nodes']==[] and s['old_node_errors']
    assert not s['new_nodes'] and not s['new_relations'] and s['old_hard']==b['hard']


def test_order_legal_wrong_one_frame_counts_wrong():
    b=fixture([[10,10],[10,13],[10,13],[13,13]], [10,12,12,13], {0:10,3:13})
    s=submit(b);e=evaluate_view(b,s,[10,11,12,13],4)
    assert s['status']=='CONDITIONAL_OUTPUTS_SAVED'
    assert e['counts']['new_nodes']['WRONG']==1 and e['counts']['new_relations']['WRONG']==2


def test_shared_shift_delta_right_absolute_edges_wrong():
    b=fixture([[20,22],[21,23]], [21,22]);e=evaluate_view(b,submit(b),[20,21],2)
    assert e['relations'][0]['new_status']=='WRONG'
    assert e['relations'][0]['new_delta_only_status']=='CORRECT'


def test_wrong_old_H_endpoint_counts_new_edge_wrong():
    b=fixture([[21,21],[21,23]], [21,22], {0:21});e=evaluate_view(b,submit(b),[20,22],2)
    assert e['counts']['old_nodes']['WRONG']==1
    assert e['counts']['new_nodes']['CORRECT']==1
    assert e['counts']['new_relations']['WRONG']==1


def test_same_nondegenerate_node_must_touch_new_correct_edge():
    b=fixture([[10,12],[12,14],[15,15],[16,16]], [11,None,15,16])
    e=evaluate_view(b,submit(b),[11,13,15,16],4)
    assert e['counts']['new_relations']['CORRECT']==1 and e['witnesses']==[]


def make_inputs(tmp_path, entry, truth=True):
    source=tmp_path/'inputs';source.mkdir()
    cfg=dict(entry=entry, views={}, output=str(tmp_path/'out'), truth=str(source/'truth.json') if truth else None)
    for view, frames in zip(rec.VIEWS,rec.LENGTHS):
        if entry=='semantic':
            path=source/(view+'.json');rec.dump(path, fixture([[10,12]]*frames, [11]*frames))
        elif entry=='raw':
            path=source/(view+'.npz');np.savez(path,**raw_fixture(frames))
        else: continue
        cfg['views'][view]=str(path)
    if entry=='media':
        cfg['media']=str(source/'source.mp4');Path(cfg['media']).write_bytes(b'artificial placeholder, never codec')
    if truth: rec.dump(cfg['truth'], {'views': {v:[11]*n for v,n in zip(rec.VIEWS,rec.LENGTHS)}})
    return cfg


def raw_fixture(frames):
    shape=(frames,5,2,63)
    return {k:np.ones(shape,dtype=bool if k=='mask' else np.float64) for k in ('q','d0','d1','soft','mask')}


class FakeBackend:
    """No actual codec, codebook enumeration, BCH or media read."""
    def codebook(self): return 'artificial table'
    @contextmanager
    def receiving_view(self): yield
    def close(self): return dict(peak_resident_bytes=1000,receiver_seconds=0,error=None)
    def probe(self,path,frames): return dict(frames=frames,artificial=True)
    def decode(self,path,frames): return np.arange(frames)[:,None]
    def encode(self, pixels,path): Path(path).write_bytes(b'artificial')
    def read(self,pixels): return raw_fixture(len(pixels))
    def receive(self,raw,table):
        R=len(raw['soft']);return fixture([[10,12]]*R,[11]*R)['hard'],np.ones((R,5,63)),np.ones((R,5,63),bool)
    def C(self,raw,hard,table):
        R=len(raw['soft'])
        return dict(C=np.tile([1.,0.,1.],R),bounds=np.tile([10,12],(R,1)),offsets=np.arange(0,3*R+1,3),
                    support=np.full(R,315),states=['DEFINED']*R,R=R,comparisons=0,graph_abnormal=False)
    def L(self,cost):
        from main.tube_state.b_soft_time_evidence_v1 import local_rank
        return local_rank(cost)
    def candidates(self,hard,cost,local):
        from main.tube_state.b_minimal_l_v1 import candidates
        return candidates(hard,cost,local)
    def semantic(self,hard,local): return method.submit(hard,local)


@pytest.mark.parametrize('entry,total', [('semantic',5),('raw',26),('media',43)])
def test_actual_runner_fixed_entry_counts_and_truth_order(tmp_path,entry,total):
    cfg=make_inputs(tmp_path,entry)
    def post(output,rows,path):
        assert all(r['status'] in rec.TERMINAL for r in rows.values())
        assert rec.load(output/'blind_terminal.json')['rows']==rows
        return evaluate(output,rows,path)
    result=run(cfg,backend_factory=FakeBackend,evaluator=post)
    assert result['failed_rows']==[] and result['errors']=={}
    counts=rec.load(Path(cfg['output'])/'counts.json')
    assert sum(c['started'] for c in counts.values())==total
    assert sum(c['returned'] for c in counts.values())==total
    assert counts['codebook']['started']==(0 if entry=='semantic' else 1)
    assert sum(c['started'] for k,c in counts.items() if k.startswith('codec_'))==(17 if entry=='media' else 0)
    assert rec.load(Path(cfg['output'])/'evaluation.json')['status']=='FINITE_ADDED_SUPPORT'
    before={str(p.relative_to(cfg['output'])):p.read_bytes() for p in Path(cfg['output']).rglob('*') if p.is_file()}
    again=run(dict(entry='raw' if entry=='semantic' else 'semantic',output=cfg['output']), backend_factory=lambda:pytest.fail('replayed'))
    after={str(p.relative_to(cfg['output'])):p.read_bytes() for p in Path(cfg['output']).rglob('*') if p.is_file()}
    assert again['physical_calls_this_invocation']==0 and before==after


def test_failure_consumed_and_saved_failures_evaluated(tmp_path):
    cfg=make_inputs(tmp_path,'semantic')
    class LateFailure(FakeBackend):
        @contextmanager
        def receiving_view(self):
            yield
            from runtime.wan.b_framewise_qim_time_recovery_v1 import ResourceExceeded
            raise ResourceExceeded('artificial deadline after save')
    run(cfg,backend_factory=LateFailure)
    e=rec.load(Path(cfg['output'])/'evaluation.json')
    assert e['status']=='EVIDENCE_INCOMPLETE' and e['counts']['new_nodes']['CORRECT']==822
    assert len(e['failed_rows'])==5
    assert run(cfg,backend_factory=lambda:pytest.fail('retry'))['physical_calls_this_invocation']==0


def test_early_failure_retains_full_denominator(tmp_path):
    cfg=make_inputs(tmp_path,'semantic');Path(cfg['views']['FULL']).write_text('{}')
    result=run(cfg,backend_factory=FakeBackend)
    e=rec.load(Path(cfg['output'])/'evaluation.json')
    assert 'FULL/semantic_submit' in result['failed_rows']
    assert e['counts']['new_nodes']['UNKNOWN']==181 and e['counts']['new_relations']['UNKNOWN']==180
    assert len(e['views']['FULL']['nodes'])==181


def test_semantic_failure_keeps_old_outputs_but_batch_remains_incomplete(tmp_path):
    cfg=make_inputs(tmp_path,'semantic')
    truth={'views':{}}
    for view,frames in zip(rec.VIEWS,rec.LENGTHS):
        b=fixture([[10,10],[11,11]]+[[12,14]]*(frames-2),[10,11]+[13]*(frames-2),{0:10,1:11})
        del b['L']['nodes'][2]['model_status']
        rec.dump(cfg['views'][view],b)
        truth['views'][view]=[10,11]+[13]*(frames-2)
    rec.dump(cfg['truth'],truth)
    result=run(cfg,backend_factory=FakeBackend)
    e=rec.load(Path(cfg['output'])/'evaluation.json')
    assert result['evaluation_status']==e['status']=='EVIDENCE_INCOMPLETE'
    assert e['old_outputs_preserved'] is True and not e['structural_errors']
    assert e['counts']['old_nodes']['CORRECT']==10 and e['counts']['old_relations']['CORRECT']==5
    assert e['counts']['new_nodes']=={'NOT_ADDED':822} and e['counts']['new_relations']=={'NOT_ADDED':817}
    assert run(cfg,backend_factory=lambda:pytest.fail('retry'))['physical_calls_this_invocation']==0


def test_fp64_conversion_precedes_original_hard_fusion(tmp_path):
    from experiments.wan_state_clock.b_conditional_time_output_v1_experiment import _load_raw
    from main.tube_state.b_framewise_qim_time_recovery_v1 import fuse
    raw=raw_fixture(2)
    for key in ('q','d0','d1','soft'): raw[key]=raw[key].astype(np.float32)
    raw['soft'].fill(np.finfo(np.float32).max)
    path=tmp_path/'raw.npz';np.savez(path,**raw)
    loaded=_load_raw(path,2)
    assert all(loaded[k].dtype==np.float64 for k in ('q','d0','d1','soft'))
    merged,known=fuse(loaded['soft'],loaded['mask'])
    assert np.isfinite(merged).all() and known.sum()==630
    raw['soft']=np.full(raw['soft'].shape,np.finfo(np.longdouble).max,dtype=np.longdouble)
    np.savez(path,**raw)
    with pytest.raises(ValueError,match='finite FP64'): _load_raw(path,2)


def test_evaluation_save_failure_cannot_report_support_or_retry(tmp_path,monkeypatch):
    cfg=make_inputs(tmp_path,'semantic');original=rec.dump
    def fail(path,value):
        if Path(path).name=='evaluation.json': raise OSError('artificial evaluation save failure')
        original(path,value)
    monkeypatch.setattr(rec,'dump',fail)
    result=run(cfg,backend_factory=FakeBackend)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE' and 'evaluation' in result['errors']
    assert rec.load(Path(cfg['output'])/'result.json')['evaluation_status']=='EVIDENCE_INCOMPLETE'
    assert not (Path(cfg['output'])/'evaluation.json').exists()
    assert run(cfg,backend_factory=lambda:pytest.fail('retry'))['physical_calls_this_invocation']==0


def test_interrupt_consumes_attempt_without_fallback(tmp_path):
    cfg=make_inputs(tmp_path,'raw')
    class Interrupted(FakeBackend):
        def codebook(self): raise KeyboardInterrupt('artificial interruption')
    result=run(cfg,backend_factory=Interrupted)
    assert result['status']=='INTERRUPTED'
    assert all(r['status'] in rec.TERMINAL for r in rec.load(Path(cfg['output'])/'state.json').values())
    assert run(cfg,backend_factory=lambda:pytest.fail('retry'))['physical_calls_this_invocation']==0


def test_unknown_truth_zero_wrong_is_not_support(tmp_path):
    cfg=make_inputs(tmp_path,'semantic');truth=rec.load(cfg['truth']);truth['views']['FULL'][3]=None;rec.dump(cfg['truth'],truth)
    run(cfg,backend_factory=FakeBackend);e=rec.load(Path(cfg['output'])/'evaluation.json')
    assert e['status']=='EVIDENCE_INCOMPLETE' and e['counts']['new_nodes']['UNKNOWN']==1
    assert e['counts']['new_nodes'].get('WRONG',0)==0


def test_full_rejection_not_pass(tmp_path):
    cfg=make_inputs(tmp_path,'semantic')
    for view,n in zip(rec.VIEWS,rec.LENGTHS):rec.dump(cfg['views'][view],fixture([[10,12]]*n,[None]*n))
    run(cfg,backend_factory=FakeBackend)
    assert rec.load(Path(cfg['output'])/'evaluation.json')['status']=='NO_ADDED_GAIN_OR_TEST_POWER'


def test_no_truth_open_before_blind_terminal(tmp_path,monkeypatch):
    cfg=make_inputs(tmp_path,'semantic');oldload=rec.load;events=[]
    def guarded(path):
        if path and Path(path)==Path(cfg['truth']):
            assert (Path(cfg['output'])/'blind_terminal.json').is_file()
            assert all(row['status'] in rec.TERMINAL for row in oldload(Path(cfg['output'])/'state.json').values())
            events.append('truth')
        return oldload(path)
    monkeypatch.setattr(rec,'load',guarded)
    run(cfg,backend_factory=FakeBackend)
    assert events==['truth']


def test_unconfigured_and_overlap_no_reservation(tmp_path):
    result=run(dict(entry=None,output=str(tmp_path/'out')),backend_factory=lambda:pytest.fail('started'))
    assert result['status']=='UNCONFIGURED' and not (tmp_path/'out').exists()
    cfg=make_inputs(tmp_path,'semantic');cfg['output']=str(tmp_path/'inputs'/'nested')
    assert run(cfg,backend_factory=lambda:pytest.fail('started'))['status']=='PREFLIGHT_UNRESOLVED'
    assert not Path(cfg['output']).exists()


def test_real_runtime_semantic_and_narrow_resource_clock(monkeypatch):
    from runtime.wan.b_conditional_time_output_v1 import Backend, ConditionalResources
    from runtime.wan.b_framewise_qim_time_recovery_v1 import Resources
    observed=[]
    @contextmanager
    def clock(self,seconds): observed.append(seconds);yield
    monkeypatch.setattr(Resources,'receiving',clock)
    resources=object.__new__(ConditionalResources);resources.receive_seconds=3200
    with resources.receiving(600): pass
    assert observed==[100]
    backend=Backend()
    try:
        b=fixture([[10,12],[11,13]],[11,12]);assert backend.semantic(b['hard'],b['L'])['new_nodes']
    finally: assert backend.close()['receiver_total_limit_seconds']==3300


def test_cli_direct_no_git_copy_and_default_unconfigured(tmp_path):
    # A source copy containing only this entry's stdlib path needs no .git, notebook or scripts.
    root=tmp_path/'editable';folder=root/'experiments/wan_state_clock';folder.mkdir(parents=True)
    for name in ('b_conditional_time_output_v1_run.py','b_conditional_time_output_v1_records.py',
                 'b_conditional_time_output_v1_experiment.py','b_minimal_l_v1_records.py'):
        (folder/name).write_text((ROOT/'experiments/wan_state_clock'/name).read_text())
    config=folder/'configs/b_conditional_time_output_v1.json';config.parent.mkdir()
    config.write_text((ROOT/'experiments/wan_state_clock/configs/b_conditional_time_output_v1.json').read_text())
    result=subprocess.run([sys.executable,'-B',str(folder/'b_conditional_time_output_v1_run.py')],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['status']=='UNCONFIGURED'
    output=tmp_path/'used';output.mkdir();rec.dump(output/'attempt.json',dict(entry='semantic'))
    result=subprocess.run([sys.executable,'-B',str(folder/'b_conditional_time_output_v1_run.py'),'--output',str(output),'--config','nonexistent.json'],cwd=tmp_path,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['status']=='READ_ONLY_EXISTING_ATTEMPT'


def test_notebook_builder_static_and_artificial_launcher(tmp_path):
    import nbformat
    def module(name):
        spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    builder=module('build_b_conditional_time_output_v1_notebook');built=builder.build(tmp_path)
    nb=nbformat.read(built['notebook'],as_version=4);nbformat.validate(nb)
    assert nb.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert nb==nbformat.read(ROOT/'notebooks/b_conditional_time_output_v1.ipynb',as_version=4)
    assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
    entry=module('b_conditional_time_output_v1_entry')
    assert entry.launch(None,None)['status']=='UNCONFIGURED'
    cfg=make_inputs(tmp_path,'semantic');config=tmp_path/'config.json';rec.dump(config,cfg)
    # Actual standalone CLI with artificial complete JSON; no media/model/codebook.
    result=entry.launch(ROOT,config)
    assert result['evaluation_status']=='FINITE_ADDED_SUPPORT'
    assert entry.launch('/missing','/missing',cfg['output'])['physical_calls_this_invocation']==0
    namespace=dict(SOURCE_DIRECTORY=None, CONFIG_PATH=None, OUTPUT=cfg['output'])
    exec(builder.RUN,namespace)
    assert namespace['result']['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert namespace['result']['saved']['result']['evaluation_status']=='FINITE_ADDED_SUPPORT'


def test_fixed_attack_and_denominators():
    mappings=attack_mappings()
    assert tuple(map(len,mappings.values()))==rec.LENGTHS
    assert mappings['CROP']==list(range(38,139))
    assert 44 not in mappings['DELETE1'] and mappings['REPEAT1'].count(44)==2
    assert not set((43,44,45))&set(mappings['DELETE3'])
    assert sum(rec.LENGTHS)==822 and sum(n-1 for n in rec.LENGTHS)==817
