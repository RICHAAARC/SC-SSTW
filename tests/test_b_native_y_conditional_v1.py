"""Artificial saved inputs only; never open or reserve the real fixed33 assets."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import pytest
from main.tube_state import b_conditional_time_output_v1 as conditional
from runtime.wan.b_native_y_conditional_v1 import Backend, validate_raw, flatten_codebook, validate_hard
from runtime.wan.b_framewise_qim_time_recovery_v1 import ResourceExceeded
from runtime.wan.b_time_layer_diagnostic_v1 import RawLimitExceeded
from experiments.wan_state_clock import b_native_y_conditional_v1_records as rec
from experiments.wan_state_clock.b_native_y_conditional_v1_experiment import run
from experiments.wan_state_clock.b_native_y_conditional_v1_evaluation import comparison, node_status, edge_status, evaluate

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.quick


def raw_value(n, qvalue=0.):
    q = np.full((n,5,2,63), qvalue, dtype=np.float64)
    d0 = np.abs(q-np.round(q/32)*32)
    d1 = np.abs(q-(np.round((q-16)/32)*32+16))
    return dict(q=q, d0=d0, d1=d1, soft=(d0*d0-d1*d1)/256, mask=np.ones(q.shape, dtype=bool))


def graph(bounds, outputs):
    hard = dict(path_status='COMPATIBLE_PATH', qualification='SUPPORTED_RELATION_SET', nodes=[], relations=[])
    for r, (bound, value) in enumerate(zip(bounds, outputs)):
        hard['nodes'].append(dict(received_frame=r, status='LOCAL_LABEL', interval=bound, output_index=value))
    for r in range(len(bounds)-1):
        left, right = bounds[r:r+2]
        delta = [max(0,right[0]-left[1]),right[1]-left[0]]
        row = dict(received_left=r, status='AMBIGUOUS', delta=delta, output_endpoints=None,
            kind='REPEAT' if delta == [0,0] else 'NORMAL' if delta == [1,1] else 'GAP' if delta[0] > 1 else 'AMBIGUOUS',
            output_missing_indices=None, output_repeated_index=None)
        if outputs[r] is not None and outputs[r+1] is not None:
            row.update(conditional.event(outputs[r], outputs[r+1]))
            row['delta'] = [row['delta']]*2
            row['status'] = 'EXACT'
        hard['relations'].append(row)
    return hard


def table_dict():
    result = {}
    for lane in range(5):
        js = np.arange(lane,92672,5,dtype=np.int64)
        words = np.zeros(len(js), dtype=np.uint64)
        words[js == 11] = (1 << 63)-1
        result.update({f'j_{lane}': js, f'word_{lane}': words})
    return result


@pytest.fixture
def artificial(tmp_path):
    source, old = tmp_path/'Y', tmp_path/'historical'
    source.mkdir()
    np.savez(source/'shared_codebook.npz', **table_dict())
    truth = {'views': {}}
    for view, n in zip(rec.VIEWS, rec.LENGTHS):
        mapping = [min(r+4,180) for r in range(n)]
        bounds, outputs = [[t,t] for t in mapping], mapping.copy()
        bounds[7], outputs[7] = [10,11], None
        hard = graph(bounds, outputs)
        folder = source/'units'/view
        rec.dump(folder/'path.json', hard)
        rec.dump(folder/'probe.json', dict(profile=dict(status='MODEL_UNCONFIRMED')))
        raw = raw_value(n)
        raw['q'][7,1] = 16
        raw['d0'][7,1] = 16
        raw['d1'][7,1] = 0
        raw['soft'][7,1] = 1
        # Poisoned alternative views must never reach the C adapter.
        raw.update(known=np.zeros(raw['mask'].shape,bool), F_known=np.zeros((n,5,63),bool), F_soft=np.zeros((n,5,63)))
        np.savez(folder/'readout.npz', **raw)
        rec.dump(old/view/'saved_public.json', {'semantic.json': {'hard': hard}, 'submission.json': {
            'final_nodes': hard['nodes'], 'final_relations': hard['relations']}})
        truth['views'][view] = dict(nodes=[dict(truth=t) for t in mapping])
    rec.dump(tmp_path/'truth-input'/'truth.json', truth)
    return dict(new_y_root=str(source), historical_public_root=str(old), truth=str(tmp_path/'truth-input'/'truth.json'))


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_raw_original_values_zero_mask_and_codebook_semantics():
    raw = raw_value(2, 8.)
    raw['mask'][0,0,0,0] = False
    original_soft, original_mask = raw['soft'], raw['mask']
    raw['soft'][1,0,0,0] = 5e-13  # allclose accepts; the saved tiny nonzero stays unchanged.
    assert validate_raw(raw,2)['soft'] is original_soft
    assert raw['mask'] is original_mask and raw['mask'][1,0,0,1] and raw['soft'][1,0,0,1] == 0
    bad = deepcopy(raw);bad['soft'][0,0,0,0] = 1e-6
    with pytest.raises(ValueError,match='soft unit'):validate_raw(bad,2)
    bad = deepcopy(raw);bad['q'] = bad['q'].astype(np.float32)
    with pytest.raises(ValueError,match='FP64'):validate_raw(bad,2)
    bad = deepcopy(raw);bad['mask'] = bad['mask'].astype(np.uint8)
    with pytest.raises(ValueError,match='boolean'):validate_raw(bad,2)
    table = table_dict();flat = flatten_codebook(table)
    assert len(flat) == 92672 and int(flat[11]) == (1 << 63)-1
    table['j_0'][1] = table['j_0'][0]
    with pytest.raises(ValueError,match='duplicate'):flatten_codebook(table)


@pytest.mark.parametrize('sign,expected', [(1.,11),(-1.,10),(0.,None)])
def test_original_C_L_submit_unique_wrong_or_tie_and_startup(sign,expected):
    hard = graph([[0,3],[10,10],[10,11],[12,12]], [None,10,None,12])
    g = validate_hard(hard,4)
    raw = raw_value(4,8.)
    raw['soft'][2,1] = sign
    # This adapter test isolates original C semantics from the separate file validator.
    backend = Backend()
    try:
        cost = backend.C(raw,g,flatten_codebook(table_dict()))
        local, candidates = backend.L(g,cost)
        submitted = backend.submit(hard,candidates)
        assert candidates['nodes'][0]['model_status'] == 'STARTUP_MODEL_UNDEFINED'
        assert candidates['nodes'][0]['candidate_index'] is None
        assert candidates['nodes'][2]['candidate_index'] == expected
        assert submitted['final_nodes'][2]['output_index'] == expected
        assert [submitted['final_nodes'][r]['output_index'] for r in (1,3)] == [10,12]
        assert cost['comparisons'] == 16 and cost['support'][2] == 315
        if sign == 0:assert candidates['nodes'][2]['best_count'] == 2
        damaged = deepcopy(hard);damaged['relations'][1]['delta'] = [20,20]
        with pytest.raises(ValueError,match='delta'):validate_hard(damaged,4)
    finally:backend.close()


def test_fixed33_actual_saved_adapter_blind_order_and_consumed(artificial,tmp_path,monkeypatch):
    output = tmp_path/'run'
    opened = []
    original = Path.open
    def guarded(path,*args,**kwargs):
        if path.is_relative_to(Path(artificial['historical_public_root'])) or path == Path(artificial['truth']):
            state = rec.load(output/'state.json')
            rec.require_blind(output,state)
            if path == Path(artificial['truth']):
                assert all(state[v+'/historical_public_load']['status'] in rec.TERMINAL for v in rec.VIEWS)
            opened.append(str(path))
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    result = run(artificial,output)
    assert result['errors'] == {} and result['evaluation_status'] == 'DIAGNOSTIC_COMPLETE',result
    assert result['computations_this_invocation'] == 33 and result['post_calls_this_invocation'] == 1
    assert len(opened) == 6 and len(rec.blind_keys()) == 26
    counts = rec.load(output/'counts.json')
    assert all(v['planned'] == v['started'] == v['returned'] for v in counts.values())
    rows = read_rows(output/'post/nodes.jsonl')
    assert len(rows) == 822 and len(read_rows(output/'post/edges.jsonl')) == 817
    assert all(r['layers']['Y_conditional_final']['output_index'] == 11 for r in rows if r['received_frame'] == 7)
    assert rec.load(output/'resources.json')['receiver_total_limit_seconds'] == 6300
    before = {str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    resumed = run({},output,backend_factory=lambda:pytest.fail('replayed'))
    assert resumed['computations_this_invocation'] == resumed['post_calls_this_invocation'] == 0
    assert before == {str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()} and len(opened) == 6


def test_local_raw_and_historical_failures_keep_denominators_unknown(artificial,tmp_path):
    class Fail(Backend):
        def raw(self,path,frames):
            if path.parent.name == 'FULL':raise ValueError('artificial bad raw')
            return super().raw(path,frames)
        def public(self,path):
            if path.parent.name == 'CROP':raise OSError('artificial unavailable comparison')
            value = super().public(path)
            if path.parent.name == 'DELETE1':value['semantic.json']['hard'] = None
            return value
    output = tmp_path/'failure'
    result = run(artificial,output,backend_factory=Fail)
    assert result['evaluation_status'] == 'EVIDENCE_INCOMPLETE'
    state = rec.load(output/'state.json')
    assert state['FULL/C']['started'] == 0 and state['DELETE3/conditional_submit']['status'] == 'RETURNED'
    rows = read_rows(output/'post/nodes.jsonl')
    assert len(rows) == 822
    assert all(r['statuses']['Y_conditional_final'] == 'NOT_EXECUTED' for r in rows if r['view'] == 'FULL')
    row = next(r for r in rows if r['view'] == 'DELETE1' and r['received_frame'] == 8)
    assert row['statuses']['historical_H'] == 'UNKNOWN' and row['statuses']['historical_final'] == 'CORRECT'
    assert row['historical_L_added_correct'] is None and row['comparisons']['historical_H']['new_wrong'] is None
    assert comparison('UNKNOWN','WRONG')['new_wrong'] is None
    assert node_status({},10,0) == 'UNKNOWN'


def test_legal_new_wrong_is_counterexample_not_repaired(artificial,tmp_path):
    class Wrong(Backend):
        def raw(self,path,frames):
            raw = super().raw(path,frames)
            # Artificial injected observation, never a real saved-array edit.
            raw['soft'][7,1] = -1
            return raw
    output = tmp_path/'counterexample'
    run(artificial,output,backend_factory=Wrong)
    report = rec.load(output/'evaluation_summary.json')
    assert report['outcome'] == 'COUNTEREXAMPLE_STOP'
    assert report['counts']['node_added/WRONG'] == 5
    assert report['counts']['edge_added/WRONG'] == 10
    assert not report['structural_errors']
    assert run({},output,backend_factory=lambda:pytest.fail('repair rerun'))['computations_this_invocation'] == 0


@pytest.mark.parametrize('failure', ['global','interrupt','view_limit','blind_save','resources_save','late_close'])
def test_stops_save_failure_and_late_resource_consumption(artificial,tmp_path,monkeypatch,failure):
    output = tmp_path/failure
    original = rec.dump
    def dump(path,value):
        if Path(path).name == {'blind_save':'blind_terminal.json','resources_save':'resources.json'}.get(failure):
            raise OSError('artificial save failure')
        original(path,value)
    monkeypatch.setattr(rec,'dump',dump)
    class Fail(Backend):
        def codebook(self,path):
            if failure == 'global':raise ResourceExceeded('artificial global limit')
            if failure == 'interrupt':raise KeyboardInterrupt('artificial stop')
            return super().codebook(path)
        def raw(self,path,frames):
            if failure == 'view_limit' and path.parent.name == 'FULL':raise RawLimitExceeded('artificial view limit')
            return super().raw(path,frames)
        def close(self):
            value = super().close()
            if failure == 'late_close':value['error'] = 'artificial late close failure'
            return value
    if failure == 'resources_save':
        with pytest.raises(OSError,match='consumed'):run(artificial,output,backend_factory=Fail)
    else:run(artificial,output,backend_factory=Fail)
    result = rec.load(output/'result.json')
    assert result['evaluation_status'] == 'EVIDENCE_INCOMPLETE'
    state = rec.load(output/'state.json')
    if failure in ('global','interrupt','blind_save'):
        assert state['post']['started'] == 0
        assert all(state[v+'/historical_public_load']['started'] == 0 for v in rec.VIEWS)
    if failure == 'view_limit':assert state['DELETE3/conditional_submit']['status'] == 'RETURNED'
    assert run({},output,backend_factory=lambda:pytest.fail('replay'))['computations_this_invocation'] == 0
    assert rec.summary(rec.recover(output),output)['evaluation_status'] == 'EVIDENCE_INCOMPLETE'


def test_resource_clocks_and_no_git_stdlib_recovery(tmp_path):
    import shutil
    import time
    backend = Backend()
    try:
        backend.started = time.monotonic()-4000
        backend.check()  # No old conditional 3300-second cap.
        backend.view_started = time.monotonic()-601
        with pytest.raises(RawLimitExceeded):backend.check()
        backend.end_view();backend.started = time.monotonic()-6301
        with pytest.raises(ResourceExceeded):backend.check()
    finally:backend.close()
    root = tmp_path/'editable'
    folder = root/'experiments/wan_state_clock'
    folder.mkdir(parents=True)
    for name in ('b_native_y_conditional_v1_run.py','b_native_y_conditional_v1_records.py',
                 'b_time_layer_diagnostic_v1_records.py','b_minimal_l_v1_records.py'):
        shutil.copyfile(ROOT/'experiments/wan_state_clock'/name,folder/name)
    output = tmp_path/'consumed'
    rec.dump(output/'attempt.json',dict(status='RESERVED_CONSUMED'))
    rows = rec.plan();rows['FULL/C'].update(status='STARTED',started=1)
    rec.dump(output/'state.json',rows)
    result = subprocess.run([sys.executable,'-S','-B',str(folder/'b_native_y_conditional_v1_run.py'),
        '--output',str(output),'--config','missing.json'],cwd=tmp_path,capture_output=True,text=True,check=True)
    summary = json.loads(result.stdout)
    assert summary['computations_this_invocation'] == 0 and summary['receipt_statuses']['UNFINISHED_RECEIPT'] == 1


@pytest.mark.parametrize('truth_missing', [False,True])
def test_v2_post_known_new_wrong_and_missing_history_are_independent(artificial,tmp_path,truth_missing):
    # Post-only synthetic bundle: no NPZ load or C/L scoring in this regression.
    output = tmp_path/'post-only'
    state = rec.plan()
    for row in state.values():row.update(status='RETURNED',started=1,returned=1)
    state['FULL/historical_public_load'].update(status='FAILED',returned=0)
    state['post'].update(status='STARTED',returned=0)
    rec.dump(output/'blind_terminal.json',dict(status='ALL_BLIND_DIAGNOSTICS_TERMINAL',rows={k:state[k] for k in rec.blind_keys()}))
    data = {}
    for view in rec.VIEWS:
        hard = rec.load(Path(artificial['new_y_root'])/'units'/view/'path.json')
        candidates = dict(qualification=hard['qualification'],epsilon=1e-12,nodes=[
            dict(received_frame=r,model_status='DEFINED',L_status='SOFT_MODEL_BEST',best_count=1,
                 model_index=node['interval'][0],candidate_index=node['interval'][0]) for r,node in enumerate(hard['nodes'])])
        data[view] = dict(hard=hard,candidates=candidates,submission=conditional.submit(hard,candidates))
        if view != 'FULL':data[view]['historical'] = rec.load(Path(artificial['historical_public_root'])/view/'saved_public.json')
    if truth_missing:
        truth = rec.load(artificial['truth'])
        for view in rec.VIEWS:truth['views'][view]['nodes'][7]['truth'] = None
        rec.dump(artificial['truth'],truth)
    report = evaluate(output,artificial,data,state)
    assert report['status'] == 'EVIDENCE_INCOMPLETE'
    assert report['counterexample_stop'] is (not truth_missing)
    assert report['confirmed_legal_new_wrong'] == dict(nodes=0 if truth_missing else 5,edges=0 if truth_missing else 10)
    assert report['outcome'] == ('EVIDENCE_INCOMPLETE' if truth_missing else 'COUNTEREXAMPLE_STOP')
    assert report['counts']['node/historical_H/UNKNOWN'] >= 181


@pytest.mark.parametrize('changes', [dict(delta='1'),dict(delta=True),dict(delta=1.),dict(delta=[1]),
    dict(delta=[True,True]),dict(delta=[-1,1]),dict(delta=[2,1]),dict(kind='BROKEN'),
    dict(output_missing_indices='5'),dict(output_missing_indices=[False]),dict(output_repeated_index=1.)])
def test_v2_malformed_edge_is_unknown_not_a_gain(changes):
    row = dict(received_left=0,output_endpoints=[5,6],delta=1,kind='NORMAL',
               output_missing_indices=None,output_repeated_index=None)
    row.update(changes)
    status = edge_status(row,5,6,0)['status']
    assert status == 'UNKNOWN'
    assert comparison(status,'CORRECT')['added_correct'] is None


@pytest.mark.parametrize('pair,delta', [([4,6],[2,2]),([5,6],2)])
def test_v2_legal_but_wrong_event_values_remain_wrong(pair,delta):
    row = dict(received_left=0,output_endpoints=pair,delta=delta,kind='GAP',
               output_missing_indices=[5],output_repeated_index=None)
    assert edge_status(row,5,6,0)['status'] == 'WRONG'
    assert edge_status(row,None,None,0)['status'] == 'UNKNOWN'
