"""Bounded CPU/fake checks for the actual local-clip delivery path."""
import io
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import urllib.request
import zipfile

import nbformat
import numpy as np
import pytest
import torch

from experiments.paper_results_v1 import local_clip_diagnostic_v1 as task
from experiments.paper_results_v1 import build_local_clip_diagnostic_v1_colab as builder
from main.tube_state.receiver_controls_v1 import receive_scores
from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span

pytestmark = pytest.mark.unit
torch.set_num_threads(1)


def fixture(tmp_path, missing=False):
    source = tmp_path/'resumable-independent-four-v1'
    rows = []
    for case in task.CASES:
        folder = source/'joint/run_state/clock/clean_mp4'/case
        if not missing:
            folder.mkdir(parents=True)
            # Column-varying rho makes editing both arrays essential.
            rho = np.broadcast_to(np.linspace(1, 2, 181), (181,181)).copy()
            q = 10*np.eye(181)
            np.savez_compressed(folder/'scores.npz', signed_projection=q*rho, rho=rho)
        rows.append(dict(case_id=case, group='CLEAN_MP4',
            media=dict(received_path=str(source/'media'/case/'received.rgb8'),
                       mp4_path=str(source/'media'/case/'candidate.mp4')),
            clock=dict(selector_record=str(folder/'blind_selection.json'),
                       latent_path=str(source/'missing-latent'/case/'framewise.npy'))))
    if not missing:
        task.atomic_json(source/'joint/run_state/residual_joint_state.json',
                         dict(schema='paper-results-v1-residual-joint-v1', key=task.KEY, rows=rows))
    return task.initialize(tmp_path/'output', source, '/unused/model', 'cpu')


class Backend:
    def __init__(self, fail_encode=None, interrupt_read=False):
        self.loads = 0
        self.inputs = []
        self.reads = []
        self.fail_encode = fail_encode
        self.interrupt_read = interrupt_read

    def load(self):
        self.loads += 1
        return object()

    def rgb(self, row):
        return torch.tensor(task.view_source_indices(row['group']), dtype=torch.uint8)[:,None,None,None].expand(-1,1,1,3).clone()

    def encode(self, vae, rgb, observer):
        self.inputs.append(rgb[:,0,0,0].tolist())
        observer('STARTED', None)
        if self.fail_encode == len(self.inputs):
            raise RuntimeError('injected encode failure')
        observer('RETURNED', None)
        return torch.zeros((1,16,(len(rgb)-1)//4+1,40,64))

    def read(self, latent, key, frames, source_map):
        self.reads.append((frames, list(source_map)))
        if self.interrupt_read and len(self.reads) == 1:
            raise KeyboardInterrupt('injected after durable encode')
        # Fake result deliberately differs from A6D39C5E: no positive-result fixture.
        R = min(44, (frames-1)//4)
        signed = np.ones((4,R,240), dtype=np.int8)
        return dict(status='READ', R=R, output_frames=frames, decoded_bits=[1]*32,
                    votes=[dict(ones=30*R, zeros=0, count=30*R) for _ in range(32)],
                    signed_votes=signed, zero_mask=(signed == 0))

    def args(self):
        return dict(load_vae=self.load, encode=self.encode, payload_read=self.read, load_rgb=self.rgb)


@pytest.mark.parametrize('field,value', [
    ('edit_positions_local', [0,10,23,35,47,59,71,88]),
    ('key', 'different-method-key'), ('payload_hex_posthoc_only', '00000000'),
])
def test_known_method_difference_rejected_before_overwriting_records(tmp_path, monkeypatch, field, value):
    store = fixture(tmp_path)
    original = {name: (store.output/name).read_bytes() for name in
                ('fixed_method_semantics.json', 'fixed_diagnostic_spec.json')}
    # Same-method continuation allows locator/device changes.
    task.initialize(store.output, store.data['source_run'], '/another/model', 'cpu')
    monkeypatch.setitem(task.SPEC, field, value)
    with pytest.raises(ValueError, match=field):
        task.initialize(store.output, store.data['source_run'])
    assert all((store.output/name).read_bytes() == data for name, data in original.items())


def test_missing_optional_prior_method_records_do_not_block_resume(tmp_path):
    store = fixture(tmp_path)
    (store.output/'fixed_method_semantics.json').unlink()
    # Older development task with its existing ordinary spec can migrate.
    assert len(task.Store(store.output).data['rows']) == 24
    (store.output/'fixed_method_semantics.json').unlink()
    (store.output/'fixed_diagnostic_spec.json').unlink()
    resumed = task.initialize(store.output, store.data['source_run'])
    assert len(resumed.data['rows']) == 24
    assert resumed.data['method_record_notes'][-1]['status'] == 'PRIOR_METHOD_RECORD_UNAVAILABLE'
    assert task.SPEC['payload_hex_posthoc_only'] not in resumed.path.read_text()


def test_truth_waits_for_complete_fixed_traversals(tmp_path):
    store = fixture(tmp_path)
    backend = Backend()
    task.run_wan(store, **backend.args())  # Direct CLI wan before prepare.
    assert task.report(store)['truth_join'] == 'NOT_JOINED'
    assert task.report(store)['path_truth_join'] == 'NOT_JOINED'
    calls = 0
    def interrupted(signed, rho, receiver):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('injected partial prepare')
        return receive_scores(signed, rho, receiver)
    with pytest.raises(KeyboardInterrupt):
        task.prepare(store, selector=interrupted)
    summary = task.report(store)  # Same as notebook finally reporting.
    rows = task.read_json(store.output/'evaluation_report.json')['rows']
    assert summary['truth_join'] == summary['path_truth_join'] == 'NOT_JOINED'
    assert all(r['path']['status'] == 'NOT_JOINED' for r in rows)
    assert all(r['path'].get('truth_start') is None for r in rows)
    assert summary['by_receiver']['RAW_U']['path_evaluated_rows'] == 0
    task.prepare(task.Store(store.output))
    store = task.Store(store.output)
    summary = task.report(store)
    assert summary['path_truth_join'] == 'POSTHOC_AFTER_FIXED_PREPARE_TRAVERSAL'
    assert summary['by_receiver']['RAW_U']['path_evaluated_rows'] == 12
    assert summary['truth_join'] == 'NOT_JOINED'  # Old empty wan traversal is insufficient.
    task.run_wan(store, **backend.args())
    assert task.report(store)['truth_join'] == 'POSTHOC_AFTER_FIXED_TRAVERSAL'


def test_malformed_optional_method_records_do_not_block_valid_state(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    (store.output/'fixed_method_semantics.json').write_text('{invalid json')
    (store.output/'fixed_diagnostic_spec.json').write_text('[]')
    original_write = task.atomic_json
    def interrupted_sidecar(path, data):
        original_write(path, data)
        if Path(path).name == 'fixed_method_semantics.json':
            raise KeyboardInterrupt('injected immediately after optional sidecar write')
    with monkeypatch.context() as patch:
        patch.setattr(task, 'atomic_json', interrupted_sidecar)
        with pytest.raises(KeyboardInterrupt):
            task.initialize(store.output, store.data['source_run'])
    # A later successful sidecar write cannot erase the earlier unavailable note.
    note = task.read_state(store.path)['method_record_notes'][-1]
    resumed = task.initialize(store.output, store.data['source_run'])
    assert len(resumed.data['rows']) == 24
    assert task.report(resumed)['status_counts'] == {'PENDING': 24}
    assert note['status'] == 'PRIOR_METHOD_RECORD_UNAVAILABLE'
    assert {r['record'] for r in note['unreadable_records']} == {
        'fixed_method_semantics.json', 'fixed_diagnostic_spec.json'}
    assert all(r['reason'] for r in note['unreadable_records'])
    assert note in resumed.data['method_record_notes']
    assert not any(resumed.data[k] for k in task.CALLS)


def test_optional_method_record_write_errors_do_not_block_valid_state(tmp_path, monkeypatch):
    store = fixture(tmp_path)
    original_write = task.atomic_json
    optional = {'fixed_method_semantics.json', 'fixed_diagnostic_spec.json'}
    def write(path, data):
        if Path(path).name in optional:
            raise PermissionError('injected optional record write failure')
        return original_write(path, data)
    monkeypatch.setattr(task, 'atomic_json', write)
    resumed = task.initialize(store.output, store.data['source_run'])
    assert task.report(resumed)['status_counts'] == {'PENDING':24}
    notes = task.read_state(resumed.path)['method_record_notes']
    failed = [n for n in notes if n['status'] == 'OPTIONAL_METHOD_RECORD_WRITE_FAILED']
    assert {n['record'] for n in failed} == optional
    assert all('PermissionError' in n['reason'] for n in failed)
    assert not any(resumed.data[k] for k in task.CALLS)


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'malformed_rows'])
def test_invalid_source_structure_keeps_fixed_rows_without_model_or_media(tmp_path, problem):
    store = fixture(tmp_path)
    source = Path(store.data['source_run'])/'joint/run_state/residual_joint_state.json'
    data = task.read_json(source)
    data['schema'] = 'optional-unrecognized-label'
    if problem == 'duplicate':
        data['rows'] *= 2
    elif problem == 'missing':
        data['rows'] = []
    else:
        data['rows'] = {'wrong': 'type'}
    task.atomic_json(source, data)
    backend = Backend()
    task.run_all(store, **backend.args())
    summary = task.report(store)
    assert summary['status_counts'] == {'MISSING' if problem == 'missing' else 'FAILED': 24}
    assert summary['planned_bits'] == summary['missing_bits'] == 768
    assert summary['truth_join'] == 'POSTHOC_AFTER_FIXED_TRAVERSAL'
    assert not any(store.data[k] for k in task.CALLS)
    assert (backend.loads, len(backend.inputs), len(backend.reads)) == (0,0,0)


@pytest.mark.parametrize('label', [None, 'optional-unrecognized-label'])
def test_optional_source_label_does_not_reject_compatible_data(tmp_path, label):
    store = fixture(tmp_path)
    source = Path(store.data['source_run'])/'joint/run_state/residual_joint_state.json'
    data = task.read_json(source)
    data.pop('schema')
    if label is not None:
        data['schema'] = label
    task.atomic_json(source, data)
    assert all(task._source_record(store, case)['case_id'] == case for case in task.CASES)
    assert store.data['source_format_label'] == label


def test_fixed_views_local_centering_real_input_maps_and_shared_inputs(tmp_path):
    store = fixture(tmp_path)
    calls = []
    def selector(signed, rho, receiver):
        calls.append((signed.copy(), rho.copy(), receiver))
        return receive_scores(signed, rho, receiver)
    task.prepare(store, selector=selector)
    assert len(calls) == 24
    assert [len(task.view_source_indices(g)) for g in task.GROUPS] == [89,81,97]
    assert task.view_source_indices('CROP_DELETE') == [i for i in range(37,126) if i not in [37,48,60,72,84,96,108,125]]
    assert task.view_source_indices('CROP_DUPLICATE').count(37) == 2
    for row, (signed, rho, receiver) in zip(store.data['rows'], calls):
        ids = task.view_source_indices(row['group'])
        assert np.array_equal(signed/rho, (10*np.eye(181))[ids])
        selection = task.read_json(row['selection_record'])
        assert selection['background'] == np.median(signed/rho, axis=0).tolist()
        assert selection['estimate']['path'] == ids
        assert row['operation']['output_frames'] == 89 and row['operation']['wan_support'] == 22
    assert task.report(store)['truth_join'] == 'NOT_JOINED'
    backend = Backend()
    task.run_wan(store, **backend.args())
    summary = task.report(store)
    assert (backend.loads, len(backend.inputs), len(backend.reads)) == (1,12,12)
    owners = [r for r in store.data['rows'] if not r.get('read_owner')]
    for row, actual in zip(owners, backend.inputs):
        source = task.view_source_indices(row['group'])
        assert actual == [source[i] for i in row['operation']['received_index_map']]
    assert summary['status_counts'] == {'READ':24}
    assert summary['planned_bits'] == 768 and summary['complete_pairs'] == 12
    assert summary['paired_error_reduction'] == 0 and summary['by_receiver']['RAW_U']['bit_errors'] > 0
    report = task.read_json(store.output/'evaluation_report.json')
    assert report['rows'][0]['path']['operation']['front_boundary_copies'] == 1
    assert report['rows'][2]['path']['operation']['front_boundary_copies'] == 2
    assert report['rows'][4]['path']['received_distinct_true_source_frames'] == 89
    assert task.SPEC['payload_hex_posthoc_only'] not in store.path.read_text()
    again = Backend()
    task.run_all(task.Store(store.output), **again.args())
    assert (again.loads, len(again.inputs), len(again.reads)) == (0,0,0)


def test_different_maps_with_equal_R_get_separate_physical_encodes(tmp_path):
    store = fixture(tmp_path)
    def selector(signed, rho, receiver):
        result = receive_scores(signed, rho, receiver)
        if receiver == 'RAW_U':
            # Controlled fake wrong path retains the same span/R but changes RGB mapping.
            path = result['estimate']['path']
            path = [path[0], *([path[0]]*(len(path)-2)), path[-1]]
            result['estimate']['path'] = path
            result['operation'] = decode_visible_span(path)
            result['status'] = result['operation']['status']
        return result
    task.prepare(store, selector=selector)
    backend = Backend()
    task.run_wan(store, **backend.args())
    assert len(backend.inputs) == len(backend.reads) == 24
    assert all(a != b for a,b in zip(backend.inputs[::2], backend.inputs[1::2]))


def test_missing_unsupported_unresolved_keep_fixed_rows_without_models(tmp_path):
    store = fixture(tmp_path)
    def selector(signed, rho, receiver):
        if receiver == 'RAW_U':
            result = receive_scores(signed, rho, receiver)
            path = [42]*len(signed)
            result['estimate']['path'] = path
            result['operation'] = decode_visible_span(path)
            result['status'] = 'UNSUPPORTED'
            return result
        return receive_scores(np.zeros_like(signed), rho, receiver)
    task.prepare(store, selector=selector)
    backend = Backend()
    task.run_wan(store, **backend.args())
    summary = task.report(store)
    assert summary['status_counts'] == {'UNSUPPORTED':12,'UNRESOLVED':12}
    assert summary['missing_bits'] == 768 and summary['paired_ber_gain'] is None
    assert backend.loads == 0
    missing = fixture(tmp_path/'missing', missing=True)
    task.run_all(missing, **backend.args())
    assert task.report(missing)['status_counts'] == {'MISSING':24}
    assert not task.report(missing)['needs_wan_model']


def test_failed_attempt_and_interrupted_reader_resume_without_reencoding(tmp_path):
    store = fixture(tmp_path)
    backend = Backend(interrupt_read=True)
    with pytest.raises(KeyboardInterrupt):
        task.run_all(store, **backend.args())
    assert len(backend.inputs) == 1 and task.report(store)['truth_join'] == 'NOT_JOINED'
    retry = Backend()
    task.run_all(task.Store(store.output), **retry.args())
    assert len(retry.inputs) == 11 and len(retry.reads) == 12
    summary = task.report(task.Store(store.output))
    assert summary['cumulative_calls']['wan_encode_clips'] == {'COMPLETE':12}
    assert summary['cumulative_calls']['payload_read_calls'] == {'INTERRUPTED':1,'COMPLETE':12}
    failed = fixture(tmp_path/'failed')
    backend = Backend(fail_encode=1)
    task.run_all(failed, **backend.args())
    assert task.report(failed)['status_counts'] == {'FAILED':1,'READ':23}
    retry = Backend()
    resumed = task.Store(failed.output)
    task.prepare(resumed)
    assert not task.report(resumed)['needs_wan_model']
    task.run_wan(resumed, **retry.args())
    # Its same-view peer already produced the identical operation after the failure.
    assert retry.loads == 0 and not retry.inputs and not retry.reads
    assert task.Store(failed.output).data['rows'][0]['history'][0]['status'] == 'FAILED'


def test_shared_mp4_fallback_and_same_view_RGB_edit(tmp_path, monkeypatch):
    from runtime.wan import variable_rgb_media
    store = fixture(tmp_path)
    task.prepare(store)
    cache, decodes = {}, []
    full = torch.arange(181, dtype=torch.uint8)[:,None,None,None].expand(-1,1,1,3).clone()
    def load(path, shape):
        if str(path) not in cache:
            raise FileNotFoundError(path)
        return cache[str(path)]
    def decode(source, target, expected_frames):
        assert expected_frames == 181
        decodes.append(str(source))
        cache[str(target)] = full
    monkeypatch.setattr(task, '_load_rgb_path', load)
    monkeypatch.setattr(variable_rgb_media, 'decode_published', decode)
    for row in store.data['rows'][:6]:
        assert task._received_rgb(store, row)[:,0,0,0].tolist() == task.view_source_indices(row['group'])
    assert len(decodes) == 1


def test_decode_failure_is_attempted_once_per_source_per_traversal(tmp_path, monkeypatch):
    from runtime.wan import variable_rgb_media
    store = fixture(tmp_path)
    cache, decodes = {}, []
    fail = [True]
    full = torch.arange(181, dtype=torch.uint8)[:,None,None,None].expand(-1,1,1,3).clone()
    def load(path, shape):
        if str(path) not in cache:
            raise FileNotFoundError(path)
        return cache[str(path)]
    def decode(source, target, expected_frames):
        decodes.append(str(source))
        if fail[0]:
            raise OSError('injected unreadable candidate MP4')
        cache[str(target)] = full
    monkeypatch.setattr(task, '_load_rgb_path', load)
    monkeypatch.setattr(variable_rgb_media, 'decode_published', decode)
    first = Backend()
    args = first.args()
    args.pop('load_rgb')
    task.run_all(store, **args)
    assert len(decodes) == 4 and len(set(decodes)) == 4
    assert task.report(store)['status_counts'] == {'FAILED':24}
    assert first.loads == 0 and not first.inputs and not first.reads
    fail[0] = False
    second = Backend()
    args = second.args()
    args.pop('load_rgb')
    task.run_all(task.Store(store.output), **args)
    assert len(decodes) == 8
    summary = task.report(task.Store(store.output))
    assert summary['cumulative_calls']['media_decodes'] == {'FAILED':4, 'COMPLETE':4}
    assert summary['status_counts'] == {'READ':24}


def test_source_score_absence_uses_saved_framewise_only_or_retains_missing(tmp_path, monkeypatch):
    from main.tube_state import video_trajectory_temporal_edit_receiver_v1 as method
    store = fixture(tmp_path)
    case = task.CASES[0]
    root = Path(store.data['source_run'])
    (root/'joint/run_state/clock/clean_mp4'/case/'scores.npz').unlink()
    target = root/'missing-latent'/case/'framewise.npy'
    target.parent.mkdir(parents=True)
    np.save(target, np.zeros((181,4,40,64), dtype=np.float32))
    observed = []
    def saved_score(latent, key):
        observed.append((latent.shape, key))
        return dict(signed_projection=10*np.eye(181), rho=np.ones((181,181)))
    monkeypatch.setattr(method, 'score_framewise', saved_score)
    task.prepare(store)
    assert len(observed) == 3 and all(r['status']=='PREPARED' for r in store.data['rows'])
    assert task.report(store)['new_framewise_encodes'] == 0


def test_delivered_notebook_portable_missing_input_run_all(tmp_path, monkeypatch):
    monkeypatch.setattr(builder,'NOTEBOOK',tmp_path/'clip.ipynb')
    monkeypatch.setattr(builder,'ZIP',tmp_path/'clip.zip')
    builder.build()
    notebook = nbformat.read(builder.NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code = [c.source for c in notebook.cells if c.cell_type == 'code']
    assert code[0] == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for text in code:
        compile(text, 'notebook', 'exec')
    with zipfile.ZipFile(builder.ZIP) as archive:
        assert 'experiments/paper_results_v1/local_clip_diagnostic_v1.adopted.json' in archive.namelist()
        assert not any('.git' in Path(n).parts or '..' in Path(n).parts or n.startswith('/') for n in archive.namelist())
        portable = tmp_path/'portable'
        archive.extractall(portable)
    result = subprocess.run([sys.executable,'-m','experiments.paper_results_v1.local_clip_diagnostic_v1_cli','--help'],
        cwd=tmp_path, env={**os.environ,'PYTHONPATH':str(portable)}, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    colab = ModuleType('google.colab')
    colab.drive = SimpleNamespace(mount=lambda _: None)
    monkeypatch.setitem(sys.modules,'google.colab',colab)
    downloads = []
    def download(url, timeout):
        downloads.append(url)
        return io.BytesIO(builder.ZIP.read_bytes())
    monkeypatch.setattr(urllib.request,'urlopen',download)
    workspace = tmp_path/'content'
    workspace.mkdir()
    namespace = {}
    for text in code:
        exec(compile(text.replace('/content',str(workspace)), 'delivered-notebook', 'exec'), namespace)
    summary = task.read_json(namespace['OUTPUT_ROOT']/'summary.json')
    assert summary['status_counts'] == {'MISSING':24} and summary['planned_rows'] == 24
    assert not summary['cumulative_calls']['vae_loads'] and summary['new_media_encodes'] == 0
    assert downloads == ['https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/paper-results-v1/notebooks/paper_results_v1_local_clip_diagnostic_v1_companion.zip']
    assert 'files.upload' not in '\n'.join(code)
