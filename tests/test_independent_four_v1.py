"""CPU/fake checks of the adopted orchestration, never real models or codec."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import nbformat
import numpy as np
import pytest
import torch

from experiments.paper_results_v1 import independent_four_v1 as task
from experiments.paper_results_v1 import build_independent_four_v1_colab as builder

pytestmark = pytest.mark.unit


def test_generate_actual_prefix_survives_failure_and_durable_arm_resume(tmp_path, monkeypatch):
    from runtime.wan import video_trajectory_conditional_joint_v1 as runtime
    cfg = task.configuration()
    task.initialize(tmp_path, cfg)
    source = task.SourceStore(tmp_path, cfg, session_id='first', stage='generate')
    case = task.CASES[0]
    monkeypatch.setattr(runtime, 'execution_device_dtype', lambda: ('cpu', torch.float32))
    monkeypatch.setattr(runtime.generation, 'prepare_generation', lambda *a, **k:
                        (SimpleNamespace(scheduler=SimpleNamespace()), torch.zeros(1), None, None, torch.float32))
    monkeypatch.setattr(runtime, 'release', lambda: None)
    def trajectory(*args, **kwargs):
        count, record = args[-2:]
        for i in range(50):
            for name in ('transformer_conditional', 'transformer_unconditional', 'native_step'):
                count(name, False); count(name, True)
            record(dict(index=i, cursor_after=i+1))
        return torch.zeros(1), {}
    monkeypatch.setattr(task.real_eval, 'run_off_trajectory', trajectory)
    def fail_payload(*args):
        count, record = args[-2:]
        for i in range(3):
            count('transformer_conditional', False); count('transformer_conditional', True)
            count('transformer_unconditional', False); count('transformer_unconditional', True)
            count('native_step', False); count('native_step', True)
        count('transformer_conditional', False)
        raise KeyboardInterrupt('simulated interruption inside fourth DiT call')
    monkeypatch.setattr(runtime, 'run_trajectory', fail_payload)
    with pytest.raises(KeyboardInterrupt):
        task.run_stage(tmp_path, cfg, 'generate:' + case, session_id='first')
    counts = task._call_log_summary(source.output / 'actual_calls.jsonl')['operations']
    assert counts['transformer_conditional'] == {'attempted': 54, 'returned': 53, 'interrupted': 1}
    assert counts['native_step']['returned'] == 53
    saved = task.read_state(source.path)
    assert task.real_eval._artifact(SimpleNamespace(data=saved), f'{case}/OFF_NATIVE/TERMINAL')['status'] == 'AVAILABLE'
    monkeypatch.setattr(runtime, 'run_trajectory', trajectory)
    # Completed OFF is skipped; only the uncommitted PAYLOAD trajectory repeats.
    task.run_stage(tmp_path, cfg, 'generate:' + case, session_id='second')
    second = task._call_log_summary(source.output / 'actual_calls.jsonl', 'second')['operations']
    assert second['native_step']['returned'] == 50
    assert second['transformer_conditional']['returned'] == 50
    assert second['generation_pipeline_load']['returned'] == 1


def test_fixed_cohort_semantics_and_nonblocking_provenance(tmp_path):
    cfg = task.configuration()
    cfg['models']['wan']['revision'] = 'optional-note-not-an-admission-gate'
    path = tmp_path / 'config.json'; task.joint.atomic_json(path, cfg)
    assert task.configuration(path)['cases'] == cfg['cases']
    cfg['cases'][0]['seed'] += 1
    task.joint.atomic_json(path, cfg)
    with pytest.raises(ValueError, match='cases semantics'):
        task.configuration(path)


def test_framewise_decode_batch_observer_records_failed_partial_batch():
    from runtime.wan import framewise_autoencoder_kl as framewise
    class FakeVAE:
        config = SimpleNamespace(scaling_factor=0.18215)
        def parameters(self): return iter([torch.zeros(1)])
        def decode(self, latent):
            if len(latent) == 1: raise RuntimeError('fake final batch failure')
            return SimpleNamespace(sample=torch.zeros(len(latent), 3, 2, 2))
    events = []
    with pytest.raises(RuntimeError, match='fake final'):
        framewise.decode_rgb_frames(FakeVAE(), torch.zeros(9, 4, 2, 2), batch_frames=8,
                                    batch_observer=lambda event, start, stop, error: events.append((event, start, stop)))
    assert events == [('STARTED', 0, 8), ('RETURNED', 0, 8), ('STARTED', 8, 9), ('FAILED', 8, 9)]


def test_new_cohort_k0_unsupported_is_retained_in_controls(tmp_path):
    cfg = task.configuration(); run = task.initialize(tmp_path, cfg)
    rows = task.joint._new_rows(task.CASES)
    for row in rows:
        row.update(status='UNSUPPORTED', reason='fake unsupported visible span')
        row['clock'].update(status='UNSUPPORTED', reason='fake unsupported visible span')
        row['payload'].update(status='UNSUPPORTED')
    task.joint.atomic_json(tmp_path / 'joint/run_state/residual_joint_state.json',
        dict(schema=task.joint.SCHEMA, key=cfg['keys']['K0'], rows=rows, sources={}))
    store = task.controls.initialize(tmp_path / 'joint', tmp_path / 'source', tmp_path / 'controls/run_state',
                                    run.effective, cases=task.CASES)
    positives = [r for r in store.data['rows'] if r['arm'] == 'WM_K0_REUSED']
    assert len(positives) == 8 and {r['status'] for r in positives} == {'UNSUPPORTED'}
    assert len(store.data['rows']) == 24 and len(store.data['quality_rows']) == 24


def fake_pipeline(tmp_path, monkeypatch, *, same_operation=False):
    cfg = task.configuration()
    for module in (task.joint, task.controls, task.residual):
        monkeypatch.setattr(module, 'RGB_SHAPE', [181, 1, 1, 3])
    monkeypatch.setattr(task.joint, 'DELETE_SHAPE', [163, 1, 1, 3])
    monkeypatch.setattr(task.controls, 'DELETE_SHAPE', [163, 1, 1, 3])
    calls = dict(codec=[], keys=[], wan=[])
    def save_rgb(source, ident, value):
        path = source.output / (ident.replace('/', '_') + '.rgb8')
        np.full((181, 1, 1, 3), value, np.uint8).tofile(path)
        source.artifact(ident, status='AVAILABLE', path=str(path), shape=[181, 1, 1, 3], bytes=543, dtype='uint8')
    def generate(source, config, case):
        for arm in ('OFF_NATIVE', 'PAYLOAD_NATIVE'):
            path = source.output / (case + arm + '.pt'); torch.save(torch.zeros(1), path)
            source.artifact(f'{case}/{arm}/TERMINAL', status='AVAILABLE', path=str(path))
    def decode(source, config, case):
        save_rgb(source, f'{case}/OFF_NATIVE/PRE', 3)
        save_rgb(source, f'{case}/PAYLOAD_NATIVE/PRE', 10)
    def framewise(source, config, case):
        save_rgb(source, f'{case}/PAYLOAD_FRAMEWISE_RECON/PRE', 9)
        save_rgb(source, f'{case}/PAYLOAD_FRAMEWISE_M05/PRE', 11)
    monkeypatch.setattr(task.real_eval, 'phase_generate', generate)
    monkeypatch.setattr(task.real_eval, 'phase_decode', decode)
    monkeypatch.setattr(task.real_eval, 'phase_framewise', framewise)
    def roundtrip(raster, digest, mp4, received, *, count, event):
        calls['codec'].append(str(raster))
        mp4, received = Path(mp4), Path(received)
        mp4.parent.mkdir(parents=True, exist_ok=True); mp4.write_bytes(b'FAKE_CODEC')
        received.write_bytes(Path(raster).read_bytes())
        for name in ('mp4_save', 'mp4_probe', 'mp4_readback'):
            count(name, False); count(name, True)
        event('mp4', {'path': str(mp4), 'status': 'SAVED'})
        event('rgb24', {'path': str(received), 'shape': [181, 1, 1, 3], 'bytes': 543, 'status': 'SAVED'})
    from runtime.wan import fixed_rgb_media
    monkeypatch.setattr(fixed_rgb_media, 'mp4_roundtrip', roundtrip)
    from runtime.wan import variable_rgb_media
    def variable_roundtrip(rgb, work, *, media_output):
        work, media_output = Path(work), Path(media_output)
        work.mkdir(parents=True, exist_ok=True); media_output.mkdir(parents=True, exist_ok=True)
        path = work / 'received.rgb8'; rgb.numpy().tofile(path)
        mp4 = media_output / 'source.mp4'; mp4.write_bytes(b'FAKE_CODEC')
        calls['codec'].append('residual_candidate')
        return rgb, {'mp4': {'path': str(mp4)}}
    monkeypatch.setattr(variable_rgb_media, 'roundtrip', variable_roundtrip)
    from experiments.paper_results_v1 import attack_matrix
    def edit(rgb, spec):
        assert spec == task.joint.delete_spec()
        keep = [i for i in range(181) if i not in spec['positions']]
        return rgb[keep], dict(status='APPLIED', positions=spec['positions'])
    monkeypatch.setattr(attack_matrix, 'apply_edit_rgb8', edit)
    class Framewise:
        def encode(self, rgb, batch_observer=None):
            for start in range(0, len(rgb), 8):
                batch_observer('STARTED', start, min(start+8, len(rgb)), None)
                batch_observer('RETURNED', start, min(start+8, len(rgb)), None)
            return np.zeros((len(rgb), 4, 40, 64), np.float32)
        def close(self): pass
    def select(latent, key):
        calls['keys'].append(key)
        mapping = [min(i, len(latent)-1) for i in range(181)]
        if key == cfg['keys']['K1'] and not same_operation: mapping[0] = 1
        op = dict(status='SUPPORTED', output_frames=181, wan_support=44,
                  received_index_map=mapping, source_coordinate_map=[[{'source_index': i, 'weight': 1.0}] for i in range(181)])
        return dict(signed_projection=np.zeros((1, 1)), rho=np.zeros((1, 1))), dict(
            estimate=dict(status='ESTIMATED', score_gap=1.0), operation=op, reason=None)
    def encode(vae, rgb, observer):
        calls['wan'].append(len(rgb)); observer('RETURNED', None)
        return torch.zeros((1, 16, 46, 40, 64))
    def read(latent, key, frames, source_map):
        return dict(status='READ', decoded_bits=cfg['payload_bits'], signed_votes=np.zeros((1,)),
                    zero_mask=np.zeros((1,), bool), votes=[], R=44, output_frames=181, original_reader_match=True)
    for module in (task.joint, task.controls):
        monkeypatch.setattr(module, '_default_framewise_loader', lambda store: Framewise())
        monkeypatch.setattr(module, '_select_clock', select)
        monkeypatch.setattr(module, '_default_wan_loader', lambda store: object())
        monkeypatch.setattr(module, '_default_wan_encode', encode)
        monkeypatch.setattr(module, '_default_payload_read', read)
    return cfg, calls


def test_complete_new_cohort_twelve_codec_and_resume(tmp_path, monkeypatch):
    cfg, calls = fake_pipeline(tmp_path, monkeypatch)
    result = task.run_all(tmp_path, cfg, invoke=lambda stage: task.run_stage(tmp_path, cfg, stage, 'fake'))
    assert result['status'] == 'COMPLETE'
    assert len(calls['codec']) == 12
    assert not any('FRAMEWISE_RECON' in p or 'FRAMEWISE_M05' in p for p in calls['codec'])
    assert len(calls['keys']) == 24 and calls['keys'].count('watermark-wrong') == 8
    assert len(calls['wan']) == 24  # same R but K1's complete map differs
    final = task.read(tmp_path / 'evaluation_report.json')
    assert len(final['rows']) == 24 and {r['case_id'] for r in final['rows']} == set(task.CASES)
    assert len(final['quality_rows']) == 24
    assert len(list((tmp_path / 'controls/run_state/quality').rglob('*.png'))) == 192
    assert result['actual_calls_cumulative']['joint']['framewise_physical_batches']['COMPLETE'] == 176
    assert result['actual_calls_cumulative']['controls']['framewise_physical_batches']['COMPLETE'] == 176
    task.run_all(tmp_path, cfg, invoke=lambda stage: pytest.fail('completed stage repeated: ' + stage))
    assert len(calls['codec']) == 12


def test_interrupt_resume_and_failures_keep_fixed_denominators(tmp_path):
    cfg = task.configuration()
    seen = []
    def interrupt(stage):
        seen.append(stage)
        if len(seen) == 2: raise KeyboardInterrupt('fake interruption')
    with pytest.raises(KeyboardInterrupt): task.run_all(tmp_path, cfg, invoke=interrupt)
    resumed = []
    result = task.run_all(tmp_path, cfg, invoke=lambda stage: resumed.append(stage) or 1)
    assert task.STAGES[0] not in resumed and resumed[0] == task.STAGES[1]
    assert result['status'] == 'COMPLETE_WITH_RETAINED_FAILURES'
    assert result['receiver_status_counts'] == {'MISSING': 24}
    assert result['quality_status_counts'] == {'MISSING': 24}
    assert task.read(tmp_path / 'independent_four_state.json')['sessions'][0]['status'] == 'INTERRUPTED'


def test_portable_cli_notebook_and_builder_agree(tmp_path):
    builder.build()
    nb = nbformat.read(builder.NOTEBOOK, as_version=4); nbformat.validate(nb)
    assert nb.cells[0].source == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in nb.cells:
        if cell.cell_type == 'code': compile(cell.source, cell.id, 'exec')
    text = '\n'.join(x.source for x in nb.cells)
    assert 'resumable-independent-four-v1' in text and "'--phase', 'all'" in text
    assert 'SOURCE_JOINT_ID' not in text
    extracted = tmp_path / 'source'; extracted.mkdir()
    with zipfile.ZipFile(builder.ZIP) as archive:
        archive.extractall(extracted)
        assert archive.read('experiments/paper_results_v1/independent_four_v1.py') == Path(task.__file__).read_bytes()
    result = subprocess.run([sys.executable, '-m', 'experiments.paper_results_v1.independent_four_v1_cli',
                             '--output', str(tmp_path / 'portable-init'), '--phase', 'init'],
                            cwd=extracted, env={**os.environ, 'PYTHONPATH': str(extracted)}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['fixed_denominator']['receiver_rows'] == 24


def test_saved_method_fields_prevent_mixing_but_provenance_is_optional(tmp_path):
    cfg = task.configuration()
    task.initialize(tmp_path, cfg)
    optional = copy.deepcopy(cfg)
    optional['cases'][0]['source'] = 'changed optional source note'
    optional['cases'][0]['status'] = 'optional status record'
    optional['models']['wan'].pop('revision')
    optional['models']['wan']['local_snapshot_path'] = '/another/readable/cache'
    optional['budget'] = {'optional_note': 'not an actual counter'}
    path = tmp_path / 'optional.json'; task.joint.atomic_json(path, optional)
    checked = task.configuration(path)
    task.initialize(tmp_path, checked)
    assert checked['budget'] == cfg['budget']
    changed = copy.deepcopy(cfg); changed['cases'][0]['seed'] += 1
    with pytest.raises(ValueError, match='saved fixed method semantics'):
        task.Run(tmp_path, changed)
    with pytest.raises(ValueError, match='saved fixed method semantics'):
        task.SourceStore(tmp_path, changed)
    changed = copy.deepcopy(cfg); changed['models']['framewise']['batch_frames'] = 4
    task.joint.atomic_json(path, changed)
    with pytest.raises(ValueError, match='framewise_batch_frames'):
        task.configuration(path)
    # Older optional records may be absent; absence does not become an identity gate.
    state_path = tmp_path / 'independent_four_state.json'
    state = task.read_state(state_path); state.pop('fixed_method_semantics')
    task.joint.atomic_json(state_path, state)
    (tmp_path / 'effective_independent_four_config.json').unlink()
    recovered = task.Run(tmp_path, cfg)
    assert recovered.data['method_record_notes'][-1]['status'] == 'PRIOR_METHOD_RECORD_UNAVAILABLE'


def test_reference_codec_interrupt_retires_prefix_and_appends_attempt(tmp_path, monkeypatch):
    cfg, calls = fake_pipeline(tmp_path, monkeypatch)
    run = task.initialize(tmp_path, cfg); case = task.CASES[0]
    task.run_stage(tmp_path, cfg, 'generate:' + case)
    task.run_stage(tmp_path, cfg, 'decode:' + case)
    def interrupted(*args, count, event):
        count('mp4_save', False); count('mp4_save', True)
        event('mp4', {'path': 'fake-partial.mp4'})
        count('mp4_probe', False)
        raise KeyboardInterrupt('interrupted inside probe')
    with pytest.raises(KeyboardInterrupt): task.reference_codec(run, case, roundtrip=interrupted)
    source = task.SourceStore(tmp_path, cfg)
    attempt = source.data['records'][case]['reference_codec']['OFF_NATIVE'][0]
    assert attempt['status'] == 'INTERRUPTED'
    # Also simulate a hard loss which could not run an exception handler.
    attempt['status'] = 'RUNNING'
    source.data['unit_attempts']['completed-artifact-before-finalization'] = [{'status': 'RUNNING'}]
    source.save()
    reopened = task.SourceStore(tmp_path, cfg)
    old = reopened.data['records'][case]['reference_codec']['OFF_NATIVE'][0]
    assert old['status'] == 'INTERRUPTED' and old['calls']['mp4_probe:attempted'] == 1
    assert old['events']['mp4']['path'] == 'fake-partial.mp4'
    assert reopened.data['unit_attempts']['completed-artifact-before-finalization'][0]['status'] == 'INTERRUPTED'
    task.reference_codec(run, case)
    state = task.read_state(source.path)
    attempts = state['records'][case]['reference_codec']['OFF_NATIVE']
    assert [x['status'] for x in attempts] == ['INTERRUPTED', 'COMPLETE']
    counts = task.call_totals(run)['reference_codec']
    assert counts['mp4_save:attempted'] == 3 and counts['mp4_save:completed'] == 3
    assert counts['mp4_probe:attempted'] == 3 and counts['mp4_probe:completed'] == 2
    assert len(calls['codec']) == 2


def test_next_run_repairs_full_dependency_chain_without_repeating_negatives(tmp_path, monkeypatch):
    cfg, calls = fake_pipeline(tmp_path, monkeypatch)
    generate = task.real_eval.phase_generate
    generated = []
    def generate_once_failure(source, config, case):
        generated.append(case)
        if case == task.CASES[0] and generated.count(case) == 1:
            raise ImportError('fake repairable generation import')
        return generate(source, config, case)
    monkeypatch.setattr(task.real_eval, 'phase_generate', generate_once_failure)
    from runtime.wan import fixed_rgb_media
    codec = fixed_rgb_media.mp4_roundtrip
    codec_attempts = []
    def codec_once_failure(*args, **kwargs):
        codec_attempts.append(str(args[0]))
        if len(codec_attempts) == 1: raise OSError('fake repairable reference codec failure')
        return codec(*args, **kwargs)
    monkeypatch.setattr(fixed_rgb_media, 'mp4_roundtrip', codec_once_failure)
    for module in (task.joint, task.controls):
        original = module._default_framewise_loader
        attempts = []
        def load_once_failure(store, original=original, attempts=attempts):
            attempts.append(1)
            if len(attempts) == 1: raise OSError('fake repairable receiver model load')
            return original(store)
        monkeypatch.setattr(module, '_default_framewise_loader', load_once_failure)
    select = task.joint._select_clock
    selected = []
    def select_with_valid_negatives(latent, key):
        selected.append(key)
        evidence, selection = select(latent, key)
        if len(selected) == 2:
            selection['operation']['status'] = 'UNSUPPORTED'
        elif len(selected) == 3:
            selection['operation'] = None
            selection['estimate']['status'] = 'UNRESOLVED'
        return evidence, selection
    monkeypatch.setattr(task.joint, '_select_clock', select_with_valid_negatives)
    invoke = lambda stage: task.run_stage(tmp_path, cfg, stage, 'fake-recovery')
    first = task.run_all(tmp_path, cfg, invoke=invoke)
    assert first['status'] == 'COMPLETE_WITH_RETAINED_FAILURES' and first['pending_stages']
    root = task.read_state(tmp_path / 'independent_four_state.json')
    assert all(len(row['attempts']) == 1 for row in root['stages'])  # no same-run retry loop
    assert generated.count(task.CASES[0]) == 1
    before = task.read_state(tmp_path / 'joint/run_state/residual_joint_state.json')
    negatives = {r['slot_id']: copy.deepcopy(r) for r in before['rows']
                 if r['clock']['status'] in ('UNSUPPORTED', 'UNRESOLVED')}
    assert len(negatives) == 2
    second = task.run_all(tmp_path, cfg, invoke=invoke)
    assert second['pending_stages'] == []
    assert second['receiver_status_counts'] == {'EVALUATED': 22, 'UNSUPPORTED': 1, 'UNRESOLVED': 1}
    assert second['quality_status_counts'] == {'EVALUATED': 24}
    assert len(calls['codec']) == 12  # no successful reference/candidate encoded again
    assert generated == [*task.CASES, task.CASES[0]]
    after = task.read_state(tmp_path / 'joint/run_state/residual_joint_state.json')
    assert all(next(r for r in after['rows'] if r['slot_id'] == ident) == row for ident, row in negatives.items())
    assert after['framewise_loads'][0]['status'] == 'FAILED'
    assert any(r.get('resume_history') for r in after['rows'])
    ctrl = task.read_state(tmp_path / 'controls/run_state/residual_controls_state.json')
    assert len(ctrl['rows']) == 24 and len(ctrl['quality_rows']) == 24
    assert ctrl['framewise_loads'][0]['status'] == 'FAILED'
    assert all(r['payload']['status'] == 'READ' for r in ctrl['rows']
               if r['case_id'] == task.CASES[0])  # initial failed K0 and OFF snapshots refreshed
    repaired = task.read_state(tmp_path / 'independent_four_state.json')['stages'][0]['attempts']
    assert [x['status'] for x in repaired] == ['FAILED', 'COMPLETE']
    task.run_all(tmp_path, cfg, invoke=lambda stage: pytest.fail('completed/negative unit replay: ' + stage))


def test_equal_operation_wan_reuse_is_counted_in_current_session(tmp_path, monkeypatch):
    cfg, calls = fake_pipeline(tmp_path, monkeypatch, same_operation=True)
    summary = task.run_all(tmp_path, cfg, invoke=lambda stage: task.run_stage(tmp_path, cfg, stage, 'fake'))
    assert summary['status'] == 'COMPLETE'
    assert len(calls['wan']) == 16
    assert summary['actual_calls_cumulative']['controls']['wan_reuses'] == {'COMPLETE': 8}
    assert summary['actual_calls_this_session']['controls']['wan_reuses'] == {'COMPLETE': 8}


def test_payload_and_quality_failure_resume_uses_saved_encodes(tmp_path, monkeypatch):
    cfg, calls = fake_pipeline(tmp_path, monkeypatch)
    for module in (task.joint, task.controls):
        original = module._default_payload_read
        attempts = []
        def read_once_failure(*args, original=original, attempts=attempts):
            attempts.append(1)
            if len(attempts) == 1: raise OSError('fake repairable read after durable Wan encode')
            return original(*args)
        monkeypatch.setattr(module, '_default_payload_read', read_once_failure)
    metrics = task.controls._quality_metrics
    quality_attempts = []
    def quality_once_failure(*args):
        quality_attempts.append(1)
        if len(quality_attempts) == 1: raise OSError('fake repairable metric input')
        return metrics(*args)
    monkeypatch.setattr(task.controls, '_quality_metrics', quality_once_failure)
    invoke = lambda stage: task.run_stage(tmp_path, cfg, stage, 'fake-read-recovery')
    first = task.run_all(tmp_path, cfg, invoke=invoke)
    assert first['status'] == 'COMPLETE_WITH_RETAINED_FAILURES'
    assert len(calls['wan']) == 24 and len(calls['keys']) == 24 and len(calls['codec']) == 12
    second = task.run_all(tmp_path, cfg, invoke=invoke)
    assert second['status'] == 'COMPLETE' and second['pending_stages'] == []
    assert len(calls['wan']) == 24 and len(calls['keys']) == 24 and len(calls['codec']) == 12
    assert second['actual_calls_cumulative']['joint']['payload_read_calls'] == {'FAILED': 1, 'COMPLETE': 8}
    assert second['actual_calls_cumulative']['controls']['payload_read_calls'] == {'FAILED': 1, 'COMPLETE': 16}
    assert second['actual_calls_cumulative']['controls']['quality_calls'] == {'FAILED': 1, 'COMPLETE': 24}
    assert second['actual_calls_this_session']['joint']['wan_encode_clips'] == {}
    assert second['actual_calls_this_session']['controls']['wan_encode_clips'] == {}
    assert second['actual_calls_this_session']['controls']['payload_read_calls'] == {'COMPLETE': 1}


@pytest.mark.parametrize('old_record_present', [True, False])
def test_method_record_commits_before_effective_config_write_interruption(tmp_path, monkeypatch, old_record_present):
    cfg = task.configuration(); task.initialize(tmp_path, cfg)
    path = tmp_path / 'independent_four_state.json'
    state = task.read_state(path); state.pop('fixed_method_semantics')
    task.joint.atomic_json(path, state)
    effective = tmp_path / 'effective_independent_four_config.json'
    if not old_record_present: effective.unlink()
    write = task.joint.atomic_json
    def interrupt_config(destination, value):
        if Path(destination) == effective: raise KeyboardInterrupt('fake loss before config replacement')
        return write(destination, value)
    monkeypatch.setattr(task.joint, 'atomic_json', interrupt_config)
    with pytest.raises(KeyboardInterrupt): task.Run(tmp_path, cfg)
    committed = task.read_state(path)
    assert committed['fixed_method_semantics'] == task.fixed_method_semantics(cfg)
    if not old_record_present:
        assert committed['method_record_notes'][-1]['status'] == 'PRIOR_METHOD_RECORD_UNAVAILABLE'
    changed = copy.deepcopy(cfg); changed['cases'][0]['seed'] += 1
    with pytest.raises(ValueError, match='saved fixed method semantics'):
        task.Run(tmp_path, changed)


@pytest.mark.parametrize('old_record', ['different', 'missing', 'same'])
def test_notebook_setup_preserves_prior_method_record_before_cli_init(tmp_path, monkeypatch, old_record):
    import shutil
    import urllib.request

    root = tmp_path / 'task'
    cfg = task.configuration(); previous = copy.deepcopy(cfg)
    if old_record == 'different': previous['cases'][0]['seed'] += 1
    task.initialize(root, previous)
    state_paths = [root / 'independent_four_state.json', root / 'source/run_state/run_state.json']
    for path in state_paths:
        state = task.read_state(path); state.pop('fixed_method_semantics')
        task.joint.atomic_json(path, state)
    effective = root / 'effective_independent_four_config.json'
    if old_record == 'missing': effective.unlink()
    prior_bytes = effective.read_bytes() if effective.exists() else None
    shutil.copyfile(builder.ZIP, root / 'companion_source.zip')
    content = tmp_path / 'content'; content.mkdir()

    # Execute the complete delivered setup cell, including the real CPU CLI.
    # Only its fixed /content paths and network availability are redirected;
    # config writes, their order, archive extraction and CLI init stay real.
    notebook = nbformat.read(builder.NOTEBOOK, as_version=4)
    code = next(cell.source for cell in notebook.cells if cell.id == 'source-and-fixed-task')
    code = code.replace('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Independent-Four-V1/resumable-independent-four-v1', str(root))
    code = code.replace('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models', str(content / 'model-cache'))
    code = code.replace("Path('/content')", f"Path({str(content)!r})")
    def unavailable(*args, **kwargs):
        raise OSError('test uses the delivered local ZIP; network disabled')
    monkeypatch.setattr(urllib.request, 'urlretrieve', unavailable)
    popen, before_init = subprocess.Popen, []
    def observed_popen(args, *a, **kw):
        if '--phase' in args and args[args.index('--phase') + 1] == 'init':
            requested = Path(args[args.index('--config') + 1])
            before_init.append((requested, effective.read_bytes() if effective.exists() else None))
        return popen(args, *a, **kw)
    monkeypatch.setattr(subprocess, 'Popen', observed_popen)
    namespace = {}
    if old_record == 'different':
        with pytest.raises(RuntimeError, match='Initialize fixed rows failed'):
            exec(compile(code, 'actual-notebook-setup', 'exec'), namespace)
        assert 'saved fixed method semantics differ' in (root / 'notebook_setup.log').read_text()
        assert effective.read_bytes() == prior_bytes
        for path in state_paths:
            assert 'fixed_method_semantics' not in task.read_state(path)
    else:
        exec(compile(code, 'actual-notebook-setup', 'exec'), namespace)
        for path in state_paths:
            assert task.read_state(path)['fixed_method_semantics'] == task.fixed_method_semantics(cfg)
        if old_record == 'missing':
            assert task.read_state(state_paths[0])['method_record_notes'][-1]['status'] == 'PRIOR_METHOD_RECORD_UNAVAILABLE'
        else:
            assert not task.read_state(state_paths[0]).get('method_record_notes')
        summary = task.read(root / 'summary.json')
        assert summary['fixed_denominator'] == dict(sources=4, receiver_rows=24, compared_bits=768, quality_pairs=24)
        assert summary['actual_calls_cumulative']['source']['operations'] == {}
    assert before_init == [(root / 'notebook_requested_config.json', prior_bytes)]
    assert namespace['CONFIG'] != effective
    assert task.read(namespace['CONFIG'])['cases'][0]['seed'] == cfg['cases'][0]['seed']


def completed_setup_fixture(tmp_path):
    cfg = task.configuration(); run = task.initialize(tmp_path, cfg)
    for row in run.data['stages']: row['status'] = 'COMPLETE'
    run.save()
    source = task.SourceStore(tmp_path, cfg)
    for row in source.data['artifacts']:
        row.update(status='AVAILABLE', path=str(tmp_path / 'completed-source-placeholder'))
    for phase in source.data['phases'].values():
        for case in phase['cases'].values(): case['status'] = 'COMPLETE'
    source.save()
    residual_path = tmp_path / 'residual/run_state/residual_clock_state.json'
    residual_state = task.read_state(residual_path)
    for row in residual_state['units'].values(): row['status'] = 'COMPLETE'
    task.joint.atomic_json(residual_path, residual_state)
    for name, filename, rows in [('joint', 'residual_joint_state.json', task.joint._new_rows(task.CASES)),
                                 ('controls', 'residual_controls_state.json', task.controls._new_rows(task.CASES))]:
        for row in rows:
            row.update(status='READ')
            row['media']['status'] = 'AVAILABLE'; row['clock']['status'] = 'CLOCK_READY'; row['payload']['status'] = 'READ'
        data = dict(rows=rows, phases={p: 'COMPLETE' for p in ('media', 'clock', 'wan', 'quality')})
        if name == 'controls':
            data['quality_rows'] = task.controls._new_quality_rows(task.CASES)
            for row in data['quality_rows']: row['status'] = 'EVALUATED'
        task.joint.atomic_json(tmp_path / name / 'run_state' / filename, data)
    return run


@pytest.mark.parametrize('remaining', ['quality', 'reference_codec', 'empty_joint_media', 'residual', 'vae', 'framewise', 'initial'])
def test_setup_routes_only_remaining_units_without_model_side_effects(tmp_path, monkeypatch, remaining):
    run = task.initialize(tmp_path, task.configuration()) if remaining == 'initial' else completed_setup_fixture(tmp_path)
    control_path = tmp_path / 'controls/run_state/residual_controls_state.json'
    if remaining in ('quality', 'reference_codec', 'vae', 'framewise'):
        data = task.read_state(control_path)
        if remaining in ('quality', 'reference_codec'):
            data['quality_rows'][0]['status'] = 'FAILED'
        else:
            row = next(x for x in data['rows'] if x['arm'] == ('WM_K1' if remaining == 'vae' else 'OFF_K0'))
            row.update(status='FAILED'); row['payload']['status'] = 'FAILED'
            if remaining == 'framewise': row['clock']['status'] = 'FAILED'
            else:
                selector = tmp_path / 'saved_blind_selection.json'
                task.joint.atomic_json(selector, {'operation': {'status': 'SUPPORTED', 'output_frames': 181}})
                row['clock']['selector_record'] = str(selector)
        task.joint.atomic_json(control_path, data)
    if remaining == 'reference_codec':
        source = task.SourceStore(tmp_path, run.config)
        task.real_eval._artifact(source, task.CASES[0] + '/PAYLOAD_NATIVE/POST')['status'] = 'FAILED'
        source.save()
    elif remaining == 'empty_joint_media':
        next(r for r in run.data['stages'] if r['stage'] == 'joint-media')['status'] = 'FAILED'; run.save()
    elif remaining == 'residual':
        path = tmp_path / 'residual/run_state/residual_clock_state.json'
        data = task.read_state(path); data['units'][task.CASES[0]]['status'] = 'FAILED'
        task.joint.atomic_json(path, data)
    planned = task.retry_stages(run); required = task.setup_requirements(run, planned)
    assert required['wan_full'] == (remaining == 'initial')
    assert required['wan_vae'] == (remaining in ('vae', 'framewise', 'initial'))
    assert required['framewise'] == (remaining in ('framewise', 'initial'))
    assert required['ffmpeg'] == (remaining in ('reference_codec', 'initial'))
    if remaining in ('quality', 'reference_codec', 'empty_joint_media', 'residual'):
        assert required['torch'] is False
    # Execute the real generated environment cell with command capture only.
    # No pip, download, process, GPU, model, codec or Drive operation is executed.
    summary = {'setup_requirements': required}
    task.joint.atomic_json(tmp_path / 'summary.json', summary)
    commands = []
    def command(args, label, check=True):
        commands.append((args, label)); return 0
    import shutil
    monkeypatch.setattr(shutil, 'which', lambda name: None)
    config = copy.deepcopy(run.config)
    for name in ('wan', 'framewise'):
        config['models'][name]['local_snapshot_path'] = str(tmp_path / 'missing-models' / name)
    notebook = nbformat.read(builder.NOTEBOOK, as_version=4)
    code = next(cell.source for cell in notebook.cells if cell.id == 'environment')
    exec(compile(code, 'actual-notebook-environment', 'exec'), dict(
        PENDING_STAGES=planned, summary=summary, command=command, sys=sys,
        OUTPUT_ROOT=tmp_path, CONFIG=tmp_path / 'notebook_requested_config.json',
        CLI=['FAKE_CAPTURE_ONLY_CLI'], config=config, Path=Path, json=json))
    text = '\n'.join(repr(args) for args, _ in commands)
    if remaining in ('quality', 'reference_codec', 'empty_joint_media', 'residual'):
        assert 'diffusers' not in text and 'snapshot_download' not in text and 'huggingface_hub' not in text
        assert 'torch' not in text
    if remaining == 'vae':
        assert 'AutoencoderKLWan' in text and 'WanPipeline' not in text
        downloads = [args[-1] for args, label in commands if label.startswith('Prepare ') and 'model files' in label]
        assert len(downloads) == 1 and "allow_patterns=['model_index.json', 'vae/*']" in downloads[0]
    if remaining == 'initial':
        assert 'WanPipeline' in text and 'AutoencoderKLWan' in text and 'AutoencoderKL' in text
        assert 'allow_patterns=None' in text
