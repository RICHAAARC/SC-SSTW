"""Artificial CPU fixtures only: no weights, datasets, codec, GPU or old evidence."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
pytestmark = pytest.mark.unit
from main.baselines import native_paper_fpr1_v3 as p
from runtime.wan.native_baseline_storage_v1 import write_json, read_json
from runtime.wan.native_baseline_storage_v3 import Unit, state, append_native, read_native, unit_ledger
from runtime.wan import native_baselines_paper_v3 as runner

ROOT = Path(__file__).resolve().parents[1]


def fixture_source(i, split=None):
    return dict(id=f'artificial_{i}', ordinal=i, split=split or ('CAL' if i <= 500 else 'TEST'),
                action=f'action_{i//25}', group=i%25+1, clip=f'v_artificial_{i}.avi',
                url=f'https://fixture.invalid/{i}.avi', dataset_path=f'fixture/{i}.avi',
                header_frames=181, fps=25, width=320, height=240)


def test_real_default_pending_and_source_semantics():
    sources = [fixture_source(i) for i in range(1, 1001)]
    assert p.validate_sources(sources) is sources
    invalid = [dict(s) for s in sources]; invalid[500].update(action=invalid[0]['action'], group=invalid[0]['group'])
    with pytest.raises(ValueError, match='Duplicate'):
        p.validate_sources(invalid)
    assert not p.eligible_header(dict(sources[0], header_frames=180))
    assert not p.eligible_header(dict(sources[0], fps=8))
    selected = p.select_sources(sources)
    assert Counter(s['split'] for s in selected) == {'CAL': 500, 'TEST': 500}
    assert selected == p.select_sources(list(reversed(sources)))


def test_actual_default_roster_and_url_case():
    config, sources = runner.load_inputs(runner.DEFAULT_CONFIG)
    assert len(sources) == 1000 and Counter(s['split'] for s in sources) == {'CAL': 500, 'TEST': 500}
    assert len({s['action'] for s in sources}) == 73
    assert len({s['action'] for s in sources[:500]}) == 69
    assert len({s['action'] for s in sources[500:]}) == 71
    for s in sources:
        assert s['url'].endswith(s['dataset_path'])
    assert all(s['ordinal'] == i for i, s in enumerate(sources, 1))
    special = [s for s in sources if s['action'] == 'HandStandPushups']
    assert [s['id'] for s in special] == ['ucf-0575', 'ucf-0838', 'ucf-0937']
    assert all('/HandstandPushups/v_HandStandPushups_' in s['dataset_path'] for s in special)


def test_queries_and_global_edit_ordinals():
    sources = [fixture_source(1), fixture_source(501)]
    q = p.queries(sources)
    for s in sources:
        for m, k in p.METHODS.items():
            item = q[f"{m}/{s['id']}"]
            assert len(item['query']) == k
            if s['split'] == 'TEST':
                assert item['wrong_query'] != item['query'] and len(item['wrong_query']) == k
    assert p.source_maps(sources[1])['seed'] == 20261105+50100
    assert p.source_maps(sources[0])['seed'] != p.source_maps(sources[1])['seed']
    # A matching query across sources is allowed; validator has no global uniqueness condition.
    assert p.view_score('wam', [[0]*32], [True], [0]*32, complete=True) == 32


def test_full_operation_budget_from_all_fixed_units():
    counts = Counter(); method_counts = {m: Counter() for m in p.METHODS}
    frames = Counter(); read_tasks = 0
    for i in range(1, 1001):
        s = fixture_source(i)
        counts.update(r['kind'] for r in p.operations(s))
        for m in p.METHODS:
            method_counts[m].update(r['kind'] for r in p.operations(s, m))
            counts.update(r['kind'] for r in p.operations(s, m))
            ts = p.tasks(s, m); read_tasks += len(ts); frames[m] += sum(t['frames'] for t in ts)
    assert read_tasks == 66000 and frames == {'wam': 5025000, 'pixelseal': 5025000}
    assert method_counts['wam']['detect'] == 645000 and method_counts['wam']['embed'] == 11500
    assert method_counts['pixelseal']['detect'] == 33000 and method_counts['pixelseal']['embed'] == 500
    for k in ('encode', 'probe', 'decode', 'normalize', 'source_probe', 'download', 'quality_precodec', 'quality_G1', 'quality_G2'):
        assert counts[k] == p.budget()[k]
    assert counts['load'] == 2000
    assert sum((n+31)//32 for n in p.CONDITIONS.values())*2*1500 == 180000


def test_calibration_order_ties_degenerate_missing_and_family():
    scores = [[20]*22 for _ in range(495)]+[[30]*22 for _ in range(5)]
    c = p.calibrate(scores, 32)
    assert c['tau'] == 30 and c['ties_at_tau'] == 5
    assert not 30 > c['tau'] and 31 > c['tau']
    assert p.calibrate([[-1]*22 for _ in range(500)], 32)['degeneration'] == 'ANY_NONEMPTY'
    assert p.calibrate([[256]*22 for _ in range(500)], 256)['degeneration'] == 'ALL_REJECT'
    scores[0][3] = None
    assert p.calibrate(scores, 32)['status'] == 'CALIBRATION_INCOMPLETE'
    assert p.family_event([None, True]+[False]*20) is True
    assert p.family_event([None]+[False]*21) is None
    assert p.family_event([False]*22) is False


def test_exact_uncertainty_and_missing_bounds():
    zero = p.rate_summary([False]*500)
    assert zero['conservative_one_sided95_upper'] == pytest.approx(1-.05**(1/500))
    two = p.rate_summary([True]*2+[False]*498)
    assert two['rate_lower'] == .004 and two['conservative_one_sided95_upper'] > .01
    missing = p.rate_summary([True]*2+[None]*3+[False]*495)
    assert missing['planned'] == 500 and missing['rate_upper'] == .01 and missing['unknown'] == 3
    assert p.action_sensitivity([False]*500, [str(i%10) for i in range(500)])['upper_rate_percentiles'] == [0, 0]


def test_native_b8_parser_threshold_row_isolation_and_empty():
    import torch
    from runtime.wan.native_baseline_models_v3 import parse_wam
    preds = torch.zeros(3, 33, 256, 256)
    preds[:, 0] = -9
    preds[0, 0, 64, 64] = 9; preds[0, 1:, 64, 64] = .4
    preds[1, 0, 10, 10:12] = 9; preds[1, 1:, 10, 10:12] = .6
    seen = []
    def native(payload, mask, method):
        seen.append(method)
        return torch.stack([payload[i].masked_select((mask[i]>.5).expand_as(payload[i])).reshape(32, -1).mean(-1)>.5 for i in range(3)])
    parsed = parse_wam(preds, native)
    assert seen == ['semihard'] and parsed['selected_count'].tolist() == [1, 2, 0]
    assert parsed['center_count'].tolist() == [1, 0, 0]
    assert parsed['valid'].tolist() == [True, True, False]
    assert parsed['bits'][0].sum() == 0 and parsed['bits'][1].sum() == 32
    assert parsed['means'][0, 0] == pytest.approx(.4) and parsed['means'][1, 0] == pytest.approx(.6)
    assert 'mask' not in parsed


def test_media_normalization_arguments_without_codec(tmp_path):
    from runtime.wan.native_baseline_media_v3 import MediaV3, crop_box
    assert crop_box(320, 240) == (0, 20, 320, 200)
    media = MediaV3(); commands = []
    def command(args, **kwargs):
        commands.append(args)
        return b''  # Real short-source handling, but no codec is launched.
    media.command = command
    with pytest.raises(ValueError, match='181 real frames'):
        media.normalize(tmp_path/'fake.avi', dict(width=320, height=240))
    cmd = commands[0]
    assert 'crop=320:200:0:20,scale=512:320:flags=bilinear' in cmd
    assert '-fps_mode' in cmd and 'passthrough' in cmd and '-r' not in cmd
    assert 'fps=' not in ' '.join(cmd) and 'loop' not in cmd


def test_unit_consumption_partial_stream_and_reuse(tmp_path):
    unit = Unit(tmp_path/'u', [dict(id='load', kind='load', status='NOT_STARTED')])
    unit.reuse(); unit.close()
    assert unit_ledger(tmp_path/'u')['load']['status'] == 'REUSED_NO_CALL'
    with pytest.raises(RuntimeError, match='Consumed'):
        Unit(tmp_path/'u', [])
    other = Unit(tmp_path/'interrupted', [])
    assert state(other.folder) == 'INTERRUPTED_CONSUMED'
    append_native(other.folder, 't', 0, dict(bits=np.zeros((8, 32), dtype=np.uint8)))
    with (other.folder/'native.bin').open('ab') as f:
        f.write(b'partial')
    records, errors = read_native(other.folder)
    assert len(records) == 1 and errors and records[0][1]['bits'].shape == (8, 32)


class FakeMedia:
    def download(self, source, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(b'artificial')
        return dict(bytes=10, requests=1)
    def source_probe(self, path):
        return dict(width=320, height=240, nb_frames=181, r_frame_rate='25/1')
    def normalize(self, path, meta):
        return np.zeros((181, 1, 1, 3), dtype=np.uint8)
    def encode(self, rgb, path, fps):
        Path(path).parent.mkdir(parents=True, exist_ok=True); Path(path).write_bytes(b'artificial')
        return dict(bytes=10)
    def probe(self, path, shape, fps):
        return dict(shape=shape, fps=fps)
    def decode(self, path, shape):
        return np.zeros((shape[0], 1, 1, 3), dtype=np.uint8)


class FakeAdapter:
    def __init__(self, method, calls):
        self.method = method; self.calls = calls
        self.counts = dict(embedder_calls=0, embedder_frames=0, detector_calls=0, detector_frames=0)
    def load(self):
        self.calls.append(('load', self.method)); return dict(artificial=True)
    def close(self):
        self.calls.append(('close', self.method))
    def resources(self):
        return dict(artificial=True)
    def embed(self, pixels, message, rectangle):
        self.calls.append(('embed', self.method, len(pixels)))
        self.counts['embedder_calls'] += 1
        self.counts['embedder_frames'] += len(pixels) if self.method == 'wam' else 23
        return pixels.copy()
    def detect(self, pixels):
        self.calls.append(('detect', self.method, len(pixels)))
        self.counts['detector_calls'] += 1 if self.method == 'wam' else (len(pixels)+31)//32
        self.counts['detector_frames'] += len(pixels)
        if self.method == 'wam':
            n = len(pixels)
            return dict(bits=np.zeros((n, 32), dtype=np.uint8), valid=np.ones(n, dtype=bool),
                        selected_count=np.ones(n, dtype=np.uint32), center_count=np.ones(n, dtype=np.uint32),
                        sums=np.zeros((n, 32)), means=np.zeros((n, 32)), mask_shape=np.array([256, 256]))
        return dict(bits=np.zeros((len(pixels), 256), dtype=np.uint8),
                    video_bits=np.zeros(256, dtype=np.uint8), logits=np.zeros((len(pixels), 257), dtype=np.float32))


def test_artificial_full_shard_budget_reuse_tail_and_resume(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'SHAPE', [181, 1, 1, 3])
    sources = [fixture_source(1), fixture_source(501)]
    write_json(tmp_path/'queries.json', p.queries(sources))
    calls = []
    factory = lambda m: FakeAdapter(m, calls)
    runner.run_shard(tmp_path, sources, factory, FakeMedia(), lambda a, b: dict(mse_rgb255=0))
    assert [c for c in calls if c[0] == 'load'] == [('load', 'wam'), ('load', 'pixelseal')]
    assert unit_ledger(tmp_path/'units/wam'/sources[1]['id'])['load']['status'] == 'REUSED_NO_CALL'
    assert all(1 <= c[2] <= 8 for c in calls if c[0] in ('detect', 'embed') and c[1] == 'wam')
    assert Counter(c[2] for c in calls if c[0] == 'embed' and c[1] == 'wam') == {8: 22, 5: 1}
    assert len([c for c in calls if c[:2] == ('detect', 'wam')]) == 1290
    assert len([c for c in calls if c[:2] == ('detect', 'pixelseal')]) == 66
    before = list(calls)
    runner.run_shard(tmp_path, sources, factory, FakeMedia(), lambda a, b: dict(mse_rgb255=0))
    assert calls == before
    from runtime.wan.native_baseline_evaluation_v3 import evaluate_unit
    ev = evaluate_unit(tmp_path, sources[1], 'wam')
    assert len(ev['rows']) == 44 and all(r['complete'] for r in ev['rows'])
    assert sum(r['planned_frames'] for r in ev['rows']) == 6700
    receipt = runner.physical_receipt(tmp_path, sources)
    assert receipt['unit_status'] == {'COMPLETED': 6}
    assert receipt['actual_forward_hooks']['wam']['detector_calls'] == 1290


def test_failed_partial_view_retains_native_and_no_retry(tmp_path, monkeypatch):
    from runtime.wan.native_baseline_evaluation_v3 import evaluate_unit
    source = fixture_source(1); write_json(tmp_path/'queries.json', p.queries([source]))
    folder = tmp_path/'units/wam'/source['id']; unit = Unit(folder, p.operations(source, 'wam'))
    adapter = FakeAdapter('wam', []); task = p.tasks(source, 'wam')[0]
    value = unit.call(f"detect/{task['id']}/0", lambda: adapter.detect(np.zeros((8, 1, 1, 3), dtype=np.uint8)))
    append_native(folder, task['id'], 0, value)
    unit.close(RuntimeError('artificial interrupt'))
    ev = evaluate_unit(tmp_path, source, 'wam'); first = ev['rows'][0]
    assert first['score'] is None and not first['complete']
    assert first['coverage'] == 8/181 and first['unknown'] == 173*32
    assert len(ev['rows']) == 22 and sum(r['planned_frames'] for r in ev['rows']) == 3350
    with pytest.raises(RuntimeError):
        Unit(folder, p.operations(source, 'wam'))


def test_missing_saved_view_is_unknown_not_empty_mask(tmp_path):
    from runtime.wan.native_baseline_evaluation_v3 import evaluate_unit
    s = fixture_source(501); write_json(tmp_path/'queries.json', p.queries([s]))
    ev = evaluate_unit(tmp_path, s, 'wam')
    assert len(ev['rows']) == 44 and all(r['score'] is None for r in ev['rows'])
    assert all(r['coverage'] == 0 and r['complete_messages_unknown'] == r['frames'] for r in ev['rows'])


def test_calibration_incomplete_still_runs_test_native(tmp_path, monkeypatch):
    from runtime.wan import native_baseline_evaluation_v3 as evaluation
    sources = [fixture_source(1), fixture_source(501)]
    write_json(tmp_path/'queries.json', p.queries(sources))
    monkeypatch.setattr(runner, 'SHAPE', [181, 1, 1, 3])
    monkeypatch.setattr(runner, 'initialize', lambda out, config, roster: (config, roster))
    # Artificial source-preparation failure consumes CAL; TEST remains runnable.
    u = Unit(tmp_path/'units/prepare'/sources[0]['id'], p.operations(sources[0])); u.close(RuntimeError('fixture'))
    calls = []
    class CalOrderAdapter(FakeAdapter):
        def load(self):
            for method in p.METHODS:
                assert read_json(tmp_path/f'calibration_{method}.json')['status'] == 'CALIBRATION_INCOMPLETE'
            return super().load()
    result = runner.run({}, sources, tmp_path, tmp_path/'no-vendor',
        adapter_factory=lambda m: CalOrderAdapter(m, calls), media=FakeMedia(), measure=lambda a,b: dict(mse_rgb255=0))
    assert result['verification'] == {'wam': 'CALIBRATION_INCOMPLETE', 'pixelseal': 'CALIBRATION_INCOMPLETE'}
    assert state(tmp_path/'units/wam'/sources[1]['id']) == 'COMPLETED'
    assert state(tmp_path/'units/pixelseal'/sources[1]['id']) == 'COMPLETED'
    assert len([c for c in calls if c[:2] == ('detect', 'wam')]) == 860
    saved = read_json(tmp_path/'evaluation.json')
    assert all(r['unknown'] == r['planned'] for r in saved['summaries'])


def test_returned_without_saved_output_and_unreadable_unit(tmp_path):
    from runtime.wan.native_baseline_evaluation_v3 import evaluate_unit, evaluated_or_unknown
    s = fixture_source(1); write_json(tmp_path/'queries.json', p.queries([s]))
    folder = tmp_path/'units/wam'/s['id']; u = Unit(folder, p.operations(s, 'wam'))
    task = p.tasks(s, 'wam')[0]
    u.call(f"detect/{task['id']}/0", lambda: None); u.close(RuntimeError('save failed'))
    ev = evaluate_unit(tmp_path, s, 'wam')
    assert ev['rows'][0]['statuses']['OUTPUT_MISSING'] == 8 and ev['rows'][0]['score'] is None
    (folder/'plan.json').write_text('broken JSON')
    retained = evaluated_or_unknown(tmp_path, s, 'wam')
    assert len(retained['rows']) == 22 and retained['read_errors']
    assert all(r['score'] is None and r['unknown'] == r['planned_bits'] for r in retained['rows'])


def test_group_cleanup_after_leader_exit(tmp_path):
    from runtime.wan.native_baseline_colab_v3 import stop_group
    pidfile = tmp_path/'pid'
    script = "import subprocess,sys; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); Path(sys.argv[1]).write_text(str(p.pid))"
    leader = subprocess.Popen([sys.executable, '-c', script, str(pidfile)], start_new_session=True)
    leader.wait(timeout=5)
    child = int(pidfile.read_text())
    stop_group(leader)
    for _ in range(30):
        stat = Path(f'/proc/{child}/stat')
        if not stat.exists() or stat.read_text().split()[2] == 'Z':
            break
        time.sleep(.02)
    else:
        pytest.fail('Orphan child survived group cleanup')


def test_actual_notebook_matches_builder_and_stays_thin(tmp_path):
    from scripts.build_native_baselines_paper_fpr1_v3_notebook import build
    (tmp_path/'notebooks').mkdir(); built = build(tmp_path)
    delivered = ROOT/'notebooks/native_baselines_paper_fpr1_v3.ipynb'
    assert json.loads(built.read_text()) == json.loads(delivered.read_text())
    nb = json.loads(delivered.read_text())
    assert ''.join(nb['cells'][0]['source']) == "from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert sum(len(c['source']) for c in nb['cells'] if c['cell_type'] == 'code') <= 20
    assert all(c.get('outputs', []) == [] for c in nb['cells'])


def test_fix1_receipt_never_started_and_consumed_missing_ledger(tmp_path):
    source = fixture_source(1)
    initial = runner.physical_receipt(tmp_path, [source])
    assert initial['unit_status'] == {'NEVER_STARTED': 3}
    for kind in ['prepare', *p.METHODS]:
        planned = Counter(r['kind'] for r in p.operations(source, None if kind == 'prepare' else kind))
        for operation, n in planned.items():
            assert initial['physical_calls'][kind+'/'+operation] == {'NOT_STARTED': n}
    folder = tmp_path/'units/wam'/source['id']; folder.mkdir(parents=True)
    write_json(folder/'state.json', dict(status='STARTED'))  # Crash before plan creation.
    result = runner.physical_receipt(tmp_path, [source])
    assert result['physical_calls']['wam/load'] == {'UNKNOWN_UNIT_LEDGER': 1}
    assert result['physical_calls']['wam/detect'] == {'UNKNOWN_UNIT_LEDGER': 430}
    assert result['physical_calls']['pixelseal/load'] == {'NOT_STARTED': 1}
    assert result['physical_accounting_errors']
    u = Unit(tmp_path/'units/pixelseal'/source['id'], p.operations(source, 'pixelseal'))
    u.reuse(); u.close()
    result = runner.physical_receipt(tmp_path, [source])
    assert result['physical_calls']['pixelseal/load'] == {'REUSED_NO_CALL': 1}


def test_fix1_bad_saved_results_remain_readonly_and_display_other_receipts(tmp_path, monkeypatch, capsys):
    from runtime.wan import native_baseline_colab_v3 as colab
    from runtime.wan import native_baseline_storage_v3 as storage
    monkeypatch.setattr(colab, 'WORK', tmp_path/'work'); monkeypatch.setattr(colab, 'OUTPUT', tmp_path/'output')
    colab.OUTPUT.mkdir()
    (colab.OUTPUT/'result.json').write_text('{bad')
    (colab.OUTPUT/'evaluation.json').write_text('{"calibration": null}')
    write_json(colab.OUTPUT/'bootstrap_failure.json', dict(status='SAVED_FAILURE', error='artificial prior failure'))
    monkeypatch.setattr(runner, 'load_inputs', lambda path: ({}, [fixture_source(1)]))
    monkeypatch.setattr(storage, 'state', lambda path: 'COMPLETED')
    monkeypatch.setattr(colab.subprocess, 'Popen', lambda *a, **k: pytest.fail('No worker allowed'))
    monkeypatch.setattr(colab, 'source_download', lambda *a, **k: pytest.fail('No download allowed'))
    result = colab.run_colab(source_directory=ROOT)
    assert result['status'] == 'READ_ONLY_OUTPUT_UNREADABLE'
    assert len(result['display_read_errors']) == 2
    assert (colab.OUTPUT/'result.json').read_text() == '{bad'
    assert 'SAVED_FAILURE' in capsys.readouterr().out
    assert len(colab.show_results()['read_errors']) == 2


def test_fix1_download_call_is_not_http_redirect_count(tmp_path, monkeypatch):
    import io
    from runtime.wan.native_baseline_media_v3 import MediaV3
    import urllib.request
    seen = []
    def urlopen(request, timeout):
        seen.append(request.full_url)
        return io.BytesIO(b'artificial bytes; no network')
    monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
    source = fixture_source(1)
    source['url'] = 'https://fixture.invalid/HandstandPushups/v_HandStandPushups_g01_c01.avi'
    result = MediaV3().download(source, tmp_path/'fixture.avi')
    assert seen == [source['url']] and result['download_calls'] == 1
    assert result['http_requests'] is None and 'NOT_MEASURED' in result['http_request_measurement']
    assert 'requests' not in result
