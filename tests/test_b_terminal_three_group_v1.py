"""Bounded integration checks. Synthetic tensors/fake physical backend only."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from runtime.wan import b_terminal_three_group_v1 as run

pytestmark = pytest.mark.unit


class FakeBackend:
    """Tags replace every physical tensor; no model, VAE, codec or old data."""
    def __init__(self, fail=None):
        self.calls = Counter()
        self.files = {}
        self.fail = fail
        self.blocks = {'encoder': 0, 'decoder': 0}

    def generation_load(self, host, config):
        self.calls['generation_load'] += 1
        self.host = deepcopy(host)
        return {'fake': True}

    def generate(self, arm, writer, count, record):
        self.calls['generate'] += 1
        for index in range(50):
            for kind in ('transformer_conditional', 'transformer_unconditional', 'native_step'):
                count(kind, False); count(kind, True)
            if arm == 'P0' and index >= 25:
                count('payload_gradient', False); count('payload_gradient', True)
            record(dict(index=index, fake=True))
        return dict(host=deepcopy(self.host), arm=arm, native=True), {'fake': True}

    def release_generation(self):
        pass

    def book(self, key):
        self.calls['book'] += 1
        return key

    def native_observe(self, z, book):
        self.calls['native_Q'] += 1
        return {'status': 'FAKE', 'key': book}

    def construct(self, z, book, pre):
        self.calls['construct'] += 1
        return dict(z, arm='PM'), {'fake': True}

    def native_audit(self, *args):
        return {'fake': True}

    def load_vae(self):
        self.calls['vae_load'] += 1
        return {'fake': True}

    def release_vae(self):
        pass

    def save_tensor(self, path, value):
        run.dump(path, value)

    def load_tensor(self, path):
        return run.read(path)

    def decode(self, z):
        self.calls['D181'] += 1
        self.blocks['decoder'] += 46
        value = dict(z, native=False, frames=list(range(181)))
        return dict(preclip=value, float_rgb=value)

    def preclip(self, x):
        return {'fake': True}

    def quantize(self, x):
        return x

    def save_mp4(self, x, path):
        self.calls['codec_encode'] += 1
        self.files[str(path)] = x
        return {'fake': True}

    def probe(self, path):
        self.calls['codec_probe'] += 1
        return {'fake': True}

    def read_media(self, path, probe, length):
        self.calls['codec_decode'] += 1
        return self.files[str(path)]

    def edit(self, x, frames):
        return dict(x, frames=frames)

    def prepare_window(self, x, w, phase):
        return dict(x, window=w, phase=phase, origin=x['frames'][w])

    def encode_view(self, x):
        self.calls['E41'] += 1
        if self.fail == 'first_cal_E' and self.calls['E41'] == 1:
            raise RuntimeError('synthetic E failure')
        self.blocks['encoder'] += 11
        return x

    def observe(self, x, book):
        self.calls['Q'] += 1
        return dict(x, key=book)

    def score(self, bank, model):
        self.calls['score'] += 1
        if any(x is None for x in bank.values()):
            return run.empty_family('TECHNICAL_UNRESOLVED')
        x = bank['0']
        p = (-x['origin']) % 4
        s = (x['origin']+p)//4+1
        positive = x['arm'] == 'PM' and x['key'] == 'watermark'
        return dict(status='SCORES_COMPLETE', candidates=[dict(phase=a, source_start=b,
            score_status='RETURNED', S=0.9 if positive and (a,b)==(p,s) else 0.05 if positive else 0.1) for a,b in run.H])

    def payload_project(self, x, rows, regions):
        self.calls['native_DCT' if len(rows)==44 else 'E41_DCT'] += 1
        return dict(x=x, rows=rows, regions=list(regions))

    def payload_decode(self, x, secret, view_id, key):
        self.calls['native_bits' if len(x['rows'])==44 else 'E41_bits'] += 1
        regions = {}
        for r in ('R0', 'R1'):
            cells = []
            for row in x['rows']:
                eid = f'{view_id}/row{row}/{r}'
                cells.append(dict(local_row=row, evidence_id=eid, bits=[dict(local_bit=b, value=1,
                                  status='AVAILABLE', count=8, evidence_id=eid) for b in range(8)]))
            regions[r] = dict(local_cells=cells)
        return dict(key=key, regions=regions, control_diagnostics={})

    def quality(self, a, b, frames, pair):
        self.calls['quality'] += 1
        return dict(status='FAKE', frame_denominator=len(frames), adjacent_denominator=len(frames)-1)


def test_actual_B1_coefficients_B2_family_to_B3(tmp_path):
    import torch
    from main.tube_state import b_local_identity_dct_v1 as payload
    from main.tube_state import b_m3_time_path_v1 as time
    from main.tube_state import b_cdc32_aggregation_v1 as cdc
    # Actual B1 DCT/Counter on a constructed CPU tensor; no model/media.
    z = torch.zeros((1, 16, 11, 40, 64), dtype=torch.float32)
    basis = payload.basis()
    patch = basis.sum(0)
    for region in ('R0', 'R1'):
        y0,y1,x0,x1 = payload.REGIONS[region]
        z[0,:4,1:11,y0:y1,x0:x1] = patch
    projection = payload.project(z, include_control=True)
    decoded = payload.decode(projection, 'watermark', view_id='v0', key_id='key00')
    cache = {}; run.flatten_cache(decoded, cache)
    assert len(cache) == 20 and all('/Rc' not in e for e in cache)
    family = dict(status='SCORES_COMPLETE', input_id='i', key_id='key00', candidates=[
        dict(phase=p, source_start=s, score_status='RETURNED', S=0.9 if (p,s)==(0,1) else 0.1) for p,s in run.H])
    prepared = time.prepare_local(family, dict(status='FROZEN', tau=0.5), 'q', 0)
    views = [run.view_metadata('i', 'q', p, 0, f'v{p}', 'RETURNED') for p in range(4)]
    plan = cdc.plan_local(prepared, views)
    run.dump(tmp_path/'plan.json', plan)
    result = cdc.read_local(run.read(tmp_path/'plan.json'), cache)
    assert [s['value'] for s in result['slots'][:8]] == [1]*8
    assert all(s['value'] is None for s in result['slots'][8:])
    assert result['used_evidence_ids'] == ['v0/row1/R0', 'v0/row1/R1']


def test_full_fake_dispatch_budget_and_readonly(tmp_path):
    backend = FakeBackend()
    value = run.run(tmp_path, backend=backend)
    assert value['status'] == 'COMPLETED_ENGINEERING_RECORDS', value
    assert backend.calls == Counter(dict(generation_load=4, generate=8, book=8, native_Q=8, construct=4,
        vae_load=4, D181=12, codec_encode=48, codec_probe=48, codec_decode=48, E41=432, Q=864,
        score=216, E41_DCT=216, E41_bits=432, native_DCT=4, native_bits=8, quality=72))
    assert run.read(tmp_path/'thresholds.json')['tau'] == 0.1
    evaluation = run.read(tmp_path/'evaluation.json')
    assert len(evaluation['tasks']) == 36 and len(evaluation['bundles']) == 72
    assert evaluation['media_P0_PM_pair']['planned'] == 2880
    assert evaluation['native_P0_PM_pair']['planned'] == 1408
    for row in evaluation['tasks'].values():
        assert row['paths']['P3_subset_P2_when_defined'] is True
        assert row['invariance'] == {'ALIAS_M_X10': True, 'REVERSE_ORDER': True}
        if row['positive']:
            assert row['paths']['M3']['true_class_retained'] is True
    assert run.read(tmp_path/'blind_complete.json')['saved_unix'] <= run.read(tmp_path/'post_truth_access.json')['opened_unix']
    before = backend.calls.copy()
    assert run.run(tmp_path, backend=backend)['status'] == 'READ_ONLY_EXISTING_ATTEMPT'
    assert backend.calls == before


def test_calibration_failure_retains_eval_and_does_not_retry(tmp_path):
    backend = FakeBackend(fail='first_cal_E')
    value = run.run(tmp_path, backend=backend)
    assert value['status'] == 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert backend.calls['E41'] == 216 and backend.calls['generate'] == 4
    assert backend.calls['E41_DCT'] == 0
    assert run.read(tmp_path/'thresholds.json')['status'] == 'CALIBRATION_UNRESOLVED'
    planned = run.read(tmp_path/'planned_outputs.json')
    assert len(planned['tasks']) == 36 and len(planned['bundles']) == 72
    assert all(v['status']=='NOT_ENTERED' for v in planned['tasks'].values())
    assert not (tmp_path/'post_truth_access.json').exists()
    receipt = run.read(tmp_path/'receipts/b2-m3-c01/OFF/CROP/L/p0/E.json')
    assert receipt['attempts'] == 1 and receipt['status'] == 'FAILED'
    assert run.run(tmp_path, backend=backend)['physical_reentry'] is False
    assert backend.calls['E41'] == 216


def test_background_failure_never_loads_model(tmp_path):
    backend = FakeBackend()
    value = run.run(tmp_path, background_path=tmp_path/'missing.json', backend=backend)
    assert value['status'] == 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert not backend.calls
    assert len(run.read(tmp_path/'planned_outputs.json')['tasks']) == 36


def test_post_exact_support_is_not_same_group_or_literal_correct():
    from runtime.wan.b_terminal_three_group_evaluation_v1 import assess
    decision = run.empty_slots('NO_COMMON_FULL_CELL')
    decision['slots'][0].update(value=1, status='CONDITIONAL_READ',
        selected_refs=[dict(evidence_id='e', nominal_received_cell=[1,2,3,4], canonical_cell=2)])
    value = assess(decision, [1]*32, True, list(range(151)))
    assert value['slots'][0]['literal_correct'] is True
    assert value['slots'][0]['any_full_correct_group'] is True
    assert value['slots'][0]['any_exact_reference'] is False
    assert value['counts']['false_attribution'] == 1
    assert value['counts']['supported_correct'] == 0


def test_failed_quality_retains_each_frame(tmp_path):
    from runtime.wan.b_terminal_three_group_evaluation_v1 import evaluate
    cfg = run.load_config()
    journal = run.Journal(tmp_path, cfg, None)
    run.dump(tmp_path/'blind_complete.json', {'synthetic': True})
    value = evaluate(cfg, tmp_path, FakeBackend(), journal)
    assert len(value['quality']) == 72
    assert sum(len(q['frames']) for q in value['quality'].values()) == 11592
    assert sum(len(q['adjacent']) for q in value['quality'].values()) == 11520
    assert all(q['status']=='NOT_ENTERED' for q in value['quality'].values())
    assert value['media_P0_PM_pair']['missing'] == 2880
    assert value['native_P0_PM_pair']['missing'] == 1408
