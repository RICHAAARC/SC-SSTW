"""Fixed synthetic CPU/fake-backend engineering checks; no saved real evidence."""
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from main.tube_state import blind_temporal_segmented_v1 as c
from runtime.wan import blind_temporal_segmented_v1 as r
from runtime.wan import blind_temporal_segmented_evaluation_v1 as e

torch.set_num_threads(1)
pytestmark = pytest.mark.unit


@pytest.mark.parametrize('n,lengths,times', [(89, [89]*4, [23]*4), (138, [137,137,137,141], [35,35,35,36])])
def test_phase_support(n, lengths, times):
    rgb = torch.arange(n).reshape(n,1,1,1).expand(-1,1,1,3).to(torch.uint8)
    cells, pairs = [], []
    for p in range(4):
        phase, plan = c.prepare_phase(rgb, p)
        assert len(phase) == lengths[p] and plan['latent_times'] == times[p]
        assert phase[:,0,0,0].tolist() == plan['received_indices']
        for a in range(n-3):
            if c.bank_address(a)[0] == p:
                _, t = c.bank_address(a)
                assert plan['received_indices'][4*t-3:4*t+1] == list(range(a,a+4))
                cells.append(a)
        for a in range(n-7):
            if c.bank_address(a)[0] == p:
                _, t = c.bank_address(a)
                assert plan['received_indices'][4*t-3:4*t+5] == list(range(a,a+8))
                pairs.append(a)
    assert sorted(cells) == list(range(n-3)) and sorted(pairs) == list(range(n-7))


@pytest.mark.parametrize('n,count,minimum', [(89,376557,8),(138,129646,14)])
def test_exhaustive_normative_scores_and_complete_zero_ties(n,count,minimum):
    # A fixed cancellation fixture distinguishes fsum from ordinary accumulation.
    q = np.zeros((n-7,22), dtype=np.float64)
    q[:,0] = 1e16
    q[:,1] = 1.0
    q[:,2] = -1e16
    result, arrays = c.select_path(q,n)
    assert len(arrays['parameters']) == count
    expected = np.array([math.fsum(float(q[a,k]) for a,k in c.pilot_terms(n,p)) for p in arrays['parameters']])
    assert np.array_equal(arrays['raw_sums'], expected)
    assert np.array_equal(arrays['scores'], expected/(n//8))
    assert np.array_equal(arrays['maximum_indices'], np.flatnonzero(expected/(n//8) == (expected/(n//8)).max()))
    assert result['minimum_support'] >= minimum
    assert np.all(arrays['support_counts']+arrays['partial_state_counts']+arrays['absent_state_counts'] == 22)
    target = (37,0,0) if n == 89 else (0,43,45)
    idx = np.flatnonzero(np.all(arrays['parameters'] == target,axis=1))[0]
    assert (arrays['support_counts'][idx],arrays['partial_state_counts'][idx],arrays['absent_state_counts'][idx]) == ((10,2,10) if n==89 else (16,2,4))
    zero_result, zero = c.select_path(np.zeros_like(q), n)
    assert zero_result['maximum_count'] == count and zero_result['frame_consensus'] == [None]*n
    assert len(zero['maximum_indices']) == count
    assert c.unit_plan(zero_result, zero['parameters'])['units'] == []


def test_division_rounding_retains_distinct_sum_tie():
    sums = np.array([1.5000000000000007,1.5000000000000009])
    scores, maximum, ids = c.score_maxima(sums, 11)
    assert sums[0] != sums[1] and scores[0] == scores[1] and ids.tolist() == [0,1]


@pytest.mark.parametrize('n,left,right,different', [(89,(37,0,0),(36,1,1),0),(138,(0,43,45),(0,43,46),45)])
def test_structural_indistinguishability_and_consensus(n,left,right,different):
    assert c.pilot_terms(n,left) == c.pilot_terms(n,right)
    paths = [c.path_frames(n,v) for v in (left,right)]
    assert [i for i in range(n) if paths[0][i] != paths[1][i]] == [different]
    consensus = [a if a == b else None for a,b in zip(*paths)]
    plan = c.unit_plan(dict(received_frames=n,frame_consensus=consensus,status='UNRESOLVED'), np.array([left,right]))
    assert plan['status'] == 'UNRESOLVED'
    assert all(not u['received_start'] <= different < u['received_start']+4 for u in plan['units'])
    assert len({u['source_t'] for u in plan['units']}) == len(plan['units'])
    if n == 89:
        assert all(u['context_flags']['after_internal_gap']=='unknown' for u in plan['units'])


def test_original_pairs_both_parities_wrong_key_and_cross_segment_units():
    book = c.public_codebook('watermark')
    z = torch.zeros((1,16,23,40,64),dtype=torch.float32)
    for roi,(y0,y1,x0,x1) in enumerate(c.ROIS):
        z[0,4:16,2:4,y0:y1,x0:x1] = book['directions'][roi,5].float()
    obs = c.pair_observations(z,89,0,book)
    idx = obs['starts'].tolist().index(5)
    assert obs['pooled_scores'][idx].argmax() == 5
    wrong = c.pair_observations(z,89,0,c.public_codebook('watermark-wrong'))
    assert not np.array_equal(obs['pooled_scores'], wrong['pooled_scores'])
    assert [c.bank_address(a)[1]%2 for a in obs['starts'][:2]] == [1,0]
    crop = c.direct_plan(89,(37,0,0))
    delete = c.direct_plan(138,(0,43,45))
    assert [sum(u['segment']==g for u in crop['units']) for g in range(4)] == [2,11,9,0]
    assert [sum(u['segment']==g for u in delete['units']) for g in range(4)] == [11,0,11,11]
    assert [u['source_t'] for u in crop['units'] if u['pilot_geometry']=='BOUNDARY_UNIT_INFERRED_FROM_PATH'] == [10,31]
    assert [u['source_t'] for u in delete['units'] if u['pilot_geometry']=='BOUNDARY_UNIT_INFERRED_FROM_PATH'] == [11]
    full = c.direct_plan(181,(0,0,0))
    assert {u['source_t']:u['segment'] for u in full['units'] if u['source_t'] in (11,12,33,34)} == {11:0,12:1,33:2,34:3}
    assert c.unit_plan(c.unavailable_path(89))['units'] == []
    with pytest.raises(ValueError,match='all four'):
        c.combine_pairs({0:obs},89)


def zero_vote_banks(n):
    return {p:dict(starts=np.array([a for a in range(n-3) if c.bank_address(a)[0]==p]),
        signed_votes=np.full((4,len([a for a in range(n-3) if c.bank_address(a)[0]==p]),240),-1,dtype=np.int8),
        zero_coefficient_mask=np.ones((4,len([a for a in range(n-3) if c.bank_address(a)[0]==p]),240),dtype=np.int8)) for p in range(4)}


def test_strict_zero_counter_order_and_false_attribution():
    n = 89
    banks = zero_vote_banks(n)
    # Channel 0/1 positive, 2/3 negative: tie resolved by first encountered vote.
    for bank in banks.values():
        bank['signed_votes'][:2] = 1
        bank['zero_coefficient_mask'][:2] = 0
    baseline = c.direct_plan(n,(0,0,0),phase_zero_only=True)
    payload = c.reduce_votes(baseline,banks)
    assert all(b['tie'] and b['decoded']==1 for s in payload['segments'][:2] for b in s['bits'])
    assert payload['segments'][0]['bits'][0]['count'] == 11*120
    checked = e.evaluate(baseline,payload,c.path_frames(n,(37,0,0)),[1]*32)
    assert checked['summary']['correct_bits'] == 16 and checked['summary']['wrong_source_units'] == 22
    assert all(b['wrong_source_unit_votes'] > 0 for b in checked['bits'][:16])
    bogus = c.direct_plan(n,(92,0,0))
    pay = c.reduce_votes(bogus,banks)
    ev = e.evaluate(bogus,pay,c.path_frames(n,(37,0,0)),[1]*32)
    assert sum(b['false_coverage'] for b in ev['bits']) == 8
    assert all(b['correct'] is None and b['message_agreement'] is True for b in ev['bits'][24:])
    assert ev['summary']['correct_bits'] == 8 and ev['summary']['read_bits'] == 16
    comparison = e.compare(checked,ev)
    assert comparison['candidate_false_coverage'] == list(range(24,32))
    assert not set(range(24,32)) & set(comparison['common_bits']+comparison['added_coverage'])
    banks[0]['signed_votes'][0,0,0] = 1
    banks[0]['zero_coefficient_mask'][0,0,0] = 1
    with pytest.raises(ValueError,match='strict-zero'):
        c.reduce_votes(baseline,banks)


def test_fft_extraction_no_synthetic_or_tail_votes():
    z = torch.zeros((1,16,36,40,64))
    spectrum = c.fft_bank(z,138,3)
    raw = c.extract_votes(spectrum,138,3,'watermark')
    assert raw['starts'][0] == 2 and raw['starts'][-1] == 134
    assert raw['signed_votes'].shape == (4,34,240)
    assert np.all(raw['signed_votes'] == -1) and np.all(raw['zero_coefficient_mask'] == 1)


class FakeBackend:
    device = 'cpu'
    blocks = {}
    def __init__(self,output,fail=None):
        self.output,self.fail,self.encodes = output,fail,0
    def load(self):
        assert (self.output/'attempt.json').is_file()
        return {'synthetic_fake_backend':True}
    def load_rgb(self,path):
        if self.fail == 'all':
            raise OSError('fixed synthetic unreadable RGB')
        return torch.zeros((181,1,1,3),dtype=torch.uint8)
    def encode(self,rgb):
        self.encodes += 1
        ledger = json.loads((self.output/'ledger.json').read_text())
        assert sum(v['status']=='STARTED' and v['kind']=='posterior_encode' for v in ledger.values()) == 1
        if self.encodes == 1 and self.fail == 'phase':
            raise RuntimeError('fixed synthetic phase failure')
        if self.encodes == 1 and self.fail == 'interrupt':
            raise KeyboardInterrupt('fixed synthetic interruption')
        return torch.zeros((1,16,(len(rgb)-1)//4+1,40,64))


@pytest.mark.parametrize('failure',[None,'phase','all','interrupt'])
def test_fake_runtime_budget_order_failures_and_read_only(tmp_path,monkeypatch,failure):
    output = tmp_path/'out'
    truth = tmp_path/'truth.json'
    truth.write_text(json.dumps({'fixed_method_semantics':{'bits':[0]*32}}))
    read = Path.read_text
    def read_spy(path,*args,**kwargs):
        if path == truth:
            assert (output/'blind_completion.json').is_file()
            assert len(list((output/'blind').rglob('payload.json'))) == 16
        return read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read_spy)
    real_fft = c.fft_bank
    def fft_spy(*args):
        assert (output/'plans_completion.json').is_file()
        assert len(list((output/'blind').rglob('plan.json'))) == 16
        return real_fft(*args)
    monkeypatch.setattr(c,'fft_bank',fft_spy)
    # Path arithmetic is exhaustively tested separately; runtime uses fixed, labeled fake scores.
    def fixed_selector(q,n):
        arrays = dict(parameters=np.array([[0,0,0]],dtype=np.int16),raw_sums=np.array([0.]),
                      scores=np.array([0.]),support_counts=np.array([n//8]),maximum_indices=np.array([0]))
        return dict(status='READ',received_frames=n,frame_consensus=list(range(n)),maximum_count=1), arrays
    monkeypatch.setattr(c,'select_path',fixed_selector)
    backend = FakeBackend(output,failure)
    result = r.run(tmp_path/'source',output,truth,backend=backend)
    assert len(result['evaluation']) == 24
    assert sum(len(v['bits']) for v in result['evaluation'].values()) == 768
    assert len(list((output/'post_blind').rglob('payload.json'))) == 8
    counts = result['counts']
    assert counts['posterior_encode']['budget'] == 16 and counts['fft']['budget'] == 16
    if failure is None:
        assert result['status'] == 'EXECUTION_COMPLETE'
        assert counts['posterior_encode']['returned'] == counts['fft']['returned'] == 16
        assert counts['pair_extract']['returned'] == counts['vote_extract']['returned'] == 32
        assert sum(np.load(p)['signed_votes'].size for p in (output/'observations').rglob('votes.npz')) == 848640
        assert sum(np.load(p)['roi_scores'].size for p in (output/'observations').rglob('pairs.npz')) == 74976
        assert len(list((output/'prepared').glob('received_*.pt'))) == 4
        maps = json.loads((output/'post_blind'/'oracle_maps.json').read_text())
        assert maps['read_after_blind'] and len(maps['received']) == 4
        assert torch.load(output/'prepared'/'received_0.pt',weights_only=True).shape[0] == 89
        assert len(list((output/'observations'/'inputs').glob('*.json'))) == 2
    elif failure == 'phase':
        assert backend.encodes == 16 and counts['posterior_encode']['failed'] == 1
        assert result['paths']['P/CROP_MID/CORRECT']['status'] == 'INCOMPLETE'
        assert result['evaluation']['P/CROP_MID/CORRECT/EST']['summary']['read_bits'] == 0
    elif failure == 'interrupt':
        assert backend.encodes == 1 and counts['posterior_encode']['interrupted'] == 1
    else:
        assert backend.encodes == 0 and all(p['status']=='INCOMPLETE' for p in result['paths'].values())
        assert not any(s['observation']=='COMPLETE_PILOT_AVAILABLE' for v in result['evaluation'].values() for s in v['source_states'])
    (output/'result.json').write_text('{broken')
    # Prove source failures, interruptions and corrupted result cannot open a new backend.
    monkeypatch.setattr(r,'WanBackend',lambda *_: pytest.fail('repeat backend construction'))
    again = r.run(tmp_path/'missing',output,tmp_path/'missing_truth')
    assert again['status'].startswith('EXISTING_ATTEMPT')
    assert again['fixed_denominator']['planned_bits'] == 768
    assert len(again['evaluation']) == 24


def test_residue_and_unreadable_truth_retains_fixed_tables(tmp_path,monkeypatch):
    out = tmp_path/'residue'
    (out/'features').mkdir(parents=True)
    (out/'features'/'partial.pt').write_bytes(b'broken')
    monkeypatch.setattr(r,'WanBackend',lambda *_: pytest.fail('residue must prevent backend'))
    assert r.run(tmp_path,out,tmp_path/'truth')['status'] == 'EXISTING_ATTEMPT_WITHOUT_CHECKPOINT'
    output = tmp_path/'unreadable'
    result = r.run(tmp_path,output,tmp_path/'missing_truth',backend=FakeBackend(output,'all'))
    assert len(result['evaluation']) == 24 and result['evaluation_error']
    assert len(list((output/'post_blind').rglob('payload.json'))) == 8


def test_hard_interruption_readonly_fixed_rows_without_dependencies(tmp_path,monkeypatch):
    (tmp_path/'attempt.json').write_text('{}')
    (tmp_path/'ledger.json').write_text(json.dumps({'P/CROP_MID/p0/encode':{'kind':'posterior_encode','status':'STARTED'}}))
    (tmp_path/'checkpoint.json').write_text(json.dumps({'status':'RUNNING','counts':{'posterior_encode':{'started':1}}}))
    monkeypatch.setattr(r,'WanBackend',lambda *_: pytest.fail('read only may not construct backend'))
    result = r.run(tmp_path/'missing',tmp_path,tmp_path/'missing_truth')
    assert len(result['evaluation']) == 24
    assert sum(len(t['bits']) for t in result['evaluation'].values()) == 768
    assert sum(len(t['frames']) for t in result['evaluation'].values()) == 908*3
    assert result['fixed_denominator']['per_mode']['source_cells'] == 352
    assert result['counts']['posterior_encode']['started'] == 1
    assert result['counts']['posterior_encode']['active'] == 1 and result['count_source'] == 'ledger.json'


def test_failed_payload_no_invented_votes_and_geometry_not_observation():
    plan = c.direct_plan(89,(0,0,0),phase_zero_only=True)
    ev = e.evaluate(plan,c.unavailable_payload('FAILED'),c.path_frames(89,(37,0,0)),[0]*32)
    assert ev['summary']['wrong_source_units'] == 22
    assert all(b['vote_count']==0 and b['wrong_source_unit_votes']==0 and b['wrong_segment_votes']==0 for b in ev['bits'])
    oracle = c.direct_plan(89,(37,0,0))
    payload = c.reduce_votes(oracle,{})
    assert payload['status']=='FAILED'
    failed = e.evaluate(oracle,payload,c.path_frames(89,(37,0,0)),[0]*32)
    assert not any(s['observation']=='COMPLETE_PILOT_AVAILABLE' for s in failed['source_states'])
    observed = e.evaluate(oracle,payload,c.path_frames(89,(37,0,0)),[0]*32,pair_available_starts=[4])
    assert [s['state'] for s in observed['source_states'] if s['observation']=='COMPLETE_PILOT_AVAILABLE'] == [5]


@pytest.mark.parametrize('status', ['INCOMPLETE','UNRESOLVED','FAILED','INTERRUPTED','NOT_ENTERED'])
def test_frame_execution_failure_separate_from_exact_ties(status):
    path = c.unavailable_path(89,status)
    plan = c.unit_plan(path)
    result = e.evaluate(plan,c.unavailable_payload(status),c.path_frames(89,(37,0,0)),[0]*32)
    assert len(result['frames']) == 89 and all(f['status']==status for f in result['frames'])
    summary = result['summary']
    assert summary['unresolved_frames'] == (89 if status=='UNRESOLVED' else 0)
    assert summary['incomplete_frames'] == (89 if status=='INCOMPLETE' else 0)
    assert summary['unavailable_frames'] == (0 if status in ('INCOMPLETE','UNRESOLVED') else 89)
    assert summary['mapped_frames'] == 0
    assert len(result['bits']) == 32 and len(result['source_cells'])==44 and len(result['source_states'])==22
