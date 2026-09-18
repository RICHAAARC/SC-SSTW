"""CPU-only checks of offline observations and retained denominators."""
import copy
import inspect
import json
import pytest
import torch
from main.tube_state import initial_noise as method
from experiments.wan_state_clock import inversion_soft_observation as analysis

pytestmark=pytest.mark.unit
torch.set_num_threads(1)
PAYLOAD=[-1,1]*8


def test_demodulation_amplitude_and_fixed_windows():
    book=method.codebook(b'offline-test')
    base=torch.full(method.SHAPE,2.)
    marked=method.write(base,book,PAYLOAD)
    amp,votes=analysis.demodulate(marked,book)
    assert amp.shape==(46,160,16) and amp.dtype==torch.float64
    assert amp[0,0].tolist()==[2*x for x in PAYLOAD]
    assert votes[0,0].tolist()==PAYLOAD
    rows=analysis.observe(marked,book)
    assert len(rows)==60
    for group,count in [('short',46),('medium_core',11),('medium_boundary',2),('full',1)]:
        assert sum(w['group']==group for w in rows)==count
    middle=[w for w in rows if w['group'].startswith('medium')]
    assert sorted(t for w in middle for t in range(w['start'],w['stop']))==list(range(46))
    assert rows[-1]['soft_vector']==[2*x for x in PAYLOAD]
    assert rows[-1]['vote_sums']==[7360*x for x in PAYLOAD]


def test_truth_only_scoring_and_soft_not_forced_to_vote():
    book=method.codebook(b'offline-test')
    z=torch.ones(method.SHAPE)
    values=torch.ones(46,160,16);values[:,:100]=-1;values[:,100:]=2
    z[0,0].reshape(46,2560)[:,book['order']]=values.reshape(46,2560)*book['pads']
    observed=analysis.observe(z,book)
    assert observed[-1]['soft_signs']==[1]*16
    assert observed[-1]['vote_signs']==[-1]*16
    assert 'truth' not in inspect.signature(analysis.observe).parameters
    payloads=[[1]*16,[-1]*16]
    scored0=analysis.score(copy.deepcopy(observed),payloads,0)
    scored1=analysis.score(copy.deepcopy(observed),payloads,1)
    for a,b in zip(scored0,scored1):
        assert {k:v for k,v in a.items() if k!='reporting_only'}=={k:v for k,v in b.items() if k!='reporting_only'}
        assert a['reporting_only']['soft']['candidate_scores']==b['reporting_only']['soft']['candidate_scores']
    off=analysis.score(copy.deepcopy(observed),payloads,None)
    assert 'true_wrong_margin' not in off[0]['reporting_only']['soft']


def test_baseline_mismatch_detected():
    z=torch.zeros(method.SHAPE);book=method.codebook(b'offline-test')
    original=method.read(z,book)
    assert analysis.baseline_check(z,book,original)['status']=='MATCH'
    original['per_time'][17]['vote_sums'][0]=1
    report=analysis.baseline_check(z,book,original)
    assert report==dict(status='MISMATCH',differences=['per_time/17/vote_sums'])


def test_all_missing_preserves_samples_and_windows(tmp_path):
    root=tmp_path/'input';root.mkdir()
    (root/'manifest.json').write_text(json.dumps(dict(development=[{'id':c} for c in analysis.CASES],
        arms=analysis.ARMS,payloads=[PAYLOAD,[-x for x in PAYLOAD]],base_config={'key_utf8':'offline-test'})))
    (root/'result.json').write_text('{}')
    out=tmp_path/'new'
    report=analysis.run(root,out)
    assert report['status']=='WITH_RETAINED_FAILURES' and len(report['samples'])==12
    assert len(report['failures'])==12
    assert all(len(s['windows'])==60 for s in report['samples'])
    core=next(s for s in report['summaries'] if s['scope']=='all' and s['group']=='medium_core' and s['arm']=='MARKED')
    assert core['window_denominator']==88 and core['missing_or_failed_windows']==88
    boundary=next(s for s in report['summaries'] if s['scope']=='all' and s['group']=='medium_boundary' and s['arm']=='MARKED')
    assert boundary['window_denominator']==16
    assert (out/'protocol.json').exists() and (out/'result.json').exists()
    with pytest.raises(FileExistsError): analysis.run(root,out)
