"""KKT/readback/physical-isolation checks. No real-data solver call or model."""
import copy
import json
from pathlib import Path
import pytest
import torch
from main.tube_state import local_joint_p48_secondary_v1 as method
from runtime.wan import local_joint_p48_secondary_v1 as runtime
from runtime.wan.local_joint_state_payload_provider_v1 import LatentSupport

pytestmark=pytest.mark.unit
SHAPE=(1,1,2,2,2)
SUPPORT=LatentSupport((1,2),((0,1,0,1),(0,1,1,2),(1,2,0,1),(1,2,1,2)))


def fixture(target=.25,gradient=(1.,0.)):
    a=torch.zeros(SHAPE);g=torch.zeros_like(a)
    g[0,0,1,0,:]=torch.tensor(gradient)
    receipt=dict(status='CANDIDATE_DEFINED',secondary_status='optimal',secondary_target=target,
        physical_candidates=1,secondary_primal=[-target],secondary_dual=dict(objective=[target/2]*4,delta_ball=0.,control_ball=0.))
    return a,[g]*4,dict(status='NUMERICALLY_UNRESOLVED',physical_candidates=0,secondary_target=target),receipt


def test_fixed_witness_certificate_and_fp32_hard_balls():
    a,g,s,r=fixture();candidate,checked=method.checked_candidate(a,g,s,r,SUPPORT)
    assert candidate is not None and checked['secondary_certificate']['stationarity_residual']==0
    assert float(candidate.double().norm())<=.02
    a,g,s,r=fixture(1.,(3.,4.))
    candidate,checked=method.checked_candidate(a,g,s,r,SUPPORT)
    assert candidate is None and checked['actual_control_l2']>.02
    assert 'no rescale' in checked['reason']


def test_inaccurate_or_bad_certificate_never_promoted():
    a,g,s,r=fixture();r['secondary_status']='optimal_inaccurate'
    r['secondary_certificate']={'stationarity_residual':123.}
    candidate,reloaded=method.checked_candidate(a,g,s,r,SUPPORT)
    assert candidate is None and reloaded['secondary_certificate']['stationarity_residual']==0
    bad=copy.deepcopy(r);bad['secondary_primal']=[float('nan')]
    with pytest.raises(ValueError,match='invalid secondary primal/dual'):
        method.checked_candidate(a,g,s,bad,SUPPORT)
    r['secondary_status']='optimal';r['secondary_dual']['objective']=[0.]*4
    c,checked=method.checked_candidate(a,g,s,r,SUPPORT)
    assert c is None and checked['secondary_certificate']['stationarity_residual']>.1
    r['secondary_target']+=1e-10
    with pytest.raises(ValueError,match='target'):method.checked_candidate(a,g,s,r,SUPPORT)


def source(tmp):
    a,g,s,r=fixture();r['secondary_status']='optimal_inaccurate';r['physical_candidates']=0
    original=tmp/'old';original.mkdir()
    torch.save(a,original/'anchor.pt')
    for n,v in zip(('S','P','W','Q'),g):
        p=original/'objectives'/n;p.mkdir(parents=True);torch.save(v,p/'control_cotangent.pt')
    for n in ('ZERO','ANCHOR'):
        p=original/n;p.mkdir();(p/'values.json').write_text(json.dumps(dict(correlations=[0.]*22,margins=[0.]*32)))
    (original/'solution.json').write_text(json.dumps(s))
    record=dict(fixed_denominators=dict(endpoints=3),calls=[dict(kind='original_call',status='RETURNED')],endpoints={n:dict(status='COMPLETE') for n in ('ZERO','ANCHOR')})
    (original/'result.json').write_text(json.dumps(record))
    decision=tmp/'cpu.json';decision.write_text(json.dumps(r))
    return original,decision


def test_report_reloads_four_gradients_without_solver_or_cuda(tmp_path,monkeypatch):
    import cvxpy
    def forbidden(*a,**k):raise AssertionError('solver/GPU/gradient entry forbidden')
    monkeypatch.setattr(cvxpy.Problem,'solve',forbidden)
    monkeypatch.setattr(torch.autograd,'grad',forbidden)
    monkeypatch.setattr(torch.cuda,'is_available',forbidden)
    src,decision=source(tmp_path);prior=(src/'result.json').read_bytes()
    def run():return runtime.run(src,tmp_path/'new',decision,shape=SHAPE,support=SUPPORT)
    r=run();assert r['status']=='NO_RESOLVED_CANDIDATE'
    assert len(r['objectives'])==4 and all(v['status']=='SAVED_GRADIENT_READ' for v in r['objectives'].values())
    assert r['calls']==[] and r['secondary_solve_count']==r['primary_solve_count']==0
    assert (src/'result.json').read_bytes()==prior and r['endpoints']['CANDIDATE']['status']=='MISSING'
    decision.write_text('{}')  # Existing stage uses its saved one-shot receipt.
    assert run()['status']=='NO_RESOLVED_CANDIDATE'
    (src/'objectives/W/control_cotangent.pt').unlink()
    failed=run();assert failed['status']=='SAVED_EVIDENCE_ERROR' and failed['calls']==[] and failed['failures']


def test_delivered_notebook_and_archive_cpu_only():
    import ast,zipfile,io
    from scripts import build_local_joint_p48_secondary_v1_notebook as b
    nb=b.build_notebook();sources=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert sources[0]=="from google.colab import drive\ndrive.mount('/content/drive')"
    for i,s in enumerate(sources):ast.parse(s,filename=f'cell{i}')
    text='\n'.join(sources)
    assert 'torch.cuda' not in text and 'solve_saved_secondary(' not in text
    assert 'secondary-direct-v1' in text and 'SAVED_SECONDARY_REPORT_ONLY' in text
    assert 'HF_TOKEN' not in text and "'diffusers'" not in text
    assert json.loads((b.ROOT/('notebooks/'+b.NAME+'_colab.ipynb')).read_text())==nb
    with zipfile.ZipFile(io.BytesIO(b.portable_archive())) as z:
        assert b.DECISION_PATH in z.namelist()
        assert json.loads(z.read(b.DECISION_PATH))['secondary_status']=='optimal_inaccurate'
