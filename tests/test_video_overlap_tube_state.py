import hashlib
import math
import json
from pathlib import Path
import numpy as np
import pytest
from main.tube_state import video_overlap_tube_state as m
from experiments.wan_state_clock import video_overlap_tube_state_run as run
pytestmark=pytest.mark.unit


@pytest.fixture(scope='module')
def source():return m.synthesize('watermark')[0,:,1:].transpose(1,0,2,3)


def test_actual_scatter_projection_budget(source):
    assert np.linalg.norm(source)==pytest.approx(1,abs=2e-15)
    values=m.extract(source,'watermark',np.ones((45,4),bool))
    np.testing.assert_allclose(values,m.composite_signs('watermark')*m.PUBLIC.alpha,atol=1e-17,rtol=0)
    assert np.count_nonzero(source[:,[i for i in range(16) if i!=4]])==0
    fp32=m.synthesize('watermark',dtype=np.float32)[0,:,1:].transpose(1,0,2,3)
    np.testing.assert_allclose(m.extract(fp32,'watermark',np.ones((45,4),bool)),values,atol=4e-9,rtol=0)
    assert not np.array_equal(m.bases('watermark'),m.bases('watermark-wrong'))
    assert not np.array_equal(m.state_code('watermark'),m.state_code('watermark-wrong'))


def test_catalog_and_fixed_roster():
    counts=[(len(m.catalog(L)),sum(r['structurally_valid'] for r in m.catalog(L))) for L in m.PUBLIC.observed_lengths]
    assert sum(n for n,v in counts)==20250
    assert [v for n,v in counts]==[944,915,882,174,89,45]
    assert sum(v for n,v in counts)==3049
    physical,conditions=run.roster(run.load_config())
    assert len(physical)==292 and len(conditions)==1704
    assert sum(c['group']=='primary' for c in conditions)==1680
    assert len({c['id'] for c in conditions})==1704


@pytest.mark.parametrize('kind,p',[('NO_EDIT',None),('DELETE',16),('REPEAT',16)])
def test_actual_edits_preserve_truth_in_top(source,kind,p):
    Y=run.edit_rows(source[6:37],kind,p)
    result=m.infer(Y,'watermark',np.ones((len(Y),4),bool))
    truth=run.edit_rows(np.arange(7,38),kind,p).tolist()
    rows=m.catalog(len(Y))
    assert truth in [rows[i]['taus'] for i in result['summary']['top_catalog_indices']]
    assert not result['summary']['state_path_accepted']
    assert not result['summary']['accepted_payload']


def test_independent_sse_same_mask_and_boundary(source):
    Y=source[:31].copy();Y[:,4,8:12,12:16]+=.003
    mask=run.availability(31,'row16');result=m.infer(Y,'watermark',mask)
    observed=m.extract(Y,'watermark',mask)
    composite=m.composite_signs('watermark');rows=m.catalog(31)
    for index in result['valid_catalog_indices'][::71]:
        row=rows[index];terms=[];M=0
        for j,tau in enumerate(row['taus']):
            for block in range(4):
                if mask[j,block]:
                    for age in range(4):
                        for bit in range(2):
                            terms.append((observed[j,block,age,bit]-m.PUBLIC.alpha*int(composite[tau-1,block,age,bit]))**2)
                            M+=1
        got=result['path_costs'][result['valid_catalog_indices'].index(index)]
        assert got==pytest.approx(math.fsum(terms)/M,abs=1e-17)
    assert result['available_dimensions']==30*32


def test_masked_equivalence_retains_unobserved_tau(source):
    mask=run.availability(31,'row16');f=m.family_receipt('watermark',31,mask);rows=m.catalog(31)
    cls=next(c for c in f['classes'] if len(c['member_catalog_indices'])>1)
    first=rows[cls['member_catalog_indices'][0]]
    result=m.infer(source[np.array(first['taus'])-1],'watermark',mask)
    assert set(cls['member_catalog_indices'])<=set(result['summary']['top_catalog_indices'])
    assert any(len(ts)>1 for ts in cls['tau_feasible_sets'])
    assert not result['summary']['unique_model_hypothesis']


def test_near_numeric_tie_keeps_classes(source):
    rows=m.catalog(31);valid=[r for r in rows if r['structurally_valid']]
    a=source[np.asarray(valid[0]['taus'])-1];b=source[np.asarray(valid[1]['taus'])-1]
    Y=.5*(a+b);result=m.infer(Y,'watermark',np.ones((31,4),bool))
    finite=np.asarray(result['class_costs']);expected=np.flatnonzero(finite-finite.min()<=1e-12).tolist()
    assert result['summary']['top_class_indices']==expected
    # All classes within global tolerance survive; a canonical representative never deletes them.
    fam=m.family_receipt('watermark',31,np.ones((31,4),bool))
    assert result['summary']['top_catalog_indices']==sorted(i for c in expected for i in fam['classes'][c]['member_catalog_indices'])


def test_noenergy_allmissing_nonfinite(source):
    Y=source[:31].copy();mask=np.ones((31,4),bool)
    zero=m.infer(np.zeros_like(Y),'watermark',mask)
    assert zero['summary']['status']=='NO_ENERGY' and zero['summary']['canonical_catalog_index'] is None
    assert zero['counts']['scored']==915
    missing=m.infer(Y,'watermark',np.zeros((31,4),bool))
    assert missing['summary']['reason']=='NO_OBSERVATIONS' and missing['counts']['catalog']==2745
    assert missing['summary']['canonical_catalog_index'] is None
    Y[0,4,8,12]=np.nan
    bad=m.infer(Y,'watermark',mask)
    assert bad['summary']['status']=='INCOMPLETE' and bad['counts']['scored']==0
    assert len(bad['path_costs'])==915
    mask[0,0]=False
    assert m.infer(Y,'watermark',mask)['summary']['status']=='COMPLETE'


def test_public_cache_isolation(source):
    mask=np.ones((45,4),bool);before=m.infer(source,'watermark',mask)
    receipt=m.family_receipt('watermark',45,mask);receipt['classes'].reverse();receipt['valid_catalog_indices'].clear()
    family=m.family('watermark',45,mask);family['means'][:]=123;family['membership'][:]=0;family['classes'].clear()
    rows=m.catalog(45);rows[0]['taus'][0]=45;rows[0]['status']='MUTATED'
    before['valid_catalog_indices'].reverse()
    after=m.infer(source,'watermark',mask)
    assert after['summary']==before['summary']
    assert after['path_costs']==before['path_costs']
    assert after['valid_catalog_indices']!=before['valid_catalog_indices']


def test_blind_api_and_input_immutability(source):
    Y=source[:31].copy();before=run.tensor_sha(Y)
    a=m.infer(Y,'watermark',np.ones((31,4),bool))
    with pytest.raises(TypeError):m.infer(Y,'watermark',np.ones((31,4),bool),source_offset=7)
    with pytest.raises(TypeError):m.infer(Y,'watermark',np.ones((31,4),bool),sample_id='truth')
    assert run.tensor_sha(Y)==before
    assert a==m.infer(Y.copy(),'watermark',np.ones((31,4),bool))


def test_unknown_insert_is_not_repeat_claim(source):
    Y=source[:31];p=16
    np.testing.assert_array_equal(run.edit_rows(Y,'EXACT_COPY_INSERT',p),run.edit_rows(Y,'REPEAT',p))
    assert not np.array_equal(run.edit_rows(Y,'MIDPOINT_INSERT',p),run.edit_rows(Y,'REPEAT',p))
    assert not np.array_equal(run.edit_rows(Y,'NULL_INSERT',p),run.edit_rows(Y,'REPEAT',p))


def test_raw_saved_before_truth_and_failure_slots(tmp_path,source,monkeypatch):
    config=run.load_config();physical,conditions=run.roster(config)
    c=conditions[0];p=physical[0];Y=np.zeros_like(source)
    result=m.infer(Y,'watermark',np.ones((45,4),bool))
    ref=run.atom(tmp_path/'raw.json.gz',dict(inference=result));before=run.sha(ref['path'])
    monkeypatch.setattr(m,'infer',lambda *a,**kw:pytest.fail('posthoc must never rerun inference'))
    evaluation=run.posthoc(run.read(ref['path'])['inference'],c,p,m.catalog(45))
    assert evaluation['status']=='EVALUATED' and evaluation['model_status']=='NO_ENERGY'
    assert run.sha(ref['path'])==before
    failure=run.failed_raw(45,'injected failure')
    assert failure['counts']==dict(catalog=4005,scorable=89,structurally_excluded=3916,scored=0)
    assert len(failure['path_costs'])==89 and failure['summary']['canonical_catalog_index'] is None
    assert run.posthoc(failure,c,p,m.catalog(45))['status']=='INCOMPLETE'


def test_frozen_config_identity():
    assert run.sha(run.CONFIG)==run.CONFIG_SHA
    root=run.ROOT
    assert run.sha(root/'docs/video_overlap_tube_state_v1.md')=='833485448f6418f4db4ae3d2f3461f95906604ff428dafc7b0359c6977c182e7'


def test_cost_integrity_detects_truncated_completed_costs(tmp_path):
    cat=run.atom(tmp_path/'catalog.json.gz',dict(rows=[dict(structurally_valid=True)]))
    fam=run.atom(tmp_path/'family.json.gz',dict(classes=[{}]))
    raw=dict(summary=dict(status='COMPLETE'),counts=dict(scored=1),valid_catalog_indices=[0],path_costs=[],class_costs=[0.])
    ref=run.atom(tmp_path/'raw.json.gz',dict(inference=raw,catalog=cat,equivalence_family=fam))
    audit=run.verify_cost_integrity(dict(conditions={'x':dict(raw=ref,group='primary')}))
    assert audit['status']=='INCOMPLETE'
    assert any('path cost slot mismatch' in r['error'] for r in audit['failures'])
    assert audit['new_extractions']==audit['new_scores']==0
