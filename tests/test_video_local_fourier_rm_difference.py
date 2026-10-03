"""Targeted NumPy engineering checks; cached observations are never regenerated."""
import ast,gzip,json
from pathlib import Path
import numpy as np
import pytest
from experiments.wan_state_clock import video_local_fourier_rm_difference_receiver as rec
from experiments.wan_state_clock import video_local_fourier_rm_difference_run as run
pytestmark=pytest.mark.unit


def fixture_for_path(key,catalog_index):
    taus=rec.frozen.catalog(44)[catalog_index]['taus']
    # Hand-constructed observed coefficient sequence plus a common per-dimension host.
    support=rec.frozen.composite_signs(key)[np.asarray(taus)-1].astype(np.float64)*rec.PUBLIC.alpha
    offset=np.arange(128,dtype=np.float64).reshape(4,4,8)/1000
    return support+offset


def test_correct_path_from_handconstructed_sequence_not_truth_input():
    q=fixture_for_path('watermark',0);inf=rec.infer_difference(q,'watermark',np.ones((44,4),bool))
    assert inf['summary']['status']=='COMPLETE' and inf['summary']['canonical_catalog_index']==0
    assert inf['summary']['top_catalog_indices']==[0] and inf['path_costs'][0]<1e-30
    assert inf['available_dimensions']==5504 and inf['edge_local']['scored']==43*132
    assert inf['summary']['state_path_accepted'] is False and inf['summary']['accepted_payload'] is False
    assert inf['edge_local']['pairs']==[[u,v] for u in range(1,46) for v in range(u,min(45,u+2)+1)]


@pytest.mark.parametrize('kind,event_i',[('REPEAT',44),('SKIP',2)])
def test_repeat_skip_candidate_uses_actual_source_transition(kind,event_i):
    catalog=rec.frozen.catalog(44);idx=next(i for i,row in enumerate(catalog) if row['tau1']==1 and row['event_type']==kind and row['event_i']==event_i)
    q=fixture_for_path('watermark',idx);inf=rec.infer_difference(q,'watermark',np.ones((44,4),bool));ids=inf['valid_catalog_indices']
    assert inf['summary']['canonical_catalog_index']==idx and inf['path_costs'][ids.index(idx)]<1e-30
    edge=inf['edge_local']['rows'][event_i-2]
    if kind=='REPEAT':
        pairs=inf['edge_local']['pairs'];stay=[i for i,(u,v) in enumerate(pairs) if u==v]
        assert len(stay)==45 and edge['top_pair_indices']==stay
        assert all(abs(edge['costs'][i])<1e-30 for i in stay)
    else:
        target=[catalog[idx]['taus'][0],catalog[idx]['taus'][1]];pairid=inf['edge_local']['pairs'].index(target)
        assert target==[1,3] and edge['costs'][pairid]<1e-30


def test_mask_intersection_requires_both_windows():
    mask=np.ones((44,4),bool);mask[7,1]=False
    q=fixture_for_path('watermark',0);q[7,1]=np.nan
    inf=rec.infer_difference(q,'watermark',mask);edge=np.asarray(inf['edge_availability'])
    expected=np.ones((43,4),bool);expected[6:8,1]=False
    assert np.array_equal(edge,expected) and inf['available_dimensions']==5504-64
    assert inf['edge_local']['rows'][6]['available_dimensions']==96 and inf['edge_local']['rows'][7]['available_dimensions']==96
    assert inf['summary']['canonical_catalog_index']==0 and np.isfinite(inf['difference_projection']).all()


def test_exact_difference_classes_preserve_sparse_and_no_support_ambiguity():
    mask=np.zeros((44,4),bool);mask[:2]=True
    family=rec.difference_family('watermark',mask)
    assert family['available_dimensions']==128 and len(family['classes'])==5
    assert sum(len(c['member_catalog_indices']) for c in family['classes'])==174
    assert any(len(c['member_catalog_indices'])>1 and any(len(t)>1 for t in c['tau_feasible_sets']) for c in family['classes'])
    empty=rec.difference_family('watermark',np.zeros((44,4),bool))
    assert len(empty['classes'])==1 and len(empty['classes'][0]['member_catalog_indices'])==174 and len(empty['classes'][0]['tau_feasible_sets'])==44


def test_no_support_zero_energy_and_invalid_observations_reject_json_safely():
    q=fixture_for_path('watermark',0);empty=rec.infer_difference(q,'watermark',np.zeros((44,4),bool))
    assert empty['summary']['reason']=='NO_OBSERVATIONS' and empty['summary']['canonical_catalog_index'] is None
    assert len(empty['classes'])==1 and len(empty['path_costs'])==174 and empty['counts']['scored']==0
    constant=np.broadcast_to(q[10],q.shape).copy();zero=rec.infer_difference(constant,'watermark',np.ones((44,4),bool))
    assert zero['summary']['status']=='NO_ENERGY' and zero['summary']['canonical_catalog_index'] is None and zero['counts']['scored']==174
    assert not zero['summary']['state_path_accepted'] and not zero['summary']['accepted_payload']
    bad=q.copy();bad[0,0,0,0]=np.nan
    for value in (bad,np.full(q.shape,1e307)*np.where(np.arange(44)[:,None,None,None]%2,1,-1)):
        result=rec.infer_difference(value,'watermark',np.ones((44,4),bool))
        assert result['summary']['reason']=='INVALID_OBSERVATION' and result['summary']['canonical_catalog_index'] is None
        json.dumps(result,allow_nan=False)


def synthetic_source(tmp_path):
    root=tmp_path/'input';root.mkdir();cfg=run.load_config();meta=dict(source_sha=cfg['input']['source_sha'],method_version=rec.PUBLIC.method_version,
        fixed_denominator={},reference_environment_limit='old MP4 reference only',baseline={},vae_setup={},reads={})
    for arm in run.ARMS:
        (root/arm).mkdir()
        for stage in run.STAGES:
            for kid,key in [('CORRECT',cfg['key']),('WRONG',cfg['wrong_key'])]:
                q=fixture_for_path(key,0);inf=rec.absolute_control(q,key,np.ones((44,4),bool));role='NEW_CHANNEL_DIAGNOSTIC' if stage in run.NEW else 'OFFLINE_REFERENCE'
                raw=dict(g=0,R=44,truth_inputs=False,role=role,inference=inf,payload=dict(status='READ',decoded_bits=run.message_bits(cfg['message']),truth_used=False,R=44,role='repeated payload'))
                sid=f'{arm}/{stage}/{kid}';path=root/arm/(stage+'.'+kid+'.raw.json.gz');run.dump_gzip(path,raw);meta['reads'][sid]=dict(status='COMPLETE',sha256=run.sha(path),path='/unused/wrong/original/path')
    run.dump(root/'result.json',meta);return root


def test_fixed_saved_run_and_truth_joins_leave_receiver_unchanged(tmp_path):
    cfg=run.load_config();root=synthetic_source(tmp_path);store=run.Store(tmp_path/'output',root);run.process(store,cfg,root)
    files=[store.output/name for name in ('receiver_channel_readouts.json','receiver_reference_readouts.json','cached_payload_readouts.json')]
    before={str(p):p.read_bytes() for p in files};run.evaluate(store,cfg)
    assert before=={str(p):p.read_bytes() for p in files} and run.finish(store,cfg)
    assert store.data['counts']['path_score_records']==48 and store.data['counts']['path_costs']==8352 and store.data['counts']['difference_edge_costs']==136224
    assert store.data['counts']['path_posthoc']==48 and store.data['counts']['cached_payload_evaluations']==48
    assert all(row['status']=='REUSED' for row in store.data['cached_payload'].values())
    assert set(store.data['calls'])=={'source_raw_read','absolute_control','difference_infer','cached_payload_reuse'}
    assert not store.data['actual_model_calls'] and not store.data['actual_media_calls']
    for p in files[:2]:
        text=p.read_text();assert 'registered_tau' not in text and 'true_path_rank' not in text and 'bit_errors' not in text
    assert all(row['saved_absolute_comparison']['status']=='MATCH' for row in store.data['inputs'].values())


def test_missing_input_retains_fixed_rows_and_other_inputs(tmp_path):
    cfg=run.load_config();root=synthetic_source(tmp_path);sid='OFF/FLOAT_VAE_ROUNDTRIP/CORRECT'
    (root/'OFF'/'FLOAT_VAE_ROUNDTRIP.CORRECT.raw.json.gz').unlink()
    store=run.Store(tmp_path/'output',root);run.process(store,cfg,root);run.evaluate(store,cfg)
    assert not run.finish(store,cfg) and store.data['inputs'][sid]['status']=='FAILED'
    assert len(store.data['inputs'])==24 and len(store.data['mode_reads'])==48 and len(store.data['path_posthoc'])==48 and len(store.data['payload_evaluations'])==48
    assert store.data['counts']['path_score_records']==46 and store.data['counts']['cached_payload_readouts']==23
    assert store.data['mode_reads']['STATE_MULTI/RGB8_VAE_ROUNDTRIP_WITHOUT_CODEC/CORRECT/ADJACENT_DIFFERENCE']['status']=='COMPLETE'
    assert all(row['status']!='PENDING' for row in store.data['mode_reads'].values())


def test_notebook_binding_fixed_direct_numpy_only(tmp_path):
    from scripts import build_video_local_fourier_rm_difference_notebook as builder
    nb=json.loads(builder.build('a'*40,tmp_path/'bound.ipynb').read_text());sources=[''.join(c['source']) for c in nb['cells']]
    assert sources[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell,text in zip(nb['cells'],sources):
        if cell['cell_type']=='code':ast.parse(text);assert cell['outputs']==[] and cell['execution_count'] is None
    code='\n'.join(text for cell,text in zip(nb['cells'],sources) if cell['cell_type']=='code')
    assert 'numpy' in code and 'torch' not in code.lower() and 'huggingface' not in code.lower() and 'venv' not in code.lower() and 'ffmpeg' not in code.lower()
    assert '--input' not in builder.RUN and 'SOURCE_SHA' in code and nb['metadata']['candidate_binding']['source_sha']=='a'*40
