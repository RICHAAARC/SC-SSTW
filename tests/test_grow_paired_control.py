"""Synthetic-only paired-budget and fixed-reader preparation tests."""
from types import SimpleNamespace
import json,math
import pytest,torch
from main.tube_state import grow_temporal_difference as carrier,grow_control_budget as budget,grow_readout_comparison as scoring
from runtime.wan import grow_paired_control as rt,grow_temporal_difference as prior
from experiments.wan_state_clock import grow_paired_control_run as gen,grow_paired_control_full as full,grow_temporal_difference_media as media,grow_hard_soft_compare as compare
from test_grow_control_transfer import native,Model
pytestmark=pytest.mark.unit
torch.set_num_threads(1)

def test_budget_semantics_and_zero_response():
    assert budget.cumulative_energy([3,4])==25 and budget.matching_scale(25,1)['scale']==5
    assert budget.matching_scale(1,0)['scale'] is None and budget.matching_scale(0,0)['scale']==0
    with pytest.raises(ValueError):budget.matching_scale(-1,2)

def test_native_multi_matches_prior_and_last_matches_energy(monkeypatch,native):
    monkeypatch.setattr(carrier,'SHAPE',(1,2,46,12,12));z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(172))*.1;book=carrier.codebook(b'paired')
    common=(z,torch.tensor(1.),torch.tensor(-1.),torch.float32,5.,book,0)
    rows=[];calls=[]
    a=rt.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*common,lambda k,d:calls.append((k,d)),rows.append,lambda *x:None,control_indices=rt.CONTROL_INDICES)
    b=prior.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*common,lambda *x:None,lambda x:None,lambda *x:None,control_indices=prior.CONTROL_INDICES)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    energy=budget.cumulative_energy([r.get('control_induced_delta_rms',0) for r in rows]);last=[];lcalls=[]
    rt.generate(SimpleNamespace(transformer=Model(),scheduler=native()),*common,lambda k,d:lcalls.append((k,d)),last.append,lambda *x:None,control_indices=(49,),target_energy=energy)
    assert sum(r['controlled'] for r in last)==1 and lcalls.count(('scheduler_step',True))==50
    assert lcalls.count(('response_probe_step',True))==1 and lcalls.count(('unit_response_probe_step',True))==1
    match=last[-1]['energy_matching'];assert match['actual_energy']==pytest.approx(energy,rel=2e-4,abs=1e-10)
    assert last[-1]['loss_after_local']==last[-1]['loss_after']
    assert 'unit_control_local' in last[-1] and last[-1]['terminal_vs_last_controlled_clean_rms']<1e-6

def test_fake_five_arm_generation_and_media(monkeypatch,native,tmp_path):
    monkeypatch.setattr(carrier,'SHAPE',(1,2,46,12,12));monkeypatch.setattr(media,'RGB_SHAPE',(3,4,4,3))
    z=torch.randn(carrier.SHAPE,generator=torch.Generator().manual_seed(43))*.1;pipe=SimpleNamespace(transformer=Model(),scheduler=native())
    monkeypatch.setattr(gen,'prepare_generation',lambda *a,**kw:(pipe,z,torch.tensor(1.),torch.tensor(-1.),torch.float32))
    root=tmp_path/'generation';case=gen.CASES[0];g=gen.run_case(case,root/case)
    assert g['status']=='EXECUTION_COMPLETE' and len(g['videos'])==5
    assert g['actual_calls']['transformer_completed']==500 and g['actual_calls']['response_probe_step_completed']==42 and g['actual_calls']['unit_response_probe_step_completed']==2
    assert g['videos']['LAST_A']['actual_response_energy']==pytest.approx(g['videos']['MULTI_A']['actual_response_energy'],rel=2e-4,abs=1e-10)
    (root/'result.json').write_text(json.dumps({'cases':{case:g}}));vae=torch.nn.Linear(1,1);vae.config=SimpleNamespace(_commit_hash='fake');cache={}
    monkeypatch.setattr(media,'load_frozen_vae',lambda *a:vae);monkeypatch.setattr(media,'decode_normalized_latent',lambda *a:torch.zeros(media.RGB_SHAPE)+.5);monkeypatch.setattr(media,'reencode_rgb24_readback',lambda *a:torch.zeros(carrier.SHAPE))
    def save(rgb,path,*a):path.parent.mkdir(exist_ok=True);path.write_bytes(b'fake');cache[str(path)]=rgb
    monkeypatch.setattr(media,'encode_rgb',save);monkeypatch.setattr(media,'read_mp4',lambda p:cache[str(p)])
    m=media.run_case(case,tmp_path/'media'/case,root,expected_source_run='generation',expected_source_commit=g['source_commit'],expected_manifest=gen.load(gen.MANIFEST),arms=gen.ARMS,messages=gen.MESSAGES,compare_readouts=True)
    assert m['status']=='EXECUTION_COMPLETE' and m['actual_calls']['vae_encode_completed']==15
    assert len(m['videos'])==5 and m['videos']['LAST_B']['layers']['mp4']['truth']==1
    assert 'hard_soft_candidates' in m['videos']['LAST_B']['layers']['mp4']

def test_full_failure_roster_and_stage_budget(tmp_path,monkeypatch):
    monkeypatch.setattr(gen,'run_all',lambda *a:(_ for _ in ()).throw(RuntimeError('synthetic failure')));calls=[]
    monkeypatch.setattr(full.subprocess,'run',lambda *a,**kw:(calls.append(a) or SimpleNamespace(returncode=1)))
    r=full.run(tmp_path/'full');assert len(calls)==4 and r['video_denominator']==20 and r['layer_denominator']==80
    assert r['fixed_calls']['transformer']==2000 and r['fixed_calls']['vae_encode']==60
    assert all(len(c['videos'])==5 for c in r['stages']['media']['cases'].values())
    assert all(v['missing_or_failed_marked']==8 for v in r['recovery_summary']['MULTI'].values())

def test_hard_soft_and_candidate_attribution_distinct():
    agg={'votes_per_bit':92,'vote_sums':[2]*16,'signs':[1]*16,'coefficient_mean_diagnostic':[-100.]+[1.]*15}
    s=scoring.score(agg);r=scoring.compare(s,[1]*16);c=scoring.candidate_scores(s,[[1]*16,[-1]*16])
    assert r['hard']['exact'] and r['soft']['bit_errors_including_erasures']==1
    assert c['soft']['top']==1 and not r['soft']['exact']
    zeros=scoring.score(dict(agg,vote_sums=[0]*16,signs=[0]*16,coefficient_mean_diagnostic=[0.]*16))
    assert scoring.candidate_scores(zeros,[[1]*16,[-1]*16])['hard']['tie']

def test_comparison_synthetic_missing_denominator(tmp_path):
    source=tmp_path/'input';source.mkdir();p=source/'result.json';p.write_text(json.dumps({'cases':{}}))
    r=compare.run(p,tmp_path/'out');assert len(r['records'])==48 and all(v['missing']==8 for v in r['summary'].values())
