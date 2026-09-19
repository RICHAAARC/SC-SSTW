"""Blind support/ranking and immutable-before-truth saved-tensor workflow."""
import copy
import os
from pathlib import Path
import pytest
import torch
from main.tube_state import inversion_blind_sync as method,inversion_state as state
from experiments.wan_state_clock import inversion_blind_sync_run as runner

pytestmark=pytest.mark.unit
torch.set_num_threads(1)


@pytest.mark.parametrize('shift',[0,4,13])
def test_ideal_encoded_slice_offset_message_equal_support(shift):
    book=state.codebook(b'blind-test')
    for message in (0,1):
        full=state.write(torch.ones(state.SHAPE),book,message)
        clip=full[:,:,shift:shift+33].clone()
        result=method.search(clip,book)
        assert result['ranking']['selected_pair']==dict(shift=shift,message=message)
        assert len(result['candidates'])==28 and len(result['ranking']['offset_profiles'])==14
        assert result['ranking']['different_offset_top_runner_gap']>0
        assert all(r['slice_denominator']==31 and r['axis_denominator']==62 for r in result['candidates'])
        assert all(min(r['nominal_source_indices'])>=1 and max(r['nominal_source_indices'])<=44 and r['complete_windows']==7 for r in result['shifts'])


def test_raw_amplitude_scaling_no_direction_normalization_and_boundaries():
    book=state.codebook(b'blind-scale');z=torch.randn(method.SHAPE,generator=torch.Generator().manual_seed(31))
    a=method.search(z,book);b=method.search(z*1e-6,book)
    for x,y in zip(a['candidates'],b['candidates']):assert y['score']==pytest.approx(x['score']*1e-6,abs=1e-14)
    z[:,:,0]=1e9;z[:,:,32]=-1e9
    c=method.search(z,book)
    assert a['candidates']==c['candidates']
    zero=method.search(torch.zeros(method.SHAPE),book)
    assert len(zero['ranking']['top_ties'])==28 and zero['ranking']['selected_pair'] is None
    assert zero['ranking']['selected_message'] is None and len(zero['ranking']['offset_top_ties'])==14
    assert 'detected' not in zero


def test_posthoc_non_aligned_no_best_map_or_off_detection():
    book=state.codebook(b'blind-report');result=method.search(torch.ones(method.SHAPE),book)
    original=copy.deepcopy(result)
    report=runner.posthoc(result,dict(source_start=17,source_arm='A'))
    assert report['nominal_floor']['shift']==4 and report['nominal_ceil']['shift']==5
    assert 'unique_aligned_shift_correct' not in report and 'best_compatible_correct' not in report
    off=runner.posthoc(result,dict(source_start=16,source_arm='OFF'))
    assert off['truth_reporting_only'] is None and 'unique_message_correct' not in off
    assert result==original


def test_search_persisted_before_metadata_truth_cannot_change_blind(tmp_path,monkeypatch):
    source=tmp_path/runner.SOURCE_RUN;source.mkdir()
    tensor=source/'fixture.pt';torch.save(torch.zeros(method.SHAPE),tensor);digest=runner.sha(tensor)
    root_result={'cases':{}}
    for case in runner.CASES:
        associations=[];clips={}
        for i,rid in enumerate(runner.IDS):
            path=source/case/'receiver'/f'{rid}_recovered.pt';path.parent.mkdir(parents=True,exist_ok=True);os.link(tensor,path)
            clips[rid]={'receiver':{'recovered_sha256':digest}}
            associations.append(dict(receiver_id=rid,source_start=(0,16,17)[i%3],source_arm=('OFF','A','B')[i//3]))
        root_result['cases'][case]={'clips':clips,'source_commit':runner.SOURCE_COMMIT}
        runner.dump(source/case/'attacker_manifest.json',dict(associations=associations))
    runner.dump(source/'result.json',root_result)
    output=tmp_path/'first';load=runner.load
    def guarded(path):
        if Path(path).is_relative_to(source):assert (output/'search_phase_complete.json').exists()
        return load(path)
    monkeypatch.setattr(runner,'load',guarded)
    first=runner.run(source,output)
    assert first['status']=='EXECUTION_COMPLETE' and first['summary']['unique_correct_messages']==0
    hashes=[r['blind_sha256'] for r in first['records']]
    for case in runner.CASES:
        data=load(source/case/'attacker_manifest.json')
        for row in data['associations']:
            row['source_start']+=4
            if row['source_arm']!='OFF':row['source_arm']='B' if row['source_arm']=='A' else 'A'
        runner.dump(source/case/'attacker_manifest.json',data)
    output=tmp_path/'second';second=runner.run(source,output)
    assert hashes==[r['blind_sha256'] for r in second['records']]


def test_missing_fixed_candidates_and_source_write_guard(tmp_path):
    source=tmp_path/runner.SOURCE_RUN
    with pytest.raises(ValueError,match='independent'):runner.run(source,source/'out')
    result=runner.run(source,tmp_path/'out')
    assert len(result['records'])==18 and sum(len(r['blind']['candidates']) for r in result['records'])==504
    assert result['summary']['complete_blind_clips']==0 and result['summary']['unmeasured_marked']==12
