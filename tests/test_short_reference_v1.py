import json
from pathlib import Path
import numpy as np
import pytest
from experiments.paper_results_v1 import short_reference_v1 as m

pytestmark=pytest.mark.unit


def test_fixed_pressure_geometry_and_majority():
    assert [len(m.view_indices(v)) for v in m.VIEWS]==[9,19,17,35]
    for v,n in [('SHORT9_HOLD',9),('SHORT17_HOLD',17)]:
        idx=m.view_indices(v)
        assert len(np.unique(idx))==n
        unique,counts=np.unique(idx,return_counts=True)
        assert unique[counts.argmax()]==37+n//2
        assert counts.max()>len(idx)/2
    assert len(m.planned_rows())==48
    assert sum(r['received_frames'] for r in m.planned_rows())==960


def test_reference_preserves_patchwise_ideal_signal():
    w,e,rho=m.public_templates()
    wp=w.reshape(181,160,4,16);ep=e.reshape(181,160,4,16)
    amplitudes=(1+np.arange(160)%7)[None,:,None,None]/7
    y=amplitudes*wp+np.arange(4)[None,None,:,None]/3
    got=np.sum(ep*y,axis=(1,2,3))/rho
    expected=np.sum(amplitudes*wp*wp,axis=(1,2,3))/rho
    np.testing.assert_allclose(got,expected,atol=1e-14,rtol=0)
    with pytest.raises(ValueError):m.observed_scores(np.zeros((1,4,40,64),np.float32),(w,e,rho))


def test_resume_preserves_complete_and_retries_only_technical(tmp_path,monkeypatch):
    source=tmp_path/'source';folder=source/'joint/run_state/latents/framewise/clean_mp4';folder.mkdir(parents=True)
    for c in m.CASES:np.save(folder/f'{c}.npy',np.zeros((181,4,40,64),np.float32))
    monkeypatch.setattr(m,'public_templates',lambda:None)
    monkeypatch.setattr(m,'observed_scores',lambda y,t:(np.zeros((181,181)),np.zeros((181,181))))
    count=[]
    def fake(q):
        count.append(len(q))
        if len(count)==1:raise OSError('fixture transient save/input failure')
        return dict(status='UNRESOLVED',reason='EXACT_BEST_PATH_TIE',operation=None,
                    estimate=dict(path=None,score_gap=0),truth_inputs=False,message_inputs=False,attack_inputs=False)
    monkeypatch.setattr(m,'blind_select',fake)
    store=m.run(tmp_path/'out',source)
    assert len(count)==48
    assert sum(r['status']=='FAILED' for r in store.data['rows'])==1
    store=m.run(tmp_path/'out',source)
    assert len(count)==49
    assert all(r['status']=='UNRESOLVED' for r in store.data['rows'])
    store=m.run(tmp_path/'out',source)
    assert len(count)==49
    assert store.data['invocations'][-1]['files_read']==0
    result=json.loads((tmp_path/'out/summary.json').read_text())
    assert result['planned_rows']==48 and result['by_arm']['CENTERED_U']['fixed_denominator_absolute_error_sum'] is None
    row=store.data['rows'][0]
    for damage in ('missing','shape','nonfinite','json'):
        if damage=='missing':Path(row['observation']).unlink()
        elif damage=='shape':np.savez_compressed(row['observation'],q=np.zeros((1,181)))
        elif damage=='nonfinite':np.savez_compressed(row['observation'],q=np.full((row['received_frames'],181),np.nan))
        else:Path(row['selection']).write_text('{}')
        before=len(count)
        store=m.run(tmp_path/'out',source)
        assert len(count)==before+1
        assert store.data['invocations'][-1]['files_read']==1
        assert any(h['status']=='EVIDENCE_INCOMPLETE' for h in store.data['rows'][0]['history'])
        store=m.run(tmp_path/'out',source)
        assert len(count)==before+1 and store.data['invocations'][-1]['files_read']==0


def test_known_actual_method_difference_and_missing_optional_record(tmp_path,monkeypatch):
    store=m.Store(tmp_path/'out',tmp_path/'source')
    original=m.view_indices
    monkeypatch.setattr(m,'view_indices',lambda view:original(view)+1)
    with pytest.raises(ValueError,match='fixed method differs'):
        m.Store(tmp_path/'out',tmp_path/'different_available_source')
    monkeypatch.setattr(m,'view_indices',original)
    store.data.pop('fixed_method_semantics');store.save()
    restored=m.Store(tmp_path/'out',tmp_path/'different_available_source')
    assert restored.data['fixed_method_semantics']==m.fixed_method_semantics()
    assert len(restored.data['rows'])==48


def test_notebook_offline_readable_zip_fallback(tmp_path,monkeypatch):
    import shutil
    import urllib.request
    from experiments.paper_results_v1 import build_short_reference_v1_colab as builder
    result=builder.build()
    output=tmp_path/'output';output.mkdir()
    shutil.copyfile(result['companion'],output/'companion_source.zip')
    def offline(*args,**kwargs):raise OSError('fixture network unavailable')
    monkeypatch.setattr(urllib.request,'urlopen',offline)
    notebook=json.loads(Path(result['notebook']).read_text())
    setup=''.join(notebook['cells'][2]['source'])
    setup=setup.replace("Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Short-Reference-V1/resumable-short-reference-v1')",repr(output))
    setup=setup.replace("Path('/content')",repr(tmp_path))
    namespace={'PosixPath':Path,'WindowsPath':Path}
    exec(compile(setup,'fixture_notebook_setup','exec'),namespace)
    assert (namespace['CODE']/'experiments/paper_results_v1/short_reference_v1_cli.py').is_file()


def test_missing_inputs_keep_all_rows_without_fallback(tmp_path):
    store=m.run(tmp_path/'out',tmp_path/'missing')
    assert len(store.data['rows'])==48
    assert all(r['status']=='MISSING' for r in store.data['rows'])
    assert store.data['invocations'][-1]['selectors']==0
