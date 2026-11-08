"""Synthetic statistics and isolated launcher tests; no scientific data or models."""
from pathlib import Path
from types import SimpleNamespace
import json
import pytest
from main.tube_state import paper_fpr_claims_v1 as f,paper_suite_v1 as p


def test_counting_bound_duplicate_words_unknowns_and_probability_spaces():
    row=dict(known_words=['0'*32]*10,unknown_inputs=0)
    assert f.saved_family_mass(row)['exact']==1/2**32  # perfectly dependent biased outputs still valid
    row=dict(known_words=['0'*32,'1'*32],unknown_inputs=3)
    assert f.saved_family_mass(row)==dict(lower=2/2**32,upper=5/2**32,exact=None,status='SAVED_BOUNDS',reason=None)
    assert f.saved_family_mass(dict(known_words=['bad'],unknown_inputs=0))['upper']==10/2**32
    assert f.uniform_query_union(360)==360/2**32<1e-6
    models=f.protocol_rows()
    assert models[1]['theoretical_bound'] is None
    assert models[2]['implemented'] is False and models[3]['implemented'] is False


def test_three_value_coverage_does_not_turn_unknown_into_rejection():
    assert f.at_least(['MATCH']+['INPUT_UNRESOLVED']*4,1) is True
    assert f.at_least(['MATCH']*4+['INPUT_UNRESOLVED'],5) is None
    assert f.at_least(['MATCH']*4+['UNRESOLVED'],5) is False
    assert f.at_least(['MATCH']*3+['INPUT_UNRESOLVED']*2,4) is None
    assert f.at_least(['MATCH']*2+['MISMATCH','UNRESOLVED','INPUT_UNRESOLVED'],4) is False


def test_fixed48_rows_finite_sources_collision_and_whole_package_union():
    claims=[dict(source=sid,arm=arm,key=key,claim=q,view=view,verdict='MATCH' if arm=='NEW' and key=='key00' and q=='q0' else 'UNRESOLVED')
        for sid in p.SOURCES for arm in p.cohort.ARMS for key in p.cohort.KEYS for q in p.cohort.CLAIMS for view in p.VIEWS]
    families=[dict(source=sid,arm=arm,known_words=['0'*32],unknown_inputs=0,correct_query_collision=False,
        false_accept_event=False,verdict_counts={'MISMATCH':10}) for sid in p.SOURCES for arm in p.cohort.ARMS]
    families[-1].update(correct_query_collision=True,false_accept_event=None)
    rows=f.measurement_rows(claims,families);assert len(rows)==48
    b=next(r for r in rows if r['panel']=='TEST-B' and r['row_kind']=='FIXED_CLAIM' and r['arm']=='NEW' and r['primary'])
    assert b['planned_sources']==8 and b['planned_claims']==40 and b['events']['ALL5']['positive']==8
    off=next(r for r in rows if r['panel']=='TEST-B' and r['row_kind']=='FIXED_CLAIM' and r['arm']=='OFF' and r['primary'])
    assert off['events']['ANY5']['positive']==0 and off['events']['ANY5']['logical_unresolved']==40
    assert off['events']['ANY5']['iid_reference_one_sided95_upper']>.31
    total=next(r for r in rows if r['row_kind']=='RANDOM_PACKAGE_UNION' and r['panel']=='ALL')
    assert total['protocol_union_upper']==360/2**32 and total['saved_mass_union_upper']==36/2**32
    assert total['whole_package_exact_probability'] is None
    new=next(r for r in rows if r['row_kind']=='RANDOM_QUERY_ARM' and r['panel']=='TEST-B' and r['arm']=='NEW')
    assert new['events']['not_applicable']==1 and new['events']['unknown']==0
    missing=f.measurement_rows([],[])
    assert len(missing)==48
    assert all(r['events']['ALL5']['unknown']==r['planned_sources'] for r in missing if r['row_kind']=='FIXED_CLAIM')


@pytest.mark.parametrize('cancel',[False,True])
def test_finite_colab_sequence_never_launches_external_models(tmp_path,monkeypatch,cancel):
    from runtime.wan import paper_suite_colab_v1 as c
    monkeypatch.setattr(c,'OUTPUT',tmp_path/'out');monkeypatch.setattr(c,'available_source',lambda *a:tmp_path)
    monkeypatch.setattr(c.subprocess,'run',lambda *a,**kw:SimpleNamespace(returncode=0))
    monkeypatch.setattr(c.os,'killpg',lambda *a:None);monkeypatch.setattr(c,'show_results',lambda:None)
    launched=[]
    class Worker:
        pid=123456789
        def __init__(self,args,**kw):
            stage=args[args.index('--stage')+1];launched.append(stage)
            self.returncode=130 if cancel and stage=='cohort' else 0
            Path(args[args.index('--receipt')+1]).write_text(json.dumps(dict(status='INTERRUPTED' if self.returncode else 'FAKE_RETURNED')))
        def poll(self):return self.returncode
        def wait(self,**kw):return self.returncode
    monkeypatch.setattr(c.subprocess,'Popen',Worker)
    result=c.run_fpr_colab(source_directory=tmp_path)
    assert launched==(['cohort'] if cancel else ['cohort','tables','queries','tables'])
    assert result['workflow']=='FINITE_FPR_VALIDATION'
    assert result['status']==('INTERRUPTED_RETAINED' if cancel else 'FINITE_FPR_RETURNED')


def test_report_integration_and_both_notebooks(tmp_path):
    from runtime.wan.paper_suite_tables_v1 import build as report
    from scripts.build_paper_fpr_validation_v1_notebook import build as finite,ROOT
    from scripts.build_paper_suite_v1_notebook import build as full
    import ast,nbformat
    out=report(tmp_path/'c',tmp_path/'w',tmp_path/'q',tmp_path/'public',tmp_path/'report')
    folder=Path(out['report']);rows=json.loads((folder/'fpr_measurements.json').read_text())
    assert len(rows)==48 and out['fpr_increment']['added_scientific_calls']==0
    for builder in (finite,full):
        path=builder(tmp_path);assert path.read_bytes()==(ROOT/'notebooks'/path.name).read_bytes()
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb)
        assert nb.cells[0].source.strip()=="from google.colab import drive\ndrive.mount('/content/drive')"
        for cell in nb.cells:
            if cell.cell_type=='code':ast.parse(cell.source);assert not cell.outputs
