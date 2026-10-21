"""Tiny/fake engineering only: no model, VAE, FFT, codec, old observations or scientific replay."""
from collections import Counter
import copy
import json
from pathlib import Path
import pytest
from main.tube_state import b_fixed_detection_confidence_v1 as m
from runtime.wan import b_fixed_detection_confidence_v1 as run
from runtime.wan import b_fixed_detection_confidence_evaluation_v1 as post

pytestmark = pytest.mark.unit


def fixture_family():
    f = m.empty_family()
    f['status'] = 'COMPLETE'
    f['phase_local_bits'] = {str(p): [0]*80 for p in range(4)}
    for row in f['candidates']:
        row.update(T=-2.0, F=-2.0, V=-2.0, score_status='RETURNED', payload_status='CONDITIONAL_REDUCED_UNCALIBRATED')
        groups = {x['group'] for x in row['cells']}
        row['decoded_bits'] = [0 if i//8 in groups else None for i in range(32)]
    f['candidates'][0]['T'] = 10.0
    return f


def threshold(tau=8, gap=0):
    return dict(status='FROZEN', tau=tau, **{'lambda':gap})


def test_gap_boundary_not_filtered_by_detection_threshold_and_consensus():
    f = fixture_family()
    f['candidates'][1]['T'] = 5.0
    d = m.decide(f, threshold(gap=5))
    assert len(d['candidate_set'])==2 and d['location_hypothesis'] is None
    assert [x['decoded'] for x in d['message'][:8]]==[0]*8
    assert all(x['decoded'] is None for x in d['message'][8:])
    f['candidates'][1]['decoded_bits'][0]=1
    assert m.decide(f,threshold(gap=5))['message'][0]['decoded'] is None
    assert m.decide(f,threshold(gap=4.999))['location']=='ACCEPT_LOCATION_PROVISIONAL'
    assert m.decide(f,threshold(tau=10))['detection']=='REJECT_SYNC'
    # Two maxima supporting disjoint groups cannot manufacture a shared message.
    f=fixture_family(); f['candidates'][11]['T']=10
    assert all(x['decoded'] is None for x in m.decide(f,threshold())['message'])


def test_nonfinite_missing_and_payload_failure_are_distinct():
    f=fixture_family(); f['candidates'].pop()
    assert m.decide(f,threshold())['status']=='UNRESOLVED_TECHNICAL'
    f=fixture_family(); f['candidates'][30]['T']=float('nan')
    assert m.decide(f,threshold())['maximum'] is None
    f=fixture_family(); f['candidates'][30]['payload_status']='FAILED'
    d=m.decide(f,threshold())
    assert d['status']=='PAYLOAD_UNRESOLVED' and d['maximum']==10
    assert d['score_only_detection']=='DETECTED_PROVISIONAL' and d['detection']=='UNRESOLVED'
    assert len(d['exact_max_ties'])==1 and all(x['decoded'] is None for x in d['message'])
    uncal=m.decide(f,dict(status='CALIBRATION_UNRESOLVED'))
    assert uncal['status']=='PAYLOAD_UNRESOLVED' and uncal['maximum']==10


def test_calibration_full_host_families_no_message_parameter():
    f=fixture_family(); hosts={h:{q:f for q in m.query_metadata()} for h in ('b93-c01','b93-c02')}
    t=m.calibrate(hosts,m.query_metadata())
    assert t['status']=='FROZEN' and t['tau']==10 and t['lambda']==12 and not t['message_used']
    assert all(len(v['nulls'])==30 and len(v['positives'])==6 for v in t['hosts'].values())
    bad=copy.deepcopy(f); bad['candidates'][0]['payload_status']='FAILED'
    hosts['b93-c02']['q00']=bad
    assert m.calibrate(hosts,m.query_metadata())['tau'] is None


def test_physical_nominal_guard_and_target_semantics():
    totals=[0,0]
    for a in m.ORIGINS:
        geo=post.geometry(a,(-a)%4)
        totals[0]+=sum(g['physical_present'] for g in geo['groups'])*8
        totals[1]+=sum(g['nominal_supported'] for g in geo['groups'])*8
    assert totals==[80,72]
    a4=post.geometry(4,0)['groups'][1]
    assert a4['physical_present'] and not a4['nominal_supported']
    assert any(r['truth_kind']=='GUARD_OR_OUT_OF_PROTOCOL' for r in post.geometry(136,3)['observed'])
    f=fixture_family(); d=m.decide(f,threshold())
    off=post.assess(d,f,dict(origin=0,arm='OFF',key='key00'),[0]*32)
    wrong=post.assess(d,f,dict(origin=0,arm='PM',key='key01'),[0]*32)
    p0=post.assess(d,f,dict(origin=0,arm='P0',key='key00'),[0]*32)
    assert off['OFF_no_target_outputs']==8 and off['present_BER'] is None
    assert wrong['wrong_key_attributions']==8 and wrong['present_BER'] is None
    assert p0['null_sync_detection'] and p0['present_BER']==0
    for r in off['message']+wrong['message']:
        assert r['correct'] is None
    # Tail wrong-phase raw evidence must never index a fifth message group.
    post.assess(d,f,dict(origin=136,arm='PM',key='key00'),[0]*32)


def test_true_phase_physical_pair_is_independent_of_chosen_peak():
    f=fixture_family(); g=copy.deepcopy(f); g['phase_local_bits']['3'][0]=1
    a=post.assess(m.decide(f,threshold()),f,dict(origin=17,arm='P0',key='key00'),[0]*32)
    b=post.assess(m.decide(g,threshold()),g,dict(origin=17,arm='PM',key='key00'),[0]*32)
    pair=post.paired_payload(a,b)
    assert pair['true_phase']==3 and pair['common_raw_read']==80
    assert pair['P0_common_errors']==0 and pair['PM_common_errors']==1 and pair['raw_correct_lost']==1
    assert pair['PM_common_BER']==1/80


def test_staging_freezes_before_confirmation_and_blocks_on_calibration_failure(tmp_path,monkeypatch):
    f=fixture_family(); events=[]
    class Fake: blocks={}
    def host_work(host,config,output,backend,journal):
        h=host['id']; events.append(h)
        assert (output/'attempt.json').is_file()
        if host['role']=='confirmation':
            t=json.loads((output/'thresholds.json').read_text())
            assert t['status']=='FROZEN' and (output/'markers/calibration_blind_complete.json').exists()
        for slot,row in journal.data.items():
            if slot.startswith(h+'/') and '/decision/' not in slot:
                row['status']='RETURNED'
        return {qid:f for qid in m.query_metadata()}
    monkeypatch.setattr(run,'run_host',host_work)
    monkeypatch.setattr(post,'evaluate',lambda *args:dict(status='FAKE_POST'))
    result=run.run(tmp_path/'ok',backend=Fake())
    assert events==['b93-c01','b93-c02','b93-v01','b93-v02']
    assert result['status']=='COMPLETED_EMPIRICAL_ONLY'
    marker=json.loads((tmp_path/'ok/markers/confirmation_blind_complete.json').read_text())
    access=json.loads((tmp_path/'ok/post_truth_access.json').read_text())
    assert access['opened_unix']>=marker['saved_unix']>=marker['threshold_freeze']
    events.clear(); f['candidates'][5]['payload_status']='FAILED'
    result=run.run(tmp_path/'bad',backend=Fake())
    assert events==['b93-c01','b93-c02'] and result['thresholds_status']=='CALIBRATION_UNRESOLVED'
    ledger=json.loads((tmp_path/'bad/ledger.json').read_text())
    assert ledger['b93-v01/generation_load']['status']=='NOT_ENTERED'
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'


def test_attempt_precedes_call_interruption_consumes_and_reentry_needs_no_config(tmp_path):
    (tmp_path/'attempt.json').write_text('{}')
    class Fake: blocks={}
    journal=run.Journal(tmp_path,run.load_config(),Fake())
    def interrupt():
        receipt=tmp_path/'receipts/b93-c01/generation_load.json'
        assert json.loads(receipt.read_text())['status']=='STARTED'
        raise KeyboardInterrupt('fake interruption')
    with pytest.raises(KeyboardInterrupt):
        journal.call('b93-c01/generation_load',interrupt)
    assert json.loads((tmp_path/'receipts/b93-c01/generation_load.json').read_text())['status']=='INTERRUPTED'
    assert run.run(tmp_path,config_path='/does/not/exist',backend=object())['physical_reentry'] is False


class FakeBackend:
    """Symbolic arrays, fake kernels and fake media; never imports torch."""
    blocks={}
    def __init__(self): self.saved={}; self.calls=Counter()
    def hit(self,k): self.calls[k]+=1
    def book(self,s): self.hit('book'); return {}
    def generation_load(self,*a): self.hit('generation_load'); return {}
    def generate(self,arm,*a): self.hit('generate'); return 1, {'fake':True}
    def release_generation(self): pass
    def construct(self,*a): self.hit('construct'); return 1,{'fake':True}
    def load_vae(self): self.hit('VAE'); return {}
    def release_vae(self): pass
    def save_tensor(self,path,v): self.saved[str(path)]=v
    def load_tensor(self,path): return self.saved[str(path)]
    def decode(self,z): self.hit('D'); return dict(preclip=1,float_rgb=1)
    def preclip(self,z): return {'fake':True}
    def quantize(self,z): return 1
    def save_mp4(self,*a): self.hit('encode_mp4'); return {}
    def probe(self,*a): self.hit('probe'); return {}
    def read_mp4(self,*a): self.hit('decode_mp4'); return 1
    def load_pixels(self,p): self.hit('load_RGB'); return self.saved[str(p)]
    def prepare_view(self,*a): return 1
    def encode_view(self,v): self.hit('E'); return 1
    def observe(self,*a): self.hit('Q'); return {'fake':True}
    def spectrum(self,z): self.hit('FFT'); return 1
    def votes(self,*a): self.hit('votes'); return dict(local_cells=[dict(bits=[dict(decoded=0) for _ in range(8)]) for _ in range(10)])
    def score(self,q,t,c):
        self.hit('score')
        for r in t['scores']: r.update(status='RETURNED',centered_min=0,F=0,V=0)
    def reduce(self,v,t,c):
        self.hit('reduce')
        for r in t['candidates']:
            r['status']='CONDITIONAL_REDUCED_UNCALIBRATED'
            gs={x['group'] for x in r['cells']}
            r['decoded_bits']=[0 if i//8 in gs else None for i in range(32)]
    def quality(self,*a): self.hit('quality'); return dict(fake=True,frame_denominator=181,adjacent_denominator=180)


def test_actual_host_orchestration_with_symbolic_backend_and_fake_io(tmp_path,monkeypatch):
    saved={}
    monkeypatch.setattr(run,'dump',lambda p,v:saved.__setitem__(str(p),v))
    # Storage-only fixture trim; all137 hypothesis cells/bits remain, no real score/reducer runs.
    def small_blank(starts):
        return dict(scores=[dict(source_start=s,status='NOT_ENTERED') for s in starts],
            candidates=[dict(source_start=s,status='NOT_ENTERED',cells=m.cells(s),decoded_bits=[None]*32) for s in starts])
    monkeypatch.setattr(run.arithmetic,'blank_table',small_blank)
    cfg=run.load_config(); backend=FakeBackend(); journal=run.Journal(tmp_path,cfg,backend)
    families=run.run_host(cfg['hosts'][0],cfg,tmp_path,backend,journal)
    assert len(families)==36 and all(len(f['candidates'])==137 and f['status']=='COMPLETE' for f in families.values())
    expected={'generation_load':1,'generate':2,'construct':1,'VAE':1,'D':3,'encode_mp4':3,'probe':3,'decode_mp4':3,'load_RGB':3,'E':72,'FFT':72,'Q':144,'votes':144,'score':144,'reduce':144,'quality':6,'book':2}
    assert dict(backend.calls)==expected
    assert saved[str(tmp_path/'hosts/b93-c01/blind_complete.json')]['status']=='COMPLETE'
    assert any('/blind/OFF/r0/p0/key00/Q.json' in p for p in saved)


def test_post_and_quality_failure_cannot_be_execution_complete(tmp_path,monkeypatch):
    f=fixture_family()
    class Fake: blocks={}
    def host_work(host,config,output,backend,journal):
        for slot,row in journal.data.items():
            if slot.startswith(host['id']+'/') and '/decision/' not in slot: row['status']='RETURNED'
        journal.data[host['id']+'/quality/OFF_to_P0/preclip']['status']='FAILED'
        return {qid:f for qid in m.query_metadata()}
    monkeypatch.setattr(run,'run_host',host_work)
    def fail(*a): raise OSError('fake post save/input failure')
    monkeypatch.setattr(post,'evaluate',fail)
    result=run.run(tmp_path,backend=Fake())
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert result['counts']['post_confirmation_evaluation']['FAILED']==1


def test_real_arithmetic_binding_on_tiny_zero_Q_and_failed_returned_payload():
    tables={}
    counts=Counter()
    q=dict(roi_q=[[[0.0]*44 for _ in range(4)] for _ in range(10)],pooled_q=[[0.0]*44 for _ in range(10)])
    fake=FakeBackend()
    for phase in range(4):
        t=run.arithmetic.blank_table(range(1 if phase==0 else 2,36))
        run.arithmetic.score(q,t,counts)  # 1760 fixed zeros, no saved/scientific observation.
        fake.reduce(None,t,counts)       # no actual vote/Counter or FFT execution.
        tables[phase]=t
    family=m.bind(tables)
    assert len(family['candidates'])==137 and m.valid_family(family)
    decision=m.decide(family,threshold(tau=0))
    assert decision['maximum']==0 and len(decision['exact_max_ties'])==137
    assert decision['detection']=='REJECT_SYNC' and counts['min_statistics']==137
    # A function that fills its rows then fails is still a failed physical/numeric unit.
    tables[0]['payload_call_status']='FAILED'
    broken=m.bind(tables)
    d=m.decide(broken,threshold(tau=0))
    assert d['status']=='PAYLOAD_UNRESOLVED' and d['maximum']==0 and d['detection']=='UNRESOLVED'


def test_notebook_bundle_readonly_and_no_git_cli(tmp_path,monkeypatch):
    import ast
    import subprocess
    import sys
    import zipfile
    from scripts import build_b_fixed_detection_confidence_v1_notebook as builder
    notebook,archive=builder.build()
    nb=json.loads(notebook.read_text())
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert all(c.get('execution_count') is None and c.get('outputs')==[] for c in nb['cells'] if c['cell_type']=='code')
    for text in code: ast.parse(text)
    assert 'SOURCE_TEXT' not in ''.join(code) and 'b64decode' not in ''.join(code)
    with zipfile.ZipFile(archive) as bundle:
        names=bundle.namelist()
        assert all(not n.startswith('/') and '..' not in Path(n).parts for n in names)
        assert not any('/data/' in n or n.endswith('.pt') for n in names)
        for name in names:
            if name.endswith('.py'): compile(bundle.read(name),name,'exec')
        bundle.extractall(tmp_path/'source')
    assert not (tmp_path/'source/.git').exists()
    complete=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_fixed_detection_confidence_v1_run','--help'],cwd=tmp_path/'source',capture_output=True,text=True)
    assert complete.returncode==0 and 'Card94' in complete.stdout
    # Existing-attempt notebook path never clones, installs or creates a worker.
    output=tmp_path/'existing';output.mkdir();(output/'attempt.json').write_text('{}')
    def blocked(*a,**k): raise AssertionError('physical/setup reentry')
    monkeypatch.setattr(subprocess,'run',blocked)
    monkeypatch.setattr(subprocess,'Popen',blocked)
    scope={}
    exec(code[1].replace(repr(builder.OUTPUT),repr(str(output))),scope)
    exec(code[2],scope)
    assert scope['EXISTING_ARTIFACTS']


def test_unit_plan_and_adopted_budget_agree():
    counts=Counter(run.units(run.load_config()).values())
    expected={'generation_pipeline_load':4,'source_50step_trajectory':8,'PM_construct':4,'VAE_load':4,
              'ordinary_D181':12,'E41':288,'receiver_FFT':288,'MP4_encode':12,'MP4_probe':12,'MP4_decode':12,
              'Q':576,'score':576,'votes':576,'reduce':576,'paired_quality':24,'family_binding':144,'threshold_and_set_read':144}
    for key,value in expected.items(): assert counts[key]==value
    budget=json.loads(run.BUDGET.read_text())['all_four_hosts_complete_path']
    assert budget['DiT_branch_forwards']==800 and budget['local_FFT_gradient_calls']==100
    assert budget['candidate_score_rows']==19728 and budget['keyed_Q_tables']==576
    assert budget['geometry_group_supported_message_slots_all_queries']==1920
    assert budget['geometry_group_absent_message_slots_all_queries']==2688
    assert budget['quality_per_frame_rows']==4344 and budget['quality_adjacent_residual_rows']==4320


def test_interrupt_after_scoring_retains_saved_score_rows(tmp_path,monkeypatch):
    saved={}
    monkeypatch.setattr(run,'dump',lambda p,v:saved.__setitem__(str(p),v))
    class Interrupted(FakeBackend):
        def spectrum(self,z):
            path=str(tmp_path/'hosts/b93-c01/blind/OFF/r0/p0/key00/table.json')
            assert path in saved and all(x['status']=='RETURNED' for x in saved[path]['scores'])
            raise KeyboardInterrupt('fake FFT boundary interrupt, no FFT executed')
    cfg=run.load_config(); backend=Interrupted(); journal=run.Journal(tmp_path,cfg,backend)
    with pytest.raises(KeyboardInterrupt):
        run.run_host(cfg['hosts'][0],cfg,tmp_path,backend,journal)
    table=saved[str(tmp_path/'hosts/b93-c01/blind/OFF/r0/p0/key00/table.json')]
    assert table['score_call_status']=='RETURNED' and table['payload_call_status']=='NOT_ENTERED'
    assert all(x['status']=='RETURNED' for x in table['scores'])
    assert journal.data['b93-c01/OFF/r0/p0/FFT']['status']=='INTERRUPTED'
    assert saved[str(tmp_path/'hosts/b93-c01/blind_complete.json')]['status']=='INCOMPLETE'


def test_decision_persistence_failure_blocks_marker_truth_and_evaluator(tmp_path,monkeypatch):
    family=fixture_family()
    class Fake: blocks={}
    def host_work(host,config,output,backend,journal):
        for slot,row in journal.data.items():
            if slot.startswith(host['id']+'/') and '/decision/' not in slot:
                row['status']='RETURNED'
        return {qid:family for qid in m.query_metadata()}
    monkeypatch.setattr(run,'run_host',host_work)
    real_dump=run.dump
    failed=tmp_path/'decisions/b93-v01/q00.json'
    attempts=[]
    def failing_dump(path,value):
        if Path(path)==failed:
            attempts.append(str(path))
            raise OSError('fake actual decision persistence failure')
        return real_dump(path,value)
    monkeypatch.setattr(run,'dump',failing_dump)
    def forbidden(*args):
        raise AssertionError('post evaluator entered after unsaved blind decision')
    monkeypatch.setattr(post,'evaluate',forbidden)
    result=run.run(tmp_path,backend=Fake())
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert result['post_evaluation_status']=='NOT_ENTERED_DECISION_SAVE_FAILURE'
    assert result['unsaved_decision_slots']==['b93-v01/decision/q00']
    assert len(attempts)==1 and not failed.exists()
    assert not (tmp_path/'markers/confirmation_blind_complete.json').exists()
    assert not (tmp_path/'post_truth_access.json').exists()
    assert not (tmp_path/'evaluation.json').exists()
    ledger=json.loads((tmp_path/'ledger.json').read_text())
    slots=[v for k,v in ledger.items() if '/decision/' in k]
    assert len(slots)==144 and sum(v['status']=='RETURNED' for v in slots)==143
    assert ledger['b93-v01/decision/q00']['status']=='FAILED'
    assert ledger['evaluation']['status']=='NOT_ENTERED'
    assert ledger['evaluation']['unavailable_from']==['b93-v01/decision/q00']
    assert run.run(tmp_path,backend=object())['physical_reentry'] is False
    assert len(attempts)==1
