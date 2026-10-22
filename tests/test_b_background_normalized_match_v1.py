"""Card101 small synthetic/stub engineering only, never scientific saved-Q replay."""
from pathlib import Path
from collections import Counter
import json,copy
import numpy as np
import pytest
from main.tube_state import b_background_normalized_match_v1 as m
from runtime.wan import b_background_normalized_cpu_v1 as cpu
from runtime.wan import b_background_normalized_confirmation_v1 as user

pytestmark=pytest.mark.unit


def q(value):return dict(roi_q=np.full((10,4,44),value).tolist())
def bank(value):return {str(p):q(value) for p in range(4)}
def off_hosts(a=1,b=3):return [[{k:bank(v) for k in m.KEY_IDS} for _ in range(6)] for v in (a,b)]


def family(value=.2):
    return dict(status='SCORES_COMPLETE',candidates=[dict(phase=p,source_start=s,S=value,score_status='RETURNED') for p,s in m.H],
        phase_local_bits={str(p):[0]*80 for p in range(4)})


def test_global_mean_covariance_and_positive_scale_match():
    model=m.fit_background(off_hosts())
    record=model['models']['key00']
    assert np.allclose(record['mean'],2) and np.allclose(record['covariance_raw'],1)
    assert np.allclose(record['covariance_regularized'],(np.ones((40,40))+np.eye(40))/2)
    assert record['host_groups']==2 and record['correlated_vectors_per_host']==[822,822]
    assert record['covariance_condition']<=41+1e-10
    pos=m.score_family(bank(3),record);scaled=m.score_family(bank(5),record);neg=m.score_family(bank(1),record)
    assert pos['status']==scaled['status']==neg['status']=='SCORES_COMPLETE'
    assert all(abs(r['S']-1)<1e-12 for r in pos['candidates']+scaled['candidates'])
    assert all(r['S']==0 for r in neg['candidates'])
    zero=m.score_family(bank(2),record)
    assert zero['status']=='SCORE_UNRESOLVED' and len(zero['candidates'])==137
    assert all(r['score_status']=='ZERO_NORM_UNRESOLVED' and r['S'] is None for r in zero['candidates'])


def test_any_background_missing_and_bad_covariance_are_not_fallbacks():
    with pytest.raises(ValueError):m.fit_background(off_hosts(0,0))
    bad=off_hosts();bad[1][5]['key01']['3']=None
    with pytest.raises((ValueError,TypeError)):m.fit_background(bad)
    f=family();f['candidates'][0]['S']=None
    d=m.decide(f,dict(status='FROZEN',tau_N=.1))
    assert d['detection']=='UNRESOLVED_SCORE' and len(d['members'])==137 and all(r['in_C'] is None for r in d['members'])


def test_null_only_threshold_no_positive_floor_and_boundary_set():
    fs={h:{qid:family() for qid in m.query_metadata()} for h in ('cal_a','cal_b')}
    for h in fs:
        for qid,meta in m.query_metadata().items():
            if meta['arm']=='PM' and meta['key']=='key00':fs[h][qid]=m.empty_family('unused_positive_score_missing')
    t=m.calibrate_null(fs)
    assert t['status']=='FROZEN' and t['tau_N']==.2 and not t['positive_floor_used']
    f=family(.1);f['candidates'][0]['S']=.7;f['candidates'][1]['S']=.2
    d=m.decide(f,t)
    assert len(d['C'])==2 and d['detection']=='DETECTED_PROVISIONAL' and d['singleton_candidate'] is None
    fs['cal_b']['q00']['candidates'][136]['S']=None
    assert m.calibrate_null(fs)['status']=='CALIBRATION_UNRESOLVED'


def test_feature_order_all137_equal40_support():
    banks={str(p):dict(roi_q=np.arange(1760).reshape(10,4,44).tolist()) for p in range(4)}
    values=m.vectors(banks)
    assert values.shape==(137,40)
    for i,(p,s) in enumerate(m.H):
        expected=[banks[str(p)]['roi_q'][r][k][s+r-1] for r in range(10) for k in range(4)]
        assert values[i].tolist()==expected


def test_missing_saved_Q_consumes_attempt_and_leaves72decisions_not_entered(tmp_path):
    out=tmp_path/'run'
    result=cpu.run_saved(tmp_path/'absentQ',out,tmp_path/'absentOld')
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert result['background_status']=='BACKGROUND_UNAVAILABLE'
    assert result['decision_not_entered']==72 and result['message_slots_not_entered']==2304
    ledger=json.loads((out/'ledger.json').read_text())
    assert len(ledger)==867
    assert sum(v['kind']=='saved_Q_load' and v['status']=='FAILED' and v['attempts']==1 for v in ledger.values())==576
    assert sum(v['kind']=='development_decision' and v['status']=='NOT_ENTERED' for v in ledger.values())==72
    assert not (out/'post_truth_access.json').exists()
    assert cpu.run_saved('/missing',out,'/missing')['physical_reentry'] is False


def stub_collection(host,config,out,backend,journal):
    h=host['id']
    assert (out/'roles.json').exists()
    if host['role']=='confirmation':
        assert json.loads((out/'background.json').read_text())['status']=='BACKGROUND_FROZEN'
        assert json.loads((out/'thresholds.json').read_text())['status']=='FROZEN'
    for slot,row in journal.data.items():
        if slot.startswith(h+'/') and row['kind'] not in ('normalized_family_score','score_family_binding','B0_unique_max_reference','absolute_compatibility_read'):
            row.update(status='RETURNED',attempts=1)
    return dict(Q={},bits={})


def stub_scores(host,collection,background,out,journal):
    fs={}
    for qid in m.query_metadata():
        def score():return family()
        v=journal.call(host['id']+'/score_family/'+qid,score)
        def save():
            user.dump(out/'hosts'/host['id']/'blind/families'/(qid+'.json'),v);return v
        fs[qid]=journal.call(host['id']+'/family/'+qid,save)
    return fs


def stub_B0(host,families,thresholds,out,backend,journal):
    fs={}
    for qid in m.query_metadata():
        def save():
            b=user.layout.b0_shell(m.compatibility_input(families[qid]),reason='EXACT_TIE_UNRESOLVED')
            g=user.layout.gate_b0(b,dict(status='FROZEN',tau=thresholds['tau_N']))
            user.dump(out/'baselines'/host['id']/(qid+'.json'),b);user.dump(out/'baselines'/host['id']/(qid+'.gated.json'),g)
            return dict(ungated=b,gated=g)
        fs[qid]=journal.call(host['id']+'/B0/'+qid,save)
    return fs


def test_future_stages_freeze_before_confirm_and_all_blind_before_truth(tmp_path,monkeypatch):
    from runtime.wan import b_background_normalized_evaluation_v1 as post
    entered=[]
    def collect(*args):entered.append(args[0]['role']);return stub_collection(*args)
    monkeypatch.setattr(user,'collect_host',collect);monkeypatch.setattr(user,'score_host',stub_scores)
    monkeypatch.setattr(user,'b0_reference',stub_B0)
    monkeypatch.setattr(user,'fit_calibration_background',lambda *a:dict(status='BACKGROUND_FROZEN',models={}))
    def fake_post(decisions,baselines,families,thresholds,bits,roles):
        assert (tmp_path/'markers/confirmation_blind_complete.json').exists()
        assert len(list((tmp_path/'decisions').rglob('*.json')))==72
        assert len(list((tmp_path/'baselines').rglob('*.gated.json')))==72
        return dict(status='FAKE_POST')
    monkeypatch.setattr(post,'evaluate_user',fake_post)
    result=user.run(tmp_path,backend=object(),role_picker=lambda pairs:pairs[0])
    assert result['status']=='COMPLETED_BACKGROUND_MATCH_EMPIRICAL_ONLY' and entered==['calibration','calibration','confirmation','confirmation']
    assert user.run(tmp_path,config_path='/absent',backend=object())['physical_reentry'] is False


def test_bad_null_stops_all_confirmation_sourceprep(tmp_path,monkeypatch):
    monkeypatch.setattr(user,'collect_host',stub_collection)
    def bad_scores(*args):
        fs=stub_scores(*args);fs['q00']['candidates'][0]['S']=None;return fs
    monkeypatch.setattr(user,'score_host',bad_scores)
    monkeypatch.setattr(user,'fit_calibration_background',lambda *a:dict(status='BACKGROUND_FROZEN',models={}))
    result=user.run(tmp_path,backend=object(),role_picker=lambda pairs:pairs[-1])
    assert result['threshold_status']=='CALIBRATION_UNRESOLVED' and result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert result['decision_not_entered']==72 and result['message_slots_not_entered']==2304
    ledger=json.loads((tmp_path/'ledger.json').read_text())
    assert sum(v['kind']=='absolute_compatibility_read' and v['status']=='NOT_ENTERED' for v in ledger.values())==72
    assert sum(v['kind']=='generation_pipeline_load' and v['status']=='RETURNED' for v in ledger.values())==2
    assert not (tmp_path/'post_truth_access.json').exists()


class FakeBackend:
    """Only scalar/symbolic arrays and pretend calls; no torch/model/FFT/codec."""
    blocks={}
    def __init__(self,output):self.saved={};self.calls=Counter();self.output=output;self.hosts=[]
    def hit(self,k):
        assert (self.output/'attempt.json').exists() and (self.output/'roles.json').exists()
        self.calls[k]+=1
    def book(self,s):self.hit('book');return {}
    def generation_load(self,host,config):
        self.hit('generation_load');self.hosts.append(host)
        receipt=json.loads((self.output/'role_draw_receipt.json').read_text())
        assert receipt['status']=='RETURNED'
        if host['role']=='confirmation':
            assert json.loads((self.output/'thresholds.json').read_text())['status']=='FROZEN'
        return dict(fake=True)
    def generate(self,*a):self.hit('generate');return 1,dict(fake=True)
    def release_generation(self):pass
    def construct(self,*a):self.hit('construct');return 1,dict(fake=True)
    def load_vae(self):self.hit('VAE');return dict(fake=True)
    def release_vae(self):pass
    def save_tensor(self,path,v):self.saved[str(path)]=v
    def load_tensor(self,path):return self.saved[str(path)]
    def decode(self,z):self.hit('D');return dict(preclip=1,float_rgb=1)
    def preclip(self,z):return dict(fake=True)
    def quantize(self,z):return 1
    def save_mp4(self,*a):self.hit('encode_mp4');return dict(fake=True)
    def probe(self,*a):self.hit('probe');return dict(fake=True)
    def read_mp4(self,*a):self.hit('decode_mp4');return 1
    def load_pixels(self,p):return self.saved[str(p)]
    def prepare_view(self,*a):return 1
    def encode_view(self,v):self.hit('E');return 1
    def observe(self,*a):self.hit('Q');return dict(fake=True)
    def spectrum(self,z):self.hit('FFT');return 1
    def votes(self,*a):
        self.hit('votes')
        return dict(local_cells=[dict(bits=[dict(decoded=0) for _ in range(8)]) for _ in range(10)])
    def score(self,q,t,c):
        self.hit('score')
        for x in t['scores']:x.update(status='RETURNED',centered_min=2 if x['source_start']==1 else 0,F=0,V=0)
    def reduce_candidate(self,v,s,c):
        self.hit('B0')
        c['candidate_reducers_returned']+=1
        groups={(s+r-2)//11 for r in range(1,11)}
        return dict(status='CONDITIONAL_REDUCED_UNCALIBRATED',decoded_bits=[0 if i//8 in groups else None for i in range(32)])
    def quality(self,*a):self.hit('quality');return dict(fake=True,frame_denominator=181,adjacent_denominator=180)


@pytest.mark.parametrize('role',['calibration','confirmation'])
def test_scoreless_collector_keeps_exact_E_Q_and_confirmation_only_FFT(tmp_path,role):
    (tmp_path/'attempt.json').write_text('{}');(tmp_path/'roles.json').write_text('{}')
    (tmp_path/'role_draw_receipt.json').write_text(json.dumps(dict(slot='role_draw',kind='uniform_role_draw',status='RETURNED',attempts=1)))
    (tmp_path/'thresholds.json').write_text(json.dumps(dict(status='FROZEN',tau_N=.2)))
    cfg=user.load_config();cfg=dict(cfg,hosts=[dict(h,role='calibration' if i<2 else 'confirmation') for i,h in enumerate(cfg['hosts'])])
    backend=FakeBackend(tmp_path);journal=user.Journal(tmp_path,cfg,backend)
    host=next(h for h in cfg['hosts'] if h['role']==role)
    result=user.collect_host(host,cfg,tmp_path,backend,journal)
    assert len(result['Q'])==144 and backend.calls['E']==72 and backend.calls['Q']==144
    assert backend.calls['score']==0 and backend.calls['B0']==0
    assert backend.calls['FFT']==(72 if role=='confirmation' else 0)
    assert backend.calls['votes']==(144 if role=='confirmation' else 0)
    assert len(result['bits'])==(36 if role=='confirmation' else 0)


def test_actual_notebook_and_no_git_bundle_readonly_bootstrap(tmp_path,monkeypatch):
    import ast,subprocess,sys,zipfile
    from scripts import build_b_background_normalized_match_v1_notebook as builder
    nbfile,archive=builder.build();nb=json.loads(nbfile.read_text());code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in code:ast.parse(c)
    assert 'SOURCE_TEXT' not in ''.join(code) and 'b64decode' not in ''.join(code)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            if name.endswith('.py'):compile(z.read(name),name,'exec')
        z.extractall(tmp_path/'source')
    completed=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_background_normalized_match_v1_run','--help'],cwd=tmp_path/'source',capture_output=True,text=True)
    assert completed.returncode==0 and 'Card101' in completed.stdout
    output=tmp_path/'existing';output.mkdir();(output/'background.json').write_text('{}')
    def blocked(*a,**k):raise AssertionError('source/setup/worker reentry')
    monkeypatch.setattr(subprocess,'run',blocked);monkeypatch.setattr(subprocess,'Popen',blocked)
    scope={};exec(code[1].replace(repr(builder.OUTPUT),repr(str(output))),scope);exec(code[2],scope)
    assert scope['EXISTING_ARTIFACTS']
