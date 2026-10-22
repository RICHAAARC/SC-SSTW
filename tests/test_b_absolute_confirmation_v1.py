"""Card99 targeted engineering fixtures only. No actual roles, sources or physical kernels."""
from collections import Counter
import copy
import json
from pathlib import Path
import pytest
from main.tube_state import b_absolute_confirmation_v1 as m
from runtime.wan import b_absolute_confirmation_v1 as run

pytestmark=pytest.mark.unit


def fixture_family():
    f=m.empty_family();f['phase_local_bits']={str(p):[0]*80 for p in range(4)}
    for x in f['candidates']:x.update(T=0,F=0,V=0,score_status='RETURNED')
    f['candidates'][0]['T']=2
    f.update(score_complete=True,status='SCORES_COMPLETE')
    return f


def test_score_only_calibration_fresh_thresholds_and_wrapper_mapping():
    f=fixture_family();f['phase_local_bits']={}  # payload intentionally absent, not a cal failure
    t=m.calibrate({h:{q:f for q in m.query_metadata()} for h in ('b97-s02','b97-s04')})
    assert t['status']=='FROZEN' and t['tau']==2 and t['L']==0 and not t['message_used']
    assert not t['inherited_development_thresholds'] and all(len(v['nulls'])==30 and len(v['positives'])==6 for v in t['hosts'].values())
    d=m.decision(f,t)
    assert len(d['C'])==137 and d['detection']=='REJECT_SYNC'  # C never filtered by tau
    broken=copy.deepcopy(f);broken['candidates'][136]['T']=None
    assert m.calibrate({h:{q:broken for q in m.query_metadata()} for h in ('b97-s02','b97-s04')})['status']=='CALIBRATION_UNRESOLVED'
    assert m.decision(broken,t)['detection']=='UNRESOLVED_SCORE'


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


def test_full_fake_path_budget_roles_persistence_and_readonly_reentry(tmp_path,monkeypatch):
    # Fixed injected pair is a synthetic fixture, never secrets/random draw.
    import secrets
    monkeypatch.setattr(secrets,'choice',lambda *a:(_ for _ in ()).throw(AssertionError('real role draw forbidden in preparation')))
    original_gate=m.gate_b0
    def before_truth_gate(*args):
        assert not (tmp_path/'post_truth_access.json').exists()
        return original_gate(*args)
    monkeypatch.setattr(m,'gate_b0',before_truth_gate)
    fake=FakeBackend(tmp_path)
    result=run.run(tmp_path,backend=fake,role_picker=lambda pairs:pairs[4])
    assert result['status']=='COMPLETED_FINITE_ROSTER_EMPIRICAL_ONLY'
    assert [(h['id'],h['role']) for h in fake.hosts]==[('b97-s02','calibration'),('b97-s04','calibration'),('b97-s01','confirmation'),('b97-s03','confirmation')]
    assert dict(fake.calls)==dict(book=8,generation_load=4,generate=8,construct=4,VAE=4,D=12,encode_mp4=12,probe=12,decode_mp4=12,E=288,Q=576,score=576,FFT=144,votes=288,B0=72,quality=24)
    assert len(list((tmp_path/'baselines').rglob('*.gated.json')))==72
    ledger=json.loads((tmp_path/'ledger.json').read_text())
    assert all(v['status']=='RETURNED' and v['attempts']==1 for v in ledger.values())
    for h in ('b97-s02','b97-s04'):
        assert not any(k.startswith(h+'/') and v['kind'] in ('receiver_FFT','votes','B0_unique_max_reference') for k,v in ledger.items())
        assert all(json.loads(p.read_text())['phase_local_bits']=={} for p in (tmp_path/'hosts'/h/'blind/families').glob('*.json'))
    marker=json.loads((tmp_path/'markers/confirmation_blind_complete.json').read_text());access=json.loads((tmp_path/'post_truth_access.json').read_text())
    assert access['opened_unix']>=marker['saved_unix']>=marker['threshold_freeze']
    evaluation=json.loads((tmp_path/'evaluation.json').read_text())
    assert evaluation['confirmation']['all_message_slots']==2304 and len(evaluation['hosts'])==2
    before=dict(fake.calls)
    assert run.run(tmp_path,config_path='/unavailable',backend=object(),role_picker=lambda p:(_ for _ in ()).throw(AssertionError('redraw')))['physical_reentry'] is False
    assert dict(fake.calls)==before


def fake_stages(monkeypatch):
    def host_work(host,config,output,backend,journal):
        assert (output/'roles.json').exists()
        for k,v in journal.data.items():
            if k.startswith(host['id']+'/') and '/B0/' not in k and '/decision/' not in k:v.update(status='RETURNED',attempts=1)
        return {q:fixture_family() for q in m.query_metadata()}
    def b0_work(host,families,thresholds,output,backend,journal):
        out={}
        for q,f in families.items():
            def save():
                v=m.b0_shell(f,reason='EXACT_TIE_UNRESOLVED');g=m.gate_b0(v,thresholds)
                run.dump(output/'baselines'/host['id']/(q+'.json'),v);run.dump(output/'baselines'/host['id']/(q+'.gated.json'),g)
                return dict(ungated=v,gated=g)
            out[q]=journal.call(host['id']+'/B0/'+q,save) or dict(ungated=m.b0_shell(f),gated=m.b0_shell(f))
        return out
    monkeypatch.setattr(run,'run_host',host_work);monkeypatch.setattr(run,'run_b0',b0_work)


@pytest.mark.parametrize('target',['decisions/b97-s03/q00.json','baselines/b97-s03/q00.gated.json'])
def test_unsaved_blind_decision_prevents_semantic_truth(tmp_path,monkeypatch,target):
    fake_stages(monkeypatch)
    original=run.dump;hits=[]
    def fail(path,value):
        if Path(path)==tmp_path/target:hits.append(1);raise OSError('fake decision save failure')
        return original(path,value)
    monkeypatch.setattr(run,'dump',fail)
    result=run.run(tmp_path,backend=object(),role_picker=lambda p:p[0])
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED' and result['post_status']=='NOT_ENTERED_UNSAVED_BLIND_OUTPUTS'
    assert hits==[1] and not (tmp_path/'post_truth_access.json').exists() and not (tmp_path/'evaluation.json').exists()
    assert not (tmp_path/'markers/confirmation_blind_complete.json').exists()


def test_calibration_failure_leaves_confirmation_sourceprep_not_entered(tmp_path,monkeypatch):
    entered=[]
    def failed_cal(host,*a):
        entered.append(host['id']);return {q:m.empty_family('FAKE_MISSING_SCORE') for q in m.query_metadata()}
    monkeypatch.setattr(run,'run_host',failed_cal)
    result=run.run(tmp_path,backend=object(),role_picker=lambda p:p[-1])
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED' and entered==['b97-s03','b97-s04']
    assert result['threshold_status']=='CALIBRATION_UNRESOLVED'
    ledger=json.loads((tmp_path/'ledger.json').read_text())
    assert all(ledger[h+'/generation_load']['status']=='NOT_ENTERED' for h in ('b97-s01','b97-s02'))


def test_interrupted_role_attempt_cannot_redraw(tmp_path):
    def interrupt(pairs):
        assert json.loads((tmp_path/'role_draw_receipt.json').read_text())['status']=='STARTED'
        raise KeyboardInterrupt('synthetic interruption')
    result=run.run(tmp_path,backend=object(),role_picker=interrupt)
    assert result['status']=='INTERRUPTED'
    assert json.loads((tmp_path/'role_draw_receipt.json').read_text())['status']=='INTERRUPTED'
    assert run.run(tmp_path,config_path='/unavailable',role_picker=lambda p:1)['physical_reentry'] is False


def test_plan_budget_and_local_notebook_bundle(tmp_path,monkeypatch):
    import ast
    import subprocess
    import sys
    import zipfile
    from scripts import build_b_absolute_confirmation_v1_notebook as builder
    cfg=run.load_config();cfg=dict(cfg,hosts=[dict(h,role='calibration' if i<2 else 'confirmation') for i,h in enumerate(cfg['hosts'])])
    counts=Counter(run.units(cfg).values())
    for kind,n in dict(generation_pipeline_load=4,source_50step_trajectory=8,PM_construct=4,VAE_load=4,ordinary_D181=12,E41=288,
                       receiver_FFT=144,Q=576,score=576,votes=288,B0_unique_max_reference=72,absolute_compatibility_read=72,paired_quality=24,score_family_binding=144).items():assert counts[kind]==n
    budget=json.loads(run.BUDGET.read_text())
    allpath=budget['all_four_hosts_complete_path_or_labeled_upper']
    assert allpath['DiT_total']==800 and allpath['native_updates']==400 and allpath['local_raw_Counter_calls']==23040
    assert allpath['reference_group_bit_Counter_calls_upper']==1152 and allpath['new_cached_cell_bit_comparisons_upper']==157824
    notebook,archive=builder.build();nb=json.loads(notebook.read_text())
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in code:ast.parse(c)
    assert 'SOURCE_TEXT' not in ''.join(code) and 'b64decode' not in ''.join(code)
    assert all(c['execution_count'] is None and c['outputs']==[] for c in nb['cells'] if c['cell_type']=='code')
    with zipfile.ZipFile(archive) as z:
        for n in z.namelist():
            assert not Path(n).is_absolute() and '..' not in Path(n).parts
            if n.endswith('.py'):compile(z.read(n),n,'exec')
        z.extractall(tmp_path/'source')
    c=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_absolute_confirmation_v1_run','--help'],cwd=tmp_path/'source',capture_output=True,text=True)
    assert c.returncode==0 and 'Card99' in c.stdout
    output=tmp_path/'existing';output.mkdir();(output/'attempt.json').write_text('{}')
    def forbidden(*a,**k):raise AssertionError('setup/worker/download on existing attempt')
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    scope={}
    exec(code[1].replace(repr(builder.OUTPUT),repr(str(output))),scope);exec(code[2],scope)
    assert scope['EXISTING_ARTIFACTS']


def test_notebook_bootstrap_roles_only_and_empty_download_stub(tmp_path,monkeypatch):
    import subprocess
    from scripts import build_b_absolute_confirmation_v1_notebook as builder
    notebook,_=builder.build();nb=json.loads(notebook.read_text());code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    existing=tmp_path/'roles_only';existing.mkdir();(existing/'roles.json').write_text('{}')
    def forbidden(*a,**k):raise AssertionError('network/source lookup for existing roles')
    monkeypatch.setattr(subprocess,'run',forbidden)
    scope={};exec(code[1].replace(repr(builder.OUTPUT),repr(str(existing))),scope)
    assert scope['EXISTING_ARTIFACTS']
    calls=[];source=tmp_path/'downloaded';fresh=tmp_path/'fresh'
    def clone_stub(args,**kwargs):
        calls.append(args)
        assert args[:2]==['git','clone']
        root=Path(args[-1]);entry=root/builder.ENTRY;entry.parent.mkdir(parents=True);entry.write_text('# fake entry presence only\n')
    monkeypatch.setattr(subprocess,'run',clone_stub)
    bootstrap=code[1].replace(repr(builder.OUTPUT),repr(str(fresh))).replace("Path('/content/SC-SSTW-b-absolute-confirmation-v1')",repr(source))
    # repr(PosixPath) is not a notebook global; replace with an explicit Path expression.
    bootstrap=bootstrap.replace(repr(source),'Path('+repr(str(source))+')')
    scope={};exec(bootstrap,scope)
    assert len(calls)==1 and scope['SOURCE_ROOT']==source and (source/builder.ENTRY).is_file()
