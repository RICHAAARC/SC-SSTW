"""Fixed-scope CPU fake/static checks. Never invoke model, VAE or media tools."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import ast
import json
import subprocess
import sys
import pytest
from runtime.wan import b_rank3_four_source_confirmation_v1 as run
from main.tube_state import b_rank3_four_source_confirmation_v1 as receiver
from runtime.wan import b_rank3_four_source_confirmation_evaluation_v1 as post
pytestmark=pytest.mark.unit

class FakeBackend:
    """Tags replace every physical tensor; no model, VAE, codec or old data."""
    def __init__(self, fail=None):
        self.calls = Counter()
        self.truth = run.load_config()['writer']['bits']
        self.files = {}
        self.fail = fail
        self.blocks = {'encoder': 0, 'decoder': 0}

    def generation_load(self, host, config):
        self.calls['generation_load'] += 1
        self.host = deepcopy(host)
        return {'fake': True}

    def generate(self, arm, writer, count, record):
        self.calls['generate'] += 1
        for index in range(50):
            for kind in ('transformer_conditional', 'transformer_unconditional', 'native_step'):
                count(kind, False); count(kind, True)
            if arm == 'P0' and index >= 25:
                count('payload_gradient', False); count('payload_gradient', True)
            record(dict(index=index, fake=True))
        return dict(host=deepcopy(self.host), arm=arm, native=True), {'fake': True}

    def release_generation(self):
        pass

    def book(self, key):
        self.calls['book'] += 1
        return key

    def native_observe(self, z, book):
        self.calls['native_Q'] += 1
        return {'status': 'FAKE', 'key': book}

    def construct(self, z, book, pre):
        self.calls['construct'] += 1
        return dict(z, arm='PM'), {'fake': True}

    def native_audit(self, *args):
        return {'fake': True}

    def load_vae(self):
        self.calls['vae_load'] += 1
        return {'fake': True}

    def release_vae(self):
        pass

    def save_tensor(self, path, value):
        run.dump(path, value)

    def load_tensor(self, path):
        return run.read(path)

    def decode(self, z):
        self.calls['D181'] += 1
        self.blocks['decoder'] += 46
        value = dict(z, native=False, frames=list(range(181)))
        return dict(preclip=value, float_rgb=value)

    def preclip(self, x):
        return {'fake': True}

    def quantize(self, x):
        return x

    def save_mp4(self, x, path):
        self.calls['codec_encode'] += 1
        self.files[str(path)] = x
        return {'fake': True}

    def probe(self, path):
        self.calls['codec_probe'] += 1
        return {'fake': True}

    def read_media(self, path, probe, length):
        self.calls['codec_decode'] += 1
        return self.files[str(path)]

    def edit(self, x, frames):
        return dict(x, frames=frames)

    def prepare_window(self, x, w, phase):
        return dict(x, window=w, phase=phase, origin=x['frames'][w])

    def encode_view(self, x):
        self.calls['E41'] += 1
        if self.fail == 'first_cal_E' and self.calls['E41'] == 1:
            raise RuntimeError('synthetic E failure')
        self.blocks['encoder'] += 11
        return x

    def observe(self, x, book):
        self.calls['Q'] += 1
        return dict(x, key=book)

    def score(self, bank, model):
        self.calls['score'] += 1
        if any(x is None for x in bank.values()):
            return run.empty_family('TECHNICAL_UNRESOLVED')
        x = bank['0']
        p = (-x['origin']) % 4
        s = (x['origin']+p)//4+1
        positive = x['arm'] == 'PM' and x['key'] == 'watermark'
        return dict(status='SCORES_COMPLETE', candidates=[dict(phase=a, source_start=b,
            score_status='RETURNED', S=0.9 if positive and (a,b)==(p,s) else 0.05 if positive else 0.1) for a,b in run.H])

    def payload_project(self, x, rows, regions):
        self.calls['native_DCT' if len(rows)==44 else 'E41_DCT'] += 1
        return dict(x=x, rows=rows, regions=list(regions))

    def payload_decode(self, x, secret, view_id, key):
        self.calls['native_bits' if len(x['rows'])==44 else 'E41_bits'] += 1
        regions = {}
        for r in ('R0', 'R1'):
            cells = []
            for row in x['rows']:
                eid = f'{view_id}/row{row}/{r}'
                cells.append(dict(local_row=row, evidence_id=eid, bits=[dict(local_bit=b, value=self.truth[8*min(3,max(0,(row-1)//11))+b],
                                  status='AVAILABLE', count=8, evidence_id=eid) for b in range(8)]))
            regions[r] = dict(local_cells=cells)
        return dict(key=key, regions=regions, control_diagnostics={})

    def quality(self, a, b, frames, pair):
        self.calls['quality'] += 1
        return dict(status='FAKE', frame_denominator=len(frames), adjacent_denominator=len(frames)-1, compared_arms=[a['arm'],b['arm']],pair=pair)

def fixture_input(path):
    model=dict(mean=[0.0]*40,factor=[[float(i==j) for j in range(40)] for i in range(40)])
    run.dump(path/'background.json',dict(status='BACKGROUND_FROZEN',models={k:model for k in run.KEYS}))
    run.dump(path/'thresholds.json',dict(status='FROZEN',tau=run.TAU,tau_rule='historical90'))


def family(origin,score=.2):
    from main.tube_state.b_m3_time_path_v1 import origin_to_state
    p,s=origin_to_state(origin)
    return dict(status='SCORES_COMPLETE',candidates=[dict(phase=a,source_start=b,score_status='RETURNED',S=score if (a,b)==(p,s) else 0.) for a,b in run.H])


def test_roster_budget_and_receiver_boundaries():
    cfg=run.load_config();assert cfg['sources']==run.SOURCES
    kinds=Counter(run.plan_units().values())
    assert {k:kinds[k] for k in ('native50_trajectory','D181','E41','E41_Q','H137_score','E41_DCT','native_DCT','quality_pair','blind_plan','blind_read')}==dict(native50_trajectory=8,D181=12,E41=432,E41_Q=864,H137_score=216,E41_DCT=432,native_DCT=8,quality_pair=76,blind_plan=72,blind_read=72)
    q=run.quality_slots();assert sum(v['frame_denominator'] for v in q.values())==12316
    assert sum(v['adjacent_denominator'] for v in q.values())==12240
    families={w:receiver.check_family(family(2+start)) for w,start in run.WINDOWS.items()}
    assert receiver.select(families,'RANK3')['status']=='DETECT'
    assert receiver.select(families,'CDC_M')['detected'] is False
    assert receiver.select({'L':families['L'],'R':families['R']},'RANK2')['windows']==['L','R']
    fixed=receiver.select({},'FIXED_ORIGIN');assert fixed['detected'] is None and fixed['status']=='NO_EXISTENCE_DECISION'
    from main.tube_state import b_cdc32_aggregation_v1 as cdc
    assert cdc.plan_local.__globals__['_family'] is cdc._family
    assert receiver._singleton_builder.__globals__['_family'] is receiver._singleton_family
    tie=deepcopy(family(2));tie['candidates'][0]['S']=.2
    families['M']=receiver.check_family(tie)
    assert receiver.select(families,'RANK3')['status']=='METHOD_AMBIGUITY_ERASURE'


def test_full_fake_budget_isolation_and_direct_quality(tmp_path,monkeypatch):
    source=tmp_path/'input';out=tmp_path/'output';fixture_input(source)
    real_read=run.read;seen_plans=set();seen_decisions=set();bit_reads=[]
    def guarded_read(path):
        path=Path(path)
        if path.name=='bits.json':
            assert len(seen_plans)==72
            assert real_read(out/'plans_complete.json')['status']=='READ_BACK'
            bit_reads.append(path)
        if path==out/'protocol.json':assert len(seen_decisions)==72
        value=real_read(path)
        if path.name=='plan.json' and 'blind' in path.parts:seen_plans.add(path)
        if path.name=='decision.json':seen_decisions.add(path)
        return value
    monkeypatch.setattr(run,'read',guarded_read)
    backend=FakeBackend();value=run.run(source,out,backend=backend)
    assert value['status']=='COMPLETED_FIXED_ATTEMPT',value
    assert backend.calls==Counter(dict(generation_load=4,generate=8,book=8,native_Q=8,construct=4,vae_load=4,D181=12,codec_encode=48,codec_probe=48,codec_decode=48,E41=432,Q=864,score=216,E41_DCT=432,E41_bits=864,native_DCT=8,native_bits=16,quality=76))
    assert all(v['started']==v['returned']==v['planned'] and not v['failed'] for v in value['physical_counts'].values())
    ev=real_read(out/'evaluation.json');assert len(ev['tasks'])==72
    assert ev['checks']['required_observations_complete'] is True
    for r in run.READERS:
        assert ev['totals'][r+'/positive']['planned']==384
        assert ev['totals'][r+'/null']['planned']==1920
    assert ev['media_P0_PM_pair']['planned']==5760 and ev['native_P0_PM_pair']['planned']==2816
    direct=[v for k,v in ev['quality'].items() if '/OFF_to_PM/' in k]
    assert len(direct)==4 and all(v['compared_arms']==['OFF','PM'] for v in direct)
    before=backend.calls.copy();assert run.run(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert backend.calls==before


def test_first_failure_and_hard_exit_recovery(tmp_path):
    source=tmp_path/'input';out=tmp_path/'failed';fixture_input(source)
    backend=FakeBackend(fail='first_cal_E');result=run.run(source,out,backend=backend)
    assert result['status']=='TECHNICAL_UNRESOLVED' and backend.calls['E41']==1 and backend.calls['generate']==2
    assert not (out/'post_truth_access.json').exists()
    assert len(run.read(out/'planned_outputs.json')['quality'])==76
    assert run.run(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    interrupted=tmp_path/'hard'
    script="import os,sys;sys.path.insert(0,sys.argv[1]);from runtime.wan import b_rank3_four_source_confirmation_v1 as r;r.dump(sys.argv[2]+'/attempt.json',{});j=r.Journal(sys.argv[2]);j.call('inputs',lambda:os._exit(77))"
    done=subprocess.run([sys.executable,'-I','-c',script,str(run.ROOT),str(interrupted)],capture_output=True,text=True)
    assert done.returncode==77,done.stderr
    before={str(p):p.read_bytes() for p in interrupted.rglob('*') if p.is_file()}
    recovered=run.run('missing',interrupted,backend=object())
    assert recovered['recovery']['ledger']['inputs']['status']=='STARTED'
    assert recovered['recovery']['ledger']['inputs']['attempts']==1
    assert len(recovered['recovery']['ledger'])==len(run.plan_units())
    assert before=={str(p):p.read_bytes() for p in interrupted.rglob('*') if p.is_file()}


def test_actual_support_and_null_detect_independent_of_outputs():
    decision=run.empty_slots('ERASURE')
    decision['slots'][0].update(value=1,status='READ',selected_refs=[dict(evidence_id='e',nominal_received_cell=[1,2,3,4],canonical_cell=1)],supporting_refs=[])
    value=post.assess(decision,[1]*32,True,list(range(151)))
    assert value['counts']['false_attribution']==0 and value['counts']['unsupported_positive_outputs']==1
    tasks={}
    for task in run.tasks():
        positive=task.split('/')[1]=='PM' and task.endswith('/key00')
        judged=post.assess(run.empty_slots('ERASURE'),[1]*32,positive,list(range(151)))
        judged.update(detection_status='DETECT' if not positive else 'REJECT_GEOMETRY',technical_detail_slots=[])
        tasks[task]=dict(positive=positive,readers={r:deepcopy(judged) for r in run.READERS})
    summary=post.summarize(tasks,True)
    assert summary['method_status']=='STOP_FIXED_RANK3'
    assert summary['totals']['RANK3/null']['emitted']==0
    assert summary['checks']['null_detect_zero'] is False


import zipfile
from scripts import build_b_rank3_four_source_confirmation_v1_notebook as builder
ROOT=run.ROOT
AUDIT=Path('/home/richar/projects/Video-WM/diagnostics/b-rank3-stage2-implementation-20261025')
def notebook_code():
    nb=run.read(ROOT/'notebooks/b_rank3_four_source_confirmation_v1.ipynb')
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert len(cells)==4 and cells[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert 'SOURCE_OVERLAY' not in cells[1] and 'apply_source_overlay' not in cells[1]
    for c in cells:ast.parse(c)
    return cells

def notebook_bootstrap(cells,output):
    return cells[1].replace("OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/fixed-four-new-v1')",'OUTPUT = Path('+repr(str(output))+')')

def test_published_source_notebook_loading(tmp_path,monkeypatch):
    import shutil, tempfile, urllib.request
    archive=AUDIT/'available-b-source.zip'
    if not archive.is_file():pytest.skip('read-only published-source fixture unavailable')
    with zipfile.ZipFile(archive) as z:z.extractall(tmp_path/'base')
    roots=list((tmp_path/'base').glob('*/experiments/wan_state_clock/b_terminal_three_group_v1_run.py'));assert len(roots)==1
    base=roots[0].parents[2]
    # Assemble the candidate publication from the available branch plus the six new source/config files.
    for rel in builder.SOURCE_FILES:
        (base/rel).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/rel,base/rel)
    candidate=Path(shutil.make_archive(str(tmp_path/'candidate'),'zip',base.parent))
    downloads=[]
    def retrieve(url,destination):
        assert url=='https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/dev/local-joint-state-payload-v1'
        downloads.append(url);shutil.copyfile(candidate,destination)
        return str(destination),None
    monkeypatch.setattr(urllib.request,'urlretrieve',retrieve)
    mkdtemp=tempfile.mkdtemp
    monkeypatch.setattr(tempfile,'mkdtemp',lambda **kw:mkdtemp(dir=tmp_path,**kw))
    cells=notebook_code();env={}
    exec(notebook_bootstrap(cells,tmp_path/'new-output'),env)
    source=env['SOURCE_ROOT'];assert len(downloads)==1 and source!=base and not env['EXISTING_ARTIFACTS']
    assert not (source/'.git').exists()
    for rel in builder.SOURCE_FILES:assert (source/rel).read_bytes()==(ROOT/rel).read_bytes()
    probe=subprocess.run([sys.executable,'-c',"from pathlib import Path; import importlib; root=Path.cwd().resolve(); names=['main.tube_state.b_rank3_four_source_confirmation_v1','runtime.wan.b_rank3_four_source_confirmation_v1','runtime.wan.b_rank3_four_source_confirmation_backend_v1','runtime.wan.b_rank3_four_source_confirmation_evaluation_v1','experiments.wan_state_clock.b_rank3_four_source_confirmation_v1_run']; modules=[importlib.import_module(n) for n in names]; assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules)"],cwd=source,capture_output=True,text=True)
    assert probe.returncode==0,probe.stderr
    cli=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_rank3_four_source_confirmation_v1_run','--help'],cwd=source,capture_output=True,text=True)
    assert cli.returncode==0,cli.stderr
    # Optional local source also loads the new entry from a copy without modifying the input tree.
    before={str(p.relative_to(base)):p.stat().st_mtime_ns for p in base.rglob('*') if p.is_file()}
    local={};boot=notebook_bootstrap(cells,tmp_path/'local-output').replace('SOURCE_DIRECTORY = None','SOURCE_DIRECTORY = '+repr(str(base)))
    exec(boot,local)
    assert len(downloads)==1 and local['SOURCE_ROOT']!=base
    assert before=={str(p.relative_to(base)):p.stat().st_mtime_ns for p in base.rglob('*') if p.is_file()}

def test_notebook_existing_attempt_is_read_only(tmp_path,monkeypatch):
    import urllib.request,shutil
    cells=notebook_code()
    output=tmp_path/'existing';output.mkdir();run.dump(output/'attempt.json',dict(started=True))
    def forbidden(*a,**k):raise AssertionError('existing attempt touched source/setup/worker')
    monkeypatch.setattr(urllib.request,'urlretrieve',forbidden);monkeypatch.setattr(shutil,'copytree',forbidden)
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    env={};exec(notebook_bootstrap(cells,output),env)
    assert env['SOURCE_ROOT'] is None and env['EXISTING_ARTIFACTS']
    before={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
    exec(cells[2],env);exec(cells[3],env)
    assert len(env['recovery']['ledger'])==len(run.plan_units())
    assert before=={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}


def test_notebook_format_and_successful_setup_unchanged():
    import nbformat
    notebook=ROOT/'notebooks/b_rank3_four_source_confirmation_v1.ipynb'
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    assert len({c.id for c in nb.cells})==8
    original=(ROOT/'scripts/build_b_gamma2_fixed_validation_v1_notebook.py').read_text()
    setup=next(ast.literal_eval(n.value) for n in ast.parse(original).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SETUP' for t in n.targets))
    assert builder.SETUP==setup
    before=notebook.read_bytes();builder.build();assert notebook.read_bytes()==before
