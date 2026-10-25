"""CPU-only fixed gamma2 checks. Never loads a VAE or invokes a real codec."""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from runtime.wan import b_gamma2_fixed_validation_v1 as r
from runtime.wan import b_gamma2_fixed_validation_evaluation_v1 as ev
from scripts import build_b_gamma2_fixed_validation_v1_notebook as builder

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT.parents[1]/'diagnostics/b-gamma2-implementation-20261025'
H=[(p,s) for p in range(4) for s in range(1 if p==0 else 2,36)]


def family(origin=None):
    phase=(-origin)%4 if origin is not None else None
    start=(origin+phase)//4+1 if origin is not None else None
    return dict(status='SCORES_COMPLETE',candidates=[dict(phase=p,source_start=s,S=(.8 if (p,s)==(phase,start) else 0.) if origin is not None else (.1 if (p,s)==(0,1) else 0.),score_status='RETURNED') for p,s in H])


def bit_record(view,key):
    return dict(status='AVAILABLE',cause_set=[],kind='E41',view_id=view,key=key,local_rows=list(range(1,11)),
        regions={region:dict(local_cells=[dict(local_row=i,evidence_id=f'{view}/row{i}/{region}',
             bits=[dict(local_bit=b,value=0,count=8,status='RETURNED') for b in range(8)]) for i in range(1,11)]) for region in ('R0','R1')})


def fixture_input(source):
    from runtime.wan.b_terminal_three_group_v1 import source_frames
    cfg=dict(model_id='no-model-fixture',sources=[dict(id=h,role=role,a=a) for h,role,a in r.HOSTS],
             generation=dict(frames=181,height=320,width=512,fps=8),writer=dict(key='watermark',gamma=1,bits=[0]*32))
    r.dump(source/'protocol.json',dict(config=cfg))
    bg=dict(status='BACKGROUND_FROZEN',models={key:dict(status='BACKGROUND_FROZEN',dimension=40,mean=[0.]*40,
        factor=[[float(i==j) for j in range(40)] for i in range(40)]) for key in r.KEYS})
    r.dump(source/'background.json',bg);r.dump(source/'thresholds.json',dict(status='FROZEN',tau=.1))
    for host in cfg['sources']:
        h,role=host['id'],host['role'];r.dump(source/'private'/h/'P0_terminal.pt',dict(h=h));r.dump(source/'private'/h/'native/P0_Q.json',dict(status='SCORE_OBSERVED'))
        r.dump(source/'media'/h/'PM/full/preclip.json',dict(status='RECORDED'))
        for scope,layers in [('full',('preclip','rgb8','readback'))]+[(f,('rgb8','readback')) for f in r.FIXTURES]:
            for layer in layers:
                r.dump(source/'media'/h/'P0'/scope/(layer+'.pt'),dict(h=h))
                r.dump(source/'quality'/h/'quality'/scope/'P0_to_PM'/(layer+'.json'),dict(status='SCORED'))
        for arm in ('OFF','P0','PM'):
            for fixture in r.FIXTURES:
                frames=source_frames(host,fixture);input_id=f'{h}-{arm}-{fixture}'
                locals_={w:dict(query_id=input_id+'-'+w,input_id=input_id,received_query_start=start) for w,start in r.WINDOWS.items()}
                for key in r.KEYS:
                    if role=='cal' and arm=='PM' and key=='key00':continue
                    for window,w in r.WINDOWS.items():
                        f=family(frames[w] if role=='eval' and arm=='PM' and key=='key00' else None)
                        # Original disk family lacks input_id/key_id; old writer added them after save.
                        r.dump(source/'families'/h/arm/fixture/window/(key+'.json'),f)
                    if role!='eval':continue
                    r.dump(source/'blind'/h/arm/fixture/key/'plan.json',dict(locals=locals_))
                    for window in r.WINDOWS:
                        for phase in range(4):
                            r.dump(source/'observations'/h/arm/fixture/window/f'p{phase}'/key/'bits.json',bit_record(input_id+f'-{window}-p{phase}',key))
                if role=='eval' and arm=='PM':
                    refs={}
                    for group in range(4):
                        for t in range(group*11+1,group*11+12):
                            fs=list(range(4*t-3,4*t+1))
                            if all(x in frames for x in fs):
                                refs[group]=dict(nominal_received_cell=[frames.index(x) for x in fs],canonical_cell=t);break
                    slots=[dict(slot=i,value=0,status='READ',selected_refs=[refs[i//8]],supporting_refs=[refs[i//8]]) for i in range(32)]
                    r.dump(source/'blind'/h/arm/fixture/'key00/decision.json',dict(aggregates=dict(CDC_M=dict(slots=slots))))


class FakeBackend:
    def __init__(self,fail_at=None,hard_exit=False):self.media={};self.h=None;self.encodes=0;self.fail_at=fail_at;self.hard_exit=hard_exit;self.hosts=[];self.blocks={}
    def preflight(self,source):
        for h,_,_ in r.HOSTS:
            assert r.read(source/'private'/h/'P0_terminal.pt')['h']==h
            assert r.read(source/'private'/h/'native/P0_Q.json')['status']=='SCORE_OBSERVED'
        return dict(status='FAKE_INPUTS_VALID')
    def book(self,key):return key
    def construct_saved(self,path,native):
        self.h=path.parent.name;self.hosts.append(self.h)
        return dict(h=self.h),dict(gamma=2),dict(gamma=2,units=[])
    def load_vae(self):self.blocks=dict(encoder=0,decoder=0);return dict(status='FAKE')
    def release_vae(self):return dict(status='RELEASED')
    def save_tensor(self,path,value):r.dump(path,value)
    def load_tensor(self,path):return r.read(path)
    def decode(self,z,event):
        event('started');self.blocks['decoder']+=1;event('returned')
        return dict(preclip=z,float_rgb=z)
    def preclip(self,x):return dict(status='RECORDED')
    def quantize(self,x):return x
    def save_mp4(self,x,path,event):event('started');self.media[str(path)]=x;event('returned');return dict(status='SAVED')
    def probe(self,path,event):event('started');event('returned');return dict(status='PROBED')
    def read_media(self,path,probe,length,event):event('started');event('returned');return self.media[str(path)]
    def edit(self,pixels,frames):return dict(pixels,frames=frames)
    def prepare_window(self,pixels,w,p):return dict(pixels,w=w,phase=p)
    def encode_view(self,pixels,event):
        self.encodes+=1;event('started')
        if self.hard_exit:os._exit(77)
        if self.encodes==self.fail_at:event('failed');raise RuntimeError('injected E41 failure')
        self.blocks['encoder']+=1;event('returned');return pixels
    def observe(self,z,book):return dict(z,key=book)
    def score(self,bank,bg):
        z=bank['0'];positive=z['h'].startswith('b2-m3-e') and z['key']=='watermark'
        return family(z['frames'][z['w']] if positive else None)
    def payload_project(self,z,rows,regions):return z
    def payload_decode(self,z,key,view,key_id):return bit_record(view,key_id)
    def quality(self,a,b,frames,pair):return dict(status='SCORED',frame_denominator=len(frames),adjacent_denominator=len(frames)-1)


def test_counts_calibration_and_supporting_refs():
    counts=Counter(r.plan_units().values())
    assert {k:counts[k] for k in r.BUDGET}==r.BUDGET
    assert counts['Q']==216 and counts['H137_score']==54 and counts['DCT']==72 and counts['key_bits']==144 and counts['quality']==36
    assert len(r.planned_outputs()['tasks'])==48 and r.denominators()['aggregate_slots_each']==1536
    from main.tube_state.b_m3_time_path_v1 import calibrate,prepare_local
    a=calibrate({str(i):family() for i in range(90)});b=calibrate({str(i):family(20) if i==89 else family() for i in range(90)})
    common=r.freeze_common(a,b);assert common['tau']==.8 and common['planned_families']==108
    local=prepare_local(family(20),common,'query',0);assert local['D'] is False and len(local['C'])==1
    exact=dict(nominal_received_cell=[1,2,3,4],canonical_cell=1)
    wrong=dict(nominal_received_cell=[5,6,7,8],canonical_cell=1)
    out=ev.assess(dict(slots=[dict(value=0,selected_refs=[exact],supporting_refs=[wrong])]),[0]*32,True,list(range(151)))
    assert out['slots'][0]['any_exact_reference'] and not out['slots'][0]['actual_supporting_exact']


def test_header_stops_before_values(tmp_path):
    path=tmp_path/'bits.json';path.write_text('{"status":"AVAILABLE","view_id":"v","key":"key00","local_rows":[1,2],"regions": INVALID BIT VALUES')
    assert r.read_bit_header(path)['view_id']=='v'
    real=AUDIT/'saved_e01_PM_CROP_L_p0_key00_bits.json'
    if real.is_file():
        header=r.read_bit_header(real);assert header['key']=='key00' and header['kind']=='E41' and 'regions' not in header


def test_full_fake_all_plans_before_bits(tmp_path,monkeypatch):
    source,out=tmp_path/'input',tmp_path/'out';fixture_input(source)
    original=r.SavedBitCache.__call__;accesses=[]
    def check(self,*args):
        if not accesses:
            plans=list((out/'blind').rglob('plan.json'));assert len(plans)==48
            assert all(r.read(p)['common_threshold']['planned_families']==108 for p in plans)
        accesses.append(args);return original(self,*args)
    monkeypatch.setattr(r.SavedBitCache,'__call__',check)
    backend=FakeBackend();result=r.run(source,out,backend)
    assert result['status']=='COMPLETED_ENGINEERING_RECORDS',result.get('stop')
    assert accesses and backend.encodes==144 and len(backend.hosts)==4
    for k,n in r.BUDGET.items():assert result['physical_counts'][k]['started']==result['physical_counts'][k]['returned']==n
    evaluation=r.read(out/'evaluation.json')
    assert len(evaluation['tasks'])==48 and len(evaluation['quality'])==36
    assert evaluation['totals']['CDC_M/positive']['planned']==384 and evaluation['totals']['CDC_M/null']['planned']==1152
    assert all(v['positive']['planned']==192 and v['null']['planned']==960 for v in evaluation['per_gamma'].values())
    before={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
    assert r.run(source,out,FakeBackend(hard_exit=True))['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert before=={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}


def test_midcal_failure_and_hard_exit(tmp_path):
    source,out=tmp_path/'input',tmp_path/'failed';fixture_input(source)
    backend=FakeBackend(fail_at=5);result=r.run(source,out,backend)
    assert result['status']=='STOPPED_FIXED_ATTEMPT' and backend.hosts==['b2-m3-c01'] and backend.encodes==5
    assert result['physical_counts']['E41']['failed']==1
    ledger=r.read(out/'ledger.json');assert ledger['b2-m3-e01/full/D']['status']=='NOT_ENTERED'
    hard=tmp_path/'hard'
    code="import runpy,sys; m=runpy.run_path(sys.argv[1]); m['r'].run(sys.argv[2],sys.argv[3],m['FakeBackend'](hard_exit=True))"
    done=subprocess.run([sys.executable,'-c',code,str(Path(__file__)),str(source),str(hard)],cwd=ROOT)
    assert done.returncode==77
    slot='b2-m3-c01/CROP/L/p0/E';ledger=r.read(hard/'ledger.json');ledger[slot].update(status='NOT_ENTERED',attempts=0);r.dump(hard/'ledger.json',ledger)
    before=(hard/'ledger.json').read_bytes();recovery=r.run(source,hard,FakeBackend())
    assert recovery['recovery']['ledger'][slot]['status']=='STARTED' and recovery['recovery']['ledger'][slot]['attempts']==1
    assert (hard/'ledger.json').read_bytes()==before


def test_gamma2_new_active_unit_and_audit(tmp_path):
    import torch
    from runtime.wan.b_gamma2_fixed_validation_backend_v1 import WanBackend,state
    torch.set_num_threads(1);book=state.codebook('watermark');z=torch.zeros((1,16,46,40,64))
    y0,y1,x0,x1=state.ROIS[0]
    z[0,4:16,1,y0:y1,x0:x1]=(1.5*book['signs'][0,0]*book['directions'][0,0]/book['norm_squared'][0,0]).float()
    native=state.observe(z,book);p=tmp_path/'P0.pt';q=tmp_path/'Q.json';torch.save(z,p);r.dump(q,native)
    changed,receipt,audit=WanBackend('no-model').construct_saved(p,q)
    assert native['roi_q'][0][0][0]>1 and audit['native_Q']['roi_q'][0][0][0]==pytest.approx(2.,abs=1e-6)
    assert not torch.equal(z,changed) and audit['units'][0]['shortage']==max(0,2-audit['units'][0]['q'])
    assert audit['gamma']==2 and audit['payload_channels_max']==audit['guard_max']==0


def notebook_code():
    nb=r.read(ROOT/'notebooks/b_gamma2_fixed_validation_v1.ipynb')
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert len(cells)==4 and cells[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert 'SOURCE_OVERLAY' not in cells[1] and 'apply_source_overlay' not in cells[1]
    for c in cells:ast.parse(c)
    return cells


def notebook_bootstrap(cells,output):
    return cells[1].replace("OUTPUT = Path('/content/drive/MyDrive/Video-WM/B-Gamma2-Fixed-Validation-V1/fixed-four-source-v1')",'OUTPUT = Path('+repr(str(output))+')')


def test_published_source_notebook_loading(tmp_path,monkeypatch):
    import shutil, tempfile, urllib.request
    archive=AUDIT/'available-branch-source.zip'
    if not archive.is_file():pytest.skip('read-only published-source fixture unavailable')
    with zipfile.ZipFile(archive) as z:z.extractall(tmp_path/'base')
    roots=list((tmp_path/'base').glob('*/experiments/wan_state_clock/b_terminal_three_group_v1_run.py'));assert len(roots)==1
    base=roots[0].parents[2]
    # Assemble the candidate publication from the available branch plus the four new source files.
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
    probe=subprocess.run([sys.executable,'-c',"from pathlib import Path; import importlib; root=Path.cwd().resolve(); names=['runtime.wan.b_gamma2_fixed_validation_v1','runtime.wan.b_gamma2_fixed_validation_backend_v1','runtime.wan.b_gamma2_fixed_validation_evaluation_v1','experiments.wan_state_clock.b_gamma2_fixed_validation_v1_run']; modules=[importlib.import_module(n) for n in names]; assert all(Path(m.__file__).resolve().is_relative_to(root) for m in modules)"],cwd=source,capture_output=True,text=True)
    assert probe.returncode==0,probe.stderr
    cli=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_gamma2_fixed_validation_v1_run','--help'],cwd=source,capture_output=True,text=True)
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
    output=tmp_path/'existing';output.mkdir();r.dump(output/'attempt.json',dict(started=True))
    def forbidden(*a,**k):raise AssertionError('existing attempt touched source/setup/worker')
    monkeypatch.setattr(urllib.request,'urlretrieve',forbidden);monkeypatch.setattr(shutil,'copytree',forbidden)
    monkeypatch.setattr(subprocess,'run',forbidden);monkeypatch.setattr(subprocess,'Popen',forbidden)
    env={};exec(notebook_bootstrap(cells,output),env)
    assert env['SOURCE_ROOT'] is None and env['EXISTING_ARTIFACTS']
    before={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
    exec(cells[2],env);exec(cells[3],env)
    assert len(env['recovery']['ledger'])==len(r.plan_units())
    assert before=={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}


def test_selective_protocol_skips_bits_and_preflight_before_physics(tmp_path,monkeypatch):
    source=tmp_path/'input';fixture_input(source)
    cfg=r.read(source/'protocol.json');cfg['config']['writer']['bits']=[987654321]
    r.dump(source/'protocol.json',cfg)
    loads=r.json.loads
    def guarded(text,*args,**kwargs):
        assert '987654321' not in text,'payload deserialized before post'
        return loads(text,*args,**kwargs)
    monkeypatch.setattr(r.json,'loads',guarded)
    selected=r.read_public_protocol(source/'protocol.json')
    assert 'bits' not in selected['config']['writer'] and selected['config']['writer']['gamma']==1
    monkeypatch.setattr(r.json,'loads',loads)
    bad=source/'private/b2-m3-e02/P0_terminal.pt';bad.write_text('unreadable as JSON')
    backend=FakeBackend();result=r.run(source,tmp_path/'out',backend)
    assert result['status']=='STOPPED_FIXED_ATTEMPT' and backend.hosts==[]
    assert all(x['started']==0 for x in result['physical_counts'].values())


def test_recovery_without_ledger_and_failed_blind_slots(tmp_path):
    slot='b2-m3-c01/full/D';r.dump(tmp_path/'receipts'/(slot+'.json'),dict(slot=slot,kind='D181',status='STARTED',attempts=1))
    recovered=r.recover_ledger(tmp_path)
    assert len(recovered['ledger'])==len(r.plan_units()) and recovered['ledger'][slot]['attempts']==1
    assert recovered['ledger']['b2-m3-e02/full/D']['status']=='NOT_ENTERED'
    r.dump(tmp_path/'receipts'/(slot+'.json.tmp'),dict(slot=slot,kind='D181',status='RETURNED',attempts=1))
    unresolved=r.recover_ledger(tmp_path);assert unresolved['ledger'][slot]['status']=='UNRESOLVED_RECEIPT' and unresolved['ledger'][slot]['attempts']==1
    outputs=r.planned_outputs();task=next(iter(outputs['tasks']))
    ledger={'blind/'+task+'/plan':dict(status='FAILED',attempts=1,error='planned write failed')}
    actual=r.output_statuses(outputs,ledger)['tasks'][task]
    assert actual['status']=='FAILED' and len(actual['aggregates']['CDC_M']['slots'])==32
    assert all(x['status']=='FAILED' for x in actual['aggregates']['CDC_M']['slots'])


def test_header_and_key_metadata_rejections(tmp_path):
    source=tmp_path/'input';fixture_input(source)
    target=source/'blind/b2-m3-e01/PM/CROP/key01/plan.json';x=r.read(target);x['locals']['L']['query_id']='other';r.dump(target,x)
    with pytest.raises(ValueError,match='key plans disagree'):r.old_context(source,'b2-m3-e01','PM1','CROP')
    bad=tmp_path/'bad.json';bad.write_text('{"status":"AVAILABLE","payload":[123],"regions":{}}')
    with pytest.raises(ValueError,match='unexpected field'):r.read_bit_header(bad)


def test_quality_failure_preserves_36_slots(tmp_path):
    source,out=tmp_path/'input',tmp_path/'out';fixture_input(source)
    r.dump(out/'blind_complete.json',dict(status='READ_BACK'));r.dump(out/'thresholds.json',dict(status='FROZEN',tau=.1))
    for task in r.planned_outputs()['tasks']:
        r.dump(out/'blind'/task/'plan.json',dict(locals={w:dict(D=False,C=[],maximum=.1,members=[]) for w in r.WINDOWS},paths={}))
        r.dump(out/'blind'/task/'decision.json',dict(local={reg:{w:{} for w in r.WINDOWS} for reg in ('R0','R1','CDC')},aggregates={name:{} for name in r.READERS}))
    for h,_,_ in r.HOSTS:
        r.dump(out/'private'/h/'native_audit.json',{});r.dump(out/'media'/h/'PM2/full/preclip.json',{})
    class BadQuality(FakeBackend):
        def load_tensor(self,path):raise FileNotFoundError('missing old quality tensor')
    journal=r.Journal(out)
    with pytest.raises(FileNotFoundError):ev.evaluate(source,out,BadQuality(),journal)
    saved=r.read(out/'evaluation.json')['quality']
    assert len(saved)==36 and Counter(x['status'] for x in saved.values())=={'FAILED':1,'NOT_ENTERED':35}
    assert sum(x['frame_denominator'] for x in saved.values())==5796
    assert sum(x['adjacent_denominator'] for x in saved.values())==5760


def test_positive_support_missing_has_separate_zero_rule():
    frames=list(range(151));exact=dict(nominal_received_cell=[1,2,3,4],canonical_cell=1)
    observed=ev.assess(dict(slots=[dict(value=0,selected_refs=[exact],supporting_refs=[])]),[0]*32,True,frames)
    assert observed['counts']['false_attribution']==0 and observed['counts']['unsupported_positive_outputs']==1
    tasks={}
    for task in r.planned_outputs()['tasks']:
        positive=task.split('/')[1] in ('PM1','PM2') and task.endswith('key00')
        value=ev.assess({},[0]*32,positive,frames)
        tasks[task]=dict(positive=positive,aggregates={name:value for name in r.READERS})
    tasks['b2-m3-e02/PM2/CROP/key00']['aggregates']['CDC_M']=observed
    hist={f:ev.assess({},[0]*32,True,frames) for f in r.FIXTURES}
    summary=ev.summarize(tasks,hist,False)
    assert summary['support_checks']['actual_supporting_attribution_zero'] is False


def test_unknown_receipts_and_invalid_saved_rows_keep_fixed_plan(tmp_path):
    slot='b2-m3-c01/full/D';other='b2-m3-c02/full/D'
    r.dump(tmp_path/'ledger.json',{slot:dict(slot=slot,kind='WRONG_KIND',status='STARTED',attempts=1)})
    r.dump(tmp_path/'receipts/unknown.json',dict(slot='unknown',kind='D181',status='STARTED',attempts=1))
    r.dump(tmp_path/'receipts'/(other+'.json'),dict(slot=other,kind='WRONG_KIND',status='STARTED',attempts=1))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    recovered=r.recover_ledger(tmp_path)
    assert len(recovered['ledger'])==len(r.plan_units()) and 'unknown' not in recovered['ledger']
    assert recovered['ledger'][slot]['status']=='UNRESOLVED_LEDGER' and recovered['ledger'][slot]['attempts']==1
    assert recovered['ledger'][other]['status']=='UNRESOLVED_RECEIPT' and recovered['ledger'][other]['attempts']==1
    assert 'receipts/unknown.json' in recovered['read_errors']
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_common_metadata_is_inside_journal_result(tmp_path):
    from main.tube_state.b_m3_time_path_v1 import calibrate
    first=calibrate({str(i):family() for i in range(90)});second=calibrate({str(i):family() for i in range(90)})
    journal=r.Journal(tmp_path)
    result=journal.call('calibration/common',lambda:r.freeze_common(first,second,.25),tmp_path/'thresholds.json')
    saved=r.read(tmp_path/'thresholds.json')
    assert saved==result and saved['status']=='FROZEN' and saved['tau']==.1
    assert saved['tau1_record']['equal'] is False and saved['tau1_record']['equality_is_gate'] is False
    assert saved['tau1_record']['delta']==pytest.approx(-.15)
    assert r.read(tmp_path/'receipts/calibration/common.json')['status']=='RETURNED'
