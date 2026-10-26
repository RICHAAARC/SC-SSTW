"""M2 CPU-only checks: fake arrays/model/media; no physical experiment."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import ast,json,os,subprocess,sys
import pytest
from runtime.wan import b_rank3r_complete_repeat_v1 as run
from main.tube_state import b_rank3r_complete_repeat_v1 as receiver
from main.tube_state import b_rank3_four_source_confirmation_v1 as old
from runtime.wan import b_rank3r_complete_repeat_evaluation_v1 as post
pytestmark=pytest.mark.unit

def fixture_input(path):
    public=dict(model_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers',generation=dict(frames=181,height=320,width=512,fps=8),
        sources=deepcopy(run.SOURCES),writer=dict(key='watermark',gamma=1,bits=[1]*32))
    run.dump(path/'protocol.json',dict(config=public))
    model=dict(mean=[0.]*40,factor=[[float(i==j) for j in range(40)] for i in range(40)])
    run.dump(path/'background.json',dict(status='BACKGROUND_FROZEN',models={k:model for k in run.KEYS}))
    run.dump(path/'thresholds.json',dict(status='FROZEN',tau=run.TAU))
    planned={};ev={}
    for task in run.old_tasks():
        positive=task.split('/')[1]=='PM' and task.endswith('/key00')
        origins=(2,55,108) if positive else (0,0,0)
        selected={w:dict(origin=o,phase=(-o)%4,source_start=(o+(-o)%4)//4+1) for w,o in zip(run.WINDOWS,origins)}
        selection=dict(status='DETECT' if positive else 'REJECT_GEOMETRY',detected=positive,selected=selected)
        planned[task]=dict(status='RETURNED',readers={'RANK3':dict(path_context=selection)})
        ev[task]=dict(readers={'RANK3':dict(detection_status=selection['status'],slots=[dict(actual_supported_correct=positive) for _ in range(32)])})
    run.dump(path/'planned_outputs.json',dict(tasks=planned))
    run.dump(path/'evaluation.json',dict(status='POST_RECORDED',tasks=ev,quality={k:dict(status='SCORED') for k in run.historical_quality_record({})['expected_rows']}))
    for h in run.SOURCES:
        for arm in run.ARMS:
            file=path/'media'/h['id']/arm/'full/video.mp4';file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'fake MP4')

class FakeBackend:
    def __init__(self,fail=None):self.calls=Counter();self.files={};self.fail=fail;self.blocks={'encoder':0,'decoder':0}
    def book(self,key):self.calls['book']+=1;return key
    def load_vae(self):self.calls['vae_load']+=1;return dict(fake=True)
    def release_vae(self):self.calls['vae_release']+=1
    def probe(self,path):self.calls['codec_probe']+=1;return dict(fake=True)
    def read_media(self,path,probe,length):
        self.calls['codec_decode']+=1
        if length==181:
            h=next(x for x in run.SOURCES if x['id'] in Path(path).parts);arm=Path(path).parts[-3]
            return dict(host=h['id'],arm=arm,frames=list(range(181)))
        return self.files[str(path)]
    def edit(self,pixels,frames):self.calls['edit']+=1;return dict(pixels,frames=frames)
    def save_mp4(self,pixels,path):
        self.calls['codec_encode']+=1;self.files[str(path)]=pixels;Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(b'fake')
        return dict(fake=True)
    def codec_quality(self,a,b):self.calls['quality']+=1;return dict(status='SCORED',frame_denominator=151,adjacent_denominator=150)
    def prepare_window(self,x,w,p):return dict(x,window=w,phase=p,origin=x['frames'][w])
    def encode_view(self,x):
        self.calls['E41']+=1
        if self.fail=='E41' and self.calls['E41']==1:raise RuntimeError('fake first encoder failure')
        self.blocks['encoder']+=11;return x
    def observe(self,x,book):self.calls['Q']+=1;return dict(x,key=book)
    def score(self,bank,model):
        self.calls['score']+=1;x=bank['0'];positive=x['arm']=='PM' and x['key']=='watermark'
        p=(-x['origin'])%4;s=(x['origin']+p)//4+1
        return dict(status='SCORES_COMPLETE',candidates=[dict(phase=a,source_start=b,score_status='RETURNED',S=.9 if positive and (a,b)==(p,s) else .01 if positive else .1) for a,b in run.H])
    def payload_project(self,z,rows,regions):self.calls['DCT']+=1;return z
    def payload_decode(self,z,secret,view,key):
        self.calls['bits']+=1
        return dict(key=key,regions={r:dict(local_cells=[dict(local_row=i,evidence_id=f'{view}/row{i}/{r}',bits=[dict(local_bit=b,value=1,status='AVAILABLE',count=8,evidence_id=f'{view}/row{i}/{r}') for b in range(8)]) for i in range(1,11)]) for r in ('R0','R1')})

def test_domain_budget_and_old_isolation(tmp_path):
    assert len(old.PATHS)==961 and len(old.ENDPOINTS)==496 and len(receiver.PATHS)==4247
    assert sum(x['seam_multiplicity'] for x in receiver.PATHS.values())==35743
    assert all(receiver.PATHS[o]==p for o,p in old.PATHS.items())
    negative=[p for p in receiver.PATHS.values() if p['d']<0];assert len(negative)==3286
    assert all(1<=-p['d']<=min(b,151-b) for p in negative for b in p['legal_seams'])
    counts=Counter(run.plan_units().values())
    assert {k:counts[k] for k in run.BUDGET}==run.BUDGET
    assert counts['E41_DCT']==288 and counts['local_and_Rc_bits']==576 and counts['E41_Q']==576 and counts['H137_score']==144
    assert counts['blind_plan']==counts['blind_read']==48 and counts['quality_pair']==24
    assert sum(x['frame_denominator'] for x in run.quality_slots().values())==3624
    assert sum(x['adjacent_denominator'] for x in run.quality_slots().values())==3600
    fixture_input(tmp_path)
    value=run.old_regression(tmp_path,tmp_path/'out');assert value['status']=='OLD72_REGRESSION_PASSED' and value['preserved_supported_correct']==384

def test_full_fake_budget_blind_order_and_no_reentry(tmp_path,monkeypatch):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    real_read=run.read;plans=set();decisions=set();bit_reads=[]
    def guarded(path):
        path=Path(path)
        if path.name=='bits.json':
            assert len(plans)==48 and real_read(out/'plans_complete.json')['reader_plans']==96;bit_reads.append(path)
        if path==out/'protocol.json':assert len(decisions)==48 and real_read(out/'blind_complete.json')['reader_decisions']==96
        value=real_read(path)
        if path.name=='plan.json' and 'blind' in path.parts:plans.add(path)
        if path.name=='decision.json':decisions.add(path)
        return value
    monkeypatch.setattr(run,'read',guarded)
    backend=FakeBackend();value=run.run(source,out,backend=backend)
    assert value['status']=='COMPLETED_FIXED_ATTEMPT',value
    assert backend.calls==Counter(book=8,vae_load=4,vae_release=4,codec_encode=24,codec_probe=36,codec_decode=36,edit=24,quality=24,E41=288,Q=576,score=144,DCT=288,bits=576)
    assert all(x['started']==x['returned']==x['planned'] and x['failed']==0 for x in value['physical_counts'].values())
    ev=real_read(out/'evaluation.json');assert ev['method_status']=='M2_METHOD_CRITERIA_MET',ev['checks']
    assert ev['totals']['RANK3R/positive']['actual_supported_correct']==256 and ev['totals']['RANK3/null']['planned']==1280
    assert len(bit_reads)>0 and len(ev['quality'])==24 and ev['historical_quality']['expected_records']==76 and len(ev['historical_quality']['expected_rows'])==76
    assert not list(out.rglob('*.pt'))
    before={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
    assert run.run(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert before=={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}

def test_failure_and_hard_interrupt_preserve_fixed_slots(tmp_path):
    source=tmp_path/'input';out=tmp_path/'failed';fixture_input(source)
    backend=FakeBackend('E41');result=run.run(source,out,backend=backend)
    assert result['status']=='TECHNICAL_UNRESOLVED' and backend.calls['E41']==1 and backend.calls['codec_encode']==1
    assert backend.calls['vae_release']==1 and not (out/'post_truth_access.json').exists()
    ledger=run.read(out/'ledger.json');assert Counter(v['status'] for v in ledger.values())['FAILED']==1
    assert ledger['b-rank3-n02/vae_load']['status']=='NOT_ENTERED'
    assert len(run.read(out/'planned_outputs.json')['tasks'])==48
    assert run.run(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    hard=tmp_path/'hard'
    script="import os,sys;sys.path.insert(0,sys.argv[1]);from runtime.wan import b_rank3r_complete_repeat_v1 as r;r.dump(sys.argv[2]+'/attempt.json',{});j=r.Journal(sys.argv[2]);j.call('inputs',lambda:os._exit(77))"
    done=subprocess.run([sys.executable,'-I','-c',script,str(run.ROOT),str(hard)],capture_output=True,text=True)
    assert done.returncode==77,done.stderr
    recovered=run.run('missing',hard,backend=object())
    assert recovered['recovery']['ledger']['inputs']['status']=='STARTED' and recovered['recovery']['ledger']['inputs']['attempts']==1
    assert len(recovered['recovery']['ledger'])==len(run.plan_units())

def test_unseen_old_null_stops_before_physical(tmp_path):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    planned=run.read(source/'planned_outputs.json');task=run.old_tasks()[0]
    p=receiver.PATHS[(2,47,100)];selected={w:dict(origin=o,**p['states'][w]) for w,o in p['query_source_origins'].items()}
    planned['tasks'][task]['readers']['RANK3']['path_context'].update(selected=selected)
    run.dump(source/'planned_outputs.json',planned)
    backend=FakeBackend();value=run.run(source,out,backend=backend)
    assert value['status']=='METHOD_NOT_SUPPORTED' and not backend.calls
    assert run.read(out/'old72_regression.json')['null_detect']==1

def test_exact_support_not_selected_only():
    row=run.empty_slots('ERASURE');row['slots'][0].update(value=1,status='READ',selected_refs=[dict(evidence_id='e',nominal_received_cell=[1,2,3,4],canonical_cell=1)],supporting_refs=[])
    value=post.assess(row,[1]*32,True,list(range(151)))
    assert value['counts']['false_attribution']==0 and value['counts']['unsupported_positive_outputs']==1


def test_technical_missing_keeps_observed_violation_separate():
    tasks={}
    for task in run.tasks():
        positive=task.split('/')[1]=='PM' and task.endswith('/key00')
        value=post.assess(run.empty_slots('ERASURE'),[1]*32,positive,list(range(151)))
        value.update(detection_status='REJECT_GEOMETRY' if positive else 'DETECT',technical_detail_slots=[])
        tasks[task]=dict(positive=positive,readers={r:deepcopy(value) for r in run.READERS})
    regression=dict(preserved_supported_correct=384,lost_supported=0,null_detect=0)
    assert post.summarize(tasks,regression,True)['method_status']=='METHOD_NOT_SUPPORTED'
    tasks[run.tasks()[0]]['readers']['RANK3R']['technical_detail_slots']=[0]
    partial=post.summarize(tasks,regression,True)
    assert partial['method_status']=='TECHNICAL_UNRESOLVED' and partial['observed_method_violation']
    assert partial['checks']['new40_null_detect_zero'] is False

def test_receiver_only_gets_key_and_anonymous_view_ids(monkeypatch):
    from main.tube_state import b_m3_time_path_v1 as geometry
    families={};queries={}
    for w,offset in run.WINDOWS.items():
        phase,start=geometry.origin_to_state(offset)
        families[w,'key00']=dict(status='SCORES_COMPLETE',candidates=[dict(phase=p,source_start=s,score_status='RETURNED',S=.9 if (p,s)==(phase,start) else .01) for p,s in run.H])
        queries[w]=dict(views=[run.view_metadata('anonymous-input','anonymous-query',p,offset,f'anonymous-view-{w}-{p}','RETURNED') for p in range(4)])
    observed=[]
    real=receiver.plan_task
    def capture(key,families,views):
        observed.append(key);assert key=='key00';return real(key,families,views)
    monkeypatch.setattr(receiver,'plan_task',capture)
    value=run.make_plan('private-source/PM/REPEAT_LEFT_GAP/key00',dict(families=families,queries=queries))
    assert observed==['key00'] and set(value['readers'])=={'RANK3','RANK3R'}
    assert all(p['selection']['status']=='DETECT' for p in value['readers'].values())


def test_bad_score_family_stops_later_physical_work(tmp_path):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    backend=FakeBackend();backend.score=lambda *args:dict(status='SCORE_UNRESOLVED',candidates=[])
    value=run.run(source,out,backend=backend)
    assert value['status']=='TECHNICAL_UNRESOLVED' and backend.calls['E41']==4 and backend.calls['codec_encode']==1
    assert backend.calls['vae_release']==1 and not (out/'plans_complete.json').exists()
    slot='b-rank3-n01/OFF/REPEAT_LEFT_GAP/L/key00/score'
    assert run.read(out/'ledger.json')[slot]['status']=='FAILED'
    assert run.read(out/'families/b-rank3-n01/OFF/REPEAT_LEFT_GAP/L/key00.json')['status']=='SCORE_UNRESOLVED'


@pytest.mark.parametrize('mode',['missing','broken','nonfinite','different'])
def test_optional_historical_tau_cannot_stop_rank_chain(tmp_path,mode):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    path=source/'thresholds.json'
    if mode=='missing':path.unlink()
    elif mode=='broken':path.write_text('{broken')
    elif mode=='nonfinite':path.write_text('{"status":"FROZEN","tau":NaN}')
    else:run.dump(path,dict(status='OTHER_HISTORICAL_STATE',tau=.123))
    backend=FakeBackend('E41');result=run.run(source,out,backend=backend)
    assert result['status']=='TECHNICAL_UNRESOLVED' and backend.calls['E41']==1
    ledger=run.read(out/'ledger.json');assert ledger['threshold']['status']==ledger['old72']['status']=='RETURNED'
    value=run.read(out/'thresholds.json')
    assert value['record_only'] and not value['used_by_two_rank_readers'] and not value['new_calibration']
    json.dumps(value,allow_nan=False)
    if mode=='different':assert value['actual_tau']==.123 and value['actual_status']=='OTHER_HISTORICAL_STATE' and value['matches_reference'] is False
    else:assert value['read_error'] and value['actual_record'] is None

@pytest.mark.parametrize('mode',['missing','nonmapping','nonfinite_row'])
def test_historical_quality_missing_is_record_only(tmp_path,mode):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    value=run.read(source/'evaluation.json')
    if mode=='missing':value.pop('quality')
    elif mode=='nonmapping':value['quality']=['unusable mapping']
    else:
        first=next(iter(value['quality']));value['quality'][first]['rms']=float('inf')
        value['quality']['extra_available']=dict(status='SCORED',note='preserve readable extra row')
    (source/'evaluation.json').write_text(json.dumps(value))
    regression=run.old_regression(source,out)
    assert regression['status']=='OLD72_REGRESSION_PASSED' and regression['preserved_supported_correct']==384 and regression['null_detect']==0
    assert regression['physical_calls']==regression['new_Q_S_or_bit_reads']==0
    quality=run.read(out/'historical_quality.json')
    assert quality['record_only'] and not quality['affects_method_checks'] and quality['expected_records']==len(quality['expected_rows'])==76
    assert quality['expected_frame_denominator']==12316 and quality['expected_adjacent_denominator']==12240
    json.dumps(quality,allow_nan=False)
    if mode=='nonfinite_row':
        assert quality['available_records']==75 and len(quality['missing_or_invalid'])==1
        assert quality['expected_rows'][first]['status']=='INVALID' and quality['available_extra_records']==1
        assert quality['records']['extra_available']['note']=='preserve readable extra row'
    else:
        assert quality['available_records']==0 and len(quality['missing_or_invalid'])==76
        assert all(v['status']=='MISSING' for v in quality['expected_rows'].values())
