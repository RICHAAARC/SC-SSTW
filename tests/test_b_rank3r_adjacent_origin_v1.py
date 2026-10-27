"""Synthetic-only tests; never open the historical candidate datasets."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from main.tube_state import b_rank3r_adjacent_origin_v1 as method
from main.tube_state import b_rank3r_complete_repeat_v1 as rank
from main.tube_state import b_rank3_four_source_confirmation_v1 as old
from main.tube_state import b_m3_time_path_v1 as geometry
from runtime.wan import b_rank3r_adjacent_origin_v1 as run
pytestmark=pytest.mark.unit


def family(top,second,delta=.01):
    rows=[]
    for phase,start in geometry.H:
        origin=geometry.state_to_origin(phase,start)
        rows.append(dict(phase=phase,source_start=start,score_status='RETURNED',S=.8 if origin==top else .8-delta if origin==second else .0))
    return old.check_family(dict(status='SCORES_COMPLETE',candidates=rows))


def families(top=(6,66,120),second=(5,67,119),deltas=(.02,.001,.02)):
    return {w:family(a,b,d) for w,a,b,d in zip(method.WINDOWS,top,second,deltas)}


def views():
    return {w:[run.view_metadata('opaque-input','opaque-query-'+w,p,start,w+'-'+str(p),'RETURNED') for p in range(4)] for w,start in method.WINDOWS.items()}


def test_two_legal_delta_choice_and_real_second_score_plan():
    f=families();before=deepcopy(f);record=method.select(f)
    assert record['selection']['origins']==[6,67,120]
    assert record['correction']['legal_count']==2 and record['correction']['status']=='MULTIPLE_LEGAL_MINIMUM_DELTA'
    assert record['correction']['selected_window']=='M' and record['correction']['delta_gap']>0
    assert record['membership_checks']==4
    plan=method.plan_task('key00',f,views(),record)
    mid=next(p for p in plan['locals'] if p['window']=='M')
    assert mid['selected_rank']==2 and mid['maximum']==.8 and f==before
    assert max(r['S'] for r in mid['members'])==.8
    assert next(r for r in mid['members'] if r['in_C'])['S']==.8-.001
    # Same geometry can legitimately choose R when its actual delta is smaller.
    other=method.select(families(deltas=(.02,.03,.001)))
    assert other['correction']['selected_window']=='R' and other['selection']['origins']==[6,66,119]
    right=method.select(families((6,58,104),(5,59,103),(.02,.001,.03)))
    assert right['correction']['selected_window']=='M' and right['correction']['legal_count']==2


def test_ties_missing_and_no_multi_window_repair():
    tied=method.select(families(deltas=(.02,.01,.01)))
    assert tied['selection']['status']=='METHOD_AMBIGUITY_ERASURE' and tied['correction']['delta_gap']==0
    f=families();target=f['M']['members'];target[0]['S']=.8
    f['M']=old.check_family(dict(status='SCORES_COMPLETE',candidates=target))
    first=method.select(f);assert not first['correction']['entered'] and first['membership_checks']==0
    f=families();target=f['M']['members'];target[0]['S']=.799
    f['M']=old.check_family(dict(status='SCORES_COMPLETE',candidates=target))
    second=method.select(f);assert next(r for r in second['correction']['alternatives'] if r['window']=='M')['status']=='SECOND_TIE'
    missing=families();missing['L']=old.check_family({})
    assert method.select(missing)['selection']['status']=='TECHNICAL_UNRESOLVED'
    no=method.select(families((0,70,136),(1,71,135)))
    assert no['selection']['status']=='REJECT_GEOMETRY' and no['correction']['legal_count']==0


def test_original_success_and_erasure_plan_unchanged():
    f=families((6,59,112),(5,58,111));metadata=views()
    new=method.plan_task('key00',f,metadata)
    original=rank.plan_task('key00',f,metadata)['RANK3R']
    assert new['selection']==original['selection'] and new['locals']==original['locals']
    assert new['correction']['status']=='ORIGINAL_BRANCH_PRESERVED'
    # The untouched read chain retains exact output/support/conflict/erasure rows.
    class Bits:
        accesses=0
        def __call__(self,evidence,key,bit):self.accesses+=1;return dict(value=bit%2,status='AVAILABLE',count=8)
    cache=Bits();new_dec=run._decision(new,cache)
    original_with_audit=dict(original,correction=dict(selected_window=None,status='ORIGINAL_BRANCH_PRESERVED'))
    old_dec=run._decision(original_with_audit,Bits())
    assert new_dec['aggregate']==old_dec['aggregate']


def test_header_only_and_fixed_denominators(tmp_path):
    rows=run.roster();assert len(rows)==312 and sum(r['read_bits'] for r in rows)==276
    assert sum(r['task'].split('/')[1]=='PM' and r['key']=='key00' for r in rows)==52
    assert sum(r['read_bits'] and r['task'].split('/')[1]=='PM' and r['key']=='key00' for r in rows)==46
    file=tmp_path/'metadata.json'
    file.write_text('{"status":"AVAILABLE","cause_set":[],"kind":"E41","view_id":"real-view","key":"key00","local_rows":[1,2,3,4,5,6,7,8,9,10],"regions":THIS_IS_NOT_PARSED')
    assert run.read_bit_header(file)['view_id']=='real-view'
    with pytest.raises(ValueError):run.read(file)
    c=run.Cache()
    with pytest.raises(RuntimeError):c('anything','key00',0)


def test_fixed_missing_roster_never_becomes_null_success(tmp_path,monkeypatch):
    catalog=tmp_path/'catalog.json';catalog.write_text('{"groups":{}}');output=tmp_path/'output'
    real_read=run.read;seen_plans=set();seen_decisions=set()
    def guarded(path):
        path=Path(path)
        if path.name=='plan.json' and 'blind' in path.parts:seen_plans.add(str(path))
        if path.name=='decision.json':seen_decisions.add(str(path))
        if path.name in ('protocol.json','evaluation-input.json'):assert len(seen_decisions)==276
        return real_read(path)
    monkeypatch.setattr(run,'read',guarded)
    result=run.run(catalog,output)
    assert result['status']=='COMPLETED_FIXED_CPU_ATTEMPT' and result['physical_calls']==0
    assert len(seen_plans)==312 and len(seen_decisions)==276
    evaluation=real_read(output/'evaluation.json')
    assert evaluation['candidate_status']=='TECHNICAL_UNRESOLVED' and len(evaluation['technical_detection_tasks'])==312
    assert len(real_read(output/'planned_outputs.json'))==312
    assert sum(len(v['slots']) for v in real_read(output/'planned_outputs.json').values())==8832
    before={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}
    assert run.run(catalog,output)['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert before=={str(p):p.stat().st_mtime_ns for p in output.rglob('*') if p.is_file()}


@pytest.mark.parametrize('top,second,window',[
    ((1,53,107),(0,40,20),'L'),
    ((0,52,106),(20,53,20),'M'),
    ((0,54,106),(20,40,107),'R')])
def test_unique_legal_replacement_can_be_any_window(top,second,window):
    record=method.select(families(top,second))
    assert record['correction']['status']=='UNIQUE_LEGAL_ALTERNATIVE'
    assert record['correction']['selected_window']==window
    assert sum(a!=b for a,b in zip(record['original_selection']['origins'],record['selection']['origins']))==1


def test_actual_needed_bits_unreadable_retain_fixed32(tmp_path,monkeypatch):
    plan=method.plan_task('key00',families((6,59,112),(5,58,111)),views())
    actual=run.read;reads=[]
    def observed(path):reads.append(path);return actual(path)
    monkeypatch.setattr(run,'read',observed)
    cache=run.Cache();cache.enabled=True
    for window in method.WINDOWS:
        for phase in range(4):cache.register(tmp_path/(window+'-'+str(phase)+'-absent.json'),'key00',window+'-'+str(phase))
    result=run._decision(plan,cache)
    assert len(result['aggregate']['slots'])==32 and run._technical(result['aggregate'],result['selection'])
    assert all(row['value'] is None for row in result['aggregate']['slots'])
    assert len(reads)==3 and len(cache.failures)==3  # A known unreadable saved file is not retried per bit.


def test_synthetic_values_wait_for_all_plans_and_truth_for_all_decisions(tmp_path,monkeypatch):
    source=tmp_path/'input';source.mkdir();out=tmp_path/'out';group=dict(root=str(source),protocol='protocol.json',evaluation='evaluation-input.json',tasks={})
    truth=[i%2 for i in range(32)]
    run.dump(source/'protocol.json',dict(config=dict(writer=dict(bits=truth),sources=[dict(id=h,a=6) for h in run.GROUPS['stage1']['hosts']])))
    for w,f in families((6,59,112),(5,58,111)).items():run.dump(source/(w+'.json'),dict(status='SCORES_COMPLETE',candidates=f['members']))
    for w in run.WINDOWS:
        for phase in range(4):
            for key in run.KEYS:
                view=w+'-'+str(phase)
                record=dict(status='AVAILABLE',cause_set=[],kind='E41',view_id=view,key=key,local_rows=list(range(1,11)),regions={})
                for region in ('R0','R1'):
                    record['regions'][region]=dict(local_cells=[dict(local_row=i,evidence_id=f'{view}/row{i}/{region}',
                        status='AVAILABLE',observed_region_id=region,carrier_region_id=region,
                        observed_region_slice=[8,24,8,24] if region=='R0' else [8,24,40,56],
                        channels=[0,4],coordinate_space='normalized_latent',diagnostic_only=False,
                        bits=[dict(local_bit=b,value=b%2,status='AVAILABLE',count=8,key=key,planned_count=8,returned_count=8,
                        diagnostic_only=False,evidence_id=f'{view}/row{i}/{region}') for b in range(8)]) for i in range(1,11)])
                run.dump(source/'bits'/w/str(phase)/key/'bits.json',record)
    historical={}
    for row in [r for r in run.roster() if r['group']=='stage1']:
        task=row['task'];entry=dict(families={w:w+'.json' for w in run.WINDOWS})
        if row['read_bits']:
            entry.update(views=views(),bits={w:{str(p):f'bits/{w}/{p}/{row["key"]}/bits.json' for p in range(4)} for w in run.WINDOWS})
            positive=task.split('/')[1]=='PM' and row['key']=='key00'
            baseline=run.assess(run.empty_slots('ERASURE'),truth,positive,list(range(6,157)))
            baseline['detection_status']='REJECT_GEOMETRY'
            for reader in ('CDC_M','RANK3'):historical[reader+'/'+task]=baseline
        group['tasks'][task]=entry
    # A missing calibration null H must stay unresolved; other null DETECTs do not stop fixed work.
    group['tasks'][next(iter(group['tasks']))]['families']['L']='missing-H.json'
    run.dump(source/'evaluation-input.json',dict(tasks=historical))
    catalog=tmp_path/'catalog.json';run.dump(catalog,dict(groups=dict(stage1=group)))
    real_read=run.read;value_reads=[];truth_reads=[]
    def guarded(path):
        path=Path(path)
        if path.name=='bits.json':
            marker=real_read(out/'plans_complete.json');assert marker['detection_plans']==312 and marker['read_plans']==276
            value_reads.append(path)
        if path in (source/'protocol.json',source/'evaluation-input.json'):
            assert real_read(out/'blind_complete.json')['decisions']==276;truth_reads.append(path)
        return real_read(path)
    monkeypatch.setattr(run,'read',guarded)
    result=run.run(catalog,out)
    assert result['status']=='COMPLETED_FIXED_CPU_ATTEMPT' and value_reads and len(truth_reads)==2
    evaluation=real_read(out/'evaluation.json');stage1=evaluation['groups']['stage1']
    assert stage1['status']=='POST_RECORDED' and len(stage1['tasks'])==36
    assert stage1['read_totals']['positive']['planned_slots']==192 and stage1['read_totals']['null']['planned_slots']==960
    assert evaluation['detection']['stage1']['null']['DETECT']==59 and evaluation['detection']['stage1']['null']['TECHNICAL_UNRESOLVED']==1
    assert evaluation['checks']['null_detect_zero'] is False and evaluation['candidate_status']=='TECHNICAL_UNRESOLVED'
    assert len(real_read(out/'planned_outputs.json'))==312


def test_adjacent_filter_before_delta_and_no_third_rank_fallback():
    # A far legal L substitution has smaller delta, but M adjacency must win.
    f=families((7,59,112),(6,60,20),(.0001,.01,.001))
    # Construct a geometry-rejected triple with legal far alternative L and
    # legal adjacent M; enumerate only synthetic public path geometry.
    found=None
    for origins in rank.PATHS:
        for window in method.WINDOWS:
            j=list(method.WINDOWS).index(window)
            for shift in (-1,1):
                bad=list(origins);bad[j]+=shift
                if not all(0<=o<=136 for o in bad) or tuple(bad) in rank.PATHS:continue
                for other in range(3):
                    if other==j:continue
                    far=next((o[other] for o in rank.PATHS if all(o[k]==bad[k] for k in range(3) if k!=other) and abs(o[other]-bad[other])>1),None)
                    if far is not None:
                        seconds=[(o+37)%137 for o in bad];seconds[j]=origins[j];seconds[other]=far
                        deltas=[.03]*3;deltas[j]=.02;deltas[other]=.001
                        found=(bad,seconds,deltas,window);break
                if found:break
            if found:break
        if found:break
    assert found
    bad,seconds,deltas,window=found
    record=method.select(families(bad,seconds,deltas))
    assert record['correction']['selected_window']==window
    assert any(r['adjacent'] is False and r['delta']==min(x['delta'] for x in record['correction']['alternatives'] if x['delta'] is not None) for r in record['correction']['alternatives'])
    # A nonadjacent unique second is discarded even if an adjacent third is legal.
    f=families((6,66,120),(40,80,100))
    rows=f['M']['members']
    next(r for r in rows if geometry.state_to_origin(r['phase'],r['source_start'])==67)['S']=.4
    f['M']=old.check_family(dict(status='SCORES_COMPLETE',candidates=rows))
    record=method.select(f)
    assert record['selection']['status']=='REJECT_GEOMETRY'
    assert record['membership_checks']==1


@pytest.mark.parametrize('target,field,bad',[
    ('cell','status','UNAVAILABLE'),('cell','observed_region_id','Rc'),
    ('cell','observed_region_slice',[8,24,40,56]),('cell','channels',[0,3]),
    ('cell','coordinate_space','RGB'),('cell','diagnostic_only',True),
    ('bit','key','key01'),('bit','evidence_id','other/row1/R0'),
    ('bit','planned_count',0),('bit','returned_count',0),('bit','diagnostic_only',True)])
def test_actual_cache_metadata_failure_is_technical(tmp_path,target,field,bad):
    view='opaque';eid=view+'/row1/R0'
    bit=dict(local_bit=0,value=1,count=8,planned_count=8,returned_count=8,key='key00',
             evidence_id=eid,status='AVAILABLE',diagnostic_only=False)
    cell=dict(local_row=1,evidence_id=eid,status='AVAILABLE',observed_region_id='R0',carrier_region_id='R0',
        observed_region_slice=[8,24,8,24],channels=[0,4],coordinate_space='normalized_latent',diagnostic_only=False,bits=[bit])
    data=dict(status='AVAILABLE',kind='E41',key='key00',view_id=view,local_rows=list(range(1,11)),regions=dict(R0=dict(local_cells=[cell])))
    path=tmp_path/'bits.json';run.dump(path,data)
    cache=run.Cache();cache.register(path,'key00',view);cache.enabled=True
    assert cache(eid,'key00',0)['value']==1
    (cell if target=='cell' else bit)[field]=bad
    run.dump(path,data);cache=run.Cache();cache.register(path,'key00',view);cache.enabled=True
    with pytest.raises(ValueError):cache(eid,'key00',0)
