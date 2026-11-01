"""Artificial CPU engineering only: no saved real arrays, media, model or GPU."""
from pathlib import Path
import ast
import importlib.util
import json
import shutil
import subprocess
import sys
import numpy as np
import pytest

from main.tube_state import b_framewise_qim_time_recovery_v1 as hard
from main.tube_state import b_time_layer_diagnostic_v1 as method
from runtime.wan.b_time_layer_diagnostic_v1 import Backend,ResourceExceeded,RawLimitExceeded
from experiments.wan_state_clock import b_time_layer_diagnostic_v1_records as rec
from experiments.wan_state_clock import b_time_layer_diagnostic_v1_evaluation as post
from experiments.wan_state_clock.b_time_layer_diagnostic_v1_experiment import run
from experiments.wan_state_clock.b_conditional_time_output_v1_experiment import attack_mappings

ROOT=Path(__file__).resolve().parents[1]
STEM='b_time_layer_diagnostic_v1'
pytestmark=pytest.mark.quick


def table_fixture():
    # Only this artificial fixture uses a small domain. Production codebook remains92672.
    return [(js,hard.words(js)) for lane in range(5) for js in [np.arange(lane,200,5,dtype=np.uint64)]]


class FixtureBackend(Backend):
    def codebook(self):return table_fixture()


def artificial_raw(mapping,wrong=True):
    q=np.zeros((len(mapping),5,2,63));mask=np.zeros(q.shape,dtype=bool);labels=[]
    for r,n in enumerate(mapping):
        frame=[]
        for lane in range(5):
            j=n-((n-lane)%5)
            if wrong and n==5 and lane==0:j=0
            if j>=0:
                word=int(hard.words([j])[0]);bits=np.array([(word>>k)&1 for k in range(62,-1,-1)])
                q[r,lane]=16*bits;mask[r,lane]=True
            frame.append(dict(lane=lane,labels=[j] if j>=0 else [],erasures=0 if j>=0 else 63,
                              status='COMPATIBLE' if j>=0 else 'NO_COMPATIBLE_CODE'))
        labels.append(frame)
    d0=np.abs(q);d1=np.abs(q-16)
    return dict(q=q,d0=d0,d1=d1,soft=(d0*d0-d1*d1)/256,mask=mask),labels


@pytest.fixture(scope='module')
def artificial_source(tmp_path_factory):
    source=tmp_path_factory.mktemp('ARTIFICIAL-nine-raw-source')
    mappings=attack_mappings()
    for descriptor in rec.units():
        rawpath=source/descriptor['raw_path'];rawpath.parent.mkdir(parents=True,exist_ok=True)
        mapping=mappings[descriptor['id'].split('/')[-1]] if descriptor['kind']=='received' else list(range(181))
        raw,labels=artificial_raw(mapping);np.savez(rawpath,**raw)
        if descriptor['kind']=='received':
            folder=rawpath.parent;h=hard.interval_paths(labels);h.update(labels=labels,comparisons=0)
            fused,known=hard.fuse(raw['soft'],raw['mask']);np.savez(folder/'fused.npz',soft=fused,known=known)
            np.savez(folder/'C.npz',bounds=[row['interval'] for row in h['nodes']],offsets=np.arange(len(mapping)+1))
            local=dict(nodes=[dict(received_frame=r,status='ARTIFICIAL',candidates=[]) for r in range(len(mapping))])
            for name,value in {'blind':h,'C':dict(states=0,R=len(mapping),comparisons=0,graph_abnormal=False),
                               'L':dict(status='ARTIFICIAL',minimum=None),'candidates':local,
                               'semantic':dict(hard=h,L=local),'submission':dict(final_nodes=h['nodes'],final_relations=h['relations'])}.items():
                rec.dump(folder/(name+'.json'),value)
    rec.dump(source/'time_evaluation.json',dict(views={v:dict(nodes=[dict(truth=n) for n in ns]) for v,ns in mappings.items()}))
    return source


def jsonlines(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_fixed_complete_run_truth_last_and_readonly(artificial_source,tmp_path,monkeypatch):
    output=tmp_path/'diagnostic';original=Path.read_text;truth_reads=[]
    before={str(p.relative_to(artificial_source)):p.read_bytes() for p in artificial_source.rglob('*') if p.is_file()}
    def guarded(path,*args,**kwargs):
        if path==artificial_source/'time_evaluation.json':
            terminal=rec.load(output/'blind_terminal.json')
            assert len(terminal['rows'])==65
            assert all(v['status']=='RETURNED' for v in terminal['rows'].values())
            assert rec.load(output/'state.json')['blind_save']['status']=='RETURNED'
            for u in rec.units():assert (output/u['id']/'compatibility_R1.json').is_file()
            truth_reads.append(path)
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    result=run(dict(source_attempt=str(artificial_source)),output,backend_factory=FixtureBackend)
    assert result['evaluation_status']=='DIAGNOSTIC_COMPLETE',result
    assert len(truth_reads)==1
    assert len(rec.plan())==67
    counts=rec.load(output/'counts.json')
    assert {k:v['planned'] for k,v in counts.items()}==dict(raw_load=9,saved_view_load=5,codebook=1,arithmetic=9,fusion=9,compatibility=27,path=5,blind_save=1,post=1)
    assert all(v['planned']==v['started']==v['returned'] for v in counts.values())
    lanes=jsonlines(output/'post/lanes.jsonl');nodes=jsonlines(output/'post/nodes.jsonl');edges=jsonlines(output/'post/edges.jsonl')
    assert (len(lanes),len(nodes),len(edges))==(7730,822,817)
    assert sum(len(x['roles']) for x in lanes)==23190
    assert sum(r['bit_slots'] for x in lanes for r in x['roles'].values())==1460970
    startup=next(x for x in lanes if x['unit']=='received/FULL' and x['received_frame']==0 and x['lane']==4)
    assert startup['expected_j']==-1 and startup['schedule_status']=='UNWRITTEN' and startup['expected_word'] is None
    assert all(r['status']=='UNWRITTEN' and r['e'] is None for r in startup['roles'].values())
    wrong=next(x for x in lanes if x['unit']=='received/FULL' and x['received_frame']==5 and x['lane']==0)
    assert wrong['expected_j']==5 and all(x['actual_hits']==[0] and x['status']=='WRONG_LABEL' for x in wrong['roles'].values())
    assert wrong['replica_comparison']['same_wrong_label']
    summary=rec.load(output/'evaluation_summary.json')
    assert sum(v.get('old_H/false_GAP',0) for v in summary['events'].values())==4
    assert sum(v.get('truth/GAP',0) for v in summary['events'].values())==2
    assert sum(v.get('truth/REPEAT',0) for v in summary['events'].values())==1
    assert all(n['reproduction'] for n in nodes)
    assert all(rec.load(output/u['id']/'path.json')['saved_fusion_comparison']['soft_equal'] for u in rec.units() if u['kind']=='received')
    assert not any((output/u['id']/'path.json').exists() for u in rec.units() if u['kind']=='material')
    assert len(jsonlines(output/'post/first_visible.jsonl'))==2715
    assert len(jsonlines(output/'post/transitions.jsonl'))==20475
    assert before=={str(p.relative_to(artificial_source)):p.read_bytes() for p in artificial_source.rglob('*') if p.is_file()}
    def forbidden():raise AssertionError('consumed output must not instantiate science')
    recovered=run(dict(source_attempt='/missing'),output,backend_factory=forbidden,evaluator=forbidden)
    assert recovered['computations_this_invocation']==recovered['post_calls_this_invocation']==0
    assert len(recovered['rows'])==67 and len(truth_reads)==1
    assert rec.denominators()['total']['comparisons_upper_bound']==429812736


def test_arithmetic_low_tie_mask_and_replica_rules():
    q=np.tile(np.array([-48.,-32.,-16.,-8.,0.,8.,16.,24.,32.,48.,64.,80.,96.,112.,128.,144.,160.,176.,192.,208.,224.]),30).reshape(1,5,2,63)
    d0=np.minimum(np.mod(q,32),32-np.mod(q,32));d1=np.minimum(np.mod(q-16,32),32-np.mod(q-16,32))
    raw=dict(q=q,d0=d0,d1=d1,soft=(d0*d0-d1*d1)/256,mask=np.ones(q.shape,dtype=bool))
    arrays,summary=method.arithmetic(raw)
    assert all(summary[k]['exact_equal'] for k in ('d0','d1','soft'))
    for role in rec.ROLES:assert not arrays[role+'_known_changed'].any()
    raw['soft']=np.ones(q.shape);raw['soft'][:,:,1]=-1
    traces=method.fusion(raw)
    assert not traces['F']['known'].any() and traces['R0']['known'].all() and traces['R1']['known'].all()
    raw['mask'][:,:,1]=False
    assert method.fusion(raw)['F']['known'].all()
    raw['mask'][:]=False
    assert not method.fusion(raw)['F']['known'].any()
    raw,_=artificial_raw([5])
    for role in rec.ROLES:
        result=method.compatibility(raw,role,table_fixture())
        baseline=hard.compatibility(raw['soft'],method.role_mask(raw['mask'],role),table_fixture())[0]
        assert [[{k:v for k,v in row.items() if k!='hit_distances'} for row in frame] for frame in result['labels']]==baseline['labels']
        assert result['labels'][0][0]['hit_distances']==[dict(j=0,e=0,u=0,two_e_plus_u=0)]


def test_missing_layer_and_empty_compatibility_are_distinct(artificial_source,tmp_path):
    class Missing(FixtureBackend):
        def load_raw(self,path,frames):
            if path.name=='preclip.npz':raise FileNotFoundError('artificial missing preclip')
            return super().load_raw(path,frames)
    output=tmp_path/'incomplete';result=run(dict(source_attempt=str(artificial_source)),output,backend_factory=Missing)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    state=rec.load(output/'state.json');assert len(state)==67
    assert state['material/preclip/raw_load']['status']=='FAILED'
    assert state['material/clip/R1/compatibility']['status']=='RETURNED'
    assert state['post']['returned']==1
    lanes=jsonlines(output/'post/lanes.jsonl');assert len(lanes)==7730
    assert all(x['roles']['F']['actual_hits'] is None for x in lanes if x['unit']=='material/preclip')
    assert any(x['roles']['F']['actual_hits']==[] for x in lanes if x['unit']=='received/FULL')
    first=jsonlines(output/'post/first_visible.jsonl')
    assert next(x for x in first if x['source_frame']==5 and x['lane']==0 and x['role']=='F')['first_visible_deviation']=='UNKNOWN'


def test_global_stop_no_hidden_compute_or_truth(tmp_path):
    calls=[]
    class Stop(FixtureBackend):
        def check(self):raise ResourceExceeded('artificial global limit')
        def load_raw(self,*args):calls.append(args);raise AssertionError('must not load')
    output=tmp_path/'stop';result=run(dict(source_attempt='/ARTIFICIAL-NOT-READ'),output,backend_factory=Stop)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE' and not calls
    rows=rec.load(output/'state.json');assert len(rows)==67
    assert rows['codebook']['status']=='RESOURCE_UNRESOLVED' and rows['codebook']['started']==0
    assert rows['post']['started']==0 and rows['blind_save']['started']==0
    assert not (output/'blind_terminal.json').exists()
    assert run({},output)['computations_this_invocation']==0


def test_local_timeout_continues_independent_units(artificial_source,tmp_path):
    class Local(FixtureBackend):
        def load_raw(self,path,frames):
            if path.name=='preclip.npz':raise RawLimitExceeded('artificial local raw timeout')
            return super().load_raw(path,frames)
    output=tmp_path/'local';result=run(dict(source_attempt=str(artificial_source)),output,backend_factory=Local)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    state=rec.load(output/'state.json')
    assert state['material/preclip/raw_load']['status']=='RESOURCE_UNRESOLVED'
    assert state['material/h264/R1/compatibility']['status']=='RETURNED'
    assert state['post']['returned']==1


def test_blocking_return_limit_and_invalid_arrays(tmp_path):
    calls=[0]
    def check():
        calls[0]+=1
        if calls[0]>1:raise ResourceExceeded('after blocking return')
    journal=rec.Journal(tmp_path,check)
    with pytest.raises(ResourceExceeded):journal.call('codebook',lambda:None)
    assert journal.rows['codebook']['returned']==1 and journal.rows['codebook']['status']=='RESOURCE_UNRESOLVED'
    assert journal.rows['codebook']['limit_observation']=='AFTER_CALL_RETURN'
    backend=Backend()
    try:
        raw,_=artificial_raw([0]);raw['q'][0,0,0,0]=np.nan;np.savez(tmp_path/'bad.npz',**raw)
        with pytest.raises(ValueError,match='finite'):backend.load_raw(tmp_path/'bad.npz',1)
        with pytest.raises(ValueError,match='geometry'):backend.load_raw(tmp_path/'bad.npz',2)
    finally:backend.close()


def test_exact_edge_endpoints_and_missing_repeat_indices():
    row=dict(output_endpoints=[4,6],delta=[2,2],kind='GAP',output_missing_indices=[5],output_repeated_index=None)
    assert post.edge_report(row,4,6)['status']=='CORRECT'
    assert post.edge_report(row,5,7)['status']=='WRONG'
    assert post.edge_report(row,5,7)['false_GAP']
    assert post.edge_report(dict(row,output_missing_indices=[99]),4,6)['status']=='WRONG'
    assert post.edge_report(row,5,6)['false_GAP']
    repeat=dict(output_endpoints=[44,44],delta=0,kind='REPEAT',output_missing_indices=None,output_repeated_index=44)
    assert post.edge_report(repeat,44,44)['status']=='CORRECT'
    assert post.edge_report(dict(repeat,output_repeated_index=None),44,44)['status']=='WRONG'


def test_unknown_compatibility_is_not_a_negative_replica_finding():
    raw,_=artificial_raw([5]);unit=dict(fusion=method.fusion(raw))
    report=post.lane_report(unit,0,0,5,table_fixture())
    assert report['replica_comparison']['same_wrong_label'] is None
    assert report['replica_comparison']['fusion_hits_differ_from_R0'] is None


def test_post_save_failure_retains_known_findings_and_no_refill(artificial_source,tmp_path,monkeypatch):
    output=tmp_path/'post-failure';original=rec.dump
    def broken(path,value):
        if Path(path)==output/'evaluation_summary.json':raise OSError('artificial post summary write failure')
        return original(path,value)
    monkeypatch.setattr(rec,'dump',broken)
    result=run(dict(source_attempt=str(artificial_source)),output,backend_factory=FixtureBackend)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    assert rec.load(output/'state.json')['post']['status']=='FAILED'
    progress=rec.load(output/'post_progress.json')
    assert sum(v.get('old_H/false_GAP',0) for v in progress['events'].values())==4
    recovered=run({},output)
    assert recovered['post_calls_this_invocation']==0
    assert recovered['rows']['post']['status']=='FAILED'


def test_malformed_receipts_stay_unknown(tmp_path):
    rec.dump(tmp_path/'attempt.json',{})
    rows=rec.plan();rows['codebook'].update(status='RETURNED',started=0,returned=1)
    rec.dump(tmp_path/'state.json',rows)
    result=rec.recover(tmp_path)
    assert len(result['rows'])==67 and result['rows']['codebook']['status']=='UNRESOLVED_RECEIPT'
    assert result['rows']['codebook']['started'] is None


def test_blind_save_failure_does_not_leak_unsaved_role_to_post(artificial_source,tmp_path,monkeypatch):
    output=tmp_path/'blind-write-failure';original=rec.dump
    def broken(path,value):
        if Path(path)==output/'received/FULL/compatibility_R0.json':raise OSError('artificial blind role write failure')
        return original(path,value)
    monkeypatch.setattr(rec,'dump',broken)
    result=run(dict(source_attempt=str(artificial_source)),output,backend_factory=FixtureBackend)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    rows=rec.load(output/'state.json')
    assert rows['received/FULL/R0/compatibility']['status']=='FAILED'
    assert rows['received/FULL/R1/compatibility']['status']=='RETURNED'
    assert rows['post']['returned']==1
    wrong=next(x for x in jsonlines(output/'post/lanes.jsonl') if x['unit']=='received/FULL' and x['received_frame']==5 and x['lane']==0)
    assert wrong['roles']['R0']['actual_hits'] is None
    assert wrong['roles']['R1']['actual_hits']==[0]
    assert wrong['replica_comparison']['same_wrong_label'] is None


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_builder_and_consumed_notebook_no_source_or_science(tmp_path,monkeypatch):
    builder=load_script('build_'+STEM+'_notebook');builder.build(tmp_path)
    assert (tmp_path/'notebooks'/(STEM+'.ipynb')).read_text()==(ROOT/'notebooks'/(STEM+'.ipynb')).read_text()
    nb=json.loads((ROOT/'notebooks'/(STEM+'.ipynb')).read_text())
    assert len(nb['cells'])==4 and ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in nb['cells'] if c['cell_type']=='code')
    output=tmp_path/'consumed';output.mkdir();rec.dump(output/'attempt.json',dict(plan=rec.plan(),fixed_slots=rec.skeleton()))
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('consumed notebook downloaded source'))
    namespace=dict(OUTPUT=str(output),SOURCE_DIRECTORY='/missing',SOURCE_ATTEMPT='/missing',SOURCE_REF='missing')
    exec(builder.RUN,namespace)
    assert namespace['result']['computations_this_invocation']==namespace['result']['post_calls_this_invocation']==0
    assert len(namespace['result']['rows'])==67
    assert all(row['status']=='UNRESOLVED_RECEIPT' for row in namespace['result']['rows'].values())


def test_entry_exact_defaults_and_no_git_cli_recovery(tmp_path,monkeypatch):
    entry=load_script(STEM+'_entry');builder=load_script('build_'+STEM+'_notebook')
    settings={};exec(builder.SETTINGS,settings)
    assert settings['SOURCE_ATTEMPT'].endswith('B-New-Source-Chain-Confirmation-V1/fixed-c02-rerun01-v1')
    assert settings['OUTPUT'].endswith('B-Time-Layer-Diagnostic-V1/fixed-c02-rerun01-v1')
    seen={}
    def launch(source_directory,config_path,output,source_ref,scratch,session_type):
        seen.update(config=rec.load(config_path),session=session_type);return dict(status='ARTIFICIAL_ENTRY_ROUTED')
    monkeypatch.setattr(entry.cpu,'launch',launch)
    output=tmp_path/'output'
    assert entry.launch('/ARTIFICIAL-SOURCE',output,ROOT)['status']=='ARTIFICIAL_ENTRY_ROUTED'
    assert seen['config']==dict(source_attempt='/ARTIFICIAL-SOURCE',output=str(output))
    assert seen['session'].stem==STEM
    copy=tmp_path/'editable-no-git';copy.mkdir()
    for relative in ('experiments/wan_state_clock/'+STEM+'_run.py','experiments/wan_state_clock/'+STEM+'_records.py','experiments/wan_state_clock/b_minimal_l_v1_records.py'):
        target=copy/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    output.mkdir();rec.dump(output/'attempt.json',dict(status='CONSUMED'))
    guard=copy/'sitecustomize.py';guard.write_text("import sys\nclass Block:\n def find_spec(self,name,*args):\n  if name.split('.')[0] in ('numpy','torch','runtime','main'): raise RuntimeError('science forbidden '+name)\nsys.meta_path.insert(0,Block())\n")
    import os
    environment=dict(os.environ,PYTHONPATH=str(copy))
    result=subprocess.run([sys.executable,'-B',str(copy/'experiments/wan_state_clock'/(STEM+'_run.py')),'--output',str(output),'--config','/missing'],cwd=copy,env=environment,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    value=json.loads(result.stdout);assert value['computations_this_invocation']==value['post_calls_this_invocation']==0
    assert not (copy/'.git').exists()


@pytest.mark.parametrize('mode',['exception','late_error','peak','total_wall'])
def test_v2_resource_close_failures_never_claim_complete(artificial_source,tmp_path,monkeypatch,mode):
    class Close(FixtureBackend):
        def close(self):
            receipt=super().close()
            if mode=='exception':raise OSError('artificial close failure')
            if mode=='late_error':receipt['error']='late watcher error'
            if mode=='peak':receipt['peak_resident_bytes']=rec.LIMITS['rss_bytes']+1
            if mode=='total_wall':
                import time
                now=time.monotonic();monkeypatch.setattr(time,'monotonic',lambda:now+6301)
            return receipt
    output=tmp_path/mode
    result=run(dict(source_attempt=str(artificial_source)),output,backend_factory=Close)
    assert result['evaluation_status']=='EVIDENCE_INCOMPLETE' and result['errors']
    assert rec.load(output/'result.json')['evaluation_status']=='EVIDENCE_INCOMPLETE'
    assert rec.load(output/'resources.json')['completion_errors']==result['errors']
    assert rec.load(output/'state.json')['post']['returned']==1
    assert rec.load(output/'evaluation_summary.json')['events']['received/FULL']['old_H/false_GAP']==1
    assert run({},output)['post_calls_this_invocation']==0


def test_v2_unfinished_known_counts_and_saved_findings_visible(tmp_path,monkeypatch):
    output=tmp_path/'existing';output.mkdir()
    rec.dump(output/'attempt.json',{})
    rows=rec.plan();rows['codebook'].update(status='STARTED',started=1,returned=0)
    rec.dump(output/'state.json',rows)
    progress=dict(status='POST_IN_PROGRESS',known_lane_findings={'received/F/WRONG_LABEL':4},
        node_status={'received/FULL':{'old_H':{'WRONG':1}}},edge_status={'received/FULL':{'old_H':{'WRONG':1}}},
        events={'received/FULL':{'old_H/false_GAP':1}})
    rec.dump(output/'post_progress.json',progress)
    recovered=rec.recover(output)
    assert recovered['rows']['codebook']['status']=='UNFINISHED_RECEIPT'
    assert (recovered['rows']['codebook']['started'],recovered['rows']['codebook']['returned'])==(1,0)
    assert recovered['saved']['post_progress']==progress
    builder=load_script('build_'+STEM+'_notebook')
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('recovery must not download'))
    namespace=dict(OUTPUT=str(output),SOURCE_DIRECTORY=str(ROOT),SOURCE_ATTEMPT='/missing',SOURCE_REF='missing')
    exec(builder.RUN,namespace)
    assert namespace['result']['rows']==recovered['rows']
    summary=namespace['summary']
    assert summary['computations_this_invocation']==summary['post_calls_this_invocation']==0
    assert summary['receipt_statuses']==dict(UNFINISHED_RECEIPT=1,UNRESOLVED_RECEIPT=66)
    assert summary['saved_diagnostic']['lane_findings']['received/F/WRONG_LABEL']==4
    assert summary['saved_diagnostic']['source']=='post_progress'
    assert summary['saved_diagnostic']['node_counts']['old_H/WRONG']==1
    assert summary['saved_diagnostic']['event_counts']['received/FULL/old_H/false_GAP']==1
    assert 'evaluation_summary' in summary['read_errors']
    (output/'attempt.json').write_text('{broken')
    namespace['SOURCE_DIRECTORY']=None
    exec(builder.RUN,namespace)
    assert len(namespace['result']['rows'])==67 and len(namespace['result']['fixed_slots'])==9
    slots=namespace['result']['fixed_slots']
    assert sum(v['positions'] for v in slots.values())==1546
    assert sum(v['edges'] for v in slots.values())==817
    assert sum(r['bit_slots'] for v in slots.values() for r in v['roles'].values())==1460970
    assert 'attempt' in namespace['summary']['read_errors']
    assert namespace['result']['rows']['codebook']['status']=='UNFINISHED_RECEIPT'
    assert namespace['summary']['unfinished']['codebook']['started']==1
    assert namespace['summary']['saved_diagnostic']['known_lane_findings']['received/F/WRONG_LABEL']==4
    rows['codebook']['kind']='wrong-kind';rec.dump(output/'state.json',rows)
    exec(builder.RUN,namespace)
    assert namespace['result']['rows']['codebook']['status']=='SAVED_RECEIPT_UNVALIDATED'
    assert namespace['result']['rows']['codebook']['saved_receipt']['started']==1
    assert 'codebook' not in namespace['summary']['unfinished']
    rec.dump(output/'result.json',dict(evaluation_status='DIAGNOSTIC_COMPLETE',errors={}))
    rec.dump(output/'resources.json',dict(error=None,completion_errors={'resource_peak':'late known peak failure'}))
    for source_directory in (None,str(ROOT)):
        namespace['SOURCE_DIRECTORY']=source_directory;exec(builder.RUN,namespace)
        assert namespace['summary']['evaluation_status']=='EVIDENCE_INCOMPLETE'
        assert namespace['summary']['recorded_evaluation_status']=='DIAGNOSTIC_COMPLETE'
        assert namespace['summary']['post_calls_this_invocation']==0
    assert namespace['summary']['resources']['completion_errors']['resource_peak']=='late known peak failure'


def test_v2_cached_stdlib_helper_keeps_notebook_thin_and_readonly(tmp_path,monkeypatch):
    output=tmp_path/'existing';output.mkdir();rec.dump(output/'attempt.json',{})
    entry=load_script(STEM+'_entry');cache=output.with_name(output.name+'-entry-cache')
    for name in entry.CACHE_FILES:
        target=cache/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    builder=load_script('build_'+STEM+'_notebook')
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('cached recovery must not download'))
    namespace=dict(OUTPUT=str(output),SOURCE_DIRECTORY=None,SOURCE_ATTEMPT='/missing',SOURCE_REF='missing')
    exec(builder.RUN,namespace)
    assert namespace['result']['status']=='READ_ONLY_EXISTING_DIAGNOSTIC'
    assert namespace['summary']['receipt_statuses']=={'UNRESOLVED_RECEIPT':67}
    assert namespace['summary']['computations_this_invocation']==namespace['summary']['post_calls_this_invocation']==0
    assert '__PLAN_KINDS__' not in builder.RUN and 'aggregate_status' not in builder.RUN
    assert len(builder.RUN)<6000 and len(builder.RUN.splitlines())<60
