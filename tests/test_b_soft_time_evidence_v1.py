"""Artificial CPU fixtures only. Never use real saved raw/fused/media."""
from contextlib import nullcontext
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from main.tube_state import b_soft_time_evidence_v1 as m
from experiments.wan_state_clock import b_soft_time_evidence_v1_records as rec
from experiments.wan_state_clock import b_soft_time_evidence_v1_experiment as exp

pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1];STEM='b_soft_time_evidence_v1'


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


@pytest.fixture(scope='module')
def table():return m.codebook()


def old_graph(bounds,statuses=None):
    statuses=statuses or ['LOCAL_LABEL']*len(bounds)
    return dict(path_status='COMPATIBLE_PATH',qualification='SUPPORTED_RELATION_SET',
                nodes=[dict(received_frame=r,status=statuses[r],interval=list(b),output_index=None) for r,b in enumerate(bounds)],
                relations=[dict(output_endpoints=None,original_sentinel=r) for r in range(len(bounds)-1)],
                labels=[[dict(lane=lane,labels=[]) for lane in range(5)] for _ in bounds])


def custom_cost(bounds,values,states=None):
    return dict(C=np.concatenate([np.asarray(v,dtype=np.float64) for v in values]),offsets=np.array([0]+list(np.cumsum([len(v) for v in values]))),
                bounds=np.asarray(bounds),states=states or ['DEFINED']*len(bounds),support=np.full(len(bounds),315),comparisons=0,R=len(bounds),graph_abnormal=False)


def test_replica_hinge_first_same_support_and_all_candidates(table):
    rng=np.random.default_rng(21);soft=rng.normal(size=(1,5,2,63));mask=rng.random(soft.shape)>.3
    soft[0,0,:,0]=[2.,-2.];mask[0,0,:,0]=True
    soft[0,1,0,1]=np.nan;mask[0,1,0,1]=True
    old=old_graph([[92668,92671]]);graph=m.graph_input(old,1)
    got=m.costs(soft,mask,graph,table)
    expected=[];support=mask[0]&np.isfinite(soft[0])
    for n in range(92668,92672):
        losses=[]
        for lane in range(5):
            j=n-(n-lane)%5;word=int(table[j])
            for k in range(63):
                vals=[max(0.,-(2*((word>>(62-k))&1)-1)*soft[0,lane,p,k]) for p in range(2) if support[lane,p,k]]
                if vals:losses.append(sum(vals)/len(vals))
        expected.append(sum(losses)/len(losses))
    np.testing.assert_allclose(got['C'],expected,atol=2e-15,rtol=2e-15)
    assert got['comparisons']==8 and got['support'][0]==support.any(axis=1).sum()
    # With exact conflicting replicas the hinge mean is positive, while hinge(mean soft)=0.
    s=np.zeros((1,5,2,63));v=np.zeros_like(s,dtype=bool);s[0,0,:,0]=[2,-2];v[0,0,:,0]=True
    c=m.costs(s,v,m.graph_input(old_graph([[5,6]]),1),table)
    np.testing.assert_array_equal(c['C'],[1.,1.])


def test_startup_conflict_no_observation_priority_and_zero(table):
    old=old_graph([[0,4],[4,5],[5,5]],['LOCAL_LABEL','LOCAL_CONFLICT','OBSERVATION_UNSUPPORTED'])
    raw=np.ones((3,5,2,63));mask=np.ones_like(raw,bool);mask[2]=False
    c=m.costs(raw,mask,m.graph_input(old,3),table);local=m.local_rank(c);g=m.global_rank(c,local)
    assert c['states']==['STARTUP_MODEL_UNDEFINED','LOCAL_CONFLICT','NO_SOFT_OBSERVATION']
    assert not c['C'].any()
    summary=m.row_summary(c,local,g)
    assert all(row['G_aligned_model_index'] is None for row in summary['nodes'])
    assert summary['formal_new_determined']==summary['new_relation_outputs']==0


def test_graph_anomaly_and_infeasibility_do_not_get_rescued(table):
    old=old_graph([[4,8],[8,12]],['IMPLEMENTATION_UNRESOLVED','LOCAL_LABEL'])
    c=m.costs(np.ones((2,5,2,63)),np.ones((2,5,2,63),bool),m.graph_input(old,2),table)
    g=m.global_rank(c,m.local_rank(c));assert g['J_star'] is None and not g['aligned'].any()
    old['path_status']='NO_COMPATIBLE_PATH'
    for node in old['nodes']:node['interval']=None
    c=m.costs(np.ones((2,5,2,63)),np.ones((2,5,2,63),bool),m.graph_input(old,2),table)
    assert not len(c['C']) and m.global_rank(c,m.local_rank(c))['J_star'] is None
    with pytest.raises(ValueError):m.graph_input(old_graph([[5,7],[4,8]]),2)


def test_forward_backward_matches_exhaustive_paths():
    bounds=[[4,6],[5,7],[6,8]];values=[[.2,.1,.8],[.7,.2,.5],[.3,.9,.1]]
    c=custom_cost(bounds,values);l=m.local_rank(c);g=m.global_rank(c,l)
    paths=[p for p in itertools.product(*[range(a,b+1) for a,b in bounds]) if all(p[i]<=p[i+1] for i in range(2))]
    scores={p:sum(values[r][n-bounds[r][0]] for r,n in enumerate(p))/3 for p in paths}
    expected=np.array([min(v for p,v in scores.items() if p[r]==n) for r,(a,b) in enumerate(bounds) for n in range(a,b+1)])
    np.testing.assert_allclose(g['Phi'],expected,atol=2e-16,rtol=2e-16)
    assert g['J_star']==pytest.approx(min(scores.values()))
    np.testing.assert_array_equal(g['delta'],g['Phi']-g['J_star'])
    np.testing.assert_array_equal(g['dG_C'],3*g['delta'])


def test_isolated_row_equal_scale_and_coupled_difference():
    isolated=custom_cost([[4,4],[4,6],[6,6]],[[3.],[.2,.1,.4],[7.]])
    l=m.local_rank(isolated);g=m.global_rank(isolated,l)
    np.testing.assert_allclose(g['dG_C'],l['dL'],atol=4e-15)
    # Local chooses (5,4), impossible; global chooses (4,4), with a clear cost gap.
    coupled=custom_cost([[4,5],[4,5]],[[1.,0.],[0.,3.]])
    l=m.local_rank(coupled);g=m.global_rank(coupled,l);summary=m.row_summary(coupled,l,g)
    assert summary['nodes'][0]['L_model_index']==5 and summary['nodes'][0]['G_aligned_model_index']==4
    assert g['attribution'][0]=='CONSTRAINT_RANKING_DIFFERENCE_AT_STORED_C'
    assert g['arithmetic_envelope_C']>0


def test_native_vs_aligned_tolerance_and_micro_residual_kept():
    c=custom_cost([[4,4],[4,5]],[[0.],[0.,1.5e-12]])
    l=m.local_rank(c);g=m.global_rank(c,l)
    assert g['native'].tolist()==[True,True,True] and g['aligned'].tolist()==[True,True,False]
    # Floating operation order can create negative residuals; preserve the exact subtraction.
    c=custom_cost([[4,4]]*3,[[.1],[.2],[.3]])
    l=m.local_rank(c);g=m.global_rank(c,l)
    np.testing.assert_array_equal(g['delta'],g['Phi']-g['J_star'])
    assert g['negative_residual_count']==int((g['delta']<0).sum())
    assert g['negative_residual_count']==1 and float(g['delta'].min())<0
    c=custom_cost([[4,5],[4,5]],[[1.,1.-1e-12],[1.,1.+2e-12]])
    l=m.local_rank(c);g=m.global_rank(c,l)
    for r in range(2):
        if not np.array_equal(l['best'][r*2:r*2+2],g['aligned'][r*2:r*2+2]):assert g['attribution'][r]=='NUMERIC_ATTRIBUTION_UNRESOLVED'


def synthetic_inputs(tmp_path):
    source=tmp_path/'old';out=tmp_path/'new';tasks=[]
    for i,R in enumerate(rec.LENGTHS):
        folder=source/'received'/f'v{i:02}';folder.mkdir(parents=True)
        raw=np.zeros((R,5,2,63));np.savez(folder/'raw.npz',soft=raw,mask=np.ones_like(raw,bool))
        rec.dump(folder/'blind.json',old_graph([[4,6]]*R))
        tasks.append(dict(id=f'v{i:02}',arm='QIM' if i<5 else 'BASE',view=str(i%5),mapping=[5]*R))
    rec.dump(source/'private/attack_truth.json',dict(views=tasks));return source,out


class ArtificialBackend:
    def __init__(self,table,fail=None):self.table=table;self.fail=fail;self.calls=[]
    def codebook(self):self.calls.append('codebook');return self.table
    def view(self):return nullcontext()
    def check(self):pass
    def read_raw(self,path):
        self.calls.append(('raw',path.parent.name))
        if self.fail==path.parent.name:raise ValueError('artificial raw failure')
        with np.load(path,allow_pickle=False) as a:return dict(soft=a['soft'],mask=a['mask'])
    def read_old(self,path):self.calls.append(('old',path.parent.name));return path.read_bytes()
    def C(self,raw,old,table):self.calls.append('C');return m.costs(raw['soft'],raw['mask'],m.graph_input(old,len(raw['soft'])),table)
    def L(self,c):self.calls.append('L');return m.local_rank(c)
    def G(self,c,l):self.calls.append('G');return m.global_rank(c,l)
    def summary(self,c,l,g):return m.row_summary(c,l,g)
    def close(self):return dict(error=None,artificial=True)


def test_fixed_budget_truth_last_old_unchanged_and_readonly_reopen(tmp_path,table,monkeypatch):
    from experiments.wan_state_clock import b_soft_time_evidence_v1_evaluation as ev
    source,out=synthetic_inputs(tmp_path);fake=ArtificialBackend(table);before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()};real_load=ev.load;truth_reads=[]
    def load(path):
        if Path(path).name=='attack_truth.json':
            assert fake.calls.count('G')==10
            assert all(r['status'] not in ('STARTED','NOT_STARTED') for r in rec.load(out/'state.json').values())
            truth_reads.append(path)
        return real_load(path)
    monkeypatch.setattr(ev,'load',load);exp.run(source,out,lambda:fake)
    assert len(truth_reads)==1
    assert all(c['started']==c['returned']==c['planned'] for c in rec.load(out/'counts.json').values())
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    for i in range(10):assert (source/'received'/f'v{i:02}'/'blind.json').read_bytes()==(out/'views'/f'v{i:02}'/'old_blind.json').read_bytes()
    evaluation=rec.load(out/'evaluation.json');assert evaluation['formal_new_determined']==evaluation['new_relations']==0
    assert sum(sum(counts.values()) for counts in evaluation['cohorts']['OLD_UNRESOLVED_NONSTARTUP'].values())==822*3
    all_before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    exp.run('/unavailable',out,lambda:pytest.fail('rerun'))
    assert all_before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}


def test_view_failure_keeps_fixed_slots_and_other_views(tmp_path,table):
    source,out=synthetic_inputs(tmp_path);fake=ArtificialBackend(table,fail='v03');exp.run(source,out,lambda:fake)
    state=rec.load(out/'state.json');counts=rec.load(out/'counts.json')
    assert state['v03/raw_read']['status']=='FAILED' and state['v03/old_blind_read']['status']=='RETURNED'
    assert state['v03/G']['status']=='DEPENDENCY_UNRESOLVED' and state['v09/G']['status']=='RETURNED'
    assert counts['raw_read']['started']==10 and counts['raw_read']['returned']==9 and counts['old_blind_read']['returned']==10
    row=rec.load(out/'evaluation.json')['views'][3]
    assert len(row['nodes'])==182 and all(n['L']=='NOT_EXECUTED' for n in row['nodes'])


def test_late_resource_failure_and_missing_truth_are_visible(tmp_path,table):
    source,out=synthetic_inputs(tmp_path);fake=ArtificialBackend(table)
    fake.close=lambda:dict(error='late artificial RSS failure',peak_resident_bytes=8*1024**3+1)
    # Rename only the synthetic truth fixture to simulate unavailable post-evaluation.
    (source/'private/attack_truth.json').rename(source/'private/artificial_truth_unavailable.json')
    exp.run(source,out,lambda:fake)
    evaluation=rec.load(out/'evaluation.json')
    assert evaluation['status']=='DIAGNOSTIC_WITH_UNRESOLVED_EXECUTION' and evaluation['truth_error']
    assert evaluation['resources']['status']=='RESOURCE_UNRESOLVED'
    assert rec.load(out/'counts.json')['G']['returned']==10


def test_no_git_cli_consumed_no_science_or_outer_modules(tmp_path):
    root=tmp_path/'copy';folder=root/'experiments/wan_state_clock';folder.mkdir(parents=True)
    for relative in (f'{STEM}_run.py',f'{STEM}_records.py','b_framewise_qim_time_recovery_v1_records.py'):
        shutil.copyfile(ROOT/'experiments/wan_state_clock'/relative,folder/relative)
    out=tmp_path/'consumed';rec.dump(out/'attempt.json',{'status':'CONSUMED'})
    script="""import sys,runpy
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','psutil','main','runtime','scripts'):raise RuntimeError('forbidden '+name)
sys.meta_path.insert(0,Block())
runpy.run_module('experiments.wan_state_clock.b_soft_time_evidence_v1_run',run_name='__main__')
"""
    result=subprocess.run([sys.executable,'-B','-c',script,'--output',str(out)],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'READ_ONLY_EXISTING_ATTEMPT' in result.stdout


@pytest.mark.parametrize('cache_mode',['none','entry_only','complete'])
def test_actual_notebook_readonly_and_builder(tmp_path,monkeypatch,cache_mode):
    builder=module(ROOT/'scripts'/f'build_{STEM}_notebook.py','soft_builder');builder.build(tmp_path)
    notebook=json.loads((ROOT/'notebooks'/f'{STEM}.ipynb').read_text())
    assert notebook==json.loads((tmp_path/'notebooks'/f'{STEM}.ipynb').read_text())
    assert 'accelerator' not in notebook['metadata']
    assert ''.join([c for c in notebook['cells'] if c['cell_type']=='code'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    entry=module(ROOT/'scripts'/f'{STEM}_entry.py','soft_entry')
    out=tmp_path/'consumed';rec.dump(out/'attempt.json',[]);rec.dump(out/'result.json',dict(status='VISIBLE'))
    cache=out.with_name(out.name+'-entry-cache')
    if cache_mode!='none':
        for relative in entry.CACHE_FILES if cache_mode=='complete' else entry.CACHE_FILES[:2]:
            p=cache/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,p)
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a,**k:pytest.fail('network'))
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('process'))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    scope=dict(Path=Path,SOURCE_ATTEMPT=tmp_path/'old',OUTPUT=out,SOURCE_DIRECTORY=None,SOURCE_REF='dev/local-joint-state-payload-v1')
    exec(builder.BOOTSTRAP,scope);exec(builder.RUN,scope);exec(builder.DISPLAY,scope)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_preflight_does_not_read_raw_and_guards_aliases(tmp_path,monkeypatch):
    source,out=synthetic_inputs(tmp_path)
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('preflight array read'))
    assert rec.preflight(source,out)['status']=='PREFLIGHT_READY' and not out.exists()
    for suffix in ('','-logs','-entry-cache','-preflight-v1'):
        target=tmp_path/'target';protected=target.with_name(target.name+suffix)
        with pytest.raises(ValueError):rec.validate_paths(protected,target)


def test_full_public_candidate_interval_no_pruning(table):
    soft=np.zeros((1,5,2,63));mask=np.zeros_like(soft,bool)
    soft[0,0,:,0]=[2,-2];mask[0,0,:,0]=True
    cost=m.costs(soft,mask,m.graph_input(old_graph([[4,92671]]),1),table)
    assert cost['comparisons']==92672 and len(cost['C'])==92668
    np.testing.assert_array_equal(cost['C'],np.ones(92668))


def test_corrupt_receipts_keep_unknown_and_continue(tmp_path,capsys):
    from experiments.wan_state_clock import b_soft_time_evidence_v1_report as report
    rec.dump(tmp_path/'attempt.json',[]);rec.dump(tmp_path/'state.json',{'v00/C':dict(status=[])})
    rec.dump(tmp_path/'result.json',dict(status='VISIBLE'))
    value=rec.recover(tmp_path);assert value['rows']['v00/C']['status']=='UNRESOLVED_RECEIPT' and 'attempt' in value['read_errors']
    report.display(tmp_path);assert 'VISIBLE' in capsys.readouterr().out


def test_fresh_no_git_cli_actual_cpu_runtime_without_outer(tmp_path):
    source,out=synthetic_inputs(tmp_path);root=tmp_path/'plain-copy'
    relatives=[f'experiments/wan_state_clock/{STEM}_{s}.py' for s in ('run','experiment','records','evaluation')]
    relatives += ['experiments/wan_state_clock/b_framewise_qim_time_recovery_v1_records.py',f'main/tube_state/{STEM}.py',
                  'main/tube_state/b_framewise_qim_time_recovery_v1.py',f'runtime/wan/{STEM}.py','runtime/wan/b_framewise_qim_time_recovery_v1.py']
    for relative in relatives:
        p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,p)
    result=subprocess.run([sys.executable,'-B','-m','experiments.wan_state_clock.'+STEM+'_run','--source-attempt',str(source),'--output',str(out)],cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    saved=rec.load(out/'result.json');assert saved['errors']=={} and saved['failed_rows']==[]
    assert rec.load(out/'resources.json')['peak_resident_bytes']>0
    assert rec.load(out/'counts.json')['G']['returned']==10


def test_ordinary_default_source_download_only_needed_files(tmp_path,monkeypatch):
    import zipfile
    entry=module(ROOT/'scripts'/f'{STEM}_entry.py','download_entry');archive=tmp_path/'fake.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for relative in entry.REQUIRED:z.writestr('source/'+relative,'# artificial source')
    urls=[]
    def download(url,destination):urls.append(url);shutil.copyfile(archive,destination)
    monkeypatch.setattr(entry.common.urllib.request,'urlretrieve',download)
    root=entry.common.prepare_source(None,'dev/local-joint-state-payload-v1',tmp_path/'scratch',entry.REQUIRED)
    assert urls==['https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/dev/local-joint-state-payload-v1']
    assert all((root/p).is_file() for p in entry.REQUIRED)


def test_nested_old_corruption_stays_per_view_unknown(tmp_path,table):
    source,out=synthetic_inputs(tmp_path)
    for index,key in ((3,'nodes'),(4,'relations'),(7,'labels')):
        path=source/'received'/f'v{index:02}'/'blind.json';value=rec.load(path);value[key][0]=None;rec.dump(path,value)
    fake=ArtificialBackend(table);result=exp.run(source,out,lambda:fake)
    assert 'evaluation' not in result['errors']
    evaluation=rec.load(out/'evaluation.json');assert len(evaluation['views'])==10
    assert evaluation['views'][3]['errors']['old_formal'] and evaluation['views'][4]['errors']['old_formal']
    assert evaluation['views'][7]['errors']['old_labels']
    assert evaluation['old_mechanical_counts']['QIM_frames_unknown']==182+178
    assert evaluation['old_mechanical_counts']['BASE_label_frames_unknown']==180
    assert evaluation['views'][9]['ranking_status']=='RETURNED' and not evaluation['views'][9]['errors']
    assert len(evaluation['views'][3]['nodes'])==182 and all(row['cohort']=='TRUTH_UNRESOLVED' for row in evaluation['views'][3]['nodes'])


def test_same_size_tie_set_change_survives_summary_and_post(tmp_path,table):
    source,out=synthetic_inputs(tmp_path);fake=ArtificialBackend(table)
    def artificial_cost(raw,old,unused):
        R=len(raw['soft'])
        return custom_cost([[4,6],[4,6]]+[[6,6]]*(R-2),[[1.,0.,0.],[0.,1.,2.]]+[[0.]]*(R-2))
    fake.C=artificial_cost
    exp.run(source,out,lambda:fake)
    summary=rec.load(out/'views/v00/blind_ranking.json')['nodes'][0]
    assert summary['L_count']==summary['G_aligned_count']==2
    assert summary['L_status']==summary['G_aligned_status']=='MODEL_TIE'
    assert summary['L_G_aligned_same_set'] is False
    evaluation=rec.load(out/'evaluation.json');row=evaluation['views'][0]['nodes'][0]
    assert row['L']==row['G_aligned']=='MODEL_TIE' and row['L_to_G_aligned']=='TIE_CHANGE'
    assert row['attribution']=='NUMERIC_ATTRIBUTION_UNRESOLVED'
    assert evaluation['transitions']['L_to_G_aligned']['TIE_CHANGE']>=5


@pytest.mark.parametrize('corrupt_local',[False,True])
def test_G_failure_keeps_saved_L_without_reading_old_inputs_or_recompute(tmp_path,table,monkeypatch,corrupt_local):
    from experiments.wan_state_clock import b_soft_time_evidence_v1_evaluation as ev
    source,out=synthetic_inputs(tmp_path);fake=ArtificialBackend(table);original_G=fake.G;calls=0
    def G(c,l):
        nonlocal calls
        calls+=1
        if calls==2:raise ValueError('artificial G-only failure after saved L')
        return original_G(c,l)
    fake.G=G
    def evaluate(unchanged_source,output,rows):
        assert rows['v01/L']['status']=='RETURNED' and rows['v01/G']['status']=='FAILED'
        if corrupt_local:(output/'views/v01/L.npz').write_bytes(b'PK\x03\x04invalid')
        # Existing saved results only: any attempted score or source raw/blind reread is forbidden here.
        for name in ('costs','local_rank','global_rank'):monkeypatch.setattr(m,name,lambda *a,**k:pytest.fail('post recomputation'))
        real_np_load=np.load;real_read=Path.read_bytes;real_load=ev.load
        def guarded_np(path,*a,**k):
            assert not Path(path).is_relative_to(source)
            return real_np_load(path,*a,**k)
        def guarded_bytes(path,*a,**k):
            assert not path.is_relative_to(source/'received')
            return real_read(path,*a,**k)
        def guarded_json(path):
            assert not Path(path).is_relative_to(source/'received')
            return real_load(path)
        monkeypatch.setattr(np,'load',guarded_np);monkeypatch.setattr(Path,'read_bytes',guarded_bytes)
        monkeypatch.setattr(ev,'load',guarded_json)
        return ev.evaluate(unchanged_source,output,rows)
    result=exp.run(source,out,lambda:fake,evaluator=evaluate)
    assert 'evaluation' not in result['errors']
    row=rec.load(out/'evaluation.json')['views'][1]
    assert row['L_receipt_status']=='RETURNED' and row['G_receipt_status']=='FAILED'
    expected='SAVED_RESULT_UNRESOLVED' if corrupt_local else 'MODEL_TIE'
    assert all(n['L']==expected and n['G_aligned']=='FAILED_UNRESOLVED' and n['L_to_G_aligned']=='UNRESOLVED_COMPARISON' for n in row['nodes'])
    assert not rec.load(out/'evaluation.json')['views'][9]['errors']
    assert rec.load(out/'counts.json')['L']['returned']==10 and rec.load(out/'counts.json')['G']['returned']==9
