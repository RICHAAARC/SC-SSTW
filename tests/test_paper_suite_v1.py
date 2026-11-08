"""Artificial CPU/interface checks only; no models, real media or saved-result replay."""
from collections import Counter
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
from main.tube_state import paper_suite_v1 as p
from runtime.wan import paper_suite_records_v1 as rec


def evidence(value=0,valid=True):
    return dict(status='READY',unit={},input_errors=[],slots=[dict(slot=i,value=value if valid else None,
        valid=valid,conflict=False,technical_missing=False,support_kind='NORMAL' if valid else None) for i in range(32)])


def test_fixed_budget_and_nominal_probability():
    assert len(p.SOURCES)==12 and len(set(p.SOURCES))==12
    assert Counter(r['kind'] for r in p.wan_plan().values())==p.WAN_BUDGET
    assert sum(p.WAN_BUDGET.values())==534
    x=p.nominal_family([evidence() for _ in range(10)])
    assert x['exact_probability']==1/2**32
    x=p.nominal_family([evidence(valid=False) for _ in range(10)])
    assert x['exact_probability']==0
    x=p.nominal_family([evidence()]*9+[None])
    assert x['exact_probability'] is None and x['probability_upper']==2/2**32
    assert p.family_event([False]*4+[None]) is None
    assert p.family_event([True]+[None]*4) is True
    assert p.rate_summary([False]*12)['iid_reference_one_sided95_upper']>.22


def test_random_queries_complete_unknown_and_collision(tmp_path):
    from runtime.wan.paper_query_statistics_v1 import run
    root=tmp_path/'cohort'
    for sid in p.SOURCES:
        folder=root/sid/'attempt01';rec.write(folder/'attempt.json',dict(status='COMPLETED'))
        rec.write(folder/'blind_terminal.json',dict(status='ALL_BLIND_TERMINAL'))
        rec.write(folder/'private/writer.json',dict(bits=[0]*32))
        for i in range(15):
            for key in p.cohort.KEYS:rec.write(folder/'evidence'/f'input{i}'/(key+'.json'),evidence())
    out=tmp_path/'query';draws=[]
    result=run(root,out,lambda n:draws.append(n) or bytes(n))
    assert result['claims_returned']==360 and result['actual_bit_comparisons']==11520
    assert draws==[4]*36
    families=rec.read(out/'random_family.json')
    assert all(f['correct_query_collision'] and f['false_accept_event'] is None for f in families if f['arm']!='OFF')
    assert all(f['false_accept_event'] is True for f in families if f['arm']=='OFF')
    before=(out/'queries.json').read_bytes()
    assert run(root,out,lambda n:(_ for _ in ()).throw(AssertionError('no redraw')))['status']=='READ_ONLY'
    assert (out/'queries.json').read_bytes()==before


def test_pending_inputs_do_not_consume_new_attempt(tmp_path):
    from runtime.wan.paper_query_statistics_v1 import run
    from runtime.wan.paper_wan_comparison_v1 import run_source
    assert run(tmp_path/'missing',tmp_path/'q')['status']=='DEPENDENCY_PENDING'
    assert not (tmp_path/'q').exists()
    assert run_source(p.SOURCES[0],tmp_path/'missing',tmp_path/'w')['status']=='DEPENDENCY_PENDING'
    assert not (tmp_path/'w').exists()


class FakeMedia:
    def __init__(self):self.calls=Counter()
    def probe(self,path,shape,fps=8,**kw):self.calls['probe']+=1;assert fps==8;return {}
    def decode(self,path,shape):self.calls['decode']+=1;return np.broadcast_to(np.uint8(0),shape)
    def encode(self,rgb,path,fps=8,**kw):
        self.calls['encode']+=1;assert fps==8
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fake');return {}


class FakeAdapter:
    def __init__(self,name):self.name=name;self.counts=Counter()
    def load(self):self.counts['load']+=1;return {'fake':True}
    def embed_batch(self,rgb,message,*,method,start):self.counts['writer']+=1;return rgb
    def detect(self,rgb):
        self.counts['reader']+=1;n=len(rgb);k=32 if self.name=='wam' else 256
        row=dict(bits=np.zeros((n,k),np.uint8),valid=np.ones(n,bool))
        if self.name=='wam':row.update(selected_count=np.ones(n,np.uint32),center_count=np.ones(n,np.uint32),
            sums=np.zeros((n,32)),means=np.zeros((n,32)),mask_shape=np.array([256,256]))
        else:row['logits']=np.zeros((n,257),np.float32)
        return row
    def close(self):self.counts['closed']+=1
    def resources(self):return {'device':'fake'}


def test_full_fake_wan_shared_off_blind_order_and_consumed_reentry(tmp_path,monkeypatch):
    from runtime.wan import paper_wan_comparison_v1 as runner
    monkeypatch.setattr(runner,'received_shape',lambda n:(n,1,1,3))
    sid=p.SOURCES[0];cohort=tmp_path/'c';output=tmp_path/'w';source=cohort/sid/'attempt01'
    rec.write(source/'result.json',dict(status='COMPLETED'));media=FakeMedia();models=[]
    def factory(name):model=FakeAdapter(name);models.append(model);return model
    monkeypatch.setattr(runner,'quality',lambda a,b:dict(frames=len(a),rgb_mse=0,psnr_db=None,perfect_match=True))
    from main.baselines import public_metrics_v4 as metrics
    actual=metrics.evaluate_view
    def check_post(*args,**kw):
        folder=output/sid/'attempt01'
        assert rec.read(folder/'blind_terminal.json')['status']=='ALL_NATIVE_RAW_SAVED'
        ops=rec.read(folder/'operations.json')
        assert all(r['status']=='RETURNED' for r in ops.values() if r['kind']=='reader')
        return actual(*args,**kw)
    monkeypatch.setattr(metrics,'evaluate_view',check_post)
    result=runner.run_source(sid,cohort,output,adapter_factory=factory,media=media)
    assert result['status']=='COMPLETED',result
    assert result['posts_returned']==40
    ops=rec.read(output/sid/'attempt01/operations.json')
    assert Counter(r['kind'] for r in ops.values() if r['status']=='RETURNED')==p.WAN_BUDGET
    assert media.calls==dict(encode=20,probe=26,decode=26)
    assert sum(m.counts['writer'] for m in models)==49 and sum(m.counts['reader'] for m in models)==350
    assert runner.run_source(sid,cohort,output,adapter_factory=lambda n:(_ for _ in ()).throw(AssertionError('reload')))['status']=='READ_ONLY'


def test_failure_is_retained_with_full_plan(tmp_path):
    from runtime.wan.paper_wan_comparison_v1 import run_source
    sid=p.SOURCES[0];root=tmp_path/'c';rec.write(root/sid/'attempt01/result.json',dict(status='STOPPED_FIXED_ATTEMPT'))
    class Bad(FakeMedia):
        def probe(self,*a,**kw):raise OSError('unreadable old media')
    result=run_source(sid,root,tmp_path/'w',media=Bad())
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    rows=rec.read(tmp_path/'w'/sid/'attempt01/operations.json')
    assert len(rows)==534 and sum(r['started'] for r in rows.values())==1


def test_public_active_runtime_is_not_modified(tmp_path,monkeypatch):
    from runtime.wan import paper_suite_worker_v1 as worker,native_baseline_colab_v4 as public
    monkeypatch.setattr(public,'check_previous_worker',lambda *a:(_ for _ in ()).throw(RuntimeError('active')))
    monkeypatch.setattr(public,'run_colab',lambda **kw:(_ for _ in ()).throw(AssertionError('must not launch')))
    assert worker.run('public')['status']=='EXTERNAL_RUNTIME_ACTIVE'


def test_report_preserves_all_missing_denominators(tmp_path):
    from runtime.wan.paper_suite_tables_v1 import build
    result=build(tmp_path/'c',tmp_path/'w',tmp_path/'q',tmp_path/'public',tmp_path/'reports')
    folder=Path(result['report'])
    assert len(rec.read(folder/'main_fixed_claims.json'))==720
    assert len(rec.read(folder/'wan_native_comparison.json'))==480
    assert len(rec.read(folder/'main_time.json'))==120
    assert all(r['unknown']==12 for r in rec.read(folder/'main_fixed_source_statistics.json'))
    assert rec.read(folder/'twelve_source_summary.json')['source_denominator']==12
    assert len(rec.read(folder/'main_random_claims.json'))==360
    assert len(rec.read(folder/'main_random_family.json'))==36
    stats=rec.read(folder/'main_random_source_statistics.json')
    assert len(stats)==3 and all(r['unknown']==12 for r in stats)
    assert not (tmp_path/'public').exists() and not (tmp_path/'c').exists()


def test_thin_notebook_and_detachable_entries(tmp_path):
    import ast,os,shutil,subprocess,sys,nbformat
    from scripts.build_paper_suite_v1_notebook import build,ROOT
    out=build(tmp_path);assert out.read_bytes()==(ROOT/'notebooks/paper_suite_v1.ipynb').read_bytes()
    nb=nbformat.read(out,as_version=4);nbformat.validate(nb)
    assert nb.cells[0].source.strip()=="from google.colab import drive\ndrive.mount('/content/drive')"
    for cell in nb.cells:
        if cell.cell_type=='code':ast.parse(cell.source);assert not cell.outputs
    source=tmp_path/'source'
    for name in ('main','runtime'):shutil.copytree(ROOT/name,source/name,ignore=shutil.ignore_patterns('__pycache__'))
    for module in ('runtime.wan.paper_suite_worker_v1','runtime.wan.native_baseline_entry_v4'):
        result=subprocess.run([sys.executable,'-m',module,'--help'],cwd=tmp_path,
            env=dict(os.environ,PYTHONPATH=str(source)),capture_output=True,text=True,timeout=30)
        assert result.returncode==0,result.stderr


def test_query_waits_for_terminal_and_retains_terminal_failure(tmp_path):
    from runtime.wan.paper_query_statistics_v1 import run
    root=tmp_path/'c';out=tmp_path/'q'
    for sid in p.SOURCES:rec.write(root/sid/'attempt01/result.json',dict(status='COMPLETED'))
    last=root/p.SOURCES[-1]/'attempt01'
    (last/'result.json').unlink();rec.write(last/'attempt.json',dict(status='RUNNING'))
    assert run(root,out)['status']=='DEPENDENCY_PENDING' and not out.exists()
    rec.write(last/'result.json',dict(status='STOPPED_FIXED_ATTEMPT'))
    result=run(root,out,lambda n:bytes(n))
    assert result['status']=='COMPLETED' and result['claims_returned']==360
    assert all(r['verdict']=='INPUT_UNRESOLVED' for r in rec.read(out/'random_claims.json'))


def test_prepare_cancellation_propagates_and_ordinary_failure_is_distinct(tmp_path,monkeypatch):
    from runtime.wan import b_paper_cohort_colab_v1 as c,paper_suite_worker_v1 as worker
    from main.tube_state import b_paper_cohort_v1 as cp
    from runtime.wan import b_paper_cohort_records_v1 as cr
    monkeypatch.setattr(c,'standalone',lambda path,*args:cp if path.startswith('main/') else cr)
    monkeypatch.setattr(c,'show_results',lambda *a:None)
    actual=c.run_colab
    for error,name,expected in [(KeyboardInterrupt('cancel'),'cancel',130),(OSError('ordinary'),'ordinary',0)]:
        def prepare(*args):raise error
        monkeypatch.setattr(c,'prepare_source',prepare)
        result=actual('TEST-A',source_directory=tmp_path,output_root=tmp_path/name)
        assert result['interrupted']==(expected==130)
        calls=[]
        monkeypatch.setattr(c,'run_colab',lambda batch,**kw:calls.append(batch) or result)
        code=worker.main(['--stage','cohort','--receipt',str(tmp_path/(name+'.json'))])
        assert code==expected and calls==['TEST-A']
    monkeypatch.setattr(c,'run_colab',actual)


def test_na_collisions_and_fixed_partial_query_rows(tmp_path):
    from runtime.wan.paper_suite_reporting_v1 import SavedReader,query_tables
    root=tmp_path/'q';sid=p.SOURCES[0]
    fields=dict(source=sid,arm='NEW',view='FULL',key='key00')
    rec.write(root/'random_claims.json',[dict(fields,verdict='MATCH',comparisons=32)])
    rec.write(root/'random_family.json',[dict(source=sid,arm='NEW',correct_query_collision=True)])
    # A returned saved individual claim remains visible even after an interrupted final table write.
    address=f'{sid}/OFF/FULL/key00'
    rec.write(root/'operations.json',{address:dict(status='RETURNED')})
    rec.write(root/'claims'/sid/'OFF/FULL/key00.json',dict(verdict='MISMATCH',comparisons=32))
    tables=query_tables(root,SavedReader())
    assert len(tables['main_random_claims'])==360 and len(tables['main_random_family'])==36
    new=next(r for r in tables['main_random_source_statistics'] if r['arm']=='NEW')
    assert new['positive']==0 and new['not_applicable']==1 and new['unknown']==11 and new['planned_sources']==12
    assert any(r['arm']=='OFF' and r['verdict']=='MISMATCH' for r in tables['main_random_claims'])
    stat=p.rate_summary([True]+[False]*11,[True]+[False]*11)
    assert stat['positive']==0 and stat['eligible_sources']==11 and stat['unknown']==0


def test_report_bad_source_does_not_hide_good_source_and_exports_saved_contributions(tmp_path):
    from runtime.wan.paper_suite_tables_v1 import build
    root=tmp_path/'c';wan=tmp_path/'w';bad=root/p.SOURCES[0]/'attempt01';good=root/p.SOURCES[1]/'attempt01'
    rec.write(bad/'result.json',[]);rec.write(bad/'decisions.json',[None,5])
    rec.write(bad/'post_status.json',dict(status='RETURNED'))
    rec.write(bad/'post/input10/B_NEW.json',[dict(slot=0,status=[])])
    rec.write(wan/p.SOURCES[1]/'attempt01/resources.json',[dict(backend=name,network_hooks=dict(detect=1)) for name in p.BACKENDS])
    rec.write(good/'result.json',dict(status='COMPLETED'));rec.write(good/'post_status.json',dict(status='RETURNED'))
    rec.write(good/'node_results.json',[dict(arm='NEW',view='FULL',received_frame=0,startup=True,status='CORRECT',original_correct=True)])
    rec.write(good/'edge_results.json',[dict(arm='NEW',view='FULL',received_edge=0,true_kind='NORMAL',status='CORRECT',original_correct=True)])
    rows=[dict(slot=i,value=0,status='CONDITIONAL_READ',literal_correct=True,correct_supported_bounds=[1,1],wrong_attribution=False) for i in range(32)]
    for offset,strategy in [(10,'B_NEW'),(10,'U_NEW'),(5,'P_PAIRED')]:rec.write(good/'post'/f'input{offset}'/(strategy+'.json'),rows)
    for view in p.VIEWS:rec.write(wan/p.SOURCES[1]/'attempt01/post/wam/WM'/(view+'.json'),
        dict(ownership=dict(decision=True),wrong_query=dict(decision=True),wrong_query_collision=True))
    result=build(root,wan,tmp_path/'q',tmp_path/'public',tmp_path/'reports');folder=Path(result['report'])
    assert any(r['source']==p.SOURCES[1] and r['status']=='COMPLETED' for r in result['components'])
    assert any('result.json' in r['path'] and 'list' in r['read_error'] for r in rec.read(folder/'read_errors.json'))
    assert any('slot 0: nonempty status' in r['read_error'] for r in rec.read(folder/'read_errors.json'))
    costs=next(r for r in rec.read(folder/'costs.json') if r['method']=='wan_baselines' and r['source']==p.SOURCES[1])
    assert len(costs['actual_network_hooks'])==3 and costs['actual_network_hooks'][0]['network_hooks']['detect']==1
    row=next(r for r in rec.read(folder/'main_payload_contributions.json') if r['source']==p.SOURCES[1] and r['view']=='FULL')
    assert all(v['correct_supported']==32 and v['complete_literal_message'] is True for v in row['absolute'].values())
    stat=next(r for r in rec.read(folder/'wan_source_statistics.json') if r['method']=='wam' and r['verification']=='wrong_query')
    assert stat['positive']==0 and stat['not_applicable']==1 and stat['unknown']==11 and stat['family']=='ANY5'
    strata=rec.read(folder/'main_time_strata.json')
    assert any(r['source']==p.SOURCES[1] and r['stratum'] is True and r['stratum_positions']==1 for r in strata)
