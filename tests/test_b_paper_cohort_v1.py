"""Artificial-only engineering checks. No actual media, model, codec or formal attempt."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from main.tube_state import b_paper_cohort_v1 as p
from main.tube_state.b_joint_time_relative_validation_v1 import construction_maps
from runtime.wan import b_paper_cohort_records_v1 as rec
from runtime.wan import b_paper_cohort_worker_v1 as work
from runtime.wan import b_paper_cohort_runner_v1 as runner
from runtime.wan import b_paper_cohort_evaluation_v1 as post


class FakeGeneration:
    def __init__(self,config,output,scratch,event,dump):
        self.config,self.output,self.scratch,self.event,self.dump=config,output,scratch,event,dump
    def generation_load(self): return {'fake':True}
    def generate_arm(self,arm):
        for kind in ('transformer_conditional','transformer_unconditional','native_step'):
            for _ in range(50): self.event(kind,False);self.event(kind,True)
        if arm=='P0':
            for _ in range(25): self.event('payload_gradient',False);self.event('payload_gradient',True)
            self.dump(self.output/'private/writer.json',self.config['writer'])
        (self.scratch/(arm+'.fixture')).write_text('artificial only')
        return {'arm':arm}
    def vae_load(self): return {'fake':True}
    def decode_off(self): self.event('D181',False);self.event('D181',True)
    def capture(self):
        self.event('D181',False)
        for _ in range(46): self.event('baseline_head',False);self.event('baseline_head',True)
        self.event('D181',True)
    def head(self): return None
    def q0(self): return None
    def target(self): return None
    def solve(self):
        for kind in ('K','KT'): self.event(kind,False);self.event(kind,True)
    def predict(self): self.event('K',False);self.event('K',True)
    def handoff(self): return None
    def replay(self):
        for _ in range(46): self.event('head_replay',False);self.event('head_replay',True)
    def close(self): return {'fake':True}


class FakeTime(FakeGeneration):
    def preclip(self,arm): return np.zeros((181,1,1,3),dtype=np.uint8)
    def quantize(self,value): return value
    def encode(self,pixels,path): path.write_text('fake media token, not video')
    def probe(self,path,n): return {'frames':n,'fake':True}
    def decode(self,path,n): return np.zeros((n,1,1,3),dtype=np.uint8)
    def decode_y(self,path,n,probe): return n
    def hard_receive(self,y,folder,uid): return {'nodes':[{} for _ in range(y)]}
    def relative(self,packet,uid):
        item=next(i for i in p.inputs() if i['id']==uid);mapping=construction_maps()[item['view']]
        positive=item['arm']=='NEW';nodes=[];old=[];edges=[]
        for r,n in enumerate(mapping):
            candidates=[n] if positive else []
            nodes.append(dict(received_frame=r,conditional_candidates=candidates,conditional_position=n if positive else None,
                status='UNCHANGED_SINGLETON' if positive else 'UNCHANGED_EMPTY',comparison_performed=False))
            old.append(dict(W=candidates))
        for a,b in zip(mapping,mapping[1:]):
            edges.append(dict(exact_singleton_pair=positive,pairs=[dict(left=a,right=b)] if positive else []))
        return dict(conditional_nodes=nodes,conditional_edges=edges,original_f42=dict(nodes=old,edges=copy.deepcopy(edges)))


class FakeIdentity:
    output=None
    def __init__(self,model):
        assert (self.output/'plans_frozen.json').exists()
        assert not (self.output/'truth_access.json').exists()
    def load(self): return {'fake':True}
    def probe(self,path,n): return {'fake':True}
    def decode(self,path,n):
        index=int(path.parent.name.removeprefix('input'))
        return np.full((n,1,1,3),index,dtype=np.uint8)
    def encode(self,pixels,start,n): return dict(input_index=int(pixels[0,0,0,0]),start=start)
    def project(self,z): return z
    def bits(self,projection,uid,key):
        assert not (self.output/'truth_access.json').exists()
        item=p.inputs()[projection['input_index']];mapping=construction_maps()[item['view']];regions={}
        for region in ('R0','R1'):
            cells=[]
            for row in range(1,11):
                t=mapping[projection['start']+4*row-3];cell=(t+3)//4;group=min(3,max(0,(cell-1)//11))
                eid=f'{uid}/row{row}/{region}'
                bits=[dict(local_bit=b,value=p.query_bits('q0')[group*8+b] ^ (key=='key01'),count=8,
                           status='AVAILABLE',key=key,evidence_id=eid) for b in range(8)]
                cells.append(dict(local_row=row,evidence_id=eid,observed_region_id=region,
                    carrier_region_id=region,status='AVAILABLE',bits=bits))
            regions[region]=dict(status='AVAILABLE',observed_region_id=region,carrier_region_id=region,local_cells=cells)
        return dict(status='AVAILABLE',kind='E41',view_id=uid,key=key,local_rows=list(range(1,11)),regions=regions)
    def close(self): return {'fake':True}


class CohortTests(unittest.TestCase):
    def test_frozen_roster_and_budget(self):
        self.assertEqual(len(p.ROSTER),12);self.assertEqual([len(p.batch_sources(b)) for b in ('TEST-A','TEST-B')],[4,8])
        self.assertEqual([s['seed'] for s in p.ROSTER],list(range(2026110701,2026110713)))
        rows=p.operation_plan();c=Counter(r['kind'] for r in rows.values())
        self.assertEqual((len(rows),len(rows)*4,len(rows)*12),(1114,4456,13368))
        self.assertEqual({k:c[k] for k in ('E41','project','bits','local','aggregate','claim','quality','encode','probe','decode_RGB','decode_Y')},
            dict(E41=180,project=180,bits=360,local=120,aggregate=40,claim=60,quality=15,encode=18,probe=33,decode_RGB=33,decode_Y=15))
        self.assertEqual(len(p.layout()),160);self.assertFalse(any('W_NEW' in k or '/q2' in k and k.endswith('/aggregate') for k in rows))

    def test_full_artificial_chain_order_closure_and_consumed_reentry(self):
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            root=Path(folder);output=root/'case_complete';scratch=root/'scratch';FakeIdentity.output=output
            from main.tube_state import qmatch_v1
            original=qmatch_v1.match_claim
            def claim(*args,**kwargs):
                rows=rec.read(output/'state.json')
                self.assertTrue(all(row['status']=='RETURNED' for row in rows.values() if row['kind']=='aggregate'))
                return original(*args,**kwargs)
            def dispatch(stage,out,workdir):
                with patch.object(qmatch_v1,'match_claim',side_effect=claim):
                    return work.worker(stage,out,workdir,generation_factory=FakeGeneration,time_factory=FakeTime,identity_factory=FakeIdentity)
            result=runner.run_source(p.source_config(p.ROSTER[0]['id']),output,scratch,worker_runner=dispatch)
            self.assertEqual(result['status'],'COMPLETED',result)
            self.assertEqual(result['operations'],{'RETURNED':1114})
            decisions=rec.read(output/'decisions.json');self.assertEqual(len(decisions),60)
            self.assertEqual(Counter(d['verdict'] for d in decisions),dict(UNRESOLVED=40,MATCH=5,MISMATCH=15))
            state=rec.read(output/'state.json')
            plan_end=max(r['returned_at'] for k,r in state.items() if k.endswith('/plan'))
            firstbit=min(r['started_at'] for r in state.values() if r['kind']=='bits')
            final_end=max(r['returned_at'] for r in state.values() if r['kind']=='aggregate')
            firstclaim=min(r['started_at'] for r in state.values() if r['kind']=='claim')
            self.assertTrue((output/'plans_frozen.json').is_file())
            # Wall time may step backwards; the claim wrapper above checks actual saved ordering.
            self.assertLess(rec.read(output/'blind_terminal.json')['saved_at'],rec.read(output/'truth_access.json')['opened_at'])
            summary=rec.read(output/'evaluation_summary.json');self.assertEqual(summary['NEW_node_statuses'],{'CORRECT':822})
            self.assertEqual(summary['NEW_edge_statuses'],{'CORRECT':817})
            self.assertEqual(summary['B_correct_supported_bounds'],[160,160])
            self.assertEqual(summary['B_unknown_slots'],0)
            self.assertEqual(len(rec.read(output/'quality.json')),15)
            self.assertEqual(sum(len(rs) for keys in rec.read(output/'raw_terminal_slots.json')['slots'].values() for rs in keys.values()),57600)
            paired=rec.read(output/'plans/input6.json')['strategies']['P_PAIRED']
            self.assertTrue(all(pl['borrowed_from']=='input11' for pl in paired))
            self.assertFalse(any(scratch.iterdir()))
            before={f.relative_to(output):(f.stat().st_size,f.stat().st_mtime_ns) for f in output.rglob('*') if f.is_file()}
            recovered=runner.run_source({},output,scratch,worker_runner=lambda *a: self.fail('reentry executed'))
            self.assertEqual(recovered['status'],'READ_ONLY')
            after={f.relative_to(output):(f.stat().st_size,f.stat().st_mtime_ns) for f in output.rglob('*') if f.is_file()}
            self.assertEqual(before,after)

    def test_failure_retains_denominators_and_unknown_comparison(self):
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            root=Path(folder);output=root/'case_failure'
            def dispatch(stage,out,scratch):
                if stage=='prepare':
                    journal=rec.Journal(out)
                    with self.assertRaises(RuntimeError): journal.call('generation_load',lambda: (_ for _ in ()).throw(RuntimeError('artificial load failure')))
                    return dict(status='STOPPED_FIXED_ATTEMPT',failure='artificial load failure')
                return work.worker(stage,out,scratch,identity_factory=lambda *a:self.fail('dependency triggered model'))
            result=runner.run_source(p.source_config(p.ROSTER[1]['id']),output,root/'scratch',worker_runner=dispatch)
            self.assertEqual(result['status'],'STOPPED_FIXED_ATTEMPT')
            summary=rec.read(output/'evaluation_summary.json');self.assertEqual(summary['delta_bounds'],[-1,1])
            self.assertEqual(summary['B_unknown_slots'],160);self.assertEqual(summary['U_unknown_slots'],160)
            self.assertEqual(len(rec.read(output/'node_results.json')),2466)
            self.assertEqual(len(rec.read(output/'edge_results.json')),2451)
            state=rec.read(output/'state.json');claim='input0/key00/q0'
            state[claim].update(status='STARTED',started=1);rec.write(output/'state.json',state)
            rec.close_blind(output,'artificial claim interruption');result=rec.finish(output,'artificial interruption')
            self.assertIsNone(result['comparisons']);self.assertEqual(result['consumed_comparison_upper_bound'],32)

    def test_technical_unknown_is_not_zero_or_message_conflict(self):
        erased=dict(value=None,status='ERASURE_OR_REJECT',cause_set=['NO_FULL_CELL'])
        self.assertEqual(post.bounds_for_slot(erased,returned=True,time_ready=True,truth_available=True)[0],[0,0])
        self.assertEqual(post.bounds_for_slot(erased,returned=True,time_ready=False,truth_available=True)[0],[0,1])
        self.assertEqual(post.bounds_for_slot(dict(erased,status='UNRESOLVED'),returned=True,time_ready=True,truth_available=True)[0],[0,1])
        b=[dict(value=1,literal_correct=True,correct_supported=True,correct_supported_bounds=[1,1]) for _ in range(32)]
        u=[dict(value=None,literal_correct=None,correct_supported=None,correct_supported_bounds=[0,1]) for _ in range(32)]
        value=post.compare_rows(b,u);self.assertEqual(value['delta_slot_sum_bounds'],[0,32]);self.assertIsNone(value['B_common_BER'])
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            value=post.cohort_summary(folder,'ALL');self.assertEqual(value['source_denominator'],12)
            self.assertEqual(value['delta']['mean'],[-1,1]);self.assertEqual(value['gain_claim'],'INSUFFICIENT_EVIDENCE')

    def test_cross_query_conflict_is_executed_erasure(self):
        from main.tube_state import b_relative_time_payload_v1 as method,b_time_payload_cdc32_v1 as previous,b_cdc32_aggregation_v1 as cdc
        plans=method.plans('fixture',101,dict(status='READY',mapping=list(range(101))),'B_NEW')
        cache={};receipts={}
        for plan in plans:
            for group in plan['groups']:
                for ref in group['references']:
                    receipts[ref['view_id']]={'status':'RETURNED'}
                    cache[ref['evidence_id']]={'key00':{str(b):dict(value=ref['query']%2,count=8,status='AVAILABLE') for b in range(8)}}
        local=[cdc.read_time_local(previous.attach_receipts(pl,receipts),cache) for pl in plans]
        final=cdc.aggregate_time(local);conflicts=[r for r in final['slots'] if r['status']=='CONFLICT_ERASURE']
        self.assertEqual(len(conflicts),16)
        for row in conflicts:
            self.assertEqual(row['conflict_refs'],[])
            self.assertEqual(post.bounds_for_slot(row,returned=True,time_ready=True,truth_available=True)[0],[0,0])
        self.assertEqual(post.bounds_for_slot(dict(conflicts[0],conflict_refs=[{'local':True}]),returned=True,time_ready=True,truth_available=True)[0],[0,0])

    def test_saved_bad_structure_keeps_fixed_source_summary(self):
        from runtime.wan import b_paper_cohort_colab_v1 as colab
        import contextlib,io
        valid=dict(B_correct_supported_bounds=[160,160],U_correct_supported_bounds=[56,56],delta_bounds=[.65,.65],
            NEW_node_statuses={'CORRECT':822},NEW_q0_key00_MATCH=5,NEW_q0_key00_unknown=0,
            B_wrong_attribution_observed=0,B_literal_errors_observed=0)
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            root=Path(folder)
            for sid in ('case_bad','case_good'):
                path=root/sid/'attempt01';path.mkdir(parents=True)
                rec.write(path/'attempt.json',{'status':'COMPLETED_FIXTURE'});rec.write(path/'post_status.json',{'status':'RETURNED'})
                rec.write(path/'evaluation_summary.json',valid);rec.write(path/'result.json',{'status':'GOOD_FIXTURE'})
            for bad in (None,7,{'status':'missing fields'},dict(valid,delta_bounds=[0,9])):
                rec.write(root/'case_bad/attempt01/evaluation_summary.json',bad)
                rec.write(root/'case_bad/attempt01/result.json',bad)
                buf=io.StringIO()
                with patch.object(colab,'ids',return_value=['case_bad','case_good']),contextlib.redirect_stdout(buf): colab.show_results(output_root=root)
                self.assertIn('GOOD_FIXTURE',buf.getvalue())
                with patch.object(p,'batch_sources',return_value=[{'id':'case_bad'},{'id':'case_good'}]): value=post.cohort_summary(root,'TEST-A')
                self.assertEqual(value['source_denominator'],2);self.assertEqual(value['sources'][0]['delta_bounds'],[-1,1])
                self.assertTrue(value['sources'][0]['read_errors']);self.assertTrue(value['sources'][1]['evaluation_available'])
            rec.write(root/'case_bad/attempt01/attempt.json',None)
            with patch.object(p,'batch_sources',return_value=[{'id':'case_bad'},{'id':'case_good'}]): value=post.cohort_summary(root,'TEST-A')
            self.assertEqual(value['sources'][0]['status'],'CONSUMED_UNKNOWN')

    def test_cancellation_stops_batch_and_colab_but_not_ordinary_failure(self):
        from types import SimpleNamespace
        from runtime.wan import b_paper_cohort_colab_v1 as colab
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            root=Path(folder)
            def cancel(*args): raise KeyboardInterrupt('artificial explicit cancellation')
            canceled=runner.run_source(p.source_config(p.ROSTER[0]['id']),root/'case_cancel',root/'scratch',worker_runner=cancel)
            self.assertTrue(canceled['interrupted']);self.assertTrue(rec.read(root/'case_cancel/result.json')['interrupted'])
            sources=[{'id':'case_one'},{'id':'case_two'}]
            for outcome,expected in [(canceled,1),(dict(canceled,interrupted=False),2)]:
                with patch.object(p,'batch_sources',return_value=sources),patch.object(p,'source_config',return_value={}),patch.object(runner,'run_source',return_value=outcome) as invoke,patch.object(post,'cohort_summary',return_value={}):
                    runner.run_batch('TEST-A',root/'batch',root/'scratch');self.assertEqual(invoke.call_count,expected)
                protocol=SimpleNamespace(source_config=lambda sid:{})
                records=SimpleNamespace(reserve=lambda *a:True,write=lambda *a:None)
                with patch.object(colab,'ids',return_value=['case_one','case_two']),patch.object(colab,'standalone',side_effect=[protocol,records]),patch.object(colab,'prepare_source',return_value=root),patch.object(colab,'show_results'),patch.object(runner,'execute_reserved',return_value=outcome),patch('runtime.wan.b_paper_cohort_resume_v1.run_source',return_value=outcome) as invoke,patch.object(post,'cohort_summary',return_value={}):
                    result=colab.run_colab(output_root=root/'colab',max_sources=2);self.assertEqual(invoke.call_count,expected)
                    self.assertEqual(result['status'],'BATCH_INTERRUPTED' if expected==1 else 'BATCH_RETURNED')

    def test_subprocess_failure_preserves_only_explicit_cancellation(self):
        from runtime.wan import b_worker_process_v1 as r02
        from runtime.wan import b_worker_process_v1 as relative_runner
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            output=Path(folder)/'case_subprocess'
            for stage in ('prepare','identity'):
                for interrupted in (True,False):
                    failure='artificial worker cancellation' if interrupted else 'artificial ordinary failure'
                    rec.write(output/'stages'/stage/'result.json',dict(status='STOPPED_FIXED_ATTEMPT',failure=failure,interrupted=interrupted))
                    with patch.object(r02,'stage_process',return_value=dict(returncode=1)) as prepare,patch.object(relative_runner,'watch',return_value=dict(worker_returncode=1)) as identity:
                        result=runner.subprocess_stage(stage,output,Path(folder)/'scratch')
                    self.assertEqual(result['status'],'STOPPED_FIXED_ATTEMPT')
                    self.assertEqual(result['interrupted'],interrupted)
                    self.assertEqual(result['failure'],failure)
                    self.assertEqual(prepare.call_count,stage=='prepare')
                    self.assertEqual(identity.call_count,stage=='identity')

    def test_worker_teardown_and_post_keep_cancellation(self):
        for point in ('backend_close','finalize_raw','close_blind','evaluator'):
            for error_type in (KeyboardInterrupt,InterruptedError,RuntimeError):
                with self.subTest(point=point,error=error_type.__name__),tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
                    output=Path(folder)/'case_worker';scratch=Path(folder)/'scratch';scratch.mkdir()
                    rec.reserve(output,p.source_config(p.ROSTER[0]['id']))
                    attempt=rec.read(output/'attempt.json');attempt['status']='RUNNING';rec.write(output/'attempt.json',attempt)
                    class ClosingGeneration(FakeGeneration):
                        def close(self): raise error_type('artificial close failure')
                    def evaluate(_):
                        if point=='evaluator': raise error_type('artificial post failure')
                    with patch.object(work,'identity'),patch.object(work,'finalize_raw',side_effect=error_type('artificial raw closure') if point=='finalize_raw' else None),patch.object(rec,'close_blind',side_effect=error_type('artificial blind closure') if point=='close_blind' else None):
                        result=work.worker('solve' if point=='backend_close' else 'identity',output,scratch,
                            generation_factory=ClosingGeneration,time_factory=FakeTime,evaluator=evaluate)
                    self.assertEqual(result['status'],'STOPPED_FIXED_ATTEMPT')
                    self.assertEqual(result['interrupted'],error_type is not RuntimeError)
                    self.assertIn(error_type.__name__,result['failure'])

    def test_scratch_cleanup_cancellation_skips_identity(self):
        for error_type in (KeyboardInterrupt,InterruptedError,RuntimeError):
            with self.subTest(error=error_type.__name__),tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
                root=Path(folder);calls=[]
                def dispatch(stage,*args):
                    calls.append(stage);return dict(status='STOPPED_FIXED_ATTEMPT',failure='artificial prepare failure')
                with patch.object(runner.shutil,'rmtree',side_effect=error_type('artificial cleanup failure')):
                    result=runner.run_source(p.source_config(p.ROSTER[0]['id']),root/'case_cleanup',root/'scratch',worker_runner=dispatch)
                self.assertEqual(calls,['prepare','identity'] if error_type is RuntimeError else ['prepare'])
                self.assertEqual(result['interrupted'],error_type is not RuntimeError)
                self.assertEqual(rec.read(root/'case_cleanup/scratch_release.json')['status'],'FAILED')
                self.assertTrue(rec.read(root/'case_cleanup/result.json')['consumed'])

    def test_generation_cuda_peak_saved_before_capture_reset_and_on_failure(self):
        import torch
        from types import SimpleNamespace
        from runtime.wan.b_paper_cohort_backend_v1 import GenerationBackend,Original
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            root=Path(folder);order=[]
            def load(backend):
                order.append('load')
                wan=SimpleNamespace(context=(None,torch.zeros(1),torch.zeros(1),torch.zeros(1),torch.float32),generate=lambda arm,*a:(torch.zeros(1),{'arm':arm}),release_generation=lambda:None,device='cuda')
                backend.native=SimpleNamespace(wan=wan,close=lambda:{});backend.generation_loaded=True
            def capture_load(backend): order.append('capture');torch.cuda.reset_peak_memory_stats();return {}
            with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'reset_peak_memory_stats',side_effect=lambda:order.append('reset')),patch.object(torch.cuda,'max_memory_allocated',side_effect=lambda:(order.append('peak') or 17)),patch.object(Original,'generation_load',load),patch.object(Original,'vae_load',capture_load):
                backend=GenerationBackend(p.source_config(p.ROSTER[0]['id']),root/'ok',root/'scratch',lambda *a:None,rec.write)
                backend.generation_load();backend.generate_arm('OFF');backend.generate_arm('P0');backend.vae_load();backend.close()
            self.assertEqual(order[:2],['reset','load']);self.assertEqual(order[-3:],['peak','capture','reset'])
            receipt=rec.read(root/'ok/generation_resources.json');self.assertTrue(receipt['generation_complete'])
            self.assertEqual(receipt['measurements'][-1]['peak_cuda_allocated_bytes'],17)
            with patch.object(torch.cuda,'is_available',return_value=True),patch.object(torch.cuda,'reset_peak_memory_stats'),patch.object(torch.cuda,'max_memory_allocated',return_value=9),patch.object(Original,'generation_load',side_effect=RuntimeError('artificial load failure')):
                backend=GenerationBackend({},root/'failed',root/'scratch',lambda *a:None,rec.write)
                with self.assertRaises(RuntimeError): backend.generation_load()
                backend.close()
            receipt=rec.read(root/'failed/generation_resources.json');self.assertFalse(receipt['generation_complete'])
            self.assertEqual(receipt['measurements'][-1]['peak_cuda_allocated_bytes'],9)

    def test_shared_resource_budget_and_original_default(self):
        from runtime.wan.b_framewise_qim_time_recovery_v1 import Resources,ResourceExceeded
        from runtime.wan.b_paper_cohort_backend_v1 import TimeBackend
        old=Resources();new=Resources(receiving_total_seconds=9450)
        try:
            old.receive_seconds=7000;new.receive_seconds=7000
            with self.assertRaises(ResourceExceeded):
                with old.receiving(600): pass
            with new.receiving(600): pass
            new.receive_seconds=9451
            with self.assertRaises(ResourceExceeded):
                with new.receiving(600): pass
        finally: old.close();new.close()
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            adapter=TimeBackend({},Path(folder),Path(folder)/'unused',lambda *a:None,rec.write)
            try:
                self.assertEqual(adapter.original.media_backend().resources.receiving_total_seconds,9450)
                adapter.bch_limit=1
                with self.assertRaisesRegex(RuntimeError,'BCH call ceiling'):
                    adapter.hard_receive(np.zeros((1,320,512),dtype=np.uint8),Path(folder)/'synthetic','fixture')
                self.assertEqual(adapter.internal['fixture']['bch_started'],1)
            finally: adapter.close()

    def test_generation_pair_reuses_context_until_both_return(self):
        import torch
        from runtime.wan.b_paper_cohort_backend_v1 import GenerationBackend
        class Wan:
            def __init__(self): self.context=object();self.calls=[];self.released=False
            def generate(self,arm,writer,event,record):
                self.assert_live();self.calls.append((arm,self.context));return torch.zeros(1),dict(arm=arm)
            def assert_live(self):
                if self.released: raise AssertionError('premature generation release')
            def release_generation(self): self.released=True
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            out=Path(folder);backend=GenerationBackend(p.source_config(p.ROSTER[0]['id']),out,out/'scratch',lambda *a:None,rec.write)
            wan=Wan();backend.native=type('Native',(),{'wan':wan})();backend.generation_loaded=True
            backend.generate_arm('OFF');self.assertFalse(wan.released)
            backend.generate_arm('P0');self.assertTrue(wan.released)
            self.assertIs(wan.calls[0][1],wan.calls[1][1]);self.assertEqual([x[0] for x in wan.calls],['OFF','P0'])

    def test_writer_bch_counter_observes_original_small_schedule(self):
        from types import SimpleNamespace
        from main.tube_state.b_joint_time_packet_a_writer_v1 import schedule
        from runtime.wan.b_paper_cohort_backend_v1 import GenerationBackend
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            out=Path(folder);events=work.Events(out,'solve')
            backend=GenerationBackend({},out,out/'scratch',events,rec.write)
            backend.cpu=SimpleNamespace(target=lambda:schedule(frames=3))
            backend.target()
            self.assertEqual(events.rows['writer_BCH_encode']['started'],12)
            self.assertEqual(events.rows['writer_BCH_encode']['returned'],12)

    def test_consumed_notebook_keeps_other_sources_after_corrupt_result(self):
        import ast
        import contextlib
        import io
        from scripts.build_b_paper_cohort_v1_notebook import build
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            root=Path(folder);book=json.loads(build(root).read_text())
            cell=next(c for c in book['cells'] if c['cell_type']=='code' and 'source_ids =' in ''.join(c['source']))
            tree=ast.parse(''.join(cell['source']))
            branch=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Call) and isinstance(n.test.func,ast.Name) and n.test.func.id=='all')
            for name,text in [('case_bad','{broken'),('case_good','{"status":"COMPLETED_FIXTURE"}')]:
                path=root/name/'attempt01';path.mkdir(parents=True);(path/'result.json').write_text(text)
            (root/'TEST-A_summary.json').write_text('{broken')
            buf=io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(ast.Module(body=branch.body,type_ignores=[]),'fixture-consumed-cell','exec'),
                     dict(OUTPUT_ROOT=root,source_ids=['case_bad','case_good'],BATCH='TEST-A',json=json))
            self.assertIn('SAVED_RESULT_UNREADABLE',buf.getvalue());self.assertIn('COMPLETED_FIXTURE',buf.getvalue())
            self.assertIn('不补后评',buf.getvalue())

    def test_notebook_builder_and_consumed_bootstrap_offline(self):
        import ast
        from scripts.build_b_paper_cohort_v1_notebook import build
        from runtime.wan import b_paper_cohort_colab_v1 as colab
        with tempfile.TemporaryDirectory(prefix='cohort-artificial-') as folder:
            root=Path(folder);book=json.loads(build(root).read_text());cells=[c for c in book['cells'] if c['cell_type']=='code']
            self.assertEqual(''.join(cells[0]['source']),"from google.colab import drive\ndrive.mount('/content/drive')\n")
            for cell in cells: ast.parse(''.join(cell['source']));self.assertEqual(cell['outputs'],[])
            self.assertEqual(book,json.loads((Path(__file__).resolve().parents[1]/'notebooks/b_paper_cohort_v1.ipynb').read_text()))
            saved=root/'case_consumed'/'attempt01';saved.mkdir(parents=True);(saved/'marker').write_text('synthetic consumed')
            with patch.object(colab,'ids',return_value=['case_consumed']),patch.object(colab,'source_text',side_effect=AssertionError('network/source access')):
                self.assertEqual(colab.run_colab(output_root=root)['status'],'READ_ONLY')


if __name__=='__main__': unittest.main()
