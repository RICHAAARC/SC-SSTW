import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from experiments.paper_results_v1 import distinct_confirmation_p2 as run
from experiments.paper_results_v1 import distinct_confirmation_p2_diagnostics as diag
from experiments.paper_results_v1.short_reference_v1 import public_templates, observed_scores

pytestmark=pytest.mark.unit


def votes_read(path,bad=False):
    plan=run.method.segment_plan(path);votes=np.full((4,plan['operation']['wan_support'],240),-1,np.int8)
    for segment in plan['segments']:
        for cell in segment['cells']:
            for slot in range(240):
                bit=run.BITS[8*segment['segment']+slot%8]
                if bad:bit=1-bit
                votes[:,cell['receiver_wan_time']-1,slot]=2*bit-1
    return run.method.decode_signed_votes(plan,votes,np.zeros_like(votes))


def test_fixed_roster_views_geometry_and_budget():
    assert [c['case_id'] for c in run.cases()]==[f'distinct_confirm_20261017_{i:02d}' for i in range(1,5)]
    assert [c['seed'] for c in run.cases()]==list(range(2026101701,2026101705))
    assert len(run.planned_rows())==196
    assert sum(len(run.view_indices(v)) for v in run.VIEWS)==508
    assert 4*4*508==8128
    present=missing=0
    for r in run.planned_rows():
        plan=run.method.segment_plan(run.view_indices(r['view']))
        n=8*sum(bool(s['cells']) for s in plan['segments']);present+=n;missing+=32-n
    assert (present,missing)==(3968,2304)
    assert run.LIMITS['wan_decode_load']==run.LIMITS['framewise_load']==run.LIMITS['wan_reader_load']==1
    assert run.CALL_LIMITS['framewise_encode_batch']==216 and run.CALL_LIMITS['framewise_decode_batch']==184
    assert run.LIMITS['selector']==128 and run.LIMITS['segment_read']==run.LIMITS['wan_encode']==196


def test_generation_reference_released_before_final_cache_cleanup():
    import weakref
    class Prepared:pass
    class Store:
        def array(self,*a):return None
        def started(self,*a):return False
        def once(self,i,k,f):return f({})
        def make_array(self,i,k,s,f):return f()
    class Fake:
        refs=[];finished=0
        def load_generation(self,case):
            value=Prepared();self.refs.append(weakref.ref(value));return value
        def generate(self,*a):return None
        def close(self,value):
            if value is None:
                assert self.refs[-1]() is None
                self.finished+=1
    fake=Fake();run.phase_generate(Store(),fake)
    assert fake.finished==4


def test_real_N_frame_projection_preserves_formula_and_view_background():
    rng=np.random.default_rng(123);y=rng.normal(size=(181,4,40,64)).astype(np.float32)
    templates=public_templates();full_raw,full_ref=observed_scores(y,templates)
    for view in ('SHORT17','SHORT17_HOLD_EXACT'):
        indices=run.view_indices(view);small=y[indices];raw,ref=observed_scores(small,templates)
        assert raw.shape==ref.shape==(len(indices),181)
        np.testing.assert_allclose(ref,full_ref[indices],atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(raw,full_raw[indices],atol=1e-13,rtol=1e-13)
        value=run.score_observation(small,templates)
        reps=value['representative_indices']
        assert len(reps)==17
        np.testing.assert_array_equal(value['background'],np.median(value['REF_U'][reps],axis=0))
        assert len(value['DISTINCT_REF_U'])==len(indices)
    with pytest.raises(ValueError):observed_scores(np.zeros((0,4,40,64),np.float32),templates)


def test_rowmax_ties_nonmonotone_and_exact_classes():
    q=np.zeros((3,181));assert diag.rowmax_diagnostic(q)['status']=='UNRESOLVED_ROW_TIE'
    q[0,4]=q[1,2]=q[2,6]=1
    value=diag.rowmax_diagnostic(q)
    assert value['status']=='ROWMAX_NONMONOTONE' and value['path']==[4,2,6]
    a=np.array([[1.,2.],[1.,2.],[np.nextafter(1.,2.),2.]])
    value=diag.equality_deltas(a)
    assert value['exact_class_count']==2 and value['class_sizes']==[2,1] and value['pairwise_max_abs']>0


def received_fixtures(store,monkeypatch):
    monkeypatch.setattr(run,'RGB_SHAPE',(181,2,2,3))
    for case in run.cases():
        cid=case['case_id'];y=np.zeros((181,4,40,64),np.float32);y[:,0,0,0]=np.arange(181)
        store.make_array(cid+'/received_y','fixture',y.shape,lambda y=y:y)
        rgb=np.broadcast_to(np.arange(181,dtype=np.uint8)[:,None,None,None],run.RGB_SHAPE).copy()
        store.make_array(cid+'/received','fixture',rgb.shape,lambda rgb=rgb:rgb,np.uint8)
        for view in run.MEDIA_VIEWS:
            indices=run.view_indices(view)
            store.make_array(cid+'/views/'+view+'/received_y','fixture',run.y_shape(view),lambda indices=indices,y=y:y[indices])
            store.make_array(cid+'/views/'+view+'/received','fixture',run.rgb_shape(view),lambda indices=indices,rgb=rgb:rgb[indices],np.uint8)
    monkeypatch.setattr(run,'public_templates',lambda:None)
    def scores(y,templates):
        q=np.zeros((len(y),181),np.float64);q[np.arange(len(y)),y[:,0,0,0].astype(int)]=1
        t=run.distinct_reference_scores(q)
        return dict(**{arm:q.copy() for arm in run.OBS_ARMS},background=t['background'],
                    representative_indices=t['representative_indices'],class_indices=t['class_indices'])
    monkeypatch.setattr(run,'score_observation',scores)


def test_blind_first_candidate_reads_alias_clean_fail_and_resume(tmp_path,monkeypatch):
    store=run.Store(tmp_path);received_fixtures(store,monkeypatch)
    class Fake:
        loads=0;encodes=0;reads=0;cleans=0;calls=[]
        def load_wan(self):self.loads+=1;return object()
        def encode_wan(self,model,rgb):
            self.encodes+=1;return np.zeros((1,16,(len(rgb)-1)//4+1,40,64),np.float32)
        def read(self,z,path):
            assert run.blind_plans_complete(store)
            assert store.used('selector')==128
            self.reads+=1;self.calls.append(path)
            if path==run.view_indices('CLEAN'):self.cleans+=1
            return votes_read(path,bad=self.cleans>1)
        def close(self,model):pass
    backend=Fake()
    run.phase_reader(store,backend);assert backend.loads==backend.reads==0
    run.phase_plan(store);assert store.used('selector')==128
    run.phase_reader(store,backend);summary=run.report(store)
    assert backend.loads==1
    assert sum(r['status']=='BLOCKED_CLEAN' for r in store.data['rows'])==144
    first=run.cases()[0]['case_id']
    for view in run.VIEWS:
        row=next(r for r in store.data['rows'] if r['case_id']==first and r['view']==view and r['receiver']=='DISTINCT_REF_U')
        assert row['status']=='READ' and row['read_alias_of'] is None
        same=next(r for r in store.data['rows'] if r['case_id']==first and r['view']==view and r['receiver']=='ORACLE')
        assert same['read_alias_of']==row['slot_id']
    assert backend.reads==store.used('segment_read')<52 and backend.encodes==store.used('wan_encode')
    assert summary['planned_rows']==196 and summary['planned_bits']==6272 and summary['read_alias_rows']>=32
    before=(backend.loads,backend.encodes,backend.reads,store.used('selector'))
    restored=run.Store(tmp_path);run.phase_plan(restored);run.phase_reader(restored,backend)
    assert before==(backend.loads,backend.encodes,backend.reads,restored.used('selector'))
    run.phase_diagnostics(restored)
    assert len(json.loads((tmp_path/'rowmax_diagnostics.json').read_text()))==32
    assert len(json.loads((tmp_path/'paired_rows.json').read_text()))==160


def test_complete_read_semantics_and_failure_consumption(tmp_path):
    store=run.Store(tmp_path);row=store.data['rows'][1];plan=run.method.segment_plan(run.view_indices(row['view']))
    p=run.physical_input(store,row,plan);a=run.read_spec(store,p,row,plan)
    second=dict(row,slot_id='another_arm');assert run.read_spec(store,p,second,plan) is a
    different=copy.deepcopy(plan);different['segments'][0]['cells']=[]
    assert run.read_spec(store,p,second,different)['unit_id']!=a['unit_id']
    calls=[]
    def fail(item):calls.append(1);raise OSError('fixture read failed')
    store.once(a['unit_id'],'segment_read',fail)
    store.once(run.read_spec(store,p,second,plan)['unit_id'],'segment_read',fail)
    assert calls==[1]
    for _ in range(2):store.load('wan_reader_load',lambda:object())
    assert store.used('wan_reader_load')==1
    store.make_array('done','wan_encode',(1,),lambda:np.ones(1,np.float32))
    Path(store.record('done')['path']).unlink()
    assert run.Store(tmp_path).make_array('done','wan_encode',(1,),lambda:pytest.fail('must not rerun')) is None


def test_framewise_one_load_all_media_actual_short_lengths_and_resume(tmp_path,monkeypatch):
    from main.tube_state import video_trajectory_payload_framewise_sync_v1 as clock
    monkeypatch.setattr(run,'RGB_SHAPE',(181,2,2,3))
    monkeypatch.setattr(clock,'apply_projection_margin',lambda y,key,target_margin:(y,dict(fixture=True)))
    store=run.Store(tmp_path)
    for case in run.cases():
        store.make_array(case['case_id']+'/native','fixture',run.RGB_SHAPE,lambda:np.zeros(run.RGB_SHAPE,np.uint8),np.uint8)
    class Fake:
        loads=0;encoded=[];decoded=[];media=[]
        def load_framewise(self):self.loads+=1;return object()
        def encode_framewise(self,model,rgb,store):
            self.encoded.append(len(rgb))
            for _ in range((len(rgb)+7)//8):store.count('framewise_encode_batch',False);store.count('framewise_encode_batch',True)
            return np.zeros((len(rgb),4,40,64),np.float32)
        def decode_framewise(self,model,y,store):
            self.decoded.append(len(y))
            for _ in range((len(y)+7)//8):store.count('framewise_decode_batch',False);store.count('framewise_decode_batch',True)
            return np.zeros(run.RGB_SHAPE,np.uint8)
        def codec(self,rgb,store,cid):
            self.media.append(len(rgb))
            for kind in ('mp4_save','mp4_probe','mp4_readback'):store.count(kind,False);store.count(kind,True)
            return rgb
        def close(self,model):pass
    fake=Fake();run.phase_framewise(store,fake)
    assert fake.loads==1 and sorted(fake.encoded)==[17]*4+[35]*4+[181]*8
    assert fake.decoded==[181]*8 and sorted(fake.media)==[17]*4+[35]*4+[181]*4
    assert store.data['counts']['framewise_encode_batch']==dict(started=216,returned=216)
    assert store.data['counts']['framewise_decode_batch']==dict(started=184,returned=184)
    run.phase_framewise(run.Store(tmp_path),fake)
    assert fake.loads==1 and len(fake.encoded)==16 and len(fake.media)==12


def test_secondary_codec_fixed_actual_frames_without_media_execution(tmp_path,monkeypatch):
    from runtime.wan import received_clip_media as media
    rgb=np.zeros((17,320,512,3),np.uint8);calls=[]
    def call(args,input=None,**kwargs):
        calls.append(args)
        if args[0]=='ffprobe':output=json.dumps({'streams':[{'height':320,'width':512,'nb_frames':'17'}]}).encode()
        elif input is not None:Path(args[-1]).write_bytes(b'fixture');output=b''
        else:output=rgb.tobytes()
        return type('Result',(),dict(returncode=0,stdout=output,stderr=b''))()
    monkeypatch.setattr(media.subprocess,'run',call)
    value=media.roundtrip(rgb,tmp_path,count=lambda *a:None,event=lambda *a:None)
    assert value.shape==rgb.shape and calls[0][calls[0].index('-crf')+1]=='18'
    assert len(calls)==3 and calls[0][calls[0].index('-r')+1]=='8'


def test_portable_notebook_and_actual_runtime_imports(tmp_path):
    from experiments.paper_results_v1.build_distinct_confirmation_p2_colab import build,ROOT
    result=build();book=json.loads(Path(result['notebook']).read_text())
    assert ''.join(book['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert run.OUTPUT_DEFAULT in ''.join(book['cells'][2]['source'])
    for cell in book['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']));assert not cell['outputs']
    with zipfile.ZipFile(result['companion']) as archive:
        assert 'experiments/paper_results_v1/distinct_segmented_p1.py' not in archive.namelist()
        assert all(not n.startswith(('docs/','diagnostics/')) for n in archive.namelist())
        for name in archive.namelist():assert archive.read(name)==(ROOT/name).read_bytes()
        archive.extractall(tmp_path/'portable')
    imports='import experiments.paper_results_v1.distinct_confirmation_p2_cli; import experiments.paper_results_v1.real_backends; import runtime.wan.segmented_payload_v1; import runtime.wan.received_clip_media'
    probe=subprocess.run([sys.executable,'-c',imports],cwd=tmp_path/'portable',capture_output=True,text=True)
    assert probe.returncode==0,probe.stderr
    command=[sys.executable,'-m','experiments.paper_results_v1.distinct_confirmation_p2_cli','--output',str(tmp_path/'output'),'--worker','init']
    completed=subprocess.run(command,cwd=tmp_path/'portable',capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    summary=json.loads((tmp_path/'output/summary.json').read_text())
    assert summary['planned_rows']==196 and sum(summary['physical_attempts'].values())==0
    assert all(summary['setup_requirements'].values())
