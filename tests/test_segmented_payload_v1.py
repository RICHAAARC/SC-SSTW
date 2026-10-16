import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import numpy as np
import pytest
import torch
from main.tube_state import grow_video_reference as original
from main.tube_state import segmented_payload_v1 as core
from experiments.paper_results_v1 import segmented_payload_v1 as run

pytestmark=pytest.mark.unit


def ideal_latent(path):
    plan=core.segment_plan(path);op=plan['operation'];times=(op['output_frames']-1)//4+1
    spectrum=torch.zeros(1,16,times,40,64)
    coords=original.coordinates(run.KEY);h=[p[0] for p in coords];w=[p[1] for p in coords]
    for r in range(times):
        source_t=min(45,op['anchor']//4+r);g=core.source_segment(source_t)
        values=torch.tensor([2*run.BITS[8*g+j%8]-1 for j in range(240)],dtype=torch.float32)
        for channel in range(4):spectrum[0,channel,r,h,w]=values
    return torch.fft.ifft2(spectrum,dim=(-2,-1),norm='ortho').real.contiguous()


def test_actual_target_has_four_bytes_same_mask_amplitude_and_redundancy():
    z=torch.zeros(original.PUBLIC.latent_shape)
    target,mask=core.build_target(z,run.KEY,run.BITS)
    old,old_mask=original.build_target(z,run.KEY,run.BITS)
    assert torch.equal(mask,old_mask) and int(mask.sum())==44160
    assert torch.all(target[mask].abs()==.5) and not torch.equal(target,old)
    coords=original.coordinates(run.KEY)
    for t in (0,1,11,12,22,23,33,34,44,45):
        g=core.source_segment(t)
        for channel in range(4):
            for j,(h,w) in enumerate(coords[:8]):
                assert target[0,channel,t,h,w].item()==(2*run.BITS[8*g+j]-1)*.5
    assert 11*4*30==44*30==1320
    assert 44*4*240==42240


@pytest.mark.parametrize('view,cells', [('CLEAN',[11,11,11,11]),('CROP_MID',[2,11,9,0]),('DELETE_S1',[11,0,11,11])])
def test_public_mapping_and_actual_missing_votes(view,cells):
    path=run.view_indices(view);plan=core.segment_plan(path)
    assert [s['observed_cells'] for s in plan['segments']]==cells
    read=core.read_segments(ideal_latent(path),run.KEY,path)
    for segment,n in enumerate(cells):
        got=read['decoded_bits'][8*segment:8*segment+8]
        assert got==(run.BITS[8*segment:8*segment+8] if n else [None]*8)
        assert all(b['count']==n*4*30 for b in read['segments'][segment]['bits'])
    if view=='DELETE_S1':
        assert plan['operation']['synthetic_output_indices']
        assert read['segments'][1]['observed_cells']==0
    assert read['signed_votes'].shape==(4,plan['operation']['wan_support'],240)


def test_generation_replaces_target_inside_all_native_steps(monkeypatch):
    from runtime.wan import segmented_payload_v1 as runtime
    class Scheduler:
        step_index=None
        sigmas=[1.]*50
    scheduler=Scheduler();seen=[]
    monkeypatch.setattr(runtime.trajectory,'validate_scheduler',lambda x:None)
    monkeypatch.setattr(runtime,'predict_branches',lambda *args:(torch.zeros_like(args[1]),torch.zeros_like(args[1])))
    def guided(z,c,u,sigma,target,mask,index,count):
        seen.append((index,target,mask));return z,dict(enabled=index>=25)
    def native(s,z,v,index,count,kind):s.step_index=index+1;return z
    monkeypatch.setattr(runtime.method,'guided_velocity',guided)
    monkeypatch.setattr(runtime.trajectory,'native_step',native)
    result,receipt=runtime.run_trajectory(None,torch.zeros(original.PUBLIC.latent_shape),scheduler,None,None,None,run.KEY,run.BITS,lambda *a:None,lambda r:None)
    expected,mask=core.build_target(result,run.KEY,run.BITS)
    assert len(seen)==50 and scheduler.step_index==50
    assert all(torch.equal(item[1],expected) and torch.equal(item[2],mask) for item in seen)
    assert receipt['active_indices']==list(range(25,50)) and receipt['net_bits']==32


def test_failed_interrupted_and_lost_artifacts_never_get_second_attempt(tmp_path):
    store=run.Store(tmp_path);calls=[]
    def fail(item):calls.append('failed');raise OSError('fixture failed call')
    store.once('one','native_decode',fail)
    store=run.Store(tmp_path);store.once('one','native_decode',fail)
    assert calls==['failed'] and store.used('native_decode')==1
    store.make_array('two','native_decode',(2,),lambda:np.ones(2,np.float32))
    Path(store.record('two')['path']).unlink()
    store=run.Store(tmp_path)
    assert store.make_array('two','native_decode',(2,),lambda:pytest.fail('must not recompute')) is None
    store.data['unit_attempts']['unknown']=[dict(status='STARTED',kind='native_decode')];store.save()
    store=run.Store(tmp_path)
    store.once('unknown','native_decode',lambda item:pytest.fail('unknown completion must remain consumed'))
    assert store.data['unit_attempts']['unknown'][0]['status']=='INTERRUPTED'
    attempts=[]
    def load_fail():attempts.append(1);raise OSError('fixture load failed')
    for _ in range(3):store.load('wan_decode_load',load_fail)
    assert len(attempts)==2 and store.used('wan_decode_load')==2


def test_exact_input_alias_never_uses_only_length(tmp_path):
    store=run.Store(tmp_path);row=store.data['rows'][1]
    first=run.physical_input(store,row,core.segment_plan(list(range(37,126))))
    other=dict(row,slot_id=row['slot_id']+'other')
    same=run.physical_input(store,other,core.segment_plan(list(range(37,126))))
    different=run.physical_input(store,dict(row,slot_id='different'),core.segment_plan(list(range(38,127))))
    assert first is same and first is not different
    assert first['output_frames']==different['output_frames']==89


def test_fixed_row_fields_and_partial_optional_method_record(tmp_path):
    store=run.Store(tmp_path)
    store.data['fixed_method_semantics'].pop('read_transform');store.save()
    restored=run.Store(tmp_path)
    assert 'method.read_transform' in restored.data['notes'][-1]['fields']
    restored.data['rows'][0]['planned_bits']=31;restored.save()
    with pytest.raises(ValueError,match='fixed row method differs'):run.Store(tmp_path)


def test_partial_blind_plans_wait_and_final_missing_does_not_use_answers(tmp_path,monkeypatch):
    store=run.Store(tmp_path);cid=run.cases()[-1]['case_id']
    for row in store.data['rows']:
        if row['oracle'] or row['case_id']==cid:continue
        path=run.view_indices(row['view'])
        store.make_json('plan/'+row['slot_id'],'fixture_plan',lambda path=path:dict(status='SUPPORTED',plan=core.segment_plan(path),truth_inputs=False,message_inputs=False))
    run.phase_plan(store)
    assert not run.blind_plans_complete(store)
    # A stale marker alone cannot allow a model or message read.
    store.data['phases']['blind_plans_committed']=dict(at=0)
    class NoModels:
        def load_wan(self):pytest.fail('partial blind plans must wait')
    run.phase_reader(store,NoModels())
    assert store.used('wan_reader_load')==0 and store.used('segment_read')==0
    # Once the upstream physical attempt is irrecoverable, freeze six missing blind slots without selecting.
    store.data['unit_attempts']['load/generation/'+cid]=[dict(kind='generation_load',status='FAILED')];store.save()
    run.phase_plan(store)
    assert run.blind_plans_complete(store) and store.used('selector')==0
    missing=[run.saved_plan(store,'plan/'+r['slot_id']) for r in store.data['rows'] if r['case_id']==cid and not r['oracle']]
    assert all(p['status']=='MISSING' and p['truth_inputs'] is False and p['message_inputs'] is False for p in missing)


@pytest.mark.parametrize('broken', ['terminal','native','writer_y','written_y','recon','margin','candidate','received','received_y'])
def test_consumed_bad_array_stops_unreachable_dependency_setup(tmp_path,monkeypatch,broken):
    monkeypatch.setattr(run,'RGB_SHAPE',(181,2,2,3))
    store=run.Store(tmp_path)
    for case in run.cases():store.data['unit_attempts']['load/generation/'+case['case_id']]=[dict(kind='generation_load',status='FAILED')]
    cid=run.cases()[0]['case_id']
    sequence=['terminal','native','writer_y','written_y','recon','margin','candidate','received','received_y']
    for name in sequence:
        if name=='terminal':shape,dtype=(1,16,46,40,64),np.float32
        elif name in ('writer_y','written_y','received_y'):shape,dtype=(181,4,40,64),np.float32
        else:shape,dtype=run.RGB_SHAPE,np.uint8
        store.make_array(cid+'/'+name,'fixture',shape,lambda shape=shape,dtype=dtype:np.zeros(shape,dtype),dtype)
        if name==broken:
            Path(store.record(cid+'/'+name)['path']).unlink();break
    # Later independent branches (e.g. P after a broken M) are already consumed for this narrow dependency test.
    for name in sequence:
        if not store.started(cid+'/'+name):store.data['unit_attempts'][cid+'/'+name]=[dict(kind='fixture',status='FAILED')]
    for row in store.data['rows']:store.data['unit_attempts']['read/'+row['slot_id']]=[dict(kind='segment_read',status='FAILED')]
    store.save()
    assert not any(run.setup_requirements(store).values())


def test_corrupt_but_valid_json_is_missing_without_model_retry(tmp_path):
    store=run.Store(tmp_path);row=store.data['rows'][0];path=run.view_indices('CLEAN')
    value=core.read_segments(ideal_latent(path),run.KEY,path)
    arrays={key:value.pop(key) for key in ('signed_votes','zero_coefficient_mask')}
    votes=tmp_path/'votes.npz';np.savez_compressed(votes,**arrays)
    value.update(votes_path=str(votes),oracle=True,truth_inputs=True)
    store.make_json('read/'+row['slot_id'],'segment_read',lambda:value)
    assert run.saved_read(store,'read/'+row['slot_id']) is not None
    value['decoded_bits'][0]=1-value['decoded_bits'][0]
    run.write_json(store.record('read/'+row['slot_id'])['path'],value)
    assert run.saved_read(store,'read/'+row['slot_id']) is None
    assert store.used('segment_read')==1
    plan_id='plan/'+row['slot_id']
    store.make_json(plan_id,'fixture_plan',lambda:dict(status='SUPPORTED',plan=core.segment_plan(list(range(89))),truth_inputs=True,message_inputs=False))
    assert run.saved_plan(store,plan_id) is None


def test_durable_unreadable_clean_without_marker_needs_no_reader_model(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'RGB_SHAPE',(181,2,2,3))
    store=run.Store(tmp_path);cid=run.cases()[0]['case_id']
    for case in run.cases():store.data['unit_attempts']['load/generation/'+case['case_id']]=[dict(kind='generation_load',status='FAILED')]
    store.make_array(cid+'/received','fixture',run.RGB_SHAPE,lambda:np.zeros(run.RGB_SHAPE,np.uint8),np.uint8)
    store.make_array(cid+'/received_y','fixture',(181,4,40,64),lambda:np.zeros((181,4,40,64),np.float32))
    row=store.data['rows'][0];value=core.read_segments(torch.zeros(original.PUBLIC.latent_shape),run.KEY,run.view_indices('CLEAN'))
    arrays={key:value.pop(key) for key in ('signed_votes','zero_coefficient_mask')}
    votes=tmp_path/'votes.npz';np.savez_compressed(votes,**arrays)
    value.update(votes_path=str(votes),oracle=True,truth_inputs=True)
    store.make_json('read/'+row['slot_id'],'segment_read',lambda:value)
    assert not store.data['clean_readability']
    assert not any(run.setup_requirements(store).values())


def test_source_load_exhaustion_retains_nine_budget_exhausted_rows(tmp_path):
    store=run.Store(tmp_path);cid=run.cases()[0]['case_id']
    for case in run.cases():store.data['unit_attempts']['load/generation/'+case['case_id']]=[dict(kind='generation_load',status='FAILED')]
    store.block(cid+'/native','native Wan load reserve consumed','BUDGET_EXHAUSTED')
    run.phase_plan(store)
    class NoModels:
        def load_wan(self):pytest.fail('missing RGB must not load reader')
    run.phase_reader(store,NoModels())
    rows=[r for r in store.data['rows'] if r['case_id']==cid]
    assert len(rows)==9 and all(r['status']=='BUDGET_EXHAUSTED' for r in rows)


@pytest.mark.skipif(sys.platform=='win32',reason='Colab process-group behavior is POSIX')
def test_notebook_process_group_interrupt_reaches_phase_worker(tmp_path):
    import os
    import signal
    import time
    pid_file=tmp_path/'worker.pid'
    code="""import sys, subprocess
from experiments.paper_results_v1 import segmented_payload_v1_cli as cli
cli.run_phase=lambda *a,**k:None
cli.PHASES={'fixture':None}
original=subprocess.Popen
def child(command,**kwargs):
    return original([sys.executable,'-c',"import os,time; from pathlib import Path; Path("+repr(sys.argv[1])+").write_text(str(os.getpid())); time.sleep(30)"],**kwargs)
cli.subprocess.Popen=child
sys.argv=[sys.argv[0]]
cli.main()
"""
    # Preserve the pid-file argument inside the closure before CLI parses its own arguments.
    code=code.replace('original=subprocess.Popen','original=subprocess.Popen\npid_target=sys.argv[1]').replace('repr(sys.argv[1])','repr(pid_target)')
    outer=subprocess.Popen([sys.executable,'-c',code,str(pid_file)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+5
        while not pid_file.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert pid_file.exists(),outer.stderr.read().decode() if outer.poll() is not None else 'worker did not start'
        pid=int(pid_file.read_text());os.killpg(outer.pid,signal.SIGTERM);outer.wait(timeout=5)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            status=Path(f'/proc/{pid}/status')
            if not status.exists() or '\nState:\tZ' in status.read_text():break
            time.sleep(.02)
        else:pytest.fail('phase worker survived notebook process-group termination')
    finally:
        if outer.poll() is None:os.killpg(outer.pid,signal.SIGKILL);outer.wait()


def test_clean_gate_blind_first_fixed_rows_and_resume_without_models(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'RGB_SHAPE',(181,2,2,3))
    store=run.Store(tmp_path)
    for index,case in enumerate(run.cases()):
        cid=case['case_id']
        store.make_array(cid+'/received_y','fixture_clock',(181,4,40,64),lambda:np.zeros((181,4,40,64),np.float32))
        if index<2:store.make_array(cid+'/received','fixture_rgb',run.RGB_SHAPE,lambda:np.zeros(run.RGB_SHAPE,np.uint8),np.uint8)
    monkeypatch.setattr(run.temporal,'score_framewise',lambda y,key:dict(signed_projection=np.zeros((len(y),181)),rho=np.ones((len(y),181))))
    monkeypatch.setattr(run,'plan_from_scores',lambda q:dict(status='SUPPORTED',plan=core.segment_plan(run.view_indices('CROP_MID' if len(q)==89 else 'DELETE_S1')),truth_inputs=False,message_inputs=False))
    run.phase_plan(store)
    assert store.used('selector')==16
    class FixtureBackend:
        loads=0
        encodes=0
        reads=0
        def load_wan(self):self.loads+=1;return object()
        def encode_wan(self,model,rgb):
            self.encodes+=1;return np.zeros((1,16,(len(rgb)-1)//4+1,40,64),np.float32)
        def read(self,z,path):
            self.reads+=1
            assert all(run.saved_plan(store,'plan/'+r['slot_id']) is not None for r in store.data['rows'] if not r['oracle'])
            value=ideal_latent(path) if self.reads<=9 else torch.zeros_like(ideal_latent(path))
            return core.read_segments(value,run.KEY,path)
        def close(self,model):pass
    backend=FixtureBackend();run.phase_reader(store,backend);summary=run.report(store)
    assert backend.loads==1 and backend.reads==10
    assert sum(r['status']=='NOT_RUN_SOURCE_UNREADABLE' for r in store.data['rows'])==8
    assert len(store.data['rows'])==36 and summary['planned_bits']==1152
    assert summary['exact_input_aliases']>=2 and backend.encodes<backend.reads
    rows=json.loads((tmp_path/'row_results.json').read_text())
    bad_claim=next(r for r in rows if r['case_id']==run.cases()[0]['case_id'] and r['view']=='DELETE_S1' and r['receiver']=='UNALIGNED')
    assert bad_claim['true_missing_segments']==[1] and bad_claim['false_attribution_bits']==8
    assert bad_claim['eligible_planned_bits']==24
    before=(backend.loads,backend.encodes,backend.reads)
    run.phase_reader(run.Store(tmp_path),backend)
    assert (backend.loads,backend.encodes,backend.reads)==before
    # Lost evidence remains missing, never spending another encode or reader call.
    done=next(r for r in store.data['rows'] if r['view']=='CROP_MID' and r['status']=='READ')
    evidence=run.saved_read(store,'read/'+done['slot_id']);Path(evidence['votes_path']).unlink()
    run.phase_reader(run.Store(tmp_path),backend)
    assert (backend.loads,backend.encodes,backend.reads)==before


def test_delivered_notebook_and_portable_cli(tmp_path,monkeypatch):
    from experiments.paper_results_v1.build_segmented_payload_v1_colab import build,ROOT
    result=build();notebook=json.loads(Path(result['notebook']).read_text())
    assert ''.join(notebook['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    with zipfile.ZipFile(result['companion']) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert '..' not in Path(name).parts and not name.startswith('/')
            assert archive.read(name)==(ROOT/name).read_bytes()
        archive.extractall(tmp_path/'portable')
    command=[sys.executable,'-m','experiments.paper_results_v1.segmented_payload_v1_cli','--output',str(tmp_path/'output'),'--worker','init']
    completed=subprocess.run(command,cwd=tmp_path/'portable',capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    summary=json.loads((tmp_path/'output/summary.json').read_text())
    assert summary['planned_rows']==36 and sum(summary['physical_attempts'].values())==0
    assert summary['setup_requirements']==dict(wan_full=True,wan_vae=True,framewise=True,ffmpeg=True,torch=True)
    import shutil
    import urllib.request
    output=tmp_path/'offline';output.mkdir()
    shutil.copyfile(result['companion'],output/'companion_source.zip')
    def offline(*args,**kwargs):raise OSError('fixture offline')
    monkeypatch.setattr(urllib.request,'urlopen',offline)
    setup=''.join(notebook['cells'][2]['source'])
    setup=setup.replace("Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Segmented-Payload-V1/resumable-segmented-payload-v1')",repr(output))
    setup=setup.replace("Path('/content')",repr(tmp_path))
    namespace={'PosixPath':Path,'WindowsPath':Path}
    exec(compile(setup,'fixture_offline_setup','exec'),namespace)
    assert namespace['NEED']==summary['setup_requirements']
