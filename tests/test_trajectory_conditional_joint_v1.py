"""Synthetic CPU evidence only: no saved-video statistics or model/codec calls."""
import ast,copy,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import video_trajectory_conditional_joint_v1 as m
from main.tube_state import payload_reader
from runtime.wan import video_trajectory_conditional_joint_v1 as rt
from experiments.wan_state_clock import video_trajectory_conditional_joint_v1_run as runner
from experiments.wan_state_clock import video_trajectory_conditional_joint_v1_prepare as prep
pytestmark=pytest.mark.unit
torch.set_num_threads(1)


def global_raw(n,key,b,tie=False):
    x=m.align.missing_sync(n,key,'synthetic fixture');x.update(status='COMPLETE')
    for row in x['candidate_rows']:row.update(status='SCORED',score=1.0 if tie or row['source_offset']==b else 0.0)
    for row in x['local_rows']:row.update(status='SCORED',q=0.0)
    top=x['candidate_offsets'] if tie else [b]
    x['summary'].update(top_offsets=top,canonical_offset=top[0] if len(top)==1 else None,unique=len(top)==1)
    x['counts']['failed_candidates']=0
    return x


def jump_raw(key,b=2,k=None):
    cache=[]
    for r in range(177):
        for d in range(5):
            rho=160.0 if r+d==180 else 40.0
            cache.append(dict(r=r,d=d,status='SCORED',rho=rho,numerator=rho*int(d==b+(k is not None and r>=k))))
    return m.deletion.reduce_cache(cache,key)


def test_native_payload_target_delta_cfg_and_50_steps_equal_old():
    from runtime.wan import video_local_fourier_rm_old8_two_state_v1 as old
    from main.tube_state import video_local_fourier_rm_old8_two_state_v1_control as old_control
    z=torch.randn((1,16,46,40,64),generator=torch.Generator().manual_seed(12))*.01
    bits=m.payload.message_bits('OKOK');target,mask=m.payload.build_target(z,'watermark',bits)
    targets=old_control.build_targets(z,'watermark',bits,'PAYLOAD_MULTI')
    assert torch.equal(target,targets['payload_target']) and torch.equal(mask,targets['payload_mask'])
    assert int(mask.sum())==44160
    class Scheduler:
        def __init__(self):
            self.config=SimpleNamespace(prediction_type='flow_prediction',thresholding=False,lower_order_final=True)
            self.predict_x0=True;self.step_index=None;self.timesteps=torch.arange(50,0,-1);self.sigmas=torch.linspace(1,0,51);self.history=[]
        def step(self,v,t,z,return_dict=False):
            i=0 if self.step_index is None else self.step_index;self.step_index=i+1
            self.history.append(float(v.flatten()[0]));return (z-.02*v,)
    class Pipe:
        def transformer(self,hidden_states,timestep,encoder_hidden_states,**kwargs):
            return (hidden_states*.01+encoder_hidden_states,)
    def run(which):
        calls={};rows=[];s=Scheduler()
        def count(k,done):calls.setdefault(k,[0,0])[int(done)]+=1
        if which=='old':value,receipt=old.run_trajectory(Pipe(),z,s,.02,.01,torch.float32,'PAYLOAD_MULTI','watermark',bits,count,rows.append,diagnostic=None)
        else:value,receipt=rt.run_trajectory(Pipe(),z,s,.02,.01,torch.float32,'watermark',bits,count,rows.append)
        return value,receipt,calls,rows,s
    a,b=run('old'),run('new')
    assert torch.equal(a[0],b[0]) and a[1]['terminal_sha256']==b[1]['terminal_sha256']
    assert a[1]['before_step25']==b[1]['before_step25'] and a[4].history==b[4].history
    assert a[2]==b[2]=={k:[n,n] for k,n in dict(transformer_conditional=50,transformer_unconditional=50,native_step=50,local_control=25,payload_gradient=25).items()}
    assert [r['enabled'] for r in b[3]]==[False]*25+[True]*25
    for x,y in zip(a[3][25:],b[3][25:]):assert x['payload']==y['payload'] and x['payload_delta_l2']==y['payload_delta_l2']


@pytest.mark.parametrize('n',[181,177,89])
def test_real_fft_stable_reader_order_zero_and_counter_tie(n):
    from main.tube_state.video_local_fourier_rm_control import payload_read as old_read
    R=m.align.support(n);z=torch.zeros((1,16,(n-1)//4+1,40,64))
    detail=rt.read_detailed(z,'watermark',n)
    assert detail['original_readout']['votes']==old_read(z,'watermark',R)['votes']
    assert np.sum(detail['zero_coefficient_mask'])==32*30*R
    assert all(x['decoded']==0 for x in detail['bit_rows'])
    # Hermitian real coefficients: first half positive, second half negative.
    spectrum=torch.zeros_like(z,dtype=torch.complex64)
    for h,w in m.payload.coordinates('watermark'):
        for t in range(1,R+1):
            value=1 if t<=R//2 else -1
            spectrum[0,:4,t,h,w]=value;spectrum[0,:4,t,(-h)%40,(-w)%64]=value
    z=torch.fft.ifft2(spectrum,dim=(-2,-1),norm='ortho').real
    detail=rt.read_detailed(z,'watermark',n);old=old_read(z,'watermark',R)
    assert detail['original_readout']['decoded_bits']==old['decoded_bits']==[1]*32
    assert all(x['tie'] and x['first_vote']==1 for x in detail['bit_rows'])
    z=-z;detail=rt.read_detailed(z,'watermark',n)
    assert detail['original_readout']['decoded_bits']==old_read(z,'watermark',R)['decoded_bits']==[0]*32


def test_original_joint_estimator_rejects_inconsistent_cache_and_rows():
    raw=jump_raw('watermark',2,88)
    assert m.estimates(raw,'SINGLE_JUMP',177,'watermark')['joint']['path']==dict(family='H1',b=2,k=88)
    bad=[]
    x=copy.deepcopy(raw);x['frame_cache'][1]=x['frame_cache'][0].copy();bad.append(x)
    x=copy.deepcopy(raw);x['status']='FAILED';bad.append(x)
    x=copy.deepcopy(raw);x['candidate_rows'][1]['score']+=.1;bad.append(x)
    x=copy.deepcopy(raw);x['local_grid'][1]['numerator']+=1;bad.append(x)
    for x in bad:
        assert all(v['status']=='UNRESOLVED' for v in m.estimates(x,'SINGLE_JUMP',177,'watermark').values())
    assert raw['frame_cache'][-1]['rho']==160


def test_two_c_selectors_maps_alias_and_fixed_budget():
    slots=m.logical_slots();assert len(slots)==44
    assert [sum(x[k] for x in slots.values()) for k in ('planned_votes','planned_time_bits','planned_final_bits')]==[1605120,53504,1408]
    plan=m.empty_plan();spec=dict(sha256='synthetic')
    a=slots['global_03/K0/EST_ALIGN'];b=slots['path_00/K0/GLOBAL_ALIGN']
    m.add_slot(plan,'a',a,m.blind_operation(a,{'global':m.align.estimate(global_raw(177,'watermark',1),177,'watermark')}),spec,'watermark')
    e=m.estimates(jump_raw('watermark',2),'SINGLE_JUMP',177,'watermark')
    m.add_slot(plan,'b',b,m.blind_operation(b,e),spec,'watermark')
    assert len(plan['encodes'])==2 and len(plan['reads'])==2 # preserve different native selector results
    m.add_slot(plan,'c',b,m.phase_operation(177,1),spec,'watermark')
    assert plan['logical_slots']['c']['alias_of']=='a'
    for p in range(4):
        op=m.phase_operation(89,p);assert op['received_index_map']==[0]*p+list(range(89-p))
    h=dict(family='H1',b=2,k=88);op=m.deletion.correction(h)
    assert op['received_index_map'][90]==87 and op['inserted_gap_output_index']==90
    assert m.operation_cost(op)['synthetic_outputs']==3
    # Distinct phases and C H0 selections construct the conservative maximum.
    plan=m.empty_plan()
    for lid,row in slots.items():
        key=row['key_label'];n=row['frames'];mode=row['mode']
        p=0 if mode in ('BASELINE','RAW') or n==181 else 1 if key=='K0' else 2
        op=m.phase_operation(n,p)
        if row['protocol']=='SINGLE_JUMP' and mode=='GLOBAL_ALIGN':op=m.deletion.correction(dict(family='H0',b=2 if key=='K0' else 3,k=None))
        if mode=='PATH_ALIGN':op=m.deletion.correction(dict(family='H1',b=1 if key=='K0' else 2,k=70 if key=='K0' else 100))
        if mode=='TRUTH_PATH':op=m.deletion.correction(dict(family='H0' if row['view_id']=='path_00' else 'H1',b=2,k=None if row['view_id']=='path_00' else 88))
        m.add_slot(plan,lid,row,op,dict(sha256=row['input_id']),key)
    assert len(plan['encodes'])<=30 and len(plan['reads'])<=40


class FakeRGB:
    def __init__(self,indices):self.indices=tuple(indices)
    def __len__(self):return len(self.indices)
class FakeInputs:
    @staticmethod
    def read(spec):return FakeRGB(range(181))
    @staticmethod
    def construct(full,indices):return FakeRGB([full.indices[i] for i in indices])
    @staticmethod
    def receipt(rgb):return dict(sha256=hashlib.sha256(bytes(rgb.indices)).hexdigest(),shape=[len(rgb),320,512,3],bytes=len(rgb)*320*512*3,dtype='uint8')
class FakeFW:
    closed=0
    def __init__(self,cfg):pass
    def encode(self,rgb):return rgb
    def score(self,rgb,key,protocol):
        if protocol=='GLOBAL':return global_raw(len(rgb),key,rgb.indices[0])
        return jump_raw(key,2,88 if 90 not in rgb.indices else None)
    def close(self):type(self).closed+=1
class FakeWan:
    encodes=0;reads=0;closed=0
    def __init__(self,cfg):pass
    @staticmethod
    def operate(rgb,indices):return FakeInputs.construct(rgb,indices)
    receipt=staticmethod(FakeInputs.receipt)
    def encode(self,rgb):
        type(self).encodes+=1;return SimpleNamespace(shape=(1,16,(len(rgb)-1)//4+1,40,64),n=len(rgb),indices=rgb.indices)
    @staticmethod
    def cache(z):return z
    @staticmethod
    def restore(z):return z
    def read(self,z,key,n):
        type(self).reads+=1;R=m.align.support(n);bits=m.payload.message_bits('OKOK')
        votes=np.array([[[2*bits[ch*8+(j%8)]-1 for j in range(240)] for t in range(R)] for ch in range(4)],np.int8)
        return m.align.detailed_votes(votes,m.payload.coordinates(key),np.zeros_like(votes),n)
    def close(self):type(self).closed+=1

def fake_source(store):
    return dict(received_sources={'M05':dict(path='synthetic',sha256='fixture',shape=[181,320,512,3])})
def fake_quality(store,*args):
    for k in runner.QUALITY:store.data['quality'][k]=dict(status='MEASURED',fixture_only=True)
    store.save()
def fake_run(path,**kwargs):
    return runner.run(path,source_fn=kwargs.pop('source_fn',fake_source),input_type=FakeInputs,framewise_type=kwargs.pop('framewise_type',FakeFW),wan_type=kwargs.pop('wan_type',FakeWan),quality_fn=fake_quality,**kwargs)


def test_production_runner_fake_success_seals_truth_aliases(tmp_path,monkeypatch):
    events=[];original=runner.companion
    def guard(cfg,name):
        if name=='oracle':
            blind=runner.read(tmp_path/'run/blind_payload_seal.json');assert len(blind['logical_reads'])==40
            assert all(x['status']=='READ' for x in blind['logical_reads'].values())
            events.append('oracle')
        if name=='posthoc':
            oracle=runner.read(tmp_path/'run/oracle_payload_seal.json');assert len(oracle['logical_reads'])==4
            assert all(x['status']=='READ' for x in oracle['logical_reads'].values());events.append('message')
        return original(cfg,name)
    monkeypatch.setattr(runner,'companion',guard)
    result=fake_run(tmp_path/'run')
    assert result['status']=='COMPLETE' and events==['oracle','message']
    assert result['counts']['logical_payload_reads']==44 and result['counts']['logical_votes']==1605120
    assert len(result['quality'])==6 and len(result['sync_reads'])==18
    assert result['calls']['framewise_receiver_encode']==dict(attempted=8,completed=8)
    assert result['calls']['sync_score']==dict(attempted=18,completed=18)
    assert result['counts']['planned_encodes']<=30 and result['counts']['physical_reads']<=40
    assert result['counts']['physical_planned_votes']<=1605120
    assert len(result['mode_differences'])==26 and all(x['status']=='DESCRIPTIVE_POSTSEAL' for x in result['mode_differences'].values())
    assert all(x['final_margin_delta']['count']==32 for x in result['mode_differences'].values())
    assert all(x['status']=='EVALUATED_TRUTH' for x in result['posthoc'].values())
    assert result['posthoc']['path_01/K0/PATH_ALIGN']['k_signed_error']==0
    assert result['posthoc']['path_00/K0/PATH_ALIGN']['k_signed_error'] is None
    assert result['payload_reads']['global_00/K0/EST_ALIGN']['alias_of']=='global_00/K0/BASELINE'
    assert result['payload_reads']['path_00/K0/RAW']['alias_of']=='global_03/K0/BASELINE'
    for name in ('blind_sync_seal','blind_plan_seal','blind_payload_seal','oracle_plan_seal','oracle_payload_seal'):runner.check_seal(result[name])
    blind=runner.read(result['blind_payload_seal']['path'])
    assert all(not x['oracle'] for x in blind['logical_reads'].values())
    assert 'message' not in json.dumps(blind) and 'truth_path' not in json.dumps(blind)
    assert all(x['status']=='READ' for x in result['payload_reads'].values())


def test_valid_tie_not_technical_and_wrong_gap_posthoc(tmp_path):
    class FW(FakeFW):
        def score(self,rgb,key,protocol):
            if protocol=='GLOBAL' and len(rgb)==89:return global_raw(89,key,0,True)
            if protocol=='SINGLE_JUMP' and 90 not in rgb.indices:return jump_raw(key,2,89)
            return super().score(rgb,key,protocol)
    result=fake_run(tmp_path/'tie',framewise_type=FW)
    assert result['posthoc']['global_02/K0/EST_ALIGN']['conditional_interpretation']=='INSUFFICIENT_UNIQUE_PATH'
    d=result['posthoc']['path_01/K0/PATH_ALIGN']
    assert d['k_signed_error']==1 and d['k_absolute_error']==1 and d['bit_errors']==0
    assert d['conditional_interpretation']=='VALID_COUNTEREXAMPLE' # correct bits cannot replace path correctness


@pytest.mark.parametrize('where',['source','framewise','wan'])
def test_interrupt_fixed_all_slots_and_release(tmp_path,where):
    class FW(FakeFW):
        def encode(self,x):raise KeyboardInterrupt('synthetic FW interrupt')
    class Wan(FakeWan):
        def encode(self,x):raise KeyboardInterrupt('synthetic Wan interrupt')
    def source(s):raise KeyboardInterrupt('synthetic source interrupt')
    output=tmp_path/where
    with pytest.raises(KeyboardInterrupt):fake_run(output,source_fn=source if where=='source' else fake_source,framewise_type=FW if where=='framewise' else FakeFW,wan_type=Wan if where=='wan' else FakeWan)
    r=runner.read(output/'result.json')
    assert r['stage']=='FINISHED' and r['status']=='RETAINED_INCOMPLETE'
    assert len(r['payload_reads'])==len(r['posthoc'])==44 and len(r['quality'])==6
    assert r['counts']['logical_time_bit_rows']==53504
    assert all(x['status'] not in ('PENDING','RUNNING','ENCODING') for x in r['payload_reads'].values())
    for x in r['posthoc'].values():assert len(runner.read(x['detail']['path'])['bit_rows'])==32
    assert len(r['sync_reads'])==18
    assert sum(len(runner.read(x['readout']['path'])['candidate_rows']) for x in r['sync_reads'].values())==3426


def test_source_and_worker_failure_retained_no_wan(tmp_path,monkeypatch):
    def fail(store):raise FileNotFoundError('synthetic missing source')
    class Never:
        def __init__(self,cfg):raise AssertionError('must not load')
    result=fake_run(tmp_path/'missing',source_fn=fail,wan_type=Never)
    assert result['counts']['logical_final_bits']==1408 and len(result['quality'])==6
    assert 'wan_vae_load' not in result['calls']
    for phase in ('source','matched'):
        cfg=runner.companion(runner.load_config(),'preparation');out=tmp_path/phase
        def fail_phase(s,c):raise RuntimeError('synthetic '+phase+' failure')
        if phase=='matched':
            x=prep.Store(out,cfg,'source');x.data['status']='SOURCE_READY';x.save()
        with pytest.raises(RuntimeError):prep.run_phase(out,phase,cfg,source_fn=fail_phase,matched_fn=fail_phase)
        assert runner.read(out/'source_preparation.json')['status']=='FAILED'


def test_failure_cache_not_retried_oracle_and_other_key_can_read(tmp_path):
    cfg=runner.load_config();store=runner.Store(tmp_path/'cache',cfg);oid='input_07';source=FakeRGB(range(177));sources={oid:source}
    rows=m.logical_slots();plan=m.empty_plan();a='path_01/K0/PATH_ALIGN';b='path_01/K0/TRUTH_PATH';c='path_01/K1/TRUTH_PATH';spec=FakeInputs.receipt(source)
    op=m.deletion.correction(dict(family='H1',b=2,k=88))
    m.add_slot(plan,a,rows[a],op,spec,cfg['key']);runner.install_plan(store,plan,{'PATH_ALIGN'})
    class ReadFail(FakeWan):
        def read(self,z,key,n):
            if key==cfg['key']:raise ValueError('cached read failure')
            return super().read(z,key,n)
    model=ReadFail(cfg);cache={};runner.run_plan(store,sources,model,cache,plan,'blind')
    assert store.data['calls']['blind/payload_read']==dict(attempted=1,completed=0)
    m.add_slot(plan,b,rows[b],op,spec,cfg['key']);m.add_slot(plan,c,rows[c],op,spec,cfg['wrong_key']);runner.install_plan(store,plan,{'TRUTH_PATH'})
    runner.run_plan(store,sources,model,cache,plan,'oracle')
    assert 'oracle/wan_receiver_encode' not in store.data['calls']
    assert store.data['calls']['oracle/payload_read']==dict(attempted=1,completed=1)
    failed=store.data['physical_reads'][plan['logical_slots'][a]['physical_read']]
    assert failed['status']=='FAILED' and b in failed['logical_slots']
    # Remember encode failures too, even when oracle needs a different key.
    store=runner.Store(tmp_path/'encode-fail',cfg);cache={};runner.install_plan(store,plan,{'PATH_ALIGN','TRUTH_PATH'})
    class EncodeFail(FakeWan):
        def encode(self,rgb):raise ValueError('cached encode failure')
    runner.run_plan(store,sources,EncodeFail(cfg),cache,plan,'blind');runner.run_plan(store,sources,EncodeFail(cfg),cache,plan,'oracle')
    assert store.data['calls']['blind/wan_receiver_encode']['attempted']==1 and 'oracle/wan_receiver_encode' not in store.data['calls']


def test_matched_preparation_same_original_two_decodes_codec_after_release(tmp_path):
    cfg=runner.companion(runner.load_config(),'preparation');store=prep.Store(tmp_path/'matched',cfg,'source');store.data['source_protocol']={}
    events=[];base=np.zeros((2,3),np.float32)
    class FW:
        def __init__(self,c):events.append('load')
        def encode(self,x):events.append('encode');return base
        def write(self,z,key):z+=1;return z,dict(target_margin=.5,rows=[{}]*7360)
        def decode(self,z):events.append('decode'+str(float(z.sum())));return z.copy()
        def close(self):events.append('close')
    class Media:
        @staticmethod
        def save_raster(x,p):return dict(path=str(p),sha256=hashlib.sha256(x.tobytes()).hexdigest())
        @staticmethod
        def mp4_roundtrip(source,sha,mp4,out,count,event):
            assert events[-1] in ('close','codec');events.append('codec')
            for k in ('mp4_save','mp4_probe','mp4_readback'):count(k,False);count(k,True)
            event('rgb24',dict(status='SAVED',path=str(out),sha256=sha,shape=[181,320,512,3]))
    prep.matched_worker(store,cfg,FW,FakeInputs,Media)
    assert np.array_equal(base,np.zeros_like(base))
    assert events==['load','encode','decode0.0','decode6.0','close','codec','codec']
    assert store.data['calls']['framewise_writer_decode']==dict(attempted=2,completed=2)
    assert store.data['calls']['mp4_save']==dict(attempted=2,completed=2)
    assert set(store.data['received_sources'])=={'P1','M05'}


def test_notebook_parent_sigterm_reaps_isolated_worker_and_grandchild(tmp_path):
    # Same actual runner.worker and SIGTERM handler, with only its child command replaced.
    pidfile=tmp_path/'pids.json';workerlog=tmp_path/'worker.json';script=tmp_path/'parent.py'
    childcode="import os,signal,subprocess,sys,time,json;signal.signal(signal.SIGTERM,signal.SIG_IGN);g=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(120)']);open(sys.argv[1],'w').write(json.dumps([os.getpid(),g.pid]));time.sleep(120)"
    script.write_text("import sys,signal,json\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(runner.ROOT))+")\nfrom experiments.wan_state_clock import video_trajectory_conditional_joint_v1_run as r\noriginal=r.subprocess.Popen\nr.subprocess.Popen=lambda cmd,**kw:original([sys.executable,'-c',"+repr(childcode)+","+repr(str(pidfile))+"],**kw)\nclass Store:\n output=Path("+repr(str(tmp_path))+ ")\n cfg={'_config_path':'unused'}\n data={'workers':{}}\n def save(self):Path("+repr(str(workerlog))+").write_text(json.dumps(self.data))\nsignal.signal(signal.SIGTERM,r.handle_termination)\nr.worker(Store(),'source')\n")
    child=subprocess.Popen([sys.executable,'-B',str(script)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        limit=time.monotonic()+15
        while not pidfile.exists() and time.monotonic()<limit:time.sleep(.05)
        assert pidfile.exists();pids=json.loads(pidfile.read_text());start=time.monotonic()
        os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=5)
        assert time.monotonic()-start<5 and json.loads(workerlog.read_text())['workers']['source']['status']=='FAILED'
        for pid in pids:
            stat=Path('/proc')/str(pid)/'stat'
            assert not stat.exists() or stat.read_text().split()[2]=='Z'
    finally:
        if child.poll() is None:os.killpg(child.pid,signal.SIGKILL);child.wait()
        if pidfile.exists():
            for pid in json.loads(pidfile.read_text()):
                try:os.kill(pid,signal.SIGKILL)
                except ProcessLookupError:pass


def test_actual_dependency_closure_config_identity_and_no_parent_git(tmp_path,monkeypatch):
    from runtime.wan.conditional_joint import provenance
    cfg=runner.load_config();identity=provenance.source_identity(runner.ROOT,'experiments/wan_state_clock/video_trajectory_conditional_joint_v1_run.py',[cfg['_config_path']]+[cfg[n+'_config'] for n in ('preparation','oracle','posthoc')])
    assert len(identity['config_files'])==4
    for name in identity['source_files']:
        assert not any(x in name for x in ('old8','payload_gt','state_run','terminal_sync_path_decision','shim'))
        if name.startswith(('main/','runtime/')):
            tree=ast.parse((runner.ROOT/name).read_text())
            for x in ast.walk(tree):
                if isinstance(x,ast.ImportFrom):assert not (x.module or '').startswith('experiments')
    root=tmp_path/'copy';root.mkdir();(root/'entry.py').write_text('x=1\n');(root/'cfg.json').write_text('{}')
    def no_git(*a,**k):raise AssertionError('must not consult external parent Git')
    monkeypatch.setattr(provenance.subprocess,'check_output',no_git)
    x=provenance.source_identity(root,'entry.py',[root/'cfg.json']);assert x['source_sha'] is None and x['source_clean'] is None
    assert x['config_files'][str(root/'cfg.json')]==hashlib.sha256(b'{}').hexdigest()


def test_notebook_schema_ast_draft_and_published_binding(tmp_path):
    import nbformat
    from scripts import build_video_trajectory_conditional_joint_notebook as b
    for sha in (None,'a'*40):
        p=b.build(sha,tmp_path/('draft.ipynb' if sha is None else 'bound.ipynb'));nb=nbformat.read(p,as_version=4);nbformat.validate(nb)
        code=[c for c in nb.cells if c.cell_type=='code'];assert len(code)==5
        assert code[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
        for c in code:ast.parse(c.source);assert c.execution_count is None and c.outputs==[]
        assert nb.metadata.candidate_binding.source_sha==sha
        assert nb.metadata.candidate_binding.status==('UNPUBLISHED_DRAFT' if sha is None else 'PUBLISHED_SHA_BOUND')
        setup=code[1].source;assert setup.index('SOURCE_SHA is None')<setup.index('mkdir')
        assert 'WanPipeline' in code[2].source and 'sentencepiece' in code[2].source
        assert 'video_trajectory_conditional_joint_v1_run' in code[3].source
        assert "result['config_sha256']!=CONFIG_SHA" in code[4].source
    actual=nbformat.read(b.OUTPUT,as_version=4);nbformat.validate(actual)
    value=actual.metadata.candidate_binding.source_sha
    assert value is None or len(value)==40 and all(x in '0123456789abcdef' for x in value)
