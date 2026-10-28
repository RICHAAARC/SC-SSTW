"""Synthetic CPU/static checks only. No model download, VAE or real codec call."""
import ast
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from main.tube_state import b_local_vae_code_v1 as code
from main.tube_state import b_local_vae_sync_v1 as sync
from main.tube_state import b_local_vae_head_v1 as head
from runtime.wan import b_local_vae_head_v1 as runtime
from runtime.wan import b_local_vae_head_evaluation_v1 as post

pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'experiments/wan_state_clock/configs/b_local_vae_head_v1.json'
torch.set_num_threads(2)

def ideal(frames,message=None,key=None):
    message=message or [i%2 for i in range(32)];key=key or code.EVAL_KEYS['key00']
    return (2*np.asarray([code.native_message(message,key,int(t)) for t in frames])-1)*20.

def save_logits(path,logits):
    path.parent.mkdir(parents=True,exist_ok=True);np.save(path,logits)
    np.save(path.with_suffix('.pilot.npy'),logits[:,:32]);np.save(path.with_suffix('.payload.npy'),logits[:,32:96])

def test_codebook_minimum_distance_and_bch_independent_library():
    words=np.array([code.pilot_word(t) for t in range(256)])
    assert len(np.unique(words,axis=0))==256
    assert min(np.count_nonzero(words[1:],axis=1))==12
    import bchlib
    independent=bchlib.BCH(5,prim_poly=0x43)
    assert independent.ecc_bits==27 and code.generator().bit_length()==28
    rng=random.Random(93)
    for _ in range(20):
        msg=[rng.randrange(2) for _ in range(32)];coded=code.encode(msg)
        # Library byte padding is low zero padding: 27 parity bits occupy the top27 of4 bytes.
        data=code.bits_to_int(msg).to_bytes(4,'big')
        parity=int.from_bytes(independent.encode(data),'big')>>5
        assert code.bits_to_int(coded[32:])==parity
        for errors in range(6):
            noisy=coded[:];positions=rng.sample(range(59),errors)
            for p in positions:noisy[p]^=1
            decoded=code.decode(noisy)
            assert decoded['message']==msg and sorted(decoded['corrected_positions'])==sorted(positions)
            assert decoded['information_observation'].count('ECC_RECONSTRUCTED')==sum(p<32 for p in positions)

@pytest.mark.parametrize('fixture',post.FIXTURES)
def test_ideal_path_exact_references_and_message(fixture):
    actual=post.source_frames(11,fixture);logits=ideal(actual)
    plan=sync.plan(logits[:,:32],code.EVAL_KEYS['key00'])
    assert plan['path_domain']==9275 and plan['detected']
    assert all(actual[x['received_frame']]==x['source_frame'] for x in plan['support'])
    read=sync.read_payload(logits[:,32:96],code.EVAL_KEYS['key00'],plan['support'])
    assert read['information_bits']==[i%2 for i in range(32)] and read['BCH_calls']==1
    assert len({r['source_frame'] for r in plan['support']})==len(plan['support'])
    assert all(r['received_frame']==actual.index(r['source_frame']) for r in plan['support'])

def test_threshold_ties_padding_and_null():
    zero=np.zeros((151,32));plan=sync.plan(zero,code.EVAL_KEYS['key00'])
    assert len(plan['retained_paths'])==9275 and not plan['detected'] and plan['support']==[]
    assert not sync.plan(zero,code.EVAL_KEYS['key00'],'I')['detected']
    actual=list(range(151));logits=ideal(actual);support=sync.oracle_support(actual,'O')
    assert sync.read_payload(np.zeros((151,64)),code.EVAL_KEYS['key00'],support)['status']=='ERASURE'
    invalid=logits[:,32:96].copy();invalid[0,0]=np.nan
    assert sync.read_payload(invalid,code.EVAL_KEYS['key00'],support)['status']=='TECHNICAL_UNREAD'
    for r,t in enumerate(actual):
        permutation,_=code.transform(code.EVAL_KEYS['key00'],'payload/'+str(t//8),64)
        for position,index in enumerate(permutation):
            if index>=59:logits[r,32+position]*=-1
    assert sync.read_payload(logits[:,32:96],code.EVAL_KEYS['key00'],support)['status']=='OUTPUT'

def test_head_zero_locality_teacher_clamp_and_gradient():
    model=head.LocalHead(3);features=torch.randn(1,3,256,256,requires_grad=True);bits=torch.zeros(1,96)
    pre=torch.randn(1,3,320,512)*.5+.5
    initial=head.compose(pre,head.bounded_delta(model,features,bits))
    assert torch.equal(initial,pre.clamp(0,1))
    torch.nn.init.constant_(model.conv3.weight,.001)
    student=head.compose(pre,head.bounded_delta(model,features,bits))
    mask=torch.ones(320,512,dtype=torch.bool);mask[32:288,:256]=False
    assert torch.equal(student[:,:,mask],pre.clamp(0,1)[:,:,mask])
    assert float((student-pre.clamp(0,1)).abs().max())<=8/255+1e-7
    detector=torch.nn.Linear(3,96).requires_grad_(False)
    logits=detector(student[:,:,32:288,:256].mean((2,3)))
    target=head.teacher_target(pre,torch.ones(1,3,256,256)*2)
    loss,_=head.fixed_loss(logits,bits,student,target,pre.clamp(0,1));loss.backward()
    assert features.grad is None and all(p.grad is None for p in detector.parameters())
    assert model.conv3.weight.grad.abs().sum()>0
    x=torch.tensor(.4,requires_grad=True);fake_decoded=torch.tensor(.25)
    st=x+(fake_decoded-x).detach();st.backward()
    assert st.item()==.25 and x.grad.item()==1

def test_hook_detached_features_original_cache_reset_and_dynamic_channels():
    class Conv(torch.nn.Module):
        in_channels=2;out_channels=3
        def forward(self,x,cache):return torch.cat((x,x[:,:1]),dim=1)
    conv=Conv();vae=SimpleNamespace(decoder=SimpleNamespace(conv_out=conv));seen=[]
    cache=torch.tensor([7.]);before=cache.clone()
    def capture(start,f,p):
        assert not f.requires_grad and p.dtype==torch.float32
        seen.append((start,len(f)))
    for _ in range(2):
        with head.VAEHeadHook(vae,capture=capture) as hook:
            for length in [1]+[4]*45:
                x=torch.zeros(1,2,length,320,512)
                original=conv.forward(x,cache);out=conv(x,cache)
                assert torch.equal(out,original) and torch.equal(cache,before)
        assert hook.frame==181 and hook.chunks==[1]+[4]*45
    assert seen[0]==seen[46]==(0,1)
    assert not conv._forward_hooks

def test_training_schedule_roster_budget_and_fixed_quality():
    cfg=json.loads(CONFIG.read_text());runtime.validate_config(cfg);rows=runtime.roster(cfg);training=runtime.schedule(cfg)
    assert len(rows)==150 and CounterKinds(rows)=={'positive':30,'null':120}
    assert len(training)==3000 and sum(r['step']%2 for r in training)==1500
    assert len(set(r['key'] for r in training))==3000
    assert not set(r['key'] for r in training)&set(code.EVAL_KEYS.values())
    assert any(r['start']==173 for r in training)
    assert 6*(181*2+151*5)==6702 and 6*(180*2+150*5)==6660
    assert sum(x.get('codec_encode',0) for x in runtime.budgets().values())==3072
    assert sum(x.get('codec_decode',0) for x in runtime.budgets().values())==3076
    assert sum(x.get('codec_probe',0) for x in runtime.budgets().values())==76

def CounterKinds(rows):
    from collections import Counter
    return Counter(r['kind'] for r in rows)

def test_blind_plans_all_saved_before_payload_and_post(tmp_path,monkeypatch):
    cfg=json.loads(CONFIG.read_text());rows=runtime.roster(cfg);journal=runtime.Journal(tmp_path);journal.initialize_tasks(rows,cfg['denominators'])
    for row in rows:
        if row['phase']!='CAL':continue
        path=tmp_path/runtime.logit_path(row);path.parent.mkdir(parents=True,exist_ok=True)
        save_logits(path,ideal(range(151)) if row['arm']=='WM' and row['roi']=='L' else np.zeros((151,256)))
    original_load=np.load
    def guarded(path,*args,**kwargs):
        if not (tmp_path/'CAL_plans_complete.json').exists():assert str(path).endswith('.pilot.npy')
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    original=sync.read_payload;calls=[]
    def checked(*args,**kwargs):
        assert (tmp_path/'CAL_plans_complete.json').is_file()
        assert len(list((tmp_path/'blind').rglob('plan.json')))==100
        assert not (tmp_path/'CAL_post_truth_access.json').exists()
        calls.append(1);return original(*args,**kwargs)
    monkeypatch.setattr(sync,'read_payload',checked)
    runtime.blind_phase(journal,rows,'CAL');assert len(calls)==100
    assessed=runtime.assessed_phase(journal,cfg,rows,'CAL',runtime.private_messages(cfg))
    assert len(assessed['STATE_SPACE'])==len(assessed['I'])==50

def test_technical_failure_full_rows_one_attempt_no_model_on_reentry(tmp_path):
    cfg=json.loads(CONFIG.read_text());calls=[]
    class FailBackend:
        def __init__(self,cfg,scratch,event):calls.append('init')
        def load_vae(self):raise RuntimeError('synthetic actual interface failure')
        def release_vae(self):pass
        def release_video(self):pass
    result=runtime.run(cfg,tmp_path/'out',{},backend_factory=FailBackend)
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    evaluation=runtime.load(tmp_path/'out/evaluation.json')
    assert len(evaluation['rows']['STATE_SPACE'])==150 and len(evaluation['rows']['I'])==150
    assert len(evaluation['post_OWU'])==90 and len(evaluation['quality'])==42
    assert evaluation['summaries']['STATE_SPACE']['information_slots']==4800
    assert evaluation['recommendation']=='CLOSE_THIS_CANDIDATE'
    assert evaluation['CAL']['checks']['null_existence_zero'] is None
    assert evaluation['CAL']['checks']['null_output_zero'] is None
    again=runtime.run(cfg,tmp_path/'out',{},backend_factory=FailBackend)
    assert again['physical_calls_this_invocation']==0 and calls==['init']

def test_CAL_failure_does_not_start_DEV_and_keeps_all_rows(tmp_path,monkeypatch):
    cfg=json.loads(CONFIG.read_text());started=[]
    class FakeHead(torch.nn.Linear):
        channels=1
    class Backend:
        def __init__(self,cfg,scratch,event):self.scratch=tmp_path;self.head=None
        def load_vae(self):pass
        def release_vae(self):pass
        def load_video(self):pass
        def release_video(self):pass
        def prepare_cache(self,path,sid):
            p=tmp_path/sid;p.mkdir();np.save(p/'features.npy',np.zeros((181,1)));np.save(p/'preclip.npy',np.zeros((181,1)))
            return dict(channels=1,cache=str(p))
        def initialize_head(self,c):self.head=FakeHead(1,1)
        def train_step(self,row,cache,optimizer):return dict(step=row['step'],synthetic=True)
    def fake_phase(backend,journal,cfg,sources,messages,quality):
        started.extend(s['role'] for s in sources)
        for row in runtime.roster(cfg):
            if row['phase']!='CAL':continue
            path=journal.output/runtime.logit_path(row);path.parent.mkdir(parents=True,exist_ok=True)
            save_logits(path,np.zeros((151,256)))
    monkeypatch.setattr(runtime,'physical_phase',fake_phase)
    result=runtime.run(cfg,tmp_path/'out',{s['id']:'synthetic' for s in cfg['training_sources']},backend_factory=Backend)
    assert started==['CAL','CAL'] and result['stop']['kind']=='METHOD'
    e=runtime.load(tmp_path/'out/evaluation.json')
    assert len([r for r in e['rows']['STATE_SPACE'] if r['phase']=='DEV' and r['status']=='NOT_STARTED'])==100
    assert e['summaries']['STATE_SPACE']['positive_erased_slots']==320
    assert e['summaries']['STATE_SPACE']['positive_unresolved_slots']==640

def test_notebook_builder_static_no_hash_gate_and_first_cell():
    path=ROOT/'scripts/build_b_local_vae_head_v1_notebook.py'
    spec=importlib.util.spec_from_file_location('builder',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    before=(ROOT/'notebooks/b_local_vae_head_v1.ipynb').read_text();module.build(ROOT)
    assert before==(ROOT/'notebooks/b_local_vae_head_v1.ipynb').read_text()
    nb=json.loads(before);cells=[c for c in nb['cells'] if c['cell_type']=='code']
    assert ''.join(cells[0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in cells:ast.parse(''.join(c['source']));assert c['execution_count'] is None and not c['outputs']
    import nbformat
    nbformat.validate(nb)
    combined='\n'.join(''.join(c['source']) for c in cells)
    assert 'expected_sha' not in combined and 'b64decode' not in combined and 'SOURCE_REF=\'dev/local-joint-state-payload-v1\'' in combined
    assert 'b_local_vae_head_v1_run import _notebook_reserved_main' in combined
    assert nb['metadata']['accelerator']=='GPU'
    tree=ast.parse(combined)
    replace_args=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='replace' and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='filename']
    assert replace_args==['\\']

def test_budget_limit_consumes_start_on_failure(tmp_path):
    j=runtime.Journal(tmp_path)
    for _ in range(4):j.event('preparation','E181',False,181)
    assert j.counts['preparation']['E181']['returned']==0
    with pytest.raises(RuntimeError,match='budget exhausted'):j.event('preparation','E181',False,181)

def test_parity_zero_erases_message_but_keeps_observed_information_and_common_BER():
    frames=list(range(151));x=ideal(frames);key=code.EVAL_KEYS['key00'];message=[i%2 for i in range(32)]
    for r,t in enumerate(frames):
        permutation,_=code.transform(key,'payload/'+str(t//8),64)
        x[r,32+permutation.index(32)]=0
    support=sync.oracle_support(frames,'O');read=sync.read_payload(x[:,32:96],key,support)
    assert read['status']=='ERASURE' and read['BCH_calls']==0 and read['raw_information_bits']==message
    row=runtime.roster(json.loads(CONFIG.read_text()))[0]
    plan=dict(support=support,detected=True,source_span=150)
    first=post.assess(row,plan,read,frames,message)
    altered=dict(read,raw_information_bits=[None]+[1-message[1]]+message[2:])
    second=post.assess(row,plan,altered,frames,message)
    pair=post.paired([first],[second])[0]
    assert pair['common_raw_information_slots']==31 and pair['main_raw_BER']==0
    assert pair['I_raw_BER']==1/31 and pair['common_raw_information_indices']==list(range(1,32))

def test_main_supported_read_can_succeed_while_post_oracle_fails():
    actual=list(range(151));key=code.EVAL_KEYS['key00'];message=[0]*32;other=[1]*32
    logits=ideal(actual,message,key);kept=set(range(30))|set(range(121,151))
    for r in range(151):
        if r not in kept:logits[r,:32]=0;logits[r,32:96]=ideal([r],other,key)[0,32:96]*5
    plan=sync.plan(logits[:,:32],key)
    main=sync.read_payload(logits[:,32:96],key,plan['support'],detected=plan['detected'])
    oracle=sync.read_payload(logits[:,32:96],key,sync.oracle_support(actual,'O'))
    assert plan['detected'] and len(plan['support'])==60 and main['information_bits']==message
    assert oracle['status']=='OUTPUT' and oracle['information_bits']==other

def test_SIGKILL_recovery_preserves_fixed_rows_without_scientific_read(tmp_path):
    import os,subprocess,sys
    output=tmp_path/'killed'
    script="""
import json,os,signal
from pathlib import Path
from runtime.wan.b_local_vae_head_v1 import run
class Kill:
    def __init__(self,*args):pass
    def load_vae(self):os.kill(os.getpid(),signal.SIGKILL)
cfg=json.loads(Path(__import__('sys').argv[1]).read_text())
run(cfg,__import__('sys').argv[2],{},backend_factory=Kill)
"""
    env=dict(os.environ,PYTHONPATH=str(ROOT));completed=subprocess.run([sys.executable,'-c',script,str(CONFIG),str(output)],env=env,timeout=30)
    assert completed.returncode==-9 and not (output/'result.json').exists()
    restored=runtime.recover(output)['fixed_recovery']
    assert len(restored['tasks'])==390 and restored['missing_state_rows']==0
    assert len(restored['quality'])==42 and restored['scientific_calls_this_recovery']==0
    assert all(r['read_status']=='NOT_STARTED' for r in restored['tasks'])

def test_relative_journal_paths_survive_vendor_chdir(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path);j=runtime.Journal('local','mirror');other=tmp_path/'vendor';other.mkdir();monkeypatch.chdir(other)
    j.write('probe.json',{'actual':True})
    assert (tmp_path/'local/probe.json').is_file() and (tmp_path/'mirror/probe.json').is_file()
    assert not (other/'local').exists()

def test_CLI_preparation_failure_precedes_reservation(tmp_path,monkeypatch):
    import sys,types
    from runtime.wan import b_local_vae_head_storage_v1 as storage,b_local_vae_head_setup_v1 as setup
    from experiments.wan_state_clock import b_local_vae_head_v1_run as cli
    google=types.ModuleType('google');auth=types.ModuleType('google.auth');transport=types.ModuleType('google.auth.transport');requests=types.ModuleType('google.auth.transport.requests')
    auth.default=lambda **kw:('fake',None);requests.AuthorizedSession=lambda c:object();google.auth=auth
    for name,module in [('google',google),('google.auth',auth),('google.auth.transport',transport),('google.auth.transport.requests',requests)]:monkeypatch.setitem(sys.modules,name,module)
    monkeypatch.setattr(storage,'preflight',lambda *a:{'can_start':True})
    calls=[];monkeypatch.setattr(storage,'reserve',lambda *a:calls.append('reserve'))
    monkeypatch.setattr(setup,'prepare_vendor',lambda *a:tmp_path)
    def fail(*args):raise RuntimeError('synthetic dependency failure')
    monkeypatch.setattr(setup,'check_dependencies',fail);monkeypatch.setattr(sys,'argv',['runner','--config',str(CONFIG)])
    with pytest.raises(RuntimeError,match='synthetic dependency failure'):cli.main()
    assert calls==[]

def test_empty_cloud_folder_keeps_preparation_error_and_display_structure(tmp_path):
    from runtime.wan.b_local_vae_head_storage_v1 import read_existing
    class Empty:
        def folder(self):return None
    saved=read_existing(Empty(),tmp_path)
    assert saved['status']=='NO_CLOUD_ATTEMPT_RECORD' and saved['errors'][0]['file']=='output_folder'
    assert saved['result.json']['status']=='NO_SAVED_COMPUTATION'
    assert saved['evaluation.json']['method_status']=='NOT_EVALUATED'
    assert saved['fixed_recovery']['missing_state_rows']==390 and saved['physical_calls']==0
