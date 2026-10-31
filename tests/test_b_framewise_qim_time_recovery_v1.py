"""Artificial engineering fixtures only. No saved media/raw/model/codec execution."""
import importlib.util
from contextlib import nullcontext
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pytest
from main.tube_state import b_framewise_qim_time_recovery_v1 as m
from experiments.wan_state_clock import b_framewise_qim_time_recovery_v1_records as rec
from experiments.wan_state_clock import b_framewise_qim_time_recovery_v1_experiment as exp

ROOT=Path(__file__).resolve().parents[1]
STEM='b_framewise_qim_time_recovery_v1'
pytestmark=pytest.mark.unit


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


@pytest.fixture(scope='module')
def table(): return m.codebook()


def frame(*js):
    return [dict(lane=l,labels=[j for j in js if j%5==l]) for l in range(5)]


def test_bit_order_full_domain_and_lane_table(table):
    from main.tube_state.b_framewise_qim_v1 import encode
    assert [len(x[0]) for x in table]==[18535,18535,18534,18534,18534]
    for j in (0,1,4,17,180,181,4096,65535,92670,92671):
        word=int(m.words([j])[0])
        assert [(word>>k)&1 for k in range(62,-1,-1)]==encode(j)
        js,codes=table[j%5];assert codes[np.flatnonzero(js==j)[0]]==word


@pytest.mark.parametrize('errors,erasures,accept',[(7,0,True),(8,0,False),(0,14,True),(0,15,False),(3,8,True),(3,9,False)])
def test_erasure_radius_and_high_j(table,errors,erasures,accept):
    j=92671;word=int(m.words([j])[0]);bits=np.array([(word>>k)&1 for k in range(62,-1,-1)])
    soft=np.zeros((1,5,2,63));mask=np.zeros_like(soft,dtype=bool)
    soft[0,j%5,:,:]=(bits*2-1)*1e-20;mask[0,j%5,:,:]=True
    soft[0,j%5,:,:errors]*=-1;mask[0,j%5,:,errors:errors+erasures]=False
    result,_,_=m.compatibility(soft,mask,table)
    assert (j in result['labels'][0][j%5]['labels'])==accept
    assert result['comparisons']<=92672


def test_replica_conflict_no_good_copy_selection():
    soft=np.ones((1,5,2,63));mask=np.ones_like(soft,dtype=bool)
    soft[:,:,1]=-1
    merged,known=m.fuse(soft,mask);assert not known.any() and not merged.any()
    soft[:,:,1]=-.5
    merged,known=m.fuse(soft,mask);assert known.all() and (merged==.5).all()
    mask[:]=False;assert not m.fuse(soft,mask)[1].any()


def test_numpy_reader_matches_original_uint8_once():
    import torch
    from main.tube_state.b_framewise_qim_v1 import read_rgb
    rgb=np.random.default_rng(1).integers(0,256,(2,320,512,3),dtype=np.uint8)
    got=m.read_rgb(rgb);expected=read_rgb(torch.from_numpy(rgb).double()/255)
    for key in ('q','d0','d1','soft'): np.testing.assert_allclose(got[key],expected[key].numpy(),atol=2e-12,rtol=2e-12)
    np.testing.assert_array_equal(got['mask'],expected['mask'].numpy())


def test_startup_free_endpoints_and_support():
    result=m.interval_paths([frame(*range(n+1)) for n in range(5)])
    assert [n['interval'] for n in result['nodes']]==[[0,4],[1,4],[2,4],[3,4],[4,4]]
    assert [n['output_index'] for n in result['nodes']]==[None,None,None,None,4]
    assert result['qualification']=='SUPPORTED_RELATION_SET'
    single=m.interval_paths([frame(0)]*8)
    assert single['qualification']=='INSUFFICIENT_MULTILANE_EVIDENCE'
    assert all(n['output_index'] is None for n in single['nodes'])


def test_conflict_inference_no_path_and_exact_events():
    exact=frame(1,5)  # intersection {5}, two lanes
    got=m.interval_paths([exact,frame(),exact])
    assert got['nodes'][1]['inference']=='INFERRED_NO_LOCAL_LABEL'
    assert all(x['repeated_index']==5 for x in got['relations'])
    got=m.interval_paths([exact,frame(0,9),exact])
    assert got['nodes'][1]['interval']==[5,5] and got['nodes'][1]['output_index'] is None
    assert got['path_status']=='PARTIAL_PATH_WITH_LOCAL_CONFLICT'
    assert all(x['output_endpoints'] is None for x in got['relations'])
    assert m.interval_paths([frame(9),frame(0)])['path_status']=='NO_COMPATIBLE_PATH'
    gap=m.interval_paths([frame(1,5),frame(5,9)])['relations'][0]
    assert gap['delta']==[4,4] and gap['missing_indices']==[6,7,8]


def test_exact_intervals_and_legal_edges_by_enumeration():
    labels=[frame(0),frame(2),frame(3)]
    paths=[p for p in itertools.product(range(5),range(2,7),range(3,8)) if p[0]<=p[1]<=p[2]]
    got=m.interval_paths(labels)
    for r,node in enumerate(got['nodes']):
        assert set(range(node['interval'][0],node['interval'][1]+1))=={p[r] for p in paths}
    for r,rel in enumerate(got['relations']):
        assert set(range(rel['delta'][0],rel['delta'][1]+1))=={p[r+1]-p[r] for p in paths}


class FakeBackend:
    def __init__(self,fail=None):self.fail=fail;self.calls=[];self.views={}
    def hit(self,op,path=''):
        key=(op,Path(path).parent.name);self.calls.append(key)
        if self.fail==key:raise ValueError('synthetic failure')
    def codebook(self):self.hit('codebook');return 'PUBLIC_FAKE'
    def probe(self,path,frames):self.hit('probe',path);return dict(frames=frames)
    def decode(self,path,frames):
        self.hit('decode',path)
        return np.arange(frames)[:,None] if frames==181 and Path(path).name!='received.mp4' else self.views[str(path)]
    def encode(self,rgb,path):self.hit('encode',path);self.views[str(path)]=rgb.copy()
    def read(self,rgb):
        self.hit('read');a=np.zeros((len(rgb),5,2,63))
        return dict(q=a,d0=a,d1=a,soft=a,mask=np.ones_like(a,dtype=bool))
    def receiving_view(self):return nullcontext()
    def receive(self,raw,table):
        self.hit('receive');assert set(raw)=={'q','d0','d1','soft','mask'} and table=='PUBLIC_FAKE'
        result=m.interval_paths([frame() for _ in raw['q']]);result['labels']=[frame() for _ in raw['q']];result['comparisons']=0
        merged,known=m.fuse(raw['soft'],raw['mask']);return result,merged,known
    def close(self):return dict(fake=True)


def inputs(tmp_path):
    qim=tmp_path/'old-qim'/'QIM.mp4';base=tmp_path/'old-base'/'BASE.mp4'
    for p in (qim,base):p.parent.mkdir();p.write_bytes(b'artificial-not-video')
    return qim,base,tmp_path/'new'


def test_fixed_plan_fake_codec_budgets_and_truth_last(tmp_path,monkeypatch):
    from experiments.wan_state_clock import b_framewise_qim_time_recovery_v1_evaluation as ev
    qim,base,out=inputs(tmp_path);fake=FakeBackend();real_load=ev.load;opened=[]
    def guarded_load(path):
        if Path(path).name=='attack_truth.json':
            state=rec.load(out/'state.json');assert all(r['status'] not in ('STARTED','NOT_STARTED') for r in state.values())
            assert len([x for x in fake.calls if x[0]=='receive'])==10;opened.append(path)
        return real_load(path)
    monkeypatch.setattr(ev,'load',guarded_load)
    exp.run(qim,base,out,lambda:fake)
    counts=rec.load(out/'counts.json')
    assert all(v['started']==v['returned']==v['planned'] for v in counts.values())
    assert len(opened)==1 and rec.denominators()['received_frames']==1644
    assert [len(fake.views[str(out/'received'/t['id']/'received.mp4')]) for t in exp.private_plan()]==list(rec.LENGTHS)
    mappings=[v[:,0].tolist() for v in fake.views.values()]
    assert mappings==[t['mapping'] for t in exp.private_plan()]
    evaluation=rec.load(out/'evaluation.json')
    assert evaluation['base_five_rejected'] and evaluation['failure_facts']==['QIM_ALL_REJECTED']
    assert sum(v['frame_counts']['unresolved'] for v in evaluation['views'][:5])==822
    assert sum(v['relation_counts']['unresolved'] for v in evaluation['views'][:5])==817
    before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    exp.run('/missing','/missing',out,lambda:pytest.fail('reentry executed'))
    assert before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    assert qim.read_bytes()==base.read_bytes()==b'artificial-not-video'


@pytest.mark.parametrize('failure',[('encode','v01'),('probe','old-qim'),('decode','v03'),('codebook','')])
def test_failures_consume_without_refill_and_keep_other_views(tmp_path,failure):
    qim,base,out=inputs(tmp_path);fake=FakeBackend(failure)
    exp.run(qim,base,out,lambda:fake)
    state=rec.load(out/'state.json');counts=rec.load(out/'counts.json')
    assert state['v09/receiver']['status']=='RETURNED' if failure[0]!='codebook' else state['v09/time_read']['status']=='RETURNED'
    assert all(v['started']<=v['planned'] for v in counts.values())
    assert all(v['status'] not in ('STARTED','NOT_STARTED') for v in state.values())
    assert any(v['status']=='FAILED' for v in state.values())


def test_preflight_and_sibling_alias_protect_inputs(tmp_path):
    qim,base,out=inputs(tmp_path)
    assert rec.preflight(qim.with_name('missing.mp4'),base,out)['status']=='PREFLIGHT_READY' and not out.exists()
    assert rec.preflight(qim.with_name('missing.mp4'),base.with_name('missing.mp4'),out)['status']=='PREFLIGHT_UNRESOLVED' and not out.exists()
    for suffix in ('','-logs','-entry-cache','-preflight-v1'):
        target=tmp_path/'target';protected=target.with_name(target.name+suffix);protected.mkdir(exist_ok=True)
        p=protected/'QIM.mp4';p.write_bytes(b'input')
        with pytest.raises(ValueError):rec.validate_paths(p,base,target)
        assert p.read_bytes()==b'input'


def test_cpu_codec_commands_and_timeout_fake_only(monkeypatch,tmp_path):
    from runtime.wan import b_framewise_qim_time_recovery_v1 as rt
    captured=[]
    class Guard:
        def check(self):pass
    backend=rt.Backend.__new__(rt.Backend);backend.resources=Guard()
    def fake_command(argv,resources,input_bytes=None):
        captured.append((argv,input_bytes))
        if argv[0]=='ffprobe':return json.dumps({'streams':[dict(width=512,height=320,r_frame_rate='8/1',nb_frames='1')]}).encode()
        return bytes(320*512*3)
    monkeypatch.setattr(rt,'command',fake_command)
    rgb=np.zeros((1,320,512,3),dtype=np.uint8);backend.encode(rgb,tmp_path/'fake.mp4');backend.probe('fake',1);backend.decode('fake',1)
    argv,data=captured[0];assert argv==['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s','512x320','-r','8','-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-n',str(tmp_path/'fake.mp4')]
    assert data==rgb.tobytes()


def test_resource_deadlines_and_actual_rss():
    from runtime.wan import b_framewise_qim_time_recovery_v1 as rt
    resources=rt.Resources()
    try:
        resources.deadline=time.monotonic()-1
        with pytest.raises(rt.ResourceExceeded):resources.check()
        resources.deadline=None;resources.error='synthetic memory limit'
        with pytest.raises(rt.ResourceExceeded):resources.check()
    finally:value=resources.close()
    assert value['peak_resident_bytes']>0 and value['peak_limit_bytes']==8*1024**3


def test_no_git_cli_without_outer_modules_science_on_recovery(tmp_path):
    root=tmp_path/'copy';folder=root/'experiments/wan_state_clock';folder.mkdir(parents=True)
    for name in ('run','experiment','records'):
        shutil.copyfile(ROOT/'experiments/wan_state_clock'/f'{STEM}_{name}.py',folder/f'{STEM}_{name}.py')
    out=tmp_path/'consumed';out.mkdir();rec.dump(out/'attempt.json',{'status':'CONSUMED'})
    code="""import runpy,sys
class Block:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','psutil','runtime','main','scripts'):raise RuntimeError('forbidden '+name)
sys.meta_path.insert(0,Block())
runpy.run_module('experiments.wan_state_clock.b_framewise_qim_time_recovery_v1_run',run_name='__main__')
"""
    command=[sys.executable,'-B','-c',code,'--qim-media','/missing','--base-media','/missing','--output',str(out)]
    result=subprocess.run(command,cwd=root,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'READ_ONLY_EXISTING_ATTEMPT' in result.stdout


def test_notebook_builder_and_consumed_zero_network_setup(tmp_path,monkeypatch):
    builder=module(ROOT/'scripts'/f'build_{STEM}_notebook.py','time_builder');builder.build(tmp_path)
    assert json.loads((tmp_path/'notebooks'/f'{STEM}.ipynb').read_text())==json.loads((ROOT/'notebooks'/f'{STEM}.ipynb').read_text())
    notebook=json.loads((ROOT/'notebooks'/f'{STEM}.ipynb').read_text());codes=[''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert 'accelerator' not in notebook['metadata']
    entry=module(ROOT/'scripts'/f'{STEM}_entry.py','time_entry')
    out=tmp_path/'consumed';out.mkdir();rec.dump(out/'attempt.json',{'status':'RESERVED'})
    for name in ('prepare_source','cache_entry'):
        monkeypatch.setattr(entry.common,name,lambda *a,**k:pytest.fail('consumed mutation'))
    monkeypatch.setattr(entry,'setup_dependencies',lambda *a:pytest.fail('setup'))
    entry.Session('/missing','/missing',out).execute()


def test_common_source_download_and_local_own_requirements(tmp_path,monkeypatch):
    import zipfile
    entry=module(ROOT/'scripts'/f'{STEM}_entry.py','entry_download')
    archive=tmp_path/'fake.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for path in entry.REQUIRED:z.writestr('checkout/'+path,'# artificial source\n')
    urls=[]
    def download(url,dest):urls.append(url);shutil.copyfile(archive,dest)
    monkeypatch.setattr(entry.common.urllib.request,'urlretrieve',download)
    root=entry.common.prepare_source(None,'dev/local-joint-state-payload-v1',tmp_path/'scratch',entry.REQUIRED)
    assert urls==['https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/dev/local-joint-state-payload-v1']
    assert entry.common.prepare_source(root,'unused',tmp_path/'scratch',entry.REQUIRED)==root


def test_missing_source_only_blocks_its_five_views(tmp_path):
    qim,base,out=inputs(tmp_path);fake=FakeBackend()
    exp.run(qim.with_name('absent.mp4'),base,out,lambda:fake)
    state=rec.load(out/'state.json');counts=rec.load(out/'counts.json')
    assert state['s0/probe']['status']=='DEPENDENCY_UNRESOLVED'
    assert all(state[f'v{i:02}/receiver']['status']=='DEPENDENCY_UNRESOLVED' for i in range(5))
    assert all(state[f'v{i:02}/receiver']['status']=='RETURNED' for i in range(5,10))
    assert counts['codec_probe']['started']==6 and counts['codec_decode']['started']==6 and counts['codec_encode']['started']==5
    evaluation=rec.load(out/'evaluation.json')
    assert sum(v['frame_counts']['not_executed'] for v in evaluation['views'][:5])==822
    assert evaluation['base_five_rejected'] and 'RUN_OR_EVALUATION_FAILURE' in evaluation['failure_facts']


def test_codec_timeout_terminates_then_kills_fake_process(monkeypatch):
    from runtime.wan import b_framewise_qim_time_recovery_v1 as rt
    calls=[]
    class Guard:
        def check(self):pass
    class Process:
        def communicate(self,input=None,timeout=None):
            calls.append(('communicate',timeout))
            return b'',b''
        def wait(self,timeout=None):
            calls.append(('wait',timeout))
            if timeout is not None:raise subprocess.TimeoutExpired('fake',timeout)
        def terminate(self):calls.append(('terminate',))
        def kill(self):calls.append(('kill',))
    ticks=iter((0,301))
    monkeypatch.setattr(rt.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(rt.subprocess,'Popen',lambda *a,**k:Process())
    with pytest.raises(rt.ResourceExceeded):rt.command(['fake-codec'],Guard())
    assert calls.count(('communicate',None))==1
    assert [c for c in calls if c[0]!='communicate']==[('terminate',),('wait',30),('kill',),('wait',None)]


def test_deadline_at_view_exit_not_reported_as_completed(tmp_path):
    from runtime.wan.b_framewise_qim_time_recovery_v1 import ResourceExceeded
    from contextlib import contextmanager
    qim,base,out=inputs(tmp_path);fake=FakeBackend()
    @contextmanager
    def window():
        yield
        raise ResourceExceeded('artificial wall timeout at close')
    fake.receiving_view=window
    exp.run(qim,base,out,lambda:fake)
    assert all(rec.load(out/'state.json')[f'v{i:02}/receiver']['status']=='RESOURCE_UNRESOLVED' for i in range(10))
    assert 'RUN_OR_EVALUATION_FAILURE' in rec.load(out/'evaluation.json')['failure_facts']


@pytest.mark.parametrize('cache_mode',['none','entry_only','complete'])
def test_actual_bootstrap_consumed_no_download_or_write(tmp_path,monkeypatch,cache_mode):
    builder=module(ROOT/'scripts'/f'build_{STEM}_notebook.py','bootstrap_builder')
    entry=module(ROOT/'scripts'/f'{STEM}_entry.py','bootstrap_cache')
    qim,base,out=inputs(tmp_path);out.mkdir();rec.dump(out/'attempt.json',{'status':'CONSUMED'})
    cache=out.with_name(out.name+'-entry-cache')
    if cache_mode!='none':
        files=entry.CACHE_FILES if cache_mode=='complete' else entry.CACHE_FILES[:2]
        for relative in files:
            target=cache/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/relative,target)
    import urllib.request
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a,**k:pytest.fail('network'))
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('process'))
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    scope=dict(OUTPUT=out,QIM_MEDIA=qim,BASE_MEDIA=base,SOURCE_DIRECTORY=None,SOURCE_REF='dev/local-joint-state-payload-v1',Path=Path)
    exec(builder.BOOTSTRAP,scope);exec(builder.RUN,scope);exec(builder.DISPLAY,scope)
    assert before=={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_wrong_attribution_and_base_accept_cannot_be_coverage_success(tmp_path):
    qim,base,out=inputs(tmp_path);fake=FakeBackend()
    def accept(raw,table):
        labels=[frame(1,5) for _ in raw['q']]
        result=m.interval_paths(labels);result['labels']=labels;result['comparisons']=0
        merged,known=m.fuse(raw['soft'],raw['mask']);return result,merged,known
    fake.receive=accept
    exp.run(qim,base,out,lambda:fake)
    evaluation=rec.load(out/'evaluation.json')
    assert evaluation['outcome']=='FAILURE_FACTS_RETAINED'
    assert set(evaluation['failure_facts'])=={'WRONG_ATTRIBUTION','BASE_FALSE_ACCEPT'}
    assert all(v['base_false_accept'] is True for v in evaluation['views'][5:])
    assert not evaluation['base_five_rejected']
    for row in evaluation['views'][:5]:assert sum(row['frame_counts'].values())==row['frames']


@pytest.mark.parametrize('closing',['late_rss','raise'])
def test_final_resource_error_preserves_returns_and_blocks_success(tmp_path,capsys,closing):
    from experiments.wan_state_clock import b_framewise_qim_time_recovery_v1_report as report
    qim,base,out=inputs(tmp_path);fake=FakeBackend()
    def close():
        if closing=='raise':raise OSError('artificial close failure')
        return dict(peak_resident_bytes=8*1024**3+1,error='peak resident memory exceeds 8 GiB')
    fake.close=close
    result=exp.run(qim,base,out,lambda:fake)
    assert result['errors']['resources']
    assert all(r['status']=='RETURNED' for r in rec.load(out/'state.json').values())
    assert all(c['returned']==c['planned'] for c in rec.load(out/'counts.json').values())
    evaluation=rec.load(out/'evaluation.json')
    assert evaluation['outcome']=='FAILURE_FACTS_RETAINED'
    assert {'RESOURCE_UNRESOLVED','RUN_OR_EVALUATION_FAILURE','QIM_ALL_REJECTED'}<=set(evaluation['failure_facts'])
    report.display(out)
    assert 'RESOURCE_UNRESOLVED' in capsys.readouterr().out


def test_actual_notebook_nonobject_fallback_keeps_next_record(tmp_path,capsys):
    notebook=json.loads((ROOT/'notebooks'/f'{STEM}.ipynb').read_text())
    display=''.join([c for c in notebook['cells'] if c['cell_type']=='code'][-1]['source'])
    out=tmp_path/'consumed';out.mkdir();rec.dump(out/'attempt.json',[]);rec.dump(out/'result.json',dict(status='VISIBLE_RESULT'))
    before={p.name:p.read_bytes() for p in out.iterdir()}
    exec(display,dict(entry=None,consumed=True,conflict=False,OUTPUT=out))
    printed=capsys.readouterr().out
    assert 'JSON对象' in printed and 'VISIBLE_RESULT' in printed
    assert before=={p.name:p.read_bytes() for p in out.iterdir()}
