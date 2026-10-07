import copy,json,hashlib,ast
from pathlib import Path
import numpy as np
import pytest
from main.tube_state import video_terminal_sync_path_decision_v1 as m
from experiments.wan_state_clock import video_terminal_sync_path_decision_v1_run as run
from experiments.wan_state_clock import video_terminal_sync_path_decision_v1_prepare as prep

pytestmark=pytest.mark.unit

KEYS=dict(run.keys(run.read(run.CONFIG)))
def cache177(level=.1,tie=False,key=KEYS['K0']):
    rows=[dict(r=r,d=d,source_index=r+d,source_tubelet=(r+d)//4,source_age=(r+d)%4,status='SCORED',numerator=(0 if tie else level if d==2 else 0)*(160 if r+d==180 else 40),rho=160 if r+d==180 else 40) for r in range(177) for d in range(5)]
    return m.deletion.reduce_cache(rows,key)
def rows89(level=.1,tie=False,key=KEYS['K0']):
    local=[];candidates=[]
    for b in range(93):
        num=rho=0.
        for support in m.prior.sync.tubelet_support_rows(89,b,m.prior.PUBLIC):
            weight=support['observed_frames']*(160 if support['source_tubelet']==45 else 40)
            q=0 if tie else level if b==38 else 0
            row=dict(source_offset=b,**support,status='SCORED',q=q,signed_projection=q*weight,rho=weight)
            # Use precisely the original float division, not an idealized fake q.
            row['q']=row['signed_projection']/row['rho'];local.append(row);num+=row['signed_projection'];rho+=weight
        candidates.append(dict(source_offset=b,status='SCORED',numerator=num,denominator=rho,score=num/rho,tubelet_rows=23))
    top=list(range(93)) if tie else [38]
    return dict(status='COMPLETE',truth_inputs=False,key_id=m.prior.sync.key_identifier(key),candidate_offsets=list(range(93)),candidate_rows=candidates,local_rows=local,counts=dict(candidate_scores=93,candidate_tubelet_rows=2139,failed_candidates=0),summary=dict(top_offsets=top,canonical_offset=None if tie else 38,unique=not tie))

def test_original_scores_unchanged_and_actions():
    for n,raw in ((177,cache177()),(89,rows89())):
        old=copy.deepcopy(raw);x=m.analyze(raw,n,KEYS['K0']);assert x['status']=='SCORED' and x['unique'];assert raw==old
        assert x['M']==max(r['score'] for r in raw['candidate_rows'])
        assert x['m']>0 and all(v['path']!=x['path'] for v in x['contrasts'])
    assert len(m.grammar(177))==709 and len({x[2] for x in m.grammar(177)})==699
    assert len(m.grammar(89))==93 and len({x[2] for x in m.grammar(89)})==4
    assert m.operation(89,dict(family='H0',b=2,k=None))==m.operation(89,dict(family='H0',b=90,k=None))
    assert m.decide(m.failed(181,'unsupported'),{})['reason']=='UNSUPPORTED_PROTOCOL'

def test_89_original_score_and_rho_algebra_cpu():
    # A fresh synthetic latent; never opens historical media, latent, or caches.
    key=KEYS['K0'];full=np.empty((181,4,40,64),np.float32)
    for t in range(46):
        for y,x in m.prior.sync.spatial_patch_coordinates(40,64,m.prior.PUBLIC):
            full[t*4:min(t*4+4,181),:,y:y+4,x:x+4]=m.prior.sync.sync_sign(key,t,m.prior.PUBLIC)*m.prior.sync.patch_direction(key,t,y,x,m.prior.PUBLIC)
    z=full[38:127].copy();raw=m.prior.sync.score_received_latent(z,key,m.prior.PUBLIC);saved=copy.deepcopy(raw)
    result=m.analyze(raw,89,key);assert raw==saved and result['status']=='SCORED'
    Q={}
    for b in range(93):
        value=0.
        for row in raw['local_rows']:
            if row['source_offset']==b:value+=row['observed_frames']*row['q']
        Q[b]=value/89
    # Check ordinary and source180-containing offsets against direct per-frame algebra.
    for b in (0,38,92):
        w=full[b:b+89].astype(np.float64);num=np.sum(z.astype(np.float64)*w,axis=(1,2,3));rho=np.sum(w*w,axis=(1,2,3))
        assert Q[b]==pytest.approx(float(np.mean(num/rho)),abs=1e-12)
        assert rho[-1]==(160 if b==92 else 40)
    h=result['path']['b']
    assert result['m']==min(Q[h]-Q[g] for g in range(93) if g%4!=h%4)

def calibration_fixture():
    rows={};ids={}
    for n,count in ((177,8),(89,6)):
        ids[str(n)]=[f'{n}_{i}' for i in range(count)]
        for sid in ids[str(n)]:rows[sid]=dict(status='SCORED',frames=n,M=.1,m=.05,unique=True)
    return rows,ids

def test_null_extrema_strict_ties_missing_no_positive_selection():
    rows,ids=calibration_fixture();rows['positive']=dict(status='SCORED',frames=177,M=100,m=100,unique=True)
    tied=rows[ids['177'][0]];tied.update(M=.9,m=None,unique=False)
    t=m.calibrate(rows,ids);assert t['status']=='AVAILABLE' and t['families']['177']['tau_M']==.9
    a=dict(status='SCORED',frames=177,M=.9,m=.5,unique=True,path={},contrasts=[],received_index_map=list(range(177)))
    assert m.decide(a,t)['state']=='REJECT';a.update(M=1,m=.05);assert m.decide(a,t)['state']=='UNCERTAIN'
    a['m']=.051;assert m.decide(a,t)['state']=='ACCEPT_ACTION'
    for sid in ids['177']:rows[sid].update(unique=False,m=None)
    assert m.calibrate(rows,ids)['status']=='UNAVAILABLE'
    rows,ids=calibration_fixture();del rows[ids['89'][0]];assert m.calibrate(rows,ids)['status']=='UNAVAILABLE'
    for n,raw in ((177,cache177(tie=True)),(89,rows89(tie=True))):
        a=m.analyze(raw,n,KEYS['K0']);assert a['status']=='SCORED' and not a['unique'] and a['M']==0

class Inputs:
    @staticmethod
    def read_source(spec):return dict(n=181,sha=spec['sha256'],indices=list(range(181)))
    @staticmethod
    def receipt(rgb):return dict(sha256=rgb['sha'],shape=[rgb['n'],320,512,3],bytes=rgb['n']*320*512*3)
    @staticmethod
    def construct(full,indices):return dict(n=len(indices),sha=hashlib.sha256(bytes(indices)).hexdigest(),indices=indices)
class FW:
    instances=0;closed=0
    def __init__(self,cfg):type(self).instances+=1;self.level=.1 if type(self).instances==1 else .8
    def encode(self,rgb):return (rgb['n'],self.level),dict(status='ENCODED_FAKE')
    def score(self,z,key):return cache177(z[1],key=key) if z[0]==177 else rows89(z[1],key=key)
    def close(self):type(self).closed+=1
@pytest.fixture(autouse=True)
def reset():FW.instances=FW.closed=0

def loader(spec):
    sid=next(s for s,x in run.read(run.CONFIG)['cached_development'].items() if x['path']==spec['path'])
    return cache177(key=KEYS[sid.split('/')[1]])
def source(store):
    run.check_seal(store.data['threshold_seal']);assert store.data['thresholds']['status']=='AVAILABLE';assert FW.closed==1
    assert not (store.output/'blind_decisions.json').exists()
    return run.companion(store.cfg,'preparation')['development_sources']

def test_runner_success_real_seal_order_and_zero_payload(tmp_path,monkeypatch):
    original=run.companion;events=[]
    def guarded(cfg,suffix):
        if suffix=='calibration_labels':assert (tmp_path/'run/development_blind_scores.json').exists()
        if suffix=='posthoc':
            assert (tmp_path/'run/blind_decisions.json').exists() and (tmp_path/'run/blind_operations.json').exists()
        events.append(suffix);return original(cfg,suffix)
    monkeypatch.setattr(run,'companion',guarded)
    x=run.run(tmp_path/'run',framewise_type=FW,input_type=Inputs,cache_loader=loader,source_fn=source)
    assert x['status']=='COMPLETE' and len(x['decisions'])==16 and len(x['development'])==20
    assert FW.instances==FW.closed==2
    for stage in ('development','confirmation'):
        assert x['calls'][stage+'/source_read']==dict(attempted=2,completed=2)
        assert x['calls'][stage+'/source_slice']==dict(attempted=8,completed=8)
        assert x['calls'][stage+'/framewise_receiver_encode']==dict(attempted=8,completed=8)
        assert x['calls'][stage+'/sync_score']==dict(attempted=16,completed=16)
    assert x['calls']['receiver_wan_encode']==x['calls']['payload_read']==dict(attempted=0,completed=0)
    assert sum(row['positive'] for row in x['posthoc'].values())==4
    assert events.index('calibration_labels')<events.index('posthoc')
    assert x['operation_plan']['receiver_wan_encodes']==x['operation_plan']['payload_reads']==0
    assert any(v['GATED']['alias_of'] for v in x['operation_plan']['operations'].values())
    assert all(v['GATED']['operation_cost'] is not None for v in x['operation_plan']['operations'].values())

@pytest.mark.parametrize('kind',['missing','hash'])
def test_bad_cached_input_retains_all_and_blocks_generation(tmp_path,kind):
    def bad(spec):raise FileNotFoundError('missing cache') if kind=='missing' else ValueError('cached read SHA mismatch')
    def forbidden(store):raise AssertionError('must not generate')
    x=run.run(tmp_path/kind,framewise_type=FW,input_type=Inputs,cache_loader=bad,source_fn=forbidden)
    assert x['thresholds']['status']=='UNAVAILABLE' and len(x['decisions'])==16
    assert all(row['execution_status']=='NOT_RUN' for row in x['confirmation'].values())
    assert set(d['state'] for d in x['decisions'].values())=={'UNCERTAIN'}
    assert FW.instances==FW.closed==1

@pytest.mark.parametrize('exc',[RuntimeError('source failed'),KeyboardInterrupt('worker interrupted')])
def test_source_failure_or_interrupt_keeps16(tmp_path,exc):
    def bad(store):raise exc
    if isinstance(exc,KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):run.run(tmp_path/'run',framewise_type=FW,input_type=Inputs,cache_loader=loader,source_fn=bad)
        x=run.read(tmp_path/'run/result.json')
    else:x=run.run(tmp_path/'run',framewise_type=FW,input_type=Inputs,cache_loader=loader,source_fn=bad)
    assert len(x['decisions'])==len(x['posthoc'])==16 and x['stage']=='FINISHED'
    assert all(d['state']=='UNCERTAIN' for d in x['decisions'].values())

def test_framewise_interrupt_release_and_retention(tmp_path):
    class Interrupted(FW):
        def encode(self,rgb):raise KeyboardInterrupt('encode interrupted')
    with pytest.raises(KeyboardInterrupt):run.run(tmp_path/'run',framewise_type=Interrupted,input_type=Inputs,cache_loader=loader,source_fn=source)
    x=run.read(tmp_path/'run/result.json');assert len(x['development'])==20 and len(x['decisions'])==16
    assert Interrupted.closed==1 and x['stage']=='FINISHED'

def test_cached_hash_production_loader(tmp_path):
    p=tmp_path/'a.json.gz';p.write_bytes(b'not the specified file')
    with pytest.raises(ValueError,match='SHA'):run.cached_read(dict(path=str(p),sha256='0'*64))
    with pytest.raises(FileNotFoundError):run.cached_read(dict(path=str(p.parent/'missing'),sha256='0'*64))

def test_matched_worker_same_latent_two_codec_cpu(tmp_path):
    cfg=run.companion(run.read(run.CONFIG),'preparation');st=prep.Store(tmp_path/'source',cfg,'source')
    st.data['source_protocol']={};events=[]
    class Backend:
        def __init__(self,cfg):pass
        def read(self,s):return 'rgb'
        def load(self):pass
        def encode(self,x):return np.zeros((2,2),np.float32)
        def write(self,z):z+=1;return z,dict(target_margin=.5,rows=[{}])
        def decode(self,z):events.append(('decode',float(z.sum())));return 'raster'
        def save(self,rgb,p):return dict(path=str(p),sha256='fake',status='SAVED')
        def close(self):events.append(('close',))
        def transport(self,raster,root,count,event):
            assert events[-1][0]=='close' or events[-1][0]=='transport';events.append(('transport',))
            for k in ('mp4_save','mp4_probe','mp4_readback'):count(k,False);count(k,True)
            event('rgb24',dict(status='SAVED',path=str(root/'received.rgb8'),sha256='fake',shape=[181,320,512,3],bytes=88965120))
    prep.matched_worker(st,cfg,Backend)
    assert events[:3]==[('decode',0.),('decode',4.),('close',)]
    assert set(st.data['received_sources'])=={'P1','M05'}
    for k,n in cfg['m05_planned_calls'].items():assert st.data['calls'][k]==dict(attempted=n,completed=n)

def test_notebook_schema_ast_draft_guard(tmp_path):
    import nbformat,re
    from scripts import build_video_terminal_sync_path_decision_notebook as b
    def validate(path,expected=None,require_published=False):
        nb=nbformat.read(path,as_version=4);nbformat.validate(nb);code=[x for x in nb.cells if x.cell_type=='code']
        assert len(code)==5 and code[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
        binding=nb.metadata.candidate_binding;sha=binding.source_sha
        assert sha is None or re.fullmatch('[0-9a-f]{40}',sha)
        assert binding.status==('UNPUBLISHED_DRAFT' if sha is None else 'PUBLISHED_SHA_BOUND')
        assignments=[n for n in ast.parse(code[1].source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_SHA' for t in n.targets)]
        assert len(assignments)==1 and ast.literal_eval(assignments[0].value)==sha
        if require_published:assert sha is not None and sha==expected
        for x in code:ast.parse(x.source);assert x.execution_count is None and x.outputs==[]
        assert code[1].source.index('if SOURCE_SHA is None')<code[1].source.index('OUTPUT.mkdir')
        assert 'WanPipeline' in code[2].source and 'sentencepiece' in code[2].source and 'ftfy' in code[2].source
        return sha
    # Draft test is temporary, so the same source test remains valid after real N binding.
    draft=b.build(None,tmp_path/'draft.ipynb');assert validate(draft) is None
    fake_sha='a'*40;published=b.build(fake_sha,tmp_path/'bound_fixture.ipynb')
    validate(published,expected=fake_sha,require_published=True)
    with pytest.raises(AssertionError):validate(draft,expected=fake_sha,require_published=True)
    validate(b.OUTPUT)


def test_development_positive_analysis_failure_does_not_expand_generation_gate(tmp_path):
    def bad_positive(spec):
        x=loader(spec)
        cfg=run.read(run.CONFIG)
        if spec['path']==cfg['cached_development']['cal_09/K0']['path']:x['candidate_rows']=[]
        return x
    x=run.run(tmp_path/'run',framewise_type=FW,input_type=Inputs,cache_loader=bad_positive,source_fn=source)
    assert x['thresholds']['status']=='AVAILABLE' and FW.instances==2
    assert x['development_analyses']['cal_09/K0']['status']=='FAILED'
    assert x['status']=='RETAINED_INCOMPLETE' and len(x['decisions'])==16
    assert x['family_summary']['89']['null_denominator']==6

@pytest.mark.parametrize('phase',['source','matched'])
def test_actual_preparation_process_sequence_failure_no_retry(tmp_path,phase):
    cfg=run.read(run.CONFIG);st=run.Store(tmp_path/'run',cfg)
    st.data['thresholds']=dict(status='AVAILABLE',families={})
    st.data['threshold_seal']=run.dump(st.output/'frozen_thresholds.json',st.data['thresholds'])
    called=[]
    def worker(store,p):
        called.append(p)
        if p==phase:raise KeyboardInterrupt(p+' interrupted')
        run.dump(store.output/'source_preparation/source_preparation.json',dict(status='SOURCE_READY'))
    with pytest.raises(KeyboardInterrupt):run.prepare_new_source(st,worker=worker)
    assert called==(['source'] if phase=='source' else ['source','matched'])
    assert len(st.data['confirmation'])==16 and all(x['execution_status']=='NOT_RUN' for x in st.data['confirmation'].values())
    if phase=='matched':assert run.read(st.output/'source_preparation/source_preparation.json')['status']=='INCOMPLETE'
