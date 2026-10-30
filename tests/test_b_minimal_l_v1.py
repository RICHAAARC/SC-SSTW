"""Artificial CPU arrays/receipts/fake media only; no real input or codec."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from main.tube_state import b_minimal_l_v1 as m
from main.tube_state import b_framewise_qim_time_recovery_v1 as hard
from main.tube_state import b_soft_time_evidence_v1 as soft
from runtime.wan.b_minimal_l_v1 import Backend
from experiments.wan_state_clock import b_minimal_l_v1_records as rec
from experiments.wan_state_clock import b_minimal_l_v1_experiment as exp
from experiments.wan_state_clock import b_minimal_l_v1_evaluation as post

pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]


def test_codebook_generated_once_and_real_hard_C_share(monkeypatch):
    original=hard.codebook;calls=[]
    monkeypatch.setattr(hard,'codebook',lambda check:(calls.append(1) or original(check)))
    monkeypatch.setattr(soft,'codebook',lambda *a:pytest.fail('no second soft codebook'))
    buckets,flat=m.shared_codebook()
    assert len(calls)==1 and len(flat)==92672
    for js,words in buckets: np.testing.assert_array_equal(flat[js.astype(int)],words)
    raw=dict(soft=np.zeros((2,5,2,63)),mask=np.ones((2,5,2,63),bool))
    for r,n in enumerate((92670,92671)):
        for lane in range(5):
            j=n-(n-lane)%5;word=int(flat[j]);bits=np.array([(word>>k)&1 for k in range(62,-1,-1)])
            raw['soft'][r,lane]=2*bits-1
    decision,_,_=hard.receive(raw,buckets)
    original_decision=copy.deepcopy(decision)
    cost=soft.costs(raw['soft'],raw['mask'],soft.graph_input(decision,2),flat)
    local=soft.local_rank(cost);candidate=m.candidates(decision,cost,local)
    assert [n['candidate_index'] for n in candidate['nodes']]==[92670,92671]
    assert decision==original_decision and candidate['formal_new_determined']==candidate['new_relation_outputs']==0
    assert len(calls)==1


@pytest.mark.parametrize('qualification,state,best,expected',[
    ('SUPPORTED_RELATION_SET','DEFINED',[True,False],5),
    ('INSUFFICIENT_MULTILANE_EVIDENCE','DEFINED',[True,False],None),
    ('SUPPORTED_RELATION_SET','DEFINED',[True,True],None),
    ('SUPPORTED_RELATION_SET','STARTUP_MODEL_UNDEFINED',[True,False],None),
    ('SUPPORTED_RELATION_SET','LOCAL_CONFLICT',[True,False],None),
    ('SUPPORTED_RELATION_SET','NO_SOFT_OBSERVATION',[True,False],None)])
def test_candidate_qualification_and_undefined_not_promoted(qualification,state,best,expected):
    cost=dict(states=[state],bounds=np.array([[5,6]]),offsets=np.array([0,2]))
    result=m.candidates(dict(qualification=qualification),cost,dict(best=np.array(best),status=[state]))
    assert result['nodes'][0]['candidate_index']==expected


class FakeBackend(Backend):
    failure=None
    def __init__(self): super().__init__();self.in_view=False;self.reads=0
    def probe(self,path,frames): assert not self.in_view;return dict(frames=frames)
    def decode(self,path,frames): assert not self.in_view;return np.zeros((frames,1),np.uint8)
    def encode(self,rgb,path): assert not self.in_view;Path(path).write_bytes(b'fake-media')
    @contextmanager
    def receiving_view(self):
        with super().receiving_view():
            self.in_view=True
            try: yield
            finally: self.in_view=False
    def read(self,rgb):
        assert self.in_view;self.reads+=1
        if self.failure=='raw' and self.reads==2: raise ValueError('artificial raw failure')
        shape=(len(rgb),5,2,63);v=np.ones(shape)
        return dict(q=v,d0=v,d1=v,soft=v,mask=np.ones(shape,bool))
    def receive(self,raw,table):
        assert self.in_view
        R=len(raw['soft']);base=self.reads>5
        decision=dict(qualification='SUPPORTED_RELATION_SET',path_status='COMPATIBLE_PATH',comparisons=R*hard.DOMAIN,
            nodes=[dict(received_frame=r,interval=[0,4] if base else [5,6],status='LOCAL_LABEL',output_index=None) for r in range(R)],
            relations=[dict(output_endpoints=None) for _ in range(R-1)])
        return decision,np.zeros((R,5,63)),np.ones((R,5,63),bool)
    def C(self,*args): assert self.in_view;return super().C(*args)
    def L(self,*args): assert self.in_view;return super().L(*args)
    def close(self):
        result=super().close()
        if self.failure=='close': raise OSError('artificial close failure')
        return result


def assets(tmp_path):
    sources=tmp_path/'media';sources.mkdir()
    for name in ('BASE','QIM'): (sources/(name+'.mp4')).write_bytes(b'fake readable media')
    return sources/'QIM.mp4',sources/'BASE.mp4',tmp_path/'output'


@pytest.mark.parametrize('failure',[None,'raw','close'])
def test_ten_view_budget_persistence_time_window_truth_last_and_consumed(tmp_path,monkeypatch,failure):
    qim,base,output=assets(tmp_path)
    class F(FakeBackend): pass
    F.failure=failure
    backend=F()
    original_save=np.savez;original_dump=rec.dump;original_load=rec.load
    def save(path,*args,**kwargs):
        assert backend.in_view;return original_save(path,*args,**kwargs)
    def dump(path,value):
        if Path(path).name in ('blind.json','C.json','L.json','candidates.json','formal_preservation.json'):
            assert backend.in_view
        return original_dump(path,value)
    def load(path):
        if Path(path).name=='attack_truth.json':
            rows=original_load(output/'state.json')
            assert all(v['status'] not in ('STARTED','NOT_STARTED') for v in rows.values())
            for i in range(10):
                if rows[f'v{i:02}/L']['status']=='RETURNED': assert (output/f'received/v{i:02}/candidates.json').is_file()
        assert Path(path).suffix!='npz'
        return original_load(path)
    monkeypatch.setattr(np,'savez',save);monkeypatch.setattr(rec,'dump',dump);monkeypatch.setattr(post,'load',load)
    monkeypatch.setattr(soft,'global_rank',lambda *a:pytest.fail('G must remain zero'))
    result=exp.run(qim,base,output,lambda:backend)
    assert result['denominators']['received_frames']==1644
    counts=original_load(output/'counts.json')
    assert [counts[k]['started'] for k in ('codec_encode','codec_probe','codec_decode','time_read','codebook','G')]==[10,12,12,10,1,0]
    assert counts['C']['returned']==counts['L']['returned']==(9 if failure=='raw' else 10)
    evaluation=original_load(output/'evaluation.json')
    assert len(evaluation['views'])==10 and evaluation['views'][9]['base_soft_evidence']=='NO_SOFT_NEGATIVE_EVIDENCE'
    assert all(v['old_determined_lost']==v['old_determined_changed']==0 for v in evaluation['views'] if v['L_status']=='RETURNED')
    if failure=='close': assert evaluation['resources']['status']=='CLOSE_FAILED' and 'resources' in result['errors']
    before={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    again=exp.run('/missing','/missing',output,lambda:pytest.fail('consumed backend'))
    assert again['physical_calls_this_invocation']==0
    assert before=={str(p):p.read_bytes() for p in output.rglob('*') if p.is_file()}


def test_missing_media_does_not_reserve_cpu(tmp_path):
    output=tmp_path/'out'
    result=exp.run(tmp_path/'media/q.mp4',tmp_path/'media/b.mp4',output,lambda:pytest.fail('not materialized'))
    assert result['status']=='PREFLIGHT_UNRESOLVED' and not output.exists()


def test_base_false_candidates_are_visible(tmp_path):
    qim,base,output=assets(tmp_path)
    exp.run(qim,base,output,FakeBackend)
    folder=output/'received/v05'
    candidate=rec.load(folder/'candidates.json')
    for node in candidate['nodes']: node.update(model_status='DEFINED',best_count=1,candidate_index=5)
    rec.dump(folder/'candidates.json',candidate)
    got=post.evaluate(output,rec.load(output/'state.json'))['views'][5]
    assert got['soft_scoreable']==181 and got['base_false_candidate_outputs']==181
    (output/'private/attack_truth.json').unlink()
    missing=post.evaluate(output,rec.load(output/'state.json'))
    assert missing['status']=='TRUTH_UNRESOLVED'
    assert sum(v['truth_unknown_frames'] for v in missing['views'])==1644


def test_standalone_cpu_cli_consumed_without_scientific_imports(tmp_path):
    out=tmp_path/'out';rec.dump(out/'attempt.json',{})
    command=[sys.executable,'-B','-c',"""
import sys
class Deny:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','psutil','scripts'): raise AssertionError(name)
sys.meta_path.insert(0,Deny())
from experiments.wan_state_clock.b_minimal_l_v1_run import main
main()
""",'--output',str(out),'--config','/absent']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['physical_calls_this_invocation']==0


@pytest.mark.parametrize('damage',['missing','malformed_json','nonobject','missing_count','invalid_count'])
def test_C_metadata_failure_preserves_saved_hard_and_L(tmp_path,monkeypatch,damage):
    # JSON-only saved-output fixture: no raw, codebook, receiver, C or L computation.
    output=tmp_path/'saved';folder=output/'received/v00';length=rec.LENGTHS[0]
    rec.dump(output/'private/attack_truth.json',dict(views=exp.private_plan()))
    rec.dump(output/'resources.json',dict(status='CLOSED',error=None))
    rows=rec.plan()
    for row in rows.values(): row['status']='DEPENDENCY_UNRESOLVED'
    for op in ('hard_receiver','C','L'): rows['v00/'+op]['status']='RETURNED'
    rec.dump(folder/'blind.json',dict(qualification='SUPPORTED_RELATION_SET',comparisons=length*92672,
        nodes=[dict(received_frame=r,output_index=r if r<2 else None,interval=[r,r+1]) for r in range(length)],
        relations=[dict(output_endpoints=[0,1] if r==0 else None) for r in range(length-1)]))
    rec.dump(folder/'candidates.json',dict(nodes=[dict(received_frame=r,candidate_index=r,
        model_status='DEFINED',best_count=1) for r in range(length)]))
    rec.dump(folder/'formal_preservation.json',dict(lost=0,changed=0))
    rec.dump(folder/'C.json',dict(comparisons=length*92672))
    for name in ('codebook','receive'):
        monkeypatch.setattr(hard,name,lambda *a,**k:pytest.fail('saved-only evaluation'))
    for name in ('costs','local_rank','global_rank'):
        monkeypatch.setattr(soft,name,lambda *a,**k:pytest.fail('saved-only evaluation'))
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('no raw or array read'))
    before=post.evaluate(output,rows)
    baseline=before['views'][0]
    assert baseline['hard']==dict(correct=2,wrong=0,unresolved=179,unknown=0)
    assert baseline['hard_relations']==dict(correct=1,wrong=0,unresolved=179,unknown=0)
    assert baseline['old_unresolved_L']['correct']==179
    assert baseline['C_replica_bit_terms']==126*length*92672
    assert baseline['C_replica_bit_terms_kind']=='DENSE_LOGICAL_EQUIVALENT_UPPER_BOUND_NOT_PHYSICAL_OPERATION_COUNT'
    target=folder/'C.json'
    if damage=='missing': target.unlink()
    elif damage=='malformed_json': target.write_text('{')
    elif damage=='nonobject': rec.dump(target,[])
    elif damage=='missing_count': rec.dump(target,{})
    else: rec.dump(target,dict(comparisons='invalid'))
    after=post.evaluate(output,rows);changed=after['views'][0]
    assert changed['C_frame_j'] is None and changed['C_replica_bit_terms'] is None and changed['C_error']
    assert 'hard_error' not in changed and 'L_error' not in changed
    skip={'C_frame_j','C_replica_bit_terms','C_error'}
    assert {k:v for k,v in baseline.items() if k not in skip}=={k:v for k,v in changed.items() if k not in skip}
    assert before['views'][1:]==after['views'][1:]
    assert after['denominators']==before['denominators']
