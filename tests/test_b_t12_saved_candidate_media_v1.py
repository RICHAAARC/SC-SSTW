"""Artificial/fake checks only; no saved t12 arrays, weights, codec or GPU."""
from pathlib import Path
from collections import Counter
import ast
import json
import importlib.util
import numpy as np
import pytest
import torch
from experiments.wan_state_clock import b_t12_saved_candidate_media_v1_records as rec
from experiments.wan_state_clock.b_t12_saved_candidate_media_v1_experiment import run
from main.tube_state import b_paper_cohort_v1 as protocol
from main.tube_state.b_joint_time_relative_validation_v1 import construction_maps

pytestmark=pytest.mark.unit


class Fake:
    output=None
    interrupt=False
    calls=Counter()
    def __init__(self,model,loads): self.loads=loads
    def native(self,candidate,host,head,output,event,dump):
        self.calls['native']+=1
        for _ in range(46): event('head_replay',False);event('head_replay',True)
        return dict(status='FINITE_COMPLETE',actual_budget='FAIL',maximum_budget_excess=2e-6)
    def quantize(self,path): return np.zeros((181,1,1,3),np.uint8)
    def encode(self,pixels,path):
        self.calls['codec_encode']+=1;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('artificial, not video')
        return dict(fake=True)
    def probe(self,path,n):
        self.calls['probe']+=1
        if not path.exists(): raise FileNotFoundError(str(path))
        return dict(fake=True,frames=n)
    def decode(self,path,n,kind):
        self.calls['decode_'+kind]+=1
        index=int(path.parent.name[5:]) if path.parent.name.startswith('input') else 0
        return np.full((n,1,1,3) if kind=='RGB' else (n,1,1),index,np.uint8)
    def read(self,y): return dict(n=np.array([len(y)]))
    def hard(self,raw): return dict(nodes=[{} for _ in range(int(raw['n'][0]))])
    def relative(self,packet,uid):
        item=next(i for i in rec.items() if i['id']==uid);mapping=construction_maps()[item['view']]
        nodes=[dict(received_frame=r,conditional_candidates=[n],conditional_position=n,status='UNCHANGED_SINGLETON') for r,n in enumerate(mapping)]
        edges=[dict(exact_singleton_pair=True,pairs=[dict(left=a,right=b)]) for a,b in zip(mapping,mapping[1:])]
        return dict(conditional_nodes=nodes,conditional_edges=edges)
    def encode_view(self,pixels,start):
        assert (self.output/'plans_frozen.json').exists()
        assert not (self.output/'truth_access.json').exists()
        self.calls['E41']+=1
        if self.interrupt:
            type(self).interrupt=False;raise KeyboardInterrupt('artificial interrupted E41')
        return torch.tensor([int(pixels[0,0,0,0]),start])
    def project(self,latent): return latent.tolist()
    def bits(self,projection,uid,key):
        assert not (self.output/'truth_access.json').exists()
        index,start=projection;item=protocol.inputs()[index];mapping=construction_maps()[item['view']];regions={}
        for region in ('R0','R1'):
            cells=[]
            for row in range(1,11):
                t=mapping[start+4*row-3];cell=(t+3)//4;group=min(3,max(0,(cell-1)//11));eid=f'{uid}/row{row}/{region}'
                bits=[dict(local_bit=b,value=protocol.query_bits('q0')[group*8+b]^(key=='key01'),count=8,status='AVAILABLE',key=key,evidence_id=eid) for b in range(8)]
                cells.append(dict(local_row=row,evidence_id=eid,observed_region_id=region,carrier_region_id=region,status='AVAILABLE',bits=bits))
            regions[region]=dict(status='AVAILABLE',observed_region_id=region,carrier_region_id=region,local_cells=cells)
        return dict(status='AVAILABLE',kind='E41',view_id=uid,key=key,local_rows=list(range(1,11)),regions=regions)
    def close(self): return dict(fake=True)
    def prepare_payload(self): pass


def config(tmp_path):
    value=dict(rec.DEFAULTS,**{k:str(tmp_path/k) for k in ('candidate','original','host','output')})
    for k in ('candidate','original','host'): Path(value[k]).mkdir()
    rec.write(Path(value['candidate'])/'result.json',dict(physical_outcome='ACTUAL_BUDGET_UNRESOLVED'))
    rec.write(Path(value['original'])/'private/writer.json',dict(bits=protocol.query_bits('q0')))
    rec.write(Path(value['original'])/'private/attack_truth.json',construction_maps())
    for item in rec.items():
        path=Path(value['original'])/'media'/item['reference']/'received.mp4';path.parent.mkdir(parents=True);path.write_text('fake reference')
    Fake.output=Path(value['output']);Fake.calls=Counter();Fake.interrupt=False
    return value


def test_full_fake_and_completed_readonly(tmp_path):
    cfg=config(tmp_path)
    result=run(cfg,backend_factory=Fake,preflight_check=lambda _:dict(status='INPUTS_READABLE'))
    assert result['engineering_status']=='COMPLETED'
    counts=result['counts']
    for name,count in dict(codec_encode=6,codec_probe=6,codec_RGB=6,codec_Y=5,time_read=5,time_hard=5,time_relative=5,time_plan=5,E41=60,project=60,bits=120,local=45,aggregate=15,claim=20,quality=5,ref_codec_probe=5,ref_codec_RGB=5).items():
        assert counts[name]==dict(planned=count,started=count,returned=count)
    out=Path(cfg['output']);evaluation=rec.read(out/'evaluation_summary.json')
    assert evaluation['native']['actual_budget']=='FAIL'
    assert evaluation['system_observation']!='T12_KNOWN_ERROR_LOCAL_REPAIR_SUPPORTED'  # No old paired evidence.
    evidence=rec.read(out/'evidence/input10/key00.json')
    assert not any(e.get('kind')=='SUPPORT_RECEIPT' for e in evidence['input_errors'])
    supported=[r for r in rec.read(out/'final/input10/B_NEW.json')['slots'] if r.get('supporting_refs')]
    assert supported
    for row in supported:
        for ref in row['supporting_refs']:
            assert (out/ref['original_bits_file']).is_file()
            assert all(rec.read(out/'operations.json')[op]['status']==status=='RETURNED'
                       for op,status in ref['original_operation_statuses'].items())
    for name,count in dict(node_results=822,edge_results=817,physical_words=8220,derived_F=4110,raw_terminal_slots=19200,local_terminal_slots=1440,payload_pairs=480).items():
        assert sum(1 for _ in (out/'post'/(name+'.jsonl')).open())==count
    before=Fake.calls.copy();run(cfg,backend_factory=lambda *a:pytest.fail('completed package loaded backend'))
    assert Fake.calls==before


def test_interrupted_call_consumed_later_calls_continue(tmp_path):
    cfg=config(tmp_path);Fake.interrupt=True
    with pytest.raises(KeyboardInterrupt): run(cfg,backend_factory=Fake,preflight_check=lambda _:dict(status='INPUTS_READABLE'))
    out=Path(cfg['output']);ops=rec.read(out/'operations.json')
    assert ops['input10/E41/a0/E41']['status']=='INTERRUPTED_CONSUMED'
    assert ops['input10/E41/a1/E41']['status']=='NOT_STARTED'
    run(cfg,backend_factory=Fake)
    ops=rec.read(out/'operations.json')
    assert ops['input10/E41/a0/E41']['started']==1 and ops['input10/E41/a0/E41']['returned']==0
    assert ops['input14/E41/a136/E41']['status']=='RETURNED'
    assert Fake.calls['native']==1 and Fake.calls['codec_encode']==6 and Fake.calls['E41']==60


def test_missing_reference_preserves_candidate_and_quality_unknown(tmp_path):
    cfg=config(tmp_path);(Path(cfg['original'])/'media/input7/received.mp4').unlink()
    run(cfg,backend_factory=Fake,preflight_check=lambda _:dict(status='INPUTS_READABLE'))
    out=Path(cfg['output']);ops=rec.read(out/'operations.json')
    assert ops['input12/ref/probe']['status']=='FAILED_CONSUMED'
    assert ops['input12/quality']['status']=='DEPENDENCY_UNRESOLVED'
    assert ops['input12/key00/q0']['status']=='RETURNED'
    assert rec.read(out/'evaluation_summary.json')['quality'][2]['comparison_status']=='UNKNOWN'


def test_receipt_unknown_started_and_paths(tmp_path):
    cfg=config(tmp_path);book=rec.Book(cfg)
    book.rows['native'].update(status='STARTED',started=1);book.save();book.close()
    book=rec.Book(cfg)
    assert book.call('native',lambda:pytest.fail('unknown started repeated')) is None
    assert book.rows['native']['status']=='INTERRUPTED_CONSUMED'
    book.json('quantize','artificial.json',lambda:dict(fake=True));(book.root/'artificial.json').unlink()
    assert book.json('quantize','artificial.json',lambda:pytest.fail('returned operation repeated')) is None
    assert not book.available('quantize')
    assert book.call('first/encode',lambda:pytest.fail('missing artifact used'),dependencies=('quantize',)) is None
    book.close()
    with pytest.raises(ValueError): rec.validate_paths(dict(cfg,output=cfg['original']+'/new'))
    with pytest.raises(ValueError): rec.validate_paths(dict(cfg,output=str(tmp_path/'Video-WM_Result/new')))
    assert rec.preflight(cfg)['physical_started']==0


def test_tiny_original_replay_modified_tail():
    from runtime.wan.b_framewise_qim_v1_operator import replay_head
    from main.tube_state import b_framewise_qim_v1 as p
    from torch.nn import functional as F
    class Causal(torch.nn.Conv3d):
        def __init__(self):
            super().__init__(1,3,3,bias=True);self._padding=(1,1,1,1,2,0)
            torch.nn.init.constant_(self.weight,.01);torch.nn.init.zeros_(self.bias)
        def forward(self,x,tail):
            if tail is not None: x=torch.cat((tail,x),dim=2)
            pad=(1,1,1,1,2 if tail is None else 2-tail.shape[2],0)
            return F.conv3d(F.pad(x,pad),self.weight,self.bias)
    conv=Causal();a=torch.zeros((5,10,1,7,9),dtype=torch.float64);a[0,0,0,0,0]=1
    active=torch.ones((5,10),dtype=torch.bool)
    features=[torch.zeros((1,1,1,320,512)),torch.zeros((1,1,4,320,512))]
    result=list(replay_head(conv,features,a,active));full=torch.cat([r for _,r in result])
    assert [s for s,_ in result]==[0,1]
    assert torch.count_nonzero(full[1]-.5)>0 and torch.count_nonzero(full[2]-.5)>0
    assert torch.equal(full[3:],torch.full_like(full[3:],.5))


def test_notebook_sync_and_stdlib_cli(tmp_path):
    from scripts.build_b_t12_saved_candidate_media_v1_notebook import build
    root=Path(__file__).resolve().parents[1];(tmp_path/'notebooks').mkdir()
    generated=json.loads(build(tmp_path).read_text());actual=json.loads((root/'notebooks/b_t12_saved_candidate_media_v1.ipynb').read_text())
    assert generated==actual
    cells=[c for c in actual['cells'] if c['cell_type']=='code']
    assert ''.join(cells[0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in cells: ast.parse(''.join(cell['source']));assert cell['outputs']==[] and cell['execution_count'] is None


def test_transition_keeps_unresolved_and_regression():
    from experiments.wan_state_clock.b_t12_saved_candidate_media_v1_evaluation import transition,indexed
    assert transition('WRONG','CORRECT')['repair']
    assert not transition('WRONG','UNRESOLVED')['repair']
    assert transition('WRONG','UNRESOLVED')['error_removed_unresolved']
    assert not transition('CORRECT','UNKNOWN')['old_correct_lost']
    assert transition('CORRECT','UNKNOWN')['old_correct_retention_unknown']
    assert transition('CORRECT','UNRESOLVED')['old_correct_lost']
    assert indexed([None],'slot',1,uid='input10') is None


def test_private_transport_fragments(tmp_path):
    from runtime.wan.b_t12_saved_candidate_media_v1 import prediction_file
    candidate=tmp_path/'candidate';candidate.mkdir();original=tmp_path/'artificial.npy'
    np.save(original,np.arange(18,dtype=np.float64).reshape(2,3,3))
    content=original.read_bytes();cut=71
    for index,fragment in enumerate((content[:cut],content[cut:])):
        (candidate/f'r_predicted_packed.npy.part{index:03}').write_bytes(fragment)
    restored=prediction_file(candidate,tmp_path/'native')
    assert np.array_equal(np.load(restored,allow_pickle=False),np.arange(18).reshape(2,3,3))
    assert restored.read_bytes()==content


def test_corrupt_torch_artifact_retains_return_receipt(tmp_path):
    cfg=config(tmp_path);book=rec.Book(cfg);key='input10/E41/a0/E41';target=book.root/'bad.pt'
    target.write_bytes(b'PK\x03\x04broken zip');book.rows[key].update(status='RETURNED',started=1,returned=1)
    assert book.call(key,lambda:pytest.fail('consumed E41 repeated'),load=lambda:torch.load(target,weights_only=True)) is None
    assert book.rows[key]['returned']==1 and book.rows[key]['status']=='RETURNED'
    assert book.observation_status(key)=='ARTIFACT_UNAVAILABLE' and not book.available(key)
    book.close()


def test_optional_history_and_failed_shared_load(tmp_path):
    from runtime.wan.b_t12_saved_candidate_media_v1 import Backend
    cfg=config(tmp_path);(Path(cfg['candidate'])/'result.json').unlink()
    required=[Path(cfg['candidate'])/'coefficients.pt',Path(cfg['candidate'])/'r_predicted_packed.npy',
        Path(cfg['original'])/'private/head.pt',Path(cfg['host'])/'BASE_preclip.npy']
    required += [Path(cfg['host'])/'features'/f'{s:03}.pt' for s in rec.STARTS]
    for p in required: p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'artificial')
    assert rec.preflight(cfg)['status']=='INPUTS_READABLE'
    calls=[];receipts=[]
    class Failure:
        def load_vae(self): calls.append(1);raise RuntimeError('artificial missing weights')
    def load_receipt(done,value):
        if not done: receipts.append(dict(status='STARTED'));return len(receipts)-1
        receipts[value['token']].update(value)
    backend=object.__new__(Backend);backend.wan=Failure();backend.loaded=False;backend.load_receipt=load_receipt
    for _ in range(2):
        with pytest.raises(RuntimeError): backend.prepare_payload()
    assert len(calls)==len(receipts)==1 and receipts[0]['status']=='FAILED'


def test_malformed_post_preserves_all_slots_and_addresses(tmp_path):
    from experiments.wan_state_clock.b_t12_saved_candidate_media_v1_evaluation import evaluate
    cfg=config(tmp_path);out=Path(cfg['output']);ops=rec.plan()
    rec.write(out/'blind_terminal.json',dict(status='ALL_BLIND_TERMINAL'))
    def saved(key,path,value):
        ops[key].update(status='RETURNED',started=1,returned=1);rec.write(out/path,value)
    relative=Fake(None,None).relative({},'input10');relative['conditional_edges'][0]['pairs']=[None]
    saved('input10/relative','blind/input10/relative.json',relative)
    saved('input10/plan','plans/input10.json',{})
    saved('input10/key00/q0','claims/input10/key00/q0.json',{})
    slots=[dict(slot=i,value=i%2,status='PARTIAL_EVIDENCE_READ',selected_refs=[],supporting_refs=[],missing_refs=[],conflict_refs=[]) for i in reversed(range(32))]
    saved('input10/B_NEW/q0','local/input10/B_NEW/q0.json',dict(slots=slots))
    saved('input10/B_NEW/aggregate','final/input10/B_NEW.json',dict(slots=slots))
    rec.write(Path(cfg['original'])/'state.json',{'input10/B_NEW/aggregate':dict(status='RETURNED')})
    rec.write(Path(cfg['original'])/'post/input10/B_NEW.json',[dict(slot=i,value=i%2,correct_supported=True) for i in reversed(range(32))])
    rec.write(out/'raw_bank.json',[None])
    rec.write(Path(cfg['candidate'])/'result.json',[])
    result=evaluate(out,cfg,ops)
    assert result['claims']['UNKNOWN']==20 and result['cpu_saved_result'] is None
    assert result['edge_statuses']['UNKNOWN']>=1
    assert result['B_old_correct_supported_lost']==0 and result['B_old_correct_retention_unknown']==32
    for name,count in dict(node_results=822,edge_results=817,local_terminal_slots=1440,payload_pairs=480).items():
        rows=[json.loads(line) for line in (out/'post'/(name+'.jsonl')).read_text().splitlines()];assert len(rows)==count
        if name=='local_terminal_slots':
            assert rows[0]['observation']['value']==0 and rows[1]['observation']['value']==1
            assert rows[32]['observation_available'] is False
        if name=='payload_pairs': assert [r['new']['value'] for r in rows[:32]]==[i%2 for i in range(32)]


def test_shared_model_failure_does_not_start_e41(tmp_path):
    cfg=config(tmp_path)
    class NoModel(Fake):
        def prepare_payload(self): raise RuntimeError('artificial shared dependency unavailable')
    result=run(cfg,backend_factory=NoModel,preflight_check=lambda _:dict(status='INPUTS_READABLE'))
    assert result['counts']['E41']==dict(planned=60,started=0,returned=0)
    assert Fake.calls['E41']==0
    assert rec.read(Path(cfg['output'])/'operations.json')['post']['status']=='RETURNED'


def test_resume_keeps_derived_data_but_marks_corrupt_raw_unknown(tmp_path):
    cfg=config(tmp_path)
    class InterruptLater(Fake):
        def encode_view(self,pixels,start):
            if self.calls['E41']==12:
                self.calls['E41']+=1;raise KeyboardInterrupt('artificial stop after first-view final')
            return super().encode_view(pixels,start)
    with pytest.raises(KeyboardInterrupt): run(cfg,backend_factory=InterruptLater,preflight_check=lambda _:dict(status='INPUTS_READABLE'))
    out=Path(cfg['output']);derived=out/'final/input10/B_NEW.json';before=derived.read_text()
    (out/'bank/input10/E41/a0/latent.pt').write_bytes(b'PK\x03\x04broken zip')
    run(cfg,backend_factory=Fake)
    assert derived.read_text()==before
    assert rec.read(out/'raw_bank.json')['input10/E41/a0']['E41']=='ARTIFACT_UNAVAILABLE'
    op=rec.read(out/'operations.json')['input10/E41/a0/E41']
    assert op['started']==op['returned']==1 and op['status']=='RETURNED'
    assert any(error['kind']=='SUPPORT_RECEIPT' for error in rec.read(out/'evidence/input10/key00.json')['input_errors'])
    assert Fake.calls['E41']==60 and Fake.calls['native']==1


def test_missing_writer_preserves_observed_final_bits(tmp_path):
    from experiments.wan_state_clock.b_t12_saved_candidate_media_v1_evaluation import evaluate
    cfg=config(tmp_path);out=Path(cfg['output']);(Path(cfg['original'])/'private/writer.json').unlink()
    rec.write(out/'blind_terminal.json',dict(status='ALL_BLIND_TERMINAL'));ops=rec.plan()
    slots=[dict(slot=i,value=i%2,status='PARTIAL_EVIDENCE_READ',selected_refs=[],supporting_refs=[]) for i in reversed(range(32))]
    ops['input10/B_NEW/aggregate'].update(status='RETURNED',started=1,returned=1)
    rec.write(out/'final/input10/B_NEW.json',dict(slots=slots));evaluate(out,cfg,ops)
    rows=[json.loads(line) for line in (out/'post/payload_pairs.jsonl').read_text().splitlines()]
    assert len(rows)==480 and [r['new']['value'] for r in rows[:32]]==[i%2 for i in range(32)]
    for row in rows[:32]:
        assert row['new']['status']=='PARTIAL_EVIDENCE_READ'
        assert row['new']['correct_supported_bounds']==[0,1]
        assert row['new']['literal_correct'] is row['new']['correct_supported'] is row['new']['wrong_attribution'] is None
    for name,count in [('node_results',822),('edge_results',817),('local_terminal_slots',1440)]:
        assert len((out/'post'/(name+'.jsonl')).read_text().splitlines())==count


def test_resume_rebuilds_invalid_raw_index_without_physical_retry(tmp_path):
    cfg=config(tmp_path);book=rec.Book(cfg)
    book.rows['native'].update(status='FAILED_CONSUMED',started=1,returned=0,error='artificial prior failure');book.save();book.close()
    out=Path(cfg['output']);rec.write(out/'raw_bank.json',[])
    class NoNative(Fake):
        def native(self,*args): pytest.fail('consumed native repeated')
        def encode_view(self,*args): pytest.fail('E41 without media')
    run(cfg,backend_factory=NoNative)
    assert len(rec.read(out/'raw_bank.json'))==60
    assert list((out/'recovery').glob('raw_bank_error_*.json'))
    assert rec.read(next((out/'recovery').glob('raw_bank_invalid_*.json')))==[]
    ops=rec.read(out/'operations.json');assert ops['native']['started']==1 and ops['post']['status']=='RETURNED'
    for name,count in [('node_results',822),('edge_results',817),('local_terminal_slots',1440),('payload_pairs',480)]:
        assert len((out/'post'/(name+'.jsonl')).read_text().splitlines())==count
