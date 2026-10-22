"""Controlled scalar/stub wiring only. No real cached tensors/model/media."""
from pathlib import Path
from collections import Counter
import copy,gzip,json
import pytest
pytestmark=pytest.mark.unit
from runtime.wan import b_full_context_local_read_v1 as runtime
from runtime.wan import b_full_context_local_read_evaluation_v1 as post
H=runtime.HOSTS
def packet(j,wrong=False,Hpilot=0):
    physical=[0]*48;data=[0]*32
    if wrong:physical[0]=data[0]=1
    q=[0.0]*48
    if wrong:q[0]=1/(12**.5)
    distances=[16]*64;distances[j-1]=int(wrong)
    decoder=dict(status='CALIBRATION_UNRESOLVED',pilot_P=.8,pilot_H=Hpilot,pilot_detected=None,pilot_status='CALIBRATION_UNRESOLVED',
        physical_bits=physical,unmasked_bits=data,code_distances=distances,nearest_ids=[j-1],nearest=j-1,nearest_distance=int(wrong),
        code_id_candidate=j-1,code_status='LEGAL_ID',tag_label=None,tau=None,technical_failure=False)
    return dict(row=j,projections=q,decoder=decoder)
def references(path):
    native=[];refs=[];local=[];oldrows=[];chips=[];null=[]
    for h in H:
        for j in range(1,45):
            native.append(dict(host=h,row=j,saved_unit=packet(j)))
            for l in range(48):refs.append(dict(host=h,cell=j,chip=l,q_N=0.0,projection_reference_status='AVAILABLE',
                reference_record=dict(bit=0,k=0,target_projection=0.0,q_after_FP32=0.0)))
        for view,phase,offset in post.WINDOWS:
            for r in range(1,11):
                j=offset+r;wrong=j in (1,3);p=packet(j,wrong,Hpilot=2 if j==2 else 0);p['row']=r
                local.append(dict(host=h,view=view,phase=phase,row=r,saved_unit=p))
                final=dict(technical=False,tag_label=None if j==2 else j,status='CODE_ONLY_REJECT' if j==2 else 'ACCEPTED_TAG_CANDIDATE')
                oldrows.append(dict(host=h,view=view,phase=phase,row=r,stored_true_distance=int(wrong),stored_nearest_distance=int(wrong),saved_final_state=final))
                for l in range(48):chips.append(dict(host=h,view=view,phase=phase,row=r,chip=l,bit_error=bool(wrong and l==0),D=0.0,epsilon_L=0.0,w=0.0,signed_same_parity_margin=.5))
        for arm,key in (('OFF','key00'),('OFF','key01'),('P0','key00'),('P0','key01'),('PT','key01')):
            for view in ('clean0','clean1','clean2','clean3','clean4','cross_center'):
                for phase in range(4):
                    for r in range(1,11):null.append(dict(coordinate=dict(host=h,arm=arm,key=key,view=view,phase=phase,row=r),status='SAVED_STATE',saved_state=dict(tag_label=None,pilot_detected=False,status='NO_DETECTION')))
    value=dict(native_packets=native,native_reference=refs,local_packets=local,local_rows=oldrows,local_chips=chips,null_saved_states=null,
        protocol=dict(delta=1/(12**.5)),thresholds=dict(tau=runtime.TAU))
    with gzip.open(path,'wt') as f:json.dump(value,f)
class Stub:
    def __init__(self,out,fail=None,interrupt=False):self.out=out;self.fail=fail;self.interrupt=interrupt;self.counts=Counter();self.blocks={'encoder':0,'decoder':0}
    def started(self,k):
        assert json.loads((self.out/'receipts'/(k+'.json')).read_text())['status']=='STARTED'
        assert (self.out/'attempt.json').exists()
        assert not (self.out/'post_reference_access.json').exists()
    def load_vae(self):self.started('vae');self.counts['vae']+=1;return dict(stub=True)
    def load_received(self,path):
        h=Path(path).parts[-4];self.started(h+'/load');self.counts['load']+=1
        if h==self.fail:raise OSError('fixture cached RGB unavailable')
        return h
    def prepare_full(self,pixels):self.started(pixels+'/prepare');self.counts['prepare']+=1;return pixels
    def encode_full(self,rgb):
        self.started(rgb+'/E');self.counts['E']+=1
        if self.interrupt:raise KeyboardInterrupt('fixture interrupted attempt')
        self.blocks['encoder']+=46;return rgb
    def save_tensor(self,path,value):Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_bytes(b'stub latent')
    def read_raw(self,z,key):
        self.started(z+'/raw');self.counts['raw']+=1
        return dict(rows=[packet(j,j in (2,3)) for j in range(1,45)],technical_rows=0,expected_rows=44,truth_used=False,message_used=False)
    def release_vae(self):self.counts['release']+=1
def test_fixed_calls_order_and_bit_truth_vs_tag(tmp_path,monkeypatch):
    out=tmp_path/'run';ref=tmp_path/'reference.gz';references(ref);stub=Stub(out)
    original_open=gzip.open;seen=[]
    def spy_open(*a,**kw):
        assert (out/'markers/raw_complete.json').exists()
        assert (out/'markers/blind_complete.json').exists()
        assert json.loads((out/'markers/blind_complete.json').read_text())['full_rows']==176
        seen.append('reference opened after full blind readback')
        return original_open(*a,**kw)
    monkeypatch.setattr(post.gzip,'open',spy_open)
    original_finish=runtime.method.finalize
    def final_spy(*a,**kw):
        assert (out/'markers/raw_complete.json').exists()
        for h in H:assert len(json.loads((out/'raw'/h/'key00.json').read_text())['rows'])==44
        return original_finish(*a,**kw)
    monkeypatch.setattr(runtime.method,'finalize',final_spy)
    result=runtime.run(tmp_path/'source',out,ref,backend=stub)
    assert result['status']=='COMPLETED_POSITIVE_CHANNEL_DESCRIPTION_ONLY'
    assert stub.counts==dict(vae=1,load=4,prepare=4,E=4,raw=4,release=1)
    assert len(seen)==1
    ledger=json.loads((out/'ledger.json').read_text());assert all(r['attempts']==1 and r['status']=='RETURNED' for r in ledger.values())
    assert sum(r.get('internal_blocks',{}).get('encoder',0) for r in ledger.values())==184
    pairchips=json.loads((out/'full_local_chip_pairs.json').read_text());assert len(pairchips)==9600
    one={x['source_cell']:x for x in pairchips if x['host']==H[0] and x['view']=='clean0' and x['chip']==0}
    assert [one[j]['truth_transition'] for j in (1,2,3,4)]==['FULL_ONLY_CORRECT_ADDED','LOCAL_ONLY_CORRECT_LOST','BOTH_WRONG','BOTH_CORRECT_RETAINED']
    assert one[3]['saved_bit_agreement'] is True and one[3]['full_correct'] is False and one[3]['local_correct'] is False
    pairs=json.loads((out/'full_local_pairs.json').read_text());assert len(pairs)==200
    assert Counter(x['anchor'] for x in pairs)=={'PREFIX40':40,'OTHER160':160}
    assert len({(x['host'],x['source_cell']) for x in pairs})==176
    before=stub.counts.copy();again=runtime.run('/absent',out,'/absent',backend=stub)
    assert again['status']=='READ_ONLY_EXISTING_ATTEMPT' and stub.counts==before
def test_missing_reference_does_not_block_E_and_keeps_post_slots(tmp_path):
    out=tmp_path/'run';stub=Stub(out)
    result=runtime.run('/source',out,tmp_path/'missing.gz',backend=stub)
    assert stub.counts['E']==4 and result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert len(json.loads((out/'full_local_pairs.json').read_text()))==200
    assert len(json.loads((out/'full_local_chip_pairs.json').read_text()))==9600
    assert result['post_unresolved_counts']==dict(full=176,pairs=200,historical_nulls=4800)
def test_cached_RGB_failure_retains44_and50_and_no_fallback(tmp_path):
    ref=tmp_path/'ref.gz';references(ref);out=tmp_path/'run';stub=Stub(out,fail=H[1])
    result=runtime.run('/source',out,ref,backend=stub)
    assert result['technical_full_rows']==44 and stub.counts['E']==3 and stub.counts['load']==4
    raw=json.loads((out/'raw'/H[1]/'key00.json').read_text())
    assert len(raw['rows'])==44 and all(len(x['decoder']['code_distances'])==64 for x in raw['rows'])
    pairs=json.loads((out/'full_local_pairs.json').read_text())
    assert sum(x['status']=='UNRESOLVED' for x in pairs)==50
    chips=json.loads((out/'full_local_chip_pairs.json').read_text())
    assert sum(x['truth_transition']=='UNDEFINED' for x in chips)==2400
def test_interrupt_consumes_attempt_and_reference_unopened(tmp_path):
    out=tmp_path/'run';stub=Stub(out,interrupt=True)
    result=runtime.run('/source',out,'/absent',backend=stub)
    assert result['status']=='INTERRUPTED' and stub.counts['E']==1
    assert not (out/'post_reference_access.json').exists()
    assert len(json.loads((out/'fixed_rows.json').read_text()))==176
    assert len(json.loads((out/'fixed_pair_status.json').read_text()))==200
    counts=stub.counts.copy();runtime.run('/absent',out,'/absent',backend=stub);assert stub.counts==counts

def test_final_failure_blocks_private_reference_and_readonly_reentry(tmp_path,monkeypatch):
    out=tmp_path/'run';ref=tmp_path/'ref.gz';references(ref);stub=Stub(out)
    original=runtime.method.finalize;calls=[]
    def fail_first(*a,**kw):
        calls.append(1)
        if len(calls)==1:raise ValueError('fixture final failed')
        return original(*a,**kw)
    def forbid_reference(*a,**kw):raise AssertionError('private reference loader must not enter')
    monkeypatch.setattr(runtime.method,'finalize',fail_first)
    monkeypatch.setattr(post.gzip,'open',forbid_reference)
    result=runtime.run('/source',out,ref,backend=stub)
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED' and stub.counts['E']==4
    assert not (out/'markers/blind_complete.json').exists() and not (out/'post_reference_access.json').exists()
    ledger=json.loads((out/'ledger.json').read_text())
    assert ledger[H[0]+'/final']['status']=='FAILED' and ledger['blind_readback']['status']=='NOT_ENTERED'
    assert len(json.loads((out/'fixed_pair_status.json').read_text()))==200
    assert len(json.loads((out/'fixed_pair_chip_status.json').read_text()))==9600
    before=stub.counts.copy();again=runtime.run('/absent',out,'/absent',backend=stub)
    assert again['status']=='READ_ONLY_EXISTING_ATTEMPT' and stub.counts==before
