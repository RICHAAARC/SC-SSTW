"""Controlled synthetic CPU/static/stub only; no real cache/media/model reads."""
from pathlib import Path
from collections import Counter
import copy,inspect,json
import pytest
from main.tube_state import b_block2_carrier_v1 as m
from runtime.wan import b_block2_carrier_v1 as r
pytestmark=pytest.mark.unit

def test_books_only_vectors_hmac_budget_and_same_chip_correlation(monkeypatch):
    m.book.cache_clear();m.core.book.cache_clear();calls=[];original=m.hmac.new
    def count(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(m.hmac,'new',count)
    report=m.initialize_books()
    assert len(calls)==7872 and len(report['rows'])==96
    assert report['dots_64D']==96 and report['coordinate_products']==6144
    for key in m.KEYS.values():
        old=m.core.book(key);new=m.book('BLOCK2-16',key)
        assert m.book('RANDOM64',key) is old
        assert new.dithers is old.dithers and new.pilot is old.pilot and new.mask is old.mask
        for u in new.vectors:
            assert sum(u)==0 and sum(x*x for x in u)==1 and set(u)=={-.125,.125}
            signs=[]
            for y in range(0,8,2):
                for x in range(0,8,2):
                    q=[u[8*(y+dy)+x+dx] for dy in range(2) for dx in range(2)]
                    assert len(set(q))==1;signs.append(q[0])
            assert signs.count(.125)==8
    assert len(calls)==7872 # no regeneration from the second family lookup

def test_explicit_tensor_book_native_and_default_unchanged(monkeypatch):
    import torch
    torch.set_num_threads(1)
    from runtime.wan import b_block2_carrier_tensor_v1 as t
    from runtime.wan import b_time_label_tensor_v1 as old
    torch.manual_seed(15);source=torch.randn(1,16,46,40,64,dtype=torch.float32)
    saved=source.clone();zR,receiptR=t.construct(source,'RANDOM64','fixture')
    legacy,_=old.construct(source,'fixture')
    assert torch.equal(zR,legacy) and torch.equal(source,saved)
    def forbid(*a,**kw):raise AssertionError('implicit default book adapter used')
    monkeypatch.setattr(m.core,'encode_label',forbid);monkeypatch.setattr(m.core,'decode_blocks',forbid)
    zB,receiptB=t.construct(source,'BLOCK2-16','fixture')
    zR2,_=t.construct(source,'RANDOM64','fixture');zB2,_=t.construct(source,'BLOCK2-16','fixture')
    assert torch.equal(zR,zR2) and torch.equal(zB,zB2)
    mask=torch.zeros_like(source,dtype=torch.bool)
    for y0,y1,x0,x1 in old.ROIS:mask[0,4:16,1:45,y0:y1,x0:x1]=True
    corrupted=zB.clone();corrupted[0,5,2,0,0]+=1
    with pytest.raises(ValueError,match='protected'):t.validate_protected(source,corrupted,mask)
    assert t.project_rows(zB,'BLOCK2-16','fixture')[0]!=t.project_rows(zB,'RANDOM64','fixture')[0]
    for z,receipt,family in ((zR,receiptR,'RANDOM64'),(zB,receiptB,'BLOCK2-16')):
        assert receipt['payload_channels_bitwise_equal'] and receipt['guard_layers_bitwise_equal'] and receipt['outside_support_bitwise_equal']
        assert len(receipt['rows'])==44 and sum(len(x['records']) for x in receipt['rows'])==2112
        assert receipt['actual_L2']**2==pytest.approx(receipt['actual_FP32_energy']) and receipt['FP32_correction_attempts']==0
        raw=m.raw_table(t.project_rows(z,family,'fixture'),family,'fixture');final=m.finalize(raw)
        assert [x['decoder']['tag_label'] for x in final['rows']]==list(range(1,45))
    assert not torch.equal(zR,zB)
    assert tuple(inspect.signature(t.project_rows).parameters)==('z','family','key')

def test_payload_shared_fft_counter_and_strict_zero(monkeypatch):
    import torch
    from main.tube_state import b_legacy_position_margin_v1 as p
    torch.set_num_threads(1);counts=Counter();fft=torch.fft.fft2
    def count_fft(*a,**kw):counts['FFT']+=1;return fft(*a,**kw)
    monkeypatch.setattr(torch.fft,'fft2',count_fft)
    orig=p.Counter;orig2=p.payload.Counter
    def count_counter(*a,**kw):counts['Counter']+=1;return orig(*a,**kw)
    monkeypatch.setattr(p,'Counter',count_counter);monkeypatch.setattr(p.payload,'Counter',count_counter)
    spectrum=p.payload_spectrum(torch.zeros(1,16,46,40,64,dtype=torch.float32))
    vals=[p.payload_from_spectrum(spectrum,key) for key in m.KEYS.values()]
    assert counts==dict(FFT=1,Counter=768)
    for v in vals:
        assert v['decoded_bits']==[0]*32
        assert len(v['local_cells'])==44
        assert all(x==1 for c in v['zero_coefficient_mask'] for t in c for x in t)
        assert all(x==-1 for c in v['signed_votes'] for t in c for x in t)

class Stub:
    def __init__(self,out,fail=None,interrupt=False):self.out=out;self.fail=fail;self.interrupt=interrupt;self.calls=Counter();self.blocks={'encoder':0,'decoder':0};self.spectra=[]
    def check(self,kind):
        rows=json.loads((self.out/'ledger.json').read_text());active=[x for x in rows.values() if x['status']=='STARTED']
        assert len(active)==1 and active[0]['kind']==kind
        assert (self.out/'attempt.json').is_file()
        receipt=json.loads((self.out/'receipts'/(active[0]['slot']+'.json')).read_text());assert receipt['status']=='STARTED'
        if not kind.startswith('post_'):assert not (self.out/'post_reference_access.json').exists()
        self.calls[kind]+=1
        if self.fail==kind and self.calls[kind]==1:raise OSError('synthetic '+kind+' failure')
    def load_vae(self):
        self.check('VAE_load')
        if self.interrupt:raise KeyboardInterrupt('synthetic stop')
        return dict(stub=True)
    def release_vae(self):pass
    def load_native(self,path):
        self.check('cached_native_load')
        if self.fail=='P0' and m.HOSTS[0] in str(path) and path.name=='P0_terminal.pt':raise OSError('synthetic unreadable pre-tag P0')
        return dict(host=path.parts[-3],arm=path.name.split('_')[0])
    def construct_variant(self,z,family,key):self.check('metadata_construct');return dict(z,family=family),dict(status='SYNTHETIC_CONSTRUCT',payload_rewritten=False)
    def save_tensor(self,path,v):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v))
    def load_tensor(self,path):return json.loads(Path(path).read_text())
    def read_variant(self,z,family,key):
        self.check('tag_raw_44')
        if self.fail=='raw' and self.calls['tag_raw_44']==1:raise ValueError('synthetic raw error')
        d=dict(status='CALIBRATION_UNRESOLVED',technical_failure=False,physical_bits=[0]*48,unmasked_bits=[0]*32,pilot_P=-.5,pilot_H=5,pilot_detected=None,pilot_status='CALIBRATION_UNRESOLVED',code_distances=[10]*64,nearest_ids=list(range(64)),nearest=None,nearest_distance=10,code_status='CODE_TIE',code_id_candidate=None,tag_label=None)
        return dict(expected_rows=44,technical_rows=0,rows=[dict(row=i,projections=[0.]*48,decoder=copy.deepcopy(d)) for i in range(1,45)])
    def load_received(self,path):self.check('cached_OFF_readback_load');return dict(pixels=True)
    def decode_PT(self,z):self.check('D181');self.blocks['decoder']+=46;return dict(preclip=dict(stub=True),float_rgb=dict(stub=True))
    def quantize(self,v):self.check('RGB8_quantize');return dict(stub=True)
    def save_mp4(self,v,path):self.check('MP4_encode');return dict(stub=True)
    def probe(self,path):self.check('MP4_probe');return dict(stub=True)
    def read_media(self,path,probe,n):self.check('MP4_decode');return dict(stub=True)
    def prepare_full(self,v):self.check('RGB_input_prepare');return dict(stub=True)
    def encode_full(self,v):self.check('E181');self.blocks['encoder']+=46;return dict(stub=True)
    def payload_spectrum(self,z):
        self.check('payload_FFT');v=dict(stub=True);self.spectra.append([v,[]]);return v
    def payload_votes(self,v,key):
        self.check('payload_keyed_read')
        assert v is self.spectra[-1][0];self.spectra[-1][1].append(key)
        return dict(status='READ',decoded_bits=[0]*32,local_cells=[dict(bits=[dict(decoded=0) for j in range(8)]) for t in range(44)],signed_votes=[[[-1]*240 for t in range(44)] for c in range(4)],zero_coefficient_mask=[[[0]*240 for t in range(44)] for c in range(4)])
    def preclip(self,v):self.check('post_clipping');assert (self.out/'markers/blind_complete.json').is_file();return dict(status='SYNTHETIC')
    def quality(self,a,b,frames,meaning):
        self.check('post_paired_quality');assert (self.out/'markers/blind_complete.json').is_file()
        return dict(status='SYNTHETIC',frame_denominator=181,adjacent_denominator=180,frames=[dict(frame=i) for i in frames],adjacent=[dict(frame_before=i,frame_after=i+1) for i in range(180)])

def fixture_source(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'protocol.json').write_text(json.dumps(dict(config=dict(writer=dict(bits=[0]*32)))));return source

def test_complete_stub_fixed_budget_blind_order_and_readonly(tmp_path,monkeypatch):
    source=fixture_source(tmp_path);out=tmp_path/'output';backend=Stub(out)
    from runtime.wan import b_block2_carrier_evaluation_v1 as post
    orig=post.read;opened=[]
    def guarded(p):
        if Path(p).name=='protocol.json':
            assert (out/'markers/raw_complete.json').is_file() and (out/'markers/blind_complete.json').is_file();opened.append(1)
        return orig(p)
    monkeypatch.setattr(post,'read',guarded)
    result=r.run(source,out,backend=backend)
    assert result['status']=='COMPLETED_DESCRIPTION_ONLY' and opened==[1]
    expected=dict(VAE_load=1,cached_native_load=8,cached_OFF_readback_load=4,metadata_construct=8,D181=12,RGB8_quantize=12,MP4_encode=12,MP4_probe=12,MP4_decode=12,RGB_input_prepare=16,E181=16,tag_raw_44=96,payload_FFT=12,payload_keyed_read=24,post_clipping=12,post_paired_quality=24)
    assert backend.calls==expected
    assert len(backend.spectra)==12 and all(x[1]==list(m.KEYS.values()) for x in backend.spectra)
    ledger=json.loads((out/'ledger.json').read_text());assert all(x['status']=='RETURNED' and x['attempts']==1 for x in ledger.values())
    assert sum(x.get('internal_blocks',{}).get('encoder',0) for x in ledger.values())==736
    assert sum(x.get('internal_blocks',{}).get('decoder',0) for x in ledger.values())==552
    assert len(json.loads((out/'fixed_rows.json').read_text()))==4224
    e=json.loads((out/'evaluation.json').read_text());assert e['payload_total']['groups']['planned_bits']==256 and e['payload_total']['cells']['planned_bits']==2816
    assert all(x['correct_ID']==0 for x in e['tag_summary'].values()) # rejecting tags did not skip any payload call
    before=backend.calls.copy();again=r.run('/missing',out,backend=backend);assert again['physical_reentry'] is False and before==backend.calls

@pytest.mark.parametrize('failure',['P0','raw'])
def test_failures_keep_slots_without_retry_or_payload_suppression(tmp_path,failure):
    source=fixture_source(tmp_path);out=tmp_path/'out';b=Stub(out,fail=failure);v=r.run(source,out,backend=b)
    assert v['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED' and v['tag_rows']==4224
    assert b.calls['cached_native_load']==8
    if failure=='P0':assert b.calls['D181']==9 and b.calls['E181']==13 and b.calls['payload_keyed_read']==18
    else:assert b.calls['D181']==12 and b.calls['E181']==16 and b.calls['payload_keyed_read']==24
    old=b.calls.copy();r.run('/none',out,backend=b);assert b.calls==old

def test_interrupt_consumes_attempt_and_retains_fixed_rows(tmp_path):
    source=fixture_source(tmp_path);out=tmp_path/'out';b=Stub(out,interrupt=True);v=r.run(source,out,backend=b)
    assert v['status']=='INTERRUPTED' and b.calls==dict(VAE_load=1)
    assert len(json.loads((out/'fixed_rows.json').read_text()))==4224
    assert not (out/'post_reference_access.json').exists()
    before=b.calls.copy();r.run('/absent',out,backend=b);assert b.calls==before

def test_missing_truth_keeps_physical_and_post_denominators(tmp_path):
    out=tmp_path/'out';b=Stub(out);v=r.run(tmp_path/'absent',out,backend=b)
    assert v['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED' and b.calls['E181']==16
    e=json.loads((out/'evaluation.json').read_text());assert e['status']=='POST_TRUTH_UNRESOLVED'
    assert e['payload_total']['groups']['undefined']==256 and e['payload_total']['cells']['undefined']==2816


def test_final_is_gate_only_with_no_new_decode_or_book(monkeypatch):
    b=m.book('BLOCK2-16','gate-fixture');q=[d+m.core.DELTA*x for d,x in zip(b.dithers,m.core.packet(3,b))]
    raw=m.raw_table([q]*44,'BLOCK2-16','gate-fixture');original=copy.deepcopy(raw)
    def forbid(*a,**kw):raise AssertionError('final recomputed core')
    monkeypatch.setattr(m,'book',forbid);monkeypatch.setattr(m.core,'projections',forbid);monkeypatch.setattr(m.core,'decode_projections',forbid);monkeypatch.setattr(m.core.math,'cos',forbid)
    final=m.finalize(raw)
    assert raw==original
    for a,b in zip(raw['rows'],final['rows']):
        for field in ('pilot_P','pilot_H','physical_bits','unmasked_bits','code_distances','nearest_ids','nearest','nearest_distance','code_status'):
            assert a['decoder'][field]==b['decoder'][field]

@pytest.mark.parametrize('folder',['raw','payload','blind'])
def test_failed_blind_dump_cannot_open_truth(tmp_path,monkeypatch,folder):
    source=fixture_source(tmp_path);out=tmp_path/'out';b=Stub(out)
    from runtime.wan import b_full_context_local_read_v1 as shared
    original=shared.save;failures=[]
    def fail_dump(path,value):
        p=Path(path)
        matches=folder in p.parts and (folder!='payload' or p.name=='key01.json')
        if matches and not failures:
            ledger=out/'ledger.json'
            rows=json.loads(ledger.read_text()) if ledger.exists() else {}
            active=[x for x in rows.values() if x['status']=='STARTED']
            wanted='tag_raw_44' if folder=='raw' else 'payload_keyed_read' if folder=='payload' else 'tag_frozen_tau_description'
            if active and active[0]['kind']==wanted:
                failures.append(active[0]['slot']);raise OSError('synthetic failed '+folder+' dump')
        return original(path,value)
    monkeypatch.setattr(shared,'save',fail_dump)
    v=r.run(source,out,backend=b)
    assert failures and v['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert not (out/'markers/blind_complete.json').exists() and not (out/'post_reference_access.json').exists()
    assert b.calls['E181']==16 and b.calls['payload_keyed_read']==24
    assert len(json.loads((out/'fixed_rows.json').read_text()))==4224
    before=b.calls.copy();r.run('/none',out,backend=b);assert b.calls==before

@pytest.mark.parametrize('failure',['VAE_load','cached_OFF_readback_load','D181','MP4_decode','E181','payload_FFT'])
def test_physical_failure_once_and_no_fallback(tmp_path,failure):
    source=fixture_source(tmp_path);out=tmp_path/'out';b=Stub(out,fail=failure);v=r.run(source,out,backend=b)
    assert v['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    ledger=json.loads((out/'ledger.json').read_text())
    failed=[x for x in ledger.values() if x['kind']==failure and x['status']=='FAILED']
    assert len(failed)==1 and failed[0]['attempts']==1
    assert all(x['attempts']<=1 for x in ledger.values())
    if failure=='VAE_load':assert b.calls['D181']==0 and b.calls['E181']==0
    elif failure=='cached_OFF_readback_load':assert b.calls['D181']==12 and b.calls['MP4_decode']==12 and b.calls['E181']==15
    elif failure=='payload_FFT':assert b.calls['payload_FFT']==12 and b.calls['payload_keyed_read']==22
    assert len(json.loads((out/'fixed_rows.json').read_text()))==4224
    before=b.calls.copy();r.run('/none',out,backend=b);assert b.calls==before

def test_notebook_builder_actual_cells_and_available_source(tmp_path):
    from scripts import build_b_block2_carrier_v1_notebook as build
    import ast
    root=Path(build.ROOT);path=build.build(root);n=json.loads(path.read_text());cells=n['cells'];code=[''.join(c['source']) for c in cells if c['cell_type']=='code']
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for source in code:ast.parse(source)
    assert 'git' in code[1] and 'b_block2_carrier_v1_run' in code[1]
    assert 'experiments.wan_state_clock.b_block2_carrier_v1_run' in code[2]
    assert "shutil.which('ffmpeg')" in code[2] and "shutil.which('ffprobe')" in code[2]
    assert 'SOURCE_TEXT' not in ''.join(code) and 'base64' not in ''.join(code)
    assert all(not c.get('outputs') and c.get('execution_count') is None for c in cells if c['cell_type']=='code')
    dependencies=build.source_files(root)
    assert 'main/tube_state/b_block2_carrier_v1.py' in dependencies and 'runtime/wan/vae.py' in dependencies
    assert all((root/p).is_file() for p in dependencies)
