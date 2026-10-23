"""Stage1 controlled CPU/fixture/stub only. No real model/cache/media/Drive execution."""
from pathlib import Path
from collections import Counter
import inspect,json,random
import numpy as np
import pytest
from main.tube_state import b_lfg_bch_reference_v1 as bch
from main.tube_state import b_lfg_posthoc_reference_v1 as core
from main.tube_state import b_lfg_posthoc_plan_v1 as plan
from runtime.wan import b_lfg_posthoc_reference_v1 as r
pytestmark=pytest.mark.unit
MESSAGE=b'Video-WM LFG posthoc reference stage1 fixed message v1'

def test_BCH_independent_C_oracle_MSB_bytes_and_zero_to_eight_errors():
    library=pytest.importorskip('bchlib');oracle=library.BCH(8,prim_poly=0x409,m=10,swap_bits=False)
    assert oracle.n==1023 and oracle.ecc_bits==80 and oracle.ecc_bytes==10
    assert bch.parameters()['generator_degree']==80
    for seed in range(3):
        message=MESSAGE if seed==0 else bytes(random.Random(seed).randbytes(54))
        word=bch.encode(message);encoded=bch.bits_to_int(word).to_bytes(64,'big')
        assert encoded==message+oracle.encode(message)
        for n in range(9):
            value=word[:]
            for pos in random.Random(50*seed+n).sample(range(512),n):value[pos]^=1
            packed=bch.bits_to_int(value).to_bytes(64,'big');data,ecc=bytearray(packed[:54]),bytearray(packed[54:])
            count=oracle.decode(data=data,recv_ecc=ecc);oracle.correct(data,ecc);own=bch.decode(value)
            assert count==n and own['correction_count']==n and bytes(data)==message and own['message_hex']==message.hex()

def test_all512_single_error_locations_and_no_message_prior():
    word=bch.encode(MESSAGE)
    for i in range(512):
        value=word[:];value[i]^=1;d=bch.decode(value)
        assert d['BCH_success'] and d['correction_positions']==[i] and d['message_hex']==MESSAGE.hex()
    other=bytes(range(54));received=bch.encode(other);decoded=bch.decode(received)
    assert decoded['BCH_success'] and decoded['message_hex']==other.hex() and decoded['message_hex']!=MESSAGE.hex()
    assert decoded['authentication'] is False # another valid word is never filtered by the known ASCII test message

def test_shortened_location_rejected_and_failure_has_no_raw_message_fallback(monkeypatch):
    exponent,_=bch._field()
    monkeypatch.setattr(bch,'_locator',lambda s:([1,exponent[700]],1))
    d=bch.decode([0]*512)
    assert d['status']=='BCH_ROOTS_OR_SHORTENING_UNRESOLVED' and not d['BCH_success']
    assert d['information_bits']==[None]*432 and d['corrected_codeword_bits'] is None

def test_Haar_orthonormal_even_boundary_and_direct_cube_formula():
    z=np.arange(8,dtype=np.float64).reshape(1,2,2,2);bands=core.haar(z)
    assert bands['LLL'].item()==pytest.approx(z.sum()/np.sqrt(8))
    assert sum(float((v*v).sum()) for v in bands.values())==pytest.approx(float((z*z).sum()))
    assert np.max(np.abs(core.inverse(bands)-z))<1e-14
    assert np.allclose(core.lll(z),bands['LLL'])
    for shape in [(2,3,4,4),(2,4,3,4),(2,4,4,3)]:
        with pytest.raises(ValueError,match='no padding'):core.haar(np.ones(shape))

def test_W_uses_sqrt512_not_M_and_sigma_population_without_floor(monkeypatch):
    monkeypatch.setattr(core,'pn_rows',lambda key,shape:iter([np.ones(shape,dtype=np.int8)]*512))
    shape=(2,2,2,2);W=core.spread([1]*512,'fixture',shape)
    assert np.allclose(W,np.sqrt(512)) # dividing by sqrt(M) gives a different amplitude
    bands={k:np.zeros(shape) for k in ('LLL','LLH','LHL','LHH','HLL','HLH','HHL','HHH')}
    bands['LLL'][0]=np.arange(8).reshape(2,2,2);bands['LLL'][1]=5
    source=core.inverse(bands).astype(np.float32);before=source.copy();marked,receipt=core.construct(source,[1]*512,'fixture')
    low=core.lll(source);expected=np.std(low,axis=(1,2,3),ddof=0)
    assert np.allclose(receipt['sigma_channels'],expected)
    assert receipt['sigma_channels'][1]==0 and receipt['zero_sigma_channels']==[1]
    assert np.array_equal(source,before) and np.array_equal(marked[1],source[1])
    observed=core.lll(marked)-low
    assert np.allclose(observed[0],.15*expected[0]*np.sqrt(512),atol=1e-5)
    assert receipt['Haar_full_forward_calls']==2 and receipt['correction_attempts']==0

def test_PN_seed_cache_bitorder_and_correlation_no_writer_inputs(monkeypatch):
    core.packed_book.cache_clear();calls=[];original=core.hmac.new
    def count(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(core.hmac,'new',count)
    shape=(2,2,2,2);a=core.packed_book('fixture',shape);b=core.packed_book('wrong',shape)
    assert core.packed_book('fixture',shape) is a and len(calls)==1024 and a!=b
    first=next(core.pn_rows('fixture',shape));expected=2*np.unpackbits(np.frombuffer(a[0],dtype=np.uint8),bitorder='big')[:16].astype(np.int8)-1
    assert np.array_equal(first.reshape(-1),expected)
    low=np.arange(16,dtype=np.float64).reshape(shape);score=core.score_from_lll(low,'fixture')
    assert score['scores'][0]==pytest.approx(float(np.sum(low*first)/16))
    zeros=core.score_from_lll(np.zeros(shape),'fixture');assert zeros['raw_codeword_bits']==[0]*512
    assert tuple(inspect.signature(core.score_from_lll).parameters)==('low','key')

def test_actual_tensor_allchannels_guards_and_FP32_roundoff_without_compensation(monkeypatch):
    import torch
    torch.set_num_threads(1)
    from runtime.wan import b_lfg_posthoc_tensor_v1 as tensor
    rng=np.random.default_rng(61);source=torch.from_numpy(rng.normal(size=plan.SHAPE).astype(np.float32));saved=source.clone()
    monkeypatch.setattr(core,'spread',lambda bits,key,shape:np.ones(shape))
    marked,receipt=tensor.construct(source,[1]*512,'fixture')
    assert torch.equal(source,saved) and receipt['source_unchanged']
    assert receipt['actual_FP32_delta']['energy']>0 and receipt['original_payload_channels_cost']['energy']>0
    assert receipt['guard0_cost']['energy']>0 and receipt['guard45_or_last_cost']['energy']>0
    assert not torch.equal(source[:,:4],marked[:,:4])
    assert set(receipt['actual_other7_drift'])=={'LLH','LHL','LHH','HLL','HLH','HHL','HHH'}
    assert receipt['FP32_materializations']==1 and receipt['correction_attempts']==0

class Stub:
    def __init__(self,out,fail=None,interrupt=False):self.out=out;self.calls=Counter();self.blocks={'encoder':0,'decoder':0};self.fail=fail;self.interrupt=interrupt;self.spectra=[];self.lows=[];self.pixel_inputs=[]
    def check(self,kind):
        rows=json.loads((self.out/'ledger.json').read_text());active=[v for v in rows.values() if v['status']=='STARTED'];assert len(active)==1 and active[0]['kind']==kind
        assert json.loads((self.out/'receipts'/(active[0]['slot']+'.json')).read_text())['status']=='STARTED'
        assert (self.out/'attempt.json').is_file()
        if not kind.startswith('post_'):assert not (self.out/'post_truth_access.json').exists()
        self.calls[kind]+=1
        if self.fail==kind and self.calls[kind]==1:raise OSError('controlled '+kind+' failure')
    def initialize_PN(self):self.check('PN_initialize');return dict(stub=True)
    def load_vae(self):
        self.check('VAE_load')
        if self.interrupt:raise KeyboardInterrupt('controlled stop')
        return dict(stub=True)
    def release_vae(self):pass
    def load_native(self,path):self.check('cached_pre_tag_native_load');return dict(host=path.parts[-3])
    def construct_message(self,z,bits,key):self.check('LLL_construct');assert len(bits)==512;return dict(z,marked=True),dict(status='SYNTHETIC',source_unchanged=True)
    def save_tensor(self,path,v):path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v))
    def load_tensor(self,path):return json.loads(Path(path).read_text())
    def message_lll(self,z):self.check('received_Haar_LLL');v=dict(LLL=True);self.lows.append([v,[]]);return v
    def message_scores(self,low,key):
        self.check('message_correlations');assert low is self.lows[-1][0];self.lows[-1][1].append(key)
        return dict(status='CORRELATIONS_SAVED',scores=[-1.]*512,raw_codeword_bits=[0]*512,N=512,M=plan.M,truth_inputs=False,message_inputs=False)
    def decode_PT(self,z):self.check('D181');self.blocks['decoder']+=46;return dict(preclip=dict(stub=True),float_rgb=dict(stub=True))
    def quantize(self,v):self.check('RGB8_quantize');value=dict(pixel_id=len(self.pixel_inputs));self.pixel_inputs.append([value,[]]);return value
    def save_mp4(self,pixels,path):self.check('MP4_encode');assert pixels is self.pixel_inputs[-1][0];return dict(stub=True,crf=23)
    def probe(self,path):self.check('MP4_probe');return dict(stub=True)
    def read_media(self,path,probe,n):self.check('MP4_decode');return dict(readback=True)
    def prepare_full(self,pixels):self.check('RGB_input_prepare');self.pixel_inputs[-1][1].append(pixels);return dict(rgb=True)
    def encode_full(self,v):self.check('E181');self.blocks['encoder']+=46;return dict(stub=True)
    def payload_spectrum(self,z):self.check('payload_FFT');v=dict(spectrum=True);self.spectra.append([v,[]]);return v
    def payload_votes(self,value,key):
        self.check('payload_keyed_read');assert value is self.spectra[-1][0];self.spectra[-1][1].append(key)
        return dict(status='READ',decoded_bits=[0]*32,local_cells=[dict(bits=[dict(decoded=0) for i in range(8)]) for t in range(44)],signed_votes=[[[-1]*240 for t in range(44)] for c in range(4)],zero_coefficient_mask=[[[0]*240 for t in range(44)] for c in range(4)])
    def preclip(self,v):self.check('post_clipping');assert (self.out/'markers/blind_complete.json').is_file();return dict(status='SYNTHETIC')
    def quality(self,a,b,frames,meaning):
        self.check('post_paired_quality');assert (self.out/'markers/blind_complete.json').is_file()
        return dict(status='SYNTHETIC',frame_denominator=181,adjacent_denominator=180,frames=[dict(frame=i) for i in range(181)],adjacent=[dict(frame_before=i,frame_after=i+1) for i in range(180)],spatial_support='global; no old ROI mask')

def source_fixture(tmp_path):
    src=tmp_path/'source';src.mkdir();(src/'protocol.json').write_text(json.dumps(dict(config=dict(writer=dict(bits=[0]*32)))));return src

def test_fixed_complete_graph_BCH_success_is_not_full_message_or_capacity(tmp_path,monkeypatch):
    src=source_fixture(tmp_path);out=tmp_path/'out';backend=Stub(out)
    original=r.read;truth=[]
    def guarded(path):
        if Path(path).name in ('writer_reference.json','protocol.json'):
            assert (out/'markers/blind_complete.json').is_file();truth.append(Path(path).name)
        return original(path)
    monkeypatch.setattr(r,'read',guarded)
    result=r.run(src,out,backend=backend)
    assert result['status']=='COMPLETED_DESCRIPTION_ONLY' and set(truth)=={'writer_reference.json','protocol.json'}
    assert backend.calls==dict(PN_initialize=1,VAE_load=1,cached_pre_tag_native_load=4,LLL_construct=4,received_Haar_LLL=24,message_correlations=48,D181=8,RGB8_quantize=8,MP4_encode=8,MP4_probe=8,MP4_decode=8,RGB_input_prepare=16,E181=16,payload_FFT=16,payload_keyed_read=32,post_clipping=8,post_paired_quality=16)
    assert len(backend.lows)==24 and all(v[1]==list(plan.KEYS.values()) for v in backend.lows)
    assert len(backend.spectra)==16 and all(v[1]==list(plan.KEYS.values()) for v in backend.spectra)
    assert all(x[1][0] is x[0] and len(x[1])==2 and x[1][1].get('readback') for x in backend.pixel_inputs)
    ledger=r.read(out/'ledger.json');assert all(v['status']=='RETURNED' and v['attempts']==1 for v in ledger.values())
    assert sum(v.get('internal_blocks',{}).get('encoder',0) for v in ledger.values())==736 and sum(v.get('internal_blocks',{}).get('decoder',0) for v in ledger.values())==368
    e=r.read(out/'evaluation.json');assert sum(v['BCH_success'] for v in e['message_summary'].values())==48
    assert sum(v['full_message_exact'] for v in e['message_summary'].values())==0 # valid all-zero code is not the fixed diagnostic message
    assert e['payload_total']['groups']['correct_retained']==256 and e['payload_total']['cells']['correct_retained']==2816
    assert e['user_net_payload_bits']==32 and not e['new_user_capacity_claim'] and not e['authentication']
    before=backend.calls.copy();again=r.run('/missing',out,backend=backend);assert again['physical_reentry'] is False and before==backend.calls

@pytest.mark.parametrize('folder',['raw','payload','blind'])
def test_attempted_blind_save_failure_cannot_open_truth(tmp_path,monkeypatch,folder):
    src=source_fixture(tmp_path);out=tmp_path/'out';backend=Stub(out);original=r.save;failed=[]
    def fail(path,value):
        path=Path(path);wanted={'raw':'message_correlations','payload':'payload_keyed_read','blind':'BCH_decode'}[folder]
        if folder in path.parts and not failed and (out/'ledger.json').is_file():
            active=[v for v in json.loads((out/'ledger.json').read_text()).values() if v['status']=='STARTED']
            if active and active[0]['kind']==wanted:failed.append(active[0]['slot']);raise OSError('controlled blind dump failure')
        return original(path,value)
    monkeypatch.setattr(r,'save',fail);v=r.run(src,out,backend=backend)
    assert failed and v['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    assert not (out/'markers/blind_complete.json').exists() and not (out/'post_truth_access.json').exists()
    assert backend.calls['E181']==16 and backend.calls['payload_keyed_read']==32
    assert len(r.read(out/'fixed_rows.json'))==48
    before=backend.calls.copy();r.run('/none',out,backend=backend);assert backend.calls==before

@pytest.mark.parametrize('failure',['cached_pre_tag_native_load','VAE_load','MP4_decode','E181','payload_FFT'])
def test_physical_failure_consumed_no_fallback_and_independent_other_view(tmp_path,failure):
    src=source_fixture(tmp_path);out=tmp_path/'out';b=Stub(out,fail=failure);v=r.run(src,out,backend=b)
    assert v['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    ledger=r.read(out/'ledger.json');assert len([x for x in ledger.values() if x['kind']==failure and x['status']=='FAILED'])==1
    assert all(x['attempts']<=1 for x in ledger.values()) and len(r.read(out/'fixed_rows.json'))==48
    if failure=='cached_pre_tag_native_load':assert b.calls['D181']==6 and b.calls['E181']==12
    if failure=='MP4_decode':assert b.calls['E181']==15 and b.calls['MP4_decode']==8 # clean view did not get discarded or substitute for mp4
    if failure=='payload_FFT':assert b.calls['E181']==16 and b.calls['payload_keyed_read']==30
    before=b.calls.copy();r.run('/none',out,backend=b);assert before==b.calls

def test_interrupt_and_scientific_residue_are_readonly(tmp_path):
    src=source_fixture(tmp_path);out=tmp_path/'out';b=Stub(out,interrupt=True);v=r.run(src,out,backend=b)
    assert v['status']=='INTERRUPTED' and b.calls==dict(PN_initialize=1,VAE_load=1)
    assert not (out/'post_truth_access.json').exists() and len(r.read(out/'fixed_rows.json'))==48
    before=b.calls.copy();r.run('/none',out,backend=b);assert before==b.calls
    residue=tmp_path/'residue';(residue/'raw').mkdir(parents=True);(residue/'raw/started.json.tmp').write_text('partial')
    b2=Stub(residue);v2=r.run('/none',residue,backend=b2);assert v2['status']=='READ_ONLY_EXISTING_ATTEMPT' and not b2.calls

def test_BCH_valid_failure_still_reads_all_original_payload(tmp_path,monkeypatch):
    src=source_fixture(tmp_path);out=tmp_path/'out';b=Stub(out)
    def decoder(bits):return dict(status='BCH_UNCORRECTABLE',BCH_success=False,raw_codeword_bits=list(bits),information_bits=[None]*432,corrected_codeword_bits=None,correction_count=None,authentication=False)
    monkeypatch.setattr(bch,'decode',decoder);v=r.run(src,out,backend=b)
    assert v['status']=='COMPLETED_DESCRIPTION_ONLY' and v['BCH_failures']==48 and b.calls['payload_FFT']==16 and b.calls['payload_keyed_read']==32
    e=r.read(out/'evaluation.json');assert all(x['information_common_bits']==0 for x in e['message_summary'].values())
    assert all(x['information_denominator']==4*432 for x in e['message_summary'].values())

def test_missing_original_truth_preserves_physical_evidence(tmp_path):
    out=tmp_path/'out';b=Stub(out);v=r.run(tmp_path/'no-source',out,backend=b)
    assert v['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED' and b.calls['E181']==16
    e=r.read(out/'evaluation.json');assert e['status']=='POST_TRUTH_UNRESOLVED' and e['payload_total']['groups']['undefined']==256 and e['payload_total']['cells']['undefined']==2816


def test_optional_environment_JSON_does_not_consume_first_attempt(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'environment.json').write_text('{}');(out/'source_note.json').write_text('{}')
    assert not r.execution_artifacts(out)
    backend=Stub(out,interrupt=True);result=r.run('/not-read',out,backend=backend)
    assert result['status']=='INTERRUPTED' and backend.calls==dict(PN_initialize=1,VAE_load=1)
    assert (out/'attempt.json').is_file()

def test_actual_notebook_builder_temporary_sync_AST_and_no_execution(tmp_path):
    import ast,shutil
    from scripts import build_b_lfg_posthoc_reference_v1_notebook as build
    root=Path(build.ROOT);actual=build.build(root)
    temp=tmp_path/'source';(temp/'runtime/wan').mkdir(parents=True);shutil.copyfile(root/'runtime/wan/b_lfg_posthoc_reference_v1.py',temp/'runtime/wan/b_lfg_posthoc_reference_v1.py')
    assert build.build(temp).read_bytes()==actual.read_bytes()
    cells=json.loads(actual.read_text())['cells'];code=[''.join(c['source']) for c in cells if c['cell_type']=='code']
    assert code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for source in code:ast.parse(source)
    assert 'experiments.wan_state_clock.b_lfg_posthoc_reference_v1_run' in code[2]
    assert "shutil.which('ffmpeg')" in code[2] and "shutil.which('ffprobe')" in code[2]
    assert 'bchlib' not in code[2] and 'SOURCE_TEXT' not in ''.join(code) and 'base64' not in ''.join(code)
    assert all(c['execution_count'] is None and c['outputs']==[] for c in cells if c['cell_type']=='code')
    assert 'runtime/wan/vae.py' in build.source_files(root)
