"""CPU algebra/geometry/blind failure and stub call-budget checks; no model/FFT/codec."""
import json,math
from collections import Counter
import numpy as np
import pytest
from main.tube_state import b_lowrate_local_time_label_v1 as c
from main.tube_state import b_lowrate_local_time_label_plan_v1 as p
from runtime.wan import b_lowrate_local_time_label_v1 as run
from runtime.wan.b_lowrate_local_time_label_evaluation_v1 import nominal

pytestmark=pytest.mark.unit
KEY=p.KEYS['key00']
def row(j,i=None,clock=None):
    b=c.book(KEY);data=[0.]*32 if i is None else [float(2*(x^m)-1) for x,m in zip(c.word(i),b['mask'])]
    pilot=[0.]*16 if clock is None else [float(2*(x^clock)-1) for x in b['pilot_bits']]
    return dict(row=j,status='OBSERVED',pilot_scores=pilot,data_scores=data)
def table(rows):return dict(status='OBSERVED',rows=rows,truth_inputs=False,message_inputs=False)
def scores(bits):return [float(2*(x^m)-1) for x,m in zip(bits,c.book(KEY)['mask'])]

def test_rm_algebra_radius_and_wrong_valid_miscorrection():
    assert len(set(c.WORDS))==64
    assert min(sum(x!=y for x,y in zip(a,b)) for i,a in enumerate(c.WORDS) for b in c.WORDS[i+1:])==16
    for i in c.VALID_IDS:
        for count in (0,7):
            bits=c.word(i)
            for k in range(count):bits[k]^=1
            assert c.decode_data(scores(bits),KEY)['decoded_id']==i
    a,b=c.word(1),c.word(3);change=[i for i,(x,y) in enumerate(zip(a,b)) if x!=y]
    for i in change[:9]:a[i]^=1
    result=c.decode_data(scores(a),KEY)
    assert result['decoded_id']==3 and result['nearest_distance']==7 # wrong valid: no known-ID rescue
    assert c.decode_data(scores(c.word(0)),KEY)['status']=='RESERVED_ID'
    assert c.decode_data([0.]*32,KEY)['status']=='ERASURE'
    with pytest.raises(ValueError):c.decode_data([float('nan')]*32,KEY)

def test_pn_exact_energy_and_independent_roles():
    b=c.book(KEY)
    for role,n,ch in [('pilot',16,4),('data',32,8)]:
        pn=b[role].astype(np.float64)
        assert pn.shape==(n,ch,640) and np.array_equal(pn.sum(2),np.zeros((n,ch)))
        for channel in range(ch):assert np.array_equal(pn[:,channel]@pn[:,channel].T,640*np.eye(n))
        w=np.tensordot(np.ones(n),pn,axes=(0,0))/math.sqrt(n)
        assert np.allclose(np.square(w).sum(1),640,rtol=0,atol=1e-11)
    assert b['pilot_bits'].count(1)==8
    assert not np.array_equal(b['data'],c.book(p.KEYS['key01'])['data'])

def test_native_energy_support_and_source_preserved():
    rng=np.random.default_rng(20261023);z=rng.standard_normal((16,46,40,64)).astype(np.float32);before=z.copy()
    marked,receipt=c.construct(z,KEY)
    assert np.array_equal(z,before) and np.array_equal(marked[:4],z[:4])
    assert np.array_equal(marked[:,[0,1,44,45]],z[:,[0,1,44,45]])
    assert receipt['ideal_LLL_delta']['energy']==pytest.approx(receipt['ideal_budget_energy'],rel=1e-12)
    assert receipt['actual_FP32_delta']['energy']==pytest.approx(receipt['ideal_budget_energy'],rel=1e-7)
    assert receipt['correction_attempts']==0 and receipt['FP32_materializations']==1
    low=c.lll(marked);observed=c.observe(low,KEY,range(1,22))
    assert len(observed['rows'])==21

def test_phase_clock_origin_and_no_data_rescue():
    tables={p:table([row(j) for j in range(1,5)]) for p in range(8)}
    tables[7]=table([row(j,j+7,(j+1)%2) for j in range(1,5)])
    f=c.bind_family(tables,KEY)
    assert f['selected_phase']==7 and f['selected_clock']==1 and f['provisional_origin']==49
    assert len(f['rows'])==4 and len(f['hypotheses'])==16
    tables[6]=table([row(j,j+7,(j+1)%2) for j in range(1,5)])
    f=c.bind_family(tables,KEY);assert f['selected_phase'] is None # pilot tie, even though data plausible
    tables[6]=c.unavailable(range(1,5),'missing');f=c.bind_family(tables,KEY)
    assert f['status']=='INCOMPLETE_PHASE_FAMILY' and f['provisional_origin'] is None

def test_pilot_zero_parity_conflict_reserved_and_wrong_key():
    r=row(1,8,0);d=c.decode_data(r['data_scores'],KEY)
    assert c.final_row(r,d,KEY,0)['accepted']
    r['pilot_scores'][0]=0;assert not c.final_row(r,d,KEY,0)['accepted']
    assert c.canonical(table([row(j,j,j%2) for j in range(1,22)]),KEY)['rows'][0]['accepted']
    tables={p:table([row(j) for j in range(1,5)]) for p in range(8)}
    tables[7]=table([row(j,j+7+(2 if j==4 else 0),(j+1)%2) for j in range(1,5)])
    f=c.bind_family(tables,KEY);assert f['status']=='ORIGIN_CONFLICT' and f['provisional_origin'] is None

def test_geometry_denominators_no_full_latent_substitution():
    full=mixed=0
    for origin in (48,49,50,51):
        for phase in range(8):
            for j in range(1,5):
                g=nominal(origin,phase,j);full+=g['kind']=='FULL';mixed+=g['kind']=='MIXED'
    assert (full,mixed)==(16,112)
    tables=p.tag_tables();assert sum(len(p.row_indices(t[0])) for t in tables)==1608
    assert len(p.families())==24 and len(p.payload_tables())==80
    assert Counter(run.units().values())['E45']==96
    import inspect
    from runtime.wan.b_lowrate_local_time_label_backend_v1 import Backend
    source=inspect.getsource(Backend.encode_E45)
    assert 'reencode_rgb24_readback' in source and '45' in source and '12' in source

class Stub:
    def __init__(self,fail=None):self.calls=Counter();self.saved={};self.blocks=dict(encoder=0,decoder=0);self.fail=fail
    def op(self,name,value):
        self.calls[name]+=1
        if self.fail==name and self.calls[name]==1:raise RuntimeError('injected '+name)
        return value
    def initialize_PN(self):return self.op('PN',{})
    def load_vae(self):return self.op('VAE',{})
    def release_vae(self):self.calls['release']+=1
    def load_native(self,path):return self.op('native',str(path))
    def construct_tag(self,z,key):return self.op('construct',('TAG',dict(status='STUB')))
    def tag_lll(self,z,local=False):return self.op('LLL',local)
    def tag_scores(self,low,key,rows):return self.op('raw',table([dict(row=j,status='OBSERVED',pilot_scores=[0.]*16,data_scores=[0.]*32) for j in rows]))
    def save_tensor(self,path,value):self.saved[str(path)]=value
    def load_tensor(self,path):return self.saved[str(path)]
    def load_pixels(self,path,n):return self.op('OFF_media','pixels')
    def decode_PT(self,z):self.blocks['decoder']+=46;return self.op('D',dict(preclip='pre',float_rgb='float'))
    def quantize(self,z):return self.op('quantize','RGB8')
    def save_mp4(self,pixels,path):return self.op('codec',{})
    def probe(self,path):return self.op('probe',{})
    def read_media(self,path,probe,n):return self.op('read','readback')
    def prepare_full(self,pixels):return 'full_rgb'
    def encode_full(self,pixels):self.blocks['encoder']+=46;return self.op('E181','E181')
    def crop_received(self,pixels,origin):return self.op('crop','crop52')
    def prepare_local(self,pixels,phase):return self.op('phase','phase45')
    def encode_E45(self,pixels):self.blocks['encoder']+=12;return self.op('E45','E45')
    def payload_spectrum(self,z):return self.op('FFT','full')
    def local_payload_spectrum(self,z):return self.op('FFT','local')
    def payload_votes(self,s,key):return self.payload(False)
    def local_payload_votes(self,s,key):return self.payload(True)
    def payload(self,local):
        r=run.payload_empty(local);r['status']='STUB';r['signed_votes']=[[[1]*240 for _ in range(11 if local else 44)] for _ in range(4)];r['zero_coefficient_mask']=[[[0]*240 for _ in range(11 if local else 44)] for _ in range(4)]
        if not local:r['decoded_bits']=[0]*32
        for cell in r['local_cells']:
            for b in cell['bits']:b['decoded']=0
        return self.op('payload',r)
    def preclip(self,v):return self.op('clip',dict(status='STUB'))
    def quality(self,*args):return self.op('quality',dict(status='STUB',frame_denominator=181,adjacent_denominator=180))

def fixture_source(tmp_path):
    source=tmp_path/'source';source.mkdir();(source/'protocol.json').write_text(json.dumps(dict(config=dict(writer=dict(bits=[0]*32)))))
    return source

def test_complete_stub_budget_blind_truth_then_read_only(tmp_path):
    source=fixture_source(tmp_path);out=tmp_path/'run';b=Stub();result=run.run(source,out,backend=b)
    assert result['status']=='COMPLETED_DESCRIPTION_ONLY',result
    assert {k:b.calls[k] for k in ('VAE','D','codec','probe','read','E181','E45','FFT','payload')}==dict(VAE=1,D=8,codec=8,probe=8,read=8,E181=8,E45=96,FFT=72,payload=80)
    assert b.blocks==dict(encoder=1520,decoder=368)
    post=json.loads((out/'evaluation.json').read_text());assert post['payload']['local_by_geometry']['FULL']['planned']==704
    assert post['payload']['full_groups']['planned']==128 and post['payload']['full_cells']['planned']==1408
    assert (out/'markers/blind_complete.json').stat().st_mtime<=(out/'post_truth_access.json').stat().st_mtime
    before=b.calls.copy();again=run.run(source,out,backend=b);assert again['status']=='READ_ONLY_EXISTING_ATTEMPT' and b.calls==before

def test_failed_E_retains_all_slots_consumes_attempt(tmp_path):
    source=fixture_source(tmp_path);out=tmp_path/'failed';b=Stub('E45');result=run.run(source,out,backend=b)
    assert result['status']=='INCOMPLETE_FIXED_DENOMINATORS_RETAINED',result
    assert result['fixed_denominators']['raw_rows']==1608
    assert sum(r['status']=='FAILED' for r in json.loads((out/'ledger.json').read_text()).values())==1
    assert run.readback(out,True)['raw_rows']==1608
    before=b.calls.copy();run.run(source,out,backend=b);assert b.calls==before

def test_optional_notes_do_not_consume_attempt(tmp_path):
    out=tmp_path/'out';out.mkdir();(out/'source.json').write_text('{}');(out/'environment.json').write_text('{}')
    assert not run.execution_artifacts(out)


def test_ideal_linear_mix_count_tie_continuous_pilot_ranking():
    tables={}
    for q in range(8):
        rows=[]
        for j in range(1,5):
            r=row(j,j+6,j%2);r['pilot_scores']=[v*(1-q/4) for v in r['pilot_scores']];rows.append(r)
        tables[q]=table(rows)
    f=c.bind_family(tables,KEY)
    top_count=max(h['pilot_passes'] for h in f['hypotheses'])
    assert top_count==4 and sum(h['pilot_passes']==top_count for h in f['hypotheses'])==7
    assert f['selected_phase']==0 and f['selected_clock']==0 and f['maximum']==1
    assert f['status']=='PROVISIONAL_ORIGIN' and f['provisional_origin']==48
    # Every raw pilot score, including an adverse sign, contributes to S.
    tables[0]['rows'][0]['pilot_scores'][0]*=-1
    f=c.bind_family(tables,KEY)
    selected=next(h for h in f['hypotheses'] if h['phase']==0 and h['clock']==0)
    assert selected['pilot_statistic']==pytest.approx(62/64)

def test_valid_but_missing_ID_is_not_truth_rescued():
    # Received-only decoder cannot know ID3 is absent from a hypothetical source.
    raw=table([row(j,3,j%2) for j in range(1,22)])
    result=c.canonical(raw,KEY)
    assert result['rows'][0]['accepted'] and result['rows'][0]['accepted_id']==3
    absent_ids={3};assert result['rows'][0]['accepted_id'] in absent_ids

def test_51_pixel_frames_refused_before_E():
    import torch
    from runtime.wan.b_lowrate_local_time_label_backend_v1 import Backend
    backend=Backend.__new__(Backend);calls=[];backend.encode_E45=lambda x:calls.append(x)
    with pytest.raises(ValueError):backend.prepare_local(torch.zeros((51,320,512,3),dtype=torch.uint8),0)
    assert calls==[]


def test_actual_notebook_builder_and_editable_source(tmp_path):
    import ast,subprocess,sys,shutil
    from scripts import build_b_lowrate_local_time_label_v1_notebook as builder
    root=builder.ROOT;files=builder.source_files();assert all((root/f).is_file() for f in files)
    nb=json.loads((root/'notebooks/b_lowrate_local_time_label_v1.ipynb').read_text());codes=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in codes:ast.parse(cell)
    assert 'b_lowrate_local_time_label_v1_run' in ''.join(codes)
    assert 'b_lfg_posthoc_reference_v1_run' not in ''.join(codes)
    for name in files:
        dest=tmp_path/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,dest)
    before=(root/'notebooks/b_lowrate_local_time_label_v1.ipynb').read_bytes()
    assert builder.build().read_bytes()==before
    completed=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_lowrate_local_time_label_v1_run','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    assert not (tmp_path/'.git').exists()


@pytest.mark.parametrize('kind,final',[('tag_correlations',False),('payload_keyed_read',False),('tag_canonical_decode',True),('pilot_phase_and_RM_decode',True)])
def test_attempted_save_failure_cannot_be_old_placeholder(tmp_path,monkeypatch,kind,final):
    out=tmp_path/'attempt';out.mkdir();target=out/'old_placeholder.json';run.save(target,dict(status='TECHNICAL_UNRESOLVED'))
    j=run.Journal(out,Stub());name=next(k for k,v in run.units().items() if v==kind);original_save=run.save
    def fail_save(path,value):
        if path==target:raise OSError('injected persistence failure')
        return original_save(path,value)
    monkeypatch.setattr(run,'save',fail_save)
    j.record(name,lambda:dict(status='TECHNICAL_UNRESOLVED'),target.name)
    assert j.rows[name]['status']=='FAILED' and j.rows[name]['attempts']==1
    completion='blind_readback' if final else 'raw_readback'
    marker='markers/blind_complete.json' if final else 'markers/raw_complete.json'
    j.record(completion,lambda:run.readback(out,final),marker)
    assert j.rows[completion]['status']=='FAILED' and not (out/marker).exists()
    assert not (out/'post_truth_access.json').exists() and not (out/'markers/blind_complete.json').exists()
    assert j.rows['post']['attempts']==0 and all(v['attempts']==0 for v in j.rows.values() if v['kind'] in ('post_clipping','post_paired_quality'))
    assert json.loads(target.read_text())['status']=='TECHNICAL_UNRESOLVED' # retained, never called persisted result
    before=j.backend.calls.copy();assert run.run(tmp_path/'source',out,backend=j.backend)['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert j.backend.calls==before

def test_post_ID_support_is_separate_from_row_geometry():
    from runtime.wan.b_lowrate_local_time_label_evaluation_v1 import id_support
    # Received48..99: ID6 support45..52 partial; ID7 support53..60 full; ID5 support37..44 absent.
    assert id_support(48,6)=='PARTIAL_PRESENT'
    assert id_support(48,7)=='FULL_PRESENT'
    assert id_support(48,5)=='ABSENT'
    assert id_support(48,None)=='UNRESOLVED_ID'
    assert id_support(48,float('nan'))=='INVALID_OR_NONFINITE_ID'
    assert id_support(48,64)=='INVALID_OR_NONFINITE_ID'
    # A MIXED observed row may coexist with a completely present decoded ID.
    assert nominal(48,1,1)['kind']=='MIXED' and id_support(48,7)=='FULL_PRESENT'
