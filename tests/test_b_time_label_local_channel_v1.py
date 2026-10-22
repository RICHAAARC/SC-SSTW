"""Card106 synthetic tensor/stub wiring only. No real assets/model/VAE/FFT/codec."""
from pathlib import Path
from collections import Counter
import inspect,json
import pytest
from main.tube_state import b_time_label_local_channel_v1 as m
from runtime.wan import b_time_label_local_channel_v1 as user
from runtime.wan import b_time_label_local_channel_evaluation_v1 as post
pytestmark=pytest.mark.unit

def test_fixed_budget_reuse_mapping_and_only_cached_bits_passed(tmp_path):
    cfg=user.load_config();k=Counter(user.units(cfg).values())
    assert {x:k[x] for x in ('E41','cached_normalized_E_load','payload_FFT','payload_votes','cached_payload_bits','ordinary_D181','MP4_encode','MP4_probe','MP4_decode')}==dict(E41=208,cached_normalized_E_load=80,payload_FFT=104,payload_votes=208,cached_payload_bits=80,ordinary_D181=4,MP4_encode=8,MP4_probe=8,MP4_decode=8)
    plan=user.plans(cfg);assert sum(p['rows'] for p in plan)==6816
    assert sum(p['rows'] for p in plan if p['scope']=='native')==1056
    for host in cfg['private_roster']:
        for arm in m.ARMS:
            for view in m.VIEWS:
                for phase in range(4):
                    r=m.reuse(arm,view,phase)
                    if r:
                        assert arm!='PT' and view!='cross_center'
                        assert m.CLEAN_ORIGINS[int(view[-1])]+phase==r['origin']+r['phase']
    p=tmp_path/'old.json';p.write_text(json.dumps(dict(phase_local_bits={str(p):[p%2]*80 for p in range(4)},scores='do not pass',path='do not pass',ties='do not pass')))
    old=user.load_cached_family(p,(0,1))
    assert set(old['phase_local_bits'])=={'0','1'} and not any(k in old for k in ('scores','path','ties'))

def test_core_exception_fixed_rows_and_threshold_only_bridge(monkeypatch):
    original=m.core.decode_projections;count=[0]
    def fail_once(*a,**kw):
        count[0]+=1
        if count[0]==1:raise RuntimeError('fixture per-unit core exception')
        return original(*a,**kw)
    monkeypatch.setattr(m.core,'decode_projections',fail_once)
    raw=m.raw_table([[0.]*48 for _ in range(10)],'fixture',10)
    assert len(raw['rows'])==10 and raw['technical_rows']==1
    d=raw['rows'][0]['decoder'];assert d['status']=='TECHNICAL_UNRESOLVED' and d['code_distances']==[None]*64
    monkeypatch.setattr(m.core,'decode_projections',original)
    b=m.core.book('fixture')
    packet=m.core.packet(6,b);q=[v+m.core.DELTA*bit for v,bit in zip(b.dithers,packet)]
    base=original(q,b,None)
    for tau in (0.,.5,1.):
        expected=original(q,b,tau);actual=m.finish(base,tau)
        assert all(actual[k]==expected[k] for k in expected)
    bad=dict(raw,rows=[{}]+raw['rows'][1:]);final=m.finalize(bad,.5)
    assert len(final['rows'])==10 and final['rows'][0]['decoder']['status']=='TECHNICAL_UNRESOLVED'

def test_local_geometry_full_mixed_guard_cross_and_all_phases_retained():
    cfg=user.load_config();counts=Counter()
    for host in cfg['private_roster']:
        c=Counter(post.geometry(host,v,p,r)['status'] for v in m.VIEWS for p in range(4) for r in range(1,11))
        assert c==Counter(FULL=59,MIXED=175,GUARD=3,CROSS_SEAM=3)
        if host['role']=='evaluation':counts.update(c)
    assert counts==Counter(FULL=118,MIXED=350,GUARD=6,CROSS_SEAM=6)
    assert len(inspect.signature(m.raw_table).parameters)==3 and 'host' not in inspect.signature(m.raw_table).parameters

def test_actual_tensor_mapping_on_synthetic_CPU_only():
    import torch
    from runtime.wan import b_time_label_tensor_v1 as tensor
    source=torch.zeros((1,16,46,40,64),dtype=torch.float32,device='cpu')
    source[:,:4]=.25;source[:,:,0]=2.;source[:,:,45]=3.
    changed,receipt=tensor.construct(source,'fixture-tensor')
    assert torch.equal(source[:,:4],changed[:,:4])
    assert torch.equal(source[:,:,0],changed[:,:,0]) and torch.equal(source[:,:,45],changed[:,:,45])
    mask=torch.zeros_like(source,dtype=torch.bool)
    for y0,y1,x0,x1 in tensor.ROIS:mask[0,4:16,1:45,y0:y1,x0:x1]=True
    assert torch.equal(source[~mask],changed[~mask])
    assert receipt['ideal_energy']<=176 and receipt['FP32_correction_attempts']==0
    q=tensor.project_rows(changed,'fixture-tensor',native=True)
    decoded=m.raw_table(q,'fixture-tensor',44);final=m.finalize(decoded,.5)
    assert [r['decoder']['tag_label'] for r in final['rows']]==list(range(1,45))
    local=tensor.project_rows(changed[:,:,:11].clone(),'fixture-tensor',native=False)
    assert [r['projections'] for r in q[:10]]==[r['projections'] for r in local]
    with pytest.raises(ValueError):tensor.validate(source.double(),46)

class Fake:
    """Token arrays only. Counts calls, never invokes torch or external programs."""
    def __init__(self,missing_cache=False,interrupt=False,out=None):
        self.calls=Counter();self.blocks={'encoder':0,'decoder':0}
        self.missing_cache=missing_cache;self.interrupt=interrupt;self.out=out
    def load_vae(self):
        self.calls['VAE']+=1
        if self.out:
            receipt=json.loads((self.out/'receipts/vae.json').read_text());assert receipt['status']=='STARTED'
            assert (self.out/'attempt.json').is_file()
        if self.interrupt:raise KeyboardInterrupt('fixture')
        return dict(status='FAKE')
    def release_vae(self):pass
    def load_native(self,path):return dict(native_token=True)
    def construct(self,source,key):
        self.calls['construct']+=1;return dict(PT_token=True),dict(status='FAKE',payload_rewritten=False)
    def project_native(self,z,key):return [[0.]*48 for _ in range(44)]
    def project_local(self,z,key):return [[0.]*48 for _ in range(10)]
    def load_pixels(self,path,length):return list(range(length))
    def load_E(self,path):
        self.calls['cached_E']+=1
        if self.missing_cache and 'b101-s01/features/OFF/r0/p0.pt' in str(path):raise FileNotFoundError('fixture missing cachedE')
        return dict(E_token=True)
    def save_tensor(self,path,v):
        p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
    def load_tensor(self,path):
        p=Path(path)
        if p.is_file():return json.loads(p.read_text())
        return list(range(121 if 'deleted_source' in str(p) else 181))
    def decode_PT(self,z):
        self.calls['D']+=1;self.blocks['decoder']+=46
        return dict(preclip=list(range(181)),float_rgb=list(range(181)))
    def quantize(self,x):return x
    def save_mp4(self,x,path):
        self.calls['encode']+=1;self.save_tensor(path,x);return dict(status='FAKE')
    def probe(self,path):self.calls['probe']+=1;return dict(status='FAKE')
    def read_media(self,path,probe,length):
        self.calls['decode']+=1;v=json.loads(Path(path).read_text());assert len(v)==length;return v
    def edit(self,x,private):
        a,b,d=(private[k] for k in ('a','b','d'));z=x[a:a+121+d];return z[:b]+z[b+d:]
    def crop45(self,x,origin):return x[origin:origin+45]
    def prepare(self,x,p):return x[p:p+41]
    def encode_local(self,x):
        assert len(x)==41;self.calls['new_E']+=1;self.blocks['encoder']+=11;return dict(E_token=True)
    def spectrum(self,z):self.calls['FFT']+=1;return dict(spectrum_token=True)
    def votes(self,z,key):
        self.calls['votes']+=1
        return dict(local_cells=[dict(bits=[dict(decoded=0) for _ in range(8)]) for _ in range(10)])
    def preclip(self,x):return dict(status='FAKE')
    def quality(self,left,right,frames,meaning):
        assert len(left)==len(right)==len(frames)
        return dict(status='FAKE',frame_denominator=len(frames),adjacent_denominator=len(frames)-1,quality_pass=None)

def source_fixture(tmp_path):
    source=tmp_path/'original_source';deletion=tmp_path/'deleted_source'
    source.mkdir();deletion.mkdir()
    user.dump(source/'protocol.json',dict(config=dict(writer=dict(bits=[0]*32))))
    for host in user.load_config()['private_roster']:
        h=host['id']
        for layer in ('preclip','rgb8','mp4'):user.dump(source/f'hosts/{h}/quality/OFF_to_P0/{layer}.json',dict(status='FAKE_REFERENCE'))
        for layer in ('input_rgb8','edited_mp4'):user.dump(deletion/f'quality/{h}/OFF_to_P0/{layer}.json',dict(status='FAKE_REFERENCE'))
        if host['role']=='evaluation':
            for q in (0,1,8,9,10,11,12,13,20,21,22,23):
                user.dump(source/f'hosts/{h}/blind/families/q{q:02}.json',dict(phase_local_bits={str(p):[0]*80 for p in range(4)}))
    return source,deletion

def test_fixed_stub_full_path_counts_freeze_blind_save_and_read_only(tmp_path):
    source,deletion=source_fixture(tmp_path);out=tmp_path/'out';fake=Fake(out=out)
    result=user.run(source,deletion,out,backend=fake)
    assert result['status']=='FINISHED_SEEN_LOCAL_NO_SCIENTIFIC_PASS',result.get('error')
    assert fake.calls==Counter(VAE=1,construct=4,D=4,encode=8,probe=8,decode=8,cached_E=80,new_E=208,FFT=104,votes=208)
    ledger=json.loads((out/'ledger.json').read_text());assert all(r['status']=='RETURNED' and r['attempts']==1 for r in ledger.values())
    frozen=json.loads((out/'thresholds.json').read_text())['frozen_unix']
    assert frozen<min(v['started_unix'] for k,v in ledger.items() if k.startswith(('b101-s02/','b101-s04/')))
    marker=json.loads((out/'markers/blind_complete.json').read_text())
    assert marker['actual_tag_tables']==600 and marker['actual_tag_rows']==6816
    assert marker['saved_unix']<json.loads((out/'post_truth_access.json').read_text())['opened_unix']
    status=json.loads((out/'unit_status.json').read_text());assert len(status)==6816
    evaluation=json.loads((out/'evaluation.json').read_text())
    assert evaluation['payload_summary']['FULL_target_bits']==944
    assert evaluation['payload_summary']['planned_pair_bits']==3840 and evaluation['payload_summary']['cross_undefined']==496
    assert evaluation['summary']['evaluation/local']['planned_rows']==2880
    assert evaluation['scientific_PASS'] is None
    before=fake.calls.copy();assert user.run('/absent','/absent',out,config='/absent',backend=fake)['physical_reentry'] is False
    assert fake.calls==before

def test_cache_failure_has_no_reencode_and_interrupt_consumes_attempt(tmp_path):
    source,deletion=source_fixture(tmp_path)
    for name,fake in [('missing',Fake(missing_cache=True)),('interrupt',Fake(interrupt=True))]:
        out=tmp_path/name;fake.out=out
        result=user.run(source,deletion,out,backend=fake)
        assert result['status'] in ('INCOMPLETE_FIXED_ROWS_RETAINED','INTERRUPTED')
        before=fake.calls.copy();again=user.run(source,deletion,out,backend=fake);assert again['physical_reentry'] is False and fake.calls==before
        assert not (out/'post_truth_access.json').exists()
        assert len(json.loads((out/'unit_status.json').read_text()))==6816
        if name=='missing':
            assert fake.calls['new_E']==104 # cal only; never a105th replacement E
            ledger=json.loads((out/'ledger.json').read_text())
            assert ledger['b101-s01/OFF/clean0/p0/E']['status']=='FAILED'
            assert all(v['status']=='NOT_ENTERED' for k,v in ledger.items() if k.startswith(('b101-s02/','b101-s04/')))
            raw=json.loads((out/'raw/b101-s01/OFF/clean0/p0/key00.json').read_text())
            assert len(raw['rows'])==10 and raw['technical_rows']==10

def test_readonly_overlays_started_receipt_without_any_input_or_backend(tmp_path):
    out=tmp_path/'partial';out.mkdir()
    user.dump(out/'attempt.json',dict(one_attempt=True))
    user.dump(out/'ledger.json',{'vae':dict(slot='vae',status='NOT_ENTERED')})
    user.dump(out/'receipts/vae.json',dict(slot='vae',status='STARTED',attempts=1))
    fake=Fake()
    result=user.run('/absent','/absent',out,config='/absent',backend=fake)
    assert result['status']=='READ_ONLY_EXISTING_ATTEMPT' and result['ledger_counts']=={'STARTED':1} and not fake.calls
