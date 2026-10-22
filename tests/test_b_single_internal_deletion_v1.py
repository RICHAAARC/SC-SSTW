"""Card103 geometry/fake engineering only; no saved-Q replay or physical imports."""
import copy,json
from pathlib import Path
import pytest
pytestmark=pytest.mark.unit
from main.tube_state import b_single_internal_deletion_v1 as m
from runtime.wan import b_single_internal_deletion_evaluation_v1 as post
from runtime.wan import b_single_internal_deletion_v1 as user

def family(value=.2):
    return dict(status='SCORES_COMPLETE',candidates=[dict(phase=p,source_start=s,S=value,score_status='RETURNED') for p,s in m.H],phase_local_bits={str(p):[0]*80 for p in range(4)})
def test_complete_factorization_and_zero_based_endpoint_identity():
    pp=m.pairs();assert len(pp)==1891 and sum(x['multiplicity'] for x in pp)==58621
    assert len({(x['a'],x['d']) for x in pp})==1891
    assert sum(x['d']==0 for x in pp)==61
    assert {4*(s-1)-p for p,s in m.H}==set(range(137))
    for row in pp:
        a,d=row['a'],row['d']
        for b in ([45,76] if d else [45]):
            actual=post.mapping(dict(a=a,b=b,d=d))
            assert len(actual)==121 and actual[0]==a and actual[-1]==a+120+d
            assert actual[:45]==list(range(a,a+45))
            assert actual[76:]==list(range(a+76+d,a+121+d))
            for ref in m.available(row).values():
                assert ref['phase']==(- (a if ref['anchor']=='left' else a+76+d))%4

def test_fixed_geometry_guard_boundary_and_partial_groups():
    cfg=user.load_config();expected=[[11,7,0,10],[6,6,9,7],[10,7,0,10],[4,10,5,10]]
    for row,full in zip(cfg['private_roster'],expected):
        geo=post.geometry(row)
        assert [g['full_cells'] for g in geo['groups']]==full
        assert all(c['retained_frames']+c['INTERNAL_GAP']+c['SOURCE_SEGMENT_BOUNDARY']==4 for c in geo['cells'])
    a=post.geometry(cfg['private_roster'][0]);assert a['groups'][2]['presence']=='STRUCTURAL_MISSING'
    assert a['groups'][2]['INTERNAL_GAP']==44 and a['groups'][2]['SOURCE_SEGMENT_BOUNDARY']==0
    b=post.geometry(cfg['private_roster'][3]);assert b['groups'][2]['presence']=='PRESENT' and not b['groups'][2]['nominal_observed']
    clean=post.geometry(dict(a=0,b=60,d=0))
    assert clean['source_frames'][0]==0 and clean['cells'][0]['retained_frames']==4
    assert clean['groups'][3]['cause']=='SOURCE_SEGMENT_BOUNDARY'

def test_threshold_full_domain_ties_and_required_failure():
    f=m.pair_family(family(),family())
    fs={h:{q:copy.deepcopy(f) for q in range(6)} for h in ('a','b')}
    fs['a'][4]=m.empty_family() # positives do not affect calibration
    t=m.calibrate(fs);assert t['status']=='FROZEN' and t['tau']==.2
    prepared=m.prepare(f,t);assert not prepared['detection'] and prepared['C_paths']==58621
    out=m.read(prepared,{})
    assert all(x['decoded'] is None for x in out['message']) and out['cached_bit_accesses']==0
    bad=family();p,s=m.state(60)
    next(x for x in bad['candidates'] if (x['phase'],x['source_start'])==(p,s))['S']=None
    g=m.pair_family(bad,family());assert g['status']=='TECHNICAL_UNRESOLVED'
    assert len(g['pairs'])==1891 and g['path_denominator']==58621
    assert all(x['in_C'] is None for x in m.prepare(g,t)['members'])
    fs['a'][0]=g;assert m.calibrate(fs)['status']=='CALIBRATION_UNRESOLVED'

def test_true_gap_aliases_common_cells_freeze_and_no_bit_switch():
    row=next(dict(r,pair_index=i,J=.8,score_status='RETURNED') for i,r in enumerate(m.pairs()) if (r['a'],r['d'])==(2,58))
    f=m.summarize([row]);p=m.prepare(f,dict(status='FROZEN',tau=.5))
    assert p['C_paths']==32 and not p['exact_seam_certified'] and p['coarse_jump_only']
    assert [g['selected_cell'] for g in p['groups']]==[2,None,None,35]
    bb={a:{str(phase):[0]*80 for phase in range(4)} for a in m.ANCHORS}
    out=m.read(p,bb);assert sum(x['decoded'] is not None for x in out['message'])==16
    before=copy.deepcopy(p);bb['left']['2'][0]=None
    out=m.read(p,bb);assert out['message'][0]['status']=='TECHNICAL_CACHE'
    assert p==before and out['groups'][0]['selected_cell']==2
    # A second retained explanation disagrees: no deleting it or trying next cell.
    other=dict(row,a=3,d=57,pair_index=999)
    p=m.prepare(m.summarize([row,other]),dict(status='FROZEN',tau=.5))
    bb={a:{str(phase):[0]*80 for phase in range(4)} for a in m.ANCHORS};bb['left']['1']=[1]*80
    out=m.read(p,bb);assert out['message'][0]['decoded'] is None and out['message'][0]['status']=='DISAGREEMENT'

def blank(value=None):
    return dict(message=[dict(slot=i,decoded=value,status='TEST') for i in range(32)],groups=[])
def test_literal_coincidence_is_not_canonical_coverage_and_union_cannot_borrow():
    private=dict(a=4,b=60,d=0);meta=dict(arm='PM',key='key00');bits=[0]*32
    wrong=blank(0);wrong['groups']=[dict(group=0,conditional_references=[
        dict(canonical_cell=1,received_interval_inclusive=[1,4],values=[0]*8)])]
    right=blank(None);right['groups']=[dict(group=0,conditional_references=[
        dict(canonical_cell=2,received_interval_inclusive=[1,4],values=[0]*8)])]
    bad=post.assess(wrong,meta,private,bits,anchor='left')
    assert bad['message'][0]['literal_correct'] is True and bad['message'][0]['supported_canonical_correct'] is False
    good=copy.deepcopy(right);good['message']=blank(0)['message']
    good=post.assess(good,meta,private,bits,anchor='left')
    c=post.compare(bad,good);assert 0 in c['literal']['correct_retained'] and 0 in c['supported_canonical']['correct_lost']
    # Right's valid local1..4 actually maps to source81..84, while it claims cell1.
    parallel=dict(left=right,right=wrong)
    parallel['right']['groups'][0]['conditional_references'][0]['received_interval_inclusive']=[1,4]
    union=m.independent_union(parallel['left'],parallel['right'])
    evaluated=post.assess(union,meta,private,bits,parallel=parallel)
    assert evaluated['message'][0]['literal_correct'] is True
    assert evaluated['message'][0]['supported_canonical_correct'] is False

class FakeBackend:
    """Tiny token lists, fake counts. Never imports torch/NumPy/FFmpeg."""
    def __init__(self,interrupt=False,fail_encode=False):
        self.blocks={'encoder':0,'decoder':0};self.interrupt=interrupt;self.fail_encode=fail_encode;self.calls=[]
    def load_vae(self):
        self.calls.append('vae')
        if self.interrupt:raise KeyboardInterrupt('fake interrupted')
        return dict(status='FAKE')
    def release_vae(self):pass
    def book(self,key):return key
    def load_source(self,path):return list(range(181))
    def edit(self,x,row):
        a,b,d=(row[k] for k in ('a','b','d'));s=x[a:a+121+d]
        return s[:b]+s[b+d:]
    def save_tensor(self,path,x):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(x))
    def load_tensor(self,path):return json.loads(Path(path).read_text())
    def save_mp4(self,pixels,path):
        self.calls.append('encode');Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(pixels));return dict(status='FAKE')
    def probe(self,path):return dict(status='FAKE')
    def read_mp4(self,path,probe):return json.loads(Path(path).read_text())
    def prepare(self,received,anchor,p):return received[anchor+p:anchor+p+41]
    def encode(self,view):
        self.calls.append('E')
        if self.fail_encode:raise ValueError('fake E failure')
        self.blocks['encoder']+=11;return view
    def observe(self,z,book):return dict(status='FAKE_Q')
    def spectrum(self,z):self.calls.append('FFT');return z
    def votes(self,spectrum,secret):return dict(local_cells=[dict(bits=[dict(decoded=0) for _ in range(8)]) for _ in range(10)])
    def score(self,banks,model):
        return family() if all(x is not None for x in banks.values()) else user.empty_anchor()
    def independent(self,fam):return blank()
    def quality(self,left,right,frames,meaning):
        assert len(left)==len(right)==len(frames)==121
        return dict(status='FAKE',frame_denominator=121,adjacent_denominator=120)

def fake_source(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'background.json').write_text(json.dumps(dict(status='BACKGROUND_FROZEN',models={k:dict(mean=[0.]*40,factor=[[float(i==j) for j in range(40)] for i in range(40)]) for k in user.KEYS})))
    (source/'protocol.json').write_text(json.dumps(dict(config=dict(writer=dict(bits=[0]*32)))))
    return source

def test_user_fake_order_complete_counts_and_read_only(tmp_path):
    source=fake_source(tmp_path);backend=FakeBackend();out=tmp_path/'out'
    result=user.run(source,out,backend=backend)
    assert result['status']=='COMPLETED_DEVELOPMENT_ONLY'
    counts=result['kinds_returned']
    assert {k:counts[k] for k in ['E41','receiver_FFT','Q','votes','MP4_encode','MP4_probe','MP4_decode','paired_quality','anchor137_score','joint1891_score']}==dict(E41=96,receiver_FFT=48,Q=192,votes=96,MP4_encode=12,MP4_probe=12,MP4_decode=12,paired_quality=16,anchor137_score=48,joint1891_score=24)
    ledger=json.loads((out/'ledger.json').read_text())
    frozen=json.loads((out/'thresholds.json').read_text())['frozen_unix']
    assert frozen<min(v['started_unix'] for k,v in ledger.items() if k.startswith(('b101-s02/','b101-s04/')) and v['kind']!='paired_quality')
    marker=json.loads((out/'markers/blind_complete.json').read_text())['saved_unix'];truth=json.loads((out/'post_truth_access.json').read_text())['opened_unix']
    assert max(v['finished_unix'] for v in ledger.values() if v['kind'] in ('joint_decision','equal_observation_references'))<=marker<truth
    assert all(v['attempts']==1 for v in ledger.values())
    prior=list(backend.calls)
    assert user.run('/absent-source',out,config='/absent-config',backend=backend)['physical_reentry'] is False
    assert backend.calls==prior
    evaluation=json.loads((out/'evaluation.json').read_text())
    assert evaluation['fixed_evaluation']['physical_present']==56 and evaluation['fixed_evaluation']['present_unobserved']==16
    assert evaluation['totals']['positive_detected']==0 # all-reject fake is NOT scientific success

def test_interruption_and_bad_E_consume_attempt_without_replay(tmp_path):
    source=fake_source(tmp_path)
    for name,backend in [('interrupted',FakeBackend(interrupt=True)),('failed',FakeBackend(fail_encode=True))]:
        out=tmp_path/name;r=user.run(source,out,backend=backend)
        assert r['status'] in ('INTERRUPTED','INCOMPLETE_FIXED_DENOMINATORS_RETAINED')
        before=list(backend.calls);assert user.run(source,out,backend=backend)['status']=='READ_ONLY_EXISTING_ATTEMPT';assert backend.calls==before
        assert not (out/'post_truth_access.json').exists()
        if name=='failed':
            assert r['threshold_status']=='CALIBRATION_UNRESOLVED'
            ledger=json.loads((out/'ledger.json').read_text());assert all(v['status']=='NOT_ENTERED' for k,v in ledger.items() if k.startswith(('b101-s02/','b101-s04/')))

def test_truth_interfaces_and_static_no_new_physical_test_imports():
    import ast,inspect
    assert list(inspect.signature(m.pair_family).parameters)==['left','right']
    assert list(inspect.signature(m.prepare).parameters)==['family','thresholds']
    source=inspect.getsource(m)
    assert not any(isinstance(n,ast.ImportFrom) and (n.module or '').startswith(('runtime','experiments')) for n in ast.walk(ast.parse(source)))
    assert 'private' not in inspect.signature(m.read).parameters

def test_quality_pair_coordinates_and_seam_are_zero_based_post_only():
    frames=post.mapping(dict(a=2,b=74,d=58));rows,pairs=post.quality_coordinates(frames)
    assert rows[0]==dict(frame=0,source_frame=2) and rows[-1]==dict(frame=120,source_frame=180)
    seam=[r for r in pairs if r['spans_private_deletion_seam']]
    assert len(seam)==1 and seam[0]['frame_before']==73 and seam[0]['frame_after']==74
    assert seam[0]['source_frame_before']==75 and seam[0]['source_frame_after']==134


def test_blind_entry_has_only_host_id_and_call_drops_private_roster():
    import ast,inspect
    assert next(iter(inspect.signature(user.blind_host).parameters))=='host_id'
    function=ast.parse(inspect.getsource(user.blind_host)).body[0]
    assert not any(isinstance(n,ast.Name) and n.id=='host' for n in ast.walk(function))
    calls=[n for n in ast.walk(ast.parse(inspect.getsource(user.run))) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='blind_host']
    assert len(calls)==1 and ast.unparse(calls[0].args[0])=="host['id']"

def test_comparison_aggregation_separates_positive_from_payload_null_gains():
    rows=[]
    names=('continuous','unaligned','independent_left','independent_right','independent_union')
    for host in range(2):
        for meta in user.QUERIES.values():
            positive=meta==dict(arm='PM',key='key00')
            payload_null=meta==dict(arm='P0',key='key00')
            c=dict(correct_retained=[],correct_lost=[0,1,2] if positive else [],
                correct_added=list(range(8)) if payload_null else [],common_count=0,new_errors=0,old_errors=0,
                literal=dict(correct_retained=[],correct_lost=[0] if positive else [],correct_added=list(range(8)) if payload_null else []))
            rows.append(dict(main=dict(metadata_post_only=meta),positive=positive,comparisons={k:c for k in names}))
    summaries=post.comparison_summaries(rows)
    for key in names:
        pos=summaries['comparisons_positive_only'][key];allq=summaries['comparisons_all_queries'][key]
        assert pos['correct_lost']==6 and pos['correct_added']==0 and pos['literal_correct_lost']==2
        assert allq['correct_lost']==6 and allq['correct_added']==16
        assert pos['query_slots']==2 and pos['message_slots']==64
        assert allq['query_slots']==12 and allq['message_slots']==384
    assert all(v['query_slots']==2 and v['message_slots']==64 for v in summaries['comparisons_by_query_class'].values())
    assert summaries['comparisons']==summaries['comparisons_positive_only']
