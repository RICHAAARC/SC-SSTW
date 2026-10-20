"""Card85 CPU/static/fake only; no actual VAE/FFT/media experiment."""
import ast,json,subprocess,sys,zipfile
from pathlib import Path
import pytest
import torch
from main.tube_state import b_real_local_wan_v1 as m
from runtime.wan import b_real_local_wan_v1 as runtime
from runtime.wan.b_real_local_wan_evaluation_v1 import geometry
pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module',autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('start',[60,61])
def test_fixed_geometry(start):
    g=geometry(start,start+45)
    assert g['all45_frames_enter_E'] and g['first_frame_context_only']==start and g['first_latent_excluded']==0
    assert [x['source_cell'] for x in g['source_cells'] if x['physical_status']=='FULL']==list(range(16,27))
    assert sum(x['physical_status']=='PARTIAL' for x in g['source_cells'])==1
    assert sum(x['physical_status']=='ABSENT' for x in g['source_cells'])==32
    assert g['physically_present_message_groups']==[1,2] and g['present_message_bits']==g['structurally_missing_message_bits']==16
    if start==60:
        assert [x['true_source_cell'] for x in g['observed_cells']]==list(range(16,27))
        assert [x['true_group'] for x in g['observed_cells']]==[1]*7+[2]*4
    else:
        assert all(x['true_source_cell'] is None for x in g['observed_cells'])
        assert [x['true_group'] for x in g['observed_cells']]==[1]*6+[None]+[2]*4
        assert g['observed_cells'][6]['nominal_source_cells']==[22,23]
        assert all(sorted(x['nominal_frame_counts'].values())==[1,3] for x in g['observed_cells'])


def test_local_match_and_precomputed_votes(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('actual FFT not permitted in CPU test')
    monkeypatch.setattr(torch.fft,'fft2',forbidden)
    x=[[float(j==15+i) for j in range(44)] for i in range(11)]
    matched=m.match(x);assert matched['provisional_start']==16 and len(matched['scores'])==34
    mapping=m.local_map(matched);assert [r['eligible_source_cell'] for r in mapping['cells']]==list(range(16,27))
    spec=torch.zeros((4,11,40,64),dtype=torch.float32);spec[2:]=1
    raw=m.read_spectrum(spec,m.KEYS[0]);assert all(b['decoded']==0 and b['tie'] for c in raw['local_cells'] for b in c['bits'])
    result=m.reduce_payload(raw,mapping)
    assert result['decoded_bits']==[None]*8+[0]*16+[None]*8
    assert [g['observed_cells'] for g in result['segments']]==[0,7,4,0]
    assert result['segments'][1]['bits'][0]['count']==840 and result['segments'][1]['bits'][0]['zero_coefficients']==420
    assert result['trusted_full_message'] is None
    ties=m.match([[0.0]*44 for _ in range(11)]);assert ties['exact_max_ties']==list(range(1,35))
    unread=m.reduce_payload(raw,m.local_map(ties));assert unread['decoded_bits']==[None]*32 and len(unread['local_cells'])==11


class Fake:
    device='cpu'
    def __init__(self,fail_arm=None,interrupt_E=None):self.blocks=dict(encoder=0,decoder=0);self.events=[];self.n=0;self.fail_arm=fail_arm;self.interrupt_E=interrupt_E
    def load_pixels(self,path):
        arm=Path(path).parent.name;self.events.append('load/'+arm)
        if arm==self.fail_arm:raise OSError('synthetic missing readback')
        return torch.arange(181,dtype=torch.uint8)[:,None,None,None].expand(181,320,512,3)
    def load(self):self.events.append('model');return dict(fake=True)
    def encode_received(self,rgb):
        self.n+=1;self.events.append('E')
        if self.n==self.interrupt_E:raise KeyboardInterrupt('synthetic interruption')
        assert tuple(rgb.shape)==(45,320,512,3)
        assert (int(rgb[0,0,0,0]),int(rgb[-1,0,0,0])) in ((60,104),(61,105))
        self.blocks['encoder']+=12
        return torch.zeros(m.LOCAL_SHAPE,dtype=torch.float32)


def inputs(tmp_path):
    source=tmp_path/'source';truth=source/'references/truth.json';truth.parent.mkdir(parents=True);truth.write_text(json.dumps(dict(bits=[0]*32)))
    for arm in ('P0','PM'):
        for key in m.KEY_IDS:
            p=source/'blind/sync'/arm/'mp4'/(key+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(pooled_q=[[0.0]*44 for _ in range(44)])))
            p=source/'blind/payload'/arm/(key+'.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(local_cells=[dict(bits=[dict(decoded=0) for _ in range(8)]) for _ in range(44)])))
    return source,truth


def mock_fft(monkeypatch):
    calls=[]
    def fake(value,**kwargs):calls.append((tuple(value.shape),kwargs));return torch.zeros_like(value,dtype=torch.complex64)
    monkeypatch.setattr(torch.fft,'fft2',fake);return calls


def test_fake_budget_isolation_and_readonly_reentry(tmp_path,monkeypatch):
    source,truth=inputs(tmp_path);out=tmp_path/'out';backend=Fake();calls=mock_fft(monkeypatch)
    read_text=Path.read_text
    def guarded(path,*a,**k):
        if path.is_relative_to(source/'blind') or path==truth:
            assert (out/'numeric_completion.json').is_file()
            assert len(list((out/'blind').rglob('*.json')))==40
        return read_text(path,*a,**k)
    monkeypatch.setattr(Path,'read_text',guarded)
    result=runtime.run(source,out,truth,backend=backend)
    assert result['status']=='EXECUTION_COMPLETE'
    assert backend.events.count('model')==1 and backend.events.count('E')==4
    assert len(calls)==4 and all(shape==(4,11,40,64) and kw==dict(dim=(-2,-1),norm='ortho') for shape,kw in calls)
    assert result['counts']['keyed_vote_read']['returned']==8 and result['counts']['conditional_vote_reduction']['returned']==8
    assert result['counts']['sequence_score']['returned']==8 and result['counts']['post_only_sequence_score']['returned']==4
    assert result['counts']['posterior_encode']['returned']==4
    assert all(row['internal_block_forward_delta']==dict(encoder=12,decoder=0) for row in result['calls'].values() if row['kind']=='posterior_encode')
    assert len(result['calls'])==runtime.fixed_denominator()['ledger_units']==71
    marker=json.loads((out/'numeric_completion.json').read_text());assert marker['truth_read'] is False and marker['reference_read'] is False
    e=json.loads((out/'evaluation.json').read_text())
    assert e['tables']['c00/key00']['singleton_geometry_rows']==11
    b=e['tables']['c01/key00'];assert b['mixed_identity_rows']==11 and b['post_singleton_true_start'] is None and b['undefined_bit_references']==8
    assert all(row['conditional_cell_correct'] is None for row in b['observed_cells'])
    assert b['cell_identity_evaluable'] is False and b['conditional_singleton_correct_fraction'] is None
    assert b['raw_group_only_readable_bits']==80 and b['raw_single_cell_readable_bits']==0
    assert all(t['structurally_missing_bits']==16 and t['trusted_full_message'] is None for t in e['tables'].values())
    assert sum(len(t['local_bits']) for t in e['tables'].values())==704
    before=list(backend.events);second=runtime.run(source,out,truth,backend=backend)
    assert second['status']=='EXISTING_ATTEMPT_NOT_REEXECUTED' and before==backend.events and len(calls)==4


@pytest.mark.parametrize('failure',['missing','interrupt'])
def test_failure_interruption_fixed_rows_and_reentry(tmp_path,monkeypatch,failure):
    source,truth=inputs(tmp_path);out=tmp_path/'out'
    backend=Fake(fail_arm='P0' if failure=='missing' else None,interrupt_E=2 if failure=='interrupt' else None);calls=mock_fft(monkeypatch)
    result=runtime.run(source,out,truth,backend=backend);assert result['status']=='INCOMPLETE'
    assert len(list((out/'blind').rglob('*.json')))==40
    e=json.loads((out/'evaluation.json').read_text());assert len(e['tables'])==8
    assert sum(len(t['message']) for t in e['tables'].values())==256
    assert sum(len(t['observed_cells']) for t in e['tables'].values())==88
    assert sum(len(t['source_cell_rows']) for t in e['tables'].values())==352
    if failure=='missing':
        assert backend.events.count('E')==2 and len(calls)==2
        assert json.loads((out/'blind/Q/c00/key00.json').read_text())['status']=='UPSTREAM_FAILED'
    else:
        assert backend.events.count('E')==2 and len(calls)==1
        assert json.loads((out/'blind/Q/c01/key00.json').read_text())['status']=='UPSTREAM_INTERRUPTED'
    case='c00' if failure=='missing' else 'c01'
    root='load/P0' if failure=='missing' else 'E/c01'
    status='UPSTREAM_FAILED' if failure=='missing' else 'UPSTREAM_INTERRUPTED'
    mapped=json.loads((out/'blind/map'/case/'key00.json').read_text())
    conditioned=json.loads((out/'blind/payload'/case/'key00.json').read_text())
    assert mapped['status']==conditioned['status']==status
    assert mapped['unavailable_from']==conditioned['unavailable_from']==root
    assert len(mapped['cells'])==11 and all(row['status']==status and row['unavailable_from']==root for row in mapped['cells'])
    assert len(conditioned['decoded_bits'])==32
    assert all(bit['status']==status and bit['unavailable_from']==root for row in conditioned['local_cells'] for bit in row['bits'])
    before=list(backend.events);runtime.run(source,out,truth,backend=backend);assert before==backend.events


@pytest.mark.parametrize('failed_stage',['match','map'])
def test_match_map_failure_propagates_root_cause(tmp_path,monkeypatch,failed_stage):
    source,truth=inputs(tmp_path);out=tmp_path/'out';backend=Fake();fft_calls=mock_fft(monkeypatch)
    target='match' if failed_stage=='match' else 'local_map';original=getattr(m,target);invocations=[]
    def fail_once(*args,**kwargs):
        invocations.append(target)
        if len(invocations)==1:raise RuntimeError('synthetic '+failed_stage+' failure')
        return original(*args,**kwargs)
    monkeypatch.setattr(m,target,fail_once)
    read_text=Path.read_text
    def guarded(path,*a,**k):
        if path.is_relative_to(source/'blind') or path==truth:
            assert (out/'numeric_completion.json').is_file()
            assert len(list((out/'blind').rglob('*.json')))==40
        return read_text(path,*a,**k)
    monkeypatch.setattr(Path,'read_text',guarded)
    result=runtime.run(source,out,truth,backend=backend)
    assert result['status']=='INCOMPLETE'
    root=failed_stage+'/c00/key00'
    assert result['calls'][root]['status']=='FAILED'
    assert result['calls']['map/c00/key00']['status']==('FAILED' if failed_stage=='map' else 'NOT_ENTERED')
    assert result['calls']['payload/c00/key00']['status']=='NOT_ENTERED'
    mapped=json.loads((out/'blind/map/c00/key00.json').read_text())
    expected='FAILED' if failed_stage=='map' else 'UPSTREAM_FAILED'
    assert mapped['status']==expected and mapped['unavailable_from']==root
    assert len(mapped['cells'])==11 and len(mapped['source_occupancy'])==44
    assert all(r['status']==expected and r['unavailable_from']==root for r in mapped['cells']+mapped['source_occupancy'])
    payload=json.loads((out/'blind/payload/c00/key00.json').read_text())
    assert payload['status']=='UPSTREAM_FAILED' and payload['unavailable_from']==root and payload['decoded_bits']==[None]*32
    assert len(payload['local_cells'])==11
    for row in [b for seg in payload['segments'] for b in seg['bits']]+[b for cell in payload['local_cells'] for b in cell['bits']]:
        assert row['status']=='UPSTREAM_FAILED' and row['unavailable_from']==root
    # Raw observations/votes survive; only the failed conditional chain is unavailable.
    assert json.loads((out/'blind/votes/c00/key00.json').read_text())['status']=='RAW_READ'
    assert result['calls']['map/c00/key01']['status']=='RETURNED'
    assert result['calls']['payload/c00/key01']['status']=='RETURNED'
    evaluation=json.loads((out/'evaluation.json').read_text())
    assert sum(len(v['observed_cells']) for v in evaluation['tables'].values())==88
    assert sum(len(v['local_bits']) for v in evaluation['tables'].values())==704
    assert sum(len(v['message']) for v in evaluation['tables'].values())==256
    assert len(fft_calls)==4 and backend.events.count('E')==4
    before=(list(backend.events),len(invocations),len(fft_calls))
    second=runtime.run(source,out,truth,backend=backend)
    assert second['status']=='EXISTING_ATTEMPT_NOT_REEXECUTED'
    assert before==(backend.events,len(invocations),len(fft_calls))


def test_package_schema_local_interface_and_editable_cli(tmp_path):
    import nbformat
    from scripts.build_b_real_local_wan_v1_notebook import build,SOURCE_FILES
    core=ast.parse((ROOT/'main/tube_state/b_real_local_wan_v1.py').read_text());functions={n.name:n for n in core.body if isinstance(n,ast.FunctionDef)}
    assert [a.arg for a in functions['match'].args.args]==['local_matrix']
    for name in ('observe','match','local_map','read_spectrum','reduce_payload'):
        names={n.id for n in ast.walk(functions[name]) if isinstance(n,ast.Name)}
        assert not names&{'PREPARATIONS','metadata','phase','start','stop','truth'}
        assert 'bit_index' not in {n.value for n in ast.walk(functions[name]) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    notebook,archive=build();nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for code in codes:ast.parse(code)
    tree=ast.parse(codes[1]);out=tmp_path/'out';src=tmp_path/'src'
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('OUTPUT','SOURCE_ROOT'):
            node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(out if node.targets[0].id=='OUTPUT' else src))],keywords=[])
    ast.fix_missing_locations(tree);exec(compile(tree,'source-only','exec'),{})
    for name in SOURCE_FILES:assert (src/name).read_text()==(ROOT/name).read_text()
    (out/'attempt.json').write_text('{}');sentinel=src/SOURCE_FILES[0];sentinel.write_text('edited')
    exec(compile(tree,'source-only','exec'),{});assert sentinel.read_text()=='edited'
    portable=tmp_path/'editable';portable.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(portable)
    help_result=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_real_local_wan_v1_run','--help'],cwd=portable,capture_output=True,text=True)
    assert help_result.returncode==0,help_result.stderr
    assert not (portable/'.git').exists()
    setup=ast.parse(codes[2]);strings={n.value for n in ast.walk(setup) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
    assert 'experiments.wan_state_clock.b_real_local_wan_v1_run' in strings
    assert 'ffmpeg' not in strings and 'ffprobe' not in strings
