"""Bounded synthetic/dispatch checks only; no actual model/media/FFT execution."""
import ast,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
import pytest
import torch
from main.tube_state import b_legacy_position_margin_v1 as m
from runtime.wan import b_legacy_position_margin_v1 as runmod

pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module",autouse=True)
def single_thread():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.fixture(scope="module")
def book():return m.codebook(m.KEYS[0])


def test_legacy_rounding_seed_and_key_position(book):
    from main.tube_state.legacy_reference_v1.codebook import _build_integer_seed,_build_normalized_direction
    d=book['directions'][0,0].reshape(-1)
    seed=_build_integer_seed(m.KEYS[0]+'|direction',m.REFERENCE,0)
    assert d.tolist()==list(_build_normalized_direction(seed,768,'l2'))
    assert torch.equal(book['norm_squared'],book['directions'].flatten(2).square().sum(2))
    assert bool((book['norm_squared']!=1).any())
    assert not torch.equal(d,book['directions'][0,1].flatten())
    assert not torch.equal(d,m.codebook(m.KEYS[1])['directions'][0,0].flatten())


def test_nearest_active_inactive_FP32_support(book):
    z=torch.zeros((1,16,46,40,64),dtype=torch.float32)
    y0,y1,x0,x1=m.ROIS[0]
    # Deliberately sufficient unit must remain bitwise unchanged.
    z[0,4:16,1,y0:y1,x0:x1]=(2*book['signs'][0,0]*book['directions'][0,0]).float()
    raw=m.observe(z,book);out,receipt=m.construct(z,book,native_observation=raw)
    after=m.observe(out,book)
    assert torch.equal(z[0,4:16,1,y0:y1,x0:x1],out[0,4:16,1,y0:y1,x0:x1])
    q=raw['roi_q'][1][0][1];n2=float(book['norm_squared'][0,1])
    expected=((1-q)/n2*book['signs'][0,1]*book['directions'][0,1]).float()
    assert torch.equal(expected,out[0,4:16,2,y0:y1,x0:x1])
    assert abs(after['roi_q'][1][0][1]-1)<1e-6
    assert torch.equal(out[:,:4],z[:,:4]) and torch.equal(out[:,:,[0,45]],z[:,:,[0,45]])
    mask=torch.zeros_like(z,dtype=torch.bool)
    for a,b,c,d in m.ROIS:mask[0,4:16,1:45,a:b,c:d]=True
    assert torch.equal(out[~mask],z[~mask])
    assert sum(x['active'] for x in receipt['units'])==175
    assert abs(receipt['actual_FP32_squared_energy']-float((out.double()-z.double()).square().sum()))<1e-9
    assert receipt['FP32_correction_attempts']==0 and not receipt['old22_constraint']


def test_score_only_ties_and_competitor_sign():
    q=torch.zeros((44,4,44),dtype=torch.float64)
    raw=m.ranks(q)
    assert raw['rows'][0]['exact_max_ties']==list(range(1,45))
    assert raw['rows'][0]['provisional_best'] is None and raw['reliable_placement'] is None
    q[:,:,3]=2;raw=m.ranks(q)
    assert raw['rows'][0]['provisional_best']==4
    assert raw['rows'][0]['reliable_identity'] is None and raw['identity_status']=='UNCALIBRATED'
    assert raw['competitor_margins'][0][3]==2 and raw['competitor_margins'][0][0]==-2


def test_payload_precomputed_votes_zero_and_encounter_tie():
    signed=np.full((4,44,240),-1,dtype=np.int8);signed[2:]=1
    zeros=np.zeros_like(signed);zeros[0]=1
    raw=m.payload_from_votes(signed,zeros)
    assert raw['decoded_bits']==[0]*32  # equal counts; first encountered channel wins
    assert all(x['tie'] for c in raw['local_cells'] for x in c['bits'])
    assert raw['segments'][0]['bits'][0]['tie']
    assert raw['segments'][0]['bits'][0]['zero_coefficients']==330
    assert raw['reliable_sync_placement'] is False
    bad=signed.copy();bad[0,0,0]=1
    with pytest.raises(ValueError,match='strict'):m.payload_from_votes(bad,zeros)


def test_payload_precomputed_spectrum_matches_canonical_reducer(monkeypatch):
    def forbidden(*a,**k):raise AssertionError('No real FFT allowed')
    monkeypatch.setattr(torch.fft,'fft2',forbidden)
    spectrum=torch.arange(4*44*40*64,dtype=torch.float32).reshape(4,44,40,64)%3-1
    for key in m.KEYS:
        coords=m.grow.coordinates(key);v=spectrum[:,:,[c[0] for c in coords],[c[1] for c in coords]]
        expected=m.payload_from_votes((2*(v>0).to(torch.int8)-1).numpy(),(v==0).to(torch.int8).numpy())
        assert m.payload_from_spectrum(spectrum,key)==expected


class FakeBackend:
    device='cpu';blocks={'encoder':0,'decoder':0}
    def __init__(self,fail=None,interrupt=None):self.events=[];self.fail=fail;self.interrupt=interrupt;self.latest=None
    def event(self,name):
        self.events.append(name)
        if name==self.interrupt:raise KeyboardInterrupt('synthetic interrupt')
        if name==self.fail:raise RuntimeError('synthetic failure')
    def load_terminal(self,path):self.event('load_terminal');return torch.zeros((1,16,46,40,64),dtype=torch.float32)
    def load(self):self.event('model');return dict(fake=True)
    def decode(self,z):self.event('decode');self.latest=z;return dict(preclip=torch.tensor([0.]),float_rgb=torch.tensor([0.]))
    def quantize(self,rgb):self.event('quantize');return torch.tensor([0],dtype=torch.uint8)
    def save_mp4(self,pixels,path):self.event('codec');return dict(fake=True)
    def probe(self,path):self.event('probe');return dict(fake=True)
    def read_mp4(self,path,probe):self.event('readback');return torch.tensor([0],dtype=torch.uint8)
    def encode_full(self,pixels):self.event('encode');return self.latest


def install_fake_io(monkeypatch):
    monkeypatch.setattr(runmod,'preclip_record',lambda v:dict(status='MEASURED',frame_denominator=181,rows=[dict(frame=t) for t in range(181)]))
    monkeypatch.setattr(runmod,'distortion',lambda *a,**k:dict(status='SCORED',frames=[dict(frame=t) for t in range(181)],adjacent=[dict(pair=t) for t in range(180)]))
    calls=[]
    def fake_fft(value,**kw):
        calls.append((tuple(value.shape),kw));return torch.zeros_like(value,dtype=torch.complex64)
    monkeypatch.setattr(torch.fft,'fft2',fake_fft)
    return calls


def test_one_shot_dispatch_shared_FFT_and_truth_isolation(tmp_path,monkeypatch):
    calls=install_fake_io(monkeypatch);backend=FakeBackend();truth=tmp_path/'truth.json'
    truth.write_text(json.dumps(dict(fixed_method_semantics=dict(bits=[0]*32))))
    original=runmod.read_truth
    def guarded(path):
        assert (tmp_path/'output/numeric_completion.json').is_file()
        files=list((tmp_path/'output/blind').rglob('*.json'));assert len(files)==12
        assert all(json.loads(p.read_text())['status'] not in ('NOT_ENTERED',) for p in files)
        return original(path)
    monkeypatch.setattr(runmod,'read_truth',guarded)
    result=runmod.run('unused',tmp_path/'output',truth,backend=backend)
    assert result['status']=='EXECUTION_COMPLETE'
    assert len(calls)==2 and all(x[0]==(4,44,40,64) for x in calls)
    assert all(x[1]==dict(dim=(-2,-1),norm='ortho') for x in calls)
    for name,n in [('decode',2),('codec',2),('readback',2),('encode',2),('model',1)]:assert backend.events.count(name)==n
    counts=result['counts'];assert counts['canonical_payload_read']['returned']==4
    assert counts['direction_observation']['returned']==8
    assert counts['shared_payload_fft']['returned']==2
    assert result['native_audit']['outside_support_max']==0
    assert result['native_audit']['payload_channels_max']==0
    evaluation=json.loads((tmp_path/'output/evaluation.json').read_text())
    assert evaluation['identity_status']=='UNCALIBRATED'
    assert evaluation['payload_pairs']['key00']['PM_common_BER']==0
    old=list(backend.events)
    second=runmod.run('unused',tmp_path/'output',truth,backend=backend)
    assert second['status']=='EXISTING_ATTEMPT_NOT_REEXECUTED' and old==backend.events


@pytest.mark.parametrize('kind',['failure','interrupt'])
def test_failed_interrupted_attempt_retains_all_rows_no_reentry(tmp_path,monkeypatch,kind):
    calls=install_fake_io(monkeypatch)
    backend=FakeBackend(fail='decode' if kind=='failure' else None,interrupt='encode' if kind=='interrupt' else None)
    output=tmp_path/'output';result=runmod.run('unused',output,tmp_path/'missing_truth.json',backend=backend)
    assert result['status']=='INCOMPLETE'
    files=list((output/'blind').rglob('*.json'));assert len(files)==12
    for p in files:
        value=json.loads(p.read_text())
        assert len(value.get('pooled_q',value.get('decoded_bits')))==(44 if 'pooled_q' in value else 32)
    assert len(list((output/'quality').glob('*.json')))==2
    assert len(list((output/'quality/preclip').glob('*.json')))==2
    assert json.loads((output/'numeric_completion.json').read_text())['truth_read'] is False
    assert result['truth_status']=='UNAVAILABLE'
    evaluation=json.loads((output/'evaluation.json').read_text())
    for table in evaluation['payload'].values():
        assert table['structural_missing']==0 and table['local_structural_missing']==0
        assert table['unread_bits']==sum(x['decoded'] is None for x in table['rows'])
        assert table['local_unread_bits']==sum(x['decoded'] is None for x in table['local_bits'])
    assert any(v['status']==('FAILED' if kind=='failure' else 'INTERRUPTED') for v in result['calls'].values())
    target=json.loads((output/'blind/sync/P0/mp4/key00.json').read_text())
    assert target['status'].startswith('UPSTREAM_')
    before=list(backend.events);runmod.run('unused',output,'missing',backend=backend);assert before==backend.events


def test_package_source_only_schema_and_editable_CLI(tmp_path):
    import nbformat
    from scripts.build_b_legacy_position_margin_v1_notebook import SOURCE_FILES,build
    notebook,archive=build();nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for source in codes:ast.parse(source)
    tree=ast.parse(codes[1]);out=tmp_path/'out';src=tmp_path/'source'
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('OUTPUT','SOURCE_ROOT'):
            node.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(out if node.targets[0].id=='OUTPUT' else src))],keywords=[])
    ast.fix_missing_locations(tree);exec(compile(tree,'notebook-source','exec'),{})
    for name in SOURCE_FILES:assert (src/name).read_text()==(ROOT/name).read_text()
    # Reentry skips rewriting available source; no model/Drive execution.
    (out/'attempt.json').write_text('{}');sentinel=src/SOURCE_FILES[0];sentinel.write_text('edited')
    exec(compile(tree,'notebook-source','exec'),{});assert sentinel.read_text()=='edited'
    portable=tmp_path/'editable';portable.mkdir()
    with zipfile.ZipFile(archive) as z:z.extractall(portable)
    result=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_legacy_position_margin_v1_run','--help'],cwd=portable,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert not (portable/'.git').exists()
    setup=ast.parse(codes[2]);literals=[n.value for n in ast.walk(setup) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
    assert 'experiments.wan_state_clock.b_legacy_position_margin_v1_run' in literals
    assert 'ffmpeg' in literals
