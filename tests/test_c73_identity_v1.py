"""Card74 targeted CPU engineering tests. No model, VAE, codec, FFT or saved host."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
import torch
import nbformat
from main.tube_state import c73_identity_v1 as method
from main.tube_state import local_segmented_terminal_sync_v1 as legacy
from runtime.wan import c73_identity_v1 as runtime
from scripts.build_c73_identity_v1_notebook import build, ROOT, SOURCE_FILES, SOURCE

pytestmark = pytest.mark.unit
torch.set_num_threads(1)


@pytest.fixture(scope="module")
def book():
    return method.codebook("watermark")


def test_basis_words_support_and_closest_pooled_correction(book):
    old = legacy.public_codebook("watermark")
    assert book["legacy_selected_basis"] == old["selected_basis"]
    assert all(set(a).isdisjoint(b) for a,b in zip(book["selected_basis"],old["selected_basis"]))
    assert torch.equal(book["words"] @ book["words"].T, torch.eye(46,dtype=torch.float64))
    for r in range(4):
        w = book["directions"][r].reshape(64,-1)
        assert torch.equal(w @ w.T, torch.eye(64,dtype=torch.float64))
        old_w = old["directions"][r,:, :,0].reshape(22,-1)
        assert float((w @ old_w.T).abs().max()) < 1e-14
    z = torch.zeros(method.LATENT_SHAPE)
    # Fixed nonzero ROI-specific host; pooled-only projection retains differences.
    for r,(y0,y1,x0,x1) in enumerate(method.ROIS):
        z[0,4:16,:,y0:y1,x0:x1] = book["directions"][r,0,None].permute(1,0,2,3).float()*(r+1)/8
    host = method.observe(z,book)
    candidate,receipt = method.construct(z,host,book)
    observed = method.observe(candidate,book)
    assert receipt["fp32_materializations"] == 1 and receipt["payload_writes"] == 0
    assert receipt["outside_written_rectangles_linf"] == receipt["payload_channels_linf"] == 0
    assert receipt["target_energy"] == 46
    assert abs(receipt["continuous_distance_squared"]-receipt["target_energy"]-receipt["host_removal_energy"]-receipt["cross_energy"]) < 1e-12
    assert max(abs(x) for row,target in zip(observed["pooled_scores"],.5*book["words"]) for x in (torch.tensor(row)-target).tolist()) < 1e-7
    assert [v["members"] for v in observed["vectors"]] == [[t] for t in range(46)]
    assert all(v["phase_status"] == "NOT_CERTIFIED" for v in observed["vectors"])
    b = torch.tensor([v["roi_scores"] for v in host["vectors"]])
    a = torch.tensor([v["roi_scores"] for v in observed["vectors"]])
    assert torch.allclose((a[:,1]-a[:,0]),(b[:,1]-b[:,0]),atol=1e-7)
    old_before=legacy.observe_state(z,"watermark",old)
    old_after=legacy.observe_state(candidate,"watermark",old)
    assert float((torch.tensor(old_before["pooled_scores"])-torch.tensor(old_after["pooled_scores"])).abs().max()) < 1e-7


def test_residual_rejects_quarter_but_same_ID_is_not_phase(book):
    words=book["words"]
    assert method.classify(words[16],words)["members"] == [16]
    quarter=.75*words[16]+.25*words[17]
    result=method.classify(quarter,words)
    assert result["members"] == []
    assert abs(result["identities"][16]["rho"]-.1) < 1e-15
    assert torch.equal(quarter>0,words[16]>0)  # sign identity alone misses phase error
    assert method.classify(.75*words[16]+.25*words[16],words)["members"] == [16]
    boundary=15*words[0]+words[1:16].sum(0)  # exact dyadic y, e240, residual15
    assert method.classify(boundary,words)["identities"][0]["status"] == "BORDERLINE_UNRESOLVED"
    zero=method.classify(torch.zeros(64),words)
    assert zero["members"] == [] and all(x["rho"] is None for x in zero["identities"])
    assert method.classify(-words[16],words)["members"] == []
    with pytest.raises(ValueError,match="finite"):
        method.classify(torch.full((64,),float('nan')),words)


def test_preparation_target_context_and_exact_D_G_reuse():
    # Tiny spatial fixtures exercise only received-frame indexing, no video/model.
    rgb=torch.arange(181,dtype=torch.uint8)[:,None,None,None].expand(-1,1,1,3)
    deleted=torch.cat((rgb[:63],rgb[64:]))
    d,rd=method.prepare_local(rgb,109)
    g,rg=method.prepare_local(deleted,108)
    assert torch.equal(d,g) and d[:,0,0,0].tolist() == list(range(104,113))
    f,rf=method.prepare_local(deleted,64)
    assert f[:5,0,0,0].tolist() == [59,60,61,62,64]
    assert f[-4:,0,0,0].tolist() == [65,66,67,68]
    p,rp=method.prepare_local(rgb,0)
    assert rp["context_real_mask"] == [False]*5 and rp["target_real_mask"] == [True]*4
    assert p[-4:,0,0,0].tolist() == [0,1,2,3]


def install_stub(monkeypatch,tmp_path,failure=None):
    output=tmp_path/'output'
    events=[]
    pixels=torch.zeros((),dtype=torch.uint8).expand(method.VIDEO_SHAPE)
    floats=torch.zeros(()).expand(method.VIDEO_SHAPE)
    def enter():
        ledger=json.loads((output/'ledger.json').read_text())
        active=[k for k,v in ledger.items() if v["status"] == "STARTED"]
        assert len(active) == 1 and (output/'attempt.json').is_file()
        assert not (output/'numeric_completion.json').exists()
        events.append(active[0])
        if active[0] == failure:
            raise RuntimeError('fixed synthetic dispatch failure')
        if failure == 'interrupt' and active[0] == 'encode/00':
            raise KeyboardInterrupt('fixed synthetic dispatch interruption')
    class Backend:
        device='cpu'
        blocks={"encoder":0,"decoder":0}
        def load_terminal(self,path): enter();return torch.zeros(method.LATENT_SHAPE)
        def load(self): enter();return {"synthetic":True}
        def decode(self,value): enter();self.blocks['decoder']+=46;return {"preclip":floats,"float_rgb":floats}
        def quantize(self,value): enter();return pixels
        def save_mp4(self,value,path): enter();path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'stub');return {"synthetic":True}
        def probe(self,path): enter();return {"synthetic":True}
        def read_mp4(self,path,probe): enter();return pixels
        def encode_local(self,clip): enter();assert clip.shape == (9,320,512,3);self.blocks['encoder']+=3;return torch.zeros((1,16,3,40,64))
    def quality(*a,**kw):
        return runtime.shared.unavailable_quality('STUB_NOT_MEASURED')
    monkeypatch.setattr(runtime,'distortion',quality)
    monkeypatch.setattr(runtime,'preclip_record',lambda value:dict(status='STUB_NOT_MEASURED',frame_denominator=181,rows=[dict(frame=t) for t in range(181)]))
    original_evaluate=runtime.evaluate
    def post_only(values):
        assert (output/'numeric_completion.json').exists()
        assert len(list((output/'blind/local').rglob('*.json'))) == 36
        for path in (output/'blind/local').rglob('*.json'):
            assert len(json.loads(path.read_text())['vectors'][0]['identities']) == 46
        events.append('post_only_evaluation')
        return original_evaluate(values)
    monkeypatch.setattr(runtime,'evaluate',post_only)
    return output,Backend(),events


@pytest.mark.parametrize('failure',[None,'model','P0/decode','encode/00','interrupt'])
def test_fixed_dispatch_retains_slots_and_never_reenters(monkeypatch,tmp_path,failure):
    output,backend,events=install_stub(monkeypatch,tmp_path,failure)
    result=runtime.run('P','S',output,backend=backend)
    assert result['status'] == ('EXECUTION_COMPLETE' if failure is None else 'INCOMPLETE')
    evaluated=json.loads((output/'evaluation.json').read_text())
    assert len(evaluated['rows']) == 1656
    assert sum(x['geometric_role']=='POSITIVE' for x in evaluated['rows']) == 3
    assert evaluated['negative_control_vectors'] == 30
    assert len(evaluated['logical_G_roles']) == 6 and evaluated['reused_identity_slots'] == 276
    assert evaluated['retained_positive'] == 0  # whole rejection is not scientific success
    assert evaluated['scientific_pass'] is None
    assert all(not x['phase_certified'] for x in evaluated['rows'])
    roles=[x for x in evaluated['roles'] if x['arm']=='S_ID' and x['case']=='F' and x['key_role']=='CORRECT']
    assert roles[0]['target_complete'] and not roles[0]['context_continuous'] and roles[0]['expected_identity']==17
    aliases=list((output/'blind/logical_reuse').glob('*.json'))
    assert len(aliases) == 6
    for p in aliases:
        alias=json.loads(p.read_text())
        assert not alias['new_physical_observation']
        assert alias['observation'] == json.loads((output/'blind'/(alias['reuse_of']+'.json')).read_text())
    if failure is None:
        counts=result['counts']
        assert counts['ordinary_decode']['returned']==3 and counts['posterior_encode']['returned']==18
        assert counts['local_identity_observation']['returned']==36
        assert counts['native_identity_observation']['returned']==6
        assert counts['legacy22_native_observation']['returned']==6
        assert not any('fft' in k.lower() for k in counts)
    before={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    prior=list(events)
    repeated=runtime.run('P','S',output,backend=backend)
    assert repeated['status']=='EXISTING_ATTEMPT_NOT_REEXECUTED' and events==prior
    assert before=={str(p.relative_to(output)):p.read_bytes() for p in output.rglob('*') if p.is_file()}
    if failure=='interrupt':
        assert result['calls']['encode/00']['status']=='INTERRUPTED'
        assert result['counts']['posterior_encode']['started']==1


def test_early_spent_attempt_and_notebook_portability(tmp_path):
    output=tmp_path/'spent';output.mkdir();(output/'attempt.json').write_text('{broken')
    assert runtime.run('missing','missing',output)['status']=='EXISTING_ATTEMPT_NOT_REEXECUTED'
    notebook,archive=build()
    nb=nbformat.read(notebook,as_version=4);nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in codes: ast.parse(c)
    assert 'https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1/notebooks/c73_identity_v1_portable_source.zip' in codes[1] and 'c73_identity_v1_run' in codes[2]
    assert all('files.upload' not in code and 'from google.colab import files' not in code for code in codes)
    assert all(not c.outputs and c.execution_count is None for c in nb.cells if c.cell_type=='code')
    extracted=tmp_path/'editable_no_git';extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        assert set(bundle.namelist())==set(SOURCE_FILES)
        for name in SOURCE_FILES: assert bundle.read(name)==(ROOT/name).read_bytes()
        bundle.extractall(extracted)
    result=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.c73_identity_v1_run','--help'],cwd=extracted,capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_optional_sync_capability_is_not_filesystem_admission(monkeypatch,tmp_path):
    import errno
    def unsupported(fd):
        raise OSError(errno.EOPNOTSUPP,'fixed unsupported fsync fixture')
    monkeypatch.setattr(runtime.os,'fsync',unsupported)
    runtime.dump(tmp_path/'record.json',{'written':True})
    assert json.loads((tmp_path/'record.json').read_text())=={'written':True}
    assert errno.EOPNOTSUPP in runtime.SYNC_UNAVAILABLE_ERRNOS
    def io_failure(fd):
        raise OSError(errno.EIO,'fixed real I/O failure fixture')
    monkeypatch.setattr(runtime.os,'fsync',io_failure)
    with pytest.raises(OSError):
        runtime.dump(tmp_path/'io_failure.json',{'written':False})


def test_nine_frame_backend_uses_posterior_mode_normalization_and_cache_reset():
    from types import SimpleNamespace
    events=[]
    class FakeVAE:
        config=SimpleNamespace(latents_mean=[1.]*16,latents_std=[2.]*16)
        def parameters(self):
            yield torch.zeros(())
        def clear_cache(self):
            events.append('clear')
        def encode(self,rgb):
            events.append(('encode',tuple(rgb.shape),float(rgb.min()),float(rgb.max())))
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=lambda:torch.full((1,16,3,40,64),3.)))
    backend=object.__new__(runtime.WanBackend)
    backend.vae=FakeVAE()
    out=backend.encode_local(torch.zeros((9,320,512,3),dtype=torch.uint8))
    assert out.shape==(1,16,3,40,64) and bool((out==1).all())
    assert events==['clear',('encode',(1,3,9,320,512),-1.,-1.),'clear']


def test_guard_inclusive_quality_and_clipping_records():
    from runtime.wan.c73_identity_quality_v1 import distortion,preclip_record
    base=torch.zeros((),dtype=torch.uint8).expand(method.VIDEO_SHAPE)
    changed=torch.zeros(method.VIDEO_SHAPE,dtype=torch.uint8)
    # Source0 is a newly written guard; nominal ROI starts at y64,x96 pixels.
    changed[0,64,96,0]=255
    old=distortion(base,changed,includes_guard=False,meaning='synthetic quality fixture')
    new=distortion(base,changed,includes_guard=True,meaning='synthetic quality fixture')
    assert len(new['frames'])==181 and len(new['adjacent'])==180
    assert old['frames'][0]['outside_nominal_support']['sum_squared']==1
    assert new['frames'][0]['outside_nominal_support']['sum_squared']==0
    assert new['frames'][0]['whole']['sum_squared']==1
    assert new['adjacent'][0]['outside_union_nominal_support']['sum_squared']==0
    floats=torch.zeros(method.VIDEO_SHAPE)
    floats[0,0,0,0]=-1;floats[-1,0,0,0]=2
    clipping=preclip_record(floats)
    assert clipping['below_zero']==clipping['above_one']==1
    assert clipping['clipping_squared_distance']==2 and len(clipping['rows'])==181


def test_notebook_source_download_local_inputs_and_spent_attempt(monkeypatch,tmp_path):
    import shutil
    import urllib.request
    notebook,archive=build()
    url="https://raw.githubusercontent.com/RICHAAARC/SC-SSTW/dev/local-joint-state-payload-v1/notebooks/c73_identity_v1_portable_source.zip"
    calls=[]
    def download(location,target):
        calls.append(location)
        Path(target).parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(archive,target)
        return str(target),None
    monkeypatch.setattr(urllib.request,'urlretrieve',download)
    # Execute only the source-loading cell with a local download stub.
    # No setup/runner/model/notebook execution and no live network.
    source=SOURCE.replace('/content/',str(tmp_path)+'/')
    env=dict(Path=Path,OUTPUT=tmp_path/'output',execution_artifacts=lambda output:[])
    exec(source,env)
    assert calls==[url]
    extracted=env['SOURCE_ROOT']
    assert (extracted/'experiments/wan_state_clock/c73_identity_v1_run.py').is_file()
    calls.clear()
    for location in (str(extracted),str(archive)):
        local_env=dict(Path=Path,OUTPUT=tmp_path/'output',execution_artifacts=lambda output:[])
        exec(source.replace(url,location),local_env)
        assert (local_env['SOURCE_ROOT']/'main/tube_state/c73_identity_v1.py').is_file()
    assert calls==[]
    spent_env=dict(Path=Path,OUTPUT=tmp_path/'spent',execution_artifacts=lambda output:['attempt.json'])
    exec(source,spent_env)
    assert calls==[] and 'SOURCE_ROOT' not in spent_env
