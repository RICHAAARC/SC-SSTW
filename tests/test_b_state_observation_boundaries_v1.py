"""CPU-only semantic, fixed-attempt and ordinary-source delivery checks."""
import ast
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest
from runtime.wan import b_state_observation_boundaries_v1 as r
from scripts import build_b_state_observation_boundaries_v1_notebook as builder

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT.parents[1]/'diagnostics/b-terminal-three-group-result-20261025'


def record(native, value=1.):
    return dict(status='SCORE_OBSERVED' if native else 'NEW_E41_OBSERVED',
                retained_rows=list(range(1,11)),roi_q=[[[value]*44 for _ in range(4)] for _ in range(44 if native else 10)])


def inputs(path):
    r.dump(path/'protocol.json',dict(config=dict(model_id='fixture-no-model',writer=dict(key='watermark'),
        generation=dict(frames=181,height=320,width=512),sources=[dict(id=c['source'],a=c['a'],role='eval') for c in r.CASES])))
    r.dump(path/'background.json',dict(status='BACKGROUND_FROZEN',models=dict(key00=dict(status='BACKGROUND_FROZEN',
        dimension=40,mean=[0.]*40,factor=[[float(i==j) for j in range(40)] for i in range(40)]))))
    r.dump(path/'thresholds.json',dict(status='FROZEN',tau=.49,tau_rule='max(0,max_S_of_90_complete_null_window_families)'))
    for case in r.CASES:
        for arm in ('P0','PM'):
            r.dump(path/'blind'/case['source']/arm/'CROP/key00/plan.json',dict(locals=dict(M=dict(
                received_query_start=53,members=[dict(phase=case['phase'],s=case['s'],O_h=case['a']+53)]))))
    for row in r.make_plan(path)['units'].values():
        if row['layer'] in ('N','D'):r.dump(row['input_path'],record(row['layer']=='N',1. if row['arm']=='P0' else 2.))


class FakeBackend:
    def __init__(self, fail_load=False, hard_exit=False):
        self.calls=0;self.released=False;self.fail_load=fail_load;self.hard_exit=hard_exit
        self.blocks=dict(encoder=0,decoder=0)
    def load(self):
        if self.fail_load:raise RuntimeError('model load failure')
        return dict(status='FAKE_CPU_ONLY')
    def encode_q40(self,row,event):
        self.calls+=1;event('started')
        if self.hard_exit:os._exit(77)
        self.blocks['encoder']+=1;event('returned')
        return [1. if row['arm']=='P0' else 2.]*40
    def release(self):self.released=True


def test_plan_and_semantic_status():
    plan=r.make_plan('/input')
    assert len(plan['units'])==20 and plan['denominators']['coordinates']==800
    assert sum(x['encoder_frames'] for x in plan['units'].values())==1052
    for case in r.CASES:
        q=record(True)
        for i in range(10):
            for roi in range(4):q['roi_q'][case['s']-1+i][roi][case['s']-1+i]=100*i+roi
        assert r.q40_from_saved(q,case['s'],True)==[100*i+j for i in range(10) for j in range(4)]
    for native in (True,False):
        q=record(native);q['status']='FAILED'
        with pytest.raises(ValueError,match='status'):r.q40_from_saved(q,19,native)
    q=record(False);q['retained_rows']=list(range(10))
    with pytest.raises(ValueError,match='retain'):r.q40_from_saved(q,19,False)


def test_saved_real_small_json_reproduction():
    if not (AUDIT/'saved/background.json').is_file():pytest.skip('original external saved evidence unavailable')
    saved=AUDIT/'saved';model=r.compile_background(r.read(saved/'background.json')['models']['key00'])
    for short,s,expected in [('e01',19,.6714865840360436),('e02',16,.2711058652747822)]:
        q=r.q40_from_saved(r.read(saved/f'{short}_PM_CROP_M_truephase_Q.json'),s)
        observed=r.describe(q,model)
        assert observed['S']==pytest.approx(expected,abs=1e-14)
        assert observed['residual_energy']>=0
        q0=r.q40_from_saved(r.read(saved/f'{short}_P0_CROP_M_truephase_Q.json'),s)
        assert r.describe(q0,model)['alpha_signed']<0


def test_zero_energy_and_denominators():
    model=dict(mean=[0.]*40,factor=[[float(i==j) for j in range(40)] for i in range(40)])
    assert r.describe([0.]*40,model)['status']=='ZERO_ENERGY_SCORE_UNRESOLVED'
    assert r.describe([1.]*40,model)['S']<=1
    row=dict(status='RETURNED',q40=[1.]*40)
    p=r.paired_metrics(row,row,row,row,model)
    assert p['status']=='PARTIAL_ZERO_DENOMINATOR' and p['cosine_native'] is None and p['mean_gain'] is None


def test_geometry_failure_spends_attempt_without_model(tmp_path):
    source=tmp_path/'input';out=tmp_path/'out';inputs(source)
    p=source/'blind/b2-m3-e02/PM/CROP/key00/plan.json';d=r.read(p);d['locals']['M']['received_query_start']=54;r.dump(p,d)
    backend=FakeBackend();result=r.run(source,out,backend)
    assert result['status']=='STOPPED_FIXED_ATTEMPT' and backend.calls==0
    assert len(result['layers'])==20 and result['counts']=={'NOT_ENTERED':20}
    before={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}
    assert r.run(source,out,FakeBackend())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert before=={str(p):p.stat().st_mtime_ns for p in out.rglob('*') if p.is_file()}


def test_complete_fake_and_load_failure(tmp_path):
    source=tmp_path/'input';inputs(source)
    backend=FakeBackend();result=r.run(source,tmp_path/'ok',backend)
    assert result['counts']=={'RETURNED':20} and backend.calls==12 and backend.released
    assert result['physical_counts']['started']==result['physical_counts']['returned']==12
    assert len(r.read(tmp_path/'ok/evaluation.json')['pairs'])==10
    backend=FakeBackend(fail_load=True);result=r.run(source,tmp_path/'fail',backend)
    assert result['counts']=={'RETURNED':8,'NOT_ENTERED':12} and backend.calls==0 and backend.released
    assert r.read(tmp_path/'fail/receipts/model_load.json')['attempts']==1


def test_hard_exit_receipt_overlays_stale_ledger(tmp_path):
    source=tmp_path/'input';out=tmp_path/'out';inputs(source)
    script="import runpy,sys; m=runpy.run_path(sys.argv[1]); m['r'].run(sys.argv[2],sys.argv[3],m['FakeBackend'](hard_exit=True))"
    done=subprocess.run([sys.executable,'-c',script,str(Path(__file__)),str(source),str(out)],cwd=ROOT)
    assert done.returncode==77
    slot='b2-m3-e01/P0/A';ledger=r.read(out/'ledger.json');ledger[slot].update(status='NOT_ENTERED',attempts=0);r.dump(out/'ledger.json',ledger)
    before=(out/'ledger.json').read_bytes();report=r.run(source,out,FakeBackend())
    assert report['layers'][slot]['status']=='STARTED' and report['layers'][slot]['attempts']==1
    assert report['physical_counts']['started']==1 and len(report['layers'])==20
    assert (out/'ledger.json').read_bytes()==before


def test_drive_fsync_unavailable_is_tolerated(tmp_path,monkeypatch):
    def unsupported(_):raise OSError(errno.EINVAL,'FUSE fsync unsupported')
    monkeypatch.setattr(r.os,'fsync',unsupported)
    result=r.run(tmp_path/'missing',tmp_path/'out',FakeBackend())
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    assert (tmp_path/'out/attempt.json').is_file() and len(result['layers'])==20


def test_q_projection_and_actual_adapter_cpu():
    import torch
    from runtime.wan import b_state_observation_boundaries_backend_v1 as b
    torch.set_num_threads(1)
    book=b.state.codebook('watermark')
    z=torch.randn((1,16,46,40,64),generator=torch.Generator().manual_seed(19))
    legacy=b.state.observe(z,book)
    assert b.project_q40(z,book,19,True)==pytest.approx(r.q40_from_saved(legacy,19,True),abs=1e-12)
    short=torch.cat([z[:,:,0:1],z[:,:,19:29]],dim=2)
    assert b.project_q40(short,book,19,False)==pytest.approx(b.project_q40(z,book,19,True),abs=1e-12)
    class VAE:
        from types import SimpleNamespace
        config=SimpleNamespace(latents_mean=[0.]*16,latents_std=[1.]*16)
        def __init__(self):self.clears=0;self.input=None;self.modes=0
        def parameters(self):return iter([torch.nn.Parameter(torch.tensor(0.))])
        def clear_cache(self):self.clears+=1
        def encode(self,value):
            from types import SimpleNamespace
            self.input=value
            def mode():self.modes+=1;return torch.zeros((1,16,1,1,1))
            return SimpleNamespace(latent_dist=SimpleNamespace(mode=mode))
    vae=VAE();events=[];rgb=torch.tensor([[[[.1,.3,.8]]]])
    result=b.reencode_rgb24_readback(b.CountedVAE(vae,events.append),rgb)
    assert events==['started','returned'] and vae.clears==2 and vae.modes==1
    assert torch.equal(vae.input,rgb.permute(3,0,1,2).unsqueeze(0)*2-1) and result.dtype==torch.float32


def test_notebook_and_plain_zip_no_git(tmp_path):
    nb=r.read(ROOT/'notebooks/b_state_observation_boundaries_v1.ipynb')
    code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    assert len(code)==4 and code[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for c in code:ast.parse(c)
    assert 'files.upload' not in code[1] and 'is_relative_to(target)' in code[1]
    assert 'FqnToConfig' in code[2] and 'ftfy' in code[2]
    assert 'https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/' in code[1]
    assert "SOURCE_REF = 'dev/local-joint-state-payload-v1'" in code[1]
    with zipfile.ZipFile(ROOT/'notebooks/b_state_observation_boundaries_v1_user_bundle.zip') as archive:
        expected=builder.source_files()+['notebooks/b_state_observation_boundaries_v1.ipynb']
        assert sorted(archive.namelist())==sorted(expected)
        for name in expected:assert archive.read(name)==(ROOT/name).read_bytes()
        archive.extractall(tmp_path)
    assert not (tmp_path/'.git').exists()
    done=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_state_observation_boundaries_v1_run','--help'],cwd=tmp_path,capture_output=True,text=True)
    assert done.returncode==0 and '--input' in done.stdout
    probe=subprocess.run([sys.executable,'-c','import runtime.wan.b_state_observation_boundaries_backend_v1 as b; print(b.__file__)'],cwd=tmp_path,capture_output=True,text=True)
    assert probe.returncode==0 and str(tmp_path) in probe.stdout


def test_default_branch_download_and_existing_attempt_is_read_only(tmp_path,monkeypatch):
    import urllib.request
    requested=[]
    def download(url, target):
        requested.append(url)
        # GitHub archives wrap the source in one repository directory.
        with zipfile.ZipFile(ROOT/'notebooks/b_state_observation_boundaries_v1_user_bundle.zip') as source:
            with zipfile.ZipFile(target,'w') as dest:
                for name in source.namelist():dest.writestr('SC-SSTW-branch/'+name,source.read(name))
        return str(target),None
    monkeypatch.setattr(urllib.request,'urlretrieve',download)
    out=tmp_path/'output'
    bootstrap=builder.BOOTSTRAP.replace('{HELPERS}',builder.recovery_helpers()).replace(r.DEFAULT_OUTPUT,str(out))
    namespace={}
    exec(bootstrap,namespace)
    assert requested==['https://codeload.github.com/RICHAAARC/SC-SSTW/zip/refs/heads/dev/local-joint-state-payload-v1']
    assert (namespace['SOURCE']/builder.ENTRY).is_file() and namespace['EXISTING'] is False
    (out/'attempt.json').write_text('{}')
    original={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
    def forbidden(*args,**kwargs):raise AssertionError('existing attempt must not download, install or launch')
    monkeypatch.setattr(urllib.request,'urlretrieve',forbidden)
    monkeypatch.setattr(subprocess,'run',forbidden)
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    namespace={}
    exec(bootstrap,namespace)
    exec(builder.RUN,namespace)
    exec(builder.DISPLAY,namespace)
    assert namespace['EXISTING'] is True and namespace['SOURCE'] is None
    assert len(namespace['report']['layers'])==20
    assert original=={p.name:p.read_bytes() for p in out.iterdir() if p.is_file()}
