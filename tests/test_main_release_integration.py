"""Main integration CPU contracts: stable reader, M05 production flow, portable layers."""
import ast,copy,hashlib,json,os,shutil,subprocess,sys
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import torch
import nbformat
from main.tube_state import payload_reader,grow_video_reference as layout
from runtime.wan import provenance
from runtime.wan.video_trajectory_payload_gt_v1 import read_payload
from runtime.wan.video_trajectory_receiver_estimated_align_v1 import read_detailed
from experiments.wan_state_clock import video_trajectory_payload_framewise_sync_m05_v1_run as m05
from experiments.wan_state_clock import receiver_records
pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]
EXPECTED_AST="dc894b959343840ca607277a420343dc5189ea9f688987cb31368f93a7ba32b0"

def test_stable_reader_exact_ast_fft_sequence_zero_and_ties():
    node=next(n for n in ast.parse((ROOT/'main/tube_state/payload_reader.py').read_text()).body if isinstance(n,ast.FunctionDef))
    assert hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()==EXPECTED_AST
    torch.set_num_threads(2)
    for frames,R in ((181,44),(177,44),(89,22)):
        z=torch.randn(1,16,(frames-1)//4+1,40,64,generator=torch.Generator().manual_seed(frames));z[0,0]=0
        for key in ('watermark','watermark-wrong'):
            row=read_payload(z,key,frames);d=read_detailed(z,key,frames);coords=layout.coordinates(key)
            values=torch.fft.fft2(z.float(),dim=(-2,-1),norm='ortho').real
            expected=[];counts=[]
            for ch in range(4):
                for bit in range(8):
                    votes=[int(values[0,ch,t,h,w]>0) for t in range(1,R+1) for h,w in coords[bit::8]]
                    expected.append(Counter(votes).most_common(1)[0][0]);counts.append(dict(ones=sum(votes),zeros=len(votes)-sum(votes),count=len(votes)))
            assert row['decoded_bits']==expected and row['votes']==counts and d['original_reader_match']
            assert row['R']==R and row['received_frames']==frames and all(x==0 for x in expected[:8])
            assert np.asarray(d['zero_coefficient_mask'])[0].all() and (np.asarray(d['signed_votes'])[0]==-1).all()
    with pytest.raises(ValueError):read_payload(z,'watermark',90)

class WriterFixture:
    closed=False
    def __init__(self,c):WriterFixture.closed=False
    @staticmethod
    def read_source(s):return torch.zeros(181,2,2,3,dtype=torch.uint8)
    def encode(self,x):return np.zeros((181,4,40,64),np.float32)
    write=staticmethod(m05.M05Backend.write)
    def decode(self,z):return torch.zeros(181,2,2,3,dtype=torch.uint8)
    @staticmethod
    def transport(rgb,out,count,event):
        for name in ('mp4_save','mp4_probe','mp4_readback'):count(name,False);count(name,True)
        event('fixture',dict(status='CPU_FAKE_CODEC_NOT_MEDIA',shape=list(rgb.shape)))
        return rgb.clone()
    score=staticmethod(m05.M05Backend.score)
    def close(self):WriterFixture.closed=True
class WanFixture:
    closed=False
    def __init__(self,c):assert WriterFixture.closed;WanFixture.closed=False
    def encode(self,x):return torch.zeros(1,16,46,40,64)
    read=staticmethod(read_detailed)
    def close(self):WanFixture.closed=True

def test_m05_actual_runner_success_message_after_seal_and_missing(tmp_path,monkeypatch):
    cfg=m05.load_config();out=tmp_path/'m05';original=receiver_records.read
    def guarded(path):
        if str(path)==cfg['posthoc_config']:assert (out/'blind_readouts.json').is_file()
        return original(path)
    monkeypatch.setattr(receiver_records,'read',guarded)
    result=m05.run(out,cfg=cfg,writer_type=WriterFixture,wan_type=WanFixture)
    assert result['status']=='COMPLETE' and all(x['match'] for x in result['call_integrity'].values())
    assert result['writer']['target']==.5 and WriterFixture.closed and WanFixture.closed
    assert all(x['status']=='READ' for x in result['payload_reads'].values())
    assert sum(len(x['bit_rows']) for x in result['posthoc'].values())==64
    class Missing(WriterFixture):
        @staticmethod
        def read_source(s):raise FileNotFoundError('CPU fixture missing input')
    bad=m05.run(tmp_path/'missing',cfg=cfg,writer_type=Missing,wan_type=WanFixture)
    assert bad['status']=='INCOMPLETE' and all(x['status']=='FAILED' for x in bad['payload_reads'].values())
    assert all(x['status']=='FAILED' for x in bad['sync_reads'].values()) and sum(len(x['bit_rows']) for x in bad['posthoc'].values())==64
    assert bad['calls']['framewise_load']['attempted']==bad['calls']['wan_load']['attempted']==0

def copy_release(destination):
    for folder in ('main','runtime','experiments'):shutil.copytree(ROOT/folder,destination/folder)
    shutil.copy2(ROOT/'release_manifest.json',destination/'release_manifest.json');return destination

def isolated(root,code):
    result=subprocess.run([sys.executable,'-I','-c','import sys;sys.path.insert(0,'+repr(str(root))+');'+code],cwd=root,text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr

def test_no_outer_layers_and_no_git_parent_are_real_subprocesses(tmp_path):
    root=copy_release(tmp_path/'release')
    subprocess.run(['git','init','--quiet',str(tmp_path)],check=True)
    subprocess.run(['git','-c','user.name=CPU Fixture','-c','user.email=cpu-fixture@example.invalid','commit','--allow-empty','-m','temporary parent trap','--quiet'],cwd=tmp_path,check=True)
    parent_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tmp_path,text=True).strip()
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==parent_sha
    assert not (root/'.git').exists()
    isolated(root,"from runtime.wan.provenance import source_identity;p=source_identity('.');assert p['git_commit'] is None and p['manifest_status']=='MATCH';from experiments.wan_state_clock import video_trajectory_receiver_estimated_align_v1_run as a,video_trajectory_internal_single_deletion_v1_run as d,video_trajectory_payload_framewise_sync_m05_v1_run as w;assert a.ROOT==d.ROOT==w.ROOT")
    for module in ('video_trajectory_payload_framewise_sync_m05_v1_run','video_trajectory_receiver_estimated_align_v1_run','video_trajectory_internal_single_deletion_v1_run'):
        r=subprocess.run([sys.executable,'-I','-c',"import runpy,sys;sys.path.insert(0,"+repr(str(root))+");sys.argv=['entry','--help'];runpy.run_module('experiments.wan_state_clock."+module+"',run_name='__main__')"],cwd=root,text=True,capture_output=True)
        assert r.returncode==0 and '--config' in r.stdout
        cfg='experiments/wan_state_clock/configs/'+module.removesuffix('_run')+'.json';out=tmp_path/module
        code="import runpy,sys;sys.path.insert(0,"+repr(str(root))+");sys.argv=['entry','--config',"+repr(str(root/cfg))+",'--output',"+repr(str(out))+"];runpy.run_module('experiments.wan_state_clock."+module+"',run_name='__main__')"
        r=subprocess.run([sys.executable,'-I','-c',code],cwd=root,text=True,capture_output=True)
        assert r.returncode==1,r.stdout+r.stderr
        saved=json.loads((out/'result.json').read_text());assert saved['status']=='INCOMPLETE'
        assert saved['source_sha'] is None and saved['source_provenance']['manifest_status']=='MATCH'
        assert len(saved['payload_reads'])==({'video_trajectory_payload_framewise_sync_m05_v1_run':2,'video_trajectory_receiver_estimated_align_v1_run':12,'video_trajectory_internal_single_deletion_v1_run':16}[module])
    shutil.rmtree(root/'experiments')
    isolated(root,"import torch;from main.tube_state.payload_reader import payload_read;from runtime.wan.video_trajectory_receiver_estimated_align_v1 import read_detailed;from main.tube_state.video_trajectory_internal_single_deletion_v1 import correction;assert len(correction(dict(family='H1',b=2,k=88))['received_index_map'])==177;assert read_detailed(torch.zeros(1,16,23,40,64),'watermark',89)['original_reader_match']")

def test_all_notebook_schema_and_draft_publication_gate():
    from governance.harness.checks import notebook_binding
    policy=json.loads((ROOT/'governance/policies/validation.json').read_text())
    for name in policy['published_release_notebooks']:
        nb=nbformat.read(ROOT/name,as_version=4);nbformat.validate(nb)
        binding=nb.metadata['candidate_binding']
        assert binding['status']==('UNPUBLISHED_DRAFT' if binding['source_sha'] is None else 'PUBLISHED_SHA_BOUND')
        for cell in nb.cells:
            if cell.cell_type=='code':ast.parse(cell.source);assert cell.outputs==[] and cell.execution_count is None
    errors=notebook_binding(ROOT,policy)
    drafts=sum(json.loads((ROOT/name).read_text())['metadata']['candidate_binding']['source_sha'] is None for name in policy['published_release_notebooks'])
    assert len(errors)==drafts and all('missing fixed published SOURCE_SHA' in e for e in errors)


@pytest.mark.parametrize('stage',['writer_close','wan_close','posthoc'])
def test_m05_catchable_interrupt_persists_and_rethrows(tmp_path,monkeypatch,stage):
    cfg=m05.load_config();out=tmp_path/'interrupted'
    class IW(WriterFixture):
        def close(self):
            super().close()
            if stage=='writer_close':raise KeyboardInterrupt('fixture writer release')
    class IV(WanFixture):
        def close(self):
            super().close()
            if stage=='wan_close':raise KeyboardInterrupt('fixture Wan release')
    original=receiver_records.read
    def interrupted(path):
        if stage=='posthoc' and str(path)==cfg['posthoc_config']:raise KeyboardInterrupt('fixture posthoc')
        return original(path)
    monkeypatch.setattr(receiver_records,'read',interrupted)
    with pytest.raises(KeyboardInterrupt):m05.run(out,cfg=cfg,writer_type=IW,wan_type=IV)
    saved=json.loads((out/'result.json').read_text());assert saved['stage']=='FINISHED' and saved['status']=='INCOMPLETE'
    assert sum(len(x['bit_rows']) for x in saved['posthoc'].values())==64
    if stage=='writer_close':assert saved['calls']['wan_load']['attempted']==0


@pytest.mark.parametrize('message',['','OK','OKOKX'])
def test_m05_invalid_message_retains_64_postseal_failures(tmp_path,monkeypatch,message):
    torch.set_num_threads(2)
    cfg=m05.load_config();path=tmp_path/'posthoc.json';path.write_text(json.dumps(dict(message=message)))
    cfg.update(posthoc_config=str(path),posthoc_config_sha256=receiver_records.digest(path));out=tmp_path/'run'
    original=receiver_records.read
    def guarded(p):
        if Path(p)==path:assert (out/'blind_readouts.json').is_file()
        return original(p)
    monkeypatch.setattr(receiver_records,'read',guarded)
    result=m05.run(out,cfg=cfg,writer_type=WriterFixture,wan_type=WanFixture)
    assert result['status']=='INCOMPLETE' and result['stage']=='FINISHED'
    assert any(x['stage']=='POSTHOC' for x in result['failures'])
    assert all(x['status']=='READ' for x in result['payload_reads'].values())
    assert all(x['status']=='FAILED' and len(x['bit_rows'])==32 and all(b['status']=='FAILED' for b in x['bit_rows']) for x in result['posthoc'].values())


@pytest.mark.parametrize('bits',[[0]*31,[0]*33,[2]+[0]*31])
def test_m05_invalid_decoded_shape_is_not_partial_evaluation(tmp_path,bits):
    torch.set_num_threads(2)
    class Broken(WanFixture):
        @staticmethod
        def read(z,key,n):
            row=read_detailed(z,key,n);row['original_readout']['decoded_bits']=bits;return row
    result=m05.run(tmp_path/'run',writer_type=WriterFixture,wan_type=Broken)
    assert result['status']=='INCOMPLETE'
    assert all(x['status']=='FAILED' and len(x['bit_rows'])==32 for x in result['posthoc'].values())


@pytest.mark.parametrize('interrupt_stage',['mp4','probe','rgb24'])
def test_m05_production_media_interrupt_closes_only_unfinished(tmp_path,monkeypatch,interrupt_stage):
    from runtime.wan import fixed_rgb_media as media
    from types import SimpleNamespace
    torch.set_num_threads(2)
    monkeypatch.setattr(media,'SHAPE',(181,2,2,3));monkeypatch.setattr(media,'RGB_BYTES',181*2*2*3)
    raw=bytes(media.RGB_BYTES);stages=[];success={};receipt_bytes={};original_run=media.subprocess.run
    def process(command,**kw):
        if command[0] not in ('ffmpeg','ffprobe'):return original_run(command,**kw)
        stage='mp4' if '-c:v' in command else ('probe' if command[0]=='ffprobe' else 'rgb24');stages.append(stage)
        if stage==interrupt_stage:raise KeyboardInterrupt('fixture '+stage)
        if stage=='mp4':Path(command[-1]).write_bytes(b'fixture-mp4');out=b''
        elif stage=='probe':out=json.dumps(dict(streams=[dict(height=2,width=2,nb_frames='181')])).encode()
        else:out=raw
        return SimpleNamespace(stdout=out,stderr=b'',returncode=0)
    monkeypatch.setattr(media.subprocess,'run',process)
    class Transport(WriterFixture):
        @staticmethod
        def transport(rgb,out,count,event):
            source=media.save_raster(rgb,out/'source.rgb8')
            def capture(name,row):
                event(name,row)
                if row['status'] in ('SAVED','COMPLETE'):
                    success[name]=copy.deepcopy(row)
                    for k,v in row.items():
                        if k.endswith('_path') and Path(v).is_file():receipt_bytes[v]=Path(v).read_bytes()
            return media.mp4_roundtrip(source['path'],source['sha256'],out/'clip.mp4',out/'received.rgb8',count=count,event=capture)
    out=tmp_path/'run'
    with pytest.raises(KeyboardInterrupt):m05.run(out,writer_type=Transport,wan_type=WanFixture)
    result=json.loads((out/'result.json').read_text());expected=['mp4','probe','rgb24'][:['mp4','probe','rgb24'].index(interrupt_stage)+1]
    assert stages==expected and result['media'][interrupt_stage]['status']=='INTERRUPTED'
    assert all(result['media'][name]==row for name,row in success.items())
    assert all(Path(p).read_bytes()==b for p,b in receipt_bytes.items())
    assert result['status']=='INCOMPLETE' and result['stage']=='FINISHED' and result['calls']['wan_load']['attempted']==0
    assert sum(len(x['bit_rows']) for x in result['posthoc'].values())==64
    for i,name in enumerate(('mp4_save','mp4_probe','mp4_readback')):
        n=len(expected);assert result['calls'][name]==dict(attempted=int(i<n),completed=int(i<n-1))


@pytest.mark.parametrize('name',['video_trajectory_payload_framewise_sync_m05_v1','video_trajectory_receiver_estimated_align_v1','video_trajectory_internal_single_deletion_v1'])
def test_notebook_real_config_binding_before_run_and_display(tmp_path,name):
    from scripts import notebook_common as nb
    cfg=json.loads((ROOT/'experiments/wan_state_clock/configs'/f'{name}.json').read_text())
    path=tmp_path/'input.json';path.write_text(json.dumps(cfg));fixed=cfg['fixed_denominator'];calls=[]
    def namespace(label):return dict(SOURCE_SHA='a'*40,CONFIG_PATH=path,ENTRYPOINT='experiments.wan_state_clock.'+name+'_run',OUTPUT_ROOT=str(tmp_path/label),FIXED=fixed)
    env=namespace('good');exec(nb.SETUP,env)
    expected_hash=hashlib.sha256(path.read_bytes()).hexdigest();assert env['CONFIG_SHA256']==expected_hash
    result=dict(source_sha='a'*40,config_sha256=expected_hash,fixed_denominator=fixed,status='COMPLETE',failures=[])
    def logged(command,*a,**kw):
        calls.append(command);env['RUN_OUTPUT'].mkdir();(env['RUN_OUTPUT']/'result.json').write_text(json.dumps(result));return 0
    env['logged']=logged;exec(nb.RUN,env);assert len(calls)==1;exec(nb.DISPLAY,env)
    for field,value in [('config_sha256','b'*64),('fixed_denominator',{})]:
        bad=dict(result);bad[field]=value;env['RESULT_PATH'].write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):exec(nb.DISPLAY,env)
    path.write_text(json.dumps(dict(cfg,unexpected_edit=True)))
    with pytest.raises(RuntimeError,match='changed after setup'):exec(nb.RUN,env)
    assert len(calls)==1
    for field,value in [('name','another_official_entry'),('fixed_denominator',{})]:
        path.write_text(json.dumps(dict(cfg,**{field:value})));bad_env=namespace(field)
        with pytest.raises(RuntimeError):exec(nb.SETUP,bad_env)
        assert not Path(bad_env['OUTPUT_ROOT']).exists()
