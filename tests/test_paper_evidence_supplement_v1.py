"""Targeted engineering fixtures only: no model, real media, or random query draws."""
from pathlib import Path
import copy
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import types
import pytest
from main.tube_state import paper_evidence_supplement_v1 as p
from runtime.wan import paper_evidence_supplement_records_v1 as rec
from runtime.wan import paper_evidence_supplement_quality_v1 as quality
from runtime.wan import paper_evidence_supplement_report_v1 as report

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.unit


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def evidence(bit=0,valid=True):
    return dict(status='READY',slots=[dict(slot=i,value=bit if valid else None,valid=valid) for i in range(32)])


def test_boolean_decisions_keep_null_missing_invalid_separate():
    values=[True,False,None,1,0,'ACCEPT','False',[],{}]
    result=report.decision_counts([{'ownership':{'decision':v}} for v in values]+[{}, {'ownership':{}}],'ownership')
    assert result['accept']==result['nonaccept']==1
    assert result['unknown']==9
    assert result['states']==dict(ACCEPT=1,NONACCEPT=1,NULL=1,INVALID_TYPE=6,MISSING=2)


def test_frozen_dedup_empty_unknown_and_true_word_exclusion():
    items=[evidence()]*3+[evidence(1)]*2+[evidence(valid=False)]*4+[None]
    counted=p.family_counts(items,actual_message=[0]*32,arm='NEW')
    assert (counted['d'],counted['u'])==(2,1)
    assert counted['uniform32_lower']==2/2**32 and counted['uniform32_upper']==3/2**32
    assert counted['wrong_query_distinct']==1 and counted['wrong_query_upper']==2/(2**32-1)
    assert p.family_counts(items,arm='OFF')['wrong_query_denominator']==2**32
    assert p.family_counts(items,actual_message=[True]*32,arm='NEW')['wrong_query_status']=='TRUTH_UNKNOWN'
    assert p.protocol_bounds()['new_query_draws']==0
    assert p.protocol_bounds()['package_uniform32_upper']==360/2**32


def test_preprocessing_matches_official_float_resize_then_divide():
    import torch
    import torch.nn.functional as f
    torch.set_num_threads(4)
    frame=(torch.arange(3*600*900).reshape(3,600,900)%256).to(torch.uint8)
    expected=f.interpolate(frame.float()[None],size=(341,512),mode='bilinear',align_corners=False,antialias=False)/255
    assert torch.equal(quality.preprocess(frame),expected)
    small=frame[:,:40,:80]
    result=quality.preprocess(small)
    assert result.shape==(1,3,40,80) and result.dtype==torch.float32
    with pytest.raises(ValueError): quality.preprocess(small.float())


def test_decoder_never_requests_missing_fixed_index(monkeypatch):
    import numpy as np
    requested=[]
    class Video:
        def __init__(self,*a,**k): assert 0<k['fault_tol']*180<1
        def __len__(self): return 180
        def get_batch(self,indices):
            requested.extend(indices)
            return types.SimpleNamespace(asnumpy=lambda:np.zeros((len(indices),4,6,3),dtype=np.uint8))
    monkeypatch.setitem(sys.modules,'decord',types.SimpleNamespace(VideoReader=Video))
    backend=quality.Backend('/unused')
    decoded=backend.decode('fake.mp4')
    assert requested==list(p.INDICES[:-1]) and 180 not in decoded['frames']
    assert backend.calls['decode_batches']==1


def test_decoder_recovery_failure_retains_all_slots(tmp_path,monkeypatch):
    class RecoveringVideo:
        def __init__(self,*a,**k): self.tolerance=k['fault_tol']
        def __len__(self): return 181
        def get_batch(self,indices):
            assert 1>self.tolerance*len(self)
            raise RuntimeError('fixture first cached-frame substitution rejected')
    monkeypatch.setitem(sys.modules,'decord',types.SimpleNamespace(VideoReader=RecoveringVideo))
    class Backend(quality.Backend):
        def load(self): return dict(model='FAKE_CPU_ENGINEERING_ONLY')
        def score(self,frame): pytest.fail('recovered frame must not reach scoring')
    result=quality.run(**args(tmp_path),backend_factory=Backend)
    assert result['complete_clips']==0 and result['frame_statuses']=={'DECODE_UNAVAILABLE':1344}
    assert result['operation_counts']['frame_score']['started']==0


def test_event_categories_include_gap_repeat_and_keep_invalid_unknown():
    edges=dict(adjacent_planned=10,adjacent_correct=6,adjacent_incorrect=3,adjacent_unknown=1,
               gap_planned=2,gap_correct=1,gap_incorrect=1,repeat_planned=1,repeat_correct=1)
    result=report.event_totals([dict(planned_edges=13,edges=edges)],13)
    assert (result['planned'],result['correct'],result['incorrect'],result['unknown'],result['technical_unknown_planned'])==(13,8,4,1,0)
    edges['gap_correct']=None
    assert report.event_totals([dict(planned_edges=13,edges=edges)],13)['technical_unknown_planned']==13


@pytest.mark.parametrize('invalid',[None,-1,True,'0',5793])
def test_invalid_payload_numerator_never_becomes_zero_ber(invalid):
    saved=dict(planned_frames=181,planned_bits=5792,observed_bits=5792,error_bits=invalid,
               missing_bits=0,erased_bits=0,full_message_frames=181,full_owner_frames=181)
    result=report.payload_totals([dict(source='fixture',view='FULL',payload=saved)],'wam')
    assert result['observed_ber'] is None and result['unknown_payload_records']==1
    assert result['unknown_payload_planned_bits']==5792 and result['planned_bits']==315648


class FakeBackend:
    def __init__(self,cache): self.calls=dict(load=0,decode=0,score=0)
    def load(self): self.calls['load']+=1;return dict(model='FAKE_CPU_ENGINEERING_ONLY')
    def decode(self,path):
        self.calls['decode']+=1
        return dict(actual_frames=181,frames={i:i for i in p.INDICES})
    def score(self,frame): self.calls['score']+=1;return float(frame)


def args(tmp_path):
    return dict(result_root=str(tmp_path/'retained'),output=str(tmp_path/'package/quality'),cache=str(tmp_path/'cache'))


def test_full_fake_budget_sibling_report_and_readonly(tmp_path):
    config=args(tmp_path)
    rec.write(tmp_path/'package/saved_evidence/summary.json',{'scope':'separate derived table'})
    backend=FakeBackend(None)
    result=quality.run(**config,backend_factory=lambda _:backend)
    assert result['complete_clips']==84 and result['planned_frame_slots']==1344
    assert backend.calls==dict(load=1,decode=84,score=1344)
    assert result['operation_counts']['frame_score']==dict(started=1344,returned=1344)
    out=Path(config['output']);pairs=rec.read(out/'quality_pairs.json')
    assert len(pairs)==84 and sum(r['complete'] for r in pairs)==36
    before={str(f):f.stat().st_mtime_ns for f in out.rglob('*') if f.is_file()}
    def prohibited(_): raise AssertionError('consumed package constructed a backend')
    again=quality.run(**config,backend_factory=prohibited)
    assert again['status']=='READ_ONLY_CONSUMED'
    assert before=={str(f):f.stat().st_mtime_ns for f in out.rglob('*') if f.is_file()}


def test_decode_failure_missing_frame_and_nonfinite_keep_slots(tmp_path):
    class Partial(FakeBackend):
        def decode(self,path):
            self.calls['decode']+=1
            if self.calls['decode']==1: raise OSError('fixture decode failure')
            return dict(actual_frames=180,frames={i:i for i in p.INDICES[:-1]})
        def score(self,frame): return float('nan') if frame==12 else float(frame)
    config=args(tmp_path);result=quality.run(**config,backend_factory=Partial)
    assert result['logical_streams']==84 and result['planned_frame_slots']==1344 and result['complete_clips']==0
    statuses=result['frame_statuses']
    assert statuses['DECODE_UNAVAILABLE']==16 and statuses['MISSING_FIXED_FRAME']==83
    assert statuses['FAILED_CONSUMED']==83
    assert all(r['complete_mean'] is None for r in rec.read(Path(config['output'])/'quality_clips.json'))


@pytest.mark.parametrize('interrupt',[False,True])
def test_model_failure_or_interrupt_consumes_package(tmp_path,interrupt):
    class Failed(FakeBackend):
        def load(self):
            if interrupt: raise KeyboardInterrupt('fixture interruption')
            raise RuntimeError('fixture load failure')
    config=args(tmp_path)
    if interrupt:
        with pytest.raises(KeyboardInterrupt): quality.run(**config,backend_factory=Failed)
    else: quality.run(**config,backend_factory=Failed)
    result=rec.read(Path(config['output'])/'quality_summary.json')
    assert result['planned_frame_slots']==1344 and result['complete_clips']==0
    assert result['operation_counts']['model_load']['started']==1
    assert result['operation_counts']['media_decode']['started']==0
    assert quality.run(**config,backend_factory=lambda _:pytest.fail('retry'))['status']=='READ_ONLY_CONSUMED'


def test_abrupt_interruption_receipt_is_readonly(tmp_path):
    config=args(tmp_path);out=Path(config['output'])
    book=rec.QualityBook(out,p.streams(config['result_root']),{})
    book.rows['model_load'].update(status='STARTED',started=1);book.save('model_load')
    result=quality.run(**config,backend_factory=lambda _:pytest.fail('model retry'))
    assert result['operation_statuses']['INTERRUPTED_CONSUMED']==1
    assert result['fixed_frame_slots']==1344 and result['saved'] is None


def test_reference_path_required_and_matched(tmp_path):
    sid=p.SOURCES[0];wan=tmp_path/'wan';path=wan/sid/'attempt01/quality.json'
    rows=[dict(source=sid,method=m,view='FULL',reference='saved OFF same second-generation view') for m in p.METHODS]
    rec.write(path,rows);refs=rec.baseline_references('/root',wan)
    assert all(r['off_media'] is None for r in refs)
    streams=p.streams('/root');off=streams[0]['media'];rows[0]['off_media']=off
    rec.write(path,rows);refs=rec.baseline_references('/root',wan)
    clips=[p.clip_summary(s,[dict(frame_index=i,status='RETURNED',score=2.) for i in p.INDICES]) for s in streams]
    pairs=p.paired_scores(clips,refs)
    assert sum(r['complete'] for r in pairs)==37
    assert len(pairs)==84


def test_missing_saved_inputs_keep_all_denominators_and_no_input_writes(tmp_path):
    inputs=[tmp_path/n for n in ('main','wan','frozen')]
    for path in inputs: path.mkdir()
    result=report.assemble(*inputs,tmp_path/'derived')
    assert result['fixed_counts']['fixed_claims']==720
    assert result['theory']['saved_unknown_slots']==360
    assert result['theory']['saved_family_mass_union_upper']==360/2**32
    for row in result['baseline_summaries']:
        assert row['positive_detection']['unknown']==60
        assert row['native_payload']['planned_frames']==9864
    assert all(not list(path.iterdir()) for path in inputs)
    with pytest.raises(ValueError): report.assemble(*inputs,inputs[0]/'derived')


def test_notebook_builder_entry_cli_fake_run_all(tmp_path,monkeypatch):
    import nbformat
    from runtime.wan import paper_evidence_supplement_v1 as cli
    builder=load_script('build_paper_evidence_supplement_v1_notebook')
    built=builder.build(tmp_path)
    assert built.read_bytes()==(ROOT/'notebooks/paper_evidence_supplement_v1.ipynb').read_bytes()
    notebook=nbformat.read(built,as_version=4);nbformat.validate(notebook)
    cells=[c.source for c in notebook.cells if c.cell_type=='code']
    assert cells[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    mounts=[];monkeypatch.setitem(sys.modules,'google.colab',types.SimpleNamespace(drive=types.SimpleNamespace(mount=mounts.append)))
    state={};exec(cells[0],state);exec(cells[1],state)
    assert state['SOURCE_REF']=='main'
    state.update(CONFIG=args(tmp_path),SOURCE_DIRECTORY=str(ROOT))
    exec(cells[2],state)
    original=quality.run;backend=FakeBackend(None)
    monkeypatch.setattr(quality,'run',lambda **kw:original(**kw,backend_factory=lambda _:backend))
    called=[]
    def worker(command,**kwargs):
        called.append(command)
        if '-m' in command and 'runtime.wan.paper_evidence_supplement_v1' in command:
            cli.main(command[command.index('runtime.wan.paper_evidence_supplement_v1')+1:])
        return types.SimpleNamespace(returncode=0)
    monkeypatch.setattr(state['entry'].subprocess,'run',worker)
    exec(cells[3],state)
    assert mounts==['/content/drive'] and backend.calls['score']==1344
    assert len(called)==2
    # A second Run all reads the package before loading an entry or installing packages.
    exec(cells[2],state);exec(cells[3],state)
    assert state['entry'] is None and len(called)==2


def test_main_reference_reaches_source_archive(tmp_path,monkeypatch):
    import inspect
    import tarfile
    import io
    entry=load_script('paper_evidence_supplement_v1_entry')
    reference=inspect.signature(entry.execute).parameters['source_ref'].default
    assert reference=='main'
    requested=[]
    def download(url,destination):
        requested.append(url)
        with tarfile.open(destination,'w:gz') as tar:
            body=b'# editable fixture source\n'
            info=tarfile.TarInfo('source/runtime/wan/paper_evidence_supplement_v1.py');info.size=len(body)
            tar.addfile(info,io.BytesIO(body))
    monkeypatch.setattr(entry.urllib.request,'urlretrieve',download)
    root=entry.prepare_source(None,reference,tmp_path/'scratch',tmp_path/'retained',tmp_path/'output')
    assert requested==['https://github.com/RICHAAARC/SC-SSTW/archive/refs/heads/main.tar.gz']
    assert (root/'runtime/wan/paper_evidence_supplement_v1.py').is_file() and not (root/'.git').exists()


def test_cli_imports_from_editable_source_without_git(tmp_path):
    from runtime.wan import paper_evidence_supplement_v1
    destination=tmp_path/'source';destination.mkdir();(tmp_path/'.git').mkdir()
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if file and (name=='main' or name.startswith('main.') or name=='runtime' or name.startswith('runtime.')):
            source=Path(file).resolve()
            if source.is_relative_to(ROOT):
                target=destination/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    env=os.environ.copy();env.update(PYTHONPATH=str(destination),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-B','-m','runtime.wan.paper_evidence_supplement_v1','--help'],cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'assemble' in result.stdout and 'quality' in result.stdout
