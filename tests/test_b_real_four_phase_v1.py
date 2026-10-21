"""Card89 engineering only: fake observations/backend/spectrum, never saved real replay."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import ast
import json
import subprocess
import sys
import tempfile
import zipfile
from unittest.mock import Mock, patch
import pytest
import torch
from main.tube_state import b_phase_centered_saved_v1 as cpu
from main.tube_state import b_real_four_phase_v1 as physical
from runtime.wan import b_real_four_phase_v1 as runtime
from runtime.wan import b_real_four_phase_evaluation_v1 as post

pytestmark=pytest.mark.unit


def fake_q():
    return dict(status='ENGINEERING_FAKE_Q',roi_q=[[[0.0]*44 for _ in range(4)] for _ in range(10)],
        pooled_q=[[0.0]*44 for _ in range(10)],truth_inputs=False)


def fake_votes():
    return dict(status='ENGINEERING_FAKE_RAW',signed_votes=[[[1]*240 for _ in range(10)] for _ in range(4)],
        zero_coefficient_mask=[[[0]*240 for _ in range(10)] for _ in range(4)],
        local_cells=[dict(local_row=r,bits=[dict(slot=b,decoded=1,ones=120,count=120,tie=False,zero_coefficients=0) for b in range(8)]) for r in range(1,11)],
        truth_inputs=False,message_inputs=False)


def fake_saved_table():
    # Saved-status stub, not recomputed old280 scores or Counter reductions.
    t=cpu.blank_table(range(1,36))
    t['status']='NUMERIC_COMPLETE_UNCALIBRATED'
    for row in t['scores']:
        row.update(status='RETURNED',F=0.0,V=0.0,raw_pooled=0.0,centered_pooled=0.0,centered_min=0.0)
    for row in t['candidates']:
        row['status']='CONDITIONAL_REDUCED_UNCALIBRATED'
        for g,segment in enumerate(row['segments']):
            cells=[x for x in row['cells'] if x['group']==g]
            segment.update(local_rows=[x['local_row'] for x in cells],observed_cells=len(cells))
            for b,bit in enumerate(segment['bits']):
                value=1 if cells else None
                bit.update(status='UNCALIBRATED_CONDITIONAL_READ' if cells else 'UNREAD_LOCATION',
                    decoded=value,count=120*len(cells),ones=120*len(cells),tie=False if cells else None,zero_coefficients=0)
                row['decoded_bits'][8*g+b]=value
    return t


class FakeBackend:
    def __init__(self,fail=None):
        self.fail=fail;self.loads=0;self.encodes=0
        self.blocks=dict(encoder=0,decoder=0)
    def load(self):
        self.loads+=1
        return dict(engineering_fake=True,device='cpu')
    def load_pixels(self,path):
        return torch.arange(181,dtype=torch.uint8).reshape(181,1,1,1).expand(181,320,512,3)
    def encode_received(self,pixels):
        expected=[62,63,64,62,63,64][self.encodes]
        self.encodes+=1
        assert tuple(pixels.shape)==(41,320,512,3)
        assert torch.equal(pixels[:,0,0,0],torch.arange(expected,expected+41,dtype=torch.uint8))
        self.blocks['encoder']+=11
        if self.encodes==1 and self.fail:
            raise self.fail('engineering fake E failure')
        return torch.zeros(physical.LOCAL_SHAPE,dtype=torch.float32)


def exercise(output,fail=None):
    output=Path(output)
    backend=FakeBackend(fail)
    truth=output.parent/'engineering_truth.json'
    truth.write_text(json.dumps(dict(status='ENGINEERING_FAKE_TRUTH',bits=[0]*32)))
    stored=fake_saved_table()
    reads=[]
    def reader(archive,member):
        reads.append(member)
        if member.startswith('post/'):
            assert (output/'numeric_completion.json').is_file()
            return None
        if member.startswith('blind/Q/'):
            return fake_q()
        if member.startswith('blind/votes/'):
            return fake_votes()
        return deepcopy(stored)
    score,reducer=cpu.score,cpu.reduce_all
    def new_score(q,table,counts):
        assert len(table['scores'])==34 and table['scores'][0]['source_start']==2
        return score(q,table,counts)
    def new_reduce(votes,table,counts):
        assert len(table['candidates'])==34
        return reducer(votes,table,counts)
    with patch.object(physical.carrier,'codebook',return_value={}),patch.object(physical,'observe',side_effect=lambda z,b:fake_q()),\
         patch.object(physical,'spectrum',return_value=object()) as fft,patch.object(physical,'read_spectrum',side_effect=lambda v,k:fake_votes()),\
         patch.object(torch.fft,'fft2',side_effect=AssertionError('no real FFT')),patch.object(torch.cuda,'is_available',return_value=False),\
         patch.object(cpu,'score',side_effect=new_score) as scores,patch.object(cpu,'reduce_all',side_effect=new_reduce) as reducers:
        result=runtime.run('engineering-only-not-real-data',output,'engineering-reader-only',truth,backend=backend,sidecar_reader=reader)
    return result,backend,reads,fft.call_count,scores.call_count,reducers.call_count


def test_fake_complete_budget_reuse_isolation_and_readonly(tmp_path):
    out=tmp_path/'engineering_run'
    result,backend,reads,fft,scores,reducers=exercise(out)
    assert result['status']=='EXECUTION_COMPLETE'
    assert (backend.loads,backend.encodes,fft,scores,reducers)==(1,6,6,12,12)
    counts=result['actual_new_CPU']
    assert (counts['half_means'],counts['half_terms'],counts['raw_means'],counts['raw_terms'])==(816,8160,408,4080)
    assert (counts['candidate_reducers_returned'],counts['candidate_message_rows_returned'],counts['group_bit_counters_returned'],counts['candidate_vote_terms'])==(408,13056,5856,3916800)
    assert len(reads)==25 and sum(x.startswith('blind/') for x in reads)==24
    assert len(result['calls'])==107
    for n in [f'c{i:02}/key{k:02}' for i in range(4) for k in range(2)]:
        family=json.loads((out/'blind/families'/(n+'.json')).read_text())
        assert len(family['candidates'])==137
        assert all(x['candidate_ref']['observation'].endswith('/'+n.split('/')[1]) for x in family['candidates'])
        assert len(family['maxima']['centered_min']['exact_max_ties'])==137
        assert family['maxima']['centered_min']['singleton_conditional_ref'] is None
    marker=json.loads((out/'numeric_completion.json').read_text())
    assert all(row['started_unix']>marker['completed_unix'] for n,row in result['calls'].items() if n.startswith('post/') or n=='evaluation')
    with patch.object(runtime,'WanBackend',side_effect=AssertionError('no model reentry')),patch.object(cpu,'score',side_effect=AssertionError('no score reentry')):
        again=runtime.run('unused',out,'unused','unused')
    assert again['status']=='EXISTING_ATTEMPT_NOT_REEXECUTED'


@pytest.mark.parametrize('failure',[RuntimeError,KeyboardInterrupt])
def test_fake_failure_or_interrupt_keeps_complete_denominators(tmp_path,failure):
    out=tmp_path/'engineering_failed_run'
    result,backend,_,fft,_,_=exercise(out,failure)
    assert result['status']=='INCOMPLETE'
    assert backend.encodes==(6 if failure is RuntimeError else 1)
    assert fft==(5 if failure is RuntimeError else 0)
    assert result['calls']['E/u02']['status']==('FAILED' if failure is RuntimeError else 'INTERRUPTED')
    for n in [f'c{i:02}/key{k:02}' for i in range(4) for k in range(2)]:
        family=json.loads((out/'blind/families'/(n+'.json')).read_text())
        assert len(family['candidates'])==137
        if n.startswith(('c00/','c01/')) or failure is KeyboardInterrupt:
            assert family['status']=='INCOMPLETE_PHASE_FAMILY'
            assert family['maxima']['centered_min']['singleton_conditional_ref'] is None
    bad=json.loads((out/'blind/unique/u02/key00/table.json').read_text())
    assert bad['scores'][0]['unavailable_from']=='E/u02'
    assert bad['candidates'][0]['unavailable_from']=='E/u02'
    assert len(bad['candidates'])==34
    assert sum(len(x['decoded_bits']) for x in bad['candidates'])==1088


def test_precomputed_spectrum_strict_zero_and_counter_order():
    value=torch.full((4,10,40,64),-1.0)
    coords=physical.carrier.grow.coordinates(physical.KEYS[0])
    value[:2,:,[y for y,x in coords],[x for y,x in coords]]=1
    y,x=coords[0]
    value[2,0,y,x]=0
    with patch.object(torch.fft,'fft2',side_effect=AssertionError('no FFT')):
        raw=physical.read_spectrum(value,physical.KEYS[0])
    assert raw['local_cells'][0]['bits'][0]['decoded']==1
    assert raw['local_cells'][0]['bits'][0]['tie'] is True
    assert raw['local_cells'][0]['bits'][0]['zero_coefficients']==1
    out=dict(candidates=[cpu.candidate(12)])
    cpu.reduce_all(raw,out,Counter())
    assert out['candidates'][0]['segments'][1]['bits'][0]['decoded']==1
    assert out['candidates'][0]['segments'][1]['bits'][0]['tie'] is True


def test_phase_specific_support_comparison_and_mixed_undefined():
    a=dict(phase=0,support_scope='same-received',truth_available=True,message=[],exact_source_evaluable=False,
        exact_correct_source_cells=None,group_correct_frame_supports=[[1,5]],missing_group_false_coverage=0,absent_source_claims=0,partial_source_claims=0)
    b=dict(a,phase=1,exact_source_evaluable=True,exact_correct_source_cells=[16],group_correct_frame_supports=[[2,6]])
    result=post.compare(a,b)
    assert result['group_support_comparison_status']=='UNRESOLVED_DIFFERENT_PHASE_SUPPORT'
    assert result['group_correct_support_retained'] is None
    assert result['exact_source_coverage_status']=='UNDEFINED_MIXED_IDENTITY'
    assert result['correct_source_added'] is None


@pytest.mark.parametrize('unavailable',[
    FileNotFoundError('transient Drive visibility'),
    OSError('transient Drive I/O'),
    '{"model":',
    '[]',
    '{"model":null}',
])
def test_notebook_progress_read_failure_keeps_same_worker(tmp_path,capsys,unavailable):
    from scripts.build_b_real_four_phase_v1_notebook import SETUP
    tree=ast.parse(SETUP)
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='report_progress')
    launch=next(n for n in tree.body if isinstance(n,ast.If))
    monitor=next(n for n in launch.body if isinstance(n,ast.Try))
    returned=next(n for n in launch.body if isinstance(n,ast.Expr)
        and isinstance(n.value,ast.Call) and n.value.args
        and isinstance(n.value.args[0],ast.Constant) and n.value.args[0].value=='Runner return code:')
    code=compile(ast.Module(body=[helper,monitor,returned],type_ignores=[]),'<actual notebook monitor>','exec')
    process=Mock(returncode=7)
    process.poll.side_effect=[None,None,7]
    saved=json.dumps({'input':{'status':'RETURNED'},'model':{'status':'STARTED'}})
    scope=dict(json=json,OUTPUT=tmp_path,process=process,time=Mock())
    with patch.object(Path,'read_text',side_effect=[unavailable,saved]) as read,\
         patch.object(subprocess,'Popen',side_effect=AssertionError('no worker relaunch')):
        exec(code,scope)
    text=capsys.readouterr().out
    assert 'Progress temporarily unavailable' in text
    assert "Completed core units: 1/2; active: ['model']" in text
    assert 'Runner return code: 7' in text
    assert read.call_count==2 and scope['time'].sleep.call_count==2
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    process.wait.assert_not_called()


def test_notebook_package_readonly_and_portable_cli(tmp_path):
    import nbformat
    root=Path(__file__).resolve().parents[1]
    nb=nbformat.read(root/'notebooks/b_real_four_phase_v1.ipynb',as_version=4)
    nbformat.validate(nb)
    codes=[c.source for c in nb.cells if c.cell_type=='code']
    assert codes[0]=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert len(codes)==4
    assert all(c.execution_count is None and not c.outputs for c in nb.cells if c.cell_type=='code')
    assert 'SOURCE_TEXT' not in '\n'.join(codes) and 'base64' not in '\n'.join(codes)
    for code in codes:ast.parse(code)
    output=tmp_path/'saved';output.mkdir();(output/'attempt.json').write_text('{}')
    bootstrap=codes[1].replace("/content/drive/MyDrive/Video-WM/B-Real-Four-Phase-V1/fixed-seen-prefix-completion-v1",str(output))
    scope={}
    with patch.object(subprocess,'run',side_effect=AssertionError('no clone/setup reentry')),patch.object(subprocess,'Popen',side_effect=AssertionError('no model process')):
        exec(bootstrap,scope)
        exec(codes[2],scope)
    source=tmp_path/'portable';source.mkdir()
    with zipfile.ZipFile(root/'notebooks/b_real_four_phase_v1_user_bundle.zip') as bundle:
        bundle.extractall(source)
    assert not (source/'.git').exists()
    result=subprocess.run([sys.executable,'-m','experiments.wan_state_clock.b_real_four_phase_v1_run','--help'],cwd=source,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    with zipfile.ZipFile(source/'experiments/wan_state_clock/data/b_real_four_phase_v1_saved.zip') as bundle:
        assert len(bundle.namelist())==25
        assert all(not p.startswith('post/') for p in bundle.namelist() if p.startswith('blind/'))
