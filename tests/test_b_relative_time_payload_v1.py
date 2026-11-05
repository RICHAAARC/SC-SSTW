"""Artificial CPU engineering fixtures only: no saved real media/model/experiment."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from main.tube_state import b_relative_time_payload_v1 as method
from main.tube_state import b_cdc32_aggregation_v1 as cdc
from runtime.wan import b_relative_time_payload_records_v1 as rec
from runtime.wan import b_relative_time_payload_worker_v1 as execution


def relative(mapping):
    return dict(conditional_nodes=[dict(received_frame=r, conditional_position=n,
        conditional_candidates=[n] if n is not None else [3,4],
        status='UNCHANGED_SINGLETON' if n is not None else 'COMPARISON_TIE',
        forbidden_truth='must not propagate') for r,n in enumerate(mapping)])


def test_relative_allowlist_inverse_and_ambiguity():
    x = relative([20,19,None]+list(range(3,181)))
    frozen = method.saved_relative(x,181)
    assert frozen['status']=='READY' and frozen['mapping'][:3]==[20,19,None]
    assert 'forbidden_truth' not in json.dumps(frozen)
    x['conditional_nodes'][0]['conditional_position']=21
    assert method.saved_relative(x,181)['status']=='SEMANTIC_UNRESOLVED'
    del x['conditional_nodes'][1]['conditional_position']
    assert method.saved_relative(x,181)['mapping']==[None]*181


def refs(plans):
    return [r for p in plans for g in p['groups'] for r in g['references']]


def test_four_phase_u_and_actual_plus_one_w():
    new = method.new_input('input0',180,relative(list(range(180))))
    u = new['strategies']['U_NEW']
    assert any(r['query']==1 and r['phase']==1 for r in refs(u))
    b = new['strategies']['B_NEW']; w = new['strategies']['W_NEW']
    assert {r['evidence_id'] for r in refs(b)} != {r['evidence_id'] for r in refs(w)}
    assert all(p['mapping_scope']=='FIXED_PLUS_ONE_TIME_SHIFT' for p in w)
    assert method.shifted(dict(status='READY',mapping=[179,180,None]))['mapping']==[180,None,None]
    assert all(0<=r['start']<=139 for r in refs(w))


def test_paired_base_real_ids_and_key_geometry():
    new=method.new_input('input0',181,relative(list(range(181))))
    pair=method.paired_input('input5',181,new,'input0')
    b=refs(new['strategies']['B_NEW']); p=refs(pair['strategies']['P_BASE'])
    assert len(b)==len(p)>0
    for x,y in zip(b,p):
        assert (x['start'],x['phase'],x['local_row'],x['canonical_cell']) == (y['start'],y['phase'],y['local_row'],y['canonical_cell'])
        assert x['evidence_id'].startswith('input0/') and y['evidence_id'].startswith('input5/')
    assert {r['evidence_id'] for r in b} == {r['evidence_id'] for r in refs(new['strategies']['K_NEW'])}
    assert new['strategies']['K_NEW'][0]['key_id']=='key01'


def test_mixed_context_preserves_four_frame_full():
    mapping=list(range(181));mapping[20]=None;mapping[30]=100
    plans=method.new_input('input0',181,relative(mapping))['strategies']['B_NEW']
    first=plans[0]['groups'][0]
    assert first['selected_canonical_cell']==1
    assert any(r['context']['has_unknown'] and r['context']['has_known_jump'] for r in first['references'])
    assert not any(r['context']['native_181_context_equivalent'] for r in first['references'])


def test_fixed_551_and_slots():
    counts=rec.counts(rec.operation_plan())
    assert {k:v['planned'] for k,v in counts.items()} == dict(plan=10,probe=10,decode_RGB=10,E41=120,project=120,bits=180,local=75,aggregate=25,load=1)
    assert len(rec.operation_plan())==551 and len(rec.layout())==100
    assert sum(i['kind']=='local' for i in rec.layout())*32==2400
    assert sum(i['kind']=='final' for i in rec.layout())*32==800
    assert sum(len(bits) for keys in rec.raw_slots().values() for bits in keys.values())==28800


def test_real_key_permutations_are_different():
    from main.tube_state import b_local_identity_dct_v1 as dct
    for region in ('R0','R1'):
        assert dct.coordinates('watermark',region) != dct.coordinates('watermark-wrong',region)
    projection=dict(kind='E41',local_rows=list(range(1,11)),include_control=False,status='PROJECTED',cause_set=[],
        coefficients={region: np.broadcast_to(np.array([1.]*8+[-1.]*8),(4,10,16)).tolist() for region in ('R0','R1')})
    a=dct.decode(projection,'watermark',view_id='input0/E41/a0',key_id='key00')
    b=dct.decode(projection,'watermark-wrong',view_id='input0/E41/a0',key_id='key01')
    assert a['regions']['R0']['local_cells'][0]['bits'][0]['coefficient_refs'] != b['regions']['R0']['local_cells'][0]['bits'][0]['coefficient_refs']
    assert a['regions']['R0']['coefficients']==b['regions']['R0']['coefficients']


def test_actual_projection_adapter_artificial_cpu_only():
    import torch
    from runtime.wan.b_relative_time_payload_backend_v1 import Backend
    from runtime.wan.b_terminal_three_group_backend_v1 import WanBackend
    backend=object.__new__(Backend);backend.wan=object.__new__(WanBackend)
    # No constructor/load/encode: only the actual DCT adapter on artificial CPU zeros.
    projection=backend.project(torch.zeros((1,16,11,40,64),dtype=torch.float32))
    restored=json.loads(json.dumps(projection,allow_nan=False))
    assert restored['coefficient_slots']==1280 and restored['local_rows']==list(range(1,11))
    for key in ('key00','key01'):
        decoded=backend.bits(restored,'input0/E41/a0',key)
        assert decoded['key']==key and decoded['returned_bits']==160
        assert all(bit['value']==0 and bit['count']==8 for region in ('R0','R1')
                   for row in decoded['regions'][region]['local_cells'] for bit in row['bits'])


def test_new_scopes_keep_conflict_and_missing_rules():
    plan=method.new_input('input0',181,relative(list(range(181))))['strategies']['B_NEW'][0]
    for group in plan['groups']:
        for ref in group['references']:ref['status']='RETURNED'
    def conflict(eid,key,bit): return dict(value=int(eid.endswith('R1')),count=8,status='AVAILABLE')
    read=cdc.read_time_local(plan,conflict)
    assert read['slots'][0]['status']=='LOCAL_CONFLICT'
    assert cdc.aggregate_time([read])['slots'][0]['status']=='CONFLICT_ERASURE'
    def missing(eid,key,bit):return dict(value=0,count=7,status='AVAILABLE')
    assert cdc.read_time_local(plan,missing)['slots'][0]['status']=='LOCAL_TECHNICAL_UNREAD'
    plan['mapping_scope']='UNKNOWN_NEW_SCOPE'
    with pytest.raises(ValueError):cdc.read_time_local(plan,missing)


class FakeBackend:
    def __init__(self,output):self.output=output;self.calls=[]
    def load(self):
        assert rec.read(self.output/'plans_frozen.json')['message_visible'] is False
        self.calls.append('load');return dict(fake=True)
    def probe(self,path,n):self.calls.append('probe');return dict(fake=True,frames=n)
    def decode(self,path,n):self.calls.append('decode_RGB');return np.zeros((n,1,1,3),dtype=np.uint8)
    def encode(self,pixels,start,n):self.calls.append('E41');return dict(fake=True)
    def project(self,latent):self.calls.append('project');return dict(fake=True)
    def bits(self,projection,uid,key):
        self.calls.append('bits')
        return dict(status='AVAILABLE',key=key,regions={region:dict(local_cells=[dict(
            evidence_id=f'{uid}/row{row}/{region}',local_row=row,bits=[dict(local_bit=b,key=key,
            value=(b+int(key=='key01'))%2,count=8,status='AVAILABLE') for b in range(8)]) for row in range(1,11)])
            for region in ('R0','R1')})
    def close(self):return dict(fake=True,peak_cuda_allocated_bytes=None)


def artificial_source(tmp_path):
    source=tmp_path/'source';output=tmp_path/'output'
    maps=dict(FULL=list(range(181)),CROP=list(range(38,139)),DELETE1=[n for n in range(181) if n!=44],
              REPEAT1=list(range(45))+list(range(44,181)),DELETE3=[n for n in range(181) if n not in (43,44,45)])
    for view,mapping in maps.items():rec.write(source/'blind/joint-relative-r01/NEW'/view/'relative.json',relative(mapping))
    rec.write(source/'private/attack_truth.json',maps)
    rec.write(source/'protocol.json',dict(writer=dict(bits=[0,1]*16)))
    config=dict(source=str(source),output=str(output),model_id='FAKE_NOT_A_MODEL')
    assert rec.reserve(output,config)
    return source,output,config


def test_full_fake_worker_551_blind_order_and_readonly(tmp_path,monkeypatch):
    source,output,config=artificial_source(tmp_path);backend=FakeBackend(output)
    original=rec.read;truth_reads=[]
    def guarded(path):
        if Path(path) in (source/'private/attack_truth.json',source/'protocol.json'):
            assert (output/'blind_terminal.json').is_file();truth_reads.append(str(path))
        return original(path)
    monkeypatch.setattr(rec,'read',guarded)
    result=execution.worker(output,lambda _:backend)
    assert result['status']=='COMPLETED',result
    assert len(truth_reads)==2 and sum(r['returned'] for r in rec.read(output/'state.json').values())==551
    assert backend.calls.count('E41')==120 and backend.calls.count('project')==120 and backend.calls.count('bits')==180
    assert len(rec.read(output/'blind_terminal.json')['artifacts'])==100
    evaluation=rec.read(output/'evaluation_summary.json')
    assert len(evaluation['rows'])==25 and evaluation['scientific_pass'] is None
    assert all(v['zero_difference'] and v['psnr_is_positive_infinity'] for v in rec.read(output/'quality.json').values())
    before=len(backend.calls)
    assert execution.worker(output,lambda _:pytest.fail('must not construct backend'))['status']=='READ_ONLY'
    from runtime.wan.b_relative_time_payload_runner_v1 import run
    assert run(config)['status']=='READ_ONLY' and len(truth_reads)==2 and len(backend.calls)==before


def test_model_failure_keeps_all_slots_and_never_posts(tmp_path):
    source,output,_=artificial_source(tmp_path)
    class Fail(FakeBackend):
        def load(self):raise RuntimeError('artificial load failure')
    result=execution.worker(output,lambda _:Fail(output),evaluator=lambda *a:pytest.fail('no post after failure'))
    assert result['status']=='STOPPED_FIXED_ATTEMPT'
    terminal=rec.read(output/'blind_terminal.json')
    assert sum(len(x['slots']) for x in terminal['artifacts'] if x['kind']=='final')==800
    assert not (output/'truth_access.json').exists()
    k=[a for a in terminal['artifacts'] if a['strategy']=='K_NEW']
    assert all(a['key_id']=='key01' and len(a['slots'])==32 for a in k)
    raw=rec.read(output/'raw_bank.json')
    assert len(raw)==120 and all(r['raw_bit_slots_per_key']==160 for r in raw.values())
    assert all(r['operation_statuses']['E41']=='NOT_ENTERED' for r in raw.values())


def test_hard_stop_receipt_closure_retains_started_unknown(tmp_path):
    source,output,_=artificial_source(tmp_path)
    state=rec.read(output/'state.json')
    state['input0/E41/a0/E41'].update(status='STARTED',started=1)
    rec.write(output/'state.json',state)
    terminal=rec.finalize_blind(output,'ARTIFICIAL_HARD_STOP')
    assert rec.read(output/'state.json')['input0/E41/a0/E41']['status']=='INTERRUPTED'
    assert len(terminal['artifacts'])==100 and not (output/'truth_access.json').exists()
    raw=rec.read(output/'raw_bank.json')
    assert raw['input0/E41/a0']['operation_statuses']['E41']=='INTERRUPTED'


def test_colab_readonly_and_safe_extraction(tmp_path,monkeypatch):
    import zipfile
    from runtime.wan import b_relative_time_payload_colab_v1 as colab
    output=tmp_path/'output';output.mkdir();(output/'attempt.json').write_text('partial reserved record')
    monkeypatch.setattr(colab,'OUTPUT',output)
    monkeypatch.setattr(colab,'show_results',lambda:None)
    monkeypatch.setattr(colab,'source_text',lambda *a:pytest.fail('no source/setup on repeat'))
    assert colab.run_colab()['status']=='READ_ONLY'
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside','bad')
    with pytest.raises(ValueError):colab.safe_extract(archive,tmp_path/'unpacked')
    assert not (tmp_path/'outside').exists()


def test_cli_editable_no_git_copy_readonly(tmp_path):
    import shutil,subprocess
    root=Path(__file__).resolve().parents[1]
    copied=tmp_path/'unrelated'/'source';(tmp_path/'unrelated'/'.git').mkdir(parents=True)
    names=['experiments/wan_state_clock/b_relative_time_payload_v1_run.py',
           'experiments/wan_state_clock/configs/b_relative_time_payload_v1.json',
           'runtime/wan/b_relative_time_payload_records_v1.py',
           'runtime/wan/b_relative_time_payload_runner_v1.py']
    for name in names:
        target=copied/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/name,target)
    source,output,_=artificial_source(tmp_path/'inputs')
    process=subprocess.run([sys.executable,str(copied/names[0]),'--output',str(output),'--config',str(tmp_path/'missing-config.json')],
        cwd=tmp_path,env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'},capture_output=True,text=True)
    assert process.returncode==0,process.stderr
    assert json.loads(process.stdout)['status']=='READ_ONLY'
    assert not (output/'supervisor_started.json').exists() and not (output/'truth_access.json').exists()


def test_geometry_and_label_error_not_hidden_by_literal():
    from runtime.wan.b_relative_time_payload_evaluation_v1 import slot_rows
    new=method.new_input('input0',181,relative(list(range(181))))
    ref=deepcopy(refs(new['strategies']['B_NEW'])[0]);ref['canonical_cell']=2
    raw={ref['view_id']:dict(E41='RETURNED',project='RETURNED',keys=dict(key00='RETURNED'),encoder_input_support=ref['encoder_input_support'])}
    slots=rec.empty_slots();slots[0].update(value=0,selected_refs=[ref],supporting_refs=[ref])
    out=slot_rows(dict(slots=slots),list(range(181)),raw,[0]*32,'key00')[0]
    assert out['literal_correct'] and out['wrong_attribution'] and out['same_group_wrong_cell'] and not out['correct_supported']


def test_quality_normalized_and_infinite_semantics():
    x=np.zeros((3,2,2,3),dtype=np.uint8);y=np.full_like(x,255)
    result=execution.quality_pair(x,y)
    assert result['mse']==1 and result['maxabs']==1 and result['psnr_db']==0
    assert execution.quality_pair(x,x)['psnr_is_positive_infinity']


def test_external_watchdog_wall_kills_group(tmp_path):
    from runtime.wan.b_relative_time_payload_runner_v1 import watch
    import psutil
    childfile=tmp_path/'child.txt'
    program="import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
    limits=dict(worker_seconds=.5,process_tree_rss_bytes=1024**3,poll_seconds=.02,terminate_grace_seconds=.1)
    result=watch([sys.executable,'-c',program,str(childfile)],tmp_path,cwd=tmp_path,limits=limits)
    assert result['failure']=='WORKER_WALL_LIMIT' and result['worker_wall_seconds']<4
    child=int(childfile.read_text())
    for _ in range(30):
        if not psutil.pid_exists(child) or psutil.Process(child).status()==psutil.STATUS_ZOMBIE:break
        time.sleep(.02)
    assert not psutil.pid_exists(child) or psutil.Process(child).status()==psutil.STATUS_ZOMBIE


def test_external_watchdog_counts_child_rss(tmp_path):
    from runtime.wan.b_relative_time_payload_runner_v1 import watch
    program="import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; x=bytearray(100*1024**2); time.sleep(30)']); time.sleep(30)"
    limits=dict(worker_seconds=5,process_tree_rss_bytes=60*1024**2,poll_seconds=.02,terminate_grace_seconds=.1)
    result=watch([sys.executable,'-c',program],tmp_path,cwd=tmp_path,limits=limits)
    assert result['failure']=='PROCESS_TREE_RSS_LIMIT' and result['peak_process_tree_rss_bytes']>60*1024**2


def test_notebook_builder_and_layering(tmp_path):
    from scripts.build_b_relative_time_payload_v1_notebook import build,ROOT
    path=build(tmp_path); notebook=json.loads(path.read_text())
    assert ''.join(notebook['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert notebook==json.loads((ROOT/'notebooks/b_relative_time_payload_v1.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
    for path in (ROOT/'runtime/wan').glob('b_relative_time_payload*.py'):
        tree=ast.parse(path.read_text())
        assert not any(isinstance(node,ast.ImportFrom) and (node.module or '').startswith('experiments') for node in ast.walk(tree))


def _saved_key_v2(output, uid, key):
    decoded=FakeBackend(output).bits({},uid,key)
    decoded.update(view_id=uid,kind='E41',local_rows=list(range(1,11)))
    return decoded


def _mark_returned_v2(rows,uid,key):
    for op in ('E41','project',key):
        rows[uid+'/'+op].update(status='RETURNED',started=1,returned=1)


@pytest.mark.parametrize('source_directory',[None,'/unreadable/local/source'])
def test_delivered_notebook_readonly_before_any_source_v2(tmp_path,monkeypatch,capsys,source_directory):
    import builtins,types,urllib.request
    root=Path(__file__).resolve().parents[1]
    notebook=json.loads((root/'notebooks/b_relative_time_payload_v1.ipynb').read_text())
    output=tmp_path/'saved';output.mkdir()
    rec.write(output/'attempt.json',dict(status='CONSUMED',consumed=True))
    rec.write(output/'result.json',dict(status='COMPLETED',marker='SAVED_RESULT_VISIBLE'))
    rec.write(output/'raw_terminal_summary.json',dict(actual_slots=28800,marker='SAVED_RAW_VISIBLE'))
    before={p.name:p.stat().st_mtime_ns for p in output.iterdir()}
    fixed='/content/drive/MyDrive/Video-WM/B-Relative-Time-Payload-CDC32-V1/fixed-r01-v1/attempt01'
    original_import=builtins.__import__; mounts=[];calls=[]
    fake_pathlib=types.ModuleType('pathlib')
    fake_pathlib.Path=lambda p: output if str(p)==fixed else Path(p)
    google=types.ModuleType('google');colab=types.ModuleType('google.colab')
    colab.drive=types.SimpleNamespace(mount=lambda p:mounts.append(p));google.colab=colab
    def forbidden(*a,**k):calls.append(a);raise AssertionError('source/network must not be touched')
    def imports(name,*a,**k):
        if name.startswith(('runtime','experiments','main.')):raise AssertionError('no scientific module import')
        return original_import(name,*a,**k)
    monkeypatch.setitem(sys.modules,'pathlib',fake_pathlib)
    monkeypatch.setitem(sys.modules,'google',google);monkeypatch.setitem(sys.modules,'google.colab',colab)
    monkeypatch.setattr(urllib.request,'urlopen',forbidden);monkeypatch.setattr(urllib.request,'urlretrieve',forbidden)
    monkeypatch.setattr(builtins,'__import__',imports)
    original_read=Path.read_text
    def read_text(path,*a,**k):
        if 'bootstrap' in path.name or path.name=='b_relative_time_payload_colab_v1.py':forbidden(str(path))
        return original_read(path,*a,**k)
    monkeypatch.setattr(Path,'read_text',read_text)
    namespace={}
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            code=''.join(cell['source']);exec(compile(code,'delivered-notebook-cell','exec'),namespace)
            if 'SOURCE_DIRECTORY = None' in code:namespace['SOURCE_DIRECTORY']=source_directory
    assert namespace['RUN']['status']=='READ_ONLY' and namespace['entry'] is None
    assert mounts==['/content/drive'] and calls==[]
    assert {p.name:p.stat().st_mtime_ns for p in output.iterdir()}==before
    shown=capsys.readouterr().out
    assert 'SAVED_RESULT_VISIBLE' in shown and 'SAVED_RAW_VISIBLE' in shown


def test_raw_partial_return_hard_stop_v2(tmp_path,monkeypatch):
    _,output,_=artificial_source(tmp_path);rows=rec.read(output/'state.json')
    uid='input0/E41/a0';_mark_returned_v2(rows,uid,'key00')
    rec.write(output/'bank'/uid/'key00.json',_saved_key_v2(output,uid,'key00'))
    # Saved key01 bytes do not confirm the interrupted operation.
    rows[uid+'/key01'].update(status='STARTED',started=1)
    rec.write(output/'bank'/uid/'key01.json',_saved_key_v2(output,uid,'key01'))
    rec.write(output/'state.json',rows)
    terminal=rec.finalize_blind(output,'ARTIFICIAL_HARD_STOP')
    raw=rec.read(output/'raw_terminal_slots.json');slots=raw['slots'][uid]
    assert raw['summary']['actual_slots']==28800
    assert raw['summary']['statuses']==dict(AVAILABLE=160,INTERRUPTED=160,NOT_ENTERED=28480)
    assert all(r['value']==r['local_bit']%2 and r['bit_status']=='AVAILABLE' and r['operation_status']=='RETURNED'
               for r in slots['key00'])
    assert all(r['value'] is None and r['status']=='INTERRUPTED' and r['unconfirmed_artifact'].endswith('key01.json')
               for r in slots['key01'])
    assert terminal['raw_terminal_summary']['confirmed_value_slots']==160
    assert rec.read(output/'raw_planned_slots.json')[uid]['key00'][0]['status']=='NOT_ENTERED'
    before=(output/'raw_terminal_slots.json').stat().st_mtime_ns
    monkeypatch.setattr(rec,'finalize_blind',lambda *a:pytest.fail('repeat must not close again'))
    monkeypatch.setattr(rec,'finalize_raw',lambda *a:pytest.fail('repeat must not close again'))
    assert rec.recover(output)['status']=='READ_ONLY'
    assert (output/'raw_terminal_slots.json').stat().st_mtime_ns==before
    assert not (output/'truth_access.json').exists()


def test_raw_returned_invalid_or_missing_receipts_v2(tmp_path):
    _,output,_=artificial_source(tmp_path);rows=rec.read(output/'state.json')
    units=rec.bank(rec.inputs()[0]);uid=units[0]['view_id'];other=units[1]['view_id']
    _mark_returned_v2(rows,uid,'key00');_mark_returned_v2(rows,uid,'key01')
    _mark_returned_v2(rows,other,'key00');_mark_returned_v2(rows,other,'key01')
    folder=output/'bank'/uid;folder.mkdir(parents=True)
    (folder/'key00.json').write_text('{broken-json')
    # key01 has a RETURNED receipt but no file.
    wrong=_saved_key_v2(output,other,'key00');wrong['key']='key01'
    rec.write(output/'bank'/other/'key00.json',wrong)
    duplicate=_saved_key_v2(output,other,'key01')
    first=duplicate['regions']['R0']['local_cells'][0]
    first['bits'].append(deepcopy(first['bits'][0]))
    rec.write(output/'bank'/other/'key01.json',duplicate)
    rec.write(output/'state.json',rows);rec.finalize_blind(output)
    raw=rec.read(output/'raw_terminal_slots.json')['slots']
    assert all(r['status']=='UNRESOLVED_RECEIPT' and r['value'] is None for key in ('key00','key01') for r in raw[uid][key])
    assert all(r['status']=='UNRESOLVED_RECEIPT' for r in raw[other]['key00'])
    bad=[r for r in raw[other]['key01'] if r['status']=='UNRESOLVED_RECEIPT']
    assert len(bad)==1 and bad[0]['local_row']==1 and bad[0]['region']=='R0' and bad[0]['local_bit']==0
    assert sum(r['status']=='AVAILABLE' for r in raw[other]['key01'])==159


def test_raw_complete_and_original_failures_v2(tmp_path):
    _,output,_=artificial_source(tmp_path);rows=rec.read(output/'state.json')
    for item in rec.inputs():
        for unit in rec.bank(item):
            uid=unit['view_id']
            for key in (('key00','key01') if item['arm']=='NEW' else ('key00',)):
                _mark_returned_v2(rows,uid,key)
                decoded=_saved_key_v2(output,uid,key)
                if uid=='input0/E41/a0' and key=='key00':
                    bit=decoded['regions']['R0']['local_cells'][0]['bits'][0]
                    bit.update(value=None,status='UNRESOLVED',returned_count=0,cause_set=['ARTIFICIAL_FAILED_BIT'])
                    decoded['status']='INCOMPLETE'
                rec.write(output/'bank'/uid/(key+'.json'),decoded)
    rec.write(output/'state.json',rows);terminal=rec.finalize_blind(output)
    raw=rec.read(output/'raw_terminal_slots.json')
    assert raw['summary']['actual_slots']==28800 and raw['summary']['operation_statuses']==dict(RETURNED=28800)
    assert raw['summary']['statuses']==dict(UNRESOLVED=1,AVAILABLE=28799)
    failed=raw['slots']['input0/E41/a0']['key00'][0]
    assert failed['status']==failed['bit_status']=='UNRESOLVED' and failed['operation_status']=='RETURNED'
    assert failed['value'] is None and failed['cause_set']==['ARTIFICIAL_FAILED_BIT']
    assert terminal['raw_terminal_summary']==raw['summary']
    assert not (output/'truth_access.json').exists()
