"""Small CPU-only dispatch, persistence and portable delivery checks."""
from __future__ import annotations
import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import types
import zipfile

import numpy as np
import pytest
import torch

from experiments.paper_results_v1 import receiver_payload_followup as f
from main.tube_state.receiver_controls_v1 import receive_scores
from main.tube_state.video_trajectory_temporal_edit_receiver_v1 import decode_visible_span


def payload(normalized, key, frames, coordinates):
    R = (frames-1)//4
    return dict(status="READ", R=R, output_frames=frames, decoded_bits=[0]*32,
        votes=[dict(ones=0, zeros=R, count=R)]*32, bit_rows=[],
        time_bit_rows=[dict(receiver_latent_index=t, output_stride_coordinate=4*t,
            estimated_source_coordinate=coordinates[4*t], bit_index=b, ones=0, zeros=1, count=1)
            for b in range(32) for t in range(1,R+1)], original_reader_match=True,
        signed_votes=-np.ones((4,R,8),np.int8), zero_mask=np.zeros((4,R,8),bool))


def fixture(tmp_path, groups=1):
    source = tmp_path / "source"; source.mkdir(parents=True)
    config = source / "effective_attack_config.json"
    f.atomic_json(config, dict(keys=dict(K0="key"), payload_bits=[0]*32, models=dict(wan=dict(local_snapshot_path="unused"))))
    f.atomic_json(source / "attack_run_state.json", dict(receiver_rows=[], artifacts=[]))
    store = f.initialize(source, tmp_path / "out", config)
    for index in range(groups):
        rows = store.data["rows"][index*3+1:index*3+3]
        op = decode_visible_span(list(range(5)))
        for row in rows:
            row.update(status="PENDING_READ", operation=copy.deepcopy(op), physical_id=f"map_{index:03d}",
                observation=dict(artifact_id=f"input{index}", shape=[5,2,2,3], path="/missing"))
        store.data["physical_inputs"].append(dict(physical_id=f"map_{index:03d}", artifact_id=f"input{index}",
            received_index_map=list(range(5)), output_frames=5, status="PLANNED", logical_slots=[r["slot_id"] for r in rows]))
    store.save()
    return store


def run_stub(store, **kwargs):
    defaults = dict(load_vae=lambda: object(), encode=lambda vae,rgb: torch.zeros(1,16,2,2,2),
                    payload_read=payload, load_rgb=lambda row: torch.full((5,2,2,3),255,dtype=torch.uint8))
    defaults.update(kwargs)
    substitute = defaults["encode"]
    def observed(vae, rgb, *, encode_observer):
        encode_observer("STARTED", None)
        try:
            value = substitute(vae, rgb)
        except BaseException as exc:
            encode_observer("FAILED", exc)
            raise
        encode_observer("RETURNED", None)
        return value
    defaults["encode"] = observed
    return f.read(store, **defaults)


def test_shared_input_normalization_and_resume(tmp_path):
    store=fixture(tmp_path); calls=[]
    def encode(vae,rgb):
        assert rgb.dtype == torch.float32 and torch.all(rgb == 1)
        calls.append(1); return torch.zeros(1,16,2,2,2)
    run_stub(store, encode=encode)
    assert len(calls)==1
    assert sum(r['status']=='EVALUATED' for r in store.data['rows'])==2
    assert store.data['errors']==[]
    run_stub(store, encode=lambda *x: pytest.fail('completed model was repeated'))
    assert len(store.data['vae_loads'])==1
    assert f.report(store)['actual_wan_calls']['attempted']==1


def test_preprocess_failure_is_not_model_attempt(tmp_path):
    store=fixture(tmp_path)
    run_stub(store, load_rgb=lambda row: (_ for _ in ()).throw(ValueError('bad RGB')))
    assert f.report(store)['actual_wan_calls']['attempted']==0
    before=store.data['rows'][1]['reason']
    f.record_external_failure(store,'outer')
    assert store.data['rows'][1]['reason']==before
    assert store.data['errors']==[]


def test_return_then_cpu_transfer_failure_counts_complete(tmp_path):
    store=fixture(tmp_path)
    class FailedCPU:
        def detach(self): return self
        def cpu(self): raise RuntimeError('CPU transfer failed')
    run_stub(store, encode=lambda *x: FailedCPU())
    assert f.report(store)['actual_wan_calls']==dict(attempted=1,statuses={'COMPLETE':1})
    run_stub(store, encode=lambda *x: pytest.fail('repeated'))
    assert store.data['rows'][1]['status']=='FAILED'


def test_interrupt_retains_attempt_and_continues_independent_map(tmp_path):
    store=fixture(tmp_path,2)
    def interrupt(*args): raise KeyboardInterrupt('stop')
    with pytest.raises(KeyboardInterrupt): run_stub(store,encode=interrupt)
    assert store.data['physical_inputs'][0]['status']=='INTERRUPTED'
    run_stub(store)
    assert store.data['physical_inputs'][1]['status']=='COMPLETE'
    assert f.report(store)['actual_wan_calls']['attempted']==2


def test_external_takeover_fans_out_durable_shared_read(tmp_path):
    store=fixture(tmp_path); group=store.data['physical_inputs'][0]
    group.update(status='COMPLETE',started_at=1)
    f._save_shared_read(store,group,payload(None,'key',5,list(range(5))))
    f.record_external_failure(store,'child killed before row checkpoint')
    assert store.data['rows'][1]['status']=='EVALUATED'
    assert store.data['rows'][2]['status']=='EVALUATED'
    run_stub(store,encode=lambda *x: pytest.fail('model repeated'))


def test_vae_load_failure_recorded_and_no_encode(tmp_path):
    store=fixture(tmp_path)
    def failure(): raise RuntimeError('load failure')
    with pytest.raises(RuntimeError): run_stub(store,load_vae=failure)
    summary=f.report(store)
    assert summary['actual_vae_loads']==dict(attempted=1,statuses={'FAILED':1})
    assert summary['actual_wan_calls']['attempted']==0
    assert len(json.loads((store.output/'evaluation_report.json').read_text())['rows'])==90


class BoundaryVAE:
    """Tiny CPU double exercising the real shared adapter, with no model weights."""
    def __init__(self, failure=None):
        self.failure=failure; self.calls=0; self.returns=0; self.clears=0
        self.parameter=torch.nn.Parameter(torch.zeros(1))
        self.config=types.SimpleNamespace(latents_mean=[1.0]*16,latents_std=[2.0]*16)
        if failure=='scaling': self.config.latents_std=[0.0]*16
    def parameters(self):
        if self.failure=='preparation': raise RuntimeError('before actual encode')
        return iter([self.parameter])
    def clear_cache(self): self.clears+=1
    def encode(self,video):
        self.calls+=1
        assert video.shape==(1,3,5,2,2) and video.dtype==torch.float32
        if self.failure=='encode': raise RuntimeError('actual encode failed')
        if self.failure=='interrupt': raise KeyboardInterrupt('actual encode interrupted')
        self.returns+=1
        def mode():
            if self.failure=='mode': raise RuntimeError('returned encode, mode failed')
            if self.failure=='shape': return torch.zeros(1,16,2,2)
            raw=torch.arange(128,dtype=torch.float32).reshape(1,16,2,2,2)
            if self.failure=='finite': raw[0,0,0,0,0]=float('nan')
            return raw
        return types.SimpleNamespace(latent_dist=types.SimpleNamespace(mode=mode))


@pytest.mark.parametrize('failure,attempted,status,returned',[
    ('preparation',0,'PREPROCESS_FAILED',0),
    ('mode',1,'COMPLETE',1), ('shape',1,'COMPLETE',1),
    ('scaling',1,'COMPLETE',1), ('finite',1,'COMPLETE',1),
    ('encode',1,'FAILED',0), ('interrupt',1,'INTERRUPTED',0),
])
def test_actual_vae_encode_boundary_and_no_repeat(tmp_path,failure,attempted,status,returned):
    store=fixture(tmp_path); vae=BoundaryVAE(failure)
    args=dict(load_vae=lambda:vae, payload_read=payload,
              load_rgb=lambda row:torch.full((5,2,2,3),255,dtype=torch.uint8))
    if failure=='interrupt':
        with pytest.raises(KeyboardInterrupt):f.read(store,**args)
    else:
        f.read(store,**args)
    assert (vae.calls,vae.returns)==(attempted,returned)
    assert store.data['physical_inputs'][0]['status']==status
    summary=f.report(store)
    assert summary['actual_wan_calls']['attempted']==attempted
    assert summary['actual_wan_calls']['statuses']==({status:1} if attempted else {})
    assert all(r['status']=='FAILED' for r in store.data['rows'][1:3])
    f.read(store,**args)
    assert (vae.calls,vae.returns)==(attempted,returned)


def test_adapter_observer_preserves_successful_default_result():
    from runtime.wan.vae import reencode_rgb24_readback
    rgb=torch.full((5,2,2,3),0.75)
    original=BoundaryVAE(); observed=BoundaryVAE(); events=[]
    default=reencode_rgb24_readback(original,rgb)
    result=reencode_rgb24_readback(observed,rgb,encode_observer=lambda event,error:events.append((event,error)))
    expected=(torch.arange(128,dtype=torch.float32).reshape(1,16,2,2,2)-1)/2
    assert torch.equal(default,expected) and torch.equal(result,default)
    assert original.calls==observed.calls==1 and original.clears==observed.clears==2
    assert events==[('STARTED',None),('RETURNED',None)]


def test_reannotation_preserves_bits_and_original(tmp_path):
    store=fixture(tmp_path); row=store.data['rows'][1]
    old=payload(None,'key',5,[None]*5); original=copy.deepcopy(old)
    result=f.reannotate_detail(old,row['operation'])
    assert result['decoded_bits']==old['decoded_bits']
    assert result['votes']==old['votes']
    assert all(r['estimated_source_coordinate']==4 for r in result['time_bit_rows'])
    assert old['time_bit_rows']==original['time_bit_rows']


def test_map_reuse_checks_all_indices_not_just_length():
    op=decode_visible_span(list(range(5)))
    assert f.same_input(dict(operation=op),op)
    wrong=copy.deepcopy(op); wrong['received_index_map'][2]=1
    assert not f.same_input(dict(operation=wrong),op)


def test_receiver_dispatch_tie_and_duration_boundary():
    tied=receive_scores(np.zeros((2,181)),np.ones((2,181)),'CENTERED_D4')
    assert tied['status']=='UNRESOLVED' and tied['operation'] is None
    long=receive_scores(np.zeros((725,181)),np.ones((725,181)),'CENTERED_D4')
    assert long['status']=='UNSUPPORTED' and long['operation'] is None


def test_runtime_uses_shared_dispatch_and_original_default(monkeypatch):
    from runtime.wan import video_trajectory_temporal_edit_receiver_v1 as runtime
    from main.tube_state import video_trajectory_temporal_edit_receiver_v1 as method
    q=np.zeros((5,181)); q[np.arange(5),np.arange(5)]=10
    monkeypatch.setattr(method,'score_framewise',lambda *a:dict(signed_projection=q,rho=np.ones_like(q)))
    class Model:
        def encode(self,received): return received
    _, legacy=runtime.encode_and_score_framewise(Model(),object(),'key')
    _, candidate=runtime.encode_and_score_framewise(Model(),object(),'key','CENTERED_D4')
    assert legacy['receiver']=='RAW_U' and legacy['estimate']['path']==list(range(5))
    assert candidate['operation']==legacy['operation']


def test_published_locator_falls_back_to_original_attack_run(tmp_path,monkeypatch):
    store=fixture(tmp_path); row=store.data['rows'][1]
    source=f.read_json(store.data['source_state']); source['recovery']={'source_run':str(tmp_path/'original')}
    f.atomic_json(store.data['source_state'],source)
    from runtime.wan import variable_rgb_media
    seen=[]
    def decode(path,target,expected_frames):
        seen.append(str(path)); return np.zeros((5,2,2,3),np.uint8),dict(path=str(target))
    monkeypatch.setattr(variable_rgb_media,'decode_published',decode)
    assert f._received_rgb(store,row).dtype==torch.uint8
    assert '/original/run_state/media/pilot_01/PAYLOAD_FRAMEWISE_M05/full/published.mp4' in seen[0]
    assert len(store.data['media_decodes'])==1


def test_missing_old_vote_is_missing_not_new_encode(tmp_path,monkeypatch):
    store=fixture(tmp_path,0)
    source=f.read_json(store.data['source_state'])
    q=np.zeros((181,181)); np.fill_diagonal(q,10)
    op=decode_visible_span(list(range(181)))
    source['receiver_rows']=[dict(case_id='pilot_01',arm=f.ARM,attack_id='full',key_label='K0',mode='BLIND_PATH',
        slot_id='old',status='EVALUATED',operation=op,detail_record={'path':'/missing/read.json'})]
    source['artifacts']=[dict(artifact_id='pilot_01/PAYLOAD_FRAMEWISE_M05/full/RECEIVED')]
    f.atomic_json(store.data['source_state'],source)
    monkeypatch.setattr(f,'_load_evidence',lambda *args:(q,np.ones_like(q),'stub'))
    f.prepare(store)
    assert all(r['status']=='MISSING' for r in store.data['rows'][:3])
    assert store.data['physical_inputs']==[]


def test_notebook_source_zip_and_no_git_cli(tmp_path):
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'notebooks/paper_results_v1_receiver_payload_followup_colab.ipynb').read_text())
    assert ''.join(nb['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    for cell in nb['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['outputs']==[] and cell['execution_count'] is None
    with zipfile.ZipFile(root/'notebooks/paper_results_v1_receiver_payload_followup_companion.zip') as z:
        for name in z.namelist(): assert z.read(name)==(root/name).read_bytes()
        z.extractall(tmp_path/'portable')
    command=[sys.executable,'-B',str(tmp_path/'portable/experiments/paper_results_v1/receiver_payload_followup_cli.py'),'--help']
    result=subprocess.run(command,cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0, result.stderr


def test_final_notebook_cell_order_repair_child_failure_and_report(tmp_path,monkeypatch):
    """Execute final cells with subprocess substitutes; never install/load/decode."""
    store=fixture(tmp_path/'fixture')
    root=Path(__file__).resolve().parents[1]
    nb=json.loads((root/'notebooks/paper_results_v1_receiver_payload_followup_colab.ipynb').read_text())
    content=tmp_path/'content'; portable=content/'paper-results-v1-receiver-followup-source'
    sentinel=portable/'experiments/paper_results_v1/receiver_payload_followup_cli.py'
    sentinel.parent.mkdir(parents=True); sentinel.write_text('# ordinary editable source')
    drive=content/'drive/MyDrive/Video-WM'
    source=drive/'Paper-Results-V1-Temporal-Attack-Recovery/20261010T063205774076Z-74da28d6'
    source.mkdir(parents=True)
    for name in ['effective_attack_config.json','attack_run_state.json']:
        (source/name).write_bytes((Path(store.data['source_state']).parent/name).read_bytes())
    cache=drive/'Paper-Results-V1-Cache/models/Wan2.1-T2V-1.3B-Diffusers/0fad780a534b6463e45facd96134c9f345acfa5b/vae'
    cache.mkdir(parents=True);(cache/'config.json').write_text('{}');(cache/'stub.safetensors').write_bytes(b'not loaded')
    commands=[]; probe_failed=False
    class Child:
        def __init__(self,command,**kwargs):
            nonlocal probe_failed
            commands.append(command);self.returncode=None;self.pid=123
            if '--phase' in command:
                phase=command[command.index('--phase')+1]
                output=Path(command[command.index('--output')+1])
                if phase=='init':
                    config=command[command.index('--config')+1]
                    f.initialize(source,output,config)
                elif phase=='prepare': f.report(f.Store(output))
                elif phase=='read': self.rc=-9;return
                elif phase=='record-failure':f.record_external_failure(f.Store(output),'child exited -9')
                elif phase=='report':f.report(f.Store(output))
            elif '-c' in command and 'AutoencoderKLWan' in command[-1] and not probe_failed:
                probe_failed=True;self.rc=1;return
            elif command[-2:]==['pip','check']:
                self.rc=7;return
            self.rc=0
        def wait(self,timeout=None):self.returncode=self.rc;return self.rc
        def poll(self):return self.returncode
    monkeypatch.setattr(subprocess,'Popen',Child)
    scope={}
    for cell in nb['cells']:
        if cell['cell_type']=='code' and cell['id']!='drive-mount':
            exec(compile(''.join(cell['source']).replace('/content',str(content)),cell['id'],'exec'),scope)
    phases=[c[c.index('--phase')+1] for c in commands if '--phase' in c]
    assert phases==['init','prepare','read','record-failure','report']
    assert any('install' in c for c in commands)
    receipts=f.read_json(scope['RECEIPTS'])
    assert any(r['stage']=='WAN_IMPORT_PROBE' and r['returncode']==1 for r in receipts)
    assert any(r['stage']=='PIP_CHECK_NONFATAL' and r['returncode']==7 for r in receipts)
    assert f.read_json(scope['OUTPUT_ROOT']/'attempts.json')[-1]['returncode']==-9
    assert f.read_json(scope['OUTPUT_ROOT']/'handoff_summary.json')['summary']['fixed_denominator']['rows']==90
