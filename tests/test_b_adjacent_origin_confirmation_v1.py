"""ADJ confirmation CPU/stub checks only; no model/codec/Drive/network calls."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import ast,json
import pytest
from runtime.wan import b_adjacent_origin_confirmation_v1 as run
from runtime.wan import b_adjacent_origin_confirmation_evaluation_v1 as post
from runtime.wan import b_adjacent_origin_confirmation_storage_v1 as storage
from main.tube_state import b_adjacent_origin_confirmation_v1 as receiver
from main.tube_state import b_rank3_four_source_confirmation_v1 as old_receiver
pytestmark=pytest.mark.unit
class FakeBackend:
    """Tags replace every physical tensor; no model, VAE, codec or old data."""
    def __init__(self, fail=None):
        self.calls = Counter()
        self.truth = run.load_config()['writer']['bits']
        self.files = {}
        self.fail = fail
        self.blocks = {'encoder': 0, 'decoder': 0}

    def generation_load(self, host, config):
        self.calls['generation_load'] += 1
        self.host = deepcopy(host)
        return {'fake': True}

    def generate(self, arm, writer, count, record):
        self.calls['generate'] += 1
        for index in range(50):
            for kind in ('transformer_conditional', 'transformer_unconditional', 'native_step'):
                count(kind, False); count(kind, True)
            if arm == 'P0' and index >= 25:
                count('payload_gradient', False); count('payload_gradient', True)
            record(dict(index=index, fake=True))
        return dict(host=deepcopy(self.host), arm=arm, native=True), {'fake': True}

    def release_generation(self):
        self.calls['generation_release']+=1

    def book(self, key):
        self.calls['book'] += 1
        return key

    def native_observe(self, z, book):
        self.calls['native_Q'] += 1
        return {'status': 'FAKE', 'key': book}

    def construct(self, z, book, pre):
        self.calls['construct'] += 1
        return dict(z, arm='PM'), {'fake': True}

    def native_audit(self, *args):
        return {'fake': True}

    def load_vae(self):
        self.calls['vae_load'] += 1
        self.blocks={'encoder':0,'decoder':0}
        return {'fake': True}

    def release_vae(self):
        self.calls['vae_release']+=1

    def save_tensor(self, path, value):
        run.dump(path, value)

    def load_tensor(self, path):
        return run.read(path)

    def decode(self, z):
        self.calls['D181'] += 1
        self.blocks['decoder'] += 46
        value = dict(z, native=False, frames=list(range(181)))
        return dict(preclip=value, float_rgb=value)

    def preclip(self, x):
        return {'fake': True}

    def quantize(self, x):
        return x

    def save_mp4(self, x, path):
        self.calls['codec_encode'] += 1
        self.files[str(path)] = x
        return {'fake': True}

    def probe(self, path):
        self.calls['codec_probe'] += 1
        return {'fake': True}

    def read_media(self, path, probe, length):
        self.calls['codec_decode'] += 1
        return self.files[str(path)]

    def edit(self, x, frames):
        return dict(x, frames=frames)

    def prepare_window(self, x, w, phase):
        return dict(x, window=w, phase=phase, origin=x['frames'][w])

    def encode_view(self, x):
        self.calls['E41'] += 1
        if self.fail == 'first_cal_E' and self.calls['E41'] == 1:
            raise RuntimeError('synthetic E failure')
        self.blocks['encoder'] += 11
        return x

    def observe(self, x, book):
        self.calls['Q'] += 1
        return dict(x, key=book)

    def score(self, bank, model):
        self.calls['score'] += 1
        if any(x is None for x in bank.values()):
            return run.empty_family('TECHNICAL_UNRESOLVED')
        x = bank['0']
        p = (-x['origin']) % 4
        s = (x['origin']+p)//4+1
        positive = x['arm'] == 'PM' and x['key'] == 'watermark'
        return dict(status='SCORES_COMPLETE', candidates=[dict(phase=a, source_start=b,
            score_status='RETURNED', S=0.9 if positive and (a,b)==(p,s) else 0.05 if positive else 0.1) for a,b in run.H])

    def payload_project(self, x, rows, regions):
        self.calls['native_DCT' if len(rows)==44 else 'E41_DCT'] += 1
        return dict(x=x, rows=rows, regions=list(regions))

    def group_for(self,x,row):
        z=x['x']
        if len(x['rows'])==44:return (row-1)//11
        start=z['frames'][z['window']+z['phase']+4*row-3]
        cell=(start+3)//4
        return min(3,max(0,(cell-1)//11))

    def payload_decode(self, x, secret, view_id, key):
        self.calls['native_bits' if len(x['rows'])==44 else 'E41_bits'] += 1
        regions = {}
        for r in ('R0', 'R1'):
            cells = []
            for row in x['rows']:
                eid = f'{view_id}/row{row}/{r}'
                cells.append(dict(local_row=row, evidence_id=eid,status='AVAILABLE',
                    observed_region_id=r,carrier_region_id=r,observed_region_slice=[8,24,8,24] if r=='R0' else [8,24,40,56],
                    channels=[0,4],coordinate_space='normalized_latent',diagnostic_only=False,
                    bits=[dict(local_bit=b, value=self.truth[8*self.group_for(x,row)+b],
                                  status='AVAILABLE', count=8,planned_count=8,returned_count=8,key=key,diagnostic_only=False,evidence_id=eid) for b in range(8)]))
            regions[r] = dict(local_cells=cells)
        return dict(status='AVAILABLE',cause_set=[],kind='E41' if len(x['rows'])==10 else 'NATIVE',view_id=view_id,key=key,local_rows=x['rows'],regions=regions,control_diagnostics={})

    def quality(self, a, b, frames, pair):
        self.calls['quality'] += 1
        return dict(status='FAKE', frame_denominator=len(frames), adjacent_denominator=len(frames)-1, compared_arms=[a['arm'],b['arm']],pair=pair)

def fixture_input(path):
    model=dict(mean=[0.0]*40,factor=[[float(i==j) for j in range(40)] for i in range(40)])
    run.dump(path/'background.json',dict(status='BACKGROUND_FROZEN',models={k:model for k in run.KEYS}))
    run.dump(path/'thresholds.json',dict(status='FROZEN',tau=run.TAU,tau_rule='historical90'))

def family(origin,score=.2):
    from main.tube_state.b_m3_time_path_v1 import origin_to_state
    p,s=origin_to_state(origin)
    return dict(status='SCORES_COMPLETE',candidates=[dict(phase=a,source_start=b,score_status='RETURNED',S=score if (a,b)==(p,s) else 0.) for a,b in run.H])

def test_roster_budget_entry_and_plan_actual_phase():
    cfg=run.load_config()
    assert [s['seed'] for s in run.SOURCES]==list(range(2026102701,2026102705))
    assert [s['a'] for s in run.SOURCES]==[2,11,17,6]
    assert run.READERS==('ADJ','RANK3R','CDC_M') and len(run.tasks())==120
    units=Counter(run.plan_units().values())
    assert {k:units[k] for k in ('native50_trajectory','D181','E41','E41_Q','H137_score','E41_DCT','native_DCT','local_and_Rc_bits','native_local_bits','quality_pair','blind_plan','blind_read')}==dict(native50_trajectory=8,D181=12,E41=720,E41_Q=1440,H137_score=360,E41_DCT=720,native_DCT=8,local_and_Rc_bits=1440,native_local_bits=16,quality_pair=108,blind_plan=120,blind_read=120)
    assert len(storage.MEDIA)==72 and storage.RGB_ARCHIVES==storage.RGB_SIDECARS==()
    assert not any(k.startswith('OFF_RGB24') for k in run.DENOMINATORS)
    q=run.quality_slots()
    assert sum(v['frame_denominator'] for v in q.values())==17148
    assert sum(v['adjacent_denominator'] for v in q.values())==17040
    from main.tube_state import b_rank3r_adjacent_origin_v1 as adj
    families={}
    for window,first,second in zip(run.WINDOWS,(6,66,120),(5,67,119)):
        raw=family(first,.8)
        from main.tube_state.b_m3_time_path_v1 import origin_to_state
        target=origin_to_state(second)
        for r in raw['candidates']:
            if (r['phase'],r['source_start'])==target:r['S']=.799 if window=='M' else .77
        families[window]=old_receiver.check_family(raw)
    views={w:[run.view_metadata('input','q-'+w,p,start,w+'-'+str(p),'RETURNED') for p in range(4)] for w,start in run.WINDOWS.items()}
    plans=receiver.plan_task('key00',families,views)
    assert plans['ADJ']['selection']['origins']==[6,67,120]
    assert plans['RANK3R']['selection']['status']=='REJECT_GEOMETRY'
    m=next(p for p in plans['ADJ']['locals'] if p['window']=='M')
    assert all(ref['phase']==1 and ref['view_id']=='M-1' for g in m['groups'] for ref in g['references'])
    assert set(plans)==set(run.READERS)

def test_full_stub_once_blind_order_and_two_layer_no_gain(tmp_path,monkeypatch):
    source=tmp_path/'source';out=tmp_path/'out';fixture_input(source)
    real_read=run.read;plans=set();decisions=set()
    def guarded(path):
        path=Path(path)
        if path.name=='bits.json':assert len(plans)==120 and real_read(out/'plans_complete.json')['reader_plans']==360
        if path==out/'protocol.json':assert len(decisions)==120 and real_read(out/'blind_complete.json')['reader_decisions']==360
        value=real_read(path)
        if path.name=='plan.json' and 'blind' in path.parts:plans.add(path)
        if path.name=='decision.json':decisions.add(path)
        return value
    monkeypatch.setattr(run,'read',guarded)
    backend=FakeBackend();result=run.run(source,out,backend=backend)
    assert result['status']=='COMPLETED_FIXED_ATTEMPT',result
    assert backend.calls==Counter(generation_load=4,generation_release=4,generate=8,book=8,native_Q=8,construct=4,vae_load=4,vae_release=4,D181=12,codec_encode=72,codec_probe=72,codec_decode=72,E41=720,Q=1440,score=360,E41_DCT=720,E41_bits=1440,native_DCT=8,native_bits=16,quality=108)
    ev=real_read(out/'evaluation.json')
    assert ev['method_status']=='ADJ_SYSTEM_METHOD_CRITERIA_MET'
    assert ev['contribution_status']=='CONTRIBUTION_NOT_MET'
    assert ev['adj_incremental']['status']=='THIS_BATCH_HAS_NO_NEW_CORRECTION_EVIDENCE'
    assert ev['totals']['ADJ/positive']['actual_supported_correct']==640
    assert ev['totals']['RANK3R/positive']['actual_supported_correct']==640
    assert ev['native_P0_PM_pair']['planned']==2816 and ev['media_P0_PM_pair']['planned']==9600
    assert len(ev['quality'])==108 and not list(out.rglob('*.pt')) and not (out/'m4_inputs').exists()
    assert not Path(real_read(out/'storage.json')['tensors']).exists()
    before=backend.calls.copy()
    assert run.run(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'
    assert before==backend.calls

def test_failure_retains_budget_no_retry(tmp_path):
    source=tmp_path/'source';out=tmp_path/'out';fixture_input(source)
    backend=FakeBackend(fail='first_cal_E');result=run.run(source,out,backend=backend)
    assert result['status']=='TECHNICAL_UNRESOLVED' and backend.calls['E41']==1 and backend.calls['generate']==2
    assert run.read(out/'ledger.json')['b-adj-v02/generation_load']['status']=='NOT_ENTERED'
    assert sum(len(r['slots']) for t in run.read(out/'planned_outputs.json')['tasks'].values() for r in t['readers'].values())==11520
    assert run._start_reserved(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'


def test_gpu_cache_semantics_and_adopted_cpu_validator_same():
    import inspect
    from runtime.wan import b_rank3r_adjacent_origin_v1 as cpu
    assert inspect.getsource(run.SavedBitCache).replace('class SavedBitCache:','class Cache:')==inspect.getsource(cpu.Cache)

def test_cloud_operations_isolated_from_old_m3(tmp_path):
    assert storage.DEFAULT_OUTPUT==run.DEFAULT_OUTPUT
    assert storage.DRIVE_COMPONENTS==('Video-WM','B-Adjacent-Origin-Confirmation-V1','fixed-four-five-v1')
    class Drive:
        def __init__(self):
            self.queried=[];self.created=[];self.reads=[];self.receipt=None
            self.rows=[
                dict(id='video',name='Video-WM',mimeType=storage.FOLDER_MIME,parents=['root']),
                dict(id='old',name='B-Rank3R-New-Source-Confirmation-V1',mimeType=storage.FOLDER_MIME,parents=['video']),
                dict(id='old-results',name='fixed-four-five-v1',mimeType=storage.FOLDER_MIME,parents=['old']),
                dict(id='sentinel',name='attempt.json',mimeType='application/json',parents=['old-results']),
                dict(id='new',name='B-Adjacent-Origin-Confirmation-V1',mimeType=storage.FOLDER_MIME,parents=['video']),
                dict(id='new-results',name='fixed-four-five-v1',mimeType=storage.FOLDER_MIME,parents=['new'])]
        def quota(self):return {}
        def children(self,parents):
            self.queried.extend(parents)
            assert 'old' not in parents and 'old-results' not in parents
            return [r for r in self.rows if any(p in parents for p in r['parents'])]
        def create_start(self,parent,payload):
            assert parent=='new-results';self.created.append(parent);self.receipt=payload
            row=dict(id='new-start',name='repeat_start.json',mimeType='application/json',parents=[parent])
            self.rows.append(row);return row
        def read_to(self,entry,path):
            assert entry['id']=='new-start';self.reads.append(entry['id'])
            Path(path).write_text(json.dumps(self.receipt));return Path(path).stat().st_size
    client=Drive();output=tmp_path/'new-results';output.mkdir()
    assert storage.preflight(client,output)['safe_to_start']
    assert storage.reserve_start(client,output,attempts=1)['status']=='START_RECEIPT_CLOUD_READ_BACK'
    assert storage.preflight(client,output)['status']=='EXISTING_ATTEMPT'
    report=storage.confirm_and_backup(client,output,backup_directory=tmp_path/'backup')
    assert report['method_status_modified'] is False and report['physical_calls']==0
    assert report['cloud_confirmation']['required_media']==72 and report['cloud_confirmation']['required_core_json']==15
    assert client.created==['new-results'] and 'old' not in client.queried and 'old-results' not in client.queried
    assert all(x=='new-start' for x in client.reads)
