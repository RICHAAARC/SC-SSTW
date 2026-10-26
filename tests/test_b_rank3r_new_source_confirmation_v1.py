"""M3 targeted CPU/fake checks. No real model, VAE, codec or Drive work."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import ast,json,subprocess,sys
import pytest
from runtime.wan import b_rank3r_new_source_confirmation_v1 as run
from runtime.wan import b_rank3r_new_source_confirmation_evaluation_v1 as post
from main.tube_state import b_rank3r_new_source_confirmation_v1 as receiver
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
                cells.append(dict(local_row=row, evidence_id=eid, bits=[dict(local_bit=b, value=self.truth[8*self.group_for(x,row)+b],
                                  status='AVAILABLE', count=8, evidence_id=eid) for b in range(8)]))
            regions[r] = dict(local_cells=cells)
        return dict(key=key, regions=regions, control_diagnostics={})

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

def test_fixed_contract_and_three_receiver_isolation():
    cfg=run.load_config();assert cfg['sources']==run.SOURCES
    assert [h['seed'] for h in cfg['sources']]==[2026102601,2026102602,2026102603,2026102604]
    kinds=Counter(run.plan_units().values())
    assert {k:kinds[k] for k in ('native50_trajectory','D181','E41','E41_Q','H137_score','E41_DCT','native_DCT','quality_pair','blind_plan','blind_read')}==dict(native50_trajectory=8,D181=12,E41=720,E41_Q=1440,H137_score=360,E41_DCT=720,native_DCT=8,quality_pair=108,blind_plan=120,blind_read=120)
    q=run.quality_slots();assert sum(x['frame_denominator'] for x in q.values())==17148 and sum(x['adjacent_denominator'] for x in q.values())==17040
    for host in run.SOURCES:
        for fixture in run.FIXTURES:
            frames=run.source_frames(host,fixture)
            assert len(frames)==151 and min(frames)>=0 and max(frames)<=180
            assert all(frames[w:w+45]==list(range(frames[w],frames[w]+45)) for w in run.WINDOWS.values())
            if fixture.startswith('REPEAT'):
                seam=run.FIXTURES[fixture][1];assert frames[seam-8:seam]==frames[seam:seam+8]
    from main.tube_state import b_rank3r_complete_repeat_v1 as rank
    assert len(rank.PATHS)==4247 and len(old_receiver.PATHS)==961
    families={w:old_receiver.check_family(family(o,.2)) for w,o in zip(run.WINDOWS,(2,47,100))}
    views={w:[run.view_metadata('anonymous-input','query-'+w,p,start,'view-'+w+str(p),'RETURNED') for p in range(4)] for w,start in run.WINDOWS.items()}
    plan=receiver.plan_task('key00',families,views)
    assert set(plan)==set(run.READERS) and plan['RANK3R']['selection']['detected'] is True and plan['RANK3']['selection']['detected'] is False and plan['CDC_M']['selection']['detected'] is False
    assert all(p['threshold_used'] is False for p in plan['RANK3R']['locals'])
    assert all('full_local_C' in p for p in plan['CDC_M']['locals'])

def test_full_fake_physical_blind_protocol_and_optional_archive(tmp_path,monkeypatch):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    real_read=run.read;plans=set();decisions=set();bit_reads=[];archives=[]
    def guarded(path):
        path=Path(path)
        if path.name=='bits.json':assert len(plans)==120 and real_read(out/'plans_complete.json')['reader_plans']==360;bit_reads.append(path)
        if path==out/'protocol.json':assert len(decisions)==120 and real_read(out/'blind_complete.json')['reader_decisions']==360
        value=real_read(path)
        if path.name=='plan.json' and 'blind' in path.parts:plans.add(path)
        if path.name=='decision.json':decisions.add(path)
        return value
    def archive(backend,rgb,out,h):
        assert rgb['arm']=='OFF' and rgb['frames']==list(range(181));archives.append(h)
        return dict(run.archive_slots()[h],status='ARCHIVE_MISSING',error='synthetic storage failure; method continues')
    monkeypatch.setattr(run,'read',guarded);monkeypatch.setattr(run,'archive_off_rgb24',archive)
    backend=FakeBackend();result=run.run(source,out,backend=backend)
    assert result['status']=='COMPLETED_FIXED_ATTEMPT',result
    expected=Counter(generation_load=4,generation_release=4,generate=8,book=8,native_Q=8,construct=4,vae_load=4,vae_release=4,D181=12,codec_encode=72,codec_probe=72,codec_decode=72,E41=720,Q=1440,score=360,E41_DCT=720,E41_bits=1440,native_DCT=8,native_bits=16,quality=108)
    assert backend.calls==expected
    assert all(v['started']==v['returned']==v['planned'] and v['failed']==0 for v in result['physical_counts'].values())
    ev=real_read(out/'evaluation.json');assert len(ev['tasks'])==120 and ev['checks']['required_observations_complete']
    assert ev['method_status']=='M3_METHOD_CRITERIA_MET',ev['checks']
    assert ev['contribution_status']=='CONTRIBUTION_NOT_MET' and ev['net_actual_supported_gain_vs_CDC_M']==0
    assert ev['totals']['RANK3R/positive']['actual_supported_correct']==640 and ev['totals']['RANK3/positive']['actual_supported_correct']==384
    assert ev['media_P0_PM_pair']['planned']==9600 and ev['native_P0_PM_pair']['planned']==2816 and len(ev['quality'])==108
    assert len(archives)==4 and all(v['status']=='ARCHIVE_MISSING' for v in result['off_rgb24_archives'].values())
    current=real_read(out/'protocol.json');assert current['stage']=='M3_NEW_SOURCE_FIVE_FIXTURE_CONFIRMATION' and current['config']['sources']==run.SOURCES
    assert real_read(out/'source_protocol.json')['role']=='HISTORICAL_SOURCE_ONLY'
    ledger=real_read(out/'ledger.json');assert ledger['b-rank3r-m02/vae_load']['internal_counter_record']['reset'] is True
    assert ledger['b-rank3r-m02/vae_load']['observed_internal_blocks']=={}
    assert all(v>=0 for row in ledger.values() for v in row.get('observed_internal_blocks',{}).values())
    assert not list(out.rglob('*.pt')) and not Path(real_read(out/'storage.json')['tensors']).exists()
    before=backend.calls.copy();assert run.run(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT' and backend.calls==before

def test_first_failure_keeps_fixed_not_entered_and_spent(tmp_path,monkeypatch):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    monkeypatch.setattr(run,'archive_off_rgb24',lambda backend,rgb,out,h:dict(run.archive_slots()[h],status='ARCHIVE_MISSING'))
    backend=FakeBackend(fail='first_cal_E');result=run.run(source,out,backend=backend)
    assert result['status']=='TECHNICAL_UNRESOLVED' and backend.calls['E41']==1 and backend.calls['generate']==2
    ledger=run.read(out/'ledger.json');assert ledger['b-rank3r-m02/generation_load']['status']=='NOT_ENTERED'
    assert ledger['b-rank3r-m01/OFF/CROP/L/p0/E']['attempts']==1 and not (out/'post_truth_access.json').exists()
    assert len(run.read(out/'planned_outputs.json')['quality'])==108
    assert run._start_reserved(source,out,backend=object())['status']=='READ_ONLY_EXISTING_ATTEMPT'

def synthetic_tasks(cdc_missing=32,main_missing=()):
    tasks={};positive_index=0
    for task in run.tasks():
        positive=task.split('/')[1]=='PM' and task.endswith('/key00');readers={}
        for reader in run.READERS:
            raw=run.empty_slots('ERASURE')
            for i,row in enumerate(raw['slots']):
                absolute=positive_index*32+i
                emit=positive and not(reader=='CDC_M' and absolute<cdc_missing) and not(reader=='RANK3R' and absolute in main_missing) and not(reader=='RANK3' and 'REPEAT' in task)
                if emit:
                    cell=(i//8)*11+1;ref=dict(evidence_id=f'e-{i}',nominal_received_cell=list(range(4*cell-3,4*cell+1)),canonical_cell=cell)
                    row.update(value=1,status='READ',selected_refs=[ref],supporting_refs=[ref])
            value=post.assess(raw,[1]*32,positive,list(range(151)))
            value.update(detection_status='DETECT' if positive and not(reader=='RANK3' and 'REPEAT' in task) else 'REJECT_GEOMETRY',technical_detail_slots=[])
            readers[reader]=value
        tasks[task]=dict(positive=positive,readers=readers)
        if positive:positive_index+=1
    return tasks

def test_method_contribution_pair_loss_and_undefined_common_BER():
    below=post.summarize(synthetic_tasks(31),True);met=post.summarize(synthetic_tasks(32),True)
    assert below['method_status']=='M3_METHOD_CRITERIA_MET' and below['contribution_status']=='CONTRIBUTION_NOT_MET'
    assert met['method_status']=='M3_METHOD_CRITERIA_MET' and met['contribution_status']=='M3_CONTRIBUTION_CRITERIA_MET'
    assert met['paired_summary']['RANK3']['repeat']['common_emitted_slots']==0 and met['paired_summary']['RANK3']['repeat']['main_common_BER'] is None
    lost=post.summarize(synthetic_tasks(64,main_missing=(100,)),True)
    assert lost['paired_summary']['CDC_M']['overall']['net_gain']>0 and not lost['checks']['loss_vs_CDC_M_zero'] and lost['method_status']=='METHOD_NOT_SUPPORTED'
    mixed=synthetic_tasks();null=next(t for t in mixed.values() if not t['positive']);null['readers']['RANK3R'].update(detection_status='DETECT',technical_detail_slots=[0])
    value=post.summarize(mixed,True)
    assert value['method_status']=='TECHNICAL_UNRESOLVED' and 'null_detect_zero' in value['observed_method_violations']


def test_optional_history_read_and_write_failure_do_not_gate_attempt(tmp_path,monkeypatch):
    source=tmp_path/'input';out=tmp_path/'out';fixture_input(source)
    (source/'thresholds.json').write_text('{invalid history')
    actual=run.source_records
    def conflicting_records(source,out):
        # Introduce only these optional file-path conflicts after the fresh guard.
        (out/'source_protocol.json').mkdir();(out/'source_thresholds.json').mkdir()
        return actual(source,out)
    monkeypatch.setattr(run,'source_records',conflicting_records)
    class StopAtLoad(FakeBackend):
        def generation_load(self,host,config):
            self.calls['generation_load']+=1
            assert config['sources']==run.SOURCES and config['tau']['CDC_M']==run.TAU and config['tau']['RANK3R'] is None
            raise RuntimeError('controlled first backend stop')
    backend=StopAtLoad();result=run.run(source,out,backend=backend)
    assert result['stop']['message']=='controlled first backend stop'
    assert backend.calls==Counter(book=2,generation_load=1,generation_release=1)
    ledger=run.read(out/'ledger.json');assert ledger['threshold']['status']=='RETURNED'
    assert run.read(out/'thresholds.json')['tau']==run.TAU
    history=result['optional_source_records'];assert history['status']=='RECORD_ONLY' and history['affects_method'] is False
    assert set(history['errors'])=={'source_protocol.json','source_thresholds.json'}
    assert all(row['read_error'] and row['write_error'] for row in history['records'].values())
    assert run.read(out/'result.json')['optional_source_records']==history
    assert run.read(out/'protocol.json')['stage']=='M3_NEW_SOURCE_FIVE_FIXTURE_CONFIRMATION'
