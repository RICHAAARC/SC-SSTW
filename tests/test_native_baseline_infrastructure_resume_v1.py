"""Artificial CPU regressions for checkpoint continuation, never real media/model work."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from runtime.wan import native_baselines_public_v4 as public
from runtime.wan import native_baselines_compact_v1 as compact
from runtime.wan.native_baseline_storage_v1 import read_json,write_json
from runtime.wan.native_baseline_storage_v4 import (Budget,UnitStore,ViewEpochs,Unavailable,
    RecoveryExhausted,append_json,events,runtime_session,retired_recovery_gate,recovery_pending)

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.unit


class FakeModel:
    def __init__(self,error=None):self.loads=0;self.closed=0;self.error=error
    def load(self):
        self.loads+=1
        if self.error:raise self.error
        return {'synthetic':True}
    def close(self):self.closed+=1


def test_three_historical_loads_next_pack_and_same_runtime_reentry(tmp_path):
    path=tmp_path/'model_loads.jsonl'
    for key in ('TEST-10/wam','TEST-10/pixelseal','TEST-10/videoseal','TEST-14/wam'):
        for status in ('STARTED','RETURNED'):
            append_json(path,dict(key=key,status=status,category='model_load',runtime_id='old'))
    old=path.read_bytes()
    budget=Budget(tmp_path,'new',limits={'base':{'model_load':0},'recovery':{'model_load':0}})
    loaders=public.Loaders(SimpleNamespace(output=tmp_path,budget=budget),None,None,None)
    keys=['TEST-10/wam','TEST-10/pixelseal','TEST-10/videoseal','TEST-14/wam','TEST-10/wam']
    for key in keys:
        model=loaders.perform(key,'model_load',FakeModel());model.close()
        assert model.loads==1
    assert budget.extra['model_load']==5 and loaders.extra_watermark==5
    assert path.read_bytes().startswith(old)
    assert len([r for r in events(path) if r['status']=='RETURNED'])==9


def test_real_load_failures_recorded_without_hidden_retry(tmp_path):
    ops=SimpleNamespace(output=tmp_path,budget=Budget(tmp_path,'one'))
    loaders=public.Loaders(ops,None,None,None);model=FakeModel(TimeoutError('network unavailable'))
    with pytest.raises(Unavailable):loaders.perform('P/wam','model_load',model)
    assert model.loads==1 and model.closed==1 and loaders.rows[-1]['status']=='INFRA_FAILED'
    with pytest.raises(Unavailable):loaders.perform('P/wam','model_load',model)
    assert model.loads==1
    again=public.Loaders(SimpleNamespace(output=tmp_path,budget=Budget(tmp_path,'two')),None,None,None)
    assert again.perform('P/wam','model_load',FakeModel()).loads==1
    bad=FakeModel(ValueError('unsupported actual model shape'))
    with pytest.raises(Unavailable):again.perform('P/pixelseal','model_load',bad)
    third=public.Loaders(SimpleNamespace(output=tmp_path,budget=Budget(tmp_path,'three')),None,None,None)
    with pytest.raises(Unavailable):third.perform('P/pixelseal','model_load',FakeModel())
    assert third.rows[-1]['status']=='FAILED_PERMANENT'


def test_old_recovery_counters_and_runtime_allowance_are_records(tmp_path):
    limits={'base':{'detect_wam':1},'recovery':{'detect_wam':0}}
    b=Budget(tmp_path,'one',limits);b.charge('detect_wam','u','base')
    for i in range(5):b.charge('detect_wam','u',str(i),recovery=True)
    saved=b.resume_state();again=Budget(tmp_path,'two',limits,saved=saved)
    again.charge('detect_wam','u','sixth',recovery=True)
    assert again.base['detect_wam']==1 and again.extra['detect_wam']==6
    with pytest.raises(RecoveryExhausted):again.charge('detect_wam','u','new scientific task')
    for i in range(70):append_json(tmp_path/'runtimes.jsonl',dict(id=str(i)))
    with runtime_session(tmp_path,runtime_limit=0) as record:assert record['ordinal']==70
    assert read_json(tmp_path/'runtime_owner.json')['status']=='CLOSED'


def test_retired_unit_gate_resumes_and_committed_response_is_not_replaced(tmp_path):
    b=Budget(tmp_path,'old');store=UnitStore(tmp_path,'u',b)
    for _ in range(4):store.event('batch','STARTED',physical_key='batch')
    store.event('batch','BUDGET_EXHAUSTED',reason='two extra infrastructure calls exhausted')
    before=store.path.read_bytes()
    restored=UnitStore(tmp_path,'u',Budget(tmp_path,'new'))
    calls=[]
    result=restored.perform('batch','detect_wam',lambda:(calls.append(1),{'x':np.array([2])})[1],arrays=True)
    assert result['x'].tolist()==[2] and restored.attempts['batch']==5
    assert restored.path.read_bytes().startswith(before)
    complete=UnitStore(tmp_path,'u',Budget(tmp_path,'later'))
    assert complete.perform('batch','detect_wam',lambda:pytest.fail('completed response replayed'),arrays=True)['x'].tolist()==[2]
    assert not recovery_pending(complete.row('batch')) and calls==[1]
    complete.event('bad','FAILED_PERMANENT',error='ValueError: invalid native shape')
    with pytest.raises(Unavailable):complete.perform('bad','detect_wam',lambda:pytest.fail('real failure retried'))
    complete.event('wrapped','FAILED_PERMANENT',error='Unavailable: previous: BUDGET_EXHAUSTED')
    assert complete.perform('wrapped','detect_wam',lambda:'recovered')=='recovered'


def test_lost_epoch_recovers_after_legacy_quota_without_selecting_old_scores(tmp_path):
    store=UnitStore(tmp_path,'u',Budget(tmp_path,'new'))
    epochs=ViewEpochs(store,'G1/T00',tmp_path/'cache')
    epochs.rows=[dict(id=f'epoch{i:02}',input=str(tmp_path/f'missing{i}.npy'),status='BUDGET_EXHAUSTED') for i in range(1,5)]
    write_json(epochs.path,epochs.rows)
    store.event('G1/T00/epoch04/batch0','INFRA_FAILED',physical_key='G1/T00/batch0')
    selected,fresh=epochs.choose([(0,8)],'detect_wam')
    assert fresh and selected['id']=='epoch05' and selected['whole_view_recovery']
    assert len(read_json(epochs.path))==5
    assert list(events(epochs.path.with_name('recovery.jsonl')))[0]['previous_epoch']['status']=='BUDGET_EXHAUSTED'


def test_only_quota_stopped_checkpoint_slots_rejoin_once(tmp_path):
    tasks=[dict(id='native/P/wam/S/OFF',kind='native',source='S',backend='wam',arm='OFF'),
           dict(id='quality/P/wam/S/wam',kind='quality',source='S',backend='wam',arm='wam'),
           dict(id='quality/P/wam/U/wam',kind='quality',source='U',backend='wam',arm='wam')]
    state=dict(status='DONE',next_task=3,pending={},retained_unknown_views={'0':{'2':{'view':'G2/T01'},'3':{'view':'G2/T02'}}},
               retained_unknown_tasks={'1':tasks[1],'2':tasks[2]})
    append_json(tmp_path/'failures.jsonl',dict(where='read/S/G2/T01',error='RecoveryExhausted: three additional watermark loads used in this runtime'))
    append_json(tmp_path/'failures.jsonl',dict(where='native/S/wam/OFF/G2/T02/8',error='RecoveryExhausted: detect_wam recovery budget exhausted'))
    append_json(tmp_path/'failures.jsonl',dict(where='quality/S/wam/precodec',error='RecoveryExhausted: model_load recovery budget exhausted'))
    append_json(tmp_path/'failures.jsonl',dict(where=tasks[2]['id'],error='ValueError: actual invalid input'))
    before=(tmp_path/'failures.jsonl').read_bytes()
    cursor=SimpleNamespace(state=state,tasks=tasks,save=lambda:write_json(tmp_path/'compute_checkpoint.json',state))
    assert compact.restore_recovery_pending(tmp_path,cursor)==['view:0:2','view:0:3','task:1']
    assert state['pending']=={'0':[2,3],'1':[0]} and state['next_task']==3
    assert compact.restore_recovery_pending(tmp_path,cursor)==[]
    assert (tmp_path/'failures.jsonl').read_bytes()==before
    assert len(state['retained_unknown_tasks'])==2 and len(state['retained_unknown_views']['0'])==2


def test_complete_cpu_quality_skips_models_and_keeps_missing_lpips(tmp_path):
    folder=tmp_path/'units/quality/S/wam/quality'
    for stage in ('precodec','G1','G2-identity'):
        write_json(folder/(stage+'.json'),dict(frames=[dict(mse=0.,ssim=1.,psnr_db=None)]*181,
            edges=[dict(residual_difference_rms=0.)]*180,lpips_observed_frames=0))
    before={f.name:f.read_bytes() for f in folder.iterdir()}
    run=object.__new__(public.Runner);run.ops=SimpleNamespace(output=tmp_path)
    assert public.quality_complete(tmp_path,dict(id='S'),'wam')
    run.quality(dict(id='S'),'wam',None)  # No ops.source/loader/measure attributes: any replay fails.
    assert before=={f.name:f.read_bytes() for f in folder.iterdir()}
    assert not public.quality_stage_complete(dict(frames=[dict(mse='bad',ssim=1.)]*181,
        edges=[dict(residual_difference_rms=0.)]*180))


def test_loader_base_quota_is_retired_but_scientific_base_is_not(tmp_path):
    assert retired_recovery_gate(dict(status='FAILED_PERMANENT',error='RecoveryExhausted: model_load base budget exhausted'))
    assert retired_recovery_gate(dict(status='BUDGET_EXHAUSTED',error='lpips_load base budget exhausted'))
    assert not retired_recovery_gate(dict(status='BUDGET_EXHAUSTED',error='detect_wam base budget exhausted'))
    path=tmp_path/'model_loads.jsonl'
    append_json(path,dict(key='P/wam',status='FAILED_PERMANENT',error='RecoveryExhausted: model_load base budget exhausted',runtime_id='old'))
    loader=public.Loaders(SimpleNamespace(output=tmp_path,budget=Budget(tmp_path,'new')),None,None,None)
    assert loader.perform('P/wam','model_load',FakeModel()).loads==1


def test_old_config_quota_fields_are_optional_but_method_settings_are_fixed():
    from main.baselines import public_compact12_v1 as cp,public_fpr001_v4 as vp
    for module,filename in [(cp,'public_compact12_v1.json'),(vp,'public_native_fpr001_v4.json')]:
        value=read_json(ROOT/'main/baselines/protocols'/filename)
        assert module.validate_config(value)==value
        for key in ('recovery_per_unit','recovery_fraction','extra_runtimes','extra_watermark_loads_per_runtime','extra_lpips_loads_per_runtime'):
            value.pop(key,None)
        module.validate_config(value)
        value['fps']=24
        with pytest.raises(ValueError):module.validate_config(value)


def test_shard_inherits_costs_above_old_infrastructure_allowance():
    from main.tube_state import paper_execution_v1 as schedule
    from main.baselines.public_compact12_v1 import selected_sources
    roster=read_json(ROOT/'main/baselines/protocols/public_native_fpr001_v4.sources.json')['sources']
    spent=[dict(base={},recovery={}) for _ in range(4)];spent[0]['recovery']['model_load']=300
    limits=schedule.shard_budget(selected_sources(roster),0,spent,parent_runtimes=100)
    assert limits['inherited']['recovery']['model_load']==300 and limits['runtime_limit']==0
    assert not limits['infrastructure_limits_enforced']
    assert not retired_recovery_gate(dict(status='FAILED_PERMANENT',error='ValueError: model input invalid'))
