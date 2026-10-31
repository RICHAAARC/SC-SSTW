"""Artificial received maps/caches only. Never reads c01 assets or invokes codecs/models."""
from copy import deepcopy
from pathlib import Path
import importlib.util
import json
import subprocess
import sys
import pytest
from main.tube_state import b_time_payload_cdc32_v1 as method
from main.tube_state import b_cdc32_aggregation_v1 as cdc
from experiments.wan_state_clock import b_time_payload_cdc32_v1_records as rec
from experiments.wan_state_clock.b_time_payload_cdc32_v1_experiment import run

pytestmark=pytest.mark.quick
ROOT=Path(__file__).resolve().parents[1]
STEM='b_time_payload_cdc32_v1'


def submission(mapping):
    return dict(status='CONDITIONAL_OUTPUTS_SAVED',final_nodes=[dict(received_frame=i,output_index=n,
        origin='OLD' if n is not None else 'UNRESOLVED') for i,n in enumerate(mapping)])


def ready(plan):
    receipts={r['view_id']:dict(status='RETURNED') for g in plan['groups'] for r in g['references']}
    return method.attach_receipts(plan,receipts)


def bit(value=1):return dict(status='AVAILABLE',value=value,count=8)


def test_fixed_bank_budget_and_no_legacy_qualification_fields():
    assert [list(dict.fromkeys(u['query_start'] for u in method.bank('x',n))) for n in rec.LENGTHS]==[[0,68,136],[0,28,56],[0,67,135],[0,68,137],[0,66,133]]
    assert len(rec.plan())==sum(rec.budgets().values())==256
    assert rec.budgets()['E41']==60 and rec.denominators()['final_slots']==480
    plans=method.blind_plans('x',181,submission(list(range(181))))
    read=cdc.read_time_local(ready(plans['B'][0]),lambda *a:bit())
    result=cdc.aggregate_time([read])
    def scan(value):
        if isinstance(value,dict):
            assert not set(value)&{'D','S','tau','members','full_local_C','detection_status','support_under_h'}
            assert value.get('mapping_scope')!='LOCAL_FULL_C'
            for v in value.values():scan(v)
        elif isinstance(value,list):
            for v in value:scan(v)
    scan(result);assert result['existence_test']=='NOT_EVALUATED'


@pytest.mark.parametrize('mutation', ['status','short','bool','missing_index','coordinate','inversion'])
def test_invalid_or_failed_saved_time_preserves_B_W_failure_U_independence(mutation):
    s=submission(list(range(181)))
    if mutation=='status':s['status']='OLD_GRAPH_UNSUPPORTED'
    if mutation=='short':s['final_nodes'].pop()
    if mutation=='bool':s['final_nodes'][4]['output_index']=True
    if mutation=='missing_index':del s['final_nodes'][4]['output_index']
    if mutation=='coordinate':s['final_nodes'][4]['received_frame']=5
    if mutation=='inversion':s['final_nodes'][4]['output_index']=10
    p=method.blind_plans('x',181,s)
    assert all(not g['references'] for plan in p['B']+p['W'] for g in plan['groups'])
    assert p['U']==method.unaligned_plans('x',181)
    assert any(g['references'] for plan in p['U'] for g in plan['groups'])


def test_cell_geometry_and_context_unknown_jumps_are_separate():
    mapping=list(range(181));mapping[:4]=[None]*4;mapping[24]=23
    p=method.blind_plans('x',181,submission(mapping))['B'][0]
    refs=p['groups'][0]['references']
    assert refs and p['groups'][0]['selected_canonical_cell']==2
    assert any(r['context']['has_unknown'] and r['context']['has_known_jump'] for r in refs)
    assert all(r['qualification']=='CELL_GEOMETRY_FULL' for r in refs)
    assert all(r['context']['native_181_context_equivalent'] is False for r in refs)
    assert method.geometry([1,2,2,4])[1]=='NON_FULL_OR_MIXED_CELL'
    assert method.geometry([1,2,4,5])[1]=='NON_FULL_OR_MIXED_CELL'
    assert method.geometry([177,178,179,180])[1]=='GUARD_OR_OUTSIDE_WRITER_CELLS'
    assert method.geometry([None,2,3,4])[1]=='UNKNOWN_TIME_OUTPUT'


def test_W_complete_permutation_keeps_actual_refs_failures_and_no_reselection():
    b=method.blind_plans('x',181,submission(list(range(181))))['B']
    b[0]['groups'][0]['status']='UNRESOLVED_MAPPING_CONFLICT'
    b[0]['groups'][0]['cause_set']=['artificial']
    b[0]['groups'][0]['references'][0]['status']='FAILED'
    w=method.permuted(b)
    for left,right in zip(b,w):
        for g in left['groups']:
            shifted=right['groups'][(g['group']+1)%4]
            assert shifted['status']==g['status'] and shifted['cause_set']==g['cause_set']
            assert len(shifted['references'])==len(g['references'])
            for a,z in zip(g['references'],shifted['references']):
                for k in ('evidence_id','view_id','local_row','observed_region_id','encoder_input_support','nominal_received_cell','status'):
                    assert z[k]==a[k]
                assert z['canonical_cell']==1+(a['canonical_cell']-1+11)%44
    assert w[2]['groups'][0]['selected_canonical_cell']==2  # B group3 at t35 wraps to t2, no new choice.


def test_U_phase_zero_misaligned_query_is_geometry_erasure_not_floor():
    plans=method.unaligned_plans('x',180)
    assert any(g['references'] for g in plans[0]['groups'])
    assert all(not g['references'] for p in plans[1:] for g in p['groups'])
    assert all(r['phase']==0 for p in plans for r in p['geometry_rows'])


def test_alias_conflicts_technical_missing_and_arm_isolation_use_original_kernel():
    plans=method.blind_plans('x',181,submission(list(range(181))))
    p=ready(plans['B'][0]);alias=deepcopy(p);alias['groups'][0]['references']*=3
    calls=[]
    def cache(e,k,b):calls.append((e,k,b));return bit()
    a=cdc.read_time_local(alias,cache);assert len(calls)==len(set(calls))
    conflict=cdc.read_time_local(p,lambda e,*a:bit(int(e.endswith('R1'))))
    assert all(s['value'] is None and s['status']=='LOCAL_CONFLICT' for s in conflict['slots'][:8])
    def missing(e,*a):
        if e.endswith('R1'):raise OSError('artificial missing region')
        return bit()
    lost=cdc.read_time_local(p,missing)
    assert all(s['status']=='LOCAL_TECHNICAL_UNREAD' for s in lost['slots'][:8])
    result=cdc.aggregate_time([a]);assert result['dependency_refs']
    wrong=cdc.read_time_local(ready(plans['W'][0]),lambda *a:bit())
    with pytest.raises(ValueError,match='one time arm'):cdc.aggregate_time([a,wrong])


def fixture(tmp):
    cfg=rec.load(ROOT/'experiments/wan_state_clock/configs'/f'{STEM}.json')
    cfg.update(source=str(tmp/'source'),output=str(tmp/'out'))
    for view,length in zip(rec.VIEWS,rec.LENGTHS):
        folder=Path(cfg['source'])/'receiver/received'/view
        rec.dump(folder/'submission.json',submission(list(range(length))))
        (folder/'received.mp4').write_bytes(b'artificial marker only')
    rec.dump(Path(cfg['source'])/'receiver/private/attack_truth.json',dict(views={v:[min(n,180) for n in range(length)] for v,length in zip(rec.VIEWS,rec.LENGTHS)}))
    return cfg


class Fake:
    mode='ok';output=None;calls=[]
    def __init__(self,model):self.calls=[]
    def check(self):pass
    def load(self):
        assert (self.output/'plans.json').is_file()
        plans=rec.load(self.output/'plans.json');assert len(plans)==5 and all(set(p)=={'B','W','U'} for p in plans.values())
        return dict(status='ARTIFICIAL_MODEL')
    def probe(self,path,length):return dict(frames=length,artificial=True)
    def decode(self,path,length):return length
    def encode(self,pixels,start,length):
        assert (self.output/'plans.json').exists()
        if self.mode=='interrupt':raise KeyboardInterrupt('artificial E41 interrupt')
        if self.mode=='failure':raise ValueError('artificial E41 failure')
        return start
    def project(self,latent):return latent
    def bits(self,projection,uid):
        return dict(key='key00',regions={region:dict(local_cells=[dict(evidence_id=cdc.evidence_id(uid,row,region),
            bits=[dict(local_bit=b,**bit()) for b in range(8)]) for row in range(1,11)]) for region in cdc.REGIONS})
    def close(self):return dict(status='ARTIFICIAL_CLOSED')


@pytest.mark.parametrize('mode',['ok','failure','interrupt'])
def test_real_runner_artificial_256_fixed_slots_plan_first_and_no_rerun(tmp_path,mode):
    cfg=fixture(tmp_path)
    class Backend(Fake):pass
    Backend.output=Path(cfg['output']);Backend.mode=mode
    observed=[]
    def evaluate(output,truth):
        blind=rec.load(output/'blind_terminal.json')
        assert sum(len(v[a]['slots']) for v in blind['final'].values() for a in rec.ARMS)==480
        assert sum(len(p['slots']) for v in blind['local'].values() for a in rec.ARMS for p in v[a])==1440
        assert len(blind['raw_bank'])==60
        assert all(r['status'] not in ('STARTED','NOT_STARTED') for r in blind['rows'].values())
        observed.append(True);return dict(status='ARTIFICIAL_POST_ONLY')
    result=run(cfg,backend_factory=Backend,evaluator=evaluate)
    assert observed and result['evaluation_status']==('ARTIFICIAL_POST_ONLY' if mode=='ok' else 'EVIDENCE_INCOMPLETE')
    counts=rec.load(Backend.output/'counts.json');assert sum(v['planned'] for v in counts.values())==256
    if mode=='ok':
        assert sum(v['started'] for v in counts.values())==256
        assert sum(v['returned'] for v in counts.values())==256
        assert counts['E41']['started']==60
    elif mode=='interrupt':
        assert result['status']=='INTERRUPTED' and counts['E41']['started']==1 and counts['E41']['returned']==0
    else:assert counts['E41']['started']==60 and counts['bits']['started']==0
    before={str(p):p.stat().st_mtime_ns for p in Backend.output.rglob('*') if p.is_file()}
    again=run(dict(output=cfg['output']),backend_factory=lambda *a:pytest.fail('rerun'))
    assert again['physical_calls_this_invocation']==0
    assert before=={str(p):p.stat().st_mtime_ns for p in Backend.output.rglob('*') if p.is_file()}


def test_post_equal_bit_wrong_cell_and_mixed_support_cannot_be_washed():
    from experiments.wan_state_clock.b_time_payload_cdc32_v1_evaluation import _reference,compare
    refs=method.blind_plans('x',181,submission(list(range(181))))['B'][0]['groups'][0]['references']
    a=deepcopy(refs[0]);raw={a['view_id']:dict(status='RETURNED',encoder_input_support=a['encoder_input_support'])}
    correct=_reference(a,list(range(181)),raw,0);assert correct['correct_geometry_support']
    a['canonical_cell']+=1
    wrong=_reference(a,list(range(181)),raw,0);assert not wrong['four_frame_exact'] and wrong['group_exact']
    assert not all(r['correct_geometry_support'] for r in [correct,wrong])
    same=[dict(literal_correct=True,correct_supported=True) for _ in range(32)]
    rotated=[dict(literal_correct=True,correct_supported=False) for _ in range(32)]
    out=compare(same,rotated)
    assert out['B_common_BER']==out['control_common_BER']==0 and not out['wrong_to_correct']
    assert len(out['added_correct_support'])==32


def test_returned_resource_overrun_is_distinct_from_call_interruption(tmp_path,monkeypatch):
    journal=rec.Journal(tmp_path);clock=[0.]
    monkeypatch.setattr(rec.time,'monotonic',lambda:clock[0])
    def slow():clock[0]=901.;return {}
    with pytest.raises(rec.ResourceExceeded):journal.call('model_load',slow)
    row=journal.rows['model_load']
    assert row['started']==row['returned']==1 and row['status']=='RESOURCE_UNRESOLVED'
    assert row['limit_observation']=='AFTER_CALL_RETURN'


def test_consumed_standalone_cli_before_missing_config_and_scientific_imports(tmp_path):
    rec.dump(tmp_path/'attempt.json',{})
    code="""import runpy,sys
class Deny:
 def find_spec(self,name,*args):
  if name.split('.')[0] in ('numpy','torch','psutil','runtime','main','scripts'):raise AssertionError(name)
sys.meta_path.insert(0,Deny())
runpy.run_path(sys.argv.pop(1),run_name='__main__')
"""
    p=subprocess.run([sys.executable,'-B','-c',code,str(ROOT/'experiments/wan_state_clock'/f'{STEM}_run.py'),'--output',str(tmp_path),'--config','/missing'],cwd=tmp_path,capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['physical_calls_this_invocation']==0


def test_notebook_builder_sync_first_mount_and_readonly_before_download(tmp_path,monkeypatch):
    import nbformat,urllib.request
    spec=importlib.util.spec_from_file_location('builder',ROOT/'scripts'/f'build_{STEM}_notebook.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    built=nbformat.read(m.build(tmp_path)['notebook'],as_version=4);nbformat.validate(built)
    assert built==nbformat.read(ROOT/'notebooks'/f'{STEM}.ipynb',as_version=4)
    assert built.cells[0].source=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    out=tmp_path/'out';rec.dump(out/'attempt.json',{})
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('readonly download'))
    scope=dict(OUTPUT=str(out));exec(m.RUN,scope)
    assert scope['result']['physical_calls_this_invocation']==0


def test_failed_call_elapsed_is_consumed_in_resource_totals(tmp_path,monkeypatch):
    clock=[0.];monkeypatch.setattr(rec.time,'monotonic',lambda:clock[0])
    journal=rec.Journal(tmp_path)
    def slow_failure():clock[0]=601.;raise OSError('artificial late failure')
    with pytest.raises(rec.ResourceExceeded):journal.call('FULL/q0/p0/E41',slow_failure)
    assert journal.encode_seconds==601 and journal.rows['FULL/q0/p0/E41']['returned']==0
    assert journal.rows['FULL/q0/p0/E41']['limit_observation']=='AFTER_CALL_FAILURE'


def test_actual_adapter_slices_received_order_and_original_divide_once(monkeypatch):
    import numpy as np,torch
    from runtime.wan.b_time_payload_cdc32_v1 import Backend
    from runtime.wan.b_terminal_three_group_backend_v1 import WanBackend
    from runtime.wan import vae
    adapter=object.__new__(Backend);adapter.wan=WanBackend('artificial model not loaded');adapter.wan.vae=object()
    seen=[]
    def encode(_vae,rgb):
        seen.append((tuple(rgb.shape),rgb.dtype,rgb[:,0,0,0].tolist()))
        return torch.zeros((1,16,11,40,64),dtype=torch.float32)
    monkeypatch.setattr(vae,'reencode_rgb24_readback',encode)
    monkeypatch.setattr(adapter.wan,'prepare_window',lambda *a:pytest.fail('old fixed151 wrapper'))
    pixels=np.broadcast_to(np.arange(101,dtype=np.uint8)[:,None,None,None],(101,320,512,3))
    z=adapter.encode(pixels,29,101)
    assert z.dtype==torch.float32 and tuple(z.shape)==(1,16,11,40,64)
    assert seen[0][:2]==((41,320,512,3),torch.float32)
    assert seen[0][2]==pytest.approx([n/255 for n in range(29,70)])


def test_full_post_evaluator_rejects_wrong_cell_even_with_right_literal(tmp_path):
    from experiments.wan_state_clock.b_time_payload_cdc32_v1_evaluation import evaluate
    cfg=fixture(tmp_path)
    plans={v:method.blind_plans(f'input{i}',n,submission(list(range(n)))) for i,(v,n) in enumerate(zip(rec.VIEWS,rec.LENGTHS))}
    final={};raw={}
    message=[int(b) for byte in bytes.fromhex('A6D39C5E') for b in f'{byte:08b}']
    for v in rec.VIEWS:
        final[v]={}
        for arm in rec.ARMS:
            reads=[cdc.read_time_local(ready(p),lambda e,k,b:bit(message[b])) for p in plans[v][arm]]
            final[v][arm]=cdc.aggregate_time(reads)
            for p in plans[v][arm]:
                for g in p['groups']:
                    for ref in g['references']:raw[ref['view_id']]=dict(status='RETURNED',encoder_input_support=ref['encoder_input_support'])
    s=final['FULL']['B']['slots'][0];assert s['value']==message[0] and s['supporting_refs']
    s['supporting_refs'][0]['canonical_cell']+=1
    terminal=dict(status='ALL_BLIND_SLOTS_TERMINAL',plans=plans,final=final,raw_bank=raw,rows={k:dict(r,status='RETURNED') for k,r in rec.plan().items()},errors={})
    out=Path(cfg['output']);rec.dump(out/'blind_terminal.json',terminal)
    result=evaluate(out,Path(cfg['source'])/'receiver/private/attack_truth.json')
    slot=result['views']['FULL']['arms']['B']['rows'][0]
    assert slot['literal_correct'] and slot['wrong_attribution'] and not slot['correct_supported']
    assert result['status']=='COUNTEREXAMPLE'
    # Missing/upstream-invalid time evidence cannot be promoted by a control result.
    for v in rec.VIEWS:
        for p in terminal['plans'][v]['B']:p['status']='SEMANTIC_UNRESOLVED'
        for arm in rec.ARMS:
            for s in terminal['final'][v][arm]['slots']:s.update(value=None,supporting_refs=[])
    rec.dump(out/'blind_terminal.json',terminal)
    assert evaluate(out,Path(cfg['source'])/'receiver/private/attack_truth.json')['status']=='EVIDENCE_INCOMPLETE'


def test_notebook_remote_bootstrap_and_entry_default_offline(tmp_path,monkeypatch):
    import shutil,urllib.request
    from types import SimpleNamespace
    def module(relative):
        spec=importlib.util.spec_from_file_location('artificial_'+Path(relative).stem,ROOT/relative)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    builder=module('scripts/build_'+STEM+'_notebook.py');seen=[]
    def retrieve(url,path):
        seen.append(url)
        if str(path).endswith(STEM+'_entry.py'):
            Path(path).write_text("def launch(*args):\n return {'status':'ARTIFICIAL_BOOTSTRAP'}\n")
        else:shutil.copyfile(ROOT/url.split('/artificial-ref/')[1],path)
    monkeypatch.setattr(urllib.request,'urlretrieve',retrieve)
    scope=dict(SOURCE='/artificial/c01',OUTPUT=str(tmp_path/'boot-out'),SOURCE_DIRECTORY=None,SOURCE_REF='artificial-ref',MODEL_ID=None)
    exec(builder.RUN,scope)
    assert len(seen)==6 and scope['result']['status']=='ARTIFICIAL_BOOTSTRAP'
    entry=module('scripts/'+STEM+'_entry.py');common=entry.cpu._common();cfg=fixture(tmp_path)
    monkeypatch.setattr(entry.cpu,'_common',lambda:common)
    monkeypatch.setattr(common,'load_file',lambda *a:SimpleNamespace(setup_dependencies=lambda *a:seen.append('setup')))
    def worker(session,root):
        assert seen[-1]=='setup'
        config=rec.load(session.config_path);assert config['source']==cfg['source']
        assert session.stem==STEM and 'materialization' not in str(session._command(root))
        rec.dump(session.output/'attempt.json',{});rec.dump(session.output/'result.json',dict(status='ARTIFICIAL_WORKER'))
    monkeypatch.setattr(common.Session,'_worker',worker)
    assert entry.launch(cfg['source'],cfg['output'],source_directory=ROOT,scratch=tmp_path/'scratch')['status']=='ARTIFICIAL_WORKER'
    monkeypatch.setattr(common,'prepare_source',lambda *a,**k:pytest.fail('consumed preparation'))
    assert entry.launch('/missing',cfg['output'],source_directory='/missing')['physical_calls_this_invocation']==0


def test_setup_repairs_actual_RSS_import_before_probe_without_model_loading(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('setup_test',ROOT/'scripts'/f'{STEM}_setup.py')
    setup=importlib.util.module_from_spec(spec);spec.loader.exec_module(setup);calls=[]
    class Loader:
        def create_module(self,spec):return None
        def exec_module(self,value):
            value.ensure_imports=lambda *a:calls.append(('import',a[0],a[1]))
            value.setup_dependencies=lambda *a:calls.append(('wan_setup',))
            value.command=lambda *a:calls.append(('probe',a[0]))
    monkeypatch.setattr(setup.importlib.util,'spec_from_file_location',lambda *a:importlib.util.spec_from_loader('fake_setup',Loader()))
    setup.setup_dependencies(ROOT,tmp_path)
    assert calls[:2]==[('import','import psutil',['psutil']),('wan_setup',)]
    assert 'import runtime.wan.'+STEM in calls[2][1][-1]
    assert 'load(' not in calls[2][1][-1]


def assert_recovered_slots(result):
    artifacts=result['recovered_artifacts']
    assert sum(len(r['value']['slots']) for r in artifacts.values() if r['kind']=='final')==480
    assert sum(len(r['value']['slots']) for r in artifacts.values() if r['kind']=='local')==1440
    assert len(artifacts['raw_bank.json']['value'])==60
    assert sum(r['raw_bit_slots'] for r in artifacts['raw_bank.json']['value'].values())==9600
    assert sum(r['vote_slots'] for r in artifacts['raw_bank.json']['value'].values())==76800
    return artifacts


def test_hard_interruption_fixed_receipts_CLI_entry_and_sourceless_notebook(tmp_path,monkeypatch,capsys):
    import urllib.request
    out=tmp_path/'out'
    rec.dump(out/'attempt.json',dict(plan=rec.plan(),denominators=rec.denominators(),recovery_layout=rec.recovery_layout()))
    state=rec.plan();state['FULL/B/aggregate'].update(status='RETURNED',started=1,returned=1)
    state['FULL/U/aggregate'].update(status='STARTED',started=1)
    rec.dump(out/'state.json',state)
    good=dict(slots=[dict(slot=i,value=1,status='CONDITIONAL_READ',selected_refs=[],supporting_refs=[]) for i in range(32)])
    good['slots'][0]['selected_refs']=[dict(marker='DO_NOT_EXPAND_REF_IN_DISPLAY')]*100
    rec.dump(out/'final/FULL/B.json',good);rec.dump(out/'final/FULL/U.json',good)
    # A damaged single row must not delete other fixed slots or create a successful value.
    good['slots'][3]['value']=True;rec.dump(out/'final/FULL/B.json',good)
    before={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    result=rec.recover(out);a=assert_recovered_slots(result)
    assert a['final/FULL/B.json']['value']['slots'][0]['value']==1
    assert a['final/FULL/B.json']['value']['slots'][3]['status']=='UNRESOLVED_RECEIPT'
    assert a['final/FULL/U.json']['value']['slots'][0]['value'] is None and a['final/FULL/U.json']['unconfirmed_artifact']
    def module(relative):
        spec=importlib.util.spec_from_file_location('recovery_'+Path(relative).stem,ROOT/relative)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    entry=module('scripts/'+STEM+'_entry.py')
    assert_recovered_slots(entry.launch('/missing',out,source_directory='/missing'))
    builder=module('scripts/build_'+STEM+'_notebook.py')
    monkeypatch.setattr(urllib.request,'urlretrieve',lambda *a:pytest.fail('recovery download'))
    scope=dict(OUTPUT=str(out));exec(builder.RUN,scope)
    assert assert_recovered_slots(scope['result'])==a
    assert scope['summary']['denominators']['final_slots']==480
    assert before=={str(p):p.read_bytes() for p in out.rglob('*') if p.is_file()}
    cli=subprocess.run([sys.executable,'-B',str(ROOT/'experiments/wan_state_clock'/f'{STEM}_run.py'),
        '--output',str(out),'--config','/missing'],cwd=tmp_path,capture_output=True,text=True)
    assert cli.returncode==0,cli.stderr
    assert_recovered_slots(json.loads(cli.stdout))
    raw=rec.recovery_layout()[-1]['fallback'];ids=list(raw)
    state['FULL/q0/p1/bits']=dict(status='RETURNED',started=1,returned=1)
    rec.dump(out/'state.json',state)
    rec.dump(out/'raw_bank.json',{ids[0]:dict(status='RETURNED'),ids[1]:dict(raw[ids[1]],status='RETURNED'),
                               ids[2]:dict(raw[ids[2]],status='RETURNED')})
    recovered=rec.recover(out);a=assert_recovered_slots(recovered);bank=a['raw_bank.json']['value']
    assert bank[ids[0]]['status']=='UNRESOLVED_RECEIPT'
    assert bank[ids[1]]['status']=='RETURNED'
    assert bank[ids[2]]['status']=='UNRESOLVED_RECEIPT' and bank[ids[2]]['unconfirmed_raw_metadata']
    scope=dict(OUTPUT=str(out));exec(builder.RUN,scope)
    assert assert_recovered_slots(scope['result'])==a
    assert scope['result']['recovered_artifacts']['final/FULL/B.json']['value']['slots'][0]['selected_refs']
    displayed=capsys.readouterr().out
    assert 'DO_NOT_EXPAND_REF_IN_DISPLAY' not in displayed and 'recovered_artifacts' not in displayed
    assert 'selected_refs' not in displayed and len(displayed)<12000
    damaged=tmp_path/'damaged';rec.dump(damaged/'attempt.json',dict(recovery_layout=rec.recovery_layout()))
    rec.dump(damaged/'state.json',[])
    assert_recovered_slots(rec.recover(damaged))
    scope=dict(OUTPUT=str(damaged));exec(builder.RUN,scope)
    assert_recovered_slots(scope['result']);assert 'state' in scope['result']['read_errors']


@pytest.mark.parametrize('boundary',['plan_rss','terminal_rss','terminal_cpu','terminal_io'])
def test_monitor_covers_plans_and_terminal_save_with_post_only_failure(tmp_path,monkeypatch,boundary):
    cfg=fixture(tmp_path);out=Path(cfg['output']);flags=dict(active=False,rss=False,closed=False,load=0);clock=[0.]
    real_load,real_dump=rec.load,rec.dump
    monkeypatch.setattr(rec.time,'monotonic',lambda:clock[0])
    class Backend(Fake):
        output=out
        def __init__(self,model):super().__init__(model);flags['active']=True
        def check(self):
            assert flags['active']
            if flags['rss']:raise rec.ResourceExceeded('artificial RSS violation at '+boundary)
        def load(self):flags['load']+=1;return super().load()
        def close(self):
            assert flags['active'];flags.update(active=False,closed=True)
            return dict(peak_resident_bytes=8*1024**3+1 if flags['rss'] else 1)
    def load(path):
        if Path(path).name=='submission.json':
            assert flags['active']
            if boundary=='plan_rss':flags['rss']=True
        return real_load(path)
    def dump(path,value):
        if Path(path).name=='blind_terminal.json':
            assert flags['active']
            if boundary=='terminal_cpu':clock[0]+=1801.
            if boundary=='terminal_rss':flags['rss']=True
            if boundary=='terminal_io':raise OSError('artificial final save failure')
        return real_dump(path,value)
    monkeypatch.setattr(rec,'load',load);monkeypatch.setattr(rec,'dump',dump)
    def evaluator(output,truth):
        assert boundary!='terminal_io','truth/evaluator must not open without a saved blind terminal'
        assert flags['closed'] and (output/'resources.json').exists()
        return dict(status='FINITE_CONDITIONAL_READBACK')
    result=run(cfg,backend_factory=Backend,evaluator=evaluator)
    assert flags['closed'] and result['evaluation_status']=='EVIDENCE_INCOMPLETE'
    resources=real_load(out/'resources.json');assert resources['completion_errors']
    if boundary=='terminal_cpu':assert resources['cpu_receive_save_seconds']==1801 and resources['measurement_end']=='AFTER_BLIND_TERMINAL_SAVE_AND_MONITOR_CLOSE'
    if boundary=='plan_rss':assert flags['load']==0
    # Reading consumed records cannot instantiate a monitor or repeat physical work.
    assert_recovered_slots(run(dict(output=str(out)),backend_factory=lambda *a:pytest.fail('rerun')))
