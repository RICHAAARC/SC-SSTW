"""Artificial orchestration and transport checks. No model, real media or solver."""
from pathlib import Path
from collections import Counter
import importlib.util
import json
import zipfile
import numpy as np
import pytest
from experiments.wan_state_clock import b_joint_time_packet_a_physical_v1_records as rec
from experiments.wan_state_clock import b_joint_time_packet_a_gpu_v1 as gpu
from experiments.wan_state_clock import b_joint_time_packet_a_cpu_media_v1 as media
from main.tube_state import b_joint_time_packet_a_identity_v1 as identity
from main.tube_state import b_joint_time_packet_a_receiver_v1 as receiver

pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def decoded(uid):
    return dict(key='key00',regions={region:dict(local_cells=[dict(local_row=row,evidence_id=f'{uid}/row{row}/{region}',
        bits=[dict(local_bit=b,evidence_id=f'{uid}/row{row}/{region}',value=b%2,planned_count=8,
                   returned_count=8,status='AVAILABLE',cause_set=[],tie=True,ones=4)
              for b in range(8)]) for row in range(1,11)]) for region in ('R0','R1')})


def test_exact_catalog_domain_and_bank_denominators(monkeypatch):
    cpu=Counter(x['kind'] for x in rec.plan('cpu_media').values())
    assert (cpu['codec_encode'],cpu['codec_probe'],cpu['codec_decode'])==(12,17,27)
    g1=Counter(x['kind'] for x in rec.plan('gpu1').values());g2=Counter(x['kind'] for x in rec.plan('gpu2').values())
    assert (g1['model_load'],g1['D181'],g1['baseline_head'],g1['head_replay'])==(1,1,46,92)
    assert (g2['model_load'],g2['E41'],g2['project'],g2['bits'],g2['old_cache_load'])==(1,180,180,180,60)
    assert len(rec.identity_units())==240 and len(rec.time_units())==30
    assert sum(u['nodes'] for u in rec.time_units())==4932 and sum(u['edges'] for u in rec.time_units())==4902
    checked=[]
    def artificial_score(terms,states,coding):
        checked.extend(states.tolist());return dict(scores=np.zeros((1,len(states)),np.float64))
    monkeypatch.setattr(receiver,'score_states',artificial_score)
    soft=np.zeros((1,5,2,63));mask=np.ones(soft.shape,bool)
    chunks=list(receiver.complete_scores(soft,mask,'NEW'))
    assert checked==list(range(4,92672)) and len(chunks)==46
    assert all(b-a<=2048 for a,b,c in chunks)


def test_gpu1_one_load_partial_source_and_consumed_reentry(tmp_path):
    cpu=tmp_path/'cpu';rec.dump(cpu/'cpu_handoff.json',dict(stage_terminal=True,source_eligibility=dict(n01=True,e02=True)))
    class Fake:
        def __init__(self,model):pass
        def load(self):return dict(status='FAKE_LOADED')
        def capture(self,path,out,event):
            event('D181',False)
            for _ in range(46):event('baseline_head',False);event('baseline_head',True)
            event('D181',True);return dict(base='fakebase',features='fakefeatures')
        def replay(self,cpu,base,features,out,event,dump):
            if Path(cpu).name=='n01':raise FileNotFoundError('artificial missing original cache')
            for _ in range(46):event('head_replay',False);event('head_replay',True)
            return dict(status='NATIVE_MEASURED')
        def close(self):return dict(fake=True)
    output=tmp_path/'gpu1'
    value=gpu.gpu1(dict(cpu_handoff=str(cpu),e02_terminal='fakeP0',n01_base='missing',n01_features='missing'),output,backend_factory=Fake)
    assert value['sources']['e02']['status']=='NATIVE_MEASURED' and value['sources']['n01']['status']=='UNRESOLVED'
    counts=rec.load(output/'counts.json');assert counts['head_replay']['returned']==46 and counts['D181']['returned']==1
    assert gpu.gpu1({},output,backend_factory=lambda _:pytest.fail('reentry loaded'))['physical_calls_this_invocation']==0


def test_cpu_media_partial_source_fixed_codec_and_relative_paths(tmp_path):
    native=tmp_path/'gpu1/n01/NEW_preclip.npy';native.parent.mkdir(parents=True);np.save(native,np.zeros((181,1),np.float32))
    rec.dump(tmp_path/'gpu1/blind_terminal.json',dict(stage_terminal=True,sources=dict(n01=dict(status='NATIVE_MEASURED',preclip='n01/NEW_preclip.npy'),e02=dict(status='UNRESOLVED'))))
    class Fake:
        def quantize(self,value):return np.zeros((181,1),np.uint8)
        def rgb_raw(self,value,layer):return dict(soft=np.zeros((len(value),5,2,63)))
        def encode(self,value,path):Path(path).write_bytes(b'artificial')
        def probe(self,path,n):return dict(artificial=True,frames=n)
        def decode(self,path,n):return np.zeros((n,1),np.uint8)
        def decode_y(self,path,n,probe):return np.zeros((n,1),np.uint8)
        def receive(self,path,out,coding,frames,dump,*,native_y=None):return dict(status='BLIND_SAVED')
        def close(self):return dict(artificial=True)
    cfg=dict(gpu1=str(tmp_path/'gpu1'),e02_old_media={v:'old-saved' for v in rec.VIEWS},layer_raw={},
             time_raw={u['id']:'saved-old-raw' for u in rec.time_units() if u['arm']!='NEW'})
    output=tmp_path/'media';value=media.run(cfg,output,backend_factory=Fake)
    counts=rec.load(output/'counts.json')
    assert tuple(counts['codec_'+k]['returned'] for k in ('encode','probe','decode'))==(6,11,16)
    assert len(value['time_units'])==30 and value['time_units']['e02/NEW/FULL']['status']=='UNRESOLVED'
    assert value['time_units']['n01/NEW/FULL']['status']=='BLIND_SAVED'
    for item in value['rgb'].values():
        if item['status']=='SAVED':assert not Path(item['file']).is_absolute() and (output/item['file']).is_file()


def test_moved_cpu_zip_to_gpu2_partial_bank_no_extra_calls(tmp_path):
    entry=module('b_joint_time_packet_a_physical_v1_entry')
    before=tmp_path/'original-local-cpu';path=before/'media/e02/NEW/CROP/received.RGB.npy';path.parent.mkdir(parents=True)
    # Sparse artificial input with the real format/shape, not real video pixels or a codec call.
    pixels=np.lib.format.open_memmap(path,mode='w+',dtype=np.uint8,shape=(101,320,512,3));pixels.flush();del pixels
    rec.dump(before/'blind_terminal.json',dict(stage_terminal=True,rgb={'e02/NEW/CROP':dict(status='SAVED',file=str(path.relative_to(before)))}))
    archive=tmp_path/'cpu-media.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for f in before.rglob('*'):
            if f.is_file():z.write(f,str(f.relative_to(before)))
    moved=entry.unpack(archive,tmp_path/'other-machine','blind_terminal.json')
    bank=tmp_path/'oldbank';uid='input0/E41/a0';rec.dump(bank/uid/'bits.json',decoded(uid))
    calls=[]
    class Fake:
        def __init__(self,model):pass
        def load(self):calls.append('load');return dict(fake=True)
        def encode(self,pixels,start,source):
            assert pixels.shape==(101,320,512,3) and pixels[0,0,0,0]==0
            calls.append('E41');return None
        def project(self,z):calls.append('project');return None
        def bits(self,p,uid):calls.append('bits');return decoded(uid)
        def close(self):return dict(fake=True)
    value=gpu.gpu2(dict(cpu_media=str(moved),n01_old_bank=str(bank)),tmp_path/'gpu2',backend_factory=Fake)
    assert Counter(calls)==dict(load=1,E41=12,project=12,bits=12)
    assert len(value['bank'])==240 and sum(x['status']=='RETURNED' for x in value['bank'].values())==13
    assert len(rec.load(tmp_path/'gpu2/addressed_raw_bits.json'))==38400
    counts=rec.load(tmp_path/'gpu2/counts.json');assert counts['old_cache_load']['started']==60
    assert gpu.gpu2({},tmp_path/'gpu2')['physical_calls_this_invocation']==0


def test_archive_escape_and_identity_context_rejected(tmp_path):
    entry=module('b_joint_time_packet_a_physical_v1_entry')
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:z.writestr('../outside','bad')
    with pytest.raises(ValueError,match='unsafe'):entry.unpack(path,tmp_path/'dest','blind_terminal.json')
    unit=rec.identity_units()[0]
    rows=identity.addressed(decoded(unit['old_id']),unit,unit['old_id'])
    assert len(rows)==160 and rows[0]['raw']['tie'] is True
    with pytest.raises(ValueError,match='address'):identity.addressed(decoded('wrong-window'),unit,unit['old_id'])


def test_notebook_builder_and_all_consumed_without_source_or_config(tmp_path):
    builder=module('build_b_joint_time_packet_a_physical_v1_notebook');builder.build(tmp_path)
    book=json.loads((tmp_path/'notebooks/b_joint_time_packet_a_physical_v1.ipynb').read_text())
    assert ''.join(book['cells'][0]['source'])=="from google.colab import drive\ndrive.mount('/content/drive')\n"
    assert len(book['cells'])==4 and all(not c.get('outputs') for c in book['cells'])
    output=tmp_path/'attempts'
    for stage in ('gpu1','gpu2'):rec.dump(output/stage/'attempt.json',dict(status='CONSUMED'))
    scope=dict(OUTPUT=str(output),CPU_HANDOFF='/missing/cpu.zip',CPU_MEDIA='/missing/media.zip',
               SOURCE_DIRECTORY='/missing/source',SOURCE_REF='irrelevant',MODEL_ID='irrelevant')
    exec(builder.RUN,scope)
    assert all(v['physical_calls_this_invocation']==0 for v in scope['result']['stages'].values())
    bootstrap=module('b_joint_time_packet_a_physical_v1_bootstrap')
    value=bootstrap.launch('/missing/cpu','/missing/media',output,source_directory='/missing/source')
    assert all(v['physical_calls_this_invocation']==0 for v in value['stages'].values())


def test_minimal_media_zip_excludes_truth_and_stage_resource_counts(tmp_path):
    pack=module('b_joint_time_packet_a_handoff_v1');source=tmp_path/'media'
    rec.dump(source/'blind_terminal.json',dict(stage_terminal=True,rgb={'e02/NEW/FULL':dict(status='SAVED',file='pixels.npy')}))
    (source/'pixels.npy').write_bytes(b'artificial pixels')
    rec.dump(source/'private/attack_truth.json',dict(private='must not enter GPU2 package'))
    rec.dump(source/'received/scores.json',dict(large='local only'))
    result=pack.package('cpu_media',source,tmp_path/'handoff.zip')
    assert set(result['included_files'])=={'blind_terminal.json','pixels.npy'}
    with zipfile.ZipFile(tmp_path/'handoff.zip') as z:assert set(z.namelist())==set(result['included_files'])
    with pytest.raises(ValueError,match='relative'):rec.package_file(source,'/another-machine/old-path.npy')
    journal=rec.Journal(tmp_path/'journal','gpu2')
    def returned_over_limit():
        error=TimeoutError('artificial model elapsed');error.physical_returned=True;raise error
    with pytest.raises(TimeoutError):journal.call('model_load',returned_over_limit)
    assert journal.rows['model_load']['returned']==1 and journal.rows['model_load']['status']=='FAILED'


def test_post_barrier_and_saved_scores_do_not_rescore(tmp_path):
    from experiments.wan_state_clock import b_joint_time_packet_a_post_v1 as post
    output=tmp_path/'post'
    assert post.run(dict(cpu_media=str(tmp_path/'missing'),gpu2=str(tmp_path/'missing2')),output)['status']=='WAITING_INPUT'
    assert not output.exists()
    folder=tmp_path/'scores';folder.mkdir()
    # One small synthetic frame, all states saved with an unambiguous truth and old competitor.
    values=np.ones((1,92668),np.float64);values[0,10-4]=.1;values[0,11-4]=.2
    np.save(folder/'all.npy',values,allow_pickle=False)
    rec.dump(folder/'scores.json',dict(frames=1,chunks=[dict(start=4,stop=92672,file='all.npy')]))
    np.savez(folder/'startup.npz',numerator=np.zeros((1,4)),denominator=np.array([[63,126,189,252]]),
             partial=np.zeros((1,4)),active=np.arange(5)[None,:]<=np.arange(4)[:,None])
    result=post.score_report(folder,[10],[dict(old_final=11)])
    assert result[0]['truth_rank']==1 and result[0]['mature_best_count']==1
    assert result[0]['queried_mature_scores']['old_final']==.2
    assert all(x['mature_rank'] is None for x in result[0]['startup'])


def test_post_missing_evidence_keeps_every_fixed_row(tmp_path):
    from experiments.wan_state_clock import b_joint_time_packet_a_post_v1 as post
    cpu,gpu=tmp_path/'cpu',tmp_path/'gpu'
    for root in (cpu,gpu):rec.dump(root/'blind_terminal.json',dict(stage_terminal=True))
    rec.dump(cpu/'private/attack_truth.json',dict(views=media.construction_maps()))
    out=tmp_path/'post';result=post.run(dict(cpu_media=str(cpu),gpu2=str(gpu),old_time={},writer_protocol={}),out)
    assert result['stage_terminal'] is True
    nodes,edges=rec.load(out/'time_nodes.json'),rec.load(out/'time_edges.json')
    assert len(nodes)==4932 and len(edges)==4902
    assert all(x['original_final_status']=='BASELINE_UNAVAILABLE' and x['local_status'] in ('UNKNOWN','DESCRIPTIVE_GEOMETRY_UNKNOWN') for x in nodes)
    assert all(x['geometry_status']=='UNKNOWN' for x in edges)
    assert not any(x['diagnostic_corrected_old_wrong'] or x['diagnostic_damaged_old_correct'] or x['diagnostic_lost_old_correct'] for x in nodes)
    bits=rec.load(out/'identity_rows.json');assert len(bits)==38400 and all(x['value'] is None for x in bits)
    assert result['results']['identity']['comparisons']['n01/FULL']['common_comparable']==0
    assert post.run({},out)['physical_calls_this_invocation']==0


def test_post_unknown_unresolved_and_saved_packets_after_score_failure(tmp_path):
    from experiments.wan_state_clock import b_joint_time_packet_a_post_v1 as post
    cpu=tmp_path/'cpu';out=tmp_path/'post';errors={};mappings=media.construction_maps()
    states=np.array([0,1,2,3,9,5,6,7,8,9,10,11,12])
    bits=receiver.template(states,'NEW')['bits'];soft=np.repeat((2*bits.astype(float)-1)[:,:,None,:],2,axis=2)
    soft[2]=0.  # Real saved erasure, unlike missing packet evidence.
    alternate=receiver.template(np.array([11]),'NEW')['bits'];soft[10,:,1,:]=2*alternate[0].astype(float)-1
    packet=receiver.diagnostics(soft,np.ones(soft.shape,bool),'NEW')
    packet['nodes'][0]=None;packet['nodes'][5]['received_frame']=50;packet['nodes'][8].pop('compatible_n')
    packet['edges'][0]=None;packet['edges'][2]['received_edge']=200
    rec.dump(cpu/'received/n01/NEW/FULL/packets.json',packet)
    rec.dump(cpu/'received/e02/NEW/FULL/packets.json',[])  # Invalid top-level saved object.
    def old_node(r,n):return dict(received_frame=r,output_index=n,interval=[n,n] if n is not None else [0,92671],origin='OLD' if n is not None else 'UNRESOLVED')
    baseline=[old_node(r,r) for r in range(13)]
    baseline[1]=None;baseline[3]=old_node(3,2);baseline[6]['output_index']='6';baseline[7]['received_frame']=70;baseline[9]=old_node(9,None)
    # Historical conditional NEW output preserves its original non-singleton H interval.
    baseline[11].update(origin='NEW',interval=[10,12])
    baseline[12].update(origin='NEW',interval=[13,14])  # Output 12 lies outside its saved interval.
    old_edges=[dict(received_left=0,output_endpoints=None,status='AMBIGUOUS',kind='AMBIGUOUS'),None,
               dict(received_left=2,output_endpoints=[2,3],status='EXACT',kind='NORMAL'),
               dict(received_left=300,output_endpoints=[3,4],status='EXACT',kind='NORMAL')]
    rec.dump(tmp_path/'old-full.json',dict(final_nodes=baseline,final_relations=old_edges))
    rec.dump(tmp_path/'old-crop.json',dict(final_nodes=[old_node(r,n) for r,n in enumerate(mappings['CROP'])],final_relations=[]))
    rec.dump(tmp_path/'invalid-old.json',[])
    config=dict(old_time={'n01/FULL':dict(final=str(tmp_path/'old-full.json')),
                         'n01/CROP':dict(final=str(tmp_path/'old-crop.json')),
                         'e02/FULL':dict(final=str(tmp_path/'invalid-old.json'))})
    terminal=dict(time_units={'n01/NEW/FULL':dict(status='UNRESOLVED',error='artificial score failure after packets saved')})
    summary=post.time_report(config,cpu,terminal,out,mappings,errors)
    all_nodes=rec.load(out/'time_nodes.json');nodes=[x for x in all_nodes if x['unit']=='n01/NEW/FULL']
    assert nodes[0]['local_status']=='UNKNOWN' and nodes[0]['original_final_status']=='CORRECT' and not nodes[0]['diagnostic_lost_old_correct']
    assert nodes[1]['local_status']=='CORRECT' and nodes[1]['original_final_status']=='BASELINE_UNAVAILABLE'
    assert nodes[2]['local_status']=='UNRESOLVED' and nodes[2]['diagnostic_lost_old_correct']
    assert nodes[3]['local_status']=='CORRECT' and nodes[3]['diagnostic_corrected_old_wrong']
    assert nodes[4]['local_status']=='WRONG' and nodes[4]['diagnostic_damaged_old_correct']
    assert nodes[5]['local_status']==nodes[8]['local_status']=='UNKNOWN'
    assert nodes[6]['original_final_status']==nodes[7]['original_final_status']=='BASELINE_UNAVAILABLE'
    assert nodes[9]['original_final_status']=='UNRESOLVED' and not nodes[9]['diagnostic_corrected_old_wrong']
    assert nodes[10]['local_status']=='UNRESOLVED' and nodes[10]['diagnostic_lost_old_correct']
    assert nodes[11]['original_final_status']=='CORRECT' and nodes[11]['original_final']==11 and nodes[11]['baseline_available']
    assert nodes[12]['original_final_status']=='BASELINE_UNAVAILABLE' and not nodes[12]['baseline_available']
    assert nodes[11]['local_status']==nodes[12]['local_status']=='CORRECT'
    assert not any(nodes[12][k] for k in ('diagnostic_corrected_old_wrong','diagnostic_damaged_old_correct','diagnostic_lost_old_correct'))
    saved_history=rec.load(out/'historical_H_L_unchanged.json')['n01/FULL']['final']['final_nodes']
    assert saved_history[11]==baseline[11] and saved_history[11]['interval']==[10,12]
    assert saved_history[12]==baseline[12]
    assert nodes[3]['evidence_status']=='UNRESOLVED' and nodes[3]['local_record_available']
    crop=[x for x in all_nodes if x['unit']=='n01/NEW/CROP']
    assert all(x['local_status']=='UNKNOWN' and x['original_final_status']=='CORRECT' and not x['diagnostic_lost_old_correct'] for x in crop)
    edges=rec.load(out/'time_edges.json');new_edges=[x for x in edges if x['unit']=='n01/NEW/FULL'];old_edges=[x for x in edges if x['unit']=='n01/OLD/FULL']
    assert [x['geometry_status'] for x in new_edges[:4]]==['UNKNOWN','UNRESOLVED','UNKNOWN','WRONG']
    assert [x['geometry_status'] for x in old_edges[:4]]==['UNRESOLVED','UNKNOWN','CORRECT','UNKNOWN']
    assert summary['preservation']['n01']['lost_old_correct']==2
    assert len(all_nodes)==4932 and len(edges)==4902


def test_owned_worker_normal_exit_receipt(tmp_path):
    import sys
    entry=module('b_joint_time_packet_a_physical_v1_entry')
    receipt=tmp_path/'worker.json';stage=tmp_path/'stage'
    code=entry.run_worker([sys.executable,'-c',"print('artificial worker')"],tmp_path,tmp_path/'worker.log',receipt,stage)
    saved=json.loads(receipt.read_text())
    assert code==0 and saved['status']=='WORKER_EXITED' and saved['reaped']
    assert not saved['terminate_sent'] and not saved['kill_sent'] and not saved['blind_terminal_present']
    assert not stage.exists()


@pytest.mark.parametrize('force_kill',[False,True])
def test_owned_worker_interrupt_term_timeout_kill_reap(tmp_path,force_kill):
    import subprocess
    import sys
    import time
    entry=module('b_joint_time_packet_a_physical_v1_entry')
    ready=tmp_path/'ready';stage=tmp_path/'stage';rec.dump(stage/'attempt.json',dict(status='CONSUMED'))
    program="import sys,time,signal;from pathlib import Path;"+("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if force_kill else '')+"Path(sys.argv[1]).write_text('ready');time.sleep(60)"
    owned=[];waits=[]
    def factory(*args,**kwargs):
        process=subprocess.Popen(*args,**kwargs);owned.append(process);began=time.monotonic()
        while not ready.exists() and time.monotonic()-began<5:time.sleep(.01)
        if not ready.exists():process.kill();process.wait();raise AssertionError('artificial child did not become ready')
        class InterruptOnce:
            interrupted=False
            def __getattr__(self,name):return getattr(process,name)
            def wait(self,timeout=None):
                waits.append(timeout)
                if not self.interrupted:self.interrupted=True;raise KeyboardInterrupt('artificial parent interruption')
                # Exercise the real owned-child KILL/reap path without a 30-second test delay.
                if force_kill and timeout==30:raise subprocess.TimeoutExpired(args[0],30)
                return process.wait(timeout=timeout)
        return InterruptOnce()
    receipt=tmp_path/'worker.json'
    with pytest.raises(KeyboardInterrupt):
        entry.run_worker([sys.executable,'-c',program,str(ready)],tmp_path,tmp_path/'worker.log',receipt,stage,process_factory=factory)
    saved=json.loads(receipt.read_text())
    assert len(owned)==1 and owned[0].poll() is not None and saved['reaped']
    assert saved['status']=='INTERRUPTED' and saved['terminate_sent'] and saved['kill_sent']==force_kill
    assert 30 in waits and saved['attempt_present'] and not saved['blind_terminal_present']
    assert not saved['worker_restarted'] and sorted(p.name for p in stage.iterdir())==['attempt.json']
