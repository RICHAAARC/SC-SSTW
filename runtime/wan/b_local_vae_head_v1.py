"""One user-run fixed attempt. Imports are inert; subsequent run is read-only."""
from pathlib import Path
from collections import Counter
import json
import random
import shutil
import time
import traceback
import numpy as np
from main.tube_state import b_local_vae_code_v1 as code
from main.tube_state import b_local_vae_sync_v1 as sync
from runtime.wan import b_local_vae_head_evaluation_v1 as post

READERS=('STATE_SPACE','I')
CASES=(('positive','WM','L','key00'),('null','OFF','L','key00'),('null','OFF','L','key01'),
       ('null','WM','L','key01'),('null','WM','R','key00'))

def load(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError('invalid JSON number '+x)))

def roster(cfg):
    result=[]
    for source in cfg['sources']:
        for fixture in post.FIXTURES:
            for kind,arm,roi,key in CASES:
                task='/'.join((source['id'],fixture,arm+'-'+roi+'-'+key))
                result.append(dict(task=task,source=source['id'],phase=source['role'],fixture=fixture,
                    kind=kind,arm=arm,roi=roi,key=key,information_slots=32))
    return result

def schedule(cfg):
    rng=random.Random(cfg['training']['schedule_seed']);rows=[]
    for step in range(3000):
        source=rng.choice(cfg['training_sources'])['id'];start=rng.randrange(174)
        message=[rng.randrange(2) for _ in range(32)];key='train/'+str(step)+'/'+str(rng.getrandbits(128))
        if key in code.EVAL_KEYS.values():raise ValueError('train/eval key overlap')
        rows.append(dict(step=step,source=source,start=start,message=message,key=key))
    return rows

def private_messages(cfg):
    rng=random.Random(cfg['private_evaluation']['message_seed'])
    return {x['id']:[rng.randrange(2) for _ in range(32)] for x in cfg['sources']}

def validate_config(cfg):
    # Check actual fixed method semantics, never config identity or a digest.
    expected=dict(height=320,width=512,frames=181,fps=8,steps=50,guidance_scale=5.0,max_sequence_length=512)
    if cfg['generation']!=expected:raise ValueError('fixed native generation parameters required')
    if len(cfg['training_sources'])!=4 or len(cfg['sources'])!=6:raise ValueError('four TRAIN and six CAL/DEV sources required')
    if [s['role'] for s in cfg['sources']]!=['CAL']*2+['DEV']*4:raise ValueError('fixed two CAL then four DEV sources required')
    if (cfg['training']['steps'],cfg['training']['batch_frames'],cfg['training']['lr'])!=(3000,8,1e-4):raise ValueError('fixed3000 Adam1e-4 batch8 required')
    if cfg['writer']!=dict(L=[32,288,0,256],R=[32,288,256,512],boundary=16,RGB_limit=8/255,hidden_channels=64,condition_bits=96,block_frames=8):raise ValueError('fixed local head geometry/strength/conditioning required')
    if cfg['receiver']!=dict(path_margin=8,frame_e=16,mean_e=8,min_support=24,min_span=64,all_edited_side_support=8,paths=9275,extract_batch=32):raise ValueError('fixed receiver rules and extractor batch32 required')
    if cfg['private_evaluation']['key00']!=code.EVAL_KEYS['key00'] or cfg['private_evaluation']['key01']!=code.EVAL_KEYS['key01']:raise ValueError('configured keys differ from actual receiver keys')
    if len(roster(cfg))!=150:raise ValueError('fixed150 task roster required')

def budgets():
    return {'preparation':dict(codec_probe=4,codec_decode=4,E181=4,D181=4),
        'training':dict(teacher_ROI=3000,student_ROI=3000,codec_encode=3000,codec_decode=3000,head_step=3000),
        'validation':dict(D181=12,codec_encode=72,codec_decode=72,codec_probe=72,
            extractor_ROI=600,transformer_conditional=300,transformer_unconditional=300,native_step=300)}

class Journal:
    def __init__(self,output,mirror=None):
        self.output=Path(output).expanduser().resolve();self.mirror=Path(mirror).expanduser().resolve() if mirror else None
        self.output.mkdir(parents=True,exist_ok=True)
        self.counts={phase:{kind:dict(planned=n,started=0,returned=0,frames_started=0,frames_returned=0) for kind,n in values.items()} for phase,values in budgets().items()}
        self.active=None
        self.task_states={};self.denominators={}
    def initialize_tasks(self,rows,denominators):
        self.denominators=denominators
        for reader in READERS:
            for row in rows:
                ident=reader+'/'+row['task']
                self.task_states[ident]=dict(row,reader=reader,plan_status='NOT_STARTED',read_status='NOT_STARTED',decision_status='NOT_STARTED')
        for row in rows:
            if row['kind']=='positive':
                for mode in ('O','W_GROUP','U'):
                    ident=mode+'/'+row['task']
                    self.task_states[ident]=dict(row,reader=mode,plan_status='POST_ONLY',read_status='NOT_STARTED',decision_status='NOT_STARTED')
        self.save_tasks()
    def save_tasks(self):
        self.write('task_state.json',dict(denominators=self.denominators,tasks=list(self.task_states.values()),
            actual_errors_unknown_without_post_evaluation=True))
    def transition(self,reader,row,operation,state,decision=None):
        item=self.task_states[reader+'/'+row['task']];item[operation+'_status']=state
        if decision is not None:item['decision_status']=decision
        self.save_tasks()
    def publish(self,relative):
        if self.mirror and self.mirror.resolve()!=self.output.resolve():
            source=self.output/relative;target=self.mirror/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    def write(self,relative,value):
        path=self.output/relative;path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)
        self.publish(relative)
        return load(path)
    def append(self,relative,value):
        line=json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'
        targets=[self.output] if self.mirror is None or self.mirror.resolve()==self.output.resolve() else [self.output,self.mirror]
        for root in targets:
            path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('a') as stream:stream.write(line);stream.flush()
    def event(self,phase,kind,done,frames=0):
        row=self.counts[phase][kind];state='returned' if done else 'started'
        if not done and row['started']>=row['planned']:raise RuntimeError('fixed physical budget exhausted: '+phase+'/'+kind)
        if done and row['returned']>=row['started']:raise RuntimeError('return without physical start')
        row[state]+=1;row['frames_'+state]+=int(frames)
        self.append('physical_events.jsonl',dict(phase=phase,kind=kind,state=state,frames=frames,unix=time.time()))
        # Counts/progress are small and durable, including an in-flight failed call.
        self.write('physical_counts.json',self.counts)
        self.write('progress.json',dict(active=self.active,phase=phase,kind=kind,state=state,ordinal=row[state]))

def existing(output):
    root=Path(output)
    return [name for name in ('attempt.json','result.json','physical_events.jsonl','repeat_start.json','repeat_start_unknown.json') if (root/name).exists()]

def recover(output):
    root=Path(output);result={'status':'READ_ONLY_EXISTING_ATTEMPT','physical_calls_this_invocation':0,'errors':[]}
    for name in ('result.json','evaluation.json','physical_counts.json'):
        try:result[name]=load(root/name)
        except Exception as e:result['errors'].append(dict(file=name,error=f'{type(e).__name__}: {e}'))
    from runtime.wan.b_local_vae_head_storage_v1 import recovery_snapshot
    values={}
    for name in ('task_state.json','quality.json','physical_counts.json'):
        try:values[name]=load(root/name)
        except Exception:pass
    result['fixed_recovery']=recovery_snapshot(values)
    return result

def blank_read(status='NOT_STARTED',error=None):
    return dict(status=status,error=error,information_bits=[None]*32,raw_information_bits=[None]*32,
        planned_information_slots=32,BCH_calls=0,support=[])

def logit_path(row):return Path('logits')/row['source']/row['arm']/row['fixture']/(row['roi']+'.npy')

def blind_phase(journal,rows,phase):
    """No geometry/message arguments; save every plan before any payload read."""
    selected=[r for r in rows if r['phase']==phase]
    for reader in READERS:
        for row in selected:
            journal.active=f'{phase}/{reader}/plan/{row["task"]}'
            journal.transition(reader,row,'plan','STARTED')
            logits=np.load(journal.output/logit_path(row).with_suffix('.pilot.npy'),allow_pickle=False)
            plan=sync.plan(logits,code.EVAL_KEYS[row['key']],reader)
            journal.write(Path('blind')/reader/row['task']/'plan.json',plan)
            journal.transition(reader,row,'plan','RETURNED')
    journal.write(f'{phase}_plans_complete.json',dict(status='ALL_PLANS_SAVED_AND_READ_BACK',plans=2*len(selected),payload_read=False))
    for reader in READERS:
        for row in selected:
            journal.active=f'{phase}/{reader}/read/{row["task"]}'
            journal.transition(reader,row,'read','STARTED')
            plan=load(journal.output/'blind'/reader/row['task']/'plan.json')
            logits=np.load(journal.output/logit_path(row).with_suffix('.payload.npy'),allow_pickle=False)
            read=sync.read_payload(logits,code.EVAL_KEYS[row['key']],plan['support'],detected=plan['detected'])
            journal.write(Path('blind')/reader/row['task']/'read.json',read)
            journal.transition(reader,row,'read','RETURNED',read['status'])
    journal.write(f'{phase}_blind_complete.json',dict(status='ALL_BLIND_READS_SAVED_AND_READ_BACK',reads=2*len(selected)))

def assessed_phase(journal,cfg,rows,phase,messages):
    if not (journal.output/f'{phase}_blind_complete.json').is_file():raise RuntimeError('post truth before phase blind seal')
    journal.write(f'{phase}_post_truth_access.json',dict(status='AFTER_BLIND',unix=time.time()))
    sources={s['id']:s for s in cfg['sources']};result={}
    for reader in READERS:
        result[reader]=[]
        for row in rows:
            if row['phase']!=phase:continue
            folder=journal.output/'blind'/reader/row['task']
            result[reader].append(post.assess(row,load(folder/'plan.json'),load(folder/'read.json'),
                post.source_frames(sources[row['source']]['a'],row['fixture']),messages[row['source']]))
    return result

def physical_phase(backend,journal,cfg,sources,messages,quality_rows):
    terminals={}
    for source in sources:
        journal.active='generate/'+source['id'];terminals[source['id']]=backend.generate(source)
        journal.write(Path('generation')/(source['id']+'.json'),terminals[source['id']])
    journal.active='validation/load_vae';backend.load_vae()
    try:
        for source in sources:
            sid=source['id'];journal.active='native_decode/'+sid
            off=backend.native_decode(terminals[sid]['path'])
            wm=backend.native_decode(terminals[sid]['path'],messages[sid],code.EVAL_KEYS['key00'])
            q=post.quality(off,wm,precodec=True);quality_rows[sid+'/precodec']=q;journal.write('quality.json',quality_rows)
            if not q['outside_L_exact_equal']:raise ValueError('precodec outside L changed')
            full={}
            for arm,canvas in (('OFF',off),('WM',wm)):
                path=journal.output/'media'/sid/arm/'full.mp4'
                full[arm]=backend.roundtrip(canvas,path,'validation');journal.publish(path.relative_to(journal.output))
            del off,wm
            quality_rows[sid+'/full']=post.quality(full['OFF'],full['WM'],precodec=False);journal.write('quality.json',quality_rows)
            for fixture in post.FIXTURES:
                frames=post.source_frames(source['a'],fixture);received={}
                for arm in ('OFF','WM'):
                    path=journal.output/'media'/sid/arm/(fixture+'.mp4')
                    received[arm]=backend.roundtrip(full[arm][frames].float()/255,path,'validation')
                    journal.publish(path.relative_to(journal.output))
                    cache=backend.scratch/'received'/sid/arm/(fixture+'.npy');cache.parent.mkdir(parents=True,exist_ok=True)
                    np.save(cache,received[arm].numpy(),allow_pickle=False)
                quality_rows[sid+'/'+fixture]=post.quality(received['OFF'],received['WM'],precodec=False)
                journal.write('quality.json',quality_rows)
    finally:backend.release_vae()
    journal.active='validation/load_extractor';backend.load_video()
    try:
        import torch
        for source in sources:
            sid=source['id']
            for fixture in post.FIXTURES:
                for arm in ('OFF','WM'):
                    pixels=torch.from_numpy(np.load(backend.scratch/'received'/sid/arm/(fixture+'.npy'),allow_pickle=False))
                    for roi in ('L','R'):
                        journal.active=f'extract/{sid}/{fixture}/{arm}/{roi}'
                        logits=backend.extract(pixels,roi)
                        path=Path('logits')/sid/arm/fixture/(roi+'.npy');(journal.output/path).parent.mkdir(parents=True,exist_ok=True)
                        np.save(journal.output/path,logits,allow_pickle=False);journal.publish(path)
                        # Planning never opens the payload-containing archive.
                        for suffix,values in (('.pilot.npy',logits[:,:32]),('.payload.npy',logits[:,32:96])):
                            part=path.with_suffix(suffix);np.save(journal.output/part,values,allow_pickle=False);journal.publish(part)
                        journal.write(path.with_suffix('.windows.json'),dict(native_logits=256,head_bits=96,
                            windows=[dict(start=r,length=min(8,151-r),mean_logits=logits[r:r+8].mean(axis=0).tolist()) for r in range(0,151,8)]))
    finally:backend.release_video()

def final_evaluation(journal,cfg,rows,messages,quality_rows,stop):
    sources={s['id']:s for s in cfg['sources']};evaluated={reader:[] for reader in READERS};failures=[]
    # Explicit pending reads retain every denominator, including a failed in-flight logical operation.
    for reader in READERS:
        for row in rows:
            folder=Path('blind')/reader/row['task'];plan_path=journal.output/folder/'plan.json';read_path=journal.output/folder/'read.json'
            if not plan_path.is_file():journal.write(folder/'plan.json',dict(status='NOT_STARTED',support=[],detected=False))
            if not read_path.is_file():journal.write(folder/'read.json',blank_read('UNRESOLVED' if plan_path.is_file() and load(plan_path).get('status')!='NOT_STARTED' else 'NOT_STARTED',stop))
    journal.write('blind_complete.json',dict(status='ALL_FIXED_ROWS_RECORDED',main_tasks=150,I_tasks=150,includes_unresolved=True))
    journal.write('post_truth_access.json',dict(status='AFTER_ALL_FIXED_BLIND_ROWS',unix=time.time()))
    for reader in READERS:
        for row in rows:
            folder=journal.output/'blind'/reader/row['task']
            value=post.assess(row,load(folder/'plan.json'),load(folder/'read.json'),post.source_frames(sources[row['source']]['a'],row['fixture']),messages[row['source']])
            evaluated[reader].append(value)
            if value['status'] not in ('OUTPUT','ERASURE') or (row['kind']=='positive' and (not value['full_message_correct'] or not value['all_used_references_exact'])) or value['null_detection'] or value['null_output']:
                failures.append(dict(reader=reader,task=row['task'],status=value['status'],full_message_correct=value['full_message_correct'],wrong_references=value['erroneous_reference_frames'],null_detection=value['null_detection'],null_output=value['null_output']))
    diagnostics=[]
    for row in rows:
        if row['kind']!='positive':continue
        actual=post.source_frames(sources[row['source']]['a'],row['fixture']);path=journal.output/logit_path(row)
        for mode in ('O','W_GROUP','U'):
            support=sync.oracle_support(actual,mode)
            if path.is_file():journal.transition(mode,row,'read','STARTED')
            try:value=sync.read_payload(np.load(path,allow_pickle=False)[:,32:96],code.EVAL_KEYS['key00'],support) if path.is_file() else blank_read()
            except Exception as error:value=blank_read('TECHNICAL_UNREAD',f'{type(error).__name__}: {error}')
            truth=messages[row['source']];raw=value['raw_information_bits'];known=[i for i,x in enumerate(raw) if x is not None]
            report=dict(task=row['task'],mode=mode,status=value['status'],information_slots=32,
                full_message_correct=value.get('information_bits')==truth if value['status']=='OUTPUT' else False,
                wrong_decoded_message=value['status']=='OUTPUT' and value['information_bits']!=truth,
                raw_observed_slots=len(known),raw_errors=sum(raw[i]!=truth[i] for i in known),
                raw_BER=sum(raw[i]!=truth[i] for i in known)/len(known) if known else None,read=value,
                presence_claim=False,post_only=True)
            journal.write(Path('post_OWU')/row['task']/(mode+'.json'),report);diagnostics.append(report)
            if path.is_file():journal.transition(mode,row,'read','RETURNED',value['status'])
    cal=post.criteria([r for r in evaluated['STATE_SPACE'] if r['phase']=='CAL'],'CAL')
    dev=post.criteria([r for r in evaluated['STATE_SPACE'] if r['phase']=='DEV'],'DEV')
    ow=[r for r in diagnostics if r['mode']=='O'];oracle_pass=all(r['full_message_correct'] for r in ow)
    qcomplete=all(q.get('status')=='RECORDED' for q in quality_rows.values())
    physical_complete=all(row['started']==row['returned']==row['planned'] for phase in journal.counts.values() for row in phase.values())
    method=cal['passed'] and dev['passed'] and qcomplete and physical_complete and stop is None
    oracle_complete=all(r['status'] in ('OUTPUT','ERASURE') for r in ow)
    per_source={reader:{sid:post.summary([r for r in rs if r['source']==sid]) for sid in sources} for reader,rs in evaluated.items()}
    report=dict(status='POST_RECORDED',method_status='FIXED_SCOPE_CRITERIA_MET' if method else 'METHOD_NOT_SUPPORTED',
        recommendation='PROPOSE_NEW_SOURCE_CONFIRMATION_ONLY' if method and oracle_pass else 'CLOSE_THIS_CANDIDATE',
        writer_diagnostic='ORACLE_ALL_MESSAGES_RECOVERED' if oracle_pass else 'ORACLE_RECOVERY_FAILED' if oracle_complete else 'ORACLE_UNRESOLVED',
        oracle_is_main_method_criterion=False,
        CAL=cal,DEV=dev,oracle_all_messages_correct=oracle_pass,quality_complete=qcomplete,physical_budget_complete=physical_complete,denominators=cfg['denominators'],
        summaries={k:post.summary(v) for k,v in evaluated.items()},per_source=per_source,
        rows=evaluated,paired_STATE_SPACE_vs_I=post.paired(evaluated['STATE_SPACE'],evaluated['I']),post_OWU=diagnostics,
        failures=failures,quality=quality_rows,stop=stop,
        claim_ceiling='fixed seen-source native writer development evidence; ECC is not authentication or per-bit physical support; no general FPR or perceptual PASS')
    journal.write('evaluation.json',report)
    return report

def run(cfg,output,inputs,*,scratch=None,mirror=None,backend_factory=None):
    if existing(output):return recover(output)
    validate_config(cfg);journal=Journal(output,mirror);rows=roster(cfg);messages=private_messages(cfg)
    journal.write('attempt.json',dict(status='STARTED_ONE_FIXED_ATTEMPT',unix=time.time(),retry=False))
    journal.write('protocol.json',cfg);journal.write('plan.json',dict(main=rows,I=rows,post_OWU=[dict(task=r['task'],mode=m,slots=32) for r in rows if r['kind']=='positive' for m in ('O','W_GROUP','U')],physical=budgets()))
    journal.initialize_tasks(rows,cfg['denominators'])
    training=schedule(cfg);journal.write('training_plan.json',training)
    journal.write('private_writer_truth.json',messages)
    quality_rows={s['id']+'/'+scope:dict(status='NOT_STARTED',frames=181 if scope in ('precodec','full') else 151,adjacent=180 if scope in ('precodec','full') else 150) for s in cfg['sources'] for scope in ('precodec','full',*post.FIXTURES)}
    journal.write('quality.json',quality_rows);journal.write('physical_counts.json',journal.counts)
    if backend_factory is None:
        from runtime.wan.b_local_vae_head_backend_v1 import Backend
        backend_factory=Backend
    backend=None;stop=None
    try:
        import torch
        inputs={key:str(Path(value).expanduser().resolve()) for key,value in inputs.items()}
        backend=backend_factory(cfg,str(Path(scratch or cfg['scratch']).expanduser().resolve()),journal.event)
        journal.active='preparation/load_vae';backend.load_vae();caches={};channels=set()
        try:
            for source in cfg['training_sources']:
                sid=source['id'];journal.active='prepare/'+sid
                receipt=backend.prepare_cache(inputs[sid],sid);journal.write(Path('preparation')/(sid+'.json'),receipt);channels.add(receipt['channels'])
                folder=Path(receipt['cache']);caches[sid]=dict(features=np.load(folder/'features.npy',mmap_mode='r'),preclip=np.load(folder/'preclip.npy',mmap_mode='r'))
        finally:backend.release_vae()
        if len(channels)!=1:raise ValueError('actual feature channels differ across training sources')
        backend.initialize_head(channels.pop());optimizer=torch.optim.Adam(backend.head.parameters(),lr=1e-4)
        journal.active='training/load_teacher_and_extractor';backend.load_video()
        try:
            for row in training:
                journal.active='training/'+str(row['step']);journal.event('training','head_step',False,8)
                receipt=backend.train_step(row,caches[row['source']],optimizer)
                journal.append('training_steps.jsonl',receipt);journal.event('training','head_step',True,8)
        finally:backend.release_video()
        checkpoint=journal.output/'head_final.pt';torch.save(dict(channels=backend.head.channels,state_dict={k:v.detach().cpu() for k,v in backend.head.state_dict().items()},steps=3000),checkpoint);journal.publish('head_final.pt')
        del caches,optimizer
        for phase in ('CAL','DEV'):
            sources=[s for s in cfg['sources'] if s['role']==phase]
            physical_phase(backend,journal,cfg,sources,messages,quality_rows)
            blind_phase(journal,rows,phase)
            assessed=assessed_phase(journal,cfg,rows,phase,messages)
            judgment=post.criteria(assessed['STATE_SPACE'],phase);journal.write(phase+'_criteria.json',judgment)
            if phase=='CAL' and not judgment['passed']:
                stop=dict(kind='METHOD',stage='CAL',reason='CAL_FIXED_CRITERIA_FAILED_DEV_NOT_STARTED');break
    except BaseException as error:
        stop=dict(kind='TECHNICAL',active=journal.active,error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc())
    finally:
        if backend is not None:backend.release_video();backend.release_vae()
    evaluation=final_evaluation(journal,cfg,rows,messages,quality_rows,stop)
    result=dict(status='COMPLETED_FIXED_ATTEMPT' if stop is None else 'STOPPED_FIXED_ATTEMPT',stop=stop,
        denominators=cfg['denominators'],physical_counts=journal.counts,method_status=evaluation['method_status'],
        physical_calls_this_invocation=sum(r['started'] for p in journal.counts.values() for r in p.values()),retry=False)
    journal.write('result.json',result);return result
