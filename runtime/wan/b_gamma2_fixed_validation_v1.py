"""Fixed gamma2 paired validation; import and existing-output recovery are stdlib only."""
from collections import Counter
import errno
import json
import math
import os
from pathlib import Path
import time
import uuid

STEM='b_gamma2_fixed_validation_v1'
DEFAULT_INPUT='/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1'
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-Gamma2-Fixed-Validation-V1/fixed-four-source-v1'
HOSTS=(('b2-m3-c01','cal',2),('b2-m3-c02','cal',11),('b2-m3-e01','eval',17),('b2-m3-e02','eval',6))
ARMS=('OFF','P0','PM1','PM2')
KEYS={'key00':'watermark','key01':'watermark-wrong'}
WINDOWS={'L':0,'M':53,'R':106}
FIXTURES={'CROP':(0,None),'DELETE_LEFT_GAP':(8,49),'DELETE_RIGHT_GAP':(8,102)}
READERS=('CDC_E','CDC_M','UNION_M','ALIAS_M_X10','REVERSE_ORDER')
BUDGET={'D181':4,'E41':144,'codec_encode':16,'codec_probe':16,'codec_decode':16}
COMMON_RULE='max(tau(shared72+PM1wrong18),tau(shared72+PM2wrong18)); unique_null_families=108'

def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        try:
            os.fsync(stream.fileno())
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
                raise
    temp.replace(path)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def recover_ledger(output):
    """Read-only receipt overlay; also embedded verbatim in the notebook.

    STARTED remains spent and incomplete. Invalid or uncommitted receipts
    remain unresolved, never an inferred successful or unentered call.
    """
    from collections import Counter
    import json
    from pathlib import Path
    output = Path(output)
    ledger={slot:dict(slot=slot,kind=kind,status='NOT_ENTERED',attempts=0) for slot,kind in plan_units().items()}
    errors, receipt_slots = {}, []
    try:
        saved = json.loads((output/'ledger.json').read_text(encoding='utf-8'))
        if not isinstance(saved,dict):raise ValueError('ledger must map slots to records')
        for slot,row in saved.items():
            if slot not in ledger:
                errors['ledger/'+slot]='unknown saved slot';continue
            if not isinstance(row,dict) or row.get('kind')!=ledger[slot]['kind']:
                ledger[slot].update(status='UNRESOLVED_LEDGER',attempts=1 if isinstance(row,dict) and row.get('attempts')==1 else None)
                errors['ledger/'+slot]='invalid saved slot';continue
            status,attempts=row.get('status'),row.get('attempts')
            if status not in ('NOT_ENTERED','STARTED','RETURNED','FAILED','INTERRUPTED') or type(attempts) is not int or attempts!=(0 if status=='NOT_ENTERED' else 1):
                ledger[slot].update(status='UNRESOLVED_LEDGER',attempts=1 if attempts==1 else None);errors['ledger/'+slot]='invalid status/attempt';continue
            ledger[slot]=row
    except (OSError, ValueError) as error:
        errors['ledger.json'] = str(error)
    directory = output/'receipts'
    paths = sorted(directory.rglob('*.json')) + sorted(directory.rglob('*.json.tmp'))
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        pending = relative.endswith('.tmp')
        slot = relative[:-9] if pending else relative[:-5]
        if slot not in ledger:
            errors['receipts/'+relative]='receipt outside fixed plan';continue
        previous = ledger[slot]
        receipt = None
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
            if pending:
                raise ValueError('uncommitted receipt; completion is unresolved')
            if not isinstance(receipt, dict) or receipt.get('slot') != slot:
                raise ValueError('receipt slot does not match its address')
            status, attempts = receipt.get('status'), receipt.get('attempts')
            if status not in ('NOT_ENTERED', 'STARTED', 'RETURNED', 'FAILED', 'INTERRUPTED'):
                raise ValueError('invalid receipt status')
            if type(attempts) is not int or attempts != (0 if status == 'NOT_ENTERED' else 1):
                raise ValueError('invalid receipt attempt count')
            if not isinstance(receipt.get('kind'), str) or not receipt['kind']:
                raise ValueError('receipt kind required')
            if previous.get('kind') is not None and previous['kind'] != receipt['kind']:
                raise ValueError('receipt kind disagrees with saved slot')
            if previous.get('attempts') == 1 and attempts == 0:
                raise ValueError('receipt cannot erase a spent call')
            ledger[slot] = receipt
            receipt_slots.append(slot)
        except (OSError, ValueError) as error:
            name = 'receipts/'+relative
            errors[name] = str(error)
            ledger[slot] = dict(previous, slot=slot, status='UNRESOLVED_RECEIPT',
                                attempts=1 if previous.get('attempts') == 1 or isinstance(receipt,dict) and receipt.get('attempts')==1 else None,
                                saved_status=previous.get('status'), recovery_error=name)
    counts = dict(Counter(row.get('status', 'UNRESOLVED_LEDGER') for row in ledger.values()))
    kinds = sorted({row.get('kind', 'UNKNOWN') for row in ledger.values()})
    unit_counts = {kind: dict(Counter(row.get('status', 'UNRESOLVED_LEDGER') for row in ledger.values()
                                     if row.get('kind', 'UNKNOWN') == kind)) for kind in kinds}
    return dict(ledger=ledger, counts=counts, unit_counts=unit_counts,
                receipt_slots=receipt_slots, read_errors=errors, physical_reentry=False,denominators=denominators())

def denominators():
    return dict(sources=4,cal_sources=2,eval_sources=2,arms=4,gamma_candidates=1,
        D181=4,E41=144,codec_encode=16,codec_probe=16,codec_decode=16,E41_frames=5904,
        new_Q=216,new_families=54,new_score_slots=7398,eval_DCT=72,eval_key_decodes=144,
        calibration_unique_families=108,calibration_unique_score_slots=14796,
        eval_tasks=48,aggregate_slots_each=1536,positive_slots=384,null_unique_slots=1152,
        per_gamma_positive=192,per_gamma_null=960,shared_null=768,local_CDC_slots=4608,
        path_classes_each_method=46128,quality_new=36,quality_frames=5796,quality_adjacent=5760,bundles=0)


def execution_artifacts(output):
    from pathlib import Path
    output=Path(output)
    names=('attempt.json','plan.json','ledger.json','inputs.json','background.json','thresholds.json',
           'physical_counts.json','result.json','evaluation.json','planned_outputs.json','blind_complete.json','post_truth_access.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for name in ('receipts','private','media','observations','families','blind','quality','calibration'):
        folder=output/name
        if folder.is_dir():found.extend(str(p.relative_to(output)) for p in folder.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def existing(output):
    recovery=recover_ledger(output);saved={};errors=dict(recovery['read_errors'])
    for name in ('result.json','thresholds.json','evaluation.json','planned_outputs.json','physical_counts.json'):
        try:saved[name]=read(Path(output)/name)
        except (OSError,ValueError) as error:errors[name]=str(error)
    saved['planned_outputs.json']=output_statuses(saved.get('planned_outputs.json',planned_outputs()),recovery['ledger'])
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_reentry=False,saved=saved,recovery=recovery,read_errors=errors)


def plan_units():
    units={'inputs':'input_semantics','preflight':'saved_input_preflight','controls':'saved_control_contexts','calibration/PM1':'calibrate90',
           'calibration/PM2':'calibrate90','calibration/common':'freeze_common','blind_complete':'blind_readback','post':'post_evaluation'}
    for h,role,_ in HOSTS:
        for suffix,kind in [('construct','construct_gamma2'),('native','native_audit'),('vae_load','model_load'),('release','model_release')]:units[h+'/'+suffix]=kind
        units[h+'/full/D']='D181';units[h+'/full/preclip']='clipping';units[h+'/full/quantize']='quantize'
        for scope in ('full',*FIXTURES):
            if scope!='full':units[h+'/'+scope+'/edit']='fixture151'
            for suffix,kind in [('encode','codec_encode'),('probe','codec_probe'),('decode','codec_decode')]:units[h+'/'+scope+'/'+suffix]=kind
        for fixture in FIXTURES:
            for window in WINDOWS:
                stem=f'{h}/{fixture}/{window}'
                for phase in range(4):
                    view=stem+f'/p{phase}';units[view+'/E']='E41'
                    for key in (('key01',) if role=='cal' else KEYS):units[view+'/'+key+'/Q']='Q'
                    if role=='eval':
                        units[view+'/DCT']='DCT'
                        for key in KEYS:units[view+'/'+key+'/bits']='key_bits'
                for key in (('key01',) if role=='cal' else KEYS):units[stem+'/'+key+'/score']='H137_score'
        for scope,layers in [('full',('preclip','rgb8','readback'))]+[(f,('rgb8','readback')) for f in FIXTURES]:
            for layer in layers:units[f'{h}/quality/{scope}/{layer}']='quality'
    for h,role,_ in HOSTS:
        if role!='eval':continue
        for arm in ARMS:
            for fixture in FIXTURES:
                for key in KEYS:
                    task='/'.join((h,arm,fixture,key))
                    units['blind/'+task+'/plan']='blind_plan';units['blind/'+task+'/read']='blind_read'
    return units


def planned_outputs():
    from runtime.wan.b_terminal_three_group_v1 import empty_slots
    tasks={}
    for h,role,_ in HOSTS:
        if role!='eval':continue
        for arm in ARMS:
            for fixture in FIXTURES:
                for key in KEYS:
                    task='/'.join((h,arm,fixture,key))
                    tasks[task]=dict(status='NOT_ENTERED',local={r:{w:empty_slots() for w in WINDOWS} for r in ('R0','R1','CDC')},
                        aggregates={name:empty_slots() for name in READERS},path_classes_each=961,path_aliases_each=8401)
    return dict(denominators=denominators(),tasks=tasks,quality=quality_slots(),bundles=[])


class Journal:
    def __init__(self,out):
        self.out=Path(out)
        self.data={s:dict(slot=s,kind=k,status='NOT_ENTERED',attempts=0) for s,k in plan_units().items()}
        self.physical={k:dict(planned=n,started=0,returned=0,failed=0,per_slot={}) for k,n in BUDGET.items()}
        dump(self.out/'ledger.json',self.data);dump(self.out/'physical_counts.json',self.physical)
    def call(self,slot,fn,path=None,backend=None):
        row=self.data[slot]
        if row['status']!='NOT_ENTERED':raise RuntimeError('fixed slot spent: '+slot)
        row.update(status='STARTED',attempts=1,started_unix=time.time())
        dump(self.out/'receipts'/(slot+'.json'),row);dump(self.out/'ledger.json',self.data)
        dump(self.out/'progress.json',dict(active=slot,counts=dict(Counter(r['status'] for r in self.data.values()))))
        before=dict(getattr(backend,'blocks',{}))
        try:
            value=fn()
            if path is not None:dump(path,value)
            row['status']='RETURNED';return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            after=dict(getattr(backend,'blocks',{}));row['internal_block_forward_delta']={k:after[k]-v for k,v in before.items() if type(v) is int and type(after.get(k)) is int}
            row['finished_unix']=time.time()
            dump(self.out/'receipts'/(slot+'.json'),row);dump(self.out/'ledger.json',self.data)
    def event(self,slot,kind,event):
        counts=self.physical[kind];row=counts['per_slot'].setdefault(slot,dict(started=0,returned=0,failed=0))
        if event=='started' and (row['started'] or counts['started']>=counts['planned']):raise RuntimeError('fixed physical budget spent')
        row[event]+=1;counts[event]+=1;dump(self.out/'physical_counts.json',self.physical)
    def callback(self,slot,kind):return lambda event:self.event(slot,kind,event)


def require_family(family):
    from main.tube_state.b_m3_time_path_v1 import _score_rows
    _,causes,_,_,_=_score_rows(family)
    if causes:raise ValueError('complete H137 family required: '+','.join(causes))
    return family


def load_inputs(source,out):
    cfg=read_public_protocol(source/'protocol.json')['config']
    roster=[{k:r[k] for k in ('id','role','a')} for r in cfg['sources']]
    expected=[dict(id=h,role=role,a=a) for h,role,a in HOSTS]
    if roster!=expected:raise ValueError('adopted four-source roles/origins required')
    if any(cfg['generation'][k]!=v for k,v in dict(frames=181,height=320,width=512,fps=8).items()):raise ValueError('original media geometry required')
    if cfg['writer']['key']!=KEYS['key00'] or cfg['writer']['gamma']!=1:raise ValueError('original gamma1/key semantics required')
    meta=dict(model_id=cfg['model_id'],writer_key=cfg['writer']['key'],sources=roster)
    del cfg  # Payload truth is selected only by the post evaluator.
    from runtime.wan.b_terminal_three_group_v1 import load_background
    bg=load_background(source/'background.json')
    old=read(source/'thresholds.json')
    if old.get('status')!='FROZEN' or not isinstance(old.get('tau'),(int,float)) or not 0<=old['tau']<=1:raise ValueError('original frozen tau required')
    for h,_,_ in HOSTS:
        for rel in (f'private/{h}/P0_terminal.pt',f'private/{h}/native/P0_Q.json'):
            if not (source/rel).is_file():raise FileNotFoundError(source/rel)
    dump(out/'inputs.json',meta);dump(out/'background.json',bg)
    return meta,bg,old


def cal_families(source):
    shared,pm1={},{}
    for h,role,_ in HOSTS:
        if role!='cal':continue
        for arm in ('OFF','P0','PM'):
            for fixture in FIXTURES:
                for key in KEYS:
                    if arm=='PM' and key=='key00':continue
                    for window in WINDOWS:
                        identity='/'.join((h,arm,fixture,window,key))
                        family=require_family(read(source/'families'/h/arm/fixture/window/(key+'.json')))
                        (pm1 if arm=='PM' else shared)[identity]=family
    if len(shared)!=72 or len(pm1)!=18:raise ValueError('fixed old null roster incomplete')
    return shared,pm1


class SavedBitCache:
    """Address metadata before plan; bit values are selected only after persisted plan."""
    def __init__(self):self.addresses={};self.loaded={};self.enabled=False
    def register(self,path,key,view_id):
        for row in range(1,11):
            for region in ('R0','R1'):
                address=(f'{view_id}/row{row}/{region}',key)
                if address in self.addresses and self.addresses[address]!=str(path):raise ValueError('physical evidence has contradictory files')
                self.addresses[address]=str(path)
    def __call__(self,evidence,key,bit):
        if not self.enabled:raise RuntimeError('bits cannot be consumed before plan readback')
        path=self.addresses[evidence,key]
        if path not in self.loaded:
            from runtime.wan.b_terminal_three_group_v1 import flatten_cache
            cache={};flatten_cache(read(path),cache);self.loaded[path]=cache
        return self.loaded[path][evidence][key][str(bit)]


def read_bit_header(path):
    """Read only the old writer's leading metadata; stop before regions/bit values."""
    with Path(path).open(encoding='utf-8') as stream:
        def nonspace():
            c=stream.read(1)
            while c and c.isspace():c=stream.read(1)
            if not c:raise ValueError('incomplete bit metadata')
            return c
        if nonspace()!='{':raise ValueError('bit record must be an object')
        header={}
        while True:
            if nonspace()!='"':raise ValueError('bit metadata key required')
            token='"';escaped=False
            while True:
                c=stream.read(1)
                if not c:raise ValueError('incomplete metadata key')
                token+=c
                if c=='"' and not escaped:break
                escaped=(not escaped) if c=='\\' else False
            key=json.loads(token)
            if nonspace()!=':':raise ValueError('metadata separator required')
            if key=='regions':return header
            if key not in ('status','cause_set','kind','view_id','key','local_rows'):raise ValueError('unexpected field before bit regions')
            value='';depth=0;quoted=False;escaped=False
            while True:
                c=stream.read(1)
                if not c:raise ValueError('incomplete metadata value')
                if not quoted and depth==0 and c in ',}':break
                value+=c
                if quoted:
                    if c=='"' and not escaped:quoted=False
                    escaped=(not escaped) if c=='\\' else False
                elif c=='"':quoted=True
                elif c in '[{':depth+=1
                elif c in ']}':depth-=1
            header[key]=json.loads(value)
            if c=='}':raise ValueError('bit regions missing')


def old_context(source,h,arm,fixture):
    from runtime.wan.b_terminal_three_group_v1 import view_metadata
    disk_arm='PM' if arm=='PM1' else arm
    plans={k:read(source/'blind'/h/disk_arm/fixture/k/'plan.json') for k in KEYS}
    context=dict(input_id=None,queries={},families={},cache=SavedBitCache())
    for window,w in WINDOWS.items():
        local=plans['key00']['locals'][window]
        if local['received_query_start']!=w:raise ValueError('original received window changed')
        query=local['query_id'];input_id=local['input_id']
        if not all(isinstance(v,str) and v for v in (query,input_id)):raise ValueError('nonempty anonymous IDs required')
        other=plans['key01']['locals'][window]
        if any(other.get(k)!=local[k] for k in ('query_id','input_id','received_query_start')):raise ValueError('key plans disagree on anonymous geometry')
        if context['input_id'] not in (None,input_id):raise ValueError('mixed original input identities')
        context['input_id']=input_id;views=[]
        for phase in range(4):
            view_id=None
            for key in KEYS:
                path=source/'observations'/h/disk_arm/fixture/window/f'p{phase}'/key/'bits.json'
                header=read_bit_header(path)
                if header.get('status')!='AVAILABLE' or header.get('kind')!='E41' or header.get('key')!=key or header.get('local_rows')!=list(range(1,11)):raise ValueError('saved complete E41 bit cache required')
                if not isinstance(header.get('view_id'),str) or not header['view_id']:raise ValueError('nonempty physical view_id required')
                if view_id not in (None,header['view_id']):raise ValueError('keys disagree on physical view')
                view_id=header['view_id'];context['cache'].register(path,key,view_id)
                del header
            views.append(view_metadata(input_id,query,phase,w,view_id,'RETURNED'))
        if len({v['view_id'] for v in views})!=4:raise ValueError('four distinct phase view identities required')
        context['queries'][window]=dict(query_id=query,views=views)
        for key in KEYS:
            family=require_family(read(source/'families'/h/disk_arm/fixture/window/(key+'.json')))
            if family.get('input_id',input_id)!=input_id or family.get('key_id',key)!=key:raise ValueError('saved score identity mismatch')
            family=dict(family,input_id=input_id,key_id=key)
            context['families'][window,key]=family
    return context


def load_controls(source):
    shared,pm1=cal_families(source);contexts={}
    for h,role,_ in HOSTS:
        if role=='eval':
            for arm in ('OFF','P0','PM1'):
                for fixture in FIXTURES:contexts[h,arm,fixture]=old_context(source,h,arm,fixture)
    return shared,pm1,contexts


def freeze_common(first,second,historical_tau=None):
    if any(x.get('status')!='FROZEN' or x.get('planned_families')!=90 or x.get('submitted_families')!=90 for x in (first,second)):
        raise ValueError('both complete exactly90 calibrations required')
    value=dict(status='FROZEN',tau=max(first['tau'],second['tau']),tau1=first['tau'],tau2=second['tau'],
        tau_rule=COMMON_RULE,planned_families=108,submitted_families=108,planned_score_slots=14796,
        shared_families=72,pm1_wrong=18,pm2_wrong=18,positive_used=False,payload_used=False,CFAR=False,population_FPR=None)
    if historical_tau is not None:
        value['historical_tau1']=historical_tau
        value['tau1_record']=dict(recomputed=first['tau'],historical=historical_tau,delta=first['tau']-historical_tau,equal=first['tau']==historical_tau,equality_is_gate=False)
    return value


def replan(context,key,threshold):
    from runtime.wan.b_terminal_three_group_v1 import plan_input
    plan=plan_input(context,key,threshold)
    for local in plan['locals'].values():local['tau_rule']=COMMON_RULE
    plan['common_threshold']=threshold
    return plan

def collect_pm2(host,source,out,backend,journal,background):
    from runtime.wan.b_terminal_three_group_v1 import source_frames,view_metadata
    h,role=host['id'],host['role'];base=out/'private'/h
    keys=('key01',) if role=='cal' else tuple(KEYS)
    books={key:backend.book(KEYS[key]) for key in keys}
    def tensor(path,value):backend.save_tensor(path,value);return value
    def construct():
        z,receipt,audit=backend.construct_saved(source/'private'/h/'P0_terminal.pt',source/'private'/h/'native/P0_Q.json')
        tensor(base/'PM2_terminal.pt',z);dump(base/'construction.json',receipt)
        return z,audit
    terminal,audit=journal.call(h+'/construct',construct)
    journal.call(h+'/native',lambda:audit,base/'native_audit.json')
    contexts={}
    try:
        journal.call(h+'/vae_load',backend.load_vae,base/'model.json',backend)
        media=out/'media'/h/'PM2'/'full'
        def decode():
            value=backend.decode(terminal,journal.callback(h+'/full/D','D181'))
            tensor(media/'preclip.pt',value['preclip']);tensor(media/'float_rgb.pt',value['float_rgb']);return value
        decoded=journal.call(h+'/full/D',decode,backend=backend)
        journal.call(h+'/full/preclip',lambda:backend.preclip(decoded['preclip']),media/'preclip.json')
        rgb=journal.call(h+'/full/quantize',lambda:tensor(media/'rgb8.pt',backend.quantize(decoded['float_rgb'])))
        journal.call(h+'/full/encode',lambda:backend.save_mp4(rgb,media/'video.mp4',journal.callback(h+'/full/encode','codec_encode')),media/'codec.json')
        probe=journal.call(h+'/full/probe',lambda:backend.probe(media/'video.mp4',journal.callback(h+'/full/probe','codec_probe')),media/'probe.json')
        pixels=journal.call(h+'/full/decode',lambda:tensor(media/'readback.pt',backend.read_media(media/'video.mp4',probe,181,journal.callback(h+'/full/decode','codec_decode'))))
        del decoded,rgb,terminal
        for fixture in FIXTURES:
            prefix=h+'/'+fixture;dest=out/'media'/h/'PM2'/fixture
            edited=journal.call(prefix+'/edit',lambda:tensor(dest/'rgb8.pt',backend.edit(pixels,source_frames(host,fixture))))
            journal.call(prefix+'/encode',lambda:backend.save_mp4(edited,dest/'video.mp4',journal.callback(prefix+'/encode','codec_encode')),dest/'codec.json')
            probe=journal.call(prefix+'/probe',lambda:backend.probe(dest/'video.mp4',journal.callback(prefix+'/probe','codec_probe')),dest/'probe.json')
            received=journal.call(prefix+'/decode',lambda:tensor(dest/'readback.pt',backend.read_media(dest/'video.mp4',probe,151,journal.callback(prefix+'/decode','codec_decode'))))
            del edited
            context=dict(input_id=uuid.uuid4().hex,queries={},families={},cache=SavedBitCache())
            for window,w in WINDOWS.items():
                query=uuid.uuid4().hex;views=[];banks={key:{} for key in keys}
                for phase in range(4):
                    slot=prefix+f'/{window}/p{phase}';folder=out/'observations'/h/'PM2'/fixture/window/f'p{phase}'
                    view_id=uuid.uuid4().hex
                    # Prepare before the actual encoder callback; preparation failure still spends the attempt.
                    def encode():
                        input_pixels=backend.prepare_window(received,w,phase)
                        return tensor(folder/'E41.pt',backend.encode_view(input_pixels,journal.callback(slot+'/E','E41')))
                    z=journal.call(slot+'/E',encode,backend=backend)
                    for key in keys:
                        banks[key][str(phase)]=journal.call(slot+'/'+key+'/Q',lambda key=key:backend.observe(z,books[key]),folder/key/'Q.json')
                    if role=='eval':
                        projected=journal.call(slot+'/DCT',lambda:tensor(folder/'DCT.pt',backend.payload_project(z,list(range(1,11)),('R0','R1','Rc'))))
                        for key in keys:
                            def bits(key=key):
                                value=backend.payload_decode(projected,KEYS[key],view_id,key)
                                if value.get('status')!='AVAILABLE':raise ValueError('incomplete physical bit cache')
                                return value
                            journal.call(slot+'/'+key+'/bits',bits,folder/key/'bits.json')
                            context['cache'].register(folder/key/'bits.json',key,view_id)
                        del projected
                    views.append(view_metadata(context['input_id'],query,phase,w,view_id,'RETURNED'))
                    del z
                context['queries'][window]=dict(query_id=query,views=views)
                for key in keys:
                    def score(key=key):
                        family=require_family(backend.score(banks[key],background['models'][key]))
                        family.update(input_id=context['input_id'],key_id=key);return family
                    context['families'][window,key]=journal.call(prefix+'/'+window+'/'+key+'/score',score,out/'families'/h/'PM2'/fixture/window/(key+'.json'))
            contexts[h,'PM2',fixture]=context
            del received
        del pixels
    finally:
        # Release is required even when load or a physical unit fails; it never restarts work.
        journal.call(h+'/release',backend.release_vae,base/'release.json')
    return contexts


def run(input_directory=DEFAULT_INPUT,output=DEFAULT_OUTPUT,backend=None):
    source,out=Path(input_directory).expanduser().resolve(),Path(output).expanduser().resolve()
    if execution_artifacts(out):return existing(out)
    if source==out or source.is_relative_to(out) or out.is_relative_to(source):raise ValueError('new output must be separate from original inputs')
    out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as stream:
            json.dump(dict(started_unix=time.time(),one_shot=True,failure_spends_attempt=True),stream);stream.flush()
            try:os.fsync(stream.fileno())
            except OSError as error:
                if error.errno not in (errno.EINVAL,errno.ENOSYS,errno.ENOTSUP):raise
    except FileExistsError:return existing(out)
    dump(out/'plan.json',dict(units=plan_units(),denominators=denominators(),gamma=2,new_sources=0,bundles=0))
    journal=Journal(out);outputs=planned_outputs();dump(out/'planned_outputs.json',outputs)
    threshold=dict(status='CALIBRATION_UNRESOLVED',tau=None);stop=None;post=None
    try:
        meta,bg,old_threshold=journal.call('inputs',lambda:load_inputs(source,out))
        if backend is None:
            from runtime.wan.b_gamma2_fixed_validation_backend_v1 import WanBackend
            backend=WanBackend(meta['model_id'])
        journal.call('preflight',lambda:backend.preflight(source),out/'preflight.json')
        shared,pm1,contexts=journal.call('controls',lambda:load_controls(source))
        from main.tube_state.b_m3_time_path_v1 import calibrate
        first=journal.call('calibration/PM1',lambda:calibrate(dict(shared,**pm1)),out/'calibration/PM1.json')
        if first['status']!='FROZEN':raise ValueError('old complete calibration required')
        for host in meta['sources']:
            if host['role']=='cal':contexts.update(collect_pm2(host,source,out,backend,journal,bg))
        new={}
        for h,role,_ in HOSTS:
            if role=='cal':
                for fixture in FIXTURES:
                    for window in WINDOWS:new['/'.join((h,'PM2',fixture,window,'key01'))]=contexts[h,'PM2',fixture]['families'][window,'key01']
        second=journal.call('calibration/PM2',lambda:calibrate(dict(shared,**new)),out/'calibration/PM2.json')
        threshold=journal.call('calibration/common',lambda:freeze_common(first,second,old_threshold['tau']),out/'thresholds.json')
        for host in meta['sources']:
            if host['role']=='eval':contexts.update(collect_pm2(host,source,out,backend,journal,bg))
        from runtime.wan.b_terminal_three_group_v1 import read_input
        for task in outputs['tasks']:
            h,arm,fixture,key=task.split('/');context=contexts[h,arm,fixture];folder=out/'blind'/task
            journal.call('blind/'+task+'/plan',lambda:replan(context,key,threshold),folder/'plan.json')
        # Every new blind plan is persisted and read back before any bit cache opens.
        for task in outputs['tasks']:read(out/'blind'/task/'plan.json')
        for task in outputs['tasks']:
            h,arm,fixture,key=task.split('/');context=contexts[h,arm,fixture];folder=out/'blind'/task
            frozen=read(folder/'plan.json');context['cache'].enabled=True
            decision=journal.call('blind/'+task+'/read',lambda:read_input(frozen,context['cache']),folder/'decision.json')
            outputs['tasks'][task]=dict(status='RETURNED',decision=str(folder/'decision.json'),path_classes_each=961,path_aliases_each=8401)
            dump(out/'planned_outputs.json',outputs)
        def blind_complete():
            for task in outputs['tasks']:
                read(out/'blind'/task/'plan.json');read(out/'blind'/task/'decision.json')
            return dict(status='READ_BACK',eval_tasks=48,truth_selected=False,saved_unix=time.time())
        journal.call('blind_complete',blind_complete,out/'blind_complete.json')
        from runtime.wan.b_gamma2_fixed_validation_evaluation_v1 import evaluate
        post=journal.call('post',lambda:evaluate(source,out,backend,journal),out/'evaluation.json')
    except BaseException as error:
        stop=dict(type=type(error).__name__,message=str(error))
    finally:
        if backend is not None:
            try:backend.release_vae()
            except BaseException as error:
                if stop is None:stop=dict(type=type(error).__name__,message='final release: '+str(error))
        outputs=output_statuses(outputs,journal.data)
        dump(out/'planned_outputs.json',outputs)
        result=dict(status='COMPLETED_ENGINEERING_RECORDS' if stop is None else 'STOPPED_FIXED_ATTEMPT',
            stop=stop,denominators=denominators(),physical_counts=journal.physical,
            counts=dict(Counter(row['status'] for row in journal.data.values())),threshold_status=threshold['status'],
            development_support=post.get('development_support') if post else 'UNRESOLVED',scientific_pass=None)
        dump(out/'result.json',result)
    return result


def read_public_protocol(path):
    """Select only physical scheduling fields; skip payload JSON without decoding it."""
    schema={'config':{'model_id':True,'generation':{k:True for k in ('frames','height','width','fps')},
        'sources':[{k:True for k in ('id','role','a')}],'writer':{'key':True,'gamma':True}}}
    with Path(path).open(encoding='utf-8') as stream:
        pending=['']
        def peek():
            if not pending[0]:pending[0]=stream.read(1)
            return pending[0]
        def take():
            c=peek();pending[0]=''
            if not c:raise ValueError('incomplete protocol')
            return c
        def spaces():
            while peek() and peek().isspace():take()
        def expect(c):
            spaces()
            if take()!=c:raise ValueError('invalid protocol syntax')
        def string(decode=True):
            expect('"');token='"';escaped=False
            while True:
                c=take();token+=c
                if c=='"' and not escaped:break
                escaped=(not escaped) if c=='\\' else False
            return json.loads(token) if decode else None
        def skip():
            spaces();c=peek()
            if c=='"':string(False);return
            if c in ('{','['):
                opening=take();closing='}' if opening=='{' else ']';spaces()
                if peek()==closing:take();return
                while True:
                    if opening=='{':string(False);expect(':')
                    skip();spaces();separator=take()
                    if separator==closing:return
                    if separator!=',':raise ValueError('invalid skipped protocol structure')
            else:
                token=''
                while peek() and peek() not in ',}] \r\n\t':token+=take()
                if not token:raise ValueError('missing protocol value')
                # Do not deserialize ignored values, including writer.bits.
        def value(selected):
            spaces()
            if selected is None:skip();return None
            if isinstance(selected,dict):
                expect('{');out={};spaces()
                if peek()=='}':take();return out
                while True:
                    key=string();expect(':');part=value(selected.get(key))
                    if key in selected:out[key]=part
                    spaces();separator=take()
                    if separator=='}':return out
                    if separator!=',':raise ValueError('invalid selected object')
            if isinstance(selected,list):
                expect('[');out=[];spaces()
                if peek()==']':take();return out
                while True:
                    out.append(value(selected[0]));spaces();separator=take()
                    if separator==']':return out
                    if separator!=',':raise ValueError('invalid selected array')
            if peek()=='"':return string()
            token=''
            while peek() and peek() not in ',}] \r\n\t':token+=take()
            return json.loads(token)
        return value(schema)


def quality_slots():
    rows={}
    for h,_,_ in HOSTS:
        for scope,layers in [('full',('preclip','rgb8','readback'))]+[(f,('rgb8','readback')) for f in FIXTURES]:
            n=181 if scope=='full' else 151
            for layer in layers:rows[f'{h}/quality/{scope}/{layer}']=dict(status='NOT_ENTERED',frame_denominator=n,adjacent_denominator=n-1,
                frames=[dict(frame=i,status='NOT_ENTERED') for i in range(n)],adjacent=[dict(frame_before=i,frame_after=i+1,status='NOT_ENTERED') for i in range(n-1)],quality_pass=None)
    return rows


def output_statuses(outputs,ledger):
    for task,row in outputs['tasks'].items():
        p=ledger.get('blind/'+task+'/plan',{});r=ledger.get('blind/'+task+'/read',{})
        row['plan_state']={k:p.get(k) for k in ('status','attempts','error')}
        row['read_state']={k:r.get(k) for k in ('status','attempts','error')}
        state=r if r.get('status') not in (None,'NOT_ENTERED') else p if p.get('status') not in (None,'NOT_ENTERED','RETURNED') else None
        if state is not None and state.get('status')!='RETURNED':
            status=state.get('status','UNRESOLVED');row['status']=status
            for decision in list(row.get('aggregates',{}).values())+[v for windows in row.get('local',{}).values() for v in windows.values()]:
                decision['status']=status
                for bit in decision.get('slots',[]):bit.update(status=status,cause_set=[state.get('error') or status])
    for slot,row in outputs.get('quality',{}).items():
        state=ledger.get(slot,{})
        if state.get('status') not in (None,'NOT_ENTERED'):
            row['status']=state['status'];row['attempts']=state.get('attempts');row['error']=state.get('error')
            if state['status']=='RETURNED':
                row.pop('frames',None);row.pop('adjacent',None);row['record']='quality/'+slot+'.json'
            else:
                for item in row.get('frames',[])+row.get('adjacent',[]):item['status']=state['status']
    return outputs


def compact_summary(result,output):
    saved=result.get('saved',{});base=saved.get('result.json',result)
    threshold=saved.get('thresholds.json');evaluation=saved.get('evaluation.json')
    if threshold is None:
        try:threshold=read(Path(output)/'thresholds.json')
        except (OSError,ValueError):threshold={}
    if evaluation is None:
        try:evaluation=read(Path(output)/'evaluation.json')
        except (OSError,ValueError):evaluation={}
    return dict(status=result['status'],saved_status=base.get('status'),denominators=denominators(),
        physical_counts={k:{n:v.get(n) for n in ('planned','started','returned','failed')} for k,v in base.get('physical_counts',{}).items()},
        threshold={k:threshold.get(k) for k in ('status','tau1','tau2','tau','planned_families')},
        development_support=evaluation.get('development_support'),support_checks=evaluation.get('support_checks'),
        CDC_M={k:v for k,v in evaluation.get('totals',{}).items() if k.startswith('CDC_M/')},
        paired={k:{n:v.get(n) for n in ('common_count','added_actual_supported_correct','lost_actual_supported_correct')} for k,v in evaluation.get('paired',{}).items()},
        stop=base.get('stop'),read_error_count=len(result.get('read_errors',{})),output=str(output))
