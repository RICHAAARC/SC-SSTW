"""M2 complete replay: existing MP4 inputs only, stdlib import/recovery."""
from collections import Counter
import errno,json,math,os,shutil,time,uuid
from pathlib import Path
STEM='b_rank3r_complete_repeat_v1'
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs'/f'{STEM}.json'
DEFAULT_INPUT='/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1'
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-Rank3R-Complete-Repeat-V1/fixed-four-v1'
READERS=('RANK3','RANK3R')
FIXTURES={'REPEAT_LEFT_GAP':(-8,49),'REPEAT_RIGHT_GAP':(-8,102)}
BUDGET={'codec_encode':24,'codec_probe':36,'codec_decode':36,'E41':288}
KEYS={'key00':'watermark','key01':'watermark-wrong'}
ARMS=('OFF','P0','PM')
WINDOWS={'L':0,'M':53,'R':106}
H=tuple((p,s) for p in range(4) for s in range(1 if p==0 else 2,36))
TAU=0.49221603782894896
SOURCES=[{'id': 'b-rank3-n01', 'role': 'eval', 'seed': 2026102501, 'a': 2, 'prompt': 'A fixed wide shot of a red tram moving slowly along a tree-lined avenue. Pedestrians wait beside the track and leaves move gently in the breeze.', 'negative_prompt': ''}, {'id': 'b-rank3-n02', 'role': 'eval', 'seed': 2026102502, 'a': 11, 'prompt': 'A locked-off medium-wide shot of a person folding a blue cotton cloth on a workbench. Both hands and the entire cloth remain visible throughout the shot.', 'negative_prompt': ''}, {'id': 'b-rank3-n03', 'role': 'eval', 'seed': 2026102503, 'a': 17, 'prompt': 'A fixed wide shot of a small waterfall flowing over mossy rocks into a shallow forest stream. Water ripples and nearby fern leaves move naturally.', 'negative_prompt': ''}, {'id': 'b-rank3-n04', 'role': 'eval', 'seed': 2026102504, 'a': 6, 'prompt': 'A fixed wide shot of a white wind turbine rotating slowly above a grassy meadow under moving clouds. The whole turbine and foreground grass remain in frame.', 'negative_prompt': ''}]

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

def load_background(path):
    value = read(path)
    if value.get('status') != 'BACKGROUND_FROZEN':
        raise ValueError('saved usable background required')
    for key in KEYS:
        m = value['models'][key]
        mean, factor = m['mean'], m['factor']
        if len(mean) != 40 or len(factor) != 40 or any(len(r) != 40 for r in factor):
            raise ValueError('background mean40 / lower factor40x40 required')
        if not all(type(v) in (float, int) and math.isfinite(v) for v in mean+[v for r in factor for v in r]):
            raise ValueError('nonfinite background')
        if any(factor[i][i] <= 0 or any(factor[i][j] != 0 for j in range(i+1, 40)) for i in range(40)):
            raise ValueError('positive lower triangular background factor required')
    return value

def empty_family(reason):
    return dict(status=reason, candidates=[dict(phase=p, source_start=s, S=None, score_status=reason) for p, s in H])

def empty_slots(reason='NOT_ENTERED'):
    return dict(status=reason, planned_slots=32, slots=[dict(slot=i, value=None, status=reason,
                cause_set=[reason], selected_refs=[], conflict_refs=[], missing_refs=[]) for i in range(32)])

def flatten_cache(decoded, target):
    """Per-unit production values only. Rc never enters CDC cache."""
    key = decoded['key']
    for region in ('R0', 'R1'):
        for row in decoded['regions'][region]['local_cells']:
            for bit in row['bits']:
                target.setdefault(row['evidence_id'], {}).setdefault(key, {})[str(bit['local_bit'])] = bit

def view_metadata(input_id, query_id, phase, w, view_id, status):
    return dict(input_id=input_id, query_id=query_id, view_id=view_id, phase=phase,
                received_query_start=w, encoder_input_support=list(range(w+phase, w+phase+41)),
                encoder_context=dict(reset=True, posterior_mode=True, preprocessing='uint8 / 255 once',
                                     spatial_input=[320, 512]), status=status)

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

def output_statuses(outputs,ledger):
    for task,row in outputs['tasks'].items():
        p=ledger.get('blind/'+task+'/plan',{});r=ledger.get('blind/'+task+'/read',{})
        row['plan_state']={k:p.get(k) for k in ('status','attempts','error')}
        row['read_state']={k:r.get(k) for k in ('status','attempts','error')}
        state=r if r.get('status') not in (None,'NOT_ENTERED') else p if p.get('status') not in (None,'NOT_ENTERED','RETURNED') else None
        if state is not None and state.get('status')!='RETURNED':
            status=state.get('status','UNRESOLVED');row['status']=status
            for value in list(row['readers'].values())+list(row['local_CDC'].values()):
                value['status']=status
                for bit in value['slots']:bit.update(status=status,cause_set=[state.get('error') or status])
    for slot,row in outputs['quality'].items():
        state=ledger.get(slot,{})
        if state.get('status') not in (None,'NOT_ENTERED'):
            row.update(status=state['status'],attempts=state.get('attempts'),error=state.get('error'))
            if state['status']=='RETURNED':
                row.pop('frames',None);row.pop('adjacent',None);row['record']='quality/'+slot+'.json'
            else:
                for item in row['frames']+row['adjacent']:item['status']=state['status']
    return outputs

def execution_artifacts(output):
    output=Path(output)
    names=('attempt.json','plan.json','ledger.json','inputs.json','protocol.json','background.json','thresholds.json',
           'physical_counts.json','physical_events.json','progress.json','result.json','evaluation.json','planned_outputs.json','plans_complete.json','blind_complete.json','post_truth_access.json','candidate_stop.json','repeat_start.json','repeat_start_unknown.json','old72_regression.json','historical_quality.json','storage.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for name in ('receipts','private','media','observations','families','blind','quality','source_media','models'):
        folder=output/name
        if folder.is_dir():found.extend(str(p.relative_to(output)) for p in folder.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)

def existing(output):
    recovery=recover_ledger(output);saved={};errors=dict(recovery['read_errors'])
    for name in ('result.json','evaluation.json','planned_outputs.json','physical_counts.json'):
        try:saved[name]=read(Path(output)/name)
        except (OSError,ValueError) as error:errors[name]=str(error)
    saved['planned_outputs.json']=output_statuses(saved.get('planned_outputs.json',planned_outputs()),recovery['ledger'])
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_reentry=False,saved=saved,recovery=recovery,read_errors=errors)

class Journal:
    def __init__(self,out,backend=None):
        self.out=Path(out);self.backend=backend
        self.data={s:dict(slot=s,kind=k,status='NOT_ENTERED',attempts=0) for s,k in plan_units().items()}
        self.physical={k:dict(planned=n,started=0,returned=0,failed=0) for k,n in BUDGET.items()}
        self.events={};self.save()
    def save(self):
        dump(self.out/'ledger.json',self.data);dump(self.out/'physical_counts.json',self.physical)
    def event(self,host,kind,returned):
        field='returned' if returned else 'started'
        item=self.events.setdefault(host+'/'+kind,dict(started=0,returned=0));item[field]+=1
        target='DiT_forward' if kind in ('transformer_conditional','transformer_unconditional') else kind
        if target in self.physical:
            row=self.physical[target]
            if not returned and row['started']>=row['planned']:raise RuntimeError('fixed event budget spent: '+target)
            row[field]+=1
        dump(self.out/'physical_events.json',self.events);dump(self.out/'physical_counts.json',self.physical)
    def call(self,slot,function,deps=()):
        row=self.data[slot]
        if row['status']!='NOT_ENTERED':raise RuntimeError('fixed slot spent: '+slot)
        if any(self.data[d]['status']!='RETURNED' for d in deps):raise RuntimeError('required dependency unavailable: '+slot)
        row.update(status='STARTED',attempts=1,started_unix=time.time(),dependencies=list(deps))
        dump(self.out/'receipts'/(slot+'.json'),row);dump(self.out/'physical_counts.json',self.physical)
        dump(self.out/'progress.json',dict(active=slot,counts=dict(Counter(x['status'] for x in self.data.values()))))
        count=self.physical.get(row['kind']);before=dict(getattr(self.backend,'blocks',{}))
        try:
            if count is not None:
                if count['started']>=count['planned']:raise RuntimeError('fixed call budget spent: '+row['kind'])
                count['started']+=1;dump(self.out/'physical_counts.json',self.physical)
            value=function();row['status']='RETURNED'
            if count is not None:count['returned']+=1
            return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            if count is not None:count['failed']+=1
            raise
        finally:
            row['finished_unix']=time.time();after=dict(getattr(self.backend,'blocks',{}))
            row['observed_internal_blocks']={k:after[k]-v for k,v in before.items() if type(v) is int and type(after.get(k)) is int}
            dump(self.out/'receipts'/(slot+'.json'),row);dump(self.out/'physical_counts.json',self.physical)
    def record(self,slot,function,path,deps=()):
        def op():
            value=function();dump(path,value);return value
        return self.call(slot,op,deps)

class SavedBitCache:
    def __init__(self):self.addresses={};self.loaded={};self.enabled=False;self.accesses=0
    def register(self,path,key,view):
        for row in range(1,11):
            for region in ('R0','R1'):self.addresses[(f'{view}/row{row}/{region}',key)]=Path(path)
    def __call__(self,evidence,key,bit):
        if not self.enabled:raise RuntimeError('bit consumption before every blind plan was persisted/read back')
        self.accesses+=1;path=self.addresses[evidence,key]
        if path not in self.loaded:
            values={};flatten_cache(read(path),values);self.loaded[path]=values
        return self.loaded[path][evidence][key][str(bit)]


class MethodStop(RuntimeError):
    pass

def source_frames(host,fixture):
    d,b=FIXTURES[fixture]
    return [host['a']+u+d*(u>=b) for u in range(151)]

def task_id(h,arm,fixture,key):return '/'.join((h,arm,fixture,key))

def tasks():return [task_id(h['id'],a,f,k) for h in SOURCES for a in ARMS for f in FIXTURES for k in KEYS]

def old_tasks():return [task_id(h['id'],a,f,k) for h in SOURCES for a in ARMS for f in ('CROP','DELETE_LEFT_GAP','DELETE_RIGHT_GAP') for k in KEYS]

def denominators():
    return dict(sources=4,full_source_MP4=12,received_MP4=24,trajectories=0,DiT_forward=0,local_gradient=0,state_construct=0,
        D181=0,E181=0,native_Q=0,native_DCT=0,codec_encode=24,codec_probe=36,codec_decode=36,CPU_edit=24,
        E41=288,Q=576,H_families=144,H_score_slots=19728,E41_DCT=288,E41_key_bits=576,
        codebook=8,vae_load=4,vae_release=4,tasks_per_reader=48,reader_plans=96,reader_decisions=96,
        slots_per_reader=1536,all_reader_slots=3072,positive_tasks=8,positive_slots=256,null_tasks=40,null_slots=1280,
        wrong_key_PM_tasks=8,wrong_key_PM_slots=256,domain_classes={'RANK3':961,'RANK3R':4247},
        domain_membership_checks=96,quality_records=24,quality_frames=3624,quality_adjacent=3600,historical_quality_records=76,bundles=0)

def quality_specs():return {f'{h["id"]}/{a}/{f}/quality':(h['id'],a,f) for h in SOURCES for a in ARMS for f in FIXTURES}

def quality_slots():
    return {slot:dict(status='NOT_ENTERED',frame_denominator=151,adjacent_denominator=150,
        frames=[dict(frame=i,status='NOT_ENTERED',whole=None) for i in range(151)],
        adjacent=[dict(frame_before=i,frame_after=i+1,status='NOT_ENTERED',whole=None) for i in range(150)],
        quality_pass=None,comparison='edited_uint8_vs_second_codec_readback') for slot in quality_specs()}

def plan_units():
    units=dict(inputs='input_semantics',background='background_load',threshold='old_tau_record',old72='saved_CPU_regression',
        plans_complete='plan_readback',blind_complete='blind_readback',post='post_evaluation')
    for host in SOURCES:
        h=host['id']
        for k in KEYS:units[h+'/book/'+k]='codebook'
        units[h+'/vae_load']='vae_load';units[h+'/vae_release']='vae_release'
        for a in ARMS:
            for op in ('probe','decode'):units[f'{h}/{a}/source/{op}']='codec_'+op
            for fixture in FIXTURES:
                stem=f'{h}/{a}/{fixture}'
                units[stem+'/edit']='fixture151'
                for op in ('encode','probe','decode'):units[stem+'/'+op]='codec_'+op
                units[stem+'/quality']='quality_pair'
                for win in WINDOWS:
                    for phase in range(4):
                        view=f'{stem}/{win}/p{phase}';units[view+'/E']='E41';units[view+'/DCT']='E41_DCT'
                        for k in KEYS:units[view+'/'+k+'/Q']='E41_Q';units[view+'/'+k+'/bits']='local_and_Rc_bits'
                    for k in KEYS:units[f'{stem}/{win}/{k}/score']='H137_score'
    for task in tasks():units['blind/'+task+'/plan']='blind_plan';units['blind/'+task+'/read']='blind_read'
    return units

def planned_outputs():
    return dict(denominators=denominators(),tasks={t:dict(status='NOT_ENTERED',readers={r:empty_slots() for r in READERS},local_CDC={}) for t in tasks()},quality=quality_slots(),bundles=[])

def load_inputs(source,config_path):
    from runtime.wan.b_gamma2_fixed_validation_v1 import read_public_protocol
    cfg=read(config_path)
    if cfg.get('readers')!=list(READERS) or cfg.get('fixtures')!={k:list(v) for k,v in FIXTURES.items()} or cfg.get('physical_budget')!=BUDGET:
        raise ValueError('fixed adopted M2 readers, fixtures and physical budget required')
    public=read_public_protocol(source/'protocol.json')['config']
    if public.get('sources')!=[{k:h[k] for k in ('id','role','a')} for h in SOURCES]:raise ValueError('fixed four eval sources and original a required')
    if public.get('model_id')!='Wan-AI/Wan2.1-T2V-1.3B-Diffusers' or public.get('generation')!=dict(frames=181,height=320,width=512,fps=8):raise ValueError('adopted Wan source geometry required')
    if public.get('writer')!=dict(key=KEYS['key00'],gamma=1):raise ValueError('saved gamma1/key source required')
    media=[]
    for h in SOURCES:
        for arm in ARMS:
            path=source/'media'/h['id']/arm/'full/video.mp4'
            if not path.is_file() or path.stat().st_size<=0:raise ValueError('missing or empty source MP4: '+str(path))
            with path.open('rb') as stream:
                if not stream.read(1):raise ValueError('unreadable source MP4: '+str(path))
            media.append(str(path))
    for name in ('planned_outputs.json','evaluation.json','background.json'):
        if not (source/name).is_file():raise ValueError('required saved input missing: '+name)
    return dict(status='INPUTS_READY',public_protocol=public,full_source_files=media,writer_bits_deserialized=False,
        input_directory=str(source),new_generation=False)

def historical_tau_record(path):
    """Optional historical context, never an input to either rank receiver."""
    record=dict(status='HISTORICAL_TAU_READ_ERROR',record_only=True,reference_tau=TAU,
        actual_status=None,actual_tau=None,matches_reference=None,actual_record=None,
        read_error=None,used_by_two_rank_readers=False,new_calibration=False)
    try:
        value=read(path)
        # json.loads permits NaN/Infinity and overflow by default. Do not let an
        # optional bad record turn the subsequent strict dump into a hard stop.
        json.dumps(value,allow_nan=False)
        if not isinstance(value,dict):raise ValueError('historical threshold root is not an object')
        record.update(actual_record=value,actual_status=value.get('status'),actual_tau=value.get('tau'))
        tau=value.get('tau')
        valid=type(tau) in (int,float) and math.isfinite(tau) and 0<=tau<=1
        record['matches_reference']=bool(valid and tau==TAU)
        record['status']='HISTORICAL_TAU_RECORDED' if valid else 'HISTORICAL_TAU_INVALID'
        if not valid:record['read_error']='historical tau is missing or not a finite number in[0,1]'
    except (OSError,ValueError,TypeError,OverflowError) as error:
        record['read_error']=f'{type(error).__name__}: {error}'
    return record

def historical_quality_record(saved_evaluation):
    """Keep the fixed76 descriptive rows without using availability as a gate."""
    from runtime.wan.b_rank3_four_source_confirmation_v1 import quality_specs as original_quality_specs
    specs=original_quality_specs();records={};issues=[];expected={};invalid={}
    raw=saved_evaluation.get('quality')
    read_error=None
    if not isinstance(raw,dict):
        read_error='historical quality field is missing or not a mapping: '+type(raw).__name__
        raw={}
    for slot,value in raw.items():
        try:
            if not isinstance(value,dict):raise ValueError('historical quality row is not an object')
            json.dumps(value,allow_nan=False)
            records[slot]=value
        except (ValueError,TypeError,OverflowError) as error:
            invalid[slot]=f'{type(error).__name__}: {error}'
    for slot,(_,scope,_,_) in specs.items():
        length=181 if scope=='full' else 151
        status='AVAILABLE' if slot in records else 'INVALID' if slot in invalid else 'MISSING'
        expected[slot]=dict(status=status,frame_denominator=length,adjacent_denominator=length-1,
            record_key=slot if status=='AVAILABLE' else None)
        if status!='AVAILABLE':issues.append(dict(slot=slot,status=status,error=invalid.get(slot,read_error or 'historical row absent')))
    unexpected=sorted(set(raw)-set(specs))
    for slot in unexpected:
        if slot in invalid:issues.append(dict(slot=slot,status='INVALID_EXTRA',error=invalid[slot]))
    available=sum(v['status']=='AVAILABLE' for v in expected.values())
    return dict(status='HISTORICAL_QUALITY_RECORDED' if available==76 else 'HISTORICAL_QUALITY_PARTIAL',
        record_only=True,expected_records=76,available_records=available,available_extra_records=sum(k in records for k in unexpected),
        expected_frame_denominator=sum(v['frame_denominator'] for v in expected.values()),
        expected_adjacent_denominator=sum(v['adjacent_denominator'] for v in expected.values()),
        expected_rows=expected,records=records,missing_or_invalid=issues,unexpected_keys=unexpected,read_error=read_error,
        affects_method_checks=False,new_physical_calls=0)

def old_regression(source,out):
    from main.tube_state.b_rank3r_complete_repeat_v1 import regress_saved
    saved_evaluation=read(source/'evaluation.json')
    result=regress_saved(read(source/'planned_outputs.json'),saved_evaluation,old_tasks())
    dump(out/'historical_quality.json',historical_quality_record(saved_evaluation))
    return result


def collect_host(host,source,out,backend,journal,background):
    h=host['id'];books={};contexts={}
    for key,secret in KEYS.items():books[key]=journal.call(h+'/book/'+key,lambda secret=secret:backend.book(secret),['old72'])
    try:
        journal.record(h+'/vae_load',backend.load_vae,out/'models'/h/'load.json',['old72'])
        for arm in ARMS:
            stem=f'{h}/{arm}/source';path=source/'media'/h/arm/'full/video.mp4'
            probe=journal.record(stem+'/probe',lambda:backend.probe(path),out/'source_media'/h/arm/'probe.json')
            pixels=journal.call(stem+'/decode',lambda:backend.read_media(path,probe,181),[stem+'/probe'])
            for fixture in FIXTURES:
                prefix=f'{h}/{arm}/{fixture}';dest=out/'media'/h/arm/fixture
                edited=journal.call(prefix+'/edit',lambda:backend.edit(pixels,source_frames(host,fixture)),[stem+'/decode'])
                journal.record(prefix+'/encode',lambda:backend.save_mp4(edited,dest/'video.mp4'),dest/'codec.json',[prefix+'/edit'])
                probe=journal.record(prefix+'/probe',lambda:backend.probe(dest/'video.mp4'),dest/'probe.json',[prefix+'/encode'])
                received=journal.call(prefix+'/decode',lambda:backend.read_media(dest/'video.mp4',probe,151),[prefix+'/probe'])
                journal.record(prefix+'/quality',lambda:backend.codec_quality(edited,received),out/'quality'/(prefix+'/quality.json'),[prefix+'/decode'])
                del edited
                context=dict(input_id=uuid.uuid4().hex,queries={},bit_paths={},families={})
                for window,w in WINDOWS.items():
                    query=uuid.uuid4().hex;views=[];banks={key:{} for key in KEYS}
                    for phase in range(4):
                        v=prefix+f'/{window}/p{phase}';view=uuid.uuid4().hex
                        z=journal.call(v+'/E',lambda:backend.encode_view(backend.prepare_window(received,w,phase)),[prefix+'/decode',h+'/vae_load'])
                        for key in KEYS:
                            banks[key][str(phase)]=journal.record(v+'/'+key+'/Q',lambda key=key:backend.observe(z,books[key]),out/'observations'/v/key/'Q.json',[v+'/E'])
                        projection=journal.call(v+'/DCT',lambda:backend.payload_project(z,list(range(1,11)),('R0','R1','Rc')),[v+'/E'])
                        for key,secret in KEYS.items():
                            bits_path=out/'observations'/v/key/'bits.json'
                            journal.record(v+'/'+key+'/bits',lambda key=key,secret=secret:backend.payload_decode(projection,secret,view,key),bits_path,[v+'/DCT'])
                            context['bit_paths'][view,key]=bits_path
                        del projection,z
                        views.append(view_metadata(context['input_id'],query,phase,w,view,'RETURNED'))
                    context['queries'][window]=dict(query_id=query,views=views)
                    for key in KEYS:
                        slot=f'{prefix}/{window}/{key}/score'
                        family_path=out/'families'/prefix/window/(key+'.json')
                        def score(key=key):
                            from main.tube_state.b_rank3_four_source_confirmation_v1 import check_family
                            raw=backend.score(banks[key],background['models'][key]);dump(family_path,raw)
                            if check_family(raw)['status']=='TECHNICAL_UNRESOLVED':raise ValueError('incomplete/nonnumeric H137; stop before later physical work')
                            return raw
                        family=journal.record(slot,score,family_path,['background'])
                        context['families'][window,key]=dict(family,input_id=context['input_id'],key_id=key)
                contexts[arm,fixture]=context
                del received
            del pixels
    finally:
        def release():
            backend.release_vae();return dict(status='RELEASED',new_inference_calls=0)
        journal.record(h+'/vae_release',release,out/'models'/h/'release.json')
        journal.save()
    return contexts

def make_plan(task,context):
    from main.tube_state import b_rank3_four_source_confirmation_v1 as old
    from main.tube_state import b_rank3r_complete_repeat_v1 as receiver
    key=task.split('/')[-1]
    families={w:old.check_family(context['families'][w,key]) for w in WINDOWS}
    if any(v['status']=='TECHNICAL_UNRESOLVED' for v in families.values()):raise ValueError('incomplete or nonnumeric H137 family')
    return dict(readers=receiver.plan_task(key,families,{w:context['queries'][w]['views'] for w in WINDOWS}),
        families=families,message_used=False,private_labels_used=False)

def read_plan(plan,cache):
    from main.tube_state import b_cdc32_aggregation_v1 as cdc
    result={}
    for reader in READERS:
        pack=plan['readers'][reader];memo={}
        locals_=[cdc.read_local(p,cache,memo=memo) for p in pack['locals']]
        aggregate=cdc.aggregate(locals_,path_context=pack['selection'])
        aggregate.update(mapping_scope=reader+'_SINGLETON_C',selection_status=pack['selection']['status'],detection_status=pack['selection']['status'])
        result[reader]=dict(selection=pack['selection'],aggregate=aggregate)
    return dict(status='RETURNED',readers=result,truth_accessed=False)

def _execute_once(input_directory,output,*,config_path=CONFIG,backend=None):
    out=Path(output);source=Path(input_directory)
    out.mkdir(parents=True,exist_ok=True)
    with (out/'attempt.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(status='STARTED',attempts=1,started_unix=time.time(),fixed_budget=BUDGET),stream);stream.flush()
    dump(out/'plan.json',dict(units=plan_units(),denominators=denominators(),sources=[{k:h[k] for k in ('id','role','a')} for h in SOURCES],readers=READERS,physical_budget=BUDGET))
    outputs=planned_outputs();dump(out/'planned_outputs.json',outputs)
    journal=Journal(out,backend);dump(out/'physical_events.json',{})
    cache=SavedBitCache();contexts={};stop=None
    try:
        inputs=journal.record('inputs',lambda:load_inputs(source,config_path),out/'inputs.json')
        # Raw copying does not deserialize payload truth. Post opens this JSON only after blind readback.
        shutil.copyfile(source/'protocol.json',out/'protocol.json')
        dump(out/'storage.json',dict(tensors='transient memory / /content',durable='JSON and24 final MP4',missing_inputs='never regenerated'))
        background=journal.record('background',lambda:load_background(source/'background.json'),out/'background.json')
        journal.record('threshold',lambda:historical_tau_record(source/'thresholds.json'),out/'thresholds.json')
        regression=journal.record('old72',lambda:old_regression(source,out),out/'old72_regression.json')
        if regression['status']!='OLD72_REGRESSION_PASSED':raise MethodStop('expanded domain fails fixed old72 regression; no physical calls')
        if backend is None:
            from runtime.wan.b_rank3r_complete_repeat_backend_v1 import WanBackend
            backend=WanBackend(model_id=inputs['public_protocol']['model_id'])
        journal.backend=backend
        for host in SOURCES:
            packs=collect_host(host,source,out,backend,journal,background)
            for (arm,fixture),context in packs.items():
                contexts[host['id'],arm,fixture]=context
                for (view,key),path in context['bit_paths'].items():cache.register(path,key,view)
        for task in tasks():
            h,arm,fixture,key=task.split('/')
            value=journal.record('blind/'+task+'/plan',lambda task=task:make_plan(task,contexts[h,arm,fixture]),out/'blind'/task/'plan.json')
            if not (arm=='PM' and key=='key00'):
                detected=[r for r in READERS if value['readers'][r]['selection']['detected'] is True]
                if detected:dump(out/'candidate_stop.json',dict(status='METHOD_NOT_SUPPORTED',cause='NULL_DETECT',task=task,readers=detected,no_further_physical_calls=True,fixed_saved_diagnostics_only=True))
        def plans_back():
            for task in tasks():read(out/'blind'/task/'plan.json')
            return dict(status='READ_BACK',task_plans=48,reader_plans=96,payload_values_read=False)
        journal.record('plans_complete',plans_back,out/'plans_complete.json');cache.enabled=True
        for task in tasks():
            value=journal.record('blind/'+task+'/read',lambda task=task:read_plan(read(out/'blind'/task/'plan.json'),cache),out/'blind'/task/'decision.json')
            outputs['tasks'][task]=dict(status='RETURNED',readers={r:v['aggregate'] for r,v in value['readers'].items()},local_CDC={})
        def decisions_back():
            for task in tasks():read(out/'blind'/task/'decision.json')
            return dict(status='READ_BACK',task_decisions=48,reader_decisions=96,truth_accessed=False,payload_files_loaded=len(cache.loaded),cache_accesses=cache.accesses)
        journal.record('blind_complete',decisions_back,out/'blind_complete.json')
        from runtime.wan.b_rank3r_complete_repeat_evaluation_v1 import evaluate
        journal.record('post',lambda:evaluate(out,journal),out/'evaluation.json')
    except BaseException as error:
        stop=dict(status='METHOD_NOT_SUPPORTED' if isinstance(error,MethodStop) else 'TECHNICAL_UNRESOLVED',
            type=type(error).__name__,message=str(error),retry=False,interrupted=not isinstance(error,Exception))
    finally:
        output_statuses(outputs,journal.data);dump(out/'planned_outputs.json',outputs);journal.save()
    complete=all(v['status']=='RETURNED' for v in journal.data.values())
    result=dict(status='COMPLETED_FIXED_ATTEMPT' if complete else stop['status'] if stop else 'TECHNICAL_UNRESOLVED',
        denominators=denominators(),physical_counts=journal.physical,ledger_counts=dict(Counter(r['status'] for r in journal.data.values())),
        stop=stop,output=str(out),physical_reentry=False)
    dump(out/'result.json',result);return result

def run(input_directory=DEFAULT_INPUT,output=DEFAULT_OUTPUT,*,config_path=CONFIG,backend=None):
    if execution_artifacts(output):return existing(output)
    return _execute_once(input_directory,output,config_path=config_path,backend=backend)

def _start_reserved(input_directory=DEFAULT_INPUT,output=DEFAULT_OUTPUT,*,config_path=CONFIG,backend=None):
    """Private live Notebook handoff after its new cloud start receipt readback.

    Not a CLI retry mode: ordinary CLI/Run all treats any existing receipt as
    spent. This live branch still refuses every runtime trace and exclusively
    creates attempt.json; it cannot resume or fill a partially started attempt.
    """
    traces=[name for name in execution_artifacts(output) if name=='repeat_start_unknown.json' or name not in ('repeat_start.json',)]
    if traces:return existing(output)
    return _execute_once(input_directory,output,config_path=config_path,backend=backend)


def compact_summary(result,output):
    base=result.get('saved',{}).get('result.json',result)
    try:ev=read(Path(output)/'evaluation.json')
    except (OSError,ValueError):ev={}
    return dict(status=result['status'],saved_status=base.get('status'),denominators=denominators(),physical_counts=base.get('physical_counts'),
        method_status=ev.get('method_status'),checks=ev.get('checks'),per_source=ev.get('per_source'),
        paired_gain=ev.get('paired_gain'),stop=base.get('stop'),output=str(output))
