"""M3 fixed four-new-source, five-fixture confirmation. Stdlib recovery."""
from collections import Counter
import errno,json,math,os,tempfile,time,uuid
from pathlib import Path
STEM='b_rank3r_new_source_confirmation_v1'
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs'/f'{STEM}.json'
DEFAULT_INPUT='/content/drive/MyDrive/Video-WM/B-Rank3-Four-Source-Confirmation-V1/repeat-20261025-v1'
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-Rank3R-New-Source-Confirmation-V1/fixed-four-five-v1'
KEYS={'key00':'watermark','key01':'watermark-wrong'}
ARMS=('OFF','P0','PM')
WINDOWS={'L':0,'M':53,'R':106}
FIXTURES={'CROP':(0,None),'DELETE_LEFT_GAP':(8,49),'DELETE_RIGHT_GAP':(8,102),'REPEAT_LEFT_GAP':(-8,49),'REPEAT_RIGHT_GAP':(-8,102)}
READERS=('RANK3R','RANK3','CDC_M')
H=tuple((p,s) for p in range(4) for s in range(1 if p==0 else 2,36))
TAU=0.49221603782894896
BUDGET={'native50_trajectory':8,'native_step':400,'DiT_forward':800,'payload_gradient':100,'state_construct':4,'D181':12,'codec_encode':72,'codec_probe':72,'codec_decode':72,'E41':720}
SOURCES=[{'id': 'b-rank3r-m01', 'seed': 2026102601, 'a': 2, 'category': '室内倒茶、人与物、液体', 'prompt': 'A fixed medium shot of a person slowly pouring tea from a white teapot into a glass cup on a wooden kitchen table. Steam rises and the whole table remains visible.', 'negative_prompt': '', 'role': 'eval'}, {'id': 'b-rank3r-m02', 'seed': 2026102602, 'a': 11, 'category': '夜景、低照度、高对比', 'prompt': 'A fixed wide shot of a quiet city intersection at night with traffic lights changing and a few cars passing slowly. Street lamps and wet pavement reflections stay in frame.', 'negative_prompt': '', 'role': 'eval'}, {'id': 'b-rank3r-m03', 'seed': 2026102603, 'a': 17, 'category': '动物、自然纹理', 'prompt': 'A fixed wide shot of a brown dog walking slowly along a sandy beach while small waves roll onto the shore. The dog and the waterline remain fully visible.', 'negative_prompt': '', 'role': 'eval'}, {'id': 'b-rank3r-m04', 'seed': 2026102604, 'a': 6, 'category': '室内书架、缓慢水平摄像机平移', 'prompt': 'A slow steady horizontal pan across a bookshelf filled with colorful books and a small potted plant in a sunlit room. Lighting stays constant throughout the shot.', 'negative_prompt': '', 'role': 'eval'}]
METHOD_CRITERIA={'overall': {'positive_detect_min_per_source': 4, 'positive_tasks_per_source': 5, 'implied_total_detect_min': 16, 'supported_correct_min': 480, 'positive_slots': 640, 'supported_correct_min_per_source': 80, 'slots_per_source': 160}, 'repeat': {'positive_detect_min': 6, 'positive_tasks': 8, 'positive_detect_min_per_source': 1, 'tasks_per_source': 2, 'supported_correct_min': 192, 'positive_slots': 256, 'supported_correct_min_per_source': 32, 'slots_per_source': 64}, 'crop_delete': {'positive_detect_min': 9, 'positive_tasks': 12, 'positive_detect_min_per_source': 2, 'tasks_per_source': 3, 'supported_correct_min': 288, 'positive_slots': 384, 'supported_correct_min_per_source': 48, 'slots_per_source': 96}, 'main_reader_null_detect': 0, 'main_reader_null_outputs': 0, 'main_reader_literal_errors': 0, 'main_reader_false_attribution': 0, 'main_reader_unsupported_outputs': 0, 'actual_supported_correct_losses_vs_CDC_M': 0, 'actual_supported_correct_losses_vs_RANK3': 0, 'required_observations_complete': True}
CONTRIBUTION_CRITERIA={'prerequisite': 'all method criteria met', 'net_actual_supported_correct_gain_over_CDC_M_min': 32, 'repeat_paired_gain_over_RANK3': 'report separately; no extra numeric threshold beyond adopted method criteria', 'common_BER': 'report common emitted-slot numerator and denominator; zero denominator means undefined, not zero BER'}
CLAIM_CEILING='Newly registered fixed four-source confirmation of frozen gamma1 RANK3R for 151-frame crop, one between-window delete8 or complete replay8. n and m groups remain separate; not IID, population FPR, all r1..53 tested, arbitrary repeat/speed, exact seam, source authentication, multiple-key security or perceptual quality PASS.'
DENOMINATORS={'sources': 4, 'eval_sources': 4, 'trajectories': 8, 'native_steps': 400, 'DiT_forward': 800, 'local_gradient': 100, 'state_construct': 4, 'D181': 12, 'full_MP4': 12, 'fixture_MP4': 60, 'codec_encode': 72, 'codec_probe': 72, 'codec_decode': 72, 'CPU_edit': 60, 'E41': 720, 'Q': 1440, 'H_families': 360, 'H_score_slots': 49320, 'E41_DCT': 720, 'E41_key_bits': 1440, 'native_key00_Q': 8, 'native_DCT': 8, 'native_key_bits': 16, 'native_P0_PM_bit_pairs': 2816, 'media_P0_PM_bit_pairs': 9600, 'codebooks': 8, 'readers': 3, 'tasks_per_reader': 120, 'reader_plans': 360, 'reader_decisions': 360, 'slots_per_reader': 3840, 'all_reader_slots': 11520, 'positive_tasks': 20, 'positive_slots': 640, 'null_tasks': 100, 'null_slots': 3200, 'wrong_key_PM_tasks': 20, 'wrong_key_PM_slots': 640, 'local_CDC_slots': 11520, 'quality_records': 108, 'quality_frames': 17148, 'quality_adjacent': 17040, 'OFF_RGB24_files': 4, 'OFF_RGB24_bytes_each': 88965120, 'OFF_RGB24_bytes_total': 355860480, 'bundles': 0}

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

def source_frames(host, fixture):
    d, seam = FIXTURES[fixture]
    return [host['a']+x+(d if seam is not None and x >= seam else 0) for x in range(151)]

def task_id(h, arm, fixture, key):
    return '/'.join((h, arm, fixture, key))

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

def tasks():
    return [task_id(h['id'],a,f,k) for h in SOURCES for a in ARMS for f in FIXTURES for k in KEYS]

def quality_specs():
    records={}
    for host in SOURCES:
        h=host['id']
        for pair in ('OFF_to_P0','P0_to_PM'):
            for scope,layers in [('full',('preclip','rgb8','readback'))]+[(f,('rgb8','readback')) for f in FIXTURES]:
                for layer in layers:records[f'{h}/quality/{scope}/{pair}/{layer}']=(h,scope,pair,layer)
        records[f'{h}/quality/full/OFF_to_PM/readback']=(h,'full','OFF_to_PM','readback')
    return records

def plan_units():
    units={'inputs':'input_semantics','background':'background_load','threshold':'old_frozen_tau',
           'plans_complete':'plan_readback','blind_complete':'blind_readback','post':'post_evaluation'}
    for host in SOURCES:
        h=host['id']
        for k in KEYS:units[h+'/book/'+k]='codebook'
        units[h+'/generation_load']='generation_load';units[h+'/generation_release']='generation_release';units[h+'/vae_load']='vae_load';units[h+'/vae_release']='vae_release'
        for a in ('OFF','P0'):units[h+'/generate/'+a]='native50_trajectory'
        for a in ('P0','PM'):
            units[f'{h}/native/{a}/Q']='native_key00_Q'
            units[f'{h}/native/{a}/DCT']='native_DCT'
            for k in KEYS:units[f'{h}/native/{a}/{k}/bits']='native_local_bits'
        units[h+'/construct']='state_construct';units[h+'/native_audit']='native_state_audit'
        for a in ARMS:
            for op,kind in [('D','D181'),('preclip','preclip'),('quantize','quantize'),('encode','codec_encode'),('probe','codec_probe'),('decode','codec_decode')]:
                units[f'{h}/{a}/full/{op}']=kind
            for f in FIXTURES:
                stem=f'{h}/{a}/{f}'
                for op,kind in [('edit','fixture151'),('encode','codec_encode'),('probe','codec_probe'),('decode','codec_decode')]:units[stem+'/'+op]=kind
                for w in WINDOWS:
                    for phase in range(4):
                        view=f'{stem}/{w}/p{phase}';units[view+'/E']='E41';units[view+'/DCT']='E41_DCT'
                        for k in KEYS:
                            units[view+'/'+k+'/Q']='E41_Q';units[view+'/'+k+'/bits']='local_and_Rc_bits'
                    for k in KEYS:units[f'{stem}/{w}/{k}/score']='H137_score'
    for task in tasks():
        units['blind/'+task+'/plan']='blind_plan';units['blind/'+task+'/read']='blind_read'
    for slot in quality_specs():units[slot]='quality_pair'
    return units

def quality_slots():
    rows={}
    for slot,(h,scope,pair,layer) in quality_specs().items():
        host=next(x for x in SOURCES if x['id']==h)
        mapping=list(range(181)) if scope=='full' else source_frames(host,scope)
        rows[slot]=dict(status='NOT_ENTERED',frame_denominator=len(mapping),adjacent_denominator=len(mapping)-1,
            frames=[dict(frame=t,source_frame=f,status='NOT_ENTERED',whole=None) for t,f in enumerate(mapping)],
            adjacent=[dict(frame_before=t,frame_after=t+1,status='NOT_ENTERED',whole=None) for t in range(len(mapping)-1)],quality_pass=None)
    return rows

def planned_outputs():
    return dict(denominators=denominators(),tasks={t:dict(status='NOT_ENTERED',
        readers={r:empty_slots() for r in READERS},local_CDC={w:empty_slots() for w in WINDOWS},
        domain_classes={'RANK3R':4247,'RANK3':961},old_three_window_geometry_label='M3_not_confirmation_stage') for t in tasks()},quality=quality_slots(),bundles=[])

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
           'physical_counts.json','physical_events.json','progress.json','result.json','evaluation.json','planned_outputs.json','plans_complete.json','blind_complete.json','post_truth_access.json','candidate_stop.json','repeat_start.json','repeat_start_unknown.json','source_protocol.json','source_thresholds.json','off_rgb24_archives.json','storage.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for name in ('receipts','private','media','observations','families','blind','quality','source_media','models','m4_inputs'):
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
            reset=row['kind']=='vae_load' or any(type(v) is int and type(after.get(k)) is int and after[k]<v for k,v in before.items())
            row['internal_counter_record']=dict(before=before,after=after,reset=reset,
                delta_scope='not_comparable_across_reset' if reset else 'within_call')
            row['observed_internal_blocks']={} if reset else {k:after[k]-v for k,v in before.items() if type(v) is int and type(after.get(k)) is int}
            dump(self.out/'receipts'/(slot+'.json'),row);dump(self.out/'physical_counts.json',self.physical)
    def record(self,slot,function,path,deps=()):
        def op():
            value=function();dump(path,value);return value
        return self.call(slot,op,deps)

def _start_reserved(input_directory=DEFAULT_INPUT,output=DEFAULT_OUTPUT,*,config_path=CONFIG,backend=None):
    """Private live Notebook handoff after its new cloud start receipt readback.

    Not a CLI retry mode: ordinary CLI/Run all treats any existing receipt as
    spent. This live branch still refuses every runtime trace and exclusively
    creates attempt.json; it cannot resume or fill a partially started attempt.
    """
    traces=[name for name in execution_artifacts(output) if name=='repeat_start_unknown.json' or name not in ('repeat_start.json',)]
    if traces:return existing(output)
    return _execute_once(input_directory,output,config_path=config_path,backend=backend)

def run(input_directory=DEFAULT_INPUT,output=DEFAULT_OUTPUT,*,config_path=CONFIG,backend=None):
    if execution_artifacts(output):return existing(output)
    return _execute_once(input_directory,output,config_path=config_path,backend=backend)

def denominators():return dict(DENOMINATORS)


def load_config(path=CONFIG):
    cfg=read(path)
    if cfg.get('sources')!=SOURCES:raise ValueError('adopted four m-source prompt/seed/a/eval roster required')
    if cfg.get('model_id')!='Wan-AI/Wan2.1-T2V-1.3B-Diffusers':raise ValueError('adopted Wan model required')
    if cfg.get('generation')!=dict(height=320,width=512,frames=181,fps=8,steps=50,guidance_scale=5,max_sequence_length=512):raise ValueError('fixed native50 geometry required')
    bits=[int(b) for byte in bytes.fromhex('A6D39C5E') for b in f'{byte:08b}']
    if cfg.get('writer')!=dict(key=KEYS['key00'],gamma=1,bits=bits):raise ValueError('adopted gamma1/key/32-bit writer required')
    if cfg.get('fixtures')!={k:dict(d=d,b=b) for k,(d,b) in FIXTURES.items()} or cfg.get('readers')!=list(READERS):raise ValueError('fixed five fixtures and three readers required')
    if cfg.get('tau')!=dict(CDC_M=TAU,RANK3=None,RANK3R=None,new_calibration=False):raise ValueError('CDC fixed tau and tau-free ranks required')
    if cfg.get('method_criteria')!=METHOD_CRITERIA or cfg.get('contribution_criteria')!=CONTRIBUTION_CRITERIA or cfg.get('budget')!=DENOMINATORS:raise ValueError('adopted M3 criteria and budget required')
    return cfg

def archive_slots():
    return {h['id']:dict(status='NOT_ENTERED',source=h['id'],arm='OFF',shape=[181,320,512,3],dtype='uint8',
        order='THWC C-contiguous RGB',fps=8,expected_bytes=88965120,method_required=False,
        path=f'm4_inputs/{h["id"]}/OFF/precodec.rgb24',sidecar=f'm4_inputs/{h["id"]}/OFF/precodec.json') for h in SOURCES}

def archive_off_rgb24(backend,rgb,out,source):
    """One optional byte archive of the already quantized OFF; no media/model work."""
    record=archive_slots()[source];path=Path(out)/record['path'];sidecar=Path(out)/record['sidecar']
    try:
        value=backend.validate_pixels(rgb,181).detach().cpu().contiguous().numpy()
        if tuple(value.shape)!=(181,320,512,3) or str(value.dtype)!='uint8' or not value.flags.c_contiguous:raise ValueError('actual contiguous uint8 THWC required for RGB24 archive')
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as stream:
            for frame in value:stream.write(frame.tobytes(order='C'))
            stream.flush()
        size=path.stat().st_size
        if size!=record['expected_bytes']:raise IOError('RGB24 archive byte count incomplete')
        record.update(status='ARCHIVED',actual_bytes=size,point='after quantization, before full H264; not a fixture/receiver input')
    except Exception as error:
        record.update(status='ARCHIVE_MISSING',error=f'{type(error).__name__}: {error}',retry=False)
    try:dump(sidecar,record)
    except Exception as error:
        record.update(status='ARCHIVE_MISSING',sidecar_error=f'{type(error).__name__}: {error}',retry=False)
    return record

def source_records(source,out):
    """Optional historical context: neither read nor local record write is a gate."""
    records={}
    for name in ('source_protocol.json','source_thresholds.json'):
        input_name='protocol.json' if name=='source_protocol.json' else 'thresholds.json'
        record=dict(role='HISTORICAL_SOURCE_ONLY',source_path=str(source/input_name),record_only=True,
            method_required=False,status='HISTORICAL_READ_ERROR',read_error=None,write_error=None)
        try:
            if name=='source_protocol.json':
                from runtime.wan.b_gamma2_fixed_validation_v1 import read_public_protocol
                value=read_public_protocol(source/input_name);json.dumps(value,allow_nan=False)
                record.update(status='HISTORICAL_RECORDED',public_fields=value)
            else:
                from runtime.wan.b_rank3r_complete_repeat_v1 import historical_tau_record
                value=historical_tau_record(source/input_name);json.dumps(value,allow_nan=False)
                record.update(value)
        except Exception as error:
            record['read_error']=f'{type(error).__name__}: {error}'
        try:dump(out/name,record)
        except Exception as error:record['write_error']=f'{type(error).__name__}: {error}'
        records[name]=record
    return dict(status='RECORD_ONLY',records=records,affects_method=False,
        errors={name:{key:row[key] for key in ('read_error','write_error') if row.get(key)}
                for name,row in records.items() if row.get('read_error') or row.get('write_error')})

def collect_host(host, cfg, out, backend, journal, background, tensor_root, archives):
    h = host['id']
    evaluation = host['role'] == 'eval'
    base = out/'private'/h
    books, terminals, contexts = {}, {}, {}
    def tensor(path, value):
        backend.save_tensor(tensor_root/path.relative_to(out), value)
        return value
    for key, secret in KEYS.items():
        books[key] = journal.call(f'{h}/book/{key}', lambda secret=secret: backend.book(secret), ['background'])
    try:
        journal.record(h+'/generation_load', lambda: backend.generation_load(host, cfg), base/'generation_load.json', ['background'])
        for arm in ('OFF', 'P0'):
            def generate(arm=arm):
                traces = []
                def record(row):
                    traces.append(row)
                    dump(base/(arm+'_trace.json'), traces)
                z, receipt = backend.generate(arm, cfg['writer'], lambda k, done: journal.event(h, k, done), record)
                dump(base/(arm+'_generation.json'), receipt)
                return tensor(base/(arm+'_terminal.pt'), z)
            terminals[arm] = journal.call(f'{h}/generate/{arm}', generate, [h+'/generation_load'])
    finally:
        def release_generation():
            backend.release_generation();return dict(status='RELEASED')
        journal.record(h+'/generation_release',release_generation,base/'generation_release.json')
    pre = journal.record(h+'/native/P0/Q', lambda: backend.native_observe(terminals['P0'], books['key00']),
                         base/'native/P0_Q.json', [h+'/generate/P0', h+'/book/key00'])
    def construct():
        z, receipt = backend.construct(terminals['P0'], books['key00'], pre)
        dump(base/'construction.json', receipt)
        return tensor(base/'PM_terminal.pt', z)
    terminals['PM'] = journal.call(h+'/construct', construct, [h+'/native/P0/Q'])
    post = journal.record(h+'/native/PM/Q', lambda: backend.native_observe(terminals['PM'], books['key00']),
                          base/'native/PM_Q.json', [h+'/construct', h+'/book/key00'])
    journal.record(h+'/native_audit', lambda: backend.native_audit(terminals['P0'], terminals['PM'], post),
                   base/'native/audit.json', [h+'/native/PM/Q'])
    try:
        journal.record(h+'/vae_load', backend.load_vae, base/'vae.json', [h+'/generation_load'])
        for arm in ARMS:
            stem = f'{h}/{arm}/full'
            media = out/'media'/h/arm/'full'
            def decode(arm=arm):
                value = backend.decode(terminals[arm])
                tensor(media/'preclip.pt', value['preclip'])
                tensor(media/'float_rgb.pt', value['float_rgb'])
                return value
            decoded = journal.call(stem+'/D', decode, [h+'/vae_load', h+'/construct' if arm == 'PM' else h+'/generate/'+arm])
            journal.record(stem+'/preclip', lambda: backend.preclip(decoded['preclip']), media/'preclip.json', [stem+'/D'])
            rgb = journal.call(stem+'/quantize', lambda: tensor(media/'rgb8.pt', backend.quantize(decoded['float_rgb'])), [stem+'/D'])
            if arm=='OFF':
                archives[h]=archive_off_rgb24(backend,rgb,out,h)
                try:dump(out/'off_rgb24_archives.json',archives)
                except Exception as error:archives[h]['index_write_error']=f'{type(error).__name__}: {error}'
            journal.record(stem+'/encode', lambda: backend.save_mp4(rgb, media/'video.mp4'), media/'codec.json', [stem+'/quantize'])
            probe = journal.record(stem+'/probe', lambda: backend.probe(media/'video.mp4'), media/'probe.json', [stem+'/encode'])
            pixels = journal.call(stem+'/decode', lambda: tensor(media/'readback.pt', backend.read_media(media/'video.mp4', probe, 181)), [stem+'/probe'])
            del decoded, rgb
            for fixture in FIXTURES:
                prefix = f'{h}/{arm}/{fixture}'
                dest = out/'media'/h/arm/fixture
                edited = journal.call(prefix+'/edit', lambda: tensor(dest/'rgb8.pt', backend.edit(pixels, source_frames(host, fixture))), [stem+'/decode'])
                journal.record(prefix+'/encode', lambda: backend.save_mp4(edited, dest/'video.mp4'), dest/'codec.json', [prefix+'/edit'])
                probe = journal.record(prefix+'/probe', lambda: backend.probe(dest/'video.mp4'), dest/'probe.json', [prefix+'/encode'])
                received = journal.call(prefix+'/decode', lambda: tensor(dest/'readback.pt', backend.read_media(dest/'video.mp4', probe, 151)), [prefix+'/probe'])
                del edited
                context = dict(input_id=uuid.uuid4().hex, queries={}, bit_paths={}, families={})
                for window, w in WINDOWS.items():
                    query_id = uuid.uuid4().hex
                    views, banks = [], {key: {} for key in KEYS}
                    for phase in range(4):
                        v = prefix+f'/{window}/p{phase}'
                        view_id = uuid.uuid4().hex
                        z = journal.call(v+'/E', lambda: tensor(out/'observations'/v/'E41.pt', backend.encode_view(backend.prepare_window(received, w, phase))), [prefix+'/decode', h+'/vae_load'])
                        for key in KEYS:
                            banks[key][str(phase)] = journal.record(v+'/'+key+'/Q', lambda key=key: backend.observe(z, books[key]), out/'observations'/v/key/'Q.json', [v+'/E', h+'/book/'+key])
                        view_status = journal.data[v+'/E']['status']
                        if evaluation:
                            projected = journal.call(v+'/DCT', lambda: tensor(out/'observations'/v/'DCT.pt', backend.payload_project(z, list(range(1, 11)), ('R0', 'R1', 'Rc'))), [v+'/E'])
                            if journal.data[v+'/DCT']['status'] != 'RETURNED':
                                view_status = journal.data[v+'/DCT']['status']
                            for key, secret in KEYS.items():
                                decoded_bits = journal.record(v+'/'+key+'/bits', lambda key=key, secret=secret: backend.payload_decode(projected, secret, view_id, key), out/'observations'/v/key/'bits.json', [v+'/DCT'])
                                context['bit_paths'][view_id, key] = out/'observations'/v/key/'bits.json'
                            del projected
                        views.append(view_metadata(context['input_id'], query_id, phase, w, view_id, view_status))
                        del z
                    context['queries'][window] = dict(query_id=query_id, views=views)
                    for key in KEYS:
                        score_slot = f'{prefix}/{window}/{key}/score'
                        family_path=out/'families'/prefix/window/(key+'.json')
                        def score(key=key):
                            from main.tube_state.b_rank3_four_source_confirmation_v1 import check_family
                            raw=backend.score(banks[key],background['models'][key]);dump(family_path,raw)
                            if check_family(raw)['status']=='TECHNICAL_UNRESOLVED':raise ValueError('incomplete/nonfinite H137; later physical work stopped')
                            return raw
                        family=journal.record(score_slot,score,family_path,['background'])
                        family.update(input_id=context['input_id'], key_id=key)
                        context['families'][window, key] = family
                contexts[arm, fixture] = context
                del received
            del pixels
    finally:
        def release_vae():
            backend.release_vae();return dict(status='RELEASED')
        journal.record(h+'/vae_release',release_vae,base/'vae_release.json')
        journal.save()
    return contexts

def make_plan(key_id,context):
    from main.tube_state import b_rank3_four_source_confirmation_v1 as original
    from main.tube_state import b_rank3r_new_source_confirmation_v1 as receiver
    families={w:original.check_family(context['families'][w,key_id]) for w in WINDOWS}
    if any(v['issues'] for v in families.values()):raise ValueError('complete finite H137 required')
    views={w:context['queries'][w]['views'] for w in WINDOWS}
    return dict(readers=receiver.plan_task(key_id,families,views),families=families,message_used=False,private_labels_used=False,
        domain_size_record_only={'RANK3R':4247,'RANK3':961},geometry_diagnostic_label='old three-window M3, not confirmation stage3')

def read_plan(plan,cache):
    from main.tube_state import b_cdc32_aggregation_v1 as cdc
    result={};local_cdc={}
    for reader in READERS:
        pack=plan['readers'][reader];memo={}
        locals_=[cdc.read_local(p,cache,memo=memo) for p in pack['locals']]
        aggregate=cdc.aggregate(locals_,path_context=pack['selection'])
        aggregate.update(mapping_scope='LOCAL_FULL_C' if reader=='CDC_M' else reader+'_SINGLETON_C',selection_status=pack['selection']['status'])
        if reader!='CDC_M':aggregate['detection_status']=pack['selection']['status']
        result[reader]=dict(selection=pack['selection'],aggregate=aggregate)
        if reader=='CDC_M':local_cdc={p['window']:value for p,value in zip(pack['locals'],locals_)}
    return dict(status='RETURNED',readers=result,local_CDC=local_cdc,truth_accessed=False)

def _execute_once(input_directory,output,*,config_path=CONFIG,backend=None):
    out=Path(output);source=Path(input_directory);out.mkdir(parents=True,exist_ok=True)
    with (out/'attempt.json').open('x',encoding='utf-8') as stream:
        json.dump(dict(status='STARTED',attempts=1,started_unix=time.time(),fixed_budget=BUDGET),stream);stream.flush()
    dump(out/'plan.json',dict(units=plan_units(),denominators=denominators(),sources=SOURCES,readers=READERS,physical_budget=BUDGET))
    outputs=planned_outputs();dump(out/'planned_outputs.json',outputs)
    journal=Journal(out,backend);dump(out/'physical_events.json',{})
    archives=archive_slots();contexts={};cache=SavedBitCache();scratch=None;stop=None
    historical_records=dict(status='NOT_ENTERED',affects_method=False)
    try:
        scratch=tempfile.TemporaryDirectory(prefix='video-wm-rank3r-m3-',dir='/content' if Path('/content').is_dir() else None)
        tensor_root=Path(scratch.name)
        dump(out/'storage.json',dict(tensors=str(tensor_root),tensor_lifetime='current_attempt_only',
            durable='necessary JSON,72 MP4,4 optional OFF RGB24 and sidecars',archives_required_for_M3=False))
        cfg=journal.record('inputs',lambda:load_config(config_path),out/'inputs.json')
        dump(out/'protocol.json',dict(stage='M3_NEW_SOURCE_FIVE_FIXTURE_CONFIRMATION',config=cfg,
            receiver_truth_access=False,source_protocol='historical metadata is separately source_protocol.json'))
        background=journal.record('background',lambda:load_background(source/'background.json'),out/'background.json')
        historical_records=source_records(source,out)
        journal.record('threshold',lambda:dict(status='FROZEN',tau=TAU,used_by=['CDC_M'],rank_readers_use_tau=False,new_calibration=False,
            origin='adopted current M3 constant; source_thresholds is optional historical record'),out/'thresholds.json')
        if backend is None:
            from runtime.wan.b_rank3_four_source_confirmation_backend_v1 import WanBackend
            backend=WanBackend(model_id=cfg['model_id'])
        journal.backend=backend
        for host in SOURCES:
            packs=collect_host(host,cfg,out,backend,journal,background,tensor_root,archives)
            for (arm,fixture),context in packs.items():
                contexts[host['id'],arm,fixture]=context
                for (view,key),path in context['bit_paths'].items():cache.register(path,key,view)
        for task in tasks():
            h,arm,fixture,key=task.split('/')
            pack=journal.record('blind/'+task+'/plan',lambda key=key:make_plan(key,contexts[h,arm,fixture]),out/'blind'/task/'plan.json')
            if not (arm=='PM' and key=='key00') and pack['readers']['RANK3R']['selection']['detected'] is True:
                dump(out/'candidate_stop.json',dict(status='METHOD_NOT_SUPPORTED',cause='NULL_DETECT',task=task,no_further_physical_calls=True,fixed_saved_diagnostics_only=True))
        def plans_back():
            for task in tasks():read(out/'blind'/task/'plan.json')
            return dict(status='READ_BACK',task_plans=120,reader_plans=360,payload_values_read=False)
        journal.record('plans_complete',plans_back,out/'plans_complete.json');cache.enabled=True
        for task in tasks():
            value=journal.record('blind/'+task+'/read',lambda task=task:read_plan(read(out/'blind'/task/'plan.json'),cache),out/'blind'/task/'decision.json')
            outputs['tasks'][task]=dict(status='RETURNED',readers={r:v['aggregate'] for r,v in value['readers'].items()},local_CDC=value['local_CDC'])
        def decisions_back():
            for task in tasks():read(out/'blind'/task/'decision.json')
            return dict(status='READ_BACK',task_decisions=120,reader_decisions=360,truth_accessed=False,payload_files_loaded=len(cache.loaded),cache_accesses=cache.accesses)
        journal.record('blind_complete',decisions_back,out/'blind_complete.json')
        from runtime.wan.b_rank3r_new_source_confirmation_evaluation_v1 import evaluate
        journal.record('post',lambda:evaluate(out,backend,journal,tensor_root=tensor_root),out/'evaluation.json')
    except BaseException as error:
        stop=dict(type=type(error).__name__,message=str(error),status='TECHNICAL_UNRESOLVED',retry=False,interrupted=not isinstance(error,Exception))
    finally:
        output_statuses(outputs,journal.data);dump(out/'planned_outputs.json',outputs);journal.save()
        try:dump(out/'off_rgb24_archives.json',archives)
        except Exception as error:archives['index_write_error']=f'{type(error).__name__}: {error}'
    complete=all(row['status']=='RETURNED' for row in journal.data.values())
    result=dict(status='COMPLETED_FIXED_ATTEMPT' if complete else 'TECHNICAL_UNRESOLVED',denominators=denominators(),
        physical_counts=journal.physical,ledger_counts=dict(Counter(row['status'] for row in journal.data.values())),stop=stop,
        output=str(out),physical_reentry=False,off_rgb24_archives=archives,archive_failures_affect_method=False,
        optional_source_records=historical_records)
    dump(out/'result.json',result)
    if scratch is not None:scratch.cleanup()
    return result

def compact_summary(result,output):
    base=result.get('saved',{}).get('result.json',result)
    try:ev=read(Path(output)/'evaluation.json')
    except (OSError,ValueError):ev={}
    return dict(status=result['status'],saved_status=base.get('status'),denominators=denominators(),physical_counts=base.get('physical_counts'),
        method_status=ev.get('method_status'),contribution_status=ev.get('contribution_status'),checks=ev.get('checks'),
        per_source=ev.get('per_source'),paired=ev.get('paired_summary'),stop=base.get('stop'),off_rgb24_archives=base.get('off_rgb24_archives'),output=str(output))
