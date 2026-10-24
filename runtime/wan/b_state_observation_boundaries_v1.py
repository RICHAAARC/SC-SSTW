"""Fixed post-truth observation-boundary diagnostic. Import is stdlib only."""
from collections import Counter
import errno
import json
import math
import os
from pathlib import Path
import time

DEFAULT_INPUT = '/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1'
DEFAULT_OUTPUT = '/content/drive/MyDrive/Video-WM/B-State-Observation-Boundaries-V1/fixed-e01-e02-v1'
LAYERS = ('N', 'A', 'B', 'C', 'D')
CASES = ({'source': 'b2-m3-e01', 'a': 17, 's': 19, 'phase': 2, 'start': 72},
         {'source': 'b2-m3-e02', 'a': 6, 's': 16, 'phase': 1, 'start': 60})

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

def make_plan(input_directory=None):
    from pathlib import Path
    source_root = Path(input_directory or '/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
    units = {}
    for case in CASES:
        source,start,s,phase=(case[k] for k in ('source','start','s','phase'))
        for arm in ('P0', 'PM'):
            for layer in ('N', 'A', 'B', 'C', 'D'):
                slot = f'{source}/{arm}/{layer}'
                if layer == 'N': relative = f'private/{source}/native/{arm}_Q.json'
                elif layer == 'D': relative = f'observations/{source}/{arm}/CROP/M/p{phase}/key00/Q.json'
                else: relative = f'media/{source}/{arm}/full/'+('rgb8.pt' if layer == 'C' else 'float_rgb.pt')
                units[slot] = dict(slot=slot, source=source, arm=arm, layer=layer,
                    s=s, phase=phase, source_start=start, input_path=str(source_root/relative),
                    source_frames=[0,180] if layer == 'A' else [start,start+40],
                    encoder_frames=181 if layer == 'A' else 41 if layer in ('B','C') else 0,
                    kind='E181' if layer == 'A' else 'E41' if layer in ('B','C') else 'SAVED_Q',
                    status='NOT_ENTERED', attempts=0, q40=[None]*40, metrics=None)
    return dict(input_directory=str(source_root), units=units,
        denominators=dict(cases=4, layers=5, layer_cases=20, coordinates=800,
                          new_encodes=12, E181=4, E41=8, new_input_frames=1052,
                          new_Q40=12, reused_Q40=8, new_coordinates=480),
        post_truth_only=True, blind_receiver_changed=False)


def execution_artifacts(output):
    from pathlib import Path
    output = Path(output)
    names = ('attempt.json','plan.json','ledger.json','result.json','evaluation.json',
             'inputs.json','physical_counts.json','release.json')
    found = [n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for name in ('receipts','layers'):
        folder = output/name
        if folder.is_dir():
            found.extend(str(p.relative_to(output)) for p in folder.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def recover_output(output):
    """Read-only receipt overlay, embedded in the notebook; never resumes work."""
    from pathlib import Path
    from collections import Counter
    import json
    output = Path(output)
    errors = {}
    def load(name, default):
        try: return json.loads((output/name).read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            errors[name] = str(error)
            return default
    plan = load('plan.json', make_plan())
    if not isinstance(plan, dict) or not isinstance(plan.get('units'), dict):
        errors['plan.json'] = 'invalid saved plan'
        plan = make_plan()
    layers = {slot: dict(row) for slot,row in make_plan(plan.get('input_directory'))['units'].items()}
    ledger = load('ledger.json', {})
    if not isinstance(ledger, dict):
        errors['ledger.json'] = 'invalid ledger'
        ledger = {}
    for path in sorted((output/'receipts').rglob('*.json'))+sorted((output/'receipts').rglob('*.json.tmp')):
        rel = path.relative_to(output/'receipts').as_posix()
        slot = rel[:-9] if rel.endswith('.tmp') else rel[:-5]
        old = ledger.get(slot, {})
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
            if rel.endswith('.tmp'): raise ValueError('uncommitted receipt')
            if not isinstance(receipt, dict) or receipt.get('slot') != slot: raise ValueError('receipt slot mismatch')
            status, attempt = receipt.get('status'), receipt.get('attempts')
            if status not in ('NOT_ENTERED','STARTED','RETURNED','FAILED','INTERRUPTED'): raise ValueError('invalid receipt status')
            if type(attempt) is not int or attempt != (0 if status == 'NOT_ENTERED' else 1): raise ValueError('invalid receipt attempts')
            if isinstance(old, dict) and old.get('attempts') == 1 and attempt == 0: raise ValueError('cannot erase spent attempt')
            ledger[slot] = receipt
        except (OSError, ValueError) as error:
            errors['receipts/'+rel] = str(error)
            ledger[slot] = dict(slot=slot,status='UNRESOLVED_RECEIPT',attempts=old.get('attempts') if isinstance(old,dict) else None)
    for slot,row in layers.items():
        receipt = ledger.get(slot, {})
        if not isinstance(receipt,dict): receipt = dict(status='UNRESOLVED_RECEIPT',attempts=None)
        if receipt.get('status') == 'RETURNED':
            value = load('layers/'+slot+'.json', None)
            if isinstance(value,dict) and isinstance(value.get('q40'),list) and len(value['q40']) == 40:
                row.update(value)
            else:
                row.update(status='UNRESOLVED_RESULT', attempts=1)
        else:
            partial = output/'layers'/(slot+'.json')
            if partial.is_file():
                value = load('layers/'+slot+'.json', None)
                if isinstance(value,dict) and isinstance(value.get('q40'),list) and len(value['q40'])==40:
                    row.update(value)
            row.update({k:v for k,v in receipt.items() if k not in ('q40','metrics')})
    return dict(status='READ_ONLY_EXISTING_ATTEMPT', physical_reentry=False,
        denominators=make_plan()['denominators'], layers=layers, ledger=ledger,
        counts=dict(Counter(r['status'] for r in layers.values())),
        physical_counts=load('physical_counts.json', None),
        evaluation=load('evaluation.json', None), read_errors=errors)


def finite(value):
    return type(value) in (int,float) and math.isfinite(value)


def q40_from_saved(record, s, native=False):
    expected = 44 if native else 10
    if not isinstance(record,dict) or record.get('status') != ('SCORE_OBSERVED' if native else 'NEW_E41_OBSERVED'):
        raise ValueError('saved N/D observation status unavailable')
    if not native and record.get('retained_rows') != list(range(1,11)):
        raise ValueError('saved D must retain local rows 1..10')
    q = record.get('roi_q') if isinstance(record,dict) else None
    if not isinstance(q,list) or len(q) != expected or any(not isinstance(r,list) or len(r)!=4 or any(not isinstance(v,list) or len(v)!=44 for v in r) for r in q):
        raise ValueError('saved Q must have native44/local10 x 4 x 44 shape')
    values = [q[s-1+i if native else i][roi][s-1+i] for i in range(10) for roi in range(4)]
    if not all(finite(x) for x in values): raise ValueError('selected saved Q40 has missing/nonfinite coordinates')
    return values


def compile_background(record):
    if record.get('dimension') != 40:raise ValueError('background dimension40 required')
    mean, factor = record.get('mean'), record.get('factor')
    if not isinstance(mean,list) or len(mean)!=40 or not all(finite(x) for x in mean): raise ValueError('finite background mean40 required')
    if not isinstance(factor,list) or len(factor)!=40 or any(not isinstance(r,list) or len(r)!=40 or not all(finite(x) for x in r) for r in factor): raise ValueError('finite background factor40x40 required')
    if any(factor[i][i]<=0 or any(factor[i][j]!=0 for j in range(i+1,40)) for i in range(40)): raise ValueError('positive diagonal lower factor required')
    return dict(mean=mean,factor=factor)


def solve(factor, y):
    x=[]
    for i,row in enumerate(factor): x.append((y[i]-math.fsum(row[j]*x[j] for j in range(i)))/row[i])
    if not all(finite(v) for v in x): raise ValueError('nonfinite whitened coordinates')
    return x


def dot(a,b): return math.fsum(x*y for x,y in zip(a,b))


def describe(q, model):
    if len(q)!=40 or not all(finite(v) for v in q): raise ValueError('finite Q40 required')
    x=solve(model['factor'],[a-b for a,b in zip(q,model['mean'])])
    u=solve(model['factor'],[1.]*40); u2=dot(u,u); energy=dot(x,x)
    if u2<=0 or energy<=0:
        return dict(status='ZERO_ENERGY_SCORE_UNRESOLVED',S=None,signed_cosine=None,alpha_signed=None,
                    residual_energy=None,energy=energy,template_energy=u2)
    alpha=dot(x,u)/u2; r=[a-alpha*b for a,b in zip(x,u)]
    signed=dot(x,u)/math.sqrt(u2*energy)
    if not finite(signed) or abs(signed)>1+64*2.220446049250313e-16:
        return dict(status='COSINE_NUMERICALLY_UNRESOLVED',S=None,signed_cosine=signed if finite(signed) else None,
                    alpha_signed=alpha,residual_energy=dot(r,r),energy=energy,template_energy=u2)
    signed=max(-1.,min(1.,signed))
    raw_energy=dot(q,q)
    return dict(status='MEASURED',S=max(0.,signed),signed_cosine=signed,
        alpha_signed=alpha,residual_energy=dot(r,r),energy=energy,template_energy=u2,
        numerator=dot(x,u),mean=math.fsum(q)/40,
        raw_cosine=None if raw_energy==0 else math.fsum(q)/math.sqrt(40*raw_energy),
        raw_cosine_status='ZERO_ENERGY' if raw_energy==0 else 'MEASURED')


def paired_metrics(before, after, native_before, native_after, model):
    if any(r.get('status')!='RETURNED' for r in (before,after,native_before,native_after)):
        return dict(status='UNRESOLVED_INPUT',coordinate_denominator=40,delta=[None]*40,
                    cosine_native=None,mean_gain=None)
    delta=[b-a for a,b in zip(before['q40'],after['q40'])]
    native=[b-a for a,b in zip(native_before['q40'],native_after['q40'])]
    nd,nn=dot(delta,delta),dot(native,native); nm=math.fsum(native)/40
    d=solve(model['factor'],delta);u=solve(model['factor'],[1.]*40);u2=dot(u,u)
    alpha=dot(d,u)/u2;residual=[a-alpha*b for a,b in zip(d,u)]
    return dict(status='MEASURED' if nd>0 and nn>0 and nm!=0 else 'PARTIAL_ZERO_DENOMINATOR',
        coordinate_denominator=40,delta=delta,native_delta=native,
        delta_mean=math.fsum(delta)/40,native_delta_mean=nm,
        cosine_native=None if nd==0 or nn==0 else dot(delta,native)/math.sqrt(nd*nn),
        cosine_native_status='ZERO_NORM' if nd==0 or nn==0 else 'MEASURED',
        mean_gain=None if nm==0 else (math.fsum(delta)/40)/nm,
        mean_gain_status='ZERO_NATIVE_MEAN' if nm==0 else 'MEASURED',
        whitened_alpha_signed=alpha,whitened_residual_energy=dot(residual,residual))


def evaluate(layers, model, tau):
    pairs={}
    for case in CASES:
        source=case['source']
        for layer in LAYERS:
            get=lambda arm,l:layers[f'{source}/{arm}/{l}']
            pairs[f'{source}/{layer}']=paired_metrics(get('P0',layer),get('PM',layer),get('P0','N'),get('PM','N'),model)
    boundary_rows=[]
    for left,right in zip(LAYERS,LAYERS[1:]):
        values={}
        for case in CASES:
            source=case['source'];a=layers[f'{source}/PM/{left}'];b=layers[f'{source}/PM/{right}']
            values[source]=dict(status='MEASURED' if a['status']==b['status']=='RETURNED' else 'UNRESOLVED',
                delta_q40=[v-u for u,v in zip(a['q40'],b['q40'])] if a['status']==b['status']=='RETURNED' else [None]*40,
                S_before=a.get('metrics',{}).get('S') if a.get('metrics') else None,
                S_after=b.get('metrics',{}).get('S') if b.get('metrics') else None)
        boundary_rows.append(dict(boundary=left+'->'+right,sources=values))
    first_loss=[]
    for index in range(1,len(LAYERS)):
        e2=[layers[f'b2-m3-e02/PM/{l}'] for l in LAYERS]
        e1=[layers[f'b2-m3-e01/PM/{l}'] for l in LAYERS]
        complete=all(x['status']=='RETURNED' for x in e1+e2)
        if complete and all(x['metrics']['S']>=tau for x in e2[:index]) and all(x['metrics']['S']<tau for x in e2[index:]) and all(x['metrics']['S']>=tau for x in e1[index-1:]):
            first_loss.append(LAYERS[index-1]+'->'+LAYERS[index])
    return dict(status='POST_TRUTH_DIAGNOSTIC',pairs=pairs,boundaries=boundary_rows,
        tau_reference_only=tau,first_reference_line_loss_boundary=first_loss,
        first_loss_status='DESCRIPTIVE_FIXED_CASE_ONLY' if first_loss else 'MIXED_OR_UNRESOLVED_OR_NO_QUALIFYING_LOSS',
        evidence_ceiling='same two viewed sources, fixed M window; no causal single-factor proof or blind detection PASS',
        scientific_pass=None,blind_receiver_changed=False)


class Journal:
    def __init__(self, output, plan):
        self.output=Path(output)
        self.ledger={k:{n:r[n] for n in ('slot','kind','status','attempts')} for k,r in plan['units'].items()}
        for slot in ('inputs','model_load'):self.ledger[slot]=dict(slot=slot,kind=slot,status='NOT_ENTERED',attempts=0)
        self.physical=dict(planned=12,started=0,returned=0,failed=0,per_slot={})
        dump(self.output/'ledger.json',self.ledger);dump(self.output/'physical_counts.json',self.physical)
    def call(self, slot, function, backend=None):
        row=self.ledger[slot]
        if row['status']!='NOT_ENTERED': raise RuntimeError('spent fixed slot '+slot)
        row.update(status='STARTED',attempts=1,started_unix=time.time())
        dump(self.output/'receipts'/(slot+'.json'),row)
        dump(self.output/'ledger.json',self.ledger)
        dump(self.output/'progress.json',dict(active=slot,counts=dict(Counter(x['status'] for x in self.ledger.values()))))
        blocks=dict(getattr(backend,'blocks',{}))
        try:
            value=function();row['status']='RETURNED';return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row['finished_unix']=time.time()
            after=dict(getattr(backend,'blocks',{}))
            row['internal_block_forward_delta']={k:after[k]-v for k,v in blocks.items() if type(v) is int and type(after.get(k)) is int}
            dump(self.output/'receipts'/(slot+'.json'),row);dump(self.output/'ledger.json',self.ledger)
    def physical_event(self,slot,event):
        item=self.physical['per_slot'].setdefault(slot,dict(started=0,returned=0,failed=0))
        if event=='started' and (item['started'] or self.physical['started']>=12): raise RuntimeError('fixed encode budget already consumed')
        item[event]+=1;self.physical[event]+=1
        dump(self.output/'physical_counts.json',self.physical)


def run(input_directory=DEFAULT_INPUT, output=DEFAULT_OUTPUT, backend=None):
    input_directory,output=Path(input_directory).expanduser().resolve(),Path(output).expanduser().resolve()
    if execution_artifacts(output):return recover_output(output)
    if output==input_directory or output.is_relative_to(input_directory) or input_directory.is_relative_to(output):
        raise ValueError('new output and original input directories must be separate')
    output.mkdir(parents=True,exist_ok=True)
    # Exclusive creation spends the whole run before input loading or model import.
    try:
        with (output/'attempt.json').open('x',encoding='utf-8') as stream:
            json.dump(dict(started_unix=time.time(),physical_budget=12,resume=False),stream)
            stream.flush()
            try:os.fsync(stream.fileno())
            except OSError as error:
                if error.errno not in (errno.EINVAL,errno.ENOSYS,errno.ENOTSUP):raise
    except FileExistsError:return recover_output(output)
    plan=make_plan(input_directory);dump(output/'plan.json',plan)
    journal=Journal(output,plan);layers={k:dict(v) for k,v in plan['units'].items()}
    model=None;tau=None;stop=None;release_error=None
    try:
        def inputs():
            protocol=read(input_directory/'protocol.json')['config']
            for case in CASES:
                host=next(x for x in protocol['sources'] if x['id']==case['source'])
                if host['a']!=case['a'] or host['role']!='eval':raise ValueError('adopted source/window semantics unavailable')
                for arm in ('P0','PM'):
                    saved_plan=read(input_directory/'blind'/case['source']/arm/'CROP/key00/plan.json')
                    window=saved_plan['locals']['M'];w=window['received_query_start']
                    if w!=53:raise ValueError('saved CROP M window must start at received53')
                    origin=host['a']+w;phase=(-origin)%4;start=origin+phase;s=start//4+1
                    if (phase,start,s)!=(case['phase'],case['start'],case['s']):raise ValueError('saved source/window geometry differs')
                    true_members=[m for m in window['members'] if m.get('phase')==phase and m.get('s')==s]
                    if len(true_members)!=1 or true_members[0].get('O_h')!=origin:raise ValueError('saved true member geometry unavailable')
            if protocol['generation']['frames']!=181 or protocol['generation']['height']!=320 or protocol['generation']['width']!=512:raise ValueError('adopted source geometry unavailable')
            secret=protocol['writer']['key']
            if not isinstance(secret,str) or not secret or secret!='watermark':raise ValueError('adopted key00 unavailable')
            bg=read(input_directory/'background.json')
            if bg.get('status')!='BACKGROUND_FROZEN' or bg['models']['key00'].get('status')!='BACKGROUND_FROZEN':raise ValueError('saved frozen background unavailable')
            background=compile_background(bg['models']['key00'])
            threshold=read(input_directory/'thresholds.json')
            if threshold.get('status')!='FROZEN':raise ValueError('saved frozen tau unavailable')
            tau=threshold['tau']
            if not finite(tau) or not 0<=tau<=1:raise ValueError('finite saved reference tau in [0,1] required')
            if threshold.get('tau_rule')!='max(0,max_S_of_90_complete_null_window_families)':raise ValueError('saved reference tau rule unavailable')
            record=dict(model_id=protocol['model_id'],key=secret,background=background,tau_reference=tau,input_directory=str(input_directory))
            dump(output/'inputs.json',record);return record
        metadata=journal.call('inputs',inputs);model=metadata['background'];tau=metadata['tau_reference']
        # Read both fixed saved boundaries before any new physical call.
        for slot,row in layers.items():
            if row['layer'] not in ('N','D'):continue
            def reuse(row=row,slot=slot):
                q=q40_from_saved(read(row['input_path']),row['s'],row['layer']=='N')
                metrics=describe(q,model)
                value=dict(row,status='RETURNED' if metrics['status']=='MEASURED' else 'FAILED',attempts=1,q40=q,metrics=metrics,reused=True)
                layers[slot]=value
                dump(output/'layers'/(slot+'.json'),value)
                if metrics['status']!='MEASURED':raise ValueError(metrics['status'])
                return value
            layers[slot]=journal.call(slot,reuse)
        if backend is None:
            from runtime.wan.b_state_observation_boundaries_backend_v1 import BoundaryBackend
            backend=BoundaryBackend(metadata['model_id'],metadata['key'])
        def load_model():
            value=backend.load();dump(output/'model.json',value);return value
        journal.call('model_load',load_model,backend)
        for slot,row in layers.items():
            if row['layer'] not in ('A','B','C'):continue
            def observe(row=row,slot=slot):
                q=backend.encode_q40(row,lambda event:journal.physical_event(slot,event))
                metrics=describe(q,model)
                value=dict(row,status='RETURNED' if metrics['status']=='MEASURED' else 'FAILED',attempts=1,q40=q,metrics=metrics,reused=False)
                layers[slot]=value
                dump(output/'layers'/(slot+'.json'),value)
                if metrics['status']!='MEASURED':raise ValueError(metrics['status'])
                return value
            layers[slot]=journal.call(slot,observe,backend)
    except BaseException as error:
        stop=dict(type=type(error).__name__,message=str(error))
    finally:
        if backend is not None:
            try:
                backend.release();dump(output/'release.json',dict(status='RELEASED'))
            except BaseException as error:
                release_error=f'{type(error).__name__}: {error}'
                dump(output/'release.json',dict(status='RELEASE_FAILED',error=release_error))
        # Recover receipt status before reporting, retaining every fixed row.
        for slot,row in layers.items():
            state=journal.ledger[slot]
            if state['status']!='RETURNED':
                row.update(status=state['status'],attempts=state['attempts'],error=state.get('error'))
        evaluation=evaluate(layers,model,tau)
        dump(output/'evaluation.json',evaluation)
        result=dict(status='COMPLETED_ENGINEERING_RECORDS' if stop is None and release_error is None else 'STOPPED_FIXED_ATTEMPT',
                    denominators=plan['denominators'],layers=layers,physical_counts=journal.physical,
                    counts=dict(Counter(x['status'] for x in layers.values())),stop=stop,release_error=release_error,
                    post_truth_only=True,scientific_pass=None)
        dump(output/'result.json',result)
    return result
