"""Fixed source reservation, execution journal and read-only consumed recovery."""
from collections import Counter
import json
import os
from pathlib import Path
import time
import uuid

IO = None


def begin_io():
    global IO
    IO = Counter()


def save_io(path):
    global IO
    saved=IO;IO=None
    try: write(path,dict(scope='new record helper JSON opens/writes only; excludes native tensor/array I/O and this receipt',counts=dict(saved or {})))
    finally: IO=saved


def read(path):
    def invalid(value): raise ValueError('non-finite JSON: '+value)
    if IO is not None: IO['json_read_started']+=1
    result=json.loads(Path(path).read_text(),parse_constant=invalid)
    if IO is not None: IO['json_read_returned']+=1
    return result


def write(path,value):
    if IO is not None: IO['json_write_started']+=1
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=1)
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    if IO is not None: IO['json_write_returned']+=1


def empty_slots(reason='NOT_STARTED'):
    return [dict(slot=i,value=None,status=reason,cause_set=[reason],selected_refs=[],
                 supporting_refs=[],missing_refs=[],conflict_refs=[]) for i in range(32)]


def reserve(output,config,protocol=None):
    if protocol is None:
        from main.tube_state import b_paper_cohort_v1 as protocol
    output=Path(output)
    if output.exists() and any(output.iterdir()): return False
    protocol.validate(config)
    output.mkdir(parents=True,exist_ok=True)
    attempt=dict(status='RESERVED',consumed=True,reserved_at=time.time(),source=config['source'],
                 denominators=protocol.DENOMINATORS,rerun='READ_ONLY_INCLUDING_POST')
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(attempt,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    except FileExistsError: return False
    write(output/'private/config.json',config)
    write(output/'public.json',dict(source=config['source']['id'],model_id=config['model_id'],
        generation=config['generation'],inputs=protocol.inputs(),limits=protocol.LIMITS))
    write(output/'state.json',protocol.operation_plan())
    write(output/'planned_slots.json',protocol.layout())
    for entry in protocol.layout():
        write(output/entry['path'],dict(entry,status='NOT_STARTED',slots=empty_slots()))
    raw={b['view_id']:dict(b,input_id=item['id'],source=config['source']['id'],status='NOT_STARTED',keys={})
         for item in protocol.inputs() for b in protocol.bank(item)}
    write(output/'raw_bank.json',raw)
    raw_slots={uid:{key:[dict(local_row=r,region=region,local_bit=b,value=None,status='NOT_STARTED')
        for r in range(1,11) for region in ('R0','R1') for b in range(8)] for key in protocol.KEYS} for uid in raw}
    write(output/'raw_terminal_slots.json',dict(slots=raw_slots,planned_slots=57600,status='NOT_STARTED'))
    write(output/'decisions.json',[dict(input_id=i['id'],arm=i['arm'],view=i['view'],key_id=key,
        claim_id=q,status='NOT_STARTED',verdict=None,comparisons=0)
        for i in protocol.inputs() for key in protocol.KEYS for q in protocol.CLAIMS])
    write(output/'time_terminal.json',[dict(input_id=i['id'],arm=i['arm'],view=i['view'],status='NOT_STARTED',
        nodes=[dict(received_frame=r,status='NOT_STARTED',conditional_candidates=None,conditional_position=None)
               for r in range(i['length'])],
        edges=[dict(received_edge=r,status='NOT_STARTED') for r in range(i['length']-1)]) for i in protocol.inputs()])
    write(output/'quality.json',{f'{pair}/{v}':dict(status='NOT_STARTED')
        for pair in ('NEW_P0','P0_OFF','NEW_OFF') for v in protocol.VIEWS})
    write(output/'post_status.json',dict(status='NOT_STARTED',started=0,returned=0))
    return True


def counts(rows):
    return {kind:dict(planned=sum(r['kind']==kind for r in rows.values()),
        started=sum(r['started'] for r in rows.values() if r['kind']==kind),
        returned=sum(r['returned'] for r in rows.values() if r['kind']==kind)) for kind in sorted({r['kind'] for r in rows.values()})}


class Journal:
    def __init__(self,output):
        self.output=Path(output);self.rows=read(self.output/'state.json')
    def save(self):
        write(self.output/'state.json',self.rows)
    def call(self,address,function):
        row=self.rows[address]
        if row['status']!='NOT_STARTED': raise RuntimeError('operation consumed: '+address)
        row.update(status='STARTED',started=1,started_at=time.time());self.save();tick=time.monotonic()
        try: value=function()
        except BaseException as error:
            row.update(status='FAILED',error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-tick)
            self.save();raise
        row.update(status='RETURNED',returned=1,returned_at=time.time(),seconds=time.monotonic()-tick)
        self.save();return value


def close_blind(output,reason=None):
    """Saved receipt closure only. It neither fills missing method outputs nor evaluates truth."""
    output=Path(output);rows=read(output/'state.json')
    for row in rows.values():
        if row['status']=='STARTED': row.update(status='INTERRUPTED_CONSUMED',error=reason or 'missing return')
    write(output/'state.json',rows);write(output/'counts.json',counts(rows))
    for entry in read(output/'planned_slots.json'):
        receipt=rows[entry['operation']];path=output/entry['path']
        if receipt['status']=='RETURNED':
            try:
                saved=read(path)
                if [s['slot'] for s in saved['slots']]!=list(range(32)): raise ValueError('fixed ordered32 slots required')
                continue
            except (OSError,ValueError,KeyError,TypeError) as error:
                state='OUTPUT_MISSING';message=str(error)
        else: state=receipt['status'];message=receipt.get('error',reason)
        write(path,dict(entry,status=state,operation_status=receipt['status'],error=message,slots=empty_slots(state)))
    decisions=read(output/'decisions.json')
    for d in decisions:
        op=rows[f"{d['input_id']}/{d['key_id']}/{d['claim_id']}"]
        if op['status']!='RETURNED':
            d.update(status=op['status'],verdict=None,comparisons=None if op['started'] else 0,
                     failure=op.get('error',reason))
    write(output/'decisions.json',decisions)
    terminal=read(output/'time_terminal.json')
    for t in terminal:
        address=f"time/{t['input_id']}/relative_compare";state=rows[address]['status']
        if state=='RETURNED':
            try:
                value=read(output/'blind'/t['input_id']/'relative.json')
                if len(value['conditional_nodes'])!=len(t['nodes']) or len(value['conditional_edges'])!=len(t['edges']):
                    raise ValueError('fixed time denominator mismatch')
                t.update(status='RETURNED',nodes=value['conditional_nodes'],edges=value['conditional_edges']);continue
            except (OSError,ValueError,KeyError,TypeError) as error: state='OUTPUT_MISSING';t['error']=str(error)
        t['status']=state
        for row in t['nodes']+t['edges']: row['status']=state
    write(output/'time_terminal.json',terminal)
    write(output/'blind_terminal.json',dict(status='ALL_BLIND_TERMINAL',saved_at=time.time(),failure=reason,
        claims=len(decisions),time_nodes=sum(len(t['nodes']) for t in terminal),
        time_edges=sum(len(t['edges']) for t in terminal),final_slots=1280,truth_opened=False))


def finish(output,failure=None,*,interrupted=False):
    output=Path(output);rows=read(output/'state.json');attempt=read(output/'attempt.json')
    decisions=read(output/'decisions.json');unknown=[i for i,d in enumerate(decisions) if d['comparisons'] is None]
    known=sum(d['comparisons'] for d in decisions if type(d['comparisons']) is int)
    complete=all(r['status']=='RETURNED' for r in rows.values())
    post=read(output/'post_status.json')
    result=dict(status='COMPLETED' if complete and post['status']=='RETURNED' and not failure else 'STOPPED_FIXED_ATTEMPT',
        consumed=True,failure=failure,interrupted=interrupted,denominators=attempt['denominators'],counts=counts(rows),
        operations=dict(Counter(r['status'] for r in rows.values())),post=post,
        claims=dict(Counter(str(d['verdict']) for d in decisions)),comparisons=None if unknown else known,
        known_completed_comparisons=known,unknown_comparison_claims=unknown,
        consumed_comparison_upper_bound=known+32*len(unknown),scientific_pass=None)
    write(output/'result.json',result);attempt.update(status=result['status'],finished_at=time.time(),failure=failure)
    write(output/'attempt.json',attempt);return result


def recover(output):
    output=Path(output);result=dict(status='READ_ONLY',consumed=True,output=str(output),saved={})
    for name in ('attempt.json','result.json','post_status.json','evaluation_summary.json','preparation_failure.json'):
        if (output/name).is_file():
            try: result['saved'][name]=read(output/name)
            except (OSError,ValueError) as error: result['saved'][name]=dict(read_error=str(error))
    return result
