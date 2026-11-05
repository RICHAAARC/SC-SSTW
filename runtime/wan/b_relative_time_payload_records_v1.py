"""Standalone stdlib reservation, fixed551 ledger and read-only recovery."""
import json
import os
from pathlib import Path
import time
import uuid
from collections import Counter

VIEWS = ('FULL', 'CROP', 'DELETE1', 'REPEAT1', 'DELETE3')
LENGTHS = (181, 101, 180, 182, 178)
STRATEGIES = ('B_NEW', 'U_NEW', 'W_NEW', 'K_NEW', 'P_BASE')
LIMITS = dict(worker_seconds=12600, process_tree_rss_bytes=8*1024**3, poll_seconds=.1, terminate_grace_seconds=2)
DENOMINATORS = dict(logical_operations=551, E41=120, project=120, keyed_decode=180,
                    raw_bit_slots=28800, vote_uses=230400, physical_coefficients=153600,
                    local_slots=2400, final_slots=800, per_strategy_final=160, sources=1)


def inputs():
    return [dict(id=f'input{5*a+i}', arm=arm, view=view, length=n)
            for a, arm in enumerate(('NEW', 'BASE')) for i, (view, n) in enumerate(zip(VIEWS, LENGTHS))]


def strategies(item):
    return STRATEGIES[:4] if item['arm'] == 'NEW' else STRATEGIES[4:]


def bank(item):
    n = item['length']
    return [dict(view_id=f"{item['id']}/E41/a{w+p}", start=w+p, query=q, phase=p,
                 encoder_input_support=list(range(w+p, w+p+41)))
            for q, w in enumerate((0, (n-45)//2, n-45)) for p in range(4)]


def operation_plan():
    rows = {}
    def add(key, kind):
        rows[key] = dict(kind=kind, status='NOT_STARTED', planned=1, started=0, returned=0)
    for item in inputs():
        stem = item['id']
        add(stem+'/plan', 'plan')
        for op in ('probe', 'decode_RGB'): add(stem+'/'+op, op)
        for unit in bank(item):
            for op in ('E41', 'project', 'key00') + (('key01',) if item['arm'] == 'NEW' else ()):
                add(unit['view_id']+'/'+op, 'bits' if op.startswith('key') else op)
        for strategy in strategies(item):
            for q in range(3): add(f'{stem}/{strategy}/q{q}/local', 'local')
            add(f'{stem}/{strategy}/aggregate', 'aggregate')
    add('model_load', 'load')
    return rows


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=1)
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def empty_slots(reason='NOT_ENTERED'):
    return [dict(slot=i, value=None, status=reason, cause_set=[reason], selected_refs=[],
                 supporting_refs=[], missing_refs=[], conflict_refs=[]) for i in range(32)]


def layout():
    result = []
    for item in inputs():
        for strategy in strategies(item):
            stem = f"{item['id']}/{strategy}"
            for q in range(3):
                result.append(dict(path=f'local/{stem}/q{q}.json', operation=f'{stem}/q{q}/local',
                                   input_id=item['id'], strategy=strategy, query=q, kind='local'))
            result.append(dict(path=f'final/{stem}.json', operation=stem+'/aggregate',
                               input_id=item['id'], strategy=strategy, kind='final'))
    return result


def raw_slots():
    return {unit['view_id']: {key: [dict(local_row=row, region=region, local_bit=bit, value=None, status='NOT_ENTERED')
                                  for row in range(1,11) for region in ('R0','R1') for bit in range(8)]
                             for key in (('key00','key01') if item['arm']=='NEW' else ('key00',))}
            for item in inputs() for unit in bank(item)}


def reserve(output, config):
    output = Path(output)
    if output.exists() and any(output.iterdir()): return False
    output.mkdir(parents=True, exist_ok=True)
    receipt = dict(consumed=True, status='RESERVED', reserved_at=time.time(), operations=operation_plan(),
                   slots=[dict(item, slots=empty_slots()) for item in layout()],
                   raw_slots=raw_slots(),
                   denominators=DENOMINATORS, limits=LIMITS, config=config, rerun='READ_ONLY_INCLUDING_POST')
    try:
        with (output/'attempt.json').open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        return False
    write(output/'state.json', operation_plan())
    write(output/'protocol.json', config)
    write(output/'planned_slots.json', receipt['slots'])
    write(output/'raw_planned_slots.json', receipt['raw_slots'])
    return True


def counts(rows):
    kinds = Counter(r['kind'] for r in operation_plan().values())
    return {kind: dict(planned=n, started=sum(r.get('started', 0) for r in rows.values() if r['kind']==kind),
                       returned=sum(r.get('returned', 0) for r in rows.values() if r['kind']==kind))
            for kind, n in kinds.items()}


class Journal:
    def __init__(self, output):
        self.output = Path(output); self.rows = read(self.output/'state.json')

    def save(self):
        write(self.output/'state.json', self.rows); write(self.output/'counts.json', counts(self.rows))

    def call(self, key, function):
        row = self.rows[key]
        if row['status'] != 'NOT_STARTED': raise RuntimeError('consumed operation: '+key)
        row.update(status='STARTED', started=1, started_at=time.time()); self.save()
        start = time.monotonic()
        try:
            value = function()
        except BaseException as error:
            row.update(status='FAILED', error=f'{type(error).__name__}: {error}', seconds=time.monotonic()-start)
            self.save(); raise
        row.update(status='RETURNED', returned=1, seconds=time.monotonic()-start, returned_at=time.time())
        self.save(); return value


def _decoded_index(saved, uid, key):
    """Index already-decoded records by explicit physical addresses; never decode."""
    if (not isinstance(saved, dict) or saved.get('view_id') != uid or saved.get('key') != key
            or saved.get('kind') != 'E41' or saved.get('local_rows') != list(range(1,11))
            or not isinstance(saved.get('regions'), dict)):
        raise ValueError('saved decode unit/key/E41 row addresses invalid')
    indexed, errors = {}, {}
    for region in ('R0','R1'):
        regional = saved['regions'].get(region)
        cells = regional.get('local_cells') if isinstance(regional,dict) else None
        if not isinstance(cells,list): continue
        for cell in cells:
            if not isinstance(cell,dict) or type(cell.get('local_row')) is not int or cell['local_row'] not in range(1,11):
                continue
            row = cell['local_row']; eid = f'{uid}/row{row}/{region}'
            bits = cell.get('bits')
            if (cell.get('evidence_id') != eid or cell.get('observed_region_id',region) != region
                    or not isinstance(bits,list)):
                for bit in range(8): errors[(row,region,bit)] = 'invalid saved row/region/evidence address'
                continue
            for bit in bits:
                if not isinstance(bit,dict) or type(bit.get('local_bit')) is not int or bit['local_bit'] not in range(8):
                    continue
                address = (row,region,bit['local_bit'])
                if address in indexed:
                    errors[address] = 'duplicate saved bit address'
                indexed[address] = bit
                if bit.get('key') != key or bit.get('evidence_id',eid) != eid:
                    errors[address] = 'invalid saved bit key/evidence address'
    return indexed, errors


def _saved_decoded(path):
    import math
    def number(text):
        value = float(text)
        if not math.isfinite(value): raise ValueError('nonfinite saved JSON number')
        return value
    def constant(text):
        raise ValueError('invalid saved JSON constant: '+text)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_float=number, parse_constant=constant)


def finalize_raw(output, operations):
    """Fixed28800 actual bit receipts from saved keyed results, without scientific calls."""
    output = Path(output); slots = raw_slots()
    for uid, keys in slots.items():
        for key, planned in keys.items():
            operation = uid+'/'+key; receipt = operations[operation]
            operation_status = receipt['status']
            path = Path('bank')/uid/(key+'.json'); exists = (output/path).is_file()
            indexed, errors, file_error, saved = {}, {}, None, None
            if operation_status == 'RETURNED':
                try:
                    saved = _saved_decoded(output/path); indexed, errors = _decoded_index(saved, uid, key)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    file_error = f'{type(error).__name__}: {error}'
            for row in planned:
                address = (row['local_row'],row['region'],row['local_bit'])
                row.update(view_id=uid, key_id=key, evidence_id=f"{uid}/row{row['local_row']}/{row['region']}",
                    decode_operation=operation, operation_status=operation_status, receipt_file='state.json',
                    decoded_file=str(path), bit_status=None, status=operation_status,
                    operation_error=receipt.get('error'))
                if operation_status != 'RETURNED':
                    if exists: row['unconfirmed_artifact'] = str(path)
                    continue
                bit = indexed.get(address)
                error = file_error or errors.get(address) or ('missing saved bit address' if bit is None else None)
                if bit is not None:
                    row.update(bit_status=bit.get('status'), original_bit_value=bit.get('value'),
                               count=bit.get('count'), returned_count=bit.get('returned_count'),
                               cause_set=bit.get('cause_set',[]), decoded_status=saved.get('status'))
                    available = bit.get('status') in ('AVAILABLE','RETURNED','READ','CONDITIONAL_READ')
                    failed = bit.get('status') in ('FAILED','NOT_ENTERED','UNRESOLVED','UNSUPPORTED','INTERRUPTED')
                    value = bit.get('value')
                    if not (type(bit.get('count')) is int and bit['count'] == 8 and
                            ((available and type(value) is int and value in (0,1)) or (failed and value is None))):
                        error = error or 'invalid saved bit value/count/status'
                if error:
                    row.update(status='UNRESOLVED_RECEIPT', value=None, receipt_error=error)
                    if exists: row['unconfirmed_artifact'] = str(path)
                else:
                    row.update(status=bit['status'], value=bit['value'])
    flat = [row for keys in slots.values() for rows in keys.values() for row in rows]
    summary = dict(status='ALL_RAW_SLOTS_TERMINAL', planned_slots=28800, actual_slots=len(flat),
        statuses=dict(Counter(row['status'] for row in flat)),
        operation_statuses=dict(Counter(row['operation_status'] for row in flat)),
        confirmed_value_slots=sum(row['value'] is not None for row in flat),
        unconfirmed_artifact_slots=sum('unconfirmed_artifact' in row for row in flat),
        scientific_calls=0, truth_read=False)
    write(output/'raw_terminal_slots.json', dict(summary=summary, slots=slots, saved_at=time.time()))
    write(output/'raw_terminal_summary.json', summary)
    return summary


def finalize_blind(output, reason=None):
    """Engineering saved-result closure only; no method, bit recomputation or truth."""
    output = Path(output)
    rows = read(output/'state.json') if (output/'state.json').is_file() else operation_plan()
    for row in rows.values():
        if row['status'] == 'STARTED': row.update(status='INTERRUPTED', error=reason or 'missing return')
        elif row['status'] == 'NOT_STARTED': row.update(status='NOT_ENTERED', error=reason or 'not entered')
    write(output/'state.json', rows); write(output/'counts.json', counts(rows))
    raw_path = output/'raw_bank.json'
    raw = read(raw_path) if raw_path.is_file() else {u['view_id']: dict(u) for item in inputs() for u in bank(item)}
    for item in inputs():
        for unit in bank(item):
            uid = unit['view_id']; entry = raw[uid]
            keys = ('key00','key01') if item['arm']=='NEW' else ('key00',)
            entry.update(planned_keys=list(keys), raw_bit_slots_per_key=160, vote_slots_per_key=1280,
                         physical_coefficient_slots=1280,
                         operation_statuses={op: rows[uid+'/'+op]['status'] for op in ('E41','project')+keys})
            entry.setdefault('status','NOT_ENTERED')
    write(raw_path,raw)
    raw_summary = finalize_raw(output, rows)
    artifacts = []
    for item in layout():
        path = output/item['path']; record = dict(item, slots=empty_slots(), key_id='key01' if item['strategy']=='K_NEW' else 'key00')
        if rows[item['operation']]['status'] == 'RETURNED':
            try:
                saved = read(path)
                if len(saved['slots']) != 32 or [r['slot'] for r in saved['slots']] != list(range(32)):
                    raise ValueError('ordered fixed32 slots required')
                record.update(saved)
            except (OSError, ValueError, KeyError, TypeError) as error:
                record.update(slots=empty_slots('UNRESOLVED_RECEIPT'), error=str(error))
        else:
            record['operation_status'] = rows[item['operation']]['status']
            if path.exists(): record['unconfirmed_artifact'] = str(path)
        write(path, record); artifacts.append(record)
    terminal = dict(status='ALL_BLIND_SLOTS_TERMINAL', saved_at=time.time(), artifacts=artifacts,
                    denominators=DENOMINATORS, counts=counts(rows), failure=reason,
                    raw_terminal_file='raw_terminal_slots.json', raw_terminal_summary=raw_summary)
    write(output/'blind_terminal.json', terminal)
    return terminal


def recover(output):
    """No setup, planner, cache, quality, post, model or codec calls."""
    output = Path(output); result = dict(status='READ_ONLY', output=str(output), consumed=True,
                                        denominators=DENOMINATORS, saved={})
    for name in ('result.json', 'counts.json', 'resources.json', 'raw_terminal_summary.json',
                 'evaluation_summary.json', 'quality.json', 'preparation_failure.json'):
        path = output/name
        if path.is_file():
            try: result['saved'][name] = read(path)
            except (OSError, ValueError) as error: result['saved'][name] = dict(error=str(error))
    return result
