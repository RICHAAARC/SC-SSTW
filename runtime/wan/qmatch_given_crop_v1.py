"""Once-consumed, stdlib-only given-CROP screen over existing keyed bits.

The header pass parses ordinary JSON then discards everything outside its small
allowlist. It does not claim byte-level isolation: only the planner's actual
data dependencies are isolated. Cache values are loaded after all plans save.
"""
from collections import Counter
from pathlib import Path
import time

from main.tube_state import b_cdc32_aggregation_v1 as cdc
from main.tube_state import qmatch_given_crop_v1 as method
from main.tube_state import qmatch_v1 as qmatch
from .qmatch_saved_v1 import CLAIMS, read_json, write_json, recover

HEADER_FIELDS = ('status', 'kind', 'view_id', 'key', 'local_rows')


def claims_from_task(task):
    expected = dict(scope=qmatch.SCOPE, geometry_scope='GIVEN_CROP_MAP',
                    sources=[dict(source=s, a=a) for s, a in method.SOURCES],
                    view='CROP', received_frames=151, arms=['OFF', 'P0'], keys=['key00', 'key01'],
                    query_starts=[0, 53, 106], phases=[0, 1, 2, 3], regions=['R0', 'R1'],
                    bit_order='MSB_FIRST', limits=method.LIMITS, attempt='run01')
    if not isinstance(task, dict) or any(task.get(k) != v for k, v in expected.items()):
        raise ValueError('fixed adopted given-CROP scope/roster/budget required')
    claims = task.get('claims')
    if not isinstance(claims, list) or len(claims) != 2:
        raise ValueError('exactly two adopted literal claims required')
    for claim, (name, hexadecimal) in zip(claims, CLAIMS):
        bits = [int(b) for b in f'{int(hexadecimal, 16):032b}']
        if (not isinstance(claim, dict) or claim.get('claim_id') != name or claim.get('hex') != hexadecimal
                or claim.get('bits') != bits or not all(qmatch._binary(b) for b in claim['bits'])):
            raise ValueError('claim differs from adopted MSB-first literal')
    return [dict(claim_id=c['claim_id'], bits=list(c['bits'])) for c in claims]


def header_only(decoded, entry):
    """An allowlisted object is the only bits-file input to the planner."""
    if not isinstance(decoded, dict):
        raise ValueError('bits object required')
    header = {k: decoded.get(k) for k in HEADER_FIELDS}
    if (header['status'] not in ('AVAILABLE', 'INCOMPLETE') or header['kind'] != 'E41'
            or header['key'] != entry['key_id'] or not isinstance(header['view_id'], str)
            or not header['view_id'] or header['local_rows'] != list(range(1, 11))
            or any(type(r) is not int for r in header['local_rows'])):
        raise ValueError('invalid E41/key/view/row header')
    return header


def load_record(path, kind, reads, transform=None):
    record = dict(path=str(path), kind=kind, status='STARTED')
    reads.append(record)
    try:
        value = read_json(path)
        if transform is not None:
            value = transform(value)
        record['status'] = 'RETURNED'
        return value, []
    except (OSError, ValueError, TypeError, KeyError) as error:
        record.update(status='UNREADABLE', reason=f'{type(error).__name__}: {error}')
        return None, [dict(kind=kind, path=str(path), reason=record['reason'])]


def cache_from_payload(decoded, entry, header):
    """Flatten only R0/R1, keeping illegal/missing cells out of the cache.

Bit value/count/status legality remains the original CDC reader's decision.
Address and diagnostic checks prevent joining an unrelated or control cell.
"""
    current = header_only(decoded, entry)
    if any(current[k] != header.get(k) for k in ('kind', 'view_id', 'key', 'local_rows')):
        raise ValueError('bits header no longer addresses the saved plan')
    cache, errors = {}, []
    regions = decoded.get('regions')
    if not isinstance(regions, dict):
        raise ValueError('R0/R1 region objects required')
    for region in ('R0', 'R1'):
        block = regions.get(region)
        if (not isinstance(block, dict) or block.get('diagnostic_only', False)
                or block.get('status') not in ('AVAILABLE', 'INCOMPLETE')
                or block.get('observed_region_id') != region or block.get('carrier_region_id') != region
                or not isinstance(block.get('local_cells'), list)):
            errors.append(dict(region=region, reason='missing/invalid production region'))
            continue
        cells = block['local_cells']
        for row in range(1, 11):
            matching = [c for c in cells if isinstance(c, dict) and type(c.get('local_row')) is int and c['local_row'] == row]
            eid = f"{header['view_id']}/row{row}/{region}"
            try:
                if len(matching) != 1:
                    raise ValueError('one cell per local row required')
                cell = matching[0]
                if (cell.get('evidence_id') != eid or cell.get('observed_region_id') != region
                        or cell.get('carrier_region_id') != region or cell.get('diagnostic_only', False)
                        or cell.get('status') != 'AVAILABLE'):
                    raise ValueError('cell address or production scope invalid')
                bits = cell.get('bits')
                if not isinstance(bits, list):
                    raise ValueError('bit address list required')
                admitted = {}
                for bit in range(8):
                    matching_bits = [b for b in bits if isinstance(b, dict)
                                     and type(b.get('local_bit')) is int and b['local_bit'] == bit]
                    if (len(matching_bits) != 1 or matching_bits[0].get('key') != entry['key_id']
                            or matching_bits[0].get('evidence_id') != eid):
                        errors.append(dict(evidence_id=eid, local_bit=bit, reason='one matching production bit address required'))
                        continue
                    admitted[str(bit)] = matching_bits[0]
                cache[eid] = {entry['key_id']: admitted}
            except (ValueError, TypeError, KeyError) as error:
                errors.append(dict(evidence_id=eid, reason=str(error)))
    return cache, errors


def run_once(task_path, input_root, output):
    task_path, root, output = Path(task_path), Path(input_root), Path(output)
    if output.exists():
        return recover(output)
    resolved = output.resolve()
    for source in (task_path.resolve(), root.resolve()):
        if source == resolved or source in resolved.parents or resolved in source.parents:
            raise ValueError('output must be separate from saved inputs and task')
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return recover(output)
    units = method.units()
    attempt = dict(status='STARTED', consumed=True, started_at=time.time(), limits=method.LIMITS,
                   phase='RESERVED', unit_states=['NOT_STARTED'] * 16, rerun='READ_ONLY')
    decisions = [dict(scope=qmatch.SCOPE, geometry_scope='GIVEN_CROP_MAP', unit=u, claim_id=name,
                      status='NOT_STARTED', verdict=None, comparisons=0) for u in units for name, _ in CLAIMS]
    reads, plans, headers, metadata_errors = [], [], {}, {}
    calls = dict(plans=0, local=0, aggregate=0, claims=0)
    failure = None
    try:
        write_json(output / 'attempt.json', attempt)
        write_json(output / 'decisions.json', decisions)
        task, errors = load_record(task_path, 'TASK', reads)
        if errors:
            raise ValueError(errors[0]['reason'])
        claims = claims_from_task(task)
        write_json(output / 'query_task.json', task)
        operations, ledger_errors = load_record(root / 'ledger.json', 'LEDGER', reads)
        if not isinstance(operations, dict):
            ledger_errors = ledger_errors or [dict(kind='LEDGER', reason='operation object required')]
            operations = {}
        addresses = {e['stem']+'/'+suffix for u in units for e in method.bank(u)
                     for suffix in ('E', 'DCT', u['key_id']+'/bits')}
        operations = {a: dict(status=operations[a].get('status')) for a in addresses
                      if isinstance(operations.get(a), dict)}
        for unit in units:
            for entry in method.bank(unit):
                header, errors = load_record(root / entry['file'], 'HEADER', reads,
                                             lambda obj, e=entry: header_only(obj, e))
                headers[entry['file']] = header or {}
                metadata_errors[entry['file']] = errors
        attempt['phase'] = 'HEADERS_LOADED'
        write_json(output / 'attempt.json', attempt)
        for index, unit in enumerate(units):
            calls['plans'] += 1
            planned = method.given_plans(unit, headers, operations)
            write_json(output / f'plans/unit{index:02d}.json', dict(unit=unit, local_plans=planned))
            plans.append(planned)
        attempt.update(phase='PLANS_SAVED', plans_saved=16, calls=dict(calls))
        write_json(output / 'attempt.json', attempt)
        # No payload cache loader is called before all 16 geometry plans save.
        for index, unit in enumerate(units):
            attempt.update(phase='PAYLOAD_READS', calls=dict(calls))
            attempt['unit_states'][index] = 'STARTED'
            write_json(output / 'attempt.json', attempt)
            cache, blocked, technical, readable = {}, set(), [], 0
            for entry in method.bank(unit):
                technical.extend(metadata_errors[entry['file']])
                loaded, errors = load_record(root / entry['file'], 'PAYLOAD', reads,
                    lambda obj, e=entry: cache_from_payload(obj, e, headers[e['file']]))
                technical.extend(errors)
                if loaded is None:
                    continue
                readable += 1
                cells, cell_errors = loaded
                technical.extend(dict(kind='CELL', path=entry['file'], **error) for error in cell_errors)
                for eid, cell in cells.items():
                    if eid in cache or eid in blocked:
                        blocked.add(eid)
                        cache.pop(eid, None)
                        technical.append(dict(kind='DUPLICATE_CACHE_ADDRESS', evidence_id=eid))
                    else:
                        cache[eid] = cell
            local, memo = [], {}
            for q, plan in enumerate(plans[index]):
                calls['local'] += 1
                value = cdc.read_time_local(plan, cache, memo=memo)
                value['geometry_scope'] = 'GIVEN_CROP_MAP'
                write_json(output / f'local/unit{index:02d}/q{q}.json', value)
                local.append(value)
            calls['aggregate'] += 1
            final = cdc.aggregate_time(local)
            final.update(unit=unit, geometry_scope='GIVEN_CROP_MAP', path_context='PREDECLARED_GIVEN_CROP_MAP')
            write_json(output / f'final/unit{index:02d}.json', final)
            input_errors = list(ledger_errors)
            if not readable:
                input_errors.append(dict(kind='BITS_INPUT', reason='no readable bits files in fixed unit'))
            evidence = method.evidence_from_aggregate(final, unit, input_errors=input_errors, technical_errors=technical)
            evidence_path = f'evidence/unit{index:02d}.json'
            write_json(output / evidence_path, evidence)
            for offset, claim in enumerate(claims):
                calls['claims'] += 1
                decision = qmatch.match_claim(evidence, claim['claim_id'], claim['bits'])
                decisions[2*index+offset] = dict(decision, status='RETURNED', geometry_scope='GIVEN_CROP_MAP', evidence_file=evidence_path)
            write_json(output / 'decisions.json', decisions)
            attempt['unit_states'][index] = 'RETURNED'
            attempt['calls'] = dict(calls)
            write_json(output / 'attempt.json', attempt)
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        for i, state in enumerate(attempt['unit_states']):
            if state == 'STARTED':
                attempt['unit_states'][i] = 'FAILED_CONSUMED'
    comparisons = sum(d['comparisons'] for d in decisions)
    returned = [d for d in decisions if d['status'] == 'RETURNED']
    if comparisons > 1024 or (not failure and (calls != dict(plans=16, local=48, aggregate=16, claims=32)
                                               or len(reads) != 386 or len(returned) != 32)):
        failure = failure or 'internal fixed denominator/budget violation'
    status = 'FAILED_CONSUMED' if failure else 'COMPLETED_WITH_INPUT_UNRESOLVED' if any(
        d['verdict'] == 'INPUT_UNRESOLVED' for d in returned) else 'COMPLETED'
    result = dict(status=status, consumed=True, scope=qmatch.SCOPE, geometry_scope='GIVEN_CROP_MAP', failure=failure,
                  limits=method.LIMITS, calls=calls, planned_decisions=32, returned_decisions=len(returned),
                  unreturned_decisions=32-len(returned), comparisons=comparisons, max_comparisons=1024,
                  verdict_counts=dict(Counter(d['verdict'] for d in returned)), logical_inputs_planned=194,
                  unique_logical_inputs_attempted=len({r['path'] for r in reads}), file_read_attempts=len(reads),
                  reads_by_kind={kind: dict(Counter(r['status'] for r in reads if r['kind'] == kind))
                                 for kind in ('TASK', 'LEDGER', 'HEADER', 'PAYLOAD')},
                  planned_shared_final_slot_records=512, new_physical_calls=0, truth_used=False,
                  independent_statistical_samples=None, scientific_pass=None,
                  elapsed_seconds=time.time()-attempt['started_at'],
                  limitations=['given experiment alignment, not blind time recovery',
                               'four seen development sources; two keys and claims share observations',
                               'q1 is slot0 sensitivity; no calibrated FPR or authentication claim',
                               'unreadable/mismatching P0 cannot be rescued by OFF nonMATCH'])
    attempt.update(status=status, calls=calls, finished_at=time.time(), failure=failure)
    write_json(output / 'decisions.json', decisions)
    write_json(output / 'input_reads.json', reads)
    write_json(output / 'attempt.json', attempt)
    write_json(output / 'result.json', result)
    return result
