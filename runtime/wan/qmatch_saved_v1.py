"""Stdlib-only, once-consumed QMatch runner over exactly 53 logical inputs."""
from collections import Counter
import json
import math
import os
from pathlib import Path
import time
import uuid

from main.tube_state import qmatch_v1 as method

LIMITS = dict(source_packages=2, views_per_source=5, strategy_view_inputs=50,
              claims_per_input=2, decisions=100, max_slot_query_comparisons=3200,
              logical_input_records=53)
CLAIMS = (('q0', 'A6D39C5E'), ('q1', '26D39C5E'))


def read_json(path):
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return value
    def constant(text):
        raise ValueError('invalid JSON constant: ' + text)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_float=number,
                      parse_constant=constant, object_pairs_hook=pairs)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def claims_from_task(task):
    if not isinstance(task, dict) or task.get('scope') != method.SCOPE or task.get('bit_order') != 'MSB_FIRST':
        raise ValueError('conditional MSB-first task required')
    if task.get('limits') != LIMITS or task.get('q1_rule') != 'flip_global_slot_0_only':
        raise ValueError('fixed adopted budget and slot0-only comparison required')
    claims = task.get('claims')
    if not isinstance(claims, list) or len(claims) != 2:
        raise ValueError('exactly two predeclared claims required')
    for claim, (name, hexadecimal) in zip(claims, CLAIMS):
        bits = [int(c) for c in f'{int(hexadecimal, 16):032b}']
        if (not isinstance(claim, dict) or claim.get('claim_id') != name or claim.get('hex') != hexadecimal
                or claim.get('bits') != bits or any(type(b) is not int for b in claim.get('bits', []))):
            raise ValueError('claim differs from adopted literal message/bit order')
    return [dict(claim_id=c['claim_id'], bits=list(c['bits'])) for c in claims]


def recover(output):
    """Existing run directories are read-only, including failed/interrupted runs."""
    output = Path(output)
    saved = {}
    for name in ('attempt.json', 'result.json'):
        try:
            saved[name] = read_json(output / name)
        except (OSError, ValueError) as error:
            saved[name] = dict(error=f'{type(error).__name__}: {error}')
    return dict(status='READ_ONLY_CONSUMED', output=str(output), consumed=True, saved=saved)


def _load(path, kind, reads):
    record = dict(path=str(path), kind=kind, status='STARTED')
    reads.append(record)
    try:
        value = read_json(path)
        record['status'] = 'RETURNED'
        return value, []
    except (OSError, ValueError) as error:
        reason = f'{type(error).__name__}: {error}'
        record.update(status='UNREADABLE', reason=reason)
        return None, [dict(kind=kind, reason=reason)]


def run_once(query_task, input_roots, output):
    """Reserve before reading inputs. Do not replay or repair an existing attempt."""
    output = Path(output)
    if output.exists():
        return recover(output)
    roots = {k: Path(v) for k, v in input_roots.items()}
    if set(roots) != {'r01', 'r02'}:
        raise ValueError('two explicit r01/r02 payload roots required')
    resolved_output = output.resolve()
    for path in [*roots.values(), Path(query_task)]:
        resolved_input = path.resolve()
        if (resolved_output == resolved_input or resolved_output in resolved_input.parents
                or resolved_input in resolved_output.parents):
            raise ValueError('output must be separate from saved inputs and task')
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return recover(output)
    units = method.fixed_units()
    attempt = dict(status='STARTED', consumed=True, started_at=time.time(), limits=LIMITS,
                   unit_states=['NOT_STARTED'] * 50, rerun='READ_ONLY')
    decisions = [dict(scope=method.SCOPE, unit=unit, claim_id=name, status='NOT_STARTED',
                      verdict=None, comparisons=0) for unit in units for name, _ in CLAIMS]
    reads, failure = [], None
    try:
        write_json(output / 'attempt.json', attempt)
        write_json(output / 'decisions.json', decisions)
        task, task_errors = _load(Path(query_task), 'QUERY_TASK', reads)
        if task_errors:
            raise ValueError(task_errors[0]['reason'])
        claims = claims_from_task(task)
        # Persist supplied claims before opening either state or any blind final.
        write_json(output / 'query_task.json', task)
        states = {source: _load(roots[source] / 'state.json', 'RECEIPT_INPUT', reads)
                  for source in ('r01', 'r02')}
        for index, unit in enumerate(units):
            attempt['unit_states'][index] = 'STARTED'
            write_json(output / 'attempt.json', attempt)
            source, stem, strategy = (unit[k] for k in ('source_package', 'input_id', 'strategy'))
            path = roots[source] / 'final' / stem / (strategy + '.json')
            final, final_errors = _load(path, 'FINAL_INPUT', reads)
            state, state_errors = states[source]
            evidence = method.prepare_saved(final, state, unit, input_errors=final_errors + state_errors)
            evidence_path = f'evidence/{source}/{stem}/{strategy}.json'
            write_json(output / evidence_path, evidence)
            for offset, claim in enumerate(claims):
                result = method.match_claim(evidence, claim['claim_id'], claim['bits'])
                decisions[2 * index + offset] = dict(result, status='RETURNED', evidence_file=evidence_path)
            write_json(output / 'decisions.json', decisions)
            attempt['unit_states'][index] = 'RETURNED'
            write_json(output / 'attempt.json', attempt)
            del final, evidence
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        for i, state in enumerate(attempt['unit_states']):
            if state == 'STARTED':
                attempt['unit_states'][i] = 'INTERRUPTED_CONSUMED' if isinstance(error, KeyboardInterrupt) else 'FAILED_CONSUMED'
    comparisons = sum(d['comparisons'] for d in decisions)
    returned = [d for d in decisions if d['status'] == 'RETURNED']
    status = 'FAILED_CONSUMED' if failure else 'COMPLETED_WITH_INPUT_UNRESOLVED' if any(
        d['verdict'] == 'INPUT_UNRESOLVED' for d in returned) else 'COMPLETED'
    if comparisons > 3200 or (not failure and (len(returned) != 100 or len(reads) != 53)):
        failure = failure or 'internal fixed denominator/budget violation'
        status = 'FAILED_CONSUMED'
    result = dict(status=status, consumed=True, scope=method.SCOPE, failure=failure,
                  planned_decisions=100, returned_decisions=len(returned), unreturned_decisions=100-len(returned),
                  comparisons=comparisons, max_comparisons=3200, verdict_counts=dict(Counter(d['verdict'] for d in returned)),
                  logical_inputs_planned=53, logical_input_read_attempts=len(reads),
                  readable_input_records=sum(r['status'] == 'RETURNED' for r in reads),
                  planned_shared_final_slot_records=1600, independent_statistical_samples=None,
                  new_physical_calls=0, new_aggregation_calls=0, truth_used=False,
                  scientific_pass=None, elapsed_seconds=time.time()-attempt['started_at'],
                  limitations=['known-answer development; q1 is algebraic sensitivity only',
                               'MATCH does not authenticate existence, time, or source',
                               'MISMATCH does not prove absence; partial-only contradiction is not reliable rejection',
                               'related views and two claims share saved evidence; no FPR estimate'])
    attempt.update(status=status, finished_at=time.time(), failure=failure)
    write_json(output / 'decisions.json', decisions)
    write_json(output / 'input_reads.json', reads)
    write_json(output / 'attempt.json', attempt)
    write_json(output / 'result.json', result)
    return result
