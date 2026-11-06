"""Stdlib joint reservation/closure. Optional injected stdlib modules support bootstrap."""
from collections import Counter
import json
import os
from pathlib import Path
import time

DENOMINATORS = dict(outer_operations=629, time_operations=78, payload_operations=551,
    time_nodes=1644, time_edges=1634, time_hard_bits=1035720, raw_bits=28800,
    vote_uses=230400, physical_coefficients=153600, local_slots=2400, final_slots=800,
    per_strategy_final=160, independent_sources=1)
INNER = dict(VAE_load=3, E41=120, project=120, keyed_decode=180, codec_encode=12,
    codec_probe=22, RGB_decode=22, Y_decode=10, conditional_DiT=50, unconditional_DiT=50,
    payload_gradient=25, D181=1, baseline_head=46, replay_head=46,
    solver_K_upper=4200, solver_KT_upper=4200, prediction_K=1,
    writer_BCH=1790, raw24_decode_upper=24660, template_encode_upper=82200,
    hard_bit_comparison_upper=10357200)


def modules(time_protocol=None, payload_records=None):
    if time_protocol is None:
        from main.tube_state import b_joint_time_relative_validation_v1 as time_protocol
    if payload_records is None:
        from . import b_relative_time_payload_records_v1 as payload_records
    return time_protocol, payload_records


def read(path):
    def invalid(value): raise ValueError('non-finite JSON number: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name+'.pending')
    with pending.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=1, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def reserve(output, config, *, time_protocol=None, payload_records=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()): return False
    tp, payload = modules(time_protocol, payload_records)
    receipt = dict(status='RESERVED', consumed=True, reserved_at=time.time(), config=config,
        denominators=DENOMINATORS, inner_budget=INNER, rerun='READ_ONLY_INCLUDING_POST',
        time_operations=tp.operation_plan(config), payload_operations=payload.operation_plan(),
        time_slots=[dict(unit=u, nodes=[dict(received_frame=i, status='NOT_STARTED', value=None)
                                       for i in range(u['frames'])],
                        edges=[dict(received_edge=i, status='NOT_STARTED', value=None)
                               for i in range(u['edges'])]) for u in tp.units(config)],
        payload_slots=[dict(item, slots=payload.empty_slots()) for item in payload.layout()],
        raw_slots=payload.raw_slots())
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x', encoding='utf-8') as stream:
            json.dump(receipt, stream, ensure_ascii=False, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
    except FileExistsError: return False
    write(output/'protocol.json', config)
    # Both fixed accounts are already in the exclusive root reservation.
    time_output = output/'time_chain'
    write(time_output/'attempt.json', dict(status='RESERVED', consumed=True))
    write(time_output/'protocol.json', config)
    write(time_output/'unit_statuses.json', {u['id']: u for u in tp.units(config)})
    payload.reserve(output/'payload', dict(source=str(time_output), source_id=config['source']['id'],
        model_id=config['model_id'], output=str(output/'payload')))
    return True


def close_time(output, reason=None):
    """Close saved receipts only; never call a time receiver or compare candidates."""
    output = Path(output); time_output = output/'time_chain'; attempt = read(output/'attempt.json')
    rows = {r['id']: r for r in attempt['time_operations']}; errors = []
    for stage in ('prepare', 'solve', 'replay', 'receive'):
        path = time_output/'stages'/stage/'events.jsonl'
        if path.is_file():
            for index, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    event = json.loads(line); rows[event['id']].update(event)
                except (ValueError, KeyError, TypeError) as error:
                    errors.append(dict(stage=stage, line=index, error=str(error)))
    for row in rows.values():
        if row['status'] == 'STARTED': row['status'] = 'INTERRUPTED_CONSUMED'
        if reason and row['status'] == 'NOT_STARTED': row['dependency_status'] = 'DEPENDENCY_UNRESOLVED'
    terminal = attempt['time_slots']
    for item in terminal:
        unit = item['unit']; uid = unit['id']; state = rows[uid+'/relative_compare']['status']
        value = None
        if state == 'RETURNED':
            try:
                value = read(time_output/'blind'/uid/'relative.json')
                if ([len(value['original_f42']['nodes']), len(value['conditional_nodes']),
                     len(value['original_f42']['edges']), len(value['conditional_edges'])]
                        != [unit['frames'], unit['frames'], unit['edges'], unit['edges']]):
                    raise ValueError('fixed node/edge lengths required')
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(dict(unit=uid, error=str(error))); value = None
                state = 'UNRESOLVED_RECEIPT'
        item['execution_status'] = state
        for kind, name in (('nodes', 'conditional_nodes'), ('edges', 'conditional_edges')):
            for i, slot in enumerate(item[kind]):
                slot.update(status='SAVED' if value is not None else state,
                    value=dict(original=value['original_f42'][kind][i], conditional=value[name][i])
                    if value is not None else None)
                if reason and value is None: slot['dependency_status'] = 'DEPENDENCY_UNRESOLVED'
    write(time_output/'ledger.json', list(rows.values()))
    write(time_output/'blind_terminated.json', dict(status='BLIND_TERMINATED', time=time.time(), failure=reason))
    result = dict(status='ALL_TIME_SLOTS_TERMINAL', saved_at=time.time(), units=terminal,
                  accounting_errors=errors, failure=reason, nodes=1644, edges=1634)
    write(output/'time_terminal_slots.json', result)
    return result


def close_blind(output, reason=None, *, payload_records=None):
    output = Path(output)
    if (output/'joint_blind_terminal.json').is_file(): return read(output/'joint_blind_terminal.json')
    if payload_records is None:
        from . import b_relative_time_payload_records_v1 as payload_records
    payload = payload_records
    time_terminal = close_time(output, reason)
    if not (output/'payload/blind_terminal.json').is_file():
        payload.finalize_blind(output/'payload', reason)
    result = dict(status='JOINT_BLIND_TERMINAL', saved_at=time.time(), failure=reason,
        time='time_terminal_slots.json', payload='payload/blind_terminal.json',
        denominators=DENOMINATORS, time_accounting_errors=time_terminal['accounting_errors'],
        both_posts_not_started=not (output/'post_started.json').exists(), scientific_pass=None)
    write(output/'joint_blind_terminal.json', result)
    return result


def finish(output, failure=None):
    output = Path(output); attempt = read(output/'attempt.json')
    time_path, payload_path = output/'time_chain/ledger.json', output/'payload/state.json'
    time_rows = read(time_path) if time_path.is_file() else attempt['time_operations']
    payload_rows = read(payload_path) if payload_path.is_file() else attempt['payload_operations']
    counters = dict(time=dict(planned=78, status_counts=dict(Counter(r['status'] for r in time_rows))),
                    payload=dict(planned=551, status_counts=dict(Counter(r['status'] for r in payload_rows.values()))))
    posts = read(output/'posts.json') if (output/'posts.json').is_file() else {}
    complete = (not failure and all(r['status']=='RETURNED' for r in time_rows)
        and all(r['status']=='RETURNED' for r in payload_rows.values())
        and all(posts.get(name, {}).get('status')=='RETURNED' for name in ('time', 'payload')))
    result = dict(status='COMPLETED' if complete else 'STOPPED_FIXED_ATTEMPT', consumed=True,
        failure=failure, denominators=DENOMINATORS, operations=counters, posts=posts,
        scientific_pass=None, evidence_ceiling='single new content; conditional time and payload; no population or FPR claim')
    write(output/'result.json', result)
    attempt.update(status=result['status'], finished_at=time.time())
    write(output/'attempt.json', attempt)
    return result


def recover(output):
    output = Path(output); result = dict(status='READ_ONLY', output=str(output), consumed=True, saved={})
    for name in ('result.json', 'joint_blind_terminal.json', 'posts.json', 'preparation_failure.json',
                 'time_chain/evaluation.json', 'payload/evaluation_summary.json', 'payload/resources.json',
                 'payload/quality.json', 'payload/raw_terminal_summary.json'):
        if (output/name).is_file():
            try: result['saved'][name] = read(output/name)
            except (OSError, ValueError) as error: result['saved'][name] = dict(read_error=str(error))
    return result
