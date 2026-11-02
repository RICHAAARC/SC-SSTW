"""One consumed n01 A/B -> ten Y views -> five payload views -> three posts."""
from pathlib import Path
import json
import time
from . import b_native_y_source_confirmation_v1_records as rec


def run(config, output=None, *, backend_factories=None, receiver_backend_factory=None, payload_backend_factory=None,
        phase_runner=None, receiver_runner=None, payload_runner=None, time_evaluator=None, payload_evaluator=None):
    output = Path(output or config['output']).expanduser().resolve()
    state = rec.output_state(output)
    if state == 'CONSUMED': return rec.recover(output)
    if state == 'CONFLICT': return dict(status='OUTPUT_CONFLICT', physical_calls_this_invocation=0)
    check = rec.preflight(config, output)
    if check['status'] != 'PREFLIGHT_READY': return check
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED_WHOLE_CHAIN', source_id=rec.SOURCE_ID,
                plan=rec.plan(), budgets=rec.budgets(), denominators=rec.denominators(),
                rerun='METADATA_ONLY_NO_PHYSICAL_OR_POST'), stream, indent=2, allow_nan=False)
    except FileExistsError: return rec.recover(output)
    journal = rec.Journal(output)
    results, errors, measurements = {}, {}, {}
    start, interrupted = time.monotonic(), False
    material_output = output/'time_chain/materialization'
    time_output = output/'time_chain/receiver'
    payload_output = output/'payload'
    from .b_new_source_chain_confirmation_v1_experiment import StageMeasurements

    def measured(name, function):
        monitor = StageMeasurements()
        try:
            with monitor: return journal.call(name, function)
        finally:
            measurements[name] = monitor.values
            rec.dump(output/'stage_measurements.json', measurements)

    def failed(key, error):
        nonlocal interrupted
        errors[key] = f'{type(error).__name__}: {error}'
        interrupted |= not isinstance(error, Exception)

    try:
        if phase_runner is None:
            from .b_minimal_l_materialization_v1_experiment import run_stage as phase_runner
        if receiver_runner is None:
            from .b_native_y_source_confirmation_v1_time import run as receiver_runner
        if payload_runner is None:
            from .b_time_payload_cdc32_v1_experiment import run as payload_runner
        factories = backend_factories or (None, None)
        protocol = dict(source_id=rec.SOURCE_ID, terminal_relative=rec.TERMINAL, frames=181,
                        global_start=0, model_id=config['model_id'])
        for i, stage in enumerate(('A', 'B')):
            if stage == 'B':
                dependency = rec.material.ready_B(material_output)
                if dependency:
                    errors['B_dependency'] = dependency
                    break
            results[stage] = measured(stage, lambda stage=stage, i=i: phase_runner(config['source'], material_output, stage,
                model_id=config['model_id'], backend_factory=factories[i], protocol=protocol, receiver_codec_calls=44))
            if journal.rows[stage]['status'] == 'INTERRUPTED':
                raise KeyboardInterrupt(stage+' interrupted; fixed whole attempt consumed')
        if 'B' in results:
            native = rec.load(material_output/'B/native.json')
            if native.get('finite_complete') is not True or native.get('frames') != 181:
                raise ValueError('finite complete 181-frame native unavailable')
            media = dict(QIM=str(material_output/'B/QIM.mp4'), BASE=str(material_output/'A/BASE.mp4'))
            for arm, name in media.items():
                path = Path(name)
                if not path.is_file() or path.stat().st_size == 0:
                    raise FileNotFoundError(arm+' first-generation media unavailable; no retry')
            results['time'] = measured('time', lambda: receiver_runner(dict(media=media), time_output,
                backend_factory=receiver_backend_factory))
            if journal.rows['time']['status'] == 'INTERRUPTED':
                raise KeyboardInterrupt('time interrupted; no payload continuation')
            time_rows = rec.load(time_output/'state.json')
            rec.require_time_blind(time_output, time_rows)
            results['payload'] = measured('payload', lambda: payload_runner(dict(source=str(output/'time_chain'),
                output=str(payload_output), model_id=config['model_id']), backend_factory=payload_backend_factory, defer_post=True))
            interrupted |= journal.rows['payload']['status'] == 'INTERRUPTED'
    except BaseException as error:
        failed('execution', error)
    def terminalize_safely(key, phases):
        try:
            journal.terminalize(phases)
            return True
        except BaseException as error:
            failed(key, error)
            # Preserve consumption in memory even if the state receipt is unwritable.
            for phase in phases:
                row = journal.rows[phase]
                if row['status'] == 'NOT_STARTED':
                    row.update(status='DEPENDENCY_UNRESOLVED', error='whole attempt consumed; no later execution')
                elif row['status'] == 'STARTED':
                    row.update(status='INTERRUPTED', error='unfinished consumed phase')
            return False
    physical_terminal_saved = terminalize_safely('physical_terminal_state_save', ('A', 'B', 'time', 'payload'))
    blind_saved, stages, stage_rows, counts = False, {}, {}, {}
    stage_budgets = dict(A=rec.budgets()['materialization_A'], B=rec.budgets()['materialization_B'],
                        time=rec.time_budgets(), payload=rec.payload.budgets())
    for stage, folder in rec.FOLDERS.items():
        stages[stage], receipt_errors = rec.read_metadata(output/folder)
        started = journal.rows[stage]['started']
        if started:
            for name in ('state', 'counts', 'resources', 'result'):
                if name in receipt_errors: errors[stage+'/'+name] = receipt_errors[name]
        resource = stages[stage].get('resources', {})
        if resource.get('error') or resource.get('completion_errors'):
            errors[stage+'/resource'] = resource.get('error') or resource['completion_errors']
        counts[stage] = stages[stage].get('counts', dict(status='UNRESOLVED_RECEIPT' if started else 'NOT_EXECUTED',
            operations={k: dict(planned=v, started=None if started else 0, returned=None if started else 0)
                        for k,v in stage_budgets[stage].items()}))
        if stage in ('time', 'payload'):
            fixed = rec.time_plan() if stage == 'time' else rec.payload.plan()
            actual = stages[stage].get('state', {})
            for key, row in fixed.items():
                if key in actual:
                    row.update(actual[key])
                    if row['status'] in ('NOT_STARTED','STARTED'):
                        row['status'] = 'UNRESOLVED_RECEIPT'
                else:
                    row.update(status='UNRESOLVED_RECEIPT' if started else 'DEPENDENCY_UNRESOLVED',
                               started=None if started else 0, returned=None if started else 0)
            stage_rows[stage] = fixed
    try:
        rec.dump(output/'counts.json', counts)
        rec.dump(output/'fixed_slots.json', dict(denominators=rec.denominators(),
            time=rec.time_fixed_slots(stage_rows['time']), payload=dict(denominators=rec.payload.denominators(),
            rows=stage_rows['payload'])))
        rec.dump(output/'blind_summary.json', dict(phases=journal.rows, errors=dict(errors), denominators=rec.denominators()))
        if not physical_terminal_saved:
            raise ValueError('physical phase terminal receipt failed; no post permitted')
        rec.dump(output/'blind_terminal.json', dict(status='ALL_CHAIN_BLIND_SLOTS_TERMINAL',
            phases={k: journal.rows[k] for k in ('A','B','time','payload')}, fixed_rows=stage_rows,
            denominators=rec.denominators(), errors=dict(errors)))
        rec.require_whole_blind(output)
        blind_saved = True
    except BaseException as error: failed('whole_blind_save', error)
    timing = payload = None
    summary = dict(status='EVIDENCE_INCOMPLETE', evidence_complete=False)
    if blind_saved:
        from . import b_native_y_source_confirmation_v1_evaluation as post
        truth = time_output/'private/attack_truth.json'
        def time_post():
            nonlocal timing
            rec.require_whole_blind(output)
            value = (time_evaluator or post.evaluate_time)(time_output, stages['time'].get('state', {}), truth, whole_output=output)
            timing = value
            rec.dump(output/'time_evaluation.json', value)
            return value
        def payload_post():
            nonlocal payload
            rec.require_whole_blind(output)
            terminal = rec.load(payload_output/'blind_terminal.json')
            if terminal.get('status') != 'ALL_BLIND_SLOTS_TERMINAL':
                raise ValueError('payload blind terminal unavailable')
            fixture = post.private_message(config['source'], output)
            evaluator = payload_evaluator
            if evaluator is None:
                from .b_time_payload_cdc32_v1_evaluation import evaluate as evaluator
            value = evaluator(payload_output, truth, message_bits=fixture['message_bits'], message_source=fixture['message_source'])
            payload = value
            rec.dump(output/'payload_evaluation.json', value)
            return value
        for phase, dependency, fn in (('time_post', 'time', time_post), ('payload_post', 'payload', payload_post)):
            if journal.rows[dependency]['started'] and not (phase == 'payload_post' and journal.rows['time_post']['status'] == 'INTERRUPTED'):
                try: journal.call(phase, fn)
                except BaseException as error: failed(phase, error)
            else: terminalize_safely(phase+'_terminal_state_save', (phase,))
        def summarize():
            nonlocal summary
            summary = post.summarize(timing, payload, results, errors)
            rec.dump(output/'evaluation.json', summary)
            return summary
        try: journal.call('summary', summarize)
        except BaseException as error:
            failed('summary', error)
            summary.update(evidence_complete=False)
            if summary.get('status') != 'COUNTEREXAMPLE': summary['status'] = 'EVIDENCE_INCOMPLETE'
    if not terminalize_safely('final_terminal_state_save', rec.PHASES):
        summary['evidence_complete'] = False
        if summary.get('status') != 'COUNTEREXAMPLE': summary['status'] = 'EVIDENCE_INCOMPLETE'
    resources = dict(total_wall_seconds=time.monotonic()-start, scope='original per-stage resource limits; no new full-chain cap',
        stages={k: dict(native=stages[k].get('resources', {}), observed=measurements.get(k, {})) for k in rec.FOLDERS})
    result = dict(status='PACKAGE_INTERRUPTED' if interrupted else 'PACKAGE_FIXED_ATTEMPT_TERMINATED', source_id=rec.SOURCE_ID,
        evaluation_status=summary['status'], errors=errors, phases=journal.rows, scientific_pass=None,
        stage_results={k: {f:v.get(f) for f in ('status','errors','native_status','solver_status')} for k,v in results.items()},
        budgets=rec.budgets(), denominators=rec.denominators(), blind_terminal_saved=blind_saved,
        known_counterexamples=summary.get('counterexamples', []))
    def incomplete_receipt(name, value):
        try: rec.dump(output/(name+'.json'), value)
        except BaseException as error:
            failed(name+'_save', error)
            summary['evidence_complete'] = False
            if summary['status'] != 'COUNTEREXAMPLE': summary['status'] = 'EVIDENCE_INCOMPLETE'
            result.update(status='FINAL_RECORD_SAVE_FAILED', evaluation_status=summary['status'], evidence_complete=False)
    incomplete_receipt('resources', resources)
    incomplete_receipt('post_status', dict(status=summary['status'], evidence_complete=summary.get('evidence_complete'),
        phases={k:journal.rows[k] for k in ('time_post','payload_post','summary')}, errors=dict(errors)))
    result['evidence_complete'] = bool(summary.get('evidence_complete'))
    incomplete_receipt('result', result)
    return result
