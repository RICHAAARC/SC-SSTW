"""One user-run P0 -> inherited A/B -> V3 fixed media receiver package."""
from pathlib import Path
import json
import time
from . import b_conditional_time_validation_v1_records as records


def run(config,output=None,backend_factories=None,receiver_backend_factory=None,phase_runner=None,receiver_runner=None):
    output=Path(output or config['output']).expanduser().resolve()
    state=records.output_state(output)
    if state=='CONSUMED': return records.recover(output)
    if state=='CONFLICT': return dict(status='OUTPUT_CONFLICT',physical_calls_this_invocation=0)
    check=records.preflight(config,output)
    if check['status']!='PREFLIGHT_READY':return check
    output.mkdir(parents=True,exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED_PACKAGE',source_id=config['source_id'],
                           terminal_relative=config['terminal_relative'],budgets=records.budgets(),
                           denominators=records.denominators(config),rerun='READ_ONLY_ALL_PHASES'),stream,indent=2)
    except FileExistsError:return records.recover(output)
    journal=records.Journal(output);results={};errors={};interrupted=False;start=time.monotonic()
    factories=backend_factories or (None,None)
    material_output=output/'materialization'
    try:
        if phase_runner is None:
            from .b_minimal_l_materialization_v1_experiment import run_stage
            phase_runner=run_stage
        def materialize(stage,factory):
            return phase_runner(config['source'],material_output,stage,model_id=config['model_id'],backend_factory=factory,
                                protocol=records.protocol(config),receiver_codec_calls=17)
        results['A']=journal.call('A',lambda:materialize('A',factories[0]))
        if results['A'].get('errors',{}).get('interrupted'):
            journal.rows['A']['status']='INTERRUPTED';journal.save()
            raise KeyboardInterrupt('A interrupted; package remains consumed')
        dependencies=records.material.ready_B(material_output)
        if dependencies:
            errors['B_dependency']=dependencies
        else:
            results['B']=journal.call('B',lambda:materialize('B',factories[1]))
            if results['B'].get('errors',{}).get('interrupted'):
                journal.rows['B']['status']='INTERRUPTED';journal.save()
                raise KeyboardInterrupt('B interrupted; package remains consumed')
            # Native tolerance failure is retained. Only an actually finite saved native and
            # available QIM file can enter the already-budgeted source probe/decode of receiver43.
            media=material_output/'B/QIM.mp4'
            try:
                native=records.load(material_output/'B/native.json')
                if native.get('finite_complete') is not True or native.get('frames')!=181:
                    raise ValueError('finite complete 181-frame native unavailable')
                if not media.is_file() or media.stat().st_size==0:
                    raise FileNotFoundError('QIM media unavailable; no materialization retry')
            except (OSError,ValueError,TypeError,AttributeError) as error:
                errors['receiver_dependency']=str(error)
            else:
                if receiver_runner is None:
                    from .b_conditional_time_output_v1_experiment import run as receiver_runner
                receiver_config=dict(entry='media',media=str(media),truth=None,
                                     source_note='new time packet from '+config['source_id']+' existing DCT2 P0; materialization failures remain separate')
                results['receiver']=journal.call('receiver',lambda:receiver_runner(receiver_config,output/'receiver',backend_factory=receiver_backend_factory))
                if results['receiver'].get('status')=='INTERRUPTED':
                    journal.rows['receiver']['status']='INTERRUPTED';journal.save()
                    raise KeyboardInterrupt('receiver interrupted; package remains consumed')
    except BaseException as error:
        interrupted=not isinstance(error,Exception)
        errors['package']=f'{type(error).__name__}: {error}'
    journal.terminalize()
    counts={}
    receiver_reserved=(output/'receiver/attempt.json').exists()
    for name,folder in (('A',material_output/'A'),('B',material_output/'B'),('receiver',output/'receiver')):
        if name=='receiver' and not receiver_reserved:
            counts[name]=dict(status='NOT_EXECUTED',operations={k:dict(planned=v,started=0,returned=0) for k,v in records.budgets()['receiver'].items()})
            continue
        try:counts[name]=records.load(folder/'counts.json')
        except (OSError,ValueError) as error:
            if name!='receiver' and journal.rows[name]['status']=='DEPENDENCY_UNRESOLVED':
                budget=records.budgets()[{'A':'materialization_A','B':'materialization_B','receiver':'receiver'}[name]]
                counts[name]=dict(status='NOT_EXECUTED',operations={k:dict(planned=v,started=0,returned=0) for k,v in budget.items()})
            else:counts[name]=dict(status='UNREADABLE_COUNTS',error=str(error))
    records.dump(output/'counts.json',counts)
    if not receiver_reserved:
        planned=records.receiver.plan('media')
        for row in planned.values():row.update(status='DEPENDENCY_UNRESOLVED',error='receiver attempt not reserved; see exit_reason; no later execution')
        exit_reason=dict(receiver_attempt_reserved=False,phase=dict(journal.rows['receiver']),
                         receiver_result=results.get('receiver'),package_errors=dict(errors))
        records.dump(output/'receiver_not_executed.json',dict(rows=planned,denominators=records.receiver.denominators(),exit_reason=exit_reason))
    b=results.get('B',{});receiver=results.get('receiver',{})
    result=dict(status='PACKAGE_INTERRUPTED' if interrupted else 'PACKAGE_FIXED_ATTEMPT_TERMINATED',
                source_id=config['source_id'],phases=journal.rows,stage_results=results,errors=errors,
                materialization_native_status=b.get('native_status'),
                materialization_errors={stage:value.get('errors',{}) for stage,value in results.items() if stage in ('A','B')},
                receiver_evaluation_status=receiver.get('evaluation_status'),
                budgets=records.budgets(),denominators=records.denominators(config),wall_seconds=time.monotonic()-start,
                scientific_pass=None,claim_ceiling='Receiver conditional evidence is separate from materialization target/numeric failures; no identity/null/general reliability claim')
    records.dump(output/'result.json',result)
    return result
