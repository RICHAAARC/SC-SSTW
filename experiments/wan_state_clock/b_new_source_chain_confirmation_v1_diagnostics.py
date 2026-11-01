"""Bounded execution-receipt display only. No files, science, post or model calls.

The builder embeds this same stdlib-free code for source-free notebook recovery.
"""


def _receipt_object(value):
    return value if isinstance(value,dict) else {}


def operation_not_executed(row):
    row=_receipt_object(row)
    # Explicit operation state is evidence; a fabricated/missing fallback is not.
    if row.get('status') not in ('NOT_STARTED','DEPENDENCY_UNRESOLVED','NOT_EXECUTED'):return False
    return all(k not in row or type(row[k]) is int and row[k]==0 for k in ('started','returned'))


def _count_rows(value):
    value=_receipt_object(value)
    return _receipt_object(value.get('operations',value))


def execution_diagnostics(result,counts=None,stages=None,phase_rows=None):
    result=_receipt_object(result);stages=_receipt_object(stages);counts=_receipt_object(counts)
    chain=_receipt_object(_receipt_object(result.get('stage_results')).get('time_chain'))
    children=_receipt_object(chain.get('stage_results'))
    phases=_receipt_object(phase_rows) or _receipt_object(result.get('phases'))
    stage_results=dict(A=children.get('A'),B=children.get('B'),time=children.get('receiver'),
                       payload=_receipt_object(result.get('stage_results')).get('payload'))
    first=None;phase_summary={};calls={}
    for name in ('A','B','time','payload'):
        stored=_receipt_object(stages.get(name));stage=_receipt_object(stage_results[name]) or _receipt_object(stored.get('result'))
        phase=_receipt_object(phases.get(name))
        phase_summary[name]={k:phase.get(k) for k in ('status','started','returned')}
        phase_summary[name]['result_status']=stage.get('status',phase.get('result_status'))
        errors=_receipt_object(stage.get('errors'))
        if not errors:errors=_receipt_object(_receipt_object(chain.get('materialization_errors')).get(name))
        dependency={'B':'B_dependency','time':'receiver_dependency'}.get(name)
        if not errors and dependency in _receipt_object(chain.get('errors')):
            errors={'dependency':chain['errors'][dependency]}
        if not errors and phase.get('status') in ('FAILED','INTERRUPTED') and phase.get('error'):
            errors={'phase':phase['error']}
        if first is None and errors:
            key='stage' if 'stage' in errors else next(iter(errors))
            first=dict(stage=name,operation=str(key)[:80],error=str(errors[key])[:800],
                       phase_status=phase.get('status'),result_status=stage.get('status'))
        rows=_count_rows(counts.get(name)) or _count_rows(stored.get('counts')) or _count_rows(stage.get('counts'))
        calls[name]=dict(recorded_kinds=len(rows),started_nonzero={},unstarted=[],unknown=[])
        for key,value in list(rows.items())[:32]:
            row=_receipt_object(value);started=row.get('started');returned=row.get('returned');label=str(key)[:64]
            if type(started) is int and started>0:
                calls[name]['started_nonzero'][label]=dict(started=started,returned=returned if type(returned) is int else None)
            elif type(started) is int and started==0 and type(returned) is int and returned==0:
                calls[name]['unstarted'].append(label)
            else:calls[name]['unknown'].append(label)
    if first is None:
        errors=_receipt_object(result.get('errors'))
        if errors:
            key=next(iter(errors));first=dict(stage='package',operation=str(key)[:80],error=str(errors[key])[:800])
    if first and ('insufficient space for FP32 native feature cache' in first['error'] or 'disk cache capacity' in first['error']):
        first.update(failure_kind='DISK_CACHE_CAPACITY_CHECK',
                     interpretation='filesystem disk-cache check; not a CUDA or RAM allocation failure')
    for name in ('time_post','payload_post','summary'):
        phase=_receipt_object(phases.get(name))
        phase_summary[name]={k:phase.get(k) for k in ('status','started','returned','result_status')}
    phase=_receipt_object(phases.get('time'));rows=_count_rows(counts.get('time'))
    positive=any(any(type(r.get(k)) is int and r[k]>0 for k in ('started','returned'))
                 for r in rows.values() if isinstance(r,dict))
    unentered=(operation_not_executed(phase) and phase.get('started')==0 and not positive)
    receiver='NOT_EXECUTED' if unentered else 'EXECUTION_RECORDED' if phase.get('started')==1 or positive else 'UNKNOWN'
    zeros={k:0 for k in ('CORRECT','WRONG','UNRESOLVED','NOT_EXECUTED','UNKNOWN')}
    time_counts=dict(nodes=dict(zeros,NOT_EXECUTED=822),relations=dict(zeros,NOT_EXECUTED=817)) if unentered else None
    capacity=_receipt_object(_receipt_object(stages.get('A')).get('disk_cache_capacity'))
    return dict(earliest_stage_error=first,phase_states=phase_summary,operation_receipts=calls,
        receiver_execution=receiver,time_execution_counts=time_counts,
        time_counts_basis='READ_ONLY_EXECUTION_RECEIPTS_NOT_REEVALUATION' if unentered else None,
        disk_cache_capacity={k:capacity.get(k) for k in ('status','check_path','actual_free_bytes','channels',
            'shape_estimate_bytes','minimum_bytes','required_bytes')} if capacity else None,
        missing_capacity_measurements='not recorded; historical free/channels/required cannot be reconstructed' if not capacity else None)


def display_summary(result,output,saved=None):
    result=_receipt_object(result);saved=_receipt_object(saved) or _receipt_object(result.get('saved'))
    actual=_receipt_object(saved.get('result')) or result
    evaluation=_receipt_object(saved.get('evaluation'));transfer=_receipt_object(evaluation.get('payload_transfer'))
    diagnostic=_receipt_object(result.get('execution_diagnostics')) or _receipt_object(actual.get('execution_diagnostics')) or execution_diagnostics(actual,saved.get('counts'),
        result.get('stages'),saved.get('state'))
    times=diagnostic.get('time_execution_counts') or _receipt_object(evaluation.get('time')).get('counts')
    errors=_receipt_object(actual.get('errors'))
    return dict(status=result.get('status'),output=str(output),evaluation_status=actual.get('evaluation_status',evaluation.get('status')),
        earliest_stage_error=diagnostic['earliest_stage_error'],phase_states=diagnostic['phase_states'],
        operation_receipts=diagnostic['operation_receipts'],disk_cache_capacity=diagnostic['disk_cache_capacity'],
        missing_capacity_measurements=diagnostic['missing_capacity_measurements'],receiver_execution=diagnostic['receiver_execution'],
        time_counts=times,time_counts_basis=diagnostic['time_counts_basis'] or 'SAVED_EVALUATION',
        B_correct_supported=transfer.get('correct_supported'),B_fixed_slots=160,five_view_target=transfer.get('full_five_view_target_observed'),
        synchronization_effect=_receipt_object(evaluation.get('synchronization_effect')).get('status'),
        physical_calls_this_invocation=result.get('physical_calls_this_invocation'),post_calls_this_invocation=result.get('post_calls_this_invocation'),
        fixed_denominators=dict(time_nodes=822,time_edges=817,raw_bits=9600,votes=76800,local=1440,final=480),
        arithmetic_upper_bound=8813,not_model_inference_count=True,codec=dict(encode=7,probe=13,decode=13),
        total_wall_seconds=_receipt_object(saved.get('resources')).get('total_wall_seconds',actual.get('wall_seconds')),
        recovered_records=len(_receipt_object(result.get('recovered_artifacts'))),
        downstream_errors={str(k)[:80]:str(v)[:240] for k,v in list(errors.items())[:8]},
        full_records='Complete saved records remain unchanged in OUTPUT; display does not rerun post.')
