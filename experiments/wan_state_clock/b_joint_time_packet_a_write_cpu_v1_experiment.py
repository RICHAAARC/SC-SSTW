"""Fixed one-shot local CPU stage P; independent of the consumed structural A-CPU."""
from pathlib import Path
from . import b_joint_time_packet_a_write_cpu_v1_records as rec


def prepare_head(model_directory, output, *, prepare=None):
    output = Path(output).expanduser().resolve()
    if (output/'attempt.json').exists(): return rec.recover(output)
    if not rec.reserve(output, 'head', dict(model_directory=str(model_directory))): return rec.recover(output)
    journal = rec.Journal(output, 'head'); result = dict(status='HEAD_PREPARATION_FAILED')
    try:
        if prepare is None:
            from runtime.wan.b_joint_time_packet_a_write_cpu_v1 import prepare_head as prepare
        result = journal.call('head_prepare', lambda: prepare(model_directory, output))
    except BaseException as error:
        result.update(error=type(error).__name__+': '+str(error), consumed=True)
    finally: journal.terminalize()
    rec.dump(output/'result.json', result)
    return result


def validate_config(config, output):
    if set(config['q0']) != set(rec.SOURCES): raise ValueError('fixed n01/e02 roster required')
    paths = [Path(config['head']).expanduser().resolve()]
    for source in rec.SOURCES:
        spec = config['q0'][source]
        if spec['source_id'] != rec.SOURCE_IDS[source]: raise ValueError('wrong original source role')
        if (spec.get('role'), spec.get('layer'), spec.get('units'), spec.get('frames')) != ('BASE','preclip','255*RGB_luma_DCT',181):
            raise ValueError('original BASE/preclip RGB q0 required')
        paths.append(Path(spec['file']).expanduser().resolve())
    for path in paths:
        if path == output or path.is_relative_to(output) or output.is_relative_to(path):
            raise ValueError('new stage output overlaps a read-only input')
    # A missing shared head is a preparatory wait. A missing source q0 is that
    # source's consumed read failure, not an all-source readiness gate.
    return [str(paths[0])] if not paths[0].is_file() else []


def run(config, output, *, backend_factory=None, head_loader=None):
    output = Path(output).expanduser().resolve()
    if (output/'attempt.json').exists(): return rec.recover(output)
    missing = validate_config(config, output)
    if missing: return dict(status='WAITING_INPUT', missing=missing, attempt_consumed=False, physical_calls_this_invocation=0)
    if not rec.reserve(output, 'solve', config): return rec.recover(output)
    journal = rec.Journal(output); errors, results = {}, {}
    try:
        rec.dump(output/'operation_plan.json', rec.operation_catalog())
        rec.dump(output/'file_plan.json', rec.output_files())
        if backend_factory is None or head_loader is None:
            from runtime.wan.b_joint_time_packet_a_write_cpu_v1 import Backend, load_head
            backend_factory = backend_factory or Backend; head_loader = head_loader or load_head
        def shared_head():
            import shutil
            value = head_loader(config['head'])
            shutil.copyfile(config['head'], output/'head.pt')
            return value
        head = journal.call('head_load', shared_head)
        for source in rec.SOURCES:
            try:
                backend = backend_factory(head, output/source, lambda kind,done: journal.event(source,kind,done), rec.dump)
                journal.call(source+'/q0_load', lambda: backend.read_q0(config['q0'][source]))
                journal.call(source+'/target', backend.target)
                def progress(row):
                    rec.dump(output/source/'solver_progress.json', row); journal.persist()
                answer = journal.call(source+'/solver', lambda: backend.solve(progress))
                if journal.budget_error and journal.budget_error.startswith(source+'/'):
                    raise rec.BudgetExceeded(journal.budget_error)
                if not answer.get('native_handoff_eligible'):
                    errors[source] = 'solver candidate not eligible; finite artifacts and original status retained'
                    continue
                journal.call(source+'/prediction', backend.predict)
                results[source] = journal.call(source+'/handoff', lambda: backend.handoff(source))
            except Exception as error:
                errors[source] = type(error).__name__+': '+str(error)
            finally:
                if 'backend' in locals(): del backend
    except BaseException as error:
        errors['stage'] = type(error).__name__+': '+str(error)
    finally: journal.terminalize()
    handoff = dict(status='CPU_HANDOFF_COMPLETE' if len(results)==2 and not errors else 'CPU_HANDOFF_PARTIAL' if results else 'CPU_HANDOFF_UNAVAILABLE',
        stage_terminal=True, source_eligibility={s:s in results for s in rec.SOURCES},
        head='head.pt', paths_relative_to='this CPU output directory',
        sources={s:results.get(s, dict(status='UNRESOLVED', error=errors.get(s, errors.get('stage','not entered')))) for s in rec.SOURCES},
        source_directories={s:s for s in rec.SOURCES}, forward=0, codec=0, GPU=0,
        next='USER_GPU_1; unavailable inputs must wait, never solve in Colab')
    rec.dump(output/'cpu_handoff.json', handoff)
    result = dict(status='CPU_FIXED_ATTEMPT_TERMINATED', handoff_status=handoff['status'],
                  errors=errors, budgets=rec.budgets(), native_measured=False, scientific_pass=None)
    rec.dump(output/'result.json', result)
    return result

