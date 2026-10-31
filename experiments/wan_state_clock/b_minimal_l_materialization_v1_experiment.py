"""Serial independent A/B attempts; scientific imports follow reservation."""
from pathlib import Path
import json
import time
from . import b_minimal_l_materialization_v1_records as records


def run_stage(source,output,stage,model_id=None,backend_factory=None,protocol=None,receiver_codec_calls=34):
    output=Path(output).expanduser().resolve();folder=output/stage
    state=records.output_state(folder)
    if state=='CONSUMED': return records.recover_stage(output,stage)
    if state=='CONFLICT': return dict(status='OUTPUT_CONFLICT',stage=stage,physical_calls_this_invocation=0)
    source,output=records.validate_paths(source,output)
    cfg=dict(records.CONFIG if protocol is None else protocol)
    if model_id is not None: cfg['model_id']=str(model_id)
    preflight=records.preflight(source,output,stage,cfg,receiver_codec_calls)
    if preflight['status']!='PREFLIGHT_READY': return preflight
    folder.mkdir(parents=True,exist_ok=True)
    try:
        with (folder/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED',stage=stage,unix=time.time(),retry=False,
                           budgets=records.A_BUDGETS if stage=='A' else records.B_BUDGETS,
                            denominators=records.denominators(cfg,receiver_codec_calls)),stream,indent=2)
    except FileExistsError: return records.recover_stage(output,stage)
    journal=records.Journal(folder,stage);backend=None;errors={};answer={};native={};complete=False
    started=time.monotonic()
    if stage=='B':
        # Consume the saved A model location. This is metadata, not an equality gate.
        try: cfg.update(records.load(output/'A/protocol.json'))
        except (OSError,ValueError,TypeError) as error: errors['protocol']=str(error)
    if model_id is not None: cfg['model_id']=str(model_id)
    records.dump(folder/'protocol.json',cfg)
    try:
        if backend_factory is None:
            from runtime.wan.b_minimal_l_materialization_v1 import CaptureBackend,CandidateBackend
            backend_factory=CaptureBackend if stage=='A' else CandidateBackend
        backend=backend_factory(cfg,folder,journal.event)
        from .b_framewise_qim_v1_experiment import save_read
        def read_all(preclip,arm):
            q0=None
            def one(path,layer):
                nonlocal q0
                def persist():
                    nonlocal q0
                    raw=backend.read_layer(path,layer)
                    save_read(folder,arm+'/'+layer,raw)
                    if arm=='BASE' and layer=='preclip': q0=raw['q']
                journal.call('time_reads',persist)
            for layer in records.LAYERS[:3]:
                try: journal.operation(arm+'/'+layer,lambda layer=layer:one(preclip,layer))
                except Exception as error: errors[arm+'/'+layer]=f'{type(error).__name__}: {error}'
            try:
                media=journal.operation('media',lambda:backend.media(preclip,arm))
                journal.operation(arm+'/h264',lambda:one(media,'h264'))
            except Exception as error: errors[arm+'/media']=f'{type(error).__name__}: {error}'
            return q0
        if stage=='A':
            baseline=journal.operation('capture',lambda:backend.capture(source/cfg['terminal_relative']))
            q0=read_all(baseline,'BASE')
            if q0 is None: raise ValueError('saved BASE/preclip q missing; no additional read permitted')
            def solve_once():
                result=journal.call('solver',lambda:backend.solve_cached(baseline,q0,lambda row:records.dump(folder/'solver_progress.json',row)))
                records.dump(folder/'solver.json',result)
                return result
            answer=journal.operation('solver',solve_once)
            cache_complete=bool(answer.get('finite_candidate_saved')) and all((folder/name).is_file() for name in records.cache_paths())
            if not cache_complete: errors['candidate']='finite candidate or complete A cache unavailable; no refill'
            complete=cache_complete and not errors
        else:
            journal.operation('restore',lambda:backend.restore_saved(output/'A'))
            def check_once():
                result=backend.check_candidate();records.dump(folder/'linear.json',result);return result
            linear=journal.operation('linear_constraints',check_once)
            if linear.get('status')!='DIRECT_CONSTRAINTS_WITHIN_TOLERANCE':
                raise ArithmeticError('direct constraints failed; B model/K/P consumed, dependent head/media not executed')
            def native_once():
                path,result=backend.apply_saved();records.dump(folder/'native.json',result);return path,result
            preclip,native=journal.operation('native',native_once)
            if not native.get('finite_complete') or native.get('frames')!=181:
                raise ValueError('native incomplete/nonfinite; dependent reads not executed')
            read_all(preclip,'QIM');complete=True
            if native.get('status')!='NATIVE_RESIDUAL_WITHIN_TOLERANCE':
                errors['native']='finite native target/budget failure retained; fixed layer reads attempted'
    except BaseException as error:
        complete=False;errors['stage']=f'{type(error).__name__}: {error}'
        if not isinstance(error,Exception): errors['interrupted']=True
    finally:
        resource_error=None
        if backend is not None:
            try: backend.close()
            except Exception as error:
                resource_error=str(error);errors['close']=resource_error;complete=False
        records.dump(folder/'resources.json',dict(stage_wall_seconds=time.monotonic()-started,
            codec_calls=getattr(backend,'codec_times',[]),codec_call_limit_seconds=300,codec_terminate_seconds=30,
            solver_seconds_limit=7200,solver_memory_bytes_limit=16*1024**3,error=resource_error))
    journal.terminalize('fixed stage ended; no retry or refill')
    result=dict(status='COMPLETED_FIXED_RECORDS' if not errors else 'STOPPED_FIXED_ATTEMPT',stage=stage,
                stage_complete=complete,finite_candidate_saved=bool(answer.get('finite_candidate_saved')) if stage=='A' else None,
                solver_status=answer.get('status'),native_status=native.get('status'),errors=errors,counts=journal.counts,
                denominators=records.denominators(cfg,receiver_codec_calls),optimality='UNPROVEN',new_identity_claim=False,scientific_pass=None)
    records.dump(folder/'result.json',result)
    return result


def run(source,output,model_id=None,backend_factories=None):
    output=Path(output).expanduser().resolve()
    if not records.needs_work(output): return records.recover(output)
    factories=backend_factories or (None,None)
    a=run_stage(source,output,'A',model_id,factories[0])
    if records.ready_B(output):
        return dict(A=a,B=dict(status='DEPENDENCY_UNRESOLVED',errors=records.ready_B(output),physical_calls_this_invocation=0),
                    cpu_confirmation='NOT_EXECUTED',denominators=records.denominators())
    b=run_stage(source,output,'B',model_id,factories[1])
    return dict(A=a,B=b,cpu_confirmation='NOT_EXECUTED_SEPARATE_LOCAL_ENTRY',denominators=records.denominators())
