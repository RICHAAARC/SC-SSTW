"""Fixed nine-raw/67-item CPU diagnostic. Truth is opened only by the final post."""
from pathlib import Path
import json
import time
from . import b_time_layer_diagnostic_v1_records as rec


def run(config,output=None,*,backend_factory=None,evaluator=None):
    output=Path(output or config['output']).expanduser().resolve()
    if rec.output_state(output)=='CONSUMED':return rec.recover(output)
    if rec.output_state(output)=='CONFLICT':return dict(status='OUTPUT_CONFLICT',computations_this_invocation=0)
    preflight=rec.preflight(config,output)
    if preflight['status']!='PREFLIGHT_READY':return preflight
    output.mkdir(parents=True,exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as stream:
            json.dump(dict(status='RESERVED_CONSUMED',source_attempt=str(config['source_attempt']),plan=rec.plan(),
                budgets=rec.budgets(),denominators=rec.denominators(),limits=rec.LIMITS,fixed_slots=rec.skeleton(),rerun='READ_ONLY_NO_POST'),stream,indent=2)
    except FileExistsError:return rec.recover(output)
    start=time.monotonic();backend=None;errors={};data={};table=None;global_stop=False;interrupted=False;evaluation={};resources={}
    journal=rec.Journal(output,lambda:None)
    try:
        import numpy as np
        from main.tube_state import b_time_layer_diagnostic_v1 as method
        if backend_factory is None:
            from runtime.wan.b_time_layer_diagnostic_v1 import Backend
            backend_factory=Backend
        backend=backend_factory();journal.check=backend.check
        if hasattr(backend,'started'):backend.started=start
        source=Path(config['source_attempt']).expanduser().resolve()
        def attempt(key,fn,unit=None):
            try:return journal.call(key,fn)
            except Exception as error:
                errors[key]=f'{type(error).__name__}: {error}'
                if unit is not None:unit.setdefault('errors',{})[key]=errors[key]
                if type(error).__name__ in ('ResourceExceeded','RawLimitExceeded'):raise
                return None
        try:table=attempt('codebook',backend.codebook)
        except Exception as error:
            if type(error).__name__!='RawLimitExceeded':raise
        for descriptor in rec.units():
            uid=descriptor['id'];unit=data.setdefault(uid,{});folder=output/uid;folder.mkdir(parents=True,exist_ok=True)
            backend.start_raw()
            try:
                def raw_load():
                    value=backend.load_raw(source/descriptor['raw_path'],descriptor['frames'])
                    np.savez(folder/'raw_saved.npz',**value);unit['raw']=value;return dict(status='RAW_SAVED')
                attempt(uid+'/raw_load',raw_load,unit)
                if descriptor['kind']=='received':
                    def saved_load():
                        value=backend.saved_view(source/descriptor['saved_path'],descriptor['frames']);rec.dump(folder/'saved_public.json',value);unit['saved']=value;return dict(status='SAVED_PUBLIC_READ')
                    attempt(uid+'/saved_view_load',saved_load,unit)
                if 'raw' not in unit:continue
                def arithmetic():
                    arrays,summary=method.arithmetic(unit['raw'])
                    np.savez(folder/'arithmetic.npz',**arrays);rec.dump(folder/'arithmetic.json',summary);unit['arithmetic']=summary;return summary
                attempt(uid+'/arithmetic',arithmetic,unit)
                def fusion():
                    value=method.fusion(unit['raw'])
                    np.savez(folder/'fusion.npz',**{role+'_'+key:array for role,trace in value.items() for key,array in trace.items()});unit['fusion']=value
                    return dict(status='FIXED_F_R0_R1_TRACE_SAVED')
                attempt(uid+'/fusion',fusion,unit)
                if table is None:continue
                for role in rec.ROLES:
                    def compatibility(role=role):
                        value=method.compatibility(unit['raw'],role,table,backend.check)
                        rec.dump(folder/('compatibility_'+role+'.json'),value);unit.setdefault('compat',{})[role]=value;return dict(status='FULL_DOMAIN_SCANNED',comparisons=value['comparisons'])
                    attempt(uid+'/'+role+'/compatibility',compatibility,unit)
                if descriptor['kind']=='received' and 'F' in unit.get('compat',{}):
                    def path():
                        value=method.path_trace(unit['compat']['F'],unit.get('saved',{}))
                        value['saved_fusion_comparison']=method.compare_saved_fusion(unit.get('fusion',{}).get('F',{}),unit.get('saved',{}))
                        rec.dump(folder/'path.json',value);unit['path']=value;return dict(status='ORIGINAL_F_PATH_DIAGNOSTIC_SAVED')
                    attempt(uid+'/path',path,unit)
            except Exception as error:
                if type(error).__name__!='RawLimitExceeded':raise
                unit.setdefault('errors',{})['raw_resource']=str(error)
            finally:backend.end_raw()
    except BaseException as error:
        global_stop=True;interrupted=not isinstance(error,Exception);errors['attempt']=f'{type(error).__name__}: {error}'
    journal.terminalize()
    if not global_stop:
        try:
            def blind_save():
                value=dict(status='ALL_BLIND_DIAGNOSTICS_TERMINAL',rows={k:v for k,v in journal.rows.items() if k not in ('blind_save','post')},
                    fixed_slots=rec.skeleton(),denominators=rec.denominators(),errors=dict(errors),
                    unit_records={u['id']:dict(directory=u['id'],available=[k for k in data.get(u['id'],{}) if k!='raw'],errors=data.get(u['id'],{}).get('errors',{})) for u in rec.units()})
                rec.dump(output/'blind_terminal.json',value);return dict(status=value['status'])
            journal.call('blind_save',blind_save)
            def post():
                nonlocal evaluation
                selected=evaluator
                if selected is None:
                    from .b_time_layer_diagnostic_v1_evaluation import evaluate as selected
                evaluation=selected(output,source,data,table,journal.rows,backend.check);return evaluation
            journal.call('post',post)
        except BaseException as error:
            errors['completion']=f'{type(error).__name__}: {error}';interrupted |= not isinstance(error,Exception)
    journal.terminalize(include_post=True)
    result=dict(status='INTERRUPTED' if interrupted else 'FIXED_DIAGNOSTIC_TERMINATED',
        evaluation_status='EVIDENCE_INCOMPLETE' if errors else evaluation.get('status','EVIDENCE_INCOMPLETE'),
        errors=errors,denominators=rec.denominators(),budgets=rec.budgets(),scientific_pass=None,
        new_CPU_diagnostic=True,physical_observations_added=0,wall_seconds=time.monotonic()-start)
    try:
        rec.dump(output/'result.json',result)
        if backend is not None:backend.check()
    except Exception as error:
        errors['final_save_or_resource']=str(error);result['evaluation_status']='EVIDENCE_INCOMPLETE'
    finally:
        if backend is not None:
            try:
                resources=backend.close()
                if not isinstance(resources,dict):raise ValueError('resource close receipt must be an object')
            except Exception as error:resources=dict(error=f'{type(error).__name__}: {error}')
    elapsed=time.monotonic()-start
    if resources.get('error'):errors['resource_close']=resources['error']
    if resources.get('peak_resident_bytes',0)>rec.LIMITS['rss_bytes']:errors['resource_peak']='closed resource receipt exceeds 8 GiB'
    if elapsed>rec.LIMITS['total_wall_seconds']:errors['resource_total_wall']='whole diagnostic 6300s limit observed on close'
    if errors:result['evaluation_status']='EVIDENCE_INCOMPLETE'
    resources.update(limits=rec.LIMITS,total_wall_seconds=elapsed,completion_errors=dict(errors))
    rec.dump(output/'resources.json',resources);rec.dump(output/'result.json',result);return result
