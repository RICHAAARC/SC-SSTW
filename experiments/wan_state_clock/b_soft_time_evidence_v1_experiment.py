"""One reserved offline attempt, each saved input once, all blind work before truth."""
from pathlib import Path
import json
from . import b_soft_time_evidence_v1_records as rec


def run(source,output,backend_factory=None,evaluator=None):
    output=Path(output).expanduser().resolve()
    if rec.output_state(output)=='CONSUMED':return rec.recover(output)
    if rec.output_state(output)=='CONFLICT':return dict(status='OUTPUT_CONFLICT',scientific_calls_this_invocation=0)
    source,output=rec.validate_paths(source,output);preflight=rec.preflight(source,output)
    if preflight['status']!='PREFLIGHT_READY':return preflight
    output.mkdir(parents=True,exist_ok=True)
    try:
        with (output/'attempt.json').open('x') as f:
            json.dump(dict(status='RESERVED_CONSUMED',source=str(source),plan=rec.plan(),budgets=rec.BUDGETS,
                           denominators=rec.denominators(),new_formal_determined=0),f,indent=2)
    except FileExistsError:return rec.recover(output)
    journal=rec.Journal(output);backend=None;errors={}
    try:
        import numpy as np
        if backend_factory is None:
            from runtime.wan.b_soft_time_evidence_v1 import Backend
            backend_factory=Backend
        backend=backend_factory();table=None
        try:table=journal.call('codebook',backend.codebook)
        except Exception as error:errors['codebook']=f'{type(error).__name__}: {error}'
        for i,expected_R in enumerate(rec.LENGTHS):
            ident=f'v{i:02}';folder=output/'views'/ident;folder.mkdir(parents=True,exist_ok=True)
            raw=old=cost=local=global_result=None
            try:
                with backend.view():
                    def read_raw():
                        value=backend.read_raw(source/'received'/ident/'raw.npz')
                        if len(value['soft'])!=expected_R:raise ValueError('actual received length differs from fixed view plan')
                        return value
                    def read_old():
                        content=backend.read_old(source/'received'/ident/'blind.json')
                        (folder/'old_blind.json').write_bytes(content)
                        value=json.loads(content,parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
                        if not isinstance(value,dict):raise ValueError('old blind must be an object')
                        return value
                    for name,fn in (('raw_read',read_raw),('old_blind_read',read_old)):
                        try:
                            value=journal.call(ident+'/'+name,fn)
                            if name=='raw_read':raw=value
                            else:old=value
                        except Exception as error:errors[ident+'/'+name]=f'{type(error).__name__}: {error}'
                    if raw is None or old is None or table is None:continue
                    def C():
                        value=backend.C(raw,old,table)
                        np.savez(folder/'C.npz',C=value['C'],offsets=value['offsets'],bounds=value['bounds'],support=value['support'])
                        rec.dump(folder/'C.json',dict(states=value['states'],R=value['R'],comparisons=value['comparisons'],graph_abnormal=value['graph_abnormal']))
                        backend.check();return value
                    cost=journal.call(ident+'/C',C)
                    def L():
                        value=backend.L(cost);np.savez(folder/'L.npz',dL=value['dL'],best=value['best'])
                        rec.dump(folder/'L.json',dict(minimum=value['minimum'],status=value['status']));backend.check();return value
                    local=journal.call(ident+'/L',L)
                    def G():
                        value=backend.G(cost,local)
                        np.savez(folder/'G.npz',**{k:value[k] for k in ('Phi','delta','dG_C','native','aligned')})
                        summary=backend.summary(cost,local,value)
                        summary.update(old_qualification=old.get('qualification'),old_path_status=old.get('path_status'),old_formal_copy='old_blind.json')
                        rec.dump(folder/'blind_ranking.json',summary);backend.check();return value
                    global_result=journal.call(ident+'/G',G)
            except Exception as error:
                errors[ident]=f'{type(error).__name__}: {error}'
                if type(error).__name__=='ResourceExceeded':
                    for stage in reversed(rec.STAGES):
                        row=journal.rows[ident+'/'+stage]
                        if row['status']!='NOT_STARTED':row.update(status='RESOURCE_UNRESOLVED',error=errors[ident]);journal.save();break
            finally:raw=old=cost=local=global_result=None
    except BaseException as error:
        errors['attempt']=f'{type(error).__name__}: {error}'
        if not isinstance(error,Exception):
            journal.terminalize();rec.dump(output/'result.json',dict(status='INTERRUPTED',errors=errors,denominators=rec.denominators()));raise
    finally:
        resources=dict(status='NOT_STARTED',error='backend unavailable')
        if backend is not None:
            try:
                resources=backend.close()
                if not isinstance(resources,dict):raise ValueError('resource object required')
                failure=resources.get('error')
                if resources.get('peak_resident_bytes',0)>8*1024**3:failure=failure or 'RSS exceeds 8 GiB'
                if resources.get('receiver_seconds',0)>6300:failure=failure or 'total exceeds 6300 seconds'
                resources.update(status='RESOURCE_UNRESOLVED' if failure else 'CLOSED',error=failure)
                if failure:errors['resources']=str(failure)
            except Exception as error:errors['resources']=str(error);resources=dict(status='CLOSE_FAILED',error=str(error))
        rec.dump(output/'resources.json',resources)
    journal.terminalize()
    try:
        if evaluator is None:
            from .b_soft_time_evidence_v1_evaluation import evaluate
            evaluator=evaluate
        rec.dump(output/'evaluation.json',evaluator(source,output,journal.rows))
    except Exception as error:errors['evaluation']=f'{type(error).__name__}: {error}'
    result=dict(status='FIXED_DIAGNOSTIC_TERMINATED',errors=errors,denominators=rec.denominators(),new_formal_determined=0,
                failed_rows=[k for k,v in journal.rows.items() if v['status']!='RETURNED'],
                claim_ceiling='Conditional node model rankings only; no new reliable correspondence, relation, identity or general-null evidence')
    rec.dump(output/'result.json',result);return result
