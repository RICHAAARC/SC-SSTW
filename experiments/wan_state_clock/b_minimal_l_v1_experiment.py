"""Fixed user-run ten-view experiment. Recovery/import remain standard-library only."""
from pathlib import Path
import json
import time
from . import b_minimal_l_v1_records as records


def private_plan():
    """Attack/evaluation side only. Never supplied to the received-only method."""
    mappings=[list(range(181)),list(range(38,139)),[n for n in range(181) if n!=44],
              list(range(45))+[44]+list(range(45,181)),[n for n in range(181) if n not in (43,44,45)]]
    return [dict(id=f'v{i:02}',source=i//5,arm=('QIM','BASE')[i//5],source_id='b2-m3-e02',
                 view=('FULL','CROP','DELETE1','REPEAT1','DELETE3')[i%5],mapping=mappings[i%5]) for i in range(10)]


def run(qim,base,output,backend_factory=None,evaluator=None):
    # A consumed attempt returns before checking inputs, importing science, or creating any directory.
    output=Path(output).expanduser().resolve()
    if records.output_state(output)=='CONSUMED': return records.recover(output)
    if records.output_state(output)=='CONFLICT': return dict(status='OUTPUT_CONFLICT',physical_calls_this_invocation=0)
    qim,base,output=records.validate_paths(qim,base,output)
    preflight=records.preflight(qim,base,output)
    if preflight['status']!='PREFLIGHT_READY': return preflight
    output.mkdir(parents=True,exist_ok=True)
    attempt=dict(status='RESERVED_CONSUMED',plan=records.plan(),budgets=records.BUDGETS,
                 denominators=records.denominators(),qim_media=str(qim),base_media=str(base),
                 context='Original native target failure and optimality UNPROVEN retained; no new writer or E41')
    try:
        with (output/'attempt.json').open('x') as f: json.dump(attempt,f,ensure_ascii=False,indent=2)
    except FileExistsError: return records.recover(output)
    journal=records.Journal(output);backend=None;errors={};wall_start=time.monotonic()
    try:
        import numpy as np
        if backend_factory is None:
            from runtime.wan.b_minimal_l_v1 import Backend
            backend_factory=Backend
        backend=backend_factory()
        attack=private_plan();records.dump(output/'private/attack_truth.json',dict(views=attack))
        table=None
        try: table=journal.call('codebook',backend.codebook)
        except Exception as error: errors['codebook']=str(error)
        for source,path in enumerate((qim,base)):
            if not preflight['inputs'][source]['readable']:
                reason=preflight['inputs'][source]['error'];errors[f's{source}']=reason
                keys=[f's{source}/probe',f's{source}/decode']
                keys += [f'v{i:02}/{op}' for i in range(source*5,source*5+5) for op in ('encode','probe','decode','time_read','hard_receiver','C','L')]
                for key in keys: journal.rows[key].update(status='DEPENDENCY_UNRESOLVED',error=reason)
                journal.save();continue
            source_rgb=None
            try:
                metadata=journal.call(f's{source}/probe',lambda:backend.probe(path,181))
                records.dump(output/f's{source}_probe.json',metadata)
                source_rgb=journal.call(f's{source}/decode',lambda:backend.decode(path,181))
            except Exception as error: errors[f's{source}']=f'{type(error).__name__}: {error}'
            if source_rgb is None: continue
            for task in attack[source*5:source*5+5]:
                ident=task['id'];folder=output/'received'/ident;folder.mkdir(parents=True,exist_ok=True)
                pixels=None;raw=None;edited=None;decision=None;cost=None;local=None
                try:
                    # Mapping is used only by the attack. Reader gets newly decoded pixels.
                    edited=source_rgb[task['mapping']]
                    journal.call(ident+'/encode',lambda:backend.encode(edited,folder/'received.mp4'))
                    edited=None
                    metadata=journal.call(ident+'/probe',lambda:backend.probe(folder/'received.mp4',len(task['mapping'])))
                    records.dump(folder/'probe.json',metadata)
                    pixels=journal.call(ident+'/decode',lambda:backend.decode(folder/'received.mp4',len(task['mapping'])))
                    def read_and_save():
                        observations=backend.read(pixels)
                        np.savez(folder/'raw.npz',**observations)
                        return None
                    def receive_and_save():
                        nonlocal raw,decision
                        # Raw file is durably closed and reopened before any semantic fusion.
                        with np.load(folder/'raw.npz',allow_pickle=False) as saved:
                            raw={k:saved[k] for k in ('q','d0','d1','soft','mask')}
                        decision,merged,known=backend.receive(raw,table)
                        np.savez(folder/'fused.npz',soft=merged,known=known)
                        records.dump(folder/'blind.json',decision)
                    def cost_and_save():
                        nonlocal cost
                        cost=backend.C(raw,decision,table)
                        np.savez(folder/'C.npz',**{k:cost[k] for k in ('C','bounds','offsets','support')})
                        records.dump(folder/'C.json',{k:cost[k] for k in ('states','R','comparisons','graph_abnormal')})
                    def local_and_save():
                        nonlocal local
                        from main.tube_state.b_minimal_l_v1 import candidates
                        local=backend.L(cost)
                        np.savez(folder/'L.npz',dL=local['dL'],best=local['best'])
                        records.dump(folder/'L.json',dict(status=local['status'],minimum=local['minimum']))
                        records.dump(folder/'candidates.json',candidates(decision,cost,local))
                        saved=records.load(folder/'blind.json')
                        before=[n['output_index'] for n in decision['nodes']]
                        after=[n['output_index'] for n in saved['nodes']]
                        records.dump(folder/'formal_preservation.json',dict(
                            lost=sum(a is not None and b is None for a,b in zip(before,after)),
                            changed=sum(a is not None and b is not None and a!=b for a,b in zip(before,after)),
                            relation_changes=int(decision['relations']!=saved['relations']),
                            qualification_changed=decision['qualification']!=saved['qualification']))
                    with backend.receiving_view():
                        journal.call(ident+'/time_read',read_and_save)
                        pixels=None
                        if table is not None:
                            journal.call(ident+'/hard_receiver',receive_and_save)
                            journal.call(ident+'/C',cost_and_save)
                            journal.call(ident+'/L',local_and_save)
                except Exception as error:
                    errors[ident]=f'{type(error).__name__}: {error}'
                    if type(error).__name__=='ResourceExceeded':
                        # Includes deadline observed while closing the view context after saved IO.
                        for op in ('L','C','hard_receiver','time_read','decode','probe','encode'):
                            row=journal.rows[ident+'/'+op]
                            if row['status']!='NOT_STARTED':
                                row.update(status='RESOURCE_UNRESOLVED',error=errors[ident]);journal.save();break
                finally: pixels=None;raw=None;edited=None;decision=None;cost=None;local=None
            source_rgb=None
    except BaseException as error:
        errors['attempt']=f'{type(error).__name__}: {error}'
        if not isinstance(error,Exception):
            journal.terminalize('attempt interrupted; no retry')
            records.dump(output/'result.json',dict(status='INTERRUPTED',errors=errors,denominators=records.denominators()))
            raise
    finally:
        resources=dict(status='NOT_STARTED',error='backend unavailable')
        if backend is not None:
            try:
                resources=backend.close()
                if not isinstance(resources,dict): raise ValueError('resource close must return an object')
                failure=resources.get('error')
                if resources.get('peak_resident_bytes',0)>8*1024**3: failure=failure or 'peak resident memory exceeds 8 GiB'
                if resources.get('receiver_seconds',0)>6300: failure=failure or 'receiver total exceeds 6300 seconds'
                resources.update(status='RESOURCE_UNRESOLVED' if failure else 'CLOSED',error=failure)
                if failure: errors['resources']=str(failure)
            except Exception as error:
                errors['resources']=f'{type(error).__name__}: {error}'
                resources=dict(status='CLOSE_FAILED',error=errors['resources'])
        records.dump(output/'resources.json',resources)
    journal.terminalize('dependency failed or fixed attempt ended; no retry')
    # All executable blind reads/receivers are terminal before opening attack truth.
    try:
        if evaluator is None:
            from .b_minimal_l_v1_evaluation import evaluate
            evaluator=evaluate
        evaluation=evaluator(output,journal.rows)
        records.dump(output/'evaluation.json',evaluation)
    except Exception as error: errors['evaluation']=f'{type(error).__name__}: {error}'
    result=dict(status='FIXED_ATTEMPT_TERMINATED',errors=errors,denominators=records.denominators(),
                source_id='b2-m3-e02', full_wall_seconds=time.monotonic()-wall_start,
                formal_new_determined=0,new_relation_outputs=0,
                failed_rows=[k for k,v in journal.rows.items() if v['status']!='RETURNED'],
                claim_ceiling='Single-source fixed edits, same second-generation H264; no identity/general-null/general-edit claim')
    records.dump(output/'result.json',result)
    return result
