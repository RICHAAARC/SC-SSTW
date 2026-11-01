"""Whole c02 reservation and record-only recovery. No scientific dependencies."""
from pathlib import Path
from copy import deepcopy
import time
from .b_minimal_l_v1_records import load, dump, output_state
from . import b_conditional_time_validation_v1_records as chain
from . import b_time_payload_cdc32_v1_records as payload

PHASES=('A','B','time','payload','time_post','payload_post','summary')
FOLDERS=dict(A='time_chain/materialization/A',B='time_chain/materialization/B',
             time='time_chain/receiver',payload='payload')


def budgets():
    return dict(materialization_A=dict(chain.material.A_BUDGETS),materialization_B=dict(chain.material.B_BUDGETS),
                time=chain.receiver.budgets('media'),payload=payload.budgets(),
                stage_arithmetic_upper_bounds=dict(A=8457,B=57,time=43,payload=256),
                heterogeneous_arithmetic_upper_bound=8813,arithmetic_is_model_inference_count=False,
                codec=dict(encode=7,probe=13,decode=13,total=33),
                materialization_time_reads=8,receiver_time_reads=5,
                post=dict(time=1,payload=1,summary=1),wrapper_phases_add_physical_calls=0)


def denominators():
    return dict(materialization=chain.material.denominators(dict(source_id='b2-m3-c02',frames=181),27),
                time=chain.receiver.denominators(),payload=payload.denominators())


def plan():
    return {k:dict(planned=1,started=0,returned=0,status='NOT_STARTED') for k in PHASES}


def recovery_layout():
    layout=payload.recovery_layout()
    for item in layout:item.update(path='payload/'+item['path'],state_path='payload/state.json')
    for view,n in zip(payload.VIEWS,payload.LENGTHS):
        layout.append(dict(kind='time',path=f'time_chain/receiver/received/{view}/submission.json',
            state_path='time_chain/receiver/state.json',operation=view+'/submit',fallback=dict(
                view=view,final_nodes=[dict(received_frame=r,output_index=None,status='UNRESOLVED_RECEIPT') for r in range(n)],
                final_relations=[dict(received_left=r,status='UNRESOLVED_RECEIPT') for r in range(n-1)])))
    return layout


def preflight(config,output):
    if config.get('source_id')!='b2-m3-c02' or config.get('terminal_relative')!='private/b2-m3-c02/P0_terminal.pt':
        return dict(status='PREFLIGHT_UNRESOLVED',error='fixed c02 source and P0 required',physical_calls_this_invocation=0)
    result=chain.preflight(config,output)
    if result['status']=='PREFLIGHT_READY':result.update(budgets=budgets(),denominators=denominators())
    return result


def read_receipts(output):
    """Only operation/result/resource metadata; never caches, truth, or private protocol."""
    saved={};errors={}
    for name in ('attempt','state','counts','resources','evaluation','result'):
        try:
            value=load(Path(output)/(name+'.json'))
            if not isinstance(value,dict):raise ValueError('receipt object required')
            saved[name]=value
        except (OSError,ValueError) as error:errors[name]=str(error)
    return saved,errors


def recover(output):
    output=Path(output);saved,errors=read_receipts(output);stages={};artifacts={}
    for stage,folder in FOLDERS.items():
        receipts,err=read_receipts(output/folder);stages[stage]=receipts
        errors.update({stage+'/'+k:v for k,v in err.items()})
    # Canonical fixed skeleton is metadata, not a planner or another observation.
    layout=recovery_layout()
    artifacts.update(payload.recover_artifacts(output,[i for i in layout if i['kind']!='time'],stages['payload'].get('state',{})))
    for item in layout:
        if item['kind']!='time':continue
        value=deepcopy(item['fallback']);unconfirmed=None;error=None
        try:
            actual=load(output/item['path']);row=stages['time'].get('state',{}).get(item['operation'],{})
            if not isinstance(row,dict) or row.get('status')!='RETURNED':
                unconfirmed=actual;raise ValueError('time submission not returned')
            if not isinstance(actual,dict):raise ValueError('time submission object required')
            for key,coordinate in (('final_nodes','received_frame'),('final_relations','received_left')):
                rows=actual.get(key)
                if not isinstance(rows,list):raise ValueError('fixed time rows missing')
                for r,fixed in enumerate(value[key]):
                    if r<len(rows) and isinstance(rows[r],dict) and rows[r].get(coordinate)==r:value[key][r]=rows[r]
                    else:error='incomplete time receipt; fixed rows retained'
                if len(rows)!=len(value[key]):error='time row count mismatch'
        except (OSError,ValueError,TypeError) as exc:error=str(exc)
        artifacts[item['path']]=dict(kind='time',value=value,error=error,unconfirmed_artifact=unconfirmed)
    phases=plan()
    for k,row in phases.items():
        actual=saved.get('state',{}).get(k)
        if isinstance(actual,dict):row.update(actual)
        if row['status'] in ('NOT_STARTED','STARTED'):row.update(status='UNRESOLVED_RECEIPT')
    return dict(status='READ_ONLY_EXISTING_PACKAGE',physical_calls_this_invocation=0,post_calls_this_invocation=0,
                saved=saved,stages=stages,phases=phases,recovered_artifacts=artifacts,
                denominators=denominators(),read_errors=errors)


class Journal:
    def __init__(self,output):self.output=Path(output);self.rows=plan();self.save()
    def save(self):dump(self.output/'state.json',self.rows)
    def call(self,name,fn):
        row=self.rows[name]
        if row['status']!='NOT_STARTED':raise RuntimeError('whole-package phase consumed: '+name)
        row.update(status='STARTED',started=1);self.save();start=time.monotonic();cpu=time.process_time()
        try:
            result=fn();row.update(status='RETURNED',returned=1,result_status=result.get('status'))
            if result.get('status')=='INTERRUPTED' or result.get('errors',{}).get('interrupted'):
                row['status']='INTERRUPTED'
            return result
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            row.update(wall_seconds=time.monotonic()-start,process_cpu_seconds=time.process_time()-cpu);self.save()
    def terminalize(self,names=PHASES):
        for k in names:
            row=self.rows[k]
            if row['status']=='NOT_STARTED':row.update(status='DEPENDENCY_UNRESOLVED',error='whole attempt consumed; no later execution')
            elif row['status']=='STARTED':row.update(status='INTERRUPTED',error='unfinished consumed phase')
        self.save()
