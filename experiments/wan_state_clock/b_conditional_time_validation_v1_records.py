"""One package reservation around inherited A/B materialization and 43 receiver calls."""
from pathlib import Path
from .b_minimal_l_v1_records import load,dump,output_state
from . import b_minimal_l_materialization_v1_records as material
from . import b_conditional_time_output_v1_records as receiver

PHASES=('A','B','receiver')


def protocol(config):
    return dict(source_id=config['source_id'],terminal_relative=config['terminal_relative'],
                frames=181,global_start=0,model_id=config['model_id'])


def budgets():
    return dict(materialization_A=dict(material.A_BUDGETS),materialization_B=dict(material.B_BUDGETS),
                receiver=receiver.budgets('media'),receiver_core_calls=43,
                materialization_codec_calls=6,receiver_codec_calls=17,chain_codec_calls=23)


def denominators(config):
    return dict(materialization=material.denominators(protocol(config),17),receiver=receiver.denominators())


def preflight(config,output):
    try:
        source=Path(config['source']).expanduser().resolve()
        output=Path(output).expanduser().resolve()
        fixed=protocol(config)
        if not all(isinstance(fixed[k],str) and fixed[k].strip() for k in ('source_id','terminal_relative','model_id')):
            raise ValueError('source role/path and model location must be configured')
        terminal=(source/fixed['terminal_relative']).resolve()
        if not terminal.is_relative_to(source): raise ValueError('P0 path escapes read-only source')
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('package output overlaps read-only source')
        if not terminal.is_file(): raise FileNotFoundError('configured '+fixed['source_id']+' P0 unavailable: '+str(terminal))
        if output_state(output)!='NEW': raise ValueError('output already contains records')
        return dict(status='PREFLIGHT_READY',input_paths=[str(terminal)],protected_roots=[str(source)],
                    physical_attempt_reserved=False,budgets=budgets(),denominators=denominators(config))
    except (OSError,ValueError,KeyError,TypeError) as error:
        return dict(status='PREFLIGHT_UNRESOLVED',error=f'{type(error).__name__}: {error}',physical_calls_this_invocation=0)


def plan():
    return {name:dict(status='NOT_STARTED') for name in PHASES}


def recover(output):
    output=Path(output);saved={};errors={}
    for name in ('attempt','state','counts','result'):
        try:
            value=load(output/(name+'.json'))
            if not isinstance(value,dict): raise ValueError('saved object required')
            saved[name]=value
        except (OSError,ValueError) as error: errors[name]=str(error)
    state=plan()
    for name,row in state.items():
        actual=saved.get('state',{}).get(name)
        if not isinstance(actual,dict) or actual.get('status') not in ('RETURNED','FAILED','INTERRUPTED','DEPENDENCY_UNRESOLVED'):
            row.update(status='UNRESOLVED_RECEIPT',error='missing/unfinished phase; consumed without resume')
        else: row.update(actual)
    stages={}
    for name,folder in (('A',output/'materialization/A'),('B',output/'materialization/B'),('receiver',output/'receiver')):
        stages[name]={}
        for file in ('attempt','state','counts','result','resources','native','solver','linear','evaluation'):
            path=folder/(file+'.json')
            if not path.exists(): continue
            try: stages[name][file]=load(path)
            except (OSError,ValueError) as error: errors[name+'/'+file]=str(error)
    return dict(status='READ_ONLY_EXISTING_PACKAGE',physical_calls_this_invocation=0,
                saved=saved,phases=state,stages=stages,read_errors=errors)


class Journal:
    def __init__(self,output):
        self.output=Path(output);self.rows=plan();self.save()
    def save(self): dump(self.output/'state.json',self.rows)
    def call(self,name,fn):
        row=self.rows[name]
        if row['status']!='NOT_STARTED': raise RuntimeError('package phase consumed: '+name)
        row['status']='STARTED';self.save()
        try:
            result=fn();row['status']='RETURNED';row['result_status']=result.get('status');return result
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            raise
        finally:self.save()
    def terminalize(self):
        for row in self.rows.values():
            if row['status']=='NOT_STARTED':row.update(status='DEPENDENCY_UNRESOLVED',error='package attempt ended; no later Run-all execution')
            elif row['status']=='STARTED':row.update(status='INTERRUPTED',error='package phase unfinished; no retry')
        self.save()
