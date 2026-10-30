"""Independent A/B reservations, fixed budgets and read-only recovery (stdlib)."""
from pathlib import Path
import json
import time
from .b_framewise_qim_v1_records import load,dump
from .b_framewise_qim_saved_candidate_v1_records import output_state,STARTS,LAYERS

CONFIG=dict(source_id='b2-m3-e02',terminal_relative='private/b2-m3-e02/P0_terminal.pt',
            frames=181,global_start=0,model_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
A_BUDGETS=dict(model_load=1,D181=1,baseline_head=46,solver=1,targets=1,K=4200,KT=4200,
               codec_encode=1,codec_probe=1,codec_decode=1,time_reads=4,
               head_replay=0,BASE_constraint_P=0,Aa_constraint_P=0,E41=0,DiT=0,training=0)
B_BUDGETS=dict(model_load=1,D181=0,baseline_head=0,solver=0,targets=0,K=1,KT=0,
               codec_encode=1,codec_probe=1,codec_decode=1,time_reads=4,
               head_replay=46,BASE_constraint_P=1,Aa_constraint_P=1,E41=0,DiT=0,training=0)


def denominators():
    return dict(source_id='b2-m3-e02',frames=181,arms=2,time_layers=4,time_read_slots=912240,
                identity_reads=0,new_identity_claim=False,codec_calls=6,receiver_codec_calls=34,chain_codec_calls=40)


def plan(stage):
    arm='BASE' if stage=='A' else 'QIM'
    keys=('capture','media','solver') if stage=='A' else ('restore','linear_constraints','native','media')
    rows={k:dict(status='NOT_STARTED',slots=0) for k in keys}
    rows.update({arm+'/'+layer:dict(status='NOT_STARTED',slots=114030) for layer in LAYERS})
    return rows


def validate_paths(source,output,extra=()):
    source,output=Path(source).expanduser().resolve(),Path(output).expanduser().resolve()
    paths=[output]+[output.with_name(output.name+s).resolve() for s in ('-preflight-v1','-logs','-entry-cache')]
    paths.extend(Path(p).expanduser().resolve() for p in extra)
    for path in paths:
        if path==source or path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError('write destination overlaps read-only source: '+str(path))
    for name in ('A','B'):
        stage=(output/name).resolve()
        if stage.parent!=output: raise ValueError('stage output must remain inside new output')
    return source,output


def cache_paths():
    return ['coefficients.pt','fixed_targets.npy','BASE_preclip.npy']+[f'features/{s:03}.pt' for s in STARTS]


def ready_B(output):
    a=Path(output)/'A';errors={}
    try:
        result=load(a/'result.json')
        if not isinstance(result,dict) or result.get('stage_complete') is not True or result.get('finite_candidate_saved') is not True:
            errors['A']='A did not complete with a finite saved candidate; no recapture or re-solve'
    except (OSError,ValueError) as error: errors['A']=str(error)
    for name in cache_paths():
        if not (a/name).is_file(): errors[name]='missing saved A input'
    return errors


def preflight(source,output,stage):
    source,output=validate_paths(source,output)
    errors={}
    if output.exists() and not output.is_dir(): errors['output']='output is not a directory'
    if stage=='A':
        if not (source/CONFIG['terminal_relative']).is_file(): errors['terminal']='fixed e02 P0 missing; tensor format/finite check occurs after reservation'
    elif stage=='B': errors.update(ready_B(output))
    else: raise ValueError('stage A or B required')
    if output_state(output/stage)!='NEW': errors['stage']='stage output is '+output_state(output/stage)
    result=dict(status='PREFLIGHT_READY' if not errors else 'PREFLIGHT_UNRESOLVED',stage=stage,errors=errors,
                physical_attempt_reserved=False,denominators=denominators())
    dump(output.with_name(output.name+'-preflight-v1')/(stage+'.json'),result)
    return result


def needs_work(output):
    output=Path(output)
    if output_state(output/'B')!='NEW': return False
    state=output_state(output/'A')
    return state=='NEW' or state=='CONSUMED' and not ready_B(output)


def recover_stage(output,stage):
    saved={};errors={};rows=plan(stage)
    for name in ('attempt','state','counts','solver','native','linear','resources','result'):
        try:
            value=load(Path(output)/stage/(name+'.json'))
            if not isinstance(value,dict): raise ValueError('saved object required')
            saved[name]=value
        except (OSError,ValueError) as error: errors[name]=str(error)
    for key,row in rows.items():
        value=saved.get('state',{}).get(key)
        if isinstance(value,dict):
            row.update(value)
            if row.get('status')=='STARTED': row['status']='UNRESOLVED_INTERRUPTED'
        else: row['status']='UNRESOLVED_RECEIPT'
    return dict(status='READ_ONLY_EXISTING_STAGE',stage=stage,saved=saved,rows=rows,read_errors=errors,physical_calls_this_invocation=0)


def recover(output):
    return dict(status='READ_ONLY_EXISTING_PACKAGE',A=recover_stage(output,'A'),B=recover_stage(output,'B'),
                physical_calls_this_invocation=0,denominators=denominators())


class Journal:
    def __init__(self,output,stage):
        self.output=Path(output);self.rows=plan(stage)
        self.counts={k:dict(planned=v,started=0,returned=0) for k,v in (A_BUDGETS if stage=='A' else B_BUDGETS).items()}
        self.save()
    def save(self):
        dump(self.output/'state.json',self.rows);dump(self.output/'counts.json',self.counts)
    def event(self,kind,done):
        row=self.counts[kind]
        if not done and row['started']>=row['planned']: raise RuntimeError('fixed call consumed: '+kind)
        if done and row['returned']>=row['started']: raise RuntimeError('return without start: '+kind)
        row['returned' if done else 'started']+=1;self.save()
    def call(self,kind,fn):
        self.event(kind,False);value=fn();self.event(kind,True);return value
    def operation(self,key,fn):
        row=self.rows[key]
        if row['status']!='NOT_STARTED': raise RuntimeError('operation consumed: '+key)
        row['status']='STARTED';self.save();start=time.monotonic()
        try:
            value=fn();row['status']='RETURNED';return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            raise
        finally: row['wall_seconds']=time.monotonic()-start;self.save()
    def terminalize(self,reason):
        for row in self.rows.values():
            if row['status']=='NOT_STARTED': row.update(status='DEPENDENCY_UNRESOLVED',error=reason)
        self.save()
