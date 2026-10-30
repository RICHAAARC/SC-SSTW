"""Standard-library fixed plan, durable receipts and read-only recovery."""
from pathlib import Path
import json

LENGTHS = (181,101,180,182,178)*2
BUDGETS = dict(codec_encode=10, codec_probe=12, codec_decode=12, time_read=10,
               hard_receiver=10, C=10, L=10, G=0, codebook=1, A=0, model_load=0, VAE=0, head=0, E41=0,
               K=0, KT=0, solver=0, DiT=0, training=0, writer=0)


def load(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def dump(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n');tmp.replace(path)


def denominators():
    return dict(views=10,received_frames=1644,time_bit_slots=1035720,lane_slots=8220,relations=1634,
                per_arm_frames=822,per_arm_relations=817,codewords=92672,max_hard_comparisons=152352768, max_C_frame_j=152352768,
                max_combined_kernel_items=304705536, max_C_replica_bit_terms=19196448768)


def plan():
    rows = {'codebook':dict(kind='codebook',status='NOT_STARTED')}
    for source in range(2):
        for op in ('probe','decode'): rows[f's{source}/{op}']=dict(kind='codec_'+op,status='NOT_STARTED')
    for i,length in enumerate(LENGTHS):
        for op in ('encode','probe','decode','time_read','hard_receiver','C','L'):
            rows[f'v{i:02}/{op}']=dict(kind='codec_'+op if op in ('encode','probe','decode') else op,
                                     status='NOT_STARTED',frames=length,bit_slots=length*630,lane_slots=length*5,relations=length-1)
    return rows


def output_state(output):
    p=Path(output)
    if (p/'attempt.json').exists(): return 'CONSUMED'
    return 'CONFLICT' if p.exists() and (not p.is_dir() or any(p.iterdir())) else 'NEW'


def validate_paths(qim,base,output,extra=()):
    inputs=[Path(p).expanduser().resolve() for p in (qim,base)]
    output=Path(output).expanduser().resolve()
    writes=[output]+[output.with_name(output.name+s).resolve() for s in ('-logs','-entry-cache','-preflight-v1')]+[Path(p).expanduser().resolve() for p in extra]
    for path in writes:
        for source in inputs:
            protected=source.parent
            if path==protected or path.is_relative_to(protected) or protected.is_relative_to(path):
                raise ValueError('write destination overlaps read-only input directory: '+str(path))
    return *inputs,output


def preflight(qim,base,output):
    qim,base,output=validate_paths(qim,base,output)
    errors={};inputs=[]
    for name,path in (('qim_media',qim),('base_media',base)):
        item=dict(name=name,path=str(path),readable=False)
        try:
            with path.open('rb') as stream:
                if not stream.read(1): raise ValueError('empty media file')
            item['readable']=True
        except (OSError,ValueError) as error: item['error']=str(error)
        inputs.append(item)
    if not all(x['readable'] for x in inputs): errors['inputs']='e02 BASE/QIM media incomplete; CPU confirmation NOT_EXECUTED'
    if output_state(output)!='NEW': errors['output']='existing output; no overwrite'
    result=dict(status='PREFLIGHT_READY' if not errors else 'PREFLIGHT_UNRESOLVED',errors=errors,
                inputs=inputs,rows=plan(),denominators=denominators(),physical_attempt_reserved=False)
    dump(output.with_name(output.name+'-preflight-v1')/'preflight.json',result)
    return result


def recover(output):
    saved={};errors={};rows=plan()
    for name in ('attempt','state','counts','evaluation','result','resources'):
        try:
            value=load(Path(output)/(name+'.json'))
            if not isinstance(value,dict): raise ValueError('object required')
            saved[name]=value
        except (OSError,ValueError) as e: errors[name]=str(e)
    for key,row in rows.items():
        value=saved.get('state',{}).get(key)
        if not isinstance(value,dict) or value.get('status') not in ('NOT_STARTED','STARTED','RETURNED','FAILED','INTERRUPTED','DEPENDENCY_UNRESOLVED','RESOURCE_UNRESOLVED'):
            row['status']='UNRESOLVED_RECEIPT'
        else:
            fixed=row.copy();row.update(value)
            for field in ('kind','frames','bit_slots','lane_slots','relations'):
                if field in fixed: row[field]=fixed[field]
            if row['status']=='STARTED': row['status']='UNRESOLVED_INTERRUPTED'
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_calls_this_invocation=0,saved=saved,rows=rows,
                read_errors=errors,denominators=denominators())


class Journal:
    def __init__(self,output):
        self.output=Path(output);self.rows=plan()
        self.counts={k:dict(planned=v,started=0,returned=0) for k,v in BUDGETS.items()}
        self.save()

    def save(self):
        dump(self.output/'state.json',self.rows);dump(self.output/'counts.json',self.counts)

    def call(self,key,fn):
        row=self.rows[key];kind=row['kind'];count=self.counts[kind]
        if row['status']!='NOT_STARTED' or count['started']>=count['planned']: raise RuntimeError('consumed fixed call: '+key)
        row['status']='STARTED';count['started']+=1;self.save()
        try:
            value=fn();row['status']='RETURNED';count['returned']+=1;return value
        except BaseException as e:
            row.update(status='RESOURCE_UNRESOLVED' if type(e).__name__=='ResourceExceeded' else 'FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
            raise
        finally: self.save()

    def terminalize(self,reason):
        for row in self.rows.values():
            if row['status']=='NOT_STARTED': row.update(status='DEPENDENCY_UNRESOLVED',error=reason)
        self.save()
