"""Fixed offline ledger and stdlib-only recovery. Inventory is not admission."""
from pathlib import Path
from .b_framewise_qim_time_recovery_v1_records import load,dump,output_state

LENGTHS=(181,101,180,182,178)*2
STAGES=('raw_read','old_blind_read','C','L','G')
BUDGETS=dict(raw_read=10,old_blind_read=10,codebook=1,C=10,L=10,G=10,
             old_BCH=0,old_receiver=0,codec=0,model=0,VAE=0,head=0,E41=0,K=0,KT=0,
             solver=0,DiT=0,training=0,writer=0,new_formal=0)


def denominators():
    return dict(views=10,frames=1644,bit_slots=1035720,lane_slots=8220,relations=1634,
                QIM_frames=822,QIM_relations=817,BASE_views=5,max_frame_j=152352768,
                max_replica_bit_terms=19196448768,formal_new_determined=0)


def plan():
    rows={'codebook':dict(kind='codebook',status='NOT_STARTED')}
    for i,R in enumerate(LENGTHS):
        for stage in STAGES:rows[f'v{i:02}/{stage}']=dict(kind=stage,status='NOT_STARTED',frames=R,bit_slots=R*630,lane_slots=R*5,relations=R-1)
    return rows


def validate_paths(source,output,extra=()):
    source=Path(source).expanduser().resolve();output=Path(output).expanduser().resolve()
    writes=[output]+[output.with_name(output.name+s).resolve() for s in ('-logs','-entry-cache','-preflight-v1')]+[Path(p).expanduser().resolve() for p in extra]
    if any(p==source or p.is_relative_to(source) or source.is_relative_to(p) for p in writes):raise ValueError('write path overlaps read-only old attempt')
    return source,output


def preflight(source,output):
    source,output=validate_paths(source,output);available=[]
    for i in range(10):
        folder=source/'received'/f'v{i:02}'
        available.append(dict(id=f'v{i:02}',raw_exists=(folder/'raw.npz').is_file(),old_blind_exists=(folder/'blind.json').is_file()))
    errors=[]
    if not any(x['raw_exists'] or x['old_blind_exists'] for x in available):errors.append('no planned saved inputs exist')
    if output_state(output)!='NEW':errors.append('output is not new')
    result=dict(status='PREFLIGHT_READY' if not errors else 'PREFLIGHT_UNRESOLVED',errors=errors,inputs=available,
                input_contents_read=False,rows=plan(),denominators=denominators())
    dump(output.with_name(output.name+'-preflight-v1')/'preflight.json',result);return result


def recover(output):
    saved={};errors={};rows=plan()
    for name in ('attempt','state','counts','resources','evaluation','result'):
        try:
            value=load(Path(output)/(name+'.json'))
            if not isinstance(value,dict):raise ValueError('saved receipt must be a JSON object')
            saved[name]=value
        except (OSError,ValueError) as error:errors[name]=str(error)
    legal={'NOT_STARTED','STARTED','RETURNED','FAILED','INTERRUPTED','DEPENDENCY_UNRESOLVED','RESOURCE_UNRESOLVED'}
    for key,row in rows.items():
        item=saved.get('state',{}).get(key)
        if not isinstance(item,dict) or not isinstance(item.get('status'),str) or item['status'] not in legal:row['status']='UNRESOLVED_RECEIPT'
        else:
            fixed=row.copy();row.update(item)
            for k in ('kind','frames','bit_slots','lane_slots','relations'):
                if k in fixed:row[k]=fixed[k]
            if row['status']=='STARTED':row['status']='UNRESOLVED_INTERRUPTED'
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',scientific_calls_this_invocation=0,rows=rows,saved=saved,read_errors=errors,denominators=denominators())


class Journal:
    def __init__(self,output):
        self.output=Path(output);self.rows=plan();self.counts={k:dict(planned=v,started=0,returned=0) for k,v in BUDGETS.items()};self.save()
    def save(self):dump(self.output/'state.json',self.rows);dump(self.output/'counts.json',self.counts)
    def call(self,key,fn):
        row=self.rows[key];count=self.counts[row['kind']]
        if row['status']!='NOT_STARTED' or count['started']>=count['planned']:raise RuntimeError('consumed fixed budget: '+key)
        row['status']='STARTED';count['started']+=1;self.save()
        try:
            value=fn();row['status']='RETURNED';count['returned']+=1;return value
        except BaseException as error:
            row.update(status='RESOURCE_UNRESOLVED' if type(error).__name__=='ResourceExceeded' else 'FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}');raise
        finally:self.save()
    def terminalize(self):
        for row in self.rows.values():
            if row['status']=='NOT_STARTED':row.update(status='DEPENDENCY_UNRESOLVED',error='fixed attempt ended; no retry')
        self.save()
