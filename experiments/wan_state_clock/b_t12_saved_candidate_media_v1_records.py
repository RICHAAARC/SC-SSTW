"""Fixed single-candidate plan and durable, non-retrying operation receipts."""
from collections import Counter
from pathlib import Path
import json
import os
import time
import uuid

ROOT = '/content/drive/MyDrive/Video-WM/T12-Saved-Candidate-Media-V1'
DEFAULTS = dict(candidate=ROOT+'/inputs/run02', output=ROOT+'/attempt01',
    original='/content/drive/MyDrive/Video-WM_Result/main_method/paper_cohort_v1/paper-joint-t12/attempt01',
    host='/content/drive/MyDrive/Video-WM_Cache/paper-checkpoints-v1/cohort/paper-joint-t12/prepare-b3ccfc5ef961476abdedb6607b64a736/host',
    model_id='Wan-AI/Wan2.1-T2V-1.3B-Diffusers')
VIEWS = ('FULL','CROP','DELETE1','REPEAT1','DELETE3')
LENGTHS = (181,101,180,182,178)
STARTS = (0,*range(1,181,4))
LABEL = 'DECLARED_OVER_BUDGET_EXPLORATORY_CANDIDATE'
DENOMINATORS = dict(nodes=822,edges=817,physical_words=8220,physical_bits=517860,
    active_words=8140,unwritten_words=80,derived_F_roles=4110,role_packets=12330,
    raw_payload_slots=19200,local_payload_slots=1440,final_payload_slots=480,
    B_slots=160,K_slots=160,U_slots=160,claims=20,quality_pairs=5)


def read(path):
    def invalid(value): raise ValueError('non-finite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with tmp.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=1)
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def items():
    return [dict(id=f'input{i+10}',arm='NEW',view=v,length=n,reference=f'input{i+5}')
            for i,(v,n) in enumerate(zip(VIEWS,LENGTHS))]


def banks(item):
    return [dict(view_id=f"{item['id']}/E41/a{w+phase}",start=w+phase,query=q,
        query_start=w,phase=phase,encoder_input_support=list(range(w+phase,w+phase+41)))
        for q,w in enumerate((0,(item['length']-45)//2,item['length']-45)) for phase in range(4)]


def plan():
    rows={}
    def add(key,kind): rows[key]=dict(kind=kind,status='NOT_STARTED',started=0,returned=0)
    add('native','native_replay');add('quantize','RGB8')
    for kind in ('encode','probe','RGB'): add('first/'+kind,'codec_'+kind)
    for item in items():
        uid=item['id']
        for kind in ('encode','probe','RGB','Y'): add(uid+'/'+kind,'codec_'+kind)
        for kind in ('read','hard','relative','plan'): add(uid+'/'+kind,'time_'+kind)
        for b in banks(item):
            for kind in ('E41','project','key00','key01'):
                add(b['view_id']+'/'+kind,'bits' if kind.startswith('key') else kind)
        for strategy in ('B_NEW','K_NEW','U_NEW'):
            for q in range(3): add(f'{uid}/{strategy}/q{q}','local')
            add(f'{uid}/{strategy}/aggregate','aggregate')
        for key in ('key00','key01'):
            for claim in ('q0','q1'): add(f'{uid}/{key}/{claim}','claim')
        for kind in ('probe','RGB'): add(uid+'/ref/'+kind,'ref_codec_'+kind)
        add(uid+'/quality','quality')
    add('post','post')
    return rows


def validate_paths(config):
    output=Path(config['output']).resolve()
    sources=[Path(config[k]).resolve() for k in ('candidate','original','host')]
    if 'Video-WM_Result' in output.parts or any(output==s or output.is_relative_to(s) or s.is_relative_to(output) for s in sources):
        raise ValueError('diagnostic output must be separate from retained inputs and Video-WM_Result')
    return output


def preflight(config):
    validate_paths(config)
    candidate=Path(config['candidate']);prediction=candidate/'r_predicted_packed.npy'
    required=[candidate/'coefficients.pt']
    required += [prediction] if prediction.is_file() else [candidate/f'r_predicted_packed.npy.part{i:03}' for i in range(2)]
    required += [Path(config['original'])/'private/head.pt',Path(config['host'])/'BASE_preclip.npy']
    required += [Path(config['host'])/'features'/f'{s:03}.pt' for s in STARTS]
    missing=[]
    for path in required:
        try:
            with path.open('rb') as stream:
                if not stream.read(1): raise ValueError('empty file')
        except (OSError,ValueError) as error: missing.append(dict(file=str(path),error=str(error)))
    return dict(status='INPUTS_READABLE' if not missing else 'DEPENDENCY_UNRESOLVED',
        errors=missing,physical_started=0,tensor_contents_checked=False)


class Book:
    def __init__(self,config):
        self.root=validate_paths(config);self.config=config;self.changed=False
        self.root.mkdir(parents=True,exist_ok=True)
        self.lock=(self.root/'worker.lock').open('a+')
        import fcntl
        try: fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: self.lock.close();raise RuntimeError('another worker is using this diagnostic output')
        if (self.root/'attempt.json').is_file():
            self.config=read(self.root/'attempt.json')['inputs'];self.rows=read(self.root/'operations.json')
            for row in self.rows.values():
                if row['status']=='STARTED': row.update(status='INTERRUPTED_CONSUMED',error='prior invocation has no return receipt')
            native=read(self.root/'native_calls.json')
            for row in native.values():
                if row['status']=='STARTED': row.update(status='INTERRUPTED_CONSUMED')
            write(self.root/'native_calls.json',native)
            self.save()
        else:
            if any(p.name!='worker.lock' for p in self.root.iterdir()): raise ValueError('nonempty unreserved diagnostic output')
            self.rows=plan()
            write(self.root/'attempt.json',dict(label=LABEL,inputs=config,created_at=time.time(),denominators=DENOMINATORS,
                old_cpu_budget='FAIL_RETAINED',scientific_pass=None))
            write(self.root/'native_calls.json',{str(s):dict(started=0,returned=0,status='NOT_STARTED') for s in STARTS})
            write(self.root/'model_loads.json',[]);self.save()

    def save(self): write(self.root/'operations.json',self.rows)

    def available(self,key): return self.rows[key]['status']=='RETURNED' and not self.rows[key].get('artifact_error')

    def observation_status(self,key):
        return 'ARTIFACT_UNAVAILABLE' if self.rows[key].get('artifact_error') else self.rows[key]['status']

    def call(self,key,function,*,load=None,save=None,dependencies=()):
        row=self.rows[key]
        if row['status']=='RETURNED':
            try: return load() if load else row.get('value')
            except Exception as error:
                row['artifact_error']=f'{type(error).__name__}: {error}';self.save();return None
        if row['status']!='NOT_STARTED': return None
        unavailable=[d for d in dependencies if not self.available(d) or self.rows[d].get('artifact_error')]
        if unavailable:
            row['dependencies_unavailable']=unavailable;self.save();return None
        row.update(status='STARTED',started=1,started_at=time.time());self.changed=True;self.save();tick=time.monotonic()
        try:
            value=function()
            if save: save(value)
            row.update(status='RETURNED',returned=1,seconds=time.monotonic()-tick)
            if not save: row['value']=value
            self.save();return value
        except BaseException as error:
            row.update(status='INTERRUPTED_CONSUMED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED_CONSUMED',
                error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-tick)
            self.save()
            if isinstance(error,(KeyboardInterrupt,SystemExit)): raise
            return None

    def json(self,key,path,function,dependencies=()):
        target=self.root/path
        return self.call(key,function,load=lambda:read(target),save=lambda v:write(target,v),dependencies=dependencies)

    def close_blind(self):
        for key,row in self.rows.items():
            if key!='post' and row['status']=='NOT_STARTED': row.update(status='DEPENDENCY_UNRESOLVED')
        self.save();write(self.root/'blind_terminal.json',dict(status='ALL_BLIND_TERMINAL',truth_opened=False,time=time.time()))

    def finish(self):
        result=dict(label=LABEL,scientific_pass=None,denominators=DENOMINATORS,
            statuses=dict(Counter(r['status'] for r in self.rows.values())),
            counts={k:dict(planned=sum(r['kind']==k for r in self.rows.values()),
                started=sum(r['started'] for r in self.rows.values() if r['kind']==k),
                returned=sum(r['returned'] for r in self.rows.values() if r['kind']==k)) for k in sorted({r['kind'] for r in self.rows.values()})})
        result['engineering_status']='COMPLETED' if all(r['status']=='RETURNED' for r in self.rows.values()) else 'INCOMPLETE_RETAINED'
        write(self.root/'result.json',result);return result

    def close(self): self.lock.close()
