"""Card103 once-only USER-run task. Import is stdlib + pure receiver only."""
import json,time
from pathlib import Path
from collections import Counter
from main.tube_state import b_single_internal_deletion_v1 as method
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/b_single_internal_deletion_v1.json'
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-Single-Internal-Deletion-V1/fixed-viewed-development-v1'
DEFAULT_SOURCE='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1'
KEYS={'key00':'watermark','key01':'watermark-wrong'}
ARMS=('OFF','P0','PM')
QUERIES={f'q{i:02}':dict(arm=arm,key=key) for i,(arm,key) in enumerate((a,k) for a in ARMS for k in KEYS)}

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)

def execution_artifacts(output):
    output=Path(output)
    names=('attempt.json','result.json','ledger.json','thresholds.json','protocol.json','roles.json','background.json','post_truth_access.json','evaluation.json','progress.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for name in ('private','media','observations','families','blind','receipts','markers','quality'):
        p=output/name
        if p.exists():found.extend(str(x.relative_to(output)) for x in p.rglob('*') if x.is_file() or x.is_symlink())
    return sorted(found)

def existing(output):
    saved={};errors={}
    for name in ('result.json','thresholds.json','evaluation.json','ledger.json','protocol.json','roles.json'):
        p=Path(output)/name
        if p.is_file():
            try:saved[name]=json.loads(p.read_text())
            except (OSError,ValueError) as e:errors[name]=str(e)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_reentry=False,saved=saved,read_errors=errors,artifacts=execution_artifacts(output))

def load_config(path=CONFIG):
    cfg=json.loads(Path(path).read_text());expected=[
        ('b101-s01','calibration',1,74,59),('b101-s03','calibration',18,53,23),
        ('b101-s02','evaluation',2,74,58),('b101-s04','evaluation',29,57,25)]
    if [(h['id'],h['role'],h['a'],h['b'],h['d']) for h in cfg['private_roster']]!=expected:
        raise ValueError('fixed Card103 geometry/roles required; no scan')
    return cfg

def load_background(path):
    value=json.loads(Path(path).read_text())
    if value.get('status')!='BACKGROUND_FROZEN':raise ValueError('usable frozen background required')
    for key in KEYS:
        model=value['models'][key];mean=model['mean'];factor=model['factor']
        if len(mean)!=40 or len(factor)!=40 or any(len(r)!=40 for r in factor):raise ValueError('background mean40/factor40x40')
        if not all(method.finite(x) for x in mean+[x for r in factor for x in r]):raise ValueError('nonfinite background')
        if any(factor[i][i]<=0 or any(factor[i][j]!=0 for j in range(i+1,40)) for i in range(40)):raise ValueError('lower positive background factor')
    return value

def units(cfg):
    u=dict(background='saved_background_load',vae='VAE_load',calibration='complete_pair_null_calibration',post='private_post')
    for k in KEYS:u['book/'+k]='public_book'
    for host in cfg['private_roster']:
        h=host['id'];evaluation=host['role']=='evaluation'
        for arm in ARMS:
            prefix=h+'/'+arm
            for n,k in [('source','RGB_source_load'),('edit','real_frame_deletion'),('encode','MP4_encode'),('probe','MP4_probe'),('decode','MP4_decode')]:u[prefix+'/'+n]=k
            for anchor in method.ANCHORS:
                for p in range(4):
                    v=prefix+'/'+anchor+'/p'+str(p)
                    u[v+'/prepare']='E41_prepare';u[v+'/E']='E41'
                    for key in KEYS:u[v+'/'+key+'/Q']='Q'
                    if evaluation:
                        u[v+'/FFT']='receiver_FFT'
                        for key in KEYS:u[v+'/'+key+'/votes']='votes'
                for key in KEYS:u[prefix+'/'+anchor+'/'+key+'/score']='anchor137_score'
        for q in QUERIES:
            u[h+'/'+q+'/pair']='joint1891_score'
            if evaluation:
                u[h+'/'+q+'/decision']='joint_decision'
                u[h+'/'+q+'/references']='equal_observation_references'
        for pair in ('OFF_to_P0','P0_to_PM'):
            for layer in ('input_rgb8','edited_mp4'):u[h+'/quality/'+pair+'/'+layer]='paired_quality'
    return u

class Journal:
    def __init__(self,out,cfg,backend):
        self.out,self.backend=Path(out),backend
        self.data={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units(cfg).items()}
        self.save()
    def save(self):
        dump(self.out/'ledger.json',self.data)
        c=Counter(x['status'] for x in self.data.values())
        dump(self.out/'progress.json',dict(planned=len(self.data),returned=c['RETURNED'],failed=c['FAILED'],
            active=next((k for k,v in self.data.items() if v['status']=='STARTED'),None)))
    def call(self,slot,fn,deps=()):
        row=self.data[slot]
        if row['status']!='NOT_ENTERED':raise RuntimeError('attempt already consumed: '+slot)
        bad=[k for k in deps if self.data[k]['status']!='RETURNED']
        if bad:row['unavailable_from']=bad;self.save();return None
        row.update(status='STARTED',attempts=1,started_unix=time.time());self.save()
        before=dict(getattr(self.backend,'blocks',{}))
        try:
            value=fn();row['status']='RETURNED';return value
        except BaseException as e:
            row.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
            if not isinstance(e,Exception):raise
        finally:
            row['finished_unix']=time.time();row['seconds']=row['finished_unix']-row['started_unix']
            if row['kind']=='E41':
                after=getattr(self.backend,'blocks',{})
                row['internal_blocks']={k:(after.get(k)-before.get(k) if isinstance(before.get(k),int) and isinstance(after.get(k),int) else None) for k in ('encoder','decoder')}
            dump(self.out/'receipts'/(slot+'.json'),row);self.save()
    def record(self,slot,fn,path,deps=()):
        def op():
            v=fn();dump(self.out/path,v);return v
        return self.call(slot,op,deps)
    def tensor(self,slot,fn,path,deps=()):
        def op():
            v=fn();self.backend.save_tensor(self.out/path,v);return v
        return self.call(slot,op,deps)

def empty_anchor():
    return dict(status='TECHNICAL_UNRESOLVED',candidates=[dict(phase=p,source_start=s,S=None,score_status='NOT_ENTERED') for p,s in method.H],phase_local_bits={})

def collect(host,source,out,backend,journal,books,background):
    """Private actuator ends before prepare(received,...); pure methods get no host/edit."""
    h=host['id'];evaluation=host['role']=='evaluation';families={}
    for arm in ARMS:
        prefix=h+'/'+arm;base='media/'+prefix
        original=journal.call(prefix+'/source',lambda:backend.load_source(Path(source)/'hosts'/h/'media'/arm/'readback.pt'))
        deleted=journal.tensor(prefix+'/edit',lambda:backend.edit(original,host),base+'/input_rgb8.pt',[prefix+'/source'])
        path=Path(out)/base/'video.mp4'
        journal.record(prefix+'/encode',lambda:backend.save_mp4(deleted,path),base+'/codec.json',[prefix+'/edit'])
        probe=journal.record(prefix+'/probe',lambda:backend.probe(path),base+'/probe.json',[prefix+'/encode'])
        received=journal.tensor(prefix+'/decode',lambda:backend.read_mp4(path,probe),base+'/received.pt',[prefix+'/probe'])
        del original,deleted
        for anchor,w in method.ANCHORS.items():
            qs={k:{} for k in KEYS};bitbanks={k:{} for k in KEYS}
            for p in range(4):
                v=prefix+'/'+anchor+'/p'+str(p);rel='observations/'+v
                view=journal.call(v+'/prepare',lambda:backend.prepare(received,w,p),[prefix+'/decode'])
                z=journal.tensor(v+'/E',lambda:backend.encode(view),rel+'/E41.pt',[v+'/prepare','vae'])
                for key in KEYS:
                    qs[key][str(p)]=journal.record(v+'/'+key+'/Q',lambda key=key:backend.observe(z,books[key]),rel+'/'+key+'/Q.json',[v+'/E','book/'+key])
                if evaluation:
                    spec=journal.call(v+'/FFT',lambda:backend.spectrum(z),[v+'/E'])
                    for key,secret in KEYS.items():
                        votes=journal.record(v+'/'+key+'/votes',lambda secret=secret:backend.votes(spec,secret),rel+'/'+key+'/votes.json',[v+'/FFT'])
                        bitbanks[key][str(p)]=[b.get('decoded') for r in votes['local_cells'] for b in r['bits']] if votes is not None else [None]*80
                    del spec
                del z,view
            for key in KEYS:
                def score(key=key):
                    value=backend.score(qs[key],background['models'].get(key))
                    value['phase_local_bits']=bitbanks[key] if evaluation else {}
                    return value
                families[arm,key,anchor]=journal.record(prefix+'/'+anchor+'/'+key+'/score',score,
                    'families/'+prefix+'/'+anchor+'/'+key+'.json') or empty_anchor()
        del received
    result={}
    for q,meta in QUERIES.items():
        left,right=[families[meta['arm'],meta['key'],a] for a in ('left','right')]
        value=journal.record(h+'/'+q+'/pair',lambda:method.pair_family(left,right),'families/'+h+'/'+q+'.pairs.json') or method.empty_family()
        result[q]=dict(family=value,anchors=dict(left=left,right=right))
    return result

def blind_host(host_id,collection,out,backend,journal,thresholds):
    h=host_id
    for q,pack in collection.items():
        bitbanks={a:pack['anchors'][a].get('phase_local_bits',{}) for a in method.ANCHORS}
        def main():
            prepared=method.prepare(pack['family'],thresholds)
            # Persist actual canonical choices before the separate cache read.
            dump(Path(out)/'blind'/h/(q+'.choices.json'),prepared)
            return method.read(prepared,bitbanks)
        journal.record(h+'/'+q+'/decision',main,'blind/'+h+'/'+q+'.main.json')
        def controls():
            cont=method.read(method.prepare(method.subset(pack['family'],'continuous'),thresholds),bitbanks)
            unaligned=method.read(method.prepare(method.subset(pack['family'],'unaligned'),thresholds),bitbanks)
            independent={a:backend.independent(pack['anchors'][a]) for a in method.ANCHORS}
            union=method.independent_union(independent['left'],independent['right'])
            return dict(continuous=cont,unaligned=unaligned,independent=independent,independent_union=union)
        journal.record(h+'/'+q+'/references',controls,'blind/'+h+'/'+q+'.references.json')

def run(source=DEFAULT_SOURCE,output=DEFAULT_OUTPUT,config=CONFIG,*,backend=None):
    out=Path(output)
    if execution_artifacts(out):return existing(out)
    cfg=load_config(config);out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as stream:json.dump(dict(started_unix=time.time(),one_attempt=True,development_only=True),stream)
    except FileExistsError:return existing(out)
    dump(out/'private/roster.json',cfg['private_roster'])
    dump(out/'roles.json',{h['id']:h['role'] for h in cfg['private_roster']})
    dump(out/'protocol.json',method.public_protocol())
    result=dict(status='STARTED',development_only=True,physical_reentry=False,
        fixed_evaluation_queries=12,fixed_evaluation_paths=703452,fixed_evaluation_message_slots=384,
        fixed_query_rows=[dict(host=h['id'],role=h['role'],query=q,metadata_control_only=meta,pair_slots=1891,path_slots=58621,message_slots=32) for h in cfg['private_roster'] for q,meta in QUERIES.items()])
    journal=None;thresholds=dict(status='CALIBRATION_UNRESOLVED',tau=None)
    try:
        if backend is None:
            from runtime.wan.b_single_internal_deletion_backend_v1 import WanBackend
            backend=WanBackend(cfg['model_id'])
        journal=Journal(out,cfg,backend)
        background=journal.record('background',lambda:load_background(Path(source)/'background.json'),'background.json')
        books={k:journal.call('book/'+k,lambda secret=s:backend.book(secret),['background']) for k,s in KEYS.items()}
        journal.record('vae',backend.load_vae,'vae.json',['background'])
        if background is not None and background.get('status')=='BACKGROUND_FROZEN':
            cal={}
            for host in cfg['private_roster']:
                if host['role']=='calibration':cal[host['id']]=collect(host,source,out,backend,journal,books,background)
            def calibrate():
                value=method.calibrate({h:{int(q[1:]):r['family'] for q,r in fs.items()} for h,fs in cal.items()})
                value['frozen_unix']=time.time() if value['status']=='FROZEN' else None
                return value
            thresholds=journal.record('calibration',calibrate,'thresholds.json') or thresholds
            if thresholds['status']=='FROZEN':
                for host in cfg['private_roster']:
                    if host['role']=='evaluation':
                        collection=collect(host,source,out,backend,journal,books,background)
                        blind_host(host['id'],collection,out,backend,journal,thresholds)
                required=[h['id']+'/'+q+'/'+label for h in cfg['private_roster'] if h['role']=='evaluation' for q in QUERIES for label in ('decision','references')]
                # Read actual saved JSON before allowing semantic post.
                if all(journal.data[k]['status']=='RETURNED' for k in required):
                    actual={}
                    for host in cfg['private_roster']:
                        h=host['id']
                        if host['role']!='evaluation':continue
                        actual[h]={}
                        for q,meta in QUERIES.items():
                            main=json.loads((out/'blind'/h/(q+'.main.json')).read_text())
                            ref=json.loads((out/'blind'/h/(q+'.references.json')).read_text())
                            anchor_bits={a:json.loads((out/'families'/h/meta['arm']/a/(meta['key']+'.json')).read_text()).get('phase_local_bits',{}) for a in method.ANCHORS}
                            actual[h][q]=dict(ref,main=main,anchor_bits=anchor_bits,metadata_post_only=meta)
                    dump(out/'markers/blind_complete.json',dict(saved_unix=time.time(),actual_decisions=12,actual_reference_bundles=12,semantic_truth_opened=False))
                    def post():
                        dump(out/'post_truth_access.json',dict(opened_unix=time.time(),actual_blind_saves_read_back=True))
                        private={r['id']:r for r in json.loads((out/'private/roster.json').read_text())}
                        bits=json.loads((Path(source)/'protocol.json').read_text())['config']['writer']['bits']
                        from runtime.wan.b_single_internal_deletion_evaluation_v1 import evaluate,mapping
                        value=evaluate(actual,private,bits)
                        dump(out/'evaluation.json',value)
                        for host in cfg['private_roster']:
                            h=host['id'];source_frames=mapping(private[h])
                            for layer,filename in [('input_rgb8','input_rgb8.pt'),('edited_mp4','received.pt')]:
                                # Exactly six quality tensor reads/host; no source/model reentry.
                                arrays={arm:backend.load_tensor(out/'media'/h/arm/filename) for arm in ARMS}
                                for pair in ('OFF_to_P0','P0_to_PM'):
                                    left,right=pair.split('_to_')
                                    journal.record(h+'/quality/'+pair+'/'+layer,
                                        lambda left=left,right=right,pair=pair:backend.quality(arrays[left],arrays[right],source_frames,pair+'/'+layer),
                                        'quality/'+h+'/'+pair+'/'+layer+'.json')
                                del arrays
                        return dict(status=value['status'])
                    journal.call('post',post)
        if not (out/'thresholds.json').exists():dump(out/'thresholds.json',thresholds)
        if thresholds['status']!='FROZEN':
            for k,row in journal.data.items():
                if row['status']=='NOT_ENTERED':row['unavailable_from']=['CALIBRATION_OR_BACKGROUND_UNAVAILABLE']
        numeric_files=list((out/'families').rglob('*.json'))
        result['numeric_unresolved_files']=[str(p.relative_to(out)) for p in numeric_files if json.loads(p.read_text()).get('status')!='SCORES_COMPLETE']
        result['status']='COMPLETED_DEVELOPMENT_ONLY' if all(x['status']=='RETURNED' for x in journal.data.values()) and not result['numeric_unresolved_files'] else 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    except BaseException as error:
        result.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
    finally:
        if backend is not None:
            try:backend.release_vae()
            except Exception as e:result['release_error']=str(e)
        if journal is not None:
            journal.save();result['counts']=dict(Counter(r['status'] for r in journal.data.values()))
            result['kinds_returned']=dict(Counter(r['kind'] for r in journal.data.values() if r['status']=='RETURNED'))
            result['not_entered']=[k for k,v in journal.data.items() if v['status']=='NOT_ENTERED']
        result['threshold_status']=thresholds['status'];dump(out/'result.json',result)
    return result
