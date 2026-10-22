"""Card106 USER-only fixed seen-local channel task. Import is stdlib/pure core."""
import errno,json,os,time
from pathlib import Path
from collections import Counter
from main.tube_state import b_time_label_local_channel_v1 as method
ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'experiments/wan_state_clock/configs/b_time_label_local_channel_v1.json'
DEFAULT_SOURCE='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1'
DEFAULT_DELETION='/content/drive/MyDrive/Video-WM/B-Single-Internal-Deletion-V1/fixed-viewed-development-v1'
DEFAULT_OUTPUT='/content/drive/MyDrive/Video-WM/B-Time-Label-Local-Channel-V1/fixed-viewed-stage1-v1'
KEYS=method.KEYS;ARMS=method.ARMS

def flush(stream):
    stream.flush()
    try:os.fsync(stream.fileno())
    except OSError as error:
        if error.errno not in {errno.EINVAL,errno.ENOSYS,errno.ENOTSUP,errno.EOPNOTSUPP}:raise

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');flush(stream)
    tmp.replace(path)

def execution_artifacts(output):
    output=Path(output);found=[]
    for name in ('attempt.json','ledger.json','progress.json','result.json','thresholds.json','protocol.json','roles.json','plan.json','evaluation.json','post_truth_access.json'):
        for suffix in ('','.tmp'):
            if (output/(name+suffix)).exists() or (output/(name+suffix)).is_symlink():found.append(name+suffix)
    for name in ('private','native','raw','blind','payload','media','cached','receipts','markers','quality'):
        p=output/name
        if p.exists():found.extend(str(x.relative_to(output)) for x in p.rglob('*') if x.is_file() or x.is_symlink())
    return sorted(found)

def existing(output):
    saved={};errors={}
    for name in ('result.json','thresholds.json','evaluation.json'):
        p=Path(output)/name
        if p.is_file():
            try:
                v=json.loads(p.read_text())
                saved[name]=v if name!='evaluation.json' else {k:v.get(k) for k in ('status','summary','fixed_denominators','payload_summary')}
            except (OSError,ValueError) as e:errors[name]=str(e)
    ledger={}
    try:
        if (Path(output)/'ledger.json').is_file():ledger=json.loads((Path(output)/'ledger.json').read_text())
    except (OSError,ValueError) as e:errors['ledger.json']=str(e)
    for p in (Path(output)/'receipts').rglob('*.json') if (Path(output)/'receipts').is_dir() else ():
        try:
            row=json.loads(p.read_text());ledger[row['slot']]=row
        except (OSError,ValueError,KeyError,TypeError) as e:errors[str(p)]=str(e)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_reentry=False,saved=saved,read_errors=errors,
        ledger_counts=dict(Counter(r['status'] for r in ledger.values())))

def load_config(path=CONFIG):
    cfg=json.loads(Path(path).read_text())
    expected=[('b101-s01','calibration',1,74,59),('b101-s03','calibration',18,53,23),
              ('b101-s02','evaluation',2,74,58),('b101-s04','evaluation',29,57,25)]
    if [(h['id'],h['role'],h['a'],h['b'],h['d']) for h in cfg['private_roster']]!=expected:raise ValueError('adopted viewed roster/roles/edit geometry required')
    if cfg['clean_origins']!=list(method.CLEAN_ORIGINS):raise ValueError('five fixed clean views')
    return cfg

def E_source(h,arm,view,phase):
    r=method.reuse(arm,view,phase)
    return None if r is None else f"hosts/{h}/features/{arm}/r{r['crop']}/p{r['phase']}.pt"

def family_source(arm,reuse,key):
    index=(0 if arm=='OFF' else 12)+2*reuse['crop']+list(KEYS).index(key)
    return f'q{index:02}'

def units(cfg):
    u={'vae':'VAE_load','calibration':'full_pilot_null_calibration','post':'private_post'}
    for key in KEYS:u['book/'+key]='metadata_codebook'
    for host in cfg['private_roster']:
        h=host['id'];evaluation=host['role']=='evaluation'
        for arm in ('OFF','P0'):u[h+'/native/'+arm+'/load']='native_source_load'
        u[h+'/construct']='QIM_metadata_construct'
        for arm in ARMS:
            for key in KEYS:
                u[h+'/native/'+arm+'/'+key+'/raw']='native_tag_projection_decode'
                u[h+'/native/'+arm+'/'+key+'/final']='native_threshold_apply'
        for arm in ('OFF','P0'):
            for stream in ('full','cross'):u[h+'/'+arm+'/'+stream+'/load']='cached_RGB_load'
        for label,kind in [('D','ordinary_D181'),('quantize','RGB8_quantize')]:u[h+'/PT/'+label]=kind
        for stream in ('full','cross'):
            if stream=='cross':u[h+'/PT/cross/edit']='real_pixel_deletion'
            for label,kind in [('encode','MP4_encode'),('probe','MP4_probe'),('decode','MP4_decode')]:u[h+'/PT/'+stream+'/'+label]=kind
        for arm in ARMS:
            for view in method.VIEWS:
                u[h+'/'+arm+'/'+view+'/crop']='received45_formation'
                for p in range(4):
                    v=f'{h}/{arm}/{view}/p{p}';r=method.reuse(arm,view,p)
                    if r:u[v+'/E']='cached_normalized_E_load'
                    else:u[v+'/prepare']='E41_prepare';u[v+'/E']='E41'
                    for key in KEYS:
                        u[v+'/'+key+'/raw']='local_tag_projection_decode';u[v+'/'+key+'/final']='local_threshold_apply'
                    if evaluation:
                        if not r:u[v+'/FFT']='payload_FFT'
                        for key in KEYS:u[v+'/'+key+'/payload']='cached_payload_bits' if r else 'payload_votes'
        if evaluation:
            for q in (0,1,8,9,10,11,12,13,20,21,22,23):u[h+f'/family/q{q:02}']='cached_original_family_load'
        u[h+'/quality/preclip']='PT_preclip_record'
        for layer in ('preclip','rgb8','mp4'):u[h+'/quality/full/'+layer]='paired_quality'
        for layer in ('input_rgb8','edited_mp4'):u[h+'/quality/cross/'+layer]='paired_quality'
        for layer in ('preclip','rgb8','mp4'):u[h+'/quality/reference/full/'+layer]='cached_quality_baseline'
        for layer in ('input_rgb8','edited_mp4'):u[h+'/quality/reference/cross/'+layer]='cached_quality_baseline'
    return u

def plans(cfg):
    rows=[]
    for host in cfg['private_roster']:
        h=host['id']
        for arm in ARMS:
            for key in KEYS:rows.append(dict(host=h,role=host['role'],scope='native',arm=arm,key=key,rows=44,initial_status='NOT_ENTERED',code_slots_per_row=64))
            for view in method.VIEWS:
                for p in range(4):
                    for key in KEYS:rows.append(dict(host=h,role=host['role'],scope='local',arm=arm,view=view,phase=p,key=key,rows=10,initial_status='NOT_ENTERED',code_slots_per_row=64))
    return rows

class Journal:
    def __init__(self,out,cfg,backend):
        self.out,self.backend=Path(out),backend
        self.data={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units(cfg).items()};self.save()
    def save(self,progress_only=False):
        if not progress_only:dump(self.out/'ledger.json',self.data)
        c=Counter(x['status'] for x in self.data.values())
        dump(self.out/'progress.json',dict(planned=len(self.data),returned=c['RETURNED'],failed=c['FAILED'],
            active=next((k for k,v in self.data.items() if v['status']=='STARTED'),None)))
    def call(self,slot,fn,deps=()):
        row=self.data[slot]
        if row['status']!='NOT_ENTERED':raise RuntimeError('consumed slot: '+slot)
        bad=[d for d in deps if self.data[d]['status']!='RETURNED']
        if bad:
            row['unavailable_from']=bad;dump(self.out/'receipts'/(slot+'.json'),row);self.save(progress_only=True);return None
        row.update(status='STARTED',attempts=1,started_unix=time.time())
        dump(self.out/'receipts'/(slot+'.json'),row);self.save(progress_only=True)
        before=dict(getattr(self.backend,'blocks',{}))
        try:value=fn();row['status']='RETURNED';return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
            if not isinstance(error,Exception):raise
        finally:
            row['finished_unix']=time.time();row['seconds']=row['finished_unix']-row['started_unix']
            if row['kind'] in ('E41','ordinary_D181'):
                after=getattr(self.backend,'blocks',{})
                row['internal_blocks']={k:after.get(k)-before.get(k) if type(after.get(k)) is int and type(before.get(k)) is int else None for k in ('encoder','decoder')}
            dump(self.out/'receipts'/(slot+'.json'),row);self.save(progress_only=True)
    def record(self,slot,fn,rel,deps=()):
        def op():v=fn();dump(self.out/rel,v);return v
        return self.call(slot,op,deps)
    def tensor(self,slot,fn,rel,deps=()):
        def op():v=fn();self.backend.save_tensor(self.out/rel,v);return v
        return self.call(slot,op,deps)

def raw_rel(h,arm,view,p,key):
    return f'raw/{h}/{arm}/{view}/p{p}/{key}.json'

def save_raw(journal,slot,fn,rel,n,deps=()):
    table=journal.record(slot,fn,rel,deps)
    if table is None:
        table=method.unavailable_table(n,'UPSTREAM_FAILED_OR_NOT_ENTERED');dump(journal.out/rel,table)
    return table

def payload_shell(reason):
    return dict(status='TECHNICAL_UNRESOLVED',bits=[None]*80,planned_bits=80,error=reason)

def load_cached_family(path,phases):
    data=json.loads(Path(path).read_text());bits={}
    for p in phases:
        row=data['phase_local_bits'][str(p)]
        if len(row)!=80 or any(x is not None and (type(x) is not int or x not in (0,1)) for x in row):raise ValueError('cached original80bit table format')
        bits[str(p)]=row
    return dict(phase_local_bits=bits,source_file=str(path),old_scores_passed_to_receiver=False)

def collect(host,source,deletion,out,backend,journal):
    """Private actuator knows edit/crop; blind projection/decode receives only z/key."""
    h=host['id'];evaluation=host['role']=='evaluation';native={}
    for arm in ('OFF','P0'):
        native[arm]=journal.call(h+'/native/'+arm+'/load',lambda arm=arm:backend.load_native(Path(source)/'hosts'/h/'private'/(arm+'_terminal.pt')))
    def construct():
        z,receipt=backend.construct(native['P0'],KEYS['key00'])
        backend.save_tensor(out/'private'/h/'PT_terminal.pt',z)
        dump(out/'private'/h/'construction.json',receipt);return z
    native['PT']=journal.call(h+'/construct',construct,[h+'/native/P0/load'])
    for arm in ARMS:
        dep=h+'/construct' if arm=='PT' else h+'/native/'+arm+'/load'
        for key,secret in KEYS.items():
            rel=f'native/{h}/{arm}/{key}.raw.json'
            save_raw(journal,h+'/native/'+arm+'/'+key+'/raw',
                lambda arm=arm,secret=secret:method.raw_table(backend.project_native(native[arm],secret),secret,44),rel,44,[dep,'book/'+key])
    pixels={}
    for arm in ('OFF','P0'):
        pixels[arm,'full']=journal.call(h+'/'+arm+'/full/load',lambda arm=arm:backend.load_pixels(Path(source)/'hosts'/h/'media'/arm/'readback.pt',181))
        pixels[arm,'cross']=journal.call(h+'/'+arm+'/cross/load',lambda arm=arm:backend.load_pixels(Path(deletion)/'media'/h/arm/'received.pt',121))
    def decode():
        data=backend.decode_PT(native['PT'])
        backend.save_tensor(out/'media'/h/'PT/full/preclip.pt',data['preclip'])
        backend.save_tensor(out/'media'/h/'PT/full/float_rgb.pt',data['float_rgb']);return data
    decoded=journal.call(h+'/PT/D',decode,[h+'/construct','vae'])
    rgb=journal.tensor(h+'/PT/quantize',lambda:backend.quantize(decoded['float_rgb']),f'media/{h}/PT/full/rgb8.pt',[h+'/PT/D'])
    for stream,length in [('full',181),('cross',121)]:
        if stream=='cross':
            rgb=journal.tensor(h+'/PT/cross/edit',lambda:backend.edit(pixels['PT','full'],host),f'media/{h}/PT/cross/input_rgb8.pt',[h+'/PT/full/decode'])
        base=f'media/{h}/PT/{stream}';path=out/base/'video.mp4'
        dep=h+'/PT/quantize' if stream=='full' else h+'/PT/cross/edit'
        journal.record(h+'/PT/'+stream+'/encode',lambda:backend.save_mp4(rgb,path),base+'/codec.json',[dep])
        probe=journal.record(h+'/PT/'+stream+'/probe',lambda:backend.probe(path),base+'/probe.json',[h+'/PT/'+stream+'/encode'])
        pixels['PT',stream]=journal.tensor(h+'/PT/'+stream+'/decode',lambda:backend.read_media(path,probe,length),base+'/received.pt',[h+'/PT/'+stream+'/probe'])
    del decoded,rgb,native
    families={}
    if evaluation:
        for q in (0,1,8,9,10,11,12,13,20,21,22,23):
            phases=(0,1) if q in (8,9,20,21) else tuple(range(4))
            families[f'q{q:02}']=journal.record(h+f'/family/q{q:02}',
                lambda q=q,phases=phases:load_cached_family(Path(source)/'hosts'/h/'blind/families'/f'q{q:02}.json',phases),
                f'cached/{h}/q{q:02}.json')
    for arm in ARMS:
        for view in method.VIEWS:
            stream='cross' if view=='cross_center' else 'full'
            origin=38 if stream=='cross' else method.CLEAN_ORIGINS[int(view[-1])]
            dep=h+'/'+arm+'/'+stream+'/load' if arm!='PT' else h+'/PT/'+stream+'/decode'
            received=journal.call(h+'/'+arm+'/'+view+'/crop',lambda:backend.crop45(pixels[arm,stream],origin),[dep])
            for p in range(4):
                prefix=f'{h}/{arm}/{view}/p{p}';cached=method.reuse(arm,view,p)
                if cached:
                    z=journal.call(prefix+'/E',lambda:backend.load_E(Path(source)/E_source(h,arm,view,p)))
                else:
                    actual=journal.call(prefix+'/prepare',lambda:backend.prepare(received,p),[h+'/'+arm+'/'+view+'/crop'])
                    z=journal.tensor(prefix+'/E',lambda:backend.encode_local(actual),f'observations/{prefix}/E41.pt',[prefix+'/prepare','vae'])
                    del actual
                for key,secret in KEYS.items():
                    save_raw(journal,prefix+'/'+key+'/raw',
                        lambda secret=secret:method.raw_table(backend.project_local(z,secret),secret,10),
                        raw_rel(h,arm,view,p,key),10,[prefix+'/E','book/'+key])
                if evaluation:
                    if not cached:spec=journal.call(prefix+'/FFT',lambda:backend.spectrum(z),[prefix+'/E'])
                    for key,secret in KEYS.items():
                        rel=f'payload/{prefix}/{key}.json'
                        if cached:
                            q=family_source(arm,cached,key)
                            def oldbits(q=q,key=key):
                                table=families[q]
                                bits=table['phase_local_bits'][str(cached['phase'])]
                                return dict(status='CACHED_DECODED_ONLY',bits=bits,planned_bits=80,
                                    source_file=table['source_file'],source_phase=cached['phase'],
                                    signed_and_zero_statistics_reconstructed=False)
                            val=journal.record(prefix+'/'+key+'/payload',oldbits,rel,[h+'/family/'+q])
                        else:
                            def votes(secret=secret):
                                v=backend.votes(spec,secret)
                                v['bits']=[b['decoded'] for row in v['local_cells'] for b in row['bits']]
                                v['planned_bits']=80
                                return v
                            val=journal.record(prefix+'/'+key+'/payload',votes,rel,[prefix+'/FFT'])
                        if val is None:dump(out/rel,payload_shell('UPSTREAM_FAILED_OR_NOT_ENTERED'))
                    if not cached:del spec
                del z
            del received
    del pixels
    journal.save()


def read_raw_local(out,h):
    return {(arm,view,p,key):json.loads((out/raw_rel(h,arm,view,p,key)).read_text())
        for arm in ARMS for view in method.VIEWS for p in range(4) for key in KEYS}

def read_table(path,n):
    value=json.loads(Path(path).read_text())
    if value['expected_rows']!=n or len(value['rows'])!=n or [r['row'] for r in value['rows']]!=list(range(1,n+1)):
        raise ValueError('fixed row table shape')
    return value

def finalize_host(h,out,journal,tau):
    for arm in ARMS:
        for key in KEYS:
            rel=f'blind/native/{h}/{arm}/{key}.json';slot=h+'/native/'+arm+'/'+key+'/final'
            value=journal.record(slot,lambda:method.finalize(read_table(out/f'native/{h}/{arm}/{key}.raw.json',44),tau),rel)
            if value is None:dump(out/rel,method.finalize(method.unavailable_table(44,'RAW_SAVE_OR_THRESHOLD_APPLY_FAILED'),tau))
        for view in method.VIEWS:
            for p in range(4):
                for key in KEYS:
                    rel=f'blind/local/{h}/{arm}/{view}/p{p}/{key}.json';slot=f'{h}/{arm}/{view}/p{p}/{key}/final'
                    value=journal.record(slot,lambda:method.finalize(read_table(out/raw_rel(h,arm,view,p,key),10),tau),rel)
                    if value is None:dump(out/rel,method.finalize(method.unavailable_table(10,'RAW_SAVE_OR_THRESHOLD_APPLY_FAILED'),tau))

def blind_readback(out,cfg):
    paths=[]
    for host in cfg['private_roster']:
        h=host['id']
        for arm in ARMS:
            for key in KEYS:
                rel=f'blind/native/{h}/{arm}/{key}.json';read_table(out/rel,44);paths.append(rel)
            for view in method.VIEWS:
                for p in range(4):
                    for key in KEYS:
                        rel=f'blind/local/{h}/{arm}/{view}/p{p}/{key}.json';read_table(out/rel,10);paths.append(rel)
                        if host['role']=='evaluation':
                            payload=json.loads((out/f'payload/{h}/{arm}/{view}/p{p}/{key}.json').read_text())
                            if len(payload['bits'])!=80:raise ValueError('fixed80 payload rows')
    return paths

def post_quality(cfg,source,deletion,out,backend,journal):
    from runtime.wan.b_time_label_local_channel_evaluation_v1 import cross_map
    for host in cfg['private_roster']:
        h=host['id']
        journal.record(h+'/quality/preclip',lambda:backend.preclip(backend.load_tensor(out/f'media/{h}/PT/full/preclip.pt')),
            f'quality/{h}/PT_preclip.json')
        for layer,file in [('preclip','preclip.pt'),('rgb8','rgb8.pt'),('mp4','readback.pt')]:
            ptfile='received.pt' if layer=='mp4' else file
            def paired(file=file,ptfile=ptfile,layer=layer):
                return backend.quality(backend.load_tensor(Path(source)/f'hosts/{h}/media/P0/{file}'),
                    backend.load_tensor(out/f'media/{h}/PT/full/{ptfile}'),list(range(181)),'P0_to_PT/full/'+layer)
            journal.record(h+'/quality/full/'+layer,paired,f'quality/{h}/P0_to_PT/full/{layer}.json')
        for layer,file in [('input_rgb8','input_rgb8.pt'),('edited_mp4','received.pt')]:
            def paired(file=file,layer=layer):
                return backend.quality(backend.load_tensor(Path(deletion)/f'media/{h}/P0/{file}'),
                    backend.load_tensor(out/f'media/{h}/PT/cross/{file}'),cross_map(host),'P0_to_PT/cross/'+layer)
            journal.record(h+'/quality/cross/'+layer,paired,f'quality/{h}/P0_to_PT/cross/{layer}.json')
        for layer in ('preclip','rgb8','mp4'):
            journal.record(h+'/quality/reference/full/'+layer,lambda layer=layer:json.loads((Path(source)/f'hosts/{h}/quality/OFF_to_P0/{layer}.json').read_text()),f'quality/{h}/OFF_to_P0_reference/full/{layer}.json')
        for layer in ('input_rgb8','edited_mp4'):
            journal.record(h+'/quality/reference/cross/'+layer,lambda layer=layer:json.loads((Path(deletion)/f'quality/{h}/OFF_to_P0/{layer}.json').read_text()),f'quality/{h}/OFF_to_P0_reference/cross/{layer}.json')

def fixed_status(out,cfg):
    """All6816 units retained even when calibration prevents eval physical entry."""
    status=[]
    for plan in plans(cfg):
        h,arm,key=plan['host'],plan['arm'],plan['key'];n=plan['rows']
        if plan['scope']=='native':
            final=out/f'blind/native/{h}/{arm}/{key}.json';raw=out/f'native/{h}/{arm}/{key}.raw.json'
        else:
            view,p=plan['view'],plan['phase']
            final=out/f'blind/local/{h}/{arm}/{view}/p{p}/{key}.json';raw=out/raw_rel(h,arm,view,p,key)
        value=None;error=None
        try:
            if final.is_file():value=read_table(final,n)
            elif raw.is_file():value=read_table(raw,n)
        except (OSError,ValueError,KeyError,TypeError) as exc:error=str(exc)
        for i in range(n):
            d=value['rows'][i]['decoder'] if value is not None else None
            status.append(dict(plan,row=i+1,status=d['status'] if d else 'TECHNICAL_UNRESOLVED' if error else 'NOT_ENTERED',
                code_slots=64,tag_label=d.get('tag_label') if d else None,
                technical=bool(d and d.get('technical_failure')) or error is not None,error=error))
    dump(out/'unit_status.json',status)
    return status

def run(source=DEFAULT_SOURCE,deletion=DEFAULT_DELETION,output=DEFAULT_OUTPUT,config=CONFIG,*,backend=None):
    out=Path(output)
    if execution_artifacts(out):return existing(out)
    cfg=load_config(config);out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x',encoding='utf-8') as stream:
            json.dump(dict(card=106,started_unix=time.time(),one_attempt=True,viewed_development=True),stream);flush(stream)
    except FileExistsError:return existing(out)
    journal=None;thresholds=dict(status='CALIBRATION_UNRESOLVED',tau=None)
    result=dict(status='STARTED',physical_reentry=False,independent_confirmation=False,scientific_PASS=None)
    try:
        dump(out/'private/roster.json',cfg['private_roster']);dump(out/'protocol.json',method.protocol())
        dump(out/'roles.json',{h['id']:h['role'] for h in cfg['private_roster']});dump(out/'plan.json',plans(cfg))
        journal=Journal(out,cfg,backend)
        # No backend/torch/diffusers or real input read before exclusive attempt/plan.
        if backend is None:
            from runtime.wan.b_time_label_local_channel_backend_v1 import WanBackend
            backend=WanBackend(cfg['model_id']);journal.backend=backend
        for key,secret in KEYS.items():journal.call('book/'+key,lambda secret=secret:method.core.book(secret))
        journal.record('vae',backend.load_vae,'vae.json')
        for host in cfg['private_roster']:
            if host['role']=='calibration':collect(host,source,deletion,out,backend,journal)
        def calibrate():
            records={h['id']:method.null_records(read_raw_local(out,h['id'])) for h in cfg['private_roster'] if h['role']=='calibration'}
            value=method.core.calibrate_complete_nulls(records)
            value['frozen_unix']=time.time() if value['status']=='FROZEN' else None
            return value
        thresholds=journal.record('calibration',calibrate,'thresholds.json') or thresholds
        if thresholds['status']=='FROZEN':
            for host in cfg['private_roster']:
                if host['role']=='calibration':finalize_host(host['id'],out,journal,thresholds['tau'])
            for host in cfg['private_roster']:
                if host['role']=='evaluation':
                    collect(host,source,deletion,out,backend,journal)
                    finalize_host(host['id'],out,journal,thresholds['tau'])
            paths=blind_readback(out,cfg)
            dump(out/'markers/blind_complete.json',dict(saved_unix=time.time(),actual_tag_tables=len(paths),
                actual_tag_rows=6816,actual_payload_tables=288,semantic_truth_opened=False))
            def post():
                dump(out/'post_truth_access.json',dict(opened_unix=time.time(),actual_blind_tables_read_back=True))
                private=json.loads((out/'private/roster.json').read_text())
                bits=json.loads((Path(source)/'protocol.json').read_text())['config']['writer']['bits']
                if len(bits)!=32 or any(type(v) is not int or v not in (0,1) for v in bits):raise ValueError('actual source writer bits binary32')
                from runtime.wan.b_time_label_local_channel_evaluation_v1 import evaluate
                value=evaluate(out,dict(cfg,private_roster=private),bits);dump(out/'evaluation.json',value)
                post_quality(cfg,source,deletion,out,backend,journal)
                return dict(status=value['status'])
            journal.call('post',post)
        else:
            for k,row in journal.data.items():
                if row['status']=='NOT_ENTERED':row['unavailable_from']=['CALIBRATION_UNAVAILABLE']
        if not (out/'thresholds.json').exists():dump(out/'thresholds.json',thresholds)
        result['status']='FINISHED_SEEN_LOCAL_NO_SCIENTIFIC_PASS'
    except BaseException as error:
        result.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
    finally:
        if backend is not None:
            try:backend.release_vae()
            except Exception as error:result['release_error']=str(error)
        if journal is not None:
            journal.save()
            statuses=fixed_status(out,cfg)
            result.update(counts=dict(Counter(r['status'] for r in journal.data.values())),
                kinds_returned=dict(Counter(r['kind'] for r in journal.data.values() if r['status']=='RETURNED')),
                row_status_counts=dict(Counter(r['status'] for r in statuses)),technical_tag_rows=sum(r['technical'] for r in statuses),
                not_entered=[k for k,v in journal.data.items() if v['status']=='NOT_ENTERED'])
            if result['status']=='FINISHED_SEEN_LOCAL_NO_SCIENTIFIC_PASS' and (
                any(v['status']!='RETURNED' for v in journal.data.values()) or result['technical_tag_rows'] or any(r['status']=='NOT_ENTERED' for r in statuses)):
                result['status']='INCOMPLETE_FIXED_ROWS_RETAINED'
        result['threshold_status']=thresholds['status'];dump(out/'result.json',result)
    return result
