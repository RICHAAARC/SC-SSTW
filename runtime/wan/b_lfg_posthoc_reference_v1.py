"""Fixed one-shot stage1 LFG reference runner. No model/tensor import before attempt."""
from pathlib import Path
from collections import Counter
import json,os,time,math
from main.tube_state import b_lfg_posthoc_plan_v1 as p

def reject(x):raise ValueError('nonfinite JSON '+x)
def flush(f):
    f.flush()
    try:os.fsync(f.fileno())
    except OSError as e:
        if e.errno not in (22,38,95):raise

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=1);f.write('\n');flush(f)
    temporary.replace(path)

def read(path):return json.loads(Path(path).read_text(),parse_constant=reject)

def execution_artifacts(output):
    out=Path(output);found=[]
    for name in ('attempt.json','ledger.json','result.json','evaluation.json','plan.json','fixed_rows.json','progress.json','post_truth_access.json','message_post_rows.json','payload_group_pairs.json','payload_cell_pairs.json'):
        for suffix in ('','.tmp'):
            path=out/(name+suffix)
            if path.exists() or path.is_symlink():found.append(name+suffix)
    for name in ('raw','blind','payload','private','PN','construction','native','latents','media','quality','markers','receipts'):
        root=out/name
        if root.exists():found.extend(str(x.relative_to(out)) for x in root.rglob('*') if x.is_file() or x.is_symlink())
    return sorted(found)

def existing(output):
    out=Path(output);saved={};errors={};ledger={}
    for name in ('result.json','evaluation.json','ledger.json'):
        try:
            if (out/name).is_file():saved[name]=read(out/name)
        except (OSError,ValueError) as e:errors[name]=str(e)
    ledger=saved.get('ledger.json',{})
    for path in (out/'receipts').rglob('*.json') if (out/'receipts').is_dir() else ():
        try:r=read(path);ledger[r['slot']]=r
        except (OSError,ValueError,KeyError,TypeError) as e:errors[str(path)]=str(e)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_reentry=False,saved=saved,read_errors=errors,ledger_status=dict(Counter(x['status'] for x in ledger.values())))

def slot(t):return '/'.join(t)
def message_path(out,folder,t):return Path(out)/folder/(slot(t)+'.json')
def payload_path(out,v,h,a,k):return Path(out)/'payload'/v/h/a/(k+'.json')

def units():
    u={'PN':'PN_initialize','fixture':'BCH_encode_writer_fixture','vae':'VAE_load','raw_readback':'raw_payload_JSON_readback','blind_readback':'BCH_message_JSON_readback','post':'truth_post_evaluation'}
    for h in p.HOSTS:
        u[h+'/load']='cached_pre_tag_native_load';u[h+'/construct']='LLL_construct'
        for a in p.ARMS:
            pre=h+'/'+a;u[pre+'/native/LLL']='received_Haar_LLL'
            for step,kind in (('D','D181'),('quantize','RGB8_quantize'),('save_mp4','MP4_encode'),('probe','MP4_probe'),('read_mp4','MP4_decode')):u[pre+'/'+step]=kind
            u['clipping/'+pre]='post_clipping'
            for v in p.MEDIA_VIEWS:
                for step,kind in (('prepare','RGB_input_prepare'),('E','E181'),('LLL','received_Haar_LLL'),('FFT','payload_FFT')):u[pre+'/'+v+'/'+step]=kind
                for key in p.KEYS:u[pre+'/'+v+'/payload/'+key]='payload_keyed_read'
        for layer in ('preclip','float_rgb','rgb8','readback'):u['quality/'+h+'/'+layer]='post_paired_quality'
    for t in p.tables():u['raw/'+slot(t)]='message_correlations';u['BCH/'+slot(t)]='BCH_decode'
    return u

class Journal:
    def __init__(self,out,backend):
        self.out=out;self.backend=backend;self.rows={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units().items()};self.snapshot()
    def snapshot(self):
        save(self.out/'ledger.json',self.rows);save(self.out/'progress.json',dict(planned=len(self.rows),counts=dict(Counter(r['status'] for r in self.rows.values())),active=next((k for k,r in self.rows.items() if r['status']=='STARTED'),None)))
    def call(self,name,fn,deps=()):
        row=self.rows[name]
        if row['status']!='NOT_ENTERED' or row['attempts']:raise RuntimeError('consumed operation '+name)
        missing=[x for x in deps if self.rows[x]['status']!='RETURNED']
        if missing:
            row['unavailable_from']=missing;save(self.out/'receipts'/(name+'.json'),row);self.snapshot();return None
        row.update(status='STARTED',attempts=1,started_unix=time.time());save(self.out/'receipts'/(name+'.json'),row);self.snapshot()
        before=dict(getattr(self.backend,'blocks',{}))
        try:
            value=fn();row['status']='RETURNED';return value
        except BaseException as e:
            row.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
            if not isinstance(e,Exception):raise
        finally:
            row['finished_unix']=time.time();row['seconds']=row['finished_unix']-row['started_unix']
            if row['kind'] in ('D181','E181'):
                after=getattr(self.backend,'blocks',{});row['internal_blocks']={k:after[k]-before[k] if type(after.get(k)) is int and type(before.get(k)) is int else None for k in ('encoder','decoder')}
            save(self.out/'receipts'/(name+'.json'),row);self.snapshot()
    def record(self,name,fn,path,deps=()):
        def op():v=fn();save(self.out/path,v);return v
        return self.call(name,op,deps)

def raw_unavailable(reason):return dict(status='TECHNICAL_UNRESOLVED',error=reason,scores=[None]*512,raw_codeword_bits=[None]*512,N=512,truth_inputs=False,message_inputs=False)
def BCH_unavailable(reason):return dict(status='TECHNICAL_UNRESOLVED',error=reason,BCH_success=None,raw_codeword_bits=[None]*512,information_bits=[None]*432,corrected_codeword_bits=None,correction_count=None,authentication=False)
def payload_unavailable(reason):
    return dict(status='TECHNICAL_UNRESOLVED',error=reason,decoded_bits=[None]*32,local_cells=[dict(bits=[dict(decoded=None,bit_index=8*(t//11)+j) for j in range(8)]) for t in range(44)],signed_votes=None,zero_coefficient_mask=None,truth_inputs=False,message_inputs=False)

def placeholders(out):
    for t in p.tables():save(message_path(out,'raw',t),raw_unavailable('NOT_ENTERED'));save(message_path(out,'blind',t),BCH_unavailable('NOT_ENTERED'))
    for v in p.MEDIA_VIEWS:
        for h in p.HOSTS:
            for a in p.ARMS:
                for k in p.KEYS:save(payload_path(out,v,h,a,k),payload_unavailable('NOT_ENTERED'))
    for h in p.HOSTS:
        for layer in ('preclip','float_rgb','rgb8','readback'):save(out/'quality'/h/(layer+'.json'),dict(status='NOT_ENTERED',frame_denominator=181,adjacent_denominator=180,frames=[dict(frame=i,status='NOT_ENTERED') for i in range(181)],adjacent=[dict(frame_before=i,frame_after=i+1,status='NOT_ENTERED') for i in range(180)]))

def readback(out,journal,final=False):
    attempted_bad=[k for k,v in journal.rows.items() if v['kind'] in (('message_correlations','payload_keyed_read','BCH_decode') if final else ('message_correlations','payload_keyed_read')) and v['attempts'] and v['status']!='RETURNED']
    if attempted_bad:raise ValueError('attempted blind products not saved: '+repr(attempted_bad))
    for t in p.tables():
        r=read(message_path(out,'raw',t))
        if len(r['scores'])!=512 or len(r['raw_codeword_bits'])!=512:raise ValueError('512-score/codeword raw geometry')
        if r['status']!='TECHNICAL_UNRESOLVED' and (not all(type(x) in (float,int) and math.isfinite(x) for x in r['scores']) or any(type(x) is not int or x not in (0,1) for x in r['raw_codeword_bits'])):raise ValueError('finite scores and binary coded bits required')
        if final:
            b=read(message_path(out,'blind',t))
            if len(b['raw_codeword_bits'])!=512 or len(b['information_bits'])!=432:raise ValueError('512/432 BCH/message geometry')
            if b['status']!='TECHNICAL_UNRESOLVED' and b['raw_codeword_bits']!=r['raw_codeword_bits']:raise ValueError('BCH received different saved hard bits')
    for v in p.MEDIA_VIEWS:
        for h in p.HOSTS:
            for a in p.ARMS:
                for k in p.KEYS:
                    b=read(payload_path(out,v,h,a,k))
                    if len(b['decoded_bits'])!=32 or len(b['local_cells'])!=44 or any(len(x['bits'])!=8 for x in b['local_cells']):raise ValueError('existing payload geometry')
                    for field in ('signed_votes','zero_coefficient_mask'):
                        val=b[field]
                        if val is None:
                            if b['status']!='TECHNICAL_UNRESOLVED':raise ValueError('missing returned full payload votes')
                        elif len(val)!=4 or any(len(x)!=44 or any(len(y)!=240 for y in x) for x in val):raise ValueError('full vote geometry')
    return dict(message_tables=48,coded_bits=24576,information_slots=20736,payload_tables=32,final=final,truth_opened=False)

def fixed_status(out):
    rows=[]
    for t in p.tables():
        try:r=read(message_path(out,'raw',t));b=read(message_path(out,'blind',t))
        except (OSError,ValueError):r={};b={}
        rows.append(dict(view=t[0],host=t[1],arm=t[2],key=t[3],score_status=r.get('status','NOT_ENTERED'),BCH_status=b.get('status','NOT_ENTERED'),BCH_success=b.get('BCH_success'),raw_coded_denominator=512,information_denominator=432))
    save(out/'fixed_rows.json',rows);return rows

def run(source=p.SOURCE,output=p.OUTPUT,*,backend=None):
    out=Path(output);source=Path(source)
    if execution_artifacts(out):return existing(out)
    out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as f:json.dump(dict(started_unix=time.time(),one_attempt=True,physical_reentry=False),f);flush(f)
    except FileExistsError:return existing(out)
    journal=None;result=dict(status='STARTED',physical_reentry=False,scientific_PASS=None,author_equivalence=False,capacity_claim=False)
    try:
        journal=Journal(out,backend);placeholders(out);save(out/'plan.json',dict(fixed=p.fixed(),alpha=p.ALPHA,K=0,N=p.N,M=p.M,views=p.VIEWS,hosts=p.HOSTS,arms=p.ARMS,CRF=p.CRF))
        from runtime.wan.b_lfg_posthoc_fixture_v1 import prepare
        fixture=journal.record('fixture',prepare,'private/writer_reference.json')
        if backend is None:
            from runtime.wan.b_lfg_posthoc_backend_v1 import LFGBackend
            backend=LFGBackend(p.MODEL);journal.backend=backend
        journal.record('PN',backend.initialize_PN,'PN/parameters.json')
        journal.record('vae',backend.load_vae,'vae.json')
        def scores(z,view,h,arm,dep):
            prefix=h+'/'+arm+'/'+view;low=journal.call(prefix+'/LLL',lambda:backend.message_lll(z),[dep])
            for key,secret in p.KEYS.items():
                t=(view,h,arm,key);journal.record('raw/'+slot(t),lambda secret=secret:backend.message_scores(low,secret),str(message_path('','raw',t)),[prefix+'/LLL','PN'])
        for h in p.HOSTS:
            original=journal.call(h+'/load',lambda:backend.load_native(source/'hosts'/h/'private/P0_terminal.pt'))
            for arm in p.ARMS:
                prefix=h+'/'+arm;dep=h+'/load';z=original
                if arm=='MSG':
                    dep=h+'/construct'
                    def construct():
                        z,receipt=backend.construct_message(original,fixture['codeword_bits'],p.KEYS['key00']);backend.save_tensor(out/'native'/h/'MSG.pt',z);save(out/'construction'/h/'MSG.json',receipt);return z
                    z=journal.call(dep,construct,[h+'/load','fixture','PN'])
                scores(z,'native',h,arm,dep)
                def decode():
                    values=backend.decode_PT(z)
                    for name,v in values.items():backend.save_tensor(out/'media'/h/arm/(name+'.pt'),v)
                    return values
                decoded=journal.call(prefix+'/D',decode,[dep,'vae'])
                def quantize():
                    v=backend.quantize(decoded['float_rgb']);backend.save_tensor(out/'media'/h/arm/'rgb8.pt',v);return v
                pixels=journal.call(prefix+'/quantize',quantize,[prefix+'/D']);del decoded
                for view in p.MEDIA_VIEWS:
                    received=pixels;received_dep=prefix+'/quantize'
                    if view=='mp4':
                        path=out/'media'/h/arm/'video.mp4'
                        journal.record(prefix+'/save_mp4',lambda:backend.save_mp4(pixels,path),str(path.parent.relative_to(out)/'codec.json'),[prefix+'/quantize'])
                        probe=journal.record(prefix+'/probe',lambda:backend.probe(path),str(path.parent.relative_to(out)/'probe.json'),[prefix+'/save_mp4'])
                        def read_mp4():
                            v=backend.read_media(path,probe,181);backend.save_tensor(path.parent/'readback.pt',v);return v
                        received_dep=prefix+'/read_mp4';received=journal.call(received_dep,read_mp4,[prefix+'/probe'])
                    viewpre=prefix+'/'+view;rgb=journal.call(viewpre+'/prepare',lambda:backend.prepare_full(received),[received_dep])
                    def encode():
                        v=backend.encode_full(rgb);backend.save_tensor(out/'latents'/view/h/(arm+'.pt'),v);return v
                    encoded=journal.call(viewpre+'/E',encode,[viewpre+'/prepare','vae'])
                    scores(encoded,view,h,arm,viewpre+'/E')
                    spectrum=journal.call(viewpre+'/FFT',lambda:backend.payload_spectrum(encoded),[viewpre+'/E'])
                    for key,secret in p.KEYS.items():journal.record(viewpre+'/payload/'+key,lambda secret=secret:backend.payload_votes(spectrum,secret),str(payload_path('',view,h,arm,key)),[viewpre+'/FFT'])
                    del rgb,encoded,spectrum,received
                del z,pixels
            del original
        journal.record('raw_readback',lambda:readback(out,journal),'markers/raw_complete.json')
        from main.tube_state import b_lfg_bch_reference_v1 as bch
        for t in p.tables():
            journal.record('BCH/'+slot(t),lambda t=t:bch.decode(read(message_path(out,'raw',t))['raw_codeword_bits']),str(message_path('','blind',t)),['raw_readback','raw/'+slot(t)])
        journal.record('blind_readback',lambda:readback(out,journal,True),'markers/blind_complete.json',['raw_readback'])
        if journal.rows['blind_readback']['status']=='RETURNED':
            save(out/'post_truth_access.json',dict(opened_unix=time.time(),all_blind_scores_BCH_messages_payload_readback=True))
            from runtime.wan.b_lfg_posthoc_evaluation_v1 import evaluate
            post=journal.record('post',lambda:evaluate(out,source/'protocol.json'),'evaluation.json',['blind_readback']);result['post_status']=post['status'] if post else 'UNRESOLVED'
            for h in p.HOSTS:
                for arm in p.ARMS:journal.record('clipping/'+h+'/'+arm,lambda h=h,arm=arm:backend.preclip(backend.load_tensor(out/'media'/h/arm/'preclip.pt')),f'quality/{h}/{arm}_clipping.json',[h+'/'+arm+'/D','blind_readback'])
                for layer in ('preclip','float_rgb','rgb8','readback'):
                    needed='D' if layer in ('preclip','float_rgb') else 'quantize' if layer=='rgb8' else 'read_mp4'
                    journal.record('quality/'+h+'/'+layer,lambda h=h,layer=layer:backend.quality(backend.load_tensor(out/'media'/h/'P0'/(layer+'.pt')),backend.load_tensor(out/'media'/h/'MSG'/(layer+'.pt')),list(range(181)),'same-runtime P0 to LFG MSG '+layer),f'quality/{h}/{layer}.json',[h+'/P0/'+needed,h+'/MSG/'+needed,'blind_readback'])
        result['status']='COMPLETED_DESCRIPTION_ONLY'
    except BaseException as e:result.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=f'{type(e).__name__}: {e}')
    finally:
        if backend is not None:
            try:backend.release_vae()
            except Exception as e:result['release_error']=str(e)
        fixed=fixed_status(out)
        if journal is not None:
            journal.snapshot();result['ledger_status']=dict(Counter(r['status'] for r in journal.rows.values()));result['kinds_started']=dict(Counter(r['kind'] for r in journal.rows.values() if r['attempts']));result['kinds_returned']=dict(Counter(r['kind'] for r in journal.rows.values() if r['status']=='RETURNED'))
            if result['status']=='COMPLETED_DESCRIPTION_ONLY' and (any(r['status']!='RETURNED' for r in journal.rows.values()) or result.get('post_status')!='POST_DESCRIPTION_ONLY'):result['status']='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
        result.update(fixed_denominators=p.fixed(),technical_score_tables=sum(r['score_status']=='TECHNICAL_UNRESOLVED' for r in fixed),BCH_failures=sum(r['BCH_success'] is False for r in fixed),user_net_payload_bits=32,new_user_capacity_claim=False,local_or_sync_claim=False)
        save(out/'result.json',result)
    return result
