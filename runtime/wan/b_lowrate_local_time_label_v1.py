"""User-only one-attempt physical runner. Imports model backend only after durable attempt."""
from pathlib import Path
from collections import Counter
import json,time
from main.tube_state import b_lowrate_local_time_label_plan_v1 as p
from runtime.wan.b_lfg_posthoc_reference_v1 import save,read,flush,existing,execution_artifacts
_PREP_ORIGINS=dict(zip(p.HOSTS,(48,49,50,51))) # fixture preparation only, never passed to core

def token(t):return '/'.join(map(str,t))
def raw_path(t):return 'raw/'+token(t)+'.json'
def blind_path(t):return 'blind/'+token(t)+'.json'
def pay_path(t):return 'payload/'+token(t)+'.json'
def fam_path(t):return 'families/'+token(t)+'.json'
def prefix(h,a,view='',phase=-1):return '/'.join((h,a,view,str(phase)))

def units():
    u={'PN':'PN_initialize','vae':'VAE_load','raw_readback':'saved_raw_readback','blind_readback':'saved_blind_readback','post':'truth_post_evaluation'}
    for t in p.tag_tables():
        u['raw/'+token(t)]='tag_correlations'
        if t[0]!='local':u['blind/'+token(t)]='tag_canonical_decode'
    for t in p.families():u['family/'+token(t)]='pilot_phase_and_RM_decode'
    for t in p.payload_tables():u['payload/'+token(t)]='payload_keyed_read'
    for h in p.HOSTS:
        for a in ('OFF','P0'):u[h+'/'+a+'/load']='native_input_load'
        u[h+'/TAG/construct']='tag_construct'
        u[h+'/OFF/media']='historical_OFF_media_load'
        for a in p.ARMS:
            pre=prefix(h,a,'native');u[pre+'/LLL']='received_Haar_LLL'
            u[h+'/'+a+'/crop']='actual_pixel_crop52'
            for phase in range(8):
                pre=prefix(h,a,'local',phase)
                for name,kind in [('prepare','actual_pixel_phase45'),('E','E45'),('LLL','received_Haar_LLL')]:u[pre+'/'+name]=kind
                if a in p.NEW_ARMS:u[pre+'/FFT']='payload_FFT'
        for a in p.NEW_ARMS:
            pre=h+'/'+a
            for name,kind in [('D','D181'),('quantize','RGB8_quantize'),('codec','MP4_encode'),('probe','MP4_probe'),('read','MP4_decode')]:u[pre+'/'+name]=kind
            pre=prefix(h,a,'mp4')
            for name,kind in [('prepare','RGB_input_prepare'),('E','E181'),('LLL','received_Haar_LLL'),('FFT','payload_FFT')]:u[pre+'/'+name]=kind
            u['clipping/'+h+'/'+a]='post_clipping'
        for layer in ('preclip','float_rgb','rgb8','readback'):u['quality/'+h+'/'+layer]='post_paired_quality'
    return u

class Journal:
    def __init__(self,out,backend):
        self.out=out;self.backend=backend;self.rows={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units().items()};self.snapshot()
    def snapshot(self):
        save(self.out/'ledger.json',self.rows);save(self.out/'progress.json',dict(planned=len(self.rows),counts=dict(Counter(r['status'] for r in self.rows.values())),active=next((k for k,r in self.rows.items() if r['status']=='STARTED'),None)))
    def call(self,name,fn,deps=()):
        r=self.rows[name]
        if r['status']!='NOT_ENTERED' or r['attempts']:raise RuntimeError('consumed operation '+name)
        missing=[d for d in deps if self.rows[d]['status']!='RETURNED']
        if missing:
            r.update(status='UNAVAILABLE_DEPENDENCY',unavailable_from=missing);save(self.out/'receipts'/(name+'.json'),r);self.snapshot();return None
        r.update(status='STARTED',attempts=1,started_unix=time.time());save(self.out/'receipts'/(name+'.json'),r);self.snapshot()
        before=dict(getattr(self.backend,'blocks',{}))
        try:
            value=fn();r['status']='RETURNED';return value
        except BaseException as e:
            r.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=type(e).__name__+': '+str(e))
            if not isinstance(e,Exception):raise
        finally:
            r['finished_unix']=time.time()
            if r['kind'] in ('D181','E181','E45'):
                after=getattr(self.backend,'blocks',{});r['internal_blocks']={k:after[k]-before[k] if type(after.get(k)) is int and type(before.get(k)) is int else None for k in ('encoder','decoder')}
            save(self.out/'receipts'/(name+'.json'),r);self.snapshot()
    def record(self,name,fn,path,deps=()):
        def operation():v=fn();save(self.out/path,v);return v
        return self.call(name,operation,deps)

def payload_empty(local):
    cells=[dict(local_row=i+1,observed_time=i+1,bits=[dict(slot=j,decoded=None) for j in range(8)]) for i in range(11 if local else 44)]
    return dict(status='TECHNICAL_UNRESOLVED',decoded_bits=None if local else [None]*32,local_cells=cells,signed_votes=None,zero_coefficient_mask=None,truth_inputs=False,message_inputs=False)

def placeholders(out):
    from main.tube_state import b_lowrate_local_time_label_v1 as c
    for t in p.tag_tables():
        r=c.unavailable(p.row_indices(t[0]));save(out/raw_path(t),r)
        if t[0]!='local':save(out/blind_path(t),c.canonical(r,p.KEYS[t[-1]]))
    for t in p.families():save(out/fam_path(t),c.bind_family({i:c.unavailable(range(1,5)) for i in range(8)},p.KEYS[t[-1]]))
    for t in p.payload_tables():save(out/pay_path(t),payload_empty(t[0]=='local'))
    for h in p.HOSTS:
        for layer in ('preclip','float_rgb','rgb8','readback'):save(out/'quality'/h/(layer+'.json'),dict(status='NOT_ENTERED',frame_denominator=181,adjacent_denominator=180,frames=[dict(frame=i,status='NOT_ENTERED') for i in range(181)],adjacent=[dict(frame_before=i,frame_after=i+1,status='NOT_ENTERED') for i in range(180)]))
        for a in p.NEW_ARMS:save(out/'quality'/h/(a+'_clipping.json'),dict(status='NOT_ENTERED',frame_denominator=181))

def readback(out,final=False):
    # An attempted product that failed to persist cannot be replaced by an old placeholder.
    ledger=read(out/'ledger.json')
    kinds={'tag_correlations','payload_keyed_read'}
    if final:kinds.update(('tag_canonical_decode','pilot_phase_and_RM_decode'))
    bad=[name for name,row in ledger.items() if row['kind'] in kinds and row['attempts'] and row['status']!='RETURNED']
    if bad:raise ValueError('attempted blind products did not persist: '+repr(bad))
    from main.tube_state import b_lowrate_local_time_label_v1 as c
    n=0
    for t in p.tag_tables():
        raw=c.validate_raw(read(out/raw_path(t)),p.row_indices(t[0]));n+=len(raw['rows'])
        if final and t[0]!='local':
            b=read(out/blind_path(t))
            if len(b['rows'])!=21:raise ValueError('canonical blind21 rows')
            if any(r['data']['raw_bits']!=[int(x>0) if x is not None and x!=0 else None for x in s['data_scores']] for r,s in zip(b['rows'],raw['rows'])):raise ValueError('blind different saved data bits')
    for t in p.payload_tables():
        b=read(out/pay_path(t));count=11 if t[0]=='local' else 44
        if len(b['local_cells'])!=count or any(len(x['bits'])!=8 for x in b['local_cells']):raise ValueError('fixed original payload geometry')
        if t[0]=='mp4' and len(b['decoded_bits'])!=32:raise ValueError('original32 groups')
        for field in ('signed_votes','zero_coefficient_mask'):
            v=b[field]
            if v is None:
                if b['status']!='TECHNICAL_UNRESOLVED':raise ValueError('returned payload has no votes')
            elif len(v)!=4 or any(len(x)!=count or any(len(y)!=240 for y in x) for x in v):raise ValueError('payload vote geometry')
    if final:
        for h,a,k in p.families():
            b=read(out/fam_path((h,a,k)))
            if len(b['rows'])!=4 or len(b['hypotheses'])!=16 or set(b['data_decodes'])!={str(i) for i in range(8)}:raise ValueError('fixed family geometry')
            for phase in range(8):
                raw=read(out/raw_path(('local',h,a,phase,k)))
                for r,s in zip(b['data_decodes'][str(phase)],raw['rows']):
                    if r['raw_bits']!=[int(x>0) if x is not None and x!=0 else None for x in s['data_scores']]:raise ValueError('family used different saved bits')
    if n!=1608:raise ValueError('1608 raw row slots')
    return dict(raw_rows=n,correlations=48*n,final=final,truth_opened=False)

def run(source=p.SOURCE,output=p.OUTPUT,*,backend=None):
    out=Path(output);source=Path(source)
    if execution_artifacts(out) or (out/'families').exists():return existing(out)
    out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as f:json.dump(dict(started_unix=time.time(),one_attempt=True,physical_reentry=False),f);flush(f)
    except FileExistsError:return existing(out)
    j=None;result=dict(status='STARTED',scientific_PASS=None,independent_confirmation=False,physical_reentry=False)
    try:
        j=Journal(out,backend);placeholders(out);save(out/'plan.json',dict(fixed=p.fixed(),unit_frames=8,valid_IDs=[1,21],alpha=.15,RM_radius=7,pilot_mismatches=0,local_phases=list(range(8)),local_rows=[1,2,3,4]))
        save(out/'private/preparation.json',dict(origins=_PREP_ORIGINS,seen_development_sources=True,receiver_access=False))
        if backend is None:
            from runtime.wan.b_lowrate_local_time_label_backend_v1 import Backend
            backend=Backend(p.MODEL);j.backend=backend
        j.record('PN',backend.initialize_PN,'PN/parameters.json');j.record('vae',backend.load_vae,'vae.json')
        def scores(z,view,h,a,phase,dep):
            pre=prefix(h,a,view,phase);low=j.call(pre+'/LLL',lambda:backend.tag_lll(z,view=='local'),[dep])
            for k,key in p.KEYS.items():
                t=(view,h,a,phase,k);j.record('raw/'+token(t),lambda key=key:backend.tag_scores(low,key,p.row_indices(view)),raw_path(t),[pre+'/LLL','PN'])
        def local_reads(received,h,a,dep):
            crop=j.call(h+'/'+a+'/crop',lambda:backend.crop_received(received,_PREP_ORIGINS[h]),[dep])
            if crop is not None:backend.save_tensor(out/'received'/h/(a+'.pt'),crop)
            for phase in range(8):
                pre=prefix(h,a,'local',phase);pixels=j.call(pre+'/prepare',lambda:backend.prepare_local(crop,phase),[h+'/'+a+'/crop'])
                def encode():v=backend.encode_E45(pixels);backend.save_tensor(out/'latents/local'/h/a/(str(phase)+'.pt'),v);return v
                z=j.call(pre+'/E',encode,[pre+'/prepare','vae']);scores(z,'local',h,a,phase,pre+'/E')
                if a in p.NEW_ARMS:
                    spectrum=j.call(pre+'/FFT',lambda:backend.local_payload_spectrum(z),[pre+'/E']);t=('local',h,a,phase,'key00')
                    j.record('payload/'+token(t),lambda:backend.local_payload_votes(spectrum,p.KEYS['key00']),pay_path(t),[pre+'/FFT']);del spectrum
                del z,pixels
        for h in p.HOSTS:
            for a in p.ARMS:
                pre=h+'/'+a
                if a=='TAG':
                    # P0 is loaded once, retained only until its TAG arm is constructed.
                    def construct():
                        z,r=backend.construct_tag(p0,p.KEYS['key00']);backend.save_tensor(out/'native'/h/'TAG.pt',z);save(out/'construction'/h/'TAG.json',r);return z
                    dep=pre+'/construct';z=j.call(dep,construct,[h+'/P0/load','PN'])
                else:
                    dep=pre+'/load';z=j.call(dep,lambda:backend.load_native(source/'hosts'/h/'private'/(a+'_terminal.pt')))
                    if a=='P0':p0=z
                scores(z,'native',h,a,-1,dep)
                if a=='OFF':
                    received_dep=pre+'/media';received=j.call(received_dep,lambda:backend.load_pixels(source/'hosts'/h/'media/OFF/readback.pt',181))
                else:
                    def decode():
                        values=backend.decode_PT(z)
                        for name,v in values.items():backend.save_tensor(out/'media'/h/a/(name+'.pt'),v)
                        return values
                    decoded=j.call(pre+'/D',decode,[dep,'vae'])
                    def quantize():v=backend.quantize(decoded['float_rgb']);backend.save_tensor(out/'media'/h/a/'rgb8.pt',v);return v
                    pixels=j.call(pre+'/quantize',quantize,[pre+'/D']);del decoded
                    path=out/'media'/h/a/'video.mp4'
                    j.record(pre+'/codec',lambda:backend.save_mp4(pixels,path),str(path.parent.relative_to(out)/'codec.json'),[pre+'/quantize'])
                    probe=j.record(pre+'/probe',lambda:backend.probe(path),str(path.parent.relative_to(out)/'probe.json'),[pre+'/codec'])
                    def read_media():v=backend.read_media(path,probe,181);backend.save_tensor(path.parent/'readback.pt',v);return v
                    received_dep=pre+'/read';received=j.call(received_dep,read_media,[pre+'/probe']);del pixels
                    full=prefix(h,a,'mp4');rgb=j.call(full+'/prepare',lambda:backend.prepare_full(received),[received_dep])
                    def encode_full():v=backend.encode_full(rgb);backend.save_tensor(out/'latents/mp4'/h/(a+'.pt'),v);return v
                    encoded=j.call(full+'/E',encode_full,[full+'/prepare','vae']);scores(encoded,'mp4',h,a,-1,full+'/E')
                    spectrum=j.call(full+'/FFT',lambda:backend.payload_spectrum(encoded),[full+'/E'])
                    for k,key in p.KEYS.items():
                        t=('mp4',h,a,-1,k);j.record('payload/'+token(t),lambda key=key:backend.payload_votes(spectrum,key),pay_path(t),[full+'/FFT'])
                    del encoded,rgb,spectrum
                local_reads(received,h,a,received_dep);del z,received
            del p0
        j.record('raw_readback',lambda:readback(out),'markers/raw_complete.json')
        from main.tube_state import b_lowrate_local_time_label_v1 as c
        for t in p.tag_tables():
            if t[0]!='local':j.record('blind/'+token(t),lambda t=t:c.canonical(read(out/raw_path(t)),p.KEYS[t[-1]]),blind_path(t),['raw_readback'])
        for t in p.families():
            h,a,k=t;j.record('family/'+token(t),lambda h=h,a=a,k=k:c.bind_family({i:read(out/raw_path(('local',h,a,i,k))) for i in range(8)},p.KEYS[k]),fam_path(t),['raw_readback'])
        j.record('blind_readback',lambda:readback(out,True),'markers/blind_complete.json',['raw_readback'])
        if j.rows['blind_readback']['status']=='RETURNED':
            save(out/'post_truth_access.json',dict(opened_unix=time.time(),all_blind_saved_and_read_back=True))
            from runtime.wan.b_lowrate_local_time_label_evaluation_v1 import evaluate
            post=j.record('post',lambda:evaluate(out,source/'protocol.json'),'evaluation.json',['blind_readback']);result['post_status']=post['status'] if post else 'UNRESOLVED'
            for h in p.HOSTS:
                for a in p.NEW_ARMS:j.record('clipping/'+h+'/'+a,lambda h=h,a=a:backend.preclip(backend.load_tensor(out/'media'/h/a/'preclip.pt')),f'quality/{h}/{a}_clipping.json',[h+'/'+a+'/D','blind_readback'])
                for layer in ('preclip','float_rgb','rgb8','readback'):
                    dep='D' if layer in ('preclip','float_rgb') else 'quantize' if layer=='rgb8' else 'read'
                    j.record('quality/'+h+'/'+layer,lambda h=h,layer=layer:backend.quality(backend.load_tensor(out/'media'/h/'P0'/(layer+'.pt')),backend.load_tensor(out/'media'/h/'TAG'/(layer+'.pt')),list(range(181)),'same-runtime P0 to TAG '+layer),f'quality/{h}/{layer}.json',[h+'/P0/'+dep,h+'/TAG/'+dep,'blind_readback'])
        result['status']='COMPLETED_DESCRIPTION_ONLY'
    except BaseException as e:result.update(status='FAILED' if isinstance(e,Exception) else 'INTERRUPTED',error=type(e).__name__+': '+str(e))
    finally:
        if backend is not None:
            try:backend.release_vae()
            except Exception as e:result['release_error']=str(e)
        if j is not None:
            j.snapshot();result.update(ledger_status=dict(Counter(r['status'] for r in j.rows.values())),kinds_started=dict(Counter(r['kind'] for r in j.rows.values() if r['attempts'])),kinds_returned=dict(Counter(r['kind'] for r in j.rows.values() if r['status']=='RETURNED')))
            if result['status']=='COMPLETED_DESCRIPTION_ONLY' and (any(r['status']!='RETURNED' for r in j.rows.values()) or result.get('post_status')!='POST_DESCRIPTION_ONLY'):result['status']='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
        result.update(fixed_denominators=p.fixed(),user_net_bits=32,new_user_capacity_claim=False,temporal_path_complete=False)
        save(out/'result.json',result)
    return result
