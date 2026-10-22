"""One consumed attempt, full-only fixed four-arm run. Stdlib import until attempt."""
from pathlib import Path
from collections import Counter
import json,time
from main.tube_state import b_block2_carrier_v1 as m
from runtime.wan.b_full_context_local_read_v1 import save,flush,reject,Journal as BaseJournal
SOURCE='/content/drive/MyDrive/Video-WM/B-Background-Normalized-Match-V1/fixed-four-unseen-v1'
OUTPUT='/content/drive/MyDrive/Video-WM/B-Block2-Carrier-V1/fixed-viewed-full-v1'
MODEL='Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
MEDIA_ARMS=('P0','RANDOM','BLOCK2')
QUALITY_LAYERS=('preclip','rgb8','readback')

def execution_artifacts(output):
    out=Path(output);found=[]
    for name in ('attempt.json','ledger.json','result.json','evaluation.json','plan.json','fixed_rows.json','progress.json','post_reference_access.json','tag_post_rows.json','tag_post_chips.json','carrier_full_pairs.json','payload_post_rows.json','payload_group_pairs.json','payload_cell_pairs.json'):
        for suffix in ('','.tmp'):
            p=out/(name+suffix)
            if p.exists() or p.is_symlink():found.append(name+suffix)
    for folder in ('raw','blind','payload','native','construction','latents','media','quality','markers','receipts','books'):
        p=out/folder
        if p.exists():found.extend(str(x.relative_to(out)) for x in p.rglob('*') if x.is_file() or x.is_symlink())
    return sorted(found)

def existing(output):
    out=Path(output);saved={};errors={};ledger={}
    for name in ('result.json','evaluation.json','ledger.json'):
        try:
            p=out/name
            if p.is_file():saved[name]=json.loads(p.read_text(),parse_constant=reject)
        except (OSError,ValueError) as e:errors[name]=str(e)
    ledger=saved.get('ledger.json',{})
    for p in (out/'receipts').rglob('*.json') if (out/'receipts').is_dir() else ():
        try:r=json.loads(p.read_text(),parse_constant=reject);ledger[r['slot']]=r
        except (OSError,ValueError,KeyError,TypeError) as e:errors[str(p)]=str(e)
    return dict(status='READ_ONLY_EXISTING_ATTEMPT',physical_reentry=False,saved=saved,read_errors=errors,ledger_counts=dict(Counter(r.get('status') for r in ledger.values())))

def slot(t):return '/'.join(t)
def tag_path(out,folder,t):return Path(out)/folder/(slot(t)+'.json')
def payload_path(out,h,a,k):return Path(out)/'payload'/h/a/(k+'.json')
def units():
    u={'books':'book_initialize_and_same_chip_correlation','vae':'VAE_load','raw_readback':'blind_raw_and_payload_readback','blind_readback':'blind_final_readback','post':'post_truth_evaluation'}
    for h in m.HOSTS:
        for a in ('OFF','P0'):u[f'{h}/{a}/load']='cached_native_load'
        u[h+'/OFF/received']='cached_OFF_readback_load'
        for a in ('RANDOM','BLOCK2'):u[f'{h}/{a}/construct']='metadata_construct'
        for a in m.ARMS:
            if a!='OFF':
                for s,k in (('D','D181'),('quantize','RGB8_quantize'),('save_mp4','MP4_encode'),('probe','MP4_probe'),('read_mp4','MP4_decode')):u[f'{h}/{a}/{s}']=k
                u[f'{h}/{a}/FFT']='payload_FFT'
                for key in m.KEYS:u[f'{h}/{a}/votes/{key}']='payload_keyed_read'
                u[f'clipping/{h}/{a}']='post_clipping'
            for s,k in (('prepare','RGB_input_prepare'),('E','E181')):u[f'{h}/{a}/{s}']=k
        for a in ('RANDOM','BLOCK2'):
            for layer in QUALITY_LAYERS:u[f'quality/{h}/{a}/{layer}']='post_paired_quality'
    for t in m.tables():u['raw/'+slot(t)]='tag_raw_44';u['final/'+slot(t)]='tag_frozen_tau_description'
    return u

class Journal(BaseJournal):
    def __init__(self,out,backend):
        self.out=out;self.backend=backend;self.rows={k:dict(slot=k,kind=v,status='NOT_ENTERED',attempts=0) for k,v in units().items()};self.snapshot()
    def call(self,slot,fn,deps=()):
        # Same durable STARTED behavior, including dependency failures and interrupts.
        before=dict(getattr(self.backend,'blocks',{}))
        try:return super().call(slot,fn,deps)
        finally:
            row=self.rows[slot]
            if row['kind'] in ('D181','E181') and row['attempts']:
                after=getattr(self.backend,'blocks',{})
                row['internal_blocks']={k:after.get(k)-before.get(k) if type(after.get(k)) is int and type(before.get(k)) is int else None for k in ('encoder','decoder')}
                save(self.out/'receipts'/(slot+'.json'),row);self.snapshot()

def unavailable_payload(reason):
    return dict(status='TECHNICAL_UNRESOLVED',error=reason,decoded_bits=[None]*32,
        local_cells=[dict(observed_time=t,bits=[dict(bit_index=8*((t-1)//11)+s,decoded=None) for s in range(8)]) for t in range(1,45)],
        signed_votes=None,zero_coefficient_mask=None,truth_inputs=False,message_inputs=False)

def placeholders(out):
    for t in m.tables():
        for folder in ('raw','blind'):save(tag_path(out,folder,t),m.unavailable('NOT_ENTERED'))
    for h in m.HOSTS:
        for a in MEDIA_ARMS:
            for k in m.KEYS:save(payload_path(out,h,a,k),unavailable_payload('NOT_ENTERED'))
        for a in ('RANDOM','BLOCK2'):
            for layer in QUALITY_LAYERS:save(out/'quality'/h/a/(layer+'.json'),dict(status='NOT_ENTERED',frame_denominator=181,adjacent_denominator=180,frames=[dict(frame=i,status='NOT_ENTERED') for i in range(181)],adjacent=[dict(frame_before=i,frame_after=i+1,status='NOT_ENTERED') for i in range(180)]))

def readback(out,final=False,journal=None):
    if journal is not None and not final:
        failed=[k for k,v in journal.rows.items() if (k.startswith('raw/') or v['kind']=='payload_keyed_read') and v['attempts'] and v['status']!='RETURNED']
        if failed:raise ValueError('attempted blind product not successfully saved: '+repr(failed))
    for t in m.tables():
        table=json.loads(tag_path(out,'blind' if final else 'raw',t).read_text(),parse_constant=reject)
        if len(table['rows'])!=44:raise ValueError('actual tag row count')
        for j,r in enumerate(table['rows'],1):
            d=r['decoder']
            if r['row']!=j or len(d['physical_bits'])!=48 or len(d['unmasked_bits'])!=32 or len(d['code_distances'])!=64:raise ValueError('actual tag raw geometry')
            if not final and r['projections'] is not None and len(r['projections'])!=48:raise ValueError('actual q geometry')
    for h in m.HOSTS:
        for a in MEDIA_ARMS:
            for k in m.KEYS:
                p=json.loads(payload_path(out,h,a,k).read_text(),parse_constant=reject)
                if len(p['decoded_bits'])!=32 or len(p['local_cells'])!=44 or any(len(x['bits'])!=8 for x in p['local_cells']):raise ValueError('actual full payload geometry')
                for field in ('signed_votes','zero_coefficient_mask'):
                    v=p[field]
                    if v is None:
                        if p.get('status')!='TECHNICAL_UNRESOLVED':raise ValueError('missing returned votes')
                    elif len(v)!=4 or any(len(x)!=44 or any(len(y)!=240 for y in x) for x in v):raise ValueError('actual signed/zero votes geometry')
    return dict(tag_rows=4224,tag_tables=96,payload_tables=24,private_truth_opened=False,final=final)

def fixed_rows(out):
    rows=[]
    for t in m.tables():
        try:table=json.loads(tag_path(out,'blind',t).read_text(),parse_constant=reject)
        except (OSError,ValueError):table={}
        for i in range(44):
            r=table.get('rows',[]);d=r[i]['decoder'] if i<len(r) else {}
            rows.append(dict(layer=t[0],host=t[1],arm=t[2],family=t[3],key=t[4],cell=i+1,status=d.get('status','NOT_ENTERED'),technical=d.get('technical_failure',True)))
    save(out/'fixed_rows.json',rows);return rows

def run(source=SOURCE,output=OUTPUT,*,backend=None):
    out=Path(output);source=Path(source)
    if execution_artifacts(out):return existing(out)
    out.mkdir(parents=True,exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as f:json.dump(dict(started_unix=time.time(),one_attempt=True,physical_reentry=False),f);flush(f)
    except FileExistsError:return existing(out)
    journal=None;result=dict(status='STARTED',scientific_PASS=None,independent_confirmation=False,authentication_claim=False,physical_reentry=False)
    try:
        journal=Journal(out,backend);placeholders(out)
        save(out/'plan.json',dict(hosts=m.HOSTS,arms=m.ARMS,layers=m.LAYERS,tag_rows=4224,positive_rows=704,null_rows=3520,D181=12,E181=16,payload_FFT=12,payload_keyed_reads=24,writer_extra_dots=33792,readout_dots=202752,old_tau_descriptive_only=True))
        journal.record('books',m.initialize_books,'books/same_chip_wrong_key.json')
        if backend is None:
            from runtime.wan.b_block2_carrier_backend_v1 import Block2Backend
            backend=Block2Backend(MODEL);journal.backend=backend
        journal.record('vae',backend.load_vae,'vae.json')
        for h in m.HOSTS:
            native={}
            for a in ('OFF','P0'):
                name=f'{h}/{a}/load'
                native[a]=journal.call(name,lambda a=a:backend.load_native(source/'hosts'/h/'private'/(a+'_terminal.pt')))
            for a in m.ARMS:
                prefix=f'{h}/{a}'
                native_dep=prefix+'/load'
                if a in ('RANDOM','BLOCK2'):
                    native_dep=prefix+'/construct';family=m.family_for(a)[0]
                    def construct(family=family,a=a):
                        z,receipt=backend.construct_variant(native['P0'],family,m.KEYS['key00'])
                        backend.save_tensor(out/'native'/h/(a+'.pt'),z);save(out/'construction'/h/(a+'.json'),receipt);return z
                    z=journal.call(native_dep,construct,[h+'/P0/load','books'])
                else:z=native[a]
                for family in m.family_for(a):
                    for key,secret in m.KEYS.items():
                        t=('native',h,a,family,key)
                        journal.record('raw/'+slot(t),lambda z=z,family=family,secret=secret:backend.read_variant(z,family,secret),str(tag_path('', 'raw',t)),[native_dep,'books'])
                if a=='OFF':
                    received_dep=prefix+'/received'
                    pixels=journal.call(received_dep,lambda:backend.load_received(source/'hosts'/h/'media/OFF/readback.pt'))
                else:
                    def decode():
                        v=backend.decode_PT(z)
                        for name,x in v.items():backend.save_tensor(out/'media'/h/a/(name+'.pt'),x)
                        return v
                    decoded=journal.call(prefix+'/D',decode,[native_dep,'vae'])
                    def quantize():
                        v=backend.quantize(decoded['float_rgb']);backend.save_tensor(out/'media'/h/a/'rgb8.pt',v);return v
                    pixels=journal.call(prefix+'/quantize',quantize,[prefix+'/D'])
                    path=out/'media'/h/a/'video.mp4'
                    journal.record(prefix+'/save_mp4',lambda:backend.save_mp4(pixels,path),str(path.parent.relative_to(out)/'codec.json'),[prefix+'/quantize'])
                    probe=journal.record(prefix+'/probe',lambda:backend.probe(path),str(path.parent.relative_to(out)/'probe.json'),[prefix+'/save_mp4'])
                    def read_mp4():
                        v=backend.read_media(path,probe,181);backend.save_tensor(path.parent/'readback.pt',v);return v
                    received_dep=prefix+'/read_mp4';pixels=journal.call(received_dep,read_mp4,[prefix+'/probe'])
                    del decoded
                rgb=journal.call(prefix+'/prepare',lambda:backend.prepare_full(pixels),[received_dep])
                def encode():
                    v=backend.encode_full(rgb);backend.save_tensor(out/'latents'/h/(a+'.pt'),v);return v
                encoded=journal.call(prefix+'/E',encode,[prefix+'/prepare','vae'])
                for family in m.family_for(a):
                    for key,secret in m.KEYS.items():
                        t=('full',h,a,family,key)
                        journal.record('raw/'+slot(t),lambda family=family,secret=secret:backend.read_variant(encoded,family,secret),str(tag_path('','raw',t)),[prefix+'/E','books'])
                if a!='OFF':
                    spectrum=journal.call(prefix+'/FFT',lambda:backend.payload_spectrum(encoded),[prefix+'/E'])
                    for key,secret in m.KEYS.items():
                        journal.record(prefix+'/votes/'+key,lambda secret=secret:backend.payload_votes(spectrum,secret),str(payload_path('',h,a,key)),[prefix+'/FFT'])
                    del spectrum
                del z,pixels,rgb,encoded
            del native
        journal.record('raw_readback',lambda:readback(out,journal=journal),'markers/raw_complete.json')
        for t in m.tables():
            journal.record('final/'+slot(t),lambda t=t:m.finalize(json.loads(tag_path(out,'raw',t).read_text(),parse_constant=reject)),str(tag_path('','blind',t)),['raw_readback'])
        journal.record('blind_readback',lambda:readback(out,True),'markers/blind_complete.json',['raw_readback']+['final/'+slot(t) for t in m.tables()])
        # Truth/protocol and quality are post-only, even if some physical inputs failed.
        if journal.rows['blind_readback']['status']=='RETURNED':
            save(out/'post_reference_access.json',dict(opened_unix=time.time(),protocol_file=str(source/'protocol.json'),after_all_tag_payload_final_readback=True))
            from runtime.wan.b_block2_carrier_evaluation_v1 import evaluate
            post=journal.record('post',lambda:evaluate(out,source/'protocol.json'),'evaluation.json',['blind_readback'])
            result['post_status']=post['status'] if post is not None else 'UNRESOLVED'
            for h in m.HOSTS:
                for a in MEDIA_ARMS:
                    journal.record(f'clipping/{h}/{a}',lambda h=h,a=a:backend.preclip(backend.load_tensor(out/'media'/h/a/'preclip.pt')),f'quality/{h}/{a}/clipping.json',[f'{h}/{a}/D','blind_readback'])
                for a in ('RANDOM','BLOCK2'):
                    for layer in QUALITY_LAYERS:
                        def quality(h=h,a=a,layer=layer):
                            return backend.quality(backend.load_tensor(out/'media'/h/'P0'/(layer+'.pt')),backend.load_tensor(out/'media'/h/a/(layer+'.pt')),list(range(181)),'same-runtime P0 to '+a+' '+layer)
                        needed='D' if layer=='preclip' else 'quantize' if layer=='rgb8' else 'read_mp4'
                        journal.record(f'quality/{h}/{a}/{layer}',quality,f'quality/{h}/{a}/{layer}.json',[f'{h}/P0/{needed}',f'{h}/{a}/{needed}','blind_readback'])
        result['status']='COMPLETED_DESCRIPTION_ONLY'
    except BaseException as error:
        result.update(status='FAILED' if isinstance(error,Exception) else 'INTERRUPTED',error=f'{type(error).__name__}: {error}')
    finally:
        if backend is not None:
            try:backend.release_vae()
            except Exception as e:result['release_error']=str(e)
        rows=fixed_rows(out)
        if journal is not None:
            journal.snapshot();result['ledger_counts']=dict(Counter(r['status'] for r in journal.rows.values()))
            result['kinds_started']=dict(Counter(r['kind'] for r in journal.rows.values() if r['attempts']))
            result['kinds_returned']=dict(Counter(r['kind'] for r in journal.rows.values() if r['status']=='RETURNED'))
            if result['status']=='COMPLETED_DESCRIPTION_ONLY' and (any(r['status']!='RETURNED' for r in journal.rows.values()) or any(r['technical'] for r in rows) or result.get('post_status')!='POST_DESCRIPTION_ONLY'):result['status']='INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
        result.update(tag_rows=4224,positive_rows=704,null_rows=3520,technical_tag_rows=sum(r['technical'] for r in rows),old_tau_descriptive_only=True,local_reset_unknown_phase_tested=False)
        save(out/'result.json',result)
    return result
