"""Card85 fixed user-run local45 E4/FFT4; no new decode/codec/writer."""
from pathlib import Path
import json,time,resource
import torch
from main.tube_state import b_real_local_wan_v1 as method
from runtime.wan import c73_identity_v1 as shared
from runtime.wan.b_legacy_migration_v1 import WanBackend as PreviousBackend
from runtime.wan.b_real_local_wan_evaluation_v1 import evaluate

dump=shared.dump
DEFAULT_MODEL_ID=shared.DEFAULT_MODEL_ID
# Preparation sidecar only. Received observation/matcher/payload functions never accept this table.
PREPARATIONS=(('c00','P0',0,60,105),('c01','P0',1,61,106),('c02','PM',0,60,105),('c03','PM',1,61,106))


def fixed_denominator():
    return dict(cached_RGB_loads=2,clips=4,frames_per_clip=45,VAE_load=1,D=0,E=4,nominal_encoder_chunks=48,decoder_chunks=0,
        MP4_decode=0,codec=0,quantization=0,DiT=0,writer=0,gradients=0,keys=2,books=2,Q_tables=8,ROI_dots=15488,
        pooled_Q=3872,new_sequence_scores=272,new_sequence_terms=2992,reference_queries=4,reference_scores=136,
        reference_terms=1496,total_sequence_scores=408,total_sequence_terms=4488,FFT=4,vote_reads=8,conditional_reducers=8,
        signed_vote_slots=84480,zero_mask_slots=84480,raw_local_bits=704,message_bits=256,source_cell_rows=352,
        observed_cell_rows=88,single_source_rows=44,mixed_identity_rows=44,local_single_cell_bit_references=352,
        local_group_only_bit_references=320,mixed_group_undefined_bits=32,present_message_bits=128,missing_message_bits=128,
        old_raw_bit_reference_slots=352,new_reference_FFT=0,new_reference_E=0,new_reference_vote_reducers=0,ledger_units=len(units()))


def protocol():
    return dict(adopted_card=85,scope='fixed local readability diagnostic, not phase recovery/identity authentication',
        L=11,source_starts=list(range(1,35)),new_writer=False,gamma=1.0,net_bits=32,
        normalization='posterior.mode, existing(raw-mean)/std, FP32; adapter cache reset before/after every E',
        first_frame='all45 enter E; only local latent0 excluded from Q and votes',local_last_latent11_read=True,
        phase_labels_post_only=True,no_oracle_rephase=True,no_cross_clip_collision=True,
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',trusted_full_messages=None,FPR=None,
        resource_hard_caps=None,fixed_denominator=fixed_denominator(),scientific_pass=None)


def units():
    u={'model':'vae_load','truth':'post_numeric_truth_read','evaluation':'post_numeric_evaluation'}
    for arm in ('P0','PM'):u['load/'+arm]='cached_RGB_load'
    for key in method.KEY_IDS:u['book/'+key]='public_position_book'
    for case,arm,variant,a,b in PREPARATIONS:
        u['prepare/'+case]='received_RGB_preparation';u['E/'+case]='posterior_encode';u['FFT/'+case]='shared_payload_fft'
        for key in method.KEY_IDS:
            for kind,name in [('Q','direction_observation'),('match','sequence_score'),('map','conditional_mapping'),('votes','keyed_vote_read'),('payload','conditional_vote_reduction')]:u[kind+'/'+case+'/'+key]=name
    for arm in ('P0','PM'):
        for key in method.KEY_IDS:
            u['reference_Q/'+arm+'/'+key]='post_only_saved_Q_load'
            u['reference_match/'+arm+'/'+key]='post_only_sequence_score'
            u['reference_bits/'+arm+'/'+key]='post_only_saved_local_bit_read'
    return u


class Journal(shared.Journal):
    def __init__(self,output,backend):
        self.output,self.backend=output,backend;self.data={k:dict(kind=v,status='NOT_ENTERED') for k,v in units().items()};self.save()


class WanBackend(PreviousBackend):
    def encode_received(self,pixels):
        from runtime.wan.vae import reencode_rgb24_readback
        if tuple(pixels.shape)!=(45,320,512,3) or pixels.dtype!=torch.uint8:raise ValueError('actual45-frame RGB uint8 clip required')
        z=reencode_rgb24_readback(self.vae,pixels.float()/255).detach().cpu().float()
        method.validate_local(z);return z


def prepare_received(pixels,start,stop):
    shared.shared.validate_rgb(pixels,pixels=True)
    if stop-start!=45 or start<0 or stop>len(pixels):raise ValueError('actual fixed45 frames required; no padding')
    return pixels[start:stop].clone()


def execution_artifacts(output):
    output=Path(output);names=('attempt.json','protocol.json','ledger.json','result.json','checkpoint.json','numeric_completion.json','evaluation.json')
    found=[n+s for n in names for s in ('','.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ('blind','features','preparation','references'):
        root=output/folder
        if root.is_dir():found.extend(str(p.relative_to(output)) for p in root.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def run(source_root,output,truth_path,model_id=DEFAULT_MODEL_ID,*,backend=None):
    source_root=Path(source_root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    def existing(found):
        value=shared.existing_attempt(output,found);value['fixed_denominator']=fixed_denominator();return value
    found=execution_artifacts(output)
    if found:return existing(found)
    try:
        with (output/'attempt.json').open('x') as f:
            json.dump(dict(started_unix=time.time(),scope='Card85 one fixed user attempt; failure consumes it',source_root=str(source_root)),f);shared.flush_file(f)
    except FileExistsError:return existing(['attempt.json'])
    backend=backend or WanBackend(model_id);ledger=Journal(output,backend);started=time.perf_counter()
    names=[c+'/'+k for c,_,_,_,_ in PREPARATIONS for k in method.KEY_IDS]
    obs={n:method.unavailable_observation('NOT_ENTERED') for n in names}
    matches={n:method.blank_match() for n in names};maps={n:method.unavailable_map('NOT_ENTERED') for n in names}
    raw={n:method.unavailable_votes('NOT_ENTERED') for n in names};payload={n:method.unavailable_payload('NOT_ENTERED') for n in names}
    refs={a+'/'+k:None for a in ('P0','PM') for k in method.KEY_IDS}
    refmatches={n:method.blank_match() for n in refs};refbits={n:None for n in refs}
    metadata={c:dict(arm=a,variant=v,start=s,stop=e) for c,a,v,s,e in PREPARATIONS}
    result=dict(status='RUNNING',fixed_denominator=fixed_denominator(),protocol=protocol(),scientific_pass=None)
    books={};pixels={};deps={}
    def save_tensor(path,value):
        p=output/path;p.parent.mkdir(parents=True,exist_ok=True);torch.save(value,p);return value
    def checkpoint():
        result['counts']=ledger.counts();dump(output/'checkpoint.json',result)
    def call(slot,fn,dependencies=()):
        deps[slot]=list(dependencies)
        if any(ledger.data[d]['status']!='RETURNED' for d in dependencies):return None
        return ledger.call(slot,fn)
    def record(slot,fn,path,dependencies=()):
        def execute():
            value=fn();dump(output/path,value);return value
        return call(slot,execute,dependencies)
    def reason(slot):
        def visit(s,seen):
            if s in seen:return None
            seen.add(s)
            if ledger.data[s]['status'] in ('FAILED','INTERRUPTED','STARTED'):return s
            for d in deps.get(s,[]):
                found=visit(d,seen)
                if found:return found
            return None
        failed=visit(slot,set())
        return dict(status=('UPSTREAM_' if failed!=slot else '')+ledger.data[failed]['status'],unavailable_from=failed) if failed else dict(status='NOT_ENTERED',attempt_interruption=result.get('fatal_error'))
    def save_blind():
        for kind,values in [('Q',obs),('match',matches),('map',maps),('votes',raw),('payload',payload)]:
            for name,value in values.items():dump(output/'blind'/kind/(name+'.json'),value)
    for case,arm,_,_,_ in PREPARATIONS:
        deps['prepare/'+case]=['load/'+arm];deps['E/'+case]=['prepare/'+case,'model'];deps['FFT/'+case]=['E/'+case]
        for key in method.KEY_IDS:
            n=case+'/'+key
            deps['Q/'+n]=['E/'+case,'book/'+key];deps['match/'+n]=['Q/'+n]
            deps['map/'+n]=['match/'+n]
            deps['votes/'+n]=['FFT/'+case];deps['payload/'+n]=['votes/'+n,'map/'+n]
    dump(output/'protocol.json',protocol());dump(output/'preparation/sidecar.json',metadata);save_blind()
    for name,value in refmatches.items():dump(output/'references/match'/(name+'.json'),value)
    checkpoint()
    try:
        for key,secret in zip(method.KEY_IDS,method.KEYS):
            value=call('book/'+key,lambda secret=secret:method.carrier.codebook(secret))
            if value is not None:books[key]=value
        for arm in ('P0','PM'):
            value=call('load/'+arm,lambda arm=arm:backend.load_pixels(source_root/'media'/arm/'readback.pt'))
            if value is not None:pixels[arm]=value
        if pixels:
            model=call('model',backend.load)
            if model is not None:result['model']=model
        for case,arm,_,start,stop in PREPARATIONS:
            rgb=call('prepare/'+case,lambda arm=arm,start=start,stop=stop:prepare_received(pixels[arm],start,stop),deps['prepare/'+case])
            z=call('E/'+case,lambda rgb=rgb,case=case:save_tensor(Path('features')/(case+'.pt'),backend.encode_received(rgb)),deps['E/'+case])
            del rgb
            for key in method.KEY_IDS:
                n=case+'/'+key
                value=record('Q/'+n,lambda z=z,key=key:method.observe(z,books[key]),Path('blind/Q')/(n+'.json'),deps['Q/'+n])
                if value is not None:obs[n]=value
                value=record('match/'+n,lambda n=n:method.match(obs[n]['pooled_q']),Path('blind/match')/(n+'.json'),deps['match/'+n])
                if value is not None:matches[n]=value
                value=record('map/'+n,lambda n=n:method.local_map(matches[n]),Path('blind/map')/(n+'.json'),deps['map/'+n])
                if value is not None:maps[n]=value
            spectrum=call('FFT/'+case,lambda z=z:method.spectrum(z),deps['FFT/'+case])
            for key,secret in zip(method.KEY_IDS,method.KEYS):
                n=case+'/'+key
                value=record('votes/'+n,lambda spectrum=spectrum,secret=secret:method.read_spectrum(spectrum,secret),Path('blind/votes')/(n+'.json'),deps['votes/'+n])
                if value is not None:raw[n]=value
                value=record('payload/'+n,lambda n=n:method.reduce_payload(raw[n],maps[n]),Path('blind/payload')/(n+'.json'),deps['payload/'+n])
                if value is not None:payload[n]=value
            del z,spectrum;checkpoint()
    except BaseException as exc:result['fatal_error']=f'{type(exc).__name__}: {exc}'
    for name in names:
        if obs[name]['status']=='NOT_ENTERED':
            why=reason('Q/'+name);obs[name]=method.unavailable_observation(why['status']);obs[name].update(why)
        if matches[name]['status']=='NOT_ENTERED':
            why=reason('match/'+name);matches[name]=method.blank_match(why['status']);matches[name].update(why)
        if maps[name]['status']=='NOT_ENTERED':
            why=reason('map/'+name);maps[name]=method.unavailable_map(why['status'],why.get('unavailable_from'));maps[name].update(why)
        if raw[name]['status']=='NOT_ENTERED':
            why=reason('votes/'+name);raw[name]=method.unavailable_votes(why['status']);raw[name].update(why)
        if payload[name]['status']=='NOT_ENTERED':
            why=reason('payload/'+name);payload[name]=method.unavailable_payload(why['status'],why.get('unavailable_from'));payload[name].update(why)
        maps[name]['from_match_status']=matches[name]['status']
    save_blind()
    dump(output/'numeric_completion.json',dict(completed_unix=time.time(),blind_Q_tables=8,blind_sequence_score_slots=272,
        vote_tables=8,local_bit_slots=704,message_slots=256,all_fixed_blind_rows_persisted=True,truth_read=False,reference_read=False))
    # Known-origin old observations and all true geometry/messages are post-only.
    for arm in ('P0','PM'):
        for key in method.KEY_IDS:
            n=arm+'/'+key
            def reference(arm=arm,key=key):
                value=json.loads((source_root/'blind/sync'/arm/'mp4'/(key+'.json')).read_text())['pooled_q']
                if len(value)!=44 or any(len(row)!=44 for row in value):raise ValueError('stored full44x44 Q required')
                return value[15:26]
            value=record('reference_Q/'+n,reference,Path('references/Q')/(n+'.json'))
            if value is not None:refs[n]=value
            value=record('reference_match/'+n,lambda n=n:method.match(refs[n]),Path('references/match')/(n+'.json'),('reference_Q/'+n,))
            if value is not None:refmatches[n]=value
            def reference_bits(arm=arm,key=key):
                rows=json.loads((source_root/'blind/payload'/arm/(key+'.json')).read_text())['local_cells'][15:26]
                values=[[b['decoded'] for b in row['bits']] for row in rows]
                if len(values)!=11 or any(len(row)!=8 for row in values):raise ValueError('saved same-cell11x8 raw bits required')
                if any(b is not None and (type(b) is not int or b not in (0,1)) for row in values for b in row):raise ValueError('invalid saved reference bit')
                return values
            value=record('reference_bits/'+n,reference_bits,Path('references/raw_bits')/(n+'.json'))
            if value is not None:refbits[n]=value
    for name in refmatches:
        if refmatches[name]['status']=='NOT_ENTERED':
            why=reason('reference_match/'+name);refmatches[name]=method.blank_match(why['status']);refmatches[name].update(why)
            dump(output/'references/match'/(name+'.json'),refmatches[name])
    def read_truth():
        raw=json.loads(Path(truth_path).read_text());bits=raw['bits']
        if len(bits)!=32 or any(type(x) is not int or x not in (0,1) for x in bits):raise ValueError('post-only32 binary bits required')
        return dict(status='AVAILABLE',bits=bits,source=str(truth_path))
    truth=record('truth',read_truth,Path('references/truth.json')) or dict(status='UNAVAILABLE',bits=None)
    post=record('evaluation',lambda:evaluate(metadata,obs,matches,maps,raw,payload,refs,refmatches,refbits,truth),Path('evaluation.json'))
    result.update(status='EXECUTION_COMPLETE' if all(v['status']=='RETURNED' for n,v in ledger.data.items() if n!='truth') and post is not None and not result.get('fatal_error') else 'INCOMPLETE',
        calls=ledger.data,counts=ledger.counts(),seconds=time.perf_counter()-started,truth_status=truth['status'],
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED',trusted_full_messages=None,FPR=None,
        optional_fsync_unavailable=sorted(shared.SYNC_UNAVAILABLE_ERRNOS))
    if backend.device=='cuda':result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    dump(output/'result.json',result);checkpoint();return result
