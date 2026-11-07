"""Durable media/writer checkpoints and pixel-only reader calls for the v4 runner."""
import os
from pathlib import Path
import shutil
import time
from collections import OrderedDict

from main.baselines.public_fpr001_v4 import batches, read_batch, SHAPE, writer_batches, frame_messages
from .native_baseline_storage_v1 import read_json, write_json
from .native_baseline_storage_v4 import UnitStore, ViewEpochs, Unavailable, SpaceUnavailable, append_json
from .native_baseline_media_v4 import edit_temporal, spatial


def check_space(path, required=1_000_000_000):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(path).free
    if free<required:raise SpaceUnavailable(f'actual available bytes {free} < next-unit reserve {required}; retain all results')
    return free


def validate_rgb(value,n=181):
    import numpy as np
    if value.shape!=(n,320,512,3) or value.dtype!=np.uint8:raise ValueError('received RGB8 frame count/geometry differs')


def save_array(path,value):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.pending')
    with temporary.open('wb') as stream:
        np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return dict(path=str(path),shape=list(value.shape),bytes=path.stat().st_size)


def load_rgb(path,n=181):
    import numpy as np
    value=np.load(path,mmap_mode='r',allow_pickle=False);validate_rgb(value,n);return value


def file_available(value):return isinstance(value,dict) and Path(value.get('path','')).is_file()


def remove_cache(path,cache):
    """Only the task's cache is eligible; never remove any protected result ancestor."""
    path=Path(path).resolve();cache=Path(cache).resolve()
    if path==cache or not path.is_relative_to(cache) or 'Video-WM_Result' in path.parts:
        raise ValueError('cleanup target must be strictly within the separate cache')
    if path.is_dir():shutil.rmtree(path)
    elif path.exists():path.unlink()


def validate_native(value,n,backend):
    import numpy as np
    bits=np.asarray(value['bits']);valid=np.asarray(value['valid'])
    if bits.shape!=(n,32 if backend=='wam' else 256) or valid.shape!=(n,) or valid.dtype!=np.bool_:
        raise ValueError('native batch shape/validity invalid')
    if not np.isin(bits[valid],[0,1]).all():raise ValueError('nonbinary returned bits')
    if backend!='wam':
        logits=np.asarray(value['logits'])
        if logits.shape!=(n,257) or logits.dtype!=np.float32 or not np.isfinite(logits).all() or not valid.all():
            raise ValueError('actual complete finite FP32 Seal logits required')
        if not np.array_equal(bits,(logits[:,1:]>0).astype(np.uint8)):raise ValueError('bits differ from raw logits')
    else:
        for key in ('selected_count','center_count','sums','means'):
            if key not in value:raise ValueError('missing native WAM statistic '+key)
        for key in ('sums','means'):
            if not np.isfinite(value[key]).all():raise ValueError('nonfinite WAM statistic')


def empty_raw(n,backend):
    import numpy as np
    out=dict(bits=np.zeros((n,32 if backend=='wam' else 256),np.uint8),valid=np.zeros(n,bool),observed=np.zeros(n,bool))
    if backend!='wam':out['logits']=np.zeros((n,257),np.float32)
    return out


def saved_raw(store,view,n,backend):
    """Only one selected epoch contributes to a received sequence; older partials remain on disk."""
    path=store.folder/'views'/view/'epochs.json';raw=empty_raw(n,backend)
    epochs=read_json(path) if path.is_file() else []
    epoch=epochs[-1] if epochs else None
    if epoch:
        prefix=view+'/'+epoch['id']
        for start,k in batches(n,read_batch(backend)):
            record=store.arrays.get(prefix+f'/batch{start}')
            if record is None:continue
            value=record['arrays'];validate_native(value,k,backend)
            raw['observed'][start:start+k]=True
            for key in ('bits','valid','logits'):
                if key in raw:raw[key][start:start+k]=value[key]
    return raw,epoch


class MediaOperations:
    def __init__(self,output,cache,budget,media):
        self.output=Path(output);self.cache=Path(cache);self.budget=budget;self.media=media
        self._stores=OrderedDict()

    def store(self,unit,*,cache_arrays=False):
        key=(unit,cache_arrays)
        if key not in self._stores:
            self._stores[key]=UnitStore(self.output,unit,self.budget,array_directory=self.cache/'arrays'/unit if cache_arrays else None)
        self._stores.move_to_end(key)
        while len(self._stores)>8:self._stores.popitem(last=False)
        return self._stores[key]

    def failure(self,where,error):
        append_json(self.output/'failures.jsonl',dict(where=where,error=f'{type(error).__name__}: {error}',at=time.time(),runtime=self.budget.runtime_id))

    def source(self,source):
        sid=source['id'];store=self.store('source/'+sid)
        canonical=self.output/'canonical'/f'{sid}.npy'
        if canonical.is_file() and store.row('normalize')['status']=='COMMITTED':return load_rgb(canonical)
        check_space(self.output);check_space(self.cache)
        def download():
            # Preserve partial attempts; no overwrite and no replacement source.
            path=self.cache/'sources'/sid/f"attempt{store.attempts['download']:02}.avi"
            value=self.media.download(source,path);return dict(value,path=str(path))
        value=store.perform('download','download',download,output_available=file_available)
        metadata=store.perform('source_probe','source_probe',lambda:self.media.source_probe(value['path']))
        def normalize():
            rgb=self.media.normalize(value['path'],metadata);validate_rgb(rgb)
            return save_array(canonical,rgb)
        store.perform('normalize','normalize',normalize,output_available=file_available)
        return load_rgb(canonical)

    def encoded(self,store,key,pixels,directory,*,codec='h264',crf=18):
        def encode():
            folder=Path(directory);folder.mkdir(parents=True,exist_ok=True)
            path=folder/f"clip-{store.attempts[key]:02}.mp4"
            # Prior failed outputs stay in the formal failure package before cache cleanup.
            try:return self.media.encode(pixels(),path,25,codec=codec,crf=crf)
            except BaseException:
                if path.is_file() and path.resolve().is_relative_to(self.cache.resolve()):
                    keep=self.output/'failures'/store.unit/key/path.name;keep.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,keep)
                raise
        return store.perform(key,'video_encode',encode,output_available=file_available)['path']

    def decoded(self,store,key,clip,n,cache_path,*,codec='h264'):
        probe_key=key.split('/decode')[0]+'/probe'
        # Each physical clip has one probe; cache-loss reconstruction gets its own probe address.
        clip_token=Path(clip).name
        store.perform(probe_key+'/'+clip_token,'video_probe',lambda:self.media.probe(clip,[n,320,512,3],25,codec=codec),physical_key=probe_key)
        def decode():
            value=self.media.decode(clip,[n,320,512,3]);validate_rgb(value,n)
            return save_array(cache_path,value)
        store.perform(key,'video_decode',decode,output_available=file_available)
        return load_rgb(cache_path,n)

    def g1(self,source,arm,pixels=None):
        sid=source['id'];store=self.store('media/'+sid+'/'+arm)
        clip=self.encoded(store,'G1/encode',pixels,self.output/'media'/sid/arm/'G1')
        cache=self.cache/'g1'/sid/(arm+'.npy')
        return self.decoded(store,'G1/decode',clip,181,cache)

    def g1_saved(self,source,arm):
        # Never regenerates a missing donor writer implicitly.
        store=self.store('media/'+source['id']+'/'+arm);row=store.row('G1/encode')
        if row['status']!='COMMITTED' or not file_available(row.get('value')):raise Unavailable('fixed DEV donor G1 missing: '+source['id']+'/'+arm)
        return self.decoded(store,'G1/decode',row['value']['path'],181,self.cache/'g1'/source['id']/(arm+'.npy'))

    def writer(self,source,method,canonical,query,adapter):
        import numpy as np
        unit='writer/'+source['id']+'/'+method;store=self.store(unit,cache_arrays=True)
        parts=[];missing=[]
        for start,n in writer_batches(method):
            message=frame_messages(query,start,n) if method=='frameindex' else query
            try:
                value=store.perform(f'batch{start}','embed_'+method,
                    lambda s=start,k=n,msg=message:dict(rgb=adapter.embed_batch(canonical[s:s+k],msg,method=method,start=s)),
                    arrays=True,validate=lambda x,k=n:validate_rgb(x['rgb'],k))
                parts.append(value['rgb'])
            except SpaceUnavailable:raise
            except Unavailable as error:missing.append(dict(start=start,frames=n,error=str(error)))
        if missing:raise Unavailable('writer incomplete: '+str(missing))
        return np.concatenate(parts,axis=0)

    def received(self,source,arm,view,plan,g1,donor_lookup,*,decode_tag):
        sid=source['id'];operation=view['operation'];store=self.store('media/'+sid+'/'+arm)
        def edited():
            if operation.startswith('T'):
                donor=donor_lookup(plan['donor'],arm) if operation=='T12' else None
                return edit_temporal(g1,plan['maps'][operation],donor)
            value=spatial(g1,operation,plan['noise_seed_sequence'])
            if operation=='S07':
                jstore=self.store('jpeg/'+sid+'/'+arm,cache_arrays=True);frames=[];missing=[]
                for i,frame in enumerate(value):
                    try:
                        encoded=jstore.perform(f'encode/{i}','jpeg_encode',lambda f=frame:dict(bytes=self.media.jpeg_encode(f)),arrays=True)
                        decoded=jstore.perform(f'decode/{i}','jpeg_decode',lambda e=encoded:dict(rgb=self.media.jpeg_decode(e['bytes'])),
                            arrays=True,validate=lambda x:validate_rgb(x['rgb'][None],1))
                        frames.append(decoded['rgb'])
                    except SpaceUnavailable:raise
                    except Unavailable as error:missing.append(dict(frame=i,error=str(error)))
                if missing:raise Unavailable('JPEG incomplete: '+str(missing))
                import numpy as np
                value=np.stack(frames)
            return value
        if view['generation']=='G1':return edited()
        codec='h265' if operation=='S01' else 'h264';crf=28 if operation in ('S00','S01') else 18
        key=view['id'];directory=self.cache/'derived'/sid/arm/key
        clip=self.encoded(store,key+'/encode',edited,directory,codec=codec,crf=crf)
        # Decoder's own output cache lives only until the reader epoch has its input.
        return self.decoded(store,key+'/decode/'+decode_tag,clip,view['frames'],directory/(decode_tag+'.npy'),codec=codec)


def extract_view(ops,source,backend,arm,view,adapter,pixels):
    """Raw extraction has a pixel supplier only. Queries/maps enter evaluation after this returns."""
    unit='native/'+source['id']+'/'+backend+'/'+arm;store=ops.store(unit)
    n=view['frames'];size=read_batch(backend);category='detect_'+backend
    epochs=ViewEpochs(store,view['id'],ops.cache/'received'/source['id']/backend/arm/view['id'])
    try:
        epoch,_=epochs.choose(batches(n,size),category);prefix=view['id']+'/'+epoch['id']
        pending=not all(store.reusable(prefix+f'/batch{s}') for s,k in batches(n,size))
        if pending:
            if not Path(epoch['input']).is_file():
                value=pixels();validate_rgb(value,n);save_array(epoch['input'],value)
            received=load_rgb(epoch['input'],n)
            for start,k in batches(n,size):
                if store.reusable(prefix+f'/batch{start}'):continue
                try:
                    store.perform(prefix+f'/batch{start}',category,
                        lambda s=start,count=k:adapter.detect(received[s:s+count]),arrays=True,
                        validate=lambda x,count=k:validate_native(x,count,backend),
                        recovery=epoch['whole_view_recovery'],physical_key=view['id']+f'/batch{start}')
                except SpaceUnavailable:raise
                except Unavailable as error:ops.failure(unit+'/'+view['id']+f'/{start}',error)
            del received
        raw,epoch=saved_raw(store,view['id'],n,backend)
        # Do not stop a view after a positive witness: every planned batch above was attempted.
        if raw['observed'].all() and Path(epoch['input']).is_file() and Path(epoch['input']).resolve().is_relative_to(ops.cache.resolve()):remove_cache(epoch['input'],ops.cache)
        elif Path(epoch['input']).is_file():
            # A failed input is scientific evidence too. Resume directly from its protected copy.
            keep=ops.output/'failures'/unit/view['id']/(epoch['id']+'.npy')
            keep.parent.mkdir(parents=True,exist_ok=True)
            if not keep.is_file():shutil.copy2(epoch['input'],keep)
            old=epoch['input'];epoch['input']=str(keep);epochs.rows[-1]['input']=str(keep);write_json(epochs.path,epochs.rows)
            if Path(old).resolve().is_relative_to(ops.cache.resolve()):remove_cache(old,ops.cache)
        write_json(store.folder/'views'/view['id']/'status.json',dict(epoch=epoch,planned=n,
            observed=int(raw['observed'].sum()),complete=bool(raw['observed'].all()),record_errors=store.record_errors))
        return raw,epoch
    except SpaceUnavailable:raise
    except Exception as error:
        ops.failure(unit+'/'+view['id'],error)
        return saved_raw(store,view['id'],n,backend)
