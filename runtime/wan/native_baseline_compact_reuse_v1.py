"""Read-only old-package references, snapshotted per current unit; no old writes."""
from collections import Counter
from pathlib import Path
from .native_baseline_storage_v1 import read_json,write_json
from .native_baseline_storage_v4 import UnitStore,LazyArrays,events,append_json,Unavailable
from .native_baseline_io_v4 import MediaOperations,load_rgb,file_available


class ReadOnlyStore:
    def __init__(self,root,unit,array_directory=None):
        self.folder=Path(root)/'units'/unit;self.rows={};self.record_errors=[];self.attempts=Counter()
        for row in events(self.folder/'calls.jsonl'):
            if 'operation' in row:
                self.rows[row['operation']]=row
                if row['status']=='STARTED':self.attempts[row.get('physical_key',row['operation'])]+=1
        self.arrays=LazyArrays(array_directory or self.folder,self.record_errors,self.rows)
    def row(self,key):return self.rows.get(key,dict(status='NOT_STARTED'))


class ImportedStore(UnitStore):
    def perform(self,key,*args,**kwargs):
        if self.record_errors:raise Unavailable('unreadable actual unit receipts; consumption unknown: '+self.unit)
        row=self.row(key)
        if row.get('imported_from'):
            if row['status']!='COMMITTED':raise Unavailable('old consumed operation retained without a new attempt: '+key)
            if kwargs.get('output_available') and not kwargs['output_available'](row.get('value')):
                raise Unavailable('old committed output unavailable; no implicit reconstruction: '+key)
        return super().perform(key,*args,**kwargs)


class Reuse:
    def __init__(self,output,old,old_cache):self.output=Path(output);self.old=Path(old);self.old_cache=Path(old_cache)

    def import_unit(self,unit,newfolder):
        marker=newfolder/'old_import.json'
        if marker.is_file():return read_json(marker)
        oldfolder=self.old/'units'/unit
        category=unit.split('/')[0]
        records=list(events(oldfolder/'calls.jsonl'))
        errors=[r for r in records if r.get('status')=='UNREADABLE_EVENT']
        # Snapshot all recorded statuses, but only encoded media (own-cache decodes are new calls).
        if category=='media':records=[r for r in records if r.get('operation','').endswith('/encode')]
        directory=self.old_cache/'arrays'/unit if category in ('writer','jpeg') else oldfolder
        converted=[]
        for row in records:
            if 'operation' not in row:continue
            row=dict(row,imported_from=str(oldfolder))
            if row.get('array_file'):row['array_file']=str(directory/row['array_file'])
            elif row['status'] in ('COMMITTED','RETURNED_NOT_COMMITTED') and category in ('native','writer','quality'):
                row['array_file']=str(directory/('arrays-'+str(row.get('runtime_id'))+'.bin'))
            converted.append(row)
        # This is a single atomic snapshot file. Store loading reads it before own receipts.
        value=dict(old_root=str(self.old),unit=unit,events=converted,old_unit_present=oldfolder.is_dir(),read_errors=errors)
        write_json(marker,value)
        append_json(self.output/'imports.jsonl',dict(unit=unit,old_root=str(self.old),
            imported_events=len(converted),old_started=sum(r['status']=='STARTED' for r in converted),
            new_physical_calls=0))
        return value

    def view(self,store,view):
        marker=store.folder/'views'/view['id']/'old_import.json'
        if marker.is_file():return read_json(marker)
        oldpath=self.old/'units'/store.unit/'views'/view['id']/'epochs.json'
        epochs=read_json(oldpath) if oldpath.is_file() else []
        # Fix the latest recorded epoch, never select using query score or post truth.
        if epochs and not (store.folder/'views'/view['id']/'epochs.json').exists():
            write_json(store.folder/'views'/view['id']/'epochs.json',epochs)
        value=dict(imported=bool(epochs),epoch=epochs[-1] if epochs else None,source=str(oldpath))
        write_json(marker,value);return value

    def quality(self,sid,method):
        new=self.output/'units/quality'/sid/method/'quality'
        for stage in ('precodec','G1','G2-identity'):
            target=new/(stage+'.json');old=self.old/'units/quality'/sid/method/'quality'/(stage+'.json')
            if not target.exists() and old.is_file():
                try:
                    value=read_json(old)
                    if not isinstance(value,dict):raise ValueError('quality record must be an object')
                    write_json(target,dict(value,imported_from=str(old)))
                except (OSError,ValueError,TypeError) as error:
                    write_json(target,dict(status='UNREADABLE_OLD_QUALITY',read_error=str(error),imported_from=str(old)))


class CompactOperations(MediaOperations):
    def __init__(self,output,cache,budget,media,reuse):
        super().__init__(output,cache,budget,media);self.reuse=reuse

    def store(self,unit,*,cache_arrays=False):
        key=(unit,cache_arrays)
        if key not in self._stores:
            folder=self.output/'units'/unit;folder.mkdir(parents=True,exist_ok=True)
            imported=self.reuse.import_unit(unit,folder)
            store=ImportedStore(self.output,unit,self.budget,array_directory=self.cache/'arrays'/unit if cache_arrays else None)
            own=dict(store.rows);own_attempts=store.attempts.copy();store.rows.clear();store.attempts.clear()
            for row in imported['events']:
                store.rows[row['operation']]=row
                if row['status']=='STARTED':store.attempts[row.get('physical_key',row['operation'])]+=1
            store.rows.update(own);store.attempts.update(own_attempts)
            store.record_errors.extend(imported.get('read_errors',[]))
            store.record_errors.extend(r for r in events(store.path) if r.get('status')=='UNREADABLE_EVENT')
            self._stores[key]=store
        self._stores.move_to_end(key)
        while len(self._stores)>8:self._stores.popitem(last=False)
        return self._stores[key]

    def source(self,source):
        old=self.reuse.old/'canonical'/(source['id']+'.npy')
        if old.is_file():return load_rgb(old)
        unit='source/'+source['id'];store=self.store(unit)
        if any(r.get('imported_from') for r in store.rows.values()):
            raise Unavailable('old source work consumed but canonical unavailable: '+source['id'])
        return super().source(source)

    def writer(self,source,method,canonical,query,adapter):
        # Old unavailable writer batches never silently become fresh work in this package.
        unit='writer/'+source['id']+'/'+method;store=self.store(unit,cache_arrays=True)
        if any(r.get('imported_from') for r in store.rows.values()):
            from main.baselines.public_fpr001_v4 import writer_batches
            for start,n in writer_batches(method):
                row=store.row('batch'+str(start))
                if row['status']!='COMMITTED':raise Unavailable('old writer incomplete; retain missing quality/media')
        return super().writer(source,method,canonical,query,adapter)


def read_saved_view(store,view,backend):
    """Isolate actual unreadable batch records, retaining all other fixed observations."""
    from main.baselines.public_fpr001_v4 import batches,read_batch
    from .native_baseline_io_v4 import empty_raw,validate_native
    raw=empty_raw(view['frames'],backend);errors=[];epoch=None
    path=store.folder/'views'/view['id']/'epochs.json'
    try:
        epochs=read_json(path) if path.is_file() else []
        if not isinstance(epochs,list):raise ValueError('epochs must be a list')
        epoch=epochs[-1] if epochs else None
        if epoch is not None and (not isinstance(epoch,dict) or not isinstance(epoch.get('id'),str)):
            raise ValueError('invalid selected epoch')
    except (OSError,ValueError,TypeError) as error:return raw,None,[str(error)]
    if epoch:
        for start,n in batches(view['frames'],read_batch(backend)):
            try:
                record=store.arrays.get(view['id']+'/'+epoch['id']+'/batch'+str(start))
                if record is None:continue
                value=record['arrays'];validate_native(value,n,backend)
                for key in ('bits','valid','logits'):
                    if key in raw:raw[key][start:start+n]=value[key]
                raw['observed'][start:start+n]=True
            except (OSError,ValueError,KeyError,TypeError,EOFError) as error:
                errors.append(dict(start=start,frames=n,error=str(error)))
    return raw,epoch,errors
