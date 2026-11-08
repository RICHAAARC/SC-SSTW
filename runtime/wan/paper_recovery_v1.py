"""Separate infrastructure attempts; never reset the original scientific rows."""
from contextvars import ContextVar
from pathlib import Path
import time
import uuid
from .paper_suite_records_v1 import read,write

ACTIVE=ContextVar('paper_infrastructure_attempt',default=None)


def reserve(directory,output,unit,original):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if len(list(directory.glob('attempt-*.json')))>=2:raise RuntimeError('two shared source infrastructure recoveries consumed')
    record=dict(id=uuid.uuid4().hex,output=str(output),unit=unit,status='STARTED',started_at=time.time(),
        original=original,original_unknown_return_retained=True,scientific_denominator_unchanged=True)
    path=directory/('attempt-'+record['id']+'.json');write(path,record)
    return path,record


def target(path):
    path=Path(path);record=ACTIVE.get()
    if record is None or not path.exists():return path
    return path.with_name(path.stem+'.recovery-'+record['id']+path.suffix)


def publish_media(logical,actual):
    logical,actual=Path(logical),Path(actual)
    if logical!=actual:write(logical.with_suffix(logical.suffix+'.selected.json'),dict(actual=actual.name,original_preserved=True))


def selected_media(path):
    path=Path(path);pointer=path.with_suffix(path.suffix+'.selected.json')
    if not pointer.is_file():return path
    name=read(pointer)['actual']
    if Path(name).name!=name:raise ValueError('selected media must remain in result directory')
    actual=path.with_name(name)
    if not actual.is_file():raise FileNotFoundError('saved recovery media unavailable: '+str(actual))
    return actual
