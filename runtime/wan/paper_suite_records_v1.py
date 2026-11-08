"""Ordinary new-package receipts. Old results are read, never rewritten."""
import csv
import json
import os
from pathlib import Path
import time
import uuid


def read(path):
    def invalid(value):raise ValueError('non-finite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=invalid)


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.pending-'+uuid.uuid4().hex)
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=1)+'\n')
    os.replace(temp,path)


def optional(path,default=None):
    try:return read(path)
    except (OSError,ValueError):return default


class Journal:
    def __init__(self,folder,plan):
        self.folder=Path(folder);self.rows=plan;write(self.folder/'operations.json',self.rows)
    def call(self,key,function):
        row=self.rows[key]
        if row['started']:raise RuntimeError('fixed operation already consumed: '+key)
        row.update(status='STARTED',started=1);write(self.folder/'operations.json',self.rows);tick=time.monotonic()
        try:
            value=function();row.update(status='RETURNED',returned=1);return value
        except BaseException as error:
            row.update(status='FAILED_CONSUMED',error=f'{type(error).__name__}: {error}');raise
        finally:
            row['seconds']=time.monotonic()-tick;write(self.folder/'operations.json',self.rows)


def reserve(folder,protocol):
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    if any(folder.iterdir()):return False
    try:
        with (folder/'attempt.json').open('x') as stream:json.dump(dict(status='CONSUMED',protocol=protocol),stream)
    except FileExistsError:return False
    return True


def table(folder,name,rows):
    folder=Path(folder);write(folder/(name+'.json'),rows)
    keys=list(dict.fromkeys(key for row in rows for key in row)) or ['status']
    with (folder/(name+'.csv')).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,keys);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
