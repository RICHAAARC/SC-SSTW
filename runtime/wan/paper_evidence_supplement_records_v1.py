"""Small records and once-consumed quality receipts; retained results are never cleaned."""
from pathlib import Path
import csv
import json
import os
import time
import uuid


def read(path):
    def invalid(value): raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'),parse_constant=invalid)


def optional(path,errors=None):
    try: return read(path)
    except (OSError,ValueError,TypeError) as error:
        if errors is not None: errors.append(dict(path=str(path),error=f'{type(error).__name__}: {error}'))
        return None


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp-'+uuid.uuid4().hex)
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def table(root,name,rows):
    root=Path(root);write(root/(name+'.json'),rows)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (root/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,columns);writer.writeheader()
        for row in rows: writer.writerow({k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream: return list(csv.DictReader(stream))


def separate_output(output,inputs):
    output=Path(output).resolve()
    for value in inputs:
        source=Path(value).resolve()
        if output==source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('output overlaps a retained input: '+str(source))
    return output


def baseline_references(result_root, wan_root=None):
    """Read saved references; generic 'same OFF view' alone cannot prove a path."""
    from main.tube_state import paper_evidence_supplement_v1 as p
    root=Path(result_root)
    wan=Path(wan_root) if wan_root is not None else root/'baselines/wan-cohort-five-view-v1'
    rows=[]
    for sid in p.SOURCES:
        path=wan/sid/'attempt01/quality.json';saved=optional(path)
        for method in p.METHODS:
            found=[r for r in saved if isinstance(r,dict) and r.get('source')==sid and r.get('method')==method and r.get('view')=='FULL'] if isinstance(saved,list) else []
            record=found[0] if len(found)==1 else {}
            reference=record.get('reference');off=record.get('off_media')
            if isinstance(reference,dict): off=reference.get('media',off)
            if not isinstance(off,str) or not Path(off).is_absolute(): off=None
            rows.append(dict(source=sid,method=method,off_media=off,
                status='SAVED_EXPLICIT_PATH' if off else 'REFERENCE_PATH_UNKNOWN',
                saved_reference=reference,record_path=str(path),
                scope='same-source label or current default runner path alone is not an actual saved media pointer'))
    return rows


class QualityBook:
    """Reservation consumes one package; subsequent invocations are read-only."""
    def __init__(self,output,streams,config):
        self.root=Path(output);self.reused=(self.root/'attempt.json').exists()
        if self.reused: return
        self.root.mkdir(parents=True,exist_ok=True)
        if any(self.root.iterdir()): raise ValueError('choose an unused quality output directory')
        self.rows={'model_load':dict(kind='model_load',status='NOT_STARTED',started=0,returned=0)}
        for stream in streams:
            self.rows[stream['id']+'/decode']=dict(kind='media_decode',status='NOT_STARTED',started=0,returned=0)
            for index in stream['planned_indices']:
                self.rows[stream['id']+f'/frame{index}']=dict(kind='frame_score',status='NOT_STARTED',started=0,returned=0)
        with (self.root/'attempt.json').open('x',encoding='utf-8') as handle:
            json.dump(dict(status='CONSUMED',config=config,streams=streams,started_at=time.time()),handle,ensure_ascii=False,allow_nan=False)
            handle.flush();os.fsync(handle.fileno())
        self.save()

    def save(self,key=None):
        # Small per-operation updates avoid rewriting the 1429-row ledger per frame on Drive.
        if key is None: write(self.root/'operations.json',self.rows)
        else: write(self.root/'operation_updates'/(key+'.json'),self.rows[key])

    def call(self,key,fn):
        row=self.rows[key]
        if row['status']!='NOT_STARTED': raise ValueError('operation already consumed: '+key)
        row.update(status='STARTED',started=1,started_at=time.time());self.save(key);tick=time.monotonic()
        try:
            value=fn();row.update(status='RETURNED',returned=1,seconds=time.monotonic()-tick)
            if row['kind']=='frame_score': row['score']=value
            self.save(key);return value
        except BaseException as error:
            row.update(status='INTERRUPTED_CONSUMED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED_CONSUMED',
                error=f'{type(error).__name__}: {error}',seconds=time.monotonic()-tick);self.save(key)
            if isinstance(error,(KeyboardInterrupt,SystemExit)): raise
            return None
