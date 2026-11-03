"""Ordinary minimal stage transport. No hash, manifest or scientific recalculation."""
from pathlib import Path
import argparse
import json
import zipfile

COMMON=('attempt.json','state.json','operations.json','counts.json','resources.json','result.json',
        'blind_terminal.json','fixed_time_units.json','fixed_identity_units.json')


def package(stage,source,destination):
    source=Path(source).resolve();destination=Path(destination).resolve()
    receipt='cpu_handoff.json' if stage=='cpu' else 'blind_terminal.json'
    data=json.loads((source/receipt).read_text())
    if data.get('stage_terminal') is not True:raise ValueError('stage terminal receipt required for transport')
    if destination.exists():raise FileExistsError(destination)
    if destination.is_relative_to(source):raise ValueError('archive output must not be inside input package')
    if stage=='cpu':names=[p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()]
    elif stage=='gpu1':
        names=list(COMMON)+[s+'/'+f for s in ('n01','e02') for f in ('NEW_preclip.npy','native.json','quality.json','host_q0_difference.npy')]
    elif stage=='cpu_media':
        names=list(COMMON)+[v['file'] for v in data.get('rgb',{}).values() if v.get('status')=='SAVED']
    elif stage=='gpu2':names=list(COMMON)+['addressed_raw_bits.json','raw_bank.json']
    else:raise ValueError('unknown transport stage')
    names=sorted(set(names));files=[];missing=[]
    for name in names:
        p=Path(name)
        if p.is_absolute() or not (source/p).resolve().is_relative_to(source):raise ValueError('handoff uses non-relative file path')
        if (source/p).is_file():files.append(p.as_posix())
        else:missing.append(p.as_posix())
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_name(destination.name+'.partial')
    with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as archive:
        for name in files:archive.write(source/name,name)
    temporary.replace(destination)
    result=dict(status='TRANSPORT_PACKAGED',stage=stage,file=str(destination),bytes=destination.stat().st_size,
                included_files=files,missing_files_retained=missing,physical_calls=0,
                location_record_only=True,no_integrity_admission=True)
    destination.with_suffix(destination.suffix+'.transport.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('cpu','gpu1','cpu_media','gpu2'))
    parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    print(json.dumps(package(args.stage,args.source,args.output),indent=2))
