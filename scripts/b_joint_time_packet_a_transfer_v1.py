"""Copy existing GPU-1 preclip bytes into ordinary transport parts; stdlib only."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import tempfile

PROJECT = Path('/content/drive/MyDrive/Video-WM/B-Joint-Time-Packet-A-Physical-V1')
CHUNK_BYTES = 96 * 1024 * 1024
SOURCES = ('n01', 'e02')


def split_file(source, destination, *, chunk_bytes=CHUNK_BYTES):
    """The runner uses 96 MiB; a small byte fixture may use a smaller chunk."""
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError('positive integer chunk size required')
    source, destination = Path(source), Path(destination)
    result = dict(source=str(source), status='COPYING', parts=[], copied_bytes=0)
    print(f'Copying existing file: {source}', flush=True)
    try:
        destination.mkdir(parents=True, exist_ok=False)
        with source.open('rb') as stream:
            while True:
                block = stream.read(chunk_bytes)
                if not block:
                    break
                path = destination / f'{source.name}.part-{len(result["parts"]):05d}'
                part = dict(file=str(path), bytes=0, status='WRITING')
                result['parts'].append(part)
                with path.open('xb') as target:
                    written = target.write(block)
                    if written != len(block):
                        raise OSError(f'short file write: {path}; {written}/{len(block)} bytes')
                part.update(bytes=path.stat().st_size, status='COPIED')
                result['copied_bytes'] += part['bytes']
                print(f'Part saved: {path} ({part["bytes"]} bytes)', flush=True)
        result['status'] = 'COPIED'
    except OSError as error:
        result.update(status='IO_ERROR', error=f'{type(error).__name__}: {error}')
        if result['parts'] and result['parts'][-1]['status'] == 'WRITING':
            part = result['parts'][-1]
            part['status'] = 'PARTIAL_IO_ERROR'
            try:
                part['bytes'] = Path(part['file']).stat().st_size
            except OSError:
                part['bytes'] = None
        print(f'Copy failed: {source}; {result["error"]}', flush=True)
    return result


def run(project=PROJECT):
    project = Path(project).expanduser().resolve()
    gpu1 = project / 'fixed-n01-e02-v1/gpu1'
    transport = project / 'transport'
    result = dict(status='PREPARING_TRANSFER', chunk_bytes=CHUNK_BYTES, files={},
                  scientific_calls=0, source_files_modified=False)
    try:
        transport.mkdir(parents=True, exist_ok=True)
        prefix = 'gpu1-preclip-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-')
        output = Path(tempfile.mkdtemp(prefix=prefix, dir=transport))
    except OSError as error:
        result.update(status='IO_ERROR', operation='create transport directory',
                      error=f'{type(error).__name__}: {error}', transport=str(transport))
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    result['output_directory'] = str(output)
    print(f'Transport output directory: {output}', flush=True)
    for source in SOURCES:
        result['files'][source] = split_file(gpu1 / source / 'NEW_preclip.npy', output / source)
    result['status'] = ('TRANSFER_COPIED' if all(v['status'] == 'COPIED' for v in result['files'].values())
                        else 'TRANSFER_WITH_IO_ERRORS')
    result['receipt'] = str(output / 'transfer_receipt.json')
    try:
        Path(result['receipt']).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    except OSError as error:
        result['receipt_error'] = f'{type(error).__name__}: {error}'
        print(f'Transfer receipt write failed: {result["receipt_error"]}', flush=True)
    print(f'Transfer finished: {result["status"]}; directory={output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(PROJECT))
    args = parser.parse_args()
    print(json.dumps(run(args.project), ensure_ascii=False, indent=2), flush=True)
