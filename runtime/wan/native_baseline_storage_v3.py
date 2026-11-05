"""Consumed units and compact append-only native arrays, independent of source identity."""
import io
import json
import os
from pathlib import Path
import struct
import zipfile
from runtime.wan.native_baseline_storage_v1 import Journal, now, read_json, write_json, recover


def state(folder):
    p = Path(folder)
    if not p.exists() or not any(p.iterdir()):
        return 'NEVER_STARTED'
    try:
        s = read_json(p/'state.json')['status']
        return 'INTERRUPTED_CONSUMED' if s == 'STARTED' else s
    except (OSError, ValueError, KeyError):
        return 'INTERRUPTED_CONSUMED'


class Unit:
    def __init__(self, folder, planned):
        self.folder = Path(folder)
        if state(folder) != 'NEVER_STARTED':
            raise RuntimeError('Consumed unit cannot be restarted')
        self.folder.mkdir(parents=True, exist_ok=True)
        # Exclusive claim; scientific calls can only follow this durable record.
        with (self.folder/'state.json').open('x') as f:
            json.dump(dict(status='STARTED', at=now()), f); f.flush(); os.fsync(f.fileno())
        self.journal = Journal(self.folder, planned)

    def call(self, key, function):
        return self.journal.call(key, function)

    def reuse(self):
        row = self.journal.rows['load']
        if row['status'] != 'NOT_STARTED':
            raise RuntimeError('Load slot already consumed')
        row.update(status='REUSED_NO_CALL')
        self.journal.event(row)

    def close(self, error=None, **details):
        snapshot = self.journal.snapshot()
        write_json(self.folder/'blind_closed.json', dict(status='BLIND_CLOSED', at=now()))
        write_json(self.folder/'state.json', dict(status='COMPLETED' if error is None else 'FAILED',
                   error=None if error is None else f'{type(error).__name__}: {error}',
                   at=now(), **details, **snapshot))


def append_native(folder, task, start, arrays):
    """One file per unit, one durable record per returned native batch/view."""
    import numpy as np
    data = io.BytesIO()
    np.savez(data, **{k: np.asarray(v) for k, v in arrays.items()})
    header = json.dumps(dict(task=task, start=start)).encode()
    payload = data.getvalue()
    with (Path(folder)/'native.bin').open('ab') as f:
        f.write(struct.pack('<IQ', len(header), len(payload)))
        f.write(header); f.write(payload); f.flush(); os.fsync(f.fileno())


def read_native(folder):
    import numpy as np
    records, errors = [], []
    path = Path(folder)/'native.bin'
    if not path.exists():
        return records, errors
    with path.open('rb') as f:
        while True:
            offset = f.tell(); prefix = f.read(12)
            if not prefix:
                break
            try:
                if len(prefix) != 12:
                    raise ValueError('Interrupted record prefix')
                h, n = struct.unpack('<IQ', prefix)
                if h > 4096 or n > 16_000_000:
                    raise ValueError('Invalid native record length')
                header = json.loads(f.read(h)); payload = f.read(n)
                if len(payload) != n:
                    raise ValueError('Interrupted native record payload')
                with np.load(io.BytesIO(payload), allow_pickle=False) as values:
                    arrays = {k: values[k] for k in values.files}
                records.append((header, arrays))
            except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as e:
                errors.append(dict(offset=offset, error=str(e)))
                break  # Keep earlier records; never infer missing bits or rerun.
    return records, errors


def unit_ledger(folder):
    return {r['id']: r for r in recover(folder)['operations']}
