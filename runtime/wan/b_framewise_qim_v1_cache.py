"""FP32 cache disk measurement; unchanged capacity formula, no placement/retry policy."""
from pathlib import Path
import errno
import json
import shutil
import sys


def check_disk_cache_capacity(output,channels):
    path=Path(output).resolve()
    feature_bytes=4*181*channels*320*512
    auxiliary_bytes=4*181*320*512*3*5
    minimum=32*1024**3
    required=max(minimum,feature_bytes+auxiliary_bytes)
    receipt=dict(check_path=str(path),cache_path=str(path/'features'),channels=channels,
        frames=181,height=320,width=512,dtype='FP32',element_bytes=4,feature_blocks=46,
        feature_estimate_bytes=feature_bytes,auxiliary_estimate_bytes=auxiliary_bytes,
        shape_estimate_bytes=feature_bytes+auxiliary_bytes,minimum_bytes=minimum,required_bytes=required,
        measurement='shutil.disk_usage(check_path).free; not Drive quota or a guarantee of later writes')
    receipt_path=path/'disk_cache_capacity.json'
    try:
        receipt['actual_free_bytes']=shutil.disk_usage(path).free
    except OSError as error:
        receipt.update(status='MEASUREMENT_FAILED',error=str(error))
        try:receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
        except OSError:pass
        raise OSError(error.errno,f'disk cache capacity measurement failed at {path}: {error}') from error
    enough=receipt['actual_free_bytes']>=required
    receipt['status']='CAPACITY_CHECK_PASSED' if enough else 'INSUFFICIENT_DISK_CACHE_CAPACITY'
    receipt_error=None
    try:receipt_path.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    except OSError as error:receipt_error=error
    detail=(f'path={path}; actual_free_bytes={receipt["actual_free_bytes"]}; channels={channels}; '
            f'shape_estimate_bytes={receipt["shape_estimate_bytes"]}; minimum_bytes={minimum}; required_bytes={required}')
    if not enough:
        if receipt_error is not None:detail+=f'; capacity_receipt_write_error={receipt_error}'
        raise OSError(errno.ENOSPC,'disk cache capacity insufficient for FP32 native feature cache; '+detail)
    if receipt_error is not None:
        # The diagnostic record is optional. Real feature/preclip writes still
        # execute next and retain their own actual I/O failures.
        receipt['receipt_write_error']=str(receipt_error)
        try:print('Non-blocking disk cache capacity receipt write failure; '+detail+'; '+str(receipt_error),file=sys.stderr,flush=True)
        except Exception:pass
    return receipt
