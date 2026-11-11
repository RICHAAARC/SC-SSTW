"""Stage saved compact records locally, reuse the evaluator, publish one ZIP.

No model, codec, raw-array replay, threshold fitting or denominator changes.
Original evidence is read-only; an unreadable selected overlay never falls back
to an older success. Local failures and interrupted uploads remain inspectable.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import copy
from pathlib import Path
import json
import os
import shutil
import tempfile
import time
import uuid
import zipfile

from main.baselines import public_compact12_v1 as p
from main.tube_state import paper_execution_v1 as schedule
from .native_baseline_storage_v1 import read_json, write_json


METADATA = ('queries.json', 'inherited_budget.json', 'physical_calls.jsonl',
            'physical_counts.json', 'model_resources.jsonl',
            'quality_resources.jsonl', 'model_loads.jsonl', 'imports.jsonl',
            'cpu_package_imports.jsonl', 'cpu_recovery.json',
            'physical_recovery.json')


def stage_records(ctx, work, progress, workers=8):
    """Copy only table inputs; preserve each slot's existing root precedence."""
    local = copy(ctx)
    local.parent = work / 'inputs/original'
    local.output = work / 'inputs/repair'
    local.old = work / 'inputs/old_v4'
    local.query_cache = {}
    locations = [(ctx.output, local.output), (ctx.parent, local.parent),
                 (ctx.old, local.old)]
    locations.sort(key=lambda pair: len(str(pair[0])), reverse=True)

    def destination(path):
        for original, staged in locations:
            if path.is_relative_to(original):
                return staged / path.relative_to(original)
        raise ValueError('table input outside the supplied result roots: ' + str(path))

    jobs = []
    for shard in range(4):
        for pack in schedule.owned_packs(ctx.sources, shard):
            for source in pack['sources']:
                roots = ctx.roots(shard)
                sid = source['id']
                for method in p.METHODS:
                    arms = ('OFF',) if source['split'] == 'CAL' else ('OFF', method)
                    for arm in arms:
                        for view in p.VIEW_IDS:
                            rel = Path('post') / sid / method / arm / (view + '.json')
                            jobs.append([root / rel for root in roots])
                    if source['split'] != 'CAL':
                        for stage in ('precodec', 'G1', 'G2-identity'):
                            rel = Path('units/quality') / sid / method / 'quality' / (stage + '.json')
                            jobs.append([root / rel for root in roots])
    for source in ctx.roster:
        if source['split'] != 'DEV':
            continue
        for method in p.METHODS:
            for arm in ('OFF', method):
                for view in p.VIEW_IDS:
                    jobs.append([ctx.old / 'post' / source['id'] / method / arm / (view + '.json')])
            for stage in ('precodec', 'G1', 'G2-identity'):
                jobs.append([ctx.old / 'units/quality' / source['id'] / method / 'quality' / (stage + '.json')])
    roots = [ctx.parent, ctx.old, ctx.output]
    roots += [ctx.parent / 'shards' / f'part{i}' for i in range(4)]
    roots += [ctx.part(i) for i in range(4)]
    for root in dict.fromkeys(roots):
        jobs.extend([[root / name] for name in METADATA])

    def transfer(candidates):
        selected = next((path for path in candidates if path.is_file()), None)
        if selected is None:
            return False
        target = destination(selected)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.partial')
        # Copy failures stop staging. Do not replace failed/new evidence with old data.
        shutil.copyfile(selected, temporary)
        os.replace(temporary, target)
        return True

    copied = missing = 0
    total = len(jobs)
    progress('STAGING', completed=0, planned=total, copied=0, absent=0)
    last = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(transfer, job): job for job in jobs}
        try:
            for done, future in enumerate(as_completed(pending), 1):
                if future.result():
                    copied += 1
                else:
                    missing += 1
                if done == total or time.monotonic() - last >= 10:
                    progress('STAGING', completed=done, planned=total, copied=copied, absent=missing)
                    last = time.monotonic()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return local, locations, dict(planned=total, copied=copied, absent=missing)


def publish_archive(work, output, run_id, progress):
    """Finish the ZIP locally; upload to a new name, then publish its final name."""
    archive = work / 'global_results.zip'
    files = [path for folder in ('inputs', 'global') for path in (work / folder).rglob('*') if path.is_file()]
    files += [work / 'stage_report.json']
    progress('PACKING', files=len(files))
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
        for path in files:
            package.write(path, path.relative_to(work).as_posix())
    folder = Path(output) / 'aggregation_runs' / run_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / archive.name
    partial = folder / (archive.name + '.partial')
    progress('UPLOADING', bytes=archive.stat().st_size, archive=str(target))
    with archive.open('rb') as source, partial.open('xb') as dest:
        shutil.copyfileobj(source, dest, length=1024 * 1024)
        dest.flush()
        os.fsync(dest.fileno())
    os.replace(partial, target)
    return target


def collect_local(ctx, local_root=None):
    """Aggregate on local disk and preserve a single self-contained result ZIP."""
    from .native_baseline_compact_recovery_v1 import collect
    base = Path(local_root) if local_root else Path('/content/video-wm-compact-aggregation') if Path('/content').is_dir() else Path(tempfile.gettempdir()) / 'video-wm-compact-aggregation'
    # A local workspace must never be placed in the formal/Drive result tree.
    resolved = base.resolve()
    if any(resolved.is_relative_to(Path(root).resolve()) for root in (ctx.parent, ctx.output, ctx.old)) or resolved.is_relative_to(Path('/content/drive')):
        raise ValueError('aggregation workspace must be local and outside formal result roots')
    base.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8]
    work = base / run_id
    work.mkdir()
    def progress(stage, **details):
        record = dict(stage=stage, at=time.time(), local_work=str(work), **details)
        write_json(work / 'progress.json', record)
        print('[local tables] ' + json.dumps(record, ensure_ascii=False), flush=True)
    try:
        local, locations, staged = stage_records(ctx, work, progress)
        write_json(work / 'stage_report.json', dict(status='STAGED', **staged,
            locations=[dict(original=str(a), local=str(b)) for a, b in locations],
            missing_slots='retained by the existing evaluator', new_model_calls=0, new_codec_calls=0))
        progress('AGGREGATING', inputs=staged['copied'])
        result = collect(local, progress=progress)
        shutil.move(str(local.output / 'global'), str(work / 'global'))
        # Keep locating fields pointing at original evidence, not disposable /content.
        for path in (work / 'global').rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.csv', '.md'):
                text = path.read_text()
                for original, staged_path in locations:
                    text = text.replace(str(staged_path), str(original))
                path.write_text(text)
        result = read_json(work / 'global/result.json')
        archive = publish_archive(work, ctx.output, run_id, progress)
        receipt = dict(status='GLOBAL_TABLES_SAVED', run_id=run_id, archive=str(archive),
                       local_work=str(work), global_result=result, new_model_calls=0, new_codec_calls=0)
        write_json(archive.parent / 'receipt.json', receipt)
        write_json(ctx.output / 'aggregation_latest.json', receipt)
        progress('SAVED', archive=str(archive), measurement_status=result['status'])
        return dict(result, delivery=receipt)
    except BaseException as error:
        progress('FAILED', error=f'{type(error).__name__}: {error}')
        raise
