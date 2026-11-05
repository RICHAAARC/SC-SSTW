"""Source-unit durable orchestration. Only this explicit runner performs physical work."""
from collections import Counter
import json
from pathlib import Path
import time
from main.baselines.native_paper_fpr1_v3 import (SHAPE, METHODS, CHANNELS, CONDITIONS, EXPERIMENT,
    validate_config, validate_sources, queries, source_maps, operations, tasks, budget)
from runtime.wan.native_baseline_storage_v1 import read_json, write_json, now
from runtime.wan.native_baseline_storage_v3 import Unit, state, append_native, unit_ledger

DEFAULT_CONFIG = Path(__file__).resolve().parents[2]/'experiments/wan_state_clock/configs/native_baselines_paper_fpr1_v3.json'


def load_inputs(config_path):
    config_path = Path(config_path)
    config = validate_config(read_json(config_path))
    source_path = config_path.parent/config['source_list']
    if not source_path.is_file():
        raise FileNotFoundError(f'PENDING_INPUT: actual 1000-source roster missing: {source_path}')
    data = read_json(source_path)
    return config, validate_sources(data['sources'] if isinstance(data, dict) else data)


def initialize(output, config, sources):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        # Resume saved method/data semantics, not a code/hash identity comparison.
        config = validate_config(read_json(output/'protocol.json'))
        sources = validate_sources(read_json(output/'sources.json'))
    else:
        validate_config(config); validate_sources(sources)
        write_json(output/'sources.json', sources)
        write_json(output/'protocol.json', config)
    if not (output/'queries.json').exists():
        if any((output/'units').glob('*/*/state.json')):
            raise ValueError('Saved preassigned queries unreadable/missing after physical units started')
        write_json(output/'queries.json', queries(sources))
    write_json(output/'physical_budget.json', budget())
    write_json(output/'unit_plan.json', [dict(source=s['id'], split=s['split'], ordinal=s['ordinal'],
        shard=(s['ordinal']-1)//10, units=['prepare', *METHODS]) for s in sources])
    return config, sources


def media_path(output, source, carrier, name):
    return Path(output)/'media'/source['id']/carrier/(name+'.mp4')


def encode(unit, media, rgb, output, source, carrier, name, *, decode=False):
    path = media_path(output, source, carrier, name)
    # Physical slot uses WM independent of the method directory name.
    arm = 'OFF' if carrier == 'OFF' else 'WM'; key = f'{arm}/{name}'
    metadata = unit.call('encode/'+key, lambda: media.encode(rgb, path, 25))
    probe = unit.call('probe/'+key, lambda: media.probe(path, list(rgb.shape), 25))
    write_json(path.with_suffix('.json'), dict(encode=metadata, probe=probe, shape=list(rgb.shape), fps=25))
    if decode:
        return unit.call('decode/'+key, lambda: media.decode(path, list(rgb.shape)))


def prepare(output, source, media):
    import numpy as np
    folder = Path(output)/'units'/'prepare'/source['id']
    if state(folder) != 'NEVER_STARTED':
        return
    unit = Unit(folder, operations(source)); error = None
    try:
        # clip is the catalog's numeric clip ordinal; preserve the actual filename,
        # including HandStandPushups capitalization, from dataset_path.
        filename = Path(source['dataset_path']).name
        avi = Path(output)/'sources'/source['id']/filename
        if not filename.lower().endswith('.avi') or '\\' in filename or filename in ('.', '..'):
            raise ValueError('Unsafe or invalid source filename')
        download = unit.call('download', lambda: media.download(source, avi))
        metadata = unit.call('source_probe', lambda: media.source_probe(avi))
        write_json(folder/'source.json', dict(metadata=metadata, download=download, avi=str(avi)))
        raw = unit.call('normalize', lambda: media.normalize(avi, metadata))
        if list(raw.shape) != SHAPE or raw.dtype != np.uint8:
            raise ValueError('Actual source normalization shape/dtype invalid')
        # CPU descriptive motion only; no content selection or threshold.
        motion = [float(np.abs(raw[i].astype(np.int16)-raw[i-1].astype(np.int16)).mean()) for i in range(1, 181)]
        write_json(folder/'motion.json', dict(adjacent_rgb_abs_mean=motion, mean=float(np.mean(motion))))
        off = encode(unit, media, raw, output, source, 'OFF', 'G1', decode=True)
        del raw
        mapping = source_maps(source)
        write_json(folder/'edit_truth.json', mapping)
        for condition in CONDITIONS:
            view = off[mapping['conditions'][condition]['indices']]
            encode(unit, media, view, output, source, 'OFF', condition)
            del view
    except BaseException as exc:
        error = exc
    finally:
        unit.close(error)
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        raise error


def blind_read(adapter, pixels, task, unit):
    """Native receives pixels only; labels locate output. Query/truth absent."""
    import numpy as np
    if len(pixels) != task['frames']:
        raise ValueError('Received frame count does not match fixed view')
    if task['method'] == 'wam':
        for start in range(0, len(pixels), 8):
            value = unit.call(f"detect/{task['id']}/{start}", lambda s=start: adapter.detect(pixels[s:s+8]))
            append_native(unit.folder, task['id'], start, value)
    else:
        value = unit.call(f"detect/{task['id']}/0", lambda: adapter.detect(pixels))
        append_native(unit.folder, task['id'], 0, {k: np.asarray(v) for k, v in value.items()})


def run_source_method(output, source, method, unit, adapter, media, measure):
    import numpy as np
    output = Path(output)
    prepared = output/'units'/'prepare'/source['id']
    if state(prepared) != 'COMPLETED':
        raise ValueError('DEPENDENCY_FAILED: source preparation did not complete')
    mapping = read_json(prepared/'edit_truth.json')  # Editor only; never passed to native.
    off = unit.call('decode/OFF/G1', lambda: media.decode(media_path(output, source, 'OFF', 'G1'), SHAPE))
    quality_rows = []
    def quality(stage, left, right):
        value = unit.call('quality/'+stage, lambda: measure(left, right))
        quality_rows.append(dict(stage=stage, **value))
        write_json(unit.folder/'quality.json', quality_rows)
    wm = None
    if source['split'] == 'TEST':
        stored = read_json(prepared/'source.json')
        raw = unit.call('normalize', lambda: media.normalize(Path(stored['avi']), stored['metadata']))
        if list(raw.shape) != SHAPE or raw.dtype != np.uint8:
            raise ValueError('Invalid normalized writer input')
        message = read_json(output/'queries.json')[f"{method}/{source['id']}"]['query']
        if method == 'wam':
            marked = np.empty_like(raw)
            for start in range(0, 181, 8):
                marked[start:start+8] = unit.call(f'embed/{start}',
                    lambda s=start: adapter.embed(raw[s:s+8], message, [80, 128, 240, 384]))
        else:
            marked = unit.call('embed/0', lambda: adapter.embed(raw, message, [80, 128, 240, 384]))
        del message
        quality('precodec', raw, marked)
        wm = encode(unit, media, marked, output, source, method, 'G1', decode=True)
        quality('G1', off, wm)
        del raw, marked
    scheduled = tasks(source, method)
    for condition, length in CONDITIONS.items():
        indices = mapping['conditions'][condition]['indices']
        off_g1 = off[indices]
        off_g2 = unit.call('decode/OFF/'+condition,
                          lambda c=condition, n=length: media.decode(media_path(output, source, 'OFF', c), [n, 320, 512, 3]))
        wm_g1 = wm_g2 = None
        if wm is not None:
            wm_g1 = wm[indices]
            wm_g2 = encode(unit, media, wm_g1, output, source, method, condition, decode=True)
            quality(condition, off_g2, wm_g2)
        for task in scheduled:
            if task['condition'] != condition:
                continue
            pixels = {('OFF', CHANNELS[0]): off_g1, ('OFF', CHANNELS[1]): off_g2,
                      ('WM', CHANNELS[0]): wm_g1, ('WM', CHANNELS[1]): wm_g2}[task['arm'], task['channel']]
            blind_read(adapter, pixels, task, unit)
        del off_g1, off_g2, wm_g1, wm_g2, pixels
        write_json(output/'progress.json', dict(at=now(), source=source['id'], method=method,
                   condition=condition, state='NATIVE_VIEW_SAVED', split=source['split']))


def run_shard(output, sources, adapter_factory, media, measure):
    output = Path(output)
    for source in sources:
        prepare(output, source, media)
    for method in METHODS:
        adapter = None
        try:
            for source in sources:
                folder = output/'units'/method/source['id']
                if state(folder) != 'NEVER_STARTED':
                    continue
                unit = Unit(folder, operations(source, method)); error = None
                before = {}; counts = {}; resource = None
                try:
                    if state(output/'units'/'prepare'/source['id']) != 'COMPLETED':
                        raise ValueError('DEPENDENCY_FAILED: source preparation unavailable')
                    if adapter is None:
                        adapter = adapter_factory(method)
                        details = unit.call('load', adapter.load)
                        write_json(folder/'model.json', details)
                    else:
                        unit.reuse()
                    before = dict(adapter.counts)
                    run_source_method(output, source, method, unit, adapter, media, measure)
                except BaseException as exc:
                    error = exc
                finally:
                    if adapter is not None:
                        counts = {k: v-before.get(k, 0) for k, v in adapter.counts.items()}
                        try:
                            resource = adapter.resources()
                        except Exception as e:
                            resource = {'error': str(e)}
                    unit.close(error, internal_forwards=counts, resources=resource)
                if error is not None and adapter is not None:
                    adapter.close(); adapter = None
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise error
                print(source['split'], source['ordinal'], source['id'], method, state(folder), flush=True)
        finally:
            if adapter is not None:
                adapter.close()


def physical_receipt(output, sources):
    output = Path(output); units = Counter(); calls = {}; forwards = {m: Counter() for m in METHODS}; errors = []
    unexpected = []
    for source in sources:
        for kind in ['prepare', *METHODS]:
            folder = output/'units'/kind/source['id']; unit_state = state(folder); units[unit_state] += 1
            planned = operations(source, None if kind == 'prepare' else kind)
            fallback = 'NOT_STARTED' if unit_state == 'NEVER_STARTED' else 'UNKNOWN_UNIT_LEDGER'
            merged = {r['id']: dict(r, status=fallback) for r in planned}
            try:
                ledger = unit_ledger(folder)
            except (OSError, ValueError, KeyError) as error:
                errors.append(dict(source=source['id'], unit=kind, error=str(error)))
                ledger = {}
            for key, row in ledger.items():
                if key in merged:
                    merged[key].update(row)
                else:
                    unexpected.append(dict(source=source['id'], unit=kind, operation=row))
            missing = sum(r['status'] == 'UNKNOWN_UNIT_LEDGER' for r in merged.values())
            if missing:
                errors.append(dict(source=source['id'], unit=kind, missing_call_slots=missing,
                                   error='Consumed unit has unreadable/missing fixed call records'))
            for row in merged.values():
                key = kind+'/'+row['kind']
                calls.setdefault(key, Counter())[row['status']] += 1
            if kind in METHODS and (folder/'state.json').exists():
                try:
                    forwards[kind].update(read_json(folder/'state.json').get('internal_forwards', {}))
                except (OSError, ValueError):
                    pass
    return dict(unit_status=dict(units), physical_calls={k: dict(v) for k, v in calls.items()},
                actual_forward_hooks={k: dict(v) for k, v in forwards.items()},
                physical_accounting_errors=errors,
                unexpected_saved_operations=unexpected,
                hook_limit='Only saved completed/failed-unit hook counters; interrupted counters may be missing')


def run(config, sources, output, vendor, *, adapter_factory=None, media=None, measure=None):
    from runtime.wan.native_baseline_models_v3 import NativeAdapterV3
    from runtime.wan.native_baseline_media_v3 import MediaV3
    from runtime.wan.native_baseline_media_v1 import quality
    from runtime.wan.native_baseline_evaluation_v3 import calibration, final_evaluation
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    # A local advisory lock prevents concurrent workers on the same mounted folder;
    # this is not a distributed Drive locking service. Use one notebook instance.
    import fcntl
    with (output/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config, sources = initialize(output, config, sources)
        adapter_factory = adapter_factory or (lambda m: NativeAdapterV3(m, vendor))
        media = media or MediaV3(); measure = measure or quality
        started = time.monotonic()
        for split in ('CAL', 'TEST'):
            selected = [s for s in sources if s['split'] == split]
            for start in range(0, len(selected), 10):
                run_shard(output, selected[start:start+10], adapter_factory, media, measure)
            if split == 'CAL':
                calibration(output, sources)  # Each method closes tau/status before any TEST statistics.
        result = final_evaluation(output, sources)
        receipt = dict(status='PHYSICAL_PLAN_CLOSED', scientific_pass=None, elapsed_seconds=time.monotonic()-started,
                       **physical_receipt(output, sources), verification={m: result['calibration'][m]['status'] for m in METHODS})
        write_json(output/'result.json', receipt)
        return receipt
