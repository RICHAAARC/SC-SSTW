"""One adopted four-source DCT2/M3/CDC32 USER-run attempt.

Import is stdlib only. Physical failures spend their planned unit; another Run
all reads existing records, including interrupted and missing results.
"""
from collections import Counter
from copy import deepcopy
import errno
import json
import math
import os
from pathlib import Path
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
STEM = 'b_terminal_three_group_v1'
CONFIG = ROOT/'experiments/wan_state_clock/configs'/f'{STEM}.json'
DEFAULT_OUTPUT = '/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1'
DEFAULT_BACKGROUND = str(ROOT/'experiments/wan_state_clock/configs'/f'{STEM}_background.json')
KEYS = {'key00': 'watermark', 'key01': 'watermark-wrong'}
ARMS = ('OFF', 'P0', 'PM')
WINDOWS = {'L': 0, 'M': 53, 'R': 106}
FIXTURES = {'CROP': (0, None), 'DELETE_LEFT_GAP': (8, 49), 'DELETE_RIGHT_GAP': (8, 102)}
H = tuple((p, s) for p in range(4) for s in range(1 if p == 0 else 2, 36))
READERS = ('CDC_E', 'CDC_M', 'UNION_M', 'ALIAS_M_X10', 'REVERSE_ORDER')


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False)
        stream.flush()
        try:
            os.fsync(stream.fileno())
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP):
                raise
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def execution_artifacts(output):
    output = Path(output)
    names = ('attempt.json', 'result.json', 'ledger.json', 'protocol.json', 'thresholds.json',
             'planned_outputs.json', 'path_domain.json', 'background.json', 'blind_complete.json', 'evaluation.json')
    found = [n+s for n in names for s in ('', '.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for name in ('private', 'receipts', 'media', 'observations', 'families', 'blind', 'bundles', 'quality'):
        folder = output/name
        if folder.is_dir():
            found.extend(str(p.relative_to(output)) for p in folder.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def recover_ledger(output):
    """Read-only receipt overlay; also embedded verbatim in the notebook.

    STARTED remains spent and incomplete. Invalid or uncommitted receipts
    remain unresolved, never an inferred successful or unentered call.
    """
    from collections import Counter
    import json
    from pathlib import Path
    output = Path(output)
    ledger, errors, receipt_slots = {}, {}, []
    try:
        ledger = json.loads((output/'ledger.json').read_text(encoding='utf-8'))
        if not isinstance(ledger, dict) or any(not isinstance(row, dict) for row in ledger.values()):
            raise ValueError('ledger must map slots to records')
    except (OSError, ValueError) as error:
        ledger = {}
        errors['ledger.json'] = str(error)
    directory = output/'receipts'
    paths = sorted(directory.rglob('*.json')) + sorted(directory.rglob('*.json.tmp'))
    for path in paths:
        relative = path.relative_to(directory).as_posix()
        pending = relative.endswith('.tmp')
        slot = relative[:-9] if pending else relative[:-5]
        previous = ledger.get(slot, {})
        try:
            receipt = json.loads(path.read_text(encoding='utf-8'))
            if pending:
                raise ValueError('uncommitted receipt; completion is unresolved')
            if not isinstance(receipt, dict) or receipt.get('slot') != slot:
                raise ValueError('receipt slot does not match its address')
            status, attempts = receipt.get('status'), receipt.get('attempts')
            if status not in ('NOT_ENTERED', 'STARTED', 'RETURNED', 'FAILED', 'INTERRUPTED'):
                raise ValueError('invalid receipt status')
            if type(attempts) is not int or attempts != (0 if status == 'NOT_ENTERED' else 1):
                raise ValueError('invalid receipt attempt count')
            if not isinstance(receipt.get('kind'), str) or not receipt['kind']:
                raise ValueError('receipt kind required')
            if previous.get('kind') is not None and previous['kind'] != receipt['kind']:
                raise ValueError('receipt kind disagrees with saved slot')
            if previous.get('attempts') == 1 and attempts == 0:
                raise ValueError('receipt cannot erase a spent call')
            ledger[slot] = receipt
            receipt_slots.append(slot)
        except (OSError, ValueError) as error:
            name = 'receipts/'+relative
            errors[name] = str(error)
            ledger[slot] = dict(previous, slot=slot, status='UNRESOLVED_RECEIPT',
                                attempts=1 if previous.get('attempts') == 1 else None,
                                saved_status=previous.get('status'), recovery_error=name)
    counts = dict(Counter(row.get('status', 'UNRESOLVED_LEDGER') for row in ledger.values()))
    kinds = sorted({row.get('kind', 'UNKNOWN') for row in ledger.values()})
    unit_counts = {kind: dict(Counter(row.get('status', 'UNRESOLVED_LEDGER') for row in ledger.values()
                                     if row.get('kind', 'UNKNOWN') == kind)) for kind in kinds}
    return dict(ledger=ledger, counts=counts, unit_counts=unit_counts,
                receipt_slots=receipt_slots, read_errors=errors, physical_reentry=False)


def existing(output):
    saved, errors = {}, {}
    for name in ('result.json', 'thresholds.json', 'evaluation.json', 'planned_outputs.json'):
        path = Path(output)/name
        if path.exists():
            try:
                saved[name] = read(path)
            except (OSError, ValueError) as error:
                errors[name] = str(error)
    recovery = recover_ledger(output)
    saved['ledger.json'] = recovery.pop('ledger')
    errors.update(recovery['read_errors'])
    return dict(status='READ_ONLY_EXISTING_ATTEMPT', physical_reentry=False,
                saved=saved, ledger_recovery=recovery, read_errors=errors,
                artifacts=execution_artifacts(output))


def load_config(path=CONFIG):
    cfg = read(path)
    expected = [('b2-m3-c01', 'cal', 2026102441, 2), ('b2-m3-c02', 'cal', 2026102442, 11),
                ('b2-m3-e01', 'eval', 2026102443, 17), ('b2-m3-e02', 'eval', 2026102444, 6)]
    if [(h['id'], h['role'], h['seed'], h['a']) for h in cfg['sources']] != expected:
        raise ValueError('adopted four source/role/seed/origin tuples required')
    if any(not h['prompt'] or h['negative_prompt'] != '' for h in cfg['sources']):
        raise ValueError('fixed prompts and empty negative prompts required')
    if cfg['generation'] != dict(height=320, width=512, frames=181, fps=8, steps=50,
                                 guidance_scale=5.0, max_sequence_length=512):
        raise ValueError('adopted native50 and media geometry required')
    expected_bits = [int(b) for byte in bytes.fromhex('A6D39C5E') for b in f'{byte:08b}']
    if cfg['writer'] != dict(key=KEYS['key00'], bits=expected_bits, gamma=1):
        raise ValueError('adopted fixed writer bits/key/gamma required')
    return cfg


def load_background(path):
    value = read(path)
    if value.get('status') != 'BACKGROUND_FROZEN':
        raise ValueError('saved usable background required')
    for key in KEYS:
        m = value['models'][key]
        mean, factor = m['mean'], m['factor']
        if len(mean) != 40 or len(factor) != 40 or any(len(r) != 40 for r in factor):
            raise ValueError('background mean40 / lower factor40x40 required')
        if not all(type(v) in (float, int) and math.isfinite(v) for v in mean+[v for r in factor for v in r]):
            raise ValueError('nonfinite background')
        if any(factor[i][i] <= 0 or any(factor[i][j] != 0 for j in range(i+1, 40)) for i in range(40)):
            raise ValueError('positive lower triangular background factor required')
    return value


def source_frames(host, fixture):
    d, seam = FIXTURES[fixture]
    return [host['a']+x+(d if seam is not None and x >= seam else 0) for x in range(151)]


def task_id(h, arm, fixture, key):
    return '/'.join((h, arm, fixture, key))


def planned_units(cfg):
    units = dict(background='background_load', calibration='tau_once', post='post_evaluation')
    for host in cfg['sources']:
        h, evaluation = host['id'], host['role'] == 'eval'
        for key in KEYS:
            units[f'{h}/book/{key}'] = 'codebook'
        units[f'{h}/generation_load'] = 'generation_load'
        units[f'{h}/vae_load'] = 'vae_load'
        for arm in ('OFF', 'P0'):
            units[f'{h}/generate/{arm}'] = 'native50_trajectory'
        for arm in ('P0', 'PM'):
            units[f'{h}/native/{arm}/Q'] = 'native_key00_Q'
            if evaluation:
                units[f'{h}/native/{arm}/DCT'] = 'native_DCT'
                for key in KEYS:
                    units[f'{h}/native/{arm}/{key}/bits'] = 'native_local_bits'
        units[f'{h}/construct'] = 'state_construct'
        units[f'{h}/native_audit'] = 'native_state_audit'
        for arm in ARMS:
            base = f'{h}/{arm}'
            for op, kind in [('D', 'D181'), ('preclip', 'preclip'), ('quantize', 'quantize'),
                             ('encode', 'codec_encode'), ('probe', 'codec_probe'), ('decode', 'codec_decode')]:
                units[f'{base}/full/{op}'] = kind
            for fixture in FIXTURES:
                stem = f'{base}/{fixture}'
                for op, kind in [('edit', 'fixture151'), ('encode', 'codec_encode'),
                                 ('probe', 'codec_probe'), ('decode', 'codec_decode')]:
                    units[f'{stem}/{op}'] = kind
                for window in WINDOWS:
                    for phase in range(4):
                        view = f'{stem}/{window}/p{phase}'
                        units[view+'/E'] = 'E41'
                        for key in KEYS:
                            units[view+'/'+key+'/Q'] = 'E41_Q'
                        if evaluation:
                            units[view+'/DCT'] = 'E41_DCT'
                            for key in KEYS:
                                units[view+'/'+key+'/bits'] = 'local_and_Rc_bits'
                    for key in KEYS:
                        units[f'{stem}/{window}/{key}/score'] = 'H137_score'
                if evaluation:
                    for key in KEYS:
                        q = task_id(h, arm, fixture, key)
                        units[q+'/plan'] = 'eval_local_and_path_plan'
                        units[q+'/read'] = 'eval_local_and_aggregate_read'
        for pair in ('OFF_to_P0', 'P0_to_PM'):
            for scope, layers in [('full', ('preclip', 'rgb8', 'readback'))]+[(f, ('rgb8', 'readback')) for f in FIXTURES]:
                for layer in layers:
                    units[f'{h}/quality/{scope}/{pair}/{layer}'] = 'quality_pair'
    for i in range(72):
        units[f'bundle/{i:02}'] = 'bundle_independent_plan_read'
    return units


def denominators():
    return dict(sources=4, full_media=12, received151=36, trajectories=8, native_steps=400,
                dit_forwards=800, local_backward=100, state_construct=4, D181=12, E41=432,
                codec_each_direction=48, state_Q=864, native_key00_Q=8, H_families=216,
                H_slots=29592, calibration_null_families=90, eval_tasks=36,
                path_classes_each_method=34596, path_aliases_each_method=302436,
                local_CDC_slots=3456, each_independent_region_slots=3456,
                each_aggregate_slots=1152, bundles=72, bundle_SEP_slots=4608,
                bundle_JOIN_slots=2304, quality_pairs=72, quality_frames=11592,
                quality_adjacent=11520, media_P0_PM_bit_pairs=2880, native_P0_PM_bit_pairs=1408)


class Journal:
    def __init__(self, output, cfg, backend):
        self.output, self.backend = Path(output), backend
        self.data = {k: dict(slot=k, kind=v, status='NOT_ENTERED', attempts=0) for k, v in planned_units(cfg).items()}
        self.events = {}
        self.save()

    def save(self):
        dump(self.output/'ledger.json', self.data)

    def event(self, host, kind, returned):
        item = self.events.setdefault(host+'/'+kind, dict(started=0, returned=0))
        item['returned' if returned else 'started'] += 1
        dump(self.output/'physical_events.json', self.events)

    def call(self, slot, function, deps=()):
        row = self.data[slot]
        if row['status'] != 'NOT_ENTERED':
            raise RuntimeError('planned unit already spent: '+slot)
        row['dependencies'] = list(deps)
        bad = [d for d in deps if self.data[d]['status'] != 'RETURNED']
        if bad:
            row['unavailable_from'] = bad
            dump(self.output/'receipts'/(slot+'.json'), row)
            return None
        row.update(status='STARTED', attempts=1, started_unix=time.time())
        dump(self.output/'receipts'/(slot+'.json'), row)
        dump(self.output/'progress.json', dict(active=slot, counts=dict(Counter(r['status'] for r in self.data.values()))))
        before = dict(getattr(self.backend, 'blocks', {}))
        try:
            value = function()
            row['status'] = 'RETURNED'
            return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error, Exception) else 'INTERRUPTED',
                       error=f'{type(error).__name__}: {error}')
            if not isinstance(error, Exception):
                raise
        finally:
            row['finished_unix'] = time.time()
            after = dict(getattr(self.backend, 'blocks', {}))
            row['observed_blocks'] = {k: after[k]-before[k] for k in before
                                      if type(before[k]) is int and type(after.get(k)) is int}
            dump(self.output/'receipts'/(slot+'.json'), row)

    def record(self, slot, function, path, deps=()):
        def operation():
            value = function()
            dump(path, value)
            return value
        return self.call(slot, operation, deps)


def empty_family(reason):
    return dict(status=reason, candidates=[dict(phase=p, source_start=s, S=None, score_status=reason) for p, s in H])


def empty_slots(reason='NOT_ENTERED'):
    return dict(status=reason, planned_slots=32, slots=[dict(slot=i, value=None, status=reason,
                cause_set=[reason], selected_refs=[], conflict_refs=[], missing_refs=[]) for i in range(32)])


def flatten_cache(decoded, target):
    """Per-unit production values only. Rc never enters CDC cache."""
    key = decoded['key']
    for region in ('R0', 'R1'):
        for row in decoded['regions'][region]['local_cells']:
            for bit in row['bits']:
                target.setdefault(row['evidence_id'], {}).setdefault(key, {})[str(bit['local_bit'])] = bit


def view_metadata(input_id, query_id, phase, w, view_id, status):
    return dict(input_id=input_id, query_id=query_id, view_id=view_id, phase=phase,
                received_query_start=w, encoder_input_support=list(range(w+phase, w+phase+41)),
                encoder_context=dict(reset=True, posterior_mode=True, preprocessing='uint8 / 255 once',
                                     spatial_input=[320, 512]), status=status)


def collect_host(host, cfg, out, backend, journal, background):
    h = host['id']
    evaluation = host['role'] == 'eval'
    base = out/'private'/h
    books, terminals, contexts = {}, {}, {}
    def tensor(path, value):
        backend.save_tensor(path, value)
        return value
    for key, secret in KEYS.items():
        books[key] = journal.call(f'{h}/book/{key}', lambda secret=secret: backend.book(secret), ['background'])
    journal.record(h+'/generation_load', lambda: backend.generation_load(host, cfg), base/'generation_load.json', ['background'])
    try:
        for arm in ('OFF', 'P0'):
            def generate(arm=arm):
                traces = []
                def record(row):
                    traces.append(row)
                    dump(base/(arm+'_trace.json'), traces)
                z, receipt = backend.generate(arm, cfg['writer'], lambda k, done: journal.event(h, k, done), record)
                dump(base/(arm+'_generation.json'), receipt)
                return tensor(base/(arm+'_terminal.pt'), z)
            terminals[arm] = journal.call(f'{h}/generate/{arm}', generate, [h+'/generation_load'])
    finally:
        backend.release_generation()
    pre = journal.record(h+'/native/P0/Q', lambda: backend.native_observe(terminals['P0'], books['key00']),
                         base/'native/P0_Q.json', [h+'/generate/P0', h+'/book/key00'])
    def construct():
        z, receipt = backend.construct(terminals['P0'], books['key00'], pre)
        dump(base/'construction.json', receipt)
        return tensor(base/'PM_terminal.pt', z)
    terminals['PM'] = journal.call(h+'/construct', construct, [h+'/native/P0/Q'])
    post = journal.record(h+'/native/PM/Q', lambda: backend.native_observe(terminals['PM'], books['key00']),
                          base/'native/PM_Q.json', [h+'/construct', h+'/book/key00'])
    journal.record(h+'/native_audit', lambda: backend.native_audit(terminals['P0'], terminals['PM'], post),
                   base/'native/audit.json', [h+'/native/PM/Q'])
    journal.record(h+'/vae_load', backend.load_vae, base/'vae.json', [h+'/generation_load'])
    try:
        for arm in ARMS:
            stem = f'{h}/{arm}/full'
            media = out/'media'/h/arm/'full'
            def decode(arm=arm):
                value = backend.decode(terminals[arm])
                tensor(media/'preclip.pt', value['preclip'])
                tensor(media/'float_rgb.pt', value['float_rgb'])
                return value
            decoded = journal.call(stem+'/D', decode, [h+'/vae_load', h+'/construct' if arm == 'PM' else h+'/generate/'+arm])
            journal.record(stem+'/preclip', lambda: backend.preclip(decoded['preclip']), media/'preclip.json', [stem+'/D'])
            rgb = journal.call(stem+'/quantize', lambda: tensor(media/'rgb8.pt', backend.quantize(decoded['float_rgb'])), [stem+'/D'])
            journal.record(stem+'/encode', lambda: backend.save_mp4(rgb, media/'video.mp4'), media/'codec.json', [stem+'/quantize'])
            probe = journal.record(stem+'/probe', lambda: backend.probe(media/'video.mp4'), media/'probe.json', [stem+'/encode'])
            pixels = journal.call(stem+'/decode', lambda: tensor(media/'readback.pt', backend.read_media(media/'video.mp4', probe, 181)), [stem+'/probe'])
            del decoded, rgb
            for fixture in FIXTURES:
                prefix = f'{h}/{arm}/{fixture}'
                dest = out/'media'/h/arm/fixture
                edited = journal.call(prefix+'/edit', lambda: tensor(dest/'rgb8.pt', backend.edit(pixels, source_frames(host, fixture))), [stem+'/decode'])
                journal.record(prefix+'/encode', lambda: backend.save_mp4(edited, dest/'video.mp4'), dest/'codec.json', [prefix+'/edit'])
                probe = journal.record(prefix+'/probe', lambda: backend.probe(dest/'video.mp4'), dest/'probe.json', [prefix+'/encode'])
                received = journal.call(prefix+'/decode', lambda: tensor(dest/'readback.pt', backend.read_media(dest/'video.mp4', probe, 151)), [prefix+'/probe'])
                del edited
                context = dict(input_id=uuid.uuid4().hex, queries={}, cache={}, families={})
                for window, w in WINDOWS.items():
                    query_id = uuid.uuid4().hex
                    views, banks = [], {key: {} for key in KEYS}
                    for phase in range(4):
                        v = prefix+f'/{window}/p{phase}'
                        view_id = uuid.uuid4().hex
                        z = journal.call(v+'/E', lambda: tensor(out/'observations'/v/'E41.pt', backend.encode_view(backend.prepare_window(received, w, phase))), [prefix+'/decode', h+'/vae_load'])
                        for key in KEYS:
                            banks[key][str(phase)] = journal.record(v+'/'+key+'/Q', lambda key=key: backend.observe(z, books[key]), out/'observations'/v/key/'Q.json', [v+'/E', h+'/book/'+key])
                        view_status = journal.data[v+'/E']['status']
                        if evaluation:
                            projected = journal.call(v+'/DCT', lambda: tensor(out/'observations'/v/'DCT.pt', backend.payload_project(z, list(range(1, 11)), ('R0', 'R1', 'Rc'))), [v+'/E'])
                            if journal.data[v+'/DCT']['status'] != 'RETURNED':
                                view_status = journal.data[v+'/DCT']['status']
                            for key, secret in KEYS.items():
                                decoded_bits = journal.record(v+'/'+key+'/bits', lambda key=key, secret=secret: backend.payload_decode(projected, secret, view_id, key), out/'observations'/v/key/'bits.json', [v+'/DCT'])
                                if decoded_bits is not None:
                                    flatten_cache(decoded_bits, context['cache'])
                            del projected
                        views.append(view_metadata(context['input_id'], query_id, phase, w, view_id, view_status))
                        del z
                    context['queries'][window] = dict(query_id=query_id, views=views)
                    for key in KEYS:
                        score_slot = f'{prefix}/{window}/{key}/score'
                        family = journal.record(score_slot, lambda key=key: backend.score(banks[key], background['models'][key]), out/'families'/prefix/window/(key+'.json'), ['background'])
                        family = family or empty_family('TECHNICAL_UNRESOLVED')
                        family.update(input_id=context['input_id'], key_id=key)
                        context['families'][window, key] = family
                contexts[arm, fixture] = context
                del received
            del pixels
    finally:
        backend.release_vae()
        journal.save()
    return contexts


def plan_input(context, key, threshold):
    from main.tube_state import b_m3_time_path_v1 as time_method
    from main.tube_state import b_cdc32_aggregation_v1 as aggregate
    locals_, regions = {}, {r: {} for r in ('R0', 'R1', 'CDC')}
    for window, w in WINDOWS.items():
        q = context['queries'][window]
        local = time_method.prepare_local(context['families'][window, key], threshold, q['query_id'], w)
        locals_[window] = local
        for r, required in [('R0', ('R0',)), ('R1', ('R1',)), ('CDC', ('R0', 'R1'))]:
            regions[r][window] = aggregate.plan_local(local, q['views'], required_regions=required)
    paths = {mode: time_method.paths(locals_, mode) for mode in ('E2', 'M3')}
    return dict(locals=locals_, regions=regions, paths=paths,
                comparisons=aggregate.plan_comparisons(regions['CDC']), message_used=False)


def read_input(plan, cache):
    from main.tube_state import b_cdc32_aggregation_v1 as method
    independent = {r: {w: method.read_local(p, cache) for w, p in plans.items()}
                   for r, plans in plan['regions'].items()}
    contexts = {mode: dict(mode=mode, status=p['status'], compatible_path_ids=p['compatible_path_ids'],
                           domain_count=p['domain_count'], compatible_class_count=p['compatible_class_count'],
                           record_ref='plan.json#paths/'+mode)
                for mode, p in plan['paths'].items()}
    comparisons = method.read_comparisons(plan['comparisons'], cache, path_context=contexts)
    return dict(status='RETURNED', local=independent, aggregates=comparisons)


def endpoint_cache(cache, views, key):
    """Select production addresses only; do not read or compare bit values."""
    restricted = {}
    for view in views:
        for row in range(1, 11):
            for region in ('R0', 'R1'):
                eid = f"{view['view_id']}/row{row}/{region}"
                keyed = cache.get(eid, {})
                if key in keyed:
                    # Keep the lazy bit mapping untouched until the bundle plan
                    # has been saved and read back. No parent mapping escapes.
                    restricted[eid] = {key: keyed[key]}
    return restricted


def bundle_requests(cfg, contexts, threshold):
    """Only raw local families/views/caches enter each new bundle boundary."""
    from main.tube_state import b_m3_time_path_v1 as time_method
    from runtime.wan.b_cdc32_bundle_v1 import RawFragment
    sources = [h for h in cfg['sources'] if h['role'] == 'eval']
    for fixture in FIXTURES:
        for arm in ARMS:
            for key in KEYS:
                for a in sources:
                    for b in sources:
                        fragments = []
                        for host, window in ((a, 'L'), (b, 'R')):
                            c = contexts.get(host['id'], {}).get((arm, fixture))
                            if c is None:
                                fragments = None
                                break
                            q = c['queries'][window]
                            local = time_method.prepare_local(c['families'][window, key], threshold, q['query_id'], WINDOWS[window])
                            cache = endpoint_cache(c['cache'], q['views'], key)
                            fragments.append(RawFragment(local, q['views'], cache))
                        yield fragments, dict(sources=[a['id'], b['id']], arm=arm, fixture=fixture,
                                              key=key, same_source=a['id'] == b['id'])


def planned_outputs(cfg):
    tasks = {}
    for h in cfg['sources']:
        if h['role'] != 'eval':
            continue
        for arm in ARMS:
            for fixture in FIXTURES:
                for key in KEYS:
                    tasks[task_id(h['id'], arm, fixture, key)] = dict(
                        status='NOT_ENTERED', path_classes_each=961, path_aliases_each=8401,
                        paths={m: dict(status='NOT_ENTERED', all_members_status='NOT_ENTERED',
                                       domain_ref='path_domain.json', planned=961, returned=0, not_entered=961)
                               for m in ('E2', 'M3')},
                        local={r: {w: empty_slots() for w in WINDOWS} for r in ('R0', 'R1', 'CDC')},
                        aggregates={name: empty_slots() for name in READERS})
    return dict(denominators=denominators(), tasks=tasks,
                bundles=[dict(index=i, status='NOT_ENTERED', outputs={n: empty_slots() for n in ('SEP_A', 'SEP_B', 'JOIN_CONDITIONAL')}) for i in range(72)])


def run(output=None, *, config_path=CONFIG, background_path=DEFAULT_BACKGROUND, backend=None):
    out = Path(output or DEFAULT_OUTPUT)
    if execution_artifacts(out):
        return existing(out)
    cfg = load_config(config_path)
    out.mkdir(parents=True, exist_ok=True)
    try:
        with (out/'attempt.json').open('x') as stream:
            json.dump(dict(started_unix=time.time(), failure_consumes_attempt=True, scope=STEM), stream)
            stream.flush()
    except FileExistsError:
        return existing(out)
    journal = Journal(out, cfg, backend)
    outputs = planned_outputs(cfg)
    dump(out/'planned_outputs.json', outputs)
    dump(out/'protocol.json', dict(config=cfg, denominators=denominators(), scientific_pass=None))
    result = dict(status='RUNNING', denominators=denominators(), scientific_pass=None, new_physical_run_entered=False)
    threshold = dict(status='CALIBRATION_UNRESOLVED', tau=None)
    contexts, decisions = {}, {}
    try:
        from main.tube_state import b_m3_time_path_v1 as time_method
        from runtime.wan import b_cdc32_bundle_v1 as bundles
        dump(out/'path_domain.json', dict(classes=time_method.path_domain(), class_count=961,
                                         seam_hypothesis_count=8401, public_domain_only=True))
        background = journal.record('background', lambda: load_background(background_path), out/'background.json')
        if backend is None and background is not None:
            from runtime.wan.b_terminal_three_group_backend_v1 import WanBackend
            backend = WanBackend(cfg['model_id'])
            journal.backend = backend
        if background is not None:
            for host in cfg['sources']:
                if host['role'] == 'cal':
                    contexts[host['id']] = collect_host(host, cfg, out, backend, journal, background)
            nulls = {}
            for host in cfg['sources']:
                if host['role'] != 'cal':
                    continue
                for arm in ARMS:
                    for fixture in FIXTURES:
                        for key in KEYS:
                            if arm == 'PM' and key == 'key00':
                                continue
                            for window in WINDOWS:
                                c = contexts.get(host['id'], {}).get((arm, fixture))
                                nulls[uuid.uuid4().hex] = c['families'][window, key] if c else empty_family('NOT_ENTERED')
            threshold = journal.record('calibration', lambda: time_method.calibrate(nulls), out/'thresholds.json', ['background']) or threshold
        dump(out/'thresholds.json', threshold)
        if threshold['status'] == 'FROZEN':
            for host in cfg['sources']:
                if host['role'] == 'eval':
                    contexts[host['id']] = collect_host(host, cfg, out, backend, journal, background)
            for host in cfg['sources']:
                if host['role'] != 'eval':
                    continue
                for arm in ARMS:
                    for fixture in FIXTURES:
                        c = contexts[host['id']][arm, fixture]
                        for key in KEYS:
                            q = task_id(host['id'], arm, fixture, key)
                            directory = out/'blind'/q
                            plan = journal.record(q+'/plan', lambda: plan_input(c, key, threshold), directory/'plan.json')
                            value = journal.record(q+'/read', lambda: read_input(read(directory/'plan.json'), c['cache']), directory/'decision.json', [q+'/plan'])
                            if value is None:
                                value = deepcopy(outputs['tasks'][q])
                                value['status'] = 'TECHNICAL_UNRESOLVED'
                                dump(directory/'decision.json', value)
                            decisions[q] = dict(file=str(directory/'decision.json'), status=value['status'])
                            outputs['tasks'][q] = dict(status=journal.data[q+'/read']['status'], decision=str(directory/'decision.json'),
                                                       path_classes_each=961, path_aliases_each=8401)
            for i, (fragments, private) in enumerate(bundle_requests(cfg, contexts, threshold)):
                dump(out/'private/bundles'/(f'{i:02}.json'), private)
                value = journal.record(f'bundle/{i:02}', lambda: bundles.run_bundle(fragments, out/'bundles'/f'{i:02}')
                                       if fragments else bundles._failure('INPUT_NOT_ENTERED'),
                                       out/'bundles'/f'{i:02}'/'result.json')
                outputs['bundles'][i] = value or dict(status='TECHNICAL_UNRESOLVED', outputs={n: empty_slots('TECHNICAL_UNRESOLVED') for n in ('SEP_A', 'SEP_B', 'JOIN_CONDITIONAL')})
            # The only transition to post: read back every saved blind decision
            # and successful plan, including failed/unresolved outputs.
            blind_files = list((out/'blind').rglob('*.json'))+list((out/'bundles').rglob('*.json'))
            for path in blind_files:
                read(path)
            dump(out/'blind_complete.json', dict(saved_unix=time.time(), eval_tasks=len(decisions), bundles=72,
                                                files_read_back=len(blind_files), truth_opened=False))
            def post():
                dump(out/'post_truth_access.json', dict(opened_unix=time.time(), blind_record=str(out/'blind_complete.json')))
                from runtime.wan.b_terminal_three_group_evaluation_v1 import evaluate
                return evaluate(cfg, out, backend, journal)
            journal.record('post', post, out/'evaluation.json')
        incomplete_families = sum(f.get('status') != 'SCORES_COMPLETE' for cs in contexts.values()
                                  for c in cs.values() for f in c['families'].values())
        complete = threshold['status'] == 'FROZEN' and all(r['status'] == 'RETURNED' for r in journal.data.values()) and not incomplete_families
        result.update(status='COMPLETED_ENGINEERING_RECORDS' if complete else 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED',
                      incomplete_H_families=incomplete_families)
    except BaseException as error:
        result.update(status='FAILED' if isinstance(error, Exception) else 'INTERRUPTED', error=f'{type(error).__name__}: {error}')
    finally:
        if threshold['status'] != 'FROZEN':
            eval_ids = [h['id']+'/' for h in cfg['sources'] if h['role']=='eval']
            for slot, row in journal.data.items():
                if row['status']=='NOT_ENTERED' and (slot=='post' or slot.startswith('bundle/') or any(slot.startswith(h) for h in eval_ids)):
                    row.setdefault('unavailable_from', ['calibration'])
            for row in outputs['tasks'].values():
                row['cause_set'] = ['CALIBRATION_UNRESOLVED']
            for row in outputs['bundles']:
                row['cause_set'] = ['CALIBRATION_UNRESOLVED']
        journal.save()
        dump(out/'planned_outputs.json', outputs)
        result.update(threshold_status=threshold['status'], counts=dict(Counter(r['status'] for r in journal.data.values())),
                      unit_counts={kind: dict(Counter(r['status'] for r in journal.data.values() if r['kind']==kind))
                                   for kind in sorted({r['kind'] for r in journal.data.values()})},
                      physical_events=journal.events,
                      new_physical_run_entered=any(r['kind']=='generation_load' and r['attempts'] for r in journal.data.values()))
        dump(out/'result.json', result)
    return result
