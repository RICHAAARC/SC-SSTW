"""Card94 once-only staged user execution. Importing this module loads no model/tensor runtime."""
from collections import Counter
from pathlib import Path
import errno
import json
import os
import time
from main.tube_state import b_fixed_detection_confidence_v1 as method
from main.tube_state import b_phase_centered_saved_v1 as arithmetic

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT/'experiments/wan_state_clock/configs/b_fixed_detection_confidence_v1.json'
BUDGET = ROOT/'experiments/wan_state_clock/configs/b_fixed_detection_confidence_v1_budget.json'
KEYS = {'key00': 'watermark', 'key01': 'watermark-wrong'}
DEFAULT_OUTPUT = '/content/drive/MyDrive/Video-WM/B-Fixed-Detection-Confidence-V1/fixed-2cal-2confirm-v1'


def flush(stream):
    stream.flush()
    try:
        os.fsync(stream.fileno())
    except OSError as error:
        if error.errno not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP}:
            raise


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with tmp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        flush(stream)
    tmp.replace(path)


def execution_artifacts(output):
    output = Path(output)
    names = ('attempt.json', 'result.json', 'checkpoint.json', 'ledger.json', 'thresholds.json', 'evaluation.json', 'protocol.json')
    found = [n+s for n in names for s in ('', '.tmp') if (output/(n+s)).exists() or (output/(n+s)).is_symlink()]
    for folder in ('hosts', 'receipts', 'markers', 'blind', 'private', 'decisions'):
        root = output/folder
        if root.is_dir():
            found.extend(str(p.relative_to(output)) for p in root.rglob('*') if p.is_file() or p.is_symlink())
    return sorted(found)


def existing_attempt(output, found=None):
    output = Path(output)
    result = dict(status='EXISTING_ATTEMPT_NOT_REEXECUTED', physical_reentry=False,
        artifacts=found if found is not None else execution_artifacts(output), saved={}, read_errors={})
    for name in ('result.json', 'checkpoint.json', 'ledger.json', 'thresholds.json', 'evaluation.json', 'protocol.json'):
        try:
            if (output/name).is_file():
                result['saved'][name] = json.loads((output/name).read_text())
        except (OSError, ValueError) as error:
            result['read_errors'][name] = f'{type(error).__name__}: {error}'
    # Process death can leave a STARTED receipt newer than the ledger snapshot.
    for path in (output/'receipts').rglob('*.json') if (output/'receipts').is_dir() else ():
        try:
            row = json.loads(path.read_text())
            result['saved'].setdefault('ledger.json', {})[row['slot']] = row
        except (OSError, ValueError, KeyError) as error:
            result['read_errors'][str(path.relative_to(output))] = f'{type(error).__name__}: {error}'
    return result


def load_config(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    ids = ['b93-c01', 'b93-c02', 'b93-v01', 'b93-v02']
    if [h['id'] for h in cfg['hosts']] != ids or [h['role'] for h in cfg['hosts']] != ['calibration']*2+['confirmation']*2:
        raise ValueError('fixed two calibration and two confirmation source roles required')
    if cfg['origins'] != list(method.ORIGINS) or cfg['writer']['key'] != KEYS['key00'] or cfg['writer']['gamma'] != 1.0:
        raise ValueError('adopted crop/key/gamma semantics required')
    if len(cfg['writer']['bits']) != 32 or any(type(x) is not int or x not in (0, 1) for x in cfg['writer']['bits']):
        raise ValueError('exact32 binary writer interface required')
    if cfg['generation'] != dict(height=320, width=512, frames=181, fps=8, steps=50, guidance_scale=5.0, max_sequence_length=512):
        raise ValueError('fixed native generation geometry and scheduler protocol required')
    if [h['seed'] for h in cfg['hosts']] != list(range(2026102101, 2026102105)) or any(not isinstance(h['prompt'], str) or not h['prompt'] or h['negative_prompt'] != '' for h in cfg['hosts']):
        raise ValueError('fixed source seed roster, nonempty prompts and empty negatives required')
    return cfg


def units(config):
    u = {'calibration': 'supervised_calibration', 'evaluation': 'post_confirmation_evaluation'}
    for host in config['hosts']:
        h = host['id']
        for k, kind in [('generation_load', 'generation_pipeline_load'), ('generate/OFF', 'source_50step_trajectory'),
                        ('generate/P0', 'source_50step_trajectory'), ('construct', 'PM_construct'), ('vae_load', 'VAE_load')]:
            u[f'{h}/{k}'] = kind
        for key in KEYS:
            u[f'{h}/book/{key}'] = 'public_codebook'
        for arm in method.ARMS:
            for label, kind in [('D', 'ordinary_D181'), ('preclip', 'preclip_record'), ('quantize', 'rgb8_quantize'),
                                ('encode_mp4', 'MP4_encode'), ('probe', 'MP4_probe'), ('decode_mp4', 'MP4_decode'), ('load_RGB', 'RGB_cache_load')]:
                u[f'{h}/{arm}/{label}'] = kind
            for oi in range(6):
                for p in range(4):
                    v = f'{h}/{arm}/r{oi}/p{p}'
                    for label, kind in [('prepare', 'E41_view_preparation'), ('E', 'E41'), ('FFT', 'receiver_FFT')]:
                        u[v+'/'+label] = kind
                    for key in KEYS:
                        for label in ('Q', 'score', 'votes', 'reduce'):
                            u[f'{v}/{key}/{label}'] = label
        for pair in ('OFF_to_P0', 'P0_to_PM'):
            for layer in ('preclip', 'rgb8', 'mp4'):
                u[f'{h}/quality/{pair}/{layer}'] = 'paired_quality'
        for qid in method.query_metadata():
            u[f'{h}/family/{qid}'] = 'family_binding'
            u[f'{h}/decision/{qid}'] = 'threshold_and_set_read'
    return u


class Journal:
    def __init__(self, output, config, backend):
        self.output, self.backend = Path(output), backend
        self.data = {k: dict(slot=k, kind=v, status='NOT_ENTERED') for k, v in units(config).items()}
        self.events = {}
        self.save()

    def save(self):
        dump(self.output/'ledger.json', self.data)

    def progress(self):
        dump(self.output/'progress.json', dict(returned=sum(x['status']=='RETURNED' for x in self.data.values()),
            planned=len(self.data), active=[k for k, x in self.data.items() if x['status']=='STARTED']))

    def event(self, host, kind, returned):
        key = host+'/'+kind
        row = self.events.setdefault(key, dict(started=0, returned=0))
        row['returned' if returned else 'started'] += 1
        dump(self.output/'physical_events.json', self.events)

    def call(self, slot, function, dependencies=()):
        row = self.data[slot]
        if row['status'] != 'NOT_ENTERED':
            raise RuntimeError('one attempt per fixed unit; cannot reenter '+slot)
        bad = [d for d in dependencies if self.data[d]['status'] != 'RETURNED']
        if bad:
            row['unavailable_from'] = bad
            return None
        row.update(status='STARTED', started_unix=time.time())
        path = self.output/'receipts'/(slot+'.json')
        dump(path, row)  # BEFORE model/IO/numeric function; consumes this slot on death.
        self.progress()
        before = dict(getattr(self.backend, 'blocks', {}))
        started = time.monotonic()
        try:
            value = function()
            row['status'] = 'RETURNED'
            return value
        except BaseException as error:
            row.update(status='FAILED' if isinstance(error, Exception) else 'INTERRUPTED', error=f'{type(error).__name__}: {error}')
            if not isinstance(error, Exception):
                raise
            return None
        finally:
            row.update(seconds=time.monotonic()-started, finished_unix=time.time())
            if row['kind'] in ('E41', 'ordinary_D181'):
                after = getattr(self.backend, 'blocks', {})
                row['internal_blocks'] = {k: None if v is None or after.get(k) is None else after[k]-v for k, v in before.items()}
            dump(path, row)
            self.progress()

    def counts(self):
        out = {}
        for row in self.data.values():
            bucket = out.setdefault(row['kind'], Counter())
            bucket['planned'] += 1
            bucket[row['status']] += 1
        return {k: dict(v) for k, v in out.items()}


def run_host(host, config, output, backend, journal):
    h = host['id']
    directory = output/'hosts'/h
    directory.mkdir(parents=True, exist_ok=True)
    counters, terminals, books = Counter(), {}, {}
    def call(name, function, deps=()):
        return journal.call(h+'/'+name, function, [h+'/'+d for d in deps])
    def record(name, function, rel, deps=()):
        def execute():
            value = function()
            dump(directory/rel, value)
            return value
        return call(name, execute, deps)
    def tensor(rel, value):
        backend.save_tensor(directory/rel, value)
        return value
    phase_tables, local_bits = {}, {}
    families = {qid: method.empty_family() for qid in method.query_metadata()}
    dump(directory/'preparation.json', dict(host=host, queries=method.query_metadata()))
    try:
        for key, secret in KEYS.items():
            book = call('book/'+key, lambda secret=secret: backend.book(secret))
            if book is not None:
                books[key] = book
        loaded = record('generation_load', lambda: backend.generation_load(host, config), 'private/generation_load.json')
        try:
            for arm in ('OFF', 'P0'):
                def generate(arm=arm):
                    traces = []
                    def step(row):
                        traces.append(row)
                        dump(directory/'private/traces'/(arm+'.json'), traces)
                    z, receipt = backend.generate(arm, config['writer'], lambda k, done: journal.event(h, k, done), step)
                    tensor('private/'+arm+'_terminal.pt', z)
                    dump(directory/'private'/(arm+'_generation.json'), receipt)
                    return z
                z = call('generate/'+arm, generate, ['generation_load'])
                if z is not None:
                    terminals[arm] = z
        finally:
            backend.release_generation()
        def construct():
            z, receipt = backend.construct(terminals['P0'], books['key00'], config['writer']['gamma'])
            tensor('private/PM_terminal.pt', z)
            dump(directory/'private/construction.json', receipt)
            return z
        pm = call('construct', construct, ['generate/P0', 'book/key00'])
        if pm is not None:
            terminals['PM'] = pm
        vae = record('vae_load', backend.load_vae, 'private/vae.json') if terminals else None
        try:
            # Materialize each arm once; no full181 E or latent reader.
            for arm in method.ARMS:
                def decode(arm=arm):
                    value = backend.decode(terminals[arm])
                    tensor('media/'+arm+'/preclip.pt', value['preclip'])
                    tensor('media/'+arm+'/float_rgb.pt', value['float_rgb'])
                    return value
                deps = ['vae_load', 'construct' if arm == 'PM' else 'generate/'+arm]
                decoded = call(arm+'/D', decode, deps)
                record(arm+'/preclip', lambda: backend.preclip(decoded['preclip']), 'quality/'+arm+'_preclip.json', [arm+'/D'])
                rgb = call(arm+'/quantize', lambda: tensor('media/'+arm+'/rgb8.pt', backend.quantize(decoded['float_rgb'])), [arm+'/D'])
                path = directory/'media'/arm/'video.mp4'
                record(arm+'/encode_mp4', lambda: backend.save_mp4(rgb, path), 'media/'+arm+'/codec.json', [arm+'/quantize'])
                probe = record(arm+'/probe', lambda: backend.probe(path), 'media/'+arm+'/probe.json', [arm+'/encode_mp4'])
                call(arm+'/decode_mp4', lambda: tensor('media/'+arm+'/readback.pt', backend.read_mp4(path, probe)), [arm+'/probe'])
                del decoded, rgb
                pixels = call(arm+'/load_RGB', lambda: backend.load_pixels(directory/'media'/arm/'readback.pt'), [arm+'/decode_mp4'])
                for oi, origin in enumerate(method.ORIGINS):
                    for p in range(4):
                        prefix = f'{arm}/r{oi}/p{p}'
                        view = call(prefix+'/prepare', lambda: backend.prepare_view(pixels, origin, p), [arm+'/load_RGB'])
                        z = call(prefix+'/E', lambda: tensor('features/'+prefix+'.pt', backend.encode_view(view)), [prefix+'/prepare', 'vae_load'])
                        tables = {key: arithmetic.blank_table(range(1 if p == 0 else 2, 36)) for key in KEYS}
                        for key in KEYS:
                            phase_tables[arm, oi, key, p] = tables[key]
                        for key in KEYS:
                            rel = 'blind/'+prefix+'/'+key
                            q = record(prefix+'/'+key+'/Q', lambda key=key: backend.observe(z, books[key]), rel+'/Q.json', [prefix+'/E', 'book/'+key])
                            call(prefix+'/'+key+'/score', lambda key=key: backend.score(q, tables[key], counters), [prefix+'/'+key+'/Q'])
                            slot = h+'/'+prefix+'/'+key+'/score'
                            tables[key]['score_call_status'] = journal.data[slot]['status']
                            if journal.data[slot]['status'] != 'RETURNED':
                                arithmetic.mark_unfinished(tables[key], 'score', journal.data[slot]['status'], slot)
                            dump(directory/(rel+'/table.json'), tables[key])
                        spec = call(prefix+'/FFT', lambda: backend.spectrum(z), [prefix+'/E'])
                        for key, secret in KEYS.items():
                            rel = 'blind/'+prefix+'/'+key
                            votes = record(prefix+'/'+key+'/votes', lambda secret=secret: backend.votes(spec, secret), rel+'/votes.json', [prefix+'/FFT'])
                            call(prefix+'/'+key+'/reduce', lambda key=key: backend.reduce(votes, tables[key], counters), [prefix+'/'+key+'/votes'])
                            slot = h+'/'+prefix+'/'+key+'/reduce'
                            tables[key]['payload_call_status'] = journal.data[slot]['status']
                            if journal.data[slot]['status'] != 'RETURNED':
                                arithmetic.mark_unfinished(tables[key], 'payload', journal.data[slot]['status'], slot)
                            dump(directory/(rel+'/table.json'), tables[key])
                            phase_tables[arm, oi, key, p] = tables[key]
                            if votes is not None:
                                local_bits[arm, oi, key, p] = [bit.get('decoded') for row in votes['local_cells'] for bit in row['bits']]
                        del z, spec, view, tables
                del pixels
        finally:
            backend.release_vae()
        for pair in ('OFF_to_P0', 'P0_to_PM'):
            left, right = pair.split('_to_')
            for layer, label in [('preclip', 'preclip'), ('rgb8', 'rgb8'), ('mp4', 'readback')]:
                ld = left+'/D' if layer == 'preclip' else left+('/quantize' if layer == 'rgb8' else '/decode_mp4')
                rd = right+'/D' if layer == 'preclip' else right+('/quantize' if layer == 'rgb8' else '/decode_mp4')
                record('quality/'+pair+'/'+layer, lambda: backend.quality(
                    backend.load_tensor(directory/'media'/left/(label+'.pt')),
                    backend.load_tensor(directory/'media'/right/(label+'.pt')), pair+'/'+layer),
                    'quality/'+pair+'/'+layer+'.json', [ld, rd])
    finally:
        # Bind every slot even after an interrupt; missing observations stay fixed rows.
        for (arm, oi, key, p), table in phase_tables.items():
            prefix = f'{h}/{arm}/r{oi}/p{p}/{key}'
            for field, leaf, section in [('score_call_status','score','score'),('payload_call_status','reduce','payload')]:
                status = journal.data[prefix+'/'+leaf]['status']
                table[field] = status
                if status != 'RETURNED':
                    arithmetic.mark_unfinished(table, section, status, prefix+'/'+leaf)
                    dump(directory/f'blind/{arm}/r{oi}/p{p}/{key}/table.json', table)
        for qid, meta in method.query_metadata().items():
            oi = method.ORIGINS.index(meta['origin'])
            tables = {p: phase_tables.get((meta['arm'], oi, meta['key'], p), {}) for p in range(4)}
            raw = {str(p): local_bits.get((meta['arm'], oi, meta['key'], p), [None]*80) for p in range(4)}
            family = method.bind(tables, raw)
            families[qid] = family
            dump(directory/'blind/families'/(qid+'.json'), family)
            # Family save itself is engineering binding, not an extra observation.
            call('family/'+qid, lambda: family)
        dump(directory/'cpu_counts.json', dict(counters))
        complete = all(f['status'] == 'COMPLETE' for f in families.values())
        dump(directory/'blind_complete.json', dict(status='COMPLETE' if complete else 'INCOMPLETE',
            saved_unix=time.time(), query_denominator=36, hypothesis_denominator=4932,
            semantic_roles_not_read_by_receiver=True, message_truth_not_read_by_receiver=True))
        journal.save()
    return families


def run(output=None, *, config_path=CONFIG, backend=None):
    output = Path(output or DEFAULT_OUTPUT)
    found = execution_artifacts(output)
    if found:
        return existing_attempt(output, found)
    config = load_config(config_path)
    budget = json.loads(BUDGET.read_text())
    output.mkdir(parents=True, exist_ok=True)
    try:
        with (output/'attempt.json').open('x', encoding='utf-8') as stream:
            json.dump(dict(card=94, started_unix=time.time(), scope='single entire fixed2cal2confirm attempt; no resume/retry'), stream)
            flush(stream)
    except FileExistsError:
        return existing_attempt(output)
    journal = Journal(output, config, backend)
    result = dict(status='RUNNING', scientific_pass=None, protocol=dict(card=94, hypotheses=137,
        net_bits=32, gamma=1.0, source_sampling='fixed_hand_chosen_roster', FPR_guarantee=None,
        default_receiver_changed=False, A_changed=False), fixed_denominator=budget['all_four_hosts_complete_path'])
    dump(output/'protocol.json', dict(result['protocol'], config=config, planned_budget=budget))
    families, decisions = {}, {}
    thresholds = dict(status='CALIBRATION_UNRESOLVED', tau=None, **{'lambda': None})
    try:
        if backend is None:
            from runtime.wan.b_fixed_detection_confidence_backend_v1 import WanBackend
            backend = WanBackend(config['model_id'])
            journal.backend = backend
        for host in config['hosts'][:2]:
            families[host['id']] = run_host(host, config, output, backend, journal)
        dump(output/'markers/calibration_blind_complete.json', dict(saved_unix=time.time(),
            hosts=['b93-c01', 'b93-c02'], confirmation_physical_entered=False))
        # Deliberate supervised role/origin use starts here, after both blind saves.
        thresholds = journal.call('calibration', lambda: method.calibrate(families, method.query_metadata()))
        thresholds = thresholds or dict(status='CALIBRATION_UNRESOLVED', tau=None, **{'lambda': None})
        thresholds['frozen_unix'] = time.time() if thresholds['status']=='FROZEN' else None
        dump(output/'thresholds.json', thresholds)  # Durable BEFORE any confirmation preparation.
        if thresholds['status'] == 'FROZEN':
            for host in config['hosts'][2:]:
                families[host['id']] = run_host(host, config, output, backend, journal)
        else:
            for host in config['hosts'][2:]:
                for slot, row in journal.data.items():
                    if slot.startswith(host['id']+'/'):
                        row['unavailable_from'] = ['CALIBRATION_UNRESOLVED']
        for host in config['hosts']:
            h = host['id']
            values = families.get(h, {qid: method.empty_family('NOT_ENTERED_CALIBRATION_UNRESOLVED') for qid in method.query_metadata()})
            families[h] = values
            decisions[h] = {}
            for qid, family in values.items():
                def decide():
                    value = method.decide(family, thresholds)
                    dump(output/'decisions'/h/(qid+'.json'), value)
                    return value
                value = journal.call(h+'/decision/'+qid, decide)
                decisions[h][qid] = value or method._output('UNRESOLVED_TECHNICAL')
        decision_slots = [h['id']+'/decision/'+qid for h in config['hosts'] for qid in method.query_metadata()]
        unsaved = [slot for slot in decision_slots if journal.data[slot]['status'] != 'RETURNED']
        if len(decision_slots) != 144 or unsaved:
            # RETURNED includes the actual decision dump. In-memory fallbacks do not
            # establish blind persistence and may never authorize post-truth access.
            result['post_evaluation_status'] = 'NOT_ENTERED_DECISION_SAVE_FAILURE'
            result['unsaved_decision_slots'] = unsaved
            journal.data['evaluation']['unavailable_from'] = unsaved or ['DECISION_DENOMINATOR_MISMATCH']
        else:
            dump(output/'markers/confirmation_blind_complete.json', dict(saved_unix=time.time(),
                threshold_freeze=thresholds.get('frozen_unix'), all_decisions_saved=True,
                confirmation_message_truth_not_read=True))
            # Post evaluator receives writer bits only after the confirmation marker.
            from runtime.wan.b_fixed_detection_confidence_evaluation_v1 import evaluate
            def post():
                dump(output/'post_truth_access.json', dict(opened_unix=time.time()))
                value = evaluate(families, decisions, method.query_metadata(), config['writer']['bits'])
                dump(output/'evaluation.json', value)
                return dict(status=value['status'])
            journal.call('evaluation', post)
        all_complete = thresholds['status']=='FROZEN' and all(f['status']=='COMPLETE' for fs in families.values() for f in fs.values()) and all(row['status']=='RETURNED' for row in journal.data.values())
        result['status'] = 'COMPLETED_EMPIRICAL_ONLY' if all_complete else 'INCOMPLETE_FIXED_DENOMINATORS_RETAINED'
    except BaseException as error:
        result.update(status='INTERRUPTED' if not isinstance(error, Exception) else 'FAILED', error=f'{type(error).__name__}: {error}')
    finally:
        result.update(counts=journal.counts(), physical_events=journal.events, thresholds_status=thresholds['status'])
        journal.save()
        dump(output/'result.json', result)
    return result
