"""Adopted four fresh sources, twelve codecs, and fixed blind receiver controls.

This module orchestrates the successful source and receiver implementations.
It never imports old experiment results or changes the underlying method.
"""
from __future__ import annotations

from collections import Counter
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from experiments.paper_results_v1 import real_eval, residual_clock_v1 as residual
from experiments.paper_results_v1 import residual_joint_v1 as joint, residual_controls_v1 as controls
from experiments.paper_results_v1.resume_state import StateJournal, read_state, state_exists, unit_attempt, remember

SCHEMA = 'paper-results-v1-independent-four-v1'
CONFIG_PATH = Path(__file__).with_name('independent_four_v1.adopted.json')
CASES = tuple(f'independent_20261015_{i:02d}' for i in range(1, 5))
SOURCE_PHASES = ('generate', 'decode', 'framewise')
STAGES = tuple(f'{phase}:{case}' for case in CASES for phase in
               (*SOURCE_PHASES, 'residual', 'reference-codec')) + (
    'joint-init', 'joint-media', 'joint-clock', 'joint-wan',
    'controls-init', 'controls-media', 'controls-clock', 'controls-wan', 'controls-quality')
TERMINAL = ('COMPLETE', 'FAILED', 'BLOCKED')
TECHNICAL = ('FAILED', 'MISSING', 'BLOCKED', 'INTERRUPTED', 'RUNNING')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fixed_method_semantics(cfg):
    return dict(cases=[{key: case[key] for key in ('case_id', 'prompt', 'negative_prompt', 'seed')}
                       for case in cfg['cases']],
                **{key: copy.deepcopy(cfg[key]) for key in
                   ('keys', 'payload_bits', 'codec', 'attacks', 'generation', 'writer')},
                framewise_batch_frames=cfg['models']['framewise']['batch_frames'])


def _bind_method(data, cfg, previous_config):
    current = fixed_method_semantics(cfg)
    saved = data.get('fixed_method_semantics')
    if saved is None:
        # Recover ordinary fields from older readable configuration records.
        # Missing optional historical records are reported, never identity gates.
        try:
            saved = fixed_method_semantics(read(previous_config))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            data.setdefault('method_record_notes', []).append(dict(
                at=time.time(), status='PRIOR_METHOD_RECORD_UNAVAILABLE', reason=str(exc)))
    if saved is not None and saved != current:
        raise ValueError('saved fixed method semantics differ; do not mix completed experiment units')
    data['fixed_method_semantics'] = current


def configuration(path=CONFIG_PATH):
    cfg = read(path)
    adopted = read(CONFIG_PATH)
    # Check method/data semantics directly, never source/config digests or versions.
    actual, expected = fixed_method_semantics(cfg), fixed_method_semantics(adopted)
    for field in expected:
        if actual[field] != expected[field]:
            raise ValueError(f'adopted {field} semantics differ')
    cfg['budget'] = copy.deepcopy(adopted['budget'])  # nominal plan, never an actual-call counter
    joint._validate_config(cfg)
    controls._validate_config(cfg)
    return cfg


class CallLog:
    """Small durable events at actual call boundaries, separate from snapshots."""
    def __init__(self, path, session_id, stage):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id, self.stage, self.pending = session_id, stage, {}

    def __call__(self, name, event, error=None):
        if event == 'STARTED':
            ident = uuid.uuid4().hex
            self.pending[name] = ident
        else:
            ident = self.pending.pop(name, None)
            if ident is None:
                raise RuntimeError(f'call returned without a start: {name}')
        row = dict(call_id=ident, operation=name, event=event, session_id=self.session_id,
                   stage=self.stage, at=time.time(), error=error)
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush(); os.fsync(stream.fileno())

    def close_pending(self, event, error):
        for name in list(self.pending):
            self(name, event, error)


class SourceStore(real_eval.RunStore):
    """Only source artifacts, using the existing RunStore phase persistence API."""
    def __init__(self, root, config, *, session_id='', stage=''):
        self.output = Path(root).resolve() / 'source/run_state'
        self.output.mkdir(parents=True, exist_ok=True)
        self.path = self.output / 'run_state.json'
        self.config = config
        self.journal = StateJournal(self.path)
        self.observe_call = CallLog(self.output / 'actual_calls.jsonl', session_id, stage)
        if state_exists(self.path):
            self.data = read_state(self.path)
            if self.data.get('case_ids') != list(CASES):
                raise ValueError('source roster differs from adopted four sources')
            self.journal.seed_cache(self.data)
            _bind_method(self.data, config, Path(root) / 'effective_independent_four_config.json')
            for phase in self.data['phases'].values():
                for case in phase['cases'].values():
                    if case['status'] == 'RUNNING':
                        case['status'] = 'INTERRUPTED'
            for attempts in self.data.get('unit_attempts', {}).values():
                for attempt in attempts:
                    if attempt.get('status') == 'RUNNING':
                        attempt.update(status='INTERRUPTED', finished_at=time.time(),
                                       reason='previous process ended before unit attempt completion')
            for record in self.data.get('records', {}).values():
                for attempts in record.get('reference_codec', {}).values():
                    for attempt in attempts:
                        if attempt.get('status') == 'RUNNING':
                            attempt.update(status='INTERRUPTED', finished_at=time.time(),
                                           reason='previous process ended before reference codec completion')
            self.save()
        else:
            artifacts = []
            for case in CASES:
                for arm in ('OFF_NATIVE', 'PAYLOAD_NATIVE'):
                    for phase in ('TERMINAL', 'PRE', 'POST'):
                        artifacts.append(dict(artifact_id=f'{case}/{arm}/{phase}', status='PLANNED'))
                for arm, phase in [('PAYLOAD_FRAMEWISE_RECON', 'PRE'), ('PAYLOAD_FRAMEWISE_M05', 'PRE'),
                                   ('SHARED_FRAMEWISE', 'LATENT')]:
                    artifacts.append(dict(artifact_id=f'{case}/{arm}/{phase}', status='PLANNED'))
            self.data = dict(schema_version=SCHEMA, case_ids=list(CASES), status='INITIALIZED',
                fixed_method_semantics=fixed_method_semantics(config),
                records={}, artifacts=artifacts, unit_attempts={},
                phases={phase: dict(status='PLANNED', cases={c: dict(status='PLANNED', failures=[]) for c in CASES})
                        for phase in SOURCE_PHASES},
                costs=[dict(cost_id=f'{case}/{phase}', status='PLANNED', seconds=None)
                       for case in CASES for phase in SOURCE_PHASES])
            self.save()

    def cost(self, *args, **kwargs):
        super().cost(*args, **kwargs)
        for row in self.data['costs']:
            row['budget_status'] = 'ADOPTED_FIXED_FOUR_SOURCE_BUDGET'
        self.save()


class Run:
    def __init__(self, root, config):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.path = self.root / 'independent_four_state.json'
        self.journal = StateJournal(self.path)
        if state_exists(self.path):
            self.data = read_state(self.path)
            if self.data.get('case_ids') != list(CASES) or [x['stage'] for x in self.data['stages']] != list(STAGES):
                raise ValueError('fixed independent task rows/stages changed')
            self.journal.seed_cache(self.data)
            _bind_method(self.data, config, self.root / 'effective_independent_four_config.json')
            self.save()  # commit recovered fields before replacing the old config record
        else:
            self.data = dict(schema=SCHEMA, status='INITIALIZED', case_ids=list(CASES),
                fixed_method_semantics=fixed_method_semantics(config),
                rows=controls._new_rows(CASES), quality_rows=controls._new_quality_rows(CASES),
                stages=[dict(stage=s, status='PENDING', attempts=[]) for s in STAGES],
                sessions=[], unit_attempts={}, phases={}, adopted_budget=config['budget'],
                fixed_denominator=dict(sources=4, receiver_rows=24, compared_bits=768, quality_pairs=24),
                old_confirmation=dict(untouched=True, rows=2880, EVALUATED=2493, UNSUPPORTED=386, UNRESOLVED=1))
            self.save()
        self.effective = self.root / 'effective_independent_four_config.json'
        joint.atomic_json(self.effective, config)

    def save(self):
        self.journal.save(self.data)


def initialize(root, config):
    run = Run(root, config)
    SourceStore(root, config)
    residual_root = run.root / 'residual'
    joint.atomic_json(residual_root / 'effective_residual_clock_config.json', config)
    residual_path = residual_root / 'run_state/residual_clock_state.json'
    if not state_exists(residual_path):
        joint.atomic_json(residual_path, dict(schema='paper-results-v1-residual-clock-v1', key=config['keys']['K0'],
            checkpoint_state=str(run.root / 'source/run_state/run_state.json'), case_ids=list(CASES),
            effective_config=str(residual_root / 'effective_residual_clock_config.json'), phases={},
            units={case: dict(status='PENDING') for case in CASES}))
    return run


def _source_dependencies(store, phase, case):
    if phase == 'decode':
        ids = [f'{case}/{arm}/TERMINAL' for arm in ('OFF_NATIVE', 'PAYLOAD_NATIVE')]
    elif phase == 'framewise':
        ids = [f'{case}/PAYLOAD_NATIVE/PRE']
    else:
        ids = []
    for ident in ids:
        row = real_eval._artifact(store, ident)
        if row.get('status') != 'AVAILABLE' or not Path(row.get('path', '')).is_file():
            raise FileNotFoundError(f'source dependency unavailable: {ident}')


def construct_residual(run, case):
    path = run.root / 'residual/run_state/residual_clock_state.json'
    data = read_state(path)
    if data['units'][case]['status'] == 'COMPLETE':
        return
    if data['units'][case]['status'] in TECHNICAL:
        remember(data['units'][case], 'new Run all resumes unfinished residual construction')
    journal = StateJournal(path)
    store = SimpleNamespace(output=path.parent, data=data, resolve=lambda p: Path(p),
                            journal=journal, save=lambda: journal.save(data))
    try:
        with journal.periodic(), unit_attempt(store, 'construct/' + case, operation='CPU_RGB8_residual'):
            destination, quality = residual._construct_residual(store, case)
            data['units'][case].update(status='COMPLETE', reason=None, path=str(destination),
                quality_record=str(residual._quality_path(store, case)),
                tail=quality['framewise'][-1], tail_transition=quality['temporal_framewise'][-1])
            store.save()
    except Exception as exc:
        data['units'][case].update(status='FAILED', reason=joint.reason(exc)); store.save()
        raise


def reference_codec(run, case, *, roundtrip=None):
    from runtime.wan.fixed_rgb_media import mp4_roundtrip
    roundtrip = roundtrip or mp4_roundtrip
    source = SourceStore(run.root, run.config)
    records = source.data['records'].setdefault(case, {}).setdefault('reference_codec', {})
    with source.journal.periodic():
        # Deliberately do not call real_eval.phase_codec (four main arms).
        for arm in ('OFF_NATIVE', 'PAYLOAD_NATIVE'):
            row = real_eval._artifact(source, f'{case}/{arm}/PRE')
            post = real_eval._artifact(source, f'{case}/{arm}/POST')
            if post['status'] == 'AVAILABLE':
                continue
            record = dict(status='RUNNING', started_at=time.time(), events={}, calls={})
            records.setdefault(arm, []).append(record); source.save()
            counts = Counter()
            def count(name, done):
                counts[f'{name}:{"completed" if done else "attempted"}'] += 1
                record['calls'] = dict(counts); source.save()
            try:
                if row['status'] != 'AVAILABLE':
                    raise FileNotFoundError(f'{case}/{arm}/PRE unavailable')
                folder = source.output / 'artifacts' / case / arm / 'codec' / f'attempt-{len(records[arm])}'
                with unit_attempt(source, f'{case}/{arm}/POST', operation='reference_mp4_roundtrip'):
                    roundtrip(row['path'], None, folder / 'source.mp4', folder / 'received.rgb8',
                              count=count, event=lambda name, value: record['events'].__setitem__(name, value))
                    receipt = record['events']['rgb24']
                    if receipt.get('shape') != joint.RGB_SHAPE or receipt.get('bytes') != __import__('math').prod(joint.RGB_SHAPE):
                        raise ValueError('reference codec readback geometry changed')
                    source.artifact(post['artifact_id'], status='AVAILABLE', reason=None, path=receipt['path'],
                        shape=receipt['shape'], bytes=receipt['bytes'], dtype='uint8',
                        mp4=record['events']['mp4'], codec=run.config['codec'])
                record['status'] = 'COMPLETE'
            except BaseException as exc:
                status = 'FAILED' if isinstance(exc, Exception) else 'INTERRUPTED'
                record.update(status=status, reason=joint.reason(exc))
                source.artifact(post['artifact_id'], status=status, reason=joint.reason(exc))
                if not isinstance(exc, Exception):
                    raise
            finally:
                record['finished_at'] = time.time()
                source.save()


def _current_joint(run):
    store = joint.Store(run.root / 'joint/run_state')
    if store.cases != CASES or Path(store.data['source_run']).resolve() != run.root / 'residual':
        raise ValueError('joint input must be the current independent cohort')
    return store


def _current_controls(run):
    store = controls.Store(run.root / 'controls/run_state')
    if (store.cases != CASES or Path(store.data['source_joint_run']).resolve() != run.root / 'joint'
            or Path(store.data['checkpoint_run']).resolve() != run.root / 'source'):
        raise ValueError('controls must reuse only this independent cohort')
    return store


def retry_stages(run):
    """Read-only next-session work plan, including workers with failed inner units."""
    selected = {row['stage'] for row in run.data['stages'] if row['status'] != 'COMPLETE'}
    residual_path = run.root / 'residual/run_state/residual_clock_state.json'
    if state_exists(residual_path):
        data = read_state(residual_path)
        selected.update('residual:' + case for case, row in data['units'].items()
                        if row['status'] != 'COMPLETE')
    source_path = run.root / 'source/run_state/run_state.json'
    if state_exists(source_path):
        data = read_state(source_path)
        selected.update('reference-codec:' + row['artifact_id'].split('/')[0]
                        for row in data['artifacts'] if row['artifact_id'].endswith('/POST')
                        and row['status'] != 'AVAILABLE')
    for name, file in (('joint', 'residual_joint_state.json'), ('controls', 'residual_controls_state.json')):
        path = run.root / name / 'run_state' / file
        if not state_exists(path):
            continue
        data = read_state(path)
        for row in data['rows']:
            if row['payload']['status'] == 'READ' or row['clock']['status'] in ('UNSUPPORTED', 'UNRESOLVED'):
                continue
            if row['media']['status'] != 'AVAILABLE': selected.add(name + '-media')
            if row['clock']['status'] != 'CLOCK_READY': selected.add(name + '-clock')
            selected.add(name + '-wan')
        if name == 'controls' and any(row['status'] != 'EVALUATED' for row in data['quality_rows']):
            selected.add('controls-quality')
    # Revisit dependent wrappers. Their durable COMPLETE/AVAILABLE/READ units
    # still skip: this does not authorize repeating successful model/codec calls.
    for stage in tuple(selected):
        phase, _, case = stage.partition(':')
        if case:
            later = {'generate': ('decode', 'framewise', 'residual', 'reference-codec'),
                     'decode': ('framewise', 'residual', 'reference-codec'),
                     'framewise': ('residual',), 'residual': (), 'reference-codec': ()}[phase]
            selected.update(p + ':' + case for p in later)
            selected.update(STAGES[20:] if phase != 'reference-codec' else STAGES[24:])
        elif stage.startswith('joint-'):
            selected.update(STAGES[STAGES.index(stage):])
        elif stage.startswith('controls-'):
            selected.update(STAGES[STAGES.index(stage):])
    # Controls inputs are snapshots of this batch, and must be refreshed after
    # an upstream repair before any resumed K1/OFF work is attempted.
    if any(s.startswith('controls-') for s in selected): selected.add('controls-init')
    return [stage for stage in STAGES if stage in selected]


def prepare_new_session(run, planned):
    """Reopen only unfinished technical units once, before the stage loop."""
    for name, file, cls in (('joint', 'residual_joint_state.json', joint.Store),
                            ('controls', 'residual_controls_state.json', controls.Store)):
        path = run.root / name / 'run_state' / file
        if not state_exists(path): continue
        store = cls(path.parent)
        for row in store.data['rows']:
            if row['payload']['status'] == 'READ' or row['clock']['status'] in ('UNSUPPORTED', 'UNRESOLVED'):
                continue
            changed = any(row[component]['status'] in TECHNICAL for component in ('media', 'clock', 'payload'))
            if changed: remember(row, 'new Run all retains prior row failure')
            for component in ('media', 'clock', 'payload'):
                part = row[component]
                if part['status'] in TECHNICAL:
                    remember(part, 'new Run all resumes unfinished technical unit')
                    part.update(status='PENDING', reason=None)
            if changed:
                row.update(status='PENDING', reason=None)
        for row in store.data.get('quality_rows', []):
            if row['status'] in TECHNICAL:
                remember(row, 'new Run all resumes unfinished quality record')
                row.update(status='PENDING', reason=None)
        for stage in planned:
            if stage.startswith(name + '-') and stage != name + '-init':
                store.data['phases'][stage.split('-')[1]] = 'PENDING'
        store.save()
    for row in run.data['stages']:
        if row['stage'] in planned:
            row['status'] = 'PENDING'
    run.save()


def setup_requirements(run, planned):
    """Actual unfinished units determine setup; empty wrappers need no models."""
    required = dict(numpy=False, pillow=False, torch=False, ffmpeg=False,
                    wan_full=False, wan_vae=False, framewise=False)
    source = read_state(run.root / 'source/run_state/run_state.json')
    artifacts = {row['artifact_id']: row for row in source['artifacts']}
    for stage in planned:
        phase, _, case = stage.partition(':')
        if not case: continue
        if phase == 'residual':
            units = read_state(run.root / 'residual/run_state/residual_clock_state.json')['units']
            required['numpy'] |= units[case]['status'] != 'COMPLETE'
        if phase in SOURCE_PHASES: required['torch'] = True
        arms = ('PAYLOAD_FRAMEWISE_RECON', 'PAYLOAD_FRAMEWISE_M05') if phase == 'framewise' else ('OFF_NATIVE', 'PAYLOAD_NATIVE')
        output = 'TERMINAL' if phase == 'generate' else 'POST' if phase == 'reference-codec' else 'PRE'
        if phase in (*SOURCE_PHASES, 'reference-codec') and any(
                artifacts[f'{case}/{arm}/{output}']['status'] != 'AVAILABLE' for arm in arms):
            required[{'generate': 'wan_full', 'decode': 'wan_vae',
                      'framewise': 'framewise', 'reference-codec': 'ffmpeg'}[phase]] = True
    for name, file in (('joint', 'residual_joint_state.json'), ('controls', 'residual_controls_state.json')):
        path = run.root / name / 'run_state' / file
        if not state_exists(path):
            required['framewise'] |= name + '-clock' in planned
            required['wan_vae'] |= name + '-wan' in planned
            required['ffmpeg'] |= name == 'joint' and name + '-media' in planned
            required['pillow'] |= name == 'controls' and 'controls-quality' in planned
            continue
        data = read_state(path)
        store = SimpleNamespace(output=path.parent, data=data)
        if name == 'controls' and 'controls-quality' in planned:
            required['pillow'] |= any(row['status'] != 'EVALUATED' for row in data['quality_rows'])
        for row in data['rows']:
            if row['payload']['status'] == 'READ' or row['clock']['status'] in ('UNSUPPORTED', 'UNRESOLVED'):
                continue
            if name == 'controls' and row['arm'] == 'WM_K0_REUSED': continue
            if name + '-media' in planned and row['media']['status'] != 'AVAILABLE':
                required['torch'] = True
                required['ffmpeg'] |= name == 'joint' and row['group'] == 'CLEAN_MP4'
            if name + '-clock' in planned and row['clock']['status'] != 'CLOCK_READY':
                required['torch'] = True
                if name == 'joint' or row['arm'] == 'OFF_K0':
                    cache = joint._clock_latent_path(store, row) if name == 'joint' else controls._control_framewise_path(store, row)
                    try: joint._load_npy(cache, row['received_frames'])
                    except Exception: required['framewise'] = True
            if name + '-wan' in planned:
                required['torch'] = True
                cache = joint._wan_latent_path(store, row) if name == 'joint' else controls._control_wan_path(store, row)
                operation = None
                try:
                    operation = read(row['clock']['selector_record'])['operation']
                    if not isinstance(operation, dict) or operation.get('status') != 'SUPPORTED':
                        raise ValueError('blind operation remains unfinished')
                    joint._load_torch(cache, operation['output_frames'])
                    continue
                except Exception:
                    pass
                # This is the same public operation equality used by K1 runtime.
                if name == 'controls' and row['arm'] == 'WM_K1' and operation is not None:
                    try:
                        original = read(row['clock']['source_k0_selector_record'])['operation']
                        if operation != original: raise ValueError('different complete operation')
                        cache = run.root / 'joint/run_state/latents/wan' / row['group'].lower() / (row['case_id'] + '.pt')
                        joint._load_torch(cache, operation['output_frames'])
                        continue
                    except Exception:
                        pass
                required['wan_vae'] = True
    required['torch'] |= required['wan_full'] or required['wan_vae'] or required['framewise']
    required['numpy'] |= required['torch'] or required['ffmpeg'] or required['pillow']
    return required


def _refresh_joint_inputs(run, store):
    for case, source in store.data['sources'].items():
        if source['status'] == 'AVAILABLE': continue
        path = Path(source['path'])
        if path.is_file() and path.stat().st_size == __import__('math').prod(joint.RGB_SHAPE):
            remember(source, 'current-batch residual became readable after repair')
            source.update(status='AVAILABLE', reason=None)
    store.save()


def _refresh_controls_inputs(run, store):
    source = _current_joint(run)
    checkpoint = read_state(run.root / 'source/run_state/run_state.json')
    for row in store.data['rows']:
        if row['payload']['status'] == 'READ' or row['clock']['status'] in ('UNSUPPORTED', 'UNRESOLVED'):
            continue
        parent = next(x for x in source.data['rows'] if x['case_id'] == row['case_id'] and x['group'] == row['group'])
        if row['arm'] == 'WM_K0_REUSED':
            remember(row, 'refresh incomplete K0 copy from current batch')
            for component in ('media', 'clock', 'payload'):
                row[component] = {**copy.deepcopy(parent[component]), 'reuse': 'SOURCE_RESIDUAL_JOINT'}
            row.update(status=parent['status'], reason=parent.get('reason'),
                       missing_bits=0 if parent['payload']['status'] == 'READ' else 32)
        elif row['arm'] == 'WM_K1':
            if row['media']['status'] != 'AVAILABLE' and parent['media']['status'] == 'AVAILABLE':
                remember(row['media'], 'refresh repaired current-batch received media')
                row['media'] = {**copy.deepcopy(parent['media']), 'reuse': 'SAME_RECEIVED_RGB_AS_WM_K0',
                                'resume_history': row['media'].get('resume_history', [])}
            if row['clock']['status'] != 'CLOCK_READY':
                row['clock']['source_framewise_latent'] = parent['clock'].get('latent_path')
                row['clock']['source_k0_selector_record'] = parent['clock'].get('selector_record')
        elif row['arm'] == 'OFF_K0' and row['media']['status'] != 'AVAILABLE':
            artifact = real_eval._artifact(SimpleNamespace(data=checkpoint), row['case_id'] + '/OFF_NATIVE/POST')
            row['media']['source_artifact'] = copy.deepcopy(artifact)
    store.save()


def run_stage(root, config, stage, session_id=''):
    run = Run(root, config)
    if stage not in STAGES:
        raise ValueError('unknown fixed task stage')
    phase, _, case = stage.partition(':')
    if phase in SOURCE_PHASES:
        source = SourceStore(root, config, session_id=session_id, stage=stage)
        _source_dependencies(source, phase, case)
        try:
            getattr(real_eval, 'phase_' + phase)(source, config, case)
        except BaseException as exc:
            source.observe_call.close_pending('FAILED' if isinstance(exc, Exception) else 'INTERRUPTED', str(exc))
            raise
    elif phase == 'residual':
        construct_residual(run, case)
    elif phase == 'reference-codec':
        reference_codec(run, case)
    elif stage == 'joint-init':
        store = joint.initialize(run.root / 'residual', run.root / 'joint/run_state', run.effective, cases=CASES)
        _refresh_joint_inputs(run, store)
        store.data['phases']['quality'] = 'COMPLETE'
        store.data['quality_inspection'] = dict(status='NOT_RUN_SEPARATELY',
            reason='PRE rows retained during construction; full POST quality executed in controls')
        store.save()
    elif stage.startswith('joint-'):
        getattr(joint, 'run_' + stage.split('-')[1])(_current_joint(run))
    elif stage == 'controls-init':
        # Parent orchestration completes or records all K0 stages before aggregation.
        parent = _current_joint(run)
        if joint.pending_model_units(parent):
            raise RuntimeError('current-cohort K0 work remains pending; controls are not initialized early')
        store = controls.initialize(run.root / 'joint', run.root / 'source', run.root / 'controls/run_state',
                                    run.effective, cases=CASES)
        _refresh_controls_inputs(run, store)
    elif stage.startswith('controls-'):
        getattr(controls, 'run_' + stage.split('-')[1])(_current_controls(run))


def _call_log_summary(path, session_id=None):
    calls, malformed = {}, []
    if Path(path).is_file():
        for number, line in enumerate(Path(path).read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                if session_id is None or row['session_id'] == session_id:
                    calls[row['call_id']] = row
            except (ValueError, KeyError, TypeError):
                malformed.append(number)
    result = {}
    for row in calls.values():
        counts = result.setdefault(row['operation'], Counter())
        counts['attempted'] += 1
        counts[row['event'].lower()] += 1
    return dict(operations={k: dict(v) for k, v in sorted(result.items())},
                unreadable_event_lines=malformed,
                started_without_return='unfinished/abruptly interrupted; never counted as returned')


def call_totals(run, *, since=None, session_id=None):
    result = {'source': _call_log_summary(run.root / 'source/run_state/actual_calls.jsonl', session_id)}
    source = read_state(run.root / 'source/run_state/run_state.json')
    result['reference_codec'] = dict(Counter(
        key for rec in source['records'].values() for attempts in rec.get('reference_codec', {}).values()
        for attempt in attempts if since is None or attempt.get('started_at', 0) >= since
        for key, count in attempt['calls'].items() for _ in range(count)))
    for name, file in [('joint', 'residual_joint_state.json'), ('controls', 'residual_controls_state.json')]:
        path = run.root / name / 'run_state' / file
        if state_exists(path):
            data = read_state(path)
            result[name] = {key: dict(Counter(x.get('status', 'UNKNOWN') for x in value
                           if since is None or x.get('started_at', x.get('finished_at', 0)) >= since))
                           for key, value in data.items() if isinstance(value, list) and
                           key in ('codec_calls', 'edit_calls', 'framewise_loads', 'framewise_encode_clips',
                                   'framewise_physical_batches', 'clock_calls', 'wan_loads', 'wan_encode_clips',
                                   'payload_read_calls', 'wan_reuses', 'quality_calls')}
    return result


def report(run):
    receiver_path = run.root / 'controls/run_state/evaluation_report.json'
    if receiver_path.is_file():
        current = read(receiver_path)
        run.data['rows'] = current['rows'] if 'rows' in current else current['control_rows']
        store = _current_controls(run)
        run.data['quality_rows'] = store.data['quality_rows']
    completed = all(x['status'] in TERMINAL for x in run.data['stages'])
    bad = any(x['status'] != 'COMPLETE' for x in run.data['stages'])
    evaluated = sum(x['status'] == 'EVALUATED' for x in run.data['rows'])
    quality_evaluated = sum(x['status'] == 'EVALUATED' for x in run.data['quality_rows'])
    if completed:
        for row in run.data['rows']:
            if row['status'] in ('PENDING', 'MEDIA_READY', 'CLOCK_READY'):
                row.update(status='MISSING', reason='upstream stage failed; retained fixed slot')
        for row in run.data['quality_rows']:
            if row['status'] == 'PENDING':
                row.update(status='MISSING', reason='upstream stage failed; retained fixed slot')
    run.data['status'] = ('COMPLETE' if completed and not bad and evaluated == 24 and quality_evaluated == 24 else
                          'COMPLETE_WITH_RETAINED_FAILURES' if completed else 'IN_PROGRESS')
    cumulative = call_totals(run)
    current_session = run.data['sessions'][-1] if run.data['sessions'] else None
    planned = retry_stages(run)
    summary = dict(schema=SCHEMA, status=run.data['status'], fixed_denominator=run.data['fixed_denominator'],
        receiver_status_counts=dict(Counter(x['status'] for x in run.data['rows'])),
        quality_status_counts=dict(Counter(x['status'] for x in run.data['quality_rows'])),
        adopted_budget=run.config['budget'], actual_calls_cumulative=cumulative,
        actual_calls_this_session=call_totals(run, since=current_session['started_at'],
            session_id=current_session['session_id']) if current_session else None,
        old_confirmation=run.data['old_confirmation'],
        actual_count_limits={
            'generation_pipeline_load': 'actual prepare_generation attempt/return including prompt preparation; internal load/text calls not separately instrumented',
            'CFG_combinations': 'UNKNOWN as a separate actual counter; nominal 400 only, not filled from the native-step count',
            'internal_Wan_chunks_text_forwards': 'UNKNOWN; clip/API calls only',
            'candidate_codec_subcalls': 'see durable media receipts; interrupted roundtrip does not imply every subcall returned'},
        quality_reference='NATIVE_PRE/POST = PAYLOAD_NATIVE; residual-clock incremental quality, not total versus OFF',
        evidence_ceiling='four prelisted fresh sources; fixed keys/payload/codec/delete_g0; no threshold, FPR, generalization or scientific PASS',
        recovery='60s last committed state plus immediate complete units; next Run all retries unfinished technical units once with history; completed/scientific terminal units skip; real cross-Colab recovery not tested',
        pending_stages=planned, setup_requirements=setup_requirements(run, planned))
    run.save()
    joint.atomic_json(run.root / 'summary.json', summary)
    joint.atomic_json(run.root / 'evaluation_report.json', dict(summary=summary, rows=run.data['rows'],
                                                               quality_rows=run.data['quality_rows']))
    controls._atomic_csv(run.root / 'receiver_rows.csv', [
        {k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in run.data['rows']])
    return summary


def run_all(root, config, *, invoke=None):
    run = initialize(root, config)
    planned = retry_stages(run)
    prepare_new_session(run, planned)
    session_id = uuid.uuid4().hex
    session = dict(session_id=session_id, started_at=time.time(), calls_before=call_totals(run), status='RUNNING')
    for prior in run.data['sessions']:
        if prior['status'] == 'RUNNING':
            prior.update(status='INTERRUPTED', reason='prior Run all ended without completion')
    run.data['sessions'].append(session); run.save()
    def execute(stage):
        command = [sys.executable, '-u', '-m', 'experiments.paper_results_v1.independent_four_v1_cli',
                   '--output', str(run.root), '--config', str(run.effective), '--phase', stage,
                   '--session-id', session_id]
        with (run.root / 'execution.log').open('a') as stream:
            stream.write('COMMAND ' + repr(command) + '\n'); stream.flush()
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            try:
                return child.wait()
            except BaseException:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
                raise
    invoke = invoke or execute
    try:
        with run.journal.periodic():
            for row in run.data['stages']:
                if row['status'] in TERMINAL:
                    continue
                for old in row['attempts']:
                    if old['status'] == 'RUNNING':
                        old.update(status='INTERRUPTED', reason='prior process ended before completion')
                attempt = dict(status='RUNNING', session_id=session_id, started_at=time.time())
                row['attempts'].append(attempt); row['status'] = 'RUNNING'; run.save()
                print('RUN', row['stage'], flush=True)
                try:
                    code = invoke(row['stage'])
                    if code is None: code = 0
                    if code < 0 or code in (130, 137, 143):
                        raise KeyboardInterrupt(f'worker interrupted: {code}')
                    row['status'] = 'COMPLETE' if code == 0 else 'FAILED'
                    attempt.update(status=row['status'], returncode=code)
                except BaseException as exc:
                    row['status'] = 'FAILED' if isinstance(exc, Exception) else 'INTERRUPTED'
                    attempt.update(status=row['status'], reason=joint.reason(exc))
                    if not isinstance(exc, Exception): raise
                finally:
                    attempt['finished_at'] = time.time(); run.save()
            session['status'] = 'COMPLETE'
    except BaseException as exc:
        session.update(status='INTERRUPTED' if not isinstance(exc, Exception) else 'FAILED', reason=joint.reason(exc))
        raise
    finally:
        session['finished_at'] = time.time()
        report(run)
    return report(run)
