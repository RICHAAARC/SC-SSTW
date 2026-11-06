"""One root attempt: time blind -> payload blind -> joint closure -> two posts."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from main.tube_state.b_joint_time_payload_r02_v1 import SOURCE, validate
from . import b_joint_time_payload_r02_records_v1 as rec
from . import b_joint_time_relative_validation_v1 as temporal
from . import b_relative_time_payload_records_v1 as payload_rec
from .b_relative_time_payload_runner_v1 import stop_group, watch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/'experiments/wan_state_clock/configs/b_joint_time_payload_r02_v1.json'
CLI = 'experiments.wan_state_clock.b_joint_time_payload_r02_v1_run'


@contextmanager
def termination_handler():
    # The old workers intentionally catch BaseException. Keep cancellation outside
    # their exception/result conversion so it cannot become an ordinary IO failure.
    interruption = dict(requested=False, signal=None)
    def terminate(signum, frame):
        interruption.update(requested=True, signal=signum)
        raise InterruptedError('joint attempt terminated by signal '+str(signum))
    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try: yield interruption
    finally:
        for sig, handler in previous.items(): signal.signal(sig, handler)


def stage_process(command, log_path, *, cwd=ROOT):
    """Keep original stage limits; record generation/capture/replay wall/RSS, no invented cap."""
    import psutil
    process = None; start = time.monotonic(); peak = 0
    environment = os.environ.copy(); environment['PYTHONPATH'] = str(cwd)
    try:
        with Path(log_path).open('a') as log:
            process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            root = psutil.Process(process.pid)
            while process.poll() is None:
                rss = 0
                try:
                    for member in [root]+root.children(recursive=True):
                        try: rss += member.memory_info().rss
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                peak = max(peak, rss)
                time.sleep(.1)
            return dict(returncode=process.returncode, wall_seconds=time.monotonic()-start,
                        peak_process_tree_rss_bytes=peak, whole_stage_limit_certified=False)
    finally:
        if process is not None: stop_group(process)


def time_stage(stage, output, scratch):
    command = [sys.executable, '-u', '-m', CLI, '--time-stage', stage,
               '--output', str(Path(output).parent), '--scratch', str(scratch)]
    resource = stage_process(command, Path(output)/(stage+'.log'))
    rec.write(Path(output)/(stage+'_process_resources.json'), resource)
    result = rec.read(Path(output)/'stages'/stage/'stage_result.json')
    if resource['returncode'] != 0 or result['status'] != 'COMPLETED':
        raise RuntimeError(f'{stage} failed: {result.get("failure")}')
    return result


def close_posts(output, reason):
    """Accounting only; interrupted or unstarted posts are never retried."""
    output = Path(output)
    posts = rec.read(output/'posts.json') if (output/'posts.json').is_file() else {}
    for name in ('time', 'payload'):
        row = posts.setdefault(name, dict(status='NOT_STARTED', started=0, returned=0))
        if row['status'] == 'STARTED': row['status'] = 'INTERRUPTED_CONSUMED'
        if row['status'] != 'RETURNED': row.update(dependency_status='DEPENDENCY_UNRESOLVED', reason=reason)
    rec.write(output/'posts.json', posts)
    return posts


def post_once(output, *, time_evaluator=None, payload_evaluator=None, interruption=None):
    output = Path(output)
    if rec.read(output/'joint_blind_terminal.json')['status'] != 'JOINT_BLIND_TERMINAL':
        raise ValueError('joint blind closure required before either post')
    try:
        with (output/'post_started.json').open('x') as stream: stream.write('{"consumed":true}')
    except FileExistsError:
        return rec.read(output/'posts.json') if (output/'posts.json').is_file() else dict(status='INTERRUPTED_CONSUMED')
    if time_evaluator is None:
        from .b_joint_time_relative_evaluation_v1 import evaluate_saved
        time_evaluator = lambda path: evaluate_saved(path, require_returned=True)
    if payload_evaluator is None:
        from .b_relative_time_payload_evaluation_v1 import evaluate
        payload_evaluator = lambda path, source: evaluate(path, source, retain_incomplete=True,
                                               source_scope='one new ocean content; five related views')
    posts = {name: dict(status='NOT_STARTED', started=0, returned=0) for name in ('time', 'payload')}
    rec.write(output/'posts.json', posts)
    for name, call in (('time', lambda: time_evaluator(output/'time_chain')),
                       ('payload', lambda: payload_evaluator(output/'payload', output/'time_chain'))):
        posts[name].update(status='STARTED', started=1, started_at=time.time())
        rec.write(output/'posts.json', posts)
        start = time.monotonic()
        try:
            value = call()
            posts[name].update(status='RETURNED', returned=1, evaluation_status=value.get('status'))
        except BaseException as error:
            if isinstance(error, KeyboardInterrupt) or (interruption and interruption['requested']):
                if interruption is not None: interruption['requested'] = True
                posts[name].update(status='INTERRUPTED_CONSUMED', error=f'{type(error).__name__}: {error}')
                rec.write(output/'external_interruption.json', dict(interruption or {}, requested=True,
                          failure=posts[name]['error'], phase=name+'_post'))
                rec.write(output/'posts.json', posts)
                return close_posts(output, posts[name]['error'])
            posts[name].update(status='FAILED', error=f'{type(error).__name__}: {error}')
        posts[name]['seconds'] = time.monotonic()-start
        rec.write(output/'posts.json', posts)
    return posts


def payload_stage(output, *, backend_factory=None, time_evaluator=None, payload_evaluator=None):
    output = Path(output)
    if rec.read(output/'attempt.json')['status'] != 'RUNNING': return rec.recover(output)
    if not (output/'time_terminal_slots.json').is_file():
        raise ValueError('time receipt closure required before payload stage')
    try:
        with (output/'payload_stage_started.json').open('x') as stream: stream.write('{"consumed":true}')
    except FileExistsError: return rec.recover(output)
    failure = None
    dependency_path = output/'dependency_failure.json'
    if dependency_path.is_file(): failure = rec.read(dependency_path)['failure']
    with termination_handler() as interruption:
        try:
            if failure is None:
                from .b_relative_time_payload_worker_v1 import worker
                result = worker(output/'payload', backend_factory=backend_factory, source_id=SOURCE, defer_post=True)
                failure = result.get('failure')
                if failure and failure.startswith('KeyboardInterrupt:'): interruption['requested'] = True
        except BaseException as error:
            failure = f'{type(error).__name__}: {error}'
            if isinstance(error, KeyboardInterrupt): interruption['requested'] = True
        rec.close_blind(output, failure)
        if interruption['requested']:
            rec.write(output/'external_interruption.json', dict(interruption, failure=failure, phase='payload'))
            close_posts(output, failure)
        else:
            post_once(output, time_evaluator=time_evaluator, payload_evaluator=payload_evaluator,
                      interruption=interruption)
        if interruption['requested']: failure = failure or 'external interruption consumed the attempt'
    return rec.finish(output, failure)


def payload_process(output):
    command = [sys.executable, '-u', '-m', CLI, '--payload-stage', '--output', str(output)]
    environment = os.environ.copy(); environment['PYTHONPATH'] = str(ROOT)
    return watch(command, Path(output)/'payload', environment=environment)


def execute_reserved(output, scratch, *, worker_runner=None, payload_runner=None):
    output, scratch = Path(output), Path(scratch)
    attempt = rec.read(output/'attempt.json')
    if attempt['status'] != 'RESERVED': return rec.recover(output)
    try:
        with (output/'supervisor_started.json').open('x') as stream: stream.write('{"consumed":true}')
    except FileExistsError: return rec.recover(output)
    config = rec.read(output/'protocol.json'); attempt['status'] = 'RUNNING'
    rec.write(output/'attempt.json', attempt)
    start = time.monotonic(); failure = None
    with termination_handler() as interruption:
        try:
            validate(config)
            # Scratch must be outside every retained result subtree.
            if scratch.resolve() == output.resolve() or scratch.resolve().is_relative_to(output.resolve()):
                raise ValueError('scratch must be outside retained results')
            result = temporal.execute_reserved(config, output/'time_chain', scratch,
                worker_runner=worker_runner or time_stage, validator=validate, defer_post=True)
            if (result.get('failure') or {}).get('type') == 'KeyboardInterrupt': interruption['requested'] = True
            if result['status'] != 'COMPLETED': failure = str(result.get('failure') or result.get('accounting_errors'))
            terminal = rec.close_time(output, failure)
            if terminal['accounting_errors']: failure = failure or str(terminal['accounting_errors'])
            if failure is None:
                for item in terminal['units']:
                    if item['execution_status'] != 'RETURNED':
                        raise ValueError('time dependency not returned: '+item['unit']['id'])
                    media = output/'time_chain/media'/item['unit']['id']/'received.mp4'
                    if not media.is_file(): raise FileNotFoundError(media)
        except BaseException as error:
            failure = f'{type(error).__name__}: {error}'
            if isinstance(error, KeyboardInterrupt): interruption['requested'] = True
        if not (output/'time_terminal_slots.json').is_file(): rec.close_time(output, failure)
        if failure: rec.write(output/'dependency_failure.json', dict(status='DEPENDENCY_UNRESOLVED', failure=failure))
        if interruption['requested']:
            failure = failure or 'external interruption consumed the attempt'
            rec.write(output/'external_interruption.json', dict(interruption, failure=failure, phase='time'))
            resource = dict(status='NOT_STARTED', worker_started=0, failure=failure, worker_returncode=None)
        else:
            try:
                resource = (payload_runner or payload_process)(output)
            except BaseException as error:
                resource = dict(failure=f'{type(error).__name__}: {error}', worker_returncode=None)
        rec.write(output/'payload/resources.json', resource)
        if resource.get('failure') or resource.get('worker_returncode') != 0:
            failure = failure or resource.get('failure') or 'payload worker exited '+str(resource.get('worker_returncode'))
            rec.close_blind(output, failure)
            # The fixed payload wall/RSS attempt is exhausted; no parent-side post retry.
            close_posts(output, failure)
        if not (output/'joint_blind_terminal.json').is_file():
            failure = failure or 'payload process returned without joint closure'
            rec.close_blind(output, failure)
    rec.write(output/'chain_resources.json', dict(wall_seconds=time.monotonic()-start,
        whole_chain_wall_and_peak_certified=False,
        payload_worker_scope='payload scientific work, joint closure and both posts only'))
    if (output/'result.json').is_file():
        failure = failure or rec.read(output/'result.json').get('failure')
    return rec.finish(output, failure)


def run(config, output=None, scratch='/content/video-wm-joint-time-payload-r02/work', **kwargs):
    output = Path(output or config['output']).expanduser().resolve()
    if output.exists() and any(output.iterdir()): return rec.recover(output)
    if not rec.reserve(output, config): return rec.recover(output)
    return execute_reserved(output, scratch, **kwargs)
