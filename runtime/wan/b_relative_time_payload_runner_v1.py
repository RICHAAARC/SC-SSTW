"""External wall/RSS watchdog for the whole consumed scientific worker."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from . import b_relative_time_payload_records_v1 as rec

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT/'experiments/wan_state_clock/configs/b_relative_time_payload_v1.json'


def stop_group(worker, grace=2):
    start = time.monotonic()
    try: os.killpg(worker.pid, signal.SIGTERM)
    except ProcessLookupError: pass
    try: worker.wait(timeout=grace)
    except subprocess.TimeoutExpired: pass
    finally:
        try: os.killpg(worker.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    worker.wait()
    return time.monotonic()-start


def watch(command, output, *, cwd=ROOT, environment=None, limits=None):
    """Independent supervisor; injectable limits/command only for artificial tests."""
    import psutil
    limits = limits or rec.LIMITS
    start = time.monotonic(); peak = 0; failure = None; worker = None; termination = 0.
    try:
        with (Path(output)/'worker.log').open('a') as log:
            worker = subprocess.Popen(command, cwd=cwd, env=environment, stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(worker.pid)
            while worker.poll() is None:
                if time.monotonic()-start >= limits['worker_seconds']:
                    failure = 'WORKER_WALL_LIMIT'; break
                rss = 0
                try:
                    members = [process]+process.children(recursive=True)
                    for member in members:
                        try: rss += member.memory_info().rss
                        except psutil.NoSuchProcess: pass
                except psutil.NoSuchProcess: pass
                peak = max(peak, rss)
                if rss > limits['process_tree_rss_bytes']:
                    failure = 'PROCESS_TREE_RSS_LIMIT'; break
                time.sleep(limits['poll_seconds'])
            termination = stop_group(worker, limits['terminate_grace_seconds'])
    except BaseException as error:
        failure = f'{type(error).__name__}: {error}'
        if worker is not None: termination = stop_group(worker, limits['terminate_grace_seconds'])
    return dict(worker_returncode=worker.returncode if worker else None, failure=failure,
                worker_wall_seconds=time.monotonic()-start, peak_process_tree_rss_bytes=peak,
                termination_seconds=termination, limits=limits,
                enforcement='parent process-group watchdog; sampled RSS; finite polling/termination delay')


def execute_reserved(output):
    output = Path(output)
    if not (output/'attempt.json').is_file(): raise ValueError('existing reservation required')
    try:
        with (output/'supervisor_started.json').open('x') as stream: stream.write('{"consumed":true}')
    except FileExistsError:
        return rec.recover(output)
    command = [sys.executable, '-u', '-m', 'experiments.wan_state_clock.b_relative_time_payload_v1_run',
               '--worker', '--output', str(output)]
    environment = os.environ.copy(); environment['PYTHONPATH'] = str(ROOT)
    try: resources = watch(command, output, environment=environment)
    except BaseException as error:
        resources = dict(failure=f'{type(error).__name__}: {error}', worker_returncode=None)
    rec.write(output/'resources.json', resources)
    result_path = output/'result.json'
    result = rec.read(result_path) if result_path.is_file() else None
    if resources.get('failure') or resources.get('worker_returncode') != 0 or result is None:
        reason = resources.get('failure') or f"worker exit {resources.get('worker_returncode')} without successful terminal"
        if not (output/'blind_terminal.json').is_file(): rec.finalize_blind(output, reason)
        result = dict(result or {}, status='STOPPED_FIXED_ATTEMPT', supervisor_failure=reason,
                      denominators=rec.DENOMINATORS, consumed=True, scientific_pass=None)
        rec.write(result_path, result)
    return result


def run(config, output=None):
    output = Path(output or config['output']).expanduser().resolve()
    if output.exists() and any(output.iterdir()): return rec.recover(output)
    source = Path(config['source']).expanduser().resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('input and new output directories must not overlap')
    if not rec.reserve(output, config): return rec.recover(output)
    return execute_reserved(output)
