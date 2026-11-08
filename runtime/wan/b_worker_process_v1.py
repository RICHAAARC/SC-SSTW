"""Shared process helpers extracted unchanged from the validated r02/runtime path.

No experiment CLI or historical runner is needed by C4 workers.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import subprocess
import time
from . import b_relative_time_payload_records_v1 as rec
ROOT = Path(__file__).resolve().parents[2]


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
