"""One supervised attempt; CUDA/model work exists only in the child worker."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from runtime.public_statistic.phase2_execution import initial_result, validate_config, worker, write


class RunCancelled(BaseException):
    pass


def install_lifetime(expected_parent):
    """Linux parent-death signal plus post-install check closes startup race."""
    import ctypes
    interrupted = [False]
    def cancel(signum, frame):
        if interrupted[0]:return
        interrupted[0] = True
        raise RunCancelled('PARENT_LOST' if os.getppid() != expected_parent else 'CANCELLED')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0):raise OSError(ctypes.get_errno(), 'parent-death signal')
    if os.getppid() != expected_parent:raise RunCancelled('PARENT_LOST')


def stop_group(child):
    """Allow worker ledger cleanup, then bound cancellation and kill descendants."""
    try:os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:pass
    try:child.wait(timeout=3)
    except subprocess.TimeoutExpired:pass
    try:os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:pass
    child.wait()


def finalize_stopped(out, c, reason):
    path = Path(out)/'result.json'
    result = json.loads(path.read_text()) if path.exists() else initial_result(c)
    result['status'] = 'FATAL_STOP'; result['supervisor_stop'] = reason
    for arm in result['arms'].values():
        if arm['status'] == 'RUNNING':
            arm['status'] = 'FAILED_' + reason
            for row in arm['rows']:
                if row['status'] != 'COMPLETE': row['status'] = 'FAILED_' + reason
    write(path, result)


def process_tree_rss(pid):
    """Linux/Colab resident bytes for the worker and its current descendants."""
    seen = set(); pending = [pid]; total = 0
    while pending:
        current = pending.pop()
        if current in seen: continue
        seen.add(current)
        try:
            folder = Path('/proc')/str(current)
            resident_pages = int((folder/'statm').read_text().split()[1])
            total += resident_pages * os.sysconf('SC_PAGE_SIZE')
            for task in (folder/'task').iterdir():
                try: children = (task/'children').read_text().split()
                except FileNotFoundError:
                    if task.exists(): raise
                    continue
                pending.extend(int(child) for child in children)
        except FileNotFoundError:
            if folder.exists(): raise  # live process became unreadable, not an exit race
            continue
    return total


def supervise(config_path, output):
    c = json.loads(Path(config_path).read_text()); validate_config(c)
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    initial = initial_result(c); initial['status']='RUNNING'; initial['active']='LAUNCHER_START'
    write(out/'result.json', initial); write(out/'config.json', c)
    command = [sys.executable, '-m', 'experiments.public_statistic.run_phase2', '--config', str(Path(config_path).resolve()), '--output', str(out.resolve()), '--worker', '--parent-pid', str(os.getpid())]
    start = time.monotonic(); reason = None; child = None
    memory = dict(status="unavailable", peak_tree_rss_bytes=None, error=None)
    try:
        with (out/'execution.log').open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                if time.monotonic()-start >= c['budget']['wall_seconds']: reason = 'WALL_BUDGET'; break
                try:
                    rss = process_tree_rss(child.pid)
                    memory.update(status='available', peak_tree_rss_bytes=max(memory['peak_tree_rss_bytes'] or 0, rss))
                except Exception as exc:
                    memory.update(status='unavailable', error=repr(exc))
                time.sleep(.2)
            if reason is not None:
                stop_group(child)
                finalize_stopped(out, c, reason)
            elif child.returncode:
                # Kill any surviving codec/download descendant on child failure.
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                finalize_stopped(out, c, 'WORKER_FAILURE')
    except BaseException as exc:
        reason = str(exc) if isinstance(exc, RunCancelled) else 'LAUNCHER_INTERRUPTED'
        if child is not None:
            stop_group(child)
        finalize_stopped(out, c, reason)
        raise
    finally:
        write(out/'execution_exit.json', dict(command=command, returncode=child.returncode if child else None, stop=reason, elapsed_seconds=time.monotonic()-start, automatic_retries=0, host_memory_diagnostic=memory))
    return child.returncode if child.returncode else (1 if reason else 0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--output', required=True); p.add_argument('--worker', action='store_true'); p.add_argument('--parent-pid',type=int); a=p.parse_args()
    try:
        install_lifetime(a.parent_pid if a.parent_pid is not None else os.getppid())
        if a.worker: worker(a.config, a.output)
        else: sys.exit(supervise(a.config, a.output))
    except RunCancelled as exc:
        # Parent may already be gone; the worker owns its final ledger cleanup.
        out = Path(a.output)
        if out.exists():finalize_stopped(out,json.loads(Path(a.config).read_text()),str(exc))
        if a.worker and os.getpgrp() == os.getpid():
            # Persist first; then terminate this worker and any codec descendants.
            os.killpg(os.getpid(), signal.SIGKILL)
        sys.exit(130)
if __name__ == '__main__': main()
