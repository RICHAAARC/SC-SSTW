"""Fixed user-run entry; a separate blind process never imports evaluation truth."""
import argparse
import json
import signal
import subprocess
import sys
from experiments.paper_results_v1.single_read_v1 import OUTPUT, has_attempt, saved_status, run_blind


def main():
    parser = argparse.ArgumentParser(description='One saved-plan RAW read; no retry or automatic continuation')
    parser.add_argument('--output', default=OUTPUT)
    parser.add_argument('--blind-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.blind_worker:
        def stop(signum, frame): raise InterruptedError(f'process signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        result = run_blind(args.output)
        print(json.dumps(result, ensure_ascii=False))
        if 'action' in result: raise SystemExit(8)
        return
    if has_attempt(args.output):
        print(json.dumps(saved_status(args.output), ensure_ascii=False, indent=2)); return
    child = subprocess.Popen([sys.executable, '-u', '-m', __spec__.name, '--output', args.output, '--blind-worker'])
    interrupted = None
    try: child.wait()
    except BaseException as exc:
        interrupted = exc; child.terminate()
        try: child.wait(timeout=10)
        except subprocess.TimeoutExpired: child.kill(); child.wait()
    if child.returncode == 8:
        print(json.dumps(saved_status(args.output), ensure_ascii=False, indent=2)); return
    from experiments.paper_results_v1.single_read_v1_evaluation import evaluate_saved
    try:
        summary = evaluate_saved(args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(json.dumps(dict(evaluation_error=f'{type(exc).__name__}: {exc}', saved=saved_status(args.output)), ensure_ascii=False))
        raise
    if interrupted is not None: raise interrupted
    if summary['status'] != 'READ': raise SystemExit('Fixed row retained without retry; inspect saved summary and state.')


if __name__ == '__main__': main()
