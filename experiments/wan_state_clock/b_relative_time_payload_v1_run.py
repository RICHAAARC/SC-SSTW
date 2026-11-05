"""Complete user-run fixed551 experiment; repeat invocation is read-only including post."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main():
    from runtime.wan import b_relative_time_payload_records_v1 as rec
    from runtime.wan.b_relative_time_payload_runner_v1 import DEFAULT_CONFIG, run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if args.output is None: parser.error('reserved output required')
        from runtime.wan.b_relative_time_payload_worker_v1 import worker
        value = worker(args.output)
    elif args.output is not None and args.output.exists() and any(args.output.iterdir()):
        value = rec.recover(args.output)
    else:
        value = run(rec.read(args.config), args.output)
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0 if value.get('status') in ('COMPLETED','READ_ONLY') else 1


if __name__ == '__main__':
    raise SystemExit(main())
