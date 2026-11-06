"""Complete new-source r02 chain; a consumed output is always read-only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))


def main():
    from runtime.wan import b_joint_time_payload_r02_records_v1 as rec
    from runtime.wan import b_joint_time_payload_r02_v1 as runner
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=runner.DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scratch', type=Path, default=Path('/content/video-wm-joint-time-payload-r02/work'))
    parser.add_argument('--reserved', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--time-stage', choices=('prepare','solve','replay','receive'), help=argparse.SUPPRESS)
    parser.add_argument('--payload-stage', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.time_stage or args.payload_stage or args.reserved:
        if args.output is None: parser.error('internal entry requires explicit reserved output')
        with runner.termination_handler():
            if args.time_stage:
                config = rec.read(args.output/'protocol.json'); runner.validate(config)
                result = runner.temporal.worker(args.time_stage, args.output/'time_chain', args.scratch)
            elif args.payload_stage: result = runner.payload_stage(args.output)
            else: result = runner.execute_reserved(args.output, args.scratch)
    elif args.output and args.output.exists() and any(args.output.iterdir()):
        result = rec.recover(args.output)
    else:
        config = rec.read(args.config)
        result = runner.run(config, args.output, args.scratch)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get('status') in ('COMPLETED','READ_ONLY','STOPPED_FIXED_ATTEMPT') else 1


if __name__ == '__main__': raise SystemExit(main())
