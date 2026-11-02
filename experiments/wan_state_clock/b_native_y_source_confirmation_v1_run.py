"""Standalone n01 user-run package; consumption recovery precedes dependencies."""
from pathlib import Path
import sys
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = 'experiments.wan_state_clock'


def main(argv=None):
    import argparse
    import json
    import signal
    from . import b_native_y_source_confirmation_v1_records as rec
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(Path(__file__).with_name('configs')/'b_native_y_source_confirmation_v1.json'))
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    if args.output and rec.output_state(args.output) == 'CONSUMED':
        result = rec.recover(args.output)
    else:
        from .b_native_y_source_confirmation_v1_experiment import run
        def stop(signum, frame): raise KeyboardInterrupt('fixed n01 attempt terminated')
        signal.signal(signal.SIGTERM, stop)
        result = run(rec.load(args.config), args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('saved','stages')}, ensure_ascii=False, allow_nan=False))
    return result


if __name__ == '__main__': main()
