"""CPU-only e02 fixed ten-view receiver. Consumed outputs recover before science."""
import argparse
import json
import signal
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qim-media')
    parser.add_argument('--base-media')
    parser.add_argument('--output')
    parser.add_argument('--config')
    args=parser.parse_args()
    from . import b_minimal_l_v1_records as records
    if args.output and records.output_state(args.output)=='CONSUMED':
        result=records.recover(args.output)
    else:
        cfg={} if args.qim_media and args.base_media and args.output else records.load(
            args.config or Path(__file__).with_name('configs')/'b_minimal_l_v1.json')
        def interrupted(signum,frame): raise KeyboardInterrupt('worker terminated')
        signal.signal(signal.SIGTERM,interrupted)
        from .b_minimal_l_v1_experiment import run
        result=run(args.qim_media or cfg['qim_media'],args.base_media or cfg['base_media'],args.output or cfg['output'])
    if result.get('status')=='READ_ONLY_EXISTING_ATTEMPT':
        result={k:result[k] for k in ('status','physical_calls_this_invocation','denominators','read_errors')}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__': main()
