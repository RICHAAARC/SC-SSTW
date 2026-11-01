"""Standalone saved-media reader; consumed output recovers before scientific imports."""
import argparse
import json
import signal
import sys
from pathlib import Path
if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]));__package__='experiments.wan_state_clock'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config');parser.add_argument('--output')
    args=parser.parse_args(argv)
    from . import b_new_source_chain_confirmation_v1_records as records
    if args.output and records.output_state(args.output)=='CONSUMED':result=records.recover(args.output)
    else:
        cfg=records.load(args.config or Path(__file__).with_name('configs')/'b_new_source_chain_confirmation_v1.json')
        def interrupted(signum,frame):raise KeyboardInterrupt('user-run receiver terminated')
        signal.signal(signal.SIGTERM,interrupted)
        from .b_new_source_chain_confirmation_v1_experiment import run
        result=run(cfg,args.output)
    print(json.dumps(dict(status=result.get('status'),evaluation_status=result.get('evaluation_status'),output=args.output or result.get('output'),physical_calls_this_invocation=result.get('physical_calls_this_invocation')),ensure_ascii=False));return result


if __name__=='__main__':main()
