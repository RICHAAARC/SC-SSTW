"""User-run full package; any consumed package returns before config/scientific imports."""
import argparse
import json
import signal
import sys
from pathlib import Path

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
    __package__='experiments.wan_state_clock'


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--output');args=parser.parse_args(argv)
    from . import b_conditional_time_validation_v1_records as records
    if args.output and records.output_state(args.output)=='CONSUMED':
        result=records.recover(args.output)
    else:
        config=records.load(args.config or Path(__file__).with_name('configs')/'b_conditional_time_validation_v1.json')
        def interrupted(signum,frame):raise KeyboardInterrupt('user-run package terminated')
        signal.signal(signal.SIGTERM,interrupted)
        from .b_conditional_time_validation_v1_experiment import run
        result=run(config,args.output)
    if result.get('status')=='READ_ONLY_EXISTING_PACKAGE':
        result={k:result[k] for k in ('status','physical_calls_this_invocation','phases','read_errors')}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    return result


if __name__=='__main__':main()
