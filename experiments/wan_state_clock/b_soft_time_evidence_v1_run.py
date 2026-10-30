"""Independent user-run CPU offline diagnostic CLI; recovery is stdlib only."""
import argparse
import json
from pathlib import Path
import signal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('configs')/'b_soft_time_evidence_v1.json')
    parser.add_argument('--source-attempt');parser.add_argument('--output');args=parser.parse_args()
    from .b_soft_time_evidence_v1_records import output_state,recover,load
    if args.output and output_state(args.output)=='CONSUMED':result=recover(args.output)
    else:
        config={} if args.source_attempt and args.output else load(args.config)
        source=args.source_attempt or config['source_attempt'];output=args.output or config['output']
        def stop(signum,frame):raise KeyboardInterrupt('worker terminated')
        signal.signal(signal.SIGTERM,stop)
        from .b_soft_time_evidence_v1_experiment import run
        result=run(source,output)
    if result.get('status')=='READ_ONLY_EXISTING_ATTEMPT':result={k:result[k] for k in ('status','scientific_calls_this_invocation','denominators','read_errors')}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
