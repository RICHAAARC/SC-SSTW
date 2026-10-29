"""Independent CPU ten-view user-run CLI; stdlib help and consumed recovery."""
import argparse
import json
import signal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qim-media',required=True)
    parser.add_argument('--base-media',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    def interrupted(signum,frame): raise KeyboardInterrupt('worker terminated')
    signal.signal(signal.SIGTERM,interrupted)
    from .b_framewise_qim_time_recovery_v1_experiment import run
    result=run(args.qim_media,args.base_media,args.output)
    if result.get('status')=='READ_ONLY_EXISTING_ATTEMPT':
        result={k:result[k] for k in ('status','physical_calls_this_invocation','denominators','read_errors')}
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__': main()
