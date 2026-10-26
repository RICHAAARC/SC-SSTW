"""Run the single fixed M3 attempt; existing results are always read-only."""
import argparse,json
from runtime.wan import b_rank3r_new_source_confirmation_v1 as run

def _dispatch(start):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=run.DEFAULT_INPUT)
    parser.add_argument('--output',default=run.DEFAULT_OUTPUT)
    args=parser.parse_args();result=start(args.input,args.output)
    print(json.dumps(run.compact_summary(result,args.output),ensure_ascii=False,indent=2))
    return 0 if result['status'] in ('COMPLETED_FIXED_ATTEMPT','READ_ONLY_EXISTING_ATTEMPT') else 1

def main():return _dispatch(run.run)

def _notebook_reserved_main():
    """Internal live handoff only; never selected by a public CLI flag."""
    return _dispatch(run._start_reserved)

if __name__=='__main__':raise SystemExit(main())
