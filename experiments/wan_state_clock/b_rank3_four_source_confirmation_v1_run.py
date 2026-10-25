"""Thin user-run entry; existing results never import the physical adapter."""
import argparse
import json
from runtime.wan import b_rank3_four_source_confirmation_v1 as run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=run.DEFAULT_INPUT)
    parser.add_argument('--output',default=run.DEFAULT_OUTPUT)
    args=parser.parse_args()
    result=run.run(args.input,args.output)
    print(json.dumps(run.compact_summary(result,args.output),ensure_ascii=False,indent=2))
    return 0 if result['status'] in ('COMPLETED_FIXED_ATTEMPT','READ_ONLY_EXISTING_ATTEMPT') else 1

if __name__=='__main__':raise SystemExit(main())
