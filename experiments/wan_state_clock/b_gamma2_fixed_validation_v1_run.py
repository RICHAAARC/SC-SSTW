"""Fixed gamma2 user-run entry; --help and existing recovery are offline."""
import argparse
import json
import sys
from pathlib import Path
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.wan.b_gamma2_fixed_validation_v1 import DEFAULT_INPUT,DEFAULT_OUTPUT,run,compact_summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=DEFAULT_INPUT)
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    args=parser.parse_args();result=run(args.input,args.output)
    print(json.dumps(compact_summary(result,args.output),ensure_ascii=False,allow_nan=False))
    return 0 if result['status'] in ('COMPLETED_ENGINEERING_RECORDS','READ_ONLY_EXISTING_ATTEMPT') else 1
if __name__=='__main__':raise SystemExit(main())
