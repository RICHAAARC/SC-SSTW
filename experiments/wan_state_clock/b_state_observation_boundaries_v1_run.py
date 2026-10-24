"""Run the adopted saved-input 12-encode diagnostic once; later calls only read."""
import argparse
import json
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.wan.b_state_observation_boundaries_v1 import DEFAULT_INPUT,DEFAULT_OUTPUT,run

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',default=DEFAULT_INPUT)
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    result=run(args.input,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0 if result['status'] in ('COMPLETED_ENGINEERING_RECORDS','READ_ONLY_EXISTING_ATTEMPT') else 1
if __name__=='__main__':raise SystemExit(main())
