"""CPU saved-JSON entry; no model, media, download or scoring interface."""
import argparse,json
from runtime.wan.b_rank3r_adjacent_origin_v1 import run,compact

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',required=True,help='local fixed four-group JSON path roster')
    parser.add_argument('--output',required=True,help='new independent one-attempt output directory')
    parser.add_argument('--baseline',required=True,help='saved previous attempt, original-DETECT branches only')
    args=parser.parse_args(argv)
    result=run(args.catalog,args.output,args.baseline)
    print(json.dumps(compact(result,args.output),ensure_ascii=False,indent=2))
    return 0

if __name__=='__main__':raise SystemExit(main())
