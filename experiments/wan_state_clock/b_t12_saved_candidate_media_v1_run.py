"""Independent saved-candidate CLI; --help and preflight use only stdlib."""
import argparse
import json
from . import b_t12_saved_candidate_media_v1_records as rec


def main():
    parser=argparse.ArgumentParser(description='Declared over-budget saved t12 candidate; user-run native/media observation')
    for key,value in rec.DEFAULTS.items(): parser.add_argument('--'+key.replace('_','-'),default=value)
    parser.add_argument('--preflight',action='store_true',help='Only inspect input readability; no physical calls')
    args=vars(parser.parse_args());preflight=args.pop('preflight')
    if preflight: result=rec.preflight(args)
    else:
        from .b_t12_saved_candidate_media_v1_experiment import run
        result=run(args)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))


if __name__=='__main__': main()
