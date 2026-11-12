"""Standalone CLI; assemble saved records or execute the user-run quality package."""
import argparse
import json
from main.tube_state import paper_evidence_supplement_v1 as p


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    saved=commands.add_parser('assemble',help='read-only arithmetic over saved small records')
    for key in ('main-assembly','wan-root','frozen-root','output'): saved.add_argument('--'+key,required=True)
    saved.add_argument('--compact-root')
    quality=commands.add_parser('quality',help='one consumed MUSIQ-SPAQ package; user Colab execution')
    quality.add_argument('--result-root',default=p.ROOT)
    quality.add_argument('--output',default=p.ROOT+'/paper_evidence_supplement_v1/attempt01/quality')
    quality.add_argument('--cache',default='/content/paper-evidence-supplement-cache')
    args=vars(parser.parse_args(argv));command=args.pop('command')
    if command=='assemble':
        from .paper_evidence_supplement_report_v1 import assemble
        value=assemble(**args)
        display={k:value[k] for k in ('status','fixed_counts','theory','new_execution','real_quality')}
    else:
        from .paper_evidence_supplement_quality_v1 import run
        value=run(**args);display=value
    print(json.dumps(display,ensure_ascii=False,indent=2,allow_nan=False))
    return value


if __name__=='__main__': main()
