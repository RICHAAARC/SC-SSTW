"""Card99 single fixed four-unseen-source user run; existing attempt is read-only."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from runtime.wan.b_absolute_confirmation_v1 import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=None,help='durable output directory; keep the same directory for later read-only Run all')
    args=parser.parse_args()
    print(json.dumps(run(args.output),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
