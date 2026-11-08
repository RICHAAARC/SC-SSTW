"""CLI for the fixed compact public package."""
import argparse
import json
from pathlib import Path
from .native_baselines_compact_v1 import run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vendor',required=True,type=Path)
    parser.add_argument('--previous-runtime-stopped',action='store_true')
    args=parser.parse_args(argv)
    try:
        result=run(vendor=args.vendor,previous_runtime_stopped=args.previous_runtime_stopped)
        print(json.dumps(result),flush=True);return 0
    except KeyboardInterrupt:return 130


if __name__=='__main__':raise SystemExit(main())
