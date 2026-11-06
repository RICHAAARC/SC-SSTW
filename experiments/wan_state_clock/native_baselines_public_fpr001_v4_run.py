"""Run/resume the fixed four-method public v4 experiment and its CPU tables."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from runtime.wan.native_baselines_public_v4 import DEFAULT_CONFIG,load_inputs,run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--vendor',type=Path,required=True)
    parser.add_argument('--previous-runtime-stopped',action='store_true',help='Confirm a previous different-host runtime has actually stopped; never a heartbeat TTL takeover')
    args=parser.parse_args(argv);config,_=load_inputs(args.config)
    try:
        result=run(output=args.output or config['output'],cache=args.cache or config['cache'],vendor=args.vendor,
            config=args.config,previous_runtime_stopped=args.previous_runtime_stopped)
    except KeyboardInterrupt:return 130
    except Exception as error:
        print(json.dumps(dict(status='INTERRUPTED_RETAINED',error=f'{type(error).__name__}: {error}'),ensure_ascii=False),flush=True)
        return 1
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True);return 0


if __name__=='__main__':raise SystemExit(main())
