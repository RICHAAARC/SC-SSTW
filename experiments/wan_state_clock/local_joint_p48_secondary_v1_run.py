"""P48 fixed-secondary continuation report; the recorded CPU attempt is unresolved."""
import argparse
import json
from pathlib import Path
from runtime.wan.local_joint_p48_secondary_v1 import run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--decision',type=Path,required=True)
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    r=run(config['saved_run'],args.output,args.decision)
    print(json.dumps(dict(status=r['status'],new_stage_calls=r['actual_new_stage_calls'],scientific_pass=False)),flush=True)
    return 0 if r['status']=='NO_RESOLVED_CANDIDATE' else 1


if __name__=='__main__':raise SystemExit(main())
