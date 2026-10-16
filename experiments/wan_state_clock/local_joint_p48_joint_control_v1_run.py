"""Fixed P48 joint-control runner; usable from an editable ZIP without Git."""
import argparse
import json
from pathlib import Path
import signal
from runtime.wan.local_joint_p48_joint_control_v1 import run


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    def interrupted(signum,_frame): raise SystemExit("external signal "+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
    result=run(json.loads(args.config.read_text()),args.output)
    print(json.dumps(dict(status=result["status"],counts=result["counts"],output=str(args.output))),flush=True)
    return 0 if result["status"] in ("COMPLETE","CONSTRUCTION_UNDEFINED","NO_RESOLVED_CANDIDATE") else 1


if __name__=="__main__": raise SystemExit(main())
