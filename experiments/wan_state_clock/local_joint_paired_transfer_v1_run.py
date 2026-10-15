"""Standalone paired-transfer CLI; no notebook or Git dependency."""
import argparse
import json
from pathlib import Path
import signal
from runtime.wan.local_joint_paired_transfer_v1 import run,FINISHED

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    def interrupted(signum,_frame):raise SystemExit(f"external signal {signum}")
    signal.signal(signal.SIGTERM,interrupted)
    result=run(json.loads(args.config.read_text()),args.output,
        source_identity=dict(workspace=str(Path(__file__).resolve().parents[2]),config_path=str(args.config)))
    print(json.dumps(dict(status=result["status"],counts=result["counts"],output=str(args.output))),flush=True)
    return 0 if result["status"] in FINISHED else 1

if __name__=="__main__":raise SystemExit(main())
