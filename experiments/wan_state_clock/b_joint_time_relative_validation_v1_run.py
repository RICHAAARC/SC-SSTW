"""User-run fresh single-content time confirmation; consumed attempts are read-only."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.wan.b_joint_time_relative_validation_v1 import DEFAULT_CONFIG, execute_reserved, run, worker
from runtime.wan.native_baseline_storage_v1 import read_json


def main(*, notebook_reservation=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/content/video-wm-joint-relative-v1/work"))
    parser.add_argument("--worker-stage", choices=("prepare", "solve", "replay", "receive"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_stage:
        if args.output is None:
            parser.error("Internal worker requires explicit reserved output")
        result = worker(args.worker_stage, args.output, args.scratch)
    else:
        config = read_json(args.config)
        result = (execute_reserved if notebook_reservation else run)(config, args.output or config["output"], args.scratch)
    print(json.dumps({k: v for k, v in result.items() if k != "operations"}, ensure_ascii=False, indent=2))
    return 0 if result.get("mode") == "READ_ONLY" or result.get("status") in ("COMPLETED", "READ_ONLY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
