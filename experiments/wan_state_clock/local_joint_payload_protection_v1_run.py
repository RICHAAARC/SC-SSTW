"""Card51 user-run fixed three-arm terminal/MP4 package."""
import argparse
import json
from pathlib import Path

from runtime.wan.local_joint_payload_protection_v1 import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).parent/"configs/local_joint_payload_protection_v1.json")
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    result=run(config["source_p"],config["source_j0"],config["output"],config["model_id"])
    print(json.dumps(dict(status=result["status"],output=config["output"],counts=result.get("counts",{}),
                          cpu_counts=result.get("cpu_counts",{})),ensure_ascii=False,indent=2))
    return 0 if result["status"] in ("EXECUTION_COMPLETE","EXISTING_ATTEMPT_NOT_REEXECUTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
