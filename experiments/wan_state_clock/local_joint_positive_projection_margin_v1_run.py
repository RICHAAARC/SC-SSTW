"""Fixed user-run positive-margin VAE/MP4 pair; no generation or retry."""
import argparse
import json
from pathlib import Path

from runtime.wan.local_joint_positive_projection_margin_v1 import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent/"configs/local_joint_positive_projection_margin_v1.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run(config["source_terminal"], config["output"], config["model_id"])
    print(json.dumps(dict(status=result["status"], output=config["output"],
                         counts=result.get("counts", result.get("saved", {}).get("counts"))), indent=2))
    return 0 if result["status"] in ("EXECUTION_COMPLETE", "EXISTING_ATTEMPT_NOT_REEXECUTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
