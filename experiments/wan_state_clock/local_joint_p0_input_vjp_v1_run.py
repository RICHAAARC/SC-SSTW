"""Standalone P0 CLI: saved input, two branch VJPs, no notebook/Git dependency."""
import argparse
import json
from pathlib import Path
import signal

from runtime.wan.local_joint_p0_input_vjp_v1 import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, _frame):
        raise SystemExit("external signal " + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    result = run(json.loads(args.config.read_text()), args.output)
    print(json.dumps(dict(status=result["status"], counts=result["counts"],
                          blocks=result["block_counts"], output=str(args.output))), flush=True)
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
