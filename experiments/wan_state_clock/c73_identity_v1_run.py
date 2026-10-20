"""Card74 fixed C73 user run, fresh D3/local reset E18/FFT0."""
import argparse
import json
from pathlib import Path
from runtime.wan.c73_identity_v1 import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent/"configs/c73_identity_v1.json")
    config = json.loads(parser.parse_args().config.read_text())
    result = run(config["source_p"], config["source_s"], config["output"], config["model_id"])
    print(json.dumps(dict(status=result["status"], output=config["output"], counts=result.get("counts")), indent=2))
    return 0 if result["status"] in ("EXECUTION_COMPLETE", "EXISTING_ATTEMPT_NOT_REEXECUTED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
