"""Run the fixed native baseline batch or read an already-consumed attempt."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.wan.native_baselines_temporal_v2 import DEFAULT_CONFIG, execute_reserved, run
from runtime.wan.native_baseline_storage_v1 import read_json


def main(*, notebook_reservation=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scratch", type=Path, default=Path("/content/video-wm-native-baselines-temporal-v2/work"))
    parser.add_argument("--vendor", type=Path, default=Path("/content/video-wm-native-baselines-temporal-v2/vendor"))
    args = parser.parse_args()
    config = read_json(args.config)
    output = args.output or Path(config["output"])
    function = execute_reserved if notebook_reservation else run
    result = function(config, output, args.scratch, args.vendor)
    print(json.dumps({k: v for k, v in result.items() if k != "operations"}, ensure_ascii=False, indent=2))
    return 0 if result.get("mode") == "READ_ONLY" or result.get("status") == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
