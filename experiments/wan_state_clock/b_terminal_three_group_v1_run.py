"""Run the adopted fixed four-source B chain once; existing attempts are read-only."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.wan.b_terminal_three_group_v1 import CONFIG, DEFAULT_BACKGROUND, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None)
    parser.add_argument('--background', default=DEFAULT_BACKGROUND)
    parser.add_argument('--config', default=str(CONFIG))
    args = parser.parse_args()
    value = run(args.output, config_path=args.config, background_path=args.background)
    print(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if value['status'] in ('COMPLETED_ENGINEERING_RECORDS', 'READ_ONLY_EXISTING_ATTEMPT') else 1


if __name__ == '__main__':
    raise SystemExit(main())
