"""Standalone user CLI; no notebook, Git metadata, or previous scientific cache required."""
from pathlib import Path
import argparse
import json
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from runtime.wan.b_fixed_detection_confidence_v1 import run, CONFIG, DEFAULT_OUTPUT


def main(argv=None):
    parser = argparse.ArgumentParser(description='Card94 fixed2cal2confirm, one attempt; existing outputs are read-only')
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--config', default=str(CONFIG))
    args = parser.parse_args(argv)
    result = run(args.output, config_path=args.config)
    saved = result.get('saved', {})
    record = saved.get('result.json', result)
    print(json.dumps(dict(status=result.get('status'), saved_status=record.get('status'),
        thresholds_status=record.get('thresholds_status'), counts=record.get('counts'),
        scientific_pass=None, output=args.output), ensure_ascii=False, indent=2))
    return 0 if result['status'] in ('COMPLETED_EMPIRICAL_ONLY', 'EXISTING_ATTEMPT_NOT_REEXECUTED') else 1


if __name__ == '__main__':
    raise SystemExit(main())
