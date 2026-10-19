"""Card58 fixed saved-RGB four-phase blind temporal receiver, one attempt."""
import argparse
import json
from pathlib import Path
from runtime.wan.blind_temporal_segmented_v1 import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'configs/blind_temporal_segmented_v1.json')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    result = run(**config)
    print(json.dumps(dict(status=result['status'], output=config['output'], counts=result.get('counts')), indent=2))
    return 0 if result['status'] in ('EXECUTION_COMPLETE', 'EXISTING_ATTEMPT_NOT_REEXECUTED', 'EXISTING_ATTEMPT_WITHOUT_CHECKPOINT') else 1


if __name__ == '__main__':
    raise SystemExit(main())
