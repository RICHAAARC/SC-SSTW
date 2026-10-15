"""Standalone fixed four-source Run all and staged workers; no notebook required."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_results_v1 import independent_four_v1 as task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=task.CONFIG_PATH)
    parser.add_argument('--phase', choices=('init', 'all', 'report', *task.STAGES), default='all')
    parser.add_argument('--session-id', default='manual-stage')
    args = parser.parse_args(argv)
    config = task.configuration(args.config)
    if args.phase == 'all':
        result = task.run_all(args.output, config)
    elif args.phase in ('init', 'report'):
        result = task.report(task.initialize(args.output, config))
    else:
        task.run_stage(args.output, config, args.phase, args.session_id)
        result = {'stage': args.phase, 'status': 'RETURNED'}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
