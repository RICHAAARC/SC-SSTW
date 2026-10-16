"""Run the adopted 24-row local-clip diagnosis; real models are user-run."""
import argparse
import json
from pathlib import Path
from experiments.paper_results_v1 import local_clip_diagnostic_v1 as run


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--source-run', default=run.SOURCE_DEFAULT)
    p.add_argument('--model-path', default='')
    p.add_argument('--device', default='cuda')
    p.add_argument('--phase', choices=['init','prepare','wan','report','all'], default='all')
    args = p.parse_args(argv)
    if args.phase == 'init' or (args.phase == 'all' and not run.state_exists(Path(args.output)/'local_clip_state.json')):
        store = run.initialize(args.output, args.source_run, args.model_path, args.device)
    else:
        store = run.Store(args.output)
    if args.phase == 'prepare':
        run.prepare(store)
    elif args.phase == 'wan':
        run.run_wan(store)
    elif args.phase == 'all':
        run.run_all(store)
    print(json.dumps(run.report(store), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
