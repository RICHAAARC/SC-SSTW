"""User-run native baseline paper protocol; same Drive resumes only never-started units."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.wan.native_baselines_paper_v3 import DEFAULT_CONFIG, load_inputs, run, physical_receipt
from runtime.wan.native_baseline_storage_v1 import read_json, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    p.add_argument('--output', type=Path)
    p.add_argument('--vendor', type=Path, default=Path('/content/video-wm-native-baselines-paper-v3/vendor'))
    p.add_argument('--evaluate-only', action='store_true')
    args = p.parse_args()
    config, sources = load_inputs(args.config)
    output = args.output or Path(config['output'])
    if args.evaluate_only:
        import fcntl
        from runtime.wan.native_baseline_evaluation_v3 import final_evaluation
        with (output/'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            sources = read_json(output/'sources.json')
            result = final_evaluation(output, sources)
        print(json.dumps(dict(status=result['status'], output=str(output))))
        return 0
    try:
        result = run(config, sources, output, args.vendor)
    except BaseException as e:
        # Preserve unit records and any specific saved failure; no retry here.
        failure = dict(status='INTERRUPTED_OR_SETUP_FAILED', error=f'{type(e).__name__}: {e}',
                       scientific_pass=None, **physical_receipt(output, sources))
        write_json(output/'interruption.json', failure)
        print(json.dumps(failure, ensure_ascii=False))
        if isinstance(e, KeyboardInterrupt):
            return 130
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
