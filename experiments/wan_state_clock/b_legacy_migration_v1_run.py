"""Card77 fixed user-run saved-pixel operator comparison, E12 only."""
import argparse,json
from pathlib import Path
from runtime.wan.b_legacy_migration_v1 import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/b_legacy_migration_v1.json')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    history=Path(config['legacy_summary'])
    if not history.is_absolute():history=Path(__file__).resolve().parents[2]/history
    result=run(config['c73_root'],config['output'],history,config['model_id'])
    print(json.dumps(dict(status=result['status'],output=config['output'],counts=result.get('counts')),indent=2))
    return 0 if result['status'] in ('EXECUTION_COMPLETE','EXISTING_ATTEMPT_NOT_REEXECUTED') else 1

if __name__=='__main__':raise SystemExit(main())
