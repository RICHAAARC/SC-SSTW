"""Fixed clean P0/PM directional-transfer evidence run; user model/media execution."""
import argparse,json
from pathlib import Path
from runtime.wan.b_legacy_position_margin_v1 import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/b_legacy_position_margin_v1.json')
    args=parser.parse_args();c=json.loads(args.config.read_text())
    result=run(c['source_p'],c['output'],c['truth_path'],c['model_id'],c['gamma'])
    print(json.dumps(dict(status=result['status'],output=c['output'],counts=result.get('counts'),
                         identity_status='UNCALIBRATED',phase_status='UNCALIBRATED'),indent=2))
    return 0 if result['status'] in ('EXECUTION_COMPLETE','EXISTING_ATTEMPT_NOT_REEXECUTED') else 1

if __name__=='__main__':raise SystemExit(main())
