"""Card89 fixed user-only E6/FFT6 completion; no calibration or publication."""
import argparse
import json
from pathlib import Path
from runtime.wan.b_real_four_phase_v1 import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).parent/'configs/b_real_four_phase_v1.json')
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    sidecar=Path(config['saved_sidecar'])
    if not sidecar.is_absolute():
        sidecar=Path(__file__).resolve().parents[2]/sidecar
    result=run(config['source_root'],config['output'],sidecar,config['truth_path'],config['model_id'])
    print(json.dumps(dict(status=result['status'],output=config['output'],counts=result.get('counts'),
        new_CPU=result.get('actual_new_CPU'),new_observation=result.get('actual_new_observation'),
        identity_status='UNCALIBRATED',phase_status='UNCALIBRATED'),ensure_ascii=False,indent=2))
    return 0 if result['status'] in ('EXECUTION_COMPLETE','NUMERIC_COMPLETE_POST_UNAVAILABLE','EXISTING_ATTEMPT_NOT_REEXECUTED') else 1


if __name__=='__main__':
    raise SystemExit(main())
