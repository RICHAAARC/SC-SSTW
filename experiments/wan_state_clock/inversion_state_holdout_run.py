"""Two previously fixed holdout cases; state method and complete media path unchanged."""
import argparse
import hashlib
from pathlib import Path
from runtime.wan.io import dump
from . import inversion_state_run as state

CASES=('holdout_p0_s0','holdout_p1_s0')
MANIFEST=Path(__file__).parent/'configs/inversion_state_holdout.json'
ROSTER_SOURCE=MANIFEST.with_name('velocity_calibration.json')
MODULE='experiments.wan_state_clock.inversion_state_holdout_run'


def validate():
    manifest=state.base.load(MANIFEST);development=state.base.load(state.MANIFEST)
    fixed=state.base.load(ROSTER_SOURCE)['holdout']
    if manifest['holdout']!=fixed or tuple(c['id'] for c in fixed)!=CASES:
        raise ValueError('holdout must equal the previously locked two-case roster')
    for key,value in development.items():
        if key not in ('protocol','development') and manifest.get(key)!=value:
            raise ValueError('holdout may not change state method parameters: '+key)
    prompts={c['prompt'] for c in development['development']}
    seeds={c['seed'] for c in development['development']}
    if any(c['prompt'] in prompts or c['seed'] in seeds for c in fixed):
        raise ValueError('holdout content and seeds must differ from development')
    if len({c['prompt'] for c in fixed})!=2 or len({c['seed'] for c in fixed})!=2:
        raise ValueError('two distinct new contents and seeds required')
    return manifest


def provenance(result):
    result['split']='holdout'
    result['protocol']='inversion_state_holdout_v1'
    result.setdefault('source_sha256',{}).update({str(p):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__),MANIFEST,ROSTER_SOURCE)})
    result['holdout_claim']='fixed independent contents/seeds for the unchanged state candidate; no tuning, OFF is not calibrated detection, quality and observer gain remain separate'
    return result


def run_case(case_id,output):
    validate()
    result=state.run_case(case_id,output,manifest_path=MANIFEST,cases=CASES,roster_key='holdout')
    provenance(result);dump(Path(output)/'result.json',result);return result


def run_all(output):
    validate()
    result=state.run_all(output,manifest_path=MANIFEST,cases=CASES,child_module=MODULE)
    provenance(result);dump(Path(output)/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case-id',choices=CASES)
    args=parser.parse_args();result=run_case(args.case_id,args.output) if args.case_id else run_all(args.output)
    if result['status']!='EXECUTION_COMPLETE':raise SystemExit(1)
