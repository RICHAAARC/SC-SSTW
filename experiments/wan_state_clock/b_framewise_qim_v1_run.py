"""Future real CLI. Import/--help are inert; running spends one fixed attempt."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,help='Existing B-Terminal-Three-Group-V1/fixed-four-new-v1 directory')
    parser.add_argument('--output',required=True,help='New output folder; existing attempt is read-only')
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs/b_framewise_qim_v1.json'))
    args = parser.parse_args()
    from experiments.wan_state_clock.b_framewise_qim_v1_records import load
    from experiments.wan_state_clock.b_framewise_qim_v1_experiment import run
    print(json.dumps(run(load(args.config),args.source,args.output),ensure_ascii=False,allow_nan=False))


if __name__ == '__main__': main()
