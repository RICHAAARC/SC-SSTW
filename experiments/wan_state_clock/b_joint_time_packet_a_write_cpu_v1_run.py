"""Explicit local stage-P entry. Neither subcommand runs a model forward."""
import argparse
import json
from pathlib import Path
from . import b_joint_time_packet_a_write_cpu_v1_experiment as experiment
from . import b_joint_time_packet_a_write_cpu_v1_records as records


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='stage', required=True)
    head = sub.add_parser('prepare-head'); head.add_argument('--model-directory', required=True); head.add_argument('--output', required=True)
    solve = sub.add_parser('solve'); solve.add_argument('--config', required=True); solve.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.stage == 'prepare-head': result = experiment.prepare_head(args.model_directory, args.output)
    elif (Path(args.output).expanduser().resolve()/'attempt.json').exists():
        result = records.recover(Path(args.output).expanduser().resolve())
    else:
        with open(args.config) as stream: config = json.load(stream)
        result = experiment.run(config, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__': main()
