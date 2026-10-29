"""User-run saved-candidate diagnostic. Import/help and consumed recovery need only stdlib."""
import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--saved-attempt', required=True)
    parser.add_argument('--truth-source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-id', default=None, help='Optional actual available model id/path')
    args = parser.parse_args()
    from .b_framewise_qim_saved_candidate_v1_experiment import run
    print(json.dumps(run(args.saved_attempt, args.truth_source, args.output, args.model_id), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__': main()
