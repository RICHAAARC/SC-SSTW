"""User-run GPU materialization A then B. No local receiver auto-execution."""
import argparse
import json
import signal


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='/content/drive/MyDrive/Video-WM/B-Terminal-Three-Group-V1/fixed-four-new-v1')
    parser.add_argument('--output',default='/content/drive/MyDrive/Video-WM/B-Minimal-L-V1/fixed-e02-materialization-v1')
    parser.add_argument('--model-id')
    args=parser.parse_args()
    def interrupted(signum,frame): raise KeyboardInterrupt('worker terminated')
    signal.signal(signal.SIGTERM,interrupted)
    from .b_minimal_l_materialization_v1_experiment import run
    print(json.dumps(run(args.source,args.output,args.model_id),ensure_ascii=False,allow_nan=False))


if __name__=='__main__': main()
