"""Fixed user-run entrypoint. --help never imports model backend."""
import argparse,json
from runtime.wan.b_affine_margin_observation_v3 import run
from main.tube_state import b_affine_margin_plan_v3 as p
def main():
 parser=argparse.ArgumentParser(description='One attempt s01 P0/OLD/NEW V3; no retry, no source generation.')
 parser.add_argument('--p0',default=p.P0);parser.add_argument('--old',default=p.OLD);parser.add_argument('--output',default=p.OUTPUT);parser.add_argument('--truth',default=p.TRUTH)
 args=parser.parse_args();print(json.dumps(run(args.p0,args.old,args.output,args.truth),indent=2))
if __name__=='__main__':main()
