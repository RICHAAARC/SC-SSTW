"""Fixed stage1 reference user CLI; model work only inside consumed-attempt run."""
import argparse,json
from runtime.wan.b_lfg_posthoc_reference_v1 import run
from main.tube_state.b_lfg_posthoc_plan_v1 import SOURCE,OUTPUT

def main():
    parser=argparse.ArgumentParser(description='Fixed full181 LFG public-formula K0 reference, no temporal IDs or sync')
    parser.add_argument('--source',default=SOURCE);parser.add_argument('--output',default=OUTPUT)
    args=parser.parse_args();print(json.dumps(run(args.source,args.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
