"""Direct user CLI; no implicit execution on import."""
import argparse,json
from runtime.wan.b_single_internal_deletion_v1 import run,DEFAULT_OUTPUT,DEFAULT_SOURCE,CONFIG
def main():
    p=argparse.ArgumentParser(description='One fixed viewed-development real deletion task; user physical execution only.')
    p.add_argument('--source',default=DEFAULT_SOURCE);p.add_argument('--output',default=DEFAULT_OUTPUT)
    p.add_argument('--config',default=str(CONFIG));args=p.parse_args()
    print(json.dumps(run(args.source,args.output,args.config),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
