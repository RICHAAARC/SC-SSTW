"""Fixed adopted full-only user CLI. No model work on import."""
import argparse,json
from runtime.wan.b_block2_carrier_v1 import run,SOURCE,OUTPUT

def main():
    p=argparse.ArgumentParser(description='Four seen-host same-P0 RANDOM versus BLOCK2 full-only carrier comparison')
    p.add_argument('--source',default=SOURCE);p.add_argument('--output',default=OUTPUT)
    a=p.parse_args();print(json.dumps(run(a.source,a.output),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
