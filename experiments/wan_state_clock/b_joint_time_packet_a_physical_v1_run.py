"""Standalone staged CLI. Consumed outputs recover before reading input config."""
from pathlib import Path
import argparse
import json
from . import b_joint_time_packet_a_physical_v1_records as rec


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('gpu1','cpu_media','gpu2','post'))
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();output=Path(args.output).expanduser().resolve()
    if (output/'attempt.json').exists():result=rec.recover(output)
    else:
        config=rec.load(args.config)
        if args.stage in ('gpu1','gpu2'):
            from . import b_joint_time_packet_a_gpu_v1 as experiment
            result=getattr(experiment,args.stage)(config,output)
        elif args.stage=='cpu_media':
            from .b_joint_time_packet_a_cpu_media_v1 import run
            result=run(config,output)
        else:
            from .b_joint_time_packet_a_post_v1 import run
            result=run(config,output)
    print(json.dumps(dict(stage=args.stage,status=result.get('status'),stage_terminal=result.get('stage_terminal'),
        errors=result.get('errors',{}),result_file=str(output/'result.json')),ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':main()
