"""Standalone public segmented-payload entry; no synchronization estimator.

Run as a module or directly from a source copy, including without .git.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from main.tube_state import segmented_payload_v1 as payload


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path,value):
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def save_read(output,result):
    import numpy as np
    arrays={name:result[name] for name in ('signed_votes','zero_coefficient_mask')}
    np.savez(output/'votes.npz',**arrays)
    save_json(output/'read.json',{k:v for k,v in result.items() if k not in arrays})


def validate_generation_config(config):
    generation=config['generation']
    fixed={'steps':50,'frames':181,'height':320,'width':512}
    if any(type(generation.get(k)) is not int or generation[k]!=v for k,v in fixed.items()):
        raise ValueError('fixed 50-step / 181x320x512 generation required')
    if type(generation.get('seed')) is not int or generation['seed']<0:
        raise ValueError('nonnegative integer generation seed required')
    if type(generation.get('max_sequence_length')) is not int or generation['max_sequence_length']<=0:
        raise ValueError('positive prompt sequence length required')
    if any(not isinstance(generation.get(k),str) for k in ('prompt','negative_prompt')):
        raise ValueError('prompt and negative_prompt strings required')
    if not isinstance(config['model'].get('id'),str) or not config['model']['id']:
        raise ValueError('loadable Wan model id/path required')
    writer=config['writer']
    if not isinstance(writer.get('key'),str) or not writer['key']:
        raise ValueError('nonempty public coordinate key required')
    bits=writer.get('bits')
    if not isinstance(bits,list) or len(bits)!=32 or any(type(b) is not int or b not in (0,1) for b in bits):
        raise ValueError('exactly 32 binary writer bits required')
    return writer


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','read-votes','read-latent','read-rgb'):
        p=sub.add_parser(name)
        p.add_argument('--source-map',required=True,help='JSON array of received-to-source frame indices, caller supplied')
        p.add_argument('--output',required=True,help='new output directory; existing results are not overwritten')
        if name!='plan':p.add_argument('--input',required=True)
        if name in ('read-latent','read-rgb'):p.add_argument('--key',required=True)
        if name=='read-rgb':
            p.add_argument('--config',required=True,help='JSON containing model.id (optional model.revision)')
            p.add_argument('--device',default='cuda')
    p=sub.add_parser('generate-latent')
    p.add_argument('--config',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--dtype',choices=('float32','float16','bfloat16'),default='bfloat16')
    args=parser.parse_args(argv)
    # Validate the actual inputs before model loading or creating output artifacts.
    if args.command=='generate-latent':
        config=load_json(args.config);writer=validate_generation_config(config)
    else:
        path=load_json(args.source_map)
        if not isinstance(path,list):raise ValueError('source map must be a JSON array')
        plan=payload.segment_plan(path)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    counts={}
    def count(kind,returned):
        row=counts.setdefault(kind,{'attempted':0,'returned':0})
        row['returned' if returned else 'attempted']+=1
    try:
        if args.command=='plan':
            save_json(output/'plan.json',plan)
        elif args.command=='read-votes':
            import numpy as np
            with np.load(args.input,allow_pickle=False) as votes:
                result=payload.decode_signed_votes(plan,votes['signed_votes'],votes['zero_coefficient_mask'])
            save_read(output,result)
        elif args.command=='read-latent':
            import numpy as np
            import torch
            normalized=torch.from_numpy(np.load(args.input,allow_pickle=False))
            save_read(output,payload.read_segments(normalized,args.key,path))
        elif args.command=='read-rgb':
            import numpy as np
            import torch
            from runtime.wan.conditional_joint.generation import load_frozen_vae
            from runtime.wan.segmented_payload_v1 import read_received_frames
            rgb=torch.from_numpy(np.load(args.input,allow_pickle=False))
            if rgb.dtype!=torch.uint8 or tuple(rgb.shape)!=(len(path),320,512,3):
                raise ValueError('uint8 received RGB [len(map),320,512,3] required')
            count('vae_load',False)
            vae=load_frozen_vae(load_json(args.config),device=args.device)
            count('vae_load',True);count('vae_encode',False)
            result=read_received_frames(vae,rgb,args.key,path)
            count('vae_encode',True);save_read(output,result)
        else:
            import numpy as np
            import torch
            from runtime.wan.conditional_joint.generation import prepare_generation
            from runtime.wan.segmented_payload_v1 import run_trajectory
            count('prepare_generation',False)
            pipe,initial,prompt,negative,dtype=prepare_generation(config,load_vae=False,
                device=args.device,model_dtype=getattr(torch,args.dtype))
            count('prepare_generation',True)
            with (output/'steps.jsonl').open('x',encoding='utf-8') as steps:
                def record(row):
                    steps.write(json.dumps(row,allow_nan=False)+'\n');steps.flush()
                terminal,receipt=run_trajectory(pipe,initial,pipe.scheduler,prompt,negative,dtype,
                    writer['key'],writer['bits'],count,record)
            np.save(output/'terminal.npy',terminal.float().numpy(),allow_pickle=False)
            save_json(output/'generation.json',receipt)
        save_json(output/'execution.json',dict(status='COMPLETE',command=args.command,calls=counts,
            scientific_pass=None,blind_sync_performed=False))
    except Exception as exc:
        save_json(output/'execution.json',dict(status='FAILED',command=args.command,calls=counts,
            error=f'{type(exc).__name__}: {exc}',automatic_retry=False))
        raise
    print(str(output))


if __name__=='__main__':
    main()
