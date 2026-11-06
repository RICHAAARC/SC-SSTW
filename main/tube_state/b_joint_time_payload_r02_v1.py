"""Adopted single new ocean source; existing time and identity methods unchanged."""
from .b_joint_time_relative_validation_v1 import VIEWS, ARMS

EXPERIMENT = 'joint-time-payload-fresh-r02-v1'
SOURCE = 'joint-relative-r02'
SEED = 2026110602
PROMPT = ('A locked-off wide shot of ocean waves washing over dark rocks under an overcast sky. '
          'Natural continuous water motion, documentary style, with the entire rocky shoreline visible throughout the shot.')
MODEL = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'
OUTPUT = '/content/drive/MyDrive/Video-WM/B-Joint-Time-Payload-R02-V1/fresh-r02-v1/attempt01'


def validate(config):
    source = config['source']
    if (config['experiment'] != EXPERIMENT or source['id'] != SOURCE or source['seed'] != SEED
            or source['prompt'] != PROMPT or source['negative_prompt'] != ''):
        raise ValueError('adopted r02 source/prompt/seed required')
    if config['model_id'] != MODEL or config['generation'] != dict(height=320, width=512, frames=181,
            fps=8, steps=50, guidance_scale=5., max_sequence_length=512):
        raise ValueError('original generation model/geometry/schedule required')
    bits = [int(c) for c in f'{0xA6D39C5E:032b}']
    if config['writer'] != dict(key='watermark', gamma=1, bits=bits):
        raise ValueError('adopted DCT2 key/message/gamma required')
    if config['views'] != list(VIEWS) or config['arms'] != list(ARMS):
        raise ValueError('all five views and paired arms required')
    return config
