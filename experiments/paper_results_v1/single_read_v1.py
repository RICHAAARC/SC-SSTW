"""One adopted blind read. Durable attempt consumption; no automatic continuation."""
from __future__ import annotations
import io
import json
import os
from pathlib import Path
import time

SLOT = 'distinct_confirm_20261017_01/SHORT17_HOLD_MEDIA/DISTINCT_RAW_U'
OUTPUT = '/content/drive/MyDrive/Video-WM/Paper-Results-V1-Single-Read-20261018/fixed-single-read'
P2 = Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Distinct-Confirmation-P2/resumable-distinct-confirmation-p2')
MODEL_CACHE = Path('/content/drive/MyDrive/Video-WM/Paper-Results-V1-Cache/models/Wan2.1-T2V-1.3B-Diffusers')
PLAN_FILE = Path(__file__).with_name('single_read_v1.plan.json')
RGB_SHAPE = (35, 320, 512, 3)
NORMALIZED_SHAPE = (1, 16, 5, 40, 64)
STAGES = ('input', 'load', 'encode', 'read')
SCIENTIFIC_FILES = ('attempt.json', 'state.json', 'blind_plan.json', 'normalized.npy',
                    'votes.npz', 'read_result.json', 'row_result.json', 'summary.json')


def atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def has_attempt(output):
    output = Path(output)
    return any((output / name).exists() or (output / (name + '.tmp')).exists() for name in SCIENTIFIC_FILES)


def saved_status(output):
    """Read only small persisted JSON; never restore arrays or continue the chain."""
    output = Path(output)
    for name in ('summary.json', 'state.json', 'attempt.json'):
        if (output / name).exists():
            try:
                value = json.loads((output / name).read_text())
                return dict(saved_file=str(output / name), saved=value,
                            action='READ_SAVED_ONLY; no continuation or retry')
            except (OSError, ValueError) as exc:
                return dict(status='UNREADABLE_SAVED_RECORD', file=str(output / name),
                            error=str(exc), action='No new physical calls')
    return dict(status='INCOMPLETE_SAVED_ARTIFACTS' if has_attempt(output) else 'NOT_STARTED',
                action='No continuation' if has_attempt(output) else 'One fixed attempt available')


def read_plan():
    from main.tube_state.segmented_payload_v1 import segment_plan
    public = json.loads(PLAN_FILE.read_text())
    if public.get('slot_id') != SLOT or public.get('key') != 'watermark':
        raise ValueError('fixed single-row slot and public key required')
    if public.get('truth_inputs') is not False or public.get('message_inputs') is not False:
        raise ValueError('blind plan cannot carry truth or message inputs')
    plan = public['plan']
    if segment_plan(plan['estimated_path']) != plan:
        raise ValueError('saved plan differs from the actual public segmented-reader semantics')
    if (len(plan['estimated_path']) != 35 or plan['operation']['output_frames'] != 17
            or [s['observed_cells'] for s in plan['segments']] != [2, 2, 0, 0]):
        raise ValueError('fixed 35 received / 17 mapped frames and 2/2/0/0 cells required')
    return public


def choose_input(root=P2):
    root = Path(root)
    received = root / 'data/distinct_confirm_20261017_01/views/SHORT17_HOLD_MEDIA/received.npy'
    post = root / 'media/distinct_confirm_20261017_01/SHORT17_HOLD_MEDIA/candidate_POST.rgb8'
    # Existence selects a container without opening pixels. Once selected, no fallback.
    if received.is_file(): return dict(path=str(received), container='npy')
    if post.is_file(): return dict(path=str(post), container='rgb24')
    raise FileNotFoundError('same-run received.npy and candidate_POST.rgb8 are unavailable')


def read_rgb(chosen):
    import numpy as np
    path = Path(chosen['path'])
    if chosen['container'] == 'npy':
        rgb = np.load(path, allow_pickle=False)
    else:
        payload = path.read_bytes()
        if len(payload) != 17203200: raise ValueError('POST RGB24 must contain exactly 35x320x512x3 bytes')
        rgb = np.frombuffer(payload, dtype=np.uint8).reshape(RGB_SHAPE)
    if rgb.dtype != np.uint8 or tuple(rgb.shape) != RGB_SHAPE:
        raise ValueError('received RGB must be uint8 [35,320,512,3]; no truncation or padding')
    return rgb


def validate_normalized(value):
    import numpy as np
    if value.dtype != np.float32 or tuple(value.shape) != NORMALIZED_SHAPE or not np.isfinite(value).all():
        raise ValueError('finite float32 normalized [1,16,5,40,64] required')


def save_normalized(path, value):
    import numpy as np
    validate_normalized(value)
    buffer = io.BytesIO(); np.save(buffer, value, allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def save_read(output, result, public):
    import numpy as np
    if result.get('plan') != public['plan'] or result.get('truth_inputs') is not False:
        raise ValueError('reader returned another plan or crossed the blind boundary')
    bits = result.get('decoded_bits', [])
    if (len(bits) != 32 or any(type(b) is not int or b not in (0, 1) for b in bits[:16])
            or bits[16:] != [None] * 16 or result.get('decoded_bit_count') != 16):
        raise ValueError('fixed reader must return 16 present bits and 16 structural-missing bits')
    signed, zeros = result['signed_votes'], result['zero_coefficient_mask']
    if (signed.shape != (4, 4, 240) or zeros.shape != (4, 4, 240)
            or signed.dtype != np.int8 or zeros.dtype != np.int8
            or not np.isin(signed, [-1, 1]).all() or not np.isin(zeros, [0, 1]).all()
            or np.any((zeros == 1) & (signed != -1))):
        raise ValueError('invalid new signed votes/zero mask')
    # Persist only the new reader output. No old votes are loaded or reduced.
    buffer = io.BytesIO(); np.savez_compressed(buffer, signed_votes=signed, zero_coefficient_mask=zeros)
    atomic_bytes(Path(output) / 'votes.npz', buffer.getvalue())
    value = {k: v for k, v in result.items() if k not in ('signed_votes', 'zero_coefficient_mask')}
    value.update(slot_id=SLOT, votes_file='votes.npz', persisted_before_evaluation=True)
    write_json(Path(output) / 'read_result.json', value)


class Backend:
    """Same successful P2 FP32 loader, RGB normalization and original segmented read."""
    def load(self):
        # Local existing cache only; directory labels/revisions are not admission gates.
        candidates = [MODEL_CACHE / 'available', MODEL_CACHE]
        if MODEL_CACHE.is_dir(): candidates += sorted(p for p in MODEL_CACHE.iterdir() if p.is_dir())
        path = next((p for p in candidates if (p / 'vae/config.json').is_file()
                     and (list((p / 'vae').glob('*.safetensors')) or list((p / 'vae').glob('*.bin')))), None)
        if path is None: raise FileNotFoundError('existing local Wan VAE cache unavailable; no checkpoint download')
        os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from runtime.wan.conditional_joint.generation import load_frozen_vae
        return load_frozen_vae(dict(model=dict(id=str(path.resolve()))), device='cuda')

    def encode(self, model, rgb):
        import torch
        from runtime.wan.conditional_joint.vae import reencode_rgb24_readback
        return reencode_rgb24_readback(model, torch.from_numpy(rgb).float().div(255)).detach().cpu().float().numpy()

    def read(self, normalized, public):
        import torch
        from main.tube_state.segmented_payload_v1 import read_segments
        return read_segments(torch.from_numpy(normalized), public['key'], public['plan']['estimated_path'])

    def close(self, model):
        import gc
        import torch
        del model; gc.collect(); torch.cuda.empty_cache()


def run_blind(output=OUTPUT, *, backend=None):
    """Whole-chain at-most-once attempt. Existing artifacts are display-only."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if has_attempt(output): return saved_status(output)
    # Exclusive creation precedes all input/model work and is never reset or replaced.
    try:
        with (output / 'attempt.json').open('x') as stream:
            json.dump(dict(slot_id=SLOT, status='STARTED', started_at=time.time(),
                           planned_rows=1, planned_segments=4, planned_bits=32,
                           present_bits=16, structural_missing_bits=16), stream)
            stream.flush(); os.fsync(stream.fileno())
    except FileExistsError:
        return saved_status(output)
    state = dict(slot_id=SLOT, status='STARTED', planned_rows=1, planned_segments=4,
                 planned_bits=32, present_bits=16, structural_missing_bits=16,
                 stages={s: dict(attempts=0, status='NOT_STARTED') for s in STAGES},
                 chosen_input=None, continuation='DISABLED; all existing attempts are read-only')
    model = None; backend = backend or Backend()

    def persist(): write_json(output / 'state.json', state)

    def stage(name, action):
        state['stages'][name] = dict(attempts=1, status='STARTED', started_at=time.time()); persist()
        try:
            value = action()
            state['stages'][name].update(status='COMPLETE', finished_at=time.time()); persist()
            return value
        except BaseException as exc:
            state['stages'][name].update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, InterruptedError, SystemExit)) else 'FAILED',
                                        error=f'{type(exc).__name__}: {exc}')
            persist(); raise

    try:
        persist()
        public = read_plan(); write_json(output / 'blind_plan.json', public)
        state['chosen_input'] = choose_input(); persist()
        rgb = stage('input', lambda: read_rgb(state['chosen_input']))
        mapped = rgb[public['plan']['operation']['received_index_map']]
        model = stage('load', backend.load)
        def encode_and_save():
            value = backend.encode(model, mapped)
            save_normalized(output / 'normalized.npy', value)
            return value
        normalized = stage('encode', encode_and_save)
        stage('read', lambda: save_read(output, backend.read(normalized, public), public))
        state['status'] = 'BLIND_COMPLETE'; persist()
    except BaseException as exc:
        state.update(status='INTERRUPTED' if isinstance(exc, (KeyboardInterrupt, InterruptedError, SystemExit)) else 'FAILED',
                     error=f'{type(exc).__name__}: {exc}')
        persist()
    finally:
        if model is not None:
            try: backend.close(model)
            except Exception as exc:
                state['cleanup_error'] = f'{type(exc).__name__}: {exc}'; persist()
    return state
