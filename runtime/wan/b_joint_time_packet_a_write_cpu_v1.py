"""Stage P: actual CPU-only VAE head extraction and original matrix-free solver."""
from pathlib import Path
import json
import math
import struct
import numpy as np
import torch
from main.tube_state import b_joint_time_packet_a_writer_v1 as method
from main.tube_state import b_framewise_qim_v1 as legacy
from main.tube_state.b_framewise_qim_v1_solver import solve, Limits, budget_metrics
from .b_framewise_qim_v1_operator import HeadOperator, validate_head

MODEL_ID = 'Wan-AI/Wan2.1-T2V-1.3B-Diffusers'


def validate_head_data(value):
    weight, bias = value['weight'], value['bias']
    if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu' or weight.dtype != torch.float32
            or weight.ndim != 5 or weight.shape[0] != 3 or weight.shape[1] < 1
            or tuple(weight.shape[2:]) != (3,3,3) or not bool(torch.isfinite(weight).all())):
        raise ValueError('finite CPU FP32 [3,C,3,3,3] head weight required')
    if not isinstance(bias, torch.Tensor) or bias.dtype != torch.float32 or bias.shape != (3,) or not bool(torch.isfinite(bias).all()):
        raise ValueError('finite FP32 RGB head bias required')
    expected = dict(padding=[1,1,1,1,2,0], stride=[1,1,1], dilation=[1,1,1], groups=1)
    if value.get('causal_contract') != expected: raise ValueError('unsupported actual causal head structure')
    return value


def read_head_tensors(header_file, data_file=None, range_start=0):
    """Read only complete declared head tensors; never load a partial checkpoint."""
    with Path(header_file).open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8: raise ValueError('missing safetensors header length')
        size = struct.unpack('<Q', prefix)[0]
        if not 0 < size <= 8*1024**2: raise ValueError('invalid safetensors header size')
        encoded = stream.read(size)
        if len(encoded) != size: raise ValueError('incomplete safetensors header')
        header = json.loads(encoded)
    data_file = Path(data_file or header_file)
    values, entries = {}, {}
    for name in ('weight', 'bias'):
        entry = header['decoder.conv_out.'+name]
        shape, offsets = entry['shape'], entry['data_offsets']
        if (entry['dtype'] != 'F32' or not isinstance(shape, list)
                or any(type(n) is not int or n <= 0 for n in shape)
                or (name == 'weight' and (len(shape)!=5 or shape[0]!=3 or shape[2:]!=[3,3,3]))
                or (name == 'bias' and shape != [3])
                or len(offsets)!=2 or any(type(n) is not int or n<0 for n in offsets)
                or offsets[1]-offsets[0] != 4*math.prod(shape)):
            raise ValueError('invalid official head tensor layout: '+name)
        start, stop = (8+size+n for n in offsets)
        if start < range_start or stop-range_start > data_file.stat().st_size:
            raise ValueError('requested head tensor bytes are incomplete: '+name)
        with data_file.open('rb') as stream:
            stream.seek(start-range_start); raw=stream.read(stop-start)
        if len(raw) != stop-start: raise ValueError('short head tensor read')
        values[name] = torch.from_numpy(np.frombuffer(raw,dtype='<f4').copy().reshape(shape))
        entries[name] = dict(dtype='F32', shape=shape, absolute_byte_range=[start,stop], bytes=len(raw))
    return values, dict(header_bytes=size, requested_tensors=entries,
                       source_file_scope='head tensor bytes only; rest of VAE not asserted complete')


def prepare_head(model_directory, output):
    """Construct actual VAE structure on meta; load only its head on CPU, no forward."""
    root = Path(model_directory); vae_dir = root/'vae'
    ranged = (vae_dir/'safetensors_header.bin').is_file() and (vae_dir/'head_range.bin').is_file()
    missing = [name for name in ('config.json',) if not (vae_dir/name).is_file()]
    if not ranged and not (vae_dir/'diffusion_pytorch_model.safetensors').is_file():
        missing.append('diffusion_pytorch_model.safetensors')
    if missing:
        from huggingface_hub import hf_hub_download
        for name in missing:
            hf_hub_download(repo_id=MODEL_ID, filename='vae/'+name, local_dir=root)
    if ranged:
        transport = json.loads((vae_dir/'head_transport.json').read_text())
        tensors, tensor_receipt = read_head_tensors(vae_dir/'safetensors_header.bin', vae_dir/'head_range.bin',
                                                   transport['range_start'])
    else:
        tensors, tensor_receipt = read_head_tensors(vae_dir/'diffusion_pytorch_model.safetensors')
    from diffusers import AutoencoderKLWan
    with torch.device('meta'):
        vae = AutoencoderKLWan.from_config(json.loads((vae_dir/'config.json').read_text()))
    conv = vae.decoder.conv_out
    if tuple(conv.weight.shape) != tuple(tensors['weight'].shape) or tuple(conv.bias.shape) != tuple(tensors['bias'].shape):
        raise ValueError('official head tensors do not fit actual config head')
    conv.to_empty(device='cpu'); conv.load_state_dict(tensors); conv.eval().requires_grad_(False)
    validate_head(conv)
    value = dict(weight=conv.weight.detach().cpu().clone(), bias=conv.bias.detach().cpu().clone(),
        causal_contract=dict(padding=list(conv._padding), stride=list(conv.stride), dilation=list(conv.dilation), groups=conv.groups))
    validate_head_data(value)
    path = Path(output)/'head.pt'; torch.save(value, path)
    return dict(status='CPU_HEAD_PREPARED', file=str(path), model_location=str(root), model_id=MODEL_ID,
                weight_shape=list(value['weight'].shape), dtype='float32', device='cpu',
                head_class=type(conv).__name__, causal_contract=value['causal_contract'],
                downloaded_files=missing, tensors=tensor_receipt,
                construction='actual AutoencoderKLWan config on meta; only head parameters loaded on CPU',
                model_forward=0, DiT=0, training=0)


def load_head(path): return validate_head_data(torch.load(path, map_location='cpu', weights_only=True))


def finite_metrics(value):
    if isinstance(value, float): return math.isfinite(value)
    if isinstance(value, dict): return all(finite_metrics(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite_metrics(v) for v in value)
    return True


class Backend:
    def __init__(self, head, output, event, dump):
        self.head, self.output, self.event, self.dump = head, Path(output), event, dump
        self.output.mkdir(parents=True, exist_ok=True)
        self.eligible = False

    def read_q0(self, spec):
        if (spec.get('role') != 'BASE' or spec.get('layer') != 'preclip'
                or spec.get('units') != '255*RGB_luma_DCT' or spec.get('frames') != 181):
            raise ValueError('explicit original 181-frame BASE/preclip RGB q0 role required')
        with np.load(spec['file'], allow_pickle=False) as archive:
            raw = {key: archive[key] for key in ('q','d0','d1','soft','mask')}
        self.q0 = method.validate_raw_q0(raw)
        np.save(self.output/'q0.npy', self.q0.numpy(), allow_pickle=False)
        self.dump(self.output/'input.json', dict(spec, status='Q0_READ_VALIDATED', shape=list(self.q0.shape),
            source_order='frame,lane,replica,bit', stored_order='frame,replica*5+lane,bit',
            all_slots=114030, unit_check='original RGB QIM distance/soft consistency', codec=0))

    def target(self):
        self.fields = method.schedule()
        self.targets = method.targets(self.q0, self.fields)
        method.validate_targets(self.q0, self.targets, self.fields)
        np.save(self.output/'fixed_targets.npy', self.targets.numpy(), allow_pickle=False)
        np.savez(self.output/'schedule.npz', **{k:v.numpy() for k,v in self.fields.items()})
        self.operator = HeadOperator(self.head['weight'], self.fields['active'], self.event)

    def solve(self, progress):
        active = self.fields['active']
        answer = solve(self.operator, (self.targets-self.q0)/255, active[:,:,None].expand(-1,-1,63),
                       self.operator.mask, limits=Limits(), progress=progress)
        self.coefficients = answer.pop('coefficients', None)
        finite = (isinstance(self.coefficients, torch.Tensor) and self.coefficients.dtype == torch.float64
                  and self.coefficients.device.type == 'cpu' and tuple(self.coefficients.shape) == self.operator.input_shape
                  and bool(torch.isfinite(self.coefficients).all()))
        if finite: torch.save(self.coefficients, self.output/'coefficients.pt')
        reason = str(answer.get('reason', ''))
        failed_call = reason.startswith(('MemoryError:', 'RuntimeError:')) or reason == 'nonfinite iterate'
        self.eligible = (finite and finite_metrics(answer) and not failed_call
                         and answer.get('status') in ('SOLVED_APPROXIMATE_KKT', 'NUMERICALLY_UNRESOLVED'))
        self.answer = dict(answer, finite_candidate_saved=finite, native_handoff_eligible=self.eligible,
                           original_solver_unchanged=True, optimality='UNPROVEN')
        self.dump(self.output/'solver.json', self.answer)
        return self.answer

    def predict(self):
        if not self.eligible: raise ValueError('no finite compliant candidate for native handoff')
        prediction = self.operator.forward(self.coefficients)  # The separately budgeted K1.
        if not bool(torch.isfinite(prediction).all()): raise ValueError('nonfinite K prediction')
        torch.save(prediction, self.output/'prediction.pt')
        projected = legacy.project(prediction)
        active = self.fields['active'][:,:,None]
        target_error = float(torch.where(active, self.q0+255*projected-self.targets, 0).abs().max())
        value = dict(status='LINEAR_DIAGNOSTIC_ONLY', target_max=target_error,
                     budget=budget_metrics(prediction, self.operator.mask),
                     native_measured=False, original_solver_status=self.answer['status'])
        self.dump(self.output/'linear.json', value)
        return value

    def handoff(self, source):
        required = ('input.json','q0.npy','schedule.npz','fixed_targets.npy','solver.json',
                    'coefficients.pt','prediction.pt','linear.json')
        missing = [name for name in required if not (self.output/name).is_file()]
        if not self.eligible or missing: raise ValueError('incomplete CPU handoff: '+str(missing))
        value = dict(status='CPU_READY_FOR_NATIVE_MEASUREMENT', source=source, frames=181,
            files={name:name for name in required}, msg='8*j+age', startup='UNWRITTEN_NO_FILLER',
            old_method_changed=False, solver_status=self.answer['status'], native_measured=False,
            next_stage='USER_GPU_1', no_gpu_solver=True)
        self.dump(self.output/'handoff.json', value)
        return value
