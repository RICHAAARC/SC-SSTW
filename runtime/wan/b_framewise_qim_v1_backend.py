"""Future user-run adapter, reusing the physically exercised DCT2 VAE/media path."""
from pathlib import Path
import shutil
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as p
from main.tube_state.b_framewise_qim_v1_solver import solve, budget_metrics
from runtime.wan.b_framewise_qim_v1_operator import HeadOperator, replay_head, validate_head
from runtime.wan.vae import decode_normalized_latent, quantize_rgb8_no_codec


class Backend:
    def __init__(self, cfg, output, event):
        from runtime.wan.b_terminal_three_group_backend_v1 import WanBackend
        self.cfg, self.output, self.event = cfg, Path(output), event
        self.base = WanBackend(cfg['model_id']); self.loaded = False
        self.js, self.bits = p.schedule(); self.active = self.js >= 0
        self.chunks = []

    def call(self, kind, fn):
        self.event(kind, False); value = fn(); self.event(kind, True)
        return value

    def capture(self, terminal):
        from main.tube_state.b_local_identity_dct_v1 import LATENT_SHAPE
        z = torch.load(terminal, map_location='cpu', weights_only=True)
        if not isinstance(z, torch.Tensor) or tuple(z.shape) != LATENT_SHAPE or z.dtype != torch.float32 or not bool(torch.isfinite(z).all()):
            raise ValueError('finite FP32 normalized DCT2 P0 [1,16,46,40,64] required')
        self.base.load_vae(); self.loaded = True
        vae = self.base.vae; vae.eval().requires_grad_(False); validate_head(vae.decoder.conv_out)
        channels = vae.decoder.conv_out.in_channels
        required = max(32*1024**3, 4*181*channels*320*512+4*181*320*512*3*5)
        if shutil.disk_usage(self.output).free < required: raise MemoryError('insufficient space for FP32 native feature cache')
        folder = self.output/'features'; folder.mkdir()
        pre = np.lib.format.open_memmap(self.output/'BASE_preclip.npy', mode='w+', dtype=np.float32, shape=(181,320,512,3))
        cursor = 0
        def before(module, args):
            self.event('baseline_head', False)
        def captured(module, args, value):
            nonlocal cursor
            h = args[0]; length = int(h.shape[2])
            if (h.dtype != torch.float32 or tuple(h.shape[:2]) != (1, channels) or tuple(h.shape[-2:]) != (320,512)
                    or length != (1 if cursor == 0 else 4) or cursor+length > 181):
                raise ValueError('native RGB-time head input contract')
            if not bool(torch.isfinite(h).all() and torch.isfinite(value).all()): raise FloatingPointError('native nonfinite head')
            path = folder/f'{cursor:03}.pt'; torch.save(h.detach().cpu(), path)
            self.chunks.append(path)
            pre[cursor:cursor+length] = (value.detach()[0].permute(1,2,3,0).float()/2+.5).cpu().numpy()
            cursor += length; self.event('baseline_head', True)
        h1 = vae.decoder.conv_out.register_forward_pre_hook(before)
        h2 = vae.decoder.conv_out.register_forward_hook(captured)
        try:
            rgb = self.call('D181', lambda: decode_normalized_latent(vae, z.to(self.base.device)))
            if cursor != 181: raise ValueError('incomplete181 capture')
            # Confirm the hook is on the actual preclip output, not another layer.
            for t in range(0,181,4):
                expected = torch.from_numpy(np.array(pre[t:t+4], copy=True)).clamp(0,1)
                if not torch.equal(expected, rgb[t:t+4].detach().cpu()): raise ValueError('head capture differs from native output')
            pre.flush()
        finally:
            h1.remove(); h2.remove()
        # Offload during CPU solving; apply() restores the native replay device.
        self.base.vae.to('cpu')
        if self.base.device == 'cuda': torch.cuda.empty_cache()
        return str(self.output/'BASE_preclip.npy')

    def solve(self, baseline, progress):
        self.base.vae.to('cpu')
        if self.base.device == 'cuda': torch.cuda.empty_cache()
        self.operator = HeadOperator(self.base.vae.decoder.conv_out.weight, self.active, self.event)
        raw = self.read_layer(baseline, 'preclip')
        q0 = torch.from_numpy(raw['q']).transpose(1,2).reshape(181,10,63)
        target = p.nearest(q0, self.bits)  # exactly once; persist before iteration
        np.save(self.output/'fixed_targets.npy', target.numpy(), allow_pickle=False)
        answer = solve(self.operator, (target-q0)/255, self.active[:,:,None].expand(-1,-1,63), self.operator.mask, progress=progress)
        coefficients = answer.pop('coefficients', None)
        if coefficients is not None: torch.save(coefficients, self.output/'coefficients.pt')
        return answer

    def apply(self, baseline):
        # Replay on the baseline's native device, avoiding an avoidable CPU/GPU
        # convolution change in the measured residual. The solver stays CPU FP64.
        self.base.vae.to(self.base.device)
        a = torch.load(self.output/'coefficients.pt', map_location='cpu', weights_only=True)
        prediction = self.operator.forward(a)
        base = np.load(baseline, mmap_mode='r', allow_pickle=False)
        out = np.lib.format.open_memmap(self.output/'QIM_preclip.npy', mode='w+', dtype=np.float32, shape=(181,320,512,3))
        target = torch.from_numpy(np.load(self.output/'fixed_targets.npy', allow_pickle=False))
        metrics = []; maximum_prediction_error = 0.; maximum_target_error = 0.
        chunks = (torch.load(path, map_location='cpu', weights_only=True).to(self.base.device) for path in self.chunks)
        for start, value in replay_head(self.base.vae.decoder.conv_out, chunks, a, self.active, self.event):
            value = value.detach().cpu()
            length = len(value); delta = value.double()-torch.from_numpy(np.array(base[start:start+length], copy=True)).double()
            packed = p.pack(delta)
            predicted = prediction[start:start+length]
            maximum_prediction_error = max(maximum_prediction_error, float((packed-predicted).abs().max()))
            mask = self.operator.mask[start:start+length]
            measured = budget_metrics(packed, mask)
            # Full canvas outside the geometrically allowed support must remain zero within arithmetic tolerance.
            full_mask = torch.zeros((length,320,512,3), dtype=torch.bool)
            for tile,(y,x) in enumerate(p.ROIS):
                height = min(58,321-y)
                full_mask[:,y-1:y-1+height,x-1:x+73] = mask[:,tile,:,:height].permute(0,2,3,1)
            measured['outside_full_canvas'] = float(torch.where(full_mask, 0, delta).abs().max())
            measured['excess'] = max(measured['excess'], measured['outside_full_canvas'])
            metrics.append(dict(start=start, **measured))
            q = 255*p.project(p.pack(value.double()))
            maximum_target_error = max(maximum_target_error, float(torch.where(self.active[start:start+length,:,None], q-target[start:start+length], 0).abs().max()))
            out[start:start+length] = value.numpy()
        out.flush()
        accepted = max(x['excess'] for x in metrics) <= 1e-7 and maximum_target_error <= 1e-3
        return str(self.output/'QIM_preclip.npy'), dict(status='NATIVE_RESIDUAL_WITHIN_TOLERANCE' if accepted else 'NUMERICALLY_UNRESOLVED',
            target_max=maximum_target_error, linear_prediction_max=maximum_prediction_error, frames=metrics)

    def read_layer(self, preclip, layer):
        rgb = np.load(preclip, mmap_mode='r', allow_pickle=False)
        values = {k:[] for k in ('q','d0','d1','soft','mask')}
        for t in range(0,181,4):
            v = torch.from_numpy(np.array(rgb[t:t+4], copy=True))
            if layer == 'clip': v = v.clamp(0,1)
            elif layer == 'rgb8': v = quantize_rgb8_no_codec(v).double()/255
            elif layer == 'h264': v = v.double()/255
            elif layer != 'preclip': raise ValueError('fixed observation layer')
            for k, x in p.read_rgb(v).items(): values[k].append(x.numpy())
        return {k:np.concatenate(x) for k,x in values.items()}

    def media(self, preclip, arm):
        from runtime.wan.io import encode_rgb8
        rgb = torch.from_numpy(np.array(np.load(preclip, mmap_mode='r', allow_pickle=False), copy=True))
        pixels = quantize_rgb8_no_codec(rgb); path = self.output/(arm+'.mp4')
        self.call('codec_encode', lambda: encode_rgb8(pixels, path, fps=8, crf=18))
        probe = self.call('codec_probe', lambda: self.base.probe(path))
        received = self.call('codec_decode', lambda: self.base.read_media(path, probe, 181))
        saved = self.output/(arm+'_h264.npy'); np.save(saved, received.numpy(), allow_pickle=False)
        return str(saved)

    def identity(self, media, start, phase, view):
        from main.tube_state import b_local_identity_dct_v1 as dct
        self.base.vae.to(self.base.device)
        pixels = torch.from_numpy(np.array(np.load(media, mmap_mode='r', allow_pickle=False)[start+phase:start+phase+41], copy=True))
        z = self.call('E41', lambda: self.base.encode_view(pixels))
        return dct.decode(dct.project(z), 'watermark', view_id=view, key_id='key00')

    def close(self):
        if self.loaded: self.base.release_vae()
