"""Restore one saved candidate and replay the existing head with one cached Aa."""
from pathlib import Path
import numpy as np
import torch
from main.tube_state import b_framewise_qim_v1 as p
from main.tube_state import b_framewise_qim_saved_candidate_v1 as checks
from .b_framewise_qim_v1_backend import Backend as OriginalBackend
from .b_framewise_qim_v1_operator import HeadOperator, replay_head, validate_head


class Backend(OriginalBackend):
    def restore_saved(self, saved):
        self.saved = Path(saved)
        self.loaded = True  # release a partially loaded model after an actual load failure too
        self.call('model_load', self.base.load_vae)
        self.base.vae.eval().requires_grad_(False)
        validate_head(self.base.vae.decoder.conv_out)
        self.operator = HeadOperator(self.base.vae.decoder.conv_out.weight, self.active, self.event)
        self.coefficients = torch.load(self.saved/'coefficients.pt', map_location='cpu', weights_only=True)
        if (not isinstance(self.coefficients, torch.Tensor) or self.coefficients.dtype != torch.float64
                or tuple(self.coefficients.shape) != self.operator.input_shape
                or not bool(torch.isfinite(self.coefficients).all())):
            raise ValueError('finite saved CPU FP64 coefficients matching actual head required')
        targets = np.load(self.saved/'fixed_targets.npy', allow_pickle=False)
        if targets.dtype != np.float64 or targets.shape != (181, 10, 63): raise ValueError('saved target shape/dtype')
        self.targets = torch.from_numpy(targets)
        self.baseline = np.load(self.saved/'BASE_preclip.npy', mmap_mode='r', allow_pickle=False)
        if self.baseline.dtype != np.float32 or self.baseline.shape != (181, 320, 512, 3): raise ValueError('saved BASE shape/dtype')
        self.chunks = [self.saved/'features'/f'{start:03}.pt' for start in (0, *range(1, 181, 4))]

    def check_candidate(self):
        def base_projection():
            # One full P pass; pack the mmap in small blocks without a full RGB copy.
            packed = torch.empty(self.operator.output_shape, dtype=torch.float64)
            for start in range(0, len(self.active), 4):
                block = torch.from_numpy(np.array(self.baseline[start:start+4], copy=True))
                if not bool(torch.isfinite(block).all()): raise ValueError('nonfinite saved BASE')
                packed[start:start+len(block)] = p.pack(block.double())
            return 255*p.project(packed)
        q0 = self.call('BASE_constraint_P', base_projection)
        checks.validate_targets(q0, self.targets, self.bits, self.active)
        self.prediction = self.operator.forward(self.coefficients)  # exactly one full K, 136 internal convolutions
        projected = self.call('Aa_constraint_P', lambda: p.project(self.prediction))
        return checks.direct_constraints(q0, projected, self.targets, self.prediction, self.operator.mask, self.active)

    def _features(self):
        channels = self.operator.channels
        for index, path in enumerate(self.chunks):
            feature = torch.load(path, map_location='cpu', weights_only=True)
            expected = (1, channels, 1 if index == 0 else 4, 320, 512)
            if not isinstance(feature, torch.Tensor) or tuple(feature.shape) != expected or feature.dtype != torch.float32:
                raise ValueError('saved feature chunk shape/dtype: '+str(path))
            if not bool(torch.isfinite(feature).all()): raise ValueError('nonfinite saved feature: '+str(path))
            yield feature.to(self.base.device)

    def apply_saved(self):
        # Cached Aa is required; this interface never calls K or loads another baseline.
        if not hasattr(self, 'prediction'): raise RuntimeError('single cached Aa must exist before replay')
        self.base.vae.to(self.base.device)
        path = self.output/'QIM_preclip.npy'
        expected_frames = len(self.coefficients)  # restore_saved enforces the fixed 181-frame input
        out = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(expected_frames, 320, 512, 3))
        chunks, frames, cursor = [], [], 0
        try:
            for start, value in replay_head(self.base.vae.decoder.conv_out, self._features(),
                                            self.coefficients, self.active, self.event):
                value = value.detach().cpu(); length = len(value)
                if start != cursor: raise ValueError('native frame coverage mismatch')
                baseline = torch.from_numpy(np.array(self.baseline[start:start+length], copy=True))
                measured = checks.native_chunk_metrics(value, baseline, self.prediction[start:start+length],
                    self.targets[start:start+length], self.active[start:start+length], self.operator.mask[start:start+length])
                frame_metrics = measured.pop('frames')
                frames.extend(dict(frame=start+i, chunk_start=start, **row) for i, row in enumerate(frame_metrics))
                chunks.append(dict(start=start, length=length, **measured))
                out[start:start+length] = value.numpy(); cursor += length
        finally: out.flush()
        if cursor != expected_frames: raise ValueError('incomplete native materialization')
        accepted = max(row['budget']['excess'] for row in chunks) <= 1e-7 and max(row['target_max'] for row in chunks) <= 1e-3
        starts = {1, *range(5, expected_frames, 4)}
        seam_frames = [row for row in frames if row['frame'] in starts or row['frame']+1 in starts]
        return str(path), dict(status='NATIVE_RESIDUAL_WITHIN_TOLERANCE' if accepted else 'NUMERICALLY_UNRESOLVED',
                              finite_complete=True, frames=expected_frames, chunks=chunks, frame_observations=frames,
                              seam_observations=seam_frames, prediction_error_is_observation_only=True,
                              optimality='UNPROVEN', extra_head_calls=0)

    def read_layer(self, preclip, layer):
        if layer == 'h264':
            pixels = np.load(preclip, mmap_mode='r', allow_pickle=False)
            if pixels.dtype != np.uint8 or pixels.shape != (181, 320, 512, 3):
                raise ValueError('h264 cache must be uint8 0..255; time reader divides once')
        return super().read_layer(preclip, layer)
