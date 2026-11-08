"""Real Card94 adapter. Imported only on a new user run; fake tests inject a backend."""
import copy
import gc
from pathlib import Path
import torch
from runtime.wan.local_joint_positive_projection_margin_v1 import WanBackend as MediaBackend
from main.tube_state import b_legacy_position_margin_v1 as carrier
from main.tube_state import b_real_four_phase_v1 as physical
from main.tube_state import b_phase_centered_saved_v1 as arithmetic
from runtime.wan import generation, b_segmented_payload_v1 as segmented
from runtime.wan.conditional_joint import trajectory
from runtime.wan.c73_identity_quality_v1 import distortion, preclip_record


class WanBackend(MediaBackend):
    def generation_load(self, host, config):
        dtype = torch.bfloat16 if self.device == 'cuda' else torch.float32
        cfg = dict(model=dict(id=self.model_id), generation=dict(config['generation'],
            prompt=host['prompt'], negative_prompt=host['negative_prompt'], seed=host['seed']))
        self.context = generation.prepare_generation(cfg, load_vae=False, device=self.device, model_dtype=dtype)
        return dict(device=self.device, model_dtype=str(dtype), prompt_pair_calls=1, initial_noise_draws=1, VAE_loaded=False)

    def generate(self, arm, writer, count, record):
        pipe, initial, prompt, negative, dtype = self.context
        scheduler = copy.deepcopy(pipe.scheduler)
        if arm == 'P0':
            return segmented.run_trajectory(pipe, initial, scheduler, prompt, negative, dtype,
                writer['key'], writer['bits'], count, record)
        if arm != 'OFF':
            raise ValueError('only OFF/P0 fresh trajectories allowed')
        trajectory.validate_scheduler(scheduler)
        if scheduler.step_index is not None:
            raise ValueError('fresh native OFF scheduler required')
        z = initial.detach().float().clone()
        for index in range(50):
            c, u = segmented.predict_branches(pipe, z, scheduler, prompt, negative, dtype, index, count)
            velocity = u.float()+5.0*(c.float()-u.float())
            if not bool(torch.isfinite(velocity).all()):
                raise FloatingPointError('nonfinite OFF velocity')
            z = trajectory.native_step(scheduler, z, velocity, index, count, kind='native_step')
            record(dict(index=index, cursor_after=scheduler.step_index, payload_enabled=False))
        return z.detach().cpu(), dict(arm='OFF', steps=50, payload_enabled=False, sync_enabled=False)

    def release_generation(self):
        self.context = None
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def book(self, secret):
        return carrier.codebook(secret)

    def construct(self, z, book, gamma):
        native = carrier.observe(z, book)
        changed, receipt = carrier.construct(z, book, gamma, native_observation=native)
        return changed, dict(construction=receipt, native_observation=native)

    def load_vae(self):
        self.blocks = {'encoder': None, 'decoder': None}
        self.hooks = []
        return self.load()

    def release_vae(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.vae = None
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def save_tensor(self, path, value):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)

    def load_tensor(self, path):
        return torch.load(path, map_location='cpu', weights_only=True)

    def load_pixels(self, path):
        from runtime.wan.local_joint_positive_projection_margin_v1 import validate_rgb
        return validate_rgb(self.load_tensor(path), pixels=True)

    def prepare_view(self, pixels, origin, phase):
        if tuple(pixels.shape) != (181, 320, 512, 3) or pixels.dtype != torch.uint8:
            raise ValueError('complete actual181 RGB24 cache required')
        received = pixels[origin:origin+45]
        if len(received) != 45 or phase not in range(4):
            raise ValueError('fixed R45 and phase required')
        return received[phase:phase+41].clone()

    def encode_view(self, pixels):
        from runtime.wan.vae import reencode_rgb24_readback
        if tuple(pixels.shape) != (41, 320, 512, 3) or pixels.dtype != torch.uint8:
            raise ValueError('actual E41 uint8 input required')
        z = reencode_rgb24_readback(self.vae, pixels.float()/255).detach().cpu().float()
        physical.validate_local(z)
        return z

    def observe(self, z, book):
        return physical.observe(z, book)

    def spectrum(self, z):
        return physical.spectrum(z)

    def votes(self, spectrum, secret):
        return physical.read_spectrum(spectrum, secret)

    def score(self, q, table, counters):
        arithmetic.score(q, table, counters)

    def reduce(self, votes, table, counters):
        arithmetic.reduce_all(votes, table, counters)

    def preclip(self, value):
        return preclip_record(value)

    def quality(self, left, right, meaning):
        return distortion(left, right, includes_guard=False, meaning=meaning)
