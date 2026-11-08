"""User-run physical adapter for the adopted DCT2 / M3 / CDC32 chain.

Importing this module does not load a model. The orchestrator imports it only
after recording the one physical attempt. Old FFT readers are never invoked.
"""
import copy
import math
import subprocess

import torch

from main.tube_state import b_legacy_position_margin_v1 as state
from main.tube_state import b_real_four_phase_v1 as physical
from runtime.wan.b_fixed_detection_confidence_backend_v1 import WanBackend as Base


class WanBackend(Base):
    def generate(self, arm, writer, count, record):
        if arm == 'OFF':
            return super().generate(arm, writer, count, record)
        if arm != 'P0':
            raise ValueError('only OFF and P0 have generation trajectories')
        from runtime.wan.b_local_identity_dct_v1 import run_trajectory
        pipe, initial, prompt, negative, dtype = self.context
        return run_trajectory(pipe, initial, copy.deepcopy(pipe.scheduler), prompt,
                              negative, dtype, writer['key'], writer['bits'], count, record)

    def native_observe(self, z, book):
        return state.observe(z, book)

    def construct(self, z, book, native):
        return state.construct(z, book, 1.0, native_observation=native)

    def native_audit(self, before, after, post_q):
        delta = after.double() - before.double()
        mask = torch.zeros_like(delta, dtype=torch.bool)
        for y0, y1, x0, x1 in state.ROIS:
            mask[0, 4:16, 1:45, y0:y1, x0:x1] = True
        rows = [dict(position=t+1, roi=r, q=post_q['roi_q'][t][r][t],
                     shortage=max(0.0, 1.0-post_q['roi_q'][t][r][t]))
                for t in range(44) for r in range(4)]
        return dict(status='MEASURED', key='key00', units=rows,
                    actual_squared_energy=float(delta.square().sum()),
                    outside_support_max=float(delta[~mask].abs().max()),
                    payload_channels_max=float(delta[:, :4].abs().max()),
                    guard_max=float(delta[:, :, [0, 45]].abs().max()),
                    no_reprojection=True)

    @staticmethod
    def validate_pixels(pixels, length):
        if (not isinstance(pixels, torch.Tensor) or
                tuple(pixels.shape) != (length, 320, 512, 3) or pixels.dtype != torch.uint8):
            raise ValueError(f'actual uint8[{length},320,512,3] RGB required')
        return pixels.detach().cpu()

    def edit(self, pixels, source_frames):
        self.validate_pixels(pixels, 181)
        if len(source_frames) != 151 or any(type(f) is not int or not 0 <= f < 181 for f in source_frames):
            raise ValueError('151 valid source frame indices required for private fixture construction')
        return self.validate_pixels(pixels[source_frames].clone(), 151)

    def read_media(self, path, probe, length):
        import numpy as np
        if (int(probe.get('nb_frames', -1)) != length or probe.get('r_frame_rate') != '8/1'
                or probe.get('codec_name') != 'h264' or probe.get('pix_fmt') != 'yuv420p'):
            raise ValueError('actual fixed-length 8fps H264 yuv420p stream required')
        proc = subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-noautorotate',
                               '-i', str(path), '-map', '0:v:0', '-f', 'rawvideo',
                               '-pix_fmt', 'rgb24', '-'], check=True, capture_output=True)
        if len(proc.stdout) != length*320*512*3:
            raise ValueError('decoded RGB length or geometry mismatch')
        value = torch.from_numpy(np.frombuffer(proc.stdout, dtype=np.uint8).copy()
                                 .reshape(length, 320, 512, 3))
        return self.validate_pixels(value, length)

    def prepare_window(self, pixels, w, phase):
        self.validate_pixels(pixels, 151)
        if w not in (0, 53, 106) or phase not in range(4):
            raise ValueError('fixed M3 received window and phase required')
        return self.validate_pixels(pixels[w+phase:w+phase+41].clone(), 41)

    def encode_view(self, pixels):
        # The inherited implementation divides once, uses posterior mode and
        # resets the Wan cache via reencode_rgb24_readback for each E41.
        return super().encode_view(pixels)

    def payload_project(self, z, rows, regions):
        from main.tube_state import b_local_identity_dct_v1 as payload
        return payload.project(z, regions=regions, rows=rows)

    def payload_decode(self, projection, secret, view_id, key_id):
        from main.tube_state import b_local_identity_dct_v1 as payload
        return payload.decode(projection, secret, view_id=view_id, key_id=key_id)

    def score(self, bank, background_model):
        from main.tube_state.b_m3_time_path_v1 import score_family
        return score_family(bank, background_model)

    def quality(self, left, right, source_frames, pair):
        """Post-only source coordinates; no new decoding or codec invocation."""
        from runtime.wan.local_joint_positive_projection_margin_quality_v1 import _frame, _stats, _combine
        if tuple(left.shape) != tuple(right.shape) or tuple(left.shape)[1:] != (320, 512, 3):
            raise ValueError('matching actual RGB arrays required')
        if len(source_frames) != len(left) or len(left) not in (151, 181):
            raise ValueError('complete post-only frame map required')
        regions = ((8, 24, 8, 24), (8, 24, 40, 56)) if pair == 'OFF_to_P0' else state.ROIS
        frames, adjacent = [], []
        previous = previous_mask = None
        for t, source in enumerate(source_frames):
            a, b = _frame(left[t]), _frame(right[t])
            if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
                raise ValueError('nonfinite quality input')
            delta = b-a
            mask = torch.zeros((320, 512, 3), dtype=torch.bool)
            if pair == 'OFF_to_P0' or 1 <= source <= 176:
                for y0, y1, x0, x1 in regions:
                    mask[y0*8:y1*8, x0*8:x1*8] = True
            frames.append(dict(frame=t, source_frame=source, whole=_stats(delta),
                               outside_nominal_support=_stats(delta[~mask]),
                               base_range=[float(a.min()), float(a.max())],
                               changed_range=[float(b.min()), float(b.max())]))
            if previous is not None:
                adjacent.append(dict(frame_before=t-1, frame_after=t,
                                     source_before=source_frames[t-1], source_after=source,
                                     crosses_deletion=source != source_frames[t-1]+1,
                                     whole=_stats(delta-previous),
                                     outside_union_nominal_support=_stats((delta-previous)[~(mask | previous_mask)])))
            previous, previous_mask = delta, mask
        whole = _combine([r['whole'] for r in frames])
        return dict(status='SCORED', frame_denominator=len(frames), adjacent_denominator=len(adjacent),
                    frames=frames, adjacent=adjacent, whole=whole,
                    outside_nominal_support=_combine([r['outside_nominal_support'] for r in frames]),
                    adjacent_residual=_combine([r['whole'] for r in adjacent]),
                    adjacent_outside_union_nominal_support=_combine([r['outside_union_nominal_support'] for r in adjacent]),
                    psnr_db_range1=None if whole['rms'] == 0 else -20*math.log10(whole['rms']),
                    nominal_regions=[list(r) for r in regions], mask_is_actual_causal_support=False,
                    quality_pass=None)
