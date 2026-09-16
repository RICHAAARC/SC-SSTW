"""Fixed OFF, terminal A/B, flow A/B experiment. No real execution on import."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from main.tube_state import projection_margin as carrier, state_clock, flow_control
from runtime.wan.generation import prepare_generation, load_frozen_vae
from runtime.wan.flow_generation import continue_steps
from runtime.wan.flow_step import controlled_step, measures
from runtime.wan.io import dump, read_mp4, encode_rgb
from runtime.wan.vae import decode_normalized_latent, reencode_rgb24_readback, _clear_cache

ARMS = ('OFF', 'TERMINAL_A', 'TERMINAL_B', 'FLOW_A', 'FLOW_B')


def quality(reference, candidate):
    # Per-frame accumulation avoids several full float64 RGB clips at once.
    if reference.shape != candidate.shape or reference.ndim != 4 or reference.shape[-1] != 3 or len(reference)<2:
        raise ValueError('quality requires matched multi-frame RGB clips')
    mse = residual = base = altered = 0.
    previous = previous_ref = previous_candidate = None
    for r, c in zip(reference, candidate):
        r, c = r.double(), c.double()
        d = c-r
        mse += float(d.square().mean())
        if previous is not None:
            residual += float((d-previous).square().mean())
            base += float((r-previous_ref).square().mean())
            altered += float((c-previous_candidate).square().mean())
        previous, previous_ref, previous_candidate = d, r, c
    n = len(reference)
    mse /= n
    return {'rgb_mse': mse, 'rgb_psnr_db': None if mse == 0 else float(-10*np.log10(mse)),
            'temporal_difference_energy_ratio': None if base == 0 else altered/base,
            'residual_temporal_mse': residual/(n-1)}


def terminal_record(z, book):
    values = {}
    for m in (0, 1):
        row = flow_control.projection_record(z, book, m)
        decoded = np.clip(row['projection'], -1, 1)*book['sync']*book['polarity']
        qs = decoded.reshape(11,80,2).mean(axis=1)
        row['identity_q'] = qs.tolist()
        row['identity_observer'] = state_clock.observe(qs, [True]*11, book['states'][m], book['steps'][m])
        values[str(m)] = row
    return values


def run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start_time = time.monotonic()
    result = {'status': 'RUNNING', 'formal_science_denominator': 0,
        'fixed_calls': {'transformer': 124, 'scheduler_step': 62, 'shadow_step': 12,
                        'vae_decode': 5, 'mp4_save': 5, 'vae_encode': 20},
        'actual_calls': {}, 'failures': [], 'videos': {
            a: {'status': 'NOT_RUN', 'observations': {str(g): {'status': 'NOT_RUN'} for g in range(4)}} for a in ARMS},
        'diagnostic_denominator': {'arms': 5, 'received_videos': 5, 'receiver_encodes': 20},
        'evidence_ceiling': 'single content/seed feasibility only; OFF ranking is not FPR',
        'quality_rule': {'relative_limit': 1.5, 'metrics': ['rgb_mse', 'residual_temporal_mse'],
                         'meaning': 'first-round tolerance, not validated perception threshold'}}
    for kind in result['fixed_calls']:
        result['actual_calls'].update({kind+'_attempted': 0, kind+'_completed': 0})
    def save():
        result['elapsed_seconds'] = time.monotonic()-start_time
        dump(output/'result.json', result)
    def count(kind, completed):
        result['actual_calls'][kind+('_completed' if completed else '_attempted')] += 1
        save()
    def fail(stage, exc):
        result['failures'].append({'stage': stage, 'error': repr(exc), 'classification': 'ENGINEERING_FAILURE'})
        save()
    def release():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    def store_terminal(name, z):
        z = z.detach().cpu().float()
        if not torch.isfinite(z).all():
            raise FloatingPointError('nonfinite terminal')
        torch.save(z, output/f'{name}_terminal.pt')
        dump(output/'terminal_diagnostics'/f'{name}.json', terminal_record(z.numpy(), book))
        if name != 'OFF' and (output/'OFF_terminal.pt').exists():
            off_reference = torch.load(output/'OFF_terminal.pt', map_location='cpu', weights_only=True)
            result['videos'][name]['final_offset_vs_off'] = measures(z-off_reference)
        result['videos'][name]['status'] = 'TERMINAL_PERSISTED'
        save()
    dump(output/'config.json', config)
    save()
    pipe = vae = snapshot = latent = prompt = negative = kwargs = off = off_scheduler = current = scheduler = None
    try:
        if config['generation']['steps'] != 50:
            raise ValueError('fixed candidate requires 50 steps')
        source = subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True)
        result['source_commit'] = source.stdout.strip() if source.returncode == 0 else None
        result['source_dirty'] = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()) if source.returncode == 0 else None
        result['source_snapshot'] = config.get('source_snapshot', 'working tree')
        import diffusers
        result['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
            'diffusers': diffusers.__version__, 'cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        book = state_clock.codebook(config['key_utf8'].encode())
        np.savez(output/'codebook.npz', **book)
        pipe, latent, prompt, negative, dtype = prepare_generation(config)
        vae = pipe.vae
        # The VAE is absent from continuation and may live on CPU until decode.
        vae.to('cpu')
        if type(pipe.scheduler).__name__ != 'UniPCMultistepScheduler' or pipe.scheduler.config.prediction_type != 'flow_prediction' or pipe.scheduler.config.thresholding:
            raise ValueError('actual scheduler does not implement the authorized affine flow path')
        result['scheduler'] = {'class': type(pipe.scheduler).__name__, 'config': dict(pipe.scheduler.config),
                               'sigmas': pipe.scheduler.sigmas.tolist()}
        save()
        kwargs = dict(pipe=pipe, prompt=prompt, negative=negative, input_dtype=dtype,
                      guidance_scale=config['generation']['guidance_scale'], count=count,
                      precision=result.setdefault('precision', {}))
        latent = continue_steps(scheduler=pipe.scheduler, latent=latent, start=0, stop=44, **kwargs)
        snapshot = copy.deepcopy(pipe.scheduler)
        torch.save({'latent': latent.cpu(), 'scheduler': copy.deepcopy(snapshot)}, output/'pre_intervention_state.pt')
        off_scheduler = copy.deepcopy(snapshot)
        off = continue_steps(scheduler=off_scheduler, latent=latent.clone(), start=44, stop=50, **kwargs)
        store_terminal('OFF', off)
        off_np = off.cpu().numpy()
        radius = 0.
        for m, name in enumerate(('TERMINAL_A', 'TERMINAL_B')):
            try:
                marked, evidence = carrier.write(off_np, book, m)
                radius = max(radius, flow_control.rms(marked-off_np))
                dump(output/'write_evidence'/f'{name}.json', evidence)
                store_terminal(name, torch.from_numpy(marked))
            except Exception as exc:
                result['videos'][name]['status'] = 'FAILED_TERMINAL'
                fail(name, exc)
        if not all(result['videos'][a]['status'] == 'TERMINAL_PERSISTED' for a in ('TERMINAL_A', 'TERMINAL_B')):
            raise RuntimeError('R unavailable because a terminal reference failed')
        result['budget'] = {'R': radius, 'definition': 'max terminal A/B support RMS versus same-run OFF',
                            'per_step': radius/3., 'cumulative_sum_rms': radius,
                            'active_indices': [44,45,46], 'normal_tail_indices': [47,48,49]}
        save()  # R is fixed and persisted before either Flow condition.
        del off, off_scheduler, marked
        for m, name in enumerate(('FLOW_A', 'FLOW_B')):
            current = scheduler = None
            totals = {'u': 0., 'D': 0.}
            try:
                scheduler = copy.deepcopy(snapshot)
                def control(index, sched, sample, velocity, timestep):
                    sigma = float(sched.sigmas[sched.step_index])
                    z0hat = (sample-sigma*velocity).float()
                    u_np, predicted = flow_control.request(z0hat.cpu().numpy(), book, m)
                    u = torch.from_numpy(u_np).to(sample.device)
                    updated, record, arrays = controlled_step(sched, sample, velocity, timestep, u, radius, totals, count)
                    record.update(index=index, prediction_before=predicted,
                        prediction_after=flow_control.projection_record((z0hat+record['scale']*u).cpu().numpy(), book, m))
                    dest = output/'guidance'/name
                    dest.mkdir(parents=True, exist_ok=True)
                    torch.save(arrays | {'z0hat': z0hat.cpu()}, dest/f'step{index}.pt')
                    dump(dest/f'step{index}.json', record)
                    return updated
                current = continue_steps(scheduler=scheduler, latent=latent.clone(), start=44, stop=50, control=control, **kwargs)
                store_terminal(name, current)
                result['videos'][name]['final_offset_vs_off'] = measures(current.cpu()-torch.from_numpy(off_np))
                result['videos'][name]['cumulative_sum_rms'] = totals
                for index in (44,45,46):
                    p = output/'guidance'/name/f'step{index}.pt'
                    arrays = torch.load(p, map_location='cpu', weights_only=True)
                    error = current.cpu()-arrays['z0hat']-arrays['applied_u']
                    dump(p.with_name(f'step{index}_terminal_error.json'), {
                        'predicted_after_vs_terminal': measures(error),
                        'projection_error': (np.asarray(flow_control.projection_record(current.cpu().numpy(),book,m)['projection'])-
                            np.asarray(flow_control.projection_record((arrays['z0hat']+arrays['applied_u']).numpy(),book,m)['projection'])).tolist()})
                    del arrays, error
            except Exception as exc:
                result['videos'][name]['status'] = 'FAILED_FLOW'
                fail(name, exc)
            finally:
                current = scheduler = None
                release()
        del kwargs
    except Exception as exc:
        fail('generation', exc)
    finally:
        pipe = snapshot = latent = prompt = negative = kwargs = off = off_scheduler = current = scheduler = None
        release()
    # All available conditions continue through independent media/readout paths.
    try:
        if vae is None and any((output/f'{a}_terminal.pt').exists() for a in ARMS):
            vae = load_frozen_vae(config)
        if vae is not None:
            vae.to('cuda')
        for name in ARMS:
            rgb = z = obs = encoded = None
            item = result['videos'][name]
            path = output/'received_videos'/f'{name}.mp4'
            try:
                z = torch.load(output/f'{name}_terminal.pt', map_location='cpu', weights_only=True)
                count('vae_decode', False)
                rgb = decode_normalized_latent(vae, z.to(next(vae.parameters()).device)).cpu()
                count('vae_decode', True)
                torch.save(rgb, output/f'{name}_precodec_rgb.pt')
                if name != 'OFF' and (output/'OFF_precodec_rgb.pt').exists():
                    reference = torch.load(output/'OFF_precodec_rgb.pt', weights_only=True)
                    item['precodec_quality_vs_off'] = quality(reference, rgb)
                    del reference
                count('mp4_save', False)
                encode_rgb(rgb, path, 8, 18)
                count('mp4_save', True)
                item.update(status='VIDEO_PERSISTED', path=str(path.relative_to(output)))
            except Exception as exc:
                item['status'] = 'FAILED_MEDIA'
                fail(name+'/media', exc)
            finally:
                rgb = z = None
                if vae is not None:
                    _clear_cache(vae)
                release()
            obs = {}
            try:
                rgb = read_mp4(path)
                if len(rgb) != 181:
                    raise ValueError('normal MP4 requires 181 frames')
                if name != 'OFF' and (output/'received_videos/OFF.mp4').exists():
                    reference = read_mp4(output/'received_videos/OFF.mp4')
                    item['saved_quality_vs_off'] = quality(reference, rgb)
                    del reference
                for g in range(4):
                    row = item['observations'][str(g)]
                    try:
                        count('vae_encode', False)
                        groups, tail = divmod(len(rgb)-g-1, 4)
                        encoded = reencode_rgb24_readback(vae, rgb[g:g+1+4*groups]).cpu().float()
                        count('vae_encode', True)
                        if tuple(encoded.shape) != (1,16,1+groups,40,64):
                            raise ValueError('receiver latent geometry mismatch')
                        dest = output/'receiver_observations'/name
                        dest.mkdir(parents=True, exist_ok=True)
                        torch.save(encoded, dest/f'g{g}.pt')
                        obs[g] = encoded.numpy()
                        row.update(status='COMPLETE', frames_used=1+4*groups, tail_discarded=tail)
                    except Exception as exc:
                        row.update(status='FAILED', error=repr(exc))
                        fail(name+f'/g{g}', exc)
                    finally:
                        encoded = None
                        _clear_cache(vae)
                        release()
                        save()
            except Exception as exc:
                fail(name+'/readback', exc)
            try:
                detection = state_clock.read(obs, book)
                dump(output/'detections'/f'{name}.json', detection)
                item['rankings'] = detection['rankings']
                if name != 'OFF':
                    truth = 0 if name.endswith('A') else 1
                    item['reporting_only'] = state_clock.report(detection, truth, 0)
                    item['five_methods_unique_correct'] = all(r['message_unique'] and r['best']['message']==truth for r in detection['rankings'].values()) if len(obs)==4 else None
                item['status'] = 'COMPLETE' if len(obs)==4 else 'PARTIAL_OR_FAILED'
            except Exception as exc:
                fail(name+'/detection', exc)
            rgb = obs = encoded = None
            release()
            save()
    except Exception as exc:
        fail('media_setup', exc)
    finally:
        vae = None
        release()
    comparisons = {}
    for suffix in ('A','B'):
        flow, terminal = result['videos']['FLOW_'+suffix], result['videos']['TERMINAL_'+suffix]
        f, t = flow.get('saved_quality_vs_off'), terminal.get('saved_quality_vs_off')
        comparisons[suffix] = None if f is None or t is None else {
            k: f[k] <= 1.5*t[k] for k in ('rgb_mse','residual_temporal_mse')}
    result['saved_quality_comparisons'] = comparisons
    complete = all(v['status']=='COMPLETE' for v in result['videos'].values()) and not result['failures']
    result['first_round_criteria_met'] = (all(result['videos'][a].get('five_methods_unique_correct') for a in ARMS[1:]) and
        all(v is not None and all(v.values()) for v in comparisons.values())) if complete else None
    result['status'] = 'EXECUTED_REQUIRES_METHOD_REVIEW' if complete else 'EXECUTED_WITH_RETAINED_FAILURES'
    result['method_diagnostics'] = {
        'saved_readout_failed_conditions': [a for a in ARMS[1:] if result['videos'][a].get('five_methods_unique_correct') is False],
        'quality_failed_messages': [m for m,v in comparisons.items() if v is not None and not all(v.values())],
        'prediction_and_propagation': 'inspect full guidance and terminal_diagnostics records; no proxy-loss PASS threshold',
        'engineering_failures_are_not_scientific_no_go': True}
    result['resources'] = {'cpu_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None}
    save()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(json.loads(args.config.read_text()), args.output)
    print(json.dumps({k:result[k] for k in ('status','actual_calls','first_round_criteria_met','failures')}, indent=2))
    if result['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
