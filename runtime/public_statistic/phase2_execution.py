"""Bounded real-Wan entry. Imported lazily; no model work occurs on import."""
import gc
import json
import math
import time
from pathlib import Path
from main.sc_sstw.terminal_feedback import clone_graph_state
from runtime.public_statistic.phase2 import WanTerminalAdapter, SolverState, FeedbackConfig, controlled_run

ARMS = ('OFF1', 'OFF2', 'Y_MINUS')


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    temporary.replace(path)


def initial_result(config):
    g = config['generation']
    return dict(status='NOT_EXECUTED', science_denominator=0, automatic_retries=0,
                arms={arm: dict(status='NOT_EXECUTED', rows=[dict(index=i, time=i/g['fps'], q=None, status='NOT_EXECUTED') for i in range(g['frames'])]) for arm in ARMS})


def validate_config(c):
    g, f, b = c['generation'], c['feedback'], c['budget']
    for name in ('steps', 'frames', 'height', 'width', 'fps', 'max_sequence_length'):
        if type(g[name]) is not int or g[name] <= 0: raise ValueError('positive integer ' + name)
    if type(g['seed']) is not int or not isinstance(g['prompt'], str) or not g['prompt']: raise ValueError('seed/prompt')
    if not math.isfinite(g['guidance']) or g['guidance'] <= 1: raise ValueError('CFG >1 required for this three-arm trial')
    if g['steps'] < 3: raise ValueError('two final feedback events require at least three steps')
    if f['after_indices'] != [g['steps']-3, g['steps']-2]: raise ValueError('this entry uses the final two nonterminal feedback events')
    for name in ('learning_rate','per_update_fraction','cumulative_fraction','amplitude','quality_clip_rmse'):
        if not math.isfinite(f[name]) or f[name] <= 0: raise ValueError('positive feedback ' + name)
    # These are requested workload/resource bounds, never environment matching gates.
    for name in ('transformer_calls','vae_calls','backward_calls','mp4','saved_frames','wall_seconds'):
        if not math.isfinite(b[name]) or b[name] <= 0: raise ValueError('positive budget ' + name)
    if b['mp4'] < 3 or b['saved_frames'] < 3*g['frames']: raise ValueError('budget cannot retain the fixed three-arm roster')
    if c.get('automatic_retries') != 0: raise ValueError('no automatic retry')
    if not isinstance(c['model']['id'],str) or not c['model']['id']: raise ValueError('model id')


class ResourceGuard:
    def __init__(self, budget, *, torch=None, persist=None, clock=time.monotonic):
        self.budget, self.torch, self.persist, self.clock = budget, torch, persist, clock
        self.start = clock()
        self.counts = dict(transformer_calls=0, vae_calls=0, backward_calls=0, mp4=0, saved_frames=0)
        self.completed = dict(self.counts)
        self.events = []
    def check(self):
        if self.clock()-self.start >= self.budget['wall_seconds']: raise TimeoutError('WALL_BUDGET')
    def consume(self, kind, amount=1):
        self.check()
        if self.counts[kind]+amount > self.budget[kind]: raise RuntimeError('CALL_BUDGET_' + kind)
        self.counts[kind] += amount
        self.stage(kind, 'START', attempted=self.counts[kind], completed=self.completed[kind])
    def complete(self, kind, amount=1):
        self.completed[kind] += amount
        if self.completed[kind] > self.counts[kind]:raise RuntimeError('completion without attempt')
        self.stage(kind, 'COMPLETE', attempted=self.counts[kind], completed=self.completed[kind])
    def stage(self, name, status, **details):
        self.events.append(dict(stage=name, status=status, elapsed_seconds=self.clock()-self.start, **details))
        if self.persist:self.persist(self.counts)
    def remaining(self):
        self.check()
        return max(.001, self.budget['wall_seconds']-(self.clock()-self.start))


def load_wan(c, guard, *, torch=None, pipeline_class=None, vae_class=None, record=None):
    """Reuse the prior working full-pipeline load/encode/prepare flow.

    Model CPU offload is deliberately not installed: remaining-rollout autograd
    needs resident frozen transformer and independently loaded FP32 VAE.
    Encoder is released before these two modules become GPU resident.
    Versions and scheduler settings are recorded as actually loaded.
    """
    if torch is None: import torch
    if pipeline_class is None or vae_class is None:
        from diffusers import WanPipeline, AutoencoderKLWan
        pipeline_class, vae_class = WanPipeline, AutoencoderKLWan
    g, model = c['generation'], c['model']
    kw = dict(torch_dtype=torch.bfloat16)
    if model.get('revision'): kw['revision'] = model['revision']
    guard.check()
    pipe = pipeline_class.from_pretrained(model['id'], **kw)
    if getattr(pipe, 'transformer_2', None) is not None or getattr(pipe.config, 'boundary_ratio', None) is not None or getattr(pipe.config, 'expand_timesteps', False):
        raise ValueError('dual/expanded timestep model requires a different method adapter')
    device = torch.device('cuda')
    pipe.text_encoder.to(device)
    with torch.no_grad():
        cond, uncond = pipe.encode_prompt(prompt=g['prompt'], negative_prompt=g['negative_prompt'], do_classifier_free_guidance=True, num_videos_per_prompt=1, max_sequence_length=g['max_sequence_length'], device=device)
    cond, uncond = cond.to(torch.bfloat16), uncond.to(torch.bfloat16)
    pipe.text_encoder.to('cpu'); pipe.text_encoder = None
    gc.collect(); torch.cuda.empty_cache(); guard.check()
    vae_kw = dict(subfolder='vae', torch_dtype=torch.float32)
    if model.get('revision'): vae_kw['revision'] = model['revision']
    pipe.vae = None
    gc.collect()
    vae = vae_class.from_pretrained(model['id'], **vae_kw).eval()
    # Assign the actual VAE before using pipeline geometry helpers.
    pipe.vae = vae
    pipe.vae_scale_factor_temporal = getattr(vae.config, 'scale_factor_temporal', None) or 2 ** sum(vae.config.temperal_downsample)
    pipe.vae_scale_factor_spatial = getattr(vae.config, 'scale_factor_spatial', None) or 2 ** len(vae.config.temperal_downsample)
    if g['height'] % pipe.vae_scale_factor_spatial or g['width'] % pipe.vae_scale_factor_spatial or (g['frames']-1) % pipe.vae_scale_factor_temporal:
        raise ValueError('requested video geometry is not representable by this VAE')
    for module in (pipe.transformer, vae):
        disable = getattr(module, 'disable_gradient_checkpointing', None)
        if disable: disable()
    pipe.transformer.to(device); vae.to(device)
    guard.check()
    with torch.no_grad():
        initial = pipe.prepare_latents(1, int(pipe.transformer.config.in_channels), g['height'], g['width'], g['frames'], torch.float32, device, torch.Generator(device=device).manual_seed(g['seed']), None).detach()
    pipe.scheduler.set_timesteps(g['steps'], device=device)
    if hasattr(pipe.scheduler, 'set_begin_index'): pipe.scheduler.set_begin_index(0)
    # No class/version/config equality gate. Adapter checks the actual step cursor.
    ledger = None
    if c.get('checkpoint_recomputation') is not None:
        if c.get('save_forecast_tensors_on_cpu', False):raise ValueError('checkpoint path cannot use CPU activation transfer')
        from runtime.public_statistic.checkpointing import CheckpointLedger
        ledger = CheckpointLedger(c['checkpoint_recomputation'], resource_check=guard.check)
    adapter = WanTerminalAdapter(pipe.transformer, vae, cond, uncond, guidance_scale=g['guidance'], torch=torch, resource_guard=guard, save_forecast_tensors_on_cpu=c.get("save_forecast_tensors_on_cpu", False), checkpoint_ledger=ledger)
    if record: record(dict(checkpoint_recomputation_limits=ledger.limits if ledger else None, save_forecast_tensors_on_cpu=adapter.save_forecast_tensors_on_cpu, model=model, scheduler_class=type(pipe.scheduler).__name__, scheduler_config=dict(pipe.scheduler.config), timesteps=pipe.scheduler.timesteps.detach().cpu().tolist(), latent_shape=list(initial.shape), latent_dtype=str(initial.dtype), transformer_dtype=str(adapter.input_dtype()), transformer_first_parameter_dtype=str(next(pipe.transformer.parameters()).dtype), vae_dtype=str(next(vae.parameters()).dtype), offload=False, transformer_checkpointing=bool(getattr(pipe.transformer, 'is_gradient_checkpointing', False)), vae_checkpointing=bool(getattr(vae, 'is_gradient_checkpointing', False)), transformer_resolved_revision=getattr(pipe.transformer.config, '_commit_hash', None), vae_resolved_revision=getattr(vae.config, '_commit_hash', None)))
    return adapter, initial, SolverState(pipe.scheduler, 0)


def execute_arms(adapter, initial, state, c, result, persist, save_arm):
    """Shared prefix, independent full-state OFF tails, fixed OFF1 Y-minus control."""
    torch = adapter.torch
    prefix_next = c['feedback']['after_indices'][0]
    z = initial
    result['status'] = 'RUNNING'; result['active'] = 'PREFIX'; persist()
    if adapter.resource_guard:adapter.resource_guard.stage('PREFIX', 'START')
    with torch.no_grad():
        while state.next_index < prefix_next: z, state = adapter.advance(z, state)
    if adapter.resource_guard:adapter.resource_guard.stage('PREFIX', 'COMPLETE')
    r0 = float(z.square().mean().sqrt())
    if not math.isfinite(r0) or r0 <= 0: raise ValueError('nonfinite/zero prefix RMS')
    result['prefix'] = dict(next_index=state.next_index, rms=r0, history='complete unchanged cloned per arm')
    off1 = None
    for arm in ARMS:
        result['active'] = arm; result['arms'][arm]['status'] = 'RUNNING'; persist()
        if adapter.resource_guard:adapter.resource_guard.stage(arm, 'START')
        if arm.startswith('OFF'):
            with torch.no_grad(): rgb = adapter.rollout(z.detach().clone(), clone_graph_state(state, torch))
            if not bool(torch.isfinite(rgb).all()): raise RuntimeError('NONFINITE_OFF_TERMINAL')
            if arm == 'OFF1': off1 = rgb.detach().clone()
            elif not torch.equal(rgb, off1): raise RuntimeError('OFF_SHARED_PREFIX_REPEAT_DIFFERENCE')
            feedback = []
        else:
            f = c['feedback']
            cfg = FeedbackConfig(tuple(f['after_indices']), f['learning_rate'], f['per_update_fraction']*r0, f['cumulative_fraction']*r0, f['quality_clip_rmse'], f['amplitude'], 1, -1, c['generation']['guidance'], status='EXECUTED_CONFIGURATION_NOT_SCIENTIFIC_VALIDATION')
            def event(row):
                result['arms'][arm].setdefault('feedback', []).append(row.copy()); persist()
            controlled = controlled_run(adapter, z, state, cfg, off1_terminal_rgb=off1, enabled=True, strict=True, event_callback=event)
            rgb, feedback = controlled['rgb'], controlled['feedback']
            result['arms'][arm]['feedback'] = feedback
            result['arms'][arm]['configuration'] = controlled['configuration']
            result['arms'][arm]['cumulative_accepted_rms'] = controlled['cumulative_accepted_rms']
        if tuple(rgb.shape) != (c['generation']['frames'], c['generation']['height'], c['generation']['width'], 3): raise ValueError('terminal RGB geometry')
        q = adapter.readout(rgb)
        if not bool(torch.isfinite(q).all()): raise RuntimeError('NONFINITE_PUBLIC_READOUT')
        quality = (rgb-off1).square().mean().sqrt()
        per_frame = (rgb-off1).square().mean(dim=(1,2,3)).sqrt()
        result['arms'][arm].update(float_q=q.detach().cpu().tolist(), terminal_clip_rmse=float(quality), per_frame_rmse_diagnostic=per_frame.detach().cpu().tolist(), worst_frame_rmse_diagnostic=float(per_frame.max()), feedback=feedback)
        save_arm(arm, rgb, off1)
        result['arms'][arm]['status'] = 'COMPLETE'
        if adapter.resource_guard:adapter.resource_guard.stage(arm, 'COMPLETE')
        if arm == 'Y_MINUS': result['arms'][arm]['control_outcome'] = 'ACCEPTED_PROPOSAL' if any(row['accepted'] for row in feedback) else 'NO_ACCEPTED_PROPOSAL_WITH_THIS_CONFIG'
        persist()
        # Completed arm output is persisted; OFF1 remains the fixed reference.
        del rgb, q, quality, per_frame
    result['status'] = 'EXECUTED_REQUIRES_REVIEW'
    result['active'] = None; persist()


def persist_arm_media(arm, rgb, off, *, c, result, guard, out):
    import numpy as np
    import subprocess
    folder = out/arm; folder.mkdir()
    values = rgb.detach().cpu().numpy()
    np.save(folder/'terminal_float_rgb.npy', values, allow_pickle=False)
    u8 = np.rint(values*255).astype(np.uint8)
    guard.consume('mp4'); guard.consume('saved_frames', c['generation']['frames'])
    g = c['generation']; path = folder/'saved.mp4'
    cmd = ['ffmpeg','-v','error','-threads','1','-f','rawvideo','-pix_fmt','rgb24','-s',f"{g['width']}x{g['height']}",'-r',str(g['fps']),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-threads','1','-n',str(path)]
    encoded = subprocess.run(cmd, input=u8.tobytes(), capture_output=True, timeout=guard.remaining())
    write(folder/'encode.json', dict(command=cmd, returncode=encoded.returncode, stderr=encoded.stderr.decode(errors='replace')))
    encoded.check_returncode()
    guard.complete('mp4')
    guard.stage('MP4_READBACK', 'START', arm=arm)
    decoded = subprocess.run(['ffmpeg','-v','error','-threads','1','-noautorotate','-i',str(path),'-map','0:v:0','-f','rawvideo','-pix_fmt','rgb24','-'], capture_output=True, timeout=guard.remaining())
    decoded.check_returncode()
    expected = g['frames']*g['height']*g['width']*3
    if len(decoded.stdout) != expected: raise RuntimeError('SAVED_FRAME_COUNT')
    guard.complete('saved_frames', g['frames'])
    data = np.frombuffer(decoded.stdout, np.uint8).reshape(values.shape)/255
    from main.sc_sstw.public_luma_statistic import read_rgb
    q = read_rgb(data, np)
    guard.stage('MP4_READBACK', 'COMPLETE', arm=arm)
    reference = off.detach().cpu().numpy()
    result['arms'][arm].update(mp4_q=q.tolist(), mp4_clip_rmse_vs_float_off1=float(np.sqrt(np.mean((data-reference)**2))), mp4_per_frame_rmse_diagnostic=np.sqrt(np.mean((data-reference)**2, axis=(1,2,3))).tolist(), rows=[dict(index=i,time=i/g['fps'],q=q[i].tolist(),status='COMPLETE') for i in range(g['frames'])])


def worker(config_path, output):
    import importlib.metadata
    import subprocess
    import sys
    c = json.loads(Path(config_path).read_text()); validate_config(c)
    out = Path(output); result = initial_result(c); adapter = None
    guard = ResourceGuard(c['budget'])
    def progress(counts):
        payload=dict(attempted=counts, completed=guard.completed, current=guard.events[-1] if guard.events else None)
        if adapter is not None and adapter.checkpoint_ledger is not None:
            payload['recomputation']=dict(counts=adapter.checkpoint_ledger.counts, capture='LAST_TOP_LEVEL_STAGE; internal replay counts can advance before next flush')
        write(out/'counts.json', payload)
    guard.persist = progress
    def persist(): write(out/'result.json', result)
    result['status']='RUNNING'; result['active']='LOAD_MODEL'
    persist(); write(out/'config.json', c); guard.stage('WORKER','START')
    try:
        import torch
        import numpy as np
        import diffusers
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError('CUDA BF16 device required')
        guard.torch = torch
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.reset_peak_memory_stats()
        write(out/'runtime.json', dict(python=sys.version, torch=torch.__version__, diffusers=diffusers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), packages={x: importlib.metadata.version(x) for x in ('transformers','accelerate','numpy')}, device_total_memory_bytes=total, allocator_cap_gib=None))
        guard.stage('LOAD_MODEL', 'START')
        adapter, initial, state = load_wan(c, guard, torch=torch, record=lambda record: write(out/'loaded_model.json', record))
        guard.stage('LOAD_MODEL', 'COMPLETE')
        def save_arm(arm, rgb, off):
            persist_arm_media(arm, rgb, off, c=c, result=result, guard=guard, out=out)
        execute_arms(adapter, initial, state, c, result, persist, save_arm)
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        guard.stage('WORKER','COMPLETE')
    except BaseException as exc:
        result['status'] = 'FATAL_STOP'; result['error'] = repr(exc)
        active = result.get('active')
        if active in ARMS:
            result['arms'][active]['status'] = 'FAILED'
            for row in result['arms'][active]['rows']:
                if row['status'] != 'COMPLETE': row['status'] = 'FAILED'
        raise
    finally:
        result['counts'] = dict(attempted=guard.counts, completed=guard.completed); result['elapsed_seconds'] = time.monotonic()-guard.start
        persist(); progress(guard.counts); write(out/'progress.json', guard.events)
        if adapter is not None and adapter.checkpoint_ledger is not None:write(out/'recomputation.json',dict(capture='FINAL_WORKER_CLEANUP',**adapter.checkpoint_ledger.summary()))
